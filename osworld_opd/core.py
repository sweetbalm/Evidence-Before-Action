"""Configuration, runtime alignment guards and episode traces."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get('OSWORLD_OPD_CONFIG', ROOT / 'opd/config.json')).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    for key in ('task_root', 'trace_root'):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else (ROOT / value).resolve())
    for key in ('max_steps', 'max_concurrent_envs', 'max_total_response_tokens'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['desktop'].get('path_to_vm') and config['max_concurrent_envs'] != 1:
        raise ValueError('A fixed path_to_vm requires max_concurrent_envs=1')
    if config['pause_seconds'] < 0:
        raise ValueError('pause_seconds must be nonnegative')
    if config['teacher_http_timeout_seconds'] <= 0:
        raise ValueError('teacher_http_timeout_seconds must be positive')
    return config


def load_task(root, relative_path):
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or path.suffix != '.json':
        raise ValueError('Task must be a JSON file within task_root')
    task = json.loads(path.read_text(encoding='utf-8'))
    if not task.get('id') or not task.get('instruction') or not task.get('evaluator'):
        raise ValueError(f'Task missing id/instruction/evaluator: {path}')
    return task


def parse_action(text):
    """Keep one complete <action> block; never execute model code on the host."""
    blocks = re.findall(r'<action>\s*(.*?)\s*</action>', text, re.I | re.S)
    if len(blocks) != 1:
        raise ValueError('Expected exactly one complete <action>...</action> block')
    code = blocks[0].strip()
    fence = re.fullmatch(r'```(?:python|py)?\s*(.*?)```', code, re.I | re.S)
    if fence:
        code = fence[1].strip()
    if not code:
        raise ValueError('Empty action')
    if code not in ('WAIT', 'DONE', 'FAIL'):
        ast.parse(code, mode='exec')
    return code


def evidence_regions(text):
    """Accept the local SFT visual_anchor and SCREENSHOT JSON evidence formats."""
    match = re.search(r'<evidence>(.*?)</evidence>', text, re.I | re.S)
    if not match:
        return []
    body = match[1]
    regions = []
    for anchor in re.findall(r'<visual_anchor\b([^>]+)/?>', body, re.I):
        attrs = dict(re.findall(r'(x|y|width|height)\s*=\s*[\"\'](\d+)[\"\']', anchor))
        if len(attrs) == 4:
            regions.append({k: int(v) for k, v in attrs.items()})
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = body.find('{', cursor)
        if start < 0:
            break
        try:
            item, consumed = decoder.raw_decode(body[start:])
            cursor = start + consumed
        except ValueError:
            cursor = start + 1
            continue
        if isinstance(item, dict) and item.get('evidence_type') == 'SCREENSHOT':
            regions.append(item.get('parameters', item))
    return regions


def validate_turn(sampled_ids, normalized_ids, mask, logprobs):
    if not sampled_ids or any(type(x) is not int or x < 0 for x in sampled_ids):
        raise ValueError('Missing/invalid sampled token IDs')
    if len(normalized_ids) != len(mask) or any(x not in (0, 1) for x in mask):
        raise ValueError('Invalid response loss mask')
    selected = [x for x, keep in zip(normalized_ids, mask) if keep]
    if selected != sampled_ids:
        raise ValueError('Normalized loss tokens differ from sampled IDs')
    if len(logprobs) != len(selected) or any(not math.isfinite(x) or x > 1e-5 for x in logprobs):
        raise ValueError('Missing, nonfinite or misaligned rollout logprobs')


def selected_ids(turn_ids, masks):
    if len(turn_ids) != len(masks):
        raise ValueError('Turn count mismatch')
    result = []
    for ids, mask in zip(turn_ids, masks):
        if len(ids) != len(mask):
            raise ValueError('Token/mask length mismatch')
        result.extend(t for t, keep in zip(ids, mask) if keep)
    return result


def align_teacher(input_ids, labels, parsed, expected_ids):
    """Strict full-context alignment. parsed omits position zero (swift contract).

    Unlike the upstream contiguous-response fallback this supports interleaved
    user/image tokens and arbitrary masked deterministic assistant prefixes.
    No suffix search, text re-tokenization or length-only matching is allowed.
    """
    lps, indices = parsed
    if len(input_ids) != len(labels) or len(input_ids) < 2:
        raise ValueError('Invalid student encoding')
    teacher_ids = [row[0] for row in indices if len(row) == 1]
    if teacher_ids != input_ids[1:] or len(lps) != len(teacher_ids):
        raise ValueError('Teacher full-context token IDs differ from student encoding; '
                         'check tokenizer, template, image processing, EOS and truncation')
    positions = [i for i, label in enumerate(labels) if label != -100]
    if not positions or positions[0] == 0:
        raise ValueError('Invalid supervised positions')
    if [input_ids[i] for i in positions] != expected_ids:
        raise ValueError('Encoded supervised tokens differ from sampled loss tokens')
    if any(labels[i] != input_ids[i] for i in positions):
        raise ValueError('Labels differ from input IDs')
    scores = []
    for i in positions:
        row = lps[i - 1]
        if len(row) != 1 or not math.isfinite(row[0]) or row[0] > 1e-5:
            raise ValueError('Invalid teacher sampled-token logprob')
        scores.append(float(row[0]))
    return scores, positions


class Trace:
    def __init__(self, root, trace_id=None):
        self.id = trace_id or uuid.uuid4().hex
        if not re.fullmatch(r'[a-f0-9]{32}', self.id):
            raise ValueError('Invalid trace ID')
        self.path = Path(root) / self.id
        self.path.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **data):
        record = {'event': kind, 'time_unix': time.time(), 'trace_id': self.id, **data}
        # Each trajectory has one writer; training uses separate per-rank files.
        with (self.path / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def observation(self, observation, step):
        screenshot = observation.get('screenshot')
        if not isinstance(screenshot, bytes) or not screenshot.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('OSWorld observation must contain PNG screenshot bytes')
        name = f'observation-{step:03d}.png'
        (self.path / name).write_bytes(screenshot)
        metadata = {k: v for k, v in observation.items() if k != 'screenshot'}
        # Unexpected binary observation values should fail rather than disappear.
        (self.path / f'observation-{step:03d}.json').write_text(
            json.dumps(metadata, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        self.event('observation', step=step, image=name,
                   sha256=hashlib.sha256(screenshot).hexdigest())
        return base64.b64encode(screenshot).decode('ascii')

    def snapshot(self, filename, value):
        (self.path / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
