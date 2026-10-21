"""Score full Student trajectories through vLLM 0.32 token-in/token-out HTTP."""
from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import time
from dataclasses import asdict
from functools import wraps
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from osworld_opd.core import Trace, align_teacher, load_config, selected_ids
from osworld_opd.launch import load_launch


def score_sample(sample, template, config, timeout, trace):
    """Send original IDs plus images; verify the server's image-expanded IDs."""
    from swift.rlhf_trainers.gkd_helpers import build_teacher_requests

    request = build_teacher_requests([sample], template)[0]
    request.add_eos = False
    # Shallow copy preserves the shared model/processor without copying weights.
    render_template = copy.copy(template)
    render_template.set_mode('vllm')
    rendered = render_template.encode(asdict(request), return_template_inputs=True)
    token_ids = rendered['input_ids']
    # Native template preprocessing already resized these images exactly as for
    # Student. vLLM must not resize them again. PNG preserves the resulting pixels.
    inputs = rendered['template_inputs']
    if inputs.videos or inputs.audios:
        raise ValueError('This adapter supports screenshot-only OSWorld trajectories')
    content_parts = []
    for picture in inputs.images:
        buffer = io.BytesIO()
        picture.save(buffer, format='PNG')
        data = base64.b64encode(buffer.getvalue()).decode('ascii')
        # TITO uses a flat `url`, NOT OpenAI chat's nested `image_url.url`.
        content_parts.append({'type': 'image_url', 'url': 'data:image/png;base64,' + data})
    payload = {
        'model': config['teacher_served_model'],
        # These IDs contain one placeholder per image, not already-expanded
        # vision patches. vLLM expands only those placeholders with the images.
        'token_ids': token_ids,
        'content_parts': content_parts,
        'sampling_params': {'prompt_logprobs': 0, 'max_tokens': 1,
                            'temperature': 0, 'detokenize': False},
        'return_token_ids': True, 'output_mode': 'tokens', 'stream': False,
    }
    # TITO's raw-media path does not forward per-request mm_processor_kwargs.
    # Set --mm-processor-kwargs '{"do_resize":false}' on the Teacher server.
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('OSWORLD_TEACHER_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OSWORLD_TEACHER_API_KEY']
    url = config['teacher_url'].rstrip('/') + '/inference/v1/generate'
    http_request = Request(url, json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urlopen(http_request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        hint = ('; start the vLLM 0.32 Teacher with --enable-scale-out'
                if error.code in (403, 404) else '')
        raise RuntimeError(f'Teacher HTTP {error.code}{hint}: '
                           f'{error.read().decode(errors="replace")[:4000]}') from error
    expected = sample.encoded['input_ids']
    returned = result.get('prompt_token_ids')
    rows = result.get('prompt_logprobs')
    if returned != expected or not isinstance(rows, list) or len(rows) != len(expected):
        trace.snapshot('teacher-response-error.json', result)
        raise ValueError('Teacher prompt IDs/logprobs differ from Student. Check tokenizer compatibility, '
                         'image processing, --mm-processor-kwargs and vLLM TITO support.')
    lps, indices = [], []
    for token, row in zip(expected[1:], rows[1:]):
        entry = row.get(str(token)) if isinstance(row, dict) else None
        if not isinstance(entry, dict) or 'logprob' not in entry:
            trace.snapshot('teacher-response-error.json', result)
            raise ValueError('Teacher omitted the actual input token prompt_logprob')
        lps.append([entry['logprob']])
        indices.append([token])
    return lps, indices


def assemble_osworld_teacher(self, samples, batch_encoded_inputs):
    import torch
    if (self.accelerator.num_processes != 1 or self.template.padding_free
            or self.template.sequence_parallel_size != 1):
        raise ValueError('OSWorld teacher bridge requires one trainer rank, no packing/sequence parallel')
    chunks = [batch.pop('_chunk_samples') for batch in batch_encoded_inputs]
    flat = [sample for chunk in chunks for sample in chunk]
    if not flat or any(not s.rollout_infos.get('trace_id') for s in flat):
        raise ValueError('OSWorld teacher bridge received a sample without a trace ID')
    if any(s.teacher_prompt or s.teacher_images is not None for s in flat):
        raise ValueError('This bridge uses the same observation history for teacher and student')
    config = load_config()
    launch = load_launch()
    root = config['trace_root']
    for batch, chunk in zip(batch_encoded_inputs, chunks):
        frame = batch['grpo_batch']
        teacher = torch.zeros_like(frame.old_per_token_logps)
        for row, sample in enumerate(chunk):
            trace = Trace(root, sample.rollout_infos['trace_id'])
            scored = None
            ids = selected_ids(sample.response_token_ids, sample.response_loss_mask)
            try:
                scored = score_sample(sample, self.template, launch, config['teacher_http_timeout_seconds'], trace)
                scores, positions = align_teacher(sample.encoded['input_ids'], sample.encoded['labels'], scored, ids)
                mask = frame.completion_mask[row]
                if int(mask.sum().item()) != len(scores):
                    raise ValueError('Teacher scores and training completion mask differ')
                # Verify the actual collated input frame too, including padding.
                actual_ids = batch['model_inputs']['input_ids'][row, -mask.numel():][mask].detach().cpu().tolist()
                if actual_ids != ids:
                    raise ValueError('Collated model inputs differ from scored response tokens')
                teacher[row, mask] = torch.tensor(scores, dtype=teacher.dtype, device=teacher.device)
                old = frame.old_per_token_logps[row, mask].detach().float().cpu().tolist()
                rollout = [lp for turn in sample.rollout_logprobs for lp in turn]
                if len(rollout) != len(ids) or any(not math.isfinite(lp) for lp in old + rollout):
                    raise ValueError('Invalid student logprob frame')
                record = {
                    'event': 'teacher_scoring', 'time_unix': time.time(), 'trace_id': trace.id,
                    'teacher_url': launch['teacher_url'], 'teacher_model': launch['teacher_served_model'],
                    'global_step': self.state.global_step, 'rank': self.accelerator.process_index,
                    'request_id': sample.request_id, 'prompt_id': sample.prompt_id,
                    'input_ids': sample.encoded['input_ids'], 'supervised_positions': positions,
                    'token_ids': ids, 'teacher_logprobs': scores,
                    'old_policy_logprobs': old, 'rollout_logprobs': rollout,
                    'teacher_kl_coef': self.teacher_kl_coef,
                    'distillation_advantages': [self.teacher_kl_coef * (t - s) for t, s in zip(scores, old)],
                }
                with (trace.path / f'training-rank-{self.accelerator.process_index}.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            except Exception as error:
                trace.snapshot(f'alignment-error-rank-{self.accelerator.process_index}.json', {
                    'error': str(error), 'student_input_ids': sample.encoded['input_ids'],
                    'student_labels': sample.encoded['labels'], 'teacher_parsed': scored,
                    'response_token_ids': sample.response_token_ids,
                    'response_loss_mask': sample.response_loss_mask})
                raise
        frame.teacher_per_token_logps = teacher


def install():
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    original = GRPOTrainer._assemble_teacher_api_logps
    if getattr(original, '_osworld_opd', False):
        return

    @wraps(original)
    def bridge(self, samples, batch_encoded_inputs):
        return assemble_osworld_teacher(self, samples, batch_encoded_inputs)

    bridge._osworld_opd = True
    GRPOTrainer._assemble_teacher_api_logps = bridge
