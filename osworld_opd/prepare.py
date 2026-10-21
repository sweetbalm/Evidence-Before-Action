"""Build prompt-only training rows from an explicit OSWorld training split."""
import argparse
import json
from pathlib import Path

from osworld_opd.core import ROOT, load_config, load_task


def task_paths(selection):
    if isinstance(selection, dict):
        return [f'{domain}/{task_id}.json' for domain, ids in selection.items() for task_id in ids]
    if isinstance(selection, list) and all(isinstance(x, str) for x in selection):
        return selection
    raise ValueError('Split must be an OSWorld domain-to-IDs map or list of relative JSON paths')


def prepare(config, split, system, output):
    paths = task_paths(split)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Training split is empty or contains duplicate tasks')
    rows = []
    for relative in paths:
        task = load_task(config['task_root'], relative)
        rows.append({'messages': [{'role': 'system', 'content': system.strip()},
                                  {'role': 'user', 'content': task['instruction']}],
                     'osworld_task': relative, 'osworld_task_id': task['id']})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Explicit output path, but don't accidentally replace an existing dataset.
    with output.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--split', required=True, help='Explicit training split JSON, not the held-out evaluation split')
    parser.add_argument('--system-prompt', default=str(ROOT / 'opd/system_prompt.txt'))
    parser.add_argument('--output', default=str(ROOT / 'opd/data/train.jsonl'))
    args = parser.parse_args()
    count = prepare(load_config(args.config), json.loads(Path(args.split).read_text(encoding='utf-8')),
                    Path(args.system_prompt).read_text(encoding='utf-8'), args.output)
    print(f'Wrote {count} online task prompts to {args.output}')


if __name__ == '__main__':
    main()
