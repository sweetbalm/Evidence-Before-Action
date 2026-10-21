"""Launch Student rollout and LoRA training in the existing server environment."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from osworld_opd.core import ROOT, load_config


def load_launch(path=None):
    path = Path(path or os.environ.get('OSWORLD_OPD_LAUNCH_CONFIG', ROOT / 'opd/launch.json')).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    for key in ('dataset', 'output_dir'):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else (ROOT / value).resolve())
    groups = [config[f'{role}_gpus'].split(',') for role in ('rollout', 'trainer')]
    all_devices = [d.strip() for group in groups for d in group]
    if len(set(all_devices)) != len(all_devices) or any(not x for x in all_devices):
        raise ValueError('Rollout and trainer must use disjoint, nonempty GPU sets')
    if any(len(group) != 1 for group in groups):
        raise ValueError('This integration supports one rollout GPU and one trainer GPU')
    ports = [config[k] for k in ('rollout_port', 'weight_sync_port', 'master_port')]
    if len(set(ports)) != len(ports) or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
        raise ValueError('Ports must be distinct integers in [1024, 65535]')
    if config['context_length'] <= config['max_completion_length'] or config['max_completion_length'] <= 0:
        raise ValueError('Invalid context/completion lengths')
    if config['template'] != 'qwen3_5':
        raise ValueError('This profile pins the shared qwen3_5 template; audit before changing it')
    return config


def cli_args(values):
    args = []
    for key, value in values.items():
        if value is None:
            continue
        args.append('--' + key)
        if isinstance(value, bool):
            args.append(str(value).lower())
        elif isinstance(value, (dict, list)):
            args.append(json.dumps(value, separators=(',', ':')))
        else:
            args.append(str(value))
    return args


def build_command(role, config, environment_config, smoke=False, resume=None):
    common = {
        'model': config['student_model'],
        'model_type': config['student_model_type'],
        'template': config['template'], 'torch_dtype': 'bfloat16',
        'enable_thinking': False, 'preserve_thinking': True, 'response_prefix': '',
        'add_non_thinking_prefix': False, 'disable_ignore_empty_think': True,
        'template_backend': 'swift', 'max_pixels': config['max_pixels'],
        'max_length': config['context_length'], 'truncation_strategy': 'delete',
        'loss_scale': 'default', 'strict': True,
    }
    if role == 'rollout':
        common.update({
            'host': '127.0.0.1', 'port': config[role + '_port'],
            'infer_backend': 'vllm', 'vllm_use_async_engine': True,
            'vllm_tensor_parallel_size': len(config[role + '_gpus'].split(',')),
            'vllm_data_parallel_size': 1, 'vllm_max_model_len': config['context_length'],
            'vllm_gpu_memory_utilization': 0.85, 'vllm_max_num_seqs': 2,
            'vllm_limit_mm_per_prompt': {'image': environment_config['max_steps']},
            'vllm_enable_prefix_caching': False, 'max_logprobs': 1,
        })
        common.update({'multi_turn_scheduler': 'osworld_opd',
                       'external_plugins': str(ROOT / 'opd/plugin.py'),
                       'max_turns': environment_config['max_steps'], 'vllm_enable_lora': False,
                       'vllm_max_model_len': config['context_length'] - 8})
        route = 'rollout'
    else:
        route = 'rlhf'
        common.update(config['train'])
        common.update({
            'rlhf_type': 'grpo', 'teacher_model_server': config['teacher_url'].rstrip('/'),
            'teacher_kl_coef': 1.0, 'beta': 0.0,
            'tuner_type': 'lora', 'target_modules': 'all-linear',
            'freeze_vit': True, 'freeze_aligner': True,
            'use_vllm': True, 'vllm_mode': 'server', 'vllm_enable_lora': False,
            'vllm_server_host': '127.0.0.1', 'vllm_server_port': config['rollout_port'],
            'vllm_server_group_port': config['weight_sync_port'], 'vllm_server_timeout': 600,
            'vllm_server_pass_dataset': True, 'external_plugins': str(ROOT / 'opd/plugin.py'),
            'async_generate': False, 'num_iterations': 1,
            'dataset': config['dataset'], 'split_dataset_ratio': 0,
            'remove_unused_columns': False, 'load_from_cache_file': False,
            'dataloader_num_workers': 0, 'dataset_num_proc': 1,
            'per_device_train_batch_size': 1, 'num_generations': 1,
            'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0,
            'max_completion_length': config['max_completion_length'],
            'loss_type': 'grpo', 'scale_rewards': 'none',
            'rollout_importance_sampling_mode': 'token_truncate',
            'log_rollout_offpolicy_metrics': True,
            'padding_free': False, 'sequence_parallel_size': 1,
            'overlong_filter': False,
            'gradient_checkpointing': True, 'attn_impl': 'sdpa',
            'output_dir': config['output_dir'], 'logging_steps': 1,
            'report_to': 'none', 'save_only_model': False,
        })
        # The scheduler lives ONLY in the rollout server. The server advertises
        # multi-turn support; passing its name here would initialize a second env scheduler.
        if smoke:
            common.update({'max_steps': 1, 'gradient_accumulation_steps': 1,
                           'save_steps': 1, 'output_dir': config['output_dir'] + '-smoke'})
        if resume:
            common['resume_from_checkpoint'] = str(Path(resume).resolve())
    return [sys.executable, '-m', 'swift.cli.main', route, *cli_args(common)]


def runtime_env(role, config, config_path):
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': config[role + '_gpus'],
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'ms-swift'), str(ROOT), env.get('PYTHONPATH', '')]),
        'OSWORLD_OPD_CONFIG': str(Path(config_path).resolve()), 'OSWORLD_OPD_ROLE': role,
        'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'IMAGE_MAX_TOKEN_NUM': '1024',
    })
    # Avoid inherited torchrun settings launching extra workers sharing one VM.
    for key in ('NPROC_PER_NODE', 'NNODES', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    if role == 'trainer':
        env.update({'NPROC_PER_NODE': '1', 'MASTER_PORT': str(config['master_port'])})
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['rollout', 'trainer'])
    parser.add_argument('--launch-config', default=str(ROOT / 'opd/launch.json'))
    parser.add_argument('--config', default=str(ROOT / 'opd/config.json'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = load_launch(args.launch_config)
    env_config = load_config(args.config)
    command = build_command(args.role, config, env_config, args.smoke, args.resume)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if os.name != 'posix':
        raise RuntimeError('Real launches require Linux on the H100 server; use --dry-run locally')
    if not Path(config['student_model']).is_dir():
        raise FileNotFoundError('Set student_model to the merged BF16 checkpoint in launch.json')
    if args.role == 'trainer':
        if not Path(config['dataset']).is_file():
            raise FileNotFoundError('Create the training manifest with osworld_opd.prepare first')
    env = runtime_env(args.role, config, args.config)
    env['OSWORLD_OPD_LAUNCH_CONFIG'] = str(Path(args.launch_config).resolve())
    manifest_dir = ROOT / 'opd/runs/launches'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    (manifest_dir / f'{stamp}-{args.role}.json').write_text(
        json.dumps({'command': command, 'launch_config': config, 'environment_config': env_config,
                    'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES']}, indent=2), encoding='utf-8')
    raise SystemExit(subprocess.call(command, env=env, cwd=ROOT))


if __name__ == '__main__':
    main()
