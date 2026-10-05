"""Inspect the release configuration and fail early on missing runtime assets."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess


def config():
    keys = ['BASE_MODEL', 'VALIDITY_DATASET', 'VALIDITY_MAX_STEPS', 'VALIDITY_RZERO_INITIAL_SOLVER',
            'VALIDITY_RZERO_DIVERSITY_MODE', 'VALIDITY_RZERO_NOVELTY_K', 'VALIDITY_RZERO_NOVELTY_SEED',
            'TERRA_REPLAY_RATIO', 'QUESTIONER_MAX_STEPS', 'SOLVER_MAX_STEPS', 'RZERO_NUM_ROUNDS',
            'QUESTIONER_TRAIN_GPU_IDS', 'VLLM_GPU_IDS', 'QUESTION_GPU_IDS', 'STORAGE_PATH', 'MODEL_ABBR']
    return {key: os.getenv(key) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['validity', 'rquest'], required=True)
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(config(), indent=2))
    if args.config_only:
        return
    failures = []
    for module in ['torch', 'vllm', 'transformers', 'datasets', 'ray', 'flash_attn', 'mathruler', 'stopit']:
        if importlib.util.find_spec(module) is None:
            failures.append(f'Missing Python package: {module}')
    for program in ['nvidia-smi', 'setsid']:
        if shutil.which(program) is None:
            failures.append(f'Missing Linux executable: {program}')
    if failures:
        raise SystemExit('\n'.join(failures))
    import torch
    gpu_ids = os.environ['QUESTION_GPU_IDS'].split(',')
    if not torch.cuda.is_available() or any(int(i) >= torch.cuda.device_count() for i in gpu_ids):
        failures.append('Selected GPUs are not visible to PyTorch; use four allocated GPU indices.')
    if args.stage == 'rquest':
        from validate_hf_checkpoint import validate_checkpoint
        validate_checkpoint(Path(os.environ['VALIDITY_RZERO_INITIAL_SOLVER']))
        import socket
        for port in range(int(os.getenv('VLLM_PORT_BASE', '5000')), int(os.getenv('VLLM_PORT_BASE', '5000')) + 2):
            with socket.socket() as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    failures.append(f'Solver port {port} is already in use.')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('Preflight passed.')

if __name__ == '__main__':
    main()
