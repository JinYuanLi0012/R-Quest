"""Warm the frozen-Base cache and validate a local, merged Solver checkpoint."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from validate_hf_checkpoint import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--solver', required=True)
    args = parser.parse_args()
    solver = Path(args.solver)
    if not solver.is_dir():
        parser.error('Solver must be a local merged HF directory. Train with train_validity.sh, or download the released step-15 weights first.')
    validate_checkpoint(solver)
    if not Path(args.base).is_dir():
        snapshot_download(args.base, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken'])
    print('Base snapshot ready; Solver checkpoint validated.')

if __name__ == '__main__':
    main()
