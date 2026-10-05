#!/usr/bin/env python3
"""Minimal structural validation for a merged Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_checkpoint(root: Path) -> None:
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json in {root}")
    weights = sorted(root.glob("*.safetensors")) + sorted(root.glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"no model weight files in {root}")
    empty = [path.name for path in weights if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty model weight files in {root}: {empty}")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        referenced = set((payload.get("weight_map") or {}).values())
        if not referenced:
            raise ValueError(f"empty weight map in {index_path}")
        missing = [
            name
            for name in sorted(referenced)
            if not (root / name).is_file() or (root / name).stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(f"missing indexed weight files in {root}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    root = args.checkpoint.expanduser().resolve()
    validate_checkpoint(root)
    print(root)


if __name__ == "__main__":
    main()
