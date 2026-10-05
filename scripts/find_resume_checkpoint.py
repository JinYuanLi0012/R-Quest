#!/usr/bin/env python3
"""Return the latest complete checkpoint atomically committed by verl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty resume-state file: {path}")


def find_complete_checkpoint(root: Path, world_size: int) -> Path | None:
    tracker = root / "latest_global_step.txt"
    if not tracker.is_file():
        return None
    raw_step = tracker.read_text(encoding="utf-8").strip()
    if not raw_step.isdigit() or int(raw_step) <= 0:
        raise ValueError(f"invalid checkpoint tracker: {tracker}")

    checkpoint = root / f"global_step_{int(raw_step)}"
    require_nonempty(checkpoint / "dataloader.pt")
    actor = checkpoint / "actor"
    for rank in range(world_size):
        require_nonempty(actor / f"model_world_size_{world_size}_rank_{rank}.pt")
        require_nonempty(actor / f"optim_world_size_{world_size}_rank_{rank}.pt")
        require_nonempty(actor / f"extra_state_world_size_{world_size}_rank_{rank}.pt")
    return checkpoint.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    checkpoint = find_complete_checkpoint(args.root, args.world_size)
    if checkpoint is None:
        return 1
    print(checkpoint)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"resume checkpoint error: {error}", file=sys.stderr)
        sys.exit(2)
