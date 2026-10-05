#!/usr/bin/env python3
"""Atomic stage markers for the base R-Zero pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fields(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("field name cannot be empty")
        result[key] = value
    return result


def configuration_fingerprint(fields: dict[str, str]) -> str:
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def init_state(path: Path, fields: dict[str, str]) -> str:
    signature = configuration_fingerprint(fields)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("run_fingerprint") != signature:
            raise RuntimeError(
                "run configuration changed; resume with the original settings or use a new MODEL_ABBR"
            )
    else:
        _atomic_json(
            path,
            {
                "format_version": 1,
                "created_at": _now(),
                "run_fingerprint": signature,
                "configuration": fields,
                "stages": {},
            },
        )
    return signature


def stage_complete(marker: Path, signature: str, required: list[Path]) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded = [Path(path) for path in payload.get("artifacts", [])]
    return (
        payload.get("run_fingerprint") == signature
        and all(path.exists() for path in recorded)
        and all(path.exists() for path in required)
    )


def complete_stage(
    state_path: Path,
    marker: Path,
    stage: str,
    signature: str,
    artifacts: list[Path],
    metadata: dict[str, str],
) -> None:
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot complete {stage}; missing {missing}")
    value = {
        "stage": stage,
        "completed_at": _now(),
        "run_fingerprint": signature,
        "artifacts": [str(path) for path in artifacts],
        "metadata": metadata,
    }
    _atomic_json(marker, value)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_fingerprint") != signature:
        raise RuntimeError("run fingerprint changed while completing a stage")
    state.setdefault("stages", {})[stage] = value
    state["updated_at"] = _now()
    _atomic_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--field", action="append", default=[])
    check = commands.add_parser("check")
    check.add_argument("--marker", type=Path, required=True)
    check.add_argument("--fingerprint", required=True)
    check.add_argument("--require", action="append", type=Path, default=[])
    finish = commands.add_parser("complete")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--marker", type=Path, required=True)
    finish.add_argument("--stage", required=True)
    finish.add_argument("--fingerprint", required=True)
    finish.add_argument("--artifact", action="append", type=Path, default=[])
    finish.add_argument("--meta", action="append", default=[])
    args = parser.parse_args()

    if args.command == "init":
        print(init_state(args.state, _fields(args.field)))
        return 0
    if args.command == "check":
        return 0 if stage_complete(args.marker, args.fingerprint, args.require) else 1
    complete_stage(
        args.state,
        args.marker,
        args.stage,
        args.fingerprint,
        args.artifact,
        _fields(args.meta),
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"pipeline state error: {error}", file=sys.stderr)
        sys.exit(2)
