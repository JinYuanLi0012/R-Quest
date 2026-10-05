"""One-step barrier before semantic workers borrow Questioner GPUs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid


BARRIER_DATA_KEY = "validity_rzero_semantic_gpu_ready_file"


def skip_final_validation(val_freq: int) -> bool:
    """Honor disabled validation only for the RQuest novelty treatment."""
    return (
        val_freq <= 0
        and os.getenv("VALIDITY_RZERO_ENABLED", "0") == "1"
        and os.getenv("VALIDITY_RZERO_DIVERSITY_MODE", "semantic_novelty_gate")
        == "semantic_novelty_gate"
    )


def create_barrier(step: int) -> Path:
    root = Path(os.environ["STORAGE_PATH"]) / "temp_results" / "semantic_gpu_barriers"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"step_{step}_{uuid.uuid4().hex}.json"


def _write_status(path: Path, status: str, detail: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps({"status": status, "detail": detail, "timestamp": time.time()}),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def signal_ready(path: Path) -> None:
    _write_status(path, "ready")


def signal_abort(path: Path, detail: str) -> None:
    _write_status(path, "abort", detail)


def wait_until_ready(path_value: str, timeout_seconds: int | None = None) -> float:
    path = Path(path_value)
    timeout = (
        int(os.getenv("VALIDITY_RZERO_SEMANTIC_BARRIER_TIMEOUT", "900"))
        if timeout_seconds is None
        else timeout_seconds
    )
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "ready":
                return time.monotonic() - started
            if state.get("status") == "abort":
                raise RuntimeError(
                    "Questioner GPU handoff was aborted before novelty judging: "
                    f"{state.get('detail') or 'unknown trainer failure'}"
                )
            raise RuntimeError(f"invalid semantic GPU barrier state: {state!r}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for Questioner GPU handoff barrier: {path}")


def cleanup_barrier(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)
