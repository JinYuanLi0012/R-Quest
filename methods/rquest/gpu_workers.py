"""Launch evenly sharded single-GPU frozen-base semantic workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable

from .judge import atomic_jsonl
from .novelty import UniquePairTask


def shard_tasks_by_candidate(
    tasks: list[UniquePairTask], shard_count: int
) -> list[list[UniquePairTask]]:
    """Keep every candidate's references contiguous and on one worker."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    groups: list[list[UniquePairTask]] = []
    group_positions: dict[tuple[str, int], int] = {}
    for task_position, task in enumerate(tasks):
        group_key = (
            ("candidate", task.candidate_index)
            if task.candidate_index is not None
            else ("ungrouped", task_position)
        )
        if group_key not in group_positions:
            group_positions[group_key] = len(groups)
            groups.append([])
        groups[group_positions[group_key]].append(task)
    shards = [[] for _ in range(shard_count)]
    for group_position, group in enumerate(groups):
        shards[group_position % shard_count].extend(group)
    return shards


def aggregate_prefix_cache_metrics(workers: list[dict]) -> dict:
    request_count = sum(int(item.get("generated_request_count", 0)) for item in workers)
    observed_requests = sum(
        int(item.get("prefix_cache_observed_request_count", 0)) for item in workers
    )
    prompt_tokens = sum(
        int(item.get("prefix_cache_observed_prompt_tokens", 0)) for item in workers
    )
    cached_tokens = sum(
        int(item.get("prefix_cache_hit_tokens", 0)) for item in workers
    )
    vllm_versions = sorted({
        str(item["vllm_version"])
        for item in workers
        if item.get("vllm_version") is not None
    })
    engine_modes = sorted({
        str(item["vllm_use_v1"])
        for item in workers
        if item.get("vllm_use_v1") is not None
    })
    return {
        "enabled_explicitly": True,
        "vllm_versions": vllm_versions,
        "vllm_use_v1_values": engine_modes,
        "generated_request_count_including_retries": request_count,
        "observed_request_count": observed_requests,
        "metrics_available_for_all_generated_requests": (
            observed_requests == request_count if request_count else None
        ),
        "observed_prompt_tokens": prompt_tokens,
        "hit_tokens": cached_tokens,
        "token_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else None,
    }


def aggregate_retry_metrics(workers: list[dict]) -> dict[str, int]:
    return {
        key: sum(int(item.get(key, 0)) for item in workers)
        for key in (
            "first_pass_batch_count",
            "first_pass_failure_count",
            "retry_batch_count",
            "retried_request_count",
        )
    }


def terminate_process_groups(processes: Iterable[subprocess.Popen]) -> None:
    processes = list(processes)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)


def run_gpu_tasks(
    tasks: list[UniquePairTask],
    model: str,
    gpu_ids: list[str],
    work_dir: Path,
    max_tokens: int = 1024,
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
    batch_size: int = 8192,
) -> tuple[dict[str, dict], dict]:
    if not gpu_ids:
        raise ValueError("at least one semantic GPU is required")
    if not tasks:
        return {}, {
            "wall_seconds": 0.0,
            "workers": [],
            "prefix_cache": aggregate_prefix_cache_metrics([]),
            "retry": aggregate_retry_metrics([]),
        }
    work_dir.mkdir(parents=True, exist_ok=True)
    shards = shard_tasks_by_candidate(tasks, len(gpu_ids))
    processes: list[subprocess.Popen] = []
    pid_file_value = os.getenv("VALIDITY_RZERO_SEMANTIC_PID_FILE")
    pid_file = Path(pid_file_value) if pid_file_value else None
    output_paths: list[Path] = []
    metric_paths: list[Path] = []
    wall_start = time.perf_counter()
    try:
        for shard_index, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
            input_path = work_dir / f"semantic_tasks_{shard_index}.jsonl"
            output_path = work_dir / f"semantic_results_{shard_index}.jsonl"
            metric_path = work_dir / f"semantic_metrics_{shard_index}.json"
            atomic_jsonl(input_path, (
                {
                    "cache_key": task.cache_key,
                    "candidate_index": task.candidate_index,
                    "panel_index": task.panel_index,
                    "prompt": task.prompt,
                }
                for task in shard
            ))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            command = [
                sys.executable, "-m", "methods.rquest.judge_worker",
                "--input", str(input_path), "--output", str(output_path),
                "--metrics", str(metric_path), "--model", model,
                "--max-tokens", str(max_tokens), "--seed", str(seed),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--batch-size", str(batch_size),
            ]
            processes.append(subprocess.Popen(command, env=env, start_new_session=True))
            if pid_file is not None:
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                pid_file.write_text(
                    "".join(f"{process.pid}\n" for process in processes), encoding="utf-8"
                )
            output_paths.append(output_path)
            metric_paths.append(metric_path)
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"semantic GPU worker failures: {return_codes}")
    except BaseException:
        terminate_process_groups(processes)
        if pid_file is not None:
            pid_file.write_text("", encoding="utf-8")
        raise
    if pid_file is not None:
        pid_file.write_text("", encoding="utf-8")
    wall_seconds = time.perf_counter() - wall_start
    judgments: dict[str, dict] = {}
    worker_metrics = []
    for output_path, metric_path in zip(output_paths, metric_paths):
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    judgments[row["cache_key"]] = row
        worker_metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
    if len(judgments) != len(tasks):
        raise RuntimeError(f"semantic result coverage mismatch: {len(judgments)} vs {len(tasks)}")
    return judgments, {
        "wall_seconds": wall_seconds,
        "workers": worker_metrics,
        "prefix_cache": aggregate_prefix_cache_metrics(worker_metrics),
        "retry": aggregate_retry_metrics(worker_metrics),
    }
