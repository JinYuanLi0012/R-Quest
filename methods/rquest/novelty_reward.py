"""Compute novelty feedback using the frozen Base model."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from .judge import PROMPT_TEMPLATE, PROMPT_VERSION, build_prompt, load_generation_config
from .novelty import cache_context, aggregate_novelty, build_novelty_pair_plan
from .gpu_workers import run_gpu_tasks
from .gpu_barrier import wait_until_ready
from .solver_service import SolverServiceConfig, semantic_gpu_handoff

_RESOLVED_MODEL: tuple[str, str | None] | None = None


def resolve_frozen_model() -> tuple[str, str | None]:
    global _RESOLVED_MODEL
    if _RESOLVED_MODEL is None:
        model = os.getenv("VALIDITY_RZERO_SEMANTIC_MODEL", "Qwen/Qwen3-4B-Base")
        local_only = os.getenv("VALIDITY_RZERO_SEMANTIC_LOCAL_FILES_ONLY", "1") == "1"
        _, _, resolved_path, revision = load_generation_config(model, None, local_only)
        _RESOLVED_MODEL = resolved_path, revision
        print(
            "[rquest][novelty] "
            f"resolved_frozen_snapshot={resolved_path} revision={revision}"
        )
    return _RESOLVED_MODEL


def _semantic_gpu_ids(service: SolverServiceConfig) -> list[str]:
    value = os.getenv("VALIDITY_RZERO_SEMANTIC_GPU_IDS", ",".join(service.gpu_ids))
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("VALIDITY_RZERO_SEMANTIC_GPU_IDS must contain at least one GPU")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"duplicate semantic GPU IDs are not allowed: {gpu_ids}")
    return gpu_ids

def compute_online_novelty(
    questions: list[str], gpu_ready_file: str | None = None
) -> list[dict[str, int]]:
    candidate_indices = [index for index, question in enumerate(questions) if question]
    output = [
        {"same_count": 0, "compared_count": 0, "parse_failure_count": 0, "novelty": 1}
        for _ in questions
    ]
    if not candidate_indices:
        print("[rquest][semantic_novelty_gate][WARNING] no valid questions; novelty=1")
        return output
    novelty_k = int(os.getenv("VALIDITY_RZERO_NOVELTY_K", "8"))
    novelty_seed = int(os.getenv("VALIDITY_RZERO_NOVELTY_SEED", "43"))
    if novelty_k < 0:
        raise ValueError("VALIDITY_RZERO_NOVELTY_K must be nonnegative")
    resolved_model, _ = resolve_frozen_model()
    context = cache_context(
        resolved_model,
        1024,
        42,
        prompt_version=PROMPT_VERSION,
        prompt_template=PROMPT_TEMPLATE,
        orientation="candidate_then_reference_v1",
    )
    question_map = {index: questions[index] for index in candidate_indices}
    references, instances, tasks = build_novelty_pair_plan(
        question_map,
        candidate_indices,
        novelty_k,
        novelty_seed,
        context,
        prompt_builder=build_prompt,
    )
    service = SolverServiceConfig.from_environment()
    semantic_gpu_ids = _semantic_gpu_ids(service)
    temp_root = Path(os.environ["STORAGE_PATH"]) / "temp_results"
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="semantic_novelty_gate_online_", dir=temp_root))
    try:
        barrier_wait_seconds = 0.0
        if gpu_ready_file:
            barrier_wait_seconds = wait_until_ready(gpu_ready_file)
            print(
                "[rquest][semantic_novelty_gate] Questioner GPU barrier ready "
                f"after {barrier_wait_seconds:.3f}s; semantic_gpus={','.join(semantic_gpu_ids)}"
            )
        with semantic_gpu_handoff(service):
            judgments, runtime = run_gpu_tasks(
                list(tasks.values()),
                resolved_model,
                semantic_gpu_ids,
                work_dir,
                max_tokens=1024,
                seed=42,
                gpu_memory_utilization=float(
                    os.getenv("VALIDITY_RZERO_SEMANTIC_GPU_MEMORY_UTILIZATION", "0.80")
                ),
                batch_size=int(
                    os.getenv("VALIDITY_RZERO_SEMANTIC_WORKER_BATCH_SIZE", "8192")
                ),
            )
        aggregates = aggregate_novelty(candidate_indices, instances, judgments)
        compared = sum(int(item["compared_count"]) for item in aggregates.values())
        failures = sum(int(item["parse_failure_count"]) for item in aggregates.values())
        total = compared + failures
        retry = runtime["retry"]
        prefix_cache = runtime["prefix_cache"]
        prefix_hit_rate = prefix_cache["token_hit_rate"]
        sampled_counts = [len(references[index]) for index in candidate_indices]
        print(
            "[rquest][semantic_novelty_gate] "
            f"prompt_version={PROMPT_VERSION} novelty_k={novelty_k} novelty_seed={novelty_seed} "
            f"references_per_candidate_min={min(sampled_counts)} "
            f"references_per_candidate_max={max(sampled_counts)} "
            f"semantic_gpus={','.join(semantic_gpu_ids)} "
            f"barrier_wait_seconds={barrier_wait_seconds:.3f} "
            f"pair_instances={len(instances)} unique_pairs={len(tasks)} "
            f"cache_hits={len(instances) - len(tasks)} "
            f"vllm_wall_seconds={runtime['wall_seconds']:.3f} "
            f"parse_success_after_retry={(compared / total if total else 1.0):.6f} "
            f"parse_failure_after_retry={(failures / total if total else 0.0):.6f}"
        )
        print(
            "[rquest][semantic_novelty_gate][efficiency] "
            f"first_pass_batches={retry['first_pass_batch_count']} "
            f"first_pass_parse_failures={retry['first_pass_failure_count']} "
            f"deferred_retry_batches={retry['retry_batch_count']} "
            f"retried_requests={retry['retried_request_count']} "
            f"prefix_cache_enabled={prefix_cache['enabled_explicitly']} "
            f"prefix_cache_token_hit_rate="
            f"{(f'{prefix_hit_rate:.6f}' if prefix_hit_rate is not None else 'unavailable')}"
        )
        for index in candidate_indices:
            output[index] = aggregates[index]
        return output
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
