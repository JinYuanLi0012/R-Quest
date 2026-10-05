#!/usr/bin/env python3
"""Run one single-GPU shard of frozen-base semantic-pair inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Callable

from .judge import atomic_json, atomic_jsonl
from .judge import parse_response, sampling_options


def read_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def generate_with_one_retry(
    tasks: list[dict],
    generate: Callable[[list[str]], list[str]],
    parse: Callable[[str], dict],
) -> dict[str, dict]:
    """Generate all tasks and retry only first-pass strict parse failures once."""
    results, _ = generate_with_deferred_retry(
        tasks, generate, parse, max(len(tasks), 1)
    )
    return results


def generate_with_deferred_retry(
    tasks: list[dict],
    generate: Callable[[list[str]], list[str]],
    parse: Callable[[str], dict],
    batch_size: int,
) -> tuple[dict[str, dict], dict[str, int]]:
    """Run every first-pass batch before retrying failures in large batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results: dict[str, dict] = {}
    retry: list[dict] = []
    first_pass_batch_count = 0
    retry_batch_count = 0
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        first_responses = generate([item["prompt"] for item in batch])
        first_pass_batch_count += 1
        if len(first_responses) != len(batch):
            raise RuntimeError("semantic generator returned the wrong number of responses")
        for item, raw_response in zip(batch, first_responses):
            parsed = parse(raw_response)
            results[item["cache_key"]] = {**parsed, "attempts": 1}
            if parsed["parsed_label"] is None:
                retry.append(item)
    first_pass_failure_count = len(retry)
    for start in range(0, len(retry), batch_size):
        batch = retry[start:start + batch_size]
        retry_responses = generate([item["prompt"] for item in batch])
        retry_batch_count += 1
        if len(retry_responses) != len(batch):
            raise RuntimeError("semantic retry returned the wrong number of responses")
        for item, raw_response in zip(batch, retry_responses):
            results[item["cache_key"]] = {**parse(raw_response), "attempts": 2}
    return results, {
        "first_pass_batch_count": first_pass_batch_count,
        "first_pass_failure_count": first_pass_failure_count,
        "retry_batch_count": retry_batch_count,
        "retried_request_count": first_pass_failure_count,
    }


def prefix_cache_observation(requests: list[object]) -> dict[str, int]:
    """Summarize vLLM RequestOutput.num_cached_tokens when available."""
    observed_requests = 0
    prompt_tokens = 0
    hit_tokens = 0
    for request in requests:
        cached_tokens = getattr(request, "num_cached_tokens", None)
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        if cached_tokens is None or prompt_token_ids is None:
            continue
        observed_requests += 1
        prompt_tokens += len(prompt_token_ids)
        hit_tokens += int(cached_tokens)
    return {
        "observed_request_count": observed_requests,
        "observed_prompt_tokens": prompt_tokens,
        "hit_tokens": hit_tokens,
    }


def main() -> None:
    args = arguments()
    tasks = read_tasks(args.input)
    if not tasks:
        atomic_jsonl(args.output, [])
        atomic_json(args.metrics, {
            "task_count": 0,
            "model_load_seconds": 0.0,
            "generation_seconds": 0.0,
            "enable_prefix_caching": True,
            "generated_request_count": 0,
            "prefix_cache_observed_request_count": 0,
            "prefix_cache_observed_prompt_tokens": 0,
            "prefix_cache_hit_tokens": 0,
            "prefix_cache_token_hit_rate": None,
            "first_pass_batch_count": 0,
            "first_pass_failure_count": 0,
            "retry_batch_count": 0,
            "retried_request_count": 0,
        })
        return

    import vllm
    import vllm.envs as vllm_envs
    from vllm import LLM, SamplingParams

    sampling = SamplingParams(**sampling_options(args.max_tokens, args.seed))
    load_start = time.perf_counter()
    model = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enable_prefix_caching=True,
    )
    model_load_seconds = time.perf_counter() - load_start
    results: dict[str, dict] = {}
    generation_seconds = 0.0
    generated_request_count = 0
    prefix_cache_observed_request_count = 0
    prefix_cache_observed_prompt_tokens = 0
    prefix_cache_hit_tokens = 0
    def generate(prompts: list[str]) -> list[str]:
        nonlocal generation_seconds, generated_request_count
        nonlocal prefix_cache_observed_request_count
        nonlocal prefix_cache_observed_prompt_tokens, prefix_cache_hit_tokens
        generation_start = time.perf_counter()
        generated = model.generate(prompts, sampling_params=sampling, use_tqdm=True)
        generation_seconds += time.perf_counter() - generation_start
        generated_request_count += len(generated)
        observation = prefix_cache_observation(generated)
        prefix_cache_observed_request_count += observation["observed_request_count"]
        prefix_cache_observed_prompt_tokens += observation["observed_prompt_tokens"]
        prefix_cache_hit_tokens += observation["hit_tokens"]
        return [output.outputs[0].text for output in generated]

    results, retry_metrics = generate_with_deferred_retry(
        tasks, generate, parse_response, args.batch_size
    )
    rows = [{"cache_key": item["cache_key"], **results[item["cache_key"]]} for item in tasks]
    atomic_jsonl(args.output, rows)
    atomic_json(args.metrics, {
        "task_count": len(tasks),
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
        "vllm_version": vllm.__version__,
        "vllm_use_v1": bool(vllm_envs.VLLM_USE_V1),
        "enable_prefix_caching": True,
        "generated_request_count": generated_request_count,
        "prefix_cache_observed_request_count": prefix_cache_observed_request_count,
        "prefix_cache_observed_prompt_tokens": prefix_cache_observed_prompt_tokens,
        "prefix_cache_hit_tokens": prefix_cache_hit_tokens,
        "prefix_cache_token_hit_rate": (
            prefix_cache_hit_tokens / prefix_cache_observed_prompt_tokens
            if prefix_cache_observed_prompt_tokens else None
        ),
        **retry_metrics,
        "final_parse_success_count": sum(row["parsed_label"] is not None for row in rows),
        "final_parse_failure_count": sum(row["parsed_label"] is None for row in rows),
    })


if __name__ == "__main__":
    main()
