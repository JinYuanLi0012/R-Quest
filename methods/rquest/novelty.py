"""K8 reference sampling, pair planning, and novelty rewards."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Callable, Iterable, Mapping

from .judge import PROMPT_TEMPLATE, PROMPT_VERSION, build_prompt, sampling_options

SEMANTIC_LABELS = {"SAME_TYPE", "DIFFERENT"}


@dataclass(frozen=True)
class PairInstance:
    candidate_index: int
    panel_index: int
    cache_key: str


@dataclass(frozen=True)
class UniquePairTask:
    cache_key: str
    question_a: str
    question_b: str
    prompt: str
    candidate_index: int | None = None
    panel_index: int | None = None


def cache_context(
    model_identity: str,
    max_tokens: int = 1024,
    seed: int = 42,
    *,
    prompt_version: str = PROMPT_VERSION,
    prompt_template: str = PROMPT_TEMPLATE,
    orientation: str = "lexicographic_question_text_v1",
) -> dict[str, Any]:
    if orientation not in {
        "lexicographic_question_text_v1",
        "candidate_then_reference_v1",
    }:
        raise ValueError(f"unsupported semantic pair orientation: {orientation}")
    return {
        "model_identity": model_identity,
        "prompt_version": prompt_version,
        "prompt_template": prompt_template,
        "sampling": sampling_options(max_tokens, seed),
        "orientation": orientation,
    }


def _stable_orientation(question_left: str, question_right: str) -> tuple[str, str]:
    return (
        (question_left, question_right)
        if question_left <= question_right
        else (question_right, question_left)
    )


def _cache_key(context_digest: str, question_a: str, question_b: str) -> str:
    payload = json.dumps(
        {"context": context_digest, "question_a": question_a, "question_b": question_b},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pair_plan(
    questions: Mapping[int, str],
    candidate_indices: Iterable[int],
    panel_indices: Iterable[int],
    context: Mapping[str, Any],
    *,
    prompt_builder: Callable[[str, str], str] = build_prompt,
) -> tuple[list[PairInstance], dict[str, UniquePairTask]]:
    candidate_indices = list(candidate_indices)
    panel_indices = list(panel_indices)
    context_digest = hashlib.sha256(
        json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    instances: list[PairInstance] = []
    tasks: dict[str, UniquePairTask] = {}
    for candidate_index in candidate_indices:
        candidate_question = questions[candidate_index]
        for panel_index in panel_indices:
            if candidate_index == panel_index:
                continue
            panel_question = questions[panel_index]
            if context["orientation"] == "candidate_then_reference_v1":
                question_a, question_b = candidate_question, panel_question
            else:
                question_a, question_b = _stable_orientation(
                    candidate_question, panel_question
                )
            key = _cache_key(context_digest, question_a, question_b)
            instances.append(PairInstance(candidate_index, panel_index, key))
            tasks.setdefault(
                key,
                UniquePairTask(
                    key,
                    question_a,
                    question_b,
                    prompt_builder(question_a, question_b),
                    candidate_index,
                    panel_index,
                ),
            )
    return instances, tasks

def sample_references_per_candidate(
    candidate_indices: Iterable[int], k: int, seed: int
) -> dict[int, list[int]]:
    if k < 0:
        raise ValueError("novelty K must be nonnegative")
    indices = list(candidate_indices)
    references: dict[int, list[int]] = {}
    for candidate_index in indices:
        population = [index for index in indices if index != candidate_index]
        sample_size = min(k, len(population))
        digest = hashlib.sha256(f"{seed}:{candidate_index}".encode("utf-8")).digest()
        candidate_seed = int.from_bytes(digest[:8], "big")
        references[candidate_index] = random.Random(candidate_seed).sample(
            population, sample_size
        )
    return references


def build_novelty_pair_plan(
    questions: Mapping[int, str],
    candidate_indices: Iterable[int],
    k: int,
    seed: int,
    context: Mapping[str, Any],
    *,
    prompt_builder: Callable[[str, str], str],
) -> tuple[
    dict[int, list[int]],
    list[PairInstance],
    dict[str, UniquePairTask],
]:
    candidate_indices = list(candidate_indices)
    references = sample_references_per_candidate(candidate_indices, k, seed)
    instances: list[PairInstance] = []
    tasks: dict[str, UniquePairTask] = {}
    for candidate_index in candidate_indices:
        candidate_instances, candidate_tasks = build_pair_plan(
            questions,
            [candidate_index],
            references[candidate_index],
            context,
            prompt_builder=prompt_builder,
        )
        instances.extend(candidate_instances)
        tasks.update(candidate_tasks)
    return references, instances, tasks


def aggregate_novelty(
    candidate_indices: Iterable[int],
    instances: Iterable[PairInstance],
    judgments: Mapping[str, Mapping[str, Any]],
) -> dict[int, dict[str, int]]:
    aggregates = {
        index: {
            "same_count": 0,
            "compared_count": 0,
            "parse_failure_count": 0,
            "novelty": 1,
        }
        for index in candidate_indices
    }
    for instance in instances:
        label = judgments.get(instance.cache_key, {}).get("parsed_label")
        item = aggregates[instance.candidate_index]
        if label not in SEMANTIC_LABELS:
            item["parse_failure_count"] += 1
            continue
        item["compared_count"] += 1
        item["same_count"] += int(label == "SAME_TYPE")
    for item in aggregates.values():
        item["novelty"] = int(item["same_count"] == 0)
    return aggregates


def novelty_training_diagnostics(
    final_results: list[Mapping[str, Any]],
    novelty_stats: list[Mapping[str, Any]],
    group_ids: Iterable[Any] | None,
) -> dict[str, float]:
    if len(final_results) != len(novelty_stats):
        raise ValueError("novelty diagnostics length mismatch")
    total = len(final_results)
    valid = [
        index
        for index, item in enumerate(final_results)
        if item.get("validity_decision") == "VALID"
    ]
    survivors = {
        index
        for index in valid
        if int(novelty_stats[index].get("novelty", 1)) == 1
    }
    semantic_indices = [
        index for index, item in enumerate(final_results) if item.get("question")
    ]
    compared = sum(int(novelty_stats[index].get("compared_count", 0)) for index in semantic_indices)
    failures = sum(
        int(novelty_stats[index].get("parse_failure_count", 0))
        for index in semantic_indices
    )
    diagnostics = {
        "validity_pass_rate": len(valid) / total if total else 0.0,
        "novelty_pass_rate_among_valid": (
            len(survivors) / len(valid) if valid else 0.0
        ),
        "valid_and_novel_pass_rate": len(survivors) / total if total else 0.0,
        "mean_same_hits_among_k": (
            sum(int(novelty_stats[index].get("same_count", 0)) for index in semantic_indices)
            / len(semantic_indices)
            if semantic_indices
            else 0.0
        ),
        "semantic_parse_failure_rate": (
            failures / (compared + failures) if compared + failures else 0.0
        ),
    }
    if group_ids is None:
        return diagnostics
    group_ids = list(group_ids)
    if len(group_ids) != total:
        raise ValueError("GRPO group identifier length mismatch")
    survivor_counts: dict[str, int] = defaultdict(int)
    for index, group_id in enumerate(group_ids):
        key = str(group_id)
        survivor_counts.setdefault(key, 0)
        survivor_counts[key] += int(index in survivors)
    counts = list(survivor_counts.values())
    group_count = len(counts)
    diagnostics.update({
        "mean_survivors_per_grpo_group": (
            sum(counts) / group_count if group_count else 0.0
        ),
        "zero_survivor_grpo_group_rate": (
            sum(count == 0 for count in counts) / group_count if group_count else 0.0
        ),
        "one_survivor_grpo_group_rate": (
            sum(count == 1 for count in counts) / group_count if group_count else 0.0
        ),
        "multi_survivor_grpo_group_rate": (
            sum(count >= 2 for count in counts) / group_count if group_count else 0.0
        ),
    })
    return diagnostics
