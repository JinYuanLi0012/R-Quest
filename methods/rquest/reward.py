"""Dispatch Solver rewards by the source of each mixed-dataset row."""

from __future__ import annotations

from typing import Any, Dict, List

from examples.reward_function.math import compute_score as math_compute_score
from methods.validity_rl.validity_reward import compute_score as validity_compute_score


def compute_score(
    predicts: List[str], ground_truths: List[str], source: List[str]
) -> List[Dict[str, float]]:
    if not (len(predicts) == len(ground_truths) == len(source)):
        raise ValueError("predicts, ground_truths, and source must have the same length")
    unknown = sorted(set(source).difference({"rzero", "terra"}))
    if unknown:
        raise ValueError(f"unknown Solver sample source: {unknown}")

    scores: list[dict[str, Any] | None] = [None] * len(predicts)
    for name, scorer in (("rzero", math_compute_score), ("terra", validity_compute_score)):
        indices = [index for index, value in enumerate(source) if value == name]
        if not indices:
            continue
        subset = scorer(
            [predicts[index] for index in indices],
            [ground_truths[index] for index in indices],
        )
        for index, score in zip(indices, subset):
            scores[index] = dict(score)

    if any(score is None for score in scores):
        raise RuntimeError("a mixed Solver reward was not assigned")
    resolved = [score for score in scores if score is not None]
    rzero_values = [score["overall"] for score, name in zip(resolved, source) if name == "rzero"]
    terra_values = [score["overall"] for score, name in zip(resolved, source) if name == "terra"]
    diagnostics = {
        "rzero_count": float(len(rzero_values)),
        "terra_count": float(len(terra_values)),
        "rzero_reward_mean": sum(rzero_values) / len(rzero_values) if rzero_values else 0.0,
        "terra_reward_mean": sum(terra_values) / len(terra_values) if terra_values else 0.0,
        "actual_replay_ratio": len(terra_values) / len(resolved) if resolved else 0.0,
    }
    print(
        "[rquest][mixed_reward] "
        + " ".join(f"{key}={value}" for key, value in diagnostics.items())
    )
    output = []
    for score, name in zip(resolved, source):
        output.append({
            **score,
            "source_rzero": float(name == "rzero"),
            "source_terra": float(name == "terra"),
            **diagnostics,
        })
    return output
