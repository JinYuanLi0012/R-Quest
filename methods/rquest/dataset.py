#!/usr/bin/env python3
"""Build one round's source-tagged R-Zero + Terra Solver dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable


def passes_rzero_filter(row: dict[str, Any], min_score: float, max_score: float) -> bool:
    return (
        not row.get("discarded_by_validity", False)
        and isinstance(row.get("score"), (int, float))
        and min_score <= float(row["score"]) <= max_score
        and row.get("answer") not in (None, "", "None")
    )


def require_full_training_batch(row_count: int, rollout_batch_size: int) -> None:
    if row_count < rollout_batch_size:
        raise RuntimeError(
            f"mixed Solver dataset has {row_count} rows, fewer than one complete "
            f"training batch ({rollout_batch_size}); increase SOLVER_GENERATE_SAMPLES"
        )


def build_mixed_rows(
    evaluated_rows: Iterable[dict[str, Any]],
    terra_train_rows: Iterable[dict[str, Any]],
    min_score: float,
    max_score: float,
    replay_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in (0, 1)")

    evaluated_rows = list(evaluated_rows)
    rzero_rows = [
        {
            "problem": row["question"],
            "answer": row["answer"],
            "score": float(row["score"]),
            "source": "rzero",
            "sample_id": f"rzero-{index}",
        }
        for index, row in enumerate(evaluated_rows)
        if passes_rzero_filter(row, min_score, max_score)
    ]
    if not rzero_rows:
        raise ValueError("validity and difficulty filtering produced no R-Zero rows")

    terra_rows = list(terra_train_rows)
    for index, row in enumerate(terra_rows):
        if row.get("split") != "train":
            raise ValueError(f"Terra replay row {index} is not from the train split")
        if row.get("terra_validity") not in {"VALID", "INVALID"}:
            raise ValueError(f"Terra replay row {index} has invalid terra_validity")
        if not row.get("question") or not row.get("validity_rl_target"):
            raise ValueError(f"Terra replay row {index} is missing question or target")
        if row["terra_validity"] == "INVALID" and row["validity_rl_target"] != "INVALID":
            raise ValueError(f"Terra replay row {index} has an invalid INVALID target")
        if row["terra_validity"] == "VALID" and (
            row.get("answer_verified") is not True
            or row["validity_rl_target"] != row.get("canonical_final_answer")
        ):
            raise ValueError(f"Terra replay row {index} has an unverified VALID target")

    terra_count = math.floor(len(rzero_rows) * replay_ratio / (1.0 - replay_ratio))
    if terra_count == 0:
        raise ValueError("replay ratio and R-Zero row count produce zero Terra replay rows")
    if terra_count > len(terra_rows):
        raise ValueError(
            f"replay ratio requires {terra_count} Terra rows, but train split has {len(terra_rows)}"
        )
    rng = random.Random(seed)
    selected = rng.sample(terra_rows, terra_count)
    replay_rows = [
        {
            "problem": row["question"],
            "answer": row["validity_rl_target"],
            "score": None,
            "source": "terra",
            "sample_id": str(row.get("id") or f"terra-{index}"),
        }
        for index, row in enumerate(selected)
    ]
    mixed = rzero_rows + replay_rows
    rng.shuffle(mixed)
    stats = {
        "evaluated_candidate_count": len(evaluated_rows),
        "discarded_by_validity_count": sum(
            bool(row.get("discarded_by_validity", False)) for row in evaluated_rows
        ),
        "rzero_sample_count": len(rzero_rows),
        "terra_replay_sample_count": len(replay_rows),
        "mixed_sample_count": len(mixed),
        "requested_replay_ratio": replay_ratio,
        "actual_replay_ratio": len(replay_rows) / len(mixed),
    }
    return mixed, stats


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--min-score", type=float, required=True)
    parser.add_argument("--max-score", type=float, required=True)
    parser.add_argument("--terra-dataset", required=True)
    parser.add_argument("--terra-config", default="default")
    parser.add_argument("--replay-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-train-rows", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    from datasets import Dataset, load_dataset

    storage_path = os.environ["STORAGE_PATH"]
    evaluated: list[dict[str, Any]] = []
    result_paths = []
    for shard in range(args.num_shards):
        path = Path(storage_path) / "generated_question" / f"{args.experiment_name}_{shard}_results.json"
        result_paths.append(path)
        evaluated.extend(json.loads(path.read_text(encoding="utf-8")))

    if Path(args.terra_dataset).is_dir():
        terra_train = load_dataset("parquet", data_files=str(Path(args.terra_dataset) / "train.parquet"), split="train")
    else:
        terra_train = load_dataset(args.terra_dataset, args.terra_config, split="train")
    mixed, stats = build_mixed_rows(
        evaluated, terra_train, args.min_score, args.max_score, args.replay_ratio, args.seed
    )
    for row in evaluated:
        row["passed_rzero_filter"] = passes_rzero_filter(row, args.min_score, args.max_score)
    require_full_training_batch(len(mixed), args.min_train_rows)

    dataset_path = args.receipt.with_suffix(".parquet").resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dataset_path.with_suffix(".parquet.tmp")
    Dataset.from_list(mixed).to_parquet(str(temporary))
    os.replace(temporary, dataset_path)
    audit_path = args.receipt.with_name(f"{args.receipt.stem}_phase_b.jsonl")
    _write_jsonl(audit_path, evaluated)
    _atomic_json(args.receipt, {
        "dataset_path": str(dataset_path),
        "config_name": args.experiment_name,
        "filtered_count": len(mixed),
        "num_shards": args.num_shards,
        "min_score": args.min_score,
        "max_score": args.max_score,
        "terra_dataset": args.terra_dataset,
        "terra_config": args.terra_config,
        "replay_seed": args.seed,
        "phase_b_audit": str(audit_path),
        **stats,
    })
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
