#!/usr/bin/env python3
"""Prepare the Validity-RL train and validation splits."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download


DEFAULT_DATASET = "jinyuan222/rzero-validity-rl-terra-v1-clean-v1"
DEFAULT_REVISION = "c3b930792b51d60a208a6db7d4e60051ae5b24aa"
REQUIRED_COLUMNS = {
    "id",
    "round",
    "question",
    "terra_validity",
    "canonical_final_answer",
    "answer_verified",
    "validity_rl_target",
    "split",
}


def _validate_split(name: str, split: Any) -> Dict[str, int]:
    missing = REQUIRED_COLUMNS.difference(split.column_names)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    counts = {"VALID": 0, "INVALID": 0}
    for index, row in enumerate(split):
        validity = row["terra_validity"]
        target = row["validity_rl_target"]
        if validity not in counts:
            raise ValueError(f"{name}[{index}] has unknown terra_validity={validity!r}")
        counts[validity] += 1
        if row["split"] != name:
            raise ValueError(f"{name}[{index}] carries split={row['split']!r}")
        if validity == "INVALID":
            if target != "INVALID":
                raise ValueError(f"{name}[{index}] INVALID row has target={target!r}")
        else:
            if row["answer_verified"] is not True:
                raise ValueError(f"{name}[{index}] VALID answer is not verified")
            if not row["canonical_final_answer"]:
                raise ValueError(f"{name}[{index}] VALID row has no canonical answer")
            if target != row["canonical_final_answer"]:
                raise ValueError(f"{name}[{index}] target differs from canonical answer")
    return counts


def audit_dataset(dataset_name: str) -> tuple[DatasetDict, Dict[str, Any]]:
    local = Path(dataset_name)
    dataset = DatasetDict()
    revision = DEFAULT_REVISION if dataset_name == DEFAULT_DATASET else None
    for name in ("train", "validation"):
        if local.is_dir():
            path = local / f"{name}.parquet"
            if not path.is_file():
                path = local / f"{name}.jsonl"
        else:
            path = Path(hf_hub_download(
                repo_id=dataset_name, repo_type="dataset", filename=f"{name}.jsonl",
                revision=revision,
            ))
        # Optional metadata can differ between rows or split exports. Read JSON
        # with Arrow schema inference, and do not force both splits to one schema.
        dataset[name] = load_dataset(
            "parquet" if path.suffix == ".parquet" else "json",
            data_files=str(path), split="train",
        )

    report: Dict[str, Any] = {"dataset": dataset_name, "revision": revision, "configs": ["default"], "splits": {}}
    for name, split in dataset.items():
        report["splits"][name] = {
            "rows": len(split),
            "columns": split.column_names,
            "validity_counts": _validate_split(name, split),
        }
    overlap = set(dataset["train"]["id"]) & set(dataset["validation"]["id"])
    if overlap:
        raise ValueError(f"train and validation share {len(overlap)} IDs")
    report["overlapping_split_ids"] = 0
    return dataset, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    dataset, report = audit_dataset(args.dataset)
    print(json.dumps(report, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "validation"):
            selected = dataset[split_name]
            destination = args.output_dir / f"{split_name}.parquet"
            selected.to_parquet(destination)
            print(f"wrote {len(selected)} rows to {destination}")
        (args.output_dir / "source_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
