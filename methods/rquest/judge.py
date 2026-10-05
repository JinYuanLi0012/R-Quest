"""Prompt, parsing, and file utilities for the frozen novelty judge."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

PROMPT_VERSION = "rquest-novelty-v1"


PROMPT_TEMPLATE = (
    "You are judging whether two generated math problems are repetitions of the\n"
    "same recurring exercise pattern.\n\n"
    "Choose SAME_TYPE when the two problems have essentially the same distinctive\n"
    "mathematical setup and ask the same kind of task, so that they feel like\n"
    "variations of the same exercise.\n\n"
    "Differences in constants, coefficients, variables, formulas, bounds, or other\n"
    "local details do not by themselves make the problems different.\n\n"
    "Do not choose SAME_TYPE merely because the problems share a broad topic,\n"
    "similar wording, presentation style.\n\n"
    "If the common pattern is not clear and specific, choose DIFFERENT.\n\n"
    "Briefly compare the exercise pattern of the two problems, then end with exactly\n\n"
    "\\boxed{SAME_TYPE}\n\n"
    "or\n\n"
    "\\boxed{DIFFERENT}.\n\n"
    "Question A:\n{question_a}\n\n"
    "Question B:\n{question_b}\n\n"
    "Analysis:"
)


def build_prompt(question_a: str, question_b: str) -> str:
    return PROMPT_TEMPLATE.replace("{question_a}", question_a).replace(
        "{question_b}", question_b
    )

DEFAULT_MAX_TOKENS = 1024


STOP_STRINGS = (r"\boxed{SAME_TYPE}", r"\boxed{DIFFERENT}")


SEMANTIC_BOX = re.compile(
    r"\\boxed\s*\{\s*(?:"
    r"(?P<plain>SAME(?:\\)?_TYPE|DIFFERENT)"
    r"|\\text\s*\{\s*(?P<text>SAME(?:\\)?_TYPE|DIFFERENT)\s*\}"
    r")\s*\}"
)


def parse_response(raw_response: str) -> dict[str, str | None]:
    """Accept boxed LaTeX variants, but never infer a label from ordinary prose."""
    labels = []
    for match in SEMANTIC_BOX.finditer(raw_response):
        label = (match.group("plain") or match.group("text")).replace(r"\_", "_")
        labels.append(label)
    unique = set(labels)
    if not unique:
        return {
            "predicted_label": "FORMAT_ERROR",
            "parsed_label": None,
            "format_status": "error",
            "format_error_reason": "missing_boxed_label",
        }
    if len(unique) > 1:
        return {
            "predicted_label": "FORMAT_ERROR",
            "parsed_label": None,
            "format_status": "error",
            "format_error_reason": "conflicting_boxed_labels",
        }
    label = next(iter(unique))
    return {
        "predicted_label": label,
        "parsed_label": label,
        "format_status": "ok",
        "format_error_reason": None,
    }


def sampling_options(max_tokens: int, seed: int) -> dict[str, Any]:
    """Qwen3 official anti-repetition sampling plus task-specific stop strings."""
    return {
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stop": list(STOP_STRINGS),
        "include_stop_str_in_output": True,
        "skip_special_tokens": True,
    }


def load_generation_config(
    model: str, revision: str | None, local_files_only: bool,
) -> tuple[dict[str, Any], str, str, str | None]:
    model_path = Path(model)
    if model_path.is_dir():
        config_path = model_path / "generation_config.json"
    else:
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(
            repo_id=model,
            filename="generation_config.json",
            revision=revision,
            local_files_only=local_files_only,
        ))
    if not config_path.is_file():
        raise FileNotFoundError(f"generation_config.json not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_revision = revision
    parts = config_path.parts
    if "snapshots" in parts:
        snapshot_index = parts.index("snapshots")
        if snapshot_index + 1 < len(parts):
            resolved_revision = parts[snapshot_index + 1]
    return (
        payload,
        str(config_path.resolve()),
        str(config_path.parent.resolve()),
        resolved_revision,
    )

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
