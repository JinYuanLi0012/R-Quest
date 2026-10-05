"""Outcome-only reward for validity-aware mathematical solving.

The final boxed answer is the only action inspected.  Mentions of validity in
the reasoning are deliberately ignored.
"""

import re
from typing import Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


INVALID_TARGET = "INVALID"


def _normalise_response(response: str) -> str:
    """Match the harmless Qwen tag cleanup used by the existing math reward."""
    return re.sub(r"\s*(<|>|/)\s*", r"\1", response)


def _extract_final_answer(response: str) -> tuple[str, bool]:
    """Return mathruler's final boxed content and whether it is usable."""
    answer = extract_boxed_content(response)
    # mathruler 0.1.0 returns the string "None" when no box is present.
    format_ok = bool(answer and answer != "None")
    return answer, format_ok


def _is_invalid(answer: str) -> bool:
    return answer.strip().casefold() == INVALID_TARGET.casefold()


# Shared public helpers for validity-gated pipelines.
extract_final_answer = _extract_final_answer
is_invalid_answer = _is_invalid


def compute_score(predicts: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
    """Score a batch using the fixed validity-aware reward table.

    Rewards:
      * correct mathematical answer or correct INVALID: +1.0
      * wrong mathematical answer (including an answer to an invalid item): 0.0
      * INVALID on a valid item: -0.5
      * missing/empty boxed final answer: -0.1

    Only ``overall`` is optimized by the trainer.  The other values are binary
    diagnostics which the existing reward manager averages for logging.
    """
    if len(predicts) != len(ground_truths):
        raise ValueError("predicts and ground_truths must have the same length")

    scores: List[Dict[str, float]] = []
    for raw_predict, raw_ground_truth in zip(predicts, ground_truths):
        predict = _normalise_response(raw_predict)
        ground_truth = str(raw_ground_truth).strip()
        answer, format_ok = _extract_final_answer(predict)
        pred_invalid = format_ok and _is_invalid(answer)
        target_invalid = _is_invalid(ground_truth)

        correct = False
        false_invalid = False
        if not format_ok:
            overall = -0.1
        elif pred_invalid:
            if target_invalid:
                overall = 1.0
                correct = True
            else:
                overall = -0.5
                false_invalid = True
        elif target_invalid:
            overall = 0.0
        else:
            try:
                correct = bool(grade_answer(answer, ground_truth))
            except Exception:
                correct = False
            overall = 1.0 if correct else 0.0

        scores.append(
            {
                "overall": overall,
                "correct": float(correct),
                "reward_positive": float(overall == 1.0),
                "reward_zero": float(overall == 0.0),
                "reward_false_invalid": float(overall == -0.5),
                "reward_malformed": float(overall == -0.1),
                "pred_invalid": float(pred_invalid),
                "false_invalid": float(false_invalid),
                "format_ok": float(format_ok),
                "format_failure": float(not format_ok),
            }
        )

    return scores
