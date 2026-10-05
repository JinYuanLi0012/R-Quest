"""Pure validity-vote logic shared by both R-Zero phases."""

from __future__ import annotations

from typing import Any, Iterable

from methods.validity_rl.validity_reward import extract_final_answer, is_invalid_answer


VALIDITY_VOTES = 9
INVALID_MAJORITY = 5
PHASE_A_MATH_VOTES = 10
PHASE_B_MATH_VOTES = 9


def evaluate_validity_responses(responses: Iterable[str]) -> dict[str, Any]:
    """Classify a problem from exactly nine validity-aware solver responses."""
    responses = list(responses)
    if len(responses) != VALIDITY_VOTES:
        raise ValueError(f"expected {VALIDITY_VOTES} validity responses, got {len(responses)}")

    extracted_answers: list[str] = []
    format_ok: list[bool] = []
    invalid_votes = 0
    for response in responses:
        answer, usable = extract_final_answer(response)
        extracted_answers.append(answer if usable else "")
        format_ok.append(usable)
        invalid_votes += int(usable and is_invalid_answer(answer))

    is_invalid = invalid_votes >= INVALID_MAJORITY
    return {
        "invalid_votes": invalid_votes,
        "total_votes": VALIDITY_VOTES,
        "validity_decision": "INVALID" if is_invalid else "VALID",
        "validity_penalty": 0.5 - invalid_votes / VALIDITY_VOTES if is_invalid else 0.0,
        "validity_outputs": extracted_answers,
        "validity_format_failures": format_ok.count(False),
    }


def valid_positions(gates: Iterable[dict[str, Any]]) -> list[int]:
    """Return only rows allowed to enter the fresh pure-math stage."""
    return [
        index for index, gate in enumerate(gates)
        if gate["validity_decision"] == "VALID"
    ]
