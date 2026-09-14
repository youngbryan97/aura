"""Versioned semantic grading; historical exact-wire scores stay replayable."""

from __future__ import annotations

from typing import Any

from core.reasoning.asymptotic import same_growth_class

LEGACY_GRADING_POLICY = "exact_wire_v1"
SEMANTIC_GRADING_POLICY = "semantic_values_v2"


def grade_semantic_task(task: Any, response: str, *, policy: str) -> dict[str, Any]:
    if policy not in {LEGACY_GRADING_POLICY, SEMANTIC_GRADING_POLICY}:
        raise ValueError("unknown semantic grading policy")
    verdict = task.grade(response)
    if not isinstance(verdict, dict) or type(verdict.get("correct")) is not bool:
        raise RuntimeError("semantic grader returned an invalid verdict")
    if policy == LEGACY_GRADING_POLICY or task.family != "frontier_coding":
        return verdict
    answer, expected = verdict.get("parsed"), verdict.get("expected")
    if not isinstance(answer, dict) or not isinstance(expected, dict):
        return verdict
    # Reuse the strict task grader for every structural/type/value constraint.
    # Only the asymptotic field admits equivalent mathematical notation.
    if set(answer) != set(expected) or not same_growth_class(
        answer.get("time_complexity"), expected.get("time_complexity"),
    ):
        return verdict
    import json

    canonical = {**answer, "time_complexity": expected["time_complexity"]}
    checked = task.grade("FINAL_ANSWER: " + json.dumps(canonical))
    return {**verdict, "correct": checked["correct"]}
