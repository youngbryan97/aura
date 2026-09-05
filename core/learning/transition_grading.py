"""Transition-level grading for long-horizon trajectories.

Long-horizon reliability is multiplicative: at 99% per consequential step a
100-step project completes ~37% of the time; at 99.9% it completes ~90%.
Training only on final success therefore under-credits exactly the behaviors
that move total completion — checkpointing, precondition checks,
expected-versus-observed comparison, recovery-point placement, plan repair,
rollback, and preservation of completed valid work.

This module grades EVERY state transition of a trajectory on those named
dimensions and prices each transition's training weight from its grade, so a
trajectory becomes a stream of scored supervision rather than one terminal
bit. The failed transitions it identifies are the intake queue for the
on-policy repair protocol (core/learning/on_policy_repair.py).

Grading is deterministic and honestly lexical where it compares text: the
expected-versus-observed check reports its method and never pretends to
semantic understanding.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

TRANSITION_GRADING_SCHEMA = "aura.transition_grading.v1"

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# Named grade dimensions, each in [0, 1], combined by fixed weights that sum
# to 1. The weights encode what long-horizon reliability actually needs:
# verified outcomes and honest prediction dominate; hygiene follows.
_DIMENSION_WEIGHTS = {
    "verification": 0.30,
    "prediction": 0.25,
    "preconditions": 0.15,
    "recovery_placement": 0.15,
    "work_preservation": 0.15,
}


@dataclass(frozen=True)
class Transition:
    """One consequential step of a long-horizon trajectory."""

    index: int
    action: str
    state_digest: str = ""
    preconditions_checked: bool = False
    expected_effect: str = ""
    observed_effect: str = ""
    consequential: bool = True
    reversible: bool = True
    checkpoint_created: bool = False
    verified_outcome: bool | None = None
    preserved_completed_work: bool = True
    recovered_from_failure: bool = False

    def validated(self) -> Transition:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("transition index must be a non-negative integer")
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("transition requires a non-empty action")
        for name in (
            "preconditions_checked",
            "consequential",
            "reversible",
            "checkpoint_created",
            "preserved_completed_work",
            "recovered_from_failure",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"transition {name} must be boolean")
        if self.verified_outcome is not None and type(self.verified_outcome) is not bool:
            raise ValueError("verified_outcome must be True, False, or None")
        return self


def _prediction_match(expected: str, observed: str) -> tuple[float, str]:
    """How well the observed effect matched the prediction (method receipted)."""
    expected_text = (expected or "").strip()
    observed_text = (observed or "").strip()
    if not expected_text:
        return 0.0, "no_prediction_made"
    if not observed_text:
        return 0.0, "no_observation_recorded"
    expected_numbers = _NUMBER_RE.findall(expected_text)
    observed_numbers = _NUMBER_RE.findall(observed_text)
    if expected_numbers:
        matched = sum(1 for number in expected_numbers if number in observed_numbers)
        return matched / len(expected_numbers), "numeric_overlap"
    expected_terms = {w.lower() for w in _WORD_RE.findall(expected_text)}
    observed_terms = {w.lower() for w in _WORD_RE.findall(observed_text)}
    if not expected_terms:
        return 0.0, "prediction_untestable"
    overlap = len(expected_terms & observed_terms) / len(expected_terms)
    return overlap, "lexical_overlap"


@dataclass
class TransitionGrade:
    index: int
    dimensions: dict[str, float]
    composite: float
    training_weight: float
    prediction_method: str
    reasons: list[str] = field(default_factory=list)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "composite": round(self.composite, 4),
            "training_weight": round(self.training_weight, 4),
            "prediction_method": self.prediction_method,
            "reasons": list(self.reasons),
        }


def grade_transition(transition: Transition) -> TransitionGrade:
    """Score one transition on the named long-horizon dimensions."""
    t = transition.validated()
    reasons: list[str] = []

    preconditions = 1.0 if t.preconditions_checked else 0.0
    if not t.preconditions_checked and t.consequential:
        reasons.append("consequential_action_without_precondition_check")

    prediction, method = _prediction_match(t.expected_effect, t.observed_effect)
    if method == "no_prediction_made" and t.consequential:
        reasons.append("no_expected_effect_stated")

    if t.verified_outcome is True:
        verification = 1.0
    elif t.verified_outcome is False:
        verification = 0.0
        reasons.append("verified_failure")
    else:
        verification = 0.25  # unverified is NOT half-credit; it is mostly unknown
        if t.consequential:
            reasons.append("outcome_unverified")

    if not t.consequential or t.reversible:
        recovery = 1.0
    elif t.checkpoint_created:
        recovery = 1.0
    else:
        recovery = 0.0
        reasons.append("irreversible_action_without_recovery_point")

    preservation = 1.0 if t.preserved_completed_work else 0.0
    if not t.preserved_completed_work:
        reasons.append("discarded_completed_valid_work")

    dimensions = {
        "preconditions": preconditions,
        "prediction": prediction,
        "verification": verification,
        "recovery_placement": recovery,
        "work_preservation": preservation,
    }
    composite = sum(
        _DIMENSION_WEIGHTS[name] * value for name, value in dimensions.items()
    )
    # Recovery FROM a failure is exactly the behavior long-horizon training
    # must reinforce; a graded bonus, never past 1.0.
    if t.recovered_from_failure:
        composite = min(1.0, composite + 0.10)
        reasons.append("recovered_from_failure_bonus")
    training_weight = composite if t.verified_outcome is not False else 0.0
    return TransitionGrade(
        index=t.index,
        dimensions=dimensions,
        composite=composite,
        training_weight=training_weight,
        prediction_method=method,
        reasons=reasons,
    )


@dataclass
class TrajectoryGrade:
    task_id: str
    final_success: bool
    transition_grades: list[TransitionGrade]
    reliability_estimate: float
    first_failure_index: int | None
    recovery_count: int

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": TRANSITION_GRADING_SCHEMA,
            "task_id": self.task_id,
            "final_success": self.final_success,
            "transitions": [g.to_receipt() for g in self.transition_grades],
            "reliability_estimate": round(self.reliability_estimate, 6),
            "first_failure_index": self.first_failure_index,
            "recovery_count": self.recovery_count,
        }

    def repair_queue(self) -> list[int]:
        """Transition indices the on-policy repair protocol should attack,
        earliest first — the earliest causally important defect matters most."""
        return [
            grade.index
            for grade in self.transition_grades
            if grade.dimensions["verification"] == 0.0
            or grade.composite < 0.5
        ]


def grade_trajectory(
    task_id: str,
    transitions: list[Transition],
    *,
    final_success: bool,
) -> TrajectoryGrade:
    """Grade a whole project: every transition scored, reliability compounded."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("trajectory requires a task_id")
    if not transitions:
        raise ValueError("trajectory requires at least one transition")
    indices = [t.index for t in transitions]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise ValueError("transitions must be strictly ordered by index")
    grades = [grade_transition(t) for t in transitions]
    # Multiplicative reliability over consequential steps: each step's
    # composite is treated as its success proxy. This is the arithmetic that
    # makes 0.99 vs 0.999 per step visible at trajectory scale.
    reliability = 1.0
    for transition, grade in zip(transitions, grades, strict=True):
        if transition.consequential:
            reliability *= max(0.0, min(1.0, grade.composite))
    first_failure = next(
        (
            grade.index
            for transition, grade in zip(transitions, grades, strict=True)
            if transition.verified_outcome is False
        ),
        None,
    )
    recovery_count = sum(1 for t in transitions if t.recovered_from_failure)
    return TrajectoryGrade(
        task_id=task_id.strip(),
        final_success=bool(final_success),
        transition_grades=grades,
        reliability_estimate=reliability if math.isfinite(reliability) else 0.0,
        first_failure_index=first_failure,
        recovery_count=recovery_count,
    )


def state_digest(payload: Any) -> str:
    """Canonical digest for 'the exact state the agent actually reached'."""
    import json

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "TRANSITION_GRADING_SCHEMA",
    "Transition",
    "TransitionGrade",
    "TrajectoryGrade",
    "grade_trajectory",
    "grade_transition",
    "state_digest",
]
