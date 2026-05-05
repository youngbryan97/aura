"""Outcome attribution independent of superficial screen/page diffs."""
from __future__ import annotations

from dataclasses import dataclass, field

from .prediction_error import PredictionError


@dataclass
class OutcomeAssessment:
    action: str
    expected_effect: str
    observed_events: list[str]
    success_score: float
    harm_score: float
    information_gain: float
    surprise: float
    credit_assignment: dict[str, float] = field(default_factory=dict)
    lesson: str | None = None


class OutcomeAttributor:
    def assess(
        self,
        *,
        action: str,
        expected_effect: str,
        observed_events: list[str],
        prediction_error: PredictionError | None = None,
        resource_delta: dict[str, float] | None = None,
        information_gain: float = 0.0,
    ) -> OutcomeAssessment:
        resource_delta = resource_delta or {}
        harm = sum(abs(v) for v in resource_delta.values() if v < 0)
        surprise = prediction_error.magnitude if prediction_error else 0.0
        expected_hit = expected_effect in observed_events if expected_effect else False
        info_success = information_gain > 0 and action in {"observe", "inspect", "diagnose", "inventory", "look"}
        success = 0.8 if expected_hit or info_success else max(0.0, 0.4 - surprise - harm)
        if harm:
            success = min(success, 0.5)
        lesson = None
        if surprise >= 0.5:
            lesson = "prediction mismatch: gather evidence before repeating this action"
        if harm:
            lesson = "harmful outcome: prefer stabilization or rollback before progress"
        return OutcomeAssessment(
            action=action,
            expected_effect=expected_effect,
            observed_events=list(observed_events),
            success_score=max(0.0, min(1.0, success)),
            harm_score=max(0.0, min(1.0, harm)),
            information_gain=max(0.0, min(1.0, information_gain)),
            surprise=surprise,
            credit_assignment={action: max(0.0, min(1.0, success - harm))},
            lesson=lesson,
        )


__all__ = ["OutcomeAssessment", "OutcomeAttributor"]
