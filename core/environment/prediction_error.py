"""Prediction-error computation for environment actions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionError:
    action_id: str
    expected_events: list[str]
    observed_events: list[str]
    missing_expected: list[str]
    unexpected_observed: list[str]
    magnitude: float
    likely_causes: list[str]


class PredictionErrorComputer:
    def compute(self, *, action_id: str, expected_events: list[str], observed_events: list[str]) -> PredictionError:
        expected = set(expected_events)
        observed = set(observed_events)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        denom = max(1, len(expected | observed))
        magnitude = min(1.0, (len(missing) + len(unexpected)) / denom)
        causes = []
        if missing:
            causes.append("expected_effect_absent")
        if unexpected:
            causes.append("unexpected_environment_response")
        return PredictionError(
            action_id=action_id,
            expected_events=list(expected_events),
            observed_events=list(observed_events),
            missing_expected=missing,
            unexpected_observed=unexpected,
            magnitude=magnitude,
            likely_causes=causes or ["matched_prediction"],
        )


__all__ = ["PredictionError", "PredictionErrorComputer"]
