"""core/world/forecast_store.py — Strategic Forecasting Store.

Stores and scores forecast estimations for missions including:
  success probability, expected blockers, estimated time, likely failure modes,
  confidence levels, and change triggers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.ForecastStore")


@dataclass
class Forecast:
    forecast_id: str
    mission_id: str
    success_probability: float
    expected_blocker: str
    estimated_duration_s: float
    likely_failure_mode: str
    confidence: float
    change_trigger: str  # observation that would change the forecast
    created_at: float = field(default_factory=time.time)
    actual_outcome: str | None = None  # success, failure, aborted
    brier_score: float | None = None  # forecast accuracy indicator (0 is perfect, 1 is worst)


class ForecastStore:
    """Stores forecasts and evaluates prediction accuracy over time."""

    def __init__(self) -> None:
        self.forecasts: dict[str, Forecast] = {}

    def make_forecast(
        self,
        mission_id: str,
        success_prob: float,
        blocker: str,
        est_duration: float,
        failure_mode: str,
        confidence: float,
        trigger: str,
    ) -> Forecast:
        fid = f"fc_{int(time.time())}_{hash(mission_id) % 10000}"
        fc = Forecast(
            forecast_id=fid,
            mission_id=mission_id,
            success_probability=success_prob,
            expected_blocker=blocker,
            estimated_duration_s=est_duration,
            likely_failure_mode=failure_mode,
            confidence=confidence,
            change_trigger=trigger,
        )
        self.forecasts[fid] = fc
        logger.info("🔮 Forecast made for mission %s: success_prob=%.2f, blocker='%s'",
                    mission_id, success_prob, blocker)
        return fc

    def resolve_forecast(self, forecast_id: str, actual_outcome: str) -> float | None:
        """Record the actual outcome and calculate the Brier score."""
        fc = self.forecasts.get(forecast_id)
        if not fc:
            return None

        fc.actual_outcome = actual_outcome

        # Calculate Brier score for binary success indicator (1 for success, 0 for failure/abort)
        actual_val = 1.0 if actual_outcome == "success" else 0.0
        brier = (fc.success_probability - actual_val) ** 2
        fc.brier_score = brier

        logger.info("🔮 Forecast %s resolved: outcome=%s, brier_score=%.3f",
                    forecast_id, actual_outcome, brier)
        return brier

    def average_brier_score(self) -> float:
        resolved = [f.brier_score for f in self.forecasts.values() if f.brier_score is not None]
        if not resolved:
            return 0.0
        return sum(resolved) / len(resolved)

    def get_mission_forecasts(self, mission_id: str) -> list[Forecast]:
        return [f for f in self.forecasts.values() if f.mission_id == mission_id]
