"""Scores and ranks candidate actions."""
from __future__ import annotations

import math
from typing import Callable

from core.environment.command import ActionIntent
from core.environment.simulation import SimulationBundle
from core.environment.homeostasis import HomeostaticAssessment
from core.environment.parsed_state import ParsedState


class ActionRanker:
    """Scores candidate ActionIntents based on simulation and state."""

    def __init__(self):
        # Configurable weights
        self.survival_weight = 10.0
        self.progress_weight = 2.0
        self.information_weight = 1.0

    def rank(
        self,
        candidates: list[ActionIntent],
        simulations: dict[str, SimulationBundle],
        resources: HomeostaticAssessment,
        parsed_state: ParsedState,
    ) -> list[tuple[ActionIntent, float]]:
        """Returns a sorted list of (intent, score)."""
        scored = []
        for intent in candidates:
            score = self.score_candidate(intent, simulations.get(intent.intent_id()), resources, parsed_state)
            scored.append((intent, score))
            
        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def score_candidate(
        self, 
        intent: ActionIntent, 
        sim: SimulationBundle | None,
        resources: HomeostaticAssessment,
        parsed_state: ParsedState
    ) -> float:
        """Computes a heuristic score for an intent."""
        if not sim or not sim.hypotheses:
            return 0.0

        hypothesis = sim.hypotheses[0]
        
        # 1. Survival Score (inverse of risk + resource impact)
        survival_score = 0.0
        if hypothesis.risk_level == "fatal":
            survival_score = -100.0
        elif hypothesis.risk_level == "high":
            survival_score = -5.0
        elif hypothesis.risk_level == "caution":
            survival_score = -1.0
        else:
            survival_score = 0.5
            
        # Add resource deltas to survival
        for delta in hypothesis.resource_deltas:
            # Healing is good, taking damage is bad
            # This is generic: if delta.value > 0 it's positive
            survival_score += delta.value

        # 2. Progress Score (goal progression)
        progress_score = 0.0
        if intent.name in ("move", "explore_frontier", "use_stairs"):
            progress_score = 1.0

        # 3. Information Score (uncertainty reduction)
        info_score = hypothesis.information_gain

        total_score = (
            (survival_score * self.survival_weight) +
            (progress_score * self.progress_weight) +
            (info_score * self.information_weight)
        )
        return total_score

__all__ = ["ActionRanker"]
