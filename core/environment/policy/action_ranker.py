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
        
        # 1. Survival Score — derived from risk_delta (0.0=safe, 1.0=lethal)
        # Thresholds: >= 0.7 is very dangerous, >= 0.4 is caution, < 0.2 is safe
        risk = hypothesis.risk_delta  # float, exists on SimulationHypothesis
        if risk >= 0.85:
            survival_score = -100.0   # near-certain death
        elif risk >= 0.7:
            survival_score = -5.0     # very high risk
        elif risk >= 0.4:
            survival_score = -1.0     # caution
        else:
            survival_score = max(0.5, 1.0 - risk)  # inverse of risk

        # Apply resource deltas from simulation (predicted_resource_delta: dict[str, float])
        for resource_key, delta_value in hypothesis.predicted_resource_delta.items():
            # Positive deltas (healing, food) improve survival; negative (damage) worsen it
            survival_score += delta_value

        # 2. Progress Score (goal progression)
        progress_score = 0.0
        if intent.name in ("move", "explore_frontier", "use_stairs", "use_stairs_down", "use_stairs_up"):
            progress_score = 1.0
        if intent.name in ("stabilize_resource", "eat", "pray") and resources.critical_resources:
            progress_score += 0.8
        if "threat_response" in intent.tags or intent.name in {"retreat_to_safety", "retreat"}:
            progress_score += 1.4
        if intent.name == "wait" and not resources.critical_resources:
            progress_score -= 0.6

        # 3. Information Score (uncertainty reduction)
        info_score = hypothesis.information_gain

        total_score = (
            (survival_score * self.survival_weight) +
            (progress_score * self.progress_weight) +
            (info_score * self.information_weight)
        )
        return total_score

__all__ = ["ActionRanker"]
