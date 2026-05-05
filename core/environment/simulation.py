"""General tactical simulation with uncertainty."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .belief_graph import EnvironmentBeliefGraph
from .command import ActionIntent


@dataclass
class SimulationHypothesis:
    predicted_events: list[str]
    predicted_resource_delta: dict[str, float]
    predicted_context_delta: dict[str, Any]
    information_gain: float
    risk_delta: float
    confidence: float
    assumptions: list[str] = field(default_factory=list)


@dataclass
class SimulationBundle:
    action_intent: ActionIntent
    hypotheses: list[SimulationHypothesis]
    worst_case_risk: float
    expected_value: float
    uncertainty: float


class TacticalSimulator:
    def simulate(self, belief: EnvironmentBeliefGraph, action_intent: ActionIntent) -> SimulationBundle:
        risk = {"safe": 0.05, "caution": 0.25, "risky": 0.55, "irreversible": 0.85, "forbidden": 1.0}.get(action_intent.risk, 0.4)
        info_gain = 0.25 if action_intent.name in {"observe", "inspect", "diagnose", "resolve_modal"} else 0.05
        predicted = [action_intent.expected_effect or f"{action_intent.name}_attempted"]
        if action_intent.name == "move" and action_intent.target_id in belief.hazards:
            predicted.append("risk_increase")
            risk = max(risk, 0.7)
        hypothesis = SimulationHypothesis(
            predicted_events=predicted,
            predicted_resource_delta={},
            predicted_context_delta={},
            information_gain=info_gain,
            risk_delta=risk,
            confidence=max(0.1, 1.0 - risk * 0.5),
            assumptions=["generic_transition_model"],
        )
        uncertainty = 0.2 + (0.5 if "unknown" in action_intent.tags else 0.0)
        ev = info_gain + (0.4 if action_intent.name in {"stabilize", "retreat"} else 0.2) - risk
        return SimulationBundle(
            action_intent=action_intent,
            hypotheses=[hypothesis],
            worst_case_risk=risk,
            expected_value=ev,
            uncertainty=min(1.0, uncertainty),
        )


__all__ = ["SimulationHypothesis", "SimulationBundle", "TacticalSimulator"]
