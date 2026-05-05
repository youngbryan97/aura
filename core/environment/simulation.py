"""General tactical simulation with uncertainty.

Enhancements:
- Checks belief graph for adjacent hostile entities on ``move`` intents.
- Predicts resource deltas for ``eat``, ``quaff``, ``stabilize``.
- Assigns higher information gain to observation/search intents.
- Propagates uncertainty for unknown-tagged items.
"""
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


# Intents that are purely informational and should get high info-gain
_OBSERVATION_INTENTS = {"observe", "inspect", "diagnose", "resolve_modal", "inventory", "far_look", "look", "search"}

# Intents that change resources
_RESOURCE_INTENTS = {"eat", "quaff", "stabilize", "pray"}


class TacticalSimulator:
    def simulate(self, belief: EnvironmentBeliefGraph, action_intent: ActionIntent) -> SimulationBundle:
        risk = {"safe": 0.05, "caution": 0.25, "risky": 0.55, "irreversible": 0.85, "forbidden": 1.0}.get(action_intent.risk, 0.4)
        info_gain = 0.35 if action_intent.name in _OBSERVATION_INTENTS else 0.05
        predicted = [action_intent.expected_effect or f"{action_intent.name}_attempted"]
        resource_delta: dict[str, float] = {}
        assumptions = ["generic_transition_model"]

        # --- Domain-aware risk for move intents ---
        if action_intent.name == "move":
            if action_intent.target_id in belief.hazards:
                predicted.append("risk_increase")
                risk = max(risk, 0.7)
            # Check for adjacent hostile entities in the belief graph
            hostile_nearby = sum(
                1 for node in belief.nodes.values()
                if node.kind.startswith("entity:hostile") and node.confidence >= 0.5
            )
            if hostile_nearby > 0:
                risk = max(risk, 0.3 + 0.1 * min(hostile_nearby, 4))
                assumptions.append(f"hostile_entities_nearby:{hostile_nearby}")

        # --- Resource delta predictions ---
        if action_intent.name in {"eat", "stabilize"}:
            resource_delta["nutrition"] = 0.3
            assumptions.append("food_available")

        if action_intent.name == "quaff":
            if "unknown" in action_intent.tags:
                resource_delta["health"] = 0.0  # uncertain
                risk = max(risk, 0.5)
                assumptions.append("potion_identity_unknown")
            else:
                resource_delta["health"] = 0.2
                assumptions.append("healing_potion_assumed")

        if action_intent.name == "pray":
            resource_delta["health"] = 0.5
            risk = max(risk, 0.15)
            assumptions.append("prayer_available")

        # --- Uncertainty for unknown items ---
        if "unknown" in action_intent.tags:
            info_gain = max(info_gain, 0.4)

        hypothesis = SimulationHypothesis(
            predicted_events=predicted,
            predicted_resource_delta=resource_delta,
            predicted_context_delta={},
            information_gain=info_gain,
            risk_delta=risk,
            confidence=max(0.1, 1.0 - risk * 0.5),
            assumptions=assumptions,
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

