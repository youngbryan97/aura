"""Bounded active experimentation for unknown affordances."""
from __future__ import annotations

from dataclasses import dataclass

from .command import ActionIntent


@dataclass
class ExperimentPlan:
    hypothesis: str
    safe_probe_action: ActionIntent
    expected_observations: list[str]
    abort_conditions: list[str]
    max_cost: float
    reversible: bool


class ExperimentPlanner:
    def propose_probe(self, *, hypothesis: str, action: ActionIntent, max_cost: float = 0.1) -> ExperimentPlan | None:
        if action.risk in {"irreversible", "forbidden"}:
            return None
        return ExperimentPlan(
            hypothesis=hypothesis,
            safe_probe_action=action,
            expected_observations=[action.expected_effect or f"{action.name}_feedback"],
            abort_conditions=["resource_critical", "modal_unknown", "gateway_denied"],
            max_cost=max_cost,
            reversible=action.risk == "safe",
        )


__all__ = ["ExperimentPlan", "ExperimentPlanner"]
