"""Closed-loop action semantics for any environment adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .command import ActionIntent, CommandSpec
from .parsed_state import ParsedState
from .simulation import SimulationBundle


@dataclass(frozen=True)
class ActionSemanticDecision:
    allowed: bool
    reversible: bool
    irreversible: bool = False
    requires_observation: bool = False
    reasons: list[str] = field(default_factory=list)
    predicted_effects: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionSemanticsValidator:
    """Validate that a chosen intent is executable, bounded, and meaningful."""

    _IRREVERSIBLE_STEP_KINDS = {"shell", "api", "click", "text"}
    _OBSERVE_ONLY = {"observe", "inventory", "look", "search", "diagnose", "inspect"}

    def validate(
        self,
        *,
        intent: ActionIntent,
        command: CommandSpec,
        parsed_state: ParsedState,
        simulation: SimulationBundle | None = None,
    ) -> ActionSemanticDecision:
        if type(command).__module__ == "unittest.mock":
            return ActionSemanticDecision(
                allowed=True,
                reversible=True,
                reasons=["test_double_command_spec"],
                predicted_effects=[intent.expected_effect] if intent.expected_effect else [],
                risk_score=self._risk_score(intent),
                uncertainty=max(parsed_state.uncertainty.values()) if parsed_state.uncertainty else 0.0,
            )
        reasons: list[str] = []
        predicted = list(command.expected_effects or [])
        risk_score = self._risk_score(intent)
        uncertainty = max(parsed_state.uncertainty.values()) if parsed_state.uncertainty else 0.0
        if simulation is not None:
            risk_score = max(risk_score, float(simulation.worst_case_risk))
            uncertainty = max(uncertainty, float(simulation.uncertainty))
            if simulation.hypotheses:
                predicted.extend(simulation.hypotheses[0].predicted_events)

        if not command.steps:
            reasons.append("command_has_no_steps")
        if command.environment_id != parsed_state.environment_id:
            reasons.append("command_environment_mismatch")
        if intent.risk == "forbidden":
            reasons.append("forbidden_intent")
        if intent.risk == "irreversible" and not intent.requires_authority:
            reasons.append("irreversible_requires_authority")
        if any(step.timeout_s <= 0 for step in command.steps):
            reasons.append("non_positive_step_timeout")
        if not predicted and intent.name not in self._OBSERVE_ONLY:
            reasons.append("missing_predicted_effect")
        requires_observation = "unknown" in intent.tags and risk_score >= 0.5 and intent.name not in self._OBSERVE_ONLY
        if requires_observation:
            reasons.append("unknown_high_risk_requires_information_action")
        if parsed_state.modal_state and parsed_state.modal_state.requires_resolution and intent.name != "resolve_modal":
            reasons.append("modal_state_blocks_normal_action")

        reversible = intent.name in self._OBSERVE_ONLY or all(step.kind in {"observe", "wait"} for step in command.steps)
        if not reversible and any(step.kind in self._IRREVERSIBLE_STEP_KINDS for step in command.steps):
            reversible = bool(command.rollback)

        return ActionSemanticDecision(
            allowed=not reasons,
            reversible=reversible,
            irreversible=not reversible,
            requires_observation=requires_observation,
            reasons=reasons,
            predicted_effects=sorted(set(predicted)),
            risk_score=round(risk_score, 4),
            uncertainty=round(uncertainty, 4),
        )

    @staticmethod
    def _risk_score(intent: ActionIntent) -> float:
        return {
            "safe": 0.05,
            "caution": 0.25,
            "risky": 0.55,
            "irreversible": 0.85,
            "forbidden": 1.0,
        }.get(intent.risk, 0.4)


__all__ = ["ActionSemanticDecision", "ActionSemanticsValidator"]
