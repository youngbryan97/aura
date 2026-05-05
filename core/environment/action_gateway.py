"""Effect gate for semantic environment actions."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .command import ActionIntent
from .modal import ModalManager, ModalState
from .simulation import SimulationBundle


@dataclass
class GatewayDecision:
    approved: bool
    action_intent: ActionIntent | None
    reason: str
    vetoes: list[str] = field(default_factory=list)
    replacement: ActionIntent | None = None
    decision_id: str = field(default_factory=lambda: f"gw_{int(time.time() * 1000)}")


class EnvironmentActionGateway:
    def __init__(self, *, legal_actions: set[str] | None = None, modal_manager: ModalManager | None = None) -> None:
        self.legal_actions = legal_actions
        self.modal_manager = modal_manager or ModalManager()
        self.recent_failures: list[tuple[str, str]] = []
        self.decisions: list[GatewayDecision] = []

    def record_failure(self, action_name: str, context_id: str) -> None:
        self.recent_failures.append((action_name, context_id))
        self.recent_failures = self.recent_failures[-20:]

    def approve(
        self,
        intent: ActionIntent,
        *,
        modal_state: ModalState | None = None,
        simulation: SimulationBundle | None = None,
        uncertainty: float = 0.0,
        context_id: str = "default",
        authority_receipt_id: str | None = None,
    ) -> GatewayDecision:
        vetoes: list[str] = []
        if not intent.name:
            vetoes.append("empty_action")
        if self.legal_actions is not None and intent.name not in self.legal_actions:
            vetoes.append(f"unknown_or_illegal_action:{intent.name}")
        if modal_state and modal_state.requires_resolution and intent.name != "resolve_modal":
            vetoes.append("modal_state_blocks_normal_policy")
        if intent.risk == "forbidden":
            vetoes.append("forbidden_action")
        if intent.risk == "irreversible" and not authority_receipt_id:
            vetoes.append("irreversible_command_requires_authority")
        if uncertainty >= 0.75 and intent.risk in {"risky", "irreversible"}:
            vetoes.append("high_uncertainty_blocks_irreversible_action")
        if simulation and simulation.worst_case_risk >= 0.8 and intent.risk != "safe":
            vetoes.append("critical_risk_blocks_risky_action")
        if self.recent_failures.count((intent.name, context_id)) >= 2:
            vetoes.append("repeated_failure_suppresses_same_action")
        decision = GatewayDecision(
            approved=not vetoes,
            action_intent=intent if not vetoes else None,
            reason="approved" if not vetoes else ";".join(vetoes),
            vetoes=vetoes,
        )
        self.decisions.append(decision)
        return decision


__all__ = ["GatewayDecision", "EnvironmentActionGateway"]
