"""Crisis-mode assessment for long environment runs."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrisisAssessment:
    active: bool
    crisis_type: str = ""
    severity: float = 0.0
    triggers: list[str] = field(default_factory=list)
    forbidden_actions: set[str] = field(default_factory=set)
    forced_options: list[str] = field(default_factory=list)
    exit_conditions: list[str] = field(default_factory=list)


class CrisisManager:
    def assess(
        self,
        *,
        critical_resources: list[str] | None = None,
        unknown_modal: bool = False,
        governance_available: bool = True,
        trace_ok: bool = True,
        repeated_failed_action: bool = False,
    ) -> CrisisAssessment:
        triggers: list[str] = []
        if critical_resources:
            triggers.append("critical_resource_low")
        if unknown_modal:
            triggers.append("unknown_modal")
        if not governance_available:
            triggers.append("governance_unavailable")
        if not trace_ok:
            triggers.append("trace_write_failure")
        if repeated_failed_action:
            triggers.append("repeated_failed_action")
        active = bool(triggers)
        return CrisisAssessment(
            active=active,
            crisis_type=triggers[0] if triggers else "",
            severity=min(1.0, 0.35 + 0.15 * len(triggers)) if active else 0.0,
            triggers=triggers,
            forbidden_actions={"progress", "irreversible", "submit", "delete"} if active else set(),
            forced_options=["RESOLVE_MODAL", "STABILIZE_RESOURCE", "RETREAT_FROM_HAZARD", "OBSERVE_MORE", "BACKTRACK"] if active else [],
            exit_conditions=["risk_reduced", "modal_cleared", "resource_stable"] if active else [],
        )


__all__ = ["CrisisAssessment", "CrisisManager"]
