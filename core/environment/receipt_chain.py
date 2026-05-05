"""Universal receipt chain for environment actions."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EnvironmentActionReceipt:
    receipt_id: str
    run_id: str
    sequence_id: int
    environment_id: str
    observation_id: str
    belief_hash_before: str
    action_intent_id: str
    simulation_id: str | None = None
    gateway_decision_id: str = ""
    will_receipt_id: str | None = None
    authority_receipt_id: str | None = None
    capability_token_id: str | None = None
    command_id: str | None = None
    execution_result_id: str | None = None
    outcome_assessment_id: str | None = None
    belief_hash_after: str | None = None
    status: str = "opened"
    opened_at: float = field(default_factory=time.time)
    closed_at: float | None = None

    def finalize(self, *, status: str, belief_hash_after: str | None = None, outcome_assessment_id: str | None = None) -> None:
        self.status = status
        self.belief_hash_after = belief_hash_after or self.belief_hash_after
        self.outcome_assessment_id = outcome_assessment_id or self.outcome_assessment_id
        self.closed_at = time.time()

    def can_execute_effect(self) -> bool:
        return bool(self.gateway_decision_id and (self.will_receipt_id or self.authority_receipt_id))


__all__ = ["EnvironmentActionReceipt"]
