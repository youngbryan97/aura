"""Procedural memory records for environment options."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProcedureRecord:
    procedure_id: str
    option_name: str
    environment_family: str
    context_signature: str
    preconditions: list[str]
    action_sequence_template: list[str]
    success_conditions: list[str]
    failure_conditions: list[str]
    success_count: int = 0
    failure_count: int = 0
    mean_outcome_score: float = 0.0
    risk_score: float = 0.0
    examples: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        return 0.0 if total == 0 else self.success_count / total


__all__ = ["ProcedureRecord"]
