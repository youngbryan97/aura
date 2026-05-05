"""Semantic action budgets for bounded autonomy."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionBudget:
    max_total_steps: int
    max_irreversible_actions: int
    max_unknown_actions: int
    max_repeated_failures: int
    max_modal_steps: int
    max_resource_cost: float
    used_total_steps: int = 0
    used_irreversible_actions: int = 0
    used_unknown_actions: int = 0
    used_modal_steps: int = 0
    used_resource_cost: float = 0.0
    failure_counts: dict[str, int] = field(default_factory=dict)

    def record(self, *, action_name: str, irreversible: bool = False, unknown: bool = False, modal: bool = False, cost: float = 0.0, failed: bool = False) -> None:
        self.used_total_steps += 1
        self.used_irreversible_actions += int(irreversible)
        self.used_unknown_actions += int(unknown)
        self.used_modal_steps += int(modal)
        self.used_resource_cost += float(cost)
        if failed:
            self.failure_counts[action_name] = self.failure_counts.get(action_name, 0) + 1

    def exhausted_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.used_total_steps > self.max_total_steps:
            reasons.append("max_total_steps")
        if self.used_irreversible_actions > self.max_irreversible_actions:
            reasons.append("max_irreversible_actions")
        if self.used_unknown_actions > self.max_unknown_actions:
            reasons.append("max_unknown_actions")
        if self.used_modal_steps > self.max_modal_steps:
            reasons.append("max_modal_steps")
        if self.used_resource_cost > self.max_resource_cost:
            reasons.append("max_resource_cost")
        if any(count > self.max_repeated_failures for count in self.failure_counts.values()):
            reasons.append("max_repeated_failures")
        return reasons


__all__ = ["ActionBudget"]
