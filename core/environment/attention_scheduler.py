"""Run-level attention scheduler for competing environment claims."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttentionClaim:
    source: str
    priority: float
    urgency: float
    novelty: float
    risk: float
    expected_value: float
    text: str
    goal_relevance: float = 0.0

    def computed_priority(self) -> float:
        return (
            0.35 * self.risk
            + 0.25 * self.urgency
            + 0.20 * self.expected_value
            + 0.10 * self.novelty
            + 0.10 * self.goal_relevance
            + 0.10 * self.priority
        )


@dataclass
class AttentionDecision:
    selected_claims: list[AttentionClaim]
    suppressed_claims: list[AttentionClaim]
    reason: str


class AttentionScheduler:
    def select(self, claims: list[AttentionClaim], *, limit: int = 3) -> AttentionDecision:
        ordered = sorted(claims, key=lambda c: c.computed_priority(), reverse=True)
        selected = ordered[:limit]
        suppressed = ordered[limit:]
        reason = "no_claims" if not claims else f"selected {len(selected)} highest-priority claims"
        return AttentionDecision(selected_claims=selected, suppressed_claims=suppressed, reason=reason)


__all__ = ["AttentionClaim", "AttentionDecision", "AttentionScheduler"]
