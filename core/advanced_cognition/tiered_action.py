"""Tiered action control: reflex, habit, tactical, deliberative, reflective."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

from .schemas import ActionCandidate, Observation, clamp, stable_hash


class ActionTier(IntEnum):
    REFLEX = 0
    HABIT = 1
    TACTICAL = 2
    DELIBERATIVE = 3
    REFLECTIVE = 4


@dataclass
class TieredActionDecision:
    decision_id: str
    tier: ActionTier
    selected: dict[str, Any] | None
    latency_budget_ms: int
    reason: str
    requires_system2: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tier"] = int(self.tier)
        payload["tier_name"] = self.tier.name.lower()
        return payload


class TieredActionController:
    """Chooses the cheapest adequate control tier under uncertainty and risk."""

    LATENCY_BUDGET_MS = {
        ActionTier.REFLEX: 5,
        ActionTier.HABIT: 50,
        ActionTier.TACTICAL: 500,
        ActionTier.DELIBERATIVE: 10_000,
        ActionTier.REFLECTIVE: 60_000,
    }

    def choose_tier(
        self,
        observation: Observation | Mapping[str, Any],
        actions: Sequence[ActionCandidate | Mapping[str, Any]],
        *,
        risk: float,
        uncertainty: float,
        novelty: float = 0.0,
        self_modification: bool = False,
    ) -> TieredActionDecision:
        obs = observation if isinstance(observation, Observation) else Observation(**dict(observation))
        acts = [a if isinstance(a, ActionCandidate) else ActionCandidate(**dict(a)) for a in actions]
        selected = acts[0].to_dict() if acts else None
        if self_modification:
            tier = ActionTier.REFLECTIVE
            reason = "self-modification requires proof obligations and postmortem learning"
        elif risk >= 0.75 or uncertainty >= 0.75:
            tier = ActionTier.DELIBERATIVE
            reason = "high risk or uncertainty requires System 2"
        elif risk >= 0.45 or novelty >= 0.6:
            tier = ActionTier.TACTICAL
            reason = "moderate risk/novelty requires short-horizon search"
        elif any({"probe", "safe"} & set(a.tags) for a in acts):
            tier = ActionTier.HABIT
            reason = "familiar safe policy/habit is adequate"
        else:
            tier = ActionTier.REFLEX
            reason = "low-risk immediate action"
        return TieredActionDecision(
            decision_id=stable_hash(
                {
                    "obs": obs.observation_id,
                    "actions": [a.action_id for a in acts],
                    "risk": clamp(risk),
                    "uncertainty": clamp(uncertainty),
                    "tier": int(tier),
                },
                prefix="tier_",
            ),
            tier=tier,
            selected=selected,
            latency_budget_ms=self.LATENCY_BUDGET_MS[tier],
            reason=reason,
            requires_system2=tier >= ActionTier.TACTICAL,
        )
