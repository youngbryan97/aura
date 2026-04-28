"""SemanticWeightGovernor — gates plastic updates.

Decides whether a proposed weight update is allowed and, if so, what
modulation strength to apply.  The decision blends:

  * vitality (refuse plasticity when the system is critical)
  * reward magnitude (refuse no-signal updates)
  * curiosity / arousal (raise modulation when the agent is "interested")
  * free-energy (raise modulation when prediction error is informative)

Real Aura wires this through the Will/Authority gateway; this module
is the local default and the integration target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PlasticityDecision:
    allowed: bool
    modulation: float
    reason: str
    severity: str = "normal"


class SemanticWeightGovernor:
    def __init__(
        self,
        *,
        min_reward_magnitude: float = 0.05,
        min_vitality: float = 0.30,
    ):
        self.min_reward_magnitude = float(min_reward_magnitude)
        self.min_vitality = float(min_vitality)

    def decide(
        self,
        *,
        module_name: str,
        reward: float,
        vitality: float = 1.0,
        curiosity: float = 0.5,
        arousal: float = 0.5,
        free_energy: float = 0.0,
    ) -> PlasticityDecision:
        if vitality < self.min_vitality:
            return PlasticityDecision(False, 0.0, "vitality_too_low", "critical")
        if abs(reward) < self.min_reward_magnitude:
            return PlasticityDecision(False, 0.0, "reward_too_weak")

        modulation = (
            0.20
            + 0.35 * min(1.0, max(0.0, free_energy))
            + 0.30 * min(1.0, max(0.0, curiosity))
        )
        modulation *= 0.5 + 0.5 * min(1.0, max(0.0, arousal))
        modulation = max(0.0, min(1.0, modulation))

        return PlasticityDecision(True, modulation, "allowed")
