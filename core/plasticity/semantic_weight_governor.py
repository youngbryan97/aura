"""SemanticWeightGovernor — gates plastic updates.

Decides whether a proposed weight update or codebase mutation is allowed and, if so,
what modulation strength or plasticity level to apply.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Plasticity.WeightGovernor")


@dataclass
class PlasticityDecision:
    allowed: bool
    modulation: float
    reason: str
    severity: str = "normal"


class SemanticWeightGovernor:
    # Staged Plasticity Levels
    LEVEL_0 = 0  # Prompt only
    LEVEL_1 = 1  # LoRA adapters weight update
    LEVEL_2 = 2  # Actuator/skill synthesis
    LEVEL_3 = 3  # Core codebase/module edits

    def __init__(
        self,
        *,
        min_reward_magnitude: float = 0.05,
        min_vitality: float = 0.30,
        phi_threshold: float = 0.40,
    ):
        self.min_reward_magnitude = float(min_reward_magnitude)
        self.min_vitality = float(min_vitality)
        self.phi_threshold = float(phi_threshold)
        
        # Staged Plasticity State
        self.recent_successes: list[bool] = [True] * 10  # Initial baseline
        self.trust_level = self.LEVEL_1  # Default level
        self.pending_rollback: dict[str, Any] | None = None

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

    def register_interaction(self, success: bool) -> None:
        """Register memory/actuator run success/failure to modulate system trust."""
        self.recent_successes.append(success)
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        self.update_trust_level()

    def compute_system_trust(self) -> float:
        """Compute the system trust score based on successes and homeostatic coherence."""
        success_rate = sum(1 for x in self.recent_successes if x) / max(len(self.recent_successes), 1)
        
        # Read coherence from LiquidSubstrate if online
        coherence = 0.8
        try:
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate:
                if hasattr(substrate, "microtubule_coherence"):
                    coherence = float(substrate.microtubule_coherence)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "semantic_weight_governor",
                exc,
                severity="warning",
                action="kept conservative default coherence while trust gate stayed online",
            )
            
        trust_score = 0.6 * success_rate + 0.4 * coherence
        return max(0.0, min(1.0, trust_score))

    def update_trust_level(self) -> None:
        """Dynamically modulate allowable staged plasticity levels."""
        score = self.compute_system_trust()
        if score >= 0.8:
            new_level = self.LEVEL_3
        elif score >= 0.6:
            new_level = self.LEVEL_2
        elif score >= 0.4:
            new_level = self.LEVEL_1
        else:
            new_level = self.LEVEL_0

        if new_level != self.trust_level:
            logger.info("⚡ [PLASTICITY] Trust Level modulated: Level %d -> Level %d (Trust Score: %.2f)", self.trust_level, new_level, score)
            self.trust_level = new_level

    def validate_plasticity_request(self, requested_level: int) -> bool:
        """Validate if the requesting staged plasticity level is permitted."""
        self.update_trust_level()
        return requested_level <= self.trust_level

    def trigger_rollback_if_needed(self, error_spike: bool = False, current_phi: float = 1.0) -> bool:
        """Fail closed and stage rollback instructions after unsafe plasticity signals.

        This gate intentionally does not run destructive git commands. Runtime
        plasticity can freeze itself and emit a recovery directive; promotion,
        reset, or cleanup must happen through the governed self-repair /
        self-modification path where backups, tests, receipts, and human policy
        can be enforced.
        """
        should_freeze = error_spike or current_phi < self.phi_threshold
        if not should_freeze:
            return False

        self.pending_rollback = {
            "reason": "error_spike" if error_spike else "phi_below_threshold",
            "current_phi": float(current_phi),
            "phi_threshold": self.phi_threshold,
            "created_at": time.time(),
            "required_path": "SelfRepairGateway/SelfModificationGateway with receipts and tests",
            "destructive_git_allowed": False,
        }
        self.recent_successes = [False] * 20
        self.update_trust_level()

        event = RuntimeError("System performance fell below homeostatic limits post-modification.")
        record_degradation(
            "semantic_weight_governor",
            event,
            severity="critical",
            action="plasticity frozen; rollback staged for governed self-repair path",
            extra={"pending_rollback": dict(self.pending_rollback)},
        )
        logger.critical(
            "🚨 [PLASTICITY] Degradation detected; plasticity frozen and governed rollback staged: %s",
            self.pending_rollback,
        )
        return True
