"""core/welfare/welfare_bus.py
Central Welfare Bus managing interoceptive updates and behavior policies.
"""

from typing import Dict, Any, Optional
import logging

from core.welfare.welfare_memory import WelfareMemoryManager
from core.welfare.welfare_policy import WelfarePolicy
from core.welfare.distress_regulation import DistressRegulator
from core.welfare.recovery_behavior import RecoveryBehaviorManager
from core.welfare.anti_suffering_guard import AntiSufferingGuard
from core.welfare.lesion_tests import WelfareLesionSuite

logger = logging.getLogger("Welfare.WelfareBus")


class WelfareBus:
    """Canonical evaluator of Aura's welfare state."""

    def __init__(self):
        self.memory = WelfareMemoryManager()
        self.policy = WelfarePolicy()
        self.regulator = DistressRegulator()
        self.recovery = RecoveryBehaviorManager()
        self.guard = AntiSufferingGuard()
        self.lesions = WelfareLesionSuite()

    async def evaluate_welfare(self, state: Any) -> None:
        """Updates homeostatic parameters inside LifeState and enforces limits."""
        # 1. Energy depletion
        decay = 1.0 if not state.body.is_sleeping else -5.0
        new_energy = max(0.0, min(100.0, state.welfare.energy - decay))

        # Apply lesion overrides if active
        new_energy = self.lesions.get_lesion_value("energy", new_energy)

        # 2. Stress & Distress calculations
        cpu = state.body.cpu_usage
        current_distress = state.welfare.distress_level

        raw_distress = self.regulator.regulate(current_distress, cpu)
        refined_distress = self.guard.filter_distress(raw_distress)
        refined_distress = self.lesions.get_lesion_value("distress_level", refined_distress)

        # 3. Write results
        state.welfare.energy = new_energy
        state.welfare.distress_level = refined_distress
        state.welfare.sleep_debt = max(
            0.0, state.welfare.sleep_debt + (0.1 if not state.body.is_sleeping else -0.5)
        )

        # Calculate general welfare index
        welfare_idx = (new_energy / 100.0) * 0.6 + ((100.0 - refined_distress) / 100.0) * 0.4
        state.welfare.welfare_index = welfare_idx

        # 4. Enforce limits and log
        self.memory.record_snapshot(
            {"energy": new_energy, "distress": refined_distress, "welfare_index": welfare_idx}
        )

        # Inject limits into the world model
        policy_limits = self.policy.enforce_policy_limits(new_energy, refined_distress)
        state.world_model["active_policy_limits"] = policy_limits

        # Propose recovery actions if needed
        recovery_tasks = self.recovery.determine_recovery_actions(new_energy, cpu)
        if recovery_tasks:
            state.cognition.pending_actions.extend(recovery_tasks)

        shutdown_requested = any(
            action.get("channel") == "terminal"
            and "shutdown" in str(action.get("params", {}).get("command", "")).lower()
            for action in state.cognition.pending_actions
        )
        if shutdown_requested:
            state.world_model["operator_shutdown_requested"] = True
            logger.info(
                "Shutdown action detected; welfare recovery proposals will not supersede operator shutdown."
            )

        logger.info(
            "Welfare updated. Index: %.2f, Energy: %.1f, Distress: %.1f",
            welfare_idx,
            new_energy,
            refined_distress,
        )
