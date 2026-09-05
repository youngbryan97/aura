"""core/sleep/sleep_cycle.py
Coordinates offline sleep consolidation states and offline processing routines.
"""
import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.sleep.dream_simulator import DreamSimulator
from core.sleep.identity_consolidation import IdentityConsolidator
from core.sleep.memory_consolidation import MemoryConsolidator
from core.sleep.nightly_report import NightlyReportCompiler
from core.sleep.value_consolidation import ValueConsolidator
from core.sleep.world_model_training import WorldModelTrainer

logger = logging.getLogger("Sleep.SleepCycle")

_SLEEP_CYCLE_ERRORS = (AttributeError, ImportError, LookupError, RuntimeError, TimeoutError, TypeError, ValueError)


class SleepManager:
    """Canonical manager governing Aura's offline sleep loop states."""

    def __init__(self):
        self.dreamer = DreamSimulator()
        self.memory_consolidator = MemoryConsolidator()
        self.value_consolidator = ValueConsolidator()
        self.world_trainer = WorldModelTrainer()
        self.identity_consolidator = IdentityConsolidator()
        self.reporter = NightlyReportCompiler()

    async def should_trigger_sleep(self, state: Any) -> bool:
        """Determines if the sleep criteria are satisfied (e.g. low energy)."""
        return state.welfare.energy < 15.0 or state.welfare.sleep_debt > 16.0

    async def execute_sleep_cycle(self, state: Any) -> bool:
        """Runs the complete sleep-cycle pipeline, blocking active actions."""
        logger.info("Aura entering offline sleep consolidation cycle...")
        state.body.is_sleeping = True

        try:
            # 1. Dream simulations
            dreams = self.dreamer.simulate_scenarios(state.cognition.current_goals)

            # 2. Memory compaction
            await self.memory_consolidator.consolidate_logs(state)

            # 3. Value consolidation
            state.active_preferences = self.value_consolidator.consolidate_preferences(state.active_preferences)

            # 4. World model causal alignment
            self.world_trainer.train_world_model(state.world_model)

            # 5. Identity alignment checks
            from core.identity.identity_kernel import IdentityKernel
            kernel = IdentityKernel()
            baseline = kernel.get_current_identity()
            state.identity = self.identity_consolidator.consolidate_identity(state.identity, baseline)

            # 6. Nightly report compilation
            report = self.reporter.compile_report(state, len(dreams))
            state.world_model["last_nightly_report"] = report
            logger.info("Nightly Report generated:\n%s", report)

            # Reset interoceptive metrics
            state.welfare.energy = 100.0
            state.welfare.sleep_debt = 0.0
            state.world_model.pop("last_sleep_cycle_error", None)
            logger.info("Sleep cycle complete. Energy fully restored.")
            return True

        except _SLEEP_CYCLE_ERRORS as e:
            state.world_model["last_sleep_cycle_error"] = {
                "error_type": type(e).__name__,
                "message": str(e)[:500],
            }
            record_degradation(
                "sleep.cycle",
                e,
                severity="degraded",
                action="preserved pre-sleep energy and debt, recorded the failed stage, and returned failure",
            )
            logger.error("Error during sleep cycle execution: %s", e, exc_info=True)
            return False
        finally:
            state.body.is_sleeping = False
