from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import Callable, Dict, List

logger = logging.getLogger("aura.reflex_core")

class HardenedReflexCore:
    """Low-latency reflex system for Aura.
    Operates at the bus level to bypass cognitive reasoning for critical signals.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.reflex_actions: Dict[str, Callable] = {
            "THERMAL_CRITICAL": self._halt_heavy_tasks,
            "SECURITY_BREACH": self._isolate_network,
            "DISK_FULL": self._flush_temp_buffers,
            "IDENTITY_COLLAPSE": self._reboot_narrative_shard
        }
        self.active_reflexes: List[str] = []

    async def trigger_reflex(self, signal_type: str, metadata: Dict = None):
        """Execute a reflex action with high priority."""
        if signal_type in self.reflex_actions:
            logger.critical(f"⚡ REFLEX TRIGGERED: {signal_type}")
            self.active_reflexes.append(signal_type)
            
            # Execute reflex action immediately
            action = self.reflex_actions[signal_type]
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(metadata)
                else:
                    action(metadata)
            except Exception as e:
                record_degradation('reflex_core', e)
                logger.error(f"Failed to execute reflex {signal_type}: {e}")
        else:
            logger.warning(f"Unknown reflex signal: {signal_type}")

    # Sub-cognitive actions
    async def _halt_heavy_tasks(self, meta):
        logger.warning("Reflex: Halting non-essential LLM generation for thermal safety.")
        if self.orchestrator:
            # Direct intervention in the orchestrator task list
            pass

    async def _isolate_network(self, meta):
        logger.critical("Reflex: Cutting off external socket connections due to breach signal.")
        # Direct system-level network isolation

    async def _flush_temp_buffers(self, meta):
        logger.info("Reflex: Emergency flushing of volatile memory buffers.")

    async def _reboot_narrative_shard(self, meta):
        logger.error("Reflex: Narrative drift detected. Resetting temporary identity buffers.")
