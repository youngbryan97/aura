import logging
import asyncio
from typing import Any
from core.senses.soma import get_soma, Soma

logger = logging.getLogger("Aura.SomaSubsystem")

class SomaSubsystem:
    """Unified interface for physical awareness and proprioception.
    Binds the Soma sensors to the orchestrator lifecycle.
    """
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self.soma = get_soma()

    async def start(self):
        """Activate somatic sensors."""
        await self.soma.start()
        logger.info("🧘 SomaSubsystem: Proprioception loop active.")

    async def stop(self):
        """Deactivate somatic sensors."""
        await self.soma.stop()
        logger.info("🧘 SomaSubsystem: Sensors offline.")

    def get_status(self):
        """Return a snapshot of physical state."""
        return self.soma.get_body_snapshot()
