import asyncio
import logging
from typing import Any

from core.memory.context_pruner import ContextPruner
from core.memory.governor import MemoryGovernor
from core.runtime.errors import record_degradation
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.MemorySubsystem")

class MemorySubsystem:
    """Orchestrates memory maintenance, pruning, and budget enforcement.
    v6.0: Hardened for persistent operations.
    """
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self.pruner = ContextPruner()
        self.governor = MemoryGovernor()
        self._task: asyncio.Task = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(
            self.maintenance_loop(),
            name="memory_subsystem.maintenance_loop",
        )
        logger.info("🧠 MemorySubsystem online (Pruner & Governor active).")

    async def stop(self):
        self._running = False
        if self._task:
            await cancel_and_join(self._task, owner="core.memory.memory_subsystem")
        logger.info("🧠 MemorySubsystem stopped.")

    async def maintenance_loop(self):
        """Background loop for memory health."""
        while self._running:
            try:
                # 1. Enforce Memory Budget (MLX cache, SQLite, etc.)
                self.governor.check()
                
                # 2. Context Pruning
                # If we have an orchestrator, we can check its total token counts
                if self.orchestrator and hasattr(self.orchestrator, 'get_current_context_tokens'):
                    tokens = self.orchestrator.get_current_context_tokens()
                    if self.pruner.needs_pruning(tokens):
                        logger.info("🕸️ Context threshold reached. Pruning recommended.")
                        # Actual pruning usually happens within the conversation loop
                        # but we can signal it here or run it on the last history.
                
                await asyncio.sleep(600) # Maintenance check every 10 minutes
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation('memory_subsystem', e)
                logger.error("MemorySubsystem maintenance error: %s", e)
                await asyncio.sleep(60)
