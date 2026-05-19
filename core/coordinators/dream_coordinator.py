import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Optional
from core.runtime.errors import record_degradation
from core.container import ServiceContainer

logger = logging.getLogger("Aura.DreamCoordinator")

class DreamCoordinator:
    """
    Coordinates various dream/maintenance cycles to prevent them from 
    concurrently accessing the same EpisodicMemory and locking each other out.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._last_run: dict[str, float] = {}

    async def _execute_with_lock(self, name: str, coro_func: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        if self._lock.locked():
            logger.debug("DreamCoordinator: %s deferred, another dream cycle is running.", name)
            return None
        
        async with self._lock:
            try:
                logger.info("DreamCoordinator: Starting %s...", name)
                start_t = time.time()
                result = await coro_func()
                elapsed = time.time() - start_t
                self._last_run[name] = time.time()
                logger.info("DreamCoordinator: Finished %s in %ss", name, f"{elapsed:.2f}")
                return result
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("dream_coordinator", e)
                logger.error("DreamCoordinator: Failed %s: %s", name, e)
                return None

    async def run_resilience_dream(self) -> None:
        """Priority 1: Dead Letter Queue (DLQ) re-ingestion."""
        async def _run():
            # The actual resilience cycle is handled by the orchestrator's instance
            # which usually loops, but if we coordinate it here, we should just call its _process_dreams
            orch = ServiceContainer.get("orchestrator", default=None)
            if orch and hasattr(orch, "dream_cycle") and orch.dream_cycle:
                # DreamCycle in resilience is a thread/task. If we just call process_dreams:
                if hasattr(orch.dream_cycle, "process_dreams"):
                    await orch.dream_cycle.process_dreams()
        await self._execute_with_lock("resilience", _run)

    async def run_maintenance_dream(self) -> None:
        """Priority 2: Maintenance and WAL checkpointing."""
        async def _run():
            from core.maintenance.dream_cycle import run_dream_cycle
            await run_dream_cycle()
        await self._execute_with_lock("maintenance", _run)

    async def run_dreamer_v2(self) -> None:
        """Priority 3: Full biological sleep cycle."""
        async def _run():
            dreamer = ServiceContainer.get("dreamer_v2", default=None)
            if dreamer:
                # Assume dreamer has a dream() method or similar
                if hasattr(dreamer, "dream"):
                    await dreamer.dream()
        await self._execute_with_lock("dreamer_v2", _run)

    async def run_dream_processor(self) -> None:
        """Priority 4: Legacy DreamProcessor (Deprecated, No-Op)."""
        logger.debug("DreamCoordinator: DreamProcessor is deprecated and disabled.")
        return

_instance = None

def get_dream_coordinator() -> DreamCoordinator:
    global _instance
    if _instance is None:
        _instance = DreamCoordinator()
        ServiceContainer.register_instance("dream_coordinator", _instance)
    return _instance
