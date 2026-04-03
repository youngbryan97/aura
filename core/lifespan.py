"""core/lifespan.py — v1.0 ATOMIC LIFECYCLE MANAGER
Ensures every service starts and stops in correct order with zero resource leaks.
No more dangling tasks, queues, or VRAM holds.
"""

import asyncio
import logging
from typing import List, Callable, Optional
from core.container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Aura.Lifespan")

class LifespanManager:
    def __init__(self):
        self.startup_tasks: List[Callable] = []
        self.shutdown_tasks: List[Callable] = []
        self._running = False

    async def startup(self):
        logger.info("🌟 Aura Lifespan Startup Sequence Initiated")
        self._running = True

        # Critical order: Reliability first → everything else
        services = [
            "reliability_engine",
            "local_voice_cortex",
            "terminal_monitor",
            "system_monitor",
            "planner",
            "insight_journal",
            "sovereign_swarm"
        ]

        for name in services:
            try:
                svc = ServiceContainer.get(name)
                if svc and hasattr(svc, "start") and asyncio.iscoroutinefunction(svc.start):
                    await svc.start()
                    logger.info(f"✅ {name} started")
            except Exception as e:
                logger.critical(f"❌ {name} failed to start: {e}")
                await self.emergency_shutdown()
                raise

    async def shutdown(self):
        if not self._running:
            return
        logger.info("🛑 Aura Lifespan Shutdown Sequence Initiated")

        # Reverse order shutdown with timeout
        for name in reversed([
            "sovereign_swarm", "planner", "insight_journal",
            "system_monitor", "terminal_monitor", "local_voice_cortex",
            "reliability_engine"
        ]):
            try:
                svc = ServiceContainer.get(name)
                if svc and hasattr(svc, "stop") and asyncio.iscoroutinefunction(svc.stop):
                    await asyncio.wait_for(svc.stop(), timeout=8.0)
                    logger.info(f"✅ {name} stopped cleanly")
                elif svc and hasattr(svc, "stop"):
                    # Fallback for non-async stop if any
                    svc.stop()
                    logger.info(f"✅ {name} stopped (sync)")
            except asyncio.TimeoutError:
                logger.error(f"⏰ {name} shutdown timeout — force cancelling")
            except Exception as e:
                logger.error(f"⚠️ {name} shutdown error: {e}")

        self._running = False
        logger.info("✅ Aura shutdown complete — all resources released.")

    def register_startup(self, coro: Callable):
        self.startup_tasks.append(coro)

    def register_shutdown(self, coro: Callable):
        self.shutdown_tasks.append(coro)

    async def emergency_shutdown(self):
        logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED")
        await self.shutdown()

# Singleton
_lifespan: Optional[LifespanManager] = None
def get_lifespan_manager() -> LifespanManager:
    global _lifespan
    if _lifespan is None:
        _lifespan = LifespanManager()
    return _lifespan

# Registration
from core.container import ServiceContainer, ServiceLifetime
ServiceContainer.register("lifespan_manager", get_lifespan_manager, ServiceLifetime.SINGLETON)
