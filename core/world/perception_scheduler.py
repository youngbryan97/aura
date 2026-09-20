"""core/world/perception_scheduler.py — Periodic Ingestion Perception Scheduler.
"""
from __future__ import annotations

import asyncio
import logging

from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task
from core.utils.concurrency import cancel_and_join
from core.world.perception_hub import PerceptionHub

logger = logging.getLogger("Aura.PerceptionScheduler")
_PERCEPTION_SCHEDULER_RECOVERABLE_ERRORS = (
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class PerceptionScheduler:
    """Schedules and runs background perception sweeps."""

    def __init__(
        self,
        interval_seconds: float = 3600.0,
        *,
        perception_hub: PerceptionHub | None = None,
    ) -> None:
        self.interval = interval_seconds
        self._perception_hub = perception_hub or PerceptionHub()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = create_tracked_task(self._loop(), name="world.perception_scheduler")
        logger.info("⏱️  Perception Scheduler started with interval %.1fs", self.interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await cancel_and_join(self._task, owner="core.world.perception_scheduler")
        logger.info("⏱️  Perception Scheduler stopped.")

    async def _loop(self) -> None:
        while self._running:
            try:
                # Periodic general intelligence sweep
                await self._perception_hub.perceive(
                    query="AI agents, sovereign runtime, model councils"
                )
            except _PERCEPTION_SCHEDULER_RECOVERABLE_ERRORS as e:
                record_degradation(
                    "perception_scheduler",
                    e,
                    action="continued scheduled perception loop after recoverable sweep failure",
                )
                logger.error("Error during scheduled perception sweep: %s", e)
            await asyncio.sleep(self.interval)
