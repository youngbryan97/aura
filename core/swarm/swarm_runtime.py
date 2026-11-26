"""core/swarm/swarm_runtime.py — Swarm Job Orchestration.

Dispatches distributed tasks across sandboxed worker pools.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.swarm.worker_pool import LocalWorkerPool, WorkerProposal, WorkerType

logger = logging.getLogger("Aura.SwarmRuntime")


class SwarmRuntime:
    """Manages long-running parallel tasks distributed across local and remote workers."""

    def __init__(self) -> None:
        self.pool = LocalWorkerPool()

    async def dispatch_mission_tasks(
        self,
        task_specs: list[dict[str, Any]],
    ) -> list[WorkerProposal]:
        """Distribute objective tasks to specialized workers."""
        logger.info("SwarmRuntime: dispatching %d tasks to workers", len(task_specs))

        jobs = []
        for spec in task_specs:
            wtype_str = spec.get("worker_type", "critic")
            try:
                wtype = WorkerType(wtype_str)
            except ValueError:
                wtype = WorkerType.CRITIC

            jobs.append(self.pool.run_worker_job(wtype, spec.get("payload", {})))

        results = await asyncio.gather(*jobs)
        return list(results)

    @property
    def active_workers(self) -> dict[str, Any]:
        return self.pool.active_workers
