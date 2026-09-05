"""core/swarm/ray_backend.py — Distributed Swarm Ray Integration.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SwarmRay")
_RAY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    ray = None
    _RAY_AVAILABLE = False


class RayBackend:
    """Interfaces with a Ray distributed cluster to dispatch tasks across worker nodes."""

    def __init__(self) -> None:
        self.active = False
        if _RAY_AVAILABLE:
            try:
                # Eagerly initialize ray if not already initialized — with
                # BOUNDED resources. Ray's defaults reserve ~30% of host RAM
                # for the object store; on the 64GB host already carrying a
                # wired 32B model that is a memory bomb, not a default.
                if not ray.is_initialized():
                    from core.runtime.flags import FlagKind, declare

                    cpus = max(1, min(int(declare(
                        "AURA_RAY_CPUS", kind=FlagKind.INT, default=2,
                        description="Ray worker CPUs (capped 8)",
                        owner="core.swarm.ray_backend",
                    ).value()), 8))
                    store_mb = max(128, min(int(declare(
                        "AURA_RAY_OBJECT_STORE_MB", kind=FlagKind.INT, default=256,
                        description="Ray object store MB (capped 2048)",
                        owner="core.swarm.ray_backend",
                    ).value()), 2048))
                    ray.init(
                        ignore_reinit_error=True,
                        num_cpus=cpus,
                        object_store_memory=store_mb * 1024 * 1024,
                        include_dashboard=False,
                        logging_level=logging.WARNING,
                    )
                self.active = True
                logger.info("⚡ Ray distributed backend connected successfully.")
            except _RAY_RECOVERABLE_ERRORS as e:
                record_degradation(
                    "ray_backend",
                    e,
                    action="used local thread execution after optional Ray backend initialization failed",
                )
                logger.warning("Failed to initialize Ray cluster: %s. Falling back to local.", e)

    def is_available(self) -> bool:
        return self.active

    async def execute_parallel(self, tasks: list[Callable[[], Any]]) -> list[Any]:
        """Dispatches tasks in parallel across Ray actors."""
        if not self.active:
            # Fallback local execute
            import asyncio
            futures = [asyncio.to_thread(t) for t in tasks]
            return await asyncio.gather(*futures)

        # Ray remote task execution
        @ray.remote
        def ray_task_runner(fn: Callable[[], Any]) -> Any:
            return fn()

        logger.info("⚡ Swarm Ray: dispatching %d tasks to cluster...", len(tasks))
        ray_refs = [ray_task_runner.remote(t) for t in tasks]
        # A worker exception propagates here as a RayTaskError (which
        # subclasses the original error type). The cluster and this driver
        # survive it — callers handle or re-raise as with any task failure.
        return ray.get(ray_refs)
