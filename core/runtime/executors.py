"""core/runtime/executors.py — Bounded executor pools for CPU/IO offload.

The audit demands that no CPU-bound math or blocking IO runs on the
asyncio event loop.  This module provides two shared, bounded pools:

  HEAVY_CPU_POOL  — for SVD, PCA, phi sweeps, scale sweeps, semantic
                    defrag, neural ODEs, and similar O(N²)+ work.
                    Max 2 workers to avoid saturating compute during
                    LoRA training or MLX inference.

  BLOCKING_IO_POOL — for synchronous file IO, YAML/JSON loads, large
                     vector-DB queries, subprocess probes, and any
                     other call that would block the loop > 5 ms.
                     Max 4 workers.

Usage from any async context::

    from core.runtime.executors import run_heavy_cpu, run_blocking_io

    result = await run_heavy_cpu(np.linalg.svd, matrix, timeout_s=2.0)
    data   = await run_blocking_io(Path("big.json").read_text, timeout_s=5.0)

Both helpers copy the call into the appropriate pool, enforce a timeout,
and raise ``asyncio.TimeoutError`` if the worker exceeds the budget.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

logger = logging.getLogger("Aura.Executors")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

HEAVY_CPU_POOL = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="aura-heavy-cpu",
)

BLOCKING_IO_POOL = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="aura-blocking-io",
)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def run_heavy_cpu(
    fn: Callable[..., T],
    *args: Any,
    timeout_s: float = 2.0,
    label: str = "",
    **kwargs: Any,
) -> T:
    """Offload *fn* to the bounded CPU pool with timeout.

    Parameters
    ----------
    fn : callable
        Synchronous function to run (e.g. ``np.linalg.svd``).
    timeout_s : float
        Maximum wall-time before ``asyncio.TimeoutError``.
    label : str
        Optional human-readable label for logging.
    """
    loop = asyncio.get_running_loop()
    tag = label or getattr(fn, "__qualname__", str(fn))

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                HEAVY_CPU_POOL,
                functools.partial(fn, *args, **kwargs),
            ),
            timeout=timeout_s,
        )
        elapsed = (time.monotonic() - t0) * 1000
        if elapsed > 500:
            logger.info(
                "Heavy CPU work '%s' completed in %.1f ms (budget %.0f ms)",
                tag, elapsed, timeout_s * 1000,
            )
        return result
    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning(
            "Heavy CPU work '%s' timed out after %.1f ms (budget %.0f ms)",
            tag, elapsed, timeout_s * 1000,
        )
        raise


async def run_blocking_io(
    fn: Callable[..., T],
    *args: Any,
    timeout_s: float = 5.0,
    label: str = "",
    **kwargs: Any,
) -> T:
    """Offload *fn* to the bounded IO pool with timeout.

    Parameters
    ----------
    fn : callable
        Synchronous function that blocks on IO (file read, subprocess, etc).
    timeout_s : float
        Maximum wall-time before ``asyncio.TimeoutError``.
    label : str
        Optional human-readable label for logging.
    """
    loop = asyncio.get_running_loop()
    tag = label or getattr(fn, "__qualname__", str(fn))

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                BLOCKING_IO_POOL,
                functools.partial(fn, *args, **kwargs),
            ),
            timeout=timeout_s,
        )
        elapsed = (time.monotonic() - t0) * 1000
        if elapsed > 1000:
            logger.info(
                "Blocking IO '%s' completed in %.1f ms (budget %.0f ms)",
                tag, elapsed, timeout_s * 1000,
            )
        return result
    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning(
            "Blocking IO '%s' timed out after %.1f ms (budget %.0f ms)",
            tag, elapsed, timeout_s * 1000,
        )
        raise


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def pool_status() -> dict:
    """Return worker status for both pools (for dashboards/health checks)."""
    def _stats(pool: ThreadPoolExecutor, name: str) -> dict:
        return {
            "name": name,
            "max_workers": pool._max_workers,
            "threads_alive": len([t for t in (pool._threads or set()) if t.is_alive()]),
            "pending_items": pool._work_queue.qsize() if hasattr(pool, "_work_queue") else -1,
        }

    return {
        "heavy_cpu": _stats(HEAVY_CPU_POOL, "heavy_cpu"),
        "blocking_io": _stats(BLOCKING_IO_POOL, "blocking_io"),
    }


def shutdown_pools(wait: bool = False) -> None:
    """Gracefully shutdown both pools.  Called during runtime teardown."""
    HEAVY_CPU_POOL.shutdown(wait=wait, cancel_futures=True)
    BLOCKING_IO_POOL.shutdown(wait=wait, cancel_futures=True)
    logger.info("Executor pools shut down (wait=%s).", wait)
