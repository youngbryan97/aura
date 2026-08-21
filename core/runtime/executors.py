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
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("Aura.Executors")

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

DURABLE_RECEIPT_POOL = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="aura-durable-receipt",
)

_POOL_REBUILD_LOCK = threading.Lock()


def _live_pool(kind: str) -> ThreadPoolExecutor:
    """Return the shared pool for *kind*, rebuilding it if it was shut down.

    The pools are process-lifetime resources, but shutdown hygiene closes
    them as registered resources. When the process then CONTINUES serving
    (an aborted shutdown; hermetic tests that drive hygiene teardown), every
    later client died with 'cannot schedule new futures after shutdown' —
    the order-dependence family across the attention gates, lag budgets,
    and executor lifecycle (2026-07-12). A shut-down pool in a process that
    keeps running earns a fresh pool; real shutdown is still refused by the
    latch check in _register_pool.
    """
    global HEAVY_CPU_POOL, BLOCKING_IO_POOL, DURABLE_RECEIPT_POOL
    pool = (
        HEAVY_CPU_POOL
        if kind == "heavy_cpu"
        else DURABLE_RECEIPT_POOL
        if kind == "durable_receipt"
        else BLOCKING_IO_POOL
    )
    if not getattr(pool, "_shutdown", False):
        return pool
    with _POOL_REBUILD_LOCK:
        pool = (
            HEAVY_CPU_POOL
            if kind == "heavy_cpu"
            else DURABLE_RECEIPT_POOL
            if kind == "durable_receipt"
            else BLOCKING_IO_POOL
        )
        if getattr(pool, "_shutdown", False):
            if kind == "heavy_cpu":
                HEAVY_CPU_POOL = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="aura-heavy-cpu"
                )
                pool = HEAVY_CPU_POOL
            elif kind == "durable_receipt":
                DURABLE_RECEIPT_POOL = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="aura-durable-receipt"
                )
                pool = DURABLE_RECEIPT_POOL
            else:
                BLOCKING_IO_POOL = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="aura-blocking-io"
                )
                pool = BLOCKING_IO_POOL
            logger.warning(
                "Rebuilt %s pool after external shutdown while the process "
                "continued serving.",
                kind,
            )
    return pool


def _register_pool(pool: ThreadPoolExecutor, *, name: str) -> None:
    from core.runtime.shutdown_coordinator import is_shutdown_requested

    if is_shutdown_requested():
        raise RuntimeError("runtime_shutdown")
    from core.runtime.runtime_hygiene import get_runtime_hygiene

    get_runtime_hygiene().register_shutdown_resource(
        pool,
        kind="executor",
        name=name,
        source="core.runtime.executors",
        closer=functools.partial(pool.shutdown, wait=False, cancel_futures=True),
        timeout_s=1.0,
        required=True,
    )


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def run_heavy_cpu[T](
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
    pool = _live_pool("heavy_cpu")
    _register_pool(pool, name="heavy_cpu_thread_pool")
    loop = asyncio.get_running_loop()
    tag = label or getattr(fn, "__qualname__", str(fn))

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                pool,
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
    except TimeoutError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning(
            "Heavy CPU work '%s' timed out after %.1f ms (budget %.0f ms)",
            tag, elapsed, timeout_s * 1000,
        )
        raise


async def run_blocking_io[T](
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
    pool = _live_pool("blocking_io")
    _register_pool(pool, name="blocking_io_thread_pool")
    loop = asyncio.get_running_loop()
    tag = label or getattr(fn, "__qualname__", str(fn))

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                pool,
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
    except TimeoutError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning(
            "Blocking IO '%s' timed out after %.1f ms (budget %.0f ms)",
            tag, elapsed, timeout_s * 1000,
        )
        raise


async def run_durable_receipt_io[T](
    fn: Callable[..., T],
    *args: Any,
    timeout_s: float = 10.0,
    label: str = "",
    **kwargs: Any,
) -> T:
    """Serialize user-facing durable receipts outside shared I/O lanes.

    A completed answer must not wait behind model probes, filesystem scans, or
    other default-executor work before its audit receipt becomes durable. One
    worker preserves receipt ordering; ReceiptStore still owns the database,
    process lock, audit-chain append, and durability policy.
    """

    pool = _live_pool("durable_receipt")
    _register_pool(pool, name="durable_receipt_thread_pool")
    loop = asyncio.get_running_loop()
    tag = label or getattr(fn, "__qualname__", str(fn))
    started = time.monotonic()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                pool,
                functools.partial(fn, *args, **kwargs),
            ),
            timeout=timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "Durable receipt IO '%s' timed out after %.1f ms (budget %.0f ms)",
            tag,
            (time.monotonic() - started) * 1000.0,
            timeout_s * 1000.0,
        )
        raise


def submit_blocking_io[T](
    fn: Callable[..., T],
    *args: Any,
    label: str = "",
    **kwargs: Any,
) -> Future[T]:
    """Submit bounded-owner blocking I/O from synchronous or worker code.

    Unlike :func:`run_blocking_io`, this does not require a running asyncio
    loop. The shared pool remains a registered shutdown resource, and callers
    must bound their own submission rate. This is the correct path for sync
    hot paths that enqueue one coalesced background flush.
    """
    pool = _live_pool("blocking_io")
    _register_pool(pool, name="blocking_io_thread_pool")
    tag = label or getattr(fn, "__qualname__", str(fn))

    def _run() -> T:
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = (time.monotonic() - started) * 1000.0
            if elapsed > 1000.0:
                logger.info("Blocking IO '%s' completed in %.1f ms", tag, elapsed)

    return pool.submit(_run)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def pool_status() -> dict[str, Any]:
    """Return worker status for both pools (for dashboards/health checks)."""
    def _stats(pool: ThreadPoolExecutor, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "max_workers": pool._max_workers,
            "threads_alive": len([t for t in (pool._threads or set()) if t.is_alive()]),
            "pending_items": pool._work_queue.qsize() if hasattr(pool, "_work_queue") else -1,
        }

    return {
        "heavy_cpu": _stats(HEAVY_CPU_POOL, "heavy_cpu"),
        "blocking_io": _stats(BLOCKING_IO_POOL, "blocking_io"),
        "durable_receipt": _stats(DURABLE_RECEIPT_POOL, "durable_receipt"),
    }


def shutdown_pools(wait: bool = False) -> None:
    """Gracefully shut down all bounded runtime pools."""
    HEAVY_CPU_POOL.shutdown(wait=wait, cancel_futures=True)
    BLOCKING_IO_POOL.shutdown(wait=wait, cancel_futures=True)
    DURABLE_RECEIPT_POOL.shutdown(wait=wait, cancel_futures=True)
    logger.info("Executor pools shut down (wait=%s).", wait)
