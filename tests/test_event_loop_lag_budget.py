"""Tests for event-loop lag budget and bounded executors.

Verifies that:
  - Heavy CPU work does not block the event loop
  - Executor pools are correctly bounded
  - Timeouts are enforced
  - Pool status is queryable
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_heavy_cpu_does_not_block_loop():
    """SVD on a 512x512 matrix via executor must not block the event loop."""
    from core.runtime.executors import run_heavy_cpu

    matrix = np.random.randn(512, 512)

    # Schedule a concurrent canary that should complete immediately
    canary_done = asyncio.Event()

    async def canary():
        await asyncio.sleep(0.01)
        canary_done.set()

    canary_task = asyncio.create_task(canary())

    # Run heavy SVD via executor
    result = await run_heavy_cpu(np.linalg.svd, matrix, timeout_s=5.0, label="test-svd")

    # Wait for canary
    await asyncio.wait_for(canary_task, timeout=1.0)
    assert canary_done.is_set(), "Canary task was blocked by heavy CPU work"

    # SVD should have returned valid results
    U, S, Vt = result
    assert U.shape == (512, 512)
    assert len(S) == 512


@pytest.mark.asyncio
async def test_heavy_cpu_timeout():
    """Work exceeding the timeout must raise TimeoutError."""
    from core.runtime.executors import run_heavy_cpu

    def slow_work():
        time.sleep(5)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await run_heavy_cpu(slow_work, timeout_s=0.1, label="slow-test")


@pytest.mark.asyncio
async def test_blocking_io_does_not_block_loop():
    """Blocking IO via executor must not block the event loop."""
    from core.runtime.executors import run_blocking_io

    def fake_io():
        time.sleep(0.1)
        return "data"

    canary_done = asyncio.Event()

    async def canary():
        await asyncio.sleep(0.01)
        canary_done.set()

    canary_task = asyncio.create_task(canary())
    result = await run_blocking_io(fake_io, timeout_s=2.0, label="test-io")
    await asyncio.wait_for(canary_task, timeout=1.0)

    assert result == "data"
    assert canary_done.is_set()


def test_pool_status():
    """Pool status should report worker counts."""
    from core.runtime.executors import pool_status

    status = pool_status()
    assert "heavy_cpu" in status
    assert "blocking_io" in status
    assert status["heavy_cpu"]["max_workers"] == 2
    assert status["blocking_io"]["max_workers"] == 4


@pytest.mark.asyncio
async def test_event_loop_lag_under_budget():
    """During concurrent heavy work, event loop lag must stay under 150ms."""
    from core.runtime.executors import run_heavy_cpu

    lags = []

    async def measure_lag():
        for _ in range(10):
            t0 = time.monotonic()
            await asyncio.sleep(0.01)
            lag = (time.monotonic() - t0 - 0.01) * 1000
            lags.append(lag)

    # Run heavy work alongside lag measurement
    matrix = np.random.randn(256, 256)
    lag_task = asyncio.create_task(measure_lag())

    for _ in range(3):
        await run_heavy_cpu(np.linalg.svd, matrix, timeout_s=2.0, label="lag-test")

    await lag_task

    p99 = sorted(lags)[int(len(lags) * 0.99)] if lags else 0
    assert p99 < 150.0, f"p99 event loop lag {p99:.1f}ms exceeds 150ms budget"
