"""Tests for event-loop lag budget and bounded executors.

Verifies that:
  - Heavy CPU work does not block the event loop
  - Executor pools are correctly bounded
  - Timeouts are enforced
  - Pool status is queryable
"""
from __future__ import annotations

import asyncio
import threading
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
    assert "durable_receipt" in status
    assert status["heavy_cpu"]["max_workers"] == 2
    assert status["blocking_io"]["max_workers"] == 4
    assert status["durable_receipt"]["max_workers"] == 1


@pytest.mark.asyncio
async def test_durable_receipt_lane_isolated_from_saturated_blocking_io():
    """User-visible delivery evidence cannot queue behind unrelated scans."""

    from core.runtime import executors

    blockers = []
    release = threading.Event()

    def occupy() -> None:
        while not release.is_set():
            time.sleep(0.005)

    blocking_pool = executors._live_pool("blocking_io")
    for _ in range(4):
        blockers.append(blocking_pool.submit(occupy))
    try:
        result = await executors.run_durable_receipt_io(
            lambda: "durable",
            timeout_s=0.5,
        )
        assert result == "durable"
    finally:
        release.set()
        for blocker in blockers:
            blocker.result(timeout=1.0)


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


def test_hypervisor_explains_a_failed_liveness_check_instead_of_naming_the_probe():
    """`is_alive() returned False` names the probe, not the fault.

    Two very different faults make the hypervisor's liveness False: the
    watchdog task is gone, or the watchdog is running perfectly and reporting
    event-loop lag that has not yet re-confirmed healthy. Health reported both
    as "hypervisor (is_alive() returned False)", which reads as a dead thread —
    measured live on a runtime whose watchdog was alive and supervising, and it
    sent the investigation straight at thread liveness.
    """
    import asyncio

    from core.ops.hypervisor import Hypervisor

    hv = Hypervisor()

    # Never started.
    assert hv.is_alive() is False
    assert "not running" in hv.liveness_failure_reason()

    async def scenario():
        await hv.start()
        try:
            assert hv.is_alive() is True
            # Healthy: nothing to explain.
            assert hv.liveness_failure_reason() == ""

            # Alive, but carrying an unrecovered severe-lag verdict.
            hv._last_severe_lag_at = time.time()
            hv._last_failure_reason = "severe event-loop lag 8.411s"
            hv._healthy_lag_samples_after_failure = 0
            hv._last_lag = 0.004
            assert hv.is_alive() is False
            reason = hv.liveness_failure_reason()
            assert "alive and supervising" in reason, reason
            assert "8.411s" in reason, reason
            assert "0/3 healthy samples" in reason, reason
            assert "current lag 0.004s" in reason, reason
        finally:
            await hv.stop()

    asyncio.run(scenario())

    # Stopped: back to naming the real cause.
    assert "not running" in hv.liveness_failure_reason()


def test_health_contract_prefers_a_services_own_liveness_explanation():
    from core.runtime import health_contract

    requirement = health_contract.ServiceRequirement(
        "Widget",
        "widget",
        health_contract.ServiceTier.IMPORTANT,
        "test widget",
        liveness_check="is_alive",
        liveness_reason_check="liveness_failure_reason",
    )

    class _Explains:
        def is_alive(self):
            return False

        def liveness_failure_reason(self):
            return "widget spindle unseated"

    class _Silent:
        def is_alive(self):
            return False

    class _Raises:
        def is_alive(self):
            return False

        def liveness_failure_reason(self):
            raise RuntimeError("explaining is broken")

    assert (
        health_contract._liveness_failure_reason(_Explains(), requirement)
        == "widget spindle unseated"
    )
    # No explanation available, or a broken one, must fall back — never raise.
    assert health_contract._liveness_failure_reason(_Silent(), requirement) == ""
    assert health_contract._liveness_failure_reason(_Raises(), requirement) == ""
    assert health_contract._liveness_failure_reason(None, requirement) == ""

    without_reason = health_contract.ServiceRequirement(
        "Widget", "widget", health_contract.ServiceTier.IMPORTANT, "t",
        liveness_check="is_alive",
    )
    assert health_contract._liveness_failure_reason(_Explains(), without_reason) == ""
