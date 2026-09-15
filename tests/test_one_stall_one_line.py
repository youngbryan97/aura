"""One stall of the loop is one line in the feed.

The hypervisor and the EventLoopMonitor each slept a second on the same loop
and each announced what they measured, so every stall was two warnings —
"HIGH EVENT LOOP LAG" from one and "EVENT LOOP LAG DETECTED" from the other —
and neither said anything the other did not. The monitor is the instrument;
the hypervisor reads its sample and keeps its own measurement for a runtime
that booted without one.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.ops.hypervisor import Hypervisor
from core.utils.concurrency import EventLoopMonitor


@pytest.fixture(autouse=True)
def _no_monitor_registered():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def test_the_monitors_sample_is_the_reading_and_it_is_already_announced(monkeypatch):
    monitor = EventLoopMonitor()
    monitor._capture_lag_sample(3.5, sampled_monotonic=time.perf_counter())
    ServiceContainer.register_instance("event_loop_monitor", monitor, required=False)

    lag, announced = Hypervisor()._lag_reading(0.2)
    assert lag == pytest.approx(3.5)
    assert announced is True


def test_a_stale_sample_hands_the_reading_back_to_the_hypervisor():
    monitor = EventLoopMonitor(interval=1.0)
    monitor._capture_lag_sample(3.5, sampled_monotonic=time.perf_counter() - 10.0)
    ServiceContainer.register_instance("event_loop_monitor", monitor, required=False)

    lag, announced = Hypervisor()._lag_reading(1.9)
    assert lag == pytest.approx(1.9)
    assert announced is False


def test_no_monitor_means_the_hypervisor_measures_and_announces():
    lag, announced = Hypervisor()._lag_reading(1.9)
    assert lag == pytest.approx(1.9)
    assert announced is False


@pytest.mark.asyncio
async def test_the_watchdog_does_not_warn_for_a_lag_the_monitor_announced(monkeypatch, caplog):
    monitor = EventLoopMonitor()
    monitor._capture_lag_sample(3.5, sampled_monotonic=time.perf_counter())
    ServiceContainer.register_instance("event_loop_monitor", monitor, required=False)
    hypervisor = Hypervisor(lag_threshold_s=1.5)
    hypervisor._running = True
    hypervisor._start_time = time.time() - 600.0
    monkeypatch.setattr(hypervisor, "_lag_threshold_for_context", lambda: (1.5, "idle"))

    calls = {"n": 0}

    async def one_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] >= 1:
            hypervisor._running = False

    monkeypatch.setattr("core.ops.hypervisor.asyncio.sleep", one_sleep)
    monkeypatch.setattr(
        "core.ops.hypervisor.get_resource_observer",
        lambda: SimpleNamespace(process=lambda _pid: None),
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="Aura.Hypervisor"):
        await hypervisor._watchdog_loop()
    assert hypervisor._last_lag == pytest.approx(3.5)
    assert not [r for r in caplog.records if "HIGH EVENT LOOP LAG" in r.getMessage()]
