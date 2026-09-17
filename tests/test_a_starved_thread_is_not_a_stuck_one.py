"""One reading tells a blocked thread from a starved one from a busy one.

A watchdog on its own thread could read only the wall clock, and on a host
three times oversubscribed it dumped eighty stacks a minute for a loop that
was running, just slowly (2026-09-16, 33 dumps an hour). The loop thread's
CPU time, read from the watchdog's thread, is the measurement it lacked.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from core.runtime.thread_cpu import thread_cpu_seconds, thread_cpu_share

pytestmark = pytest.mark.skipif(
    not (sys.platform == "darwin" or sys.platform.startswith("linux")),
    reason="per-thread CPU is read through mach or /proc",
)


def test_a_thread_reading_itself_matches_the_interpreters_clock():
    ours = thread_cpu_seconds(threading.get_ident(), native_id=threading.get_native_id())
    assert ours is not None
    assert abs(ours - time.thread_time()) < 0.05


def test_a_busy_thread_and_a_sleeping_one_read_differently():
    stop = threading.Event()

    def busy():
        x = 0
        while not stop.is_set():
            x += 1

    def sleeper():
        while not stop.is_set():
            time.sleep(0.02)

    busy_thread = threading.Thread(target=busy, daemon=True)
    sleeping_thread = threading.Thread(target=sleeper, daemon=True)
    busy_thread.start()
    sleeping_thread.start()
    try:
        time.sleep(0.1)
        b0 = thread_cpu_seconds(busy_thread.ident, native_id=busy_thread.native_id)
        s0 = thread_cpu_seconds(sleeping_thread.ident, native_id=sleeping_thread.native_id)
        started = time.monotonic()
        time.sleep(0.5)
        wall = time.monotonic() - started
        b1 = thread_cpu_seconds(busy_thread.ident, native_id=busy_thread.native_id)
        s1 = thread_cpu_seconds(sleeping_thread.ident, native_id=sleeping_thread.native_id)
    finally:
        stop.set()
    busy_share = thread_cpu_share(b0, b1, wall)
    sleeping_share = thread_cpu_share(s0, s1, wall)
    assert busy_share is not None and sleeping_share is not None
    # A busy thread gets a real share even on a loaded host; a sleeping one
    # gets almost none. The gap is the measurement.
    assert busy_share > 0.05
    assert sleeping_share < 0.05
    assert busy_share > 5 * sleeping_share


def test_an_unreadable_thread_reads_none():
    assert thread_cpu_share(None, 1.0, 1.0) is None
    assert thread_cpu_share(1.0, 1.5, 0.0) is None


def test_the_watchdog_tells_a_starved_loop_from_a_stuck_one(monkeypatch, caplog):
    """A loop thread on a fifth of a core is running, and the watchdog says so
    once a minute without a dump. One on no CPU is blocked and gets its dump.
    One on most of a core is doing on-loop work and gets its dump too."""
    import logging

    from core.resilience import stall_watchdog as sw

    watchdog = sw.StallWatchdog.__new__(sw.StallWatchdog)
    watchdog._loop_thread_id = 12345
    watchdog._loop_cpu_at_heartbeat = 100.0
    watchdog._starved_stalls = 0
    watchdog._last_starvation_log_at = 0.0
    watchdog._last_stall_dump_at = 0.0
    watchdog._last_stall_dump_path = ""
    watchdog._suppressed_stall_dumps = 0
    reported: list[tuple[float, float | None]] = []
    monkeypatch.setattr(
        sw.StallWatchdog,
        "_report_stall",
        lambda self, elapsed, share=None: reported.append((elapsed, share)),
    )

    # Starved: 0.8s of CPU over 10s of wall is 8% of a core.
    monkeypatch.setattr(sw, "thread_cpu_seconds", lambda ident: 100.8)
    share = watchdog._loop_cpu_share_since_heartbeat(10.0)
    assert share is not None and sw.LOOP_BLOCKED_CEILING_FRACTION <= share < sw.LOOP_HOLD_STARVED_FRACTION
    with caplog.at_level(logging.WARNING):
        watchdog._report_starvation(10.0, share)
    assert watchdog._starved_stalls == 1
    assert any("starved, not stuck" in r.getMessage() for r in caplog.records)
    assert reported == []

    # Blocked: no CPU at all.
    monkeypatch.setattr(sw, "thread_cpu_seconds", lambda ident: 100.0)
    assert watchdog._loop_cpu_share_since_heartbeat(10.0) == 0.0

    # Busy: most of a core.
    monkeypatch.setattr(sw, "thread_cpu_seconds", lambda ident: 109.0)
    assert watchdog._loop_cpu_share_since_heartbeat(10.0) == pytest.approx(0.9)

    # Unreadable: the wall clock alone, as before.
    watchdog._loop_cpu_at_heartbeat = None
    assert watchdog._loop_cpu_share_since_heartbeat(10.0) is None


def test_the_watchdog_loop_skips_dump_and_recovery_for_a_starved_stall():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "core/resilience/stall_watchdog.py"
    ).read_text(encoding="utf-8")
    loop = source[source.index("# Check for stall") :]
    loop = loop[: loop.index("def stop(self)")]
    assert "share = self._loop_cpu_share_since_heartbeat(elapsed)" in loop
    assert "LOOP_BLOCKED_CEILING_FRACTION <= share < LOOP_HOLD_STARVED_FRACTION" in loop
    starved = loop[loop.index("self._report_starvation(elapsed, share)") :]
    starved = starved[: starved.index("continue")]
    assert "_attempt_active_recovery" not in starved
    assert "_report_stall" not in starved


@pytest.mark.asyncio
async def test_the_loop_monitor_files_no_breach_for_a_starved_sample(monkeypatch, caplog):
    """LIVE 2026-09-16: 52 CRITICAL event_loop_monitor degradations in 22
    minutes for a loop thread on a fifth of a core. Its own clock over the
    sample says the lag was the host's."""
    import logging

    from core.utils import concurrency as cc

    monitor = cc.EventLoopMonitor(interval=1.0, threshold=0.75)
    monitor._started_at = time.perf_counter() - 600.0
    monitor.startup_grace = 0.0

    # One sleep of 6.0s wall on 0.3s of the loop thread's CPU: 5% of a core.
    walls = iter([0.0, 6.0])
    cpus = iter([100.0, 100.3])
    # cc.time is the time module itself; the fakes stay total so nothing that
    # runs after the sample (fixture teardown included) meets an exhausted one.
    monkeypatch.setattr(cc.time, "perf_counter", lambda: next(walls, 6.0))
    monkeypatch.setattr(cc.time, "thread_time", lambda: next(cpus, 100.3))
    monkeypatch.setattr(monitor, "_lag_threshold_for_context", lambda: (0.75, "idle"))

    async def one_sleep(_seconds):
        monitor._stop_event.set()

    monkeypatch.setattr(cc.asyncio, "sleep", one_sleep)
    recorded = []
    monkeypatch.setattr(cc, "record_degradation", lambda *a, **k: recorded.append(a))
    with caplog.at_level(logging.WARNING):
        await monitor._run()
    assert recorded == [], "a starved loop is not an event_loop_monitor failure"
    assert monitor._consecutive_breaches == 0
    assert monitor._starved_samples == 1
    assert any("starved, not blocked" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_hypervisor_opens_no_freeze_for_a_starved_sample(monkeypatch, caplog):
    """LIVE 2026-09-16: 38 SEVERE FREEZE criticals in 22 minutes, same cause."""
    import logging
    from types import SimpleNamespace

    from core.container import ServiceContainer
    from core.ops import hypervisor as hv

    ServiceContainer.clear()
    try:
        hypervisor = hv.Hypervisor(lag_threshold_s=1.5)
        hypervisor._running = True
        hypervisor._start_time = time.time() - 600.0
        hypervisor._required_severe_lag_samples = 1
        monkeypatch.setattr(hypervisor, "_lag_threshold_for_context", lambda: (1.5, "idle"))
        monos = iter([0.0, 9.0])
        cpus = iter([50.0, 50.4])  # 0.4s of CPU over 9s: 4% of a core
        monkeypatch.setattr(hv, "_monotonic_now", lambda: next(monos, 9.0))
        monkeypatch.setattr(hv.time, "thread_time", lambda: next(cpus, 50.4))

        async def one_sleep(_seconds):
            hypervisor._running = False

        monkeypatch.setattr(hv.asyncio, "sleep", one_sleep)
        monkeypatch.setattr(
            hv, "get_resource_observer", lambda: SimpleNamespace(process=lambda _pid: None)
        )
        recorded = []
        monkeypatch.setattr(hv, "record_degradation", lambda *a, **k: recorded.append(a))
        with caplog.at_level(logging.WARNING, logger="Aura.Hypervisor"):
            await hypervisor._watchdog_loop()
        assert recorded == [], "a starved loop is not a hypervisor freeze"
        assert hypervisor._severe_lag_streak == 0
        assert hypervisor._starved_samples == 1
        assert not [r for r in caplog.records if "SEVERE FREEZE" in r.getMessage()]
        assert any("starved, not frozen" in r.getMessage() for r in caplog.records)
    finally:
        ServiceContainer.clear()
