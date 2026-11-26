"""A hold is judged by progress, not by the clock.

LIVE, 2026-09-21: twenty-nine of these in one evening.

    DEADLOCK ALERT: Lock 'AuraKernel.StateLock' (ID: b78f35cc) held for 438.8s!
    StabilityGuardian: DEGRADED — lock_watchdog:1 active lock(s)

The lock was not deadlocked. `AuraKernel.tick` takes it, runs the phases,
and releases it in a `finally`; one of those phases was a generation on the
Brainstem, which on a loaded host takes minutes. Every alert was a CRITICAL
line, a degraded event at critical severity, and a DEGRADED card — and each
one was also eligible to fire a recovery callback that force-releases the
lock out from under live work.

The hold is real and worth knowing about. It is not a deadlock. This is the
same reading the cognitive engine renews its cycle from and the stall
watchdog uses for its starvation carve-out; the lock watchdog is the fourth
caller and the first that could have force-released something.
"""

from __future__ import annotations

import time

import pytest

from core.resilience import lock_watchdog as lw


@pytest.fixture
def a_long_hold():
    tracked = lw._TrackedLock(start_time=time.monotonic() - 400.0, name="AuraKernel.StateLock")
    return tracked


def test_a_turn_that_is_producing_excuses_the_hold(monkeypatch):
    import core.runtime.turn_progress as progress

    monkeypatch.setattr(progress, "still_producing", lambda **_k: True)
    monkeypatch.setattr(progress, "normal_gap_between_tokens", lambda *_a, **_k: 20.0)
    assert lw._the_holder_is_working() is True


def test_a_silent_turn_does_not(monkeypatch):
    import core.runtime.turn_progress as progress

    monkeypatch.setattr(progress, "still_producing", lambda **_k: False)
    monkeypatch.setattr(progress, "normal_gap_between_tokens", lambda *_a, **_k: 20.0)
    assert lw._the_holder_is_working() is False


def test_an_unreadable_signal_does_not_excuse_it(monkeypatch):
    """Unmeasurable is not working; a broken reading must not excuse forever."""
    import core.runtime.turn_progress as progress

    def _raises(**_k):
        raise RuntimeError("no progress state")

    monkeypatch.setattr(progress, "still_producing", _raises)
    monkeypatch.setattr(progress, "normal_gap_between_tokens", lambda *_a, **_k: 20.0)
    assert lw._the_holder_is_working() is False


def _one_pass_over_a_long_hold(monkeypatch, *, working: bool) -> dict:
    """Run the monitor loop over a lock held 400s, and say what it did.

    The loop runs until it has looked at the hold twice, so it has had every
    chance to alert and to force-release.
    """
    import asyncio
    import contextlib

    import core.health.degraded_events as degraded_events

    seen = {"looks": 0, "released": 0, "degraded": []}
    looked_twice = asyncio.Event()

    def holder_is_working() -> bool:
        seen["looks"] += 1
        if seen["looks"] >= 2:
            looked_twice.set()
        return working

    monkeypatch.setattr(lw, "_the_holder_is_working", holder_is_working)
    monkeypatch.setattr(
        degraded_events,
        "record_degraded_event",
        lambda *a, **k: seen["degraded"].append((a, k)),
    )

    def force_release() -> None:
        seen["released"] += 1

    # Its own watchdog: the process one keeps whatever interval it was first
    # built with, and at ten seconds its loop never looked inside this wait.
    watchdog = lw.LockWatchdog.__wrapped__(check_interval=0.01, threshold=180.0)
    watchdog.report_acquire_start("b78f35cc", "AuraKernel.StateLock", on_stall=force_release)
    watchdog._active_locks["b78f35cc"].start_time = time.monotonic() - 400.0

    async def run() -> None:
        watchdog._running = True
        task = asyncio.create_task(watchdog._monitor_loop())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(looked_twice.wait(), timeout=5.0)
        watchdog._running = False
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run())
    assert seen["looks"] >= 2, "the loop never reached the hold"
    return seen


def test_a_working_hold_is_neither_alerted_on_nor_released(monkeypatch, caplog):
    """The carve-out has to sit before the alert and before the recovery."""
    import logging

    with caplog.at_level(logging.INFO, logger=lw.logger.name):
        seen = _one_pass_over_a_long_hold(monkeypatch, working=True)
    assert seen["released"] == 0, (
        "a hold that is working was force-released out from under live work"
    )
    assert not [r for r in caplog.records if "DEADLOCK ALERT" in r.getMessage()]
    said = [r.getMessage() for r in caplog.records if "still producing" in r.getMessage()]
    assert said, "the hold was excused without a word"
    assert "(1 look(s))" in said[0]


def test_the_excused_path_does_not_record_a_degraded_event(monkeypatch):
    seen = _one_pass_over_a_long_hold(monkeypatch, working=True)
    assert seen["degraded"] == []


def test_a_silent_hold_is_still_alerted_on_and_released(monkeypatch):
    """The carve-out is for work; a hold nobody is producing through is still one."""
    seen = _one_pass_over_a_long_hold(monkeypatch, working=False)
    assert seen["released"] >= 1
    assert seen["degraded"], "a real stall reached no degraded event"
