"""A hold is judged by progress, not by the clock.

LIVE, 2026-09-21: twenty-nine of these in one evening.

    🚨 DEADLOCK ALERT: Lock 'AuraKernel.StateLock' (ID: b78f35cc) held for 438.8s!
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


def test_the_loop_checks_progress_before_it_alerts():
    """The carve-out has to sit before the alert and before the recovery."""
    import inspect

    source = inspect.getsource(lw)
    source = source[source.index("async def _monitor_loop") :]
    working = source.index("_the_holder_is_working()")
    alert = source.index("DEADLOCK ALERT")
    recovery = source.index("_attempt_recovery")
    assert working < alert < recovery, (
        "a hold that is working must be excused before it is alerted on and "
        "before anything force-releases it"
    )


def test_the_excused_path_does_not_record_a_degraded_event():
    import inspect

    source = inspect.getsource(lw)
    source = source[source.index("async def _monitor_loop") :]
    excused = source[source.index("_the_holder_is_working()") :]
    excused = excused[: excused.index("DEADLOCK ALERT")]
    assert "record_degraded_event" not in excused
    assert "continue" in excused
