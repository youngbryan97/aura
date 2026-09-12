"""Lockdep stamped a splat with `time.time()` while it held its own lock.

`time.time` is replaceable, and the Subject Core replaces it for a whole
campaign. The experiment clock's `now()` took a checked lock, so the stamp went
back into the validator from inside the validator, and the validator's lock is
not reentrant. The first splat raised while a clock was installed parked every
thread that asked the time.

The validator now stamps with the wall clock it captured when it loaded. This
drives a real splat on a validator of its own, with `time.time` replaced by a
function that goes back into that same validator, which is the shape of the
original hang. It runs on a thread with a deadline so a regression fails here
instead of hanging the suite.
"""

from __future__ import annotations

import threading
import time

import pytest

from core.runtime import lockdep
from core.runtime.lockdep import LockdepValidator, LockRank

pytestmark = pytest.mark.unit


def test_a_splat_stamped_under_the_lock_does_not_call_a_replaced_time(monkeypatch) -> None:
    validator = LockdepValidator()
    real_time = time.time
    probe: dict[str, int] = {"ident": -1, "under_the_lock": 0, "outside_it": 0}

    def reenters_the_validator() -> float:
        # What an experiment clock with a checked lock did: reading the time
        # acquires a lock, and acquiring a lock goes through the validator.
        # Whether the validator's own lock is held is read first. A call made
        # while it is held is the hang; a call made after it is released, from
        # the report's log line, is ordinary and allowed.
        if threading.get_ident() == probe["ident"]:
            if validator._lock.locked():
                probe["under_the_lock"] += 1
                return real_time()
            probe["outside_it"] += 1
            validator.on_acquire("probe.clock", rank=LockRank.UNRANKED, is_async=False, reentrant=True)
            validator.on_release("probe.clock", is_async=False)
        return real_time()

    reported: list[str] = []
    monkeypatch.setattr(lockdep, "taint", lambda *args, **kwargs: reported.append("taint"))
    monkeypatch.setattr(
        "core.runtime.errors.record_degradation",
        lambda *args, **kwargs: reported.append("degradation"),
    )
    monkeypatch.setattr(time, "time", reenters_the_validator)

    done = threading.Event()
    errors: list[BaseException] = []

    def acquire_twice() -> None:
        probe["ident"] = threading.get_ident()
        try:
            validator.on_acquire("probe.plain", rank=LockRank.UNRANKED, is_async=False, reentrant=False)
            # The same non-reentrant name again: a self-deadlock splat, which is
            # stamped while the validator's lock is held.
            validator.on_acquire("probe.plain", rank=LockRank.UNRANKED, is_async=False, reentrant=False)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test's thread
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=acquire_twice, daemon=True, name="lockdep-stamp-probe").start()
    if not done.wait(5.0):
        pytest.fail(
            "stamping a splat went back into the validator under its own lock; "
            "every thread that asks the time is now parked behind it"
        )
    if errors:
        raise errors[0]

    splats = validator.splats()
    assert [s.kind for s in splats] == ["self_deadlock"]
    assert splats[0].at > 1e9, "the stamp is not a wall-clock instant"
    assert probe["under_the_lock"] == 0, "the stamp read the replaceable clock under the lock"
    assert "taint" in reported


def test_the_captured_clock_is_the_machine_clock() -> None:
    """Captured before anything could replace it, which is the whole point."""
    assert abs(lockdep._wall_time() - time.time()) < 5.0
    assert lockdep._wall_time is not getattr(time.time, "__self__", None)
