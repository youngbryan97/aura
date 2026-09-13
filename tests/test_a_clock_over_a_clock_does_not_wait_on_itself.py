"""Installing a clock over another waited on itself, and took the process with it.

`ExperimentClock.install()` held its install lock and called `uninstall()` on
the clock already installed, which takes the same lock, and the lock is not
reentrant. Lockdep saw the second acquire and began a splat, stamped it with
`time.time()` under its own lock, and `time.time` was the old clock's `now()`,
which took a checked lock and went back into lockdep. Every thread that asked
the time parked behind it.

It showed up as the 49th test of a subject-core selection hanging with every
thread in a condition wait, and only when an earlier test had left a clock up.
A campaign runs with its clock installed for hours, so any lockdep splat in
that time would have frozen the run the same way.
"""

from __future__ import annotations

import threading
import time

import pytest

from core.subject.clock import ExperimentClock, installed_clock

pytestmark = pytest.mark.unit


def _finishes_within(seconds: float, action) -> bool:
    """Run `action` on a thread and say whether it returned in time.

    On a thread so that a regression fails this test instead of hanging the
    whole run, which is how the defect first presented.
    """
    done = threading.Event()
    errors: list[BaseException] = []

    def body() -> None:
        try:
            action()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=body, daemon=True, name="clock-install-probe").start()
    finished = done.wait(seconds)
    if errors:
        raise errors[0]
    return finished


def test_installing_a_clock_over_another_returns() -> None:
    real = time.time
    first = ExperimentClock(0.01, start=1000.0)
    second = ExperimentClock(0.02, start=2000.0)
    first.install()
    if not _finishes_within(5.0, second.install):
        pytest.fail(
            "installing a clock over an installed one waited on itself; the "
            "install lock is still held, so nothing here can be taken down"
        )
    try:
        assert installed_clock() is second
        assert time.time() == pytest.approx(2000.0)
    finally:
        second.uninstall()
        first.uninstall()
    assert time.time is real
    assert installed_clock() is None


@pytest.mark.parametrize("order", ["newest_first", "oldest_first"])
def test_either_order_of_taking_them_down_leaves_the_real_clock(order: str) -> None:
    real = time.time
    first = ExperimentClock(0.01, start=1000.0)
    second = ExperimentClock(0.02, start=2000.0)
    first.install()
    assert _finishes_within(5.0, second.install)
    for clock in ((second, first) if order == "newest_first" else (first, second)):
        clock.uninstall()
    assert time.time is real
    assert installed_clock() is None


def test_reading_the_clock_takes_no_lock() -> None:
    """Once installed, `now()` is `time.time` for the whole process.

    Lockdep reads the time while holding its own lock, so a lock taken here is
    a lock taken from inside lockdep. The clock's lock is replaced with one that
    refuses to be taken, and reading still works.
    """

    class _Refuses:
        def __enter__(self):
            raise AssertionError("now() took a lock")

        def __exit__(self, *exc):
            return False

        def acquire(self, *args, **kwargs):
            raise AssertionError("now() took a lock")

        def release(self):
            raise AssertionError("now() released a lock")

    clock = ExperimentClock(0.01, start=5.0)
    clock._lock = _Refuses()
    assert clock.now() == 5.0
    assert clock() == 5.0
