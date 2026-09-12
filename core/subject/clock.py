"""The clock the arms share.

Two arms of an intervention have to see the same computation. They already see
the same state, the same organs, the same generators and the same host, and
they still saw different amounts of time: one arm's turn took two seconds and
the other's took two and a tenth, because that is what a machine does.

That difference is not small where a phase reads elapsed time. Drive levels
decay by `decay * dt` on every motivation update, and the unity layer binds a
mind-moment over a four-second window with a 1.2-second half-life. Half a
standard deviation of deliberation's whole ordinary spread, and one and a half
of the workspace's, came from nothing but how long the machine took.

So the run gets a clock of its own. `time.time` returns it, it advances by one
fixed step per frame, and a restore rewinds it with everything else. Every
phase then sees exactly the same interval in both arms.

`time.monotonic` is left alone on purpose: asyncio's timeouts read it, and a
timeout that cannot fire is a hang rather than a measurement.

The step is measured rather than chosen. A run times its own frames on the real
clock before installing this, and the number it found is part of the campaign
fingerprint, so two runs with the same fingerprint saw the same timeline.
"""

from __future__ import annotations

import time as _time
from typing import Any
from core.runtime.lockdep import checked_lock

__all__ = ["ExperimentClock", "installed_clock", "real_time"]

#: The real `time.time`, kept so the harness can still measure itself. A
#: run's duration, a timeout on the process, the stamp on an artefact: those
#: are the machine's questions and they take the machine's answer whether or
#: not the experiment's clock is installed.
real_time = _time.time

#: The process-wide installed clock, if any. One per process, because it
#: replaces a module-level function that every subsystem shares.
_INSTALLED: ExperimentClock | None = None
_INSTALL_LOCK = checked_lock("core.subject.clock._INSTALL_LOCK")


def installed_clock() -> ExperimentClock | None:
    """The clock the process is running on, or None for the real one."""
    return _INSTALLED


class ExperimentClock:
    """A monotone clock the experiment advances by hand."""

    def __init__(self, step: float, *, start: float | None = None) -> None:
        self.step = max(1e-6, float(step))
        self._now = float(start if start is not None else _time.time())
        self._lock = checked_lock("core.subject.clock.experiment_clock")
        self._real_time: Any = None

    # ── reading ──────────────────────────────────────────────────────────

    def now(self) -> float:
        # No lock. This becomes `time.time` for the whole process, so everything
        # that reads the wall clock calls it, lockdep included, and lockdep reads
        # the time while holding its own lock. A checked lock here went back
        # into lockdep from inside lockdep: the first splat raised while a clock
        # was installed parked every thread that asked the time. Reading a float
        # attribute is atomic, and only the harness writes one.
        return self._now

    def __call__(self) -> float:
        return self.now()

    # ── driving ──────────────────────────────────────────────────────────

    def advance(self, step: float | None = None) -> float:
        """Move the clock on by one frame, or by a stated amount."""
        with self._lock:
            self._now += self.step if step is None else max(0.0, float(step))
            return self._now

    def set(self, value: float) -> None:
        """Rewind or forward the clock, which is what a restore does."""
        with self._lock:
            self._now = float(value)

    # ── installing ───────────────────────────────────────────────────────

    def install(self) -> None:
        """Make `time.time()` return this clock, process-wide.

        Nothing is patched but `time.time`. A module that bound the function
        directly with `from time import time` keeps the real one, which is a
        known and narrow gap: one module in the tree does that.
        """
        global _INSTALLED
        with _INSTALL_LOCK:
            if _INSTALLED is self:
                return
            if _INSTALLED is not None:
                # The helper, not `uninstall()`. That takes this lock again, and
                # the lock is not reentrant, so installing a clock over another
                # one waited on itself forever.
                _INSTALLED._uninstall_locked()
            self._real_time = _time.time
            _time.time = self.now  # type: ignore[assignment]
            _INSTALLED = self

    def uninstall(self) -> None:
        with _INSTALL_LOCK:
            self._uninstall_locked()

    def _uninstall_locked(self) -> None:
        """Put the real clock back. The caller holds `_INSTALL_LOCK`."""
        global _INSTALLED
        if self._real_time is not None:
            _time.time = self._real_time  # type: ignore[assignment]
            self._real_time = None
        if _INSTALLED is self:
            _INSTALLED = None

    def __enter__(self) -> ExperimentClock:
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.uninstall()
