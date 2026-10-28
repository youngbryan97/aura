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

`time.monotonic` is left alone process-wide on purpose: asyncio's timeouts read
it, and a timeout that cannot fire is a hang rather than a measurement. Her own
modules are given the clock's monotonic instead (`ExperimentClock.monotonic`),
through their own `time` name, so what she keeps as a monotonic reading rewinds
with a restore like everything else. asyncio, threads and the machinery keep the
real one.

The step is measured rather than chosen. A run times its own frames on the real
clock before installing this, and the number it found is part of the campaign
fingerprint, so two runs with the same fingerprint saw the same timeline.
"""

from __future__ import annotations

import sys
import time as _time
import types
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["ExperimentClock", "installed_clock", "real_time"]

#: The real `time.time`, kept so the harness can still measure itself. A
#: run's duration, a timeout on the process, the stamp on an artefact: those
#: are the machine's questions and they take the machine's answer whether or
#: not the experiment's clock is installed.
real_time = _time.time

#: The real `time.monotonic`. asyncio, threads and the machinery keep it.
real_monotonic = _time.monotonic


def her_monotonic() -> float:
    """`time.monotonic` as her modules read it: the installed clock's, or the real one.

    A plain function at module scope, so what holds it can be copied and pickled
    and restored and still read the live clock. A bound method of the clock was
    what her modules held first: a snapshot deep-copied the neural mesh's
    `_clock` with the whole clock behind it, a restore handed the mesh a frozen
    copy of the clock, and pickling an anchor for a carrier worker failed on the
    copy's real `time.time` (26 October).
    """
    clock = _INSTALLED
    return clock.monotonic() if clock is not None else real_monotonic()


def her_monotonic_ns() -> int:
    return int(her_monotonic() * 1e9)


class _HerTime(types.ModuleType):
    """`time` as her modules see it: the module itself, with the run's monotonic."""

    monotonic = staticmethod(her_monotonic)
    monotonic_ns = staticmethod(her_monotonic_ns)

    def __init__(self) -> None:
        super().__init__("time")

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)

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
        # Her monotonic is the experiment's time plus this offset: the real
        # reading at install minus the experiment's time then, so a reading
        # taken before the install and one taken after are on one line. It
        # travels with a snapshot, because a worker process installs a clock of
        # its own and a reading written in the coordinator has to keep its age
        # when a worker restores it.
        self.offset = real_monotonic() - self._now
        #: What the valve has added, kept so her monotonic never runs backwards
        #: within an arm and rewound with the clock by a restore.
        self.slack = 0.0
        #: Real seconds a frame may run before her monotonic follows the
        #: machine's: the time a phase is given before the harness abandons it.
        #: None keeps the valve shut.
        self.stall_after: float | None = None
        #: How many frames the valve opened in. A run where it opened was not on
        #: one timeline in every arm, and its report says so.
        self.valve_openings = 0
        #: Where her code was waiting each time the valve opened: the frame's
        #: real age and the stack that read the clock, the first few kept.
        self.valve_events: list[dict[str, Any]] = []
        self._frame_began = real_monotonic()
        self._open = False
        self._given: list[tuple[Any, str, Any]] = []
        self._cells: list[tuple[Any, Any]] = []
        self._her_time: _HerTime | None = None
        self._modules_seen = 0

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

    def monotonic(self) -> float:
        """`time.monotonic` as her modules read it.

        It advances with the experiment and rewinds with a restore, so an age she
        computes from one of her own readings is the same in every arm, whatever
        the machine took and however long a snapshot sat. Seed 71's untouched
        anchors, restored an hour into the process, read an hour of idle nobody
        lived; re-aging the readings on restore then made short time-to-lives
        depend on how fast the machine ran a turn (26 October).

        A frame that runs past `stall_after` real seconds is one where something
        of hers is waiting on a deadline this clock, frozen within a frame, would
        never reach. From then on the reading follows the real clock, so the wait
        ends, and the frame is counted in `valve_openings`.
        """
        value = self._now + self.offset + self.slack
        limit = self.stall_after
        if limit is not None:
            overdue = real_monotonic() - self._frame_began - limit
            if overdue > 0.0:
                if not self._open:
                    self._open = True
                    self.valve_openings += 1
                    if len(self.valve_events) < 8:
                        import traceback

                        self.valve_events.append({
                            "frame_seconds": round(limit + overdue, 3),
                            "waiting_in": [line.strip() for line in traceback.format_stack(limit=10)[:-1]],
                        })
                value += overdue
        return value

    # ── driving ──────────────────────────────────────────────────────────

    def advance(self, step: float | None = None) -> float:
        """Move the clock on by one frame, or by a stated amount."""
        with self._lock:
            self._close_valve()
            self._now += self.step if step is None else max(0.0, float(step))
            self._give_to_new_modules()
            return self._now

    def set(self, value: float, *, slack: float | None = None, offset: float | None = None) -> None:
        """Rewind or forward the clock, which is what a restore does."""
        with self._lock:
            self._now = float(value)
            if slack is not None:
                self.slack = float(slack)
            if offset is not None:
                self.offset = float(offset)
            self._open = False
            self._frame_began = real_monotonic()

    def _close_valve(self) -> None:
        """Keep what an open valve added, so her monotonic does not run back."""
        if self._open and self.stall_after is not None:
            self.slack += max(0.0, real_monotonic() - self._frame_began - self.stall_after)
        self._open = False
        self._frame_began = real_monotonic()

    # ── her modules ──────────────────────────────────────────────────────

    def _give_to_new_modules(self) -> None:
        """Her modules imported since the last look get her `time` too."""
        if self._her_time is not None and len(sys.modules) != self._modules_seen:
            self._give_to_her()

    def _give_to_her(self) -> None:
        """Point each of her modules' `time` (and a bare `monotonic`) at this clock.

        Her modules are the organism's packages outside the machinery, as the
        snapshot draws them (`core/subject/snapshot.py`). asyncio, threading and
        the machinery are not touched, so every timeout still fires on time.
        """
        from core.subject.snapshot import _MACHINERY_PACKAGES, _ORGANISM_PACKAGES, _in_packages

        if self._her_time is None:
            self._her_time = _HerTime()
        for name, module in list(sys.modules.items()):
            if module is None or not _in_packages(name, _ORGANISM_PACKAGES):
                continue
            if _in_packages(name, _MACHINERY_PACKAGES):
                continue
            space = getattr(module, "__dict__", None)
            if not isinstance(space, dict):
                continue
            # The module under any name (`import time as _t`), and its monotonic
            # under any name (`from time import monotonic`, `_now = time.monotonic`).
            for key, value in list(space.items()):
                if value is _time:
                    space[key] = self._her_time
                    self._given.append((space, key, _time))
                elif value is real_monotonic:
                    space[key] = her_monotonic
                    self._given.append((space, key, real_monotonic))
        # And the references her modules captured before the install: a closure
        # cell, which is where a dataclass keeps `field(default_factory=
        # time.monotonic)`. The governance token stamped its issue on the real
        # clock that way and read its expiry on this one, and since her clock
        # runs a second a turn, every token read as expired within a minute and
        # a restore's write was refused (26 October).
        import gc

        for obj in gc.get_objects():
            if type(obj) is not types.FunctionType:
                # An instance of one of her classes keeping it as an attribute:
                # the neural mesh holds its clock as `self._clock`.
                home = getattr(type(obj), "__module__", "")
                if not isinstance(home, str) or not _in_packages(home, _ORGANISM_PACKAGES):
                    continue
                if _in_packages(home, _MACHINERY_PACKAGES):
                    continue
                try:
                    fields = object.__getattribute__(obj, "__dict__")
                except (AttributeError, TypeError):
                    continue
                for key, value in list(fields.items()):
                    if value is real_monotonic:
                        fields[key] = her_monotonic
                        self._given.append((fields, key, real_monotonic))
                continue
            if not obj.__closure__:
                continue
            home = getattr(obj, "__module__", "")
            if not isinstance(home, str) or not _in_packages(home, _ORGANISM_PACKAGES):
                continue
            if _in_packages(home, _MACHINERY_PACKAGES):
                continue
            for cell in obj.__closure__:
                try:
                    held = cell.cell_contents
                except ValueError:
                    continue
                if held is real_monotonic:
                    cell.cell_contents = her_monotonic
                    self._cells.append((cell, real_monotonic))
        self._modules_seen = len(sys.modules)

    def _take_back(self) -> None:
        for space, name, original in reversed(self._given):
            if space.get(name) is not original:
                space[name] = original
        self._given.clear()
        for cell, original in reversed(self._cells):
            cell.cell_contents = original
        self._cells.clear()
        self._her_time = None

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
            self.offset = real_monotonic() - self._now
            self._frame_began = real_monotonic()
            self._give_to_her()
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
        self._take_back()
        if _INSTALLED is self:
            _INSTALLED = None

    def __enter__(self) -> ExperimentClock:
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.uninstall()
