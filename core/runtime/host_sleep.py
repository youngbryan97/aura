"""How much of the gap since an anchor was the host asleep.

A laptop closes and the machine suspends. ``time.monotonic()`` stops on
macOS, ``time.time()` does not, and every subsystem holding "when did I
last see this" wakes to an anchor hours old. Each one then reaches the
same wrong conclusion in its own vocabulary: the heartbeat says the mind
stopped, the liveness check says the worker wedged, the staleness check
says the reading is void. Nothing was wrong; the host was shut.

``core/brain/llm/mlx_client.py`` worked this out for the generation wait
and kept it: a clock that counts THROUGH suspend, paired with one that
does not, measures the suspension without consulting the wall clock at
all — so an NTP step, a manual date change, or a VM migration cannot be
mistaken for a resume. It was private to that module, so the inference
lane rebased and nothing else did.

This is the same measurement, once, for anything with an anchor.

Three states and they are not interchangeable, which is the point:

* **asleep** — both clocks disagree by a measurable amount. The gap is
  the host being suspended and is not evidence about anything else.
* **clock moved** — the wall clock jumped beyond running time and
  measured sleep. Wall anchors are void, but the machine was awake the
  whole time, so a stall really was a stall.
* **cannot tell** — the platform has no suspend-inclusive clock. Saying
  so beats attributing the gap to whichever is convenient.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HostGap",
    "sleep_inclusive_monotonic",
    "seconds_asleep_since",
    "host_sleep_report",
    "sleep_anchor",
]


def sleep_inclusive_monotonic() -> float | None:
    """A monotonic clock that keeps counting while the host is asleep.

    Returns None where the platform offers no such clock, and the caller
    then says it could not tell sleep from a stall rather than guessing.
    """

    for name in ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC"):
        clock_id = getattr(time, name, None)
        if clock_id is None:
            continue
        if name == "CLOCK_MONOTONIC" and sys.platform != "darwin":
            # Only Darwin's CLOCK_MONOTONIC includes suspend; elsewhere it
            # is what time.monotonic() already returns, so the difference
            # would be a constant zero dressed up as a measurement.
            continue
        try:
            return float(time.clock_gettime(clock_id))
        except (OSError, ValueError, AttributeError):
            continue
    return None


@dataclass(frozen=True)
class HostGap:
    """What happened between two samples."""

    #: Seconds the process actually ran.
    running_s: float
    #: Seconds the host was suspended, when that is measurable.
    slept_s: float
    #: Seconds the wall clock moved beyond running time and sleep.
    clock_shift_s: float
    #: False when the platform cannot tell sleep from a clock jump.
    measured: bool

    @property
    def unexplained_s(self) -> float:
        """Time that is neither running nor sleep — a real absence."""

        return max(0.0, self.clock_shift_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running_s": round(self.running_s, 3),
            "slept_s": round(self.slept_s, 3),
            "clock_shift_s": round(self.clock_shift_s, 3),
            "measured": self.measured,
        }


class _HostClock:
    """One sample of all three clocks, advanced on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wall = time.time()
        self._monotonic = time.monotonic()
        self._inclusive = sleep_inclusive_monotonic()
        self.slept_total_s = 0.0
        self.sleeps = 0

    def measure(self) -> HostGap:
        with self._lock:
            wall = time.time()
            monotonic = time.monotonic()
            inclusive = sleep_inclusive_monotonic()

            running = max(0.0, monotonic - self._monotonic)
            wall_delta = max(0.0, wall - self._wall)

            slept = 0.0
            measured = False
            if inclusive is not None and self._inclusive is not None:
                slept = max(0.0, (inclusive - self._inclusive) - running)
                measured = True
            clock_shift = wall_delta - running - slept
            if not measured:
                # Indistinguishable here, so the gap is reported as sleep
                # rather than as a clock anomaly nothing can substantiate.
                slept, clock_shift = max(0.0, clock_shift), 0.0

            self._wall = wall
            self._monotonic = monotonic
            self._inclusive = inclusive
            if slept > 0.0:
                self.slept_total_s += slept
                self.sleeps += 1
            return HostGap(
                running_s=running,
                slept_s=slept,
                clock_shift_s=max(0.0, clock_shift),
                measured=measured,
            )


_CLOCK = _HostClock()


def _measure_gap() -> HostGap:
    """Advance the shared sample and return what happened since the last.

    Deliberately not exported. The sample is one, so two callers sharing
    it would each see part of a gap and neither would see it whole — a
    subsystem asking "was the host asleep?" must keep its OWN anchor and
    use :func:`seconds_asleep_since`. This one belongs to the report.
    """

    return _CLOCK.measure()


def seconds_asleep_since(anchor_inclusive: float | None, anchor_monotonic: float) -> float:
    """Of the time since these anchors, how much was the host suspended.

    Take both with :func:`sleep_inclusive_monotonic` and
    :func:`time.monotonic` at the same moment, keep them beside whatever
    you are timing, and pass them back. Returns 0.0 when the platform
    cannot measure suspend, which is the honest answer and the safe one:
    a caller that subtracts it is then no worse off than before.
    """

    if anchor_inclusive is None:
        return 0.0
    now_inclusive = sleep_inclusive_monotonic()
    if now_inclusive is None:
        return 0.0
    ran = max(0.0, time.monotonic() - anchor_monotonic)
    return max(0.0, (now_inclusive - anchor_inclusive) - ran)


def sleep_anchor() -> tuple[float | None, float]:
    """Both clocks at one moment, to keep beside whatever you are timing.

    Hand the pair back to :func:`seconds_asleep_since` later.
    """

    return sleep_inclusive_monotonic(), time.monotonic()


def host_sleep_report() -> dict[str, Any]:
    """For the integrity surface: how much of this run was suspended."""

    _measure_gap()
    return {
        "measurable": sleep_inclusive_monotonic() is not None,
        "sleeps": _CLOCK.sleeps,
        "slept_total_s": round(_CLOCK.slept_total_s, 3),
    }


def reset_host_clock_for_test() -> None:
    global _CLOCK
    _CLOCK = _HostClock()
