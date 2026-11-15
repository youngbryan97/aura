"""Processor use, read so that a zero means idle and never "not looked yet".

Moved out of ``core/runtime/resource_observation.py`` on 14 November 2026, when
her event loop stopped taking this sampler's lock.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import psutil

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.walks_on_one_thread import done_by_the_walker

logger = logging.getLogger("Runtime.ResourceObservation")

#: Processor use is the difference between two samples of the kernel's
#: counters, so a reading with nothing to subtract is exactly 0.0 and looks
#: like a measurement.
#:
#: LIVE, 2026-08-25: "How hard is the machine you run on working right now?
#: Give me a number you can stand behind." The freshly taken sample said 0.0%
#: on a host sitting at 24%.
#:
#: The sampler keeps its own snapshot of the counters rather than psutil's.
#: psutil 7 keeps the baseline for cpu_percent(interval=None) per THREAD, so
#: the seed taken at import served only the importing thread, and every other
#: thread's first reading had nothing to subtract: her health pulse read "CPU
#: 0.0%" every few samples on 13 and 14 November, between readings such as 31.9% and 47.7%.
_CPU_SAMPLED_AT = 0.0
_CPU_TIMES_AT: Any = None

#: Under this many seconds of separation the counters may not have moved, and
#: a zero would mean "too soon to tell" rather than "idle".
_CPU_MINIMUM_WINDOW_S = 0.5


def _seed_cpu_counter() -> None:
    """Give the counter a baseline, without blocking anything."""
    global _CPU_SAMPLED_AT, _CPU_TIMES_AT
    try:
        _CPU_TIMES_AT = psutil.cpu_times()
    except (psutil.Error, OSError, RuntimeError, ValueError) as exc:
        logger.debug("CPU counter not seeded: %s", exc)
        return
    _CPU_SAMPLED_AT = time.monotonic()


def _busy_percent(before: Any, after: Any) -> float | None:
    """The share of processor time spent busy between two cpu_times() snapshots, as psutil counts it.

    None when the counters have not moved, which is "too soon to tell", not idle.
    """

    def total(times: Any) -> float:
        # Guest time is already counted in user time on Linux; psutil leaves it out once.
        return float(sum(times)) - float(getattr(times, "guest", 0.0)) - float(getattr(times, "guest_nice", 0.0))

    def busy(times: Any) -> float:
        return total(times) - float(times.idle) - float(getattr(times, "iowait", 0.0))

    elapsed = total(after) - total(before)
    if elapsed <= 0.0:
        return None
    return max(0.0, min(100.0, 100.0 * (busy(after) - busy(before)) / elapsed))


#: The last real reading, served to anyone who asks again before the counters
#: can have moved.
_CPU_LAST_VALUE = 0.0
_CPU_EVER_MEASURED = False
_CPU_SAMPLE_LOCK = checked_lock("resource_observation.cpu_sampler", rank=LockRank.LEAF)


def _read_against_the_snapshot(now: float) -> float | None:
    """A reading over the time since the last snapshot, which then moves here. Called holding the lock."""
    global _CPU_SAMPLED_AT, _CPU_TIMES_AT, _CPU_LAST_VALUE, _CPU_EVER_MEASURED
    times = psutil.cpu_times()
    reading = _busy_percent(_CPU_TIMES_AT, times) if _CPU_TIMES_AT is not None else None
    if reading is None:
        if _CPU_TIMES_AT is None:
            _CPU_TIMES_AT, _CPU_SAMPLED_AT = times, now
        return None
    _CPU_TIMES_AT, _CPU_SAMPLED_AT = times, now
    _CPU_LAST_VALUE, _CPU_EVER_MEASURED = reading, True
    return reading


def measured_cpu_percent() -> float | None:
    """Processor use over a window long enough for the counters to have moved.

    Never blocks twice. Asked again inside the window, it repeats the last real
    reading rather than waiting for a new one: this function is on the resource
    path that boot polls continuously, and a half-second wait per call took the
    runtime past the launcher's boot deadline and put it in a restart loop.

    On her event loop it never takes the sampler's lock: see
    ``_cpu_percent_without_waiting``. None means no reading has been taken yet.
    """
    if _on_her_event_loop():
        return _cpu_percent_without_waiting()
    # One sampler, and one snapshot for every thread that asks.
    with _CPU_SAMPLE_LOCK:
        now = time.monotonic()
        if _CPU_TIMES_AT is not None and (now - _CPU_SAMPLED_AT) >= _CPU_MINIMUM_WINDOW_S:
            reading = _read_against_the_snapshot(now)
            if reading is not None:
                return reading
        if _CPU_EVER_MEASURED:
            return _CPU_LAST_VALUE
        # Nothing has ever been measured and the snapshot is too fresh to diff
        # against. One bounded wait, once per process and never on her loop, so
        # the first answer is a reading.
        if _CPU_TIMES_AT is None:
            _read_against_the_snapshot(now)
        time.sleep(max(0.0, _CPU_MINIMUM_WINDOW_S - (time.monotonic() - _CPU_SAMPLED_AT)))
        return _read_against_the_snapshot(time.monotonic())


def _on_her_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    # not a failure: no running loop in this thread is the answer: it is not her event loop.
    except RuntimeError:
        return False
    return True


#: One refresh waits on the walker at a time; set on the loop, cleared by the walker.
_CPU_REFRESH_QUEUED = False


def _cpu_percent_without_waiting() -> float | None:
    """The reading her event loop gets: the last one taken, never a wait for the next.

    Between 12 and 14 November 2026 lockdep caught her loop holding the
    sampler's lock past its 50 ms limit 18 times, the longest 239 ms, from the
    cognitive loop's free-energy tick and the subconscious loop. The work under the lock is one counter read; the
    time was the loop thread off the CPU while it held a lock that other
    threads wait on. The loop now reads the published reading, and the walker
    thread takes the next one, including the first, bounded wait.
    """
    global _CPU_REFRESH_QUEUED
    due = time.monotonic() - _CPU_SAMPLED_AT >= _CPU_MINIMUM_WINDOW_S
    if (due or not _CPU_EVER_MEASURED) and not _CPU_REFRESH_QUEUED:
        _CPU_REFRESH_QUEUED = True
        done_by_the_walker(_refresh_the_cpu_reading)
    return _CPU_LAST_VALUE if _CPU_EVER_MEASURED else None


def _refresh_the_cpu_reading() -> None:
    global _CPU_REFRESH_QUEUED
    try:
        measured_cpu_percent()
    finally:
        _CPU_REFRESH_QUEUED = False


_seed_cpu_counter()
