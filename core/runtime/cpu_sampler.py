"""Processor use, read so that a zero means idle and never "not looked yet".

Moved out of ``core/runtime/resource_observation.py`` on 14 November 2026, when
her event loop stopped taking this sampler's lock.
"""

from __future__ import annotations

import asyncio
import logging
import time

import psutil

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.walks_on_one_thread import done_by_the_walker

logger = logging.getLogger("Runtime.ResourceObservation")

#: psutil reports processor use as the difference between two samples of the
#: kernel's counters, so the FIRST non-blocking call in a process has nothing to
#: subtract and returns exactly 0.0. Every caller in this tree reads through
#: here, which made that zero look like a measurement.
#:
#: LIVE, 2026-08-25: "How hard is the machine you run on working right now?
#: Give me a number you can stand behind." The freshly taken sample said 0.0%
#: on a host sitting at 24%.
#:
#: The baseline is seeded at import, which costs nothing and means the window
#: any later read diffs over is however long the process has been alive. A
#: short blocking sample is not an alternative: this host's counters are too
#: coarse to move in 0.12s and return a clean 0.0 anyway, which is the same
#: fabricated number by a slower route.
_CPU_SAMPLED_AT = 0.0

#: Under this many seconds of separation the counters may not have moved, and
#: a zero would mean "too soon to tell" rather than "idle".
_CPU_MINIMUM_WINDOW_S = 0.5


def _seed_cpu_counter() -> None:
    """Give the counter a baseline, without blocking anything."""
    global _CPU_SAMPLED_AT
    try:
        psutil.cpu_percent(interval=None)
    except (psutil.Error, OSError, RuntimeError, ValueError) as exc:
        logger.debug("CPU counter not seeded: %s", exc)
        return
    _CPU_SAMPLED_AT = time.monotonic()


#: The last real reading, served to anyone who asks again before the counters
#: can have moved.
_CPU_LAST_VALUE = 0.0
_CPU_EVER_MEASURED = False
_CPU_SAMPLE_LOCK = checked_lock("resource_observation.cpu_sampler", rank=LockRank.LEAF)


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
    global _CPU_SAMPLED_AT, _CPU_LAST_VALUE, _CPU_EVER_MEASURED
    # One sampler. psutil keeps a single baseline for interval=None, so two
    # threads reaching this together made the second diff against a baseline
    # the first had just reset — and read 0.0. The live health pulse showed
    # "CPU 0.0%" on a host at load 15 (2026-09-16), every few samples.
    with _CPU_SAMPLE_LOCK:
        now = time.monotonic()
        if _CPU_SAMPLED_AT and (now - _CPU_SAMPLED_AT) >= _CPU_MINIMUM_WINDOW_S:
            _CPU_LAST_VALUE = float(psutil.cpu_percent(interval=None) or 0.0)
            _CPU_SAMPLED_AT = now
            _CPU_EVER_MEASURED = True
            return _CPU_LAST_VALUE
        if _CPU_EVER_MEASURED:
            return _CPU_LAST_VALUE
        # Nothing has ever been measured and the seed is too fresh to diff
        # against. One bounded wait, once per process, so the first answer is
        # a reading.
        _CPU_LAST_VALUE = float(psutil.cpu_percent(interval=_CPU_MINIMUM_WINDOW_S) or 0.0)
        _CPU_SAMPLED_AT = time.monotonic()
        _CPU_EVER_MEASURED = True
        return _CPU_LAST_VALUE


def _on_her_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
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
