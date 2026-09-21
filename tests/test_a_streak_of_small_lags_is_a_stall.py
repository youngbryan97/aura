"""Two minutes of a loop falling behind leaves something to read.

LIVE, 2026-09-16, 22:49 to 22:51Z: a streak of lags, the shortest 1.5s
and the longest 11s,
produced no forensic dump. Each look measured less than the watchdog's 5s
threshold and the next one started fresh, so a loop that could not keep up
for two minutes left nothing behind — the one open item in R06 after the
nine named holds were fixed.

The two numbers are the watchdog's own. The window is the interval over
which it already considers a stall worth acting on, and the budget is one
reportable stall's worth of lost time.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from core.resilience.stall_watchdog import (
    _ACTIVE_RECOVERY_THRESHOLD,
    _WATCHDOG_TICK_S,
    StallWatchdog,
)


@pytest.fixture
def watchdog() -> StallWatchdog:
    loop = asyncio.new_event_loop()
    try:
        dog = StallWatchdog(loop, threshold=5.0)
        dog._should_suppress_stall = lambda _elapsed: False  # type: ignore[method-assign]
        yield dog
    finally:
        loop.close()


def _reported(dog: StallWatchdog) -> list[float]:
    seen: list[float] = []
    dog._report_stall = lambda elapsed, share=None: seen.append(elapsed)  # type: ignore[method-assign]
    dog._loop_cpu_share_since_heartbeat = lambda _elapsed: None  # type: ignore[method-assign]
    return seen


def test_one_small_lag_reports_nothing(watchdog: StallWatchdog) -> None:
    """The null. A single late look is not a streak and never was."""
    seen = _reported(watchdog)
    watchdog._note_lateness(_WATCHDOG_TICK_S + 4.0)
    assert seen == []


def test_lateness_short_of_the_budget_reports_nothing(watchdog: StallWatchdog) -> None:
    seen = _reported(watchdog)
    for _ in range(4):
        watchdog._note_lateness(_WATCHDOG_TICK_S + 1.0)
    assert seen == [], "4s of lateness is under one 5s stall's worth"


def test_a_streak_worth_a_stall_is_reported(watchdog: StallWatchdog) -> None:
    """Five looks 1.2s late each is 6s lost, which is more than the line."""
    seen = _reported(watchdog)
    for _ in range(5):
        watchdog._note_lateness(_WATCHDOG_TICK_S + 1.2)
    assert len(seen) == 1
    assert seen[0] == pytest.approx(6.0, abs=0.01)


def test_the_live_streak_is_caught(watchdog: StallWatchdog) -> None:
    """The lags that produced nothing: 1.5, 2.1, 3.4, 4.9 seconds elapsed."""
    seen = _reported(watchdog)
    for elapsed in (1.5, 2.1, 3.4, 4.9):
        watchdog._note_lateness(elapsed)
    assert len(seen) == 1
    # 0.5 + 1.1 + 2.4 + 3.9 seconds of lateness.
    assert seen[0] == pytest.approx(7.9, abs=0.01)


def test_it_reports_once_not_every_look(watchdog: StallWatchdog) -> None:
    """The window clears on report, so a long bad patch is not a flood."""
    seen = _reported(watchdog)
    for _ in range(20):
        watchdog._note_lateness(_WATCHDOG_TICK_S + 1.2)
    assert len(seen) == 4, "one report per 6s of accumulated lateness"


def test_lateness_ages_out_of_the_window(watchdog: StallWatchdog) -> None:
    """Lags an hour apart are not a streak.

    The moment is passed in rather than patched onto the `time` module: a
    watchdog is a thread, and a fake clock installed globally is read by
    everything else running in the process. That is how this test first
    broke a memory-watchdog test three files away.
    """
    seen = _reported(watchdog)
    now = 1000.0
    for _ in range(5):
        watchdog._note_lateness(_WATCHDOG_TICK_S + 1.2, now=now)
        now += _ACTIVE_RECOVERY_THRESHOLD + 1.0
    assert seen == []


def test_a_healthy_look_accumulates_nothing(watchdog: StallWatchdog) -> None:
    """An on-time look is not lateness, however many of them there are."""
    seen = _reported(watchdog)
    for _ in range(100):
        watchdog._note_lateness(_WATCHDOG_TICK_S)
    assert seen == []
    assert not watchdog._recent_lateness


def test_the_report_says_what_it_saw(watchdog: StallWatchdog, caplog) -> None:
    watchdog._report_stall = lambda elapsed, share=None: None  # type: ignore[method-assign]
    watchdog._loop_cpu_share_since_heartbeat = lambda _elapsed: None  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="Aura.Resilience.Watchdog"):
        for _ in range(5):
            watchdog._note_lateness(_WATCHDOG_TICK_S + 1.2)
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "FALLING BEHIND" in said
    assert "5 looks" in said
    assert "none of them past" in said
