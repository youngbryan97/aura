"""The loudest CRITICAL in the runtime was arithmetic on the event loop.

Live 2026-08-17: 246 "🚨 [WATCHDOG] EVENT LOOP STALL DETECTED" at 5.0-7.6s
each, and 196 "hard event-loop lag exceeded 5.00s". The stall dump names the
path exactly:

    heartbeat._tick -> qualia_synthesizer.synthesize
    -> qualia_engine.process -> earned_metric.fit -> earned_metric._ridge

EarnedAxis.fit solves a ridge regression, then solves it _AXIS_PERMUTATIONS
(200) more times to build the permutation null. That is 201 numpy solves per
axis, inline, on the heartbeat tick, on the event loop — multiplied by the
number of axes. The stall was not mysterious; it was arithmetic.

Deferring it is safe because nothing reads the result during that call:
value() reads coefficients under the axis's own lock, and fit() takes the same
lock only around the state it copies, doing the heavy solve outside it. A
refit landing a few hundred milliseconds later is invisible to every reader.
Blocking the loop for seconds is not.
"""
from __future__ import annotations

import inspect
import time

import pytest

import core.consciousness.qualia_engine as qualia_engine


def _owner():
    for _, cls in inspect.getmembers(qualia_engine, inspect.isclass):
        if hasattr(cls, "_refit_off_the_loop"):
            return cls
    raise AssertionError("no class owns _refit_off_the_loop")


class _SlowAxis:
    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay
        self.fits = 0

    def fit(self):
        time.sleep(self.delay)
        self.fits += 1


def _instance(axes):
    cls = _owner()
    inst = cls.__new__(cls)
    inst._axes = axes
    inst._refit_in_flight = False
    return inst


def test_the_caller_is_not_blocked_by_the_refit():
    """The whole defect: 201 solves per axis, inline, on the tick."""
    inst = _instance({"a": _SlowAxis(0.3), "b": _SlowAxis(0.3)})

    started = time.monotonic()
    inst._refit_off_the_loop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.1, f"the tick waited {elapsed:.2f}s on a refit"


def test_the_refit_actually_happens():
    """Off the loop must not mean not at all."""
    axes = {"a": _SlowAxis(0.05), "b": _SlowAxis(0.05)}
    inst = _instance(axes)

    inst._refit_off_the_loop()
    time.sleep(0.5)

    assert axes["a"].fits == 1
    assert axes["b"].fits == 1
    assert inst._refit_in_flight is False


def test_refits_do_not_queue_up():
    """The newest fit is the only one worth having.

    Refits arriving faster than they complete would otherwise back up, and a
    backlog of stale ones turns a deferral into a second leak.
    """
    axes = {"a": _SlowAxis(0.3)}
    inst = _instance(axes)

    for _ in range(5):
        inst._refit_off_the_loop()
    time.sleep(0.8)

    assert axes["a"].fits == 1


def test_a_failing_fit_releases_the_latch():
    """A wedged flag would stop every future refit silently."""

    class _Broken:
        def fit(self):
            raise ValueError("singular matrix")

    inst = _instance({"a": _Broken()})

    inst._refit_off_the_loop()
    time.sleep(0.3)

    assert inst._refit_in_flight is False


def test_the_tick_path_no_longer_calls_fit_directly():
    """Wiring: the stall returns the moment someone inlines it again."""
    source = inspect.getsource(qualia_engine)
    start = source.index("_REFIT_EVERY == 0")
    window = source[start : start + 260]

    assert "_refit_off_the_loop" in window
    assert "axis.fit()" not in window


def test_the_permutation_count_is_still_what_makes_this_expensive():
    """Documents WHY this cannot be inline, so nobody 'simplifies' it back."""
    assert qualia_engine._AXIS_PERMUTATIONS >= 100
