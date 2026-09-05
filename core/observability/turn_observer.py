"""What a turn actually costs, and whether it went in circles. Measurement only.

Two things were unmeasured. Nobody knew how many cognitive phases a real turn
runs or how long they take — `PassInstrumentation.report()` aggregates by pass
name across the whole process, which answers "is this pass slow" but not "what
did that turn cost". And nothing watched for a turn repeating itself; the stuck
detector's thresholds were inherited from another project's tuning, never
checked against Aura.

**This cannot change behaviour, structurally.** The instrumentation seam offers
two hook types: a *before* hook, whose False return skips the pass, and an
*after* hook, which only observes. This module registers **after-hooks only** —
it holds no mechanism by which a pass could be skipped, a turn shortened, or an
output altered. That is a property of what it registers, not a promise about
how it behaves, and `test_turn_observer.py` asserts it. `after_pass` also wraps
every hook in its own try/except, so a bug in here cannot take down a turn.

The point is calibration. Every threshold in `StuckDetector` and every ceiling
in `Budget` is currently a guess borrowed from someone else's system. Numbers
first, ceilings second — a limit chosen without knowing the distribution is an
arbitrary constant wearing a safety label.

Two honest limits on what this can see:

* **No tokens.** ``PassRecord`` carries name, ordinal, duration, skipped,
  reason and error. There is no token count at this seam, so the ledger meters
  steps and wall-clock and leaves ``max_tokens`` unmeasured rather than
  reporting a partial figure as though it were the cost.
* **Phases, not tools.** A rut here means a cognitive phase erroring over and
  over, or two phases alternating. "She read the same file four times" is tool
  level and lives elsewhere.
"""
from __future__ import annotations

import logging
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.stuck_detector import AgentStep, StuckDetector, StuckVerdict
from core.runtime.turn_budget import Budget, BudgetLedger

logger = logging.getLogger("Aura.TurnObserver")

__all__ = ["TurnSummary", "TurnObserver", "get_turn_observer", "install_turn_observer"]

#: How many completed turns to keep. Bounded because this runs for the life of
#: the process and an unbounded history is a leak with a nice name.
DEFAULT_HISTORY = 64


@dataclass
class TurnSummary:
    """One turn's cost and whether it repeated itself."""

    label: str = ""
    started_at: float = field(default_factory=time.time)
    passes: int = 0
    skipped: int = 0
    errors: int = 0
    duration_s: float = 0.0
    stuck: StuckVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "started_at": self.started_at,
            "passes": self.passes,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_s": round(self.duration_s, 6),
            "stuck": self.stuck.describe() if self.stuck and self.stuck.stuck else None,
        }


class TurnObserver:
    """Accumulates per-turn cost and stuck verdicts from pass records.

    Turn boundaries are inferred rather than signalled: ``begin_run`` resets
    pass numbering per turn, so ordinal 1 opens a turn. An ordinal that fails
    to increase also opens one, which covers paths that reset the counter
    without going through ``begin_run`` — without that fallback a missed reset
    would silently merge every subsequent turn into one ever-growing record.
    """

    def __init__(
        self,
        *,
        budget: Budget | None = None,
        detector: StuckDetector | None = None,
        history: int = DEFAULT_HISTORY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budget = budget or Budget.unlimited()
        self._detector = detector or StuckDetector()
        self._clock = clock
        self._lock = checked_lock("turn_observer.state", rank=LockRank.LEAF)
        self._history: deque[TurnSummary] = deque(maxlen=history)
        self._current: TurnSummary | None = None
        self._ledger: BudgetLedger | None = None
        self._steps: list[AgentStep] = []
        self._last_ordinal = 0

    # -- the hook ----------------------------------------------------------

    def observe(self, record: Any) -> None:
        """After-hook. Records a pass; never influences one."""
        with self._lock:
            if record.ordinal <= self._last_ordinal or self._current is None:
                self._close_locked()
                self._open_locked(record.name)
            self._last_ordinal = record.ordinal

            summary = self._current
            assert summary is not None  # opened just above
            if record.skipped:
                summary.skipped += 1
                return

            summary.passes += 1
            if record.error:
                summary.errors += 1

            # record() rather than spend(): an unlimited budget cannot be
            # exceeded, and a metered turn must never raise into the pipeline.
            if self._ledger is not None:
                self._ledger.record(steps=1)

            self._steps.append(
                AgentStep(
                    action=record.name,
                    observation=record.error or record.reason or "",
                    is_error=bool(record.error),
                    kind="tool",
                )
            )
            verdict = self._detector.check(self._steps)
            if verdict.stuck:
                summary.stuck = verdict
                logger.info("turn %r %s", summary.label, verdict.describe())

    # -- turn lifecycle ----------------------------------------------------

    def _open_locked(self, label: str) -> None:
        self._current = TurnSummary(label=label)
        self._ledger = BudgetLedger(budget=self._budget, clock=self._clock)
        self._steps = []
        self._detector.reset()

    def _close_locked(self) -> None:
        if self._current is None:
            return
        if self._ledger is not None:
            self._current.duration_s = self._ledger.elapsed
        self._history.append(self._current)
        self._current = None
        self._ledger = None

    def close_turn(self) -> TurnSummary | None:
        """Finalise the turn in flight, if any. Optional — the next turn
        closes the previous one anyway; this just makes the last one visible
        without waiting for another."""
        with self._lock:
            summary = self._current
            self._close_locked()
            return summary

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """The surface that makes this worth collecting.

        A metric nobody reads is the thing the histogram registry refuses for
        want of an owner; this is where these numbers get read.
        """
        with self._lock:
            turns = list(self._history)
            in_flight = self._current

        completed = [t for t in turns if t.passes]
        passes = sorted(t.passes for t in completed)
        durations = sorted(t.duration_s for t in completed)
        stuck = [t for t in completed if t.stuck and t.stuck.stuck]

        return {
            "turns_recorded": len(completed),
            "in_flight": in_flight.to_dict() if in_flight else None,
            "passes": _distribution(passes),
            "duration_s": _distribution(durations),
            "turns_with_errors": sum(1 for t in completed if t.errors),
            "stuck_turns": len(stuck),
            "stuck_patterns": sorted({str(t.stuck.pattern) for t in stuck}),
            # Stated rather than omitted: absent is not zero.
            "tokens": "not measured at this seam (PassRecord carries no token count)",
            "recent": [t.to_dict() for t in turns[-10:]],
        }


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(values[0], 6),
        "median": round(values[len(values) // 2], 6),
        "max": round(values[-1], 6),
        "mean": round(sum(values) / len(values), 6),
    }


_instance: TurnObserver | None = None
_install_lock = checked_lock("turn_observer.install", rank=LockRank.LEAF)
#: Which instrumentations already carry the hook. Weak so a discarded
#: instrumentation does not pin itself in memory, and per-instrumentation
#: because idempotence is a property of the *pairing*: a bare "already
#: installed" flag on the observer would silently refuse to attach to a second
#: instrumentation, leaving it unmetered while reporting success.
_installed_into: weakref.WeakSet[Any] = weakref.WeakSet()


def get_turn_observer() -> TurnObserver:
    global _instance
    with _install_lock:
        if _instance is None:
            _instance = TurnObserver()
        return _instance


def install_turn_observer(instrumentation: Any = None) -> TurnObserver:
    """Register the observer's after-hook. Idempotent per instrumentation.

    Only ``add_after_hook`` is called. There is deliberately no path here that
    registers a before-hook, because a before-hook is the only way this seam
    can change what runs.
    """
    observer = get_turn_observer()
    if instrumentation is None:
        from core.pipeline.pass_manager import get_instrumentation

        instrumentation = get_instrumentation()
    with _install_lock:
        if instrumentation in _installed_into:
            return observer
        instrumentation.add_after_hook(observer.observe)
        _installed_into.add(instrumentation)
    logger.info("TurnObserver installed (observe-only)")
    return observer
