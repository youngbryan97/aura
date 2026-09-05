"""core/interiority/attribution.py — did the firing predict anything?

O3 on the council docket, and the gap that separates a system with
states from a system that learns from having them. Faculties fire, their
effects land, and then hours or days later something turns out well or
badly — and nothing connected the two. Every faculty's parameters were
therefore frozen at the value they were written with, and no amount of
living could move them.

The mechanism is delayed credit assignment with an eligibility trace.
When a faculty fires it leaves a trace keyed to the event; when an
outcome arrives naming that event, or naming a goal the faculty moved,
the trace is still warm and the credit goes to the faculties that were
active. Traces decay, so a faculty that fired a week before an unrelated
outcome collects nothing.

Two things make this honest rather than a reward signal with extra
steps.

The credit is **directional, not evaluative**. It records whether the
faculty's own claim held — did the boundary hold after anger fired, did
the repair happen after guilt, did the practice resume after revival —
rather than whether the outcome was pleasant. A faculty that correctly
produces a painful state is right, and a scheme that scored on valence
would train it away.

And an outcome with no trace is **dropped, not spread**. Assigning
credit to whoever happened to be active is how a learner acquires
confident nonsense, so an outcome that names nothing this system did
leaves nothing behind but a counter.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from core.interiority.params import ParamKind, declare
from core.runtime.errors import record_degradation

_TRACE_HALF_LIFE = declare(
    "interiority.attribution.trace_half_life_s",
    3600.0,
    unit="s",
    basis=(
        "An eligibility trace has to outlive the gap between a state and the "
        "outcome it was about. An hour covers a conversation and the work that "
        "follows it, and is short enough that a state from yesterday collects "
        "nothing from today. Anything a faculty is about that resolves over "
        "days — grief, a dormant practice — carries its own record in the "
        "ledger and does not depend on the trace."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Longer and unrelated outcomes start collecting credit, which is how a "
        "learner acquires confident nonsense; shorter and nothing slower than a "
        "single turn can ever be learned from."
    ),
    lower=60.0,
    upper=604800.0,
    sweep_range=(600.0, 86400.0),
    owner="core/interiority/attribution.py",
)

_LEARNING_RATE = declare(
    "interiority.attribution.credit_learning_rate",
    0.08,
    unit="rate",
    basis=(
        "Rescorla-Wagner on a faculty's own hit rate. Slow enough that one "
        "outcome does not rewrite a standing, fast enough that a dozen "
        "confirmations move it visibly."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity="How fast a faculty's measured reliability moves after an outcome.",
    sweep_range=(0.01, 0.3),
    owner="core/interiority/attribution.py",
)

_MAX_TRACES = declare(
    "interiority.attribution.max_traces",
    2048,
    unit="records",
    basis=(
        "Bounded because an unbounded trace table is the 96GB state file this "
        "runtime has already had once. Two thousand covers far more open "
        "events than the half-life keeps warm."
    ),
    kind=ParamKind.DERIVED,
    sensitivity="Smaller and slow outcomes lose their trace before they arrive.",
    lower=64.0,
    upper=65536.0,
    owner="core/interiority/attribution.py",
)


@dataclass
class Trace:
    """One faculty's firing, still eligible for credit."""

    faculty: str
    event_id: str
    intensity: float
    #: What the faculty claimed would follow. The outcome is checked against
    #: this rather than against whether things went well.
    claim: str
    goals_touched: tuple[str, ...]
    at: float = field(default_factory=time.time)

    def eligibility(self, now: float) -> float:
        age = max(0.0, now - self.at)
        return math.exp(-age * math.log(2.0) / _TRACE_HALF_LIFE.value)


@dataclass
class Standing:
    """What a faculty's firing has been worth, measured."""

    faculty: str
    #: Fraction of resolved firings whose own claim held. Starts at neither
    #: 0 nor 1: an unmeasured faculty is unmeasured, and `resolved` says so.
    hit_rate: float = 0.5
    resolved: int = 0
    confirmed: int = 0
    disconfirmed: int = 0
    unresolved: int = 0

    @property
    def measured(self) -> bool:
        return self.resolved > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "faculty": self.faculty,
            "hit_rate": self.hit_rate if self.measured else None,
            "resolved": self.resolved,
            "confirmed": self.confirmed,
            "disconfirmed": self.disconfirmed,
            "unresolved": self.unresolved,
            "measured": self.measured,
        }


class Attribution:
    """Delayed credit assignment from outcomes back to the faculties."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.interiority.attribution.Attribution", reentrant=True)
        self._traces: list[Trace] = []
        self._standing: dict[str, Standing] = {}
        self._outcomes_seen = 0
        self._outcomes_dropped = 0

    # ── recording ─────────────────────────────────────────────────────
    def note_firing(
        self,
        faculty: str,
        *,
        event_id: str,
        intensity: float,
        claim: str,
        goals_touched: Iterable[str] = (),
    ) -> Trace:
        with self._lock:
            trace = Trace(
                faculty=faculty,
                event_id=event_id,
                intensity=max(0.0, min(1.0, intensity)),
                claim=claim,
                goals_touched=tuple(goals_touched),
            )
            self._traces.append(trace)
            limit = int(_MAX_TRACES.value)
            if len(self._traces) > limit:
                del self._traces[: len(self._traces) - limit]
            self._standing.setdefault(faculty, Standing(faculty)).unresolved += 1
            return trace

    def note_activations(self, state: Any) -> int:
        """Leave a trace for every faculty that fired on this state."""
        event_id = getattr(state, "event_id", "") or ""
        noted = 0
        for faculty, intensity in getattr(state, "transmitted", {}).items():
            if intensity <= 0.0:
                continue
            goals = tuple(
                g.goal for g in getattr(state, "goals", ()) if g.delta != 0.0
            )
            self.note_firing(
                faculty,
                event_id=event_id,
                intensity=intensity,
                claim=getattr(state, "dominant", ("", 0.0))[0],
                goals_touched=goals,
            )
            noted += 1
        return noted

    # ── the outcome ───────────────────────────────────────────────────
    def record_outcome(
        self,
        *,
        event_id: str = "",
        goal: str = "",
        claim_held: bool,
        detail: str = "",
    ) -> dict[str, float]:
        """Attribute an outcome back to the faculties that were eligible.

        ``claim_held`` is the faculty's own claim, not whether the outcome
        was pleasant. Anger's claim is that the boundary holds; guilt's is
        that the repair happens; revival's is that the practice resumes. A
        faculty that correctly produced a painful state is right, and
        scoring on valence would train it away.
        """
        now = time.time()
        with self._lock:
            self._outcomes_seen += 1
            eligible = [
                (trace, trace.eligibility(now))
                for trace in self._traces
                if (event_id and trace.event_id == event_id)
                or (goal and goal in trace.goals_touched)
            ]
            eligible = [(t, e) for t, e in eligible if e > 0.01]
            if not eligible:
                # An outcome that names nothing this system did leaves a
                # counter and nothing else. Spreading it over whoever
                # happened to be active is how a learner acquires
                # confident nonsense.
                self._outcomes_dropped += 1
                return {}

            total = sum(e * t.intensity for t, e in eligible) or 1.0
            moved: dict[str, float] = {}
            for trace, eligibility in eligible:
                share = (eligibility * trace.intensity) / total
                standing = self._standing.setdefault(
                    trace.faculty, Standing(trace.faculty)
                )
                before = standing.hit_rate
                target = 1.0 if claim_held else 0.0
                standing.hit_rate += _LEARNING_RATE.value * share * (target - before)
                standing.hit_rate = max(0.0, min(1.0, standing.hit_rate))
                standing.resolved += 1
                standing.unresolved = max(0, standing.unresolved - 1)
                if claim_held:
                    standing.confirmed += 1
                else:
                    standing.disconfirmed += 1
                moved[trace.faculty] = standing.hit_rate - before

            # A resolved trace is spent. Leaving it would let one outcome
            # be collected twice by the same firing.
            spent = {id(trace) for trace, _ in eligible}
            self._traces = [t for t in self._traces if id(t) not in spent]
            return moved

    # ── reading ───────────────────────────────────────────────────────
    def standing(self, faculty: str) -> Standing | None:
        with self._lock:
            return self._standing.get(faculty)

    def hit_rate(self, faculty: str) -> float | None:
        """Measured reliability, or None when nothing has resolved.

        None rather than 0.5: an unmeasured faculty is unmeasured, and a
        caller that cannot tell the difference will treat a guess as a
        finding.
        """
        with self._lock:
            standing = self._standing.get(faculty)
            if standing is None or not standing.measured:
                return None
            return standing.hit_rate

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            warm = sum(1 for t in self._traces if t.eligibility(now) > 0.01)
            return {
                "traces": len(self._traces),
                "traces_still_eligible": warm,
                "outcomes_seen": self._outcomes_seen,
                "outcomes_with_no_trace": self._outcomes_dropped,
                "standings": {
                    k: v.to_dict() for k, v in sorted(self._standing.items())
                },
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._traces.clear()
            self._standing.clear()
            self._outcomes_seen = 0
            self._outcomes_dropped = 0


_ATTRIBUTION: Attribution | None = None
_LOCK = checked_lock("core.interiority.attribution.singleton")


def get_attribution() -> Attribution:
    global _ATTRIBUTION
    with _LOCK:
        if _ATTRIBUTION is None:
            _ATTRIBUTION = Attribution()
        return _ATTRIBUTION


__all__ = ["Attribution", "Standing", "Trace", "get_attribution"]
