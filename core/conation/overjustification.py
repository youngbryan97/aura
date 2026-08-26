"""core/conation/overjustification.py — paying for something you did for free.

Deci's 1971 result, replicated many times since: pay people to do a thing they
were already doing for its own sake, and they do less of it once the pay
stops. Lepper, Greene and Nisbett found the same with children and drawing.
The reward does not add to the intrinsic motive. It substitutes for it, and
the substitution outlasts it.

The usual reading is a curiosity about human psychology. For an agent that
learns from its own outcomes it is a live hazard, because the mechanism that
produces it is exactly the mechanism a value-learning system runs on. An agent
that folds every reward into one cached value cannot help doing this to
itself: any autotelic pull that happens to coincide with a useful outcome gets
re-attributed to the outcome, and once the outcome stops, the pull is gone.
The agent has traded a reason of its own for a reason belonging to a task.

That is why ``Instrumentality`` is a field on ``ConativeState`` rather than a
note. Telling the two apart is what makes this correctable.

## What is corrected

Nothing about the behaviour at the time. An autotelic act that also earns
something is still worth doing, and refusing rewards to protect curiosity
would be a strange kind of care.

What is corrected is the *attribution*. When an autotelic motive is followed
by an extrinsic payoff, the payoff is recorded against the incentive rather
than folded into its cached value, so the intrinsic pull is neither credited
nor debited for something it did not do. The cached value keeps tracking what
the contact itself was worth.

    contaminated = extrinsic payoffs delivered to an autotelic motive
    protection   = the share of learned value that stayed intrinsic

A high contamination count with intact intrinsic value means the guard is
working. A high count with collapsed intrinsic value means it is not, and the
readout says which.

## Why this is not simply "ignore extrinsic rewards"

The agent needs to learn from payoffs; that is what the instrumental path is
for. The distinction is which predictor gets the update. An instrumental
motive's payoff belongs in its cached value, because the payoff is the point.
An autotelic motive's payoff belongs in a ledger, because the point was
elsewhere and letting the payoff teach would overwrite it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.conation.origins import Instrumentality, ValueOrigin

#: Origins whose value is the contact itself. A payoff attached to one of
#: these is the overjustification case; a payoff attached to a homeostatic or
#: vicarious motive is not, because those were never about the doing.
INTRINSIC_ORIGINS = frozenset(
    {ValueOrigin.EPISTEMIC, ValueOrigin.AESTHETIC}
)


@dataclass
class ContaminationRecord:
    """Extrinsic payoffs delivered to one autotelic incentive."""

    key: str
    origin: str
    payoffs: int = 0
    total_payoff: float = 0.0
    intrinsic_at_first_payoff: float | None = None
    intrinsic_latest: float | None = None
    last_update: float = field(default_factory=time.time)

    def observe(self, payoff: float, intrinsic_now: float) -> None:
        self.payoffs += 1
        self.total_payoff += max(0.0, float(payoff))
        if self.intrinsic_at_first_payoff is None:
            self.intrinsic_at_first_payoff = intrinsic_now
        self.intrinsic_latest = intrinsic_now
        self.last_update = time.time()

    def erosion(self) -> float | None:
        """Fall in intrinsic pull since the first payoff arrived.

        Positive means the intrinsic motive has weakened while payoffs were
        being delivered, which is the measurable form of the effect. ``None``
        before a payoff has been seen, because there is nothing to compare to.
        """
        if self.intrinsic_at_first_payoff is None or self.intrinsic_latest is None:
            return None
        return max(0.0, self.intrinsic_at_first_payoff - self.intrinsic_latest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "origin": self.origin,
            "payoffs": self.payoffs,
            "total_payoff": round(self.total_payoff, 4),
            "intrinsic_at_first_payoff": self.intrinsic_at_first_payoff,
            "intrinsic_latest": self.intrinsic_latest,
            "erosion": self.erosion(),
        }


class OverjustificationGuard:
    """Keeps an extrinsic payoff from teaching an intrinsic predictor."""

    MAX_RECORDS = 256

    def __init__(self) -> None:
        self._records: dict[str, ContaminationRecord] = {}

    @staticmethod
    def is_at_risk(instrumentality: Instrumentality, origin: ValueOrigin | None) -> bool:
        """Whether a payoff here would overwrite a reason of the agent's own."""
        return (
            instrumentality is Instrumentality.AUTOTELIC
            and origin is not None
            and origin in INTRINSIC_ORIGINS
        )

    def observe_payoff(
        self,
        key: str,
        *,
        origin: ValueOrigin | None,
        payoff: float,
        intrinsic_now: float,
    ) -> ContaminationRecord:
        """Record an extrinsic payoff against an autotelic incentive."""
        record = self._records.get(key)
        if record is None:
            if len(self._records) >= self.MAX_RECORDS:
                stalest = min(self._records.values(), key=lambda r: r.last_update)
                self._records.pop(stalest.key, None)
            record = ContaminationRecord(key=key, origin=str(origin or "unknown"))
            self._records[key] = record
        record.observe(payoff, intrinsic_now)
        return record

    def eroded(self, *, min_payoffs: int = 3) -> list[dict[str, Any]]:
        """Autotelic motives that have weakened while being paid for.

        The list a self-model reads to answer "what did I used to do for its
        own sake". It exists because instrumentality is a typed field; a
        system with one reward scalar cannot form the question.
        """
        rows = [
            record.to_dict()
            for record in self._records.values()
            if record.payoffs >= min_payoffs and (record.erosion() or 0.0) > 0.0
        ]
        rows.sort(key=lambda row: -(row["erosion"] or 0.0))
        return rows

    def status(self) -> dict[str, Any]:
        protected = sum(r.payoffs for r in self._records.values())
        return {
            "protected_payoffs": protected,
            "incentives": len(self._records),
            "eroded": self.eroded()[:5],
        }
