"""core/cognition/value_of_computation.py — is more thinking worth anything.

``cognitive_cost.py`` answers what thinking costs. This answers the other half,
and the two together are what makes a spend decision rather than a spend habit.

The idea the whole module rests on is narrow and it is the one that makes the
question answerable: **more computation is worth something only if it can
change what she does.** Uncertainty on its own buys nothing. If the leading
option is ahead by more than further deliberation could plausibly close, then
thinking longer produces a better-argued version of the same act, and the
argument is not the deliverable.

That turns an unanswerable question — how much better would the answer be? —
into a measurable one: how far has extra spend actually moved the leader
before? :class:`Swing` keeps that record, and it is checked against its own
null. Deliberations where nothing extra was spent must show no movement; if
they do, the record is measuring drift rather than thinking, and the module
says UNMEASURED rather than reporting the drift as value.

The second question is the one the module exists for as much as the first:
whether to spend on learning at all. Learning is a cost paid once against a
benefit paid per use, so it is worth it exactly when the thing will be used
enough to repay it — and a system that cannot ask that will study whatever is
in front of it forever. :func:`worth_learning` asks it, and returns NO for a
real skill with a real gain that will be needed twice.

Every verdict can come back UNMEASURED, which is the honest answer when
nothing has been recorded, and it is not the same as NO.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Cognition.VOC")


def _checked_lock(name: str, *, reentrant: bool = False):
    """The repo's instrumented lock, so lockdep can see this one too.

    A raw threading lock is invisible to the ABBA detector, and a detector
    that sees only some of the locks reports clean while the deadlock it
    exists to find is assembled out of the others.
    """

    from core.runtime.lockdep import checked_lock

    return checked_lock(name, reentrant=reentrant)


#: Observations needed before the swing record can say anything. Below this
#: the spread is one or two deliberations and calling it a distribution would
#: be dressing an anecdote as a measurement.
MIN_OBSERVATIONS = 8

#: How far above its own null the measured swing has to sit before it is
#: treated as movement caused by thinking rather than as noise.
SWING_MARGIN = 0.02

#: Percentile of the swing distribution used as "what more thinking could
#: plausibly do". The high end rather than the mean: the question is whether
#: a change is possible, not whether it is typical.
SWING_PERCENTILE = 0.9

#: How many observations to keep.
CAPACITY = 512


class Worth(StrEnum):
    """Whether a spend is worth making."""

    #: It could change the outcome and the change is worth the cost.
    WORTH = "worth"
    #: The leader is too far ahead for any plausible swing to catch it.
    SETTLED = "settled"
    #: It could change the outcome and the change is not worth the cost.
    TOO_EXPENSIVE = "too_expensive"
    #: Nothing has been recorded, so no swing is known.
    UNMEASURED = "unmeasured"

    @property
    def spend(self) -> bool:
        return self is Worth.WORTH


@dataclass(frozen=True)
class Judgement:
    """One verdict about a spend, and the numbers behind it."""

    worth: Worth
    #: How far the leader is ahead now.
    margin: float
    #: What further spend could plausibly move it, or None when unknown.
    plausible_swing: float | None
    cost: float
    #: Expected value of the spend: the chance it changes things, times what
    #: changing them is worth, minus what it costs.
    expected_value: float | None
    because: str

    @property
    def spend(self) -> bool:
        return self.worth.spend

    def to_dict(self) -> dict[str, Any]:
        return {
            "worth": str(self.worth),
            "spend": self.spend,
            "margin": round(self.margin, 4),
            "plausible_swing": (
                None if self.plausible_swing is None else round(self.plausible_swing, 4)
            ),
            "cost": round(self.cost, 4),
            "expected_value": (
                None if self.expected_value is None else round(self.expected_value, 4)
            ),
            "because": self.because,
        }


@dataclass(frozen=True)
class Observation:
    """One deliberation: what was spent, and how far the leader moved."""

    spend: float
    movement: float
    changed_decision: bool = False
    at: float = field(default_factory=time.time)


class Swing:
    """How far extra spend has actually moved the leading margin."""

    def __init__(self, name: str = "default") -> None:
        self.name = str(name)
        self._observations: list[Observation] = []
        self._lock = _checked_lock("value_of_computation", reentrant=True)

    def observe(
        self, *, spend: float, movement: float, changed_decision: bool = False
    ) -> None:
        record = Observation(
            spend=max(0.0, float(spend)),
            movement=abs(float(movement)),
            changed_decision=bool(changed_decision),
        )
        with self._lock:
            self._observations.append(record)
            if len(self._observations) > CAPACITY:
                del self._observations[0]

    def _split(self) -> tuple[list[Observation], list[Observation]]:
        """Deliberations that spent something, and those that spent nothing.

        The second group is the null. Movement without spend is drift — the
        scores wobbling for reasons that have nothing to do with thinking —
        and a record that cannot separate the two is measuring the wobble.
        """
        with self._lock:
            observations = list(self._observations)
        spent = [o for o in observations if o.spend > 0.0]
        idle = [o for o in observations if o.spend <= 0.0]
        return spent, idle

    def plausible(self) -> float | None:
        """What further thinking could plausibly do, or None if unknown."""
        spent, idle = self._split()
        if len(spent) < MIN_OBSERVATIONS:
            return None
        moved = sorted(o.movement for o in spent)
        high = moved[min(len(moved) - 1, int(len(moved) * SWING_PERCENTILE))]
        if not idle:
            # No null arm. The record cannot separate thinking from drift, so
            # it does not get to claim the movement was thinking.
            return None
        drift = statistics.median(o.movement for o in idle)
        if high - drift <= SWING_MARGIN:
            return None
        return high - drift

    def change_rate(self) -> float | None:
        """How often extra spend changed the decision, where it was recorded."""
        spent, _idle = self._split()
        if len(spent) < MIN_OBSERVATIONS:
            return None
        return sum(1 for o in spent if o.changed_decision) / len(spent)

    def snapshot(self) -> dict[str, Any]:
        spent, idle = self._split()
        return {
            "name": self.name,
            "observations": len(spent) + len(idle),
            "with_spend": len(spent),
            "null_arm": len(idle),
            "plausible_swing": self.plausible(),
            "change_rate": self.change_rate(),
        }

    def clear(self) -> None:
        with self._lock:
            self._observations.clear()


def worth_continuing(
    *,
    margin: float,
    cost: float,
    stakes: float = 1.0,
    swing: Swing | None = None,
) -> Judgement:
    """Whether another round of thinking is worth making.

    ``margin`` is how far the leading option is ahead, ``cost`` is what the
    next round costs in the same unit as ``stakes``, and ``stakes`` is what
    getting the decision right is worth. All three in one unit is what makes
    the subtraction mean anything.
    """
    record = swing if swing is not None else default_swing()
    plausible = record.plausible()
    lead = abs(float(margin))

    if plausible is None:
        return Judgement(
            worth=Worth.UNMEASURED,
            margin=lead,
            plausible_swing=None,
            cost=float(cost),
            expected_value=None,
            because=(
                "nothing recorded says how far more thinking moves a decision, "
                "and movement with no spend to compare it against is drift"
            ),
        )

    if lead > plausible:
        return Judgement(
            worth=Worth.SETTLED,
            margin=lead,
            plausible_swing=plausible,
            cost=float(cost),
            expected_value=0.0,
            because=(
                f"the leader is {lead:.3f} ahead and more thinking has moved a "
                f"decision by at most {plausible:.3f}; another round produces a "
                "better-argued version of the same act"
            ),
        )

    # It could change things. Whether that is worth paying for depends on how
    # often it actually does and on what the decision is worth.
    rate = record.change_rate()
    chance = rate if rate is not None else _overlap_chance(lead, plausible)
    value = chance * float(stakes) - float(cost)
    if value > 0.0:
        return Judgement(
            worth=Worth.WORTH,
            margin=lead,
            plausible_swing=plausible,
            cost=float(cost),
            expected_value=value,
            because=(
                f"a swing of {plausible:.3f} can still close a lead of {lead:.3f}, "
                f"and at a {chance:.0%} chance of changing a decision worth "
                f"{stakes:.3f} that is worth {float(cost):.3f}"
            ),
        )
    return Judgement(
        worth=Worth.TOO_EXPENSIVE,
        margin=lead,
        plausible_swing=plausible,
        cost=float(cost),
        expected_value=value,
        because=(
            f"it could still change the decision, but a {chance:.0%} chance at "
            f"{stakes:.3f} does not repay {float(cost):.3f}"
        ),
    )


def _overlap_chance(margin: float, plausible: float) -> float:
    """How likely a swing that large is to close a lead that size.

    Used only where the change rate has not been recorded. Linear in how much
    of the plausible swing the lead uses up, which is the weakest assumption
    that still distinguishes a near-tie from a near-miss.
    """
    if plausible <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (margin / plausible)))


def worth_learning(
    *,
    cost: float,
    gain_per_use: float,
    expected_uses: float,
    retention: float = 1.0,
) -> Judgement:
    """Whether to spend on learning something at all.

    Learning is paid once and repays per use, so it is worth it exactly when
    the thing will be needed enough times to cover the cost. A system that
    cannot ask this studies whatever is in front of it, forever, and calls the
    studying progress.

    ``retention`` is the fraction that survives to the next use. Something
    learned and forgotten before it is needed twice repays once however good
    the lesson was.
    """
    uses = max(0.0, float(expected_uses))
    per_use = float(gain_per_use)
    kept = max(0.0, min(1.0, float(retention)))
    # Each subsequent use is worth what survived to it. A geometric sum,
    # because retention compounds rather than applying once.
    if kept >= 1.0:
        total = per_use * uses
    elif kept <= 0.0:
        total = per_use * min(uses, 1.0)
    else:
        whole = math.floor(uses)
        total = per_use * (1.0 - kept**whole) / (1.0 - kept)
        total += per_use * (kept**whole) * (uses - whole)
    value = total - float(cost)
    if value > 0.0:
        return Judgement(
            worth=Worth.WORTH,
            margin=0.0,
            plausible_swing=None,
            cost=float(cost),
            expected_value=value,
            because=(
                f"{uses:.1f} expected uses at {per_use:.3f} each, {kept:.0%} "
                f"retained between them, repays {float(cost):.3f}"
            ),
        )
    return Judgement(
        worth=Worth.TOO_EXPENSIVE,
        margin=0.0,
        plausible_swing=None,
        cost=float(cost),
        expected_value=value,
        because=(
            f"{uses:.1f} expected uses at {per_use:.3f} each returns "
            f"{total:.3f}, which does not repay {float(cost):.3f}"
        ),
    )


_SWINGS: dict[str, Swing] = {}
_SWINGS_LOCK = _checked_lock("value_of_computation")


def swing_record(name: str = "default") -> Swing:
    with _SWINGS_LOCK:
        found = _SWINGS.get(name)
        if found is None:
            found = Swing(name)
            _SWINGS[name] = found
        return found


def default_swing() -> Swing:
    return swing_record("default")


def reset_swings() -> None:
    with _SWINGS_LOCK:
        _SWINGS.clear()


def observe_deliberation(
    *,
    scores_before: Sequence[float],
    scores_after: Sequence[float],
    spend: float,
    name: str = "default",
) -> None:
    """Record what one round of extra thinking did to a ranking."""
    if not scores_before or not scores_after:
        return
    before = sorted(scores_before, reverse=True)
    after = sorted(scores_after, reverse=True)
    margin_before = before[0] - (before[1] if len(before) > 1 else before[0])
    margin_after = after[0] - (after[1] if len(after) > 1 else after[0])
    changed = bool(
        scores_before
        and scores_after
        and max(range(len(scores_before)), key=lambda i: scores_before[i])
        != max(range(len(scores_after)), key=lambda i: scores_after[i])
    )
    swing_record(name).observe(
        spend=spend, movement=margin_after - margin_before, changed_decision=changed
    )


__all__ = [
    "CAPACITY",
    "MIN_OBSERVATIONS",
    "SWING_MARGIN",
    "SWING_PERCENTILE",
    "Judgement",
    "Observation",
    "Swing",
    "Worth",
    "default_swing",
    "observe_deliberation",
    "reset_swings",
    "swing_record",
    "worth_continuing",
    "worth_learning",
]
