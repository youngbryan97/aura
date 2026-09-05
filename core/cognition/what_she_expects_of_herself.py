"""What she expects a change to do, written down before it is made.

Every estimate so far has been made and then used. None of them has been made,
recorded, and afterwards compared with what happened — and an estimate never
compared with an outcome is a rule for producing numbers rather than a model of
anything. A policy scored by one of those is optimising the rule.

So this writes the prediction down first. Before a developmental action runs,
`what_she_expects` says what it will cost, whether it will be kept, and what
the families she has met will gain. Afterwards `what_actually_happened` says
what did. The gap between them is `how_well_she_knows_herself`, and it is the
one number here that cannot be gamed by the thing it measures: the prediction
is fixed before the outcome exists.

The forecaster
--------------
`how_long_this_will_take` predicts how many candidates a family will cost
before she has spent them, from families whose shape resembles it. That is the
model of her own future learning the whole question turns on, and it is
deliberately plain — a shape-weighted average over what similar families
actually took. Plain is what lets it be checked.

Where the record is thin the answer is that she does not know, and that is
returned rather than defaulted. A forecaster that always answers is a
forecaster whose answers say nothing.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

__all__ = [
    "WhatSheExpected",
    "forget_what_she_expected",
    "how_long_this_will_take",
    "how_well_she_knows_herself",
    "what_actually_happened",
    "what_she_expects",
    "what_she_expected",
]

logger = logging.getLogger("Aura.WhatSheExpectsOfHerself")


@dataclass
class WhatSheExpected:
    """A prediction, fixed before the outcome, and the outcome beside it."""

    about: str
    will_cost: int
    will_be_kept: float
    will_gain: int
    #: Filled afterwards. Nothing until it is.
    did_cost: int | None = None
    was_kept: bool | None = None
    did_gain: int | None = None

    @property
    def settled(self) -> bool:
        return self.was_kept is not None

    def describes(self) -> str:
        if not self.settled:
            return (
                f"{self.about}: expects {self.will_cost:,} to cost, "
                f"{self.will_be_kept:.2f} to keep, {self.will_gain:,} to gain"
            )
        return (
            f"{self.about}: cost {self.did_cost:,} against {self.will_cost:,}, "
            f"{'kept' if self.was_kept else 'not kept'} against "
            f"{self.will_be_kept:.2f}"
        )


_EXPECTED: list[WhatSheExpected] = []


def what_she_expected() -> tuple[WhatSheExpected, ...]:
    return tuple(_EXPECTED)


def forget_what_she_expected() -> None:
    _EXPECTED.clear()


def what_she_expects(name: str, *, costs_now: int) -> WhatSheExpected:
    """Say what a change will do, before it does it.

    From what that action has done before, and from what changes of any kind
    have done where it has no history of its own. Written down as a row, so
    the comparison afterwards is against something fixed rather than against a
    memory of what she would have said.
    """
    from core.cognition.the_record_of_her_own_work import what_it_has_cost
    from core.cognition.what_it_is_worth_doing import how_often_a_change_has_paid
    from core.cognition.what_she_could_do_next import what_it_has_done

    mine = what_it_has_done(name)
    made = WhatSheExpected(
        about=str(name),
        will_cost=int(what_it_has_cost(name) or costs_now),
        will_be_kept=(
            mine.how_often_it_pays if mine.taken else how_often_a_change_has_paid()
        ),
        will_gain=int(mine.what_it_gains) if mine.gained else 0,
    )
    _EXPECTED.append(made)
    if len(_EXPECTED) > 256:
        del _EXPECTED[:-256]
    return made


def what_actually_happened(
    name: str, *, cost: int, kept: bool, gained: int = 0
) -> WhatSheExpected | None:
    """Fill in the outcome beside the most recent unsettled prediction."""
    for one in reversed(_EXPECTED):
        if one.about == str(name) and not one.settled:
            one.did_cost = int(cost)
            one.was_kept = bool(kept)
            one.did_gain = int(gained)
            return one
    return None


def how_well_she_knows_herself() -> dict[str, Any]:
    """The gap between what she expected and what happened.

    Three readings, all over settled predictions only. How far the cost
    estimate was out, in the unit costs are in. Whether the keep estimate was
    on the right side of a half. And the correlation between predicted and
    actual gain, which is the one that says whether the model is a model or an
    average.

    Nothing here is used to decide anything. It exists so that a claim about
    her self-model can be checked, and so that a self-model getting worse is
    visible rather than quiet.
    """
    settled = [one for one in _EXPECTED if one.settled]
    if not settled:
        return {"predictions": 0}
    cost_out = [
        abs((one.did_cost or 0) - one.will_cost) / max(1, one.will_cost)
        for one in settled
    ]
    right_side = sum(
        1
        for one in settled
        if (one.will_be_kept > 0.5) == bool(one.was_kept)
    )
    gains = [(one.will_gain, one.did_gain or 0) for one in settled if one.will_gain]
    together = 0.0
    if len(gains) > 2:
        try:
            together = statistics.correlation(
                [one for one, _ in gains], [other for _, other in gains]
            )
        except (statistics.StatisticsError, ValueError):
            together = 0.0
    return {
        "predictions": len(settled),
        "cost out by": round(sum(cost_out) / len(cost_out), 3),
        "keep on the right side": round(right_side / len(settled), 3),
        "gain tracks outcome": round(together, 3),
    }


def how_long_this_will_take(cases: Any) -> tuple[int | None, str]:
    """How many candidates this family will cost, before spending them.

    A shape-weighted average over families she has met: the more a stored
    family's cases resemble these, the more its cost counts. Where nothing
    resembles them the answer is that she does not know, which is returned
    rather than replaced by an average of everything.

    This is the model of her own future learning, and it is plain on purpose.
    A forecaster nobody can check is a forecaster nobody should believe.
    """
    from core.cognition.the_record_of_her_own_work import the_record

    lengths = {len(before) for before, _after in cases}
    made = any(set(after) - set(before) for before, after in cases)
    weighed: list[tuple[float, int]] = []
    for one in the_record().kept:
        if not one.about:
            continue
        theirs = {len(before) for before, _after in one.about}
        made_too = any(set(after) - set(before) for before, after in one.about)
        # Shape rather than surface: the lengths asked about and whether values
        # were moved or made. Two families over completely different numbers
        # count as alike when both of those match.
        alike = len(lengths & theirs) / max(1, len(lengths | theirs))
        if made_too != made:
            alike *= 0.5
        if alike > 0:
            weighed.append((alike, one.walked))
    if not weighed:
        return None, "nothing she has met resembles this"
    whole = sum(alike for alike, _cost in weighed)
    return (
        int(round(sum(alike * cost for alike, cost in weighed) / whole)),
        f"from {len(weighed)} families like it",
    )
