"""What a change to herself is worth, and why that number has to exist.

Two conditions have to hold before "she decided to develop" can be a fact
rather than a way of speaking, and both are provable rather than stipulated.

**One choice set.** Suppose acting and developing are separate modes with a
switch between them. The switch is a function of something. If it is a function
of the same quantity that ranks the actions, then it IS that ranking and the
modes were never separate. If it is a function of anything else, there are
situations where the ranking says develop and the switch says act, and in those
situations the decision was made by whoever wrote the switch.
`where_a_split_disagrees_with_the_whole` finds those situations by running
both, so the argument is checkable rather than asserted. A fixed ladder of
`if this fails, try that` is the degenerate case: its switch is a constant, and
a constant disagrees with every ranking somewhere.

**One estimable value.** A choice caused by the record has to vary when the
record varies. `the_choice_follows_the_record` varies it and looks. A policy
that answers the same thing on every record is not reading the record, whatever
it reads, and that is a test a hand-written ladder fails on purpose.

Together those are necessary. They are also sufficient, given the third thing
the previous mandate already built: the actions and the ranking are terms on
the same floor, so the path that revises one revises the other. That is the
whole architecture, and everything below is the arithmetic of it.

The worth of a developmental action
-----------------------------------
Not stipulated. It falls out of what the action does. Doing `d` costs `c(d)`
now and saves `g(d)` on each later occasion where it applies, and there are
`n(d)` such occasions, and `r(d)` is what it costs when it does not work out —
the failure, plus the tax a bigger library puts on every later search. So

    worth(d) = n(d) · g(d) − c(d) − r(d)

in one unit, candidates walked, because that is what search spends and what a
better language saves. Nothing in it is a constant chosen to make an experiment
come out; each of the four is read off `the_record_of_her_own_work`, and where
the record is silent the estimate is refused rather than defaulted. A refused
estimate is what makes exploration necessary — an action nobody can price is
worth trying precisely because trying it is how it gets priced — and
`the_price_of_finding_out` is what that costs, stated rather than hidden.

The term is on the floor, for the reason `the_order_she_tries_them_in` gives:
a rule that is a Python expression is the next authored level up. This one is
installed, kept, removed and replaced by the code that installs, keeps, removes
and replaces a head.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from core.cognition.the_floor_she_stands_on import (
    Code,
    L,
    MINUS,
    N,
    TIMES,
    V,
    build,
    how_long,
    read_back,
    run,
    written_down,
)
from core.cognition.the_record_of_her_own_work import (
    Episode,
    attribution,
    how_often,
    the_record,
)

__all__ = [
    "THE_WORTH",
    "WHAT_THE_WORTH_IS_GIVEN",
    "WhatItIsWorth",
    "forget_the_worth",
    "how_much_it_is_worth",
    "how_often_it_will_come_up",
    "the_choice_follows_the_record",
    "the_price_of_finding_out",
    "the_worth_read_back",
    "the_worth_she_uses",
    "the_worth_she_wrote",
    "what_each_occasion_would_save",
    "what_it_risks",
    "what_the_record_says_is_slow",
    "where_a_split_disagrees_with_the_whole",
    "written_worth",
]

logger = logging.getLogger("Aura.WhatItIsWorthDoing")

#: What the rule is handed, outermost binder first. All four in one unit.
WHAT_THE_WORTH_IS_GIVEN: tuple[str, ...] = (
    "occasions it would apply to",
    "what it saves on each",
    "what it costs to do",
    "what it costs when it does not work",
)

#: occasions × saving − cost − risk. Written where she can reach it.
THE_WORTH: Code = build(
    L(
        "occasions",
        L(
            "saving",
            L(
                "cost",
                L(
                    "risk",
                    MINUS(
                        MINUS(
                            TIMES(V("occasions"), V("saving")),
                            V("cost"),
                        ),
                        V("risk"),
                    ),
                ),
            ),
        ),
    )
)

_IN_USE: list[Code] = [THE_WORTH]

#: What one valuation may spend. Valuing is done once per candidate action, so
#: it is cheap on purpose; a rule that cannot answer inside it values nothing,
#: which makes the action unpriced rather than worthless.
_A_VALUATION_MAY_SPEND = 20_000


@dataclass(frozen=True, slots=True)
class WhatItIsWorth:
    """The four terms and what they come to, with the silences named."""

    occasions: int
    saving: int | None
    cost: int
    risk: int
    worth: int | None
    #: Which of the four could not be read off the record.
    unknown: tuple[str, ...] = ()

    @property
    def priced(self) -> bool:
        return self.worth is not None

    def describes(self) -> str:
        if self.worth is None:
            return f"unpriced ({', '.join(self.unknown) or 'no rule'})"
        return (
            f"{self.occasions}×{self.saving} − {self.cost} − {self.risk} "
            f"= {self.worth}"
        )


def the_worth_she_uses() -> Code:
    return _IN_USE[0]


def the_worth_she_wrote(term: Code) -> Code:
    """Put a different valuation in force. Same call shape as installing a head."""
    _IN_USE[0] = term
    logger.info("she values her own work differently: %d symbols", how_long(term))
    return term


def forget_the_worth() -> Code:
    """Back to the one she started with. The lesion."""
    _IN_USE[0] = THE_WORTH
    return THE_WORTH


def how_much_it_is_worth(
    *, occasions: int, saving: int, cost: int, risk: int
) -> int | None:
    """Run the valuation in force. Nothing where it refuses."""
    work: Any = _IN_USE[0]
    given = (int(occasions), int(saving), int(cost), int(risk))
    try:
        made = run(work, fuel=_A_VALUATION_MAY_SPEND)
        for one in given:
            if not hasattr(made, "body"):
                return None
            made = run(made.body, (one, *made.env), fuel=_A_VALUATION_MAY_SPEND)
        return int(made)
    except Exception:  # noqa: BLE001 - a refusal is an unpriced action
        return None


def written_worth() -> dict[str, Any]:
    return written_down(_IN_USE[0])


def the_worth_read_back(row: Any) -> Code | None:
    return read_back(row)


# --------------------------------------------------------------------------
# The four estimates, each off the record and none off a constant.
# --------------------------------------------------------------------------


def how_often_it_will_come_up(family: str) -> int:
    """Occasions ahead where this would apply.

    Taken to be as many as there have been. The record evidences no other
    horizon, and a horizon read off anything else would be a number chosen to
    make the arithmetic come out.
    """
    return how_often(family)


def what_each_occasion_would_save(
    kind: str, *, costs_now: int
) -> tuple[int | None, str]:
    """What one occasion would save, from what admissions of this kind saved before.

    The estimator is the record's own history of this kind of change: for each
    family where something of this kind was admitted, what the family cost
    before it and after. With no such history the estimate is refused, because
    the alternative is a prior that decides the experiment.
    """
    before: dict[str, list[int]] = {}
    after: dict[str, list[int]] = {}
    admitted_at: dict[str, int] = {}
    for at, one in enumerate(the_record().kept):
        if one.admitted == kind and one.family not in admitted_at:
            admitted_at[one.family] = at
    if not admitted_at:
        return None, "nothing of this kind has been admitted before"
    for at, one in enumerate(the_record().kept):
        when = admitted_at.get(one.family)
        if when is None or at == when:
            # The occasion the change was admitted on is what admitting it
            # cost, and counting that as an occasion afterwards makes every
            # change look like it saved less than it did.
            continue
        (before if at < when else after).setdefault(one.family, []).append(
            one.walked
        )
    fractions: list[float] = []
    for family, was in before.items():
        now = after.get(family)
        if not now or not was:
            continue
        was_mean = sum(was) / len(was)
        now_mean = sum(now) / len(now)
        if was_mean <= 0:
            continue
        fractions.append(max(0.0, (was_mean - now_mean) / was_mean))
    if not fractions:
        return None, "no family was measured both before and after"
    share = sum(fractions) / len(fractions)
    return int(round(share * max(0, int(costs_now)))), ""


def what_it_risks(kind: str, *, cost: int, entries: int) -> int:
    """What it costs when it does not work out, plus the tax on every later search.

    Two parts, both measured. A change of this kind did not work out when it
    was admitted and the family it was admitted for went on costing what it
    cost, so the failure rate is a count over the record rather than a belief
    about how often invention fails. The second part is what one more library
    entry does to every later scoring pass, which is why a language that keeps
    everything gets slower at everything.
    """
    worked = missed = 0
    admitted_at: dict[str, int] = {}
    for at, one in enumerate(the_record().kept):
        if one.admitted == kind and one.family not in admitted_at:
            admitted_at[one.family] = at
    for family, when in admitted_at.items():
        was = [
            one.walked
            for at, one in enumerate(the_record().kept)
            if one.family == family and at < when
        ]
        now = [
            one.walked
            for at, one in enumerate(the_record().kept)
            if one.family == family and at > when
        ]
        if not was or not now:
            continue
        if sum(now) / len(now) < sum(was) / len(was):
            worked += 1
        else:
            missed += 1
    rate = (missed + 1) / (worked + missed + 2)
    return int(round(rate * max(0, int(cost)))) + max(0, int(entries))


def the_price_of_finding_out(cost: int) -> int:
    """What it costs to price an action nobody can price yet.

    Doing it is the only way to learn what it saves, so the price of the
    information is the price of the action. Exploration is not free and is not
    a knob; it is this number, stated.
    """
    return max(0, int(cost))


# --------------------------------------------------------------------------
# The two conditions, run rather than argued.
# --------------------------------------------------------------------------


def the_choice_follows_the_record(
    choose: Callable[[], Any],
    situations: Sequence[Callable[[], None]],
) -> bool:
    """Does what she chooses change when the record changes?

    Each situation writes a different record; the chooser runs after each. A
    chooser that answers the same thing every time is not reading the record,
    and a mechanism that is not reading the record is not deciding from it —
    whatever it says about itself.
    """
    answers = []
    for put_it_in_place in situations:
        put_it_in_place()
        answers.append(repr(choose()))
    return len(set(answers)) > 1


def where_a_split_disagrees_with_the_whole(
    *,
    ordinary: Sequence[Any],
    developmental: Sequence[Any],
    worth: Callable[[Any], int | None],
    switch: Callable[[Sequence[Any], Sequence[Any]], str],
    situations: Iterable[Callable[[], None]] = (),
) -> list[dict[str, Any]]:
    """Situations where a mode switch overrides the ranking.

    Run the split — a switch picking a mode, then the best action inside it —
    beside the whole, which ranks everything together. Every row returned is a
    situation where something other than the value decided, and the author of
    that decision is whoever wrote the switch.

    An empty list means the switch agreed everywhere, which means it computed
    the ranking, which means the two modes were one all along.
    """
    found: list[dict[str, Any]] = []
    rounds = list(situations) or [lambda: None]
    for put_it_in_place in rounds:
        put_it_in_place()
        both = list(ordinary) + list(developmental)
        priced = [(one, worth(one)) for one in both]
        together = max(
            (row for row in priced if row[1] is not None),
            key=lambda row: row[1],
            default=(None, None),
        )[0]
        side = switch(ordinary, developmental)
        inside = list(developmental if side == "develop" else ordinary)
        apart = max(
            (
                (one, worth(one))
                for one in inside
                if worth(one) is not None
            ),
            key=lambda row: row[1],
            default=(None, None),
        )[0]
        if together is not apart:
            found.append(
                {
                    "the switch said": side,
                    "the split chose": apart,
                    "the ranking chose": together,
                }
            )
    return found


def what_the_record_says_is_slow() -> list[dict[str, Any]]:
    """Which part of her spends the search, ordered worst first.

    The self-model, and it is a reading rather than a belief. A route that is
    tried often, answers seldom and costs much is where the time goes, and that
    is a fact about the record rather than something she has to be told.
    """
    rows = []
    for where, how in attribution().items():
        misses = how["episodes"] - how["answered"]
        rows.append(
            {
                "route": where,
                "each": how["each"],
                "walked": how["walked"],
                "answered": how["answered"],
                "missed": misses,
                # What it spends per answer, counting what it spent answering
                # nothing. A route that never answers is priced at everything
                # it spent, which is the honest number.
                "per answer": (
                    how["walked"] / how["answered"]
                    if how["answered"]
                    else float(how["walked"])
                ),
            }
        )
    rows.sort(key=lambda row: -row["per answer"])
    return rows


def _episodes_for(family: str) -> list[Episode]:
    return [one for one in the_record().kept if one.family == family]
