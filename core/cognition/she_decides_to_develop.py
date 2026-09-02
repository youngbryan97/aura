"""Who decided, and on what grounds — the policy, and the trace that shows it.

A ladder does not decide. It runs the next thing because the last thing
returned nothing, and if you ask it why it changed its own language the honest
answer is that a person put that line under that line. This is the replacement:
one ranking over everything she could do, ordinary and developmental together,
scored by `what_it_is_worth_doing`, with the grounds of every choice written
down beside it.

Three things it can do, and the third is the one that makes the first two
mean anything.

**Choose a priced action.** Occasions times saving, less cost, less risk, all
four off the record. The best one runs.

**Choose an unpriced one.** An action nobody can price is worth trying exactly
when finding out costs less than what the bottleneck is spending per answer,
and both of those are readings rather than settings. This is why exploration
is here at all: an action that has never been taken has no history, and the
only way to give it one is to take it.

**Refuse.** When nothing is worth doing, doing nothing is the decision and it
is recorded as one. A mechanism that can only ever say yes is not choosing, and
`the_choice_follows_the_record` would catch it.

The budget is not a constant. Developing a family may spend what answering that
family is going to spend anyway — its cost per occasion times the occasions the
record says are coming — because past that it cannot pay back and the search
for it would be the thing making her slow.

The trace
---------
Every stage of every developmental episode is written down with who started it:
`she` when the ranking chose it, `asked` when a caller named it. That is what
makes "she decided" checkable. An episode whose stages all say `asked` is a
harness call however it is described in prose, and the trace says so without
anybody having to be honest about it.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from core.cognition.the_record_of_her_own_work import (
    how_often,
    note_an_episode,
    the_record,
    what_it_has_cost,
)
from core.cognition.how_a_change_is_promoted import promote
from core.cognition.how_sure_she_is import (
    after_the_winners_curse,
    how_much_to_spend_on_developing,
    the_bar_right_now,
    which_to_try,
)
from core.cognition.what_it_is_worth_doing import (
    WhatItIsWorth,
    how_often_a_change_has_paid,
    how_much_it_is_worth,
    how_often_it_will_come_up,
    the_price_of_finding_out,
    what_each_occasion_would_save,
    what_it_risks,
    what_the_record_says_is_slow,
)
from core.cognition.what_she_expects_of_herself import (
    what_actually_happened,
    what_she_expects,
)
from core.cognition.what_she_could_do_next import (
    ADevelopmentalAction,
    note_what_it_did,
    the_actions_she_has,
    what_it_has_done,
)

__all__ = [
    "Decision",
    "Stage",
    "forget_the_trace",
    "she_decides_to_develop",
    "she_develops_herself",
    "the_trace",
    "what_it_may_spend",
    "what_is_worth_doing_now",
    "what_to_do_next",
    "who_started_it",
    "why_each_one",
]

logger = logging.getLogger("Aura.SheDecidesToDevelop")


@dataclass(frozen=True, slots=True)
class Stage:
    """One step of one developmental episode, and who started it."""

    #: trigger, diagnosis, proposal, evaluation, installation, refusal
    what: str
    #: "she" where the ranking chose it, "asked" where a caller named it.
    started_by: str
    about: str = ""
    when: float = field(default_factory=time.monotonic)

    def describes(self) -> str:
        return f"{self.what} ← {self.started_by}" + (
            f" ({self.about})" if self.about else ""
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """What she decided to do about herself, and the grounds."""

    action: ADevelopmentalAction | None
    worth: WhatItIsWorth | None
    #: chosen, exploring, refused
    because: str
    grounds: str
    started_by: str = "she"
    #: Every action she considered, priced, worst last.
    considered: tuple[tuple[str, WhatItIsWorth], ...] = ()

    def describes(self) -> str:
        named = self.action.name if self.action else "nothing"
        return f"{named}: {self.because} — {self.grounds}"


_TRACE: list[Stage] = []

#: Whether a developmental episode is already running.
#:
#: Developing can cause development. Letting go of a part retracts it, a
#: retraction leaves families that used to be sayable, and rebuilding those is
#: an ordinary developmental objective — so the policy is asked again from
#: inside a decision the policy made. Measured before this guard: eight hundred
#: and thirty-four triggers on three families, eight hundred and twenty-seven of
#: them refusals, which is a recursion bottoming out rather than a system
#: deciding eight hundred times.
#:
#: One episode at a time. What is refused here is a re-entry and not an
#: opportunity: the outer episode is still running and will record what came of
#: it, so nothing is lost by declining to start a second inside it.
_ALREADY_DECIDING = [False]


def the_trace() -> tuple[Stage, ...]:
    return tuple(_TRACE)


def forget_the_trace() -> None:
    _TRACE.clear()
    _ALREADY_DECIDING[0] = False


def _note(what: str, started_by: str, about: str = "") -> Stage:
    made = Stage(what=what, started_by=started_by, about=about)
    _TRACE.append(made)
    if len(_TRACE) > 4096:
        del _TRACE[:1024]
    return made


def who_started_it(what: str | None = None) -> dict[str, int]:
    """How many stages each initiator started, over the trace so far."""
    counted: dict[str, int] = {}
    for one in _TRACE:
        if what is not None and one.what != what:
            continue
        counted[one.started_by] = counted.get(one.started_by, 0) + 1
    return counted


def what_it_may_spend(family: str, *, costs_now: int) -> int:
    """What developing this family may spend, from what answering it will.

    Cost per occasion times the occasions the record says are coming. Past
    that a change cannot pay for itself even if it works perfectly, so the
    ceiling is derived from the family rather than set.
    """
    return max(0, int(costs_now)) * max(1, how_often_it_will_come_up(family))


def _price(
    action: ADevelopmentalAction, family: str, *, costs_now: int
) -> WhatItIsWorth:
    occasions = how_often_it_will_come_up(family)
    # What this action itself has gained comes first, and what changes of its
    # kind have gained second. An action with a history of its own is the
    # closest thing to evidence there is; falling back to the kind is a
    # generalisation and falling back to nothing is honest.
    #
    # This is also the calibration: the estimate is the outcome, so a value
    # that drifts away from what happens cannot, because it is what happens.
    mine = what_it_has_done(action.name)
    if mine.gained:
        saving, why = int(mine.what_it_gains), ""
    else:
        saving, why = what_each_occasion_would_save(
            action.kind, costs_now=costs_now
        )
    # Measured first, stated second, and the occasion in hand last. An action
    # that has run has a cost rather than an estimate of one.
    cost = what_it_has_cost(action.name) or action.price or costs_now
    risk = what_it_risks(action.kind, cost=cost, entries=len(the_record().uses))
    if saving is None:
        return WhatItIsWorth(
            occasions=occasions,
            saving=None,
            cost=cost,
            risk=risk,
            worth=None,
            unknown=("what it saves: " + why,),
        )
    worth = how_much_it_is_worth(
        occasions=occasions, saving=saving, cost=cost, risk=risk
    )
    return WhatItIsWorth(
        occasions=occasions,
        saving=saving,
        cost=cost,
        risk=risk,
        worth=worth,
        unknown=() if worth is not None else ("the rule refused",),
    )


def what_to_do_next(
    family: str,
    *,
    costs_now: int,
    among: Sequence[ADevelopmentalAction] | None = None,
    asked_for: str | None = None,
    this_one_is_lost: bool = False,
    a_question_is_waiting: bool = False,
) -> Decision:
    """Rank everything she could do, and choose. Or choose nothing.

    `this_one_is_lost` says the occasion in hand produces nothing unless
    something changes. It matters because the value below is a value of future
    savings, and an occasion that yields no answer at all is not a saving
    question. Trying is then better than not trying by dominance: the budget is
    bounded, and not answering is worth nothing.

    Without that distinction a family met for the first time is always refused,
    which is right when the answer already exists and dead wrong when it does
    not — and it was the second case that showed up, as a search that solved
    nothing in no time and looked like restraint.
    """
    actions = list(among if among is not None else the_actions_she_has())
    if asked_for is not None:
        named = [one for one in actions if one.name == asked_for]
        if named:
            _note("trigger", "asked", asked_for)
            return Decision(
                action=named[0],
                worth=_price(named[0], family, costs_now=costs_now),
                because="chosen",
                grounds=f"a caller named {asked_for}",
                started_by="asked",
            )
    if not actions:
        _note("refusal", "she", "there is nothing she could do")
        return Decision(
            action=None,
            worth=None,
            because="refused",
            grounds="nothing is registered",
        )

    _note("trigger", "she", f"{family} cost {costs_now:,}")
    priced = [(one, _price(one, family, costs_now=costs_now)) for one in actions]
    ranked = sorted(
        priced,
        key=lambda row: (
            row[1].worth is None,
            -(row[1].worth or 0),
        ),
    )
    considered = tuple((one.name, worth) for one, worth in ranked)
    # What developing may spend at all: the family's own total, times the share
    # development has earned against answering. Both are read; neither is set.
    ceiling = int(
        what_it_may_spend(family, costs_now=costs_now)
        * max(0.05, how_much_to_spend_on_developing())
    )
    # The bar. Zero with nothing waiting; the price of an answer when there is,
    # because the cost of developing then includes the answer nobody got.
    bar = the_bar_right_now(a_question_is_waiting=a_question_is_waiting)

    worths = [row[1].worth for row in ranked if row[1].worth is not None]
    spread = (
        statistics.pstdev(worths) if len(worths) > 1 else 0.0
    )
    best = next(
        (
            row
            for row in ranked
            if row[1].worth is not None
            # The largest of several noisy numbers is larger than the truth by
            # an amount that grows with how many were compared. Believing it
            # unshrunk is how a promotion policy convinces itself everything
            # it picked worked.
            and after_the_winners_curse(
                float(row[1].worth), len(worths) - 1, spread=spread
            )
            > bar
        ),
        None,
    )
    if best is not None and best[1].cost <= ceiling:
        _note("diagnosis", "she", f"{family} is the cost, {costs_now:,} a time")
        _note("proposal", "she", best[0].name)
        return Decision(
            action=best[0],
            worth=best[1],
            because="chosen",
            grounds=f"worth {best[1].describes()} against a ceiling of {ceiling:,}",
            considered=considered,
        )

    # What an action with no history is worth is not knowable, so what is
    # weighed is the information. A change cannot save more than the whole
    # search costs, so the most it could be worth is every occasion of that;
    # how likely it is to work at all is how often changes have worked, which
    # is Laplace over the record and is one half when there is no record.
    #
    # An earlier version compared the price of finding out against what the
    # slowest route spends, and both of those fall back to the occasion in
    # hand, so the test was a number against itself and passed always. A rule
    # that cannot refuse is not deciding, whatever it prints.
    paid = how_often_a_change_has_paid()
    most_it_could_save = float(what_it_may_spend(family, costs_now=costs_now))
    # Which unpriced one to try, drawn from what the counts support rather than
    # taken in the order they were written. An action that failed once early is
    # otherwise never tried again and its estimate never improves.
    unpriced = [one for one, worth in ranked if worth.worth is None]
    drawn = which_to_try(
        unpriced,
        pays=lambda one: (
            what_it_has_done(one.name).kept,
            what_it_has_done(one.name).taken,
        ),
        gains=lambda one: max(1.0, most_it_could_save),
    )
    if drawn is not None:
        ranked = [row for row in ranked if row[0] is drawn] + [
            row for row in ranked if row[0] is not drawn
        ]
    for one, worth in ranked:
        if worth.worth is not None:
            continue
        price = the_price_of_finding_out(worth.cost)
        worth_a_try = this_one_is_lost or paid * most_it_could_save > price
        if worth_a_try and price <= ceiling:
            _note("diagnosis", "she", f"{one.name} has no history")
            _note("proposal", "she", f"{one.name}, to find out what it saves")
            return Decision(
                action=one,
                worth=worth,
                because="exploring",
                grounds=(
                    "unpriced, and this one is lost without it"
                    if this_one_is_lost
                    else (
                        f"unpriced; at best it saves {most_it_could_save:,.0f} "
                        f"and {paid:.2f} of changes have paid, against "
                        f"{price:,} to find out"
                    )
                ),
                considered=considered,
            )

    _note("refusal", "she", f"nothing clears {ceiling:,}")
    return Decision(
        action=None,
        worth=None,
        because="refused",
        grounds=(
            f"nothing is worth doing: best priced "
            f"{ranked[0][1].describes()}, ceiling {ceiling:,}, and at best a "
            f"change saves {most_it_could_save:,.0f} with {paid:.2f} of them "
            "paying"
        ),
        considered=considered,
    )


def she_decides_to_develop(
    family: str,
    *,
    costs_now: int,
    situation: Any = None,
    asked_for: str | None = None,
) -> tuple[Decision, Any]:
    """Choose, do it, and write down what happened. The whole episode."""
    decided = what_to_do_next(
        family, costs_now=costs_now, asked_for=asked_for
    )
    if decided.action is None:
        note_an_episode(family, route=None, walked=0, admitted=None)
        return decided, None
    # Said before it happens, so the comparison afterwards is against something
    # fixed rather than against a memory of what she would have said.
    expected = what_she_expects(decided.action.name, costs_now=costs_now)
    _note("prediction", decided.started_by, expected.describes())
    _ALREADY_DECIDING[0] = True
    try:
        came_of_it = decided.action.do_it(situation)
    except Exception as exc:  # noqa: BLE001 - a failed action is a result
        logger.info("%s raised: %s", decided.action.name, exc)
        came_of_it = None
    finally:
        _ALREADY_DECIDING[0] = False
    _note(
        "evaluation",
        decided.started_by,
        f"{decided.action.name} gave {came_of_it!r}",
    )
    if decided.action.succeeded is not None:
        try:
            came_of_it = came_of_it if decided.action.succeeded(came_of_it) else None
        except Exception:  # noqa: BLE001 - a success test that raises fails it
            came_of_it = None
    if came_of_it:
        _note("installation", decided.started_by, decided.action.kind)
        promote(
            f"{decided.action.over}/{decided.action.name}",
            became="shadow",
            started_by=decided.started_by,
            evidence=decided.grounds,
            asked_from_outside=asked_for,
        )
    note_what_it_did(
        decided.action.name,
        kept=bool(came_of_it),
        gained=int(decided.worth.saving or 0) if decided.worth else 0,
    )
    what_actually_happened(
        decided.action.name,
        cost=decided.worth.cost if decided.worth else costs_now,
        kept=bool(came_of_it),
        gained=int(decided.worth.saving or 0) if decided.worth else 0,
    )
    note_an_episode(
        family,
        route=decided.action.name if came_of_it else None,
        walked=decided.worth.cost if decided.worth else 0,
        admitted=decided.action.kind if came_of_it else None,
    )
    return decided, came_of_it


def why_each_one(decided: Decision) -> list[str]:
    """The whole ranking as lines, for a record that has to be readable."""
    return [
        f"{name}: {worth.describes()}" for name, worth in decided.considered
    ]


def what_is_worth_doing_now() -> Decision:
    """Nobody asked anything. Is there something worth doing about herself?

    The other entry point, and the one that makes the difference between a
    system that improves when pushed and a system that improves. Nothing here
    is handed a family: the family is the one the record says costs the most in
    total, because total cost is what a change to it would be recovering.

    That choice is the diagnosis question — which part of her is limiting —
    answered by reading rather than by asking. A record with nothing in it
    yields no family, and she says so instead of picking one.
    """
    from core.cognition.what_she_notices_about_herself import what_she_notices

    kept = the_record().kept
    if not kept:
        _note("refusal", "she", "there is nothing to read")
        return Decision(
            action=None,
            worth=None,
            because="refused",
            grounds="the record is empty, so there is nothing to be worth doing",
        )
    spent: dict[str, list[int]] = {}
    for one in kept:
        spent.setdefault(one.family, []).append(one.walked)

    # What the readings turned up, strongest first. The target is whichever
    # opportunity has the most evidence behind it and is about something she
    # has actually met; a reading about a part of her rather than a family
    # still names the family that part is costing.
    #
    # Falling back to the costliest family is what happens when no reading has
    # anything to say, and that is a different situation from having nothing
    # to read at all.
    noticed = [one for one in what_she_notices() if one.about in spent]
    if noticed:
        family = noticed[0].about
        _note(
            "diagnosis",
            "she",
            f"{family} costs {sum(spent[family]):,} in all"
            f" ({noticed[0].describes()})",
        )
    else:
        family = max(spent, key=lambda one: sum(spent[one]))
        _note("diagnosis", "she", f"{family} costs {sum(spent[family]):,} in all")
    each = int(round(sum(spent[family]) / len(spent[family])))
    return what_to_do_next(
        family,
        costs_now=each,
        among=the_actions_she_has(with_a_case=False),
    )


def she_develops_herself() -> tuple[Decision, Any]:
    """Nobody asked. Choose, do it, and write down what came of it.

    One at a time. A developmental action can cause development — retracting a
    part leaves families to rebuild, and rebuilding is a developmental
    objective — so without a guard the policy is asked again from inside a
    decision it just made, and that recurses.

    The idle loop's whole episode, and the recording is the part that matters:
    an action that gives nothing has to cost something in the record, or the
    ranking picks it again next time and the loop is a loop rather than a
    development. A failure here is evidence, and the next choice is made with
    it.
    """
    if _ALREADY_DECIDING[0]:
        return (
            Decision(
                action=None,
                worth=None,
                because="refused",
                grounds="an episode is already running; one at a time",
            ),
            None,
        )
    decided = what_is_worth_doing_now()
    if decided.action is None:
        return decided, None
    family = next(
        (one.about.split(" costs")[0] for one in reversed(_TRACE)
         if one.what == "diagnosis" and " costs" in one.about),
        "herself",
    )
    each_costs = decided.worth.cost if decided.worth else 0
    expected = what_she_expects(decided.action.name, costs_now=each_costs)
    _note("prediction", "she", expected.describes())
    _ALREADY_DECIDING[0] = True
    try:
        came_of_it = decided.action.do_it(None)
    except Exception as exc:  # noqa: BLE001 - a failed action is a result
        logger.info("%s raised: %s", decided.action.name, exc)
        came_of_it = None
    finally:
        _ALREADY_DECIDING[0] = False
    _note(
        "evaluation", "she", f"{decided.action.name} gave {came_of_it!r}"
    )
    if came_of_it:
        _note("installation", "she", decided.action.kind)
        promote(
            f"{decided.action.over}/{decided.action.name}",
            became="shadow",
            started_by="she",
            evidence=decided.grounds,
        )
    note_what_it_did(
        decided.action.name,
        kept=bool(came_of_it),
        gained=int(decided.worth.saving or 0) if decided.worth else 0,
    )
    what_actually_happened(
        decided.action.name,
        cost=decided.worth.cost if decided.worth else each_costs,
        kept=bool(came_of_it),
        gained=int(decided.worth.saving or 0) if decided.worth else 0,
    )
    note_an_episode(
        family,
        route=decided.action.name if came_of_it else None,
        walked=decided.worth.cost if decided.worth else 0,
        admitted=decided.action.kind if came_of_it else None,
    )
    return decided, came_of_it
