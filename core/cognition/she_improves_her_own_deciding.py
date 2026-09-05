"""Two things she can do about the way she works, offered like any other action.

Everything the previous mandate built made the machinery of her search into
objects: the order words are tried in is a term, the proposer is a term, and
now what a change is worth is a term. Objects are not initiative. Something has
to want to change them, and that something has to be the same ranking that
decides everything else or the wanting was ours.

So these are two entries in `what_she_could_do_next`, priced and ranked beside
the eight that widen a language. Neither is reachable by failing, which is why
neither could exist under a ladder: an order that finds answers sooner is worth
having when the answers are already being found, and a better way of deciding
what to change is worth having whether or not anything is currently wrong.

**An order that finds them sooner.** Search the floor for a scoring rule that
ranks the words that have actually won earlier than the rule in force does,
judged on her own history in `how_she_learns_to_look`. Nothing is handed a
target and there is no list of scoring rules to pick from.

**A way of deciding that would have chosen better.** Search the floor for a
worth rule and replay the record under it: for each family, what would this
rule have picked, and what did that cost when it was picked? Lower is better.
This one is the recursion the question is about — it changes the rule that
chooses what to change, it is itself chosen by that rule, and the trace says
who started it.

What stops it from being circular
---------------------------------
The replay is against measurements that are already in the record. A worth rule
cannot make itself look good by changing what gets recorded, because what gets
recorded is what the search spent, and the search does not consult it. The gate
stays outside the space for the reason `a_gate_inside_the_space_cannot_hold`
already executes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.cognition.the_floor_she_stands_on import (
    Code,
    OutOfFuel,
    Stuck,
    every_code,
)
from core.cognition.what_counts_as_better import how_bad_that_is

__all__ = [
    "an_order_that_finds_them_sooner",
    "a_worth_that_would_have_chosen_better",
    "how_soon_they_are_found",
    "offer_what_she_can_do_about_herself",
    "what_the_record_would_have_cost",
]

logger = logging.getLogger("Aura.SheImprovesHerOwnDeciding")


def _closed(candidate: Code, binders: int) -> Code:
    made: Code = candidate
    for _ in range(binders):
        made = Code("given a thing", parts=(made,))
    return made


def how_soon_they_are_found(order: Code) -> float:
    """Where the word that won sits under this order, over the occasions she has lived.

    Mean rank of the winner, lower being sooner. This is the one quantity an
    order exists to move, and it is measured on her own history rather than on
    a stand-in for it: `how_she_learns_to_look` keeps what each ranking was
    computed from, so a past occasion can be ranked again under a rule that did
    not exist when it happened.

    Infinity where she has not lived any, which keeps a search from setting off
    with nothing to be judged against.
    """
    from core.cognition.how_she_learns_to_look import (
        how_often_it_worked,
        how_the_last_ones_looked,
    )
    from core.cognition.the_floor_she_stands_on import run

    lived = how_the_last_ones_looked()
    if not lived:
        return float("inf")
    ranks: list[int] = []
    for one in lived:
        scored: list[tuple[int, int, str]] = []
        for name, (told, symbols) in one["features"].items():
            before = how_often_it_worked(name)
            try:
                work: Any = run(order, fuel=20_000)
                for each in (told, one["places"], before.won, before.of, symbols):
                    work = run(work.body, (each, *work.env), fuel=20_000)
                said = int(work)
            except (OutOfFuel, Stuck, TypeError, ValueError, AttributeError):
                said = 0
            scored.append((-said, symbols, name))
        scored.sort()
        where = [at for at, (_s, _y, name) in enumerate(scored) if name == one["winner"]]
        sat = where[0] if where else len(scored)
        # What counts as bad about that is a term, not this function. The
        # default computes the rank, which is what this computed before; what
        # changed is that an experiment she designs is now the same kind of
        # thing as a word she invents.
        ranks.append(
            how_bad_that_is(sat=sat, of=len(scored), symbols=len(one["features"]))
        )
    return sum(ranks) / len(ranks)


def an_order_that_finds_them_sooner(
    *, deepest: int = 3, within: float = 20.0
) -> Code | None:
    """Search the floor for an order that ranks her winners earlier.

    Shortest first, over the five numbers an order is given. Returns nothing
    when nothing beats the rule in force, which is the honest answer and the
    one that keeps a search from installing noise.
    """
    from core.cognition.the_order_she_tries_them_in import THE_ORDER

    began = time.monotonic()
    best = how_soon_they_are_found(THE_ORDER)
    if best == float("inf"):
        return None
    found: Code | None = None
    for candidate in every_code(
        deepest=deepest, variables=5, constants=(0, 1, 2), also=()
    ):
        if time.monotonic() - began >= within:
            break
        closed = _closed(candidate, 5)
        try:
            got = how_soon_they_are_found(closed)
        except (OutOfFuel, Stuck, ArithmeticError, TypeError, ValueError):
            continue
        if got < best:
            best, found = got, closed
    return found


def what_the_record_would_have_cost(worth: Code) -> float:
    """Replay her own record under a different way of valuing, and total the bill.

    For every family in the record, ask what this rule would have chosen, and
    charge what that choice actually cost when it was made. A choice the record
    never made is charged what the family cost with nothing chosen, because
    that is what not knowing costs.

    The measurements are already there and a rule still could flatter itself,
    which this claim used to deny. Asking what to do next writes a stage into
    the decision trace and an entry into the ledger of what each action has
    done, and the pricing reads that ledger; two identical replays of the same
    rule over the same record returned 45,408 and then 42,537, the second
    cheaper for the first having happened. And choosing among actions with no
    history is a draw, so one replay is one sample of a quantity whose spread
    is about a tenth of itself.

    Both are handled here now: the replay runs inside a scope that restores
    everything it touches, and the score is the mean over seeded replays.
    """
    from core.cognition.the_record_of_her_own_work import the_record
    from core.cognition.what_it_is_worth_doing import (
        forget_the_worth,
        the_worth_she_uses,
        the_worth_she_wrote,
    )
    from core.cognition.she_decides_to_develop import what_to_do_next
    from core.cognition.what_she_could_do_next import the_actions_she_has

    from core.cognition.does_improving_compound import (
        HOW_MANY_REPLAYS,
        a_replay_that_changes_nothing,
    )

    kept = the_record().kept
    if not kept or not the_actions_she_has():
        return float("inf")
    cost_of: dict[tuple[str, str | None], list[int]] = {}
    for one in kept:
        cost_of.setdefault((one.family, one.route), []).append(one.walked)
    families = sorted({one.family for one in kept})
    was = the_worth_she_uses()
    got: list[float] = []
    try:
        the_worth_she_wrote(worth)
        for seed in range(HOW_MANY_REPLAYS):
            total = 0.0
            with a_replay_that_changes_nothing(seed=seed):
                for family in families:
                    spent = cost_of.get((family, None)) or [
                        one.walked for one in kept if one.family == family
                    ]
                    now = sum(spent) / len(spent)
                    decided = what_to_do_next(family, costs_now=int(now))
                    picked = decided.action.name if decided.action else None
                    seen = cost_of.get((family, picked))
                    total += (sum(seen) / len(seen)) if seen else now
            got.append(total)
    except (OutOfFuel, Stuck, ArithmeticError, TypeError, ValueError):
        return float("inf")
    finally:
        the_worth_she_wrote(was)
    return sum(got) / len(got) if got else float("inf")


def a_worth_that_would_have_chosen_better(
    *, deepest: int = 3, within: float = 20.0
) -> Code | None:
    """Search the floor for a way of valuing that the record says was better.

    The recursion, stated plainly: this changes the rule that decides what to
    change, and this search is itself one of the things that rule chooses
    between.
    """
    from core.cognition.what_it_is_worth_doing import THE_WORTH

    began = time.monotonic()
    best = what_the_record_would_have_cost(THE_WORTH)
    if best == float("inf"):
        return None
    found: Code | None = None
    for candidate in every_code(
        deepest=deepest, variables=4, constants=(0, 1, 2), also=()
    ):
        if time.monotonic() - began >= within:
            break
        closed = _closed(candidate, 4)
        try:
            got = what_the_record_would_have_cost(closed)
        except (OutOfFuel, Stuck, ArithmeticError, TypeError, ValueError):
            continue
        if got < best:
            best, found = got, closed
    return found


def offer_what_she_can_do_about_herself(*, within: float = 20.0) -> None:
    """Put both in the registry, so the ranking can choose them.

    Called from the same place the eight are registered. Neither is special and
    neither is tried first; both are priced off the record like everything
    else.
    """
    from core.cognition.the_order_she_tries_them_in import the_order_she_wrote
    from core.cognition.what_it_is_worth_doing import (
        the_worth_she_uses,
        the_worth_she_wrote,
    )
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )

    def a_sooner_order(situation: Any = None) -> str | None:
        found = an_order_that_finds_them_sooner(within=within)
        if found is None:
            return None
        the_order_she_wrote(found)
        return "an order that finds them sooner"

    def a_better_way_of_deciding(situation: Any = None) -> str | None:
        found = a_worth_that_would_have_chosen_better(within=within)
        if found is None:
            return None
        the_worth_she_wrote(found)
        return "a way of deciding that the record says was better"

    if "an order that finds them sooner" not in WHAT_SHE_COULD_DO:
        what_she_could_do(
            "an order that finds them sooner",
            over="the order she tries them in",
            kind="an order",
            do_it=a_sooner_order,
            needs_a_case=False,
        )
    def look_deeper(situation: Any = None) -> str | None:
        """Offer candidates one level down, where the shallow ones ran out.

        Depth two is one head over two leaves and it is where most short
        answers are. Past it the authored enumerator took over, and a proposer
        that hands off at a fixed boundary has a fixed boundary. This is the
        next shape — a head over a head and a leaf — and installing it is a
        decision rather than a default, because every candidate it offers costs
        more to build and check than a shallow one.
        """
        from core.cognition.the_proposer_she_can_replace import (
            THE_DEEPER_PROPOSER,
            the_proposer_in_use,
            the_proposer_she_wrote,
        )

        if the_proposer_in_use() is THE_DEEPER_PROPOSER:
            return None
        the_proposer_she_wrote(THE_DEEPER_PROPOSER)
        return "offered candidates one level deeper"

    if "look one level deeper" not in WHAT_SHE_COULD_DO:
        what_she_could_do(
            "look one level deeper",
            over="the proposer",
            kind="a deeper proposer",
            do_it=look_deeper,
            needs_a_case=False,
        )
    def judge_searches_differently(situation: Any = None) -> str | None:
        """Aim the search for a better order at something else.

        Every search here is a search against an objective, and while the
        objective was a Python function the space of orders she could look for
        was whatever that function could see. Mean rank of the winner is a good
        objective and not the only one: a rule putting the winner second every
        time beats one putting it first half the time and last half the time on
        the mean, and which is actually better depends on what happens next.

        Judged the way everything else is: does the record cost less afterwards.
        """
        from core.cognition.what_counts_as_better import (
            THE_OBJECTIVE,
            forget_the_objective,
            the_objective_she_wrote,
        )

        best = what_the_record_would_have_cost(the_worth_she_uses())
        if best == float("inf"):
            return None
        began = time.monotonic()
        for candidate in every_code(
            deepest=2, variables=3, constants=(0, 1, 2), also=()
        ):
            if time.monotonic() - began >= within:
                break
            closed = _closed(candidate, 3)
            try:
                the_objective_she_wrote(closed)
                got = what_the_record_would_have_cost(the_worth_she_uses())
            except (OutOfFuel, Stuck, ArithmeticError, TypeError, ValueError):
                forget_the_objective()
                continue
            if got < best:
                return "judged searches by something else"
            forget_the_objective()
        return None

    if "judge searches differently" not in WHAT_SHE_COULD_DO:
        what_she_could_do(
            "judge searches differently",
            over="what a change is worth",
            kind="an objective",
            do_it=judge_searches_differently,
            needs_a_case=False,
        )
    if "a way of deciding what to change" not in WHAT_SHE_COULD_DO:
        what_she_could_do(
            "a way of deciding what to change",
            over="what a change is worth",
            kind="a way of deciding",
            do_it=a_better_way_of_deciding,
            needs_a_case=False,
        )
