"""Looking several moves ahead, as far as there is time to look.

Once she can try a move without making it and say which result is better, the
two compose: try a move, then try every move from there, and keep going while
it is worth the arithmetic. What comes back is not a move but an ordering over
the moves available, with the reason the best one is best.

How deep is not a setting. Each level costs branching times the last, and the
cost of one level is measured rather than assumed, so the depth is whatever
fits the time this decision is worth. That is the whole of metareasoning at
this scale: work out what thinking costs, work out what is available, spend
the second on the first.

A world that adds something of its own after every act — a dealt tile, another
person, a page that refreshes — makes a deep future less trustworthy than a
shallow one, whatever the arithmetic says. So what a later level is worth is
discounted, and the discount is not a preference: it is the share of her own
predictions that have been holding.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from core.agency.how_good_is_this import how_good, why

__all__ = [
    "as_far_as_the_world_lets_her",
    "how_deep_to_look",
    "look_ahead",
    "worth_finding_out",
]

logger = logging.getLogger("Aura.LookingAhead")

#: How far ahead the arithmetic still describes something that could happen
#: is not a number anybody picks — it is how much of its own the world will
#: have added by then, against the room the thing has left. See
#: :func:`as_far_as_the_world_lets_her`, which measures it.

#: What one level of looking is assumed to cost before any has been measured.
#: Deliberately generous: being slow to look deep costs a little, and looking
#: deeper than there is time for costs the move.
_UNMEASURED_LEVEL_S = 0.02

#: What a level of looking has actually cost, measured.
_A_LEVEL: dict[str, float] = {"seconds": 0.0}


class _AlreadyWorkedOut:
    """What this search has already worked out, so it is not worked out twice.

    A search over a thing reaches the same situation by many routes — four
    moves and the world's replies fan out and fold back on each other
    constantly — and every arrival used to pay again for the same three
    things: what an act makes of a situation, what the situation is worth,
    and what the best line from it comes to.

    Kept for one search and thrown away with it, because all three answers
    are about the rule and the measure as they stand at this moment, and both
    can change between one move and the next.
    """

    __slots__ = ("becomes", "worth", "onward", "hits")

    def __init__(self) -> None:
        self.becomes: dict[tuple[str, str], Any] = {}
        self.worth: dict[str, float] = {}
        self.onward: dict[tuple[str, int], float] = {}
        self.hits = 0

    def what_it_becomes(self, expect: Any, state: Any, action: str) -> Any:
        key = (_reading(state), action)
        if key in self.becomes:
            self.hits += 1
            return self.becomes[key]
        made = expect(state, action)
        self.becomes[key] = made
        return made

    def what_it_is_worth(self, state: Any, **how: Any) -> float:
        key = _reading(state)
        if key in self.worth:
            self.hits += 1
            return self.worth[key]
        said = how_good(state, **how)
        self.worth[key] = said
        return said


def as_far_as_the_world_lets_her(state: Any, world: Any) -> int:
    """How many acts ahead the arithmetic still describes something that could
    happen.

    The world puts things of its own into a thing between her acts. Every
    level of looking is one more act, so it is one more of the world's
    additions — and once those exceed the room the thing has left, the future
    being scored is one that cannot exist, whatever the arithmetic says.

    This used to be a fixed four, with that reasoning written beside it. Four
    is right on a nearly full board and wrong on an empty one, which is
    exactly where looking deeper pays: a board with fourteen free places and a
    world that deals one thing a move has fourteen acts of honest arithmetic
    in it, and she was stopping at four.

    Nought means no limit, which is the honest answer for a world that adds
    nothing of its own and for one she has not watched.
    """
    often = getattr(world, "how_often", None)
    if not callable(often):
        return 0
    try:
        rate = float(often() or 0.0)
    except (TypeError, ValueError):
        return 0
    if rate <= 0.0:
        return 0
    try:
        room = int(state.places()) - int(state.occupied())
    except (AttributeError, TypeError, ValueError):
        return 0
    return max(1, int(room / rate))


def how_deep_to_look(
    available: int, budget_s: float, branching: int = 4, no_further_than: int = 0
) -> int:
    """How far ahead there is time to look, from what a level has cost.

    Each level multiplies the work by the branching. The answer is the deepest
    level whose cost still fits, and one is always affordable — a single level
    is the difference between choosing blind and choosing at all.

    ``no_further_than`` is where the arithmetic stops describing anything that
    could happen, which is a fact about the world rather than about the clock.
    Nought is no limit.
    """
    if available < 1 or budget_s <= 0.0:
        return 1
    a_level = _A_LEVEL["seconds"] or _UNMEASURED_LEVEL_S
    depth = 1
    spent = a_level * branching
    while no_further_than <= 0 or depth < no_further_than:
        spent = spent * branching
        if spent > budget_s:
            break
        depth += 1
    return depth


def worth_finding_out(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    ahead: dict[str, tuple[float, str]] | None = None,
    never_tried: Sequence[str] = (),
) -> dict[str, float]:
    """What each act is worth for what it would TELL her, not where it leads.

    Looking ahead asks which move is best under the rule she is using. This
    asks a different question: which move would settle which rule is right.
    They are not the same move, and early on the second is worth far more —
    a rule she is sure of improves every move after this one, and a slightly
    better position improves only this one.

    Scaled by what knowing is worth here, which is read off the futures she
    can already see: where the best and worst moves available differ by very
    little, being right about the rule is worth very little, and where they
    differ by a lot it is worth a lot. So the number comes from her own
    situation rather than from a setting, and it goes to nought by itself as
    the rule settles — at which point she stops experimenting, because there
    is nothing left to find out.

    Where she cannot see ahead at all, this is what she has: the acts are
    scored purely by what they would settle, which is the right thing to do
    when she has no model to prefer anything by.

    ``never_tried`` is the acts she has not taken here yet, and it comes
    first, because everything above needs a state she can read and rules that
    disagree about it — and in a world she has just arrived in she has
    neither. What her own acts do is the first thing there is to find out,
    and it needs no model, no grid and no reading. It empties itself once
    they have all been taken.

    LIVE 2026-09-04: no grid, so no rule, so nothing disagreed about
    anything, so this returned nothing, so there was no reason to vary — and
    a grid is worked out from what moves, which under one act is wherever
    that act puts things. Two hundred seconds of the same key.
    """
    fresh = [str(one) for one in never_tried if str(one) in {str(a) for a in actions}]
    if fresh:
        # A tie between all of them on the first act, which is right: any of
        # them settles as much as any other when none has been taken.
        return {one: 1.0 for one in fresh}
    settle = getattr(knows, "what_this_would_settle", None)
    if not callable(settle) or state is None:
        return {}
    told: dict[str, float] = {}
    for action in actions:
        try:
            told[action] = max(0.0, float(settle(state, action)))
        except (AttributeError, TypeError, ValueError):
            continue
    if not told or not any(told.values()):
        return {}
    values = [value for value, _ in (ahead or {}).values()]
    # What being right is worth here. With nothing to see ahead, finding out
    # is the only thing on offer and is worth one whole move.
    spread = (max(values) - min(values)) if len(values) > 1 else 1.0
    if spread <= 0.0:
        return {}
    return {action: share * spread for action, share in told.items()}


def look_ahead(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    *,
    toward: str = "",
    approach: str = "",
    budget_s: float = 0.5,
    world: Any = None,
    weights: Any = None,
    depth: int = 0,
) -> dict[str, tuple[float, str]]:
    """Every move available, scored by where it leads and how sure that is.

    ``knows`` is anything that can say what a state would become — the rules
    she worked out by watching. When it cannot, this returns nothing, which is
    the honest answer and not a failure.

    ``world`` is what the world does on its own between her acts, if she has
    worked that out. Without it, the search takes the best continuation at
    every level — which quietly assumes the world will cooperate, and plans a
    future that cannot happen. With it, each level averages over what the
    world might do instead, which is the difference between a plan and a wish.
    """
    if not actions or state is None or knows is None:
        return {}
    expect = getattr(knows, "expect", None)
    if not callable(expect):
        return {}
    trust = float(getattr(knows, "confidence", lambda: 0.0)() or 0.0)
    if trust <= 0.0:
        return {}

    # NOT bound to what the superlative names right now. Measured, and it
    # costs half her play. See `bound_to` for the reasoning and the numbers
    # that refute it: a superlative in a line is meant to float.

    started = time.monotonic()
    fixed_depth = bool(depth)
    # What a level really costs is her acts times the world's replies to each
    # of them. Counting only her own acts understated it by the whole of the
    # world's fan-out, so the depth the clock allowed was many times what the
    # clock could afford.
    depth = depth or how_deep_to_look(
        len(actions),
        budget_s,
        branching=max(2, len(actions)) * max(1, _how_many_ways(world, state)),
        no_further_than=as_far_as_the_world_lets_her(state, world),
    )
    known = _AlreadyWorkedOut()
    here_now = _reading(state)

    def one_pass(how_far: int) -> dict[str, tuple[float, str]]:
        found: dict[str, tuple[float, str]] = {}
        for action in actions:
            future = known.what_it_becomes(expect, state, action)
            if future is None or _reading(future) == here_now:
                # A move that would change nothing has not gone anywhere.
                #
                # Scored like any other, it collects the value of the
                # situation it left alone, once at every level of the search —
                # so standing still outscores every move that costs something
                # to make. Measured against a null on 2026-08-26: choosing
                # this way was WORSE than choosing at random, with 78% of
                # moves doing nothing.
                #
                # Ruling one out before making it is the whole point of being
                # able to try a move without making it.
                continue
            here = known.what_it_is_worth(
                future,
                toward=toward,
                approach=approach,
                weights=weights,
                knows=knows,
                acts=actions,
            )
            onward = _after_the_world(
                expect, future, actions, how_far - 1,
                toward=toward, approach=approach, trust=trust, world=world,
                weights=weights, known=known, knows=knows,
                been=frozenset({here_now}),
            )
            found[action] = (
                here + trust * onward,
                why(future, toward=toward, approach=approach),
            )
        return found

    # Deeper while the clock allows, rather than a guess at how deep it will
    # allow.
    #
    # What a level costs is a projection from an average, and it is wrong by
    # whatever the world's fan-out and what has already been worked out do to
    # it — the two together were a factor of tens. Going one level deeper and
    # stopping when the time is gone is measured by construction, and the
    # deeper pass is nearly free because the shallower one is still
    # remembered. What she hands back is always a completed pass.
    scored = one_pass(depth)
    if not fixed_depth and scored:
        ends_at = started + max(0.0, budget_s)
        a_pass = time.monotonic() - started
        ceiling = as_far_as_the_world_lets_her(state, world)
        while (ceiling <= 0 or depth < ceiling) and a_pass > 0.0:
            # Only a level there is time to FINISH. A pass abandoned halfway
            # is a pass that cost the budget and answered nothing.
            branching = max(2, len(actions)) * max(1, _how_many_ways(world, state))
            if time.monotonic() + a_pass * branching > ends_at:
                break
            deeper_at = time.monotonic()
            deeper = one_pass(depth + 1)
            if not deeper:
                break
            depth += 1
            scored = deeper
            a_pass = time.monotonic() - deeper_at

    spent = time.monotonic() - started
    if scored and depth and not fixed_depth:
        _a_level_took(spent / float(depth))
    logger.debug(
        "looked %d ahead over %d move(s) in %.3fs (%d already worked out)",
        depth, len(actions), spent, known.hits,
    )
    return scored


def _how_many_ways(world: Any, state: Any) -> int:
    """How many replies the world has to one act, as it stands here."""
    might = getattr(world, "might_do", None)
    if not callable(might):
        return 1
    try:
        return max(1, len(might(state)))
    except (AttributeError, TypeError, ValueError):
        return 1


def _after_the_world(
    expect: Any,
    state: Any,
    actions: Sequence[str],
    depth: int,
    *,
    toward: str,
    approach: str,
    trust: float,
    world: Any = None,
    weights: Any = None,
    known: Any = None,
    knows: Any = None,
    been: frozenset[str] = frozenset(),
) -> float:
    """What this comes to once the world has had its turn, and she has hers.

    Her own move is the best she can find. What the world does is not hers to
    pick, so it is averaged over rather than chosen — which is the whole
    difference between working out what will happen and hoping.
    """
    if depth <= 0:
        return 0.0
    ways = ()
    might = getattr(world, "might_do", None)
    if callable(might):
        try:
            ways = might(state)
        except (AttributeError, TypeError, ValueError):
            ways = ()
    if not ways:
        return _best_from(
            expect, state, actions, depth,
            toward=toward, approach=approach, trust=trust, world=world, weights=weights,
            known=known, knows=knows, been=been,
        )
    return sum(
        share
        * _best_from(
            expect, way, actions, depth,
            toward=toward, approach=approach, trust=trust, world=world, weights=weights,
            known=known, knows=knows, been=been,
        )
        for way, share in ways
    )


def _best_from(
    expect: Any,
    state: Any,
    actions: Sequence[str],
    depth: int,
    *,
    toward: str,
    approach: str,
    trust: float,
    world: Any = None,
    weights: Any = None,
    known: Any = None,
    knows: Any = None,
    been: frozenset[str] = frozenset(),
) -> float:
    """The best this could still come to, that many levels on.

    ``been`` is the line already walked to get here. A future already on it
    is not reached: going back somewhere is not progress, and a search that
    scores it as progress prefers pacing to arriving.

    That could not happen in the world this was written for, where every move
    is irreversible, so it was never exposed there. In a world where a move
    can be undone it is severe. Measured on a sealed world with a reading
    that rises towards the goal: she climbed the reading correctly to a
    ridge, and then stepped back and forth between the same two squares for
    the rest of the budget, because the line that stepped back could step
    forward again and collect the higher reading a second time. Two squares,
    eighty moves, a perfectly correct model of the world, and nought arrivals.
    """
    if depth <= 0:
        return 0.0
    here_now = _reading(state)
    if known is not None:
        # The same situation, the same distance from the end, is the same
        # answer — for the same line. A search folds back on itself constantly
        # and this is most of what it costs. The line is part of the key
        # because the value of a state depends on what is now behind her.
        remembered = known.onward.get((here_now, depth, been))
        if remembered is not None:
            known.hits += 1
            return remembered
    walked = been | {here_now}
    best = 0.0
    for action in actions:
        future = (
            known.what_it_becomes(expect, state, action)
            if known is not None
            else expect(state, action)
        )
        if future is None or _reading(future) == here_now:
            continue
        if _reading(future) in walked:
            continue
        here = (
            known.what_it_is_worth(
                future,
                toward=toward,
                approach=approach,
                weights=weights,
                knows=knows,
                acts=actions,
            )
            if known is not None
            else how_good(
                future,
                toward=toward,
                approach=approach,
                weights=weights,
                knows=knows,
                acts=actions,
            )
        )
        onward = _after_the_world(
            expect, future, actions, depth - 1,
            toward=toward, approach=approach, trust=trust, world=world, weights=weights,
            known=known, knows=knows, been=walked,
        )
        best = max(best, here + trust * onward)
    if known is not None:
        known.onward[(here_now, depth, been)] = best
    return best


def _a_level_took(seconds: float) -> None:
    """Record what a level of looking cost, so the next depth is chosen from it."""
    spent = float(seconds or 0.0)
    if spent <= 0.0:
        return
    before = _A_LEVEL["seconds"]
    _A_LEVEL["seconds"] = spent if before <= 0.0 else before * 0.7 + spent * 0.3


def _reading(state: Any) -> str:
    """A state as a thing that can be compared to another state."""
    as_text = getattr(state, "as_text", None)
    return as_text() if callable(as_text) else str(state)
