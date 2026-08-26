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

__all__ = ["DEEPEST", "look_ahead", "how_deep_to_look"]

logger = logging.getLogger("Aura.LookingAhead")

#: No deeper than this however much time there is. Past it the world has
#: added more of its own than she predicted, and the arithmetic is describing
#: a future that will not happen.
DEEPEST = 4

#: What one level of looking is assumed to cost before any has been measured.
#: Deliberately generous: being slow to look deep costs a little, and looking
#: deeper than there is time for costs the move.
_UNMEASURED_LEVEL_S = 0.02

#: What a level of looking has actually cost, measured.
_A_LEVEL: dict[str, float] = {"seconds": 0.0}


def how_deep_to_look(available: int, budget_s: float, branching: int = 4) -> int:
    """How far ahead there is time to look, from what a level has cost.

    Each level multiplies the work by the branching. The answer is the deepest
    level whose cost still fits, and one is always affordable — a single level
    is the difference between choosing blind and choosing at all.
    """
    if available < 1 or budget_s <= 0.0:
        return 1
    a_level = _A_LEVEL["seconds"] or _UNMEASURED_LEVEL_S
    depth = 1
    spent = a_level * branching
    while depth < DEEPEST:
        spent = spent * branching
        if spent > budget_s:
            break
        depth += 1
    return depth


def look_ahead(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    *,
    toward: str = "",
    approach: str = "",
    budget_s: float = 0.5,
) -> dict[str, tuple[float, str]]:
    """Every move available, scored by where it leads and how sure that is.

    ``knows`` is anything that can say what a state would become — the rules
    she worked out by watching. When it cannot, this returns nothing, which is
    the honest answer and not a failure.
    """
    if not actions or state is None or knows is None:
        return {}
    expect = getattr(knows, "expect", None)
    if not callable(expect):
        return {}
    trust = float(getattr(knows, "confidence", lambda: 0.0)() or 0.0)
    if trust <= 0.0:
        return {}

    started = time.monotonic()
    depth = how_deep_to_look(len(actions), budget_s, branching=max(2, len(actions)))
    scored: dict[str, tuple[float, str]] = {}
    here_now = _reading(state)
    for action in actions:
        future = expect(state, action)
        if future is None or _reading(future) == here_now:
            # A move that would change nothing has not gone anywhere.
            #
            # Scored like any other, it collects the value of the situation it
            # left alone, once at every level of the search — so standing
            # still outscores every move that costs something to make.
            # Measured against a null on 2026-08-26: choosing this way was
            # WORSE than choosing at random, with 78% of moves doing nothing.
            #
            # Ruling one out before making it is the whole point of being able
            # to try a move without making it.
            continue
        here = how_good(future, toward=toward, approach=approach)
        onward = _best_from(
            expect, future, actions, depth - 1, toward=toward, approach=approach, trust=trust
        )
        scored[action] = (here + trust * onward, why(future, toward=toward, approach=approach))

    spent = time.monotonic() - started
    if scored and depth:
        _a_level_took(spent / float(depth))
    logger.debug("looked %d ahead over %d move(s) in %.3fs", depth, len(actions), spent)
    return scored


def _best_from(
    expect: Any,
    state: Any,
    actions: Sequence[str],
    depth: int,
    *,
    toward: str,
    approach: str,
    trust: float,
) -> float:
    """The best this could still come to, that many levels on."""
    if depth <= 0:
        return 0.0
    best = 0.0
    here_now = _reading(state)
    for action in actions:
        future = expect(state, action)
        if future is None or _reading(future) == here_now:
            continue
        here = how_good(future, toward=toward, approach=approach)
        onward = _best_from(
            expect, future, actions, depth - 1, toward=toward, approach=approach, trust=trust
        )
        best = max(best, here + trust * onward)
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
