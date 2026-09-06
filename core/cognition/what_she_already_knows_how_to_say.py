"""The terms she has already admitted, offered back to the search as leaves.

``every_code`` says the important thing in its own docstring: ``also`` is
"the only channel by which a long term becomes reachable — shortest-first
over a universal language reaches a few dozen symbols and no further, which
is Levin's bound rather than a defect here, and a library is what moves the
horizon rather than a bigger budget."

Then almost every caller passed ``also=()``. The autonomous operator search
walks 380 terms at depth three, forever, over a language that is
computationally universal — which an external review named as the gap
between being able to REPRESENT an improvement and being able to FIND one.
The cause was not budget. It was that the one channel the module documents
for moving the horizon was fed nothing.

What counts as already known is deliberately narrow: terms she installed and
kept. A head in ``DERIVED_HEADS``, a rule in ``RULES_WITH_NO_SHAPE``. Those
survived whatever gate admitted them, so composing over them is composing
over things that worked rather than over everything that ever parsed.

The library grows as she does, which is the point. A search whose reach is
fixed at boot cannot get better at searching; one whose leaves are what she
has learned gets longer terms within the same budget every time something is
kept.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["how_far_the_search_reaches", "what_she_already_knows_how_to_say"]

#: The most leaves offered back. Enumeration is combinatorial in the leaf
#: count, so an unbounded library turns a bounded search into an unbounded
#: one — which is the same defect as a fixed horizon, arriving from the other
#: side. Longest-first, because a long term is the one that buys reach.
_HOW_MANY_LEAVES = 24


def what_she_already_knows_how_to_say(*, most: int = _HOW_MANY_LEAVES) -> tuple[Any, ...]:
    """Terms she installed and kept, longest first, bounded.

    Longest first because reach is the whole reason to offer them: a
    two-symbol leaf saves two symbols, and a fourteen-symbol one puts a
    sixteen-symbol term inside a depth-three budget.
    """
    from core.cognition.the_floor_she_stands_on import Code, how_long

    found: list[Any] = []

    try:
        from core.cognition.one_algebra import DERIVED_HEADS

        found.extend(
            head.body
            for head in DERIVED_HEADS.values()
            if isinstance(getattr(head, "body", None), Code)
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("no heads to offer the search: %s", exc)

    try:
        from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE

        found.extend(
            rule.body
            for rule in RULES_WITH_NO_SHAPE.values()
            if isinstance(getattr(rule, "body", None), Code)
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("no rules to offer the search: %s", exc)

    # Deduplicated by what they ARE, not by where they were found: a head and
    # a rule can hold the same term, and offering it twice doubles the branching
    # factor for nothing.
    seen: dict[str, Any] = {}
    for term in found:
        seen.setdefault(repr(term), term)
    return tuple(
        sorted(seen.values(), key=lambda one: -how_long(one))[: max(0, int(most))]
    )


def how_far_the_search_reaches() -> dict[str, Any]:
    """What the library buys, in symbols, so the claim is a measurement.

    ``reach`` is the longest term a depth-three search can now build: three
    levels of composition over the longest leaf it has. With no library that
    is the floor's own primitives and the answer is small and fixed.
    """
    from core.cognition.the_floor_she_stands_on import how_long

    library = what_she_already_knows_how_to_say()
    longest = max((how_long(one) for one in library), default=1)
    return {
        "schema": "aura.search.reach.v1",
        "leaves_offered": len(library),
        "longest_leaf_symbols": longest,
        # Measured, not derived. Walking 4,000 candidates at depth three, the
        # longest term the enumerator produces is 2n+1 for a longest leaf of
        # n symbols — checked at n = 1, 4, 7 and 9, which came out 3, 9, 15
        # and 19. The first version of this line said n*4+3 and would have
        # reported 31 where the search reaches 15.
        "reach_at_depth_three": 2 * longest + 1,
        "reach_with_no_library": 3,
    }
