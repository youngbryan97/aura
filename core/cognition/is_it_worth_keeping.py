"""Whether a way of building words earns its place, measured on what it was not built from.

A maker that makes the family in front of her sayable has proved one thing:
that it fits the evidence it was made from. That is the weakest possible test,
and it is the one a lookup table passes.

What it has to earn is a place in the language she thinks in from now on, and
that is a trade. Against it:

    every word it makes is another branch at every step of every search

For it:

    families that were unreachable become reachable
    thoughts that were long become short, and a short thought is a findable one
    both of those happen for families it was never shown

The usual way to weigh those is a sum with a weight on each term, and the
weights are where the argument goes. There are none here. Every term is
counted in the same unit — expressions she would otherwise have to walk — so
the trade settles itself:

    worth = (what it makes reachable) + (what it shortens) - (what it adds)

and a weight of one on each is not a choice, it is what having one unit means.

Nothing it was built from counts. A maker is always worth everything on the
family it was made for, and measuring it there measures the fitting rather
than the maker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from core.cognition.what_it_costs_to_say import how_many_expressions

__all__ = ["WhatItIsWorth", "what_it_is_worth"]

logger = logging.getLogger("Aura.IsItWorthKeeping")


@dataclass(frozen=True)
class WhatItIsWorth:
    """What a way of building buys and what it costs, in the one unit."""

    #: Families it makes sayable that it was not built from, and were not
    #: sayable before. Each is a whole search she would otherwise walk and
    #: fail in.
    reaches: int = 0
    #: Expressions saved because things can be said more briefly.
    shortens: int = 0
    #: Expressions added by having one more word at every position.
    costs: int = 0
    #: How many held-out families were tried, so a zero can be read.
    tried: int = 0

    @property
    def worth(self) -> int:
        return self.reaches + self.shortens - self.costs

    @property
    def keep_it(self) -> bool:
        return self.worth > 0

    def describes(self) -> str:
        verdict = "earns its place" if self.keep_it else "costs more than it buys"
        return (
            f"{verdict}: reaches {self.reaches:,} and shortens {self.shortens:,} "
            f"against {self.costs:,} added, over {self.tried} family(ies) it was "
            "not built from"
        )


def what_it_is_worth(
    *,
    now_sayable: Callable[[Sequence[Any]], bool],
    held_out: Sequence[Sequence[Any]],
    was_sayable: Sequence[bool] = (),
    vocabulary_before: int,
    vocabulary_after: int,
    longest: int,
    shorter_by: int = 0,
    used: int = 0,
) -> WhatItIsWorth:
    """Weigh a maker on families it was never shown.

    ``now_sayable`` answers whether a family is sayable with the maker in
    place; ``was_sayable`` says whether each was sayable before it. A family
    that was already sayable buys nothing, however well it goes now.
    """
    before = list(was_sayable) + [False] * max(0, len(held_out) - len(was_sayable))
    walked = how_many_expressions(max(1, vocabulary_after), max(1, longest))
    newly = 0
    for family, already in zip(held_out, before):
        if already:
            continue
        try:
            if now_sayable(family):
                newly += 1
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
            continue

    # A family made reachable is worth the search she would otherwise walk to
    # the end of and fail in. That is the size of the space, once per family.
    reaches = newly * walked
    # What brevity buys, in the same unit: the layers of the search she no
    # longer has to walk because the thing is shorter to say.
    saved = max(0, int(shorter_by)) * max(0, int(used))
    shortens = walked - how_many_expressions(
        max(1, vocabulary_after), max(0, int(longest) - saved)
    )
    # And what it costs: one more word at every position, at every length.
    costs = walked - how_many_expressions(max(1, vocabulary_before), max(1, longest))

    found = WhatItIsWorth(
        reaches=int(reaches),
        shortens=int(shortens),
        costs=int(max(0, costs)),
        tried=len(held_out),
    )
    logger.info("what that way of building is worth — %s", found.describes())
    return found
