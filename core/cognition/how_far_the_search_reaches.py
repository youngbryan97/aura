"""How much of the language the operator search actually walks.

The floor is computationally universal and the proposer for new operators is
enumeration: shortest terms first, to depth three, filtered by a probe, capped
at four thousand examined and sixty-four offered. An external review put the
problem as having a universal programming language and searching only tiny
programs, and that is right — but "tiny" was a word rather than a number, and
a search that reports no denominator implies it looked at what mattered.

So this counts both sides. The number of terms the floor admits at each size
is exact and comes from the same recurrence the generator walks, rather than
from generating them: five one-place heads, seven two-place ones, and a
three-place ``if``, over however many leaves there are. Against that goes what
a run actually examined, what computed a number, and what the kernel accepted.

Library leaves change the representable search space, not necessarily the
probability of finding a correct answer within a budget. Actual coverage must
come from the iterator; a configured limit is not a measurement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.HowFarTheSearchReaches")

__all__ = [
    "AReach",
    "how_many_at",
    "how_many_up_to",
    "what_the_search_reached",
    "what_the_library_buys",
    "how_far_it_reaches",
]

#: The generator's heads, by how many parts each takes. Read from the floor so
#: that adding a head changes the count rather than making it wrong.
ONE_PLACE = 5
THREE_PLACE = 1


def _two_place() -> int:
    from core.cognition.the_floor_she_stands_on import ARITHMETIC

    return 2 + len(ARITHMETIC)


def how_many_at(size: int, *, leaves: int, twos: int | None = None) -> int:
    """How many terms of exactly this size the floor admits.

    The same shape the generator builds: a one-place head over a term one
    smaller, a two-place head over an ordered split, and ``if`` over an
    ordered triple. Counted rather than generated, because the point of a
    denominator is to be available when generating it is what you cannot
    afford.
    """
    if size < 1:
        return 0
    twos = _two_place() if twos is None else twos
    counted = {1: max(0, int(leaves))}
    for at in range(2, int(size) + 1):
        total = ONE_PLACE * counted.get(at - 1, 0)
        for left in range(1, at - 1):
            total += twos * counted.get(left, 0) * counted.get(at - 1 - left, 0)
        for test in range(1, at - 2):
            for then in range(1, at - 1 - test):
                rest = at - 1 - test - then
                total += (
                    THREE_PLACE
                    * counted.get(test, 0)
                    * counted.get(then, 0)
                    * counted.get(rest, 0)
                )
        counted[at] = total
    return counted.get(int(size), 0)


def how_many_up_to(deepest: int, *, leaves: int, twos: int | None = None) -> int:
    """Every term the floor admits up to that size."""
    return sum(
        how_many_at(size, leaves=leaves, twos=twos)
        for size in range(1, max(1, int(deepest)) + 1)
    )


@dataclass(frozen=True, slots=True)
class AReach:
    """What one search walked, against what there was."""

    deepest: int
    leaves: int
    #: Every term the floor admits to this depth over these leaves.
    there_were: int
    #: The most the run would pull off the generator. A ceiling, not a count.
    would_examine: int
    #: How many of those computed a number on the probes.
    computed: int
    #: How many were offered to the kernel.
    offered: int
    #: None means no observed walk was supplied.
    examined: int | None = None

    def __post_init__(self) -> None:
        if min(self.would_examine, self.computed, self.offered) < 0:
            raise ValueError("search counters must be nonnegative")
        if self.examined is not None:
            if not 0 <= self.offered <= self.computed <= self.examined <= min(
                self.would_examine, self.there_were
            ):
                raise ValueError("observed search counters are inconsistent")

    @property
    def share_examined(self) -> float | None:
        if self.examined is None:
            return None
        return (self.examined / self.there_were) if self.there_were else 0.0

    @property
    def exhausted(self) -> bool | None:
        """Whether every term was actually visited, not merely affordable."""
        return None if self.examined is None else self.examined == self.there_were

    def to_dict(self) -> dict[str, Any]:
        return {
            "deepest": self.deepest,
            "leaves": self.leaves,
            "there_were": self.there_were,
            "would_examine": self.would_examine,
            "examined": self.examined,
            "computed": self.computed,
            "offered": self.offered,
            "share_examined": (
                None if self.share_examined is None else round(self.share_examined, 6)
            ),
            "exhausted": self.exhausted,
            "one_in": (
                round(self.there_were / self.examined) if self.examined else None
            ),
        }


def what_the_search_reached(
    *,
    deepest: int = 3,
    leaves: int,
    would_examine: int,
    computed: int = 0,
    offered: int = 0,
    examined: int | None = None,
) -> AReach:
    """One run's reach, with the denominator beside it."""
    return AReach(
        deepest=int(deepest),
        leaves=int(leaves),
        there_were=how_many_up_to(deepest, leaves=leaves),
        would_examine=int(would_examine),
        computed=int(computed),
        offered=int(offered),
        examined=examined,
    )


def what_the_library_buys(
    *, deepest: int = 3, floor_leaves: int, from_her_library: int
) -> dict[str, Any]:
    """How much her own terms move the horizon, against a deeper budget.

    Two ways to reach further: more depth, or better leaves. The comparison is
    the point — a library entry of fourteen symbols puts a sixteen-symbol term
    inside a depth-three budget, and no amount of depth-three enumeration over
    the bare floor gets there.
    """
    bare = how_many_up_to(deepest, leaves=floor_leaves)
    withit = how_many_up_to(deepest, leaves=floor_leaves + max(0, int(from_her_library)))
    deeper = how_many_up_to(deepest + 1, leaves=floor_leaves)
    return {
        "deepest": int(deepest),
        "floor_leaves": int(floor_leaves),
        "from_her_library": int(from_her_library),
        "terms_over_the_floor": bare,
        "terms_with_her_library": withit,
        "times_larger": (withit / bare) if bare else 0.0,
        "terms_at_one_more_depth": deeper,
        "a_library_entry_is_worth": (
            "more than a depth" if withit > deeper else "less than a depth"
        ),
    }


def how_far_it_reaches(
    *, deepest: int = 3, leaves: int, would_examine: int, from_her_library: int = 0,
    examined: int | None = None, computed: int = 0, offered: int = 0,
) -> dict[str, Any]:
    """For the health report: the search's reach and what widens it."""
    reached = what_the_search_reached(
        deepest=deepest, leaves=leaves, would_examine=would_examine,
        examined=examined, computed=computed, offered=offered,
    )
    return {
        "schema": "aura.operator_search.reach.v2",
        "reach": reached.to_dict(),
        "library": what_the_library_buys(
            deepest=deepest,
            floor_leaves=max(1, leaves - from_her_library),
            from_her_library=from_her_library,
        ),
    }
