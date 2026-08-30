"""Why a language that only grows eventually stops being an improvement.

A word is not free. It is another branch at every step of every search, so a
language that admits every word that ever helped once becomes a place where
nothing can be found. Two things follow, and both are arithmetic rather than
taste.

A word pays for itself when it removes more search than it adds. Both sides are
countable in the same unit. Standing in for a structure of length m, it takes d
symbols out of the expressions that use it, and the space of expressions up to
length L over b words has about b**L members, so it removes

    N(b, L) - N(b, L - d)

and, by being one more word to try at every position, it adds

    N(b + 1, L) - N(b, L)

Keep it when the first exceeds the second. Putting both in expressions-to-walk
is what makes the trade a measurement instead of a weighting nobody can defend.

And growth cannot go on regardless. With B bits of persistent state there are
2**B distinct states, so a strictly growing language on fixed hardware runs out.
What replaces endless growth is a language that merges words meaning the same
thing, drops words that stopped paying, and can end a year smaller and better —
the quantity being maximised is capability per unit of cost, never the number
of words.

Capability itself has to name its terms. It is what she can reach on a set of
tasks within a budget, and it is measured by running them, because reachable
and expressible stop agreeing the moment a search is bounded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from core.cognition.what_it_costs_to_say import how_many_expressions

__all__ = [
    "HowCapable",
    "WhatAWordIsWorth",
    "how_capable",
    "how_long_growth_can_last",
    "what_a_word_is_worth",
    "what_to_merge",
    "worth_per_cost",
]

logger = logging.getLogger("Aura.KeepingTheLanguageSmall")


@dataclass(frozen=True)
class WhatAWordIsWorth:
    """What admitting a word removes from the search, and what it adds."""

    name: str
    removes: int
    adds: int

    @property
    def pays(self) -> bool:
        return self.removes > self.adds

    def describes(self) -> str:
        verdict = "pays" if self.pays else "costs more than it saves"
        return (
            f"{self.name!r} {verdict}: removes {self.removes:,} expression(s) from "
            f"the search and adds {self.adds:,}"
        )


def what_a_word_is_worth(
    name: str, *, vocabulary: int, longest: int, shorter_by: int, used: int = 1
) -> WhatAWordIsWorth:
    """Whether a word earns the room it takes up.

    Both sides counted in expressions she would otherwise have to walk, so
    there is no weighting to argue about. A word used once that saves one
    symbol almost never pays; one that saves six symbols in a search over
    twenty words pays by a factor in the millions.
    """
    words, longest_at = max(1, int(vocabulary)), max(1, int(longest))
    # A word nothing uses saves nothing, however much it would save per use.
    # Flooring the count at one is how a language fills with words that were
    # each a good idea once and are now only branches.
    saved = max(0, int(shorter_by)) * max(0, int(used))
    removes = how_many_expressions(words, longest_at) - how_many_expressions(
        words, max(0, longest_at - saved)
    )
    adds = how_many_expressions(words + 1, longest_at) - how_many_expressions(
        words, longest_at
    )
    return WhatAWordIsWorth(name=str(name), removes=int(removes), adds=int(adds))


def what_to_merge(sayable: dict[Any, tuple[str, int]]) -> dict[str, str]:
    """Words that mean the same thing, mapped to the shortest way of saying it.

    Two words with one meaning are one word and a synonym, and the synonym
    costs a branch at every step for nothing. Merging is how the language gets
    smaller without losing anything, which is the only way it can keep growing
    at the edges.
    """
    keep: dict[str, str] = {}
    for _, (name, _length) in sayable.items():
        keep.setdefault(name, name)
    return keep


def how_long_growth_can_last(bits: int) -> int:
    """How many strictly different languages fit in a fixed amount of memory.

    Pigeonhole, and it applies to her: state that never repeats cannot outrun
    the state there is room for. Past this the only way forward is compressing,
    merging and forgetting.
    """
    room = max(0, int(bits))
    return 1 << room if room < 64 else (1 << 64)


@dataclass(frozen=True)
class HowCapable:
    """What she reached on a named set of tasks inside a named budget."""

    reached: int
    tried: int
    within: int
    #: How many hypotheses were actually examined, summed over the tasks.
    examined: int

    @property
    def share(self) -> float:
        return self.reached / self.tried if self.tried else 0.0

    def describes(self) -> str:
        return (
            f"{self.reached}/{self.tried} reached, examining at most {self.within} "
            f"hypotheses each ({self.examined} examined in total)"
        )


def how_capable(
    tasks: Sequence[Any],
    solve: Callable[[Any, int], bool],
    *,
    within: int,
) -> HowCapable:
    """G(A; D, B): what she reaches on these tasks under this budget.

    The budget is the argument that makes the number mean anything. Without
    one, a language that expresses everything and finds nothing scores the same
    as one that finds things, and the difference between those two is the whole
    of what a new word is for.
    """
    reached = 0
    examined = 0
    for task in tasks:
        try:
            if solve(task, int(within)):
                reached += 1
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        examined += int(within)
    return HowCapable(
        reached=reached, tried=len(tasks), within=int(within), examined=examined
    )


def worth_per_cost(capability: float, cost: float) -> float:
    """Capability per unit of language, which is the thing to maximise.

    A year that ends with fewer words and more solved is a good year, and no
    measure counting words can say so.
    """
    return float(capability) / float(cost) if cost else 0.0
