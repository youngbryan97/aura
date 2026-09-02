"""Finding a stored part by its shape, so transfer needs no label.

Reuse in this codebase has always worked by making every library entry a leaf
of every later search: a shared term is found because it is short, not because
anything noticed it was shared. That is cheaper than recognition and it works,
and it has one thing it cannot do — say WHY it helped, or decline in advance
when it will not.

This is the other half. A fingerprint is what a term looks like with the names
and the numbers taken out: the sequence of heads and the shape of the tree.
Two terms over completely different vocabularies fingerprint the same when they
compute the same shape of thing, which is exactly what "different surface, same
structure" means and is why no label is needed.

    what_this_reminds_her_of   parts ranked by how much shape they share
    what_keeps_coming_up       parts of terms that recur across the corpus
    did_it_help                whether the reminder was worth taking

The third is the one that keeps this honest. A relevance score is a prediction,
and a prediction that is never checked is a story. What is stored is the
outcome — this part helped that family, or it did not — so a reminder that
keeps not helping stops being offered.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "WHAT_HELPED_WHERE",
    "a_fingerprint",
    "did_it_help",
    "how_much_they_share",
    "what_keeps_coming_up",
    "what_this_reminds_her_of",
]

logger = logging.getLogger("Aura.WhatThisRemindsHerOf")


def _shape(term: Any) -> str:
    """The term with its names and numbers taken out.

    A leaf is a leaf whatever it holds. Two terms over vocabularies with
    nothing in common give the same shape when they compute the same shape of
    thing, and that is the whole mechanism: the surface is what was dropped.
    """
    from core.cognition.the_floor_she_stands_on import Code

    if not isinstance(term, Code):
        return "?"
    if not term.parts:
        return "." if term.head in {"a number", "the one it was given"} else term.head
    return term.head + "(" + ",".join(_shape(one) for one in term.parts) + ")"


def a_fingerprint(term: Any) -> str:
    """A short, stable name for a term's shape."""
    return hashlib.sha256(_shape(term).encode("utf-8")).hexdigest()[:16]


def _pieces(term: Any, deepest: int = 4) -> list[str]:
    """Every subterm's shape, down to a depth. What two terms are compared on."""
    from core.cognition.the_floor_she_stands_on import Code

    if not isinstance(term, Code) or deepest <= 0:
        return []
    found = [_shape(term)]
    for one in term.parts:
        found.extend(_pieces(one, deepest - 1))
    return found


def how_much_they_share(first: Any, second: Any) -> float:
    """Between nothing and one: how much of their shape is common.

    Over subterm shapes rather than whole ones, so a part appearing inside a
    bigger term still registers. Names and numbers are already gone, so this
    says nothing about what either term is over.
    """
    here, there = Counter(_pieces(first)), Counter(_pieces(second))
    if not here or not there:
        return 0.0
    common = sum((here & there).values())
    whole = max(sum(here.values()), sum(there.values()))
    return common / whole if whole else 0.0


@dataclass
class WhatHelped:
    """What a part did when it was offered, so a bad reminder stops being offered."""

    offered: int = 0
    helped: int = 0

    @property
    def worth_offering(self) -> float:
        return (self.helped + 1) / (self.offered + 2)


WHAT_HELPED_WHERE: dict[tuple[str, str], WhatHelped] = {}


def what_this_reminds_her_of(
    term: Any, *, at_least: float = 0.0
) -> list[dict[str, Any]]:
    """Parts that share shape with this, likeliest to help first.

    The ranking is shared shape times how often that part has helped when
    offered before. Nothing about either half knows what any term means.
    """
    from core.cognition.what_she_is_made_of import what_she_is_made_of

    found = []
    for part in what_she_is_made_of():
        if part.term is None:
            continue
        share = how_much_they_share(term, part.term)
        if share <= at_least:
            continue
        before = WHAT_HELPED_WHERE.get((part.at, a_fingerprint(term)))
        odds = before.worth_offering if before else 0.5
        found.append(
            {
                "at": part.at,
                "shares": round(share, 3),
                "has helped": round(odds, 3),
                "worth offering": round(share * odds, 4),
            }
        )
    found.sort(key=lambda row: -row["worth offering"])
    return found


def did_it_help(at: str, term: Any, *, helped: bool) -> WhatHelped:
    """Write down whether a reminder was worth taking.

    The half that makes the relevance score evidence rather than a story. A
    part that keeps being offered and keeps not helping falls to the bottom of
    the ranking without anybody deciding it should.
    """
    key = (str(at), a_fingerprint(term))
    held = WHAT_HELPED_WHERE.setdefault(key, WhatHelped())
    held.offered += 1
    if helped:
        held.helped += 1
    return held


def what_keeps_coming_up(
    terms: Iterable[Any], *, at_least: int = 2, biggest_first: bool = True
) -> list[tuple[str, int]]:
    """Shapes that recur across the whole corpus, not between one pair.

    Comparing terms two at a time finds what a pair share; this finds what
    everything shares, which is the thing worth naming. A shape appearing in
    six terms saves five copies of itself, and a shape appearing in two saves
    one.
    """
    counted: Counter = Counter()
    for term in terms:
        # Once per term: a shape occurring twice inside one term is not
        # evidence that it recurs, and counting it as such names local
        # repetition rather than shared structure.
        for shape in set(_pieces(term)):
            counted[shape] += 1
    found = [
        (shape, how_many)
        for shape, how_many in counted.items()
        if how_many >= at_least and shape not in {".", "?"}
    ]
    found.sort(key=lambda row: (-(len(row[0]) * row[1]) if biggest_first else -row[1]))
    return found
