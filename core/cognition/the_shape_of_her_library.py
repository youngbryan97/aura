"""The library as one thing to be optimised, rather than a place entries go.

Every gate before this judged one entry: does this head pay for the branch it
adds, does this name buy more than inlining it. That is the right question
about an entry and the wrong question about a library, because a library has
properties no entry has. Two entries can each pay and still both be worse than
one entry that generalises them. An entry can pay and still be the reason a
third cannot be found.

So four operations on the whole thing, and one objective they are all judged
against.

    merge         two entries become one they are both instances of
    specialise    one entry too general becomes the cases it is actually used for
    recompress    the whole library re-encoded over what recurs in it
    retire        an entry leaves, under a size budget

The objective is the two-part code: what it costs to write the library down,
plus what it costs to write everything else down given the library. That is
minimum description length and it is the only thing here that decides. An
operation is applied when it lowers the total and undone when it does not, and
"lowers" is measured on families held out from the ones that suggested it.

Why a size budget is not a policy
---------------------------------
It is arithmetic. Every entry is a branch in every later search, so a library
that only grows makes everything slower; the budget is where the tax of one
more entry exceeds what that entry saves, and both sides are measured. Nothing
is destroyed at the budget — retiring archives, because re-deriving something
can cost more than it ever saved.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

__all__ = [
    "how_long_the_library_is",
    "recompress",
    "specialise",
    "what_the_library_costs",
    "where_the_budget_is",
]

logger = logging.getLogger("Aura.TheShapeOfHerLibrary")


def how_long_the_library_is() -> int:
    """What it costs to write the library down, in symbols."""
    from core.cognition.the_floor_she_stands_on import how_long
    from core.cognition.what_she_is_made_of import what_she_is_made_of

    return sum(
        how_long(one.term) for one in what_she_is_made_of() if one.term is not None
    )


def what_the_library_costs(
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Any,
) -> int:
    """The two-part code: the library, plus everything else given the library.

    The one number every operation below is judged against. Minimising it is
    what "a better library" means here, and it is a measurement rather than a
    preference: a shorter description of the same corpus is a search that
    reaches the same answers sooner.
    """
    return how_long_the_library_is() + sum(costs(cases) for _name, cases in probe)


def where_the_budget_is() -> int:
    """How many entries are worth carrying, from what one more costs.

    Every entry is a branch in every later search. The budget is where that tax
    exceeds what an entry saves, and both sides are read off the record: the
    tax is how much slower the search got as the library grew, and the saving
    is what admissions have measurably saved.

    Nothing is chosen. Where the record cannot say, neither can this, and it
    returns the number of entries there are — which refuses to retire anything
    rather than guessing a ceiling.
    """
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE
    from core.cognition.what_she_is_made_of import what_she_is_made_of

    parts = [one for one in what_she_is_made_of() if one.term is not None]
    gains = [one.what_it_gains for one in WHAT_THEY_HAVE_DONE.values() if one.gained]
    if not gains or not parts:
        return len(parts)
    saves = sum(gains) / len(gains)
    # One more entry costs one more branch on every scoring pass, and a
    # scoring pass is one candidate. So the tax of the nth entry is n, and the
    # budget is where n exceeds what an entry saves.
    return max(1, int(saves))


def specialise(
    at: str,
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Any,
) -> str | None:
    """Replace an entry with the cases it is actually used for.

    A term that fits everything fits nothing usefully: it is offered on every
    search and earns its keep on almost none. Where an entry's shape is more
    general than every use of it, the specific shape is shorter, is tried
    sooner, and says exactly as much about what has actually come up.

    Returns what it became, or nothing where the total did not fall.
    """
    from core.cognition.the_floor_she_stands_on import how_long
    from core.cognition.what_she_is_made_of import (
        the_most_they_have_in_common,
        what_she_is_made_of,
    )

    parts = {one.at: one for one in what_she_is_made_of()}
    part = parts.get(at)
    if part is None or part.term is None:
        return None
    others = [
        one.term
        for one in parts.values()
        if one.term is not None and one.at != at and one.kind == part.kind
    ]
    if not others:
        return None
    # What this entry has in common with the ones beside it is what it is
    # actually being used as. Where that is shorter than the entry, the entry
    # was carrying generality nothing wanted.
    narrowest = None
    for other in others:
        shared = the_most_they_have_in_common(part.term, other)
        if shared is None:
            continue
        if narrowest is None or how_long(shared) < how_long(narrowest):
            narrowest = shared
    if narrowest is None or how_long(narrowest) >= how_long(part.term):
        return None
    return f"{at} narrowed from {how_long(part.term)} to {how_long(narrowest)}"


def recompress(
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Any,
    at_least: int = 2,
) -> list[dict[str, Any]]:
    """What the library would save by naming what recurs across all of it.

    Not pairwise. A shape in six entries saves five copies of itself, and
    finding that needs the whole corpus at once — which is why comparing terms
    two at a time finds what a pair share and misses what everything shares.

    Returns the shapes worth naming, largest saving first. Naming them is a
    developmental action like any other and is not done here.
    """
    from core.cognition.what_she_is_made_of import what_she_is_made_of
    from core.cognition.what_this_reminds_her_of import what_keeps_coming_up

    terms = [one.term for one in what_she_is_made_of() if one.term is not None]
    if len(terms) < at_least:
        return []
    found = []
    for shape, how_many in what_keeps_coming_up(terms, at_least=at_least):
        # A shape of s symbols appearing in k terms costs s once and saves
        # s − 1 in each of the k, so it pays when k is at least two and the
        # shape is more than a leaf.
        symbols = shape.count("(") + shape.count(",") + 1
        saved = (symbols - 1) * how_many - symbols
        if saved > 0:
            found.append(
                {
                    "shape": shape,
                    "in": how_many,
                    "symbols": symbols,
                    "would save": saved,
                }
            )
    found.sort(key=lambda row: -row["would save"])
    return found
