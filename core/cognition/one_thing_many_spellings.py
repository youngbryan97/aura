"""One concept, however many ways there are of writing it.

Once she can invent freely she reinvents the same thing in different clothes.
Measured after a single invention: thirty words in the language, twenty-five
distinct behaviours, five duplicates — and the plainest of them was

    if how many there are left over 2 then [here] else [here]

which is "here", spelled at six times the length. Every duplicate is another
branch at every step of every search, bought for nothing.

So the vocabulary is not a list of names. It is a set of behaviours, each with
the ways of writing it that she happens to know, and the one she uses is the
cheapest of those — cheapest on the ruler she cannot move, not the shortest
name, because names are exactly what a maker can make cheap.

That also separates a concept from its implementation. "How far apart two
things are" is one concept whether she reaches it by subtracting and taking
the size, or by a way of building she wrote last week. When a shorter way of
saying it turns up, the concept does not become a different concept.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "also_compare_at",
    "how_it_behaves",
    "one_of_each",
    "sizes_words_are_told_apart_at",
    "the_other_spellings",
]

logger = logging.getLogger("Aura.OneThingManySpellings")

#: Sizes a word is compared at when nothing has asked about any others. Two
#: words agreeing on every place of every one of these agree everywhere she has
#: looked, which is the strongest thing available and is not the same as
#: agreeing everywhere.
_AT_SIZES = (3, 4, 5)

#: And the sizes something has actually asked about, which are added to those.
#:
#: Deciding identity on three fixed sizes threw away the one word that answered
#: a family of size six: it agreed with another word at three, four and five,
#: so it was a duplicate by this measure and a duplicate is dropped. The answer
#: left the language before the search could reach it, and the search then ran
#: its whole budget looking for something no longer there.
#:
#: Sizes come from the world here as every other quantity does.
_SIZES_ASKED_ABOUT: set[int] = set()


def also_compare_at(sizes: Any) -> None:
    """Remember a size something has been asked about, for telling words apart."""
    for size in sizes or ():
        try:
            found = int(size)
        except (TypeError, ValueError):
            continue
        if found > 1:
            _SIZES_ASKED_ABOUT.add(found)


def sizes_words_are_told_apart_at() -> tuple[int, ...]:
    return tuple(sorted({*_AT_SIZES, *_SIZES_ASKED_ABOUT}))

#: The spellings she knows for each behaviour, beyond the one in use.
_ALSO_WRITTEN: dict[tuple[int, ...], tuple[str, ...]] = {}


def how_it_behaves(word: Any) -> tuple[int, ...] | None:
    """What a word does, as one comparable value, or None where it refuses."""
    try:
        return tuple(
            int(word(at, size)) % size
            for size in sizes_words_are_told_apart_at()
            for at in range(size)
        )
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
        return None


def one_of_each(words: dict[str, Any]) -> dict[str, Any]:
    """The vocabulary with each behaviour kept once, by its cheapest spelling.

    A word that refuses to say what it does at these sizes is kept as it is:
    it cannot be compared, and dropping something because it declined to be
    measured is worse than carrying it.
    """
    from core.cognition.the_ruler_she_cannot_move import what_it_costs_to_be

    best: dict[tuple[int, ...], tuple[int, str]] = {}
    kept: dict[str, Any] = {}
    others: dict[tuple[int, ...], list[str]] = {}
    for name, word in words.items():
        does = how_it_behaves(word)
        if does is None:
            kept[name] = word
            continue
        costs = what_it_costs_to_be(word, name)
        others.setdefault(does, []).append(name)
        standing = best.get(does)
        if standing is None or costs < standing[0]:
            best[does] = (costs, name)
    for does, (_costs, name) in best.items():
        kept[name] = words[name]
        rest = tuple(one for one in others.get(does, ()) if one != name)
        if rest:
            _ALSO_WRITTEN[does] = rest
    if len(kept) < len(words):
        logger.info(
            "one of each: %d word(s) say what %d already said",
            len(words) - len(kept),
            len(kept),
        )
    return kept


def the_other_spellings(word: Any) -> tuple[str, ...]:
    """The other ways she knows of writing this, if any.

    A concept keeps its alternatives. The one in use is the cheapest known,
    and finding a cheaper one later changes the spelling and not the concept.
    """
    does = how_it_behaves(word)
    if does is None:
        return ()
    return _ALSO_WRITTEN.get(does, ())
