"""Kinds of rule whose meaning she worked out, not kinds somebody wrote a branch for.

A learned rule is a node with a kind, and the interpreter knows three kinds:
apply these in turn, read the positions, read the cells. Anything else returns
None. So she can compose programs out of the meanings she was given, and the
set of meanings never grows — a node of an unknown kind has no semantics, and
acquiring one has always meant a person editing the interpreter.

This is the registry that makes the set grow. A kind admitted here carries its
own executable meaning, and the interpreter consults it exactly as it consults
the three it was born with. Adding one is not an edit to the interpreter.

Where the meaning comes from
----------------------------
Not from a name. A meaning here is a point in a small algebra over finite
states: which two places the value at each position is read from, and what is
done with the pair. Reversal, rotation, shifting, taking the larger of a pair and
combining a pair all fall out of the same two choices, and none of them is in
the space by name — the same discipline the rule space and the measure space
already follow.

What will not be admitted
-------------------------
Anything that only explains the examples it was induced from. A meaning is
admitted on transitions it was NOT built from, and refused otherwise, because
a rule fitted to everything has been tested against nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "ENOUGH_HELD_BACK",
    "Induced",
    "KINDS",
    "admit",
    "every_meaning",
    "forget",
    "induce_from",
    "interpretation_of",
]

logger = logging.getLogger("Aura.AnInventedKind")

#: How many transitions a meaning must get right that it was not induced from.
#: One is a coincidence when states are short.
ENOUGH_HELD_BACK = 2


# ── where the value at a position comes from ─────────────────────────────

WHERE_FROM: dict[str, Callable[[int, int], int]] = {
    "here": lambda index, size: index,
    "the far end": lambda index, size: size - 1 - index,
    "one along": lambda index, size: (index + 1) % size,
    "one back": lambda index, size: (index - 1) % size,
    "its partner": lambda index, size: index + 1 if index % 2 == 0 else index - 1,
}

#: A second place, chosen the same way. What makes the pair itself something
#: she works out rather than something fixed: pairing each place with the one
#: after it, with its partner, or with its mirror are different meanings, and
#: which one holds is a question for the examples.


# ── what is done with what is found there ────────────────────────────────


def _as_it_is(one: Any, _other: Any) -> Any:
    return one


def _the_larger(one: Any, other: Any) -> Any:
    try:
        return one if float(one) >= float(other) else other
    except (TypeError, ValueError):
        return one


def _the_smaller(one: Any, other: Any) -> Any:
    try:
        return one if float(one) <= float(other) else other
    except (TypeError, ValueError):
        return one


def _both_together(one: Any, other: Any) -> Any:
    try:
        return type(one)(float(one) + float(other)) if isinstance(one, (int, float)) else one
    except (TypeError, ValueError):
        return one


WHAT_OF_IT: dict[str, Callable[[Any, Any], Any]] = {
    "as it is": _as_it_is,
    "the larger of it and its neighbour": _the_larger,
    "the smaller of it and its neighbour": _the_smaller,
    "both together": _both_together,
}


@dataclass(frozen=True)
class Induced:
    """One meaning: where each value comes from, and what is done with it.

    Executable, and expressible in the same breath, so what she admitted can
    be said out loud as well as run.
    """

    where_from: str
    and_from: str
    what_of_it: str
    #: How it did on transitions it was not induced from. Nought means it was
    #: never held to any, which is not the same as failing them.
    held_back: float = 0.0
    from_examples: int = 0

    @property
    def name(self) -> str:
        if self.what_of_it == "as it is":
            return f"take {self.where_from}"
        return f"take {self.where_from} and {self.and_from}, {self.what_of_it}"

    def read(self, cells: Sequence[Any]) -> tuple[Any, ...] | None:
        """The state this meaning turns ``cells`` into, or None where it cannot."""
        found = tuple(cells)
        size = len(found)
        if size == 0:
            return ()
        try:
            where = WHERE_FROM[self.where_from]
            other = WHERE_FROM[self.and_from]
            what = WHAT_OF_IT[self.what_of_it]
        except KeyError:
            return None
        out: list[Any] = []
        for index in range(size):
            try:
                one = found[where(index, size) % size]
                two = found[other(index, size) % size]
                out.append(what(one, two))
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                return None
        return tuple(out)

    def describe(self) -> str:
        held = (
            f", right about {self.held_back:.0%} of what it was not built from"
            if self.from_examples
            else ""
        )
        return f"{self.name}{held}"


def every_meaning() -> Iterator[Induced]:
    """The whole space, so nothing in it had to be thought of in advance."""
    for where_from, and_from, what_of_it in product(WHERE_FROM, WHERE_FROM, WHAT_OF_IT):
        if what_of_it == "as it is" and and_from != where_from:
            # Reading a second place and ignoring it is the same meaning said
            # a different way, and every duplicate is another chance for a
            # coincidence to win the search.
            continue
        yield Induced(where_from=where_from, and_from=and_from, what_of_it=what_of_it)


#: Kinds she has worked out the meaning of, by name. Empty at import: nothing
#: is here that a person put here.
KINDS: dict[str, Induced] = {}


def induce_from(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> Induced | None:
    """Work out a meaning that accounts for these before-and-after pairs.

    Half to solve on, half to be judged on, because a meaning fitted to every
    example it has seen has been tested against nothing. Returns nothing when
    no meaning in the space survives the half it never saw — which is the
    honest answer and how "the language cannot say this" stays sayable.
    """
    pairs = [(tuple(before), tuple(after)) for before, after in transitions]
    if len(pairs) < ENOUGH_HELD_BACK + 1:
        return None
    solving = pairs[0::2]
    judging = pairs[1::2]
    if len(judging) < ENOUGH_HELD_BACK:
        return None
    for meaning in every_meaning():
        if not all(meaning.read(before) == after for before, after in solving):
            continue
        right = sum(1 for before, after in judging if meaning.read(before) == after)
        if right < len(judging):
            continue
        found = Induced(
            where_from=meaning.where_from,
            and_from=meaning.and_from,
            what_of_it=meaning.what_of_it,
            held_back=right / len(judging),
            from_examples=len(solving),
        )
        logger.info(
            "a meaning nobody wrote accounts for this: %s", found.describe()
        )
        return found
    return None


def admit(kind: str, meaning: Induced) -> str:
    """Give a kind of node a meaning, so the interpreter can run it.

    No edit to the interpreter. That is the whole point: the set of things a
    node can mean grows because she worked one out, not because somebody added
    a branch for it.
    """
    name = str(kind or "").strip()
    if not name or not isinstance(meaning, Induced):
        return ""
    if not meaning.from_examples:
        # A meaning that was never held to anything it did not come from is a
        # guess with an executable body, and running it would be worse than
        # saying she cannot.
        return ""
    KINDS[name] = meaning
    logger.info("she gave %r a meaning: %s", name, meaning.describe())
    return name


def forget(kind: str) -> bool:
    """Take a meaning back out. What was admitted on evidence can lose it."""
    return KINDS.pop(str(kind), None) is not None


def interpretation_of(kind: str) -> Callable[[Sequence[Any]], tuple[Any, ...] | None] | None:
    """How to run a node of that kind, or nothing if she has no meaning for it."""
    meaning = KINDS.get(str(kind or ""))
    return meaning.read if meaning is not None else None
