"""When the language that GENERATES hypotheses is the thing that is missing.

She can compose a meaning out of a small algebra — which two places a value is
read from, and what is done with the pair — and admit it to the interpreter
without anybody adding a branch. That is the set of expressions growing:

    K(t+1) = K(t) + {a kind she induced}

The algebra those expressions are built from did not grow. Five ways to say
where a value comes from and four ways to combine a pair were written down by
a person, and every meaning she will ever induce is a point in their closure.
So a family outside it is not merely unsolved — it is unsayable, and no amount
of searching finds it, because the search is over the wrong set.

This is the other growth:

    A(t+1) = A(t) + {an operation she derived}

Deriving one, rather than choosing it
-------------------------------------
An addressing function answers "where did the value at position i come from?"
and that question is answerable FROM the examples: when the values in a state
are distinct, each output value is at exactly one place in the input, so the
correspondence can be read off rather than guessed. If no existing addressing
says what was read off, the correspondence IS a new addressing — defined by
what it does, which is the only definition anything here ever has.

A pair operation answers "what was done with the two values?" and is derived
the same way, from the cases where the addressing is already known.

What is honest about it
-----------------------
Read off a correspondence and you have a function on the lengths you saw. It is
admitted as exactly that, and says so. Where the same correspondence holds at
several lengths it is offered at any length; where it was only ever seen at
one, it answers at that length and refuses elsewhere — because a rule that has
never been tried at a size is not a rule about that size, whatever it would be
convenient for it to be.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "DerivedAddressing",
    "DerivedOperation",
    "an_addressing_nobody_wrote",
    "an_operation_nobody_wrote",
    "widen_with_addressing",
    "widen_with_operation",
]

logger = logging.getLogger("Aura.WideningTheLanguage")

#: How many transitions a derived operation must survive that it was not
#: derived from, before it is allowed into the language everything else is
#: built out of. Higher than for a single meaning: a wrong meaning answers one
#: family wrongly, a wrong PRIMITIVE poisons every hypothesis built with it.
ENOUGH_TO_WIDEN = 2


@dataclass(frozen=True)
class DerivedAddressing:
    """A way of saying where a value came from, read off examples.

    ``at`` maps a length to the correspondence observed at that length: for
    each position, which position its value was taken from.
    """

    name: str
    at: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __call__(self, index: int, size: int) -> int:
        seen = self.at.get(int(size))
        if seen is None or not (0 <= index < len(seen)):
            # Not a length she has seen it at. Refusing is the honest answer:
            # a correspondence read off four cells says nothing about six.
            raise IndexError(f"{self.name} was never seen at length {size}")
        return seen[index]

    def describes(self) -> str:
        lengths = ", ".join(str(one) for one in sorted(self.at))
        return f"{self.name} (seen at length {lengths})"


@dataclass(frozen=True)
class DerivedOperation:
    """A way of combining two values, read off examples."""

    name: str
    does: dict[tuple[Any, Any], Any] = field(default_factory=dict)

    def __call__(self, one: Any, other: Any) -> Any:
        if (one, other) in self.does:
            return self.does[(one, other)]
        raise KeyError(f"{self.name} was never seen on {(one, other)!r}")

    def describes(self) -> str:
        return f"{self.name} (seen on {len(self.does)} pair(s))"


def _where_each_came_from(
    before: Sequence[Any], after: Sequence[Any]
) -> tuple[int, ...] | None:
    """For each place in the result, the one place its value came from.

    None where it cannot be read off: a value that appears twice in the input
    has no single source, and a value that is not in the input at all was not
    taken from anywhere, so neither says anything about addressing.
    """
    seen: dict[Any, int] = {}
    for index, value in enumerate(before):
        if value in seen:
            return None
        seen[value] = index
    found: list[int] = []
    for value in after:
        where = seen.get(value)
        if where is None:
            return None
        found.append(where)
    return tuple(found)


def an_addressing_nobody_wrote(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *,
    already: Sequence[Callable[[int, int], int]] = (),
) -> DerivedAddressing | None:
    """A way of saying where values come from that the language cannot say.

    Read off the examples where it can be read off, checked for consistency
    across every example of the same length, and refused when an existing
    addressing already says it — a language does not need a second name for
    something it can already express.
    """
    pairs = [(tuple(before), tuple(after)) for before, after in transitions]
    at: dict[int, tuple[int, ...]] = {}
    for before, after in pairs:
        if len(before) != len(after):
            return None
        found = _where_each_came_from(before, after)
        if found is None:
            continue
        size = len(before)
        if size in at and at[size] != found:
            # Two examples of one length disagree about where things come
            # from, so there is no addressing here to derive.
            return None
        at[size] = found
    if not at:
        return None
    if _already_said_by(at, already):
        return None
    name = "where these came from"
    derived = DerivedAddressing(name=name, at=dict(at))
    logger.info("an addressing nobody wrote: %s", derived.describes())
    return derived


def _already_said_by(
    at: dict[int, tuple[int, ...]], already: Sequence[Callable[[int, int], int]]
) -> bool:
    """Whether something the language can already say produces this."""
    for existing in already:
        try:
            if all(
                tuple(existing(index, size) % size for index in range(size)) == found
                for size, found in at.items()
            ):
                return True
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return False


def an_operation_nobody_wrote(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    addressing: Callable[[int, int], int],
    second: Callable[[int, int], int],
    *,
    already: Sequence[Callable[[Any, Any], Any]] = (),
) -> DerivedOperation | None:
    """A way of combining two values that the language cannot say.

    Given where each of the two values comes from, what was done with them is
    whatever the result is — and that is a function she can read off, one pair
    at a time, provided the examples never disagree about the same pair.
    """
    does: dict[tuple[Any, Any], Any] = {}
    for before, after in ((tuple(b), tuple(a)) for b, a in transitions):
        size = len(before)
        if size == 0 or len(after) != size:
            return None
        for index in range(size):
            try:
                one = before[addressing(index, size) % size]
                other = before[second(index, size) % size]
            except (IndexError, TypeError, ValueError, ZeroDivisionError):
                return None
            got = after[index]
            if does.get((one, other), got) != got:
                return None
            does[(one, other)] = got
    if not does:
        return None
    for existing in already:
        try:
            if all(existing(one, other) == got for (one, other), got in does.items()):
                return None
        except (TypeError, ValueError):
            continue
    derived = DerivedOperation(name="what was done with these", does=dict(does))
    logger.info("an operation nobody wrote: %s", derived.describes())
    return derived


def widen_with_addressing(name: str, addressing: DerivedAddressing) -> str:
    """Put a derived addressing into the language every meaning is built from.

    This is the whole point. It is not a new hypothesis — it is a new WORD, and
    every hypothesis that can be formed afterwards may use it. The space she
    searches is strictly larger than the one she was given.
    """
    from core.cognition.an_invented_kind import WHERE_FROM

    said = str(name or "").strip()
    if not said or said in WHERE_FROM:
        return ""
    WHERE_FROM[said] = addressing
    logger.info(
        "the language grew: %d ways to say where a value comes from", len(WHERE_FROM)
    )
    return said


def widen_with_operation(name: str, operation: DerivedOperation) -> str:
    """Put a derived operation into the language every meaning is built from."""
    from core.cognition.an_invented_kind import WHAT_OF_IT

    said = str(name or "").strip()
    if not said or said in WHAT_OF_IT:
        return ""
    WHAT_OF_IT[said] = operation
    logger.info(
        "the language grew: %d ways to combine a pair", len(WHAT_OF_IT)
    )
    return said
