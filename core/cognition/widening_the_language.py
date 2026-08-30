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
    "CONSTRUCTORS",
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
    """A way of combining two values, read off examples.

    Two kinds, and the difference is the whole difference between a memory and
    a word. With a ``rule`` it is an expression over the pair, so it answers a
    pair nobody has shown her. Without one it is the table of what she saw, and
    it refuses everything else — which is honest, and is as far as reading a
    correspondence off examples can get on its own.
    """

    name: str
    does: dict[tuple[Any, Any], Any] = field(default_factory=dict)
    #: What was done, where it was worked out rather than merely recorded.
    rule: Any = None

    def __call__(self, one: Any, other: Any) -> Any:
        if self.rule is not None:
            return self.rule(one, other)
        if (one, other) in self.does:
            return self.does[(one, other)]
        raise KeyError(f"{self.name} was never seen on {(one, other)!r}")

    def describes(self) -> str:
        if self.rule is not None:
            return f"{self.name}: {self.rule.name}, and it answers pairs it never saw"
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
    already: Any = (),
) -> DerivedAddressing | None:
    """A way of saying where values come from that the language cannot say.

    Read off the examples where it can be read off, checked for consistency
    across every example of the same length, and refused when the language can
    already say it.

    "Already" has to mean everything constructible and not merely the words
    that were written down. A candidate matching some composition of two words
    she has is a macro: it enlarges the vocabulary and leaves the set of
    meanings exactly where it was. Passing the primitives here instead of the
    closure is how a language reports growth it did not have.

    A macro is still admitted when it is short enough to pay for itself — see
    ``a_shorthand_worth_having`` — but it is admitted as brevity and never
    counted as a new meaning.
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
    if _already_said_by(at, already) is not None:
        return None
    name = "where these came from"
    derived = DerivedAddressing(name=name, at=dict(at))
    logger.info("an addressing nobody wrote: %s", derived.describes())
    return derived


def _already_said_by(at: dict[int, tuple[int, ...]], already: Any) -> str | None:
    """What the language already uses to say this, where it can say it at all.

    Returns the name so the caller can weigh the old way against the new one.
    A mapping of names is what should be passed; a bare sequence still works
    and gives back a position instead of a name.
    """
    if hasattr(already, "items"):
        candidates = list(already.items())
    else:
        candidates = [(f"word {index}", word) for index, word in enumerate(already)]
    for name, existing in candidates:
        try:
            if all(
                tuple(existing(index, size) % size for index in range(size)) == found
                for size, found in at.items()
            ):
                return str(name)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return None


def a_shorthand_worth_having(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *,
    already: Any,
    longest: int = 3,
) -> tuple[DerivedAddressing, str] | None:
    """A word for something she can already say, when saying it is long.

    This adds no meaning, and it is worth having anyway when the thing it
    stands for is long enough that the search saved beats the branch added.
    Both are counted in expressions she would have to walk, so the trade is
    settled by arithmetic rather than by preference.
    """
    from core.cognition.keeping_the_language_small import what_a_word_is_worth
    from core.cognition.what_it_costs_to_say import _symbols

    at: dict[int, tuple[int, ...]] = {}
    for before, after in ((tuple(b), tuple(a)) for b, a in transitions):
        if len(before) != len(after):
            return None
        found = _where_each_came_from(before, after)
        if found is None:
            continue
        if len(before) in at and at[len(before)] != found:
            return None
        at[len(before)] = found
    if not at:
        return None
    said_by = _already_said_by(at, already)
    if said_by is None:
        return None
    vocabulary = len(already) if hasattr(already, "__len__") else 1
    worth = what_a_word_is_worth(
        said_by,
        vocabulary=max(1, vocabulary),
        longest=max(1, int(longest)),
        shorter_by=max(0, _symbols(said_by) - 1),
    )
    if not worth.pays:
        return None
    logger.info("a shorthand that pays: %s", worth.describes())
    return DerivedAddressing(name="a shorter way of saying this", at=dict(at)), said_by


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
    # Work out the rule before settling for the record of it. A table is what
    # is left when nothing explains the pairs, and reaching for it first means
    # never finding the explanation that was there.
    from core.cognition.an_operation_that_generalises import (
        an_operation_that_generalises,
    )

    worked_out = an_operation_that_generalises(
        [(one, other, got) for (one, other), got in does.items()]
    )
    derived = DerivedOperation(
        name="what was done with these", does=dict(does), rule=worked_out
    )
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


# ── and a way of INVENTING words, which is a different thing again ───────


@dataclass(frozen=True)
class OneAfterAnother:
    """Two ways of saying where a value comes from, used one after the other."""

    first: Callable[[int, int], int]
    then: Callable[[int, int], int]

    def __call__(self, index: int, size: int) -> int:
        return self.then(self.first(index, size) % size, size)


def one_after_another(words: dict[str, Any]) -> dict[str, Any]:
    """Every pair of addressings, applied in turn.

    A way of BUILDING words rather than a word. Admitted, it applies to every
    addressing she has and every one she ever derives, so what it enlarges is
    not the language but the language's capacity to grow.
    """
    made: dict[str, Any] = {}
    for first_name, first in words.items():
        for then_name, then in words.items():
            if first_name == then_name:
                continue
            made[f"{first_name}, then {then_name}"] = OneAfterAnother(first, then)
    return made


#: Every way of building this source knows how to make. A kept language names
#: the ways it grew by, and a name is resolved here — so what comes back after
#: a restart can only ever be a constructor that is already written down.
CONSTRUCTORS: dict[str, Any] = {"one after another": one_after_another}


def a_way_of_building_nobody_wrote(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> str | None:
    """A new way of making words, when no new WORD would be enough.

    The step past deriving a primitive. Deriving one answers the family in
    front of her and composes with everything — but it is read off what she was
    shown, so it says nothing about a case it has not seen, and a language that
    grows only by memorising is not really growing.

    A way of building is different in kind. It takes the words she already has,
    including the ones she derived, and makes more out of them — so admitting
    one enlarges what she can say about families she has never met, which is
    the thing a derived constant cannot do.

    Tried only when nothing else works, and admitted only if the family that
    defeated everything becomes sayable with it. A way of building that changes
    nothing is not a discovery, it is a bigger search.
    """
    from core.cognition.an_invented_kind import WAYS_TO_BUILD, induce_from

    name = "one after another"
    if name in WAYS_TO_BUILD:
        return None
    if induce_from(transitions) is not None:
        # Already sayable. A way of building earns its place by making
        # something possible, not by being available when nothing needed it.
        return None
    WAYS_TO_BUILD[name] = one_after_another
    if induce_from(transitions) is None:
        # It did not make this sayable, so it has not earned a place in how she
        # thinks. Enlarging the search for nothing is worse than not enlarging
        # it: every hypothesis it adds is another chance for a coincidence.
        WAYS_TO_BUILD.pop(name, None)
        return None
    logger.info(
        "a way of MAKING words, not a word: %r — the language can now grow "
        "in a direction it could not before",
        name,
    )
    return name
