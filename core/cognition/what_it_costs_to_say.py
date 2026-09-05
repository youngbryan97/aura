"""What a thought costs to say, and what a new word buys.

Three different things get called "the language", and keeping them apart
decides whether growth is real:

    A(t)          the primitive words
    P(A(t))       everything constructible from them
    E(t)          the meanings those constructions actually have

Adding a word always enlarges A. It enlarges E only when the word means
something no construction already meant. A word whose meaning was already
reachable is a macro: new syntax over the same semantics, and

    E(t+1) = E(t)

however much larger A got. So a candidate is checked against the closure, not
against the primitives, and the check names the old expression that already
said it.

Once a word is admitted honestly, what it buys is not new meanings but shorter
ones. Let

    K(t, f) = the length of the shortest expression meaning f

A word standing in for a structure of length m that recurs k times takes
k(m-1) symbols out of every expression using it. That is worth measuring
because search is exponential in length: over a vocabulary of b words there are
about b**L expressions of length up to L, so shortening a solution by d shrinks
the space she must walk by b**d. Under a prior that favours short hypotheses,
the same idea becomes 2**d times more probable.

None of that makes anything newly computable. It moves thoughts from possible
to reachable, and the second is the one that decides what she can actually do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "AlreadySaid",
    "everything_sayable",
    "how_long_it_is",
    "how_many_expressions",
    "how_much_shorter",
    "in_order_of_length",
    "more_probable_by",
    "what_already_says_it",
    "what_it_means",
]

logger = logging.getLogger("Aura.WhatItCostsToSay")

#: Lengths a meaning is compared over when working out what it means. A bounded
#: witness, and it is bounded on purpose: two expressions agreeing on every
#: state of every length is not something anything finite can check, so what is
#: claimed is agreement over what was tried.
_OVER_LENGTHS = (2, 3, 4, 5)


@dataclass(frozen=True)
class AlreadySaid:
    """A candidate word whose meaning the language already had."""

    said_by: str
    #: How long the old way of saying it is, in symbols. A macro that saves
    #: nothing is worthless; one that turns nine symbols into one is worth
    #: keeping for its length even though it adds no meaning.
    instead_of: int

    def describes(self) -> str:
        return f"already said by {self.said_by!r}, in {self.instead_of} symbol(s)"


def how_long_it_is(meaning: Any) -> int:
    """How many primitive symbols a meaning is written in.

    A word built by composition costs what it is composed of. This is what
    makes a shorter way of saying the same thing measurable rather than a
    matter of taste.

    Only what is read is charged for. A meaning that takes a value as it is
    never looks at the second place, so counting that place would make "take
    the far end" cost the same as a rule reading two places and combining
    them — and then the search has no reason to prefer the simpler of two
    readings that fit the same examples.
    """
    where_from = str(getattr(meaning, "where_from", "") or "")
    and_from = str(getattr(meaning, "and_from", "") or "")
    what_of_it = str(getattr(meaning, "what_of_it", "") or "")
    if what_of_it == "as it is":
        return _symbols(where_from) + 1
    return _symbols(where_from) + _symbols(and_from) + (1 if what_of_it else 0)


def _symbols(word: str) -> int:
    """How many primitives a word name stands for."""
    if not word:
        return 0
    # A word made by a way of building names its parts, so its cost is its
    # parts. Nothing else knows how it was built and nothing else needs to.
    return word.count(", then ") + 1


def what_it_means(meaning: Any, over_lengths: Sequence[int] = _OVER_LENGTHS) -> Any:
    """What a meaning does, as one comparable value.

    Two expressions with the same value here agree everywhere she looked. That
    is the strongest thing available: see the module docstring on why it is not
    stronger.
    """
    from core.cognition.an_invented_kind import _every_telling_state

    return tuple(
        meaning.read(state)
        for size in sorted({int(size) for size in over_lengths if int(size) >= 2})
        for state in _every_telling_state(size)
    )


def everything_sayable(
    over_lengths: Sequence[int] = _OVER_LENGTHS,
) -> dict[Any, tuple[str, int]]:
    """E(t): every meaning the language currently has, and its shortest form.

    Keyed by what an expression does, so two ways of saying one thing collapse
    to one entry. Meanings are what get counted here; spellings are counted by
    ``every_meaning``, and the two numbers are rarely close.
    """
    from core.cognition.an_invented_kind import every_meaning

    found: dict[Any, tuple[str, int]] = {}
    for meaning in every_meaning():
        does = what_it_means(meaning, over_lengths)
        if all(part is None for part in does):
            continue
        length = how_long_it_is(meaning)
        standing = found.get(does)
        if standing is None or length < standing[1]:
            found[does] = (meaning.name, length)
    return found


def what_already_says_it(
    does: Any, over_lengths: Sequence[int] = _OVER_LENGTHS
) -> AlreadySaid | None:
    """Whether the language already means this, and what it uses to say it.

    The check that decides whether a candidate is growth or a macro. It is run
    against everything constructible, so composing two words she has counts as
    already being able to say it.
    """
    standing = everything_sayable(over_lengths).get(does)
    if standing is None:
        return None
    name, length = standing
    return AlreadySaid(said_by=name, instead_of=length)


def how_much_shorter(before: int, after: int, used: int = 1) -> int:
    """How many symbols a word saves, over every place it is used.

    One use of a word standing in for a structure of length m saves m-1. Used
    k times it saves k(m-1), and that multiplication is where abstraction
    stops being tidy and starts being the difference between reachable and
    not.
    """
    return max(0, int(used)) * max(0, int(before) - int(after))


def how_many_expressions(vocabulary: int, length: int) -> int:
    """How many expressions of length 1..L exist over a vocabulary of b words.

    Exactly (b**(L+1) - b) / (b - 1), which is the size of what she would have
    to walk if she walked all of it.
    """
    words, longest = int(vocabulary), int(length)
    if words <= 0 or longest <= 0:
        return 0
    if words == 1:
        return longest
    return (words ** (longest + 1) - words) // (words - 1)


def more_probable_by(shorter_by: int) -> float:
    """How much likelier a shortened hypothesis becomes under a short-first prior.

    Under P(h) proportional to 2**-K(h), taking d symbols out of a hypothesis
    multiplies its prior by 2**d. Ten symbols is a factor of a thousand: the
    idea did not change, the geometry of the space it sits in did.
    """
    return float(2 ** max(0, int(shorter_by)))


def in_order_of_length(meanings: Iterable[Any]) -> list[Any]:
    """Meanings shortest first, which is where the compression gets spent.

    Enumerating in the order the loops happen to run puts a nine-symbol
    coincidence in front of a two-symbol explanation, and whichever is checked
    first is the one that gets to win a tie. Ordering by length is the same
    prior as favouring short hypotheses, applied where it costs nothing.
    """
    return sorted(meanings, key=lambda meaning: (how_long_it_is(meaning), meaning.name))
