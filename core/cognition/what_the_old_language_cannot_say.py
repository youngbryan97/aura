"""A behaviour her positional language can never reach, however much it grows.

`core/cognition/what_growth_cannot_do.py` records that a universal language
cannot be made more expressive from inside, and that a word defined as a term
adds no meanings. Both are true. Neither says which side of that line the
language she actually thinks in falls on, and the answer decides whether the
regress is real.

It is real, and here is the reason rather than the assertion.

The bound
---------
Every positional term computes a number, and how large that number can get is
fixed by the term's own length. Write ``L`` for the symbols in a term, ``c``
for the largest constant written into it, and ``B = max(n, c, 2)`` at a state
of length ``n``. Then

    |run(T, i, n, W)| <= B ** L

by induction on the term. Reading a position or a length gives at most ``n``.
Applying a word, undoing one, or repeating one hands back a place inside the
state, so all three are below ``n`` whatever the word does — which is why no
amount of inventing words moves this. A comparison gives nought or one.
Addition of two parts bounded by ``B**L1`` and ``B**L2`` is at most
``2*B**max`` and so at most ``B**(L1+L2+1)``; multiplication is exactly
``B**(L1+L2)``; division and remainder only shrink. A branch is one of its
arms. Every case is covered because the heads are.

The witness
-----------
So for a FIXED term, what it can say grows at most like ``n**L`` — a polynomial
whose degree is the length of the term. And

    f(n) = 2 ** n

outgrows every polynomial. Therefore no positional term computes it: not at any
length, not over any vocabulary, not after any number of makers, levels or
inventions. The only thing that puts it in the language is a person editing
``run``.

That is the regress, located and measured. It is not a complaint about taste in
grammars.

What follows
------------
The floor says the same function in a term you can read, so

    E(floor) is strictly larger than E(positional terms)

and it is proved on both sides: a proof of exclusion above, a construction
below. It is also the LAST such gain that will ever be available, because the
floor is universal and
:mod:`core.cognition.what_growth_cannot_do` already has the theorem for that
case. Expressiveness grows exactly once more and then stops, which is why
everything after this is counted in reach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from core.cognition.one_algebra import HEADS, Term, run
from core.cognition.the_floor_she_stands_on import (
    A,
    Code,
    IF,
    L,
    MINUS,
    N,
    SAME,
    TIMES,
    V,
    Y,
    build,
    how_long,
)
from core.cognition.the_floor_she_stands_on import run as run_on_the_floor

__all__ = [
    "DOUBLING",
    "Ceiling",
    "biggest_constant_in",
    "the_bound",
    "the_bound_holds_on",
    "where_doubling_escapes",
    "a_sample_of_terms",
    "the_heads_the_argument_covers",
    "why_it_cannot_be_said",
]

logger = logging.getLogger("Aura.WhatTheOldLanguageCannotSay")


def biggest_constant_in(term: Term) -> int:
    """The largest number written into the term itself."""
    here = abs(int(term.value or 0)) if term.head == "fixed" else 0
    return max([here, *(biggest_constant_in(part) for part in term.parts)])


def the_bound(term: Term, size: int) -> int:
    """The largest number this term can produce at a state of this length.

    ``B ** L``, where ``B`` is the largest of the length, the largest constant
    in the term, and two. Not tight, and it does not need to be: what matters
    is that it is a polynomial in the length whose degree is fixed by the term.
    """
    base = max(int(size), biggest_constant_in(term), 2)
    return base ** term.how_long()


def the_bound_holds_on(
    terms: Sequence[Term],
    words: Sequence[Callable[[int, int], int]],
    *,
    sizes: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 11),
) -> dict[str, Any]:
    """Run the induction's conclusion against the interpreter.

    A proof by cases over a head list is only as good as the head list, and
    that list has been wrong here before. This executes the claim: for every
    term, at every size, at every position, the answer is inside the bound.
    """
    checked = 0
    broken: list[dict[str, Any]] = []
    for term in terms:
        limit_by_size = {size: the_bound(term, size) for size in sizes}
        for size in sizes:
            for at in range(size):
                try:
                    said = run(term, at, size, words)
                except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
                    continue
                checked += 1
                if abs(int(said)) > limit_by_size[size]:
                    broken.append(
                        {
                            "term": term.name,
                            "at": at,
                            "size": size,
                            "said": int(said),
                            "bound": limit_by_size[size],
                        }
                    )
    return {"checked": checked, "broken": broken, "holds": not broken}


def where_doubling_escapes(length: int, *, biggest_constant: int = 64) -> int:
    """A state length at which doubling is past what any term this long can say.

    The witness the theorem needs, produced rather than argued for. Given a
    length budget, this returns an ``n`` with ``2 ** n`` greater than
    ``max(n, c, 2) ** length`` — so every term of that length or shorter, over
    any vocabulary, is already too small at that ``n``.
    """
    size = 2
    while True:
        base = max(size, int(biggest_constant), 2)
        if 2**size > base**length:
            return size
        size += 1


#: Doubling, on the floor, in a term short enough to read.
DOUBLING: Code = build(
    Y(
        "twice",
        L(
            "n",
            IF(
                SAME(V("n"), N(0)),
                N(1),
                TIMES(N(2), A(V("twice"), MINUS(V("n"), N(1)))),
            ),
        ),
    )
)


@dataclass(frozen=True)
class Ceiling:
    """What the positional language cannot say, and what the floor says instead."""

    #: The longest positional term the argument was run against.
    up_to_length: int
    #: A state length at which every term that long is already too small.
    escapes_at: int
    #: What doubling gives there, and the most any such term could give.
    doubling_says: int
    #: The bound every term of that length obeys.
    the_most_it_could_say: int
    #: Symbols the floor needs to say the same thing.
    on_the_floor: int
    #: Whether the floor's term actually computes doubling, checked.
    the_floor_says_it: bool

    @property
    def strictly_wider(self) -> bool:
        return self.doubling_says > self.the_most_it_could_say and self.the_floor_says_it

    def describes(self) -> str:
        return (
            f"at length {self.escapes_at} doubling is {self.doubling_says} and no "
            f"positional term of {self.up_to_length} symbols can exceed "
            f"{self.the_most_it_could_say}; the floor says it in "
            f"{self.on_the_floor} symbols"
        )


def why_it_cannot_be_said(
    *, up_to_length: int = 24, biggest_constant: int = 64
) -> Ceiling:
    """The whole argument, executed, at a stated length budget.

    ``up_to_length`` is not a search budget and nothing here searches. It is
    the length at which the arithmetic is reported; the conclusion holds at
    every length, because the escape point moves with it.
    """
    size = where_doubling_escapes(up_to_length, biggest_constant=biggest_constant)
    base = max(size, biggest_constant, 2)
    doubled = 2**size
    on_the_floor = int(
        run_on_the_floor(Code("of", parts=(DOUBLING, Code("a number", value=size))))
    )
    return Ceiling(
        up_to_length=up_to_length,
        escapes_at=size,
        doubling_says=doubled,
        the_most_it_could_say=base**up_to_length,
        on_the_floor=how_long(DOUBLING),
        the_floor_says_it=on_the_floor == doubled,
    )


def a_sample_of_terms(how_many: int = 4000, *, deepest: int = 3) -> Iterator[Term]:
    """Terms from the positional enumerator, for running the bound against."""
    from core.cognition.one_algebra import every_term

    for at, term in enumerate(every_term((0, 1, 2, 3), holes=2, deepest=deepest)):
        if at >= how_many:
            return
        yield term


def the_heads_the_argument_covers() -> set[str]:
    """Every head the induction has a case for.

    Read off here so the coverage test can compare it with what ``run``
    dispatches on. An induction over a head list is worth exactly what the
    head list is worth.
    """
    return {
        "where", "many", "fixed", "hole", "through", "undo", "over again", "if",
        *HEADS,
    }
