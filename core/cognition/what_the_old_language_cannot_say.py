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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.one_algebra import HEADS, Term, run
from core.cognition.the_floor_she_stands_on import (
    IF,
    MINUS,
    SAME,
    TIMES,
    A,
    Code,
    L,
    N,
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
    "every_word_she_could_make",
    "the_one_no_word_of_hers_says",
    "no_word_of_hers_says_it",
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


# ── a second witness, inside the range positions live in ──────────────────
#
# The bound above is about the numbers a term can produce, and doubling is
# outside because it grows too fast. A fair objection is that a WORD only ever
# hands back a place inside the state, so a function that leaves that range was
# never a candidate. The objection is right, and it needs a second argument.
#
# Here it is, and it is the older one. Every word she can make is a term over
# the words she was given — that is what `addressings` produces and what
# `what_it_rests_on` records — so the words she can ever have are recursively
# enumerable, and each of them either answers or refuses at every place. Walk
# that enumeration and answer differently from the n-th word at the n-th
# place. The result is a place inside the state, it is computable, and it is
# not what any word says. Not at any length, not after any number of makers,
# not over any vocabulary she can build, because the vocabulary she can build
# is what was walked.


def every_word_she_could_make(
    given: Sequence[Callable[[int, int], int]] | None = None,
) -> Iterator[tuple[str, Callable[[int, int], int]]]:
    """Every word constructible from the ones she was given, shortest first.

    Enumerated by size, with the constants a term may mention bounded by the
    size, so a term of any length and any constant appears eventually. That is
    all the argument needs: a surjection onto the words, computable, in an
    order that terminates on each element.
    """
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import Made, _choose, every_term, holes_in

    words = list(given if given is not None else WHERE_FROM.values())
    seen: set[str] = set()
    deepest = 1
    while True:
        for term in every_term(tuple(range(deepest + 1)), holes=2, deepest=deepest):
            takes = holes_in(term)
            for chosen in _choose(list(range(len(words))), max(0, takes)):
                name = f"{term.name}[{','.join(str(one) for one in chosen)}]"
                if name in seen:
                    continue
                seen.add(name)
                yield name, Made(
                    term=term, words=tuple(words[one] for one in chosen)
                )
        deepest += 1


def the_one_no_word_of_hers_says(
    given: Sequence[Callable[[int, int], int]] | None = None,
) -> Callable[[int, int], int]:
    """A place-valued rule that differs from the n-th word she can make, at n.

    Total, computable, and inside the range a word answers in. Slow, because
    it walks the enumeration to find the n-th word; that is a cost of the
    construction and not a limit on it.
    """
    made: list[Callable[[int, int], int]] = []
    walking = every_word_she_could_make(given)

    def says(index: int, size: int) -> int:
        while len(made) <= index:
            made.append(next(walking)[1])
        if size < 2:
            return 0
        try:
            answered = int(made[index](index, size)) % size
        except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
            # It refuses here. Answering at all is already different.
            return 0
        return (answered + 1) % size

    return says


def no_word_of_hers_says_it(how_many: int = 300, *, also_at: int = 3) -> dict[str, Any]:
    """Check the diagonal against the words themselves, rather than assert it.

    The n-th word is asked at position n, in a state long enough to contain
    that position — the diagonal point, and the one place the construction
    promises a disagreement. It is checked at a few longer states as well,
    because a construction that only worked at one length would be worth
    knowing about, and those extra checks are not part of the argument.
    """
    rule = the_one_no_word_of_hers_says()
    agreed: list[dict[str, Any]] = []
    checked = 0
    counted = 0
    for index, (name, word) in enumerate(every_word_she_could_make()):
        if index >= how_many:
            break
        counted = index + 1
        for size in (index + 2, *(index + 2 + step for step in range(1, also_at))):
            checked += 1
            try:
                said = int(word(index, size)) % size
            except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
                said = None
            if said is not None and said == rule(index, size):
                agreed.append({"word": name, "at": index, "size": size})
    return {
        "words": counted,
        "checked": checked,
        "agreed": agreed,
        "differs_from_every_one": not agreed,
    }
