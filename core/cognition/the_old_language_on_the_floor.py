"""The positional language, compiled into the floor, and checked to agree.

`one_algebra` is called one algebra because a word and a way of building words
are the same kind of thing. Two algebras remained. Positional terms compute
places and are run by `one_algebra.run`; value expressions compute what is done
to a pair and are run by `an_operation_that_generalises.Expression.__call__`.
Separate primitives, separate enumerators, separate serialisers, and an
invention in one could never be material for an invention in the other.

This closes half of that. Every positional term compiles to a floor term, and a
test runs both over a grid of places and lengths and demands they agree
everywhere, including where both refuse. So the floor is the semantics and the
positional algebra is a specialised way of writing some of it — which is what
lets a head, a word and a proposal rule all be the same kind of object.

Why keep the positional algebra at all
--------------------------------------
Because it is the right language for positional families and the floor is not.
A search over the floor is Levin-hard; a search over positional terms is
directed by a correspondence read straight off the examples. Compiling gives
one semantics without giving up the frontend, which is the arrangement every
serious treatment of this converges on.

Both halves are here. Positional terms compile through
:func:`compile_positional`, and the value expressions of
`an_operation_that_generalises` compile through :func:`compile_an_operation`.
Each has a check that runs both languages over a grid and demands the same
answer, refusals included, so what is claimed is behaviour rather than shape.

What is left after that is not a third algebra. It is the SCHEMA — a rule is
still two sources and one operation, and no head or word changes that. Which
is a different ceiling, and it is named in the record rather than closed here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from core.cognition.one_algebra import Term
from core.cognition.the_floor_she_stands_on import (
    BELOW,
    FST,
    IF,
    LEFTOVER,
    MINUS,
    PAIR,
    PLUS,
    SAME,
    SND,
    TIMES,
    A,
    Code,
    L,
    N,
    V,
    Y,
    build,
)

__all__ = [
    "THE_WORDS_SHE_WAS_GIVEN",
    "as_a_floor_term",
    "compile_an_operation",
    "compile_positional",
    "operations_agree_everywhere",
    "they_agree_everywhere",
]

logger = logging.getLogger("Aura.TheOldLanguageOnTheFloor")


def _length() -> Any:
    """The length, never nought, because every place is taken modulo it."""
    return IF(BELOW(V("n"), N(1)), N(1), V("n"))


def _inside(what: Any) -> Any:
    """A number brought back inside the state."""
    return LEFTOVER(what, _length())


#: The five words she was given, as floor terms of a place and a length. Each
#: is what `an_invented_kind.WHERE_FROM` computes, written where the floor can
#: run it.
THE_WORDS_SHE_WAS_GIVEN: dict[str, Code] = {
    "here": build(L("at", L("n", V("at")))),
    "the far end": build(
        L("at", L("n", MINUS(MINUS(V("n"), N(1)), V("at"))))
    ),
    "one along": build(
        L("at", L("n", LEFTOVER(PLUS(V("at"), N(1)), IF(BELOW(V("n"), N(1)), N(1), V("n")))))
    ),
    "one back": build(
        L("at", L("n", LEFTOVER(MINUS(V("at"), N(1)), IF(BELOW(V("n"), N(1)), N(1), V("n")))))
    ),
    "its partner": build(
        L(
            "at",
            L(
                "n",
                IF(
                    SAME(LEFTOVER(V("at"), N(2)), N(0)),
                    PLUS(V("at"), N(1)),
                    MINUS(V("at"), N(1)),
                ),
            ),
        )
    ),
}


def _word_at(words: Sequence[Code], which: int, place: Any) -> Any:
    """What the word in this hole says at this place."""
    if not 0 <= which < len(words):
        raise ValueError(f"no word for hole {which}")
    return _inside(A(words[which], place, V("n")))


def _applied_at(what: Term, words: Sequence[Code], place: Any) -> Any:
    """Use a term — or a hole standing for a word — at a place.

    The floor version of ``_apply``. A hole is the word itself; anything else
    is the term evaluated with the position moved.
    """
    if what.head == "hole":
        return _word_at(words, int(what.value or 0), place)
    return _inside(_compile(what, words, place))


def _compile(term: Term, words: Sequence[Code], place: Any) -> Any:
    """One positional term, as a named floor expression at this place."""
    head = term.head
    if head == "where":
        return place
    if head == "many":
        return V("n")
    if head == "fixed":
        return N(int(term.value or 0))
    if head == "hole":
        return _word_at(words, int(term.value or 0), place)
    if head == "through":
        # Bound rather than substituted, and the difference is not cosmetic.
        #
        # The positional interpreter works the inner term out before it uses
        # it, so a term that divides by nothing there refuses even when the
        # outer part ignores its position. Substituting the expression into
        # the body makes the floor lazy in exactly that case: `how many there
        # are of where it is over where it is` refused in one language and
        # answered nought in the other, on 12,000 terms across three pairs of
        # words. Application on the floor evaluates its argument, so binding
        # it restores the strictness.
        inner = _inside(_compile(term.parts[1], words, place))
        return A(
            L("through", _applied_at(term.parts[0], words, V("through"))),
            inner,
        )
    if head == "over again":
        # The count, capped at the length: walking a set of n places n times
        # has already reached whatever cycle it will reach.
        return LET_OVER_AGAIN(term, words, place)
    if head == "undo":
        return LET_UNDO(term, words, place)
    if head == "if":
        return IF(
            _compile(term.parts[0], words, place),
            _compile(term.parts[1], words, place),
            _compile(term.parts[2], words, place),
        )
    work = {
        "plus": PLUS,
        "minus": MINUS,
        "times": TIMES,
        "over": _over,
        "left over": _left_over,
        "below": BELOW,
        "same as": SAME,
    }.get(head)
    if work is None:
        raise ValueError(f"nothing this compiler knows called {head!r}")
    return work(
        _compile(term.parts[0], words, place),
        _compile(term.parts[1], words, place),
    )


def _over(one: Any, other: Any) -> Any:
    from core.cognition.the_floor_she_stands_on import OVER

    return OVER(one, other)


def _left_over(one: Any, other: Any) -> Any:
    return LEFTOVER(one, other)


def LET_OVER_AGAIN(term: Term, words: Sequence[Code], place: Any) -> Any:
    """Doing something as many times as the term itself says to."""
    times = _compile(term.parts[0], words, place)
    body = term.parts[1]
    return A(
        L(
            "cap",
            A(
                L(
                    "k",
                    _inside(
                        A(
                            Y(
                                "go",
                                L(
                                    "left",
                                    L(
                                        "p",
                                        IF(
                                            SAME(V("left"), N(0)),
                                            V("p"),
                                            A(
                                                V("go"),
                                                MINUS(V("left"), N(1)),
                                                _applied_at(body, words, V("p")),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            V("k"),
                            place,
                        )
                    ),
                ),
                # min(max(0, times), cap)
                A(
                    L(
                        "t",
                        IF(BELOW(V("t"), V("cap")), V("t"), V("cap")),
                    ),
                    IF(BELOW(times, N(0)), N(0), times),
                ),
            ),
        ),
        _length(),
    )


def LET_UNDO(term: Term, words: Sequence[Code], place: Any) -> Any:
    """The one place it could have come from, or a refusal where there is not one.

    The positional interpreter raises where the term does not move things one
    for one. The floor has no exceptions, so the refusal is written as asking
    for the first of a number, which is the floor's way of refusing and
    propagates the same way.
    """
    wanted = _inside(_compile(term.parts[1], words, place))
    body = term.parts[0]
    return A(
        L(
            "wanted",
            A(
                L(
                    "found",
                    IF(
                        SAME(FST(V("found")), N(1)),
                        SND(V("found")),
                        FST(N(0)),
                    ),
                ),
                A(
                    Y(
                        "go",
                        L(
                            "i",
                            L(
                                "count",
                                L(
                                    "where",
                                    IF(
                                        SAME(V("i"), _length()),
                                        PAIR(V("count"), V("where")),
                                        IF(
                                            SAME(
                                                _applied_at(body, words, V("i")),
                                                V("wanted"),
                                            ),
                                            A(
                                                V("go"),
                                                PLUS(V("i"), N(1)),
                                                PLUS(V("count"), N(1)),
                                                V("i"),
                                            ),
                                            A(
                                                V("go"),
                                                PLUS(V("i"), N(1)),
                                                V("count"),
                                                V("where"),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    N(0),
                    N(0),
                    N(0),
                ),
            ),
        ),
        wanted,
    )


def compile_positional(term: Term, words: Sequence[Code]) -> Code:
    """A positional term as a closed floor term of a place and a length."""
    return build(L("at", L("n", _compile(term, words, V("at")))))


def as_a_floor_term(term: Term, word_names: Sequence[str]) -> Code:
    """Compile, with the holes filled by the words she was given, by name."""
    return compile_positional(
        term, [THE_WORDS_SHE_WAS_GIVEN[name] for name in word_names]
    )


def they_agree_everywhere(
    terms: Sequence[Term],
    word_names: Sequence[str],
    *,
    sizes: Sequence[int] = (2, 3, 4, 5, 6, 7),
    fuel: int = 200_000,
) -> dict[str, Any]:
    """Run both and demand the same answer, including the same refusals.

    Structural equality of the two representations is not the test. What is
    checked is behaviour, everywhere the pair can be asked, and a term that
    refuses in one language must refuse in the other.
    """
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import run as positional_run
    from core.cognition.the_floor_she_stands_on import OutOfFuel, Stuck
    from core.cognition.the_floor_she_stands_on import run as run_on_the_floor

    words = [WHERE_FROM[name] for name in word_names]
    compiled_words = [THE_WORDS_SHE_WAS_GIVEN[name] for name in word_names]
    checked = 0
    apart: list[dict[str, Any]] = []
    for term in terms:
        try:
            compiled = compile_positional(term, compiled_words)
        except (ValueError, RecursionError):
            apart.append({"term": term.name, "why": "did not compile"})
            continue
        for size in sizes:
            for at in range(size):
                checked += 1
                try:
                    said = int(positional_run(term, at, size, words)) % size
                except (ArithmeticError, IndexError, RecursionError, TypeError,
                        ValueError):
                    said = None
                try:
                    made = int(
                        run_on_the_floor(
                            Code(
                                "of",
                                parts=(
                                    Code(
                                        "of",
                                        parts=(compiled, Code("a number", value=at)),
                                    ),
                                    Code("a number", value=size),
                                ),
                            ),
                            fuel=fuel,
                        )
                    ) % size
                except (OutOfFuel, Stuck, TypeError, ValueError, ZeroDivisionError):
                    made = None
                if said != made:
                    apart.append(
                        {
                            "term": term.name,
                            "at": at,
                            "size": size,
                            "positional": said,
                            "floor": made,
                        }
                    )
    return {"checked": checked, "apart": apart, "agree": not apart}


# ── the other algebra ─────────────────────────────────────────────────────
#
# An operation is an expression over two values: one of them, a constant, or a
# way of combining two smaller expressions. Eight ways, and each of them is a
# floor term over the pair.


def _how_they_combine(kind: str, one: Any, other: Any) -> Any:
    """One way of combining two numbers, as a floor expression."""
    if kind == "minus":
        return MINUS(one, other)
    if kind == "added":
        return PLUS(one, other)
    if kind == "multiplied":
        return TIMES(one, other)
    if kind == "how far apart they are":
        return A(
            L("a", L("b", IF(BELOW(V("a"), V("b")), MINUS(V("b"), V("a")),
                             MINUS(V("a"), V("b"))))),
            one,
            other,
        )
    if kind == "the larger":
        return A(
            L("a", L("b", IF(BELOW(V("a"), V("b")), V("b"), V("a")))), one, other
        )
    if kind == "the smaller":
        return A(
            L("a", L("b", IF(BELOW(V("b"), V("a")), V("b"), V("a")))), one, other
        )
    if kind == "what is left over":
        return LEFTOVER(one, other)
    if kind == "how many times it goes in":
        return _over(one, other)
    raise ValueError(f"nothing this compiler knows called {kind!r}")


def _compile_operation(rule: Any) -> Any:
    """One value expression, as a named floor expression over the pair."""
    kind = rule.kind
    if kind == "the first":
        return V("one")
    if kind == "the second":
        return V("other")
    if kind == "a fixed number":
        value = rule.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"the floor holds whole numbers, not {value!r}")
        return N(value)
    return _how_they_combine(
        kind,
        _compile_operation(rule.parts[0]),
        _compile_operation(rule.parts[1]),
    )


def compile_an_operation(rule: Any) -> Code:
    """A value expression as a closed floor term of two numbers."""
    return build(L("one", L("other", _compile_operation(rule))))


def operations_agree_everywhere(
    rules: Sequence[Any],
    *,
    over: Sequence[int] = (-3, -1, 0, 1, 2, 5, 7, 12),
    fuel: int = 200_000,
) -> dict[str, Any]:
    """Run both and demand the same answer, including the same refusals."""
    from core.cognition.the_floor_she_stands_on import OutOfFuel, Stuck
    from core.cognition.the_floor_she_stands_on import run as run_on_the_floor

    checked = 0
    apart: list[dict[str, Any]] = []
    for rule in rules:
        try:
            compiled = compile_an_operation(rule)
        except (ValueError, RecursionError, AttributeError):
            apart.append({"rule": rule.name, "why": "did not compile"})
            continue
        for one in over:
            for other in over:
                checked += 1
                try:
                    said = rule(one, other)
                    said = int(said)
                except (ArithmeticError, TypeError, ValueError):
                    said = None
                try:
                    made = int(
                        run_on_the_floor(
                            Code(
                                "of",
                                parts=(
                                    Code(
                                        "of",
                                        parts=(compiled, Code("a number", value=one)),
                                    ),
                                    Code("a number", value=other),
                                ),
                            ),
                            fuel=fuel,
                        )
                    )
                except (OutOfFuel, Stuck, TypeError, ValueError, ZeroDivisionError):
                    made = None
                if said != made:
                    apart.append(
                        {
                            "rule": rule.name,
                            "one": one,
                            "other": other,
                            "expression": said,
                            "floor": made,
                        }
                    )
    return {"checked": checked, "apart": apart, "agree": not apart}
