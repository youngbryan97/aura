"""Certificates that the floor is universal, and that its arithmetic is not load-bearing.

Two things have to be shown about a bedrock before anything can be built on it,
and every response in the council document asserted the first and skipped the
second.

**It reaches everything computable.** Kleene's characterisation gives a
checkable route: the partial computable functions are exactly what you get from
zero, successor and projection by composition, primitive recursion and
unbounded search. So the certificate is not an argument, it is five terms —
:data:`ZERO`, :data:`SUCCESS`, :func:`take_the_one_at`, :func:`by_recursion`
and :func:`the_least_where` — each written in the floor's eighteen heads and
each checked against what it is supposed to compute. Unbounded search is the
one that matters: it is the construct that can fail to stop, and the test that
it exhausts its meter on a predicate with no root is the test that the floor is
genuinely universal rather than merely large.

**Its instruction set is doing no work.** Seven arithmetic heads look like a
menu, and a menu is the thing this whole line of work exists to remove. Four of
the seven are shown here to be definable from the other three, as terms, with
the derived version checked against the primitive on a grid. What is left —
addition, subtraction and a comparison, plus functions, application, a branch,
pairs and quotation — is small enough that the choice among such sets changes
constants and nothing else.

What this does not establish
----------------------------
Universality is a statement about what exists, not about what is found.
Everything that follows is about reach, and reach is measured rather than
proved.
"""

from __future__ import annotations

from typing import Any

from core.cognition.the_floor_she_stands_on import (
    BELOW,
    IF,
    LET,
    MINUS,
    PLUS,
    SAME,
    A,
    Code,
    L,
    N,
    V,
    Y,
    build,
    run,
)

__all__ = [
    "AS_MANY_AS",
    "HOW_MANY_TIMES_IT_GOES_IN",
    "IS_IT_THE_SAME",
    "SUCCESS",
    "WHAT_IS_LEFT_OVER",
    "ZERO",
    "by_recursion",
    "take_the_one_at",
    "the_least_where",
    "what_the_arithmetic_rests_on",
]


# ── Kleene's three starting points ────────────────────────────────────────

#: The constant nought, as a function of one thing.
ZERO: Code = build(L("x", N(0)))

#: One more. Note the name: `succ` reads as a word from another language, and
#: everything else here is written to be read.
SUCCESS: Code = build(L("x", PLUS(V("x"), N(1))))


def take_the_one_at(how_many: int, which: int) -> Code:
    """Of this many things given one after another, hand back that one."""
    if not 0 <= which < how_many:
        raise ValueError(f"no argument {which} among {how_many}")
    body: Any = V(f"x{which}")
    for at in reversed(range(how_many)):
        body = L(f"x{at}", body)
    return build(body)


# ── and the three ways of building ────────────────────────────────────────
#
# Composition needs no constructor: applying one term to another is a head.


def by_recursion(base: Any, step: Any) -> Code:
    """Primitive recursion, as a term.

    ``f(x, 0) = base(x)`` and ``f(x, n+1) = step(x, n, f(x, n))``. The count
    comes down by one each time, so this always stops — which is exactly why it
    is not enough on its own and why the next function exists.
    """
    return build(
        Y(
            "again",
            L(
                "x",
                L(
                    "n",
                    IF(
                        SAME(V("n"), N(0)),
                        A(base, V("x")),
                        LET(
                            "less",
                            MINUS(V("n"), N(1)),
                            A(
                                step,
                                V("x"),
                                V("less"),
                                A(V("again"), V("x"), V("less")),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def the_least_where(holds: Any) -> Code:
    """Unbounded search: the smallest count at which this says nought.

    The one construct here that can fail to stop, and the reason the machine
    has a meter. Without it the floor computes a strict subset of the total
    functions and something computable is outside it forever; with it, nothing
    is. Every claim in this area turns on that difference.
    """
    return build(
        LET(
            "from",
            Y(
                "go",
                L(
                    "k",
                    IF(
                        SAME(A(holds, V("k")), N(0)),
                        V("k"),
                        A(V("go"), PLUS(V("k"), N(1))),
                    ),
                ),
            ),
            A(V("from"), N(0)),
        )
    )


# ── the arithmetic is not the point ───────────────────────────────────────
#
# Each of these is written with addition, subtraction and one comparison, and
# nothing else. Each is checked against the head it replaces.

#: Multiplication, counted out. Agrees with ``times`` where both are at least
#: nought; below that the primitive and this part company, and the check says
#: where.
AS_MANY_AS: Code = build(
    Y(
        "over_and_over",
        L(
            "a",
            L(
                "b",
                IF(
                    BELOW(V("b"), N(1)),
                    N(0),
                    PLUS(V("a"), A(V("over_and_over"), V("a"), MINUS(V("b"), N(1)))),
                ),
            ),
        ),
    )
)

#: Division, counted out.
HOW_MANY_TIMES_IT_GOES_IN: Code = build(
    Y(
        "goes_in",
        L(
            "a",
            L(
                "b",
                IF(
                    BELOW(V("a"), V("b")),
                    N(0),
                    PLUS(N(1), A(V("goes_in"), MINUS(V("a"), V("b")), V("b"))),
                ),
            ),
        ),
    )
)

#: What is left when it will not go in again.
WHAT_IS_LEFT_OVER: Code = build(
    Y(
        "left",
        L(
            "a",
            L(
                "b",
                IF(
                    BELOW(V("a"), V("b")),
                    V("a"),
                    A(V("left"), MINUS(V("a"), V("b")), V("b")),
                ),
            ),
        ),
    )
)

#: Equality, from one comparison used twice. Neither below the other.
IS_IT_THE_SAME: Code = build(
    L(
        "a",
        L(
            "b",
            IF(
                BELOW(V("a"), V("b")),
                N(0),
                IF(BELOW(V("b"), V("a")), N(0), N(1)),
            ),
        ),
    )
)


def what_the_arithmetic_rests_on() -> dict[str, Any]:
    """Which heads are definable from which, checked rather than claimed."""
    derived = {
        "times": AS_MANY_AS,
        "over": HOW_MANY_TIMES_IT_GOES_IN,
        "left over": WHAT_IS_LEFT_OVER,
        "same as": IS_IT_THE_SAME,
    }
    from core.cognition.the_floor_she_stands_on import ARITHMETIC

    agrees: dict[str, bool] = {}
    for head, term in derived.items():
        work = ARITHMETIC[head]
        ok = True
        for a in range(0, 13):
            for b in range(0, 13):
                if head in {"over", "left over"} and b == 0:
                    continue
                got = run(Code("of", parts=(Code("of", parts=(term, Code("a number", value=a))), Code("a number", value=b))))
                if got != work(a, b):
                    ok = False
        agrees[head] = ok
    return {
        "irreducible": ["plus", "minus", "below"],
        "derived": sorted(derived),
        "agrees": agrees,
        "all_agree": all(agrees.values()),
    }
