"""The thing that says what to try next, as a term rather than as a loop.

`the_order_she_tries_them_in` made one part of the machinery an object: the
rule deciding which word goes in a hole first. What stayed authored is the
larger half — the loop that produces candidates at all. A proposer written as a
Python generator is a proposer she cannot replace, and replacing it is what
experiment H is about.

A proposer here is a term of one shape:

    which one, how many leaves  ->  the encoding of a candidate

Given a number it hands back a term, so a stream is that term asked for nought,
one, two and so on. It is homoiconic all the way down: what it returns is the
candidate written as numbers and pairs, which is what quotation already makes a
term into, so a proposer building a candidate is a term building a term with no
machinery in between.

What it covers, and what it does not
------------------------------------
:data:`THE_PROPOSER` walks the terms of one arithmetic head over two leaves —
the depth-two space, which is where nearly every short answer sits. Longer
candidates still come from `every_code`, and saying which half is which is the
point: the claim is that the proposal path has a replaceable term in it, not
that no Python remains anywhere.

Why the shape is a number in rather than a stream out
------------------------------------------------------
A stream is a thing that holds state, and a term that holds state has to hold
it in its own argument. Asking for the k-th candidate is the same enumeration
with the state on the outside, where the harness can bound it, restart it, and
compare two proposers on the same k. That is worth more than the elegance of a
generator, because the comparison is the experiment.
"""

from __future__ import annotations

import logging
from typing import Any

from core.cognition.the_floor_she_stands_on import (
    IF,
    LEFTOVER,
    MINUS,
    NIL,
    OVER,
    PAIR,
    PLUS,
    SIGNATURE,
    A,
    Code,
    L,
    N,
    OutOfFuel,
    Stuck,
    V,
    build,
    decode,
    read_back,
    run,
    written_down,
)

__all__ = [
    "HOW_MANY_ARITHMETIC_HEADS",
    "THE_DEEPER_PROPOSER",
    "THE_PROPOSER",
    "WHAT_A_PROPOSER_IS_GIVEN",
    "forget_the_proposer",
    "the_candidate_at",
    "the_proposer_in_use",
    "the_proposer_she_wrote",
    "the_proposer_read_back",
    "the_proposer_written_down",
]

logger = logging.getLogger("Aura.TheProposerSheCanReplace")

#: What a proposer is given, outermost binder first.
WHAT_A_PROPOSER_IS_GIVEN: tuple[str, ...] = ("which one", "how many leaves")

#: Where the arithmetic starts in the signature, and how many there are. Read
#: off the signature rather than written down twice, because the signature is
#: the contract quotation encodes against and a second copy of it would drift.
_WHERE_THE_ARITHMETIC_STARTS = SIGNATURE.index("plus")
HOW_MANY_ARITHMETIC_HEADS = SIGNATURE.index("same as") - _WHERE_THE_ARITHMETIC_STARTS + 1

_A_NUMBER = SIGNATURE.index("a number")
_A_VARIABLE = SIGNATURE.index("the one it was given")

#: How many of the leaves are the things a head is given. Past that they are
#: constants, counting from nought.
_HOW_MANY_BINDINGS = 7

#: What one proposal may spend. A proposer is asked once per candidate and a
#: search asks for thousands, so this is small; one that cannot answer inside
#: it proposes nothing, which sends the search to the next number rather than
#: stopping it.
_A_PROPOSAL_MAY_SPEND = 20_000


def _a_leaf(which: Any) -> Any:
    """A leaf as a written term: one of the things a head is given, or a number."""
    return IF(
        _below(which, N(_HOW_MANY_BINDINGS)),
        PAIR(N(_A_VARIABLE), PAIR(which, NIL)),
        PAIR(N(_A_NUMBER), PAIR(MINUS(which, N(_HOW_MANY_BINDINGS)), NIL)),
    )


def _below(one: Any, other: Any) -> Any:
    from core.cognition.the_floor_she_stands_on import BELOW

    return BELOW(one, other)


def _the_proposer_as_written() -> Any:
    """The default proposer before names become distances.

    One arithmetic head over two leaves, indexed by a single number: the head
    from what is left when the number is divided by how many there are, and
    the two leaves from what is left of the rest.
    """
    return L(
        "which",
        L(
            "leaves",
            A(
                L(
                    "rest",
                    A(
                        L(
                            "left",
                            A(
                                L(
                                    "right",
                                    PAIR(
                                        PLUS(
                                            N(_WHERE_THE_ARITHMETIC_STARTS),
                                            LEFTOVER(
                                                V("which"), N(HOW_MANY_ARITHMETIC_HEADS)
                                            ),
                                        ),
                                        PAIR(
                                            N(0),
                                            PAIR(
                                                _a_leaf(V("left")),
                                                PAIR(_a_leaf(V("right")), NIL),
                                            ),
                                        ),
                                    ),
                                ),
                                LEFTOVER(OVER(V("rest"), V("leaves")), V("leaves")),
                            ),
                        ),
                        LEFTOVER(V("rest"), V("leaves")),
                    ),
                ),
                OVER(V("which"), N(HOW_MANY_ARITHMETIC_HEADS)),
            ),
        ),
    )


def _the_deeper_proposer_as_written() -> Any:
    """One arithmetic head whose left side is itself an arithmetic head.

    The depth-two proposer covers one head over two leaves, and past that the
    authored enumerator took over. This covers one head over a head and a leaf,
    which is the shape almost every longer answer has: something worked out,
    then combined with something simple.

    Flat rather than recursive on purpose. A term that recurses needs the fixed
    point, and the fixed point costs fuel on every candidate — which is paid on
    the ones that do not need it. Two shapes, chosen between by whether she
    thinks the deeper one is worth it, is cheaper than one shape that is always
    deep.

    Indexed the same way: the number is divided down through the outer head,
    the inner head, and three leaves.
    """
    def a_head_over(left: Any, right: Any, which: Any) -> Any:
        return PAIR(
            PLUS(
                N(_WHERE_THE_ARITHMETIC_STARTS),
                LEFTOVER(which, N(HOW_MANY_ARITHMETIC_HEADS)),
            ),
            PAIR(N(0), PAIR(left, PAIR(right, NIL))),
        )

    return L(
        "which",
        L(
            "leaves",
            A(
                L(
                    "inner",
                    A(
                        L(
                            "rest",
                            A(
                                L(
                                    "a",
                                    A(
                                        L(
                                            "b",
                                            A(
                                                L(
                                                    "c",
                                                    a_head_over(
                                                        a_head_over(
                                                            _a_leaf(V("a")),
                                                            _a_leaf(V("b")),
                                                            V("inner"),
                                                        ),
                                                        _a_leaf(V("c")),
                                                        V("which"),
                                                    ),
                                                ),
                                                LEFTOVER(
                                                    OVER(
                                                        OVER(V("rest"), V("leaves")),
                                                        V("leaves"),
                                                    ),
                                                    V("leaves"),
                                                ),
                                            ),
                                        ),
                                        LEFTOVER(
                                            OVER(V("rest"), V("leaves")), V("leaves")
                                        ),
                                    ),
                                ),
                                LEFTOVER(V("rest"), V("leaves")),
                            ),
                        ),
                        OVER(
                            OVER(V("which"), N(HOW_MANY_ARITHMETIC_HEADS)),
                            N(HOW_MANY_ARITHMETIC_HEADS),
                        ),
                    ),
                ),
                LEFTOVER(
                    OVER(V("which"), N(HOW_MANY_ARITHMETIC_HEADS)),
                    N(HOW_MANY_ARITHMETIC_HEADS),
                ),
            ),
        ),
    )


#: The proposer she was given. A term of the floor, and nothing else.
THE_PROPOSER: Code = build(_the_proposer_as_written())

#: The same idea one level down: a head over a head and a leaf. Not in force;
#: going deeper costs more per candidate, so it is something to decide rather
#: than something to default to.
THE_DEEPER_PROPOSER: Code = build(_the_deeper_proposer_as_written())

#: The one in force. Replaced by the same call that replaces a head, and there
#: is no second mechanism for it.
_IN_USE: list[Code] = [THE_PROPOSER]


def the_proposer_in_use() -> Code:
    """The proposer in force."""
    return _IN_USE[0]


def the_proposer_she_wrote(term: Code) -> Code:
    """Put a different proposer in force."""
    from core.cognition.the_floor_she_stands_on import how_long

    _IN_USE[0] = term
    logger.info("she is proposing differently: %d symbols", how_long(term))
    return term


def forget_the_proposer() -> Code:
    """Back to the one she started with. The lesion."""
    _IN_USE[0] = THE_PROPOSER
    return THE_PROPOSER


def the_candidate_at(which: int, *, leaves: int = 10) -> Code | None:
    """The candidate this proposer offers at that number, or nothing.

    A proposer that raises, runs out of fuel, or hands back something that is
    not a written term proposes nothing at that number. The search moves to the
    next one, which is the same answer an unusable candidate would have got.
    """
    work: Any = _IN_USE[0]
    try:
        made = run(work, fuel=_A_PROPOSAL_MAY_SPEND)
        for one in (int(which), max(1, int(leaves))):
            made = run(made.body, (one, *made.env), fuel=_A_PROPOSAL_MAY_SPEND)
        return decode(made)
    except (OutOfFuel, Stuck, TypeError, ValueError, AttributeError):
        return None


def the_proposer_written_down() -> dict[str, Any]:
    """The proposer as plain data, so one she wrote survives a restart."""
    return written_down(_IN_USE[0])


def the_proposer_read_back(row: Any) -> Code | None:
    """A proposer from what was written down, or nothing where it does not read."""
    return read_back(row)
