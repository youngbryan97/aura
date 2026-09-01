"""The floor's evaluator, written as a term in the floor.

The claim everywhere in this area is that the mechanism becomes an object of
the language it works on. That sentence is cheap. This module is the price of
it: :data:`THE_INTERPRETER` is a value — a term of eighteen heads and nothing
else — which, given the encoding of any floor term, computes what that term
computes.

Why it matters, exactly
-----------------------
Without it, "she can revise the machinery of her own learning" means she can
revise things a Python function then interprets, and the Python function is the
next authored level. With it, evaluation itself is inside the hypothesis space:
a term can read a term, take it apart, build a different one, and run it. A new
way of building is then a term, a proposer is a term, and a proposer that
writes proposers is a term. There is no level at which the shape of the object
changes, so there is no level at which a new mechanism is needed to handle the
new shape.

That is the whole content of the collapse, and it is standard: Mogensen showed
a self-interpreter for the untyped lambda calculus in 1992, and reflective
towers go back to Smith's 3-LISP. Nothing here is new mathematics. What is new
is that it is the evaluator of the language Aura's rules are actually written
in, rather than a demonstration beside it.

What it does not buy
--------------------
Representability, and only that. Kleene's recursion theorem says a program can
be handed its own description; it does not say the program will find a good
rewrite, that finding one is cheap, or that the rewrite is safe. Those are
experiments, and they are the ones that can fail.
"""

from __future__ import annotations

from typing import Any, Sequence

from core.cognition.the_floor_she_stands_on import (
    A,
    ARITHMETIC,
    Code,
    ENOUGH_STEPS,
    FST,
    IF,
    ISPAIR,
    L,
    LET,
    N,
    NIL,
    PAIR,
    QUOTE,
    SAME,
    SIGNATURE,
    SND,
    V,
    Y,
    _Named,
    build,
    run,
)

__all__ = [
    "THE_INTERPRETER",
    "interpret",
    "the_interpreter_as_written",
]

#: Where each head sits in the signature. Quotation writes a head as this
#: number, so the interpreter reads it back as this number.
_AT = {head: at for at, head in enumerate(SIGNATURE)}


def _rec(what: Any) -> _Named:
    """Evaluate a sub-term in the same environment."""
    return A(V("ev"), what, V("env"))


#: The parts of the term being evaluated, reached only inside the branch that
#: knows they exist. A branch of ``if`` is not evaluated unless it is taken, so
#: asking for the third part of a term that has one is never done.
_P0 = FST(V("ps"))
_P1 = FST(SND(V("ps")))
_P2 = FST(SND(SND(V("ps"))))


def _arithmetic_cases(otherwise: Any) -> Any:
    """One branch per arithmetic head, over the two evaluated parts."""
    made = otherwise
    for head in reversed(list(ARITHMETIC)):
        made = IF(
            SAME(V("h"), N(_AT[head])),
            _Named(head, parts=(_rec(_P0), _rec(_P1))),
            made,
        )
    return made


def _the_dispatch() -> Any:
    """One branch per head, in signature order."""
    # Nothing matched. Asking for the first of a number is the floor's way of
    # refusing, and it refuses loudly rather than returning a wrong answer.
    stuck: Any = FST(N(0))
    body = _arithmetic_cases(stuck)
    body = IF(SAME(V("h"), N(_AT["as it is written"])), _P0, body)
    body = IF(
        SAME(V("h"), N(_AT["if"])),
        IF(_rec(_P0), _rec(_P1), _rec(_P2)),
        body,
    )
    body = IF(SAME(V("h"), N(_AT["of"])), A(_rec(_P0), _rec(_P1)), body)
    # A function in the interpreted term becomes a real function on the floor,
    # closing over the body and the environment. That is what makes this
    # metacircular rather than an interpreter with its own idea of a closure.
    body = IF(
        SAME(V("h"), N(_AT["given a thing"])),
        L("given", A(V("ev"), _P0, PAIR(V("given"), V("env")))),
        body,
    )
    body = IF(
        SAME(V("h"), N(_AT["the one it was given"])),
        A(V("nth"), V("env"), V("v")),
        body,
    )
    body = IF(SAME(V("h"), N(_AT["is it a pair"])), ISPAIR(_rec(_P0)), body)
    body = IF(SAME(V("h"), N(_AT["the second of"])), SND(_rec(_P0)), body)
    body = IF(SAME(V("h"), N(_AT["the first of"])), FST(_rec(_P0)), body)
    body = IF(SAME(V("h"), N(_AT["a pair"])), PAIR(_rec(_P0), _rec(_P1)), body)
    body = IF(SAME(V("h"), N(_AT["nothing"])), NIL, body)
    return IF(SAME(V("h"), N(_AT["a number"])), V("v"), body)


def the_interpreter_as_written() -> Any:
    """The interpreter before names become distances.

    Kept as a function so the shape can be read. ``env`` is a list of values,
    innermost first, exactly as the machine keeps it.
    """
    return LET(
        "nth",
        Y(
            "nth",
            L(
                "xs",
                L(
                    "k",
                    IF(
                        SAME(V("k"), N(0)),
                        FST(V("xs")),
                        A(V("nth"), SND(V("xs")), _Named("minus", parts=(V("k"), N(1)))),
                    ),
                ),
            ),
        ),
        Y(
            "ev",
            L(
                "enc",
                L(
                    "env",
                    LET(
                        "h",
                        FST(V("enc")),
                        LET(
                            "rest",
                            SND(V("enc")),
                            LET(
                                "v",
                                FST(V("rest")),
                                LET("ps", SND(V("rest")), _the_dispatch()),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


#: The evaluator, as a term. Applied to a written term and an environment, it
#: gives what that term gives.
THE_INTERPRETER: Code = build(the_interpreter_as_written())


def interpret(
    code: Code, env: Sequence[Code] = (), *, fuel: int = ENOUGH_STEPS
) -> Any:
    """Run a term by running the interpreter on it, rather than by running it.

    The two answers must agree, and the test that says so is the only reason
    to believe the interpreter is the floor's evaluator rather than something
    that resembles it.
    """
    made: Any = NIL
    for one in reversed(list(env)):
        made = PAIR(one, made)
    return run(build(A(THE_INTERPRETER, QUOTE(code), made)), fuel=fuel)
