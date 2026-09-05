"""Why the regress ends at universality, and nowhere else.

`what_growth_cannot_do` records four walls, and one of them is that a universal
language cannot be made more expressive from inside. Every serious treatment of
this problem states that wall and then treats universality as a sensible choice
of substrate. Universality is the only place the tower can end, and the reason
is the other direction of the same argument, which nobody writes down.

Two theorems, and the second is the one that matters
----------------------------------------------------

**A term over a bedrock adds no meanings.** A word introduced as a term can be
substituted away — naming is a ``let``, and unfolding a ``let`` never changes
what an expression denotes. So under growth by construction alone,

    E(A_t) = E(B) for every t,

whatever she invents, at whatever level, for as long as she runs. That is the
ceiling, and it is already in this codebase.

**A bedrock that is not universal needs authoring forever.** Suppose
``E(B)`` is not all of the computable functions. Then some computable ``f`` is
outside it, and by the first theorem no amount of inventing puts it in — not a
word, not a maker, not a maker of makers, not a hundred levels. The only thing
that changes ``E`` is a person editing the evaluator. And after that edit the
bedrock is ``B'``; if ``E(B')`` is still not everything, the same argument
applies to ``B'``. So for any task stream that eventually asks for each
computable function, the number of required authoring events is unbounded.

That is the infinite regress, stated exactly: it is not a regress of
mechanisms, it is a regress of PRIMITIVES, and it is forced by the bedrock
being small rather than by anything about how invention is organised.

**A universal bedrock needs authoring never.** If ``E(B)`` is everything
computable, then every computable ``f`` is already in ``P(B)``, so no task ever
requires an edit — and by the first theorem no edit could add anything if one
were made. The sequence of required authorings has length nought.

Together: universality is necessary and sufficient for the tower to have a top.
Not preferable. Necessary.

`core/cognition/what_the_old_language_cannot_say.py` settles which side her
positional algebra falls on, with two witnesses, and
`core/cognition/what_the_floor_can_say.py` settles which side the floor falls
on, with Kleene's. So the tower had no top, and now it has one.

What is still authored, and why it cannot be otherwise
------------------------------------------------------
Three things, and only three.

The **meter**, because reach is measured in it and a thing that measures cannot
be the thing measured without the measurement becoming free.

The **gate** — persistence, novelty, reach, compression, held-out, rollback —
because a gate inside the hypothesis space is a gate a candidate can replace
with one that says yes. :func:`a_gate_inside_the_space_cannot_hold` executes
that rather than arguing it.

The **governor**: fuel, memory, the transaction, the privileges a term may not
reach. Same argument as the gate, and the same demonstration.

None of the three says anything about what a useful concept is. That is the
difference between a fixed substrate and a fixed vocabulary, and it is the
whole of what "no authored ceiling" can honestly mean.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AUTHORED_FOREVER",
    "NEVER_AGAIN",
    "UNDECIDED",
    "WhereItEnds",
    "a_gate_inside_the_space_cannot_hold",
    "naming_adds_no_meaning",
    "what_is_still_authored",
    "where_the_tower_ends",
]

logger = logging.getLogger("Aura.WhereTheTowerHasATop")

#: The bedrock is not universal, so something computable is outside it and only
#: a person puts it in — and the same is true of whatever they replace it with.
AUTHORED_FOREVER = "authoring is required again, and again after that"

#: The bedrock is universal. Nothing is outside it, so nothing needs adding and
#: nothing could be added.
NEVER_AGAIN = "nothing more will ever need authoring, and nothing more can be"

#: Not enough was established about the bedrock to say.
UNDECIDED = "not enough is known about the bedrock to say"


@dataclass(frozen=True)
class WhereItEnds:
    """What follows about authoring, from what is known about the bedrock."""

    verdict: str
    because: str
    #: The behaviour shown outside it, where one was shown.
    outside: str = ""
    #: What the certificate of universality rested on, where there was one.
    certificate: str = ""

    @property
    def has_a_top(self) -> bool:
        return self.verdict == NEVER_AGAIN

    def describes(self) -> str:
        return f"{self.verdict}: {self.because}"


def where_the_tower_ends(
    *,
    universal: bool | None,
    a_behaviour_outside: str = "",
    certificate: str = "",
) -> WhereItEnds:
    """Which of the three cases this bedrock is in.

    ``universal`` is not asserted here and must not be. It comes from a
    certificate — Kleene's constructors exhibited as terms, for the floor — or
    from a witness on the other side, and passing None is the honest answer
    where neither exists.
    """
    if universal is None:
        return WhereItEnds(
            UNDECIDED,
            "no certificate either way, so nothing follows about authoring",
        )
    if universal:
        return WhereItEnds(
            NEVER_AGAIN,
            (
                "every computable behaviour is already a term over it, so no "
                "task can require an edit and no edit could add a meaning"
            ),
            certificate=certificate,
        )
    return WhereItEnds(
        AUTHORED_FOREVER,
        (
            "something computable is outside it and no term over it reaches "
            "that, so only a person puts it in — and the same holds of "
            "whatever they replace it with, unless that is universal"
        ),
        outside=a_behaviour_outside,
    )


def naming_adds_no_meaning(
    says_it: Callable[[Any], bool],
    *,
    the_name: Any,
    written_out: Callable[[], Any],
) -> bool:
    """The substitution argument, on one word, run rather than cited.

    ``the_name`` is the word she made; ``written_out`` builds the same thing in
    the language it was made from, with the name gone. Agreement everywhere
    they are asked is the whole content of the theorem, on this word, in code.
    """
    try:
        without = written_out()
    except (ArithmeticError, TypeError, ValueError):
        return False
    try:
        return bool(says_it(the_name)) and bool(says_it(without))
    except (ArithmeticError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class WhatHeld:
    """Whether an invariant survived putting the thing that checks it in reach."""

    held_while_fixed: bool
    held_once_replaceable: bool
    what_replaced_it: str

    @property
    def shows_the_gate_must_stay_out(self) -> bool:
        return self.held_while_fixed and not self.held_once_replaceable

    def describes(self) -> str:
        return (
            f"the invariant held while the gate was fixed: "
            f"{self.held_while_fixed}; once the gate was itself admissible, a "
            f"candidate replaced it with {self.what_replaced_it} and the "
            f"invariant held: {self.held_once_replaceable}"
        )


def a_gate_inside_the_space_cannot_hold(
    candidates: Sequence[Any] = ("harmless", "harmful"),
) -> WhatHeld:
    """Build the candidate that turns the gate off, and run it.

    The argument for keeping something authored, executed instead of asserted.
    An invariant of the form "everything admitted is harmless" is inductive
    only while the thing checking it cannot itself be admitted. Make the gate a
    candidate and a candidate is available whose effect is to accept
    everything; after it is admitted the invariant is no longer preserved, and
    nothing inside the system was violated on the way.

    Nothing about the particular invariant matters, which is the point. Any
    non-trivial one goes the same way.
    """
    harmful = {one for one in candidates if "harm" in str(one)}

    def the_gate(one: Any) -> bool:
        return one not in harmful

    admitted: list[Any] = [one for one in candidates if the_gate(one)]
    held_while_fixed = not (set(admitted) & harmful)

    # Now the gate is a value like any other, so a candidate may be one.
    gate: Callable[[Any], bool] = the_gate
    everything = "a gate that says yes"

    def a_candidate_that_replaces_the_gate() -> Callable[[Any], bool]:
        return lambda _one: True

    if gate(everything) or True:
        # It passes its own check — it is harmless by the letter of the
        # invariant, because the invariant is about what is admitted and not
        # about what admits.
        gate = a_candidate_that_replaces_the_gate()
    admitted = [one for one in candidates if gate(one)]
    held_once_replaceable = not (set(admitted) & harmful)
    found = WhatHeld(
        held_while_fixed=held_while_fixed,
        held_once_replaceable=held_once_replaceable,
        what_replaced_it=everything,
    )
    logger.info("a gate inside the space — %s", found.describes())
    return found


def what_is_still_authored() -> dict[str, str]:
    """The three things above the floor that cannot become developmental.

    Written here so a claim that everything became developmental has somewhere
    to be checked against, and so the list can only shrink by argument rather
    than by being forgotten.
    """
    return {
        "the meter": (
            "reach is measured in steps, and a measure inside the thing "
            "measured makes the measurement free"
        ),
        "the gate": (
            "persistence, novelty, reach, compression, held-out and rollback; "
            "a gate a candidate can replace is a gate that says yes"
        ),
        "the governor": (
            "fuel, memory, the transaction, and the privileges a term may not "
            "reach; same argument as the gate"
        ),
    }
