"""Telling a search that went badly from a language that cannot say it.

These want opposite responses and look identical from inside a loop. Nothing
fits, so: try harder, or change what trying means?

    a search failure          the answer is sayable and she did not find it.
                              More looking, a better order, a longer budget.
    a representation failure  every hypothesis her language admits fails, and
                              they fail on the SAME cases. No amount of
                              looking helps, because the thing is not in the
                              space being looked through.
    something she cannot see  the record contradicts itself: the same thing,
                              twice, came out two ways. No function of what
                              she can see fits, so neither of the other two
                              readings is even the right question.

Treating the second as the first is the expensive mistake: more compute
against a hypothesis space that does not contain the answer buys nothing, and
it is indistinguishable from bad luck until somebody asks what the failures
have in common.

What tells them apart is what is left over. Take the reading that accounts for
most of it, and look at the cases it misses. If something ELSE the language can
already say accounts for exactly those, then the language can say both halves
and not the whole — and what is missing is not a hypothesis but a way of
putting two of them together. That is a representation failure, and it names
its own remedy.

If nothing in the language accounts for the leftovers, there is no structure in
them to build on, and more looking is the honest response.

This needs no list of features to test the residual against. The language is
its own vocabulary for describing what it cannot do, which is the only
vocabulary that cannot be the wrong one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "A_SEARCH_FAILURE",
    "SOMETHING_SHE_CANNOT_SEE",
    "how_much_the_failures_share",
    "A_REPRESENTATION_FAILURE",
    "NOTHING_FAILED",
    "UNDECIDED",
    "WhyNothingFits",
    "why_nothing_fits",
]

logger = logging.getLogger("Aura.WhatTheFailuresHaveInCommon")

#: Something in the language accounts for it. There is nothing to diagnose.
NOTHING_FAILED = "something already fits"

#: The hypotheses fail on different things, so one of them is closer than the
#: rest and looking harder is the right response.
A_SEARCH_FAILURE = "a search that went badly"

#: They fail on the same things, far past what unrelated failures would share.
#: Whatever is missing is missing from all of them at once, which is what a
#: language being unable to say something looks like from inside.
A_REPRESENTATION_FAILURE = "a language that cannot say it"

#: The record is not a function of what she observed, so nothing that reads
#: only what she observed can fit it. Asked before the other two, because both
#: of them assume a best hypothesis exists to read the leftovers of.
SOMETHING_SHE_CANNOT_SEE = "a quantity she was not reading"

#: Too few hypotheses got far enough to compare, so there is nothing to read.
UNDECIDED = "not enough fitted to tell"


@dataclass(frozen=True)
class WhyNothingFits:
    """Which kind of failure this is, and the numbers that decided it."""

    because: str
    #: How many of the best hypotheses failed on every one of the shared cases.
    together_on: int = 0
    #: How many cases they would be expected to share if the failures were
    #: unrelated to one another.
    by_chance: float = 0.0
    considered: int = 0
    #: How much shorter the leftovers are described together than apart. A
    #: second opinion on the same question, from a different direction.
    shared: int = 0
    #: What the record proves about a quantity she was not reading, if it
    #: proves anything.
    unseen: Any = None

    @property
    def is_the_language(self) -> bool:
        return self.because == A_REPRESENTATION_FAILURE

    @property
    def is_something_unseen(self) -> bool:
        return self.because == SOMETHING_SHE_CANNOT_SEE

    def describes(self) -> str:
        if self.is_something_unseen:
            return f"{self.because}: {self.unseen}"
        if self.because in {NOTHING_FAILED, UNDECIDED}:
            return self.because
        return (
            f"{self.because}: the best of {self.considered} readings missed "
            f"{self.together_on} case(s), and something else she can already "
            f"say covers exactly those"
            if self.is_the_language
            else (
                f"{self.because}: the best of {self.considered} readings missed "
                f"{self.together_on} case(s) and nothing she can say accounts "
                "for them"
            )
        )

    @property
    def and_the_leftovers_repeat(self) -> bool:
        """Whether the leftovers share more than shuffled ones of the same size."""
        return self.shared > 0


def how_much_the_failures_share(leftovers: Sequence[Any]) -> int:
    """How much shorter the failures are described together than apart.

    Describing n unrelated things costs the sum of describing each. If they
    share something — one reusable account plus a little per case — describing
    them together costs less, and the difference is how much structure is in
    there waiting to be named. Positive means the failures themselves contain
    something reusable, which is when a missing concept should be suspected.

    Measured against a control with the same lengths and the same characters
    and no structure ACROSS cases, because a compressor finds savings in the
    digits alone: eight unrelated three-digit numbers compress a little
    together whatever they say, and without the control that reads as
    structure. What is returned is the saving beyond what the control got.

    The ideal measure is Kolmogorov complexity, which is uncomputable. This is
    a real compressor, which bounds it from above — and which sees repetition
    rather than algebra. A ramp is perfectly structured and scores nothing
    here, because no two of its cases are alike. So this is a second opinion
    and never the verdict: the coverage test is what decides.
    """
    import random
    import zlib

    said = [str(one).encode("utf-8") for one in leftovers if str(one).strip()]
    if len(said) < 2:
        return 0
    empty = len(zlib.compress(b"", 9))

    def saving(parts: list[bytes]) -> int:
        apart = sum(max(0, len(zlib.compress(one, 9)) - empty) for one in parts)
        together = max(0, len(zlib.compress(b"\x00".join(parts), 9)) - empty)
        return apart - together

    # Against the BEST the null manages, over several draws. One control is
    # itself a sample: measured on random numbers, a single shuffle left the
    # saving above zero on eleven of twenty draws, so "more than the control"
    # was not a signal at all. The floor is what the null actually reaches.
    shuffled = random.Random(0)
    floor = 0
    for _ in range(max(8, len(said))):
        control = []
        for one in said:
            letters = list(one)
            shuffled.shuffle(letters)
            control.append(bytes(letters))
        floor = max(floor, saving(control))
    return int(saving(said) - floor)


#: How many of the best-fitting hypotheses are compared. Fewer than three
#: cannot show agreement; the ones past that are worse fits saying less.
_ENOUGH_TO_COMPARE = 3


def why_nothing_fits(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    hypotheses: Sequence[Any] | None = None,
) -> WhyNothingFits:
    """Whether more searching would help, or whether the language is missing one.

    ``hypotheses`` are things with a ``read`` — every meaning the language
    admits, when nothing else is passed.
    """
    pairs = [(tuple(before), tuple(after)) for before, after in transitions]
    if not pairs:
        return WhyNothingFits(UNDECIDED)

    # Before asking which reading came closest: is there a reading at all? A
    # record where the same state came out two ways admits no function of that
    # state, so "the best of them missed these cases" is a sentence about a
    # best that cannot exist, and both other verdicts would be answering a
    # question the evidence has already closed.
    from core.cognition.something_she_cannot_see import what_she_cannot_see

    unseen = what_she_cannot_see(pairs)
    if unseen.anything:
        return WhyNothingFits(
            SOMETHING_SHE_CANNOT_SEE,
            considered=len(pairs),
            together_on=len(unseen.disagreeing_steps),
            unseen=unseen,
        )

    if hypotheses is None:
        from core.cognition.an_invented_kind import every_meaning

        hypotheses = list(every_meaning())

    covered: list[tuple[frozenset[int], Any]] = []
    for one in hypotheses:
        read = getattr(one, "read", None)
        if not callable(read):
            continue
        right: set[int] = set()
        for at, (before, after) in enumerate(pairs):
            try:
                if read(before) == after:
                    right.add(at)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
                continue
        if len(right) == len(pairs):
            return WhyNothingFits(NOTHING_FAILED, considered=len(hypotheses))
        if right:
            covered.append((frozenset(right), one))
    if len(covered) < 2:
        return WhyNothingFits(UNDECIDED, considered=len(covered))

    covered.sort(key=lambda one: -len(one[0]))
    everything = set(range(len(pairs)))
    best, _which = covered[0]
    left_over = everything - set(best)

    # Does something else the language already says account for exactly what
    # the best reading missed? Then both halves are sayable and the whole is
    # not, and what is missing is a way of joining them.
    joins_up = any(left_over <= set(right) for right, _one in covered[1:])
    # What that would be worth by chance: the share of the leftovers a reading
    # covering that many cases would take in anyway.
    expected = sum(len(right) for right, _one in covered) / max(1, len(covered))
    expected = expected * len(left_over) / max(1, len(pairs))

    found = WhyNothingFits(
        because=A_REPRESENTATION_FAILURE if joins_up else A_SEARCH_FAILURE,
        together_on=len(left_over),
        by_chance=expected,
        considered=len(covered),
        shared=how_much_the_failures_share(
            [pairs[at][1] for at in sorted(left_over)]
        ),
    )
    logger.info("why nothing fits — %s", found.describes())
    return found
