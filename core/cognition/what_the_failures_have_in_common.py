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
    a check that cannot see   every reading fails, and each says the wanted
                              answer under one consistent renaming. The
                              reading is right and the comparison is wrong,
                              and no amount of searching or widening fixes a
                              test that cannot recognise a correct answer.

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "A_CHECK_THAT_CANNOT_SEE_IT",
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

#: Every reading fails, and each of them says the wanted answer under one
#: consistent renaming. Measured live on 2026-08-30: a frontier battery scored
#: zero with the model answering Tokyo, 258662 and Gia correctly, because the
#: contract returns <answer>Tokyo</answer> and the graders compared the raw
#: text. Searching harder and widening the language are both wrong answers to
#: a test that cannot recognise a correct answer.
A_CHECK_THAT_CANNOT_SEE_IT = "a check that cannot see the answer"

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

    blind = _a_check_that_cannot_see_it(pairs, hypotheses)
    if blind is not None:
        return blind

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
    tried = 0
    for one in hypotheses:
        read = getattr(one, "read", None)
        if not callable(read):
            continue
        tried += 1
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
    if not covered and tried >= 2:
        # Every reading the language admits, wrong about every case.
        #
        # This counted how many hypotheses got SOMETHING right and called
        # fewer than two "not enough fitted to tell". That reads the strongest
        # evidence there is as the weakest: a language missing something is
        # missing it from all of its readings at once, and the limit of "they
        # fail on the same things" is that they fail on everything.
        #
        # It also closed the loop that grows the language. Eighty meanings ran
        # against a family needing a maximum of two sources, none got a single
        # case right, the verdict was undecided, and the path that writes a way
        # of building words is gated on this verdict — so the one situation the
        # writer exists for is the one situation it could never be reached in.
        # The precondition was kept true by its own failure.
        #
        # Two readable hypotheses is the floor, not a threshold: with fewer
        # than that there is no language to be short of anything. There is no
        # rival verdict available here either — a search that went badly means
        # one reading came closer, and nothing came closer than nothing.
        return WhyNothingFits(
            A_REPRESENTATION_FAILURE,
            considered=tried,
            together_on=len(pairs),
        )
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


def _a_check_that_cannot_see_it(
    pairs: Sequence[tuple[tuple[Any, ...], tuple[Any, ...]]],
    hypotheses: Sequence[Any] | None,
) -> WhyNothingFits | None:
    """Whether every reading is right and the comparison cannot tell.

    The signature is one consistent renaming. If what a reading says maps to
    what was wanted by a single map that holds across every case — the same
    thing always becoming the same other thing — then the reading has the
    answer and the equality it is being judged by does not admit it.

    Two cases at least, because one case is a renaming of anything.
    """
    if hypotheses is None or len(pairs) < 2:
        return None
    for one in hypotheses:
        read = getattr(one, "read", None)
        if not callable(read):
            continue
        renaming: dict[Any, Any] = {}
        said_anything = False
        consistent = True
        for before, after in pairs:
            try:
                got = read(before)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
                consistent = False
                break
            if got is None or len(got) != len(after):
                consistent = False
                break
            if tuple(got) == tuple(after):
                # It already passes, so there is nothing for a renaming to
                # explain and this is not the failure being looked for.
                consistent = False
                break
            said_anything = True
            for mine, wanted in zip(got, after):
                if renaming.setdefault(mine, wanted) != wanted:
                    consistent = False
                    break
            if not consistent:
                break
        if consistent and said_anything and len(renaming) > 1:
            return WhyNothingFits(
                A_CHECK_THAT_CANNOT_SEE_IT,
                considered=len(hypotheses),
                together_on=len(pairs),
            )
    return None
