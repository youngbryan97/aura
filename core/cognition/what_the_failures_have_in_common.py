"""Telling a search that went badly from a language that cannot say it.

These want opposite responses and look identical from inside a loop. Nothing
fits, so: try harder, or change what trying means?

    a search failure          the answer is sayable and she did not find it.
                              More looking, a better order, a longer budget.
    a representation failure  every hypothesis her language admits fails, and
                              they fail on the SAME cases. No amount of
                              looking helps, because the thing is not in the
                              space being looked through.

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

    @property
    def is_the_language(self) -> bool:
        return self.because == A_REPRESENTATION_FAILURE

    def describes(self) -> str:
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
    )
    logger.info("why nothing fits — %s", found.describes())
    return found
