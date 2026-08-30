"""The four walls a language that grows itself runs into.

Growing the language she thinks in is worth doing, and there are exactly four
places where it stops paying, each provable rather than cautionary. Written
down here so a claim about growth has to survive them before it is made.

**A universal language cannot be made more expressive from inside.** If she can
already express every computable function, then any new word she invents is
computable, so its meaning was already there, so E(t+1) = E(t). Growth in what
is expressible has a last day, and it arrives when the language becomes
universal. Her rule language is nowhere near that, which is exactly why
widening it still buys new meanings — and the check says so rather than
assuming it.

**No update rule improves on every environment.** For any rule, take the action
it chooses and build the environment that rewards the other one. Improvement is
therefore always improvement over a named distribution and a named budget, and
a claim without those two is not a claim.

**Representational inadequacy is not decidable in general.** Asking whether any
expression in a universal language has some semantic property reduces to
halting. What is available instead is exhaustive search in a small language,
and residuals, counterexamples and bounded search in a large one — which
answers UNKNOWN where it does not know, rather than False.

**Unrestricted self-modification and complete self-verification cannot both be
had.** Proving that an arbitrary rewrite preserves every property she cares
about is the same undecidable question. What survives is local invariants, a
sandbox, a measured trial and a rollback, and every one of those is a smaller
claim than the proof would have been.

Past all four sits the only thing that would give genuinely more computing
power: an oracle. She cannot build one out of computation, because a machine
that could would decide its own halting problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

__all__ = [
    "BOUNDED",
    "UNIVERSAL",
    "UNKNOWN",
    "HowExpressive",
    "Refutation",
    "can_be_decided",
    "how_expressive",
    "no_updater_wins_everywhere",
    "what_a_new_word_can_buy",
    "what_verification_is_available",
    "what_would_need_an_oracle",
]

logger = logging.getLogger("Aura.WhatGrowthCannotDo")

#: A language with no unbounded repetition and no branching on its own values.
#: Everything it can express is a finite set of total functions, so a word that
#: means something new is still possible.
BOUNDED = "bounded"

#: A language that can express every computable function. Nothing computable
#: she invents adds a meaning to it.
UNIVERSAL = "universal"

#: What an honest bounded search returns where an exhaustive one is impossible.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class HowExpressive:
    """What kind of language this is, and what follows from that."""

    verdict: str
    because: str
    #: How many distinct meanings it has, where that is finite and countable.
    meanings: int | None = None

    def describes(self) -> str:
        counted = f", {self.meanings} meaning(s)" if self.meanings is not None else ""
        return f"{self.verdict}{counted}: {self.because}"


def how_expressive(
    *, repeats_without_bound: bool, branches_on_its_own_values: bool,
    meanings: int | None = None,
) -> HowExpressive:
    """Whether this language is universal, and so whether new words can add meaning.

    Universality needs both unbounded repetition and a decision made on a value
    the language itself produced. A language with neither computes a fixed
    finite family however many words are added to it, which is the good case
    here: it means widening still buys meanings and not only brevity.
    """
    if repeats_without_bound and branches_on_its_own_values:
        return HowExpressive(
            verdict=UNIVERSAL,
            because=(
                "it repeats without bound and decides on values it made, so it "
                "expresses every computable function and a computable word she "
                "invents was expressible already"
            ),
        )
    missing = []
    if not repeats_without_bound:
        missing.append("no unbounded repetition")
    if not branches_on_its_own_values:
        missing.append("no branching on its own values")
    return HowExpressive(
        verdict=BOUNDED,
        because=f"{' and '.join(missing)}, so a word can still mean something new",
        meanings=meanings,
    )


def what_a_new_word_can_buy(language: HowExpressive) -> str:
    """What admitting a word could possibly do, before checking what it did."""
    if language.verdict == UNIVERSAL:
        return (
            "shorter expressions and a smaller search, and nothing newly "
            "expressible, because every computable meaning is already in it"
        )
    return "a meaning it did not have, or a shorter way of saying one it did"


@dataclass(frozen=True)
class Refutation:
    """A counterexample, with the numbers that make it one."""

    holds: bool
    scored: float
    the_other_scored: float
    on: str

    def describes(self) -> str:
        return (
            f"{self.on}: it scored {self.scored:g} where the alternative scored "
            f"{self.the_other_scored:g}"
        )


def no_updater_wins_everywhere(
    updater: Callable[[Any], Any], *, on: Sequence[Any] = (0, 1)
) -> Refutation:
    """Build the environment that beats a given update rule, and run it.

    The diagonal argument, executed rather than cited. Whatever the rule
    chooses, an environment rewarding a different choice exists and is built
    here, so the rule scores nothing on it. Nothing about the rule matters,
    which is the point: no rule improves everywhere, and every improvement
    claim has to name the distribution it improved on.
    """
    choices = tuple(on)
    try:
        chose = updater(choices)
    except (TypeError, ValueError):
        chose = choices[0]
    against = next((choice for choice in choices if choice != chose), chose)

    def worst_case(action: Any) -> float:
        return 1.0 if action == against else 0.0

    scored = worst_case(chose)
    return Refutation(
        holds=scored < worst_case(against),
        scored=scored,
        the_other_scored=worst_case(against),
        on=f"an environment that rewards {against!r} because the rule chose {chose!r}",
    )


def can_be_decided(
    *, language: HowExpressive, exhaustive_search_finished: bool
) -> bool | str:
    """Whether "the language cannot express this" is a thing she may assert.

    True or False where a small language was walked end to end. UNKNOWN where
    it was not, and in a universal language it never can be, because the
    question reduces to halting. Returning UNKNOWN rather than False is what
    keeps a bounded search from reporting absence as proof.
    """
    if language.verdict == UNIVERSAL:
        return UNKNOWN
    if not exhaustive_search_finished:
        return UNKNOWN
    return True


def what_verification_is_available(*, change_is_arbitrary: bool) -> tuple[str, ...]:
    """What can honestly be checked about a change she makes to herself.

    A proof covering every property of an arbitrary rewrite is not available at
    any price. These four are, and they are what a governed change is allowed
    to rest on.
    """
    if not change_is_arbitrary:
        return ("proof over the restricted form", "local invariants", "a measured trial", "rollback")
    return ("local invariants", "a sandbox", "a measured trial", "rollback")


def what_would_need_an_oracle(question: str) -> str:
    """Say plainly that a question sits above computation, rather than trying it.

    The Turing degrees strictly increase, and each one is unreachable from the
    one below by any amount of computing. She runs at the bottom of that
    hierarchy and no self-modification moves her up it, so the answer to a
    question that needs a jump is that she cannot answer it.
    """
    return (
        f"{question} needs an oracle strictly above ordinary computation, and "
        "no amount of self-modification builds one, so this is refused rather "
        "than approximated"
    )
