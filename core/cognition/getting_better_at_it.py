"""Whether keeping what she works out makes her better at the next thing.

Every piece needed for this exists and none of them has been asked to run
together: she can work out a word, invent a kind of thing, write a way of
building words, compose an action out of the ones she has, tell a search that
went badly from a language that cannot say it, notice a quantity she is not
reading, choose the one experiment worth doing, and drop what stopped paying.

U8 is the question those were all for. Not "can she learn something" — each of
them answers that on its own — but whether learning ACCUMULATES: does the
hundredth problem go better because of the first ninety-nine, in a domain the
first ninety-nine were not about.

Three things make that a measurement rather than a story.

It is scored on problems she has not seen. A run that reports how well she does
on what she just learned from reports that she can remember.

It is scored against a null: the same problems, in the same order, with
everything she works out thrown away after each one. Without that, a rising
score says the problems got easier, and there is no way to tell the difference
afterwards.

And the domains share nothing. Sequences of numbers and states of a world have
no vocabulary in common, so a gain in the second cannot come from having been
shown the first — only from something she built being general.

Both what she got right and what it cost her are recorded, because the first
alone stops discriminating the moment the problems are easy enough to solve
from scratch. A first run of ten scored ten out of ten either way, which says
nothing about accumulation and everything about the problems. Where the score
is at a ceiling the seconds still separate the two runs, and where the budget
binds the score separates them again — the same fact reaching the measurement
by whichever route is open.

An arm that fails fast looks cheap. Keeping solved two more problems than
forgetting and took 136ms against 9ms, because solving costs and giving up does
not — so the totals said keeping was fifteen times slower, which is true and
answers nothing. Seconds are only comparable over the problems BOTH arms got
right; anywhere else the comparison is between working and stopping.

A difference one sample cannot support is not a difference. Both runs of that
first ten came in under a millisecond, and the clock's own resolution — the
first floor tried here — is nanoseconds, so it admitted the noise and reported
"sooner at sequences" where the two runs were identical to the eye. One sample
has no spread, so it can carry no timing claim at all: the run is repeated, and
a difference must be larger than the spread of the runs themselves before it is
said out loud. With one repetition nothing about seconds is claimed, which is
the honest thing a single run can say.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "HowItWent",
    "Problem",
    "a_run_of_problems",
    "did_keeping_it_help",
]

logger = logging.getLogger("Aura.GettingBetterAtIt")


def _wider_than_the_runs_vary(mine: Sequence[float], theirs: Sequence[float]) -> bool:
    """Whether one arm was faster by more than either arm varies.

    With fewer than two repetitions of each there is no spread to compare
    against, and the answer is no — not because it was slow, but because one
    sample cannot say.
    """
    if len(mine) < 2 or len(theirs) < 2:
        return False
    apart = (sum(theirs) / len(theirs)) - (sum(mine) / len(mine))
    spread = max(max(mine) - min(mine), max(theirs) - min(theirs))
    return apart > spread


@dataclass(frozen=True)
class Problem:
    """One thing to work out, and how to tell whether she did."""

    #: Which world it belongs to. Two problems of different kinds share no
    #: vocabulary, which is what makes carrying between them mean something.
    kind: str
    name: str
    #: What she is shown.
    shown: Any
    #: What she must get right, having seen only ``shown``.
    held_back: Any


@dataclass
class HowItWent:
    """What a run of problems came to."""

    solved: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    #: What she added while working, in the order she added it.
    added: list[str] = field(default_factory=list)
    #: Whether each problem in order was solved, for reading the shape.
    in_order: list[bool] = field(default_factory=list)
    by_kind: dict[str, list[bool]] = field(default_factory=dict)
    #: Seconds each problem took, in order, with its name and kind beside it.
    took: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    cost_by_kind: dict[str, list[float]] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        total = len(self.solved) + len(self.missed)
        return len(self.solved) / total if total else 0.0

    @property
    def cost(self) -> float:
        return sum(self.took)

    def rate_for(self, kind: str) -> float:
        got = self.by_kind.get(kind) or []
        return sum(1 for one in got if one) / len(got) if got else 0.0

    def cost_for(self, kind: str) -> float:
        return sum(self.cost_by_kind.get(kind) or [])

    def cost_of(self, names: Any, kind: str = "") -> float:
        """What these particular problems cost, by name.

        The only comparable total between two arms that solved different
        things.
        """
        wanted = set(names)
        return sum(
            spent
            for named, spent, mine in zip(self.names, self.took, self.kinds)
            if named in wanted and (not kind or mine == kind)
        )

    def __str__(self) -> str:
        kinds = ", ".join(
            f"{kind} {self.rate_for(kind):.0%} in {self.cost_for(kind):.1f}s"
            for kind in sorted(self.by_kind)
        )
        return (
            f"{len(self.solved)}/{len(self.solved) + len(self.missed)} "
            f"({self.rate:.0%}) in {self.cost:.1f}s; {kinds}; kept {len(self.added)}"
        )


def a_run_of_problems(
    problems: Sequence[Problem],
    *,
    solve: Callable[[Problem], bool],
    keeping: bool,
    forget: Callable[[], None],
    what_she_has: Callable[[], list[str]],
) -> HowItWent:
    """Work through them in order, keeping what she works out or throwing it away.

    ``forget`` is what makes the null a null. It runs after every problem, so
    the run has no memory at all and each problem meets the same beginner.
    """
    went = HowItWent()
    before = set(what_she_has())
    for problem in problems:
        began = time.monotonic()
        try:
            got = bool(solve(problem))
        except (ArithmeticError, KeyError, RecursionError, TypeError, ValueError):
            logger.debug("problem %r raised", problem.name, exc_info=True)
            got = False
        spent = time.monotonic() - began
        (went.solved if got else went.missed).append(problem.name)
        went.in_order.append(got)
        went.took.append(spent)
        went.names.append(problem.name)
        went.kinds.append(problem.kind)
        went.by_kind.setdefault(problem.kind, []).append(got)
        went.cost_by_kind.setdefault(problem.kind, []).append(spent)
        if keeping:
            now = set(what_she_has())
            went.added.extend(sorted(now - before))
            before = now
        else:
            forget()
            before = set(what_she_has())
    return went


@dataclass(frozen=True)
class Verdict:
    """Whether keeping it helped, and by how much, against the null."""

    keeping: HowItWent
    forgetting: HowItWent
    #: Kinds where keeping did better, worse, or the same.
    better_at: tuple[str, ...] = ()
    worse_at: tuple[str, ...] = ()
    #: Kinds it reached sooner, whether or not the score could move. Empty
    #: unless the run was repeated, because one sample has no spread.
    sooner_at: tuple[str, ...] = ()
    #: What each repetition of each arm cost, for weighing "sooner" against
    #: how much the runs vary on their own.
    kept_costs: tuple[float, ...] = ()
    forgot_costs: tuple[float, ...] = ()

    @property
    def helped(self) -> bool:
        """More right, or as many right for visibly less.

        A ceiling on the score is a fact about the problems, and refusing to
        read the clock there is refusing to measure. Reading a difference the
        clock cannot resolve is worse than refusing.
        """
        if self.keeping.rate != self.forgetting.rate:
            return self.keeping.rate > self.forgetting.rate
        shared = self.both_solved
        return _wider_than_the_runs_vary(
            [self.keeping.cost_of(shared)], [self.forgetting.cost_of(shared)]
        )

    @property
    def both_solved(self) -> frozenset[str]:
        """The problems both arms got right, which is where seconds compare."""
        return frozenset(self.keeping.solved) & frozenset(self.forgetting.solved)

    @property
    def carried(self) -> bool:
        """Whether a gain appeared in more than one kind of problem.

        One kind improving is a better search in that kind. Two kinds sharing
        no vocabulary both improving is something general having been built.
        """
        return len(set(self.better_at) | set(self.sooner_at)) > 1

    def __str__(self) -> str:
        change = self.keeping.rate - self.forgetting.rate
        said = (
            f"keeping {self.keeping.rate:.0%} against forgetting "
            f"{self.forgetting.rate:.0%} ({change:+.0%})"
        )
        shared = self.both_solved
        said = (
            f"{said}; on the {len(shared)} both solved, "
            f"{self.keeping.cost_of(shared):.3f}s against "
            f"{self.forgetting.cost_of(shared):.3f}s"
        )
        if self.better_at:
            said = f"{said}; better at {', '.join(self.better_at)}"
        if self.sooner_at:
            said = f"{said}; sooner at {', '.join(self.sooner_at)}"
        if self.worse_at:
            said = f"{said}; worse at {', '.join(self.worse_at)}"
        return said


def did_keeping_it_help(
    problems: Sequence[Problem],
    *,
    solve: Callable[[Problem], bool],
    forget: Callable[[], None],
    what_she_has: Callable[[], list[str]],
    times: int = 1,
) -> Verdict:
    """Run it both ways — keeping what she works out, and throwing it away.

    The same problems in the same order every time. Any difference is the
    keeping, because nothing else differs. ``times`` repeats both arms, which
    is what makes any statement about seconds sayable.
    """
    kept_runs: list[HowItWent] = []
    forgot_runs: list[HowItWent] = []
    for _again in range(max(1, int(times))):
        forget()
        forgot_runs.append(
            a_run_of_problems(
                problems,
                solve=solve,
                keeping=False,
                forget=forget,
                what_she_has=what_she_has,
            )
        )
        forget()
        kept_runs.append(
            a_run_of_problems(
                problems,
                solve=solve,
                keeping=True,
                forget=forget,
                what_she_has=what_she_has,
            )
        )
    keeping, forgetting = kept_runs[-1], forgot_runs[-1]
    better, worse, sooner = [], [], []
    for kind in sorted(keeping.by_kind):
        mine, theirs = keeping.rate_for(kind), forgetting.rate_for(kind)
        if mine > theirs:
            better.append(kind)
        elif mine < theirs:
            worse.append(kind)
        # Only over what both arms got right, or this compares working with
        # giving up.
        shared = frozenset(keeping.solved) & frozenset(forgetting.solved)
        if _wider_than_the_runs_vary(
            [one.cost_of(shared, kind) for one in kept_runs],
            [one.cost_of(shared, kind) for one in forgot_runs],
        ):
            sooner.append(kind)
    found = Verdict(
        keeping=keeping,
        forgetting=forgetting,
        better_at=tuple(better),
        worse_at=tuple(worse),
        sooner_at=tuple(sooner),
        kept_costs=tuple(one.cost for one in kept_runs),
        forgot_costs=tuple(one.cost for one in forgot_runs),
    )
    logger.info("did keeping it help: %s", found)
    return found
