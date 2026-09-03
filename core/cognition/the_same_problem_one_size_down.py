"""Solving a thing by assuming a smaller one of it is already solved.

The Towers of Hanoi, drawn out: to move n disks, move the top n-1 somewhere
else, move the big one, move the n-1 back. The board says "Move top 3 disks to
rod 2" and does not say how, because how is the same question one size down.

The video puts the whole of it in two lines. Show f(1) works. Assume f(n-1)
works. Nobody traces it, and tracing it is the mistake — three disks is seven
moves, ten disks is a thousand and twenty three, and a person who follows the
moves has understood nothing while a person who accepts the assumption has
understood all of it.

Her walking-back is not this. It asks what would make a far thing near and
answers with a DIFFERENT thing, found by telling situations apart. That is the
right tool when the smaller problem is a different problem. It is the wrong
one, and unaffordable, when the smaller problem is the same problem, because
it re-derives at every size what was already known at the size below.

Three things are needed and no more. A size that cannot be reduced and can be
answered outright. A way of taking one size off. And what to do with the
answers to the smaller ones. Then the plan for any size exists without any
size being walked, and the whole of it is checkable at the bottom — if the
base case is right and one step down is right, every size is right, and that
is a proof rather than a hope.

What it will not do is pretend. Given something that does not get smaller, or
a bottom it never reaches, it says so instead of running until it stops.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["APlanBySize", "solve_by_the_size_below"]


@dataclass(frozen=True)
class APlanBySize:
    """A plan made without walking it, and what is known about it."""

    steps: tuple[Any, ...]
    #: How many sizes down it went before reaching one it could answer.
    deep: int
    settled: bool
    why: str

    def describe(self) -> str:
        if not self.settled:
            return self.why
        return f"{len(self.steps)} step(s), worked out {self.deep} size(s) down"


def solve_by_the_size_below(
    size: Any,
    *,
    smallest: Callable[[Any], bool],
    answer_outright: Callable[[Any], Sequence[Any]],
    one_size_down: Callable[[Any], Sequence[Any]],
    put_together: Callable[[Any, Sequence[Sequence[Any]]], Sequence[Any]],
    deepest: int = 64,
) -> APlanBySize:
    """The plan for this size, from the plan for the size below it.

    ``one_size_down`` hands back the smaller problems this one rests on —
    Hanoi hands back two of them, and something else might hand back one or
    five. ``put_together`` says what to do with their answers, which is where
    the move of the big disk goes.

    ``deepest`` is a bound on how far down it will go before saying it cannot
    get to the bottom. Not a budget for effort: a problem that does not shrink
    is a problem this cannot solve, and running until something stops it is
    how that gets mistaken for a hard one.
    """
    went: dict[str, int] = {"deep": 0}

    def work_out(here: Any, down: int) -> Sequence[Any] | None:
        went["deep"] = max(went["deep"], down)
        if down > deepest:
            return None
        try:
            if smallest(here):
                return list(answer_outright(here))
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            return None
        try:
            below = list(one_size_down(here))
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            return None
        if not below:
            return None
        answers: list[Sequence[Any]] = []
        for one in below:
            got = work_out(one, down + 1)
            if got is None:
                return None
            answers.append(got)
        try:
            return list(put_together(here, answers))
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            return None

    steps = work_out(size, 0)
    if steps is None:
        return APlanBySize(
            steps=(),
            deep=went["deep"],
            settled=False,
            why=(
                "it does not get smaller, or the bottom is further down than "
                f"{deepest}"
            ),
        )
    return APlanBySize(
        steps=tuple(steps), deep=went["deep"], settled=True, why="worked out"
    )
