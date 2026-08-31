"""Choosing how a thing is going to end before choosing what to do in it.

Asked about the checkers game afterwards:

    my strategy both times was to get the king, have all of my pieces back,
    and force a forfeit. even at the sacrifice of other pieces, no it isnt
    necessarily the best checkers strategy but again, you can see my logic

The admission is the interesting part. They knew it was not the strongest way
to play and played it anyway, because it was the one they could bring about
against that opponent. That is a decision taken before any move, and every
move afterwards was valued by it — pieces became spendable, a crowned piece
became the one thing that could not be spent, and a position losing by three
was winning.

Most things that play a game do not make this decision. They score a position
and take the best number, which quietly fixes the ending as whichever one the
score was built around. Something that can choose its ending can play for the
one it can actually reach, and against a stronger opponent that is usually a
different one.

Choosing is a measurement she already knows how to make. For each way it could
end, ask which property of the places she passed through predicts ending that
way. A way to win with no such property is one she has no route to, however
good it sounds. A way with a strong one is a way she can steer toward, and the
property is what to hold while she does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.something_she_keeps_true import SomethingTrue, how_well_it_predicts

__all__ = ["AWayToWin", "which_way_to_win"]


@dataclass(frozen=True)
class AWayToWin:
    """One way it could end, and whether she has a route to it."""

    name: str
    by_holding: SomethingTrue | None
    ended_this_way: int
    endings: int

    @property
    def how_often(self) -> float:
        return self.ended_this_way / self.endings if self.endings else 0.0

    @property
    def can_steer_to_it(self) -> bool:
        """Whether anything about the places she passes through predicts it."""
        return self.by_holding is not None and self.by_holding.tells_them_apart > 0

    def describe(self) -> str:
        if not self.can_steer_to_it:
            return f"{self.name}: happens, but nothing she does makes it happen"
        assert self.by_holding is not None
        return (
            f"{self.name}: by holding {self.by_holding.name} "
            f"({self.by_holding.tells_them_apart:+.2f})"
        )


def which_way_to_win(
    ways: dict[str, Callable[[Any], bool]],
    runs: Sequence[tuple[Sequence[Any], Any]],
    *,
    deepest: int = 2,
) -> list[AWayToWin]:
    """The endings she has a route to, the most steerable first.

    ``runs`` is what happened: the places she passed through, and how each one
    finished. ``ways`` are the endings worth telling apart, asked of the
    finish. An ending nothing predicts sorts last rather than being dropped,
    because "this keeps happening and nothing I do changes it" is worth
    knowing and is not the same as "this never happens".

    Ordered by how well the property tells them apart before how often the
    ending came up. A rare ending she can steer to beats a common one she
    cannot, which is the whole of what choosing a way to win is.
    """
    weighed: list[AWayToWin] = []
    endings = len(runs)
    for name, ended in ways.items():
        watched: list[tuple[Any, bool]] = []
        counted = 0
        for places, finish in runs:
            this_way = bool(ended(finish))
            counted += this_way
            for place in places:
                watched.append((place, this_way))
        found = how_well_it_predicts(watched, deepest=deepest) if watched else []
        best = found[0] if found and found[0].tells_them_apart > 0 else None
        weighed.append(
            AWayToWin(
                name=name,
                by_holding=best,
                ended_this_way=counted,
                endings=endings,
            )
        )
    weighed.sort(
        key=lambda one: (
            not one.can_steer_to_it,
            -(one.by_holding.tells_them_apart if one.by_holding else 0.0),
            -one.how_often,
        )
    )
    return weighed
