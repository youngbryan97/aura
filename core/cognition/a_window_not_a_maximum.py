"""Wanting something in a band, where more is worse past a point.

Catching a Pokémon is the clearest small lesson in any of these games. You
have to hurt it, because a healthy one will not be caught. You must not hurt
it too much, because a dead one cannot be caught at all. The goal is not to
maximise damage and it is not to minimise it — it is to land in a window, and
every act she has is judged by how far it moves you INTO that window rather
than how big it is.

Almost everything she does is scored the other way. More of what is wanted is
better, right up until the thing is achieved. That is true of some goals and
quietly false of a great many: a bid that must beat one number and stay under
another, a message that has to be long enough to say the thing and short
enough to be read, a load that must be high enough to be worth running and low
enough to survive, a temperature, a dose, a deadline that can be missed by
being early.

The failure is not that she overshoots by a little. It is that overshooting
looks like success right up to the moment it is a disaster, because the
measure that says "more" says "more" all the way past the edge — and by then
the thing she wanted is gone rather than merely not got.

What is wanted here is small. Say what the window is. Score by distance into
it rather than by size. Notice when an act would carry her past the far edge,
which is a different kind of mistake from falling short and needs saying
differently: falling short leaves the thing there to try again on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["AWindow", "how_close_to_it", "which_act_lands_in_it"]


@dataclass(frozen=True)
class AWindow:
    """A band she is trying to land in."""

    at_least: float
    at_most: float

    def holds(self, value: float) -> bool:
        return self.at_least <= float(value) <= self.at_most

    def overshot(self, value: float) -> bool:
        """Past the far edge, which is the mistake that cannot be tried again."""
        return float(value) > self.at_most

    def short_of(self, value: float) -> bool:
        return float(value) < self.at_least

    @property
    def middle(self) -> float:
        return (self.at_least + self.at_most) / 2.0

    def describe(self, value: float) -> str:
        if self.holds(value):
            return f"{value:.2f} is in it"
        if self.overshot(value):
            return f"{value:.2f} is past it — gone rather than not got"
        return f"{value:.2f} is short of it, and it is still there"


def how_close_to_it(value: float, window: AWindow) -> float:
    """How near this is to being in the window, one being in it.

    Distance in rather than size. The same number that says a thing is good
    has to stop saying so past the far edge, or she walks through it while
    being told she is doing well.
    """
    got = float(value)
    if window.holds(got):
        return 1.0
    width = max(1e-9, window.at_most - window.at_least)
    if window.short_of(got):
        return max(0.0, 1.0 - (window.at_least - got) / width)
    return max(0.0, 1.0 - (got - window.at_most) / width)


def which_act_lands_in_it(
    acts: Sequence[str],
    *,
    now: float,
    what_it_moves: Callable[[str], float],
    window: AWindow,
) -> list[tuple[str, float, bool]]:
    """Her acts, nearest to landing in the window first.

    The third value says whether the act would carry her past the far edge.
    Kept separate rather than folded into the score, because overshooting and
    falling short are not the same mistake and she should be able to refuse
    one of them outright.
    """
    weighed = []
    for one in acts:
        lands = float(now) + float(what_it_moves(one))
        weighed.append((one, how_close_to_it(lands, window), window.overshot(lands)))
    return sorted(weighed, key=lambda one: (one[2], -one[1], one[0]))
