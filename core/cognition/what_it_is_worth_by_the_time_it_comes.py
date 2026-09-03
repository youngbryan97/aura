"""Something later against something now, without a rate anybody chose.

Two things from two recordings that turn out to be one thing.

A Stellaris player, paused, choosing between agendas — and the one they take
says it pays after ten years. Ten years of a game they might not be in. Go
players do the same and have two words for it: territory is points that are
already theirs, influence is a position that will become points if the game
goes the way it looks like going. Strong players take influence early and
territory late, and never the other way round.

The usual way to weigh later against now is a rate somebody picked. Pick it
high and she never builds anything; pick it low and she starves while
investing. It is a knob, it is wrong in every world it was not tuned for, and
there is nothing to tune it against.

There is a real number underneath it. A payoff that arrives in ten years is
worth what it pays TIMES THE CHANCE SHE IS STILL THERE in ten years, and that
is not a preference, it is a fact about how her runs have gone. She has ended
runs. She can count how many lasted that long. So a thing that pays later is
discounted by exactly how often the world has let her get that far, and the
same investment is correctly worth more in a stable world and less in a
precarious one — which is what the word "precarious" means.

Early in a run, when much is still ahead, the far payoff wins. Late, when
little is, the near one does. Nobody has to tell her which phase she is in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HowFarSheUsuallyGets", "WhatItComesTo", "what_it_comes_to"]


@dataclass
class HowFarSheUsuallyGets:
    """How long her runs in this world have lasted."""

    lasted: list[float] = field(default_factory=list)

    def a_run_lasted(self, how_long: float) -> None:
        if how_long > 0:
            self.lasted.append(float(how_long))

    def still_here_in(self, how_long: float) -> float:
        """The share of her runs that got at least this far.

        Laplace, so that one long run is not a promise and nothing seen is not
        a refusal. With no runs at all it is a half — which correctly makes
        her neither a builder nor a hoarder until she has been somewhere.
        """
        if how_long <= 0:
            return 1.0
        got = sum(1 for one in self.lasted if one >= how_long)
        return (got + 1) / (len(self.lasted) + 2)

    def describe(self) -> str:
        if not self.lasted:
            return "she has not finished a run here yet"
        middle = sorted(self.lasted)[len(self.lasted) // 2]
        return f"{len(self.lasted)} run(s) here, the middling one lasting {middle:.0f}"


@dataclass(frozen=True)
class WhatItComesTo:
    """One thing she could do, and what it actually comes to."""

    name: str
    pays: float
    arrives_in: float
    #: How likely she is to still be there when it does.
    still_here: float

    @property
    def comes_to(self) -> float:
        return self.pays * self.still_here

    def describe(self) -> str:
        if self.arrives_in <= 0:
            return f"{self.name}: {self.pays:.2f}, already hers"
        return (
            f"{self.name}: {self.pays:.2f} in {self.arrives_in:.0f}, "
            f"{self.still_here:.0%} likely to see it, so {self.comes_to:.2f}"
        )


def what_it_comes_to(
    options: Mapping[str, tuple[float, float]],
    usually: HowFarSheUsuallyGets,
) -> list[WhatItComesTo]:
    """Each option as what it pays and when, ordered by what it comes to.

    ``options`` maps a name to (what it pays, how long until it does). A thing
    that pays now has nothing taken off it; a thing that pays later has taken
    off it exactly how often the world has let her get that far.
    """
    weighed = [
        WhatItComesTo(
            name=str(name),
            pays=float(pays),
            arrives_in=float(when),
            still_here=usually.still_here_in(float(when)),
        )
        for name, (pays, when) in options.items()
    ]
    return sorted(weighed, key=lambda one: (-one.comes_to, one.arrives_in, one.name))


def already_hers(things: Sequence[Any], is_settled: Any) -> tuple[float, float]:
    """What is settled and what is still only likely, kept apart.

    Territory and influence. Adding them up loses the difference, and the
    difference is the whole of what the late game is about — a lead made of
    things that might become points is not a lead when there is no time left
    for them to become anything.
    """
    settled = sum(1.0 for one in things if is_settled(one))
    return settled, float(len(things)) - settled
