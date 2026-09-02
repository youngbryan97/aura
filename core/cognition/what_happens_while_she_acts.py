"""Acts that take time, and what the world does during them.

Watching somebody play Ghosts 'n Goblins: a jump cannot be called off. Once it
starts, the knight goes where he was going, and for about a second he cannot
answer anything — so the decision to jump is not a decision about where to be,
it is a decision to be unable to respond for a second, taken while knowing
what is coming. Good players jump much less than bad ones, and that is why.

Every act she takes has this shape and none of her weighing has it. An act is
treated as a thing that happens and then is over, so pressing a key and
starting a build that runs for five minutes are the same kind of move, and the
only difference — that one of them costs her five minutes of being unable to
react — does not appear anywhere.

What it costs is not the time. It is how much the world gets to do that she
cannot answer, and those come apart: five minutes in a world that changes
hourly costs nothing, and one second in a world that deals a card every half
second costs two cards. So it is measured as one against the other, and both
sides of it are hers to measure rather than to be told.

She times her own acts by doing them. How fast the world moves on its own she
already keeps, because she has to tell what she did from what merely happened.
The exposure of an act is the second divided into the first, and it is the
part of a decision nothing was looking at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["WhatItCostsToBeBusy", "how_exposed_each_act_is"]


@dataclass
class WhatItCostsToBeBusy:
    """How long each of her acts takes, and how fast the world moves."""

    #: How long each act has taken, added up, and how many times.
    took: dict[str, float] = field(default_factory=dict)
    times: dict[str, int] = field(default_factory=dict)
    #: How long the world has been watched, and how often it did something of
    #: its own in that time.
    watched_for: float = 0.0
    it_moved: int = 0

    def an_act_took(self, act: str, seconds: float) -> None:
        """One act, and how long she could not do anything else for."""
        if seconds <= 0:
            return
        self.took[act] = self.took.get(act, 0.0) + float(seconds)
        self.times[act] = self.times.get(act, 0) + 1

    def the_world_moved(self, seconds: float, times: int = 1) -> None:
        """A stretch of time, and how often the world did something in it."""
        if seconds <= 0:
            return
        self.watched_for += float(seconds)
        self.it_moved += max(0, int(times))

    def how_long(self, act: str) -> float:
        """How long that act takes, or nothing where she has not done it."""
        many = self.times.get(act, 0)
        return (self.took.get(act, 0.0) / many) if many else 0.0

    @property
    def how_fast_the_world_is(self) -> float:
        """How many things the world does on its own in a second."""
        return (self.it_moved / self.watched_for) if self.watched_for > 0 else 0.0

    def exposure(self, act: str) -> float:
        """How much the world gets to do during that act that she cannot answer.

        Nought where she has not timed the act, or where the world has never
        been seen to do anything by itself — and nought is the honest answer
        in both, because there is nothing yet to be exposed to.
        """
        return self.how_long(act) * self.how_fast_the_world_is

    def worth_thinking_about(self, acts: Sequence[str]) -> bool:
        """Whether being busy costs enough here to be worth weighing at all.

        Where every act exposes her to less than one thing the world does,
        nothing is being missed and this is noise on the decision. Said out
        loud so that a weighing which cannot matter is not made anyway.
        """
        return any(self.exposure(one) >= 1.0 for one in acts)

    def describe(self, acts: Sequence[str]) -> str:
        if not self.worth_thinking_about(acts):
            return "nothing she does takes long enough here to miss anything"
        worst = max(acts, key=self.exposure)
        return (
            f"{worst} leaves {self.exposure(worst):.1f} of the world's own moves "
            f"unanswered ({self.how_long(worst):.2f}s at "
            f"{self.how_fast_the_world_is:.2f} a second)"
        )


def how_exposed_each_act_is(
    acts: Sequence[str],
    busy: WhatItCostsToBeBusy,
    *,
    worth: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Her acts, best first, once being unable to answer is counted.

    ``worth`` is what each act is worth on every other ground she has. The
    exposure is taken off it, because a thing the world does while she is busy
    is a thing that happens to her rather than one she chose — which is the
    whole difference between a move and a gamble.

    Without ``worth`` this ranks by exposure alone, least first, which is what
    to do when nothing else separates them.
    """
    if worth is None:
        return sorted(((one, busy.exposure(one)) for one in acts), key=lambda o: (o[1], o[0]))
    return sorted(
        ((one, float(worth.get(one, 0.0)) - busy.exposure(one)) for one in acts),
        key=lambda one: (-one[1], one[0]),
    )
