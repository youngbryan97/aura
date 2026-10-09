"""How far a thing reaches: the distance, and the side, at which being near it costs her without its touching her.

A guard sees her across a room, ahead of where it faces; an enemy fires from a
distance; a mine goes off as she comes close; a hot stove burns from a hand
away. Keeping clear of what touches her is not enough with any of these: what
costs is coming within its reach.

So a loss with nothing touching her is put down to the things near her then,
each with where she stood from it: how far, in its own sizes, and on which side
of the way it was going. A kind that has cost her this way more than once has a
reach: the distance within which it did, and whether it did only ahead of it.
Her danger counts a thing as met when she comes within its reach, on that side.

Nothing here knows what a guard, a gun or a mine is.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HowFarThingsReach"]

#: How near, in her own sizes, a thing must be to be suspected of a loss that nothing touching her explains.
NEAR = 8.0
#: Losses a kind must have cost from a distance before it is taken to reach.
TWICE = 2
#: How nearly ahead of a thing's way she must have stood, at most of the losses, for its reach to be ahead only: within
#: about 60°. Most, not every: where a loss is read a moment late, she has moved on by then.
AHEAD = 0.5
#: The share of her losses from a distance a kind must have been near at to be taken to reach: what costs is there each
#: time, and a thing that happened to be about (a coin she was going for) is there only some of the times.
MOSTLY = 0.75


@dataclass
class _Reach:
    distances: list[float] = field(default_factory=list)
    aheads: list[float] = field(default_factory=list)
    losses: int = 0


class HowFarThingsReach:
    """What each kind of thing has cost her from a distance, and how far and on which side it reaches."""

    def __init__(self) -> None:
        self.kinds: dict[int, _Reach] = defaultdict(_Reach)
        self.losses = 0

    def lost_untouched(self, mine: Any, things: Any) -> None:
        """A loss with nothing touching her: the things near her are suspected, each with where she stood from it."""
        if mine is None:
            return
        self.losses += 1
        size = max(float(mine.w), float(mine.h), 1.0)
        suspected: set[int] = set()
        for thing in things:
            if thing is mine or getattr(thing, "number", None) == getattr(mine, "number", None):
                continue
            apart = math.hypot(float(mine.x) - float(thing.x), float(mine.y) - float(thing.y))
            if apart > NEAR * size:
                continue
            reach = self.kinds[int(thing.kind)]
            if int(thing.kind) not in suspected:
                suspected.add(int(thing.kind))
                reach.losses += 1
            reach.distances.append(apart / max(float(thing.w), float(thing.h), 1.0))
            going = math.hypot(float(thing.vx), float(thing.vy))
            if going > 0:
                reach.aheads.append(((float(mine.x) - float(thing.x)) * float(thing.vx)
                                     + (float(mine.y) - float(thing.y)) * float(thing.vy)) / (going * max(apart, 1e-6)))

    def reach(self, kind: int) -> tuple[float, bool] | None:
        """How far a kind reaches, in its own sizes, and whether only ahead of its way; None where it has not been seen to."""
        found = self.kinds.get(kind)
        if found is None or found.losses < TWICE or found.losses < MOSTLY * self.losses:
            return None
        far = statistics.median(found.distances) * 1.25
        ahead = len(found.aheads) >= TWICE and sum(1 for cos in found.aheads if cos >= AHEAD) >= MOSTLY * len(found.aheads)
        return far, ahead

    def within(self, thing: Any, at_x: float, at_y: float, there: tuple[float, float]) -> bool:
        """Whether a point is within a thing's reach, the thing standing at ``there``."""
        reach = self.reach(int(thing.kind))
        if reach is None:
            return False
        far, ahead = reach
        size = max(float(thing.w), float(thing.h), 1.0)
        dx, dy = at_x - there[0], at_y - there[1]
        apart = math.hypot(dx, dy)
        if apart > far * size:
            return False
        going = math.hypot(float(thing.vx), float(thing.vy))
        return not ahead or going == 0 or (dx * float(thing.vx) + dy * float(thing.vy)) / (going * max(apart, 1e-6)) >= AHEAD
