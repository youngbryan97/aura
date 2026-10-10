"""How far her blow reaches: a key that sends nothing out strikes what is close, and how close is learned.

A punch, a kick, a sword's swing, a yo-yo on its string. A person told "S to
punch" or "Z to attack close enemies" does not stand back and press: nothing
flies. They press when what they are to hit is in front of them.

So a key the words give to hitting that has brought nothing out beside her
(core/agency/which_one_answers_to_her.py ``makes``) is a blow, and she strikes
with it when something is close. Each press is kept with the things near her
then, each with the gap between it and her in her own sizes. A press followed
soon after by a gain paid, and the nearest thing then was within reach. Her
reach is how far the paying presses reached; before any has paid it is an
arm's length, and she presses at a little more than that to find out.

A blow is not pressed at what pays on touch: that is to be met, not hit.
Nothing here knows a punch.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.agency.what_meeting_things_does import CLICK, IGNORE, MEET

__all__ = ["ARM", "BLOW_EVERY_S", "HerBlows", "the_gap"]

#: The gap a blow reaches across before any press has paid, in her own sizes: an arm's length.
ARM = 0.75
#: How much further than an arm she strikes while nothing has paid yet, so that a longer reach is found.
FINDING = 1.6
#: How soon after a press a gain must be read to be put down to it, in seconds.
PAID_WITHIN_S = 1.0
#: How often one key is struck, at most, in seconds.
BLOW_EVERY_S = 0.3
#: The most a learned reach may be, in her own sizes: further than this is a throw, not a blow.
LONGEST = 4.0
#: The least: a press that paid with the gap closed still reaches across a sliver.
SHORTEST = 0.25
#: Presses with something near that a blow which has never paid is given before she stops standing her ground for it.
TRIED = 8


def the_gap(mine: Any, thing: Any) -> float:
    """The gap between two boxes, edge to edge, in the first one's own sizes: 0 where they touch or overlap."""
    size = max(float(mine.w), float(mine.h), 1.0)
    dx = max(0.0, abs(float(thing.x) - float(mine.x)) - (float(thing.w) + float(mine.w)) / 2)
    dy = max(0.0, abs(float(thing.y) - float(mine.y)) - (float(thing.h) + float(mine.h)) / 2)
    return math.hypot(dx, dy) / size


@dataclass
class _Press:
    key: str
    at: float
    nearest: float


@dataclass
class HerBlows:
    """Each press of a key that sends nothing out, what was close when it was pressed, and how far it has paid."""

    presses: list[_Press] = field(default_factory=list)
    paid: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    last: dict[str, float] = field(default_factory=dict)

    def keys(self, fire_keys: tuple[str, ...] | list[str], makes: Any) -> list[str]:
        """The keys given to hitting that have brought nothing out beside her: struck, not fired."""
        return [key for key in fire_keys if key not in (makes or {}) and key != "mouse"]

    def reach(self, key: str) -> float:
        """How far a blow by this key reaches, edge to edge, in her own sizes."""
        paid = self.paid.get(key)
        if not paid:
            return ARM * FINDING
        return min(LONGEST, max(SHORTEST, max(paid) * 1.15))

    def stands_ground(self, keys: list[str]) -> float | None:
        """How far she can strike, in her own sizes, by a blow worth standing her ground for: one that has paid, or one
        still being tried. None where there is none, and she keeps clear of what comes at her as before."""
        reaches = [self.reach(key) for key in keys
                   if self.paid.get(key) or sum(1 for p in self.presses if p.key == key and p.nearest <= LONGEST) < TRIED]
        return max(reaches, default=None)

    def to_strike(self, key: str, mine: Any, things: Any, stance: Any, at: float) -> Any:
        """The nearest thing within this key's reach that is to be struck, if it is time to strike again; else None.

        ``stance`` gives what to do about a thing's kind. What pays on touch is met, what is to be clicked is clicked,
        and what does not matter is left alone: anything else that moves and comes close is struck.
        """
        if mine is None or at - self.last.get(key, -math.inf) < BLOW_EVERY_S:
            return None
        reach = self.reach(key)
        best, nearest = None, math.inf
        for thing in things:
            if thing is mine or getattr(thing, "number", None) == getattr(mine, "number", None):
                continue
            if not getattr(thing, "moved", True) or stance(thing) in (MEET, CLICK, IGNORE):
                continue
            gap = the_gap(mine, thing)
            if gap <= reach and gap < nearest:
                best, nearest = thing, gap
        return best

    def pressed(self, key: str, at: float, mine: Any, things: Any) -> None:
        """A press of a blow, with how near the nearest other thing was."""
        self.last[key] = at
        if mine is None:
            return
        gaps = [the_gap(mine, thing) for thing in things
                if thing is not mine and getattr(thing, "number", None) != getattr(mine, "number", None)]
        self.presses.append(_Press(key, at, min(gaps, default=math.inf)))
        del self.presses[:-40]

    def gained(self, at: float) -> None:
        """A gain read: the latest press soon before it paid, at the gap its nearest thing stood."""
        for press in reversed(self.presses):
            if 0.0 <= at - press.at <= PAID_WITHIN_S:
                if math.isfinite(press.nearest) and press.nearest <= LONGEST:
                    self.paid[press.key].append(press.nearest)
                    del self.paid[press.key][:-20]
                return
            if press.at < at - PAID_WITHIN_S:
                return
