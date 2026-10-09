"""When a press pays: where the moving things were when one did, and pressing when one comes there again.

Some worlds give her no body to steer. A needle swings across a bar and a press
counts only while it is over the right part; a hook swings and a press drops
what it holds; a meter rises and falls and a press sets the power; a beat comes
down a lane and a press must meet it. What is asked is not where to go but
when to press, and the when is a place: where the thing that keeps moving is.

So every press is put down to where each thing that keeps moving was at that
moment, a place along the way it goes, and what her counters did before the next
press is credited to those places. She presses when the thing is about to be at
the place most worth it, counting what it has paid and how little it has been
tried (an upper-confidence choice, as with where clicks pay,
core/agency/where_clicks_pay.py). Her own delay between deciding and the press
landing is allowed for: she presses for where the thing will be.

Nothing here knows what the thing is or what a press does to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhenAPressPays"]

#: The places along a thing's way a press is put down to. A place is judged with its neighbours at half weight: what
#: pays is a stretch of the way, and a stretch seldom starts where a place does (offline 2026-10-09, a green stretch an
#: eighth of a needle's swing, cut across two of eight places, left neither of them plainly worth it).
PLACES = 12
NEIGHBOURS = 0.5
#: How much less tried a place may be and still come before one that paid: the weight of the unknown, against a
#: payoff that runs from a life lost to a point won.
TRYING = 1.0
#: How far a thing must have gone along its way, in its own sizes, for its way to be cut into places.
GOES_FAR = 3.0
#: Presses at a place before what it paid says which thing to time against.
PROVEN = 2
#: Spacings without a press after which she times against the thing going furthest: the thing chosen for what it
#: paid may no longer come to where it paid.
STALLED = 4
#: The least time between two presses, in seconds: the counters' answer to one is read before the next is made (they are
#: read every half second, and the reading takes a moment).
BETWEEN_S = 1.2


@dataclass
class _Way:
    """The way one kind of thing goes: along which axis, from where to where, and its places' tries and pay."""

    axis: int = 0
    low: float = math.inf
    high: float = -math.inf
    size: float = 1.0
    places: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0] for _ in range(PLACES)])

    def saw(self, thing: Any) -> None:
        self.size = max(1.0, float(max(thing.w, thing.h)))
        across = abs(float(thing.vx)) >= abs(float(thing.vy))
        if self.high - self.low < GOES_FAR * self.size:
            self.axis = 0 if across else 1
        where = (float(thing.x), float(thing.y))[self.axis]
        self.low, self.high = min(self.low, where), max(self.high, where)

    def goes_far(self) -> bool:
        return self.high - self.low >= GOES_FAR * self.size

    def place(self, where: float) -> int:
        share = (where - self.low) / max(1e-6, self.high - self.low)
        return min(PLACES - 1, max(0, int(share * PLACES)))

    def worth(self, place: int, tries_in_all: float) -> float:
        tries = paid = 0.0
        for near, weight in ((place, 1.0), (place - 1, NEIGHBOURS), (place + 1, NEIGHBOURS)):
            if 0 <= near < PLACES:
                tries += weight * self.places[near][0]
                paid += weight * self.places[near][1]
        if tries == 0:
            return math.inf
        return paid / tries + TRYING * math.sqrt(math.log(max(1.0, tries_in_all)) / tries)


class WhenAPressPays:
    """What each place along each moving kind's way has paid when a press was made with a thing of it there."""

    def __init__(self) -> None:
        self.ways: dict[int, _Way] = {}
        self.last: dict[int, int] | None = None
        self.counted_at: tuple[int, int] = (0, 0)
        self.pressed_at = -math.inf

    def saw(self, things: Any) -> None:
        """The things in the picture now: each that keeps moving widens the way of its kind."""
        for thing in things:
            if getattr(thing, "moved", False) and math.hypot(float(thing.vx), float(thing.vy)) > 0.0:
                self.ways.setdefault(int(thing.kind), _Way()).saw(thing)

    def credit(self, gains: int, losses: int) -> None:
        """What was gained and lost since the last press, put down to the places the moving things were in at it."""
        if self.last is not None:
            paid = (gains - self.counted_at[0]) - (losses - self.counted_at[1])
            for kind, place in self.last.items():
                tried = self.ways[kind].places[place]
                tried[0] += 1
                tried[1] += paid
            self.last = None
        self.counted_at = (gains, losses)

    def press_now(self, things: Any, at: float, ahead_s: float) -> bool:
        """Whether to press now: the thing that keeps moving will be, ``ahead_s`` from now, at the place most worth it."""
        if at - self.pressed_at < BETWEEN_S:
            return False
        movers = self._movers(things)
        if not movers:
            return False
        # The kind whose places say most: one with a place that has paid over several presses, else the one going
        # furthest. One press that paid proves little: offline 2026-10-09 a flicker by the bar was credited with the
        # first point, was timed against from then on, and was never where it was wanted.
        stalled = at - self.pressed_at > STALLED * BETWEEN_S and math.isfinite(self.pressed_at)
        kind, thing = max(movers.items(), key=lambda item: (0.0 if stalled else self._proven(item[0]),
                                                           self.ways[item[0]].high - self.ways[item[0]].low))
        way = self.ways[kind]
        tries_in_all = sum(tries for tries, _paid in way.places)
        target = max(range(PLACES), key=lambda place: way.worth(place, tries_in_all))
        return way.place(_coming(thing, ahead_s)[way.axis]) == target

    def pressed(self, things: Any, at: float, ahead_s: float) -> None:
        """A press made now, put down to where each moving thing will be as it lands: where it was chosen for."""
        self.pressed_at = at
        self.last = {kind: self.ways[kind].place(_coming(thing, ahead_s)[self.ways[kind].axis])
                     for kind, thing in self._movers(things).items()}

    def a_round_s(self) -> float:
        """How long a round of presses takes to try every place once: the least time timing is given to pay."""
        return PLACES * BETWEEN_S

    def pays(self) -> bool:
        """Whether a place has paid over several presses: timing is how this world is played."""
        return any(self._proven(kind) > 0 for kind in self.ways)

    def has_something_to_time(self, things: Any) -> bool:
        """Whether something keeps moving far enough to time presses against: then every press is a timed one, and
        no other press is made to be put down to its places (offline 2026-10-09, her trials of the same key went on
        between timed presses, and their misses were put down to the timed ones)."""
        return bool(self._movers(things))

    def _proven(self, kind: int) -> float:
        """The best a place of this kind's way has paid a press, over places tried more than once."""
        return max((paid / tries for tries, paid in self.ways[kind].places if tries >= PROVEN), default=0.0)

    def _movers(self, things: Any) -> dict[int, Any]:
        """One thing of each kind that keeps moving far enough to be timed against."""
        found: dict[int, Any] = {}
        for thing in things:
            way = self.ways.get(int(thing.kind))
            if way is not None and way.goes_far() and getattr(thing, "moved", False):
                found.setdefault(int(thing.kind), thing)
        return found


def _coming(thing: Any, ahead_s: float) -> tuple[float, float]:
    """Where a thing will be ``ahead_s`` from now, going on as it goes."""
    return float(thing.x) + float(thing.vx) * ahead_s, float(thing.y) + float(thing.vy) * ahead_s
