"""What one press does over the moment after it: a curve per kind of thing, learned by watching, run forward to choose.

A key that moves her as long as it is held is a way, and her play knows those
(core/agency/which_one_answers_to_her.py). A press can do something else: lift
her and let her come back down (a jump), throw her a distance and stop (a dash),
take her out and back (a dodge). None of these is a speed. Each is a path over
the next second, the same each time, whatever else she is doing.

So every press of a key is watched, on every thing on the screen: where it is,
against where it would have been going on as it was going, picture by picture,
for as long as it is still going somewhere from the press or until the longest
a press has been seen to last. The curves are kept by key and by kind of thing,
so they are there when she finds which thing is hers, and their middle is what
the key does to that kind. Choosing, she runs that path forward against what is
coming (``path``): a press that takes her clear of something that holding her
course runs into is the move to make. A key whose path rises and comes back
down is one that lifts her (``lifts``).

Nothing here knows what a jump is for: clearing a thing that runs at her, a gap,
a closing door; a dash past a guard; stepping out of the way of a falling thing.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatAPressDoes"]

#: The longest after a press she watches what it does, in seconds: longer than a jump's hang in any world measured
#: (Flash platformers: 0.5 to 1.1 s), so the whole of a jump is seen.
LONGEST_S = 1.5
#: Pictures in a row without going anywhere that end a watch early: the press has done what it does.
SETTLED = 4
#: Curves kept per key and kind, newest last; their middle is what the key does.
KEPT = 7
#: Curves seen before a key's path is used: two that agree, else three, so their middle stands against one that
#: went wrong. Two agree where they are nowhere further apart than this share of how far either goes.
ENOUGH = 3
AGREE = 0.3
#: The most things watched from one press: a screen of things, not a swarm.
WATCHED = 24
#: Curves of the thing that was hers when the key was pressed, whatever its kind: a kind can be numbered afresh.
HERS = -1
#: How near where it began a lifted thing must come back down, as a share of how high it went.
LIFTED_BACK = 0.2
#: How far out, in working pixels, where a thing is seen may be from where it is: a lift must rise further than this.
SEEN_WITHIN_PX = 2.0


@dataclass
class _Watch:
    key: str
    number: int
    kind: int
    began: float
    x: float
    y: float
    drift: tuple[float, float]
    size: float
    samples: list[tuple[float, float, float]] = field(default_factory=list)
    still: int = 0
    let_go: bool = False
    hers: bool = False


class WhatAPressDoes:
    """The curves of what each press did to each kind of thing, and the path a press would take hers on now."""

    def __init__(self) -> None:
        self.watching: list[_Watch] = []
        self.curves: dict[tuple[str, int], deque[list[tuple[float, float, float]]]] = defaultdict(lambda: deque(maxlen=KEPT))
        #: The same curves by the number of the thing they were watched on: a kind can be read afresh as a thing's look
        #: changes (a body mid-jump), its number goes on.
        self.by_number: dict[tuple[str, int], deque[list[tuple[float, float, float]]]] = defaultdict(lambda: deque(maxlen=KEPT))
        #: How long each press of a key was held, newest last: a press is made again as it was made when watched.
        self.holds: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=KEPT))
        self._pressed_at: dict[str, float] = {}

    def pressed(self, key: str, at: float, things: Iterable[Any], hers: int | None = None) -> None:
        """A press of ``key`` at ``at``, watched on each of ``things`` from where it is and the way it is going; ``hers``
        is the number of the one that is hers, where that is known."""
        if not key:
            return
        self._pressed_at[key] = at
        self.watching = [w for w in self.watching if w.key != key]
        for thing in list(things)[:WATCHED]:
            # A thing seen too few times to know how it goes on is not watched: its own going would be read as the
            # press's (LIVE-like 2026-10-10, a block just come on at the edge, sliding at the runner, read as moved by
            # every press, and taken for hers).
            born = getattr(thing, "born", None)
            new = born is not None and at - float(born) < NEW_FOR_S
            if len(getattr(thing, "path", ()) or ()) < KNOWN_AFTER and (
                    new or math.hypot(float(thing.vx), float(thing.vy)) > GOING_PX_S):
                continue
            # Likewise one going along whose course cannot be told (its speed unsteady as read): taken as still, its own
            # going would be the press's.
            drift = _going_on(thing)
            if drift == (0.0, 0.0) and math.hypot(float(thing.vx), float(thing.vy)) > GOING_PX_S:
                continue
            self.watching.append(_Watch(key, int(thing.number), int(thing.kind), at, float(thing.x), float(thing.y),
                                        drift, max(float(thing.w), float(thing.h), 1.0),
                                        hers=hers is not None and int(thing.number) == hers))

    def released(self, key: str, at: float) -> None:
        """``key`` let go at ``at``: how long that press was held."""
        began = self._pressed_at.pop(key, None)
        if began is not None:
            self.holds[key].append(max(0.0, at - began))
        for watch in self.watching:
            if watch.key == key:
                watch.let_go = True

    def held_for(self, key: str, least: float) -> float:
        """How long to hold ``key`` to press it as it was pressed when watched; at least ``least`` seconds."""
        held = list(self.holds.get(key, ()))
        return max(least, statistics.median(held)) if held else least

    def saw(self, things: dict[int, Any], at: float) -> None:
        """The things in the picture now, by number, for every press being watched."""
        for watch in list(self.watching):
            after = at - watch.began
            thing = things.get(watch.number)
            if thing is None:
                self.watching.remove(watch)
                continue
            if after <= 0.0:
                continue
            dx = float(thing.x) - watch.x - watch.drift[0] * after
            dy = float(thing.y) - watch.y - watch.drift[1] * after
            if watch.samples:
                watch.still = watch.still + 1 if math.dist((dx, dy), watch.samples[-1][1:]) < 0.05 * watch.size else 0
            watch.samples.append((after, dx, dy))
            if after >= LONGEST_S or (watch.still >= SETTLED and len(watch.samples) > SETTLED):
                self.watching.remove(watch)
                # A key still held when the watch is over was held, not pressed: holding a key that lifts her lifts
                # her again and again, and that is not what one press does.
                if watch.let_go or after < LONGEST_S:
                    self.curves[(watch.key, watch.kind)].append(watch.samples)
                    self.by_number[(watch.key, watch.number)].append(watch.samples)
                    if watch.hers:
                        self.curves[(watch.key, HERS)].append(watch.samples)

    def _curves(self, key: str, kind: int | None, number: int | None) -> list[list[tuple[float, float, float]]]:
        """The curves that say what ``key`` does to her: of her kind, else of her as she was known, else of her thing.
        One curve that lifts her and lets her down whole is enough on its own: a curve gone wrong has no such shape,
        and her first trial of a key is often made before her thing is followed (offline 2026-10-09)."""
        found = [(key, kind) if kind is not None else None, (key, HERS)]
        stores = [self.curves.get(at, ()) if at is not None else () for at in found]
        stores.append(self.by_number.get((key, number), ()) if number is not None else ())
        for curves in stores:
            usable = [curve for curve in curves if len(curve) >= 3]
            if _enough(usable) or (len(usable) == 1 and _lifts(usable[0])):
                return usable
        return []

    def path(self, key: str, kind: int | None, number: int | None = None) -> Callable[[float], tuple[float, float]] | None:
        """Where a press of ``key`` takes a thing of ``kind`` (numbered ``number``), as a function of the seconds after
        it; None until seen enough."""
        curves = self._curves(key, kind, number)
        if not curves:
            return None

        def at(seconds: float) -> tuple[float, float]:
            points = [_where(curve, seconds) for curve in curves]
            return statistics.median(p[0] for p in points), statistics.median(p[1] for p in points)

        return at

    def lasts(self, key: str, kind: int | None, number: int | None = None) -> float:
        """How long a press of ``key`` goes on doing what it does, in seconds: the middle of its curves' lengths."""
        curves = self._curves(key, kind, number)
        return statistics.median(curve[-1][0] for curve in curves) if curves else 0.0

    def moves_anything(self, least: float) -> bool:
        """Whether a press has been seen to move a thing more than ``least`` the same way each time it was made."""
        return bool(self.kinds_it_moves(least))

    def kinds_it_moves(self, least: float) -> set[int]:
        """The kinds of thing a press of some key moves more than ``least``, the same way each time it was made."""
        moved = set()
        for (key, kind) in list(self.curves):
            if kind == HERS:
                continue
            path = self.path(key, kind)
            if path is not None and max(math.hypot(*path(step / 20 * LONGEST_S)) for step in range(21)) > least:
                moved.add(kind)
        return moved

    def lifts(self, kind: int | None, size: float, number: int | None = None) -> list[str]:
        """The keys whose press lifts a thing of ``kind`` (numbered ``number``) by more than half its size and brings it
        back down near where it began."""
        found = []
        keys = {key for key, of in self.curves if of in (kind, HERS)} | {key for key, of in self.by_number if of == number}
        for key in sorted(keys):
            path = self.path(key, kind, number)
            if path is None:
                continue
            curve = [path(step / 20 * LONGEST_S) for step in range(21)]
            if min(dy for _dx, dy in curve) < -0.5 * size and abs(curve[-1][1]) < 0.35 * size:
                found.append(key)
        return found


def _lifts(curve: list[tuple[float, float, float]]) -> bool:
    """Whether one curve rises clear of where it began and comes back down to it, against its own height.

    Clear means further than where a thing is seen can be out: offline 2026-10-09, read from screenshots, a robot
    walking flat was seen a fraction of a pixel higher mid-walk than at either end, its one curve was taken for a lift,
    and it was taken for hers in three runs of six.
    """
    highest = min(dy for _t, _dx, dy in curve)
    return (highest < -SEEN_WITHIN_PX and abs(curve[-1][2]) < LIFTED_BACK * abs(highest)
            and curve[-1][0] >= SETTLED * 0.04)


def _enough(curves: list[list[tuple[float, float, float]]]) -> bool:
    """Whether curves say what a press does: three or more, or two that agree."""
    if len(curves) >= ENOUGH:
        return True
    if len(curves) < 2:
        return False
    times = [step / 20 * LONGEST_S for step in range(21)]
    first, second = ([_where(curve, t) for t in times] for curve in curves[:2])
    reach = max(max(math.hypot(*point) for point in first), max(math.hypot(*point) for point in second), 1.0)
    return max(math.dist(a, b) for a, b in zip(first, second, strict=True)) <= AGREE * reach


#: How many pictures a thing must have been seen in for how it goes on to be known; and above what speed, in working
#: pixels a second, a thing is going along.
KNOWN_AFTER = 4
GOING_PX_S = 8.0
#: How long, in seconds, a thing that has just turned up is new: how it goes on is not yet seen.
NEW_FOR_S = 0.4


def _going_on(thing: Any) -> tuple[float, float]:
    """The way a thing would have gone on without the press: its speed, where that has held over its last steps.

    A thing falling, or flung, is gaining speed, and its speed at the press says nothing of where it would be a
    second later: taken as going on, a press made mid-fall read as lifting it two hundred pixels (offline
    2026-10-09). A speed that has held is going on; one that has not is taken as none.
    """
    path = list(getattr(thing, "path", ()) or ())[-4:]
    steps = [((x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)) for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:], strict=False)
             if t1 > t0]
    if len(steps) < 3:
        return 0.0, 0.0
    fastest = max(math.hypot(*step) for step in steps)
    if fastest > 0 and max(math.dist(step, steps[-1]) for step in steps) > 0.25 * fastest:
        return 0.0, 0.0
    return float(thing.vx), float(thing.vy)


def _where(curve: list[tuple[float, float, float]], seconds: float) -> tuple[float, float]:
    """A curve's displacement ``seconds`` after its press, between its samples; past its end, where it ended."""
    if seconds <= curve[0][0]:
        share = seconds / curve[0][0] if curve[0][0] > 0 else 1.0
        return curve[0][1] * share, curve[0][2] * share
    for (t0, x0, y0), (t1, x1, y1) in zip(curve, curve[1:], strict=False):
        if t0 <= seconds <= t1:
            share = (seconds - t0) / (t1 - t0) if t1 > t0 else 0.0
            return x0 + (x1 - x0) * share, y0 + (y1 - y0) * share
    return curve[-1][1], curve[-1][2]
