"""The physics of the world in front of her, learned by watching it, and run forward in her head.

Every world has its own physics, and none of it is written anywhere she can
read. A ball keeps its speed or slows; things fall or float; an edge sends a
thing back, lets it out, or brings it in on the other side; a paddle sends a
ball back the way it came off its middle and steeply off its ends, a little
faster each time. A player learns all of that in the first minute of play and
then stops thinking about it, because it is what they see coming.

So this keeps, for each kind of thing in this world:

- how it moves on its own: the acceleration between pictures where nothing
  touched it (gravity, if there is any), and how fast things of its kind go;
- what each edge of the picture does to it, and where that edge really is (a
  ball turned back by a scoreboard band turns back below it, not at the top
  of the picture);
- what meeting her thing does to it: the angle it leaves at, against where
  along her thing it met, and how much faster it goes.

And it runs those forward (``imagine``): where a thing will be, when it will
reach a line, what it will do at an edge, and where it would go if she met it
at one part of her thing or another (``after_meeting``). That is what lets
her play against where things will be instead of where they are, and choose
how to meet a thing by what it will do afterwards.

What is learned here is about this world only. It is kept with this world and
used elsewhere only where the same measurements say the same thing.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HowThingsMoveHere", "Imagined"]

#: Fewest samples before a learned quantity is used instead of the plain guess.
ENOUGH = 5

#: Accelerations below this, in working pixels per second squared, are noise.
STILL_AIR = 25.0

#: How close to an edge, as a share of the picture, a turn has to be to be the edge's.
NEAR_AN_EDGE = 0.15


@dataclass
class _Edge:
    bounces: int = 0
    leaves: int = 0
    wraps: int = 0
    where: list[float] = field(default_factory=list)
    kept: list[float] = field(default_factory=list)

    def does(self) -> str:
        counts = {"bounce": self.bounces, "leave": self.leaves, "wrap": self.wraps}
        best = max(counts, key=lambda name: counts[name])
        return best if counts[best] else ""


@dataclass
class _Kind:
    accelerations: deque = field(default_factory=lambda: deque(maxlen=200))
    speeds: deque = field(default_factory=lambda: deque(maxlen=200))
    edges: dict[str, _Edge] = field(default_factory=lambda: defaultdict(_Edge))
    #: (where along her thing it met, -1 at one end to 1 at the other; angle it left at; speed ratio)
    meetings: list[tuple[float, float, float]] = field(default_factory=list)


def _near(thing: Any, other: Any, margin: float) -> bool:
    return abs(thing.x - other.x) < (thing.w + other.w) / 2 + margin and abs(thing.y - other.y) < (thing.h + other.h) / 2 + margin


@dataclass
class Imagined:
    """A thing run forward: where it is at each step, and what ends the run."""

    path: list[tuple[float, float, float]]
    ends: str = ""
    at_edge: str = ""


class HowThingsMoveHere:
    """Learns the physics of one world from what the tracker sees, and runs it forward."""

    def __init__(self) -> None:
        self.kinds: dict[int, _Kind] = defaultdict(_Kind)
        #: How each kind looks, for pooling kinds that look alike: colour and size.
        self.looks: dict[int, tuple[tuple[int, int, int], float]] = {}
        self._last: dict[int, tuple[float, float, float, float, float, int]] = {}
        self.shape: tuple[int, int] = (0, 0)

    # -- learning --------------------------------------------------------------

    def numbered_afresh(self) -> None:
        """The picture's things are numbered from one again: forget what was kept of each by its number."""
        self._last.clear()

    def saw(self, moves: Any, hers: Any, happened: list[dict[str, Any]], at: float) -> None:
        """One picture's worth of motion."""
        self.shape = moves.shape
        mine = hers.thing(moves)
        seen = set()
        for thing in moves.things.values():
            if not thing.moved or (mine is not None and thing.number == mine.number):
                continue
            seen.add(thing.number)
            before = self._last.get(thing.number)
            self._last[thing.number] = (at, thing.x, thing.y, thing.vx, thing.vy, thing.kind)
            kind = self.kinds[thing.kind]
            self.looks[thing.kind] = (tuple(int(c) for c in thing.colour), float(thing.size))
            kind.speeds.append(math.hypot(thing.vx, thing.vy))
            if before is None or not 0.0 < at - before[0] < 0.12:
                continue
            others = [t for t in moves.things.values() if t.number != thing.number and (mine is None or t.number != mine.number)]
            self._learn_a_step(thing, before, at, kind, mine, others)
        self._learn_the_edges(moves, happened)
        self._last = {n: v for n, v in self._last.items() if n in seen}

    def _learn_a_step(self, thing: Any, before: tuple, at: float, kind: _Kind, mine: Any, others: list[Any]) -> None:
        dt = at - before[0]
        _t, x0, y0, vx0, vy0, _k = before
        turned_x = vx0 * thing.vx < 0 and abs(vx0) > 20 and abs(thing.vx) > 20
        turned_y = vy0 * thing.vy < 0 and abs(vy0) > 20 and abs(thing.vy) > 20
        if not (turned_x or turned_y):
            # A change of speed with something touching it is that thing's
            # doing, not the world's pull: only free flight measures gravity.
            if not any(_near(thing, other, 4.0) for other in [*others, mine] if other is not None):
                kind.accelerations.append(((thing.vx - vx0) / dt, (thing.vy - vy0) / dt))
            return
        # Seen turned a picture or two after it turned: how far it went since
        # is how near to her it may have been when it did.
        reach = 6.0 + 2.0 * math.hypot(thing.vx, thing.vy) * dt
        if mine is not None and _near(thing, mine, reach):
            self._learn_a_meeting(thing, (vx0, vy0), mine, kind, turned_x)
            return
        if any(_near(thing, other, reach) for other in others):
            return
        tall, wide = self.shape
        for turned, position, size, names in (
            (turned_x, thing.x, float(wide), ("left", "right")),
            (turned_y, thing.y, float(tall), ("top", "bottom")),
        ):
            if not turned:
                continue
            # A wall is at an edge: a thing that turns in the open turned off
            # something, and calling that a wall put walls across the middle.
            if position > size * NEAR_AN_EDGE and position < size * (1 - NEAR_AN_EDGE):
                continue
            edge = names[0] if position < size / 2 else names[1]
            record = kind.edges[edge]
            record.bounces += 1
            record.where.append(position)
            before_speed = math.hypot(vx0, vy0)
            if before_speed > 1.0:
                record.kept.append(math.hypot(thing.vx, thing.vy) / before_speed)

    def _learn_a_meeting(self, thing: Any, incoming: tuple[float, float], mine: Any, kind: _Kind, across: bool) -> None:
        """A thing that met hers: where along hers it met, the angle it left at, the speed it gained."""
        if across:
            along = (thing.y - mine.y) / max(1.0, mine.h / 2)
            angle = math.atan2(thing.vy, abs(thing.vx))
        else:
            along = (thing.x - mine.x) / max(1.0, mine.w / 2)
            angle = math.atan2(thing.vx, abs(thing.vy))
        speed_in = math.hypot(*incoming)
        ratio = math.hypot(thing.vx, thing.vy) / speed_in if speed_in > 1.0 else 1.0
        kind.meetings.append((max(-1.5, min(1.5, along)), angle, ratio))
        del kind.meetings[:-60]

    def _learn_the_edges(self, moves: Any, happened: list[dict[str, Any]]) -> None:
        tall, wide = moves.shape
        gone = [h for h in happened if h.get("what") == "gone"]
        fresh = [h for h in happened if h.get("what") == "appeared"]
        for event in gone:
            x, y = event["x"], event["y"]
            edge = min((x, "left"), (wide - x, "right"), (y, "top"), (tall - y, "bottom"))
            if edge[0] > 8:
                continue
            opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[edge[1]]
            came_back = any(
                h["kind"] == event["kind"]
                and min((h["x"], "left"), (wide - h["x"], "right"), (h["y"], "top"), (tall - h["y"], "bottom"))[1] == opposite
                for h in fresh
            )
            record = self.kinds[event["kind"]].edges[edge[1]]
            if came_back:
                record.wraps += 1
            else:
                record.leaves += 1

    # -- what it has learned ---------------------------------------------------

    def alike(self, kind: int) -> list[_Kind]:
        """This kind and the kinds that look like it: one ball seen as two kinds is one ball's physics."""
        looks = self.looks.get(kind)
        if looks is None:
            return [self.kinds[kind]] if kind in self.kinds else []
        colour, size = looks
        pooled = []
        for other, record in self.kinds.items():
            there = self.looks.get(other)
            if other == kind or (
                there is not None
                and max(abs(a - b) for a, b in zip(colour, there[0], strict=True)) <= 45
                and max(size, there[1]) / max(1e-6, min(size, there[1])) < 2.5
            ):
                pooled.append(record)
        return pooled

    def gravity(self, kind: int) -> tuple[float, float]:
        samples = [a for record in self.alike(kind) for a in record.accelerations]
        if len(samples) < ENOUGH * 4:
            return 0.0, 0.0
        ax = statistics.median(a for a, _b in samples)
        ay = statistics.median(b for _a, b in samples)
        return (ax if abs(ax) > STILL_AIR else 0.0), (ay if abs(ay) > STILL_AIR else 0.0)

    def edge(self, kind: int, name: str) -> tuple[str, float | None, float]:
        """What an edge does to a kind, where it really is, and how much speed a bounce keeps."""
        records = [r.edges[name] for r in self.alike(kind) if name in r.edges]
        if not records:
            return "", None, 1.0
        pooled = _Edge(
            sum(r.bounces for r in records), sum(r.leaves for r in records), sum(r.wraps for r in records),
            [w for r in records for w in r.where], [k for r in records for k in r.kept],
        )
        where = statistics.median(pooled.where) if len(pooled.where) >= 2 else None
        kept = statistics.median(pooled.kept) if len(pooled.kept) >= 2 else 1.0
        return pooled.does(), where, kept

    def fastest(self, kind: int) -> float:
        speeds = sorted(s for record in self.alike(kind) for s in record.speeds)
        return speeds[int(len(speeds) * 0.95)] if len(speeds) >= ENOUGH else 0.0

    def after_meeting(self, kind: int, along: float, incoming: tuple[float, float], across: bool) -> tuple[float, float] | None:
        """How a thing would leave her thing if it met her ``along`` her length, once enough meetings were seen.

        A straight line through the angles seen against where they met, and
        the median speed gained. None until there are enough meetings to say.
        """
        meetings = [m for record in self.alike(kind) for m in record.meetings]
        if len(meetings) < ENOUGH:
            return None
        xs = [m[0] for m in meetings]
        ys = [m[1] for m in meetings]
        mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
        spread = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / spread if spread > 1e-6 else 0.0
        angle = mean_y + slope * (along - mean_x)
        speed = math.hypot(*incoming) * statistics.median(m[2] for m in meetings)
        if across:
            away = -1.0 if incoming[0] > 0 else 1.0
            return away * speed * math.cos(angle), speed * math.sin(angle)
        away = -1.0 if incoming[1] > 0 else 1.0
        return speed * math.sin(angle), away * speed * math.cos(angle)

    # -- running it forward ----------------------------------------------------

    def imagine(self, thing: Any, seconds: float, *, start: tuple[float, float, float, float] | None = None,
                step: float = 1 / 60) -> Imagined:
        """Run a thing forward under this world's physics as learned so far."""
        tall, wide = self.shape
        x, y, vx, vy = start if start is not None else (thing.x, thing.y, thing.vx, thing.vy)
        ax, ay = self.gravity(thing.kind)
        path = [(0.0, x, y)]
        elapsed = 0.0
        while elapsed < seconds:
            elapsed += step
            vx, vy = vx + ax * step, vy + ay * step
            x, y = x + vx * step, y + vy * step
            for axis, low_name, high_name, limit in ((0, "left", "right", wide), (1, "top", "bottom", tall)):
                position = x if axis == 0 else y
                for name, beyond in ((low_name, position < 0.0), (high_name, position > limit)):
                    does, where, kept = self.edge(thing.kind, name)
                    if where is not None and does == "bounce":
                        beyond = position < where if name in ("left", "top") else position > where
                    if not beyond:
                        continue
                    if does == "bounce":
                        wall = where if where is not None else (0.0 if name in ("left", "top") else float(limit))
                        if axis == 0:
                            x, vx = 2 * wall - x, -vx * kept
                        else:
                            y, vy = 2 * wall - y, -vy * kept
                    elif does == "wrap":
                        if axis == 0:
                            x = x % max(1, wide)
                        else:
                            y = y % max(1, tall)
                    else:
                        path.append((elapsed, x, y))
                        return Imagined(path, ends="leaves", at_edge=name)
            path.append((elapsed, x, y))
        return Imagined(path)

    def when_it_reaches(self, thing: Any, axis: int, line: float, seconds: float = 3.0,
                        start: tuple[float, float, float, float] | None = None) -> tuple[float, float] | None:
        """When a thing run forward first crosses ``line`` on ``axis``, and where it is along the other axis then."""
        imagined = self.imagine(thing, seconds, start=start)
        previous = imagined.path[0]
        for point in imagined.path[1:]:
            a, b = previous[1 + axis], point[1 + axis]
            if (a - line) * (b - line) <= 0 and a != b:
                share = (line - a) / (b - a)
                other = previous[2 - axis] + share * (point[2 - axis] - previous[2 - axis])
                return previous[0] + share * (point[0] - previous[0]), other
            previous = point
        return None
