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
        self._last: dict[int, tuple[float, float, float, float, float, int]] = {}
        self.shape: tuple[int, int] = (0, 0)

    # -- learning --------------------------------------------------------------

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
            kind.speeds.append(math.hypot(thing.vx, thing.vy))
            if before is None or not 0.0 < at - before[0] < 0.12:
                continue
            self._learn_a_step(thing, before, at, kind, mine)
        self._learn_the_edges(moves, happened)
        self._last = {n: v for n, v in self._last.items() if n in seen}

    def _learn_a_step(self, thing: Any, before: tuple, at: float, kind: _Kind, mine: Any) -> None:
        dt = at - before[0]
        _t, x0, y0, vx0, vy0, _k = before
        turned_x = vx0 * thing.vx < 0 and abs(vx0) > 20 and abs(thing.vx) > 20
        turned_y = vy0 * thing.vy < 0 and abs(vy0) > 20 and abs(thing.vy) > 20
        if not (turned_x or turned_y):
            kind.accelerations.append(((thing.vx - vx0) / dt, (thing.vy - vy0) / dt))
            return
        near_her = mine is not None and abs(thing.x - mine.x) < (thing.w + mine.w) / 2 + 6 and abs(thing.y - mine.y) < (thing.h + mine.h) / 2 + 6
        if near_her:
            self._learn_a_meeting(thing, (vx0, vy0), mine, kind, turned_x)
            return
        tall, wide = self.shape
        for turned, position, low, high, names in (
            (turned_x, thing.x, 0.0, float(wide), ("left", "right")),
            (turned_y, thing.y, 0.0, float(tall), ("top", "bottom")),
        ):
            if not turned:
                continue
            edge = names[0] if position - low < high - position else names[1]
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

    def gravity(self, kind: int) -> tuple[float, float]:
        samples = list(self.kinds[kind].accelerations) if kind in self.kinds else []
        if len(samples) < ENOUGH * 4:
            return 0.0, 0.0
        ax = statistics.median(a for a, _b in samples)
        ay = statistics.median(b for _a, b in samples)
        return (ax if abs(ax) > STILL_AIR else 0.0), (ay if abs(ay) > STILL_AIR else 0.0)

    def edge(self, kind: int, name: str) -> tuple[str, float | None, float]:
        """What an edge does to a kind, where it really is, and how much speed a bounce keeps."""
        record = self.kinds[kind].edges.get(name) if kind in self.kinds else None
        if record is None:
            return "", None, 1.0
        where = statistics.median(record.where) if len(record.where) >= 2 else None
        kept = statistics.median(record.kept) if len(record.kept) >= 2 else 1.0
        return record.does(), where, kept

    def fastest(self, kind: int) -> float:
        speeds = sorted(self.kinds[kind].speeds) if kind in self.kinds else []
        return speeds[int(len(speeds) * 0.95)] if len(speeds) >= ENOUGH else 0.0

    def after_meeting(self, kind: int, along: float, incoming: tuple[float, float], across: bool) -> tuple[float, float] | None:
        """How a thing would leave her thing if it met her ``along`` her length, once enough meetings were seen.

        A straight line through the angles seen against where they met, and
        the median speed gained. None until there are enough meetings to say.
        """
        meetings = self.kinds[kind].meetings if kind in self.kinds else []
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
