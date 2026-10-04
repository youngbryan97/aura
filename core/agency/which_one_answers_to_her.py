"""Which of the things on screen is hers, found by what answers when she acts.

In a game that runs on its own, everything moves. The ball moves, the other
paddle moves, the clouds move. One thing moves because she pressed a key, and
it is the only thing whose movement depends on which key she is holding.

That is a measurement, and an old one in animal learning: contingency. For
each thing, its speed is collected under each key she was holding at the time.
If the speeds differ more between keys than they vary under any one key, the
thing answers to her. The ball's speed is the same whatever she holds; her
paddle's is up under one key and down under another. The ratio of those two
spreads is the F statistic of a one-way analysis of variance, and the thing
with the largest one, well clear of chance, is hers.

The test holds only while the keys are chosen without looking. Once she plays,
she presses up because the ball is going up, and then the ball's speed depends
on her keys too: her own choices make the ball look like hers. So only the
pictures taken while she is trying keys, in a fixed order whatever is on the
screen, go into the test. Every picture of her own thing still says what each
key does to it, in pixels a second, which is what a plan needs. A key that moves nothing may still make something: a shot
that appears beside her and flies off. That is found the same way, by what
turns up near her just after the key and not otherwise.

Nothing here is told which thing is hers, or which keys do what.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Makes", "WhichIsHers"]

#: How long after a key is pressed its effect shows in a picture: one frame of
#: the game and one of the picture being taken.
RESPONSE_S = 0.05

#: The F ratio a thing's speeds must reach to count as answering to her. At
#: the sample sizes here, chance stays under 5 for three keys with fifty
#: samples (the 1% point of F(2, 50) is 5.06); this is four times that.
ANSWERS = 20.0

#: And the difference has to be a movement, not a jitter: the fastest and
#: slowest key must differ by this many working pixels a second.
REALLY_MOVES = 20.0

#: Samples a key needs before it says anything about a thing.
ENOUGH = 4


@dataclass
class Makes:
    """What pressing a key brings into the picture beside her."""

    key: str
    kind: int
    vx: float
    vy: float
    times: int


@dataclass
class _Speeds:
    by_key: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))

    def add(self, key: str, vx: float, vy: float) -> None:
        kept = self.by_key[key]
        kept.append((vx, vy))
        if len(kept) > 200:
            del kept[:100]

    def ratio(self) -> tuple[float, float]:
        """F, and the largest difference between two keys' mean speeds."""
        groups = {key: values for key, values in self.by_key.items() if len(values) >= ENOUGH}
        if len(groups) < 2:
            return 0.0, 0.0
        means = {key: _mean(values) for key, values in groups.items()}
        everyone = [value for values in groups.values() for value in values]
        grand = _mean(everyone)
        between = sum(len(groups[key]) * _apart2(means[key], grand) for key in groups)
        within = sum(_apart2(value, means[key]) for key, values in groups.items() for value in values)
        k, n = len(groups), len(everyone)
        if n <= k:
            return 0.0, 0.0
        f = (between / (k - 1)) / max(1e-6, within / (n - k))
        widest = max(math.dist(a, b) for a in means.values() for b in means.values())
        return f, widest

    def typical(self, key: str) -> tuple[float, float] | None:
        """The middle speed under a key: a picture matched to the wrong thing does not move it."""
        values = self.by_key.get(key) or []
        if len(values) < ENOUGH:
            return None
        return statistics.median(v[0] for v in values), statistics.median(v[1] for v in values)


def _follows(pairs: list[tuple[float, float, float, float]]) -> tuple[bool, bool]:
    """Along which axes a thing's place goes with the pointer's: correlation above 0.9, slope near one."""
    import numpy as np

    data = np.asarray(pairs[-60:], dtype=np.float64)
    along = []
    for axis in (0, 1):
        pointer, thing = data[:, axis], data[:, 2 + axis]
        if pointer.std() < 5.0 or thing.std() < 5.0:
            along.append(False)
            continue
        r = float(np.corrcoef(pointer, thing)[0, 1])
        slope = float(np.polyfit(pointer, thing, 1)[0])
        along.append(r > 0.9 and 0.5 < slope < 2.0)
    return along[0], along[1]


def _mean(values: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(v[0] for v in values) / len(values), sum(v[1] for v in values) / len(values)


def _apart2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class WhichIsHers:
    """Keeps, for each thing and each kind of thing, how it moved under each key."""

    def __init__(self) -> None:
        self._held: deque = deque(maxlen=400)
        self._by_thing: dict[int, _Speeds] = defaultdict(_Speeds)
        self._by_kind: dict[int, _Speeds] = defaultdict(_Speeds)
        #: How whichever thing was hers at the time moved under each key. Not
        #: by kind: the other player's paddle looks just like hers.
        self._hers = _Speeds()
        self.number: int | None = None
        self.kind: int | None = None
        self.last_seen: tuple[float, float] | None = None
        self.last_size = 0.0
        self.lowest: list[float] = [math.inf, math.inf]
        self.highest: list[float] = [-math.inf, -math.inf]
        self._taps: list[tuple[str, float, tuple[float, float]]] = []
        self._made: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        self.makes: dict[str, Makes] = {}
        self._pointer: list[tuple[float, float, float]] = []
        self._followed: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        self.follows_pointer = False
        self._new_screen_at = -math.inf
        #: The axes along which it follows: a paddle under the pointer may follow only across.
        self.follows_along: tuple[bool, bool] = (False, False)

    # -- what she did ------------------------------------------------------

    def holding(self, key: str, at: float, *, trying: bool = False) -> None:
        """She is holding ``key`` from ``at`` on ("" for nothing); ``trying`` when it was not chosen by looking."""
        if not self._held or self._held[-1][1:] != (key, trying):
            self._held.append((at, key, trying))

    def tapped(self, key: str, at: float) -> None:
        if self.last_seen is not None:
            self._taps.append((key, at, self.last_seen))

    def pointed(self, x: float, y: float, at: float) -> None:
        """The pointer was taken to (x, y), in working pixels."""
        self._pointer.append((at, x, y))
        del self._pointer[:-40]

    def _pointer_at(self, at: float) -> tuple[float, float] | None:
        for when, x, y in reversed(self._pointer):
            if when <= at:
                return x, y
        return None

    def _what_follows_the_pointer(self, moves: Any, at: float) -> None:
        """A thing that goes where the pointer went is hers, the way a key's thing is."""
        if not self._pointer or self.follows_pointer:
            return
        pointer = self._pointer_at(at - 0.15)
        if pointer is None:
            return
        # By kind, the nearest of each to the pointer: a thing under a pointer
        # that moved too far in one picture comes back as a new thing.
        nearest: dict[int, Any] = {}
        for thing in moves.things.values():
            if thing.seen != at:
                continue
            best = nearest.get(thing.kind)
            if best is None or math.dist((thing.x, thing.y), pointer) < math.dist((best.x, best.y), pointer):
                nearest[thing.kind] = thing
        for kind, thing in nearest.items():
            self._followed[kind].append((pointer[0], pointer[1], thing.x, thing.y))
        for kind, pairs in self._followed.items():
            if kind not in nearest or len(pairs) < 20:
                continue
            along = _follows(pairs)
            if any(along):
                self.number, self.kind, self.follows_pointer = nearest[kind].number, kind, True
                self.follows_along = along
                return

    def _under_the_pointer(self, moves: Any, at: float) -> int | None:
        """The thing of her kind nearest the pointer: a thing that follows it is re-made when it jumps."""
        pointer = self._pointer_at(at)
        mine = [t for t in moves.things.values() if t.kind == self.kind]
        if not mine or pointer is None:
            return None
        return min(mine, key=lambda t: math.dist((t.x, t.y), pointer)).number

    def _held_at(self, at: float) -> tuple[str, bool] | None:
        for began, key, trying in reversed(self._held):
            if began <= at:
                return key, trying
        return None

    # -- what she saw ------------------------------------------------------

    def saw(self, moves: Any, happened: list[dict[str, Any]], at: float) -> None:
        if any(h.get("what") == "new screen" for h in happened):
            self._new_screen_at = at
        held = self._held_at(at - RESPONSE_S)
        # A screen being drawn afresh moves everything at once, whatever she held.
        if held is not None and at - self._new_screen_at > 0.5:
            key, trying = held
            for thing in moves.things.values():
                if thing.seen != at or thing.born == at:
                    continue
                if trying:
                    self._by_thing[thing.number].add(key, thing.vx, thing.vy)
                    self._by_kind[thing.kind].add(key, thing.vx, thing.vy)
                if thing.number == self.number:
                    self._hers.add(key, thing.vx, thing.vy)
        self._what_follows_the_pointer(moves, at)
        if not self.follows_pointer:
            self._decide(moves)
        elif self.number not in moves.things:
            self.number = self._under_the_pointer(moves, at)
        self._what_keys_make(moves, happened, at)

    def _decide(self, moves: Any) -> None:
        best, best_f = None, ANSWERS
        for number, speeds in self._by_thing.items():
            if number not in moves.things or not moves.things[number].moved:
                continue
            f, widest = speeds.ratio()
            if f > best_f and widest > REALLY_MOVES:
                best, best_f = number, f
        if best is not None:
            if best != self.number:
                for key, values in self._by_thing[best].by_key.items():
                    self._hers.by_key[key].extend(values[-50:])
            self.number = best
            self.kind = moves.things[best].kind
        elif self.number not in moves.things and self.kind is not None:
            self.number = self._one_of_her_kind(moves)
        mine = moves.things.get(self.number) if self.number is not None else None
        if mine is not None:
            self.last_seen = (mine.x, mine.y)
            self.last_size = mine.size
            self.kind = mine.kind
            for axis, value in enumerate((mine.x, mine.y)):
                self.lowest[axis] = min(self.lowest[axis], value)
                self.highest[axis] = max(self.highest[axis], value)

    def _one_of_her_kind(self, moves: Any) -> int | None:
        """After she was lost, the thing of her kind nearest where she was.

        Or, where nothing of her kind is left, a thing her size close to where
        she was last seen: a kind can be looked at again and changed under her.
        """
        candidates = [t for t in moves.things.values() if t.kind == self.kind]
        if not candidates and self.last_seen is not None and self.last_size:
            x, y = self.last_seen
            candidates = [
                t for t in moves.things.values()
                if math.hypot(t.x - x, t.y - y) < 30.0 and 0.5 < t.size / self.last_size < 2.0
            ]
        if not candidates:
            return None
        if self.last_seen is None:
            return candidates[-1].number
        x, y = self.last_seen
        return min(candidates, key=lambda t: math.hypot(t.x - x, t.y - y)).number

    def _what_keys_make(self, moves: Any, happened: list[dict[str, Any]], at: float) -> None:
        """A thing that turns up beside her just after a key, and then flies off."""
        fresh = [h for h in happened if h.get("what") == "appeared"]
        self._taps = [tap for tap in self._taps if at - tap[1] < 0.4]
        for key, when, (x, y) in self._taps:
            for event in fresh:
                if event["at"] - when < 0.0 or math.hypot(event["x"] - x, event["y"] - y) > 30.0:
                    continue
                if event["kind"] == self.kind:
                    continue
                self._made[(key, event["kind"])].append((when, event["thing"]))
        for (key, kind), births in self._made.items():
            speeds = [
                (moves.things[number].vx, moves.things[number].vy)
                for _when, number in births[-6:]
                if number in moves.things and moves.things[number].moved
            ]
            if len(births) >= 2 and speeds:
                vx, vy = _mean(speeds)
                if math.hypot(vx, vy) > REALLY_MOVES:
                    self.makes[key] = Makes(key, kind, vx, vy, len(births))

    # -- what she knows ----------------------------------------------------

    def thing(self, moves: Any) -> Any:
        return moves.things.get(self.number) if self.number is not None else None

    def way_of(self, key: str) -> tuple[float, float] | None:
        """How fast her thing goes while ``key`` is held, in working pixels a second."""
        if self.kind is None:
            return None
        return self._hers.typical(key)

    def keys_that_move_her(self, keys: list[str]) -> dict[str, tuple[float, float]]:
        moving = {}
        for key in keys:
            way = self.way_of(key)
            if way is not None and math.hypot(*way) > REALLY_MOVES:
                moving[key] = way
        return moving

    def tried(self, key: str) -> int:
        if self.kind is None:
            return max((len(s.by_key.get(key) or []) for s in self._by_thing.values()), default=0)
        return len(self._hers.by_key.get(key) or [])

    def keys_known(self, keys: list[str]) -> bool:
        """Whether every key has been held long enough on her thing to say what it does."""
        return self.kind is not None and all(self.tried(key) >= ENOUGH for key in keys)

    def axes(self, keys: list[str]) -> tuple[bool, bool]:
        """Whether her keys move her across, and up and down."""
        ways = self.keys_that_move_her(keys).values()
        return (
            any(abs(vx) > REALLY_MOVES for vx, _vy in ways),
            any(abs(vy) > REALLY_MOVES for _vx, vy in ways),
        )
