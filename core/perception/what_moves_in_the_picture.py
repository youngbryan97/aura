"""The things in a moving picture: where each is, how fast it is going, and which look alike.

A board is read as places, and a page as words. A game that runs on its own
clock is neither. What matters in it is a handful of shapes moving over a
backdrop: one of them answers to her keys, some come at her, some get away.
Reading the words on it every second and a half tells her the score and
nothing about where the ball is.

So this keeps a picture of the backdrop, the part that stays where it is, and
takes everything that differs from it as a thing. The backdrop starts as the
middle value of each pixel over the first half second, which is what a place
shows when nothing is passing over it, and then drifts toward each new picture
everywhere except under the things that have moved. Under those it holds, so a
paddle that stops is still a paddle and does not sink into the floor, and the
place a thing has just left fades back to floor within about a second.
Things are matched from one picture to the next by where they were going and
by how they look, so each keeps a name while it moves.

Things that look alike are one kind. That is what lets what she learns about
one falling bomb hold for the next.

Nothing here knows what a game is. It is measured on the pixels of whatever it
is given, at a working size small enough to read twenty times a second.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Kind", "Thing", "WhatMoves", "what_happened"]

#: The width pictures are read at. A sprite eight pixels across on a 480-wide
#: game is four here, which is still a thing, and a picture is cheap to read.
WORKING_WIDTH = 240

#: How far a pixel's colour has to be from the backdrop's, in 0..255 on its
#: most different channel, to belong to something on it. Compression noise in
#: a JPEG of a flat colour stays under a quarter of this.
DIFFERENT_ENOUGH = 40

#: The fewest working pixels a thing covers. Fewer is noise or a stray glint.
SMALLEST = 3

#: The largest share of the picture one thing may cover. A change bigger than
#: this is the picture changing, not something moving across it.
LARGEST = 0.25

#: Most of the picture changing at once is a new screen, not movement.
NEW_SCREEN = 0.45

#: How long the first look at a screen lasts before anything on it is taken
#: for a thing. Its middle value per pixel is the first backdrop.
FIRST_LOOK_S = 0.5

#: How fast the backdrop drifts to the picture where nothing that moved is
#: standing, per second. A place a thing has left is floor again in about a
#: second.
BACKDROP_DRIFT = 2.5

#: A thing missing this long, and not found by its look, has gone.
GONE_AFTER_S = 0.25


@dataclass
class Kind:
    """Things that look alike: one colour mix, one size."""

    number: int
    look: np.ndarray
    size: float
    colour: tuple[int, int, int]
    seen: int = 0

    def like(self, colour: tuple[int, int, int], size: float) -> float:
        """How far a colour and size are from this kind's; under KIND_ALIKE is the same kind."""
        ratio = max(size, self.size) / max(1e-6, min(size, self.size))
        apart = max(abs(int(a) - int(b)) for a, b in zip(colour, self.colour, strict=True)) / 255.0
        return apart + (0.0 if ratio < 2.5 else 1.0)


#: Colours nearer than this on their most different channel, as a share of
#: the whole range, are one kind when the sizes are alike too. Forty-five
#: levels apart is the difference between two colours a person names apart.
KIND_ALIKE = 45 / 255.0

#: Sizes within this ratio of each other are one size to the eye.
PLAINLY_THE_SAME_SIZE = 1.6


@dataclass
class Thing:
    """One thing in the picture, followed from picture to picture.

    Positions and sizes are in working pixels; ``share`` turns them into
    fractions of the picture for anyone outside.
    """

    number: int
    x: float
    y: float
    w: float
    h: float
    look: np.ndarray
    colour: tuple[int, int, int]
    patch: np.ndarray
    born: float
    seen: float
    kind: int = -1
    moved: bool = False
    vx: float = 0.0
    vy: float = 0.0
    still: bool = False
    path: deque = field(default_factory=lambda: deque(maxlen=40))
    sizes: deque = field(default_factory=lambda: deque(maxlen=15))

    @property
    def size(self) -> float:
        return self.w * self.h

    def box(self) -> tuple[float, float, float, float]:
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)

    def where_at(self, after_s: float) -> tuple[float, float]:
        return self.x + self.vx * after_s, self.y + self.vy * after_s


def _measured(thing: Thing) -> tuple[float, float]:
    """Where a thing was last seen, as against where it is thought to be now."""
    return (thing.path[-1][1], thing.path[-1][2]) if thing.path else (thing.x, thing.y)


def _speed_from_its_path(
    path: deque, at: float, last_step: tuple[float, float], blended: tuple[float, float]
) -> tuple[float, float]:
    """How fast a thing is going, fitted through its last few places rather than averaged.

    An average of steps lags a thing that has just sped up, and a ball that
    speeds up a little each return is aimed at where it would have been. A
    straight line through its last tenth of a second does not lag. Where the
    last step turned back against the fitted line, it has just bounced, and
    the last step is the speed.
    """
    recent = [point for point in path if at - point[0] <= 0.15][-5:]
    if len(recent) < 3:
        return blended
    times = np.asarray([point[0] for point in recent]) - recent[-1][0]
    if float(times.max() - times.min()) < 1e-3:
        return blended
    fitted = []
    for axis in (1, 2):
        slope = float(np.polyfit(times, np.asarray([point[axis] for point in recent]), 1)[0])
        step = last_step[axis - 1]
        fitted.append(step if slope * step < 0 and abs(step) > 20.0 else slope)
    return fitted[0], fitted[1]


#: The bins of a colour mix: four levels a channel.
LOOK_BINS = 64


def _look_of(pixels: np.ndarray) -> np.ndarray:
    """A thing's colour mix: 64 bins of its pixels, four levels a channel, summing to one."""
    if pixels.size == 0:
        return np.full(LOOK_BINS, 1.0 / LOOK_BINS)
    q = (pixels.astype(np.int32) // 64).clip(0, 3)
    bins = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    counts = np.bincount(bins, minlength=64).astype(np.float64)
    return counts / max(1.0, counts.sum())


def _look_apart(a: np.ndarray, b: np.ndarray) -> float:
    """Half the summed difference of two colour mixes: 0 the same, 1 nothing shared."""
    return float(0.5 * np.abs(a - b).sum())


def _without_what_stands_out(still: np.ndarray) -> np.ndarray:
    """The first backdrop, with the small things that stand out from their floor taken off it.

    A coin that never moves is in every picture of the first look, so the
    middle value of each pixel keeps it. What tells it from the floor is that
    it is small and unlike what is around it: the floor is what a wide median
    filter leaves, and anything compact that differs from that by a clear
    margin is put back to floor, so that it shows as a thing from the start.
    """
    from core.perception.picture_arithmetic import apart, median, pieces

    tall, wide = still.shape[:2]
    reach = max(3, (min(tall, wide) // 12) | 1)
    floor = median(still, reach)
    unlike = apart(still, floor).max(axis=2) > DIFFERENT_ENOUGH
    count, labels, stats, _centres = pieces(unlike)
    backdrop = still.astype(np.float32)
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < SMALLEST or w > reach or h > reach:
            continue
        inside = labels == label
        backdrop[inside] = floor[inside]
    return backdrop


def what_happened(kind: str, thing: Thing, at: float) -> dict[str, Any]:
    return {"what": kind, "thing": thing.number, "kind": thing.kind, "x": thing.x, "y": thing.y, "at": at}


class WhatMoves:
    """Reads pictures one after another and keeps the things in them.

    ``see(picture, at)`` takes an RGB array of any size and returns what
    happened in it: things that appeared, things that went, and whether the
    whole screen changed.
    """

    def __init__(self, *, width: int = WORKING_WIDTH, kinds: list[Kind] | None = None) -> None:
        self.width = width
        self.scale = 1.0
        self.shape: tuple[int, int] = (0, 0)
        self.things: dict[int, Thing] = {}
        #: Kinds met before in this world keep their numbers, so what was
        #: learned about one in an earlier stretch of play still applies.
        self.kinds: list[Kind] = list(kinds or [])
        # A kind kept from another way of looking (a colour mix of another
        # size) keeps its colour and size, and starts its mix again: mixed
        # with today's, it stopped play dead (offline 2026-10-04, 32 bins
        # against 64).
        for kind in self.kinds:
            if np.shape(kind.look) != (LOOK_BINS,):
                kind.look = np.full(LOOK_BINS, 1.0 / LOOK_BINS)
        self._numbers = 0
        self._first: list[np.ndarray] = []
        self._first_at = -math.inf
        self._backdrop: np.ndarray | None = None
        self._last: np.ndarray | None = None
        self._last_at = 0.0
        self.pictures = 0
        self.screens = 0
        #: Where each thing that went was last seen, for whoever asks what it went beside.
        self.last_box: dict[int, tuple[float, float, float, float]] = {}

    # -- the picture -------------------------------------------------------

    def _smaller(self, picture: np.ndarray) -> np.ndarray:
        from core.perception.picture_arithmetic import shrink

        tall, wide = picture.shape[:2]
        if wide <= self.width:
            self.scale = 1.0
            return picture
        self.scale = self.width / wide
        return shrink(picture, self.width, max(1, round(tall * self.scale)))

    def _first_look(self, small: np.ndarray, at: float) -> bool:
        """Whether the first look at this screen is still going on."""
        if self._backdrop is not None:
            return False
        if not self._first:
            self._first_at = at
        self._first.append(small)
        if at - self._first_at < FIRST_LOOK_S and len(self._first) < 30:
            return True
        self._backdrop = _without_what_stands_out(np.median(np.stack(self._first), axis=0).astype(np.uint8))
        self._first = []
        return False

    def _start_again(self) -> None:
        """A new screen: what was kept of the old one says nothing about this one."""
        self._first = []
        self._backdrop = None
        self.screens += 1

    def _drift(self, small: np.ndarray, dt: float) -> None:
        """Let the backdrop take the picture in, except under the things she is following."""
        if self._backdrop is None:
            return
        held = np.zeros(small.shape[:2], dtype=bool)
        tall, wide = small.shape[:2]
        for thing in self.things.values():
            left, top, right, bottom = thing.box()
            held[max(0, int(top) - 2) : min(tall, int(bottom) + 3), max(0, int(left) - 2) : min(wide, int(right) + 3)] = True
        rate = min(1.0, BACKDROP_DRIFT * max(0.0, dt))
        # A weight a pixel, nought under what she follows: the same blend as
        # picking the free pixels out, without copying them out and back.
        weight = np.where(held, np.float32(0.0), np.float32(rate))[..., None]
        self._backdrop += weight * (small.astype(np.float32) - self._backdrop)

    def _what_differs(self, small: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        from core.perception.picture_arithmetic import grow

        if self._backdrop is None:
            return None
        apart = np.abs(small.astype(np.float32) - self._backdrop).max(axis=2)
        strict = (apart > DIFFERENT_ENOUGH).astype(np.uint8)
        return strict, grow(strict)

    def _blobs(self, small: np.ndarray, masks: tuple[np.ndarray, np.ndarray]) -> list[dict[str, Any]]:
        from core.perception.picture_arithmetic import pieces

        strict, mask = masks
        count, labels, stats, _centres = pieces(mask)
        area_of_all = mask.shape[0] * mask.shape[1]
        found = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < SMALLEST or area > LARGEST * area_of_all:
                continue
            inside = (labels[y : y + h, x : x + w] == label) & (strict[y : y + h, x : x + w] > 0)
            if int(inside.sum()) < SMALLEST:
                continue
            if self._a_ghost(small, x, y, w, h, inside):
                continue
            rows, cols = np.nonzero(inside)
            top, left = y + int(rows.min()), x + int(cols.min())
            bottom, right = y + int(rows.max()) + 1, x + int(cols.max()) + 1
            pixels = small[y : y + h, x : x + w][inside]
            found.append({
                "x": (left + right) / 2.0, "y": (top + bottom) / 2.0,
                "w": float(right - left), "h": float(bottom - top),
                "look": _look_of(pixels),
                "colour": self._colour_of(small, x, y, w, h, inside, pixels),
                "patch": small[top:bottom, left:right].copy(),
            })
        return found

    def _colour_of(self, small: np.ndarray, x: int, y: int, w: int, h: int, inside: np.ndarray, pixels: np.ndarray) -> tuple[int, int, int]:
        """A thing's own colour: that of its pixels least mixed with the floor.

        A thing a few pixels across, shrunk to the working size, is mostly
        edge, and an edge pixel is the thing's colour blended with the floor's
        by however much of it the thing covered. The median of all of them
        drifts as it moves across pixel boundaries: offline 2026-10-04 a white
        ball was filed as white one moment and grey the next, and what she
        learned of one kind was never applied to the other.
        """
        floor = self._backdrop[y : y + h, x : x + w][inside]
        apart = np.abs(pixels.astype(np.float32) - floor).max(axis=1)
        core = pixels[apart >= 0.75 * float(apart.max())]
        return tuple(int(c) for c in np.median(core, axis=0))

    def _a_ghost(self, small: np.ndarray, x: int, y: int, w: int, h: int, inside: np.ndarray) -> bool:
        """Whether a difference is the backdrop being wrong rather than something being there.

        Where a thing that stood still during the first look has since moved
        away, the backdrop still shows it and the picture shows floor. What is
        there now looks like what is around it now, and nothing on screen is a
        thing that looks like its own surroundings. The backdrop is put right.
        """
        tall, wide = small.shape[:2]
        top, bottom = max(0, y - 2), min(tall, y + h + 2)
        left, right = max(0, x - 2), min(wide, x + w + 2)
        ring = np.ones((bottom - top, right - left), dtype=bool)
        ring[y - top : y - top + h, x - left : x - left + w] = False
        if not ring.any():
            return False
        around = np.median(small[top:bottom, left:right][ring].astype(np.float32), axis=0)
        now = small[y : y + h, x : x + w][inside].astype(np.float32)
        if float(np.abs(np.median(now, axis=0) - around).max()) >= DIFFERENT_ENOUGH / 2:
            return False
        region = self._backdrop[y : y + h, x : x + w]
        region[inside] = now
        return True

    # -- following things --------------------------------------------------

    def _cost(self, thing: Thing, blob: dict[str, Any], at: float) -> float:
        dt = max(1e-3, at - thing.seen)
        px, py = _measured(thing)
        px, py = px + thing.vx * dt, py + thing.vy * dt
        reach = 6.0 + 1.5 * math.hypot(thing.vx, thing.vy) * dt + 0.5 * max(thing.w, thing.h)
        distance = math.hypot(blob["x"] - px, blob["y"] - py) / reach
        if distance > 1.0:
            return math.inf
        looks = _look_apart(thing.look, blob["look"])
        if looks > 0.8:
            return math.inf
        # Equal area and colour do not make a tall object a wide one.
        # Otherwise a control can retain its identity when replaced by text.
        aspect = (blob["w"] / max(1.0, blob["h"])) / (thing.w / max(1.0, thing.h))
        if max(aspect, 1.0 / max(1e-6, aspect)) > 2.5:
            return math.inf
        ratio = max(blob["w"] * blob["h"], thing.size) / max(1.0, min(blob["w"] * blob["h"], thing.size))
        return distance + looks + 0.3 * math.log(ratio)

    def _match(self, blobs: list[dict[str, Any]], at: float) -> tuple[dict[int, int], list[int]]:
        pairs = []
        for number, thing in self.things.items():
            for index, blob in enumerate(blobs):
                cost = self._cost(thing, blob, at)
                if cost < math.inf:
                    pairs.append((cost, number, index))
        pairs.sort()
        matched: dict[int, int] = {}
        taken: set[int] = set()
        for _cost, number, index in pairs:
            if number in matched or index in taken:
                continue
            matched[number] = index
            taken.add(index)
        return matched, [index for index in range(len(blobs)) if index not in taken]

    def _moved(self, thing: Thing, blob: dict[str, Any], at: float) -> None:
        dt = max(1e-3, at - thing.seen)
        x0, y0 = _measured(thing)
        vx, vy = (blob["x"] - x0) / dt, (blob["y"] - y0) / dt
        blend = 0.5 if thing.seen > thing.born else 1.0
        thing.vx = (1 - blend) * thing.vx + blend * vx
        thing.vy = (1 - blend) * thing.vy + blend * vy
        thing.x, thing.y, thing.w, thing.h = blob["x"], blob["y"], blob["w"], blob["h"]
        thing.look = 0.8 * thing.look + 0.2 * blob["look"]
        thing.colour, thing.patch = blob["colour"], blob["patch"]
        thing.seen, thing.still = at, False
        thing.path.append((at, thing.x, thing.y))
        thing.vx, thing.vy = _speed_from_its_path(thing.path, at, (vx, vy), (thing.vx, thing.vy))
        thing.sizes.append(thing.size)
        if len(thing.sizes) % 5 == 0:
            self._kind_again(thing)
        if not thing.moved:
            _at, x0, y0 = thing.path[0]
            thing.moved = math.hypot(thing.x - x0, thing.y - y0) > 2.0

    def _still_there(self, thing: Thing, small: np.ndarray) -> bool:
        """Whether a thing nobody saw move is still where it was last seen, by its look."""
        x, y = _measured(thing)
        left, top, right, bottom = (int(round(v)) for v in (x - thing.w / 2, y - thing.h / 2, x + thing.w / 2, y + thing.h / 2))
        tall, wide = small.shape[:2]
        if left < 0 or top < 0 or right > wide or bottom > tall or right <= left or bottom <= top:
            return False
        here = small[top:bottom, left:right]
        if here.shape != thing.patch.shape or self._backdrop is None:
            return False
        if float(np.abs(here.astype(np.int16) - thing.patch.astype(np.int16)).mean()) >= 25.0:
            return False
        # And it is something: a patch that matches itself and the floor is floor.
        floor = self._backdrop[top:bottom, left:right]
        return float(np.abs(here.astype(np.float32) - floor).max(axis=2).mean()) > DIFFERENT_ENOUGH / 2

    def _kind_for(self, look: np.ndarray, size: float, colour: tuple[int, int, int]) -> int:
        best, apart = -1, KIND_ALIKE
        for kind in self.kinds:
            distance = kind.like(colour, size)
            if distance < apart:
                best, apart = kind.number, distance
        # Among kinds it plainly is (its colour, and a size well within half
        # again), the one most seen. A kind made from a ball half out from
        # behind the edge is alike enough to draw the whole ball the next
        # time it is born: offline 2026-10-04 one ball was two kinds, and what
        # was learned of one was not applied to the other.
        plainly = [kind for kind in self.kinds if kind.like(colour, size) < KIND_ALIKE and max(size, kind.size) / max(1e-6, min(size, kind.size)) < PLAINLY_THE_SAME_SIZE]
        if plainly:
            best = max(plainly, key=lambda kind: kind.seen).number
        if best < 0:
            best = len(self.kinds)
            self.kinds.append(Kind(best, look.copy(), size, colour))
        kind = self.kinds[best]
        kind.seen += 1
        kind.look = 0.9 * kind.look + 0.1 * look
        kind.size = 0.9 * kind.size + 0.1 * size
        return best

    def _kind_again(self, thing: Thing) -> None:
        """Look again at what kind a thing is, now its size has settled.

        A thing first seen half out from behind the backdrop is a sliver of
        itself: LIVE-like offline 2026-10-03, a paddle first seen the size of
        the ball was filed with the ball, and the ball was left out of
        everything she aimed at because it looked like her.
        """
        size = float(np.median(np.asarray(thing.sizes)))
        kind = self.kinds[thing.kind] if 0 <= thing.kind < len(self.kinds) else None
        if kind is not None and kind.like(thing.colour, size) < KIND_ALIKE:
            return
        thing.kind = self._kind_for(thing.look, size, thing.colour)

    def _born(self, blob: dict[str, Any], at: float) -> Thing:
        self._numbers += 1
        thing = Thing(
            self._numbers, blob["x"], blob["y"], blob["w"], blob["h"], blob["look"],
            blob["colour"], blob["patch"], born=at, seen=at,
        )
        thing.kind = self._kind_for(thing.look, thing.size, thing.colour)
        thing.path.append((at, thing.x, thing.y))
        self.things[thing.number] = thing
        return thing

    # -- one picture -------------------------------------------------------

    def see(self, picture: np.ndarray, at: float) -> list[dict[str, Any]]:
        """Take one picture in. Returns what happened in it."""
        small = self._smaller(np.asarray(picture))
        self.shape = small.shape[:2]
        happened: list[dict[str, Any]] = []
        if self._a_new_screen(small):
            self._start_again()
            happened.append({"what": "new screen", "at": at})
            happened.extend(what_happened("gone", thing, at) for thing in self.things.values())
            self.last_box.update({number: thing.box() for number, thing in self.things.items()})
            self.things.clear()
        dt = at - self._last_at if self._last is not None else 0.0
        self._last, self._last_at = small, at
        self.pictures += 1
        if self._first_look(small, at):
            return happened
        masks = self._what_differs(small)
        blobs = self._blobs(small, masks) if masks is not None else []
        matched, fresh = self._match(blobs, at)
        for number, index in matched.items():
            self._moved(self.things[number], blobs[index], at)
        happened.extend(self._the_unmatched(small, matched, at))
        for index in fresh:
            happened.append(what_happened("appeared", self._born(blobs[index], at), at))
        self._drift(small, dt)
        return happened

    def _a_new_screen(self, small: np.ndarray) -> bool:
        if self._last is None or self._last.shape != small.shape:
            return self._last is not None
        from core.perception.picture_arithmetic import apart

        return float((apart(small, self._last).max(axis=2) > DIFFERENT_ENOUGH).mean()) > NEW_SCREEN

    def _the_unmatched(self, small: np.ndarray, matched: dict[int, int], at: float) -> list[dict[str, Any]]:
        happened = []
        for number in [n for n in self.things if n not in matched]:
            thing = self.things[number]
            if self._still_there(thing, small):
                thing.x, thing.y = _measured(thing)
                thing.vx *= 0.5
                thing.vy *= 0.5
                thing.still = True
                thing.seen = at
                thing.path.append((at, thing.x, thing.y))
                continue
            if at - thing.seen >= GONE_AFTER_S:
                happened.append(what_happened("gone", thing, at))
                self.last_box[number] = thing.box()
                del self.things[number]
                continue
            # Not seen, and not gone yet: where its speed has taken it. A
            # ball that runs into a paddle draws as one blob with it for a
            # picture or two; held where it was last seen, offline 2026-10-04,
            # it was aimed at where it had been and the paddle missed it.
            x0, y0 = _measured(thing)
            thing.x, thing.y = x0 + thing.vx * (at - thing.seen), y0 + thing.vy * (at - thing.seen)
        return happened

    # -- for anyone outside --------------------------------------------------

    def share(self, x: float, y: float) -> tuple[float, float]:
        tall, wide = self.shape
        return x / max(1, wide), y / max(1, tall)

    def moving(self, faster_than: float = 4.0) -> list[Thing]:
        """Things going faster than ``faster_than`` working pixels a second."""
        return [t for t in self.things.values() if math.hypot(t.vx, t.vy) > faster_than]
