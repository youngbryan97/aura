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

#: Pictures a thing must have gone as the view goes before it is taken back into the view.
SCENERY_AFTER = 4

#: How long after the view last moved it is taken to be going by still, in seconds.
VIEW_GOING_FOR_S = 0.5

#: Where the view goes by, how many working pixels a thing's edges are closed over to make it whole.
CLOSED_OVER = 2

#: How far, in working pixels on average, a thing's places may lie off one straight line for it to have had one
#: going over them.
ONE_GOING_PX = 1.0

#: How far, in working pixels, a layer of the view must go over a stretch of a thing's path for going with it to be
#: told from standing still.
TOLD_FROM_STANDING_PX = 4.0

#: How much bigger than its own size a followed thing has grown when part of it is something it left behind.
LEFT_BEHIND_GROWTH = 1.6

#: How long a pixel inside such a thing stays unchanged before it is taken for what it left behind, in seconds.
LEFT_BEHIND_S = 0.25

#: A thing missing this long, and not found by its look, has gone.
GONE_AFTER_S = 0.25
#: How long a moving thing that went out of sight is kept, with where its going was taking it, so that a thing of its
#: kind coming back into sight near there is that thing again: a figure that walks behind a pillar and out the other
#: side, a ball under a cup. Infants expect a hidden thing to come out where its path leads (Baillargeon's
#: violation-of-expectation studies); the models of it track a thing through occlusion (ADEPT, Smith et al., 2019).
OUT_OF_SIGHT_S = 4.0


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

#: How far, in its own sizes, a thing may go between two pictures when it sets off suddenly (a jump, a dash) and
#: still be followed as itself, and how alike in look it must be to be: a jump of a 22-pixel body at 430 pixels a
#: second is two of its sizes in a picture at 15 a second.
LEAP_SIZES = 4.0
SAME_LOOK = 0.4


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
    #: Its own size: the least it has measured, over a few pictures at a time, while followed.
    least: float = math.inf
    #: Whether anyone has been told it appeared: while the view moves, not until it has shown it is not scenery.
    announced: bool = True
    #: Whether it is a thing that went out of sight and came back into it (OUT_OF_SIGHT_S).
    again: bool = False

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


#: How many of a compact thing's reach a thin bar standing out from its floor may run and still be a thing.
THIN_BAR_REACHES = 4


def _without_what_stands_out(still: np.ndarray) -> np.ndarray:
    """The first backdrop, with the small things that stand out from their floor taken off it.

    A coin that never moves is in every picture of the first look, so the
    middle value of each pixel keeps it. What tells it from the floor is that
    it is small and unlike what is around it: the floor is what a wide median
    filter leaves, and anything compact that differs from that by a clear
    margin is put back to floor, so that it shows as a thing from the start.
    So is a thin bar a few times as long: offline 2026-10-09 a landing pad that
    never moved was part of the backdrop, and could be neither seen nor gone to.
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
        # Compact, or a thin bar a few times as long (a pad to land on, a ledge, a goal line): a thing. A strip
        # running much of the way across (a floor, a wall, the ground) is the place, not a thing in it.
        if area < SMALLEST or min(w, h) > reach or max(w, h) > THIN_BAR_REACHES * reach:
            continue
        inside = labels == label
        backdrop[inside] = floor[inside]
    return backdrop


def _back_in_sight(moves: WhatMoves, thing: Thing, at: float) -> Thing | None:
    """The moving thing of this one's kind that went out of sight lately and was headed for about where this one has
    come into it, the nearest such; None where none was."""
    best, nearest = None, math.inf
    for number, (gone, when) in list(moves.out_of_sight.items()):
        away = at - when
        if away > OUT_OF_SIGHT_S:
            del moves.out_of_sight[number]
            continue
        if gone.kind != thing.kind or number in moves.things or number in moves.pending:
            continue
        x, y = gone.x + gone.vx * away, gone.y + gone.vy * away
        off = math.hypot(thing.x - x, thing.y - y)
        # Its own size twice over, and half of how far it would have gone: a thing hidden longer may have turned.
        if off <= 2 * max(gone.w, gone.h) + 0.5 * math.hypot(gone.vx, gone.vy) * away and off < nearest:
            best, nearest = number, off
    return moves.out_of_sight.pop(best)[0] if best is not None else None


def what_happened(kind: str, thing: Thing, at: float) -> dict[str, Any]:
    return {"what": kind, "thing": thing.number, "kind": thing.kind, "x": thing.x, "y": thing.y,
            "vx": thing.vx, "vy": thing.vy, "width": thing.w, "height": thing.h, "at": at,
            "last_visible": list(_measured(thing)), "last_seen_at": thing.seen, "moved": thing.moved,
            "again": bool(getattr(thing, "again", False))}


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


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
        #: Things born while the view moved and not yet shown not to be scenery: followed, and kept from everyone else.
        self.pending: dict[int, Thing] = {}
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
        self._still: np.ndarray | None = None
        self.pictures = 0
        self.screens = 0
        #: Where each thing that went was last seen, for whoever asks what it went beside.
        self.last_box: dict[int, tuple[float, float, float, float]] = {}
        #: Moving things gone out of sight lately, with when: a thing of the kind coming back near where its going was
        #: taking it is the same thing (OUT_OF_SIGHT_S).
        self.out_of_sight: dict[int, tuple[Thing, float]] = {}
        #: How the view itself last moved (core/perception/how_the_scenery_goes_by.py), and what turns its layers'
        #: shifts into pixels a second.
        self.view_moved: Any = None
        #: When the view last moved, and how: the view going by is still going by in a picture that repeats the last.
        self.view_moved_at = -math.inf
        self.view_going: Any = None
        #: How many pictures drawn anew the view's last measure spans, where this picture was one (else 0): its going,
        #: added up a picture at a time, is where the view has gone (core/perception/where_in_a_world.py).
        self.view_over = 0
        #: The last few pictures that were drawn anew, with when: the view is measured against the oldest.
        self._drawn: deque = deque(maxlen=3)
        self._now = -math.inf

    # -- the picture -------------------------------------------------------

    def _smaller(self, picture: np.ndarray) -> np.ndarray:
        from core.perception.picture_arithmetic import shrink

        tall, wide = picture.shape[:2]
        if wide <= self.width:
            self.scale = 1.0
            return picture
        self.scale = self.width / wide
        return shrink(picture, self.width, max(1, round(tall * self.scale)))

    @property
    def has_looked(self) -> bool:
        """Whether the first look at this screen is over, and what differs from it is followed."""
        return self._backdrop is not None

    @property
    def seeing(self) -> bool:
        """Whether the first look at this screen is done, so what happens on it can be told from what was there."""
        return self._backdrop is not None

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
        self._drawn.clear()
        self.screens += 1

    def _drift(self, small: np.ndarray, dt: float) -> None:
        """Let the backdrop take the picture in, except under the things she is following, and what they leave behind.

        Under a thing she follows the backdrop is held, so a thing at rest does
        not fade into the scene. But a thing that leaves marks (paint, a pen's
        line, footprints) is one patch with them, and held whole the patch only
        grows: LIVE 2026-10-07 a painter in a game was a blob 31 pixels wide at
        rest and 143 by 126 after a few seconds of painting, whose middle barely
        moved under the keys, and no key was ever found to move it. Where a
        thing has grown well past its own size, what in it has stood still is
        what it left behind, and is scene at once.
        """
        if self._backdrop is None:
            return
        held = np.zeros(small.shape[:2], dtype=bool)
        left_behind = np.zeros(small.shape[:2], dtype=bool)
        tall, wide = small.shape[:2]
        still = self._still if self._still is not None and self._still.shape == held.shape else None
        for thing in [*self.things.values(), *self.pending.values()]:
            left, top, right, bottom = thing.box()
            area = (slice(max(0, int(top) - 2), min(tall, int(bottom) + 3)), slice(max(0, int(left) - 2), min(wide, int(right) + 3)))
            if self._goes_with_the_view(thing, dt):
                # What goes as the view goes is the view: an edge of the scenery the backdrop was laid a part of a
                # pixel off. Held, it is held wrong, and it grows.
                continue
            if still is not None and thing.size > LEFT_BEHIND_GROWTH * thing.least:
                stood = still[area] >= LEFT_BEHIND_S
                held[area] |= ~stood
                left_behind[area] |= stood
            else:
                held[area] = True
        rate = min(1.0, BACKDROP_DRIFT * max(0.0, dt))
        scenery = getattr(self.view_moved, "scenery", None)
        if self.view_is_moving and scenery is not None and scenery.shape == held.shape:
            # What went by with the view is the backdrop as it is now, but for what she follows.
            left_behind |= scenery
        # A weight a pixel, nought under what she follows: the same blend as
        # picking the free pixels out, without copying them out and back.
        weight = np.where(held, np.float32(0.0), np.where(left_behind, np.float32(1.0), np.float32(rate)))[..., None]
        self._backdrop += weight * (small.astype(np.float32) - self._backdrop)

    @property
    def view_is_moving(self) -> bool:
        """Whether the view has moved lately: a game draws more slowly than she looks, and between its pictures the
        view stands still for one of hers while it is going by."""
        return self._now - self.view_moved_at <= VIEW_GOING_FOR_S

    def _back_into_the_view(self, dt: float) -> None:
        """Things that have gone as the view goes for a few pictures were scenery: they go back into it, unsaid.

        Not said to have gone, because nothing went: what she learns from things
        going (a target hit, a ball let by) must not be learned from a roof.
        """
        if not self.view_is_moving:
            return
        for kept in (self.pending, self.things):
            for number in [n for n, t in kept.items() if self._goes_with_the_view(t, dt) and len(t.path) >= SCENERY_AFTER]:
                del kept[number]

    def _shown_not_scenery(self, dt: float) -> list[dict[str, Any]]:
        """Things born while the view moved that have been followed long enough without going as it goes: they
        appeared, as of when they were first seen."""
        shown = []
        for number, thing in list(self.pending.items()):
            if len(thing.path) >= SCENERY_AFTER and not self._goes_with_the_view(thing, dt):
                thing.announced = True
                self.things[number] = self.pending.pop(number)
                shown.append(what_happened("appeared", thing, thing.born))
        return shown

    def _goes_with_the_view(self, thing: Thing, dt: float) -> bool:
        """Whether a thing has gone as one of the view's layers where it stands goes, while the view is moving: it is
        part of the view. LIVE 2026-10-09 the fences nearest the eye went by faster than the town behind them and were
        too small a share of their band to be a layer of it; they are a layer of the whole picture.

        Its going is taken over the last few pictures of its path, as one straight line, not its last step; it must
        be nearer the layer's going than standing still, and the layer must have gone far enough over that time to
        be told from standing: offline 2026-10-09 a hero that stopped in front of hills going by at a pixel every
        other picture, and then set off, was taken for the hills, and lost.
        """
        moved = self.view_moved
        if not self.view_is_moving or self.view_going is None or not hasattr(moved, "ways_going"):
            return False
        recent = [step for step in thing.path if step[0] >= thing.path[-1][0] - VIEW_GOING_FOR_S] if thing.path else []
        if len(recent) < SCENERY_AFTER:
            return False
        times = np.array([step[0] for step in recent]) - recent[0][0]
        span = float(times[-1])
        if span <= 0:
            return False
        # Its going as one straight line through where it was: a thing that stood and then set off has no one going.
        places = np.array([[step[1], step[2]] for step in recent])
        slope, start = np.polyfit(times, places, 1)
        if float(np.sqrt(np.mean((start + np.outer(times, slope) - places) ** 2))) > ONE_GOING_PX:
            return False
        vx, vy = float(slope[0]), float(slope[1])
        tall, wide = self.shape
        for dx, dy in moved.ways_going(thing.y, thing.x, tall, wide):
            lx, ly = dx * self.view_going, dy * self.view_going
            if math.hypot(lx, ly) * span < TOLD_FROM_STANDING_PX:
                continue
            off = math.hypot(vx - lx, vy - ly)
            if off < 0.35 * math.hypot(lx, ly) and off < 0.5 * math.hypot(vx, vy):
                return True
        return False

    def _what_differs(self, small: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        from core.perception.picture_arithmetic import grow

        if self._backdrop is None:
            return None
        apart = np.abs(small.astype(np.float32) - self._backdrop).max(axis=2)
        strict = (apart > DIFFERENT_ENOUGH).astype(np.uint8)
        moved = self.view_moved
        if self.view_is_moving and getattr(moved, "going_by", None) is not None and moved.going_by.shape == strict.shape:
            # Where the view goes by, a thing is what no layer of the view lays over: moving its own way, or holding
            # its place on the screen while the view goes by under it. What stood still is told by the backdrop, as
            # anywhere: a thing that came and stopped there is unlike it.
            # A thing no layer lays over is told at its edges, for a flat patch is laid over by any layer: it is closed
            # over, and its pieces a few pixels apart are one.
            from scipy import ndimage

            own = ndimage.binary_closing(moved.on_its_own, structure=np.ones((3, 3), bool), iterations=CLOSED_OVER)
            strict = np.where(moved.going_by, own | (moved.standing & strict.astype(bool)), strict.astype(bool)).astype(np.uint8)
            near = ndimage.binary_dilation(strict, structure=np.ones((3, 3), bool), iterations=CLOSED_OVER + 1)
            return strict, (grow(strict).astype(bool) | (near & moved.going_by)).astype(np.uint8)
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
        for number, thing in {**self.things, **self.pending}.items():
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
        # A thing that set off faster than it was going is still itself: what went from here and what came there
        # look the same. LIVE-like 2026-10-09 a body standing still that jumped went 15 working pixels between
        # pictures against a reach of 11, was taken for a new thing at every jump, and no press was ever seen to move it.
        leaps = sorted((cost, number, index)
                       for number, thing in {**self.things, **self.pending}.items() if number not in matched
                       for index, blob in enumerate(blobs) if index not in taken
                       if (cost := self._leap_cost(thing, blob, at)) < math.inf)
        for _cost, number, index in leaps:
            if number in matched or index in taken:
                continue
            matched[number] = index
            taken.add(index)
        return matched, [index for index in range(len(blobs)) if index not in taken]

    def _leap_cost(self, thing: Thing, blob: dict[str, Any], at: float) -> float:
        """How far a blob is from where a thing was last seen, in reaches of a sudden move; inf where it is not the
        thing: farther than a few of its sizes, or not of the same look, size and shape."""
        dt = max(1e-3, at - thing.seen)
        px, py = _measured(thing)
        reach = 6.0 + 1.5 * math.hypot(thing.vx, thing.vy) * dt + LEAP_SIZES * max(thing.w, thing.h)
        distance = math.hypot(blob["x"] - px, blob["y"] - py) / reach
        looks = _look_apart(thing.look, blob["look"])
        aspect = (blob["w"] / max(1.0, blob["h"])) / (thing.w / max(1.0, thing.h))
        ratio = max(blob["w"] * blob["h"], thing.size) / max(1.0, min(blob["w"] * blob["h"], thing.size))
        if distance > 1.0 or looks > SAME_LOOK or max(aspect, 1.0 / max(1e-6, aspect)) > PLAINLY_THE_SAME_SIZE or ratio > PLAINLY_THE_SAME_SIZE:
            return math.inf
        return distance + looks

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
        if len(thing.sizes) >= 5:
            thing.least = min(thing.least, float(np.median(list(thing.sizes)[-5:])))
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
        before = _back_in_sight(self, thing, at)
        if before is not None:
            # The same thing, come back into sight where its going was taking it: its number, its path and its way.
            thing.number, thing.path, thing.born, thing.moved = before.number, before.path, before.born, True
            thing.vx, thing.vy, thing.again = before.vx, before.vy, True
        thing.path.append((at, thing.x, thing.y))
        thing.least = thing.size  # as it first stands out, before it has left anything behind
        thing.announced = not self.view_is_moving
        (self.things if thing.announced else self.pending)[thing.number] = thing
        return thing

    # -- one picture -------------------------------------------------------

    def see(self, picture: np.ndarray, at: float) -> list[dict[str, Any]]:
        """Take one picture in. Returns what happened in it."""
        small = self._smaller(np.asarray(picture))
        self.shape = small.shape[:2]
        happened: list[dict[str, Any]] = []
        self._now = at
        if self._with_the_view(small):
            return happened
        if self._a_new_screen(small):
            self._start_again()
            happened.append({"what": "new screen", "at": at})
            happened.extend(what_happened("gone", thing, at) for thing in self.things.values() if thing.announced)
            self.last_box.update({number: thing.box() for number, thing in self.things.items()})
            self.things.clear()
            self.pending.clear()
            self.out_of_sight.clear()
        dt = at - self._last_at if self._last is not None else 0.0
        # How long each pixel has stood unchanged, picture to picture.
        if self._last is not None and self._last.shape == small.shape:
            changed = np.abs(small.astype(np.int16) - self._last.astype(np.int16)).max(axis=2) > DIFFERENT_ENOUGH // 2
            before = self._still if self._still is not None and self._still.shape == changed.shape else np.zeros(changed.shape, np.float32)
            self._still = np.where(changed, np.float32(0.0), before + np.float32(dt))
        else:
            self._still = np.zeros(small.shape[:2], np.float32)
        self._last, self._last_at = small, at
        self.pictures += 1
        if self._first_look(small, at):
            return happened
        masks = self._what_differs(small)
        blobs = self._blobs(small, masks) if masks is not None else []
        scene = getattr(picture, "drawing_scene", None)
        if scene is not None and masks is not None:
            blobs = self._drawn_objects(scene, small, masks[0])
        matched, fresh = self._match(blobs, at)
        for number, index in matched.items():
            self._moved(self.things.get(number) or self.pending[number], blobs[index], at)
        self._back_into_the_view(dt)
        happened.extend(self._the_unmatched(small, matched, at))
        for index in fresh:
            thing = self._born(blobs[index], at)
            if thing.announced:
                happened.append(what_happened("appeared", thing, at))
        happened.extend(self._shown_not_scenery(dt))
        self._drift(small, dt)
        return happened

    def _with_the_view(self, small: np.ndarray) -> bool:
        """How the view itself moved: where it goes by, the scenery going by is not things coming and going
        (core/perception/how_the_scenery_goes_by.py). Returns whether the picture is the last one again while the
        view is going by: nothing has been drawn, and nothing happened.

        The view is measured against the picture drawn two before this one,
        not the last: scenery going by at a part of a pixel a picture is laid
        over by standing still as well as by its own going, and over two
        pictures it has gone far enough to tell. A game draws more slowly than
        she looks: LIVE 2026-10-09 every other picture of a scrolling town was
        the one before, and measured against it the view stood still half the
        time.
        """
        from core.perception.how_the_scenery_goes_by import (
            STILL,
            how_the_view_moved,
            the_same_picture,
        )

        self.view_over = 0
        if self._drawn and the_same_picture(self._drawn[-1][1], small):
            if self.view_is_moving:
                return True
        else:
            self._drawn.append((self._now, small))
            # A picture drawn anew: how many drawn pictures its measure spans, for whoever adds the view's going up.
            self.view_over = len(self._drawn) - 1
        self.view_moved = STILL
        if len(self._drawn) < 2:
            return False
        then, before = self._drawn[0]
        measured = how_the_view_moved(before, small)
        if measured.still:
            return False
        # Layers are measured in pixels over the pictures between; this turns them into pixels a second.
        self.view_going = 1.0 / max(1e-3, self._now - then)
        self.view_moved, self.view_moved_at = measured, self._now
        return False

    def _drawn_objects(self, scene: dict[str, Any], small: np.ndarray, foreground: np.ndarray) -> list[dict[str, Any]]:
        """Exact painted bounds for foreground objects, checked against their pixels.

        The backdrop still distinguishes scenery from objects. The display list
        supplies their geometry, including two objects whose pixels touch.
        Text is supplied separately by the renderer and never becomes a body.
        """
        from PIL import ImageColor

        tall, wide = self.shape
        found = []
        for item in scene["objects"]:
            x, y, w, h = (float(item[key]) * self.scale for key in ("x", "y", "width", "height"))
            if w * h < SMALLEST or w * h > LARGEST * wide * tall:
                continue
            left, top = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
            right, bottom = min(wide, int(math.ceil(x + w))), min(tall, int(math.ceil(y + h)))
            if right <= left or bottom <= top:
                continue
            try:
                colour = ImageColor.getrgb(item["colour"])
            except (KeyError, TypeError, ValueError):
                continue
            patch = small[top:bottom, left:right]
            painted = np.abs(patch.astype(np.int16) - np.asarray(colour)).max(axis=2) < 50
            # Geometry describes the paint, but later paint may have covered it.
            if int(painted.sum()) < SMALLEST:
                continue
            visible = foreground[top:bottom, left:right] & painted
            held = any(_boxes_overlap(t.box(), (x, y, x + w, y + h))
                       and max(abs(a - b) for a, b in zip(t.colour, colour, strict=True)) < 45
                       for t in self.things.values())
            if int(visible.sum()) < SMALLEST and not held:
                continue
            found.append({"x": x + w / 2, "y": y + h / 2, "w": w, "h": h,
                          "look": _look_of(np.asarray([colour])), "colour": colour, "patch": patch.copy()})
        return found

    def _a_new_screen(self, small: np.ndarray) -> bool:
        if self._last is None or self._last.shape != small.shape:
            return self._last is not None
        from core.perception.picture_arithmetic import apart

        return float((apart(small, self._last).max(axis=2) > DIFFERENT_ENOUGH).mean()) > NEW_SCREEN

    def _the_unmatched(self, small: np.ndarray, matched: dict[int, int], at: float) -> list[dict[str, Any]]:
        happened = []
        for number in [n for n in [*self.things, *self.pending] if n not in matched]:
            kept = self.things if number in self.things else self.pending
            thing = kept[number]
            if self._still_there(thing, small):
                thing.x, thing.y = _measured(thing)
                thing.vx *= 0.5
                thing.vy *= 0.5
                thing.still = True
                thing.seen = at
                thing.path.append((at, thing.x, thing.y))
                continue
            if at - thing.seen >= GONE_AFTER_S:
                if thing.announced:
                    happened.append(what_happened("gone", thing, at))
                    self.last_box[number] = thing.box()
                    if thing.moved:
                        self.out_of_sight[number] = (thing, at)
                del kept[number]
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
