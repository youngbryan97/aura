"""How the view itself moved between two pictures: the scenery going by in layers, and what each pixel did.

A person watching a window that pans, a page that scrolls or a world that
slides past a flying hero does not see every house and cloud as a new thing
coming and going. They see the view moving, and the few things moving in it.

What moves in a picture is found against a backdrop of what stays put
(core/perception/what_moves_in_the_picture.py). Where the view moves, nothing
stays put: LIVE 2026-10-09 a game slid its town past her three heroes, and in
six seconds of it 266 things appeared and 242 went, one of them the whole
street; no thing answered to her keys, and she never found herself.

Scenery goes by in layers (far hills slower than the road, the road slower
than the trees in front of it), and one stretch of the picture holds more than
one: LIVE 2026-10-09 the roofs at the front of a town went by faster than the
houses behind them, and laid over by one shift, every edge of either was a
thing. So the view is measured as layers (Wang and Adelson's "Representing
moving images with layers", 1994), pixel by pixel:

1. For every shift the view could have gone between two pictures, how well
   each pixel of the new picture is laid over by the old one moved by it,
   with Birchfield and Tomasi's measure, which forgives the half pixel that
   scenery going by a part of a pixel a picture leaves at every edge.
2. A shift is a layer of a band of the picture where it lays over a good
   share of the pixels that tell one shift from another (a flat patch of sky
   is laid over by any). The few things moving on their own are too small a
   share of any band to be a layer. Standing still is a layer too where
   enough stands still.
3. Where a layer is going by, a pixel one of its band's layers lays over is
   scenery. One that none of them lays over is a thing: moving its own way,
   or holding its place on the screen while the view goes by under it (a
   score, a frame, a hero the view follows).

Nothing here knows what is being looked at. A still view measures as still.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["STILL", "ViewMoved", "how_the_view_moved", "the_same_picture"]

#: The furthest, in working pixels, the view may go between the two pictures it is measured over, along either axis.
#: Scenery that goes further than this goes by too fast to be told from a new screen.
MOST = 16

#: Bands across (each with layers going sideways) and down (each with layers going up or down).
BANDS = 8

#: The least share of a band's telling pixels one shift must lay over for that shift to be a layer of the band.
LAYER_SHARE = 0.15

#: The least share of all the picture's telling pixels one shift must lay over to be a layer wherever it goes by:
#: the trees nearest the eye go by fastest and are a small share of any one band they cross.
LAYER_SHARE_OF_ALL = 0.08

#: The least share of a band's telling pixels a layer of the whole picture must lay over to be one of the band's.
SOME_OF_A_BAND = 0.04

#: How far, in grey levels averaged over a pixel and its neighbours, a pixel may be from the span its like in the
#: other picture takes within half a pixel, and still be laid over by it.
LAID_OVER = 10.0

#: How much better than the shifts on average the best must lay a pixel over for the pixel to tell one shift from
#: another: in a flat patch every shift lays over as well as any other.
TELLING = 8.0

#: The furthest, in working pixels, one edge of a thing of one colour may be from where its other edge is
#: looked for across it.
ACROSS_A_THING = 12

#: Into how many parts a band is cut along its length, and the share of them a layer's pixels must be found in: the
#: view going by goes by all along a band.
SPREAD_OVER = 8
SPREAD = 0.6

#: The least share of a picture's pixels that must tell shifts apart for its layers to be measured at all.
ENOUGH_TELLING = 0.01

#: How far apart, in grey levels, two pictures' pixels may be with neither picture having been drawn anew.
REDRAWN = 12


@dataclass(frozen=True)
class ViewMoved:
    """How the view moved between two pictures.

    ``across`` holds the layers of each band across (how far each went sideways), ``down`` those of each band down
    (how far up or down), in whole working pixels between the two pictures; a layer at 0 is scenery standing still.
    ``everywhere`` holds the moving layers of the whole picture, as (across, down) shifts. ``going_by`` marks where
    a moving layer is; ``scenery`` the pixels there a moving layer lays over (a flat patch among them, which any
    lays over); ``on_its_own`` the pixels there no layer of theirs lays over: a thing moving its own way, or holding
    its place on the screen while the view goes by under it; ``standing`` the pixels there that only standing still
    lays over, which the view going by says nothing of.
    """

    across: tuple[tuple[int, ...], ...] = ()
    down: tuple[tuple[int, ...], ...] = ()
    everywhere: tuple[tuple[int, int], ...] = ()
    #: For each band across, every sideways shift that may lay over a pixel in it; for each band down, every up or
    #: down shift: its own layers, those of the bands beside it, and those of the whole picture some of it goes by.
    across_ways: tuple[tuple[int, ...], ...] = ()
    down_ways: tuple[tuple[int, ...], ...] = ()
    scenery: np.ndarray | None = field(default=None, compare=False)
    on_its_own: np.ndarray | None = field(default=None, compare=False)
    standing: np.ndarray | None = field(default=None, compare=False)
    going_by: np.ndarray | None = field(default=None, compare=False)

    @property
    def still(self) -> bool:
        return not any(self.across) and not any(self.down)

    def ways_going(self, y: float, x: float, tall: int, wide: int) -> list[tuple[int, int]]:
        """Every way the view goes by at a place, as (across, down) shifts: the moving layers that may lay over a
        pixel there."""
        row = self.across_ways[min(len(self.across_ways) - 1, max(0, int(y * len(self.across_ways) / max(1, tall))))] if self.across_ways else ()
        col = self.down_ways[min(len(self.down_ways) - 1, max(0, int(x * len(self.down_ways) / max(1, wide))))] if self.down_ways else ()
        return [(dx, 0) for dx in row if dx] + [(0, dy) for dy in col if dy]


STILL = ViewMoved()


def the_same_picture(before: np.ndarray | None, now: np.ndarray) -> bool:
    """Whether ``now`` is ``before`` again: a thing drawn more slowly than she looks shows the same picture twice."""
    if before is None or before.shape != now.shape:
        return False
    return int(np.abs(before.astype(np.int16) - now.astype(np.int16)).max()) <= REDRAWN


def _grey(picture: np.ndarray) -> np.ndarray:
    picture = picture.astype(np.float32)
    return picture.mean(axis=-1) if picture.ndim == 3 else picture


def _shifted(picture: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``picture`` moved ``dx`` across and ``dy`` down, its edge repeated into what comes into view."""
    out = np.roll(picture, (dy, dx), axis=(0, 1))
    if dy > 0:
        out[:dy] = picture[:1]
    elif dy < 0:
        out[dy:] = picture[-1:]
    if dx > 0:
        out[:, :dx] = picture[:, :1]
    elif dx < 0:
        out[:, dx:] = picture[:, -1:]
    return out


def _apart_within_half_a_pixel(now: np.ndarray, moved: np.ndarray, axis: int) -> np.ndarray:
    """How far each pixel of ``now`` is from the span ``moved`` takes within half a pixel along ``axis``
    (Birchfield and Tomasi, "A pixel dissimilarity measure that is insensitive to image sampling", 1998)."""
    before_it = 0.5 * (moved + np.roll(moved, 1, axis=axis))
    after_it = 0.5 * (moved + np.roll(moved, -1, axis=axis))
    low = np.minimum(np.minimum(moved, before_it), after_it)
    high = np.maximum(np.maximum(moved, before_it), after_it)
    return np.maximum(0.0, np.maximum(now - high, low - now))


def _shifts() -> list[tuple[int, int]]:
    return [(dx, 0) for dx in range(-MOST, MOST + 1)] + [(0, dy) for dy in range(-MOST, MOST + 1) if dy]


def _layers_of_bands(best: np.ndarray, telling: np.ndarray, shifts: list[tuple[int, int]], axis: int, share: float = LAYER_SHARE) -> tuple[list[tuple[int, ...]], list[tuple[int, int]]]:
    """For each band along ``axis`` (0: bands across, going sideways; 1: bands down, going up or down), the shifts
    that lay over at least ``share`` of its telling pixels, and where the band lies."""
    length = best.shape[axis]
    edges = np.linspace(0, length, BANDS + 1).astype(int)
    family = [i for i, (dx, dy) in enumerate(shifts) if (dy == 0 if axis == 0 else dx == 0)]
    layers, bounds = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        band = (slice(a, b), slice(None)) if axis == 0 else (slice(None), slice(a, b))
        tells = telling[band]
        count = int(tells.sum())
        found: list[int] = []
        if count:
            tally = np.bincount(best[band][tells], minlength=len(shifts))
            for index in family:
                if tally[index] >= share * count and _spread_along(best[band] == index, tells, axis):
                    dx, dy = shifts[index]
                    found.append(dx if axis == 0 else dy)
        layers.append(tuple(found))
        bounds.append((int(a), int(b)))
    return layers, bounds


def _spread_along(went: np.ndarray, tells: np.ndarray, axis: int) -> bool:
    """Whether the telling pixels of a band that went one way are spread along the band, as scenery is, and not in one
    place of it, as a thing is: offline 2026-10-09 a figure swaying beside a board was all its band had that moved, and
    the band was taken for scenery going by."""
    along = (went & tells).any(axis=axis)
    parts = np.array_split(along, SPREAD_OVER)
    return sum(bool(part.any()) for part in parts) >= SPREAD * SPREAD_OVER


def _changed_along_a_band(old: np.ndarray, new: np.ndarray) -> bool:
    """Whether any band changed all along its length, as a view going by changes it: where only things moved, a band
    changed in a place or two, and the layers are not measured at all. LIVE 2026-10-10 measuring them on every
    picture of a still screen took play from 17 pictures a second to 10."""
    changed = np.abs(new - old) > REDRAWN
    everywhere = np.ones_like(changed)
    for axis in (0, 1):
        length = changed.shape[axis]
        edges = np.linspace(0, length, BANDS + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            band = changed[a:b] if axis == 0 else changed[:, a:b]
            if band.any() and _spread_along(band, everywhere[a:b] if axis == 0 else everywhere[:, a:b], axis):
                return True
    return False


def _between(inside: np.ndarray, outside: np.ndarray, axis: int) -> np.ndarray:
    """Pixels with a pixel of ``inside`` on either side of them along ``axis``, within ``ACROSS_A_THING`` pixels,
    and no pixel of ``outside`` between."""
    if axis == 0:
        return _between(inside.T, outside.T, axis=1).T
    wide = inside.shape[1]
    places = np.broadcast_to(np.arange(wide), inside.shape)
    nowhere = np.full(inside.shape, -wide * 4)

    def last_seen(mask: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(np.where(mask, places, nowhere), axis=1)

    left_in, left_out = last_seen(inside), last_seen(outside)
    right_in = wide - 1 - last_seen(inside[:, ::-1])[:, ::-1]
    right_out = wide - 1 - last_seen(outside[:, ::-1])[:, ::-1]
    return (
        (left_in > left_out) & (places - left_in <= ACROSS_A_THING)
        & (right_in < right_out) & (right_in - places <= ACROSS_A_THING)
    )


def how_the_view_moved(before: np.ndarray | None, now: np.ndarray) -> ViewMoved:
    """How the view moved from ``before`` to ``now`` (pictures of one size): its layers, and what each pixel did."""
    if before is None or before.shape != now.shape:
        return STILL
    from scipy import ndimage

    old, new = _grey(before), _grey(now)
    tall, wide = new.shape
    if tall <= 4 * MOST or wide <= 4 * MOST:
        return STILL
    if not _changed_along_a_band(old, new):
        return STILL
    shifts = _shifts()
    # Each pixel alone, and each with its neighbours: the neighbours tell which way a pixel went, and the pixel
    # alone whether it went that way; where two layers meet, no one shift lays a pixel and its neighbours over.
    alone = np.stack([_apart_within_half_a_pixel(new, _shifted(old, dx, dy), 1 if dy == 0 else 0) for dx, dy in shifts])
    apart = ndimage.uniform_filter(alone, size=(1, 3, 3), mode="nearest")
    best = apart.argmin(axis=0)
    least = apart.min(axis=0)
    # A pixel tells which shift it went by only where that shift, or one beside it, lays it over and none further off
    # does as well: inside a patch of one colour every shift short of its width lays it over.
    position = np.array([dx + dy for dx, dy in shifts])
    family = np.array([0 if dy == 0 else 1 for _dx, dy in shifts])
    near_best = (family[:, None, None] == family[best][None]) & (np.abs(position[:, None, None] - position[best][None]) <= 1)
    runner_up = np.where(near_best, np.inf, apart).min(axis=0)
    telling = (least <= LAID_OVER) & (runner_up - least >= TELLING)
    if float(telling.mean()) < ENOUGH_TELLING:
        return STILL
    across, rows = _layers_of_bands(best, telling, shifts, axis=0)
    down, cols = _layers_of_bands(best, telling, shifts, axis=1)
    if not any(any(layers) for layers in across) and not any(any(layers) for layers in down):
        return STILL
    # Which shifts may lay each pixel over: the layers of its own band and the bands beside it, for scenery in one
    # layer spans bands and a band's share of it can be small, and the layers of the whole picture where its band
    # has a layer going the same way.
    allowed = np.zeros(apart.shape, dtype=bool)
    index_of = {shift: i for i, shift in enumerate(shifts)}
    tally = np.bincount(best[telling], minlength=len(shifts))
    everywhere = [shifts[i] for i in range(len(shifts)) if shifts[i] != (0, 0) and tally[i] >= LAYER_SHARE_OF_ALL * tally.sum()]
    # A layer of the whole picture is one of a band's only where some of the band goes its way: a thing going the
    # way the scenery goes, at the speed of scenery elsewhere, is not taken for it.
    across_some, _ = _layers_of_bands(best, telling, shifts, axis=0, share=SOME_OF_A_BAND)
    down_some, _ = _layers_of_bands(best, telling, shifts, axis=1, share=SOME_OF_A_BAND)
    for n, (a, b) in enumerate(rows):
        for dx, dy in everywhere:
            if dy == 0 and dx in across_some[n]:
                allowed[index_of[(dx, 0)], a:b] = True
    for n, (a, b) in enumerate(cols):
        for dx, dy in everywhere:
            if dx == 0 and dy in down_some[n]:
                allowed[index_of[(0, dy)], :, a:b] = True
    for n, (a, b) in enumerate(rows):
        for layers in across[max(0, n - 1): n + 2]:
            for dx in layers:
                allowed[index_of[(dx, 0)], a:b] = True
    for n, (a, b) in enumerate(cols):
        for layers in down[max(0, n - 1): n + 2]:
            for dy in layers:
                allowed[index_of[(0, dy)], :, a:b] = True
    standing = index_of[(0, 0)]
    moving = allowed.copy()
    moving[standing] = False
    going_by = moving.any(axis=0)
    # Whether a pixel is laid over is asked of it alone: with its neighbours, the thin edge of a thing standing still
    # is outvoted by the scenery beside it.
    laid_by_layer = np.where(allowed, alone, np.inf).min(axis=0) <= LAID_OVER
    laid_by_moving = np.where(moving, alone, np.inf).min(axis=0) <= LAID_OVER
    laid_by_standing = alone[standing] <= LAID_OVER
    scenery = going_by & laid_by_moving
    # What comes into view at an edge the view goes from is laid over by nothing, and is not a thing.
    coming_in = np.zeros((tall, wide), dtype=bool)
    for a, b in rows:
        ways = [dx for dx, dy in shifts if dy == 0 and dx and allowed[index_of[(dx, 0)], a:b].any()]
        if any(dx > 0 for dx in ways):
            coming_in[a:b, : max(ways) + 1] = True
        if any(dx < 0 for dx in ways):
            coming_in[a:b, min(ways) - 1:] = True
    for a, b in cols:
        ways = [dy for dx, dy in shifts if dx == 0 and dy and allowed[index_of[(0, dy)], :, a:b].any()]
        if any(dy > 0 for dy in ways):
            coming_in[: max(ways) + 1, a:b] = True
        if any(dy < 0 for dy in ways):
            coming_in[min(ways) - 1:, a:b] = True
    on_its_own = going_by & ~laid_by_layer & ~coming_in
    # A thing of one colour is told only at its edges: in between, every shift lays it over. A flat stretch with
    # its own going on both sides of it, near, across or down, and nothing between that the view's going lays over,
    # is the thing.
    gone_by = going_by & laid_by_moving & telling
    on_its_own |= going_by & ~telling & (_between(on_its_own, gone_by, axis=1) | _between(on_its_own, gone_by, axis=0))
    # What came into view is the view, as it is.
    scenery = (scenery & ~on_its_own) | (going_by & coming_in)
    return ViewMoved(
        across=tuple(across), down=tuple(down), scenery=scenery, on_its_own=on_its_own, standing=going_by & laid_by_standing & ~laid_by_moving & ~coming_in,
        going_by=going_by, everywhere=tuple(everywhere),
        across_ways=tuple(tuple(dx for dx, dy in shifts if dy == 0 and allowed[index_of[(dx, 0)], a:b].any()) for a, b in rows),
        down_ways=tuple(tuple(dy for dx, dy in shifts if dx == 0 and dy and allowed[index_of[(0, dy)], :, a:b].any()) for a, b in cols),
    )
