"""Keys a screen shows as pictures: a key cap drawn on it, with an arrow on its face.

A screen asks for keys in words ("press SPACE") and in pictures. LIVE 2026-10-08
a game caught her character and showed two key caps over it, ← and →, lit in
turn: tap these, one after the other, fast, to break free. Her reading of
writing made the ← a "+" and did not see the → at all, and her button finder
passes over shapes that small, so she stood held while the game took her
health.

A key cap is a small tile, nearly square and filled, light on a darker picture
or dark on a lighter one, with one mark on its face that contrasts with it. An
arrow on it is read by its shape. Seen across the arrow's length, its tail is a
shaft of even thickness and its head tapers to the point, so it points away
from the end where the shaft is. Nothing here knows what game it is in; the
same reading holds for a tutorial overlay, a help panel or a form.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["keys_drawn", "which_way_it_points"]

#: How light (or dark) a tile is against the picture, on a 0-255 scale.
LIGHT, DARK = 170.0, 60.0
#: How far the mark on a tile's face stands from the tile itself.
MARKED = 60.0
#: A key cap's size, as a share of the picture's shorter side, and how square and filled it is: the caps measured were
#: 4.4% (a game) and 8% (a test world) of it, and the enclosed part of a letter in a score line is under 3%. A cap is a
#: rounded square, filled 0.90 and 0.93 of its box as measured; a bold letter filled in has notches and is less.
SMALLEST, LARGEST = 0.03, 0.12
SQUARE = (0.7, 1.43)
FILLED = 0.85
#: The fewest pixels a mark is made of, and how much wider an arrow's head is than its shaft.
LEAST_MARK = 8
HEAD_OVER_SHAFT = 2.0
#: How many more lines a head must narrow over one way than the other to be an arrow's.
LEAST_TAPER = 2


def _one_way_taper(extents: list[int]) -> int:
    """How much longer a profile's steadiest widening is than its steadiest narrowing: above nought it widens away
    from its start (the start is a point), below nought it narrows to its end (the end is)."""
    widest_run = narrowest_run = widening = narrowing = 1
    for before, after in zip(extents, extents[1:], strict=False):
        widening = widening + 1 if after > before else 1
        narrowing = narrowing + 1 if after < before else 1
        widest_run, narrowest_run = max(widest_run, widening), max(narrowest_run, narrowing)
    return widest_run - narrowest_run


def which_way_it_points(mark: np.ndarray) -> str:
    """"left", "right", "up" or "down" where a mask is an arrow, else "".

    The one reading of an arrow's shape, for a key cap, a row of arrows to follow, or a sign. Along an arrow its head
    tapers one way, line after line narrower to the point, and the shaft does not taper back; across it the head
    widens to the shaft and narrows again, as much one way as the other. Measured on a game's key caps: a one-way
    taper of 4 lines along and 1 across. The arrow points to its narrow end. Where neither way tapers one way plainly
    more, it is no arrow: a wrong key pressed is worse than none.
    """
    ys, xs = np.nonzero(mark)
    if len(ys) < LEAST_MARK:
        return ""
    across = [int(np.ptp(ys[xs == x]) + 1) if (xs == x).any() else 0 for x in range(xs.min(), xs.max() + 1)]
    down = [int(np.ptp(xs[ys == y]) + 1) if (ys == y).any() else 0 for y in range(ys.min(), ys.max() + 1)]
    readings = []
    for extents, ways in ((across, ("left", "right")), (down, ("up", "down"))):
        taper = _one_way_taper(extents)
        if abs(taper) >= LEAST_TAPER and max(extents) >= HEAD_OVER_SHAFT * max(1, min(extents[0], extents[-1])):
            readings.append((abs(taper), ways[0] if taper > 0 else ways[1]))
    readings.sort(reverse=True)
    if not readings or (len(readings) > 1 and readings[0][0] <= readings[1][0]):
        return ""
    return readings[0][1]


def keys_drawn(picture: Any) -> list[dict[str, Any]]:
    """The key caps drawn in an RGB ``picture`` that carry an arrow, each with its key and where it is (shares)."""
    from scipy import ndimage

    pixels = np.asarray(picture, dtype=np.float32)
    if pixels.ndim != 3 or min(pixels.shape[:2]) < 40:
        return []
    gray = pixels[..., :3].mean(axis=2)
    tall, wide = gray.shape
    short = min(tall, wide)
    found: list[dict[str, Any]] = []
    for tile_mask in (gray > LIGHT, gray < DARK):
        labels, _count = ndimage.label(ndimage.binary_fill_holes(tile_mask))
        for number, place in enumerate(ndimage.find_objects(labels), 1):
            if place is None:
                continue
            high, broad = place[0].stop - place[0].start, place[1].stop - place[1].start
            if not (SMALLEST * short <= min(high, broad) and max(high, broad) <= LARGEST * short
                    and SQUARE[0] <= broad / high <= SQUARE[1]):
                continue
            tile = labels[place] == number
            if tile.mean() < FILLED:
                continue
            face = gray[place]
            ground = float(np.median(face[tile & tile_mask[place]]))
            # A cap stands apart from what it sits on: the band just outside it is unlike its face.
            top, left = max(0, place[0].start - 3), max(0, place[1].start - 3)
            bottom, right = min(tall, place[0].stop + 3), min(wide, place[1].stop + 3)
            around = np.ones((bottom - top, right - left), dtype=bool)
            around[place[0].start - top:place[0].stop - top, place[1].start - left:place[1].stop - left] = False
            if not around.any() or abs(float(np.median(gray[top:bottom, left:right][around])) - ground) < MARKED / 2:
                continue
            edge = max(1, int(0.12 * min(high, broad)))
            inner = np.zeros_like(tile)
            inner[edge:-edge, edge:-edge] = True
            mark = inner & tile & (np.abs(face - ground) > MARKED)
            if int(mark.sum()) < LEAST_MARK:
                continue
            # The mark sits on the cap's face clear of its edges; what reaches in from an edge is the outline of
            # something else (the strokes round the hole in a letter of a score line).
            rows, cols = np.nonzero(mark)
            if rows.min() <= edge or cols.min() <= edge or rows.max() >= high - edge - 1 or cols.max() >= broad - edge - 1:
                continue
            way = which_way_it_points(mark)
            if way:
                found.append({"key": way, "center_x": (place[1].start + broad / 2) / wide,
                              "center_y": (place[0].start + high / 2) / tall,
                              "width": broad / wide, "height": high / tall})
    return sorted(found, key=lambda key: (key["center_y"] // 0.05, key["center_x"]))
