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
#: A key cap's size, as a share of the picture's shorter side, and how square and filled it is.
SMALLEST, LARGEST = 0.02, 0.12
SQUARE = (0.7, 1.43)
FILLED = 0.7
#: The fewest pixels a mark is made of, and how much wider an arrow's head is than its shaft.
LEAST_MARK = 8
HEAD_OVER_SHAFT = 2.0
#: How lopsided a profile must be to run along an arrow. Across an arrow it is even (0.01 measured on a 14-pixel key
#: cap); along one, 0.17 on the same cap and more on a longer arrow.
LOPSIDED = 0.1


def _runs_even(extents: list[int]) -> tuple[int, int]:
    """How many lines at each end of a profile hold the thickness the end has: a shaft is long, a tip is one line."""
    def run(values: list[int]) -> int:
        count = 1
        while count < len(values) and abs(values[count] - values[0]) <= 1:
            count += 1
        return count
    return run(extents), run(extents[::-1])


def which_way_it_points(mark: np.ndarray) -> str:
    """"left", "right", "up" or "down" where a mask is an arrow, else "".

    The one reading of an arrow's shape, for a key cap, a row of arrows to follow, or a sign.
    """
    ys, xs = np.nonzero(mark)
    if len(ys) < LEAST_MARK:
        return ""
    across = [int(np.ptp(ys[xs == x]) + 1) if (xs == x).any() else 0 for x in range(xs.min(), xs.max() + 1)]
    down = [int(np.ptp(xs[ys == y]) + 1) if (ys == y).any() else 0 for y in range(ys.min(), ys.max() + 1)]
    readings = []
    for extents, ways in ((across, ("left", "right")), (down, ("up", "down"))):
        if len(extents) < 5:
            continue
        # Along an arrow its profile is lopsided, head at one end and shaft at the other; across it, it is even about
        # the middle (head above and below, the shaft through the middle), and says nothing of which way it points.
        lopsided = sum(abs(a - b) for a, b in zip(extents, extents[::-1], strict=True)) / (2 * sum(extents))
        first, last = _runs_even(extents)
        shaft = min(extents[0], extents[-1])
        if lopsided < LOPSIDED or max(extents) < HEAD_OVER_SHAFT * max(1, shaft) or first == last:
            continue
        # The shaft is the long even run; the arrow points to the other end.
        readings.append((lopsided, ways[0] if last > first else ways[1]))
    return max(readings)[1] if readings else ""


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
            edge = max(1, int(0.12 * min(high, broad)))
            inner = np.zeros_like(tile)
            inner[edge:-edge, edge:-edge] = True
            mark = inner & tile & (np.abs(face - ground) > MARKED)
            if int(mark.sum()) < LEAST_MARK:
                continue
            way = which_way_it_points(mark)
            if way:
                found.append({"key": way, "center_x": (place[1].start + broad / 2) / wide,
                              "center_y": (place[0].start + high / 2) / tall,
                              "width": broad / wide, "height": high / tall})
    return sorted(found, key=lambda key: (key["center_y"] // 0.05, key["center_x"]))
