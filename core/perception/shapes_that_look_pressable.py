"""The shapes on a still picture that look like something to press: buttons with no words on them.

A game's way on is often drawn, not written: a triangle to play, an arrow to
go to the next page of the rules, a round button with a tick. The screen
reading finds words, and a screen whose only way on is a shape offered her
nothing to click. LIVE-like 2026-10-05, a bowling game's rules page had an
arrow at its corner and she pressed the arrow keys at it for four minutes.

A shape that looks pressable stands out from what is around it, is a compact
thing of a button's size (neither a speck nor half the screen), and is not cut
by the picture's edge. Each is named by where it is, so the same shape on the
next look has the same name and what clicking it did can be remembered.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["pressable_shapes"]

#: The most shapes offered from one picture, the largest first.
AT_MOST = 6

#: The width a picture is looked at, for speed; shapes are found at that size.
LOOKED_AT_WIDE = 240


def pressable_shapes(picture: Any, *, apart_from: Any = (), at_most: int = AT_MOST) -> list[dict[str, Any]]:
    """The pressable-looking shapes of ``picture`` as regions (shares of the picture), named by where they are.

    A shape under writing in ``apart_from`` (regions of the same picture) is
    left out: the writing already names it, and is clicked by its words.
    """
    from scipy import ndimage

    from core.perception.picture_arithmetic import pieces

    pixels = np.asarray(picture, dtype=np.float32)
    if pixels.ndim == 3:
        pixels = pixels[..., :3].mean(axis=2)
    if pixels.ndim != 2 or min(pixels.shape) < 20:
        return []
    step = max(1, int(round(pixels.shape[1] / LOOKED_AT_WIDE)))
    small = pixels[::step, ::step]
    high, wide = small.shape
    around = ndimage.median_filter(small, size=max(9, wide // 12))
    stands_out = np.abs(small - around) > 28.0
    stands_out = ndimage.binary_closing(stands_out, iterations=2)
    stands_out = ndimage.binary_fill_holes(stands_out)
    count, _labels, stats, centres = pieces(stands_out)
    area = float(high * wide)
    found: list[tuple[float, dict[str, Any]]] = []
    for index in range(1, count):
        left, top, w, h, filled = (int(v) for v in stats[index])
        share = filled / area
        if not 0.002 <= share <= 0.06 or w < 6 or h < 6:
            continue
        if left <= 1 or top <= 1 or left + w >= wide - 1 or top + h >= high - 1:
            continue
        if not 0.25 <= w / h <= 4.0 or filled / float(w * h) < 0.3:
            continue
        cx, cy = centres[index][0] / wide, centres[index][1] / high
        if any(_under(cx, cy, region) for region in apart_from or ()):
            continue
        found.append((share, {
            "text": f"the shape at {int(round(cx * 20)) * 5}% across, {int(round(cy * 20)) * 5}% down",
            "x": round(left / wide, 5), "y": round(top / high, 5),
            "width": round(w / wide, 5), "height": round(h / high, 5),
            "center_x": round(float(cx), 5), "center_y": round(float(cy), 5),
            "shape": True,
        }))
    found.sort(key=lambda pair: -pair[0])
    named: dict[str, dict[str, Any]] = {}
    for _share, region in found:
        named.setdefault(region["text"], region)
    return list(named.values())[:at_most]


def _under(x: float, y: float, region: Any) -> bool:
    """Whether the point (shares) lies on a region of writing, or just around it."""
    try:
        left, top = float(region["x"]), float(region["y"])
        wide, high = float(region["width"]), float(region["height"])
    except (KeyError, TypeError, ValueError):
        return False
    return left - 0.02 <= x <= left + wide + 0.02 and top - 0.03 <= y <= top + high + 0.03
