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

Shapes are found two ways: by standing out from what is near them, and by
their outlines. Where many buttons sit close together (a level select, a
keypad) the first finds one block of all of them; outlines find each.

Of several alike shapes, the one whose colour stands apart from the rest is
the one that is not like the others: the lit one, the open one, the one that
is chosen. LIVE 2026-10-08 a game's level select had one bright level among
twenty-three locked ones, and she clicked the row captions for four minutes.
That one is named for it ("the one that stands out at ...") and offered first.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

__all__ = ["STANDS_OUT", "pressable_shapes"]

#: The most shapes offered from one picture, the largest first.
AT_MOST = 6

#: The width a picture is looked at, for speed; shapes are found at that size.
LOOKED_AT_WIDE = 240


#: How far a shape's colour must be from the middle of its alike ones, and how many times further than the next
#: furthest, to stand apart from them; and how many alike there must be.
APART_BY = 40.0
APART_TIMES = 2.0
ALIKE_AT_LEAST = 3

#: What the one that stands apart is called, before where it is.
STANDS_OUT = "the one that stands out at"


def pressable_shapes(picture: Any, *, apart_from: Any = (), at_most: int = AT_MOST) -> list[dict[str, Any]]:
    """The pressable-looking shapes of ``picture`` as regions (shares of the picture), named by where they are.

    A shape under writing in ``apart_from`` (regions of the same picture) is
    left out: the writing already names it, and is clicked by its words. The
    one that stands apart from its alike shapes is kept whatever is written on
    it, and comes first.
    """
    pixels = np.asarray(picture, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.ndim != 3 or min(pixels.shape[:2]) < 20:
        return []
    step = max(1, int(round(pixels.shape[1] / LOOKED_AT_WIDE)))
    small = pixels[::step, ::step, :3]
    high, wide = small.shape[:2]
    found: dict[tuple[int, int, int, int], tuple[float, np.ndarray]] = {}
    for mask in (_standing_out(small), _outlined(small)):
        for box, filled, colour in _button_like(mask, small):
            if not any(_same_place(box, other) for other in found):
                found[box] = (filled, colour)
    apart = _standing_apart(found)
    shapes: list[tuple[float, dict[str, Any]]] = []
    for box, (filled, _colour) in found.items():
        left, top, w, h = box
        cx, cy = (left + w / 2) / wide, (top + h / 2) / high
        stands = box in apart
        under = [region for region in apart_from or () if _under(cx, cy, region)]
        # Writing of words names its shape, and it is clicked by them; a number or a sign on the one that stands
        # apart ("1" on the one open level) does not stop it being offered as itself.
        if under and (not stands or any(re.search(r"[A-Za-z]{2,}", str(region.get("text") or "")) for region in under)):
            continue
        name = STANDS_OUT if stands else "the shape at"
        shapes.append((filled + (1.0 if stands else 0.0), {
            "text": f"{name} {int(round(cx * 20)) * 5}% across, {int(round(cy * 20)) * 5}% down",
            "x": round(left / wide, 5), "y": round(top / high, 5),
            "width": round(w / wide, 5), "height": round(h / high, 5),
            "center_x": round(float(cx), 5), "center_y": round(float(cy), 5),
            "shape": True, **({"stands_apart": True} if stands else {}),
        }))
    shapes.sort(key=lambda pair: -pair[0])
    named: dict[str, dict[str, Any]] = {}
    for _share, region in shapes:
        named.setdefault(region["text"], region)
    return list(named.values())[:at_most]


def _standing_out(small: np.ndarray) -> np.ndarray:
    """What stands out in brightness from what is near it, as a mask."""
    from scipy import ndimage

    gray = small.mean(axis=2)
    around = ndimage.median_filter(gray, size=max(9, small.shape[1] // 12))
    mask = ndimage.binary_closing(np.abs(gray - around) > 28.0, iterations=2)
    return ndimage.binary_fill_holes(mask)


def _outlined(small: np.ndarray) -> np.ndarray:
    """What is closed inside a strong outline, as a mask; thin lines opened away."""
    from scipy import ndimage

    edge = np.zeros(small.shape[:2], dtype=np.float32)
    for channel in range(3):
        edge += np.hypot(ndimage.sobel(small[..., channel], 0), ndimage.sobel(small[..., channel], 1))
    if not edge.any():
        return np.zeros(edge.shape, dtype=bool)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(edge > np.percentile(edge, 80), iterations=1))
    return ndimage.binary_opening(mask, iterations=2)


def _button_like(mask: np.ndarray, small: np.ndarray) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
    """The pieces of ``mask`` of a button's size and shape: their box, their share of the picture, their colour."""
    from core.perception.picture_arithmetic import pieces

    high, wide = mask.shape
    count, labels, stats, _centres = pieces(mask)
    area = float(high * wide)
    out = []
    for index in range(1, count):
        left, top, w, h, filled = (int(v) for v in stats[index])
        share = filled / area
        if not 0.002 <= share <= 0.06 or w < 6 or h < 6:
            continue
        if left <= 1 or top <= 1 or left + w >= wide - 1 or top + h >= high - 1:
            continue
        if not 0.25 <= w / h <= 4.0 or filled / float(w * h) < 0.3:
            continue
        inside = labels[top:top + h, left:left + w] == index
        out.append(((left, top, w, h), share, small[top:top + h, left:left + w][inside].mean(axis=0)))
    return out


def _same_place(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two boxes are one shape found twice: the middle of each inside the other."""
    def inside(x: float, y: float, box: tuple[int, int, int, int]) -> bool:
        return box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]

    return inside(a[0] + a[2] / 2, a[1] + a[3] / 2, b) and inside(b[0] + b[2] / 2, b[1] + b[3] / 2, a)


def _standing_apart(found: dict[tuple[int, int, int, int], tuple[float, np.ndarray]]) -> set[tuple[int, int, int, int]]:
    """The shapes whose colour stands apart from several shapes of their size: one to a set of alike shapes."""
    boxes = list(found)
    apart: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        # Alike in size, and set out in a line with it: a row of levels, a column of choices. Shapes merely of a
        # size scattered over a picture (letters of a caption, scenery) are not a set of choices.
        alike = [b for b in boxes if abs(b[2] - box[2]) <= 0.3 * box[2] and abs(b[3] - box[3]) <= 0.3 * box[3]
                 and (abs((b[1] + b[3] / 2) - (box[1] + box[3] / 2)) <= box[3] / 2
                      or abs((b[0] + b[2] / 2) - (box[0] + box[2] / 2)) <= box[2] / 2)]
        if len(alike) < ALIKE_AT_LEAST:
            continue
        colours = np.array([found[b][1] for b in alike])
        middle = np.median(colours, axis=0)
        far = np.sqrt(((colours - middle) ** 2).sum(axis=1))
        mine = float(np.sqrt(((found[box][1] - middle) ** 2).sum()))
        next_far = sorted(far)[-2]
        if mine >= far.max() and mine > APART_BY and mine >= APART_TIMES * max(float(next_far), 1.0):
            apart.add(box)
    return apart


def _under(x: float, y: float, region: Any) -> bool:
    """Whether the point (shares) lies on a region of writing, or just around it."""
    try:
        left, top = float(region["x"]), float(region["y"])
        wide, high = float(region["width"]), float(region["height"])
    except (KeyError, TypeError, ValueError):
        return False
    return left - 0.02 <= x <= left + wide + 0.02 and top - 0.03 <= y <= top + high + 0.03
