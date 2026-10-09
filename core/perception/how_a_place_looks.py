"""How a place on the screen looks, as a few numbers: the mean colour in each cell of a small grid over it.

Two places that look alike have near numbers whatever is drawn in them: two cards
turned over showing the same picture, the item a customer asks for and the one
on the shelf, two pieces of one colour. Small enough to keep for every place she
has seen, and to compare against every other.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["ALIKE", "alike", "look_of"]

#: The cells across and down a place's look is taken over.
CELLS = 4
#: How far apart two looks may be (mean difference per cell and colour, 0 to 1) and still be alike: about a tenth of
#: the way from black to white, beyond what the compression and edges of a screen picture move.
ALIKE = 0.1


def look_of(picture: Any, region: Mapping[str, Any]) -> tuple[float, ...]:
    """The look of a ``region`` (x, y, width, height as shares) of a picture: CELLS x CELLS mean colours, 0 to 1."""
    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.size == 0:
        return ()
    tall, wide = pixels.shape[:2]
    left, top = int(float(region.get("x", 0.0)) * wide), int(float(region.get("y", 0.0)) * tall)
    right = max(left + 1, int((float(region.get("x", 0.0)) + float(region.get("width", 0.0))) * wide))
    bottom = max(top + 1, int((float(region.get("y", 0.0)) + float(region.get("height", 0.0))) * tall))
    patch = pixels[max(0, top):min(tall, bottom), max(0, left):min(wide, right), :3].astype(np.float32) / 255.0
    if patch.size == 0:
        return ()
    rows = np.array_split(np.arange(patch.shape[0]), CELLS)
    cols = np.array_split(np.arange(patch.shape[1]), CELLS)
    cells = [patch[r][:, c].reshape(-1, 3).mean(axis=0) if len(r) and len(c) else np.zeros(3) for r in rows for c in cols]
    return tuple(round(float(value), 3) for cell in cells for value in cell)


def alike(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Whether two looks are alike."""
    if not a or len(a) != len(b):
        return False
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) <= ALIKE


def apart(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """How far apart two looks are (0 the same; 1 or more nothing alike); 1 where either is not known."""
    if not a or not b or len(a) != len(b):
        return 1.0
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
