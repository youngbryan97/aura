"""Places in a picture that changed and stayed changed, with nothing moving there: counters.

A game keeps score in writing it redraws, and text recognition does not
always read it. A lone digit is the worst case: offline 3 October a "0" on a
black court was read as nothing at every size. That a counter changed, and
where, can be seen without reading it. Pictures half a second apart differ
there; the next half second they do not; and nothing that moves passed over
the place in between. A change over a tenth of the picture or more is a screen
being drawn, not a counter.

Nothing here knows what a counter means. Whoever reads these places decides
that (core/agency/what_meeting_things_does.py).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

__all__ = ["WhatChangedAndStayed"]

#: How often a picture is kept, in seconds; places are compared two apart.
EVERY_S = 0.25

#: Grey levels: a change, and the most that "stayed the same" allows.
CHANGED, SAME = 40, 20

#: More than this share of the picture changing at once is a new screen.
A_NEW_SCREEN = 0.10


class WhatChangedAndStayed:
    """Takes the pictures a tracker reads and says which still places changed."""

    def __init__(self) -> None:
        self._kept: deque = deque(maxlen=8)
        self._at = -math.inf
        self.seen: list[tuple[float, float, float]] = []

    def see(self, picture: np.ndarray | None, things: Any, at: float, *, never: set[int] = frozenset()) -> list[tuple[float, float, float]]:
        """Take one picture (RGB, the tracker's working size) and the things in it.

        Returns new changes as (when, x, y), with x and y as shares of the
        picture. ``never`` names things whose places are never counters (hers).
        """
        import cv2

        if picture is None or at - self._at < EVERY_S:
            return []
        self._at = at
        tall, wide = picture.shape[:2]
        grey = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        moving = np.zeros((tall, wide), dtype=bool)
        for thing in things.values():
            if thing.moved or thing.number in never:
                left, top, right, bottom = (int(v) for v in thing.box())
                moving[max(0, top - 4) : bottom + 5, max(0, left - 4) : right + 5] = True
        if self._kept and self._kept[-1][1].shape != grey.shape:
            self._kept.clear()
        self._kept.append((at, grey, moving))
        if len(self._kept) < 5:
            return []
        (_t0, first, _m0), (when, middle, _m1), (_t2, last, _m2) = self._kept[-5], self._kept[-3], self._kept[-1]
        passed = np.zeros_like(moving)
        for _when, _grey, mask in list(self._kept)[-5:]:
            passed |= mask
        changed = (cv2.absdiff(first, middle) > CHANGED) & (cv2.absdiff(middle, last) < SAME) & ~passed
        if changed.mean() > A_NEW_SCREEN or not changed.any():
            return []
        count, _labels, stats, centres = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
        new = []
        for label in range(1, count):
            if stats[label][4] < 3:
                continue
            x, y = float(centres[label][0] / wide), float(centres[label][1] / tall)
            if any(abs(x - px) < 0.03 and abs(y - py) < 0.03 and when - t < 1.5 for t, px, py in self.seen):
                continue
            self.seen.append((when, x, y))
            new.append((when, x, y))
        del self.seen[:-50]
        return new
