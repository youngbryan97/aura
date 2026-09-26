"""What each of her acts does to the view, learned from the pictures before and after it.

On a board an act moves things between places, and `how_it_moves` works out
which way each act pushes. In a world seen through a camera nothing moves
between places: the whole picture slides when she turns, and grows from the
middle when she walks forward. Which key walks, which way the mouse turns the
camera and how far a point of mouse travel turns it are all different in
every game, and none of them is written anywhere she can read. SIMA 2 learns
them inside its weights from a great deal of human play. Here they are
measured, act by act, from what the picture did.

Two measurements do all of it. How far the picture slid between two looks,
found by phase correlation: the shift at which one frame lines up with the
next, read off the peak of their normalised cross-power spectrum. And how
much it grew, found by trying the middle of the second frame at a few
magnifications and keeping the one that matches the first best. Both run on
small grey copies of the frames, because what is asked is where the picture
went, not what is in it.

From those, per act: the mean slide and growth, and how consistent they were.
The mouse's gain — pixels of slide per point of travel, in each axis, with
its sign — is a least-squares fit through the origin over every mouse move
she has watched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ViewChange", "WhatMyHandsDoToTheView", "grey", "how_alike", "how_it_grew", "how_it_moved", "how_it_slid"]

#: The size frames are measured at. Big enough that a slide of a few pixels
#: at full size still registers; small enough that a measurement costs
#: nothing beside a look.
_WIDE = 160


def grey(frame: Any, *, wide: int = _WIDE) -> np.ndarray:
    """A small grey copy of a frame, as floats, for measuring motion."""
    pixels = np.asarray(frame, dtype=np.float32)
    if pixels.ndim == 3:
        pixels = pixels[..., :3].mean(axis=2)
    high, across = pixels.shape
    step = max(1, across // wide)
    small = pixels[: high - high % step, : across - across % step]
    small = small.reshape(small.shape[0] // step, step, small.shape[1] // step, step).mean(axis=(1, 3))
    return small - small.mean()


def how_it_slid(before: np.ndarray, after: np.ndarray) -> tuple[float, float, float]:
    """How far the picture slid from ``before`` to ``after``: (across, down, how sure).

    In pixels of the small frames. Sure is the height of the correlation peak
    over the mean of the surface: a clean slide stands far above it, a change
    that is not a slide does not.
    """
    if before.shape != after.shape or before.size == 0:
        return 0.0, 0.0, 0.0
    window = np.outer(np.hanning(before.shape[0]), np.hanning(before.shape[1]))
    one, two = np.fft.fft2(before * window), np.fft.fft2(after * window)
    cross = two * np.conj(one)
    cross /= np.abs(cross) + 1e-9
    surface = np.abs(np.fft.ifft2(cross))
    down, across = np.unravel_index(int(np.argmax(surface)), surface.shape)
    high, wide = surface.shape
    if down > high // 2:
        down -= high
    if across > wide // 2:
        across -= wide
    sure = float(surface.max() / (surface.mean() + 1e-9))
    return float(across), float(down), sure


#: The magnifications tried when asking how much the picture grew, finer near
#: one because a walking step is a few per cent. Shrinking is the same
#: question asked the other way round.
_GROWTHS = tuple(round(1.0 + 0.025 * step, 3) for step in range(13))


def _magnified(frame: np.ndarray, growth: float) -> np.ndarray | None:
    """The middle of ``frame`` magnified by ``growth`` to the frame's own size."""
    high, wide = frame.shape
    crop_h, crop_w = int(round(high / growth)), int(round(wide / growth))
    if crop_h < 4 or crop_w < 4:
        return None
    top, left = (high - crop_h) // 2, (wide - crop_w) // 2
    middle = frame[top : top + crop_h, left : left + crop_w]
    rows = np.linspace(0, crop_h - 1, high).round().astype(int)
    cols = np.linspace(0, crop_w - 1, wide).round().astype(int)
    grown = middle[np.ix_(rows, cols)]
    return grown - grown.mean()


def how_it_moved(before: np.ndarray, after: np.ndarray) -> tuple[float, float, float, float]:
    """How far the picture slid and how much it grew between two frames: (across, down, grew, sure).

    Asked together, because each spoils the other asked alone: a slide read
    off a magnified pair is off by the magnification, and a magnification read
    off a pair that slid lines up texture instead of the scene. For every
    trial magnification the best slide is found by phase correlation, and the
    pair whose correlation peak stands highest wins. Measured live, reading
    them apart gave walking steps of 0.8 and 1.1 in a row while the view grew
    five per cent a step.
    """
    if before.shape != after.shape or min(before.shape) < 8:
        return 0.0, 0.0, 1.0, 0.0
    if np.array_equal(before, after):
        # Nothing changed, and every trial ties on a picture that did not
        # move; the tie must not be broken toward motion that was not there.
        return 0.0, 0.0, 1.0, math.inf
    best = (0.0, 0.0, 1.0, -math.inf)
    for growth in _GROWTHS:
        trials = [(growth, _magnified(before, growth), after)]
        if growth != 1.0:
            trials.append((1.0 / growth, before, _magnified(after, growth)))
        for grew, one, two in trials:
            if one is None or two is None:
                continue
            across, down, sure = how_it_slid(one, two)
            if sure > best[3]:
                best = (across, down, grew, sure)
    return best


def how_alike(one: np.ndarray, two: np.ndarray) -> float:
    """How alike two views are once the slide between them is taken out; one is the same view.

    Turning the camera a whole circle in steps of whole mouse points does not
    land exactly where it began, so two raw pictures of the same place can
    share nothing pixel for pixel. Lined up first, they share nearly all of it,
    and two views of different places share nothing however they are lined up.
    """
    if one.shape != two.shape or min(one.shape) < 8:
        return 0.0
    across, down, _sure = how_it_slid(one, two)
    dx, dy = int(round(across)), int(round(down))
    back = np.roll(np.roll(two, -dy, axis=0), -dx, axis=1)
    high, wide = one.shape
    my, mx = abs(dy) + 1, abs(dx) + 1
    if high <= 2 * my + 4 or wide <= 2 * mx + 4:
        return 0.0
    a = one[my : high - my, mx : wide - mx]
    b = back[my : high - my, mx : wide - mx]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def how_it_grew(before: np.ndarray, after: np.ndarray) -> float:
    """How much the middle of the picture grew from ``before`` to ``after``; one is not at all."""
    return how_it_moved(before, after)[2]


@dataclass(frozen=True)
class ViewChange:
    """What one act did to the view."""

    across: float
    down: float
    grew: float
    sure: float


@dataclass
class WhatMyHandsDoToTheView:
    """Each act she has watched, and what it did to the picture."""

    seen: dict[str, list[ViewChange]] = field(default_factory=dict)
    #: Mouse travel and the slide it caused, for fitting the gain.
    _mouse: list[tuple[float, float, float, float]] = field(default_factory=list)

    def watched(self, act: str, before: Any, after: Any, *, mouse: tuple[int, int] = (0, 0)) -> ViewChange:
        """One act, and the frames either side of it."""
        across, down, grew, sure = how_it_moved(grey(before), grey(after))
        change = ViewChange(across, down, grew, sure)
        self.seen.setdefault(str(act), []).append(change)
        if mouse != (0, 0):
            self._mouse.append((float(mouse[0]), float(mouse[1]), across, down))
        return change

    def growth_between(self, before: Any, after: Any) -> float:
        """How much the middle of the view grew between two frames."""
        return how_it_moved(grey(before), grey(after))[2]

    def mouse_gain(self) -> tuple[float, float]:
        """Pixels of slide (in the small frames) per point of mouse travel, across and down.

        A fit through the origin, since no travel is no slide. Nought where she
        has not moved the mouse that way yet. Negative where the picture slides
        against the mouse, which is how a camera turning toward the mouse looks.
        """
        def fit(travel: list[float], slid: list[float]) -> float:
            den = sum(t * t for t in travel)
            return sum(t * s for t, s in zip(travel, slid, strict=True)) / den if den else 0.0

        return (
            fit([m[0] for m in self._mouse], [m[2] for m in self._mouse]),
            fit([m[1] for m in self._mouse], [m[3] for m in self._mouse]),
        )

    def what_it_does(self, act: str) -> ViewChange | None:
        """What an act has done on average, or None before it has been watched."""
        seen = self.seen.get(str(act)) or []
        if not seen:
            return None
        n = float(len(seen))
        return ViewChange(
            sum(c.across for c in seen) / n,
            sum(c.down for c in seen) / n,
            sum(c.grew for c in seen) / n,
            sum(c.sure for c in seen) / n,
        )

    def walks_forward(self) -> str:
        """The act that grows the middle of the view most, or empty when none does."""
        grown = {act: (self.what_it_does(act) or ViewChange(0, 0, 1.0, 0)).grew for act in self.seen}
        best = max(grown, key=grown.get, default="")
        return best if best and grown[best] > 1.0 else ""

    def turn_for(self, across_px: float) -> int:
        """Mouse travel, in points, that would slide the view by ``across_px`` small-frame pixels."""
        gain, _ = self.mouse_gain()
        return int(round(across_px / gain)) if gain else 0
