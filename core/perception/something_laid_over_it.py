"""Something laid over the window she is using: a panel drawn on top, and everything else darkened behind it.

A program that needs an answer before anything else lays a panel over its window and darkens the rest, so that
nothing under it can be pressed until the panel is gone. What the panel says has no end of forms ("Player not
found", "Save changes?", a sign-in box, an advertisement); how it is drawn has one. Everything she was looking at is
still there, darker by one common share, and one block of it has been replaced by something else.

So it is told from two pictures of one window: the last she had of it with nothing over it, and the one now. Each is
taken down to a coarse grid of how light each part is. A layer of shade laid over a picture changes every part of it
the same way: each part's lightness is multiplied by one share and moved by one amount, the same for all (a dark
layer only multiplies; a light one also lifts). So the parts outside the panel are fitted by one such line, and that
line compresses them toward one tone. The parts that do not fit it, gathered in one block, are the panel.

A game going on, a page scrolling, a window resized: none of them changes most of a window by one common line, and
none is taken for a panel. Nothing here knows a program or a word.

LIVE 23 November 2026 she pressed a heading in En Croissant, it laid "FIDE Player Info: Player not found" over the
board and darkened the rest, and she pressed the up arrow at the dimmed board until she gave up. Her reader of
dialogs knew a dialog only by words like "cookies" and "welcome".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.SomethingLaidOverIt")

__all__ = ["ACROSS", "DOWN", "LaidOver", "how_light_each_part_is", "laid_over", "laid_over_what_she_saw"]

#: The grid a picture is taken down to: fine enough that a panel a few lines tall spans several parts, coarse enough
#: that a part is an area and not a stroke of a letter.
ACROSS = 32
DOWN = 24

#: The least two pictures of one thing can differ by, in levels of an 8-bit value: what rounding leaves.
ONE_LEVEL = 1.0
#: How many times the fit's own spread a part may miss it by and still be one of the parts it fits: three spreads,
#: the usual bar past which a value is not one of its kind.
SPREADS = 3.0
#: The robust spread of a set of misses: its median absolute miss times the factor that makes it a standard
#: deviation where the misses are normal.
MAD_TO_SD = 1.4826
#: How much of its contrast the rest of the window keeps, at most, under a layer of shade. A person notices a
#: change in lightness of about one part in fifty (Weber's fraction for lightness); a layer laid over a window to
#: hold attention darkens it by a third or more. A tenth is five times what can be seen and well short of any such
#: layer.
KEPT_AT_MOST = 0.9
#: The share of the window that must be fitted by the one line: most of it, or the panel is not the smaller part.
MOST = 0.5
#: How much of the block the panel's parts span must be the panel's: a panel is one solid thing, not scattered parts.
SOLID = 0.5
#: The fewest parts a panel covers: two by two, the least that is a block.
FEWEST = 4


@dataclass(frozen=True)
class LaidOver:
    """A panel laid over the window: where it is, as shares of the window, and how much of its contrast the rest
    kept."""

    left: float
    top: float
    right: float
    bottom: float
    kept: float
    fitted: float

    def as_dict(self) -> dict[str, float]:
        return {"left": round(self.left, 4), "top": round(self.top, 4), "right": round(self.right, 4),
                "bottom": round(self.bottom, 4), "kept": round(self.kept, 3), "fitted": round(self.fitted, 3)}

    def centre(self) -> tuple[float, float]:
        return (self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0


def how_light_each_part_is(picture: Any) -> list[list[float]]:
    """How light each part of a picture is: its three channels averaged over each of DOWN by ACROSS parts."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.shape[0] < DOWN or pixels.shape[1] < ACROSS:
        return []
    grey = pixels[:, :, :3].astype(np.float32).mean(axis=2)
    small = Image.fromarray(grey).resize((ACROSS, DOWN), Image.Resampling.BOX)
    return [[round(float(v), 1) for v in row] for row in np.asarray(small)]


def laid_over(before: Any, now: Any) -> LaidOver | None:
    """The panel laid over a window between two of its grids of lightness, or None where nothing is."""
    import numpy as np  # noqa: PLC0415

    try:
        was = np.asarray(before, dtype=np.float64)
        is_now = np.asarray(now, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if was.shape != is_now.shape or was.ndim != 2 or was.size < 4 * FEWEST:
        return None
    b, a = was.ravel(), is_now.ravel()
    fits = np.ones(b.size, dtype=bool)
    slope, lift = 1.0, 0.0
    for _ in range(4):
        if fits.sum() < 2 or np.ptp(b[fits]) <= ONE_LEVEL:
            return None
        slope, lift = np.polyfit(b[fits], a[fits], 1)
        miss = np.abs(a - (slope * b + lift))
        spread = max(MAD_TO_SD * float(np.median(miss[fits])), ONE_LEVEL)
        refit = miss <= SPREADS * spread
        if (refit == fits).all():
            break
        fits = refit
    fitted = float(fits.mean())
    # Most of the window on one line that keeps less of its contrast than a layer of shade leaves at most, and that
    # does not turn it over (a theme going from light to dark).
    if fitted < MOST or not 0.0 < slope <= KEPT_AT_MOST:
        return None
    # The change along the line must be one that can be told from no change at all: at the parts fitted, the line
    # moves them further than the fit misses them by.
    moved = np.abs((slope - 1.0) * b[fits] + lift)
    if float(np.median(moved)) <= SPREADS * spread:
        return None
    from scipy import ndimage  # noqa: PLC0415

    misfit = (~fits).reshape(was.shape)
    blocks, count = ndimage.label(misfit, structure=np.ones((3, 3), dtype=int))
    if not count:
        return None
    sizes = ndimage.sum(misfit, blocks, index=range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    rows, columns = np.nonzero(blocks == largest)
    tall = rows.max() - rows.min() + 1
    wide = columns.max() - columns.min() + 1
    if len(rows) < FEWEST or len(rows) < SOLID * tall * wide:
        return None
    down, across = was.shape
    return LaidOver(left=float(columns.min() / across), top=float(rows.min() / down),
                    right=float((columns.max() + 1) / across), bottom=float((rows.max() + 1) / down),
                    kept=float(slope), fitted=fitted)


#: For each window, the last grid of it she had with nothing laid over it.
_CLEAR: dict[str, list[list[float]]] = {}


def laid_over_what_she_saw(window: str, tones: Any) -> LaidOver | None:
    """Whether a panel is laid over ``window`` in the picture whose grid is ``tones``, against the last picture of it
    she had with nothing over it. A picture with nothing over it becomes that picture."""
    if not window or not tones:
        return None
    clear = _CLEAR.get(window)
    found = laid_over(clear, tones) if clear is not None else None
    if found is None:
        _CLEAR[window] = [list(row) for row in tones]
        return None
    logger.info("a panel is laid over %r at (%.2f, %.2f)-(%.2f, %.2f): the rest kept %.0f%% of its contrast",
                window, found.left, found.top, found.right, found.bottom, 100 * found.kept)
    return found
