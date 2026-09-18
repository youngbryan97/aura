"""What a window shows, read from its pixels rather than only from its words.

A reading made of text alone cannot see a place with nothing written in it.
An empty square of a board, a blank day on a calendar, an unfilled cell of a
form are all there on the screen and absent from the words, so everything that
had to know where the places were worked it out backwards from where text had
been seen over many moves. That inference is slow and it is wrong in exactly
the cases that matter: a board whose top row happens to be empty reads as
three rows, and a thing with two items in it reads as a line.

And text recognition drops things. Measured 2026-09-17 on a clean capture of a
game window: macOS Vision returned "2" and "8" and left out a "4" and a "2"
sitting in plain sight, with language correction on or off, at every revision.
A single glyph alone in a square is not what it was built to find.

So a reading is built in three passes, none of which knows what it is looking
at:

1. **Panels.** Regions of even colour with a clear edge, found from the image
   gradient. A button, a card, a square of a board, a cell of a table.
2. **Grids.** Panels of one size set out at one pitch across and down. The
   places are there whether or not anything is written in them, so an empty
   place is empty rather than absent from the first glance.
3. **Words, placed.** One recognition pass over the whole window, each run of
   text put in the panel it sits in. A place that looks unlike the empty ones
   and read as nothing is read again, and places that look alike are
   recognised as saying the same thing, because once somebody has read a sign
   they do not read it letter by letter every time it comes round.

Recognition by appearance is also what makes looking cheap. A place whose
look has been read before costs a comparison, not a recognition pass.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatThePixelsShow")

__all__ = [
    "Grid",
    "Panel",
    "Looker",
    "grids_in",
    "panels_in",
    "recognize_text",
]

#: The longest side a picture is worked on at for finding panels. Edges of a
#: control survive halving comfortably and the work is a quarter.
_WORKING_SIDE = 760

#: How different two neighbouring pixels have to be, in Lab units, to be an
#: edge. Below this, anti-aliasing and gradients inside one surface.
_EDGE = 14.0

#: How much of its bounding box a region has to fill to count as a panel. A
#: rounded corner costs a few percent; a glyph or an irregular shape costs far
#: more.
_FILLS = 0.82

#: Two sizes are one size when they are within this share of each other.
_SAME_SIZE = 0.14

#: Two appearances are one appearance when their distance, per sample, is
#: under this many Lab units.
_SAME_LOOK = 9.0

#: The side of the small picture an appearance is kept as.
_LOOK_SIDE = 24

#: How wide the picture is made when asking whether the part of a window
#: around its grids has changed. Small enough to compare in a millisecond,
#: large enough that one digit of a score moves it further than the noise of
#: capturing the same screen twice.
_AROUND_SIDE = 160


@dataclass(frozen=True)
class Panel:
    """A region of even colour with an edge. Fractions of the window."""

    left: float
    top: float
    width: float
    height: float
    #: The colour most of it is, in Lab.
    tone: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def holds(self, x: float, y: float) -> bool:
        return (
            self.left <= x <= self.left + self.width
            and self.top <= y <= self.top + self.height
        )


@dataclass(frozen=True)
class Grid:
    """Places of one size, at one pitch, across and down."""

    rows: int
    columns: int
    #: Where each row's middle is, top to bottom, and each column's, left to
    #: right. Fractions of the window.
    down_at: tuple[float, ...]
    across_at: tuple[float, ...]
    cell_width: float
    cell_height: float

    @property
    def outline(self) -> tuple[float, float, float, float]:
        """Left, top, right, bottom of the whole grid."""
        return (
            self.across_at[0] - self.cell_width / 2.0,
            self.down_at[0] - self.cell_height / 2.0,
            self.across_at[-1] + self.cell_width / 2.0,
            self.down_at[-1] + self.cell_height / 2.0,
        )

    def place(self, row: int, column: int) -> Panel:
        return Panel(
            left=self.across_at[column] - self.cell_width / 2.0,
            top=self.down_at[row] - self.cell_height / 2.0,
            width=self.cell_width,
            height=self.cell_height,
        )

    def where(self, x: float, y: float) -> tuple[int, int] | None:
        """The row and column a point is in, or None when it is between places."""
        column = min(range(self.columns), key=lambda c: abs(self.across_at[c] - x))
        row = min(range(self.rows), key=lambda r: abs(self.down_at[r] - y))
        if abs(self.across_at[column] - x) > self.cell_width / 2.0:
            return None
        if abs(self.down_at[row] - y) > self.cell_height / 2.0:
            return None
        return (row, column)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "down_at": list(self.down_at),
            "across_at": list(self.across_at),
            "cell_width": self.cell_width,
            "cell_height": self.cell_height,
        }


def _smaller(image: Any, wide: int, tall: int) -> Any:
    """An 8-bit blue-green-red ``image`` averaged down to ``wide`` by ``tall``.

    Each pixel is the mean of the area it covers, so a thin line between two
    surfaces fades rather than vanishing between samples. Pillow's box filter
    does it in C, five times faster than the same sums in numpy.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    picture = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
    return np.asarray(picture.resize((max(1, wide), max(1, tall)), Image.Resampling.BOX))[:, :, ::-1]


_TO_LINEAR: Any = None


def _lab_of(image: Any) -> Any:
    """CIE L*a*b* of an 8-bit blue-green-red picture, each channel on a 0-255 scale.

    Differences in this space track differences a person sees, which is what
    an edge between two surfaces is.
    """
    import numpy as np  # noqa: PLC0415

    global _TO_LINEAR
    if _TO_LINEAR is None:
        level = np.arange(256, dtype=np.float64) / 255.0
        _TO_LINEAR = np.where(level <= 0.04045, level / 12.92, ((level + 0.055) / 1.055) ** 2.4).astype(
            np.float32
        )
    blue, green, red = (_TO_LINEAR[image[:, :, channel]] for channel in range(3))
    x = (0.412453 * red + 0.357580 * green + 0.180423 * blue) / 0.950456
    y = 0.212671 * red + 0.715160 * green + 0.072169 * blue
    z = (0.019334 * red + 0.119193 * green + 0.950227 * blue) / 1.088754

    def bend(t: Any) -> Any:
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = bend(x), bend(y), bend(z)
    lab = np.empty(image.shape[:2] + (3,), np.float32)
    lab[:, :, 0] = np.where(y > 0.008856, 116.0 * fy - 16.0, 903.3 * y) * 2.55
    lab[:, :, 1] = 500.0 * (fx - fy) + 128.0
    lab[:, :, 2] = 200.0 * (fy - fz) + 128.0
    return np.clip(np.rint(lab, out=lab), 0, 255, out=lab)


def panels_in(image: Any) -> list[Panel]:
    """Every region of even colour with a clear edge around it."""
    if image is None:
        return []
    import numpy as np  # noqa: PLC0415

    try:
        from scipy import ndimage  # noqa: PLC0415
    except ImportError as why:
        _say_once(f"no panels can be found without scipy: {why}")
        return []

    tall, wide = image.shape[:2]
    shrink = min(1.0, _WORKING_SIDE / float(max(tall, wide)))
    small = _smaller(image, int(wide * shrink), int(tall * shrink)) if shrink < 1.0 else image
    lab = _lab_of(small)
    across = np.zeros_like(lab)
    down = np.zeros_like(lab)
    across[:, 1:-1] = lab[:, 2:] - lab[:, :-2]
    down[1:-1, :] = lab[2:, :] - lab[:-2, :]
    strength = np.sqrt((across * across + down * down).sum(axis=2))
    edges = strength > _EDGE
    # Thicken each edge by a pixel so a surface is closed off from its neighbour.
    wider = edges.copy()
    wider[:, 1:] |= edges[:, :-1]
    thick = wider.copy()
    thick[1:, :] |= wider[:-1, :]
    # A surface is a connected stretch with no edge in it, and its outline is
    # what encloses it: a tile with a number drawn on it is one surface with
    # holes, and the holes are part of what it covers.
    surfaces, _count = ndimage.label(~thick)
    s_tall, s_wide = small.shape[:2]
    least = max(8, int(0.02 * min(s_tall, s_wide)))
    found: list[tuple[int, int, int, int]] = []
    for label, where in enumerate(ndimage.find_objects(surfaces), start=1):
        if where is None:
            continue
        y, x = where[0].start, where[1].start
        h, w = where[0].stop - y, where[1].stop - x
        if w < least or h < least or (w > 0.97 * s_wide and h > 0.97 * s_tall):
            continue
        covers = ndimage.binary_fill_holes(surfaces[where] == label)
        if int(covers.sum()) < _FILLS * w * h:
            continue
        found.append((x, y, w, h))
    found.sort(key=lambda box: box[2] * box[3])
    kept: list[tuple[int, int, int, int]] = []
    for box in found:
        if any(_nearly_the_same(box, other) for other in kept):
            continue
        kept.append(box)
    panels: list[Panel] = []
    for x, y, w, h in kept:
        inset = max(1, min(w, h) // 8)
        middle = lab[y + inset : y + h - inset, x + inset : x + w - inset]
        if middle.size == 0:
            continue
        tone = tuple(float(v) for v in np.median(middle.reshape(-1, 3), axis=0))
        panels.append(
            Panel(
                left=x / s_wide,
                top=y / s_tall,
                width=w / s_wide,
                height=h / s_tall,
                tone=tone,  # type: ignore[arg-type]
            )
        )
    return panels


def _nearly_the_same(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return False
    shared = (right - left) * (bottom - top)
    return shared >= 0.8 * max(aw * ah, bw * bh)


def _lines(values: Sequence[float], apart: float) -> list[float]:
    """The lines a set of positions sits on, when lines are at least ``apart``."""
    ordered = sorted(values)
    groups: list[list[float]] = []
    for value in ordered:
        if groups and value - groups[-1][-1] < apart:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _even(lines: Sequence[float]) -> bool:
    """Whether lines are set at one pitch, allowing a reading's own jitter."""
    if len(lines) < 2:
        return False
    gaps = [b - a for a, b in zip(lines, lines[1:], strict=False)]
    pitch = sorted(gaps)[len(gaps) // 2]
    return pitch > 0 and all(abs(gap - pitch) <= 0.2 * pitch for gap in gaps)


def grids_in(panels: Sequence[Panel]) -> list[Grid]:
    """Groups of same-sized panels set out in rows and columns.

    Two by two at least, because one row of buttons is a row and not a grid,
    and most of the places a grid should have have to be panels: a grid with
    most of its places missing is a coincidence of alignment.
    """
    remaining = sorted(panels, key=lambda p: p.width * p.height, reverse=True)
    grids: list[Grid] = []
    used: set[int] = set()
    for index, seed in enumerate(remaining):
        if index in used:
            continue
        family = [
            other_index
            for other_index, other in enumerate(remaining)
            if other_index not in used
            and abs(other.width - seed.width) <= _SAME_SIZE * seed.width
            and abs(other.height - seed.height) <= _SAME_SIZE * seed.height
        ]
        if len(family) < 4:
            continue
        members = [remaining[i] for i in family]
        width = sorted(p.width for p in members)[len(members) // 2]
        height = sorted(p.height for p in members)[len(members) // 2]
        across = _lines([p.center[0] for p in members], width * 0.5)
        down = _lines([p.center[1] for p in members], height * 0.5)
        if len(across) < 2 or len(down) < 2:
            continue
        if not (_even(across) and _even(down)):
            continue
        # Places only as far apart as a place is big, give or take a gap. Two
        # buttons at opposite ends of a window share a size and not a grid.
        if (across[1] - across[0]) > 1.6 * width or (down[1] - down[0]) > 1.6 * height:
            continue
        if len(members) < 0.6 * len(across) * len(down):
            continue
        grids.append(
            Grid(
                rows=len(down),
                columns=len(across),
                down_at=tuple(down),
                across_at=tuple(across),
                cell_width=width,
                cell_height=height,
            )
        )
        used.update(family)
    return grids


def recognize_text(image: Any) -> list[dict[str, Any]]:
    """Every run of text in a BGR picture, with where it sits. Fractions, top-left.

    Language correction off: a window is not prose, and correction turns a
    lone "4" into a guess. Measured on a game window, the same pass is four
    times faster without it and no less accurate on the words.
    """
    if image is None:
        return []
    try:
        import numpy as np  # noqa: PLC0415
        import Quartz  # noqa: PLC0415
        from Vision import (  # noqa: PLC0415
            VNImageRequestHandler,
            VNRecognizeTextRequest,
            VNRequestTextRecognitionLevelAccurate,
        )
    except ImportError:
        return []
    try:
        tall, wide = image.shape[:2]
        rgba = np.dstack([image[:, :, 2], image[:, :, 1], image[:, :, 0], np.full((tall, wide), 255, np.uint8)])
        rgba = np.ascontiguousarray(rgba)
        provider = Quartz.CGDataProviderCreateWithData(None, rgba.tobytes(), rgba.nbytes, None)
        picture = Quartz.CGImageCreate(
            wide, tall, 8, 32, wide * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
            Quartz.kCGRenderingIntentDefault,
        )
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(picture, {})
        ok, _error = handler.performRequests_error_([request], None)
        if not ok:
            return []
        regions: list[dict[str, Any]] = []
        for observation in list(request.results() or []):
            best = list(observation.topCandidates_(1) or [])
            if not best:
                continue
            said = str(best[0].string() or "").strip()
            if not said:
                continue
            box = observation.boundingBox()
            x, y = float(box.origin.x), float(box.origin.y)
            w, h = float(box.size.width), float(box.size.height)
            regions.append(
                {
                    "text": said,
                    "x": round(x, 5),
                    "y": round(1.0 - (y + h), 5),
                    "width": round(w, 5),
                    "height": round(h, 5),
                    "center_x": round(x + w / 2.0, 5),
                    "center_y": round(1.0 - (y + h / 2.0), 5),
                    "confidence": round(float(best[0].confidence()), 4),
                }
            )
        return regions
    except (AttributeError, RuntimeError, TypeError, ValueError) as why:
        logger.debug("text recognition did not run: %s", why)
        return []


@dataclass
class _Seen:
    look: Any
    says: str
    times: int = 1


@dataclass
class Looker:
    """Somebody looking at one window over time, and getting used to it.

    Holds what the things in it have been read as, by how they look, so a
    thing that looks exactly like one already read is recognised rather than
    read again.
    """

    seen: list[_Seen] = field(default_factory=list)
    #: How the empty places of each grid look, by the grid's shape.
    blank: dict[tuple[int, int], Any] = field(default_factory=dict)
    #: How the window looked around its grids when its words were last read,
    #: and the words that were outside them.
    around: Any = None
    around_says: tuple[dict[str, Any], ...] = ()

    def _look_of(self, image: Any, panel: Panel) -> Any:
        tall, wide = image.shape[:2]
        x0 = int(max(0, panel.left) * wide)
        y0 = int(max(0, panel.top) * tall)
        x1 = int(min(1.0, panel.left + panel.width) * wide)
        y1 = int(min(1.0, panel.top + panel.height) * tall)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return _lab_of(_smaller(crop, _LOOK_SIDE, _LOOK_SIDE))

    def _around(self, image: Any, grids: Sequence[Grid]) -> Any:
        """The picture with the grids taken out of it, small enough to compare."""
        import numpy as np  # noqa: PLC0415

        tall, wide = image.shape[:2]
        shrink = _AROUND_SIDE / float(max(1, max(tall, wide)))
        blanked = _smaller(
            image, max(1, int(wide * shrink)), max(1, int(tall * shrink))
        ).astype(np.int16)
        s_tall, s_wide = blanked.shape[:2]
        for grid in grids:
            left, top, right, bottom = grid.outline
            blanked[
                max(0, int(top * s_tall)) : max(0, int(math.ceil(bottom * s_tall))),
                max(0, int(left * s_wide)) : max(0, int(math.ceil(right * s_wide))),
            ] = 0
        return blanked

    def _same_around(self, image: Any, grids: Sequence[Grid]) -> bool:
        import numpy as np  # noqa: PLC0415

        now = self._around(image, grids)
        before = self.around
        if before is None or getattr(before, "shape", None) != now.shape:
            self.around = now
            return False
        same = float(np.abs(now - before).mean()) < _STILL
        self.around = now
        return same

    def _remember_around(
        self, image: Any, grids: Sequence[Grid], layout: Sequence[dict[str, Any]]
    ) -> None:
        """Keep the words that were outside the grids, and how that part looked."""
        if not grids:
            self.around, self.around_says = None, ()
            return
        outside = []
        for one in layout:
            x = float(one.get("center_x", -1.0) or -1.0)
            y = float(one.get("center_y", -1.0) or -1.0)
            within = False
            for grid in grids:
                left, top, right, bottom = grid.outline
                within = within or (left <= x <= right and top <= y <= bottom)
            if not within:
                outside.append(dict(one))
        self.around = self._around(image, grids)
        self.around_says = tuple(outside)

    @staticmethod
    def _apart(a: Any, b: Any) -> float:
        import numpy as np  # noqa: PLC0415

        if a is None or b is None:
            return math.inf
        return float(np.sqrt(((a - b) ** 2).sum(axis=2)).mean())

    def recognised(self, look: Any) -> str | None:
        """What this place says, when its look says one thing and not two.

        Two things whose surrounds differ by a few units are the same look at
        this size, and answering with the nearer of them is a guess: a 256
        was read as a 128 in half the glances of a game, because their
        backgrounds are four units apart and the digits are a tenth of the
        square. When more than one thing she has read is this close, the
        honest answer is that she cannot tell, and it is read again.
        """
        best: tuple[float, _Seen] | None = None
        others: set[str] = set()
        for one in self.seen:
            apart = self._apart(look, one.look)
            if apart >= _SAME_LOOK:
                continue
            others.add(one.says)
            if best is None or apart < best[0]:
                best = (apart, one)
        if best is None or len(others) > 1:
            return None
        return best[1].says

    def learned(self, look: Any, says: str) -> None:
        if look is None or not says:
            return
        for one in self.seen:
            if self._apart(look, one.look) < _SAME_LOOK * 0.5:
                if one.says == says:
                    one.times += 1
                    return
        self.seen.append(_Seen(look=look, says=says))
        if len(self.seen) > 400:
            self.seen.sort(key=lambda s: s.times, reverse=True)
            del self.seen[300:]

    def read(self, image: Any, *, words: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        """One reading of a window: its words, its panels and its grids.

        The shape matches every other screen reading here — ``text`` and a
        ``layout`` of positioned runs — with ``grids`` added, each carrying
        what every one of its places says, empty places included.
        """
        if image is None:
            return {"ok": False, "text": "", "layout": [], "grids": [], "error": "no picture"}
        panels = panels_in(image)
        grids = grids_in(panels)
        if words is not None:
            layout = list(words)
        elif grids and self._same_around(image, grids):
            # Words she has already read, in a part of the picture that has
            # not changed, are the same words. What is inside the grids is
            # read from the places themselves, so nothing here is kept stale.
            layout = [dict(one) for one in self.around_says]
        else:
            layout = recognize_text(image)
            self._remember_around(image, grids, layout)
        read_grids: list[dict[str, Any]] = []
        extra: list[dict[str, Any]] = []
        for grid in grids:
            says: dict[tuple[int, int], str] = {}
            looks: dict[tuple[int, int], Any] = {}
            covered = False
            for region in layout:
                # A run of text wider or taller than one place is not the
                # contents of a place. It is something lying across the grid —
                # a message over a finished board, a caption — or two places
                # read as one line, and giving it to whichever place its middle
                # lands in puts "hain 8" where an 8 is.
                if (
                    float(region.get("width", 0.0) or 0.0) > grid.cell_width * 1.05
                    or float(region.get("height", 0.0) or 0.0) > grid.cell_height * 1.05
                ):
                    covered = True
                    continue
                spot = grid.where(float(region["center_x"]), float(region["center_y"]))
                if spot is not None:
                    says[spot] = (says.get(spot, "") + " " + str(region["text"])).strip()
            for row in range(grid.rows):
                for column in range(grid.columns):
                    looks[(row, column)] = self._look_of(image, grid.place(row, column))
            for spot, text in says.items():
                self.learned(looks.get(spot), text)
            blank = self._blank_look(grid, looks, says)
            unread: list[tuple[int, int]] = []
            remembered: dict[tuple[int, int], str] = {}
            for spot, look in looks.items():
                if spot in says or look is None:
                    continue
                if self._looks_empty(look, blank):
                    continue
                known = self.recognised(look)
                if known:
                    remembered[spot] = known
                else:
                    unread.append(spot)
            if unread:
                # Every place she has something for goes into the strip beside
                # the ones she has not read. Recognition reads a line and not
                # a square: one digit alone came back empty from the same
                # picture whose four digits, laid side by side, all read. So a
                # place is never read alone, and the places she thinks she
                # knows are read again for nothing — which is how a look
                # learned as the wrong thing stops being believed for ever.
                company = sorted(set(unread) | set(remembered))
                fresh = self._read_as_a_strip(image, grid, company)
                for spot in company:
                    text = fresh.get(spot, "")
                    if not text:
                        text = remembered.get(spot, "")
                    if text:
                        says[spot] = text
                        self.learned(looks.get(spot), text)
            else:
                says.update(remembered)
            unsure = [spot for spot in unread if spot not in says]
            places = []
            for row in range(grid.rows):
                for column in range(grid.columns):
                    text = says.get((row, column), "")
                    places.append(text)
                    if text and (row, column) not in {
                        grid.where(float(r["center_x"]), float(r["center_y"])) for r in layout
                    }:
                        # Put where it was read, so a reader of the layout alone
                        # sees what the grid saw.
                        extra.append(
                            {
                                "text": text,
                                "center_x": grid.across_at[column],
                                "center_y": grid.down_at[row],
                                "x": grid.across_at[column] - grid.cell_width / 4.0,
                                "y": grid.down_at[row] - grid.cell_height / 4.0,
                                "width": grid.cell_width / 2.0,
                                "height": grid.cell_height / 2.0,
                                "confidence": 0.9,
                                "recognised": True,
                            }
                        )
            read_grids.append(
                {
                    **grid.as_dict(),
                    "says": places,
                    # Places that look like something and could not be read.
                    # Usually a thing still moving; a reading with any is not
                    # a settled picture of the grid.
                    "unsure": [list(spot) for spot in unsure],
                    # Text lying across the places, which is usually
                    # something put over the grid rather than in it.
                    "covered": covered,
                }
            )
        full = layout + extra
        return {
            "ok": True,
            "text": " ".join(str(region["text"]) for region in full),
            "layout": full,
            "grids": read_grids,
            "panels": len(panels),
        }

    def _blank_look(
        self, grid: Grid, looks: dict[tuple[int, int], Any], says: dict[tuple[int, int], str]
    ) -> Any:
        """How an empty place of this grid looks.

        The look most of the unwritten places share. Remembered per shape of
        grid, because a grid that has filled up has few empty places left to
        tell from and the ones it had earlier were the same colour.
        """
        shape = (grid.rows, grid.columns)
        silent = [
            look
            for spot, look in looks.items()
            # A place she has read something from is not a candidate for what
            # nothing looks like, however many of them there are. On a board
            # of mostly twos the commonest unread look IS a two, and taking it
            # for the empty look makes every two on the board disappear.
            if spot not in says and look is not None and self.recognised(look) is None
        ]
        best: tuple[int, Any] | None = None
        for look in silent:
            alike = sum(1 for other in silent if self._apart(look, other) < _SAME_LOOK)
            if best is None or alike > best[0]:
                best = (alike, look)
        if best is not None and best[0] >= 2:
            known = self.blank.get(shape)
            if known is None or self._apart(known, best[1]) < _SAME_LOOK or best[0] >= 3:
                self.blank[shape] = best[1]
        return self.blank.get(shape)

    def _looks_empty(self, look: Any, blank: Any) -> bool:
        """Whether a place holds nothing: it looks more like an empty one than like anything written.

        Nearness to the empty look alone is not enough. The palest thing a
        place can hold sits a few units from an empty place and well inside
        any fixed distance, so every one of them was skipped as empty — six
        at a time, whole rows of a board (offline, drawn as pixels,
        2026-09-17). What settles it is which it is nearer to.
        """
        if look is None or blank is None:
            return False
        to_blank = self._apart(look, blank)
        if to_blank >= _SAME_LOOK:
            return False
        to_written = min((self._apart(look, one.look) for one in self.seen), default=math.inf)
        return to_blank <= to_written

    def _read_as_a_strip(
        self, image: Any, grid: Grid, spots: Sequence[tuple[int, int]]
    ) -> dict[tuple[int, int], str]:
        """Read several places at once, laid side by side with space between.

        Recognition finds a line of things far more reliably than one thing
        alone in a square: the same four crops that came back with half of
        them missing one at a time were all read, laid out as a line.
        """
        import numpy as np  # noqa: PLC0415

        tall, wide = image.shape[:2]
        crops: list[Any] = []
        for row, column in spots:
            panel = grid.place(row, column)
            x0 = int(max(0.0, panel.left) * wide)
            y0 = int(max(0.0, panel.top) * tall)
            x1 = int(min(1.0, panel.left + panel.width) * wide)
            y1 = int(min(1.0, panel.top + panel.height) * tall)
            crops.append(image[y0:y1, x0:x1])
        if not crops:
            return {}
        gap = max(8, max(crop.shape[0] for crop in crops) // 2)
        height = max(crop.shape[0] for crop in crops) + 2 * gap
        pieces: list[Any] = []
        starts: list[tuple[int, int]] = []
        at = gap
        pieces.append(np.full((height, gap, 3), 255, np.uint8))
        for crop in crops:
            # Each place sits in its own white surround. Laid edge to edge,
            # the coloured squares of two places run together and come back
            # as one run of text across both — "1( 6" for a 16 beside a 2 —
            # which is worse than not reading them, because it is read into
            # one of the places as its contents.
            padded = np.full((height, crop.shape[1], 3), 255, np.uint8)
            padded[gap : gap + crop.shape[0], : crop.shape[1]] = crop
            pieces.append(padded)
            starts.append((at, at + crop.shape[1]))
            at += crop.shape[1]
            pieces.append(np.full((height, gap, 3), 255, np.uint8))
            at += gap
        strip = np.hstack(pieces)
        found: dict[tuple[int, int], str] = {}
        strip_wide = strip.shape[1]
        for region in recognize_text(strip):
            left = float(region.get("x", region["center_x"])) * strip_wide
            right = left + float(region.get("width", 0.0)) * strip_wide
            middle = float(region["center_x"]) * strip_wide
            for spot, (start, end) in zip(spots, starts, strict=False):
                if start <= middle <= end:
                    if left < start - gap / 2.0 or right > end + gap / 2.0:
                        # A run across two places belongs to neither.
                        break
                    found[spot] = (found.get(spot, "") + " " + str(region["text"])).strip()
                    break
        return found


#: The one looker per application, so what she has learned to recognise in a
#: window lasts as long as she keeps looking at it.
_LOOKERS: dict[str, Looker] = {}

#: How long a window may go on changing before she reads it anyway. A moving
#: thing that never stops moving is still worth a reading; she says it was not
#: still.
_STILL_WITHIN_S = 1.5

#: Two pictures are the same picture when their pixels differ by less than
#: this on average, in 0..255. Compression and cursor blink sit below it.
_STILL = 0.05

#: How many times running a picture has to agree with the one before it to
#: count as having stopped. Two pictures agreeing is not enough: a window
#: redraws in parts, and a picture taken across a redraw holds one part of
#: what the act did and another part of what was there before. Read as a
#: state, that is a world doing something no rule can explain — live,
#: 2026-09-17, one row of a board slid and another had not, and her rule sat
#: at 88% of what it watched because of pairs like it.
_AGREEING_LOOKS = 2


_SAID: set[str] = set()


#: The windows she has already said how she first saw.
_LOOKED_AT: set[str] = set()


def _say_once(why: str) -> None:
    """Why this way of looking stood aside, said once per reason."""
    if why not in _SAID:
        _SAID.add(why)
        logger.info("%s", why)


def _nothing_drawn(image: Any) -> bool:
    """Whether a picture has nothing in it: one colour from edge to edge.

    What a capture gives back when the process asking may not see other
    applications' pixels. It is a refusal wearing the shape of a picture, and
    read as one it is a window that has emptied.
    """
    try:
        return float(image[::8, ::8].std()) < 1.0
    except (AttributeError, TypeError, ValueError):
        return True


#: Which way of taking a window's pixels has worked, so a way that is refused
#: is not asked again on every look.
_HOW_PIXELS_COME: dict[str, str] = {"way": ""}


async def _the_pixels_of(window: Any) -> Any:
    """A window's pixels, by whichever authority this process has.

    In this process first, which is fastest. Where this process is not allowed
    to see other applications, the resident desktop bridge — the signed
    application that holds that permission — takes the frame instead, which it
    can do for the window in front.
    """
    import asyncio  # noqa: PLC0415

    from core.capabilities import window_server  # noqa: PLC0415

    if _HOW_PIXELS_COME["way"] != "bridge":
        image = await asyncio.to_thread(window_server.capture, window)
        if image is not None and not _nothing_drawn(image):
            _HOW_PIXELS_COME["way"] = "here"
            return image
        _say_once("this process cannot see other windows' pixels; asking the desktop bridge")
    image = await asyncio.to_thread(_a_frame_from_the_bridge, window)
    if image is not None:
        _HOW_PIXELS_COME["way"] = "bridge"
    return image


def _a_frame_from_the_bridge(window: Any) -> Any:
    try:
        import base64  # noqa: PLC0415
        import io  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from core.security.native_desktop_bridge import (
            invoke_native_desktop_bridge,  # noqa: PLC0415
        )

        answer = invoke_native_desktop_bridge(
            "observe_foreground_frame", read_only=True, timeout=3.0, allow_one_shot=False
        )
    except (ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as why:
        _say_once(f"the desktop bridge could not be asked for a frame: {type(why).__name__}")
        return None
    if not answer.get("ok"):
        _say_once(f"the desktop bridge gave no frame: {answer.get('error')}")
        return None
    if int(answer.get("window_id") or 0) != int(getattr(window, "number", -1)):
        # The frame is of whatever is in front, and that is not her window.
        return None
    try:
        png = base64.b64decode(str(answer.get("frame_base64") or ""))
        with Image.open(io.BytesIO(png)) as picture:
            image = np.ascontiguousarray(np.asarray(picture.convert("RGB"))[:, :, ::-1])
    except (ValueError, TypeError, OSError) as why:
        _say_once(f"the desktop bridge's frame would not decode: {why}")
        return None
    return image


def looker_for(app: str) -> Looker:
    key = " ".join(str(app or "").split()).casefold()
    if key not in _LOOKERS:
        _LOOKERS[key] = Looker()
    return _LOOKERS[key]


def how_different(a: Any, b: Any) -> float:
    """How far two pictures of the same thing are apart. Public for the reader
    that runs in its own process."""
    return _how_different(a, b)


#: How still counts as still, for whoever is doing the looking.
STILL = _STILL

#: How many agreements in a row count as stopped, for the same.
AGREEING_LOOKS = _AGREEING_LOOKS


def crop_to(image: Any, over: tuple[float, float, float, float] | None) -> Any:
    """The part of a picture a caller is interested in. Public for the reader
    that runs in its own process."""
    return _crop(image, over)


def _crop(image: Any, over: tuple[float, float, float, float] | None) -> Any:
    if image is None or over is None:
        return image
    tall, wide = image.shape[:2]
    left, top, right, bottom = over
    x0, x1 = int(max(0.0, left) * wide), int(min(1.0, right) * wide)
    y0, y1 = int(max(0.0, top) * tall), int(min(1.0, bottom) * tall)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return image
    return image[y0:y1, x0:x1]


def _how_different(a: Any, b: Any) -> float:
    import numpy as np  # noqa: PLC0415

    if a is None or b is None or a.shape != b.shape:
        return math.inf
    return float(np.abs(a[::4, ::4].astype(np.int16) - b[::4, ::4].astype(np.int16)).mean())


async def look_at_window(
    app: str,
    over: tuple[float, float, float, float] | None = None,
    *,
    wait_for_stillness: bool = True,
    still_within_s: float = _STILL_WITHIN_S,
) -> dict[str, Any] | None:
    """Read one application's window from its own pixels. None when it cannot.

    None means this way of looking is not available — no window server, no
    window by that name, no picture — and the caller should look some other
    way. A refusal on privacy grounds is not None: it is an answer, returned
    with ``ok`` False and the reason, so nobody looks round it.
    """
    import asyncio  # noqa: PLC0415
    import time  # noqa: PLC0415

    from core.capabilities import window_server  # noqa: PLC0415
    from core.security.screen_capture_policy import (  # noqa: PLC0415
        evaluate_window_capture_admission_async,
    )

    if not str(app or "").strip():
        return None
    window = await asyncio.to_thread(window_server.window_of, app)
    if window is None:
        _say_once(f"no window for {app!r} in the window list, so reading it the old way")
        return None
    admission = await evaluate_window_capture_admission_async(window.owner, window.title)
    if not admission.allowed:
        _say_once(f"reading {window.owner!r} was refused: {admission.reason}")
        return {
            "ok": False,
            "text": "",
            "layout": [],
            "grids": [],
            "error": getattr(admission, "public_error", "") or str(admission.reason),
            "refused_because": str(admission.reason),
        }
    began = time.monotonic()
    # Somewhere else if they will have it: a reading is mostly Python holding
    # the interpreter, and in her own process it waits behind everything else
    # she is doing — an eighth of a second of work took nearly a second while
    # she played (live, 2026-09-17).
    from core.perception.eyes_of_their_own import look_through_them  # noqa: PLC0415

    elsewhere = await asyncio.to_thread(
        look_through_them, window, over, wait_for_stillness, still_within_s
    )
    if elsewhere is not None:
        picture_shape = tuple(elsewhere.pop("_shape", ()) or ())
        still = bool(elsewhere.pop("_settled", True))
        reading = elsewhere
        looked_took = float(reading.pop("_looked_took", 0.0) or 0.0)
    else:
        async def take() -> Any:
            return _crop(await _the_pixels_of(window), over)

        picture = await take()
        if picture is None:
            return None
        agreed = 0
        still = not wait_for_stillness
        while not still and time.monotonic() - began < still_within_s:
            again = await take()
            if again is None:
                break
            agreed = agreed + 1 if _how_different(picture, again) < _STILL else 0
            still = agreed >= _AGREEING_LOOKS
            picture = again
        looked_took = time.monotonic() - began
        reading = await asyncio.to_thread(looker_for(window.owner).read, picture)
        picture_shape = (int(picture.shape[1]), int(picture.shape[0]))
    front = await asyncio.to_thread(window_server.front_owner)
    left, top, wide, tall = window.bounds
    bounds = [left, top, wide, tall]
    if over is not None:
        l, t, r, b = over
        bounds = [left + int(l * wide), top + int(t * tall), max(1, int((r - l) * wide)), max(1, int((b - t) * tall))]
    if window.owner not in _LOOKED_AT:
        _LOOKED_AT.add(window.owner)
        grids_seen = [(g["rows"], g["columns"]) for g in reading.get("grids") or []]
        wide_px, tall_px = (picture_shape + (0, 0))[:2]
        logger.info(
            "first look at %r by %s: %dx%d, %s panel(s), grids %s, %d run(s) of text",
            window.owner,
            "eyes of their own" if elsewhere is not None else (_HOW_PIXELS_COME["way"] or "nothing"),
            wide_px, tall_px,
            reading.get("panels", 0), grids_seen, len(reading.get("layout") or []),
        )
    reading.update(
        {
            "scoped_to": app,
            "bounds": bounds,
            "read_within": "the part" if over is not None else "the window",
            "in_front_then": front,
            "her_window_showing": bool(window.on_screen),
            "at": time.time(),
            "settled": bool(still),
            "window_number": window.number,
            "owner": window.owner,
            "pid": window.pid,
            "seconds_to_still": round(looked_took, 3),
            "seconds_reading": round(time.monotonic() - began - looked_took, 3),
        }
    )
    return reading
