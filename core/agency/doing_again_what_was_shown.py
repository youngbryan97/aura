"""Doing again what was shown: a sequence read or watched, then done in the same order.

A screen that shows the way and waits for it to be followed asks for one thing:
do again what it showed, in its order. It shows it one of two ways. It draws
it: a row of arrows, the keys to press in turn ("follow the arrow keys in the
book"). Or it plays it: places that light up one after another (a dance's
steps, a lock's buttons, a tutorial that flashes what to click), and then
waits. Either way the order is the thing.

An arrow is read from its shape alone: along its longer axis, the end that
is wider is its head, and the way it points is the way of the key with the
same name. Places that light up are found by watching each stand brighter than
it usually is, one at a time; the order they lit in is the order to click them.
Then she does the same, at a person's pace, and watches whether the screen
answers with something new (the next step, a score), which says she did it
right.

Nothing here knows what is being shown or why: a dance, a spell, a code, a
lesson in which buttons to press.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.DoingAgainWhatWasShown")

__all__ = ["arrows_in", "asks_to_follow", "do_again", "lit_in_turn", "Shown"]

#: Words that ask for what was shown to be done again.
_FOLLOW = re.compile(
    r"\b(?:follow|repeat|copy|match|mimic|do the same|same order|in order|memori[sz]e|remember the|watch and|"
    r"simon says|press the (?:arrow|keys?) (?:shown|that appear|as they appear))\b", re.I)

#: Arrow characters a reading of the screen may give for drawn arrows.
_GLYPHS = {"←": "left", "→": "right", "↑": "up", "↓": "down", "<": "left", ">": "right", "^": "up", "v": "down"}

#: How much brighter than usual a place must be to be lit, in grey levels.
LIT_BY = 40.0

#: The time between two acts of doing it again, in seconds: a person's pace, and a screen's time to answer.
BETWEEN_S = 0.35


def asks_to_follow(text: str) -> bool:
    """Whether a screen's words ask for what it shows to be done again."""
    return bool(_FOLLOW.search(" ".join(str(text or "").split())))


@dataclass
class Shown:
    """What was shown, in order: keys to press (``keys``) or places to click (``places``, shares of the picture)."""

    keys: list[str] = field(default_factory=list)
    places: list[tuple[float, float]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.keys or self.places)


def _the_way_it_points(mask: np.ndarray) -> str | None:
    """The way an arrow-shaped mask points: along its longer axis, toward its wider end."""
    rows, cols = np.nonzero(mask)
    if len(rows) < 12:
        return None
    tall, wide = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    across = wide >= tall
    along = cols - cols.min() if across else rows - rows.min()
    length = (wide if across else tall)
    widths = np.bincount(along, minlength=length)
    first, last = widths[: length // 3], widths[-(length // 3):]
    if not len(first) or not len(last):
        return None
    head_first, head_last = float(first.max()), float(last.max())
    if max(head_first, head_last) < 1.3 * min(head_first, head_last):
        return None  # no end wider than the other: not an arrow
    if across:
        return "left" if head_first > head_last else "right"
    return "up" if head_first > head_last else "down"


def arrows_in(picture: Any, regions: list[dict[str, Any]] | None = None) -> list[str]:
    """The arrows drawn on a still picture, left to right and then top to bottom, as the keys they name.

    Read from the writing first where it gives arrow characters; else from the shapes that stand out, each
    read by its own outline.
    """
    for region in sorted(regions or [], key=lambda r: (round(float(r.get("center_y", 0)) * 10), float(r.get("center_x", 0)))):
        glyphs = [_GLYPHS[ch] for ch in str(region.get("text") or "") if ch in _GLYPHS and ch not in "v^<>"]
        if len(glyphs) >= 2:
            return glyphs
    from core.perception.picture_arithmetic import pieces

    pixels = np.asarray(picture, dtype=np.float32)
    grey = pixels.mean(axis=2) if pixels.ndim == 3 else pixels
    background = float(np.median(grey))
    standing_out = np.abs(grey - background) > LIT_BY
    count, labels, stats, _centres = pieces(standing_out.astype(np.uint8))
    tall, wide = grey.shape
    found = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 20 or w * h > 0.2 * tall * wide or min(w, h) < 4:
            continue
        way = _the_way_it_points(labels[y:y + h, x:x + w] == label)
        if way:
            found.append((round((y + h / 2) / max(1, tall) * 5), (x + w / 2) / max(1, wide), way))
    return [way for _row, _x, way in sorted(found)]


async def lit_in_turn(look: Callable[[], Awaitable[Any]], seconds: float, *, places: int = 16) -> list[tuple[float, float]]:
    """Watch for ``seconds``: the places that light up one after another, in the order they lit, as shares of the
    picture. A place is a cell of a coarse grid over the picture; it is lit where it stands brighter than its usual."""
    cells = int(np.ceil(np.sqrt(places)))
    watched: list[np.ndarray] = []
    began = None
    while True:
        seen = await look()
        if seen is None:
            break
        picture, at = seen
        began = at if began is None else began
        grey = np.asarray(picture, dtype=np.float32)
        grey = grey.mean(axis=2) if grey.ndim == 3 else grey
        tall, wide = grey.shape
        watched.append(np.array([[grey[r * tall // cells:(r + 1) * tall // cells, c * wide // cells:(c + 1) * wide // cells].mean()
                                  for c in range(cells)] for r in range(cells)]))
        if at - began >= seconds:
            break
    if not watched:
        return []
    # How each place usually looks is how it looked most of the time it was watched: a place is lit only now and then,
    # and the watching may begin while one already is.
    usual = np.median(np.stack(watched), axis=0)
    lit: list[tuple[float, float]] = []
    lit_now: tuple[int, int] | None = None
    for means in watched:
        brighter = means - usual
        row, col = (int(v) for v in np.unravel_index(int(np.argmax(brighter)), brighter.shape))
        if brighter[row, col] >= LIT_BY:
            # A place lit now that was not lit a moment ago is the next in the order.
            if (row, col) != lit_now:
                lit.append(((col + 0.5) / cells, (row + 0.5) / cells))
                lit_now = (row, col)
        else:
            lit_now = None
    return lit


async def do_again(shown: Shown, hands: Any, *, say: Callable[[str], Any] | None = None) -> int:
    """Do what was shown, in its order, at a person's pace: each key pressed, or each place clicked. How many acts."""
    done = 0
    if say is not None and shown:
        what = ", ".join(shown.keys) if shown.keys else f"{len(shown.places)} places in the order they lit"
        say(f"Doing it again as it was shown: {what}.")
    for key in shown.keys:
        await hands.tap(key)
        done += 1
        await asyncio.sleep(BETWEEN_S)
    for place in shown.places:
        await hands.click(*place)
        done += 1
        await asyncio.sleep(BETWEEN_S)
    return done
