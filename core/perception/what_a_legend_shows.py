"""What a screen that draws things beside words about them says each thing is: a legend.

A game's rules screen, a map's key, a manual's parts list and a toolbar's help
all do the same thing: they draw a thing and write beside it, or under it, what
it is or what to do about it. "Candy: gives you invincibility." "Heart:
restores your health." A robot dog over "Destroy the robot dogs". A person reads
it once, and from then on knows the candy when it floats by.

LIVE 2026-10-09 a game's rules screen drew its candy, its heart and its enemies
over what each was for, and she read the words aloud and met every one of them
in play as a stranger, learning what each did by touching it.

So each caption is read for what it says to do about its thing (get it, keep
clear of it, shoot it), and the picture beside it is found: the shapes just
above the caption, or just before it on its line, that stand apart from the
panel they are drawn on. Each is kept as a colour mix measured the way things in
play are measured (core/perception/what_moves_in_the_picture.py), so a thing in
play that looks like one is known by it. Sizes are not compared: a legend draws
its things larger or smaller than play does.

Nothing here knows a game. A legend read wrongly is overruled by play
(core/agency/what_meeting_things_does.py: what she was told holds until the
evidence against it is clear).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Shown", "most_like", "stance_of", "what_a_legend_shows"]

#: What a caption says to do about its thing, by its words; the earliest that says anything decides.
_SAYS = (
    ("avoid", re.compile(r"\b(?:avoid|watch out|look out|beware|don'?t (?:touch|hit|get hit|let)|dodge|danger\w*|"
                         r"hurts?|harm\w*|deadly|poison\w*|careful|lose (?:a )?(?:life|lives|health|time))\b", re.I)),
    ("shoot", re.compile(r"\b(?:destroy|shoot|defeat|zap|blast|attack|smash|knock out|kill|fight|punch|stop)\b", re.I)),
    ("meet", re.compile(r"(?:\b(?:collect|get|grab|catch|pick up|restores?|gives?|extra|bonus|points?|pts|health|"
                        r"lives?|power|invincib\w*|speed|shield|heals?|free|save|rescue)\b|\+\s?\d+)", re.I)),
)

#: How far from the panel's own colour, on its most different channel, a pixel must be to be part of a thing drawn on it.
APART_FROM_THE_PANEL = 48

#: The least share of the picture a drawn thing covers, and how filled its box must be to be a thing, not a rule line.
SMALLEST = 0.0012
FILLED = 0.15

#: How far above a caption, and before it on its line, its thing may be drawn, as shares of the picture.
ABOVE = 0.24
BEFORE = 0.2

#: How alike two colour mixes must be (0 the same, 1 nothing shared) for a thing in play to be one a legend drew; and,
#: where the thing's own colour is the drawing's to within SAME_COLOUR levels, how far apart the mixes may be.
ALIKE = 0.45
LOOSER = 0.6
SAME_COLOUR = 30


@dataclass(frozen=True)
class Shown:
    """One thing a legend drew, and what its words say of it."""

    words: str
    stance: str
    look: Any
    colour: tuple[int, int, int]
    #: Where it was drawn, as shares of the picture: left, top, right, bottom.
    where: tuple[float, float, float, float]


def stance_of(words: str) -> str:
    """What a caption says to do about its thing: "meet", "avoid", "shoot", or "" where it says nothing."""
    found = [(match.start(), stance) for stance, pattern in _SAYS for match in [pattern.search(words or "")] if match]
    return min(found)[1] if found else ""


def _captions(regions: Sequence[dict[str, Any]]) -> list[tuple[str, str, tuple[float, float, float, float]]]:
    """Each line that says what to do about something, with the short label lines just above it joined to it."""
    lines = [r for r in regions if str(r.get("text") or "").strip()]
    out = []
    for line in lines:
        text = " ".join(str(line["text"]).split())
        stance = stance_of(text)
        if not stance:
            continue
        x, y, w, h = (float(line.get(k, 0.0)) for k in ("x", "y", "width", "height"))
        left, top, right, bottom = x, y, x + w, y + h
        for label in lines:
            lx, ly, lw, lh = (float(label.get(k, 0.0)) for k in ("x", "y", "width", "height"))
            if label is line or stance_of(str(label["text"])) or len(str(label["text"]).split()) > 3:
                continue
            just_above = -0.5 * h <= top - (ly + lh) <= 1.5 * h and lx < right and lx + lw > left
            if just_above:
                text = f"{' '.join(str(label['text']).split())}: {text}"
                top, left, right = ly, min(left, lx), max(right, lx + lw)
        out.append((text, stance, (left, top, right, bottom)))
    return out


def _things_in(picture: np.ndarray, window: tuple[float, float, float, float], text_boxes: list[tuple[float, ...]]) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    """The shapes drawn in a window of the picture that stand apart from its panel: their pixels and where they are."""
    from core.perception.picture_arithmetic import pieces

    tall, wide = picture.shape[:2]
    left, top, right, bottom = (int(round(window[0] * wide)), int(round(window[1] * tall)),
                                int(round(window[2] * wide)), int(round(window[3] * tall)))
    left, top, right, bottom = max(0, left), max(0, top), min(wide, right), min(tall, bottom)
    if right - left < 4 or bottom - top < 4:
        return []
    part = picture[top:bottom, left:right].astype(np.int16)
    edge = np.concatenate([part[0], part[-1], part[:, 0], part[:, -1]])
    panel = np.median(edge, axis=0)
    drawn = np.abs(part - panel).max(axis=2) > APART_FROM_THE_PANEL
    for tx0, ty0, tx1, ty1 in text_boxes:
        drawn[max(0, int(ty0 * tall) - top):max(0, int(ty1 * tall) - top), max(0, int(tx0 * wide) - left):max(0, int(tx1 * wide) - left)] = False
    count, labels, stats, _centres = pieces(drawn)
    found = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < SMALLEST * wide * tall or area < FILLED * w * h or max(w, h) > 6 * max(1, min(w, h)):
            continue
        mask = labels[y:y + h, x:x + w] == label
        pixels = picture[top + y:top + y + h, left + x:left + x + w][mask]
        found.append((pixels, ((left + x) / wide, (top + y) / tall, (left + x + w) / wide, (top + y + h) / tall)))
    return found


def what_a_legend_shows(picture: Any, regions: Sequence[dict[str, Any]]) -> list[Shown]:
    """The things a screen draws beside words that say what to do about them. ``picture`` is RGB, as play sees it;
    ``regions`` the screen's writing, each with ``text`` and ``x``, ``y``, ``width``, ``height`` as shares of it."""
    from core.perception.what_moves_in_the_picture import _look_of

    picture = np.asarray(picture)
    if picture.ndim != 3 or not regions:
        return []
    text_boxes = [(float(r.get("x", 0.0)), float(r.get("y", 0.0)), float(r.get("x", 0.0)) + float(r.get("width", 0.0)),
                   float(r.get("y", 0.0)) + float(r.get("height", 0.0))) for r in regions]
    shown: list[Shown] = []
    for words, stance, (left, top, right, bottom) in _captions(regions):
        middle, half = (left + right) / 2.0, max(right - left, 0.15) * 0.65
        # Up to the nearest writing above: a heading over the panel is not a thing the caption speaks of.
        ceiling = max([top - ABOVE, 0.0] + [b for (x0, _t, r, b) in text_boxes if b <= top - 0.004 and x0 < middle + half
                                            and r > middle - half])
        above = (middle - half, ceiling, middle + half, top - 0.004)
        before = (max(0.0, left - BEFORE), top - (bottom - top), left - 0.004, bottom + (bottom - top))
        for pixels, where in _things_in(picture, above, text_boxes) or _things_in(picture, before, text_boxes):
            apart = np.abs(pixels.astype(np.int16) - np.median(pixels, axis=0)).max(axis=1)
            core = pixels[apart <= np.percentile(apart, 60)] if len(pixels) > 4 else pixels
            colour = tuple(int(c) for c in np.median(core, axis=0))
            shown.append(Shown(words, stance, _look_of(pixels), colour, where))
    return shown


def most_like(look: Any, shown: Sequence[Shown], *, colour: tuple[int, int, int] | None = None) -> Shown | None:
    """The thing a legend drew that a thing seen in play is most like, where it is alike enough to be it: its colour
    mix close, or its own colour the drawing's and its mix not far off. Only one drawing may be that close."""
    from core.perception.what_moves_in_the_picture import _look_apart

    def apart(drawn: Shown) -> float:
        mixed = _look_apart(np.asarray(look), drawn.look)
        if colour is not None and max(abs(int(a) - int(b)) for a, b in zip(colour, drawn.colour, strict=True)) <= SAME_COLOUR:
            return min(mixed, max(0.0, mixed - (LOOSER - ALIKE)))
        return mixed

    close = [drawn for drawn in shown if apart(drawn) <= ALIKE]
    if not close or len({drawn.stance for drawn in close}) > 1:
        return None
    return min(close, key=apart)
