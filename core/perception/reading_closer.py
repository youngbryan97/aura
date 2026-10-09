"""Looking closer at writing that did not read as words: its place read again by itself, larger.

Lettering drawn for a game (outlined, shadowed, set over a busy picture) comes
back from a reading of the whole screen with its letters half right. LIVE
2026-10-09 a title's START was read "sitarit", and she never saw the way on
that a person reads at a glance; read again from its own place, cut out and
enlarged, it was "Start". A person who cannot make out a word looks closer at
it, and so does she: a short piece of writing that is not words of the language
is read again by itself, and where that reading is words, it is what the place
says.

Only short pieces (a label, not a paragraph) and only a few a look, since each
is a reading of its own; and what a piece came to is remembered by how it
looked, so a screen that stays up is not read closer again on every look.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["read_closer"]

#: The most pieces read again in one look.
CLOSER_AT_MOST = 4
#: The longest piece read again, in letters: a label, not a line of prose.
LABEL_LONGEST = 24
#: How much larger a piece is made before it is read again, and the margin cut around it, in its own heights.
ENLARGED = 3
MARGIN = 0.5
#: Pieces whose closer reading is remembered.
REMEMBERED = 256

_remembered: OrderedDict[tuple[str, bytes], str] = OrderedDict()


def _words(text: str) -> bool:
    """Whether every run of letters in ``text`` long enough to be a word is a word of the language."""
    from core.language.words_of_the_language import the_word

    runs = [run for run in re.findall(r"[A-Za-z]+", text) if len(run) >= 2]
    return bool(runs) and all(the_word(run) is not None for run in runs)


def _cut(pixels: np.ndarray, region: dict[str, Any]) -> np.ndarray | None:
    """The piece's place with a margin around it, as pixels; None where it has no place."""
    tall, wide = pixels.shape[:2]
    try:
        x, y = float(region["x"]), float(region["y"])
        w, h = float(region["width"]), float(region["height"])
    except (KeyError, TypeError, ValueError):
        return None
    pad = MARGIN * h
    left, top = max(0, int((x - pad) * wide)), max(0, int((y - pad) * tall))
    right, bottom = min(wide, int((x + w + pad) * wide) + 1), min(tall, int((y + h + pad) * tall) + 1)
    if right - left < 4 or bottom - top < 4:
        return None
    return pixels[top:bottom, left:right, :3]


def _enlarged(patch: np.ndarray) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(patch).astype(np.uint8))
    return np.asarray(image.resize((image.width * ENLARGED, image.height * ENLARGED), Image.LANCZOS))


def read_closer(picture: Any, layout: Sequence[dict[str, Any]],
                recognize: Callable[[Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """``layout`` with each short piece that is not words read again from its own place, where that reads as words."""
    pixels = np.asarray(picture)
    out = [dict(region) for region in layout]
    if pixels.ndim != 3 or pixels.size == 0:
        return out
    tried = 0
    for region in out:
        text = " ".join(str(region.get("text") or "").split())
        if not text or len(text) > LABEL_LONGEST or not re.search(r"[A-Za-z]{3,}", text) or _words(text):
            continue
        patch = _cut(pixels, region)
        if patch is None:
            continue
        seen = (text, np.ascontiguousarray(patch[::4, ::4]).tobytes())
        if seen not in _remembered:
            if tried >= CLOSER_AT_MOST:
                continue
            tried += 1
            again = " ".join(" ".join(str(piece.get("text") or "") for piece in recognize(_enlarged(patch))).split())
            _remembered[seen] = again if again and _words(again) else ""
            while len(_remembered) > REMEMBERED:
                _remembered.popitem(last=False)
        if _remembered[seen]:
            region["text"], region["read_closer_from"] = _remembered[seen], text
    return out
