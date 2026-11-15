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

__all__ = ["read_closer", "ways_on_at_the_edges", "written_on"]

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
    # not a failure: a region that is not four numbers is none
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


def written_on(picture: Any, region: dict[str, Any], recognize: Callable[[Any], list[dict[str, Any]]]) -> str:
    """The words written on a shape, read from the shape alone, enlarged; "" where none are words. LIVE-like
    2026-10-11 a title's PLAY, small red letters in a red frame, was not read from the whole screen, and the shape it
    stood in was offered as "the shape at 85% across, 90% down" after five others, none of them it."""
    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.size == 0:
        return ""
    tall, wide = pixels.shape[:2]
    try:
        left, top = max(0, int(float(region["x"]) * wide)), max(0, int(float(region["y"]) * tall))
        right = min(wide, int((float(region["x"]) + float(region["width"])) * wide) + 1)
        bottom = min(tall, int((float(region["y"]) + float(region["height"])) * tall) + 1)
    except (KeyError, TypeError, ValueError):
        return ""
    if right - left < 8 or bottom - top < 6:
        return ""
    patch = pixels[top:bottom, left:right, :3]
    seen = ("", np.ascontiguousarray(patch[::4, ::4]).tobytes())
    if seen not in _remembered:
        pieces = [" ".join(str(piece.get("text") or "").split()) for piece in recognize(_enlarged(patch))]
        _remembered[seen] = " ".join(piece for piece in pieces if len(re.sub(r"[^A-Za-z]", "", piece)) >= 2 and _words(piece))
        while len(_remembered) > REMEMBERED:
            _remembered.popitem(last=False)
    return _remembered[seen]


#: The strips along the bottom and the top of a picture, as shares of its height, where a way on is usually drawn; and
#: the fewest words a screen reads for it to be a page of prose that a way on leads from.
EDGE = 0.16
PROSE_AT_LEAST = 15


def ways_on_at_the_edges(picture: Any, layout: Sequence[dict[str, Any]],
                         recognize: Callable[[Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """``layout`` with the ways on read along the picture's bottom and top edges, read again by themselves and larger,
    where a page of prose was read and no way on with it. A person who has read a page of instructions and sees no way
    on looks along the bottom for one: offline 2026-10-12 an instructions screen's CONTINUE and SKIP INSTRUCTIONS, in
    a bar along its bottom, were not read with its paragraphs, an arrow drawn beside them was taken for a key, and the
    right arrow was pressed twenty-two times over four minutes."""
    from core.language.a_way_on import how_much_it_leads_on

    out = [dict(region) for region in layout]
    texts = [" ".join(str(region.get("text") or "").split()) for region in out]
    if any(0 < len(text) <= 40 and how_much_it_leads_on(text) > 1.0 for text in texts):
        return out
    if sum(len(text.split()) for text in texts) < PROSE_AT_LEAST:
        return out
    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.size == 0:
        return out
    tall = pixels.shape[0]
    for top, bottom in ((1.0 - EDGE, 1.0), (0.0, EDGE)):
        strip = np.ascontiguousarray(pixels[int(top * tall):int(bottom * tall), :, :3])
        if strip.shape[0] < 6:
            continue
        seen = ("the edge", strip[::4, ::4].tobytes())
        if seen not in _remembered:
            pieces = recognize(_enlarged(strip))
            _remembered[seen] = repr([(str(p.get("text") or ""), float(p.get("x", 0)), float(p.get("y", 0)),
                                       float(p.get("width", 0)), float(p.get("height", 0))) for p in pieces])
            while len(_remembered) > REMEMBERED:
                _remembered.popitem(last=False)
        import ast

        for text, x, y, width, height in ast.literal_eval(_remembered[seen]):
            said = " ".join(text.split())
            if 0 < len(said) <= 40 and how_much_it_leads_on(said) > 1.0:
                y_share, h_share = top + y * (bottom - top), height * (bottom - top)
                out.append({"text": said, "x": x, "y": y_share, "width": width, "height": h_share,
                            "center_x": x + width / 2, "center_y": y_share + h_share / 2, "read_at_the_edge": True})
    return out
