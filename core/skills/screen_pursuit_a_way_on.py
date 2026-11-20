"""A way on through an application, taken by what its labels mean: the card or button nearest what she was asked to do.

Asked to play a game in an application, a person who opens it and finds a page of cards ("Play Chess: play against an
engine or a friend", "Analysis Board", "Puzzles") does not press the arrow keys at it; they read the cards and press the
one that means what they came to do. LIVE 20 November 2026 she opened En Croissant on its start page, took the grid of
five cards for a board, pressed arrows at it for a minute, and gave up: "Nothing I do is changing anything here".

So where the window holds words and nothing she has done there has changed it yet, the words are gathered into the
controls they belong to (a card's heading, its line under it, its button: words stacked close in one column), each
control is read by her encoder against the first act she was asked that is still to do (acts are said in the order they
are done, and "open it" is done once she is in it), and the nearest control is pressed where it is further ahead of the
next than the next is of the one after. Within it the words nearest the act are pressed first; where the screen does not change, the next. A control
that names the application she is in is where she already is, and is left. A control is taken once. Nothing here knows
an application or a word on a button.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.ScreenPursuit.AWayOn")

__all__ = ["took_a_way_on"]

#: How often the window is read for a way on, in seconds.
LOOKED_EVERY_S = 3.0
#: How long a press is given to change the screen.
SHOWS_IN_S = 1.0
#: Words this close, as shares of the window, belong to one control: across, the width of a column of text; down, a
#: line and a half of a card.
SAME_COLUMN = 0.06
SAME_CARD = 0.09


def _regions(reading: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for region in reading.get("layout") or []:
        text = " ".join(str((region or {}).get("text") or "").split())
        if not any(ch.isalpha() for ch in text):
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        found.append({"text": text, "x": x, "y": y})
    return found


def _controls(regions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Words stacked close in one column, top to bottom, as one control."""
    joined = list(range(len(regions)))

    def root(i: int) -> int:
        while joined[i] != i:
            joined[i] = joined[joined[i]]
            i = joined[i]
        return i

    for i, one in enumerate(regions):
        for j in range(i + 1, len(regions)):
            other = regions[j]
            if abs(one["x"] - other["x"]) <= SAME_COLUMN and abs(one["y"] - other["y"]) <= SAME_CARD:
                joined[root(i)] = root(j)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, region in enumerate(regions):
        groups.setdefault(root(i), []).append(region)
    return [sorted(group, key=lambda r: r["y"]) for group in groups.values()]


def _stands_out(scores: list[float]) -> int | None:
    """The one that stands out: further ahead of the next than the next is of the one after it."""
    if len(scores) < 2:
        return 0 if scores else None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    lead = scores[order[0]] - scores[order[1]]
    behind = scores[order[1]] - scores[order[2]] if len(order) > 2 else 0.0
    return order[0] if lead > behind else None


def _names(text: str, app: str) -> bool:
    """Whether words name the application she is in: where she is already is no way further into it."""
    import re

    letters = re.sub(r"[^a-z0-9]+", "", str(app or "").lower())
    return bool(letters) and letters in re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _the_words_differ(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    one = {r["text"] for r in before}
    other = {r["text"] for r in after}
    if not one and not other:
        return False
    return len(one ^ other) > 0.3 * max(len(one), len(other))


#: The longest a reading is waited on before it is taken to show nothing.
READ_AT_MOST_S = 5.0


async def _a_reading(read_screen: Any, target_app: str) -> dict[str, Any] | None:
    """A reading of the window, or None where none comes back in time: a look that does not come back is not a look."""
    import asyncio

    try:
        return await asyncio.wait_for(read_screen(target_app), timeout=READ_AT_MOST_S)
    except (TimeoutError, OSError, RuntimeError) as why:
        logger.info("the window was not read for a way on (%s)", type(why).__name__)
        return None


async def took_a_way_on(target_app: str, goal: str, keep: dict[str, Any], *, nothing_answered: bool,
                        seen: dict[str, Any] | None = None) -> bool:
    """Press the control in ``target_app``'s window whose words mean most nearly ``goal``, where nothing she has done
    there has changed the screen yet and one stands out; whether the screen changed for it."""
    import asyncio

    from core.language.learned_matcher import embed_sentences

    from .screen_pursuit import read_screen
    from .screen_pursuit_on_a_page import on_her_page
    from .screen_pursuit_surface import click_normalized

    now = time.monotonic()
    if not target_app or not goal or not nothing_answered or on_her_page() is not None:
        return False
    if now - float(keep.get("way_on_looked_at", -1e9)) < LOOKED_EVERY_S:
        return False
    keep["way_on_looked_at"] = now
    # The reading just taken, where there is one: a second look is a second chance for a capture to wedge.
    reading = seen if seen is not None else await _a_reading(read_screen, target_app)
    if not reading or reading.get("ok") is False:
        return False
    regions = _regions(reading)
    if len(regions) < 3:
        return False
    taken: list[str] = keep.setdefault("ways_on_taken", [])
    controls = [c for c in _controls(regions)
                if " / ".join(r["text"] for r in c) not in taken
                and not _names(" ".join(r["text"] for r in c), target_app)]
    if len(controls) < 2:
        return False
    texts = [" ".join(r["text"] for r in control) for control in controls]
    # Each act she was asked, apart: "open it and play a game against X" is two, and the first is done once she is in.
    from core.language.what_she_is_asked_to_do import the_parts_that_ask_her

    acts = [act for act in (the_parts_that_ask_her(goal) or [goal]) if not _names(act, target_app)] or [goal]
    vectors = await asyncio.to_thread(embed_sentences, [*acts, *texts])
    if len(vectors) != len(acts) + len(texts):
        return False
    matrix = np.asarray(vectors, dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    by_act = matrix[len(acts):] @ matrix[:len(acts)].T
    # The act to find a way on for is the first one still to do: a person says what to do in the order it is done,
    # and "open it" is done once she is in it. "Tell me each move" and "close it when you're done" come after.
    act = 0
    scores = [float(v) for v in by_act[:, act]]
    chosen = _stands_out(scores)
    if chosen is None:
        logger.info("no way on stands out in %r for %r", target_app, acts[act][:60])
        return False
    control = controls[chosen]
    taken.append(" / ".join(r["text"] for r in control))
    bounds = list(reading.get("bounds") or [])
    logger.info("a way on in %r by what it means: %r for %r (%.2f, next %.2f)", target_app, texts[chosen][:80],
                acts[act][:60], scores[chosen], sorted(scores)[-2])
    from .screen_pursuit import _tell

    # Within it, the words nearest the act first: on a card its button or its heading, in a column of settings the
    # one that starts what was asked rather than one that changes how.
    own = await asyncio.to_thread(embed_sentences, [acts[act], *(r["text"] for r in control)])
    order = list(control)
    if len(own) == len(control) + 1:
        near = np.asarray(own, dtype=np.float64)
        near /= np.linalg.norm(near, axis=1, keepdims=True) + 1e-12
        closeness = near[1:] @ near[0]
        order = [control[i] for i in np.argsort(-closeness)]
    for region in order:
        await click_normalized(region["x"], region["y"], expect_app=target_app, bounds=bounds)
        await asyncio.sleep(SHOWS_IN_S)
        after = _regions(await _a_reading(read_screen, target_app) or {})
        if _the_words_differ(regions, after):
            _tell(f"I pressed \"{region['text']}\": it's the nearest thing here to what you asked.")
            return True
    return False
