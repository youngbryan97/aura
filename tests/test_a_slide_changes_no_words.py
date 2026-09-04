"""Waiting for a surface to settle has to watch where things are.

A thing sliding across a surface changes no text at all: the same numbers are
there throughout, in different places. Compared by text, a board mid-slide
reads identical to the board before the move — so the wait finished on the
first look, and whatever was still travelling was read at wherever it had got
to.

LIVE 2026-09-04 on a correctly read four by four board: the row nearest the
direction pressed came back unmoved, move after move, while every other row
landed exactly where the rule said. The true rule sat at 59% of 29 — under the
bar to be trusted, so nothing ever looked ahead, all game.
"""

from __future__ import annotations

import asyncio

import core.skills.screen_pursuit as sp


def _screen(*places: tuple[float, float, str]) -> dict:
    return {
        "ok": True,
        "text": " ".join(said for _x, _y, said in places),
        "layout": [
            {"text": said, "center_x": x, "center_y": y} for x, y, said in places
        ],
        "scoped_to": "Thing",
        "in_front_then": "Thing",
        "her_window_showing": True,
    }


MOVING = _screen((0.3, 0.4, "2"), (0.5, 0.4, "4"))
SLID = _screen((0.4, 0.4, "2"), (0.6, 0.4, "4"))
DONE = _screen((0.5, 0.4, "2"), (0.7, 0.4, "4"))


def test_it_waits_for_a_slide_that_changes_no_words(monkeypatch):
    frames = [SLID, DONE, DONE]

    async def read(app_name="", over=None):
        return frames.pop(0) if len(frames) > 1 else frames[0]

    monkeypatch.setattr(sp, "read_screen", read)
    seen, moved = asyncio.run(sp._settled_after(MOVING, "Thing", patience=5.0))
    assert moved is True
    assert seen is DONE, "it settled on the frame where nothing was travelling"


def test_a_surface_that_never_moves_says_so(monkeypatch):
    async def read(app_name="", over=None):
        return MOVING

    monkeypatch.setattr(sp, "read_screen", read)
    _seen, moved = asyncio.run(sp._settled_after(MOVING, "Thing", patience=1.0))
    assert moved is False


def test_the_words_alone_would_have_said_it_had_finished():
    """Which is the whole of the defect."""
    assert MOVING["text"] == SLID["text"] == DONE["text"]
