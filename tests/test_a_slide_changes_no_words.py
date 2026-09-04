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


# ── and recognising what she foretold is knowing it has landed ───────────


def test_the_arrival_of_what_she_foretold_ends_the_wait(monkeypatch):
    """Watching until it stops changing is what she does when she cannot say
    what the change will be. When she can, the arrival of exactly that is the
    same fact at one reading instead of two — and a reading is most of what a
    move costs."""
    reads = {"n": 0}

    async def read(app_name="", over=None):
        reads["n"] += 1
        return SLID

    monkeypatch.setattr(sp, "read_screen", read)
    seen, moved = asyncio.run(
        sp._settled_after(MOVING, "Thing", patience=5.0, arrived=lambda now: now is SLID)
    )
    assert moved is True
    assert seen is SLID
    assert reads["n"] == 1, "one look, because she recognised it"


def test_a_test_that_says_no_still_waits_for_stillness(monkeypatch):
    frames = [SLID, DONE, DONE]

    async def read(app_name="", over=None):
        return frames.pop(0) if len(frames) > 1 else frames[0]

    monkeypatch.setattr(sp, "read_screen", read)
    seen, moved = asyncio.run(
        sp._settled_after(MOVING, "Thing", patience=5.0, arrived=lambda now: False)
    )
    assert moved is True
    assert seen is DONE


def test_she_foretold_nothing_when_she_holds_no_grid():
    assert sp._looks_like(object(), None, None) is None
    assert sp._looks_like(None, None, None) is None


def test_the_pursuit_hands_the_wait_what_it_foretold():
    import inspect

    source = inspect.getsource(sp.pursue_on_screen)
    at = source.index("came_to_rest, _ = await _settled_after(")
    assert "arrived=_looks_like(" in source[at : at + 400]
    assert 'expected["after"]' in source[at : at + 400]
