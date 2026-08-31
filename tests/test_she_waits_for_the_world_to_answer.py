"""Waiting for the move to land before judging whether it landed.

There was no wait at all after a keystroke. The key returned, the loop came
round, and the next reading was of a board mid-slide — or of one the game had
not begun moving yet. Compared with the reading before it, that says nothing
happened.

Of what she had kept about playing this game on 2026-08-30, ninety-eight of a
hundred acts were written down as having changed nothing, and the only rule
she ever confirmed was "this does not move", ninety-eight times.
"""

from __future__ import annotations

import asyncio

import pytest

from core.skills import screen_pursuit


def _reading(text: str) -> dict:
    return {"ok": True, "text": text, "layout": [], "scoped_to": "a thing"}


@pytest.mark.asyncio
async def test_it_waits_for_the_change_and_then_for_the_stillness(monkeypatch) -> None:
    """A board mid-slide has half moved, and half a move is not a state any
    rule describes."""
    frames = iter(
        [
            _reading("before"),  # the key has landed, nothing has moved yet
            _reading("before"),
            _reading("mid slide"),  # moving
            _reading("after"),  # arrived
            _reading("after"),  # and still
            _reading("after"),
        ]
    )

    async def read(app, over=None):
        return next(frames, _reading("after"))

    monkeypatch.setattr(screen_pursuit, "read_screen", read)
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.0
    seen, moved = await screen_pursuit._settled_after(_reading("before"), "a thing")
    assert moved
    assert seen["text"] == "after", "it should not stop on a board mid-slide"


@pytest.mark.asyncio
async def test_a_move_that_really_changes_nothing_says_so(monkeypatch) -> None:
    async def read(app, over=None):
        return _reading("before")

    monkeypatch.setattr(screen_pursuit, "read_screen", read)
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.4
    seen, moved = await screen_pursuit._settled_after(_reading("before"), "a thing")
    assert not moved
    assert seen["text"] == "before"


def test_how_long_to_wait_comes_from_how_long_it_has_taken() -> None:
    """Nothing is chosen. Before she has seen a change there is no
    measurement, so the old default stands; after that it is a little more
    than the longest she has seen."""
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.0
    assert screen_pursuit._how_long_to_wait() == 4.0
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.9
    assert screen_pursuit._how_long_to_wait() == pytest.approx(1.8)
    screen_pursuit._ANSWERING_TOOK["longest"] = 0.1
    assert screen_pursuit._how_long_to_wait() == 1.0, "never less than a second"


def test_the_move_path_waits_at_all() -> None:
    """The mechanism existed and was wired only to the restart path."""
    source = screen_pursuit.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    act = text[text.index("        async def act() -> bool:") :]
    act = act[: act.index("        return Step(")]
    assert "_settled_after(" in act, "a keystroke must be given time to land"
