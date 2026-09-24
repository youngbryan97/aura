"""A look that waited on recognition is not the world being late.

LIVE 2026-09-24: her moves came a minute and more apart on a board that
answers in under a second. Her eyes give a place that will not read a whole
look to come good and then report the look as not settled. The settle loop
took that as the world still moving, looked again, and the probe of her pause
counted both pauses and the look between them as what the world had needed.
Each such move doubled the pause before the next look, and a clean first look
only halved it; by the time anyone checked she paused 51 seconds before
looking at all. The wait for a change was twice the slowest answer ever seen,
so a press that changed nothing then cost the better part of two minutes.
"""
from __future__ import annotations

import math

import pytest
from screen_pursuit_support import patch_pursuit

from core.perception.what_the_pixels_show import still_but_unread
from core.skills import screen_pursuit_looking as looking


@pytest.fixture(autouse=True)
def _fresh_pacing():
    kept = (dict(looking._WAIT), dict(looking._STILL_FLOOR), list(looking._ANSWERS),
            dict(looking._ANSWERING_TOOK))
    looking._WAIT["seconds"] = 0.02
    looking._STILL_FLOOR["seconds"] = math.inf
    looking._ANSWERS.clear()
    looking._ANSWERING_TOOK.update({"longest": 0.0, "quickest": 0.0})
    yield
    looking._WAIT.update(kept[0])
    looking._STILL_FLOOR.update(kept[1])
    looking._ANSWERS[:] = kept[2]
    looking._ANSWERING_TOOK.update(kept[3])


def _board(text: str, **extra) -> dict:
    return {"ok": True, "text": text, "layout": [], "scoped_to": "a game", **extra}


def test_the_eyes_tell_an_unread_place_from_a_moving_world():
    assert still_but_unread(False, True, [(0, 1)])
    assert not still_but_unread(True, True, [(0, 1)]), "a settled look needs no excuse"
    assert not still_but_unread(False, False, [(0, 1)]), "something was still moving"
    assert not still_but_unread(False, True, []), "nothing failed to read"


def test_pictures_spent_on_recognition_do_not_lengthen_the_wait():
    looking._WAIT["seconds"] = 1.0
    looking._it_was_ready(
        1.0, True,
        {"seconds_to_still": 1.6, "pictures_to_still": 6, "still_but_unread": True},
    )
    assert looking._WAIT["seconds"] == 0.5


@pytest.mark.asyncio
async def test_a_changed_board_at_rest_but_for_one_tile_is_the_answer(monkeypatch):
    looks: list[int] = []

    async def read(app, over=None):
        looks.append(1)
        return _board("after", settled=False, still_but_unread=True,
                      seconds_to_still=1.6, pictures_to_still=6)

    patch_pursuit(monkeypatch, "read_screen", read)
    seen, moved = await looking._settled_after(_board("before", settled=True), "a game")
    assert moved and seen["text"] == "after"
    assert len(looks) == 1, "a second look would carry the same place as unsure"
    assert looking._WAIT["seconds"] == pytest.approx(0.01, abs=0.01)


@pytest.mark.asyncio
async def test_a_look_that_timed_out_does_not_make_the_world_late(monkeypatch):
    replies = iter(["timeout", "after"])

    async def read(app, over=None):
        if next(replies) == "timeout":
            raise TimeoutError
        return _board("after", settled=True, seconds_to_still=0.2, pictures_to_still=2)

    patch_pursuit(monkeypatch, "read_screen", read)
    looking._WAIT["seconds"] = 0.04
    seen, moved = await looking._settled_after(_board("before", settled=True), "a game")
    assert moved
    # Halved from the first pause, not set to both pauses and the timeout.
    assert looking._WAIT["seconds"] == pytest.approx(0.02, abs=0.01)


@pytest.mark.asyncio
async def test_many_moves_with_an_unreadable_tile_do_not_grow_the_pause(monkeypatch):
    """The regression, as it ran live: the same unreadable tile, move after move."""

    async def read(app, over=None):
        return _board(f"after {len(boards)}", settled=False, still_but_unread=True,
                      seconds_to_still=1.6, pictures_to_still=6)

    boards: list[int] = []
    patch_pursuit(monkeypatch, "read_screen", read)
    looking._WAIT["seconds"] = 0.05
    for _ in range(12):
        boards.append(1)
        await looking._settled_after(_board("before", settled=True), "a game")
    assert looking._WAIT["seconds"] <= 0.05


def test_one_slow_answer_ages_out_of_how_long_she_waits():
    looking._answering_took(45.0)
    assert looking._how_long_to_wait() == pytest.approx(90.0)
    for _ in range(looking._A_FEW_ANSWERS):
        looking._answering_took(0.3)
    assert looking._how_long_to_wait() == 1.0


def test_the_last_few_answers_are_read_by_something():
    """They were written after every answer and read by nothing."""
    looking._answering_took(0.9)
    looking._answering_took(0.4)
    assert looking._how_long_to_wait() == pytest.approx(1.8)
