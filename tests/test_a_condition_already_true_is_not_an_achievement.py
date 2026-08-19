"""She reported the goal reached in 1.2 seconds without making a move.

LIVE 2026-08-19. Asked to play until a 128 tile, she found the game, opened
it, read "SCORE 128" from the header, and declared the objective complete.
The number was on screen. It was not a tile, and she had not played.

Two things were wrong, and both are general:

  * a bare value has to BE something on screen rather than appear inside
    something. "128" is inside "SCORE 128" and inside "1284", and neither is
    the thing being waited for. A value that is the whole of a text region is
    a value the screen is showing as a thing;
  * a condition true before she acted is not something she did. A run that
    finishes off its first reading has found the condition already met, which
    usually means it describes something other than what is being waited for,
    and saying so beats a receipt for work nobody did.
"""
from __future__ import annotations

import pytest

from core.skills import screen_pursuit as sp
from core.skills.screen_pursuit import goal_reached

BOARD_WITH_SCORE = {
    "ok": True,
    "text": "2048 SCORE 128 BEST 6068 2 4 8",
    "layout": [
        {"text": "SCORE 128", "center_y": 0.15},
        {"text": "2", "center_y": 0.4},
        {"text": "4", "center_y": 0.5},
    ],
}
BOARD_WITH_TILE = {
    "ok": True,
    "text": "2048 SCORE 1400 128 64",
    "layout": [
        {"text": "SCORE 1400", "center_y": 0.15},
        {"text": "128", "center_y": 0.5},
        {"text": "64", "center_y": 0.6},
    ],
}


def test_a_number_inside_a_label_is_not_the_thing_being_waited_for():
    assert not goal_reached(BOARD_WITH_SCORE, "128", region_top=0.12, region_bottom=1.0)


def test_a_number_that_is_a_thing_on_screen_counts():
    assert goal_reached(BOARD_WITH_TILE, "128", region_top=0.12, region_bottom=1.0)


def test_a_longer_number_containing_it_does_not_count():
    board = {"ok": True, "text": "1284", "layout": [{"text": "1284", "center_y": 0.5}]}
    assert not goal_reached(board, "128", region_top=0.12, region_bottom=1.0)


def test_a_worded_condition_still_matches_inside_a_sentence():
    """Only bare values get the stricter test."""
    board = {"ok": True, "text": "Build passed", "layout": [{"text": "Build passed", "center_y": 0.5}]}
    assert goal_reached(board, "passed", region_top=0.12, region_bottom=1.0)


def test_a_bare_value_is_checked_by_region_even_with_no_band():
    assert not goal_reached(BOARD_WITH_SCORE, "128")
    assert goal_reached(BOARD_WITH_TILE, "128")


@pytest.mark.asyncio
async def test_a_goal_already_met_before_she_moved_is_said_not_claimed(monkeypatch):
    async def read(app_name=""):
        return BOARD_WITH_TILE

    async def press(key, *, expect_app=""):
        raise AssertionError("she should not have needed to move")

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://play2048.co/", "title": "2048", "error": ""}

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)

    result = await sp.pursue_on_screen(
        goal="play until a 128 tile",
        success_when="128",
        region_top=0.12,
        max_cycles=2,
        max_seconds=5.0,
        narrate=False,
        lived=False,
    )
    assert result["outcome"] == "already_true"
    assert result["completed"] is False
    assert result["already_true_at_the_start"] == "128"
