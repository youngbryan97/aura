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


SCORED = [
    {"text": "SCORE", "x": 0.40, "width": 0.06, "center_y": 0.15, "height": 0.02},
    {"text": "128", "x": 0.48, "width": 0.04, "center_y": 0.15, "height": 0.02},
    {"text": "BEST", "x": 0.60, "width": 0.05, "center_y": 0.15, "height": 0.02},
    {"text": "6068", "x": 0.66, "width": 0.05, "center_y": 0.15, "height": 0.02},
    {"text": "2", "x": 0.30, "width": 0.03, "center_y": 0.50, "height": 0.05},
]
TILED = SCORED + [{"text": "128", "x": 0.55, "width": 0.05, "center_y": 0.55, "height": 0.05}]


def test_a_number_beside_a_word_is_that_words_number():
    """"SCORE" and "128" are separate regions, so wholeness cannot tell them
    apart — but a tile has nothing next to it saying what it counts."""
    from core.skills.screen_pursuit import labelled_by

    assert labelled_by(SCORED[1], SCORED) == "SCORE"
    assert labelled_by(SCORED[3], SCORED) == "BEST"
    assert labelled_by(TILED[-1], TILED) == ""


def test_a_labelled_total_does_not_satisfy_a_goal_about_a_thing():
    assert not goal_reached({"ok": True, "text": "x", "layout": SCORED}, "128", region_top=0.12, region_bottom=1.0)


def test_the_thing_itself_does():
    assert goal_reached({"ok": True, "text": "x", "layout": TILED}, "128", region_top=0.12, region_bottom=1.0)


def test_a_label_far_away_is_not_a_label():
    """A label sits against the value it names."""
    from core.skills.screen_pursuit import labelled_by

    distant = [
        {"text": "SCORE", "x": 0.05, "width": 0.06, "center_y": 0.15, "height": 0.02},
        {"text": "128", "x": 0.80, "width": 0.04, "center_y": 0.15, "height": 0.02},
    ]
    assert labelled_by(distant[1], distant) == ""


def test_a_label_directly_above_counts_too():
    from core.skills.screen_pursuit import labelled_by

    stacked = [
        {"text": "Total", "x": 0.50, "width": 0.06, "center_y": 0.40, "height": 0.02},
        {"text": "99", "x": 0.50, "width": 0.04, "center_y": 0.46, "height": 0.02},
    ]
    assert labelled_by(stacked[1], stacked) == "Total"


@pytest.mark.asyncio
async def test_a_goal_met_by_an_old_game_is_a_choice_to_begin_again(monkeypatch):
    """Someone else's finished board satisfies "play until a 128 tile" without
    her having played. Stopping there hands back a result she did not produce.
    """
    state = {"clicked": [], "pressed": []}
    board = dict(BOARD_WITH_TILE)
    board["layout"] = BOARD_WITH_TILE["layout"] + [
        {"text": "New Game", "x": 0.70, "width": 0.10, "center_x": 0.75, "center_y": 0.18, "height": 0.02}
    ]

    async def read(app_name=""):
        return board

    async def click(x, y, *, expect_app="", bounds=None):
        state["clicked"].append((x, y))
        return True

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        return True

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://play2048.co/", "title": "2048", "error": ""}

    async def think(objective, evidence):
        return "start over"

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "click_normalized", click)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)

    result = await sp.pursue_on_screen(
        goal="play until a 128 tile",
        success_when="128",
        region_top=0.12,
        think=think,
        max_cycles=1,
        max_seconds=5.0,
        narrate=False,
        lived=False,
    )
    assert state["clicked"], "she accepted a board she had not played"
    assert result["restarts"] >= 1
    assert result.get("outcome") != "already_true"


@pytest.mark.asyncio
async def test_accepting_a_pre_met_goal_is_still_reported_honestly(monkeypatch):
    board = dict(BOARD_WITH_TILE)
    board["layout"] = BOARD_WITH_TILE["layout"] + [
        {"text": "New Game", "x": 0.70, "width": 0.10, "center_x": 0.75, "center_y": 0.18, "height": 0.02}
    ]

    async def read(app_name=""):
        return board

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://play2048.co/", "title": "2048", "error": ""}

    async def think(objective, evidence):
        return "see it through"

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)

    result = await sp.pursue_on_screen(
        goal="play until a 128 tile",
        success_when="128",
        region_top=0.12,
        think=think,
        max_cycles=1,
        max_seconds=5.0,
        narrate=False,
        lived=False,
    )
    assert result["outcome"] == "already_true"
    assert result["completed"] is False
