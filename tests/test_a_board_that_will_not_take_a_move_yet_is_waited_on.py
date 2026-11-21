"""A board of pieces that would not take a move yet is waited on, never tried with the arrow keys.

LIVE 20 November 2026 En Croissant's setup page showed its board at the start beside "Start game". She tried a move,
the board took none, and she left it for eight seconds. She pressed "Start game" by what it means, and in those eight
seconds she pressed the up arrow at the game twice. The board was looked for again only once the wait was over.
"""
from __future__ import annotations

import time

import pytest
from screen_pursuit_support import patch_pursuit

from core.skills import screen_pursuit as sp
from core.skills import screen_pursuit_a_way_on as way_on
from core.skills import screen_pursuit_board_of_pieces as board_of_pieces


class _Store:
    def __init__(self):
        self.episodes = []

    def record(self, episode):
        self.episodes.append(episode)
        return f"ep_{len(self.episodes)}"

    def resolve(self, episode_id, outcome):
        self.episodes.append((episode_id, outcome))

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        self.episodes.append((action, outcome, success))


@pytest.fixture
def board(monkeypatch):
    """A window with a board on it that takes no move, and a control that starts the game on the third look."""
    state = {"pressed": [], "looked_for": 0, "looks": 0, "start_game_at": 3}

    async def read(app_name="", over=None):
        return {"ok": True, "text": "board 2\n4", "layout": [{"text": "2", "center_x": 0.3, "center_y": 0.4},
                                                              {"text": "4", "center_x": 0.5, "center_y": 0.4}],
                "bounds": [], "scoped_to": "TheBoard",
                "in_front_then": "TheBoard", "her_window_showing": True}

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        return True

    async def yes(*_a, **_k):
        return True

    async def in_front(*_a, **_k):
        return "TheBoard"

    async def identity():
        return {"url": "", "title": "", "error": ""}

    async def a_board(target_app, ends_at, keep, seen=None):
        # As the real one: nothing while the board is left, and a board that takes no move is left again.
        if time.monotonic() < float(keep.get("board_left_until", 0.0)):
            return None
        state["looked_for"] += 1
        keep["board_left_until"] = time.monotonic() + 60.0
        return None

    async def a_way_on(target_app, goal, keep, *, nothing_answered, seen=None):
        state["looks"] += 1
        return state["looks"] == state["start_game_at"]

    from core.security import screen_capture_policy as policy

    async def may_look(*_a, **_k):
        return policy.ScreenCaptureAdmission(allowed=True)

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", may_look)
    monkeypatch.setattr(policy, "evaluate_window_capture_admission_async", may_look)
    monkeypatch.setattr(board_of_pieces, "played_a_board_of_pieces", a_board)
    monkeypatch.setattr(way_on, "took_a_way_on", a_way_on)
    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "press", press)
    patch_pursuit(monkeypatch, "_ensure_frontmost", yes)
    patch_pursuit(monkeypatch, "current_page_identity", identity)
    patch_pursuit(monkeypatch, "_frontmost", in_front)
    patch_pursuit(monkeypatch, "_bring_the_thing_back_to_the_front", yes, raising=False)
    patch_pursuit(monkeypatch, "_how_long_to_wait", lambda: 0.05, raising=False)
    return state


async def _play(cycles: int) -> dict:
    async def think(objective, evidence):
        return ""

    return await sp.pursue_on_screen(
        goal="play the game on TheBoard",
        target_app="TheBoard",
        success_when="never happens",
        think=think,
        max_cycles=cycles,
        max_seconds=20.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )


@pytest.mark.asyncio
async def test_no_key_is_pressed_at_a_board_left_for_a_moment(board):
    await _play(8)
    assert board["looked_for"] >= 1, "the board was never looked for"
    assert board["pressed"] == [], f"keys pressed at a board waiting to be played: {board['pressed']}"


@pytest.mark.asyncio
async def test_a_board_is_looked_for_at_once_after_a_way_on_is_taken(board):
    await _play(8)
    assert board["looks"] >= board["start_game_at"], "the way on was never taken"
    assert board["looked_for"] == 2, "the board was not looked for again once the way on changed the screen"
