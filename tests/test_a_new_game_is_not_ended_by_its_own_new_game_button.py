"""A new game is not ended by the button that started it.

Whether a way to start again has appeared was measured against the run's first
reading. LIVE 2026-09-24 the run began on the last board of a finished game;
after she began again, the fresh game's own New Game was a control that had
"appeared", every cycle ended the game as it began, and she pressed New Game on
an empty board for minutes.
"""
from __future__ import annotations

import pytest
from screen_pursuit_support import patch_pursuit

from core.skills import screen_pursuit as sp


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
def finished_then_fresh(monkeypatch):
    state = {"finished": True, "pressed": [], "clicked": 0, "turn": 0}

    async def read(app_name="", over=None):
        if state["finished"]:
            text, control = "Game over 128 64 32", "Try again"
        else:
            text, control = f"board {state['turn']}", "New Game"
        return {
            "ok": True,
            "text": f"{text}\n{control}",
            "layout": [
                {"text": control, "center_x": 0.8, "center_y": 0.2},
                {"text": text, "center_x": 0.5, "center_y": 0.6},
            ],
            "bounds": [],
            "scoped_to": app_name or "TheBoard",
            "in_front_then": "TheBoard",
            "her_window_showing": True,
        }

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        if not state["finished"]:
            state["turn"] += 1
        return True

    async def click(x, y, **_k):
        state["clicked"] += 1
        state["finished"] = False
        state["turn"] = 0
        return True

    async def yes(*_a, **_k):
        return True

    async def in_front(*_a, **_k):
        return "TheBoard"

    async def nothing_on_top(*_a, **_k):
        return ""

    async def identity():
        return {"url": "", "title": "", "error": ""}

    from core.security import screen_capture_policy as policy

    async def may_look(*_a, **_k):
        return policy.ScreenCaptureAdmission(allowed=True)

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", may_look)
    monkeypatch.setattr(policy, "evaluate_window_capture_admission_async", may_look)
    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "press", press)
    patch_pursuit(monkeypatch, "click_normalized", click)
    patch_pursuit(monkeypatch, "_ensure_frontmost", yes)
    patch_pursuit(monkeypatch, "current_page_identity", identity)
    patch_pursuit(monkeypatch, "_frontmost", in_front)
    patch_pursuit(monkeypatch, "_bring_the_thing_back_to_the_front", yes, raising=False)
    patch_pursuit(monkeypatch, "_whats_on_top", nothing_on_top, raising=False)
    patch_pursuit(monkeypatch, "_how_long_to_wait", lambda: 0.05, raising=False)
    return state


@pytest.mark.asyncio
async def test_one_restart_and_then_the_new_game_is_played(finished_then_fresh):
    async def names_nothing(objective, evidence):
        return ""

    await sp.pursue_on_screen(
        goal="play the game on TheBoard until the 2048 tile",
        target_app="TheBoard",
        success_when="2048",
        think=names_nothing,
        max_cycles=14,
        max_seconds=20.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert finished_then_fresh["clicked"] == 1, "she began again on a game that had just begun"
    assert finished_then_fresh["pressed"], "the new game was never played"
