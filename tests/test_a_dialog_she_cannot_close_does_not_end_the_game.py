"""A dialog she cannot close is not a game that has ended.

LIVE 2026-09-24: four arrows in a row changed nothing while a system
notification sat in front of the board. She named it, pressed the key that
declines, and it stayed. The ending she had concluded from the silence was
kept, so starting again was offered as if the game were over, without needing
a reason in words, and the part of her that likes an act she has never tried
chose it: "acting to find out: start over 1.000". A board with a 64 on it went
to a notification somebody else had to answer.
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


def _mind_that_names_nothing():
    async def think(objective, evidence):
        return ""

    return think


@pytest.fixture
def covered(monkeypatch):
    """A board nothing moves, a New Game control, and a notification over it."""
    state = {"pressed": [], "clicked": [], "cleared": 0, "covered_by": "UserNotificationCenter"}

    async def read(app_name="", over=None):
        return {
            "ok": True,
            "text": "board 2\nNew Game",
            "layout": [{"text": "New Game", "center_x": 0.8, "center_y": 0.2}],
            "bounds": [],
            "scoped_to": app_name or "TheBoard",
            "in_front_then": "TheBoard",
            "her_window_showing": True,
        }

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        return True

    async def click(x, y, **_k):
        state["clicked"].append((x, y))
        return True

    async def yes(*_a, **_k):
        return True

    async def in_front(*_a, **_k):
        return "TheBoard"

    async def on_top(*_a, **_k):
        return state["covered_by"]

    async def will_not_close(_what):
        state["cleared"] += 1
        return False

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
    patch_pursuit(monkeypatch, "_whats_on_top", on_top, raising=False)
    patch_pursuit(monkeypatch, "clear_what_is_in_front", will_not_close, raising=False)
    patch_pursuit(monkeypatch, "_how_long_to_wait", lambda: 0.05, raising=False)
    return state


async def _play(cycles: int) -> dict:
    return await sp.pursue_on_screen(
        goal="play the game on TheBoard",
        target_app="TheBoard",
        success_when="never happens",
        think=_mind_that_names_nothing(),
        max_cycles=cycles,
        max_seconds=20.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )


@pytest.mark.asyncio
async def test_she_does_not_start_again_under_a_dialog_she_cannot_close(covered):
    await _play(16)
    assert covered["cleared"], "she never got as far as trying to move it"
    assert not covered["clicked"], "she threw the game away to a notification"


@pytest.mark.asyncio
async def test_she_keeps_trying_the_thing_itself(covered):
    await _play(16)
    tried = set(covered["pressed"])
    assert len(covered["pressed"]) > len(tried), "she stopped pressing after one round"


@pytest.mark.asyncio
async def test_with_nothing_over_it_silence_is_still_an_ending(covered):
    """The guard is about the dialog. A finished game with nothing over it ends."""
    covered["covered_by"] = ""
    await _play(16)
    assert covered["clicked"], "a finished game with nothing over it was never begun again"
