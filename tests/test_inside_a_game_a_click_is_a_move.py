"""Inside the game she was sent to play, what can be clicked is among her moves.

A Flash game in a page draws its PLAY button, its answers and its NEXT on a
canvas: no element holds them, only the screen reading sees them, and the
arrow keys may do nothing at all. Her moves were the keys she was told plus
the arrows, so in a game played with the mouse she had nothing that worked.

Clicks are offered only inside the drawing the page said it is making, and
only once what she was told is not working — a world where the keys do the job
never widens, so a board game played with arrows is offered nothing new.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.agency.what_i_can_do_here import (
    ENOUGH_TO_JUDGE,
    WhatWorksHere,
    a_click_on,
    what_is_clicked,
)
from core.skills.screen_pursuit_bearings import screen_options, things_to_click, where_to_click

pytestmark = pytest.mark.unit

_SEEN = {
    "layout": [
        {"text": "PLAY", "center_x": 0.5, "center_y": 0.6},
        {"text": "  How  to play ", "center_x": 0.5, "center_y": 0.8},
    ],
    "bounds": [100, 200, 400, 300],
}


def test_a_click_is_named_by_what_is_written_on_it():
    assert a_click_on("  How  to play ") == 'click "How to play"'
    assert what_is_clicked('click "How to play"') == "How to play"
    assert what_is_clicked("up") is None
    assert what_is_clicked('click ""') is None


def test_nothing_is_clickable_outside_a_drawing():
    assert things_to_click(_SEEN, None) == ()


def test_inside_a_drawing_each_piece_of_writing_can_be_clicked():
    assert things_to_click(_SEEN, (0.1, 0.2, 0.9, 0.8)) == ('click "PLAY"', 'click "How to play"')
    assert where_to_click(_SEEN, "how to play") == (0.5, 0.8)
    assert where_to_click(_SEEN, "QUIT") is None


def test_a_click_becomes_an_option_with_its_own_test():
    options = screen_options(["up", 'click "PLAY"'])
    assert [option.name for option in options] == ["up", 'click "PLAY"']
    assert options[1].detail == 'click "PLAY"'
    assert options[1].expectation.changed is True


def test_keys_that_work_are_never_widened_to_clicks():
    can_do = WhatWorksHere(told=("up", "down", "left", "right"))
    can_do.on_screen = ('click "2"', 'click "4"')
    for key in can_do.told:
        can_do.tried(key, True)
    assert can_do.available() == ("up", "down", "left", "right")


def test_when_the_keys_do_nothing_the_things_to_click_are_offered():
    can_do = WhatWorksHere(told=("up", "down", "left", "right"))
    can_do.on_screen = ('click "PLAY"',)
    for _ in range(ENOUGH_TO_JUDGE):
        for key in can_do.told:
            can_do.tried(key, False)
    assert 'click "PLAY"' in can_do.available()
    # And a click that never does anything is let go like a key.
    for _ in range(ENOUGH_TO_JUDGE):
        can_do.tried('click "PLAY"', False)
    assert 'click "PLAY"' not in can_do.available()


def test_a_click_lands_where_the_reading_showed_it(monkeypatch):
    from core.skills import screen_pursuit_acting, screen_pursuit_surface

    clicked: list[tuple[Any, ...]] = []

    async def _click(x, y, *, expect_app="", bounds=None):
        clicked.append((x, y, expect_app, list(bounds or [])))
        return True

    monkeypatch.setattr(screen_pursuit_surface, "click_normalized", _click)
    run = SimpleNamespace(observation=_SEEN, target_app="Google Chrome", anchor={"app": ""})
    assert asyncio.run(screen_pursuit_acting._click_what_she_named(run, "PLAY"))
    assert clicked == [(0.5, 0.6, "Google Chrome", [100, 200, 400, 300])]
    # Gone from the reading: not guessed at.
    assert not asyncio.run(screen_pursuit_acting._click_what_she_named(run, "QUIT"))
    assert len(clicked) == 1


def test_she_says_she_is_clicking(monkeypatch):
    from core.skills import screen_pursuit, screen_pursuit_looking

    said: list[str] = []
    monkeypatch.setattr(screen_pursuit, "_tell", lambda line, *_a, **_k: said.append(line))
    monkeypatch.setattr(screen_pursuit, "_publish_decision", lambda *_a, **_k: None)
    screen_pursuit_looking._say_intent('click "PLAY"', out_loud=True)
    assert said and said[0].startswith('Clicking "PLAY"')


# ── the whole loop ───────────────────────────────────────────────────────────


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
def a_title_screen(monkeypatch):
    """A game that does nothing until its PLAY, drawn in the game, is clicked."""
    from screen_pursuit_support import patch_pursuit

    from core.security import screen_capture_policy as policy
    from core.skills import screen_pursuit_surface

    state: dict[str, Any] = {"playing": False, "pressed": [], "clicked": [], "turn": 0}

    async def read(app_name="", over=None):
        if not state["playing"]:
            layout = [{"text": "PLAY", "center_x": 0.5, "center_y": 0.8}]
        else:
            layout = [{"text": f"score {state['turn']}", "center_x": 0.2, "center_y": 0.1}]
        return {
            "ok": True,
            "text": "\n".join(region["text"] for region in layout),
            "layout": layout,
            "bounds": [10, 20, 540, 360],
            "scoped_to": app_name or "TheGame",
            "in_front_then": "TheGame",
            "her_window_showing": True,
        }

    async def press(key, *, expect_app=""):
        state["pressed"].append(key)
        if state["playing"]:
            state["turn"] += 1
        return True

    async def press_many(keys, *, expect_app=""):
        for key in keys:
            await press(key, expect_app=expect_app)
        return len(keys)

    async def click(x, y, **_k):
        state["clicked"].append((round(x, 2), round(y, 2)))
        if (round(x, 2), round(y, 2)) == (0.5, 0.8):
            state["playing"] = True
        return True

    async def yes(*_a, **_k):
        return True

    async def in_front(*_a, **_k):
        return "TheGame"

    async def nothing_on_top(*_a, **_k):
        return ""

    async def identity():
        return {"url": "", "title": "", "error": ""}

    async def may_look(*_a, **_k):
        return policy.ScreenCaptureAdmission(allowed=True)

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", may_look)
    monkeypatch.setattr(policy, "evaluate_window_capture_admission_async", may_look)
    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "press", press)
    patch_pursuit(monkeypatch, "press_many", press_many)
    patch_pursuit(monkeypatch, "click_normalized", click)
    monkeypatch.setattr(screen_pursuit_surface, "click_normalized", click)
    patch_pursuit(monkeypatch, "_ensure_frontmost", yes)
    patch_pursuit(monkeypatch, "current_page_identity", identity)
    patch_pursuit(monkeypatch, "_frontmost", in_front)
    patch_pursuit(monkeypatch, "_bring_the_thing_back_to_the_front", yes, raising=False)
    patch_pursuit(monkeypatch, "_whats_on_top", nothing_on_top, raising=False)
    patch_pursuit(monkeypatch, "_how_long_to_wait", lambda: 0.05, raising=False)
    return state


@pytest.mark.asyncio
async def test_a_game_that_waits_for_its_play_button_is_started_by_a_click(a_title_screen):
    from core.skills import screen_pursuit as sp

    async def names_nothing(objective, evidence):
        return ""

    await sp.pursue_on_screen(
        goal="play the game on TheGame",
        target_app="TheGame",
        success_when="score 5",
        think=names_nothing,
        max_cycles=40,
        max_seconds=40.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
        drawn_at=(0.1, 0.1, 0.9, 0.9),
    )
    assert a_title_screen["clicked"], "nothing she could press did anything, and she never clicked PLAY"
    assert a_title_screen["clicked"][0] == (0.5, 0.8)
    assert a_title_screen["playing"] and a_title_screen["turn"] > 0, "the started game was never played"


@pytest.mark.asyncio
async def test_without_a_drawing_nothing_is_clicked(a_title_screen):
    """Outside a game a click is a press of somebody's button, and is never tried."""
    from core.skills import screen_pursuit as sp

    async def names_nothing(objective, evidence):
        return ""

    await sp.pursue_on_screen(
        goal="play the game on TheGame",
        target_app="TheGame",
        success_when="score 5",
        think=names_nothing,
        max_cycles=30,
        max_seconds=30.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert a_title_screen["clicked"] == []


@pytest.mark.asyncio
async def test_where_the_keys_work_nothing_is_clicked(a_title_screen):
    """A world played with the keys she was told is never widened to clicks."""
    from core.skills import screen_pursuit as sp

    a_title_screen["playing"] = True

    async def names_nothing(objective, evidence):
        return ""

    await sp.pursue_on_screen(
        goal="play the game on TheGame",
        target_app="TheGame",
        success_when="score 12",
        think=names_nothing,
        max_cycles=30,
        max_seconds=30.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
        drawn_at=(0.1, 0.1, 0.9, 0.9),
    )
    assert a_title_screen["turn"] > 0
    assert a_title_screen["clicked"] == []
