"""A key goes to the application it is for, and a goal is met where she acts.

Two defects from one live run, 2026-09-17, driving the 2048 desktop app.

A keystroke through System Events has no address: it goes to whatever is in
front when it arrives, so every key had to be guarded by a check of the front
window and could still be beaten to it. Addressed to the process that owns the
window, a key reaches that application and nothing else.

And the run ended on its first reading, twice, "already true after 0 moves":
the app's heading is a run of text that says 2048, which is also the number
she had been asked to reach. Where she can see the places of the thing she is
acting in, the finishing condition counts only inside them.
"""
from __future__ import annotations

import pytest
from screen_pursuit_support import patch_pursuit

from core.capabilities import window_server
from core.skills import screen_pursuit as sp
from core.skills import screen_pursuit_surface as surface


@pytest.mark.asyncio
async def test_a_key_is_posted_to_the_process_that_owns_the_window(monkeypatch):
    posted: list[tuple[int, list[str]]] = []
    a_window = window_server.Window(
        number=7, owner="Some Game", pid=4321, title="", left=0, top=0,
        width=400, height=400, layer=0, on_screen=True,
    )
    monkeypatch.setattr(window_server, "window_of", lambda app, **_: a_window)
    monkeypatch.setattr(
        window_server, "post_keys", lambda pid, keys, **_: posted.append((pid, list(keys))) or len(keys)
    )

    assert await surface.press("left", expect_app="Some Game") is True
    assert await surface.press_many(["up", "right"], expect_app="Some Game") == 2
    assert posted == [(4321, ["left"]), (4321, ["up", "right"])]


@pytest.mark.asyncio
async def test_with_no_window_to_address_the_key_goes_the_guarded_way(monkeypatch):
    from core.capabilities import host_automation

    monkeypatch.setattr(window_server, "window_of", lambda app, **_: None)
    asked: list[tuple[str, str]] = []

    class _Host:
        async def hotkey(self, key, *, expect_app=""):
            asked.append((key, expect_app))

            class _Receipt:
                success = True

            return _Receipt()

    monkeypatch.setattr(host_automation, "get_host_automation", lambda: _Host())
    assert await surface.press("down", expect_app="Somewhere") is True
    assert asked == [("down", "Somewhere")]


def test_a_named_key_has_a_code_and_a_character_does_not():
    assert window_server.key_code("Left") == 123
    assert window_server.key_code("a") is None


def test_where_the_goal_shows_says_which_runs_satisfy_it():
    reading = {
        "layout": [
            {"text": "2048", "center_x": 0.27, "center_y": 0.12},
            {"text": "SCORE", "center_x": 0.60, "center_y": 0.10, "x": 0.57, "width": 0.06},
            {"text": "2048", "center_x": 0.42, "center_y": 0.48},
        ]
    }
    assert surface.where_the_goal_shows(reading, "2048") == [(0.27, 0.12), (0.42, 0.48)]


def _with_a_grid(title_only: bool) -> dict:
    says = ["2", "4", "", ""] + [""] * 12
    if not title_only:
        says[5] = "2048"
    layout = [{"text": "2048", "center_x": 0.27, "center_y": 0.12, "width": 0.2, "height": 0.06}]
    if not title_only:
        layout.append({"text": "2048", "center_x": 0.42, "center_y": 0.48, "width": 0.05, "height": 0.04})
    return {
        "ok": True,
        "text": " ".join(r["text"] for r in layout),
        "layout": layout,
        "grids": [
            {
                "rows": 4, "columns": 4,
                "down_at": [0.34, 0.48, 0.62, 0.76],
                "across_at": [0.25, 0.42, 0.58, 0.75],
                "cell_width": 0.15, "cell_height": 0.13,
                "says": says,
            }
        ],
        "scoped_to": "Some Game",
        "owner": "Some Game",
        "window_number": 7,
    }


def _run_against(monkeypatch, reading: dict) -> dict:
    import asyncio

    from core.security.screen_capture_policy import ScreenCaptureAdmission

    async def allowed(*_a, **_k):
        return ScreenCaptureAdmission(allowed=True, reason=None)

    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async", allowed
    )

    async def read(app_name="", over=None):
        return reading

    async def press(key, *, expect_app=""):
        return True

    async def yes(*_a, **_k):
        return True

    async def identity():
        return {"url": "", "title": "", "error": ""}

    async def nobody_thinks(*_a, **_k):
        return "I will press left."

    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "press", press)
    patch_pursuit(monkeypatch, "_ensure_frontmost", yes)
    patch_pursuit(monkeypatch, "current_page_identity", identity)
    patch_pursuit(monkeypatch, "_bring_the_thing_back_to_the_front", yes, raising=False)
    return asyncio.run(
        sp.pursue_on_screen(
            goal="play until the 2048 tile",
            success_when="2048",
            think=nobody_thinks,
            target_app="Some Game",
            max_cycles=2,
            max_seconds=20.0,
            narrate=False,
            lived=False,
            research=False,
        )
    )


def test_the_goal_in_the_heading_is_not_the_goal_met(monkeypatch):
    result = _run_against(monkeypatch, _with_a_grid(title_only=True))
    assert result["outcome"] not in {"already_true", "goal_reached"}


def test_the_goal_in_the_places_she_acts_in_is_the_goal_met(monkeypatch):
    result = _run_against(monkeypatch, _with_a_grid(title_only=False))
    assert result["outcome"] in {"already_true", "goal_reached"}


def test_a_finished_thing_offers_starting_again_and_not_seeing_it_through():
    from core.skills.screen_pursuit_bearings import SEE_IT_THROUGH, START_OVER, ways_out

    reading = {"layout": [{"text": "Try again", "center_x": 0.5, "center_y": 0.6}]}
    ended = [option.name for option in ways_out(reading, ended=True)]
    going = [option.name for option in ways_out(reading, ended=False)]
    assert ended == [START_OVER]
    assert SEE_IT_THROUGH in going and START_OVER in going


def test_her_own_model_saying_no_act_changes_anything_is_an_ending():
    """Pressing every key to find out asks a question she has the answer to."""
    from types import SimpleNamespace

    from core.perception.how_it_moves import HowItMoves, shifted_and_combined
    from core.perception.what_is_there import Arrangement, Cell
    from core.perception.where_it_responds import Responsive
    from core.skills.screen_pursuit_decision import _decide_the_next_move_nothing_task_working

    def board(values):
        return Arrangement(
            4, 4, tuple(Cell(i // 4, i % 4, str(v), (0.0, 0.0)) for i, v in enumerate(values) if v)
        )

    rules = HowItMoves()
    state = board([2, 4, 0, 8, 0, 2, 4, 0, 4, 0, 0, 2, 64, 2, 0, 4])
    for move in ("left", "up", "right", "down", "left", "up"):
        after = shifted_and_combined(state, move)
        rules.watched(state, move, after)
        state = after
    assert rules.rule() is not None
    locked = board([2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256])

    class _CanDo:
        def available(self):
            return ["up", "down", "left", "right"]

    _options, ended = _decide_the_next_move_nothing_task_working(
        _CanDo(), ("up", "down", "left", "right"), {"layout": []},
        {"was_there": None, "said": False}, {"state": Responsive()},
        knows=SimpleNamespace(rules=rules), laid_out=locked,
    )
    assert ended is True
    _options, still_going = _decide_the_next_move_nothing_task_working(
        _CanDo(), ("up", "down", "left", "right"), {"layout": []},
        {"was_there": None, "said": False}, {"state": Responsive()},
        knows=SimpleNamespace(rules=rules), laid_out=state,
    )
    assert still_going is False


@pytest.mark.asyncio
async def test_whether_she_may_look_is_asked_of_the_window_she_will_read(monkeypatch):
    """A browser in front with a page she cannot name does not stop a look at a game's window."""
    import time

    from core.security import screen_capture_policy as policy
    from core.skills.screen_pursuit_looking import wait_for_a_screen_to_look_at

    a_window = window_server.Window(
        number=7, owner="Some Game", pid=4321, title="Some Game", left=0, top=0,
        width=400, height=400, layer=0, on_screen=True,
    )
    asked: list[tuple[str, str]] = []

    async def the_screen():
        return policy.ScreenCaptureAdmission(
            allowed=False, reason=policy.ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN
        )

    async def one_window(owner, title):
        asked.append((owner, title))
        return policy.ScreenCaptureAdmission(allowed=True)

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", the_screen)
    monkeypatch.setattr(policy, "evaluate_window_capture_admission_async", one_window)
    monkeypatch.setattr(window_server, "window_of", lambda app, **_: a_window if app == "Some Game" else None)

    assert await wait_for_a_screen_to_look_at(time.monotonic() + 5.0, app="Some Game") is True
    assert asked == [("Some Game", "Some Game")]
    # With no window of that name, the whole screen is what she would read.
    assert await wait_for_a_screen_to_look_at(time.monotonic() + 0.5, app="Nothing Open") is False
