"""A consequential control usually asks, and she was walking away from it.

LIVE 2026-08-25. She decided to begin the game again, clicked "New Game"
correctly, and play2048 asked "Are you sure you want to start a new game?"
Nothing answered it. The score sat unchanged at 996 for the whole run while
she reported that she had begun again.

The dialog's own button is found by the question rather than by classifying
the dialog: whatever is asking sits next to the control that answers it, and
the control that answers is not the one already pressed. On that page the
question is "Are you sure you want to start a new game?" and the answer is
"Start New Game", directly below it.

General to any consequential control that confirms before acting.
"""
from __future__ import annotations

import pytest

from core.skills import screen_pursuit as sp

BOARD = {
    "ok": True,
    "text": "2048 SCORE 996 New Game",
    "bounds": [],
    "layout": [{"text": "New Game", "center_x": 0.75, "center_y": 0.13}],
}
ASKED = {
    "ok": True,
    "text": "2048 SCORE 996 New Game Are you sure you want to start a new game? Start New Game",
    "bounds": [],
    "layout": [
        {"text": "New Game", "center_x": 0.75, "center_y": 0.13},
        {"text": "New Game", "center_x": 0.49, "center_y": 0.45},
        {"text": "Are you sure you want to start a new game?", "center_x": 0.49, "center_y": 0.49},
        {"text": "Start New Game", "center_x": 0.49, "center_y": 0.575},
    ],
}
FRESH = {
    "ok": True,
    "text": "2048 SCORE 0 New Game",
    "bounds": [],
    "layout": [{"text": "New Game", "center_x": 0.75, "center_y": 0.13}],
}


@pytest.mark.asyncio
async def test_she_answers_the_question_her_own_click_raised(monkeypatch):
    state = {"clicks": [], "phase": "asked"}

    async def read(app_name=""):
        return ASKED if state["phase"] == "asked" else FRESH

    async def click(x, y, *, expect_app="", bounds=None):
        state["clicks"].append((round(x, 2), round(y, 2)))
        state["phase"] = "fresh"
        return True

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "click_normalized", click)

    _after, began = await sp._answer_own_confirmation(BOARD, "Google Chrome", "New Game")
    assert began, "the question went unanswered"
    assert state["clicks"] == [(0.49, 0.57)], "she pressed something other than the answer"


@pytest.mark.asyncio
async def test_she_does_not_press_the_same_control_again(monkeypatch):
    """Pressing the control that raised the question just re-asks it."""
    pressed = []

    async def read(app_name=""):
        return ASKED

    async def click(x, y, *, expect_app="", bounds=None):
        pressed.append((round(x, 2), round(y, 2)))
        return True

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "click_normalized", click)

    await sp._answer_own_confirmation(BOARD, "Google Chrome", "New Game")
    assert (0.75, 0.13) not in pressed


@pytest.mark.asyncio
async def test_nothing_is_pressed_when_nothing_is_asking(monkeypatch):
    """A surface that simply did not change is not a dialog."""
    pressed = []

    async def read(app_name=""):
        return BOARD

    async def click(x, y, *, expect_app="", bounds=None):
        pressed.append((x, y))
        return True

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "click_normalized", click)

    _after, began = await sp._answer_own_confirmation(BOARD, "Google Chrome", "New Game")
    assert not began
    assert pressed == []


@pytest.mark.asyncio
async def test_a_reset_that_simply_worked_asks_nothing(monkeypatch):
    async def read(app_name=""):
        return FRESH

    async def click(x, y, *, expect_app="", bounds=None):
        raise AssertionError("nothing needed pressing")

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "click_normalized", click)

    _after, began = await sp._answer_own_confirmation(BOARD, "Google Chrome", "New Game")
    assert began


def test_the_words_that_mean_a_dialog_is_asking():
    from core.skills.screen_pursuit import ASKING_TO_CONFIRM

    assert "are you sure" in ASKING_TO_CONFIRM
    assert any("do you want" in phrase for phrase in ASKING_TO_CONFIRM)
