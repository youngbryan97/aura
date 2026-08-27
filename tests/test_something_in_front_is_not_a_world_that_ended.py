"""A thing she cannot reach and a thing that has finished look identical inside.

Keys reported sent, her window reported frontmost, nothing moving. Measured
live 2026-08-26: a system permission dialog sat above every window, took the
keyboard, and swallowed every keystroke for an hour — and what she said was
that the game had ended.

Something else holding the keyboard is a different thing from a thing that is
finished, and it has a different answer: one of them somebody can fix.
"""

from __future__ import annotations

import pytest

from core.skills import screen_pursuit
from core.skills.screen_pursuit import ABOVE_EVERYTHING, _why_nothing_answers


class _Windows:
    """What the window server would say, as it says it."""

    def __init__(self, *windows: tuple[int, str]) -> None:
        self.windows = [
            {"kCGWindowLayer": layer, "kCGWindowOwnerName": owner} for layer, owner in windows
        ]

    kCGWindowListOptionOnScreenOnly = 1
    kCGNullWindowID = 0

    def CGWindowListCopyWindowInfo(self, *_a):
        return self.windows


@pytest.fixture
def seeing(monkeypatch):
    def use(*windows: tuple[int, str]):
        monkeypatch.setitem(__import__("sys").modules, "Quartz", _Windows(*windows))
    return use


@pytest.mark.asyncio
async def test_a_dialog_over_her_window_is_named(seeing):
    seeing((0, "Google Chrome"), (8, "UserNotificationCenter"))
    said = await _why_nothing_answers("Google Chrome")
    assert "UserNotificationCenter" in said
    assert "taking the keyboard" in said


@pytest.mark.asyncio
async def test_with_nothing_over_it_the_thing_has_finished(seeing):
    seeing((0, "Google Chrome"), (0, "Notes"))
    said = await _why_nothing_answers("Google Chrome")
    assert "this attempt is over" in said


@pytest.mark.asyncio
async def test_the_menu_bar_and_the_dock_are_not_in_her_way(seeing):
    seeing((0, "Google Chrome"), (ABOVE_EVERYTHING, "Dock"), (24, "Window Server"))
    said = await _why_nothing_answers("Google Chrome")
    assert "this attempt is over" in said


@pytest.mark.asyncio
async def test_her_own_window_is_not_in_her_way(seeing):
    seeing((0, "Google Chrome"), (8, "Google Chrome"))
    said = await _why_nothing_answers("Google Chrome")
    assert "this attempt is over" in said


@pytest.mark.asyncio
async def test_not_being_able_to_look_is_not_a_finding(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "Quartz", object())
    said = await _why_nothing_answers("Google Chrome")
    assert "this attempt is over" in said


@pytest.mark.asyncio
async def test_what_it_says_is_something_a_person_can_act_on(seeing):
    seeing((0, "Google Chrome"), (8, "SecurityAgent"))
    said = await _why_nothing_answers("Google Chrome")
    assert "in front of it" in said
    assert "getting through" in said
