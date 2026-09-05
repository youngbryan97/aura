"""A dialog in front of her work is an obstacle, not a reason to stop.

She could see one and name it and had no way to move it, because every key she
can send is bound to her own window and the dialog is not in it. Measured live
2026-08-26: two system permission dialogs sat above every window for an hour,
took the keyboard, and swallowed every keystroke — and what she said was that
the game had ended.

One key only. She may clear something out of her way; she may not agree to
something on somebody's behalf, and a dialog asking for a permission is
exactly where those two come apart.
"""

from __future__ import annotations

import inspect

import pytest

from core.skills import screen_pursuit
from core.skills.screen_pursuit import (
    DECLINES_AND_NOTHING_ELSE,
    clear_what_is_in_front,
)


class _Keyboard:
    def __init__(self) -> None:
        self.pressed: list[list[str]] = []

    async def hotkeys(self, keys, **kwargs):
        self.pressed.append(list(keys))
        return type("R", (), {"success": True, "evidence": {"keys_sent": len(keys)}})()


@pytest.fixture
def keyboard(monkeypatch):
    board = _Keyboard()
    import core.capabilities.host_automation as host

    monkeypatch.setattr(host, "get_host_automation", lambda: board)
    return board


def _sees(monkeypatch, *answers: str):
    """What is above her work, answer by answer.

    Both readers are patched. The check for "did it close" used to ask
    `_whats_on_top(on_top)`, whose first argument names the window to leave
    OUT — so it excluded the very thing it was checking for and reported
    success whenever the overlay was the only thing above her. That was
    fixed to read `_everything_on_top`, and this fixture went on patching
    only the old one, so the check saw an empty screen and said yes to
    everything.
    """

    seen = list(answers)

    async def everything(_mine, over=None):
        return tuple(one for one in ([seen.pop(0)] if seen else []) if one)

    async def on_top(mine, over=None):
        above = await everything(mine, over=over)
        return above[0] if above else ""

    monkeypatch.setattr(screen_pursuit, "_whats_on_top", on_top)
    monkeypatch.setattr(screen_pursuit, "_everything_on_top", everything)


# ── what she sends ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_she_declines_what_is_in_front_of_her(keyboard, monkeypatch):
    _sees(monkeypatch, "")
    assert await clear_what_is_in_front("UserNotificationCenter")
    assert keyboard.pressed == [[DECLINES_AND_NOTHING_ELSE]]


@pytest.mark.asyncio
async def test_the_only_key_she_sends_is_the_one_that_commits_to_nothing(
    keyboard, monkeypatch
):
    _sees(monkeypatch, "")
    await clear_what_is_in_front("SecurityAgent")
    for keys in keyboard.pressed:
        assert keys == ["escape"]


def test_and_she_never_agrees_to_anything_on_anybody_s_behalf():
    said = inspect.getsource(clear_what_is_in_front)
    assert "may not agree" in said
    assert "return" in said


@pytest.mark.asyncio
async def test_something_that_will_not_close_is_reported_rather_than_pressed_forever(
    keyboard, monkeypatch
):
    _sees(monkeypatch, "SecurityAgent")
    assert not await clear_what_is_in_front("SecurityAgent")


@pytest.mark.asyncio
async def test_nothing_in_front_is_nothing_to_clear(keyboard):
    assert not await clear_what_is_in_front("")
    assert keyboard.pressed == []


# ── when she tries ───────────────────────────────────────────────────────

def test_she_clears_it_before_reading_or_acting():
    source = inspect.getsource(screen_pursuit)
    clears = source.index("in_front = await _whats_on_top(")
    blocker = source.index("blocker = await clear_blocker(observation)")
    assert clears < blocker


def test_one_that_will_not_close_is_not_pressed_at_once_a_cycle():
    source = inspect.getsource(screen_pursuit)
    where = source.index("in_front = await _whats_on_top(")
    window = source[where : where + 300]
    assert 'in_front != in_the_way["last"]' in window


@pytest.mark.asyncio
async def test_and_she_says_so_when_it_worked(keyboard, monkeypatch):
    _sees(monkeypatch, "UserNotificationCenter", "")
    said = await screen_pursuit._why_nothing_answers("Google Chrome")
    assert "closed it" in said.lower()


@pytest.mark.asyncio
async def test_and_says_what_it_is_when_it_did_not(keyboard, monkeypatch):
    _sees(monkeypatch, "SecurityAgent", "SecurityAgent")
    said = await screen_pursuit._why_nothing_answers("Google Chrome")
    assert "SecurityAgent" in said
    assert "will not close" in said
