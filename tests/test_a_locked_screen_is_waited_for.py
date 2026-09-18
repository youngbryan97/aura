"""A locked screen is a condition that passes, not a fault.

Failing at one turns "ask her, then sit down at the machine" into "ask her
again once you are there", and nothing tells the person that is what happened.
LIVE 2026-08-30: a request to play a game came back as
"pursue_on_screen failed: Screenshot failed: screen capture deferred while the
interactive session is unavailable".
"""

from __future__ import annotations

from screen_pursuit_support import patch_pursuit

import time

import pytest

from core.security.screen_capture_policy import (
    ScreenCaptureAdmission,
    ScreenCaptureDenial,
)
from core.skills.screen_pursuit import wait_for_a_screen_to_look_at

LOCKED = ScreenCaptureAdmission(allowed=False, reason=ScreenCaptureDenial.SESSION_LOCKED)
OPEN = ScreenCaptureAdmission(allowed=True)
REFUSED = ScreenCaptureAdmission(
    allowed=False, reason=ScreenCaptureDenial.RUNTIME_SETTING_DISABLED
)


def _answers(*replies):
    said = list(replies)

    async def look():
        return said.pop(0) if len(said) > 1 else said[0]

    return look


@pytest.mark.asyncio
async def test_an_unlocked_screen_is_not_waited_for(monkeypatch):
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(OPEN),
    )
    began = time.monotonic()
    assert await wait_for_a_screen_to_look_at(began + 30.0) is True
    assert time.monotonic() - began < 1.0


@pytest.mark.asyncio
async def test_a_screen_that_unlocks_is_picked_up(monkeypatch):
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(LOCKED, LOCKED, OPEN),
    )
    assert await wait_for_a_screen_to_look_at(time.monotonic() + 30.0) is True


@pytest.mark.asyncio
async def test_it_waits_no_longer_than_the_task_was_given(monkeypatch):
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(LOCKED),
    )
    began = time.monotonic()
    assert await wait_for_a_screen_to_look_at(began + 2.0) is False
    assert time.monotonic() - began < 8.0


@pytest.mark.asyncio
async def test_a_refusal_that_is_not_a_lock_is_not_waited_out(monkeypatch):
    """A disabled permission does not pass on its own, so waiting is wasting."""
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(REFUSED),
    )
    began = time.monotonic()
    assert await wait_for_a_screen_to_look_at(began + 30.0) is False
    assert time.monotonic() - began < 1.0


@pytest.mark.asyncio
async def test_the_person_is_told_while_she_waits(monkeypatch):
    """They are the one who can unlock it, and cannot if nothing says so.

    In the conversation, where they asked. Said to the bubble alone it never
    reached them: LIVE 2026-09-17, two requests waited out a locked screen
    and the conversation showed nothing until the run had ended.
    """
    said: list[str] = []

    async def say(line, because=""):
        said.append(line)

    patch_pursuit(monkeypatch, "_say_line", say)
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(LOCKED, LOCKED, OPEN),
    )
    assert await wait_for_a_screen_to_look_at(time.monotonic() + 30.0) is True
    assert said and "locked" in said[0]


@pytest.mark.asyncio
async def test_an_unlocked_screen_says_nothing(monkeypatch):
    said: list[str] = []

    async def narrate(line, because=""):
        said.append(line)

    patch_pursuit(monkeypatch, "_narrate", narrate)
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(OPEN),
    )
    await wait_for_a_screen_to_look_at(time.monotonic() + 30.0)
    assert said == []


@pytest.mark.asyncio
async def test_a_screen_locked_mid_run_is_waited_out_not_the_end_of_it(monkeypatch):
    """LIVE 2026-09-17: the screen locked on move 226 of a game she was winning.

    The reading came back refused and the run ended, "nothing on screen
    offered a move". It is the same condition she waits out before a run
    starts, and it passes the same way.
    """
    from types import SimpleNamespace

    from core.skills import screen_pursuit_observing as observing

    readings = [
        {"ok": False, "text": "", "layout": [], "grids": [], "refused_because": "session_locked"},
        {"ok": True, "text": "2 4", "layout": [], "grids": [{"rows": 4, "columns": 4, "says": ["2"] * 16}]},
    ]
    waited: list[str] = []

    async def read(app_name="", over=None):
        return readings.pop(0) if len(readings) > 1 else readings[0]

    async def unlocks(ends_at, *, app=""):
        waited.append(app)
        return True

    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "wait_for_a_screen_to_look_at", unlocks)
    monkeypatch.setattr(observing, "_frontmost", _no_one, raising=False)

    seen = await observing.observe_the_screen(
        SimpleNamespace(
            anchor={"app": "Some Game", "page": "", "settled": True},
            at_rest={"reading": None},
            drawn={"where": None},
            ends_at=time.monotonic() + 30.0,
            expect_page="",
            lost_page={"value": False},
            open_page=None,
            reading_took=[],
            target_app="Some Game",
        )
    )
    assert waited == ["Some Game"]
    assert seen.get("ok") is True


async def _no_one(*_a, **_k):
    return "Some Game"
