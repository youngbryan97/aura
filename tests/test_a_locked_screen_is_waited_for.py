"""A locked screen is a condition that passes, not a fault.

Failing at one turns "ask her, then sit down at the machine" into "ask her
again once you are there", and nothing tells the person that is what happened.
LIVE 2026-08-30: a request to play a game came back as
"pursue_on_screen failed: Screenshot failed: screen capture deferred while the
interactive session is unavailable".
"""

from __future__ import annotations

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
    """They are the one who can unlock it, and cannot if nothing says so."""
    said: list[str] = []

    async def narrate(line, because=""):
        said.append(line)

    monkeypatch.setattr("core.skills.screen_pursuit._narrate", narrate)
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

    monkeypatch.setattr("core.skills.screen_pursuit._narrate", narrate)
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        _answers(OPEN),
    )
    await wait_for_a_screen_to_look_at(time.monotonic() + 30.0)
    assert said == []
