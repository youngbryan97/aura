"""Seven ways of being refused a look, reported as one wrong sentence.

A desktop read can be refused because the session is locked, because the
setting that permits it is switched off, because something private is in
front, because she cannot tell what is in front, or because the policy could
not be consulted. All of them came back as "your screen is locked".

So the person was told to unlock a screen that was not locked, and the one
thing that would have fixed it went unsaid. LIVE 2026-09-05: the setting was
off, the screen was in use the whole time, and the run reported nothing to
look at.

They do not have the same remedy either. Waiting is right for a condition that
passes on its own — somebody unlocks a screen, closes a private window, brings
something forward. It is wrong for one that does not, and waiting out the
whole budget on a setting nobody will change during the task is failing, only
slower.
"""

from __future__ import annotations

import time

import pytest

import core.skills.screen_pursuit as sp
from core.skills.screen_pursuit import (
    PASSES_ON_ITS_OWN,
    SOMETHING_ELSE_IS_IN_FRONT,
    _what_being_refused_a_look_means,
    wait_for_a_screen_to_look_at,
)


class _Refused:
    def __init__(self, why: str) -> None:
        self.allowed = False
        self.reason = why


class _Allowed:
    allowed = True
    reason = "none"


def _refusing(monkeypatch, why: str) -> None:
    async def check():
        return _Refused(why)

    monkeypatch.setattr(sp, "evaluate_screen_capture_admission_async", check, raising=False)
    import core.security.screen_capture_policy as policy

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", check)


@pytest.mark.parametrize(
    "why",
    ["session_locked", "runtime_setting_disabled", "private_foreground",
     "foreground_unknown", "browser_title_unknown", "policy_unavailable"],
)
def test_every_refusal_is_said_in_its_own_words(why):
    said = _what_being_refused_a_look_means(why)
    assert said and said != _what_being_refused_a_look_means("session_locked") or why == "session_locked"


def test_a_disabled_setting_is_not_described_as_a_locked_screen():
    said = _what_being_refused_a_look_means("runtime_setting_disabled")
    assert "locked" not in said
    assert "switched off" in said


def test_an_unknown_reason_still_says_something_true():
    said = _what_being_refused_a_look_means("something_new_nobody_has_seen")
    assert "something_new_nobody_has_seen" in said


@pytest.mark.asyncio
async def test_a_refusal_that_waiting_cannot_change_is_not_waited_out(monkeypatch):
    """Waiting out the budget on a setting is failing, only slower."""
    _refusing(monkeypatch, "runtime_setting_disabled")
    began = time.monotonic()
    assert await wait_for_a_screen_to_look_at(began + 30.0) is False
    assert time.monotonic() - began < 2.0, "it waited for something that does not pass"


@pytest.mark.asyncio
async def test_the_reason_survives_for_whoever_has_to_report_it(monkeypatch):
    _refusing(monkeypatch, "runtime_setting_disabled")
    await wait_for_a_screen_to_look_at(time.monotonic() + 5.0)
    assert sp._WHY_SHE_CANNOT_LOOK["value"] == "runtime_setting_disabled"


@pytest.mark.asyncio
async def test_being_able_to_look_clears_the_reason(monkeypatch):
    async def allowed():
        return _Allowed()

    sp._WHY_SHE_CANNOT_LOOK["value"] = "session_locked"
    import core.security.screen_capture_policy as policy

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", allowed)
    assert await wait_for_a_screen_to_look_at(time.monotonic() + 5.0) is True
    assert sp._WHY_SHE_CANNOT_LOOK["value"] == ""


def test_not_knowing_what_is_in_front_is_not_having_no_screen():
    """There is a screen. She cannot tell what is on it, which is hers to fix."""
    assert SOMETHING_ELSE_IS_IN_FRONT <= PASSES_ON_ITS_OWN
    assert "session_locked" not in SOMETHING_ELSE_IS_IN_FRONT
    assert "runtime_setting_disabled" not in PASSES_ON_ITS_OWN


def test_the_conditions_that_pass_are_the_ones_somebody_can_change():
    """A person unlocks a screen or closes a window; nobody waits out a setting."""
    assert "session_locked" in PASSES_ON_ITS_OWN
    assert "private_foreground" in PASSES_ON_ITS_OWN
    assert "policy_unavailable" not in PASSES_ON_ITS_OWN
