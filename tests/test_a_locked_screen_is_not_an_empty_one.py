"""A locked screen is refused on purpose, and has to be reported as refused.

Nine live runs said "nothing on screen offered a move" while the truth was
that the screen was locked and no reading was possible at all. One of those is
about the screen's contents; the other is a privacy guard doing its job and
says nothing about the task.

Reported as the first, it sent me looking for a perception defect across four
separate fixes. LIVE 2026-08-27: CGSSessionScreenIsLocked was true the whole
time, and the answer was that somebody had to unlock the Mac.
"""

from __future__ import annotations

import pytest

from core.security.screen_capture_policy import (
    ScreenCaptureAdmission,
    ScreenCaptureDenial,
)


def refused(reason: ScreenCaptureDenial) -> ScreenCaptureAdmission:
    return ScreenCaptureAdmission(allowed=False, reason=reason)


# ── the refusal says which refusal it is ─────────────────────────────────

def test_a_locked_screen_says_it_is_a_session_and_not_a_screen():
    said = refused(ScreenCaptureDenial.SESSION_LOCKED).public_error
    assert "interactive session" in said
    assert "nothing" not in said


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (ScreenCaptureDenial.RUNTIME_SETTING_DISABLED, "permissions.screen"),
        (ScreenCaptureDenial.PRIVATE_FOREGROUND, "private content"),
        (ScreenCaptureDenial.PRIVATE_VISIBLE, "private content"),
        (ScreenCaptureDenial.FOREGROUND_UNKNOWN, "could not be verified"),
    ],
)
def test_and_every_other_refusal_says_which_one_it_is(reason, expected):
    assert expected in refused(reason).public_error


def test_an_admission_that_allows_says_nothing_at_all():
    assert ScreenCaptureAdmission(allowed=True).public_error == ""


def test_no_refusal_discloses_what_was_on_the_screen():
    """The whole point of the enum: a reason a person can act on, and nothing
    about what was visible."""
    for reason in ScreenCaptureDenial:
        said = refused(reason).public_error
        assert "\\n" not in said and len(said) < 120


# ── and the run reports it as its own kind of ending ─────────────────────

def test_the_pursuit_names_it_apart_from_finding_nothing():
    import inspect

    from core.skills import computer_use

    endings = inspect.getsource(computer_use)
    assert '"cannot_see": "the screen could not be read at all"' in endings
    assert '"no_move_available": "nothing on screen offered a move"' in endings
