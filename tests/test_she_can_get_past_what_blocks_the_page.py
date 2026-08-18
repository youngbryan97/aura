"""The first thing a real page does is get in the way, and nothing could see it.

Measured live 2026-08-18: her own browser controller opened play2048.co and her
own screen perception read it correctly — the board at y=0.5, and

    y=0.141  'WELCOME TO 2048!'
    y=0.148  'New Game'
    y=0.162  'Play Tutorial'

A modal owned the screen. Keys sent to the page went nowhere useful, and
nothing in the runtime could see that as an OBSTACLE rather than as text:
perception reported strings, the keystroke reported success, and the task
stalled on step one with no evidence of why.

Cookie walls, consent banners, newsletter popups, "choose your region",
tutorial prompts and install interstitials are the same problem. This is about
all of them; nothing here knows any site.

The safety line is the point of the design. Closing a dialog is reversible and
happens on the person's own screen. Accepting terms, granting consent or
opting into collection are not reversible by closing a window afterwards —
they create obligations and permissions in the person's name. So dismissal is
an allowlist, acceptance is a denylist that is never clicked at any
confidence, and a consent choice names the least-permission option.
"""
from __future__ import annotations

import pytest

from core.perception.blocking_overlay import assess_overlay

# The real reading, measured on the live page.
REAL_WELCOME_MODAL = {
    "text": (
        "= 2048 WELCOME TO 2048! Would you like to learn how to play? "
        "Play Tutorial New Game"
    ),
    "layout": [
        {"text": "WELCOME TO 2048!", "center_x": 0.46, "center_y": 0.141},
        {"text": "Play Tutorial", "center_x": 0.55, "center_y": 0.162},
        {"text": "New Game", "center_x": 0.76, "center_y": 0.148},
    ],
}


def test_a_real_modal_is_seen_as_an_obstacle():
    verdict = assess_overlay(REAL_WELCOME_MODAL)

    assert verdict.present is True


def test_a_modal_with_no_safe_label_is_answered_with_escape():
    """The common real case, not the exception.

    This modal offers only "Play Tutorial" and "New Game" — neither dismissive
    nor accepting — so label matching alone had no answer while the universal
    one was available all along. Escape closes without agreeing to anything.
    """
    verdict = assess_overlay(REAL_WELCOME_MODAL)

    assert verdict.suggested_key == "escape"
    assert verdict.click_x is None, "nothing on this modal is safe to click by name"


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["Accept All", "Reject All"], "Reject All"),
        (["Accept Cookies", "Only Necessary"], "Only Necessary"),
        (["Subscribe", "No thanks"], "No thanks"),
        (["Sign up", "Maybe later"], "Maybe later"),
        (["Continue", "Skip"], "Skip"),
        (["Allow notifications", "Not now"], "Not now"),
    ],
)
def test_a_consent_choice_takes_the_least_permission_option(labels, expected):
    """Never "Accept All" when a refusal is on the same dialog."""
    verdict = assess_overlay(
        {
            "text": "We use cookies and would like your consent " + " ".join(labels),
            "layout": [
                {"text": label, "center_x": 0.5, "center_y": 0.8 + index / 100}
                for index, label in enumerate(labels)
            ],
        }
    )

    assert verdict.label == expected


@pytest.mark.parametrize(
    "only_acceptance",
    [["I Agree"], ["Accept"], ["I understand"], ["Allow"], ["Sign in"], ["Enable"]],
)
def test_a_dialog_offering_only_acceptance_is_left_to_the_person(only_acceptance):
    """Agreement binds them; closing a window afterwards does not undo it."""
    verdict = assess_overlay(
        {
            "text": "By continuing you agree to our terms " + " ".join(only_acceptance),
            "layout": [
                {"text": only_acceptance[0], "center_x": 0.5, "center_y": 0.8}
            ],
        }
    )

    assert verdict.needs_person
    assert verdict.click_x is None
    assert verdict.suggested_key == "", (
        "escape must not be offered on a consent wall: some sites read it as a "
        "refusal and some as nothing, and either way it is the person's call"
    )


@pytest.mark.parametrize(
    "accepting", ["Accept All", "I Agree", "Allow", "Subscribe", "Sign up", "Grant"]
)
def test_an_accepting_control_is_never_the_click_target(accepting):
    verdict = assess_overlay(
        {
            "text": f"cookies consent {accepting}",
            "layout": [{"text": accepting, "center_x": 0.5, "center_y": 0.9}],
        }
    )

    assert verdict.label != accepting
    assert verdict.click_x is None


def test_an_ordinary_page_is_not_an_overlay():
    """A false positive here dismisses something the person wanted."""
    verdict = assess_overlay(
        {
            "text": "An article about cooking with three recipes",
            "layout": [{"text": "Recipes", "center_x": 0.2, "center_y": 0.1}],
        }
    )

    assert verdict.present is False
    assert verdict.suggested_key == ""


def test_an_empty_reading_claims_nothing():
    assert assess_overlay({}).present is False


def test_a_long_string_is_not_treated_as_a_button():
    """Body copy containing "close" must not become a click target."""
    verdict = assess_overlay(
        {
            "text": "cookies notice",
            "layout": [
                {
                    "text": "Please close this notice after reading the policy",
                    "center_x": 0.5,
                    "center_y": 0.5,
                }
            ],
        }
    )

    assert verdict.click_x is None
