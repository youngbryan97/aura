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


# ── Not everything that says "install" is a dialog ────────────────────────

def test_an_apps_own_toolbar_is_not_an_overlay():
    """The regression that made a loop press Escape forty times.

    Chrome carries an "Install" button in its toolbar and \\binstall\\b is a
    hint, so on the first window-scoped run every reading looked like a modal:
    the loop dismissed instead of playing and made no moves at all. A window's
    furniture is full of these words; a dialog blocking the content is IN the
    content.
    """
    verdict = assess_overlay(
        {
            "text": "Install 2048 Play the Free Online Game 2 4 8",
            "layout": [
                {"text": "C Install", "center_x": 0.67, "center_y": 0.079},
                {"text": "2048", "center_x": 0.84, "center_y": 0.043},
                {"text": "2", "center_x": 0.40, "center_y": 0.50},
                {"text": "4", "center_x": 0.50, "center_y": 0.60},
            ],
        }
    )

    assert verdict.present is False
    assert verdict.suggested_key == ""


def test_a_single_mention_is_not_a_dialog():
    """A page that mentions cookies is a page, not a cookie wall."""
    verdict = assess_overlay(
        {
            "text": "an article about cookies",
            "layout": [{"text": "Cookies and cream recipes", "center_y": 0.5}],
        }
    )

    assert verdict.present is False


def test_a_modal_in_the_content_area_is_still_caught():
    """Ignoring the top strip must not blind it to real dialogs."""
    verdict = assess_overlay(
        {
            "text": "welcome tutorial",
            "layout": [
                {"text": "WELCOME TO 2048!", "center_x": 0.46, "center_y": 0.30},
                {"text": "Would you like to learn how to play?", "center_x": 0.46, "center_y": 0.34},
                {"text": "Play Tutorial", "center_x": 0.55, "center_y": 0.38},
            ],
        }
    )

    assert verdict.present is True
    assert verdict.suggested_key == "escape"


def test_a_labelled_dismissal_needs_no_hint_quorum():
    """An explicit "No thanks" is evidence on its own."""
    verdict = assess_overlay(
        {
            "text": "join our list",
            "layout": [{"text": "No thanks", "center_x": 0.5, "center_y": 0.7}],
        }
    )

    assert verdict.label == "No thanks"


@pytest.mark.parametrize("glyph", ["X", "✕", "×", "⨯", "x"])
def test_a_bare_glyph_is_never_clicked(glyph):
    """It was, at low confidence, and it navigated away from the task.

    Driving a page, the detector found an "X" and clicked it; the browser ended
    up on x.com, because the glyph it matched was a tab label. One character
    carries no evidence of what it closes — it is a close control, a tab-close,
    a delete control, a clear-field control and a company logo, and the wrong
    guess destroys work or leaves the task entirely.

    Nothing is lost by refusing: a dialog whose only exit is a glyph is exactly
    what Escape is for, and Escape cannot close a tab.
    """
    verdict = assess_overlay(
        {
            "text": "subscribe notifications",
            "layout": [
                {"text": glyph, "center_x": 0.9, "center_y": 0.3},
                {"text": "Subscribe", "center_x": 0.5, "center_y": 0.4},
                {"text": "Notifications", "center_x": 0.5, "center_y": 0.5},
            ],
        }
    )

    assert verdict.click_x is None, f"{glyph!r} must never be a click target"


def test_a_dialog_whose_only_exit_is_a_glyph_falls_to_escape():
    verdict = assess_overlay(
        {
            "text": "welcome tutorial",
            "layout": [
                {"text": "WELCOME", "center_y": 0.30},
                {"text": "Would you like a tutorial?", "center_y": 0.35},
                {"text": "X", "center_y": 0.28},
            ],
        }
    )

    assert verdict.suggested_key == "escape"
    assert verdict.click_x is None


# ── A dialog that destroys work is not an obstacle ────────────────────────

DESTRUCTIVE_DIALOG = {
    "text": (
        "New Game Are you sure you want to start a new game? "
        "All progress will be lost. Start New Game"
    ),
    "layout": [
        {"text": "New Game", "center_x": 0.5, "center_y": 0.449},
        {"text": "Are you sure you want to start a new game?", "center_x": 0.5, "center_y": 0.489},
        {"text": "All progress will be lost.", "center_x": 0.5, "center_y": 0.512},
        {"text": "Start New Game", "center_x": 0.5, "center_y": 0.578},
    ],
}


def test_a_dialog_that_warns_of_loss_is_the_persons_decision():
    """Measured live, and the most important thing this file protects.

    Clearing the way on a real page produced "Are you sure you want to start a
    new game? All progress will be lost." Nothing could tell that apart from a
    cookie banner, so a loop told to get past obstacles would have wiped a game
    in progress — or, on another page, a draft, a cart, or an unsaved document.
    """
    verdict = assess_overlay(DESTRUCTIVE_DIALOG)

    assert verdict.needs_person
    assert verdict.click_x is None
    assert verdict.suggested_key == "", (
        "not even Escape: on some dialogs Escape cancels and on others it "
        "confirms the destructive default, and the difference is invisible"
    )


@pytest.mark.parametrize(
    "warning",
    [
        "All progress will be lost.",
        "This cannot be undone.",
        "This will permanently delete your account.",
        "You have unsaved changes.",
        "Discard your draft?",
        "This action is irreversible.",
    ],
)
def test_every_loss_warning_stops_the_loop(warning):
    verdict = assess_overlay(
        {
            "text": f"Are you sure? {warning} Continue",
            "layout": [
                {"text": warning, "center_x": 0.5, "center_y": 0.5},
                {"text": "Continue", "center_x": 0.5, "center_y": 0.6},
                {"text": "Cancel", "center_x": 0.4, "center_y": 0.6},
            ],
        }
    )

    assert verdict.needs_person, warning
    assert verdict.click_x is None, f"{warning}: nothing here is safe to click"


def test_an_ordinary_banner_is_unaffected_by_the_new_rule():
    """The guard must not swallow the cases that were working."""
    verdict = assess_overlay(
        {
            "text": "We use cookies for consent",
            "layout": [
                {"text": "Reject All", "center_x": 0.6, "center_y": 0.9},
                {"text": "Accept All", "center_x": 0.7, "center_y": 0.9},
            ],
        }
    )

    assert verdict.label == "Reject All"
    assert not verdict.needs_person


def test_a_control_label_is_not_a_warning():
    """LIVE: a finished game showing "Try again" and "Start over" could never
    be restarted — deciding to restart it produced a halt saying the decision
    was the person's to make.

    A warning is a sentence about a consequence. "Start over" is a button.
    """
    from core.perception.blocking_overlay import DESTRUCTIVE_WARNINGS

    assert not any("start" in pattern and "over" in pattern for pattern in DESTRUCTIVE_WARNINGS)


def test_a_warning_about_what_she_chose_is_a_confirmation_not_an_ambush():
    from core.perception.blocking_overlay import assess_overlay

    text = "Game over! Try again. Your progress will be lost."
    seen = {"ok": True, "text": text, "layout": [{"text": w, "center_y": 0.4} for w in text.split()]}
    assert assess_overlay(seen).needs_person, "unasked-for destruction still stops her"
    assert not assess_overlay(seen, intending="start over").needs_person


def test_permanent_deletion_stops_her_whatever_she_intended():
    """Losing this attempt is recoverable by doing it again. This is not."""
    from core.perception.blocking_overlay import assess_overlay

    text = "This will permanently delete all your files and cannot be undone"
    seen = {"ok": True, "text": text, "layout": [{"text": w, "center_y": 0.4} for w in text.split()]}
    assert assess_overlay(seen, intending="start over").needs_person


def test_an_unrelated_intent_does_not_unlock_a_warning():
    from core.perception.blocking_overlay import assess_overlay

    text = "Your unsaved changes will be lost"
    seen = {"ok": True, "text": text, "layout": [{"text": w, "center_y": 0.4} for w in text.split()]}
    assert assess_overlay(seen, intending="press up").needs_person
