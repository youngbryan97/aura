"""Continuing an action must be requested, not merely mentioned."""

import pytest

from core.conversation.request_mood import assess_request_mood
from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.watched_goal import read_watched_goal
from core.skills.desktop_task import DesktopTaskSkill


DISCUSSION = (
    "Doesn't an async lock let other tasks keep running? Would a short critical "
    "section with no awaits change your choice?",
    "Can an event loop keep running while a coroutine waits?",
    "Why would someone keep refreshing until a build passes?",
    "Can you explain how to keep refreshing until the build passes?",
    "Tell me why an installer keeps on waiting for input.",
    "I was playing a puzzle until I reached 256.",
    '"Keep refreshing until the build passes."',
    "Don't keep refreshing the page until the build passes.",
)


@pytest.mark.parametrize("message", DISCUSSION)
def test_discussion_does_not_create_a_watched_goal(message):
    assert read_watched_goal(message) is None
    assert not looks_like_desktop_objective(message)


@pytest.mark.parametrize("message", DISCUSSION)
def test_planner_cannot_turn_discussion_into_a_pursuit(message):
    steps = DesktopTaskSkill()._derive_single_objective_steps(message, {})
    assert all(step.action != "pursue_on_screen" for step in steps)


@pytest.mark.parametrize("message", (
    "Keep refreshing the page until it says passed.",
    "Just keep playing forever.",
    "Can you keep refreshing the page until it says passed?",
    "Monitor the installer until it says Finished.",
    "Wait until the download is done.",
    "Step through the installer until it says Finished.",
    "Carry on until you reach 512.",
))
def test_actual_continuation_requests_still_reach_the_planner(message):
    assert assess_request_mood(message).asks_for_action
    assert read_watched_goal(message) is not None
    steps = DesktopTaskSkill()._derive_single_objective_steps(message, {})
    assert [step.action for step in steps] == ["pursue_on_screen"]


def test_an_unrelated_directive_does_not_authorize_a_discussed_continuation():
    message = "Does the installer keep running until it is done? Open Notes."
    assert assess_request_mood(message).asks_for_action
    assert read_watched_goal(message) is None


def test_a_discussion_does_not_suppress_a_separate_continuation_request():
    message = "Would that let the build keep running? Keep refreshing its page until passed."
    assert read_watched_goal(message) is not None


def test_shared_mood_preserves_explanation_and_action_clause_ownership():
    message = "Can you explain how to keep playing? Then keep refreshing the build page."
    mood = assess_request_mood(message)
    assert mood.asks_for_action
    assert mood.actionable_clauses == ("Then keep refreshing the build page",)
    assert mood.non_action_clauses == ("Can you explain how to keep playing",)


def test_live_followup_cannot_bypass_cognition_into_desktop_execution():
    from interface.routes.chat import _desktop_objective_self_sufficient_without_cognitive_text

    assert not _desktop_objective_self_sufficient_without_cognitive_text(DISCUSSION[0])
