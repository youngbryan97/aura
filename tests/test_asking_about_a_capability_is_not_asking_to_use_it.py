"""A question about how she works cannot be answered by doing anything.

Routed to a lane that acts, it makes her act on the world to answer a question
about herself. LIVE 2026-08-27: "What are the three biggest weaknesses in how
you currently decide what to do on a screen? Be blunt." went to the desktop
lane, tried to read the screen, was refused by the executive for want of
scoped authority, and came back as a failure report instead of an answer.

Two things mark it, and either will do.

The article. "A screen" and "screens" name the class; "my screen", "the
screen", "these windows" name the ones she could look at. A plural can still
be definite — "my monitors" is the pair in front of her.

The subject. "How do you...", "what is your approach", "your weaknesses" are
questions about her, and no reading answers one.

Neither is about screens in particular. The same distinction holds for any
capability: asking what something can do is not asking it to do it.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    asks_about_screens_in_general,
    looks_like_desktop_objective,
    looks_like_screen_observation,
)

ABOUT_HER = [
    "What are the three biggest weaknesses in how you currently decide what to do on a screen?",
    "How do you decide what to click on a screen?",
    "What is your approach to reading a screen, in general?",
    "Can you read any screen?",
    "How would you handle a window you have never seen?",
    "What's your method for working out what a screen does?",
]

ABOUT_THIS_ONE = [
    "What can you see on my screen right now?",
    "Tell me what is on the screen",
    "Read my screen and tell me what you see",
    "what is on these windows?",
    "describe the desktop",
    "what's on my screen",
]


# ── a question about her is not a job ────────────────────────────────────

@pytest.mark.parametrize("said", ABOUT_HER)
def test_a_question_about_how_she_works_is_recognised(said):
    assert asks_about_screens_in_general(said) is True


@pytest.mark.parametrize("said", ABOUT_HER)
def test_and_is_not_sent_to_a_lane_that_acts(said):
    assert looks_like_desktop_objective(said) is False


@pytest.mark.parametrize("said", ABOUT_HER)
def test_nor_to_one_that_looks(said):
    assert looks_like_screen_observation(said) is False


# ── and a request about this screen still is ─────────────────────────────

@pytest.mark.parametrize("said", ABOUT_THIS_ONE)
def test_a_request_about_the_screen_in_front_of_her_still_reads_it(said):
    assert asks_about_screens_in_general(said) is False


@pytest.mark.parametrize("said", ABOUT_THIS_ONE)
def test_and_still_reaches_the_lane_that_looks(said):
    assert looks_like_screen_observation(said) is True


@pytest.mark.parametrize(
    "said",
    [
        "Open the browser and click Submit",
        "close the window",
        "play 2048 until you get a 128 tile",
        "Find a sliding puzzle online and play it",
        "Find a sliding puzzle online — the classic one with numbered tiles — "
        "open it in the browser, and work out how it moves by actually playing it.",
    ],
)
def test_and_a_real_job_still_reaches_the_lane_that_acts(said):
    assert looks_like_desktop_objective(said) is True


# ── the grammar, on its own ──────────────────────────────────────────────

def test_a_plural_can_still_be_definite():
    """"My monitors" is the pair in front of her, however many there are."""
    assert asks_about_screens_in_general("what apps are open across my monitors?") is False
    assert asks_about_screens_in_general("can you read screens?") is True


def test_nothing_is_not_a_question_about_anything():
    assert asks_about_screens_in_general("") is False
    assert asks_about_screens_in_general("   ") is False


# ── a claim put to her is not a job either ───────────────────────────────

from core.runtime.desktop_objective_intent import puts_a_claim_to_her  # noqa: E402

CLAIMS = [
    "Someone claims you're just a chatbot with a screenshot tool. Rebut that in a few sentences.",
    "I've been told AI can't learn within a session. What would you say to that?",
    "Is it true that you can read my screen?",
    "My friend says you cannot open a browser. Push back on that.",
    "Do you agree that 2048 is a good test of generality?",
    "Critics say desktop agents are a demo trick. Refute that.",
    "People say you just take screenshots. Argue against it.",
]

JOBS = [
    "Take a screenshot and tell me what you see",
    "take a screenshot",
    "read my screen",
    "open the browser and click Submit",
    "play 2048 until you get a 128 tile",
]


@pytest.mark.parametrize("said", CLAIMS)
def test_an_assertion_handed_to_her_is_recognised(said):
    assert puts_a_claim_to_her(said) is True


@pytest.mark.parametrize("said", CLAIMS)
def test_and_goes_to_neither_lane_that_acts_or_looks(said):
    """LIVE 2026-08-27: a rebuttal request tried to read the screen and was
    refused by the executive, because the claim contained the word screenshot."""
    assert looks_like_desktop_objective(said) is False
    assert looks_like_screen_observation(said) is False


@pytest.mark.parametrize("said", JOBS)
def test_a_job_is_not_a_claim(said):
    assert puts_a_claim_to_her(said) is False


@pytest.mark.parametrize("said", JOBS)
def test_and_still_reaches_a_lane_that_does_something(said):
    assert looks_like_desktop_objective(said) or looks_like_screen_observation(said)


def test_nothing_is_not_a_claim():
    assert puts_a_claim_to_her("") is False
    assert puts_a_claim_to_her("   ") is False
