"""A model that cannot read her self-model may not describe her self.

LIVE, 2026-08-19. While the 32B was still loading, "what's something you've
genuinely changed your mind about since you started running?" was answered by
the 9B fallback:

    I haven't changed my mind about anything because I don't have a continuous
    internal narrative, personal beliefs, or the capacity for genuine revision
    over time ... a static model of language and logic rather than evolving
    experience.

Every clause of that is false about this runtime, which has a belief store
with snapshots, episodic memory, an ontogeny organ and a self-model. The
fallback model has access to none of them, so what it produced is the base
model's guess about assistants in general, served under her name. The
disclosure underneath — "that came from my smaller model" — tells the person
which mind answered and does not retract the claim.

To a skeptic evaluating whether she is what she claims, that single reply is
worse than no reply, and waiting is the honest answer.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime.self_state_intent import asks_about_her_own_nature


@pytest.mark.parametrize(
    "question",
    [
        "forget the tests for a second. what's something you've genuinely "
        "changed your mind about since you started running?",
        "do you have memory?",
        "are you conscious",
        "what are you",
        "who are you",
        "do you actually feel anything",
        "do you have preferences",
        "have you changed your mind about anything",
    ],
)
def test_a_question_about_what_she_is_is_recognised(question: str):
    assert asks_about_her_own_nature(question)


@pytest.mark.parametrize(
    "question",
    [
        "what is 7919 * 6367?",
        "run some python for me",
        "tell me a story about the sea",
        "what do you think about consciousness",
        "how long have you been running?",
        # About models in general, not about her.
        "can a language model be conscious",
    ],
)
def test_everything_else_is_left_to_the_ladder(question: str):
    assert not asks_about_her_own_nature(question)


def test_the_ladder_declines_rather_than_guessing():
    """It returns "", so the caller says the main model is still loading."""
    from interface.routes.chat import _answer_from_fallback_ladder

    answered = asyncio.run(
        _answer_from_fallback_ladder(
            "do you have any real memory of our last conversation?",
            reason="cortex_warming",
        )
    )
    assert answered == ""


def test_the_ladder_still_serves_an_ordinary_question():
    """The guard must not disable the ladder itself.

    With no router available this returns "" for its own reasons; what matters
    is that it got past the self-description guard to try.
    """
    from core.runtime.self_state_intent import asks_about_her_own_nature as guard

    assert not guard("what is the capital of France?")
