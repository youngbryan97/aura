"""She played the game and answered with her autobiography.

LIVE 2026-08-19. "Go find a 2048 game online and play it until you get a 128
tile. Tell me what you are doing as you go." She found the site, opened it,
and played to a score of 888. The reply was "I'm Aura. I'm a local stateful
cognitive-agent runtime: memory, live state, tool governance, and local model
lanes feeding one user-facing voice."

The identity detector matched "tell me what you are" inside "tell me what you
are doing as you go" — the same defect _asks_only_who_you_are was written to
fix, in the branch beside it: a pattern matching the OPENING of a longer
question. It gets the same structural test rather than another list of words
that may not follow.

And when both readings fit, the action wins. A request to act on the machine
is not a question about her, and answering it as one loses the work.
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_repair import (
    _build_bounded_identity_repair_reply,
    _is_identity_request,
)


@pytest.mark.parametrize(
    "asked",
    [
        "who are you?",
        "what are you?",
        "tell me who you are",
        "tell me what you are",
        "introduce yourself",
    ],
)
def test_a_question_only_about_her_is_still_one(asked):
    assert _is_identity_request(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "tell me what you are doing as you go",
        "tell me what you are made of, honestly",
        "tell me what you are able to measure about yourself",
        "what are you doing right now with the browser",
    ],
)
def test_the_opening_of_a_longer_question_is_not_the_question(asked):
    assert not _is_identity_request(asked)


def test_the_live_request_is_not_an_identity_question():
    asked = (
        "Go find a 2048 game online and play it until you get a 128 tile. "
        "Tell me what you are doing as you go."
    )
    assert not _is_identity_request(asked)
    assert _build_bounded_identity_repair_reply(asked) == ""


def test_a_request_to_act_wins_over_a_reading_about_her():
    """Both readings fit "tell me what you are doing"; only one was asked for."""
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    asked = "Open Notes and write a haiku, then tell me what you are doing"
    assert looks_like_desktop_objective(asked)
    assert not _is_identity_request(asked)


def test_ordinary_identity_turns_still_get_their_answer():
    reply = _build_bounded_identity_repair_reply("who are you?")
    assert reply, "an identity question must still be answerable"
