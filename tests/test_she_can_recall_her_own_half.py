"""Grounding one side of an exchange and leaving the other to invention.

LIVE 2026-08-30, asked "what did I ask you about first in this conversation,
and what did you say?": she quoted their turn exactly and then said "I didn't
get a chance to answer that before this follow-up" — about a question she had
answered a minute earlier, quoting a file to do it.

The grounding block carried their half only, and told her the quoted speaker
was not her. With nothing to read for her own half, she made one up.
"""

from __future__ import annotations

import time

import pytest

from core.conversation.grounded_recall import (
    build_grounded_recall_context,
    resolve_what_she_answered,
)


def _exchange():
    """Two turns each way, marked the way the chat route marks a real one."""
    now = time.time()

    def theirs(text: str, at: float) -> dict:
        return {
            "role": "user",
            "content": text,
            "timestamp": at,
            "origin": "chat_api",
            "metadata": {"source": "chat_api"},
        }

    def hers(text: str, at: float) -> dict:
        return {"role": "assistant", "content": text, "timestamp": at}

    return [
        theirs("how do I remember names?", now - 60),
        hers("Anchor it to something specific.", now - 59),
        theirs("which rule is most often broken?", now - 30),
        hers("No monkey-patching.", now - 29),
    ]


def test_her_reply_to_a_given_turn_is_retrievable():
    assert resolve_what_she_answered("how do I remember names?", _exchange()) == (
        "Anchor it to something specific."
    )


def test_a_turn_she_never_answered_returns_nothing():
    assert resolve_what_she_answered("something never asked", _exchange()) is None


def test_a_question_about_both_halves_grounds_both():
    said = build_grounded_recall_context(
        "What did I ask you first, and what did you say?", _exchange()
    )
    assert said
    assert "how do I remember names?" in said
    assert "Anchor it to something specific." in said
    assert "That one is yours" in said


def test_a_question_about_their_half_only_does_not_drag_hers_in():
    said = build_grounded_recall_context("What did I ask you first?", _exchange())
    assert said
    assert "how do I remember names?" in said
    assert "That one is yours" not in said


def test_where_she_never_replied_it_says_so_rather_than_leaving_a_gap():
    now = time.time()
    only_theirs = [
        {
            "role": "user",
            "content": "how do I remember names?",
            "timestamp": now - 60,
            "origin": "chat_api",
            "metadata": {"source": "chat_api"},
        },
    ]
    said = build_grounded_recall_context(
        "What did I ask you first, and what did you say?", only_theirs
    )
    if said:
        assert "no reply of yours on record" in said


@pytest.mark.parametrize(
    "asked",
    [
        "What did I ask you first, and what did you say?",
        "what was my first question and how did you answer it",
    ],
)
def test_the_role_boundary_is_still_stated(asked):
    """Their quote must never be handed back as hers."""
    said = build_grounded_recall_context(asked, _exchange())
    assert said
    assert "The quoted speaker is the user, not you" in said
