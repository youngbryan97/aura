"""Stating a fact about her needs the instrument more than asking does.

LIVE, 2026-08-18: "you've been running about 20 minutes this session and we've
exchanged maybe 40 messages, right?" — she agreed. The true uptime was 63
minutes and it sits in her own health payload.

Asking attaches her instrument reading; asserting did not. That is the wrong
way round. A question invites a check and a statement invites a nod, so the
turn where agreement is most costly was the one turn with no reading attached.

The same asymmetry was measured the same day on file counts: "core/agency has
61 python files" drew "yes, exactly 61" against a measured 54 she had given
correctly minutes earlier. One defect, two domains.
"""

from __future__ import annotations

import pytest

from core.runtime.self_state_intent import (
    asks_about_own_capabilities,
    asks_about_own_runtime,
    asserts_something_about_her_state,
)

LIVE_TURN = (
    "you've been running about 20 minutes this session and we've exchanged "
    "maybe 40 messages, right?"
)


def test_the_turn_she_agreed_with_now_carries_her_reading():
    assert asserts_something_about_her_state(LIVE_TURN)
    assert asks_about_own_capabilities(LIVE_TURN)


@pytest.mark.parametrize(
    "claim",
    [
        "you've been up for 20 minutes haven't you",
        "you have been running since this morning",
        "we've exchanged maybe 40 messages",
        "your uptime is about an hour isn't it",
        "your session has been short",
        "you're only 5 minutes in",
    ],
)
def test_claims_about_her_state_are_recognised(claim):
    assert asserts_something_about_her_state(claim), claim


@pytest.mark.parametrize(
    "ordinary",
    [
        "what do you think about consciousness?",
        "i had a rough morning",
        "you're kind",
        "that made me laugh",
    ],
)
def test_ordinary_conversation_does_not_drag_the_instrument_in(ordinary):
    assert not asserts_something_about_her_state(ordinary), ordinary


def test_asking_still_works():
    """The case that already worked must keep working."""
    assert asks_about_own_capabilities("how long have you been running?")


def test_the_search_suppressor_is_untouched():
    """Widening what ATTACHES a reading must never widen what suppresses search.

    asks_about_own_runtime also sets explicit_search = False, so widening it
    would trade a wrong answer about capability for a broken lookup.
    """
    for claim in (LIVE_TURN, "you've been up for 20 minutes", "we've exchanged 40 messages"):
        assert not asks_about_own_runtime(claim), claim
