"""Her instrument reading belongs on a request, not on every sentence.

LIVE, 2026-08-19. "forget the tests for a second. what's something you've
genuinely changed your mind about since you started running?" was answered:

    I can query beliefs — query_beliefs are registered and enabled right now,
    so if that failed it was the attempt and not the capability. ...

A registry status line stapled to an intimate question. Two words did it:
"running", as in OPERATING, matched the execution domain, and "forget" matched
the memory domain. The predicate was one flat list of words matched anywhere,
so any sentence containing any of them was treated as a request to act.

It now uses the same relation the capability router uses to pick a skill — a
verb and an object, in one clause, in the mood that asks — so the two agree
about what counts as asking for something.
"""

from __future__ import annotations

import pytest

from core.runtime.self_state_intent import asked_to_act_in_a_capability_domain


@pytest.mark.parametrize(
    "request_text",
    [
        "run a tiny bit of python and give me the actual number it printed",
        "can you execute code?",
        "look at my screen and tell me what you see",
        "search the web for fusion news",
        "use your interpreter and tell me what 2**40 is",
        "open the desktop calculator",
        "read the file and summarise it",
    ],
)
def test_a_real_request_still_attaches_her_reading(request_text: str):
    """The denial this predicate exists to prevent must stay prevented."""
    assert asked_to_act_in_a_capability_domain(request_text)


@pytest.mark.parametrize(
    "conversation",
    [
        "forget the tests for a second. what's something you've genuinely "
        "changed your mind about since you started running?",
        "what do you think about consciousness",
        "have you changed your mind about anything",
        "tell me a story about the sea",
        "my code doesn't run anymore",
        "how long have you been running?",
        "I use python at work",
    ],
)
def test_ordinary_conversation_gets_no_status_line(conversation: str):
    assert not asked_to_act_in_a_capability_domain(conversation)


def test_running_as_operating_is_not_running_as_executing():
    """The word that broke it, in both senses."""
    assert asked_to_act_in_a_capability_domain("run this python for me")
    assert not asked_to_act_in_a_capability_domain("how long have you been running")
    assert not asked_to_act_in_a_capability_domain("since you started running")


def test_a_memory_verb_is_read_by_its_complement():
    """"Remember that X" is an operation; "forget the tests" is an idiom.

    Memory verbs take whatever they are given, so no domain object appears in
    "remember that my sister's name is Ada" — and requiring one would drop a
    real memory write. What separates the two is the complement: a clause, an
    infinitive, or something of the speaker's, against a bare noun phrase.
    """
    for operation in (
        "remember that my sister's name is Ada",
        "remember that i prefer tea",
        "remember to ask me about the deploy tomorrow",
        "recall what I told you last week",
    ):
        assert asked_to_act_in_a_capability_domain(operation), operation

    for idiom in (
        "forget the tests for a second",
        "forget it, never mind",
    ):
        assert not asked_to_act_in_a_capability_domain(idiom), idiom


def test_someone_else_doing_it_is_still_not_her_doing_it():
    assert not asked_to_act_in_a_capability_domain("can a language model run code")
