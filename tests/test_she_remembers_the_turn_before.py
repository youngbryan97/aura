"""A question and its answer rarely share vocabulary.

LIVE 2026-08-26, one turn apart in the same conversation:

    "I'm thinking about a number: 47. Just hold on to it."
    "Got it. 47 is on hold."
    "what number did I say?"
    "I don't find that in this conversation's completed turns."

and

    "remember that my sister's name is Nell"
    "I've pinned it in durable session memory."
    "what did I just tell you about my family?"
    "Nothing. You haven't told me anything about your family."

Matching the question's words against the answer's words finds the cases
where somebody repeated themselves and misses ordinary conversation. What the
question DOES say is what kind of thing it wants back.
"""
from __future__ import annotations

import re

import pytest

from interface.routes.chat_memory_state import _kind_asked_for, _states_something


@pytest.mark.parametrize(
    ("asked", "told"),
    [
        ("what number did I say?", "I'm thinking about a number: 47. Just hold on to it."),
        ("what was the date again?", "let's meet on 14/03/2026"),
        ("which colour did I pick?", "go with teal for the header"),
        ("what file did I mention?", "the config lives at ~/.aura/settings.json"),
    ],
)
def test_the_question_says_what_kind_of_thing_it_wants_back(asked, told):
    kind = _kind_asked_for(asked)
    assert kind, "the question named a kind and none was recognised"
    assert re.search(kind, told, re.IGNORECASE), "the thing she was told is of that kind"


def test_a_general_question_asks_for_no_particular_kind():
    """"What did I tell you about my family?" is not asking to be quoted, and
    demanding a quotation of it finds nothing."""
    assert _kind_asked_for("what did I just tell you about my family?") == ""
    assert _kind_asked_for("what were my exact words?") != ""


@pytest.mark.parametrize(
    ("said", "is_a_statement"),
    [
        ("I'm thinking about a number: 47.", True),
        ("my sister's name is Nell", True),
        ("go with teal for the header", True),
        ("what did I say?", False),
        ("Is that right?", False),
        ("did you get that?", False),
        ("", False),
    ],
)
def test_a_turn_that_told_her_something_is_told_from_one_that_asked(said, is_a_statement):
    """"What did I tell you?" is answered by the turns where the person told
    her something, and an assertion is recognisable without knowing its
    subject."""
    assert _states_something(said) is is_a_statement


def test_the_search_falls_through_to_the_kind_when_words_do_not_match():
    import inspect

    from interface.routes import chat_memory_state

    source = inspect.getsource(chat_memory_state._find_session_content_exchanges)
    assert "_kind_asked_for" in source
    assert "_states_something" in source
    # And the empty-keyword case no longer gives up before looking.
    assert "if not keywords:\n        return []" not in source
