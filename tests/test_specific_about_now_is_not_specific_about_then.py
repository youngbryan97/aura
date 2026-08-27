"""A reading of her live state answers a question about now, and nothing else.

A repair substitutes that reading for a hedged reply when the person asked for
specifics. Where the question named another time, the substitute is specific
about the wrong thing and the answer that was there is gone. Measured live
2026-08-26: "what are you able to do that you could not do a month ago — be
specific" was answered "Things feel unusually settled right now. My attention
is on internal monitoring."
"""

from __future__ import annotations

import inspect

import pytest

from interface.routes import chat_conversation_repair
from interface.routes.chat_conversation_repair import _asks_about_another_time


@pytest.mark.parametrize(
    "asked",
    [
        "what could you not do a month ago, be specific",
        "be specific: what has changed since yesterday",
        "what can you do now that you could not before — no fluff",
    ],
)
def test_a_question_naming_another_time_is_recognised(asked):
    assert _asks_about_another_time(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "be specific about what you can do",
        "be more specific about how you feel right now",
        "give me specifics",
    ],
)
def test_a_question_about_now_is_not(asked):
    assert not _asks_about_another_time(asked)


def test_the_live_reading_is_not_offered_for_another_time():
    source = inspect.getsource(chat_conversation_repair)
    guard = "_contains_phrase(user_text, _SPECIFICITY_PUSH_MARKERS) and not _asks_about_another_time("
    assert guard in source


def test_the_repair_still_fires_for_a_question_about_now():
    """Narrow on purpose: only the comparison case is held back."""
    source = inspect.getsource(chat_conversation_repair)
    assert "_SPECIFICITY_PUSH_MARKERS" in source
    assert "the grounded read I have right now is:" in source
