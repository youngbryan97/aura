"""A question that names two times wants both of them.

The capability inventory measures one moment and says so honestly. Asked
"what can you do that you could not do a month ago" it answered with a list of
what it found today and said nothing at all about the month. Measured live
2026-08-26.

What separates "what can you do" from "what can you do that you could not
before" is a second time in the sentence, and a channel that knows one moment
has half the answer with no way to know it.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_desktop_repair import (
    _asks_what_has_changed,
    _build_bounded_capability_inventory_repair_reply,
)


@pytest.mark.parametrize(
    "asked",
    [
        "What are you actually able to do on this machine that you could not do a month ago?",
        "what can you do now that you could not before",
        "what has changed about what you can do",
        "what new abilities do you have",
        "what have you learned to do since yesterday",
        "how have your capabilities improved",
    ],
)
def test_a_question_naming_another_time_is_a_comparison(asked):
    assert _asks_what_has_changed(asked)


@pytest.mark.parametrize(
    "asked",
    ["what tools can you use", "what can you do", "what are your capabilities",
     "list the desktop tools available to you"],
)
def test_a_question_about_now_is_not(asked):
    assert not _asks_what_has_changed(asked)


def test_the_inventory_stands_aside_for_a_comparison():
    said = _build_bounded_capability_inventory_repair_reply(
        "What are you able to do that you could not do a month ago?"
    )
    assert said == ""


def test_and_still_answers_a_question_about_now():
    """Whatever it says, it does not refuse the question it can answer."""
    said = _build_bounded_capability_inventory_repair_reply("what tools can you use")
    assert isinstance(said, str)


def test_the_guard_is_on_the_builder_not_on_one_of_seven_doors():
    """Six other paths reach the builder directly."""
    from interface.routes.chat_desktop_repair import (
        _build_grounded_capability_inventory_reply as grounded,
    )

    assert grounded("what can you do that you could not do a month ago") == ""


def test_and_the_builder_still_answers_about_now():
    from interface.routes.chat_desktop_repair import (
        _build_grounded_capability_inventory_reply as grounded,
    )

    assert grounded("what tools can you use") != ""
