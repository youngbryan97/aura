"""A person asking how she is has not asked for her telemetry.

LIVE 2026-08-26: "are you ok?" was answered with

    Processor 32.6%, memory 56.8%. Thermal pressure 0.00 of 1.

which is the same category error as answering a reflective question with a
log. She has an answer to how she is; it is hers to give, and it is not a
percentage.

Held apart by whether the question names an instrument, rather than by
teaching the matcher with examples: "are you ok" and "are you overloaded" sit
next to each other in any sentence embedding, and labelling one of them
dragged the other across with it.
"""
from __future__ import annotations

import pytest

from core.introspection.self_evidence import (
    _asks_after_her_rather_than_her_instruments,
    asks_about_own_operational_state,
)


@pytest.mark.parametrize(
    "asked",
    [
        "are you ok?",
        "are you okay?",
        "are you alright?",
        "you good?",
        "how are you?",
        "how are you doing?",
        "how are you holding up?",
        "everything ok on your end?",
        "is everything alright?",
    ],
)
def test_asking_after_her_is_hers_to_answer(asked):
    assert _asks_after_her_rather_than_her_instruments(asked)
    assert not asks_about_own_operational_state(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "how much memory are you using?",
        "is anything failing on your side?",
        "are you ok on memory?",
        "how are you doing on CPU?",
        "what is your current disk usage?",
        "are you throttling?",
    ],
)
def test_asking_what_she_reads_still_reaches_the_instruments(asked):
    """A question that names something an instrument reads is asking about
    the machine, whatever shape it takes."""
    assert not _asks_after_her_rather_than_her_instruments(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "my deploy is failing",
        "how much memory does a transformer need?",
        "what is the capital of Peru",
        "how are the tests looking?",
    ],
)
def test_a_question_about_something_else_is_neither(asked):
    assert not _asks_after_her_rather_than_her_instruments(asked)
    assert not asks_about_own_operational_state(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "what are you doing right now, and how are you going about it?",
        "what are you working on?",
        "what are you up to?",
        "what are you busy with?",
        "how are you going about that?",
    ],
)
def test_what_she_is_doing_is_about_her_activity_not_her_machine(asked):
    """She holds what she is working on and how she is going about it, and
    that is the answer to this.

    LIVE 2026-08-26, mid-task: "what are you doing right now, and how are you
    going about it?" was answered "The machine is at 0.0% processor and 69.3%
    memory right now."
    """
    assert _asks_after_her_rather_than_her_instruments(asked)
    assert not asks_about_own_operational_state(asked)


def test_naming_an_instrument_still_reaches_the_instruments():
    """"What are you doing with all that memory" is a question about memory."""
    assert asks_about_own_operational_state("what are you doing with all that memory?")
