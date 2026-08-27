"""Naming the machine she runs on is not asking about the machine.

"What are you able to do on this machine" names the machine the way "in this
room" names a room: it is the setting, and the question is about her. Measured
live 2026-08-26, answered "The machine is at 0.0% processor and 66.0% memory
right now." A preposition is the difference between a subject and a place, and
it is the same difference in any sentence.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import asks_about_own_operational_state as reads_instruments


@pytest.mark.parametrize(
    "asked",
    [
        "What are you actually able to do on this machine that you could not do a month ago?",
        "what can you do on this computer",
        "write me a file on this machine",
        "what have you learned working on this laptop",
        "show me what you can build with this hardware",
    ],
)
def test_the_machine_as_a_setting_is_not_the_question(asked):
    assert not reads_instruments(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "How hard is the machine you run on working right now? Give me a number.",
        "how much memory are you using",
        "is your cpu under load",
        "what are your thermals like",
    ],
)
def test_the_machine_as_the_subject_still_is(asked):
    assert reads_instruments(asked)


def test_a_question_about_her_is_still_about_her():
    assert not reads_instruments("are you ok?")
    assert not reads_instruments("what are you doing right now, and how are you going about it?")


def test_the_setting_does_not_swallow_a_real_enquiry_beside_it():
    """Two clauses, one about where she is and one about how she is."""
    assert reads_instruments(
        "you are running on this machine — how much memory are you using right now?"
    )
