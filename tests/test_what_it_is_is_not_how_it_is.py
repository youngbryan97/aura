"""Asking what a part IS is not asking how it is doing.

The self-evidence gate settles on a host reading as soon as a possessive
reaches a machine word: "your memory" needs no condition word beside it, which
is right for "how much memory are you using".

It is wrong for every question about what her memory is.

LIVE, 2026-09-07: "Describe your memory system." was answered "The machine is
at 23.9% processor and 58.3% memory right now. Thermal pressure 0.25 of 1."
The reply is not even about her memory; it is about the host's, and the
question was about architecture.

A trouble word keeps the condition reading, because "explain why your memory
is degraded" really is asking after one.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import asks_about_own_operational_state


@pytest.mark.parametrize(
    "question",
    [
        "Describe your memory system.",
        "How does your memory work?",
        "Explain your memory architecture.",
        "What is your memory system made of?",
        "Walk me through your memory design.",
        "Tell me about your memory.",
        "How is your memory implemented?",
        "What does your memory consist of?",
    ],
)
def test_an_architecture_question_is_not_a_telemetry_question(question: str) -> None:
    assert not asks_about_own_operational_state(question)


@pytest.mark.parametrize(
    "question",
    [
        "How much memory are you using?",
        "Is your memory under pressure?",
        "What's your status?",
        "Which of your subsystems is degraded right now?",
        "Are your thermals ok?",
        "How hard are you working right now?",
    ],
)
def test_a_condition_question_still_reaches_her_instruments(question: str) -> None:
    assert asks_about_own_operational_state(question)


@pytest.mark.parametrize(
    "question",
    [
        "Explain why your memory is degraded.",
        "Describe what is going wrong with your runtime.",
        "Walk me through the failures on your side.",
    ],
)
def test_a_description_of_a_fault_is_still_about_the_fault(question: str) -> None:
    """The frame does not take a trouble word away."""

    assert asks_about_own_operational_state(question)


@pytest.mark.parametrize(
    "question",
    [
        "Plan, in three steps, how you would find out whether your own episodic "
        "memory is being written to right now. Then do step one.",
        "Design an experiment to test your own memory.",
        "Work out whether your memory is being written to.",
        "Figure out how to prove your runtime is healthy.",
        "Check whether your memory writes are landing.",
    ],
)
def test_asking_her_to_work_it_out_is_not_asking_for_a_reading(question: str) -> None:
    """A method, an experiment or an action — a reading is what it is not.

    LIVE, 2026-09-08: "Plan, in three steps, how you would find out whether
    your own episodic memory is being written to right now. Then do step one."
    was answered "The machine is at 0.0% processor and 59.1% memory right now."
    """

    assert not asks_about_own_operational_state(question)


@pytest.mark.parametrize(
    "question",
    [
        "Work out why your memory is degraded.",
        "Figure out what is failing on your side.",
        "Investigate why your runtime is unhealthy.",
    ],
)
def test_working_out_a_fault_is_still_about_the_fault(question: str) -> None:
    assert asks_about_own_operational_state(question)
