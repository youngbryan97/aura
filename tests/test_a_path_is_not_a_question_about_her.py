"""A file's name says nothing about who the question is about.

Live on 2026-08-28: "Debug the failing pytest in
core/runtime/conversation_support.py and core/orchestrator/mixins/
tool_execution.py." was answered "The machine is at 10.0% processor and 50.0%
memory right now." The self-subject test matched "runtime" inside a directory
name and the trouble test matched "failing pytest", and between them a
debugging request became a request for telemetry.

The same shape as "your friend" and "something's off" turning a sourdough
question into one: a word read without what it is attached to.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import asks_about_own_operational_state


@pytest.mark.parametrize(
    "text",
    [
        "Debug the failing pytest in core/runtime/conversation_support.py and "
        "core/orchestrator/mixins/tool_execution.py.",
        "read core/brain/inference_gate.py and tell me what it does",
        "the memory tests in tests/test_memory_state.py are broken",
        "why is core/runtime/lockdep.py slow",
        "check config/settings.yaml for the timeout",
    ],
)
def test_naming_code_is_not_asking_after_her(text: str) -> None:
    assert asks_about_own_operational_state(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "is your runtime ok?",
        "how much memory are you using",
        "what is your processor doing right now",
    ],
)
def test_asking_after_her_still_works(text: str) -> None:
    assert asks_about_own_operational_state(text) is True


def test_a_path_beside_a_real_question_about_her_still_reaches_her() -> None:
    """Stripping the path must not strip the question wrapped around it."""

    assert (
        asks_about_own_operational_state(
            "your runtime is struggling — is core/runtime/lockdep.py the cause?"
        )
        is True
    )


def test_a_lead_in_about_the_answers_shape_is_not_the_question() -> None:
    """Two clauses, one word taken from each, and a check-in became a reading.

    Live on 2026-08-28: "Finish with a short status: are you still coherent, on
    the same thread, and able to continue?" was answered "The machine is at
    10.0% processor and 50.0% memory right now." The self-subject came from
    "are you", after the colon; the enquiry word came from "a short status",
    before it, where "status" describes the shape of the reply rather than a
    thing being asked about.
    """

    assert (
        asks_about_own_operational_state(
            "Finish with a short status: are you still coherent, on the same "
            "thread, and able to continue?"
        )
        is False
    )


def test_a_lead_in_that_names_her_keeps_its_subject() -> None:
    """Only the case where the question is on the far side of the colon."""

    assert asks_about_own_operational_state("about your memory: how much is left") is True
    assert (
        asks_about_own_operational_state("Give me a status: what is your processor doing")
        is True
    )
