"""A baking question answered with processor and memory percentages.

LIVE, 2026-08-28: "Something's off with my sourdough. It rose fine for months,
then the loaves came out dense. My friend says the starter has gone weak.
Design me the experiment, and say what result would prove your friend wrong."
came back "The machine is at 9.5% processor and 66.5% memory."

Two innocent words did it. The bare "your" in `_SELF_SUBJECT_RE` matched "your
friend", and the trouble pattern matched "something's off". Between them a
question about bread became a request for telemetry.

A possessive says what it is attached to, and the list of what she has was
already in the pattern. She has subsystems and lanes and a runtime. She does
not have a friend, a sourdough, or a deploy.
"""

from __future__ import annotations

from core.introspection.self_evidence import asks_about_own_operational_state as asks


def test_the_sourdough_question_is_not_about_her() -> None:
    said = (
        "Something's off with my sourdough. It rose fine for months, then the "
        "loaves came out dense. My friend says the starter has gone weak. "
        "Design me the experiment, and say what result would prove your friend "
        "wrong."
    )
    assert asks(said) is False


def test_a_possessive_attached_to_nothing_of_hers_is_not_about_her() -> None:
    for said in (
        "Tell my friend they are wrong about the oven.",
        "my deploy is failing",
        "your recipe is wrong",
        "is your friend's build broken",
    ):
        assert asks(said) is False, said


def test_a_possessive_attached_to_something_of_hers_still_is() -> None:
    for said in (
        "Which of your subsystems is degraded right now?",
        "Are any of your heartbeats failing?",
        "Is your conversation lane wedged?",
        "Is your runtime degraded?",
        "How much memory are you using?",
    ):
        assert asks(said) is True, said


def test_a_message_that_is_about_both_still_reaches_her() -> None:
    """The person's trouble first and hers second is still hers."""

    assert asks("My deploy is failing. Is your runtime degraded too?") is True


def test_the_host_she_runs_on_is_still_her_subject() -> None:
    assert asks("How hard is the machine you run on working right now?") is True
