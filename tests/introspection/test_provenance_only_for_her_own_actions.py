"""Her causal record answers "why did YOU", not "why do people".

LIVE 2026-08-17: "why do you think people find it hard to admit they were
wrong?" — a casual question about human psychology — came back with the
runtime's phase-by-phase provenance dump stapled underneath:

    From the runtime's own record of that turn, not from my impression of it:
    • ...

The subject is what separates the two. "Why did YOU pick that file" asks for
the record. "Why do you think PEOPLE lie" asks for her view, and the record has
nothing to say about it.
"""

from __future__ import annotations

import pytest

from core.introspection.decision_provenance import asks_why_she_did_that


@pytest.mark.parametrize(
    "question",
    [
        "why did you do that?",
        "why did you choose that file?",
        "why did you skip the verification step?",
        "why did you answer it that way?",
        "why do you think that is?",
    ],
)
def test_questions_about_her_own_actions_still_reach_the_record(question: str) -> None:
    assert asks_why_she_did_that(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "why do you think people find it hard to admit they were wrong?",
        "why do you think users lie?",
        "why do humans procrastinate?",
        "why does everyone hate meetings?",
        "why is the sky blue?",
    ],
)
def test_questions_about_the_world_do_not(question: str) -> None:
    assert asks_why_she_did_that(question) is False


@pytest.mark.parametrize("value", [None, "", "   ", "x" * 500])
def test_garbage_and_overlong_input_is_safe(value) -> None:
    assert asks_why_she_did_that(value) is False
