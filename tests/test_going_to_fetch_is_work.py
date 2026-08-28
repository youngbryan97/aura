"""Fetching the material is work the answer is built from, and it is not grammar.

LIVE, 2026-08-28: "Something's off in <path> ... Go through the code and tell me
what's actually happening, with the file and line" carried the closed-question
floor of 256 tokens. It was handed diagnose_repo, spent the budget reaching the
tool, and the request deadline expired before anything could be said about what
came back.
"""

from __future__ import annotations

import os

import pytest

from core.runtime.structured_input import (
    A_CLOSED_QUESTIONS_FLOOR,
    answer_surface_token_floor,
)

_HERE = os.getcwd()


def test_a_request_that_must_look_gets_more_than_a_closed_question() -> None:
    asked = (
        f"Something's off in {_HERE} and I can't put my finger on it. No error, "
        "nothing crashes, the tests such as they are pass. Go through the code "
        "and tell me what's actually happening, with the file and line."
    )
    assert answer_surface_token_floor(asked) > A_CLOSED_QUESTIONS_FLOOR


def test_a_plain_read_also_has_two_things_to_do() -> None:
    asked = f"read {_HERE}/README.md to me"
    assert answer_surface_token_floor(asked) > A_CLOSED_QUESTIONS_FLOOR


@pytest.mark.parametrize(
    "asked",
    [
        "what time is it",
        "how are you doing?",
        "explain how a binary search works",
        "are you still there",
    ],
)
def test_a_question_answered_from_here_is_unchanged(asked: str) -> None:
    assert answer_surface_token_floor(asked) == A_CLOSED_QUESTIONS_FLOOR


def test_a_path_that_does_not_exist_is_nowhere_to_go() -> None:
    asked = "Something's off in /no/such/place, go through it and tell me why"
    assert answer_surface_token_floor(asked) == A_CLOSED_QUESTIONS_FLOOR


def test_the_fetch_is_counted_rather_than_added() -> None:
    """It uses the accounting the function already does, not a new number."""

    plain = "tell me what this project does"
    looking = f"tell me what the project at {_HERE} does"
    # One more obligation, which the function prices at its own rate.
    assert answer_surface_token_floor(looking) > answer_surface_token_floor(plain)
