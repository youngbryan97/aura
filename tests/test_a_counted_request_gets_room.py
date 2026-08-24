"""A request that says how many pieces it wants has said how much work it is.

LIVE, 2026-08-22. "six slides on the ledger project" was given an answer
surface of 256 tokens — the same as "what is 2 + 2?" — because a count of
parts was not among the obligations that reserve capacity. Every attempt at
that deck came back truncated: two slides of six, then three, then one.

The measurement that would have shown it has existed since July, and its own
comment says so: "a turn whose scaffold dwarfs the question tends to be
continued as more scaffold". Nothing acted on it.
"""

from __future__ import annotations

import pytest

from core.runtime.structured_input import (
    answer_surface_planning_tokens,
    answer_surface_token_floor,
)


@pytest.mark.parametrize(
    ("asked", "least"),
    [
        ("six slides on the ledger project", 1024),
        ("break it into five steps", 1024),
        ("give me three examples", 512),
        ("write four sections on the migration", 768),
    ],
)
def test_a_counted_request_reserves_room_for_the_count(asked: str, least: int):
    assert answer_surface_token_floor(asked) >= least, asked


@pytest.mark.parametrize(
    "asked",
    ["what is 2 + 2?", "how are you feeling?", "who founded Hugging Face?"],
)
def test_a_closed_question_is_unchanged(asked: str):
    assert answer_surface_token_floor(asked) == 256, asked


def test_more_pieces_reserve_more_room():
    """The floor follows the count rather than jumping to a constant."""
    three = answer_surface_token_floor("give me three examples")
    six = answer_surface_token_floor("give me six examples")
    ten = answer_surface_token_floor("give me ten examples")
    assert three < six < ten


def test_the_planning_prior_counts_them_too():
    """Capacity and the completion-length prior ask the same question."""
    assert answer_surface_planning_tokens("six slides on the ledger") > (
        answer_surface_planning_tokens("what is 2 + 2?")
    )


@pytest.mark.parametrize(
    ("asked", "least"),
    [
        ("numbered pseudocode", 768),
        ("a worked example with five weighted edges", 768),
        ("time complexity with both a heap and an array", 640),
        ("the failure case and the correct alternative", 512),
    ],
)
def test_one_structurally_expensive_obligation_gets_its_own_room(
    asked: str,
    least: int,
):
    """Decomposing a request must not collapse each work unit to 256 tokens."""

    assert answer_surface_token_floor(asked) >= least


def test_the_count_comes_from_the_shared_reader():
    """Not a second opinion about numbers."""
    from core.runtime.structured_input import _parts_the_request_counted

    assert _parts_the_request_counted("six slides") == 6
    assert _parts_the_request_counted("break it into five steps") == 5
    assert _parts_the_request_counted("some slides") == 0
    assert _parts_the_request_counted("what is 2 + 2?") == 0
