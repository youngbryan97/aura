"""An answer cannot terminate before its own declared structure is complete."""

from __future__ import annotations

import pytest

from core.language.discourse_commitments import (
    discourse_commitments,
    unfulfilled_commitments,
)

pytestmark = pytest.mark.unit


def test_live_two_item_failure_explanation_is_not_complete_after_item_one() -> None:
    reply = (
        "So the break is likely one of two things:\n"
        "1. The automation timed out waiting for the app to launch."
    )

    commitment = unfulfilled_commitments(reply)[0]

    assert commitment.expected_count == 2
    assert commitment.observed_count == 1
    assert commitment.kind == "things"


@pytest.mark.parametrize(
    "reply",
    (
        "There are three likely causes:\n1. Load.\n2. Locking.\n3. Cache drift.",
        "The following two checks:\n- verify the PID\n- verify the revision",
        "One of two explanations:\nFirst: the file moved.\nSecond: the hash changed.",
    ),
)
def test_fully_emitted_declared_structures_are_complete(reply: str) -> None:
    commitments = discourse_commitments(reply)

    assert len(commitments) == 1
    assert commitments[0].fulfilled
    assert unfulfilled_commitments(reply) == ()


def test_duplicate_or_skipped_numbers_do_not_invent_missing_items() -> None:
    duplicate = "There are three reasons:\n1. First.\n1. Repeated.\n2. Second."
    skipped = "There are three reasons:\n1. First.\n3. Third."

    assert unfulfilled_commitments(duplicate)[0].observed_count == 2
    assert unfulfilled_commitments(skipped)[0].observed_count == 2


@pytest.mark.parametrize(
    "reply",
    (
        "I compared two reasons and chose the stronger one.",
        "The value is one of two things: yes or no.",
        "There are two sides to every comparison.",
    ),
)
def test_ordinary_cardinality_mentions_do_not_create_structure(reply: str) -> None:
    assert discourse_commitments(reply) == ()
