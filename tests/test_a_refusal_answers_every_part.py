"""You cannot name three of a thing you have just said you cannot see.

LIVE, 2026-09-21, and the person got nothing at all. Asked "What files are
in the current working directory of the process you are running in? Name
three.", she answered:

    I don't have access to the current working directory of the process
    I'm running in

which is true, complete, and the only honest answer available. The coverage
gate read the request as two parts and reported, six times in a row:

    coverage: 1 of 2 part(s) unanswered — missed ['Name three.']

The repair loop rejected the draft each time, and the error boundary
finally raised ``Foreground conversation lane produced only unsafe drafts:
unanswered_question_part``.

A reply that declines the request AS A WHOLE has engaged every part of it.
Whole is the load-bearing word: a decline with content after it, or with a
list in it, is a partial answer and keeps the per-part check.
"""

from __future__ import annotations

import pytest

from core.conversation.request_coverage import (
    declines_the_whole_request,
    unanswered_question_parts,
)


class _Contract:
    def __init__(self, segments):
        self.question_segments = tuple(segments)
        self.requires_single_reply_coverage = True
        self.numbered_parts = 0


THE_LIVE_TURN = _Contract(
    (
        "What files are in the current working directory of the process you are running in?",
        "Name three.",
    )
)


def test_the_live_turn_is_not_reported_unanswered():
    body = (
        "I don't have access to the current working directory of the process "
        "I'm running in"
    )
    assert unanswered_question_parts(body, THE_LIVE_TURN) == []


@pytest.mark.parametrize(
    "body",
    [
        "I don't have access to that.",
        "I cannot see the filesystem from here.",
        "I do not have a way to read it.",
        "I am not able to open that path.",
        "I have no access to that directory.",
    ],
)
def test_a_whole_decline_reads_as_one(body):
    assert declines_the_whole_request(body) is True


@pytest.mark.parametrize(
    "body",
    [
        "I can't open that one, but here are three others: a, b and c.",
        "I cannot read it. 1. alpha 2. beta 3. gamma",
        "I cannot see them.\n- alpha\n- beta",
        "The three files are a.py, b.py and c.py.",
        "",
        # A bare "I don't" is not a decline. This exact sentence appears in
        # tests/test_every_part_of_the_question_gets_answered.py, where the
        # reply IS supposed to fail coverage, and a first version of the
        # pattern matched it and let the reply through.
        "I'm doing fine, thanks. Just resting in the middle of a session "
        "where I don't have anything to do.",
        "I don't think that's right.",
    ],
)
def test_anything_with_content_keeps_the_per_part_check(body):
    assert declines_the_whole_request(body) is False


def test_a_partial_decline_is_still_checked_part_by_part():
    """The half-answer must not get a free pass from the decline in front of it."""
    body = "I can't list the directory, but here are three names: a.py, b.py, c.py."
    assert declines_the_whole_request(body) is False


def test_an_ordinary_answer_takes_the_path_it_always_did():
    """The new branch is entered only by a whole decline.

    The gate's verdict on everything else is whatever it was; this holds
    that nothing else reaches the early return.
    """
    contract = _Contract(("What is the capital of France?", "Name its river."))
    body = "Paris. The river is the Seine."
    assert declines_the_whole_request(body) is False
    # Same verdict as the per-segment walk gives on its own.
    assert unanswered_question_parts(body, contract) == [
        "What is the capital of France?"
    ]
