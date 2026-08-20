"""An expression has to stand on its own, not sit inside a name.

LIVE, 2026-08-19. Asked to debug an unfamiliar repository:

    there's a python project at /private/tmp/claude-501/-Users-bryan--aura-
    live-source/7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/ledger - one
    of its tests is failing. read the code, work out why, and tell me exactly
    which line is wrong and what it should be.

the whole reply was "30."

Two things combined. "Work out" satisfied the arithmetic intent gate, and the
bare-expression pattern found "7-8" INSIDE the UUID `47f7-8c38`. A request to
debug a repository was answered with a number, and because a computed answer
is served in place of the model's draft, the number replaced the entire reply
rather than merely appearing in it.

Paths, UUIDs, version strings and hyphenated names are full of digits with
operators between them, so the pattern needs boundaries.
"""

from __future__ import annotations

import pytest

from core.conversation.arithmetic_check import requested_arithmetic_result

TASK = (
    "there's a python project at /private/tmp/claude-501/-Users-bryan--aura-live-source/"
    "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/ledger - one of its tests is "
    "failing. read the code, work out why, and tell me exactly which line is wrong "
    "and what it should be."
)


def test_the_task_that_came_back_as_a_number():
    assert requested_arithmetic_result(TASK) is None


@pytest.mark.parametrize(
    "text",
    [
        "version 3.11.2 broke it",
        "call me at 555-1234",
        "the 2015 - 2020 period was rough",
        "see /tmp/a-1-2/b and work out why it failed",
        "the run id is 47f7-8c38, work out what happened",
        "ticket ABC-123-456: figure out the cause",
    ],
)
def test_digits_inside_a_name_are_not_an_expression(text: str):
    assert requested_arithmetic_result(text) is None


@pytest.mark.parametrize(
    "question,expected",
    [
        ("what is 7919 * 6367?", 7919 * 6367),
        ("2+2", 4),
        ("calculate 99 * 99", 9801),
        ("what is 2^31 - 1?", 2**31 - 1),
        ("how much is 12500 + 3750", 16250),
        ("work out 250 / 5", 50),
    ],
)
def test_real_arithmetic_still_computes(question: str, expected):
    """The guard must not cost a single genuine computation."""
    assert requested_arithmetic_result(question) == expected


def test_a_served_answer_replaces_the_whole_reply_so_the_gate_matters():
    """Why a stray match is worse here than elsewhere.

    A computed answer is served IN PLACE OF the draft, so a false match does
    not add a wrong number to a good reply — it deletes the reply.
    """
    from interface.routes.chat import _known_answer_for_this_turn
    from core.conversation.session_scope import set_user_question

    set_user_question(TASK)
    try:
        assert _known_answer_for_this_turn() == ""
    finally:
        set_user_question("")
