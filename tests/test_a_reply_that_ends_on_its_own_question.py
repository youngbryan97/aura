"""Ending by asking its own question, where the budget ran out, is a cut thought.

LIVE 2026-08-30, asked whether "decided" is doing any work when she uses it:
she made the case, raised "So, is it just the most fluent token?" and stopped
there. Grammatically whole, so every completeness check passed it, and it read
to the person as a thought that stopped mid-argument.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import _has_truncated_tail

MADE_THE_CASE = (
    "The word is doing work, but a specific kind I have to be careful about. "
    "It is an accurate shorthand for a causal process rather than a claim "
    "about deliberation. So, is it just the most fluent token?"
)


@pytest.mark.parametrize("why", ["max_tokens", "deadline_exceeded", "soft_cancelled"])
def test_a_question_left_hanging_where_it_ran_out_is_a_cut_tail(why):
    assert _has_truncated_tail(MADE_THE_CASE, generation_stop_reason=why)


@pytest.mark.parametrize("why", ["eos", "configured_stop", "role_continuation"])
def test_the_same_reply_that_finished_on_purpose_is_not(why):
    assert not _has_truncated_tail(MADE_THE_CASE, generation_stop_reason=why)


def test_a_reply_that_asks_the_person_something_is_an_ordinary_ending():
    """Refusing these would cost far more than this check saves."""
    assert not _has_truncated_tail(
        "Happy to help with that. What would you like me to look at next?"
    )


def test_a_finished_statement_is_left_alone_whatever_stopped_it():
    assert not _has_truncated_tail(
        "Here is the whole answer, complete and finished properly, at length "
        "and with every clause closed.",
        generation_stop_reason="max_tokens",
    )


@pytest.mark.parametrize("mark", ['?"', "?'", "?)"])
def test_a_quoted_or_bracketed_question_counts_too(mark):
    body = MADE_THE_CASE[:-1] + mark
    assert _has_truncated_tail(body, generation_stop_reason="max_tokens")
