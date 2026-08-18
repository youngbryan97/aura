"""Obeying "just say yes" must not be treated as failing to answer.

LIVE, 2026-08-18. Told:

    don't acknowledge that rule beyond a yes. just hold it.

the Cortex answered "Yes." — four characters, precisely what was asked for.
The gate rejected it as `too_short_for_user_turn, missing_requested_phrase`,
retried, got "Yes." again, rejected it again, and the person was told "I
couldn't get to an answer I'd stand behind" about an instruction she had
followed exactly, twice.

Two guards should have caught it and neither could. The short-answer exemption
covers arithmetic and GRAMMATICALLY polar questions — prompts opening with
is/are/do/can — and an instruction is neither, so a request FOR brevity was
the one shape that could not license brevity. The 48-character floor below it
then discarded the draft outright.

The worker had already reached the opposite conclusion on the same reasons:
`missing_requested_phrase` sits in its deliverable-residual set precisely
because a dead turn is worse than an imperfectly styled one. The gate
disagreed with it and the gate ran first.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import _should_pass_user_facing_draft_downstream
from core.conversation.surface_disposition import (
    requests_a_brief_answer,
    short_draft_answers_closed_question,
)

LIVE_REQUEST = (
    "good. harder version: from now on, if i ever mention the word 'lantern' in "
    "this conversation, i want you to immediately tell me the number again "
    "without me asking. don't acknowledge that rule beyond a yes. just hold it."
)


def test_the_turn_that_punished_obedience_now_passes():
    assert _should_pass_user_facing_draft_downstream(
        "Yes.",
        {"too_short_for_user_turn", "missing_requested_phrase"},
        user_prompt=LIVE_REQUEST,
    )


def test_a_requested_short_answer_is_a_finished_answer():
    assert short_draft_answers_closed_question("Yes.", LIVE_REQUEST)


@pytest.mark.parametrize(
    "request_text",
    [
        "just say yes or no",
        "answer in one word",
        "just the number please",
        "briefly: what is it",
        "don't elaborate, what is it",
        "no preamble — what is it",
        "in a word, how did it go",
    ],
)
def test_every_way_of_asking_for_brevity_is_recognised(request_text):
    assert requests_a_brief_answer(request_text), request_text


@pytest.mark.parametrize(
    "request_text",
    [
        "tell me everything you know about orcas",
        "walk me through the whole design",
        "what happened yesterday",
    ],
)
def test_an_open_request_does_not_license_a_stub(request_text):
    assert not requests_a_brief_answer(request_text), request_text
    assert not short_draft_answers_closed_question("Yes.", request_text)
    assert not _should_pass_user_facing_draft_downstream(
        "Yes.", {"too_short_for_user_turn"}, user_prompt=request_text
    )


def test_filler_is_still_not_an_answer():
    """"ok" answers nothing even when brevity was requested."""
    assert not short_draft_answers_closed_question("ok", LIVE_REQUEST)


def test_the_existing_exemptions_are_untouched():
    assert short_draft_answers_closed_question("68", "what's 17 times 4?")
    assert short_draft_answers_closed_question("Yes", "are you there?")


def test_an_empty_draft_is_never_served():
    assert not _should_pass_user_facing_draft_downstream(
        "", {"too_short_for_user_turn"}, user_prompt=LIVE_REQUEST
    )
