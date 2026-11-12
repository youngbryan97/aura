"""A reply made only of being there, talk about her own reply and a promise is not an answer.

LIVE 12 November 2026, three runs of a question about a subreddit whose page her turn held, each served as her
answer: "I'm here. My response was interrupted — I'll continue from where you left off.", "I'm here. My response got
cut short — I'll continue from wherever you left off in a moment." and "I'm here. My response came through in
fragments — I'll piece together what's actually on the page rather than guess at it."
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import assess_user_facing_reply
from core.conversation.response_surface_checks import _is_promise_without_answer

pytestmark = pytest.mark.unit

_ASKED = "Can you check r/AskHistorians on Reddit and tell me what the top thread is asking right now?"


@pytest.mark.parametrize("reply", [
    "I'm here. My response was interrupted — I'll continue from where you left off.",
    "I'm here. My response got cut short — I'll continue from wherever you left off in a moment.",
    "I'm here. My response came through in fragments — I'll piece together what's actually on the page rather than "
    "guess at it.",
    "I'm here. My response is running long and I've lost the thread of what you actually need from me — tell me where "
    "to pick up.",
])
def test_the_replies_served_live_are_not_answers(reply):
    assert _is_promise_without_answer(_ASKED, reply)
    assessment = assess_user_facing_reply(_ASKED, reply)
    assert not assessment.ok and "promise_without_answer" in assessment.reasons


@pytest.mark.parametrize("reply", [
    "My last answer got cut off. The top thread asks whether tourists could visit Nazi-occupied France.",
    "I'll be honest: the top thread asks whether tourists could visit Nazi-occupied France.",
    "I'm here. The top thread asks whether you could visit occupied France as a tourist.",
])
def test_a_sentence_about_the_question_makes_it_an_answer(reply):
    assert not _is_promise_without_answer(_ASKED, reply)


def test_being_there_answers_a_question_about_what_she_is_doing():
    assert not _is_promise_without_answer("what are you doing?", "I'm here. I'll be with you in a moment.")


def test_a_promise_alone_is_left_to_the_narrower_check():
    assert not _is_promise_without_answer("Ok. Just checking. I'll be back, ok?", "I'll be here.")


def test_the_runtime_notice_of_a_cut_turn_and_her_versions_of_it_stay_out_of_her_history():
    """24 copies of the chat route's cancel notice were logged as her replies on 11 November 2026, and her prompt
    replays her past replies as her own words; on 12 November she answered five questions running with her own
    versions of it."""
    from core.conversation.response_reliability import TURN_CUT_SHORT_REPLY, is_non_answer_repair_floor_reply

    assert is_non_answer_repair_floor_reply(TURN_CUT_SHORT_REPLY)
    assert is_non_answer_repair_floor_reply("I'm here. My response got cut short — I'll continue from wherever you "
                                            "left off in a moment.")
    assert not is_non_answer_repair_floor_reply("The top thread on r/AskHistorians right now is asking how the United "
                                                "States ended up with 36 of 50 gubernatorial elections in midterms.")
