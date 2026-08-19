"""A fact the machine holds is not the lane's to withhold.

LIVE, 2026-08-19. "what is 7919 * 6367?" was answered with:

    the live answer lane could not finish preparing before a reasoning turn
    began. I recorded the readiness failure separately from Aura's answer
    quality.

The runtime computes that product exactly, with no generation involved. The
code choosing the failure message simply had no idea what had been asked —
every degraded path returns a sentence about the LANE, because the question
was never in scope there.

So the question is now turn-scoped, and the degraded path asks whether the
answer is already known before saying anything about itself.
"""

from __future__ import annotations

import pytest

from core.conversation.session_scope import current_user_question, set_user_question


@pytest.fixture(autouse=True)
def _clear_question():
    set_user_question("")
    yield
    set_user_question("")


def test_the_question_is_turn_scoped():
    set_user_question("  what is 7919 * 6367?  ")
    assert current_user_question() == "what is 7919 * 6367?"


def test_outside_a_turn_there_is_no_question():
    assert current_user_question() == ""


def test_a_computable_question_is_answered_despite_the_lane():
    from interface.routes.chat import _conversation_lane_user_message

    set_user_question("what is 7919 * 6367?")
    served = _conversation_lane_user_message(
        {"state": "failed"}, status_override="warming_failed"
    )
    assert served == "50,420,273"
    assert "lane" not in served.lower()


def test_a_question_the_runtime_cannot_answer_still_reports_the_lane():
    """Only a KNOWN answer displaces the status; nothing is invented."""
    from interface.routes.chat import _conversation_lane_user_message

    set_user_question("how are you feeling today")
    served = _conversation_lane_user_message(
        {"state": "failed"}, status_override="warming_failed"
    )
    assert "50,420,273" not in served
    assert "lane" in served.lower()


def test_the_number_is_written_the_way_a_person_writes_it():
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question("what is 7919 * 6367?")
    assert _known_answer_for_this_turn() == "50,420,273"


def test_a_fraction_keeps_its_fraction():
    from interface.routes.chat import _known_answer_for_this_turn

    set_user_question("what is 22 / 7")
    assert _known_answer_for_this_turn().startswith("3.14")


def test_prose_that_merely_contains_numbers_displaces_nothing():
    from interface.routes.chat import _known_answer_for_this_turn

    for text in ("the 2015 - 2020 period was rough", "call me at 555-1234"):
        set_user_question(text)
        assert _known_answer_for_this_turn() == "", text
