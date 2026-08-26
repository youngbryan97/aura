"""The approach question waits for the screen to say what the task is.

Measured live on 2026-08-26: asked on the first cycle, when the only thing
she had seen was a whole browser window — the board, the score, the tabs, the
bookmarks, an "Ask Gemini" button and a copyright line — she answered by
reading the page back, three runs in a row.
"""

from __future__ import annotations

from core.agency.standing_strategy import RECONSIDER_AFTER
from core.skills.screen_pursuit import LANGUAGE_EVERY, _ask_again_after


def test_a_first_plan_waits_longer_than_a_later_one():
    assert _ask_again_after(-1) > _ask_again_after(0)


def test_the_first_horizon_is_the_one_past_which_an_approach_is_a_habit():
    assert _ask_again_after(-1) == RECONSIDER_AFTER


def test_a_plan_already_asked_for_is_revisited_on_the_language_horizon():
    assert _ask_again_after(0) == LANGUAGE_EVERY
    assert _ask_again_after(37) == LANGUAGE_EVERY


def test_the_first_question_is_never_put_on_the_opening_cycle():
    # asked_at starts at -1 and no move has been made.
    assert 0 - (-1) < _ask_again_after(-1)
