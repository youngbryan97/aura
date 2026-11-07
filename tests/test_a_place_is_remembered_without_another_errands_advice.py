"""A place is remembered for what it is, without what another errand did there.

LIVE 7 November 2026: her notes on YouTube read "to make progress there you Search for 'Allen Pan' to find his channel",
from an errand of October; asked to find a cooking video and then a stand-up clip, she searched "Allen Pan's cooking
videos" and "Allen Pan stand up comedy".
"""

from __future__ import annotations

import pytest

from core.skills.sovereign_browser_understanding import _what_the_place_is

pytestmark = pytest.mark.unit

HOME = ("www.youtube.com is YouTube homepage, not signed in. No feed yet.; to make progress there you Search for "
        "'Allen Pan' to find his channel, then browse his videos")
RESULTS = "www.youtube.com is YouTube search results for 'Allen Pan'. Shows his channel (2.35M subs)"
STAND_UP = "Let's find a funny stand-up comedy clip on YouTube, something under ten minutes, and watch it together."


def test_what_a_place_is_carries_and_another_errands_advice_does_not():
    assert _what_the_place_is(HOME, STAND_UP) == "www.youtube.com is YouTube homepage, not signed in. No feed yet."


def test_a_place_remembered_by_another_errands_search_is_left_out():
    assert _what_the_place_is(RESULTS, STAND_UP) == ""


def test_it_is_kept_when_this_errand_names_it():
    assert "Allen Pan" in _what_the_place_is(RESULTS, "Now let's do Allen Pan. Go to his channel and pick one.")


def test_a_title_she_names_in_a_decision_is_not_her_biography_and_speech_is_still_checked():
    # LIVE 7 November 2026: her decision naming the clip "I Grew Up In ..." was cut off as a claim to a human childhood.
    from core.conversation.ontology_grounding import detect_unsupported_embodiment_claim as claim

    page = "results: I Grew Up In | Jeff Allen, 5:12; Menopause Is Worse Than PMS"
    decision = "I Grew Up In is 5 minutes and the funniest title here"
    assert claim(decision, prompt=page, names_what_she_was_shown=True).ok
    assert not claim(decision, prompt=page).ok  # said to a person, it is checked as before
    assert not claim("I grew up in a small town by a lake.", prompt="", names_what_she_was_shown=True).ok
