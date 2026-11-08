"""What she opened and how she came to it are hers, told in her own person, and her answer does not call her "she".

LIVE 8 November 2026, rehearsal 12: handed "What she opened, and is about to play: ... How her browsing came to it:
... beside her window, she sees it every few seconds", her answer began "Oh, good pick — or rather, *her* good pick"
and asked the person to tell her what happened in it, as if they were the one watching.
"""
from __future__ import annotations

from core.skills import watching_together
from core.skills.watching_along import _without_herself_told_of

SHOWN = ('before it plays. What I opened, and am about to play: Found "CatGPT Goes Wrong" by Simon\'s Cat (4:11). '
         "How my browsing came to it: Search. " + watching_together.AS_IT_PLAYS)


def test_what_she_has_of_it_as_it_plays_is_told_in_her_own_person():
    assert " she " not in f" {watching_together.AS_IT_PLAYS} " and " her " not in f" {watching_together.AS_IT_PLAYS} "
    assert "I see it" in watching_together.AS_IT_PLAYS


def test_a_sentence_calling_her_she_is_left_out_of_her_answer_and_the_rest_is_said():
    said = "Oh, good pick — or rather, *her* good pick. CatGPT Goes Wrong. Four minutes eleven.\n\nI'm curious how it lands."
    assert _without_herself_told_of(said, SHOWN) == "CatGPT Goes Wrong. Four minutes eleven.\n\nI'm curious how it lands."


def test_a_she_on_screen_is_said():
    said = "She opens the laptop and the cat stares at her."
    assert _without_herself_told_of(said, "before it plays. A woman at a desk with a cat.") == said
