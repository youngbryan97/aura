"""A thing whose own words set a player to make something has no winner, and asked to win it she says so.

LIVE 2026-10-07 a character maker ("Create your own KND by selecting your
favorite body parts ... Then click on the GENERATE button to see your
character in action!") was played for four minutes to be won, and reported
"Played, and not won: I could not get further with it."
"""
from __future__ import annotations

import time

import pytest

from core.language.how_a_game_ended import what_it_asks_of_a_player
from core.skills.screen_pursuit_as_it_happens import MADE_NOT_WON, TRYING_OUT_S, PlayingAsItHappens
from core.skills.sovereign_browser_drawing import (
    _PLAY_OVER,
    NOTHING_TO_WIN,
    _address,
    _what_it_is_for,
    played_first,
)

pytestmark = pytest.mark.unit

MAKER = ("Create your own KND by selecting your favorite body parts, accessories and equipment. "
         "Then click on the GENERATE button to see your character in action!")


def test_its_own_words_say_whether_it_is_won_scored_or_made():
    assert what_it_asks_of_a_player(MAKER) == "make"
    assert what_it_asks_of_a_player("Dress up the princess for the ball.") == "make"
    assert what_it_asks_of_a_player("Create your own fighter, then battle to win the tournament.") == "win"
    assert what_it_asks_of_a_player("You have two minutes to collect as many points as possible.") == "score"
    assert what_it_asks_of_a_player("Use the arrow keys to move. Catch the fruit.") == ""


def test_a_thing_for_making_is_over_once_she_has_tried_it_out():
    reflexes = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="win it", ends_at=time.monotonic() + 600)
    reflexes.words = ["KND NUMBUH GENERATOR", MAKER]
    assert not reflexes.run_is_over({"text": MAKER})
    reflexes.for_making_since = time.monotonic() - TRYING_OUT_S - 1
    assert reflexes.run_is_over({"text": MAKER})
    assert reflexes.over_because == MADE_NOT_WON


def test_she_says_what_it_is_for_in_its_own_words_and_ends_it_as_played():
    assert _what_it_is_for(["KND NUMBUH GENERATOR", MAKER]).startswith("Create your own KND")
    url = "https://archive.org/details/a-maker"
    _PLAY_OVER[_address(url)] = NOTHING_TO_WIN
    done = played_first({"url": url}, "play it and win it")
    assert done["done"] and done["why"] == f"Played; {NOTHING_TO_WIN}."
