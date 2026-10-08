"""A watcher hears each kind of line once a while, and writing quoted only where she could read it.

LIVE 2026-10-07 a game's chat filled with "Going up — to see what it does",
"Going down — to see what it does" every few seconds, and quoted misread
writing as words: "Clicking "toct"", "It says: ....... 5°".
"""
from __future__ import annotations

import pytest

from core.language.what_a_watcher_hears import WhatAWatcherHears, shape_of, where_it_is


@pytest.mark.unit
def test_moves_of_one_kind_are_one_line_a_while():
    watcher = WhatAWatcherHears()
    assert watcher.heard("Going up — to see what it does", 0.0)
    assert watcher.heard("Going down — to see what it does", 5.0) is None
    assert watcher.heard("Going down — down has not been tried yet", 6.0)
    assert watcher.heard("Going left — to see what it does", 60.0)


@pytest.mark.unit
def test_misread_writing_is_not_quoted():
    watcher = WhatAWatcherHears()
    assert watcher.heard('Clicking "trelalts hop" — to see what it does', 0.0) is None
    assert watcher.heard("It says: ....... 5°", 1.0) is None
    assert watcher.heard('Clicking "PLAY" — to see what it does', 2.0) == 'Clicking "PLAY" — to see what it does'


@pytest.mark.unit
def test_a_near_miss_is_mended_and_a_word_not_made_out_left_out():
    watcher = WhatAWatcherHears()
    said = watcher.heard("It says: Game Over toct Your score: Try Again, Leuel 2", 0.0)
    assert said is not None and "Level 2" in said and "Game Over" in said and "Try Again" in said


@pytest.mark.unit
def test_names_she_was_given_are_words():
    watcher = WhatAWatcherHears()
    assert watcher.heard('Clicking "Bloo" — to see what it does', 0.0) is None
    watcher.knows("Foster’s Home for Imaginary Friends: Bloo's Bounce")
    assert watcher.heard('Clicking "Bloo" — to see what it does', 100.0)


@pytest.mark.unit
def test_two_readings_of_different_screens_are_both_said():
    watcher = WhatAWatcherHears()
    assert watcher.heard("It says: Use the arrow keys to guide the lander onto the platform.", 0.0)
    assert watcher.heard("It says: Click your mouse button to start the hamster in motion.", 3.0)


@pytest.mark.unit
def test_a_place_given_as_shares_is_said_as_a_place():
    watcher = WhatAWatcherHears()
    said = watcher.heard('Clicking "the shape at 90% across, 15% down" — to see what it does', 0.0)
    assert said == "Clicking the shape at the top right — to see what it does"
    assert where_it_is(0.5, 0.5) == "the middle"
    assert shape_of("Round 2: lost. Again.") == shape_of("Round 3: lost. Again.")


@pytest.mark.unit
def test_a_reason_that_names_the_move_again_says_it():
    watcher = WhatAWatcherHears()
    assert watcher.heard("Going up — up is the only thing available", 0.0) == "Going up — it is the only thing available"
    assert watcher.heard('Clicking "PLAY" — click "PLAY" has worked here before', 1.0) == 'Clicking "PLAY" — it has worked here before'
