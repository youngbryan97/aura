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
def test_the_names_of_keys_are_words_of_the_place():
    """Offline 2026-10-11 "USE WASD TO MOVE SPACE TO STOMP" was said as "USE … TO MOVE SPACE TO STOMP"."""
    watcher = WhatAWatcherHears()
    said = watcher.heard("It says: USE WASD TO MOVE SPACE TO STOMP, CTRL OR F2 TO DUCK", 0.0)
    assert said is not None and "WASD" in said and "CTRL" in said and "F2" in said


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


@pytest.mark.unit
def test_capitals_read_back_in_mixed_case_are_quoted_as_capitals():
    watcher = WhatAWatcherHears()
    assert watcher.heard('Clicking "no InvenToRY" — to see what it does', 0.0) == 'Clicking "NO INVENTORY" — to see what it does'
    assert watcher.heard('Clicking "Play Now" — it has worked here before', 1.0) == 'Clicking "Play Now" — it has worked here before'


@pytest.mark.unit
def test_in_plain_prose_a_name_and_a_hyphened_word_are_kept_and_a_short_unread_word_is_not_guessed():
    watcher = WhatAWatcherHears()
    said = watcher.heard("It says: Eduardo, Mac and Bloo are raiding the kitchen for a late-night snack. Click to start.", 0.0)
    assert said is not None and "Bloo" in said and "late-night" in said
    said = watcher.heard("It says: Game Over toct Your score: Try Again", 1.0)
    assert said is not None and "tact" not in said


@pytest.mark.unit
def test_stray_marks_and_another_scripts_letters_are_not_quoted():
    watcher = WhatAWatcherHears()
    assert watcher.heard('Clicking "START."" — to see what it does', 0.0) == 'Clicking "START" — to see what it does'
    said = watcher.heard("It says: Aim with your MOUSE (ог and click to sling Rigby into the sky after the fireflies!", 1.0)
    assert said is not None and "ог" not in said and "MOUSE … and click" in said


@pytest.mark.unit
def test_what_she_reads_out_is_not_judged_as_her_own_prose():
    from core.conversation.response_surface_checks import _has_function_word_starvation
    from core.language.what_a_watcher_hears import in_her_own_words

    reading = "It says: NUMBUH: 361,188 Special Talents: Super Hero Strength Hobbies: Snorkeling Equipment: B.O.O.G.S. (Back Off Or Get Slapped)"
    assert in_her_own_words(reading).strip() == "It says:"
    assert not _has_function_word_starvation(reading)
    collapsed = "Introspection signal conformance vector substrate coherence telemetry pipeline aggregation drift resonance lattice manifold"
    assert _has_function_word_starvation(collapsed)
    assert _has_function_word_starvation(f'{collapsed} "and the board says this"')


@pytest.mark.parametrize("line, heard", [
    ("What I'm after: level 3 boss (Boss 03) reaches ST_KILL, triggering the WIN sequence. It starts from main menu, "
     "then character-select screen (CH1, CH2, or CH3), then Level 1 (SCR_LEVEL01).",
     "What I'm after: It starts from main menu, then character-select screen, then Level 1."),
    ("From its code: it's won when The code checks if nLevel == 3 and the boss's mState == ST_KILL; it's lost when "
     "health hits 0.", "From its code: it's lost when health hits 0."),
    ("From its code: it's won when nLevel == 3.", None),
    ("Going left — to see what it does", "Going left — to see what it does"),
    ("I'm the pink girl at the left, moved by the arrows.", "I'm the pink girl at the left, moved by the arrows."),
])
def test_what_a_program_names_things_by_is_not_said_to_the_person_watching(line, heard):
    """LIVE 2026-10-11 "level 3 boss (Boss 03) reaches ST_KILL ... (CH1, CH2, or CH3) ... (SCR_LEVEL01)" and "it's won
    when The code checks if nLevel == 3" were said to the person watching: names out of a program are not words."""
    from core.language.what_a_watcher_hears import WhatAWatcherHears

    assert WhatAWatcherHears().heard(line, now=1.0) == heard


@pytest.mark.unit
def test_a_long_line_is_said_with_what_she_could_not_read_left_out():
    """Offline 2026-10-12 what she made in a thing for making was never told, for a caption quoted in it."""
    watcher = WhatAWatcherHears()
    said = watcher.heard("There's nothing to win or score in this one; it's for making things. What I made shows a "
                         "photo of a dog on a beach under the caption 'Xqzt Bvrk'.", 0.0)
    assert said is not None and "photo of a dog on a beach" in said and "Xqzt" not in said
    assert said.endswith("under the caption words I could not make out.")
    assert watcher.heard('Clicking "trelalts hop" — to see what it does', 1.0) is None


@pytest.mark.unit
def test_a_line_mostly_of_quotes_is_not_said_with_one_left_out():
    watcher = WhatAWatcherHears()
    assert watcher.heard('Carrying "the red cup" to "Xqzt Bvrk" — to see what it does', 0.0) is None


@pytest.mark.unit
def test_a_sentence_is_ended_once():
    watcher = WhatAWatcherHears()
    said = watcher.heard("There's nothing to win in this one; it answered: outlook good... and so is pizza!.", 0.0)
    assert said is not None and said.endswith("pizza!")
    assert watcher.heard("It said: maybe later....", 10.0).endswith("later...")


@pytest.mark.unit
def test_remarks_made_while_playing_come_at_a_human_pace_and_what_happens_is_always_said():
    """Live 2026-10-12 five remarks were said inside three seconds of play."""
    watcher = WhatAWatcherHears()
    assert watcher.heard("That's me: the red kite at the left. It goes where the mouse goes.", 0.0)
    assert watcher.heard("The green apples are worth getting: they add to my count.", 1.0) is None
    assert watcher.heard("Clicking \"PLAY\" — to see what it does", 1.5)
    assert watcher.heard("Round 2: lost. Again.", 2.0)
    assert watcher.heard("The blue birds cost me. Keeping clear of them.", 6.5)


@pytest.mark.unit
def test_how_each_go_came_out_is_news_each_time():
    """Live 2026-10-12 "Run 2 of 3: scored 10" was held back as a line like "Run 1 of 3: scored 190"."""
    watcher = WhatAWatcherHears()
    assert watcher.heard("There's nothing to win in this one, only a score. Run 1 of 3: scored 190. Again.", 0.0)
    assert watcher.heard("There's nothing to win in this one, only a score. Run 2 of 3: scored 10. Again.", 20.0)
    assert watcher.heard("Round 2: lost. Again.", 30.0) and watcher.heard("Round 3: lost. Again.", 40.0)


def test_news_is_said_as_it_happens_whatever_the_pace():
    """LIVE 21 November 2026, asked to tell each move against Stockfish: every move of hers came within four seconds of
    the move it answered and was held back as "a remark too soon after the last line"."""
    watcher = WhatAWatcherHears()
    theirs = "They played their piece that runs straight from e7 to e5."
    mine = "I'll move my piece that leaps two-and-one from g1 to f3. It looks about even."
    assert watcher.heard(theirs, now=100.0, news=True)
    assert watcher.heard(mine, now=101.0, news=True)
    # The same line word for word is still not said twice.
    assert watcher.heard(mine, now=102.0, news=True) is None
    # A remark is still paced.
    assert watcher.heard("That piece looks heavy to me.", now=103.0) is None
