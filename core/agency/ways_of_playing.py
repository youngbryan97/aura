"""Every way she has of playing something: what it asks, the function in her live play that does it, and what shows it.

A way of playing is named for what it asks of anyone, not for a game
(docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md). Each is done by one
function, called from her live play, and shown working by a test. A way with
no function her live play reaches is a way she does not have, however well it
is written: tests/test_every_way_of_playing_is_live.py holds each of these to
all three.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WAYS", "Way"]


@dataclass(frozen=True)
class Way:
    """One way of playing: what it asks, ``done_by`` (module:function), ``called_from`` (the module of her live play
    that calls it) and ``shown_by`` (the test that shows it)."""

    name: str
    asks: str
    done_by: str
    called_from: str
    shown_by: str


WAYS: tuple[Way, ...] = (
    Way("steer", "move a body to meet some things and keep clear of others, by keys or the pointer",
        "core.agency.playing_as_it_happens:play_as_it_happens", "core.skills.screen_pursuit_as_it_happens",
        "tests/test_playing_a_world_that_does_not_wait.py"),
    Way("shoot", "fire or throw at things, aimed",
        "core.agency.playing_as_it_happens:_trigger", "core.agency.playing_as_it_happens",
        "tests/test_steering_and_triggering_use_separate_inputs.py"),
    Way("click things", "click things as they show or cross",
        "core.agency.playing_as_it_happens:_click_things", "core.agency.playing_as_it_happens",
        "tests/test_inside_a_game_a_click_is_a_move.py"),
    Way("send", "press, pull or hold, and let go; how hard and which way found from where each went",
        "core.agency.playing_by_shots:play_by_shots", "core.skills.screen_pursuit_as_it_happens",
        "tests/test_a_thing_sent_is_aimed_by_where_the_last_one_went.py"),
    Way("type", "type what a screen asks for (a name, a question, an answer, something to look for), where it points",
        "core.agency.typing_what_is_asked:type_what_is_asked", "core.skills.screen_pursuit_as_it_happens",
        "tests/test_what_a_screen_asks_to_have_typed_is_typed.py"),
    Way("jump", "a press whose effect plays out over the next moment (a jump, a dash, a dodge), learned by watching and made "
        "when the path it takes her on is clear of what holding her course runs into",
        "core.agency.what_a_press_does:WhatAPressDoes", "core.agency.playing_as_it_happens",
        "tests/test_what_a_press_does_over_time.py"),
    Way("time a press", "where no body answers her keys and something keeps moving, press when it is about to be where a "
        "press has paid, learned from what each press paid by where the thing was",
        "core.agency.when_a_press_pays:WhenAPressPays", "core.agency.playing_as_it_happens",
        "tests/test_a_press_is_timed_to_where_things_are.py"),
    Way("use things", "take a thing by clicking it (it goes elsewhere or is chosen where it is), then use it on another: "
        "a click on it and a click on the other",
        "core.agency.taking_and_using:TakingAndUsing", "core.agency.what_i_can_do_here",
        "tests/test_a_thing_taken_is_used_on_another.py"),
    Way("unseen", "keep out of what reaches her without touching her (what sees, fires or goes off near her): the distance "
        "and side at which each kind has cost her, kept clear of",
        "core.agency.how_far_a_thing_reaches:HowFarThingsReach", "core.agency.playing_as_it_happens",
        "tests/test_how_far_a_thing_reaches.py"),
    Way("remember what was shown", "remember what each place showed when turned over, and turn together two places that were "
        "alike before and showed alike things",
        "core.agency.things_that_go_together:ThingsThatGoTogether", "core.agency.what_i_can_do_here",
        "tests/test_what_was_shown_is_remembered_and_paired.py"),
    Way("switch", "a key after which her keys move another of hers (a team's next member) switches her: the one she left "
        "stays hers to come back to",
        "core.agency.which_one_answers_to_her:WhichIsHers", "core.agency.playing_as_it_happens",
        "tests/test_which_thing_answers_to_her.py"),
    Way("keys shown", "press the keys a screen shows as pictures mid-play, while it shows them: in turn and fast where several are",
        "core.agency.pressing_what_is_shown:KeysShown", "core.agency.playing_as_it_happens",
        "tests/test_keys_a_screen_shows_are_pressed.py"),
    Way("copy a sequence", "do again, in order, what a screen shows: arrows drawn, or places lit one after another",
        "core.agency.doing_again_what_was_shown:do_again", "core.skills.screen_pursuit_as_it_happens",
        "tests/test_what_was_shown_is_done_again_in_its_order.py"),
    Way("a view going by", "a world that scrolls past, in layers",
        "core.perception.how_the_scenery_goes_by:how_the_view_moved", "core.perception.what_moves_in_the_picture",
        "tests/test_scenery_going_by_is_not_things.py"),
    Way("a board in turns", "a board, moves in turns, someone on the other side: looked ahead as far as there is time",
        "core.agency.looking_ahead:look_ahead", "core.skills.screen_pursuit_decision",
        "tests/test_looking_as_far_ahead_as_there_is_time_for.py"),
    Way("a grid's rule", "a grid whose rule is found by moving in it",
        "core.skills.screen_pursuit_decision_reading:the_pixels_show_a_grid", "core.skills.screen_pursuit_decision_reading",
        "tests/test_the_places_are_seen_not_inferred.py"),
    Way("make", "make something, and say what was made and why",
        "core.skills.sovereign_browser_drawing:_what_was_made", "core.skills.sovereign_browser_drawing",
        "tests/test_a_thing_for_making_has_nothing_to_win.py"),
    Way("counters", "read score, lives and time, and what each change meant",
        "core.agency.what_meeting_things_does:readouts_in", "core.agency.what_meeting_things_does",
        "tests/test_what_she_reads_off_a_screen_is_said_in_words.py"),
    Way("take stock", "read the rules; ask what she knows when stuck, failing, unsure or before she begins",
        "core.cognition.taking_stock:take_stock", "core.skills.sovereign_browser_taking_stock",
        "tests/test_she_takes_stock_when_she_keeps_failing.py"),
)
