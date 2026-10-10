"""What she takes in when she takes stock, and from the place's own words: an instruction whatever its verb, a question
about doing what it asks in general beside one about the thing by name, the time her model needs while the thing waits,
and the thing she is told to drive taken for hers. Nothing here is a game."""
from __future__ import annotations

import pytest

from core.cognition.her_bearings import instructions_in
from core.cognition.taking_stock import BEFORE, FAILING, HELPED_BEFORE, Heard, Situation, questions_for, what_to_take
from core.cognition.what_this_place_is import hers_named_by

pytestmark = pytest.mark.unit


def test_an_instruction_is_read_whatever_its_verb_after_the_screens_furniture():
    assert instructions_in("LEVEL 1 Park the red car") == ["Park the red car"]
    assert instructions_in("Rescue all the puppies before time runs out!") == ["Rescue all the puppies before time runs out"]
    assert instructions_in("TIME 01:11 PARKED 0/1 HITS LEFT 99 LEVEL 1") == []
    assert instructions_in("Mojo Jojo has kidnapped all the dogs of Townsville.") == []


def test_she_asks_how_what_the_place_asks_is_done_in_general_beside_the_thing_by_name():
    failing = Situation("Some Old Game", "play it until you win", ("LEVEL 1 Park the red car",), "TRY AGAIN?", FAILING, kind="game")
    asked = questions_for(failing)
    assert asked[0] == "how to win Some Old Game" and "how to park the red car in a game" in asked
    before = questions_for(Situation("Some Old Game", "play it", ("Park the red car",), "", BEFORE, kind="game"))
    assert "how to park the red car in a game" in before


def test_what_helped_before_comes_first_whatever_else_matches_the_questions_better():
    situation = Situation("Some Old Game", "play it", ("Use the arrow keys to steer the boat.",), "", FAILING, kind="game")
    heard = [Heard("the web", "", "Use the arrow keys to steer the boat and catch every firefly before dawn."),
             Heard(HELPED_BEFORE, "", "Press left and right in turn, fast, when caught.")]
    assert what_to_take(heard, situation, questions_for(situation))[0].source == HELPED_BEFORE


def test_what_she_is_told_to_drive_is_hers_by_the_name_it_was_seen_as():
    assert hers_named_by("LEVEL 1 Park the red car", ["the red car", "a green car", "a sign"]) == "the red car"
    assert hers_named_by("Steer your ship through the asteroids", ["ship", "asteroid"]) == "ship"
    assert hers_named_by("Park the red car") == "red car"
    assert hers_named_by("Use the arrow keys to move") == ""


def test_the_web_asked_by_programs_includes_the_encyclopedias_own_search():
    from core.skills.looking_it_up import ENGINES, _wikipedia

    assert ENGINES[0][0] == "wikipedia"
    hits = _wikipedia('{"query": {"search": [{"title": "Parallel parking"}]}}')
    assert hits == [("Parallel parking", "https://en.wikipedia.org/wiki/Parallel_parking")]


def test_an_act_is_read_in_any_of_its_forms_and_split_from_its_last_word():
    from core.language.what_an_instruction_asks import what_the_words_ask

    def acts(text):
        return [(i.act, i.thing) for i in what_the_words_ask(text)]

    assert acts("Earn points by knocking other tops out of the stadium.") == [("hit", ("other", "tops"))]
    assert acts("Knock the other tops out of the stadium.") == [("hit", ("other", "tops"))]
    assert acts("Pick the coins up before they vanish.") == [("get", ("coins",))]
    assert acts("Avoid the flying saucers.") == [("keep clear", ("flying", "saucers"))]
    assert acts("Keep the ball from getting past you.") == [("get", ("ball",))]


def test_a_legend_of_controls_is_read_as_controls_by_their_keys_and_not_as_what_the_place_is_for():
    from core.agency.the_controls_a_game_names import a_legend_of_controls, without_a_legend
    from core.cognition.a_guide_to_a_place import Guide

    said = "Forward Reverse Turn left Turn right Quit Music on/off Bump into the cars and make them park before time runs out."
    assert [key for _name, key in a_legend_of_controls(said) if key] == ["up", "down", "left", "right"]
    assert without_a_legend(said).startswith("Bump into the cars")
    assert a_legend_of_controls("Use the left and right arrow keys to move") == []
    guide = Guide(place="a car park")
    guide.take_in("what the screen says", [said])
    assert {key: control.act for key, control in guide.controls.items()} == {
        "up": "forward", "down": "reverse", "left": "turn left", "right": "turn right"}
    assert all("Forward Reverse" not in line for line in [*guide.goals, *guide.win, *guide.lose])


def test_a_word_misread_into_another_word_is_mended_to_the_one_the_place_showed_before():
    from core.language.what_a_watcher_hears import WhatAWatcherHears

    watcher = WhatAWatcherHears()
    assert watcher.heard("It says: YOU SAVED 1 ITEMS. YOUR SCORE: 120", now=0) == "It says: YOU SAVED 1 ITEMS. YOUR SCORE: 120"
    assert watcher.heard("It says: YOU SAYED 3 ITEMS. YOUR SCORE: 300", now=1000) == "It says: YOU SAVED 3 ITEMS. YOUR SCORE: 300"


def test_a_bar_is_not_named_by_writing_she_could_not_make_out():
    from core.agency.what_she_has_left import what_a_bar_measures

    class _Bar:
        start, end, row = 200, 300, 10

        def where(self, wide, tall):
            return 0.6, 0.05, 0.9, 0.07

    misread = [{"text": "Bl, Eld n Eldy", "center_x": 0.5, "center_y": 0.06}]
    legible = [{"text": "Energy", "center_x": 0.5, "center_y": 0.06}]
    assert what_a_bar_measures(_Bar(), misread, 320, 240)[0] == ""
    assert what_a_bar_measures(_Bar(), legible, 320, 240)[0] == "Energy"


def test_a_place_whose_words_name_only_the_mouse_is_not_offered_keys_its_program_alone_listens_for():
    from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
    from core.cognition.a_guide_to_a_place import PROGRAM, THE_GUIDE, TOLD, Control, Guide

    guide = Guide(place="a music maker")
    guide.take_in(TOLD, ["Hold the mouse button and drag a beat onto a dancer."])
    guide.controls["space"] = Control("space", "", PROGRAM, 0.0)
    token = THE_GUIDE.set(guide)
    try:
        here = WhatWorksHere(told=("up", "down", "left", "right"))
        here.asked_for_by("")
        here.looked_at([a_click_on("PLAY")])
        here.looked_at([a_click_on("PLAY")])
        assert here.pointer_only
    finally:
        THE_GUIDE.reset(token)
