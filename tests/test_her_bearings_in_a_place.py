"""In any place she asks: why am I here, what does it tell me, what stands out, what can I act on, should I.

LIVE 2026-10-08 she clicked a game's "MORE GAMES" and its credits as readily as
its START; a person knows the first takes them away from what they came for.
"""
from __future__ import annotations

import pytest

from core.cognition.her_bearings import Place, instructions_in, leads_away, take_bearings


@pytest.mark.unit
def test_what_tells_her_what_to_do_is_found():
    said = "Help Lazlo spruce things up. Use the arrow keys to move. Lazlo is a scout. Press Z to jump over puddles."
    assert instructions_in(said) == ["Help Lazlo spruce things up.", "Use the arrow keys to move.", "Press Z to jump over puddles."]


@pytest.mark.unit
def test_what_would_take_her_away_is_left_alone_unless_it_is_what_she_was_asked_for():
    assert leads_away("MORE GAMES") and leads_away("Download our app") and not leads_away("START")
    assert not leads_away("Download", "Download the report from this page")


@pytest.mark.unit
def test_her_bearings_name_what_stands_out_and_what_she_leaves_alone():
    place = Place(asked="Play this game and win it.", says="Use the mouse to aim.", labels=("START", "MORE GAMES", "CREDITS"),
                  stands_out=("START",), keys=())
    bearings = take_bearings(place)
    assert bearings.why == "I was asked to play this game and win it."
    assert bearings.sticks_out == ("START",) and bearings.leaving_alone == ("MORE GAMES",)
    assert "MORE GAMES" not in bearings.can_act_on and bearings.should_act
    assert bearings.said() == "“START” stands out; “MORE GAMES” would take me away from it, so I leave it be."


@pytest.mark.unit
def test_a_place_that_says_nothing_of_what_to_do_leaves_that_open():
    bearings = take_bearings(Place(asked="Fill in the form.", labels=("the shape at the middle",)))
    assert "how is this done here" in bearings.open_questions and not bearings.instructions
