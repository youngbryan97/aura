"""A screen that asks for a click and names nothing to click is clicked on the picture itself.

LIVE 2026-10-07 a game said "Click your mouse button to start the hamster in
motion. When the hamster lines up with the pillow, click again to launch",
and in four minutes nothing was clicked: the only clicks she had were on
writing.
"""
from __future__ import annotations

import pytest

from core.agency.what_i_can_do_here import (
    THE_PICTURE,
    WhatWorksHere,
    a_click_on,
    clicks_a_screen_asks_for,
)
from core.skills.screen_pursuit_bearings import where_to_click


@pytest.mark.unit
@pytest.mark.parametrize("said", [
    "Click your mouse button to start the hamster in motion.",
    "Click anywhere to continue",
    "click to start",
    "Click the left mouse button to shoot",
])
def test_a_click_on_nothing_named_is_a_click_on_the_picture(said):
    assert clicks_a_screen_asks_for(said, []) == (a_click_on(THE_PICTURE),)


@pytest.mark.unit
def test_a_click_on_a_named_button_is_on_the_button_and_keys_ask_for_no_click():
    assert clicks_a_screen_asks_for("Click on the GENERATE button", [a_click_on("GENERATE")]) == (a_click_on("GENERATE"),)
    assert clicks_a_screen_asks_for("Use the arrow keys to move", []) == ()


@pytest.mark.unit
def test_it_is_offered_and_lands_in_the_middle():
    can_do = WhatWorksHere(told=("up", "down", "left", "right"))
    can_do.asked_for_by("Click your mouse button to start the hamster in motion.", [])
    assert can_do._available()[0] == a_click_on(THE_PICTURE)
    assert where_to_click({"layout": []}, THE_PICTURE) == (0.5, 0.5)


@pytest.mark.unit
def test_a_game_named_as_played_with_the_mouse_offers_the_picture_too():
    can_do = WhatWorksHere(told=("up", "down", "left", "right"))
    can_do.asked_for_by("Use the mouse to aim and toss Bloo", [])
    assert a_click_on(THE_PICTURE) in can_do._available()


@pytest.mark.unit
def test_what_the_rules_said_of_the_mouse_holds_on_a_screen_that_says_nothing():
    can_do = WhatWorksHere(told=("up", "down", "left", "right"))
    can_do.asked_for_by("Click your mouse button to start the hamster in motion.", [])
    can_do.asked_for_by("", [])  # the game's own screen, with no words
    assert can_do.pointer_only and a_click_on(THE_PICTURE) in can_do._available()
    can_do.asked_for_by("Use the arrow keys to move.", [])
    assert not can_do.pointer_only
