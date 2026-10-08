"""A game's rules are read as instructions: an act, a thing, a polarity, a control."""
from __future__ import annotations

import pytest

from core.agency.what_meeting_things_does import AVOID, CLICK, MEET, SHOOT
from core.agency.what_the_rules_said import WhatTheRulesSaid
from core.language.what_an_instruction_asks import what_the_words_ask

pytestmark = pytest.mark.unit


def _asks(text):
    return [(i.act, i.thing, i.forbidden, i.control) for i in what_the_words_ask(text)]


def test_a_forbidden_thing_after_an_asked_one():
    assert _asks("Catch the fruit. Not the bombs.") == [("get", ("fruit",), False, ""), ("get", ("bombs",), True, "")]


def test_the_control_that_does_it():
    asked = _asks("Dodge food by pressing the space bar.")
    assert asked == [("keep clear", ("food",), False, "space")]


def test_a_thing_said_to_cost_is_kept_clear_of():
    assert ("keep clear", ("rocks",), False, "") in _asks("Rocks cost a life.")


def test_keeping_a_thing_from_getting_past_is_meeting_it():
    assert ("get", ("ball",), False, "") in _asks("Keep the ball from getting past you.")


@pytest.mark.parametrize(
    ("words", "colour", "stance"),
    [
        ("Click the orange targets. Do not hit the green ones.", "orange", CLICK),
        ("Click the orange targets. Do not hit the green ones.", "green", AVOID),
        ("Pick up the coins. The red ones hurt.", "red", AVOID),
        ("Catch the yellow stars.", "yellow", MEET),
        ("Shoot the purple invaders with space.", "purple", SHOOT),
    ],
)
def test_a_colour_ties_words_to_a_kind(words, colour, stance):
    assert WhatTheRulesSaid.read(words).stance_for_colour(colour) == stance


def test_keys_get_their_roles():
    rules = WhatTheRulesSaid.read("Take aim with your mouse and click to throw. Dodge food by pressing the space bar.")
    assert rules.dodge_keys == ("space",)
    assert rules.a_click_is_a_shot


@pytest.mark.parametrize("words", ["Click to start.", "Click to play again.", "Click for help."])
def test_a_lifecycle_click_has_no_firing_role(words):
    assert not WhatTheRulesSaid.read(words).a_click_is_a_shot


@pytest.mark.parametrize("words", [
    "Aim with your mouse; click to throw at red targets.",
    "Click to fire at red targets.",
])
def test_observed_steering_resolves_a_mouse_trigger_into_shooting(words):
    rules = WhatTheRulesSaid.read(words)
    assert rules.stance_for_colour("red") == CLICK
    assert rules.stance_for_colour("red", pointer_steers=True) == SHOOT
    assert rules.a_click_is_a_shot


def test_explicit_target_clicks_and_forbidden_colours_keep_their_roles():
    rules = WhatTheRulesSaid.read("Move with the mouse. Click the orange targets. Do not hit the green ones.")
    assert rules.stance_for_colour("orange", pointer_steers=True) == CLICK
    assert rules.stance_for_colour("green", pointer_steers=True) == AVOID
