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
