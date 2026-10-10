"""A part put into a chain is turned until what it sends out points on toward where the chain must go.

LIVE 2026-10-10 a player of a trap-building game turned each device toward the cage after putting it down; she never
turned one.
"""
from __future__ import annotations

import pytest

from core.agency.aiming_what_was_placed import a_turn_wanted, points_on, turns_a_thing
from core.agency.what_i_can_do_here import WhatWorksHere
from core.cognition.a_guide_to_a_place import Guide
from core.cognition.reading_the_rules import Frame

pytestmark = pytest.mark.unit


def test_an_output_on_the_goals_side_points_on_and_one_on_the_far_side_does_not():
    assert points_on((0.5, 0.5), (0.6, 0.5), (0.9, 0.4)) is True
    assert points_on((0.5, 0.5), (0.4, 0.5), (0.9, 0.4)) is False
    assert points_on((0.5, 0.5), None, (0.9, 0.4)) is None
    assert turns_a_thing("TURN") and turns_a_thing("Rotate left") and not turns_a_thing("DELETE")


def test_a_part_pointing_away_is_turned_and_one_turned_round_is_left(monkeypatch):
    from core.perception import where_the_words_point as eyes

    found = {"the end of another device's arrow": (0.40, 0.50), "the cage": (0.15, 0.20)}
    monkeypatch.setattr(eyes.PLACES, "where", lambda place: found.get(place))
    guide = Guide(place="a place of devices")
    goal = "Tom needs your help to build a trap to the cage."
    guide.rules.hear([goal])
    guide.rules.took([Frame(goal, act="a chain", thing="a trap", where="the cage", is_what_it_is_for=True)])
    here = WhatWorksHere()
    here.place_named, here.chain_ends_at = "the end of another device's arrow", (0.30, 0.50)
    offered = ['click "TURN"', 'click "DELETE"', 'click "TEST TRAP"']
    assert a_turn_wanted(here, guide, offered) == 'click "TURN"'             # its arrow ends away from the cage
    found["the end of another device's arrow"] = (0.25, 0.40)
    assert a_turn_wanted(here, guide, offered) == ""                         # turned: it points on
    found["the end of another device's arrow"] = (0.40, 0.50)
    here.turned_since_placed = 4
    assert a_turn_wanted(here, guide, offered) == ""                         # turned round already: left as it is
    here.chain_ends_at = None
    assert a_turn_wanted(here, guide, offered) == ""                         # nothing put down
