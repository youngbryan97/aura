"""A way that would not open is remembered with what it wants, and gone back to when she has it.

Games are built of locks and keys (a door and its key, an area and the ability that opens it), and so are programs.
"""
from __future__ import annotations

import pytest

from core.agency.where_things_lead import WhereThingsLead
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.cognition.locks_she_met import heard_on_a_screen

pytestmark = pytest.mark.unit

HALL = ['click "DOOR"', 'click "STAIRS"']
CELLAR = ['click "CHEST"', 'click "BACK"']


def test_a_locked_way_is_kept_with_what_it_wants_and_a_gain_naming_it_opens_it_to_try():
    guide = Guide(place="a house")
    said = heard_on_a_screen(guide, "This door is locked. You need the brass key.", ["DOOR", "STAIRS"], 'click "DOOR"')
    assert said == ["DOOR won't open yet: it wants the brass key. I'll come back to it."]
    assert "Ways that would not open yet: DOOR (wants the brass key)" in guide.for_thinking()
    assert heard_on_a_screen(guide, "You found a silver coin!", ["CHEST"]) == []          # not what it wanted
    assert heard_on_a_screen(guide, "You found the brass key!", ["CHEST"]) == ["That's what DOOR wanted. Going back to it."]
    assert [lock.at for lock in guide.locks.open_to_try()] == ["DOOR"]
    guide.locks.tried('click "DOOR"', changed=True)
    assert not guide.locks.met                                                          # it opened


def test_a_lock_open_to_try_is_gone_back_to_by_the_screens_she_knows():
    from core.skills.screen_pursuit_decision import _the_lesson_first

    leads = WhereThingsLead()
    leads.looked(HALL, "A HALL WITH A DOOR AND STAIRS")
    leads.in_order(HALL)
    leads.looked(CELLAR, "A CELLAR WITH A CHEST")
    leads.acted('click "STAIRS"', changed=True)
    leads.in_order(CELLAR)
    leads.looked(HALL, "A HALL WITH A DOOR AND STAIRS")
    leads.acted('click "BACK"', changed=True)
    leads.looked(CELLAR, "A CELLAR WITH A CHEST")
    leads.acted('click "STAIRS"', changed=True)
    guide = Guide(place="a house")
    heard_on_a_screen(guide, "This door is locked. You need the brass key.", ["DOOR", "STAIRS"], 'click "DOOR"')
    heard_on_a_screen(guide, "You found the brass key!", ["CHEST", "BACK"])
    token = THE_GUIDE.set(guide)
    try:
        valued = _the_lesson_first({'click "CHEST"': 0.9, 'click "BACK"': 0.2}, leads)
    finally:
        THE_GUIDE.reset(token)
    assert max(valued, key=valued.get) == 'click "BACK"'
