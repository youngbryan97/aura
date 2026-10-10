"""She finds what each thing does by doing it, reaches what a step needs when it is not on the screen, and holds each
step to what it was to show.

LIVE 2026-10-10 her plan said to use a roller from a game's device library, and the rollers were behind a tab she had
opened once: she clicked the shapes on the page instead, and tested an empty trap.
"""
from __future__ import annotations

import pytest

from core.agency.where_things_lead import WhereThingsLead
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.cognition.a_plan_to_an_end import Plan, the_screen_answered
from core.cognition.reading_the_rules import Frame

pytestmark = pytest.mark.unit

ROOM = "A ROOM OF DEVICES BUILD A TRAP TO THE CAGE"
CLOSED = ['click "DEVICE LIBRARY"', 'click "TEST TRAP"', 'click "the shape at 50% across, 80% down"']
OPEN = [*CLOSED, 'click "LAUNCHERS"', 'click "ROLLERS"', 'click "CLOSE"']


def _a_map_that_has_opened_the_library_once() -> WhereThingsLead:
    leads = WhereThingsLead()
    leads.looked(CLOSED, ROOM)
    leads.in_order(CLOSED)
    leads.looked(OPEN, ROOM)                                         # the library came up
    leads.acted('click "DEVICE LIBRARY"', changed=True)
    leads.in_order(OPEN)
    leads.looked(CLOSED, ROOM)                                       # and was closed again
    leads.acted('click "CLOSE"', changed=True)
    return leads


def test_what_each_thing_does_is_kept_as_what_it_brought_up_and_took_away():
    leads = _a_map_that_has_opened_the_library_once()
    said = leads.what_things_do()
    assert "DEVICE LIBRARY: brings up CLOSE, LAUNCHERS, ROLLERS" in said
    assert any(line.startswith("CLOSE: takes away") for line in said)
    again = WhereThingsLead.from_memory(leads.as_memory())             # kept between sittings
    assert again.what_things_do() == said


def test_what_a_step_needs_is_reached_by_the_act_that_brings_it_up():
    leads = _a_map_that_has_opened_the_library_once()
    assert leads.toward("a roller") == {'click "DEVICE LIBRARY"': 2.0}
    assert leads.toward("the test trap button") == {}                # it is on the screen
    assert leads.toward("a jetpack") == {}                           # never come across: exploring's to find


def test_the_next_steps_needs_lead_the_move_when_nothing_here_does_the_step():
    from core.skills.screen_pursuit_decision import _the_lesson_first

    leads = _a_map_that_has_opened_the_library_once()
    guide = Guide(place="a place of devices")
    plan = Plan()
    plan.took([Frame("Put a roller at the end of the arrow", act="carry", thing="a roller", where="the end of the arrow")])
    guide.plan = plan
    token = THE_GUIDE.set(guide)
    try:
        valued = _the_lesson_first({'click "DEVICE LIBRARY"': 0.2, 'click "the shape at 50% across, 80% down"': 0.9,
                                    'click "TEST TRAP"': 0.7}, leads)
    finally:
        THE_GUIDE.reset(token)
    assert max(valued, key=valued.get) == 'click "DEVICE LIBRARY"'


def test_a_step_is_held_to_what_it_was_to_show_and_two_misses_make_the_plan_again():
    guide = Guide(place="a place of devices")
    plan = Plan()
    steps = [Frame("Open the device library", act="click things", thing="DEVICE LIBRARY"),
             Frame("Test the trap", act="click things", thing="TEST TRAP"),
             Frame("Watch the cage fall", act="click things", thing="REPLAY")]
    plan.took(steps)
    plan.expects.update({steps[0].key: "the device library opens with types of device",
                         steps[1].key: "the cage drops on Jerry", steps[2].key: "the cage drops on Jerry"})
    guide.plan = plan
    plan.tried('click "DEVICE LIBRARY"', changed=True)
    the_screen_answered(guide, "DEVICE LIBRARY CHOOSE A TYPE OF DEVICE LAUNCHERS ROLLERS")
    assert plan.passed(plan.frames[steps[0].key]) and not plan.missed     # it showed what it was to
    plan.tried('click "TEST TRAP"', changed=True)
    the_screen_answered(guide, "RATS EDIT THE TRAP REPLAY THE TRAP START OVER")
    assert not plan.passed(plan.frames[steps[1].key]) and plan.missed and not guide.plan_again_because
    plan.tried('click "REPLAY"', changed=True)
    the_screen_answered(guide, "RATS EDIT THE TRAP REPLAY THE TRAP START OVER")
    assert "“Test the trap” was to show the cage drops on Jerry; the screen showed: RATS" in guide.plan_again_because


def test_a_step_that_cannot_be_done_as_written_has_the_plan_made_again_around_it():
    guide = Guide(place="a place of devices")
    plan = Plan()
    plan.took([Frame("Click the blue gear", act="click things", thing="blue gear")])
    guide.plan = plan
    for _ in range(2):
        plan.tried('click "BLUE GEAR"', changed=False)
    the_screen_answered(guide, "A ROOM OF DEVICES")
    assert "“Click the blue gear” did nothing when tried" in guide.plan_again_because and not plan.stuck


def test_through_her_whole_move_choice_the_act_that_brings_up_what_the_step_uses_comes_first():
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.skills.screen_pursuit_decision import _first_what_goes_on

    here = WhatWorksHere()
    here.looked_at(CLOSED, ROOM)
    here.looked_at(CLOSED, ROOM)
    here.looked_at(OPEN, ROOM)
    here.tried('click "DEVICE LIBRARY"', changed=True)
    here.looked_at(CLOSED, ROOM)
    here.tried('click "CLOSE"', changed=True)
    here.looked_at(CLOSED, ROOM)
    guide = Guide(place="a place of devices")
    plan = Plan()
    plan.took([Frame("Put a roller at the end of the arrow", act="carry", thing="a roller", where="the end of the arrow")])
    guide.plan = plan
    token = THE_GUIDE.set(guide)
    try:
        valued = _first_what_goes_on(here, {'click "DEVICE LIBRARY"': 0.2, 'click "TEST TRAP"': 0.7,
                                            'click "the shape at 50% across, 80% down"': 0.9})
    finally:
        THE_GUIDE.reset(token)
    assert max(valued, key=valued.get) == 'click "DEVICE LIBRARY"', valued


def test_a_later_step_does_not_lead_while_the_next_can_be_done_or_reached():
    from core.skills.screen_pursuit_decision import _the_lesson_first

    leads = _a_map_that_has_opened_the_library_once()
    guide = Guide(place="a place of devices")
    plan = Plan()
    plan.took([Frame("Put a roller at the end of the arrow", act="carry", thing="a roller", where="the end of the arrow"),
               Frame("Test the trap", act="click things", thing="TEST TRAP")])
    guide.plan = plan
    token = THE_GUIDE.set(guide)
    try:
        valued = _the_lesson_first({'click "DEVICE LIBRARY"': 0.2, 'click "TEST TRAP"': 0.9}, leads)
    finally:
        THE_GUIDE.reset(token)
    assert max(valued, key=valued.get) == 'click "DEVICE LIBRARY"', valued


def test_the_place_things_are_carried_to_is_not_offered_as_a_click():
    from core.agency.what_i_can_do_here import WhatWorksHere

    here = WhatWorksHere()
    here.place_named = "any active square"
    here.carrying_said = True
    screen = ['click "DEVICE LIBRARY"', 'click "any active square"', 'click "the shape at 20% across, 70% down"']
    here.looked_at(screen, ROOM)
    here.looked_at(screen, ROOM)
    offered = here.available()
    assert 'click "any active square"' not in offered
    assert any('to "any active square"' in move for move in offered)


def test_what_a_step_was_to_show_is_held_to_the_screen_in_the_places_own_words():
    """LIVE 2026-10-10 "A list or grid of device types appears, each with a 'choose' option" was judged not shown on a
    screen that read "CHOOSE A TYPE OF DEVICE LAUNCHERS HANGERS ROLLERS CUTTERS"."""
    guide = Guide(place="a place of devices")
    plan = Plan(vocabulary={"device", "library", "type", "choose", "launcher", "cage", "jerry", "trap", "test"})
    step = Frame("Open the device library", act="click things", thing="DEVICE LIBRARY")
    plan.took([step])
    plan.expects[step.key] = "A list or grid of device types appears, each with a 'choose' option"
    guide.plan = plan
    plan.tried('click "DEVICE LIBRARY"', changed=True)
    the_screen_answered(guide, "CLOSE X DEVICE LIBRARY CHOOSE A TYPE OF DEVICE LAUNCHERS HANGERS ROLLERS CUTTERS")
    assert plan.passed(plan.frames[step.key]) and not plan.missed
