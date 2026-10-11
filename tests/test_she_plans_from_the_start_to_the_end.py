"""She plans from where things start to what she is there to do: what she has to work with, what each does and works
with, and the steps that join them, each a use of one thing for another; followed, and made again when a try fails.

LIVE 2026-10-10 she clicked a device library's tabs and the shapes on its page about forty times and tested an empty
trap. The player in a video of the same game put a device at the end of the last one's arrow, turned it toward the
cage, added the next, tested the whole and mended the link that broke.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Guide
from core.cognition.a_plan_to_an_end import ask_for_a_plan, plan_again
from core.cognition.reading_the_rules import Frame

pytestmark = pytest.mark.unit

ON_SCREEN = ["DEVICE LIBRARY", "ROLLERS", "LAUNCHERS", "TURN", "TEST TRAP", "the mouse trap", "the cage"]
PLAN = {
    "end": "the cage drops on Jerry", "start": "the mouse trap",
    "means": [{"name": "a roller", "does": "rolls the way its arrow points", "works_with": "the end of the arrow"},
              {"name": "a jetpack", "does": "flies to the cage"},                        # not in this place
              {"name": "a launcher", "does": "throws what lands on it", "on_its_own": False}],
    "steps": [{"step": "Open the device library", "act": "click things", "thing": "DEVICE LIBRARY",
               "for_what": "to see what I can use"},
              {"step": "Put a roller at the end of the mouse trap's arrow", "act": "carry", "thing": "a roller",
               "where": "the end of the arrow", "for_what": "to carry the action toward the cage"},
              {"step": "Strap on the jetpack", "act": "click things", "thing": "jetpack"},  # invented: dropped
              {"step": "Turn it toward the cage", "act": "click things", "using": "TURN", "for_what": "so it rolls on"},
              {"step": "Test the trap", "act": "click things", "thing": "TEST TRAP", "for_what": "to see it work"}],
}


def _guide() -> Guide:
    guide = Guide(place="a place of devices")
    guide.take_in(SCREEN, ["Tom needs your help to build a trap to the cage that will catch Jerry!"])
    guide.goals.append("build a trap to the cage")
    guide.rules.hear(["Place the device at the end of another device's arrow."])
    guide.rules.took([Frame("Place the device at the end of another device's arrow.", act="carry",
                            thing="the device", where="the end of another device's arrow")])
    return guide


def _asker(answer, asked):
    async def ask(prompt, schema, most):
        asked.append(prompt)
        return schema.model_validate(answer)

    return ask


def test_a_plan_is_made_of_what_the_place_has_and_says_each_step_as_a_use():
    guide, asked, said = _guide(), [], []

    async def run():
        assert ask_for_a_plan(guide, _asker(PLAN, asked), on_screen=ON_SCREEN, tell=said.append)
        await guide.planning

    asyncio.run(run())
    plan = guide.plan
    assert [m.name for m in plan.means] == ["a roller", "a launcher"]                  # nothing invented
    assert [f.sentence for f in plan.steps()] == ["Open the device library",
                                                  "Put a roller at the end of the mouse trap's arrow",
                                                  "Turn it toward the cage", "Test the trap"]
    assert said and said[0].startswith("What I'm after: the cage drops on Jerry. It starts from the mouse trap. My plan: "
                                       "using DEVICE LIBRARY to see what I can use; then using a roller to carry the action")
    thinking = guide.for_thinking()
    assert "What I can use: a roller (rolls the way its arrow points; works with the end of the arrow)" in thinking
    assert "a launcher (throws what lands on it; not on its own)" in thinking
    assert "My plan, in order: → Open the device library" in thinking
    assert "On the screen now: DEVICE LIBRARY" in asked[0]


def test_the_plans_next_step_leads_the_move_and_a_failed_try_makes_it_again():
    from core.skills.screen_pursuit_decision import _the_lesson_first

    guide, asked = _guide(), []

    async def first():
        ask_for_a_plan(guide, _asker(PLAN, asked), on_screen=ON_SCREEN)
        await guide.planning

    asyncio.run(first())
    token = THE_GUIDE.set(guide)
    try:
        valued = _the_lesson_first({'click "DEVICE LIBRARY"': 0.3, 'click "the shape at 20% across, 40% down"': 0.9,
                                    'click "TEST TRAP"': 0.8})
    finally:
        THE_GUIDE.reset(token)
    assert max(valued, key=valued.get) == 'click "DEVICE LIBRARY"'
    guide.plan.tried('click "DEVICE LIBRARY"', changed=True)
    assert guide.plan.next_step().sentence == "Put a roller at the end of the mouse trap's arrow"
    assert not ask_for_a_plan(guide, _asker(PLAN, asked), on_screen=ON_SCREEN)        # one plan until a try says
    plan_again(guide, "it ended on “EDIT THE TRAP START OVER” (lost)")

    async def again():
        assert ask_for_a_plan(guide, _asker(PLAN, asked), on_screen=ON_SCREEN)
        await guide.planning

    asyncio.run(again())
    assert "Their last try: it ended on “EDIT THE TRAP START OVER” (lost)" in asked[-1]
    assert guide.plan.because.startswith("it ended on")


def test_no_plan_is_asked_for_without_an_end_to_aim_at():
    assert not ask_for_a_plan(Guide(place="somewhere"), _asker(PLAN, []), on_screen=ON_SCREEN)


def test_what_she_has_built_is_part_of_what_the_plan_is_made_from():
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.runtime.skill_contract import PredicateState
    from core.skills.screen_pursuit_looking import _what_she_has_built

    guide, asked = _guide(), []
    here = WhatWorksHere()
    # The guide consumes current measured occupancy, not a screen-change flag.
    here.placement_receipts = {"placed": SimpleNamespace(effect_bbox=(0.62, 0.41, 0.04, 0.06))}
    here.placement_states = {"placed": PredicateState.SATISFIED}
    here.chain_ends_at = (0.62, 0.41)
    token = THE_GUIDE.set(guide)
    try:
        _what_she_has_built(here)
    finally:
        THE_GUIDE.reset(token)
    assert guide.built == "1 part put in place, the last at 62% across and 41% down"

    async def run():
        ask_for_a_plan(guide, _asker(PLAN, asked), on_screen=ON_SCREEN)
        await guide.planning

    asyncio.run(run())
    assert "What they have built so far: 1 part put in place" in asked[0]
