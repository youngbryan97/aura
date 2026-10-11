"""The observed result closes the delivered obligation before a dependent input."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.putting_things_in_place import a_carry_of
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.cognition.a_plan_to_an_end import Plan
from core.cognition.reading_the_rules import Frame, Rules
from core.perception.observed_transfer import (
    ScreenReading,
    bind_transfer,
    finish_delivery,
    make_snapshot,
)
from core.runtime.skill_contract import PredicateOperator, PredicateState, SemanticPredicate
from core.skills.screen_pursuit_decision import _the_lesson_first
from core.skills.screen_pursuit_looking import _what_she_has_built
from core.skills.screen_step_evidence import (
    abandon_bound_step,
    admissible_step,
    bind_step,
    screen_evidence,
    settle_step,
)

pytestmark = pytest.mark.unit


def _observation(at, *, moved=False, surface="editor", other_change=False):
    image = np.full((240, 320, 3), 224, np.uint8)
    rows, cols = np.indices((32, 32))
    part = np.stack(((rows * 9 + cols * 3) % 251, (rows * 3 + cols * 11) % 241,
                     ((cols // 8 + rows // 8) % 2) * 160 + 30), axis=2).astype(np.uint8)
    x, y = (200, 130) if moved else (20, 30)
    image[y:y + 32, x:x + 32] = part
    if other_change:
        image[10:20, 100:150] = 0
    return ScreenReading({"surface_id": surface, "owner": surface, "capture_at": at,
                          "_capture_epoch": str(at), "bounds": [0, 0, 320, 240], "settled": True,
                          "text": "status changed" if other_change else "",
                          "shapes": [{"text": "component", "x": x / 320, "y": y / 240,
                                      "width": .1, "height": 32 / 240, "shape": True}]},
                         picture_snapshot=make_snapshot(image))


def _run(before):
    carry = Frame("Place the component", act="carry", thing="component", where="destination")
    test = Frame("Assess the result", act="click things", thing="Assess", order=1)
    guide = Guide()
    guide.rules = Rules(frames={carry.key: carry, test.key: test})
    guide.plan = Plan(frames={carry.key: carry, test.key: test})
    move = a_carry_of("component", "destination")
    run = SimpleNamespace(key=move, pending={"observed_before": screen_evidence(before),
                                           "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=move))})
    bound = bind_transfer(move, before, source="component", destination="destination",
                          source_at=(36 / 320, 46 / 240), destination_at=(216 / 320, 146 / 240))
    run.pending["transfer_intent"] = finish_delivery(bound, started=1.1, completed=1.2, delivered=True)
    return guide, run, carry, test


def test_actual_placement_advances_before_the_next_action_and_only_once():
    before, after = _observation(1), _observation(2, moved=True)
    guide, run, carry, test = _run(before)
    here = WhatWorksHere()
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        assert settle_step(run.pending, after, (), here)
        assert guide.plan.next_step() is test
        assert guide.rules.passed(carry)
        here.observe_placements(after)
        assert here.placed_count() == 1
        _what_she_has_built(here)
        assert guide.built.startswith("1 part put in place")
        here.tried(run.key, True, transfer=run.pending["transfer_receipt"])
        assert here.placed_count() == 1
    finally:
        THE_GUIDE.reset(token)


def test_animation_after_delivered_drag_does_not_authorize_a_test():
    before, after = _observation(1), _observation(2, other_change=True)
    guide, run, carry, _ = _run(before)
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        assert settle_step(run.pending, after, (), WhatWorksHere())
        assert not guide.plan.passed(carry)
        assert _the_lesson_first({a_click_on("Assess"): 1_000_000}) == {}
    finally:
        THE_GUIDE.reset(token)


def test_missing_measurement_gets_bounded_reobservation_without_false_completion():
    before = _observation(1)
    guide, run, carry, _ = _run(before)
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        missing = dict(_observation(2, moved=True))  # public copy has deliberately lost image custody
        assert not settle_step(run.pending, missing, ())
        assert guide.plan.pending_step is not None
        assert settle_step(run.pending, dict(_observation(3, moved=True)), ())
        assert guide.plan.states[carry.key] is PredicateState.UNKNOWN
        assert not guide.plan.passed(carry)
    finally:
        THE_GUIDE.reset(token)


def test_a_replacement_plan_cannot_receive_the_old_plans_action():
    before, after = _observation(1), _observation(2, moved=True)
    guide, run, carry, _ = _run(before)
    old_plan = guide.plan
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        guide.plan = Plan(frames=dict(old_plan.frames))
        run.pending["dispatch_completed"] = 1.2
        assert settle_step(run.pending, after, ())
        assert old_plan.passed(carry)
        assert not guide.plan.passed(carry)
    finally:
        THE_GUIDE.reset(token)


def test_retained_work_revalidates_after_resume_and_observed_removal():
    guide, run, _, _ = _run(_observation(1))
    here = WhatWorksHere()
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        settle_step(run.pending, _observation(2, moved=True), (), here)
        held = here.as_memory()
        resumed = WhatWorksHere.from_memory(held)
        assert resumed.placed_count() == 0  # retained evidence requires a current observation
        resumed.observe_placements(_observation(3, moved=True))
        assert resumed.placed_count() == 1
        resumed.observe_placements(_observation(4))
        assert resumed.placed_count() == 0
        _what_she_has_built(resumed)
        assert guide.built == ""
    finally:
        THE_GUIDE.reset(token)


def test_inventory_transition_is_measured_against_the_pre_click_labels():
    here = WhatWorksHere(using_said=True)
    before = [a_click_on("document"), a_click_on("open folder")]
    after = [a_click_on("held document"), a_click_on("open folder")]
    here.looked_at(before)
    here.looked_at(after)
    here.tried(a_click_on("document"), True, before={"clickable": before}, after={"clickable": after})
    assert here.taken == {"document": "held document"}


@pytest.mark.parametrize("after_epoch,after_at,surface", [("", 2, "editor"), ("2", .5, "editor"), ("2", 2, "other")])
def test_navigation_does_not_complete_from_stale_or_wrong_surface(after_epoch, after_at, surface):
    frame = Frame("Open the panel", act="click things", thing="Open")
    guide = Guide()
    guide.rules = Rules(frames={frame.key: frame})
    run = SimpleNamespace(key=a_click_on("Open"), pending={
        "observed_before": screen_evidence(_observation(1)),
        "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=a_click_on("Open"))),
    })
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        after = _observation(2, other_change=True, surface=surface)
        after.update(_capture_epoch=after_epoch, capture_at=after_at)
        assert settle_step(run.pending, after, ())
        assert guide.rules.states[frame.key] is PredicateState.UNKNOWN
    finally:
        THE_GUIDE.reset(token)


def test_an_ambient_response_veto_does_not_complete_navigation():
    frame = Frame("Open the panel", act="click things", thing="Open")
    guide = Guide()
    guide.rules = Rules(frames={frame.key: frame})
    run = SimpleNamespace(key=a_click_on("Open"), pending={
        "observed_before": screen_evidence(_observation(1)),
        "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=a_click_on("Open"))),
    })
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        after = _observation(2, other_change=True)
        after["answered"] = False
        settle_step(run.pending, after, ())
        assert not guide.rules.passed(frame)
    finally:
        THE_GUIDE.reset(token)


def test_binding_failure_releases_only_the_attempts_already_bound():
    guide, run, carry, _ = _run(_observation(1))

    class FailingPlan:
        pending_step = None

        def begin(self, *args, **kwargs):
            raise ValueError("the contract could not be bound")

    guide.plan = FailingPlan()
    token = THE_GUIDE.set(guide)
    try:
        with pytest.raises(ValueError, match="could not be bound"):
            bind_step(run)
        assert guide.rules.pending_step is None
        assert guide.rules.states[carry.key] is PredicateState.UNKNOWN
        assert not guide.rules.passed(carry)
    finally:
        THE_GUIDE.reset(token)


def test_delivery_cleanup_keeps_requirements_unresolved_and_allows_fresh_binding():
    guide, run, carry, _ = _run(_observation(1))
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        abandon_bound_step(run.pending, run.key)
        for procedure in (guide.rules, guide.plan):
            assert procedure.pending_step is None
            assert procedure.states[carry.key] is PredicateState.UNKNOWN
            assert not procedure.passed(carry)
        bind_step(run)
        assert guide.plan.pending_step is not None
    finally:
        THE_GUIDE.reset(token)


def test_unmet_requirement_is_enforced_even_for_a_single_or_high_score_candidate():
    guide, _, carry, test = _run(_observation(1))
    token = THE_GUIDE.set(guide)
    try:
        assert not admissible_step(a_click_on("Assess"))
        assert not _the_lesson_first({a_click_on("Assess"): 1e12})
        guide.rules.done.add(carry.key)
        guide.plan.done.add(carry.key)
        assert admissible_step(a_click_on("Assess"))
        assert guide.rules.next_step() is test
    finally:
        THE_GUIDE.reset(token)


def test_measured_placement_cannot_erase_a_required_termination_predicate():
    guide, run, carry, _ = _run(_observation(1))
    guide.plan.effects[carry.key] = [SemanticPredicate(
        "construction_complete", "after.parts", PredicateOperator.GREATER_THAN_OR_EQUAL, 3)]
    token = THE_GUIDE.set(guide)
    try:
        bind_step(run)
        run.pending["dispatch_completed"] = 1.2
        assert settle_step(run.pending, _observation(2, moved=True), ())
        assert not guide.plan.passed(carry)
        assert guide.plan.states[carry.key] is PredicateState.UNKNOWN
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_dispatch_exception_resolves_bindings_without_automatic_replay(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from core.perception import where_am_i
    from core.skills import screen_pursuit_acting as acting

    frame = Frame("Open the panel", act="click things", thing="Open")
    guide = Guide()
    guide.rules = Rules(frames={frame.key: frame})
    key = a_click_on("Open")
    run = SimpleNamespace(
        key=key, observation=_observation(1), about_to={"key": key}, anchor={"app": "test"}, at_rest={}, busy=Mock(),
        expected={"took": 1, "after": None}, follow_on=[], goal="open", in_flight=Mock(),
        laid_out=None, lattice=Mock(), made=None, moves=[], narrate=False, pacing={"brief": True},
        pending={"observed_before": screen_evidence(_observation(1)),
                 "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=key))},
        responds={"state": None, "lattice": None}, target_app="test", world=Mock(),
    )
    click = AsyncMock(side_effect=OSError("input delivery uncertain"))
    monkeypatch.setattr(where_am_i, "where_am_i", lambda *args, **kwargs: SimpleNamespace(the_thing_is_here=True))
    monkeypatch.setattr(acting, "_click_what_she_named", click)
    monkeypatch.setattr(acting, "_say_intent", Mock())
    token = THE_GUIDE.set(guide)
    try:
        with pytest.raises(OSError, match="uncertain"):
            await acting.carry_out_the_move(run)
        assert click.await_count == 1
        assert guide.rules.pending_step is None
        assert guide.rules.states[frame.key] is PredicateState.UNKNOWN
        assert not guide.rules.passed(frame)
        assert run.pending["deliberation"] is None
        assert run.moves == []
    finally:
        THE_GUIDE.reset(token)
