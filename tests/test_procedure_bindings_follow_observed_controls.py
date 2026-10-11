"""Procedural variable bindings come from owned effects and expire with their context."""
from types import SimpleNamespace

import pytest

from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.cognition.reading_the_rules import Frame, Rules
from core.runtime.skill_contract import PredicateState
from core.skills.screen_pursuit_decision import (
    _reaching_what_the_next_step_needs,
    _the_lesson_first,
)
from core.skills.screen_step_evidence import admissible_step, artifact_ready_for


def test_registered_choice_and_readiness_invariants_are_measured():
    from core.cognition.procedure_binding import _choice_binding_invariant
    from core.skills.screen_step_evidence import _construction_readiness_invariant

    assert _choice_binding_invariant() == ()
    assert _construction_readiness_invariant() == ()


def observation(epoch, labels, surface="editor"):
    return {"text": " ".join(labels), "clickable": [f'click "{label}"' for label in labels],
            "capture_epoch": str(epoch), "capture_at": float(epoch), "surface_id": surface, "bounds": [0, 0, 400, 300]}


def rules_with(*frames):
    return Rules(frames={frame.key: frame for frame in frames})


def selector_rules():
    choose = Frame("Choose the type of component you wish to use", act="click things",
                   thing="the type of component you wish to use", using="choose")
    open_panel = Frame("Click here to open the component library", act="click things",
                       thing="the component library", using="click here to open", order=1)
    place = Frame("Place the component on the workspace", act="carry", thing="component", where="workspace", order=2)
    rules = rules_with(choose, open_panel, place)
    return rules, choose, open_panel, place


def expose_choices(rules, open_panel):
    before = observation(1, ["Component Library", "Run"])
    after = observation(2, ["Component Library", "Run", "Filters", "Sources", "Delete"])
    rules.begin('click "Component Library"', before, step_key=open_panel.key)
    rules.tried('click "Component Library"', True, after=after)
    return after


def test_method_prose_uses_its_named_target_and_never_matches_the_word_here():
    rules, _, open_panel, _ = selector_rules()
    assert rules.step_of('click "Component Library"') is open_panel
    assert rules.step_of('click "Here"') is None


def test_a_bare_input_method_binds_an_ordinary_click_but_cannot_swallow_a_named_later_step():
    method = Frame("CLICK", act="click things", using="CLICK")
    next_frame = Frame("Next", act="click things", thing="Next", order=1)
    rules = rules_with(method, next_frame)
    assert rules.step_of('click "Play"') is method
    assert rules.step_of('click "Next"') is next_frame
    guide = Guide(rules=rules)
    token = THE_GUIDE.set(guide)
    try:
        assert not admissible_step('click "Next"')
    finally:
        THE_GUIDE.reset(token)


def test_opening_a_selector_reaches_its_choice_without_claiming_the_choice_was_done():
    rules, choose, open_panel, _ = selector_rules()
    assert rules.reaches_next('click "Component Library"')
    assert _reaching_what_the_next_step_needs([rules], {'click "Component Library"': 1}, None) == {'click "Component Library"': 1}
    expose_choices(rules, open_panel)
    assert rules.next_step() is choose and not rules.passed(choose)
    assert rules.step_of('click "Filters"') is choose
    assert rules.step_of('click "Delete"') is None
    assert rules.step_of('click "Run"') is None
    assert rules.step_of('click "Component Library"') is None


@pytest.mark.parametrize("damage", ["unchanged", "missing_epoch", "other_surface", "old_time", "missing_pixels_context"])
def test_unmeasured_or_wrong_context_opening_does_not_bind_choices(damage):
    rules, choose, open_panel, _ = selector_rules()
    before, after = observation(1, ["Component Library"]), observation(2, ["Filters", "Sources"])
    changed = True
    if damage == "unchanged":
        changed = False
    elif damage == "missing_epoch":
        after["capture_epoch"] = ""
    elif damage == "other_surface":
        after["surface_id"] = "other editor"
    elif damage == "old_time":
        after["capture_at"] = 0
    else:
        after["bounds"] = []
    rules.begin('click "Component Library"', before, step_key=open_panel.key)
    rules.tried('click "Component Library"', changed, after=after)
    assert rules.step_of('click "Filters"') is None
    assert not rules.passed(choose)


@pytest.mark.parametrize("change", ["hidden", "surface", "viewport", "old_capture", "epoch_reused"])
def test_a_revealed_choice_must_still_be_on_the_same_surface(change):
    rules, _, open_panel, _ = selector_rules()
    current = expose_choices(rules, open_panel)
    if change == "hidden":
        current["clickable"] = ['click "Run"']
    elif change == "surface":
        current["surface_id"] = "other editor"
    elif change == "viewport":
        current["bounds"] = [0, 0, 600, 500]
    elif change == "old_capture":
        current["capture_at"] = 1.
    else:
        current["capture_at"] = 3.
    rules.observe_context(current)
    assert rules.step_of('click "Filters"') is None


def test_an_unrelated_panel_cannot_supply_the_variables_of_a_named_entity():
    rules, _, _, _ = selector_rules()
    unrelated = Frame("Open the preferences panel", act="click things", thing="preferences panel", order=3)
    rules.frames[unrelated.key] = unrelated
    assert not rules.reaches_next('click "preferences panel"')


@pytest.mark.parametrize("label", ["Run", "Test circuit", "Simulate", "Render", "Preview", "Check my answer"])
def test_unmatched_consuming_controls_need_currently_measured_work(label):
    carry = Frame("Place the component", act="carry", thing="component", where="workspace")
    guide = Guide(rules=rules_with(carry), built="Previously placed work needs another look")
    can_do = SimpleNamespace(carried_to={"some old drag": True}, placed_count=lambda: 0)
    token = THE_GUIDE.set(guide)
    try:
        move = f'click "{label}"'
        assert not artifact_ready_for(move, can_do)
        assert not admissible_step(move, can_do=can_do)
        assert _the_lesson_first({move: 1e9, 'click "Library"': 1}, can_do=can_do) == {'click "Library"': 1}
        can_do.placed_count = lambda: 1
        assert artifact_ready_for(move, can_do)
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.parametrize("label", ["Play", "Next", "Edit", "Run"])
def test_an_existing_task_without_a_construction_requirement_can_execute(label):
    token = THE_GUIDE.set(Guide())
    try:
        assert artifact_ready_for(f'click "{label}"')
    finally:
        THE_GUIDE.reset(token)


def test_measured_selection_satisfies_its_own_step_then_leaves_placement_unresolved():
    rules, choose, open_panel, place = selector_rules()
    before = expose_choices(rules, open_panel)
    rules.begin('click "Filters"', before)
    after = observation(3, ["Component Library", "Filter A", "Filter B", "Run"])
    rules.tried('click "Filters"', True, after=after)
    assert rules.states[choose.key] is PredicateState.SATISFIED
    assert rules.next_step() is place
    assert not rules.passed(place)
