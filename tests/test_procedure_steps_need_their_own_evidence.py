"""Procedures advance on their requirements' observed effects, across unrelated interfaces."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.cognition.a_plan_to_an_end import Plan
from core.cognition.reading_the_rules import (
    EvidenceSnapshot,
    Frame,
    Rules,
    _procedure_completion_is_measured,
)
from core.runtime.skill_contract import PredicateOperator, PredicateState, SemanticPredicate

pytestmark = pytest.mark.unit


def _rules(*frames: Frame) -> Rules:
    return Rules(frames={frame.key: frame for frame in frames})


def _plan(expected: str = "", *, act: str = "click things", thing: str = "Open") -> tuple[Plan, Frame]:
    frame = Frame("Prepare the workspace", act=act, thing=thing)
    plan = Plan()
    plan.took([frame])
    if expected:
        plan.expects[frame.key] = expected
    return plan, frame


def test_a_later_answer_does_not_prove_an_earlier_requirement():
    first = Frame("Choose a document", act="click things", thing="Browse")
    last = Frame("Send the document", act="click things", thing="Send", order=1)
    rules = _rules(first, last)
    rules.tried('click "Send"', changed=True)
    assert rules.passed(last)
    assert not rules.passed(first)
    assert rules.next_step() is first


def test_a_confirmed_repeat_allows_the_later_control_to_proceed():
    repeat = Frame("Add a sample to the tray", act="carry", thing="sample", where="tray", again=True)
    later = Frame("Analyze the samples", act="click things", thing="Analyze", order=1)
    plan = Plan(frames={f.key: f for f in (repeat, later)})
    move = 'drag "sample" to "tray"'
    effect = SemanticPredicate("sample placed", "after.placement.state", PredicateOperator.EQUALS, "verified")
    plan.tried(move, True, before={"text": "empty tray"},
               after={"text": "tray", "placement": {"state": "verified"}}, effects=(effect,))
    assert plan.passed(repeat)
    assert plan.next_step() is later
    assert plan.the_step_to_do(['click "Analyze"', move]) is later
    assert plan.repeat_opportunities([move]) == [repeat]


def test_an_explicit_repeat_termination_predicate_stays_mandatory_until_satisfied():
    repeat = Frame("Add samples until two are present", act="carry", thing="sample", where="tray", again=True)
    later = Frame("Analyze", act="click things", thing="Analyze", order=1)
    rules = _rules(repeat, later)
    effect = SemanticPredicate("enough samples", "after.samples", PredicateOperator.MIN_COUNT, 2)
    move = 'drag "sample" to "tray"'
    rules.tried(move, True, before={"samples": []}, after={"samples": ["red"]}, effects=(effect,))
    assert rules.next_step() is repeat
    assert rules.the_step_to_do(['click "Analyze"']) is None
    assert rules.repeat_opportunities([move]) == []
    rules.tried(move, True, before={"samples": ["red"]},
                after={"samples": ["red", "blue"]}, effects=(effect,))
    assert rules.passed(repeat)
    assert rules.next_step() is later


def test_a_later_failure_and_a_failed_optional_repeat_keep_the_confirmed_obligation():
    repeat = Frame("Add a sample to the tray", act="carry", thing="sample", where="tray", again=True)
    later = Frame("Analyze", act="click things", thing="Analyze", order=1)
    plan = Plan(frames={f.key: f for f in (repeat, later)})
    move = 'drag "sample" to "tray"'
    plan.tried(move, True)
    for _ in range(2):
        plan.tried('click "Analyze"', False)
    assert plan.stuck
    plan.tried(move, False)
    assert plan.receipts[-1].attempt.optional
    assert plan.repeat_states[repeat.key] == PredicateState.UNSATISFIED
    assert plan.states[repeat.key] == PredicateState.SATISFIED
    assert plan.passed(repeat)
    assert plan.next_step() is later
    assert plan.stuck
    plan.tried(move, True)
    assert plan.stuck  # Optional success cannot hide a mandatory step's repair requirement.


def test_measured_misses_remain_requirements_for_local_repair():
    first = Frame("Open the component panel", act="click things", thing="Components")
    last = Frame("Run the design", act="click things", thing="Run", order=1)
    rules = _rules(first, last)
    for _ in range(3):
        rules.tried('click "Components"', changed=False)
    assert rules.steps() == [first, last]
    assert rules.next_step() is first
    assert first.key in rules.failed_steps
    assert rules.the_step_to_do(['click "Run"']) is None
    assert rules.the_step_to_do(['click "Run"'], reaching=True) is None


def test_a_short_unnamed_control_binds_to_its_own_label():
    rules = _rules(Frame("Click to open", act="click things"), Frame("Next", act="click things", order=1))
    assert rules.step_of('click "CLICK TO OPEN"').sentence == "Click to open"
    rules.tried('click "CLICK TO OPEN"', changed=True)
    assert rules.step_of('click "NEXT"').sentence == "Next"


def test_an_unnamed_long_instruction_does_not_bind_to_any_shared_word():
    rules = _rules(Frame("Choose the option needed before opening the next panel", act="click things"))
    assert rules.step_of('click "NEXT"') is None


@pytest.mark.parametrize("move", ['drag "blue document" to "Archive"', 'drag "red document" to "Trash"',
                                 'use "red document" on "Archive"', 'click "red document"'])
def test_a_transfer_binds_kind_source_and_destination(move):
    rules = _rules(Frame("Put the red document in Archive", act="carry", thing="red document", where="Archive"))
    assert rules.step_of(move) is None
    assert rules.step_of('drag "red document" to "Archive"') is not None


def test_a_position_only_source_can_bind_to_a_known_destination():
    rules = _rules(Frame("Put the part in the highlighted square", act="carry", thing="the part",
                         where="the highlighted square"))
    assert rules.step_of('drag "the shape at 20% across, 30% down" to "the highlighted square"') is not None
    assert rules.step_of('drag "Settings" to "the highlighted square"') is None


def test_a_named_control_is_separate_from_the_thing_it_operates_on():
    rules = _rules(Frame("Rotate the selected image using Rotate", act="click things",
                         thing="the selected image", using="Rotate"))
    assert rules.step_of('click "ROTATE"') is not None
    assert rules.step_of('click "the selected image"') is None


def test_a_deictic_control_uses_the_named_target_instead_of_matching_the_word_here():
    rules = _rules(Frame("Click here to open the panel", act="click things", thing="the panel", using="here"))
    assert rules.step_of('click "PANEL"') is not None
    assert rules.step_of('click "HERE"') is None


def test_causal_change_does_not_complete_a_planned_effect_before_observation():
    plan, frame = _plan("workspace ready")
    plan.begin('click "Open"', {"text": "workspace closed"})
    assert plan.tried('click "Open"', changed=True) is None
    assert frame.key not in plan.done
    assert plan.states[frame.key] == PredicateState.UNKNOWN
    assert plan.the_step_to_do(['click "Open"']) is None
    with pytest.raises(RuntimeError, match="awaiting"):
        plan.begin('click "Open"', {"text": "a different screen"})
    assert not plan.saw({"text": "workspace ready"})
    assert plan.receipts[-1].state == PredicateState.SATISFIED
    assert plan.passed(frame)


@pytest.mark.parametrize("shown", ["device library", "DEVICE OPTIONS", "device is not placed"])
def test_related_nouns_and_negative_text_do_not_verify_placement(shown):
    plan, frame = _plan("device placed")
    plan.tried('click "Open"', True, before={"text": "device library"}, after={"text": shown})
    assert not plan.passed(frame)
    assert plan.receipts[-1].state == PredicateState.UNSATISFIED


@pytest.mark.parametrize("shown", ["door not open", "not door open", "the door remains shut"])
def test_expected_phrase_polarity_is_preserved(shown):
    plan, frame = _plan("door open")
    plan.tried('click "Open"', True, before={"text": "door closed"}, after={"text": shown})
    assert plan.receipts[-1].state == PredicateState.UNSATISFIED
    assert frame.key not in plan.done


def test_an_effect_already_in_the_baseline_is_not_caused_by_a_later_click():
    plan, frame = _plan("workspace ready")
    plan.tried('click "Open"', True, before={"text": "workspace ready"},
               after={"text": "workspace ready; clock 9"})
    assert plan.receipts[-1].state == PredicateState.UNKNOWN
    assert not plan.passed(frame)


def test_a_legacy_text_only_observation_cannot_claim_an_owned_delta():
    plan, frame = _plan("workspace ready")
    plan.tried('click "Open"', True)
    plan.saw("workspace ready")
    assert plan.receipts[-1].state == PredicateState.UNKNOWN
    assert not plan.passed(frame)


def test_unobservable_expectation_does_not_inherit_delivery_success():
    plan, frame = _plan("...")
    plan.tried('click "Open"', True, before={"text": "closed"}, after={"text": "ready"})
    assert plan.receipts[-1].state == PredicateState.UNKNOWN
    assert frame.key not in plan.done


@pytest.mark.parametrize("placement,expected", [
    ({"state": "verified"}, PredicateState.SATISFIED),
    ({"state": "rejected"}, PredicateState.UNSATISFIED),
    ({}, PredicateState.UNKNOWN),
])
def test_transfer_effect_uses_the_shared_deterministic_predicate_contract(placement, expected):
    plan, frame = _plan("the photo was put where requested")
    effect = SemanticPredicate("placement", "after.placement.state", PredicateOperator.EQUALS, "verified")
    plan.begin('click "Open"', {"text": "canvas"}, effects=(effect,))
    receipt = plan.tried('click "Open"', True, after={"text": "canvas", "placement": placement})
    assert receipt.state == expected
    assert plan.passed(frame) is (expected == PredicateState.SATISFIED)


def test_required_unknown_effect_cannot_be_outvoted_by_a_satisfied_effect():
    plan, frame = _plan()
    effects = (SemanticPredicate("opened", "after.panel.open", expected=True),
               SemanticPredicate("loaded", "after.document.loaded", expected=True))
    plan.tried('click "Open"', True, before={"text": "editor"}, after={"panel": {"open": True}},
               effects=effects)
    assert plan.receipts[-1].state == PredicateState.UNKNOWN
    assert not plan.passed(frame)


def test_a_false_answer_does_not_complete_even_when_effect_text_is_present():
    plan, frame = _plan("workspace ready")
    plan.tried('click "Open"', False, before={"text": "closed"}, after={"text": "workspace ready"})
    assert plan.receipts[-1].state == PredicateState.UNSATISFIED
    assert not plan.passed(frame)


def test_receipts_bind_immutable_baselines_effects_and_results():
    plan, frame = _plan()
    before = {"text": "canvas", "parts": []}
    after = {"text": "canvas", "placement": {"state": "verified"}}
    effect = {"predicate_id": "placement", "evidence_path": "after.placement.state",
              "operator": "equals", "expected": "verified"}
    attempt = plan.begin('click "Open"', before, effects=(effect,))
    before["parts"].append("invented")
    effect["expected"] = "rejected"
    receipt = plan.tried('click "Open"', True, after=after)
    after["placement"]["state"] = "rejected"
    assert receipt.before.evidence["parts"] == []
    assert receipt.after.evidence["placement"]["state"] == "verified"
    assert receipt.state == PredicateState.SATISFIED
    copy = receipt.after.evidence
    copy["placement"]["state"] = "different"
    assert receipt.after.evidence["placement"]["state"] == "verified"
    with pytest.raises(FrozenInstanceError):
        attempt.move = "another action"
    assert receipt.step_key == frame.key


def test_a_result_from_a_different_surface_is_unknown():
    plan, frame = _plan("workspace ready")
    plan.tried('click "Open"', True, before={"surface_id": "left", "text": "closed"},
               after={"surface_id": "right", "text": "workspace ready"})
    assert plan.receipts[-1].state == PredicateState.UNKNOWN
    assert not plan.passed(frame)


def test_unknown_and_failed_requirements_request_repair_without_disappearing():
    for measurement in (None, {"state": "rejected"}):
        plan, frame = _plan()
        effect = SemanticPredicate("placement", "after.placement.state", PredicateOperator.EQUALS, "verified")
        for _ in range(2):
            after = {"text": "canvas"}
            if measurement is not None:
                after["placement"] = measurement
            plan.tried('click "Open"', True, before={"text": "canvas"}, after=after, effects=(effect,))
        assert plan.stuck
        assert plan.next_step().key == frame.key
        assert frame.key not in plan.done
        if measurement is None:
            assert frame.key not in plan.failed_steps
            assert frame.key not in plan.unanswered_steps
        else:
            assert frame.key in plan.failed_steps


def test_a_confirmed_local_retry_clears_repair_state():
    plan, frame = _plan("workspace ready")
    for _ in range(2):
        plan.tried('click "Open"', False)
    assert plan.stuck
    plan.tried('click "Open"', True, before={"text": "closed"}, after={"text": "workspace ready"})
    assert plan.passed(frame)
    assert not plan.stuck
    assert not plan.missed
    assert frame.key not in plan.failed_steps


def test_plain_causal_navigation_keeps_existing_callers_working():
    plan, frame = _plan()
    receipt = plan.tried('click "Open"', True)
    assert receipt.state == PredicateState.SATISFIED
    assert plan.passed(frame)


def test_nonfinite_evidence_is_rejected_instead_of_published():
    with pytest.raises(ValueError):
        EvidenceSnapshot.of({"coordinate": float("nan")})


def test_registered_procedure_evidence_invariant():
    assert _procedure_completion_is_measured()
