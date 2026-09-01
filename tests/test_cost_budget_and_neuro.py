"""What thinking costs, and what a biological name is entitled to claim.

Cards 009, 015, 034, 035, 036, 039, 044, 045, 070, 107-120, 177, 178, 180, 207.
"""
from __future__ import annotations

import pytest

from core.cognition.cognitive_cost import (
    CognitiveBudget,
    CognitiveCost,
    CostWeights,
    ModelTier,
    expected_information_gain,
    reset_controller_for_test,
)
from core.science.neuro_reference import (
    Abstracted,
    Grade,
    Mapping,
    Species,
    get_neuro_reference,
    reset_neuro_reference_for_test,
)

EVERYTHING = tuple(Abstracted)


# ── the cost currency ─────────────────────────────────────────────────────

def test_a_second_costs_more_while_someone_is_waiting():
    act = CognitiveCost(seconds=2.0, tokens=4000, tier=ModelTier.DEEP)
    assert act.under(CostWeights.while_waiting()).total > act.under(CostWeights.idle()).total


def test_the_weights_travel_with_the_reading():
    act = CognitiveCost(seconds=1.0, weights=CostWeights.idle())
    assert act.to_dict()["weights"]["context"] == "idle"


def test_a_cached_procedure_costs_nothing_for_its_tier_and_a_deep_pass_costs_most():
    assert ModelTier.REFLEX.penalty == 0.0
    assert ModelTier.DEEP.penalty > ModelTier.RESIDENT.penalty > ModelTier.SMALL.penalty


def test_methods_become_comparable_because_they_share_a_unit():
    weights = CostWeights.while_waiting()
    retrieval = CognitiveCost(seconds=0.2, tokens=200, tier=ModelTier.SMALL, weights=weights)
    deep = CognitiveCost(seconds=6.0, tokens=8000, tier=ModelTier.DEEP, weights=weights)
    assert deep.total > retrieval.total


# ── the budget ────────────────────────────────────────────────────────────

def test_urgent_and_important_are_not_the_same_number():
    now = 1000.0
    urgent = CognitiveBudget(priority=1.0, durability=2.0, created_at=now)
    important = CognitiveBudget(priority=1.0, durability=600.0, created_at=now)
    assert urgent.priority_at(now + 30) < important.priority_at(now + 30)


def test_an_approaching_deadline_overrides_decay():
    now = 1000.0
    budget = CognitiveBudget(priority=1.0, durability=1.0, deadline=now + 1.5, created_at=now)
    assert budget.priority_at(now + 1.0) > budget.priority_at(now + 1.0 - 0.5) * 0.0
    assert budget.priority_at(now + 2.0) == 0.0


def test_a_child_budget_derives_from_value_uncertainty_and_information():
    parent = CognitiveBudget(priority=1.0, durability=60.0)
    certain = parent.child(uncertainty=0.05, information_gain=0.9)
    uncertain = parent.child(uncertainty=0.9, information_gain=0.9)
    assert uncertain.priority > certain.priority


def test_a_harder_subgoal_gets_less_not_more():
    parent = CognitiveBudget(priority=1.0)
    easy = parent.child(uncertainty=0.8, information_gain=0.8, complexity=1.0)
    hard = parent.child(uncertainty=0.8, information_gain=0.8, complexity=8.0)
    assert hard.priority < easy.priority


def test_a_child_budget_is_strictly_smaller_than_its_parent():
    parent = CognitiveBudget(priority=1.0, durability=60.0, expected_cost=10.0)
    child = parent.child(uncertainty=1.0, information_gain=1.0)
    assert child.priority <= parent.priority
    assert child.durability < parent.durability
    assert child.expected_cost < parent.expected_cost


# ── information gain ──────────────────────────────────────────────────────

def test_an_observation_that_settles_a_coin_flip_is_worth_almost_a_bit():
    assert expected_information_gain([0.5, 0.5], [[0.99, 0.01], [0.01, 0.99]]) > 0.9


def test_an_observation_that_tells_you_nothing_is_worth_nothing():
    assert expected_information_gain([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]]) == pytest.approx(0.0)


def test_looking_when_you_are_already_certain_is_worth_nothing():
    assert expected_information_gain([1.0, 0.0], [[1.0, 0.0]]) == pytest.approx(0.0)


# ── the value of computation ──────────────────────────────────────────────

def test_an_untried_method_gets_one_bounded_trial_rather_than_a_guess():
    controller = reset_controller_for_test()
    decision = controller.should_continue("deep_search", remaining_budget=5.0, expected_gain=0.1)
    assert decision["continue"]
    assert decision["cost_allowed"] == controller.TRIAL_COST
    assert "untried" in decision["reason"]


def test_after_a_trial_a_method_competes_on_measurement():
    controller = reset_controller_for_test()
    controller.observe("deep_search", cost=2.0, gain=0.6)
    controller.observe("retrieval", cost=0.5, gain=0.4)
    ranked = controller.rank(["deep_search", "retrieval"])
    assert ranked[0]["method"] == "retrieval"


def test_an_untried_method_ranks_last_but_is_never_excluded():
    controller = reset_controller_for_test()
    controller.observe("known", cost=1.0, gain=0.9)
    ranked = controller.rank(["known", "novel"])
    assert [r["method"] for r in ranked] == ["known", "novel"]


def test_a_spent_budget_stops_everything_and_says_so():
    controller = reset_controller_for_test()
    decision = controller.should_continue("anything", remaining_budget=0.0, expected_gain=1.0)
    assert not decision["continue"] and decision["reason"] == "budget spent"


def test_a_method_that_has_never_paid_stops_being_chosen():
    controller = reset_controller_for_test()
    for _ in range(5):
        controller.observe("useless", cost=2.0, gain=0.0)
    assert not controller.should_continue("useless", remaining_budget=5.0, expected_gain=0.5)["continue"]


# ── the neuroscience register ─────────────────────────────────────────────

def _mapping(**kw):
    base = dict(
        label="x", module="m", structure="s", species=Species.HUMAN,
        hypothesis="h", grade=Grade.INSPIRED_BY, abstracted=EVERYTHING,
        falsifier="something would show it wrong",
    )
    return Mapping(**{**base, **kw})


def test_a_mapping_with_no_falsifier_is_refused():
    with pytest.raises(ValueError, match="not a claim"):
        _mapping(falsifier="  ")


def test_a_mapping_that_abstracts_nothing_is_refused():
    with pytest.raises(ValueError, match="abstracts something"):
        _mapping(abstracted=())


def test_a_functional_claim_about_an_unspecified_species_is_refused():
    with pytest.raises(ValueError, match="different findings"):
        _mapping(grade=Grade.ANALOGOUS_FUNCTION, species=Species.UNSPECIFIED)


def test_a_connectivity_claim_with_no_source_is_refused():
    with pytest.raises(ValueError, match="no source"):
        _mapping(grade=Grade.CONNECTIVITY_MATCHED, source="")


def test_a_claim_is_limited_by_its_weakest_mapping():
    reference = get_neuro_reference()
    verdict = reference.strongest_supportable_claim(["global_workspace", "hippocampus"])
    assert verdict["grade"] == "inspired_by"
    assert verdict["limited_by"] == "hippocampus"


def test_mixing_species_is_reported_rather_than_ignored():
    reference = get_neuro_reference()
    assert reference.strongest_supportable_claim(["global_workspace", "hippocampus"])["mixes_species"]


def test_an_undeclared_biological_word_licenses_nothing():
    reference = get_neuro_reference()
    verdict = reference.strongest_supportable_claim(["amygdala"])
    assert verdict["grade"] is None
    assert "not a declared mapping" in verdict["licenses"]


def test_a_circuit_query_finds_every_mechanism_attached_to_it():
    reference = get_neuro_reference()
    assert [m.label for m in reference.for_circuit("basal ganglia")] == ["basal_ganglia_selection"]


def test_every_grade_states_what_it_licenses():
    assert len({grade.licenses for grade in Grade}) == len(list(Grade))


def test_most_declared_mappings_are_design_lineage_and_say_so():
    audit = get_neuro_reference().audit()
    assert audit["by_grade"].get("inspired_by", 0) >= audit["by_grade"].get("discriminated", 0)
    assert audit["without_a_competing_hypothesis"] == []


def test_no_mapping_claims_a_neural_prediction_aura_cannot_make():
    for mapping in get_neuro_reference().mappings():
        assert mapping.grade < Grade.DISCRIMINATED, (
            "a DISCRIMINATED grade needs an experiment that ruled out a rival mapping; "
            "none has been run"
        )
