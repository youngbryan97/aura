"""One vector algebra, and a skill that has to prove it survived the crossing.

Cards 046, 047, 048, 053, 074, 075, 076, 078, 079, 082, 083, 084, 085, 087,
088, 094, 116, 158, 183, 217.
"""
from __future__ import annotations

import math

import pytest

from core.cognition.cognitive_vector import (
    DEFAULT_DIMENSION,
    CleanupMemory,
    Projection,
    bind,
    capacity,
    random_vector,
    representational_similarity,
    reset_vector_registry_for_test,
    similarity,
    subspace_angle,
    superpose,
    unbind,
)
from core.cognition.dual_knowledge import (
    CreditAssignment,
    Form,
    measure_equivalence,
    reset_knowledge_registry_for_test,
)


# ── the algebra ───────────────────────────────────────────────────────────

def test_binding_produces_something_like_neither_operand():
    a, b = random_vector(256, seed=1), random_vector(256, seed=2)
    bound = bind(a, b)
    assert abs(similarity(bound, a)) < 0.2
    assert abs(similarity(bound, b)) < 0.2


def test_unbinding_recovers_the_other_operand():
    a, b = random_vector(256, seed=1), random_vector(256, seed=2)
    assert similarity(unbind(bind(a, b), a), b) > 0.6


def test_a_superposition_resembles_everything_in_it():
    parts = [random_vector(256, seed=i) for i in range(4)]
    bundle = superpose(*parts)
    assert all(similarity(bundle, p) > 0.3 for p in parts)


def test_binding_across_dimensions_is_refused():
    with pytest.raises(ValueError, match="different dimension"):
        bind(random_vector(64), random_vector(128))


def test_novel_role_filler_pairs_work_without_retraining_anything():
    registry = reset_vector_registry_for_test(dimension=512)
    colour, shape = random_vector(512, seed=11), random_vector(512, seed=12)
    red, square = registry.concept("red"), registry.concept("square")
    for name in ("blue", "circle", "heavy", "loud"):
        registry.concept(name)
    scene = superpose(bind(colour, red), bind(shape, square))
    assert registry.cleanup.resolve(unbind(scene, colour))["name"] == "red"
    assert registry.cleanup.resolve(unbind(scene, shape))["name"] == "square"


# ── cleanup refuses rather than guessing ──────────────────────────────────

def test_cleanup_refuses_noise_instead_of_naming_its_best_match():
    registry = reset_vector_registry_for_test(dimension=256)
    for name in ("red", "square", "loud"):
        registry.concept(name)
    result = registry.cleanup.resolve(random_vector(256, seed=999))
    assert result["name"] is None
    assert "threshold" in result["reason"]


def test_cleanup_refuses_when_two_concepts_are_within_the_margin():
    memory = CleanupMemory(threshold=0.1, margin=0.2)
    base = random_vector(256, seed=1)
    memory.add("a", base)
    memory.add("b", base)
    result = memory.resolve(base)
    assert result["name"] is None and result["reason"] == "two concepts within the margin"


def test_an_empty_cleanup_memory_says_so():
    assert CleanupMemory().resolve(random_vector(64))["reason"] == "cleanup memory is empty"


# ── capacity is measured, not assumed ─────────────────────────────────────

def test_capacity_falls_as_bindings_are_added():
    few = capacity(256, pairs=4, distractors=100)["recovery_rate"]
    many = capacity(256, pairs=24, distractors=100)["recovery_rate"]
    assert few > many


def test_a_bigger_space_holds_more_bindings():
    small = capacity(128, pairs=12, distractors=200)["recovery_rate"]
    large = capacity(512, pairs=12, distractors=200)["recovery_rate"]
    assert large > small


def test_the_default_dimension_is_justified_by_that_curve():
    assert capacity(DEFAULT_DIMENSION, pairs=8, distractors=200)["recovery_rate"] == 1.0


# ── projections carry their measured fidelity ─────────────────────────────

def test_an_unmeasured_projection_is_not_treated_as_lossless():
    assert not Projection(substrate="world_model", dimension=512).lossless_enough


def test_a_measured_round_trip_becomes_a_number():
    registry = reset_vector_registry_for_test(dimension=128)
    identity = registry.measure_projection("perfect", lambda v: v, lambda v: v, samples=5)
    assert identity.fidelity == pytest.approx(1.0)
    assert identity.lossless_enough


def test_a_lossy_projection_reports_what_it_lost():
    registry = reset_vector_registry_for_test(dimension=128)
    lossy = registry.measure_projection(
        "eight_dims", lambda v: v[:8], lambda v: list(v) + [0.0] * 120, samples=5
    )
    assert lossy.fidelity < 0.5
    assert not lossy.lossless_enough


def test_a_projection_that_raises_has_fidelity_zero_rather_than_an_exception():
    registry = reset_vector_registry_for_test(dimension=64)
    broken = registry.measure_projection(
        "broken", lambda v: (_ for _ in ()).throw(RuntimeError("nope")), lambda v: v, samples=3
    )
    assert broken.fidelity == 0.0 and not broken.invertible


# ── representational diagnostics ──────────────────────────────────────────

def test_a_space_is_maximally_similar_to_itself():
    space = [random_vector(128, seed=i) for i in range(8)]
    assert representational_similarity(space, space) == pytest.approx(1.0)


def test_two_unrelated_spaces_agree_about_nothing():
    left = [random_vector(128, seed=i) for i in range(8)]
    right = [random_vector(128, seed=100 + i) for i in range(8)]
    assert abs(representational_similarity(left, right)) < 0.6


def test_a_space_compared_to_a_shorter_one_returns_zero_rather_than_guessing():
    assert representational_similarity([random_vector(8)], []) == 0.0


def test_subspace_angle_is_zero_for_a_space_against_itself():
    space = [random_vector(128, seed=i) for i in range(4)]
    assert subspace_angle(space, space) == pytest.approx(0.0, abs=1e-6)


def test_subspace_angle_is_a_right_angle_when_one_side_is_empty():
    assert subspace_angle([], [random_vector(8)]) == pytest.approx(math.pi / 2)


# ── the conversion cycle ──────────────────────────────────────────────────

def _policy(x):
    return (x * 3 + 1) if x % 2 else x // 2


def test_a_conversion_that_agrees_installs():
    registry = reset_knowledge_registry_for_test()
    registry.hold("halving", Form.IMPLICIT, _policy, benefit=1.0)
    result = registry.convert(
        "halving", source=Form.IMPLICIT, target=Form.EXPLICIT,
        build=lambda _p: _policy, cases=list(range(50)), benefit=0.5,
    )
    assert result.installed and result.equivalence.agreement == 1.0


def test_a_conversion_that_disagrees_does_not_install_and_keeps_the_cases():
    registry = reset_knowledge_registry_for_test()
    registry.hold("halving", Form.IMPLICIT, _policy)
    result = registry.convert(
        "halving", source=Form.IMPLICIT, target=Form.NEURAL,
        build=lambda _p: (lambda x: x // 2), cases=list(range(50)),
    )
    assert not result.installed
    assert result.equivalence.disagreements, "the failing cases are the specification"
    assert registry.get("halving").forms.get(Form.NEURAL) is None


def test_a_conversion_from_a_form_that_does_not_exist_is_refused():
    registry = reset_knowledge_registry_for_test()
    result = registry.convert(
        "nothing", source=Form.IMPLICIT, target=Form.EXPLICIT,
        build=lambda p: p, cases=[1],
    )
    assert not result.installed and "no implicit form" in result.reason


def test_a_conversion_that_raises_is_a_refusal_not_a_crash():
    registry = reset_knowledge_registry_for_test()
    registry.hold("x", Form.EXPLICIT, _policy)
    result = registry.convert(
        "x", source=Form.EXPLICIT, target=Form.NEURAL,
        build=lambda _p: (_ for _ in ()).throw(RuntimeError("compiler down")),
        cases=[1],
    )
    assert not result.installed and "RuntimeError" in result.reason


def test_a_skill_that_travels_all_three_forms_says_so():
    registry = reset_knowledge_registry_for_test()
    registry.hold("halving", Form.IMPLICIT, _policy, benefit=1.0)
    registry.convert("halving", source=Form.IMPLICIT, target=Form.EXPLICIT,
                     build=lambda _p: _policy, cases=list(range(40)), benefit=0.6)
    registry.convert("halving", source=Form.EXPLICIT, target=Form.NEURAL,
                     build=lambda _p: _policy, cases=list(range(40)), benefit=0.9)
    entry = registry.get("halving")
    assert entry.travelled_all_three
    assert registry.report()["in_all_three"] == 1


def test_a_form_with_no_measured_benefit_is_reported_as_idle():
    registry = reset_knowledge_registry_for_test()
    registry.hold("x", Form.IMPLICIT, _policy, benefit=1.0)
    registry.convert("x", source=Form.IMPLICIT, target=Form.EXPLICIT,
                     build=lambda _p: _policy, cases=list(range(20)))
    assert "explicit" in registry.get("x").each_form_earns_its_place()["idle_forms"]


def test_equivalence_is_measured_on_held_out_cases_by_default():
    equivalence = measure_equivalence(_policy, _policy, list(range(10)))
    assert equivalence.measured_on == "held_out" and equivalence.passes


def test_a_source_that_raises_counts_as_a_disagreement():
    def explodes(_x):
        raise ValueError("no")

    equivalence = measure_equivalence(explodes, _policy, [1, 2, 3])
    assert equivalence.agreement == 0.0


# ── credit ────────────────────────────────────────────────────────────────

def test_credit_goes_to_what_removal_actually_costs():
    credit = CreditAssignment()
    share = credit.assign("turn", full_score=1.0, without={"policy": 0.4, "rule": 0.95, "memory": 1.0})
    assert share["policy"] > share["rule"] > share["memory"] == 0.0


def test_a_component_that_would_not_have_been_missed_gets_nothing():
    credit = CreditAssignment()
    credit.assign("turn", full_score=1.0, without={"loud": 1.0, "quiet": 0.2})
    assert credit.report()["components_with_no_credit"] == ["loud"]


def test_when_nothing_would_have_been_missed_credit_is_empty_not_shared_out():
    credit = CreditAssignment()
    share = credit.assign("turn", full_score=1.0, without={"a": 1.0, "b": 1.0})
    assert share == {"a": 0.0, "b": 0.0}
