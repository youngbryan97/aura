"""Human-derived relational invariants are mechanical and evidence-bound."""

import itertools
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.cognition.agent_model import AgentModel
from core.cognition.procedural_generalization import (
    DecisionEpisode,
    ProceduralGeneralizer,
    PromotionCriteria,
    RuleTier,
)
from core.cognition.relational_generalization import (
    Interpretation,
    RelationalCase,
    RelationalGeneralizer,
)


def cases():
    return (
        RelationalCase(
            "one",
            (("alice", "person"), ("box", "container")),
            (("inside", "alice", "box"),),
            (("state", "open"),),
            "find object",
            "chat",
        ),
        RelationalCase(
            "two",
            (("bob", "person"), ("crate", "container")),
            (("inside", "bob", "crate"),),
            (("state", "open"),),
            "find object",
            "chat",
        ),
    )


def test_sameness_is_structural_not_literal():
    left, right = cases()
    generalizer = RelationalGeneralizer()
    assert generalizer.same_problem(left, right)


def test_principle_requires_independent_support_and_falsification():
    left, right = cases()
    model = RelationalGeneralizer(minimum_support=2, minimum_contexts=2)
    interpretation = Interpretation("retrieve contained person")
    candidate = model.observe(
        left, interpretation, outcome="found", context_id="a", supports=True, evidence="trace-a"
    )
    assert candidate.status(minimum_support=2, minimum_contexts=2) == "provisional"
    model.observe(
        right,
        interpretation,
        outcome="found",
        context_id="b",
        supports=True,
        falsification_attempt=True,
        evidence="trace-b",
    )
    assert candidate.status(minimum_support=2, minimum_contexts=2) == "consolidated"


def test_contradiction_requests_revision_and_records_continuity():
    left, _ = cases()
    model = RelationalGeneralizer()
    interpretation = Interpretation("retrieve contained person")
    candidate = model.observe(
        left,
        interpretation,
        outcome="missing",
        context_id="a",
        supports=False,
        evidence="trace-fail",
    )
    assert candidate.status(minimum_support=3, minimum_contexts=2) == "needs_revision"
    assert model.continuity("chat") == ("one",)
    revision = model.revise(left, interpretation, Interpretation("inspect container first"))
    assert revision["preserved_facts"] == left.invariant_facts
    assert revision["new_hypothesis"] == "inspect container first"


def test_perspective_is_not_world_truth():
    agent = AgentModel("someone")
    model = RelationalGeneralizer(agent_model=agent)
    result = model.record_perspective(
        Interpretation(
            "choose route", perspective_agent="someone", believed_facts=(("door_open", True),)
        ),
        world_truth={"door_open": False},
        evidence="conversation",
    )
    assert result == {"recorded": True, "divergences": {"door_open": True}}
    assert agent.false_belief("someone:door_open", world_truth=False)["known"]


def test_a_different_goal_is_not_silently_same_problem():
    left, _ = cases()
    altered = RelationalCase(
        "three", left.entities, left.relations, left.facts, "count objects", left.context
    )
    assert not RelationalGeneralizer().same_problem(left, altered)


def test_changed_facts_cannot_be_treated_as_the_same_problem():
    left, _ = cases()
    assert not RelationalGeneralizer().same_problem(
        left, replace(left, facts=(("state", "closed"),)))


def test_goal_order_is_semantic_not_a_bag_of_words():
    left, _ = cases()
    assert not RelationalGeneralizer().same_problem(
        replace(left, goal="person before container"),
        replace(left, goal="container before person"),
    )


@pytest.mark.parametrize(
    "left_goal,right_goal",
    [
        ("x < y", "x > y"),
        ("x = y", "x != y"),
        ("X equals y", "x equals y"),
        ("最小", "最大"),
    ],
)
def test_goal_symbols_case_and_unicode_are_not_discarded(left_goal, right_goal):
    left, _ = cases()
    assert not RelationalGeneralizer().same_problem(
        replace(left, goal=left_goal), replace(left, goal=right_goal)
    )


def test_replayed_evidence_does_not_increase_support():
    left, _ = cases()
    model = RelationalGeneralizer()
    for _ in range(4):
        candidate = model.observe(
            left,
            Interpretation("retrieve"),
            outcome="found",
            context_id="a",
            supports=True,
            evidence="same-trace",
        )
    assert candidate.support == 1


def test_unmeasured_evidence_is_rejected_without_mutation():
    left, _ = cases()
    model = RelationalGeneralizer()
    with pytest.raises(ValueError):
        model.observe(
            left,
            Interpretation("retrieve"),
            outcome="found",
            context_id="a",
            supports=None,
            evidence="trace",
        )
    assert model.candidate(left, "retrieve") is None


def test_contradiction_reaches_existing_concept_engine():
    left, _ = cases()
    engine = Mock()
    model = RelationalGeneralizer(concept_engine=engine)
    model.observe(
        left,
        Interpretation("retrieve"),
        outcome="missing",
        context_id="a",
        supports=False,
        evidence="trace",
    )
    engine.observe_prediction_error.assert_called_once()


def test_same_resolution_failure_counts_against_a_shared_rule():
    engine = ProceduralGeneralizer(
        PromotionCriteria(min_episodes=2, min_confidence_lower_bound=0.1)
    )
    for _ in range(3):
        engine.record(DecisionEpisode(frozenset({"state=open"}), "enter", True))
    engine.record(DecisionEpisode(frozenset({"state=open"}), "enter", False))
    rule = engine.derive("enter")
    assert rule.contradicting == 1
    assert rule.tier is RuleTier.CANDIDATE


def test_relational_outcomes_feed_shared_procedural_learning():
    left, _ = cases()
    engine = ProceduralGeneralizer(
        PromotionCriteria(min_episodes=2, min_confidence_lower_bound=0.1)
    )
    model = RelationalGeneralizer(procedural_generalizer=engine)
    for index in range(3):
        model.observe(
            left,
            Interpretation("retrieve"),
            outcome="found",
            context_id=str(index),
            supports=True,
            evidence=f"trace-{index}",
        )
    assert engine.derive("retrieve").supporting == 3
    model.observe(
        left,
        Interpretation("retrieve"),
        outcome="missing",
        context_id="four",
        supports=False,
        evidence="failed-trace",
    )
    assert engine.derive("retrieve").contradicting == 1


def test_relabeling_context_does_not_create_independent_evidence():
    left, _ = cases()
    model = RelationalGeneralizer()
    for context in ("a", "b", "c"):
        candidate = model.observe(
            left,
            Interpretation("retrieve"),
            outcome="found",
            context_id=context,
            supports=True,
            evidence="same-trace",
        )
    assert candidate.support == 1
    assert candidate.independent_contexts == 1


def test_typed_canonicalization_matches_exhaustive_reference():
    entities = (("a", "person"), ("b", "container"), ("c", "person"), ("d", "container"))
    relations = (("owns", "a", "b"), ("owns", "c", "d"), ("near", "b", "d"))
    case = RelationalCase("case", entities, relations)
    expected = []
    for order in itertools.permutations(entities):
        positions = {name: index for index, (name, _) in enumerate(order)}
        expected.append(
            (
                tuple(kind for _, kind in order),
                tuple(
                    sorted(
                        (relation, positions[left], positions[right])
                        for relation, left, right in relations
                    )
                ),
                (),
            )
        )
    assert case.shape_key == min(expected)
    assert replace(case, entities=tuple(reversed(entities))).shape_key == case.shape_key


def test_distinct_typed_roles_do_not_require_factorial_permutations():
    case = RelationalCase("large", tuple((str(i), f"kind-{i:02}") for i in range(12)), ())
    assert case.shape_key[0] == tuple(f"kind-{i:02}" for i in range(12))


def test_understanding_is_measured_by_independent_feedback_against_prior():
    case, _ = cases()
    agent = AgentModel("user")
    model = RelationalGeneralizer(agent_model=agent)
    ticket = model.predict_interpretation(
        case,
        Interpretation("request object", predicted_outcome="object_retrieved"),
        prior_outcome="object_counted",
    )
    assert not agent.beats_the_prior()["measurable"]
    candidate = model.resolve_interpretation(
        ticket, actual_outcome="object_retrieved", context_id="turn-2", evidence="action-receipt"
    )
    assert candidate.support == 1
    assert agent.beats_the_prior()["delta"] == 1.0
    model.resolve_interpretation(
        ticket, actual_outcome="object_retrieved", context_id="turn-3", evidence="replayed"
    )
    assert candidate.support == 1
    with pytest.raises(ValueError, match="rewritten"):
        model.resolve_interpretation(
            ticket, actual_outcome="object_counted", context_id="turn-4", evidence="revision"
        )


def test_misunderstanding_reaches_revision_without_phrase_matching():
    case, _ = cases()
    agent = AgentModel("user")
    model = RelationalGeneralizer(agent_model=agent)
    ticket = model.predict_interpretation(
        case,
        Interpretation("request object", predicted_outcome="object_retrieved"),
        prior_outcome="object_counted",
    )
    candidate = model.resolve_interpretation(
        ticket, actual_outcome="object_counted", context_id="turn-2", evidence="user-confirmation"
    )
    assert candidate.contradictions == 1
    assert agent.beats_the_prior()["delta"] == -1.0


@pytest.mark.parametrize("field,value", [("topic", "other goal"), ("prior_predicted", "changed")])
def test_prediction_ticket_binds_goal_and_prior(field, value):
    case, _ = cases()
    agent = AgentModel("user")
    model = RelationalGeneralizer(agent_model=agent)
    ticket = model.predict_interpretation(
        case, Interpretation("retrieve", predicted_outcome="found"), prior_outcome="missing"
    )
    agent.predictions[ticket] = replace(agent.predictions[ticket], **{field: value})
    with pytest.raises(ValueError, match="changed"):
        model.resolve_interpretation(ticket, actual_outcome="found", context_id="x", evidence="e")
    assert model.candidate(case, "retrieve") is None


def test_external_resolution_cannot_impersonate_evidence_delivery():
    case, _ = cases()
    agent = AgentModel("user")
    model = RelationalGeneralizer(agent_model=agent)
    ticket = model.predict_interpretation(
        case, Interpretation("retrieve", predicted_outcome="found"), prior_outcome="missing"
    )
    agent.resolve(ticket, "found")
    with pytest.raises(ValueError, match="outside its evidence path"):
        model.resolve_interpretation(ticket, actual_outcome="found", context_id="x", evidence="e")
    assert model.candidate(case, "retrieve") is None
