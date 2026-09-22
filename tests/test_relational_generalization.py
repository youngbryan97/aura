"""Human-derived relational invariants are mechanical and evidence-bound."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.cognition.agent_model import AgentModel
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


def test_goal_order_is_semantic_not_a_bag_of_words():
    left, _ = cases()
    assert not RelationalGeneralizer().same_problem(
        replace(left, goal="person before container"),
        replace(left, goal="container before person"),
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
