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
    TemporalFactEvidence,
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
        left, interpretation, outcome="found", context_id="a", supports=True,
        evidence="trace-a", source_id="observer-a",
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
        source_id="observer-b",
    )
    assert candidate.status(minimum_support=2, minimum_contexts=2) == "consolidated"


def test_relabelled_contexts_from_one_source_do_not_certify_principle():
    left, right = cases()
    model = RelationalGeneralizer(minimum_support=2, minimum_contexts=2)
    candidate = model.observe(left, Interpretation("retrieve"), outcome="found",
                              context_id="one", supports=True, evidence="trace-a",
                              source_id="one-observer")
    model.observe(right, Interpretation("retrieve"), outcome="found",
                  context_id="two", supports=True, falsification_attempt=True,
                  evidence="trace-b", source_id="one-observer")
    assert candidate.independent_contexts == 2
    assert candidate.independent_sources == 1
    assert candidate.status(minimum_support=2, minimum_contexts=2) == "provisional"


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


def test_plausible_readings_keep_unknown_consequences_open():
    model = RelationalGeneralizer()
    affordable = Interpretation("salary alone funded the purchase",
        believed_facts=(("income_is_sufficient", True), ("owns_vehicle", True)))
    prior_savings = Interpretation("purchase came from earlier savings",
        believed_facts=(("owns_vehicle", True), ("prior_savings", True)))
    audits = model.scrutinize((affordable, prior_savings), observations={
        "income_is_sufficient": (False, "budget-receipt"),
        "owns_vehicle": (True, "registration-receipt"),
    })
    assert audits[0].status == "contradicted"
    assert audits[0].contradicted == (("income_is_sufficient", "budget-receipt"),)
    assert audits[1].status == "unresolved"
    assert audits[1].unmeasured == ("prior_savings",)
    revised = model.scrutinize((prior_savings,), observations={
        "owns_vehicle": (True, "registration-receipt"),
        "prior_savings": (True, "bank-receipt"),
    })
    assert revised[0].status == "supported"


def test_scrutiny_rejects_unmeasured_or_conflicting_claim_identity():
    model = RelationalGeneralizer()
    with pytest.raises(ValueError, match="provenance"):
        model.scrutinize((Interpretation("claim", believed_facts=(("fact", True),)),),
                         observations={"fact": (True, "")})
    with pytest.raises(ValueError, match="distinct"):
        model.scrutinize((Interpretation("claim", believed_facts=(("fact", True),
                                                                    ("fact", False))),),
                         observations={})


def test_discrimination_distinguishes_decisive_from_one_sided_questions():
    first = Interpretation("new salary paid", believed_facts=(
        ("income_is_sufficient", True), ("owns_vehicle", True)))
    second = Interpretation("prior savings paid", believed_facts=(
        ("income_is_sufficient", False), ("prior_savings", True),
        ("owns_vehicle", True)))
    inquiries = RelationalGeneralizer().plan_discrimination(
        (first, second), observations={"owns_vehicle": (True, "registration")})
    assert [(item.proposition, item.decisive) for item in inquiries] == [
        ("income_is_sufficient", True), ("prior_savings", False)]
    assert inquiries[0].predictions == (("new salary paid", True),
                                        ("prior savings paid", False))
    assert RelationalGeneralizer().plan_discrimination(
        (first, second), observations={"owns_vehicle": (True, "registration"),
                                       "income_is_sufficient": (False, "budget")})[0].proposition == (
                                           "prior_savings")


def test_analogy_exposes_extra_target_constraints_without_claiming_transfer():
    boxing = RelationalCase("boxing", (("athlete", "competitor"), ("opponent", "competitor")),
                            (("strikes", "athlete", "opponent"),), goal="win")
    mma = RelationalCase("mma", (("fighter", "competitor"), ("rival", "competitor")),
                         (("punches", "fighter", "rival"),
                          ("grapples", "fighter", "rival")), goal="win")
    probe = RelationalGeneralizer().probe_transfer(boxing, mma, trials=4)
    assert probe.matched == 1
    assert len(probe.target_unmatched) == 1
    assert probe.role_consistent
    assert probe.null_separation is not None
    assert not probe.goal_equivalence_measured
    assert not probe.task_outcome_measured
    assert not probe.structural_candidate


def test_revision_separates_changed_fact_from_initially_wrong_belief():
    evidence = (
        TemporalFactEvidence("rule_applies", True, 10, 12, "archive", "old-rule"),
        TemporalFactEvidence("rule_applies", False, 20, 21, "registry", "new-rule"),
    )
    assess = RelationalGeneralizer.assess_revision
    changed = assess("rule_applies", old_belief=True, new_belief=False,
                     old_at=10, new_at=20, evidence=evidence)
    assert changed.status == "world_change_supported"
    assert changed.old_evidence == ("archive:old-rule",)
    assert changed.new_evidence == ("registry:new-rule",)
    misunderstood = assess("rule_applies", old_belief=True, new_belief=False,
                           old_at=10, new_at=20,
                           evidence=(replace(evidence[0], value=False), evidence[1]))
    assert misunderstood.status == "prior_misunderstanding_supported"
    recent_only = assess("rule_applies", old_belief=True, new_belief=False,
                         old_at=10, new_at=20, evidence=(evidence[1],))
    assert recent_only.status == "unresolved"


def test_revision_refuses_conflicting_or_replayed_temporal_evidence():
    earlier = TemporalFactEvidence("p", True, 1, 3, "archive", "one")
    later = TemporalFactEvidence("p", False, 2, 4, "registry", "two")
    assess = RelationalGeneralizer.assess_revision
    conflict = assess("p", old_belief=True, new_belief=False, old_at=1, new_at=2,
                      evidence=(earlier, replace(earlier, value=False, ref="contradiction"),
                                later))
    assert conflict.status == "unresolved"
    with pytest.raises(ValueError, match="count twice"):
        assess("p", old_belief=True, new_belief=False, old_at=1, new_at=2,
               evidence=(earlier, earlier, later))


def test_cross_domain_analogy_needs_sourced_roles_and_exposes_foil():
    schedule = RelationalCase(
        "schedule", (("doctor", "worker"), ("slot", "capacity"),
                     ("patient", "request")),
        (("claims", "patient", "slot"), ("serves", "doctor", "patient"),
         ("occupies", "doctor", "slot")), goal="serve requests",
    )
    async_work = RelationalCase(
        "async", (("job", "process"), ("worker", "executor"),
                  ("resource", "resource")),
        (("queues_for", "job", "resource"), ("handled_by", "worker", "job"),
         ("holds", "worker", "resource")), goal="serve requests",
    )
    foil = replace(async_work, case_id="async-with-preemption",
                   relations=(*async_work.relations,
                              ("preempts", "job", "worker")))
    generalizer = RelationalGeneralizer()
    unsourced = generalizer.probe_transfer(schedule, async_work, trials=8)
    assert not unsourced.role_consistent
    with pytest.raises(ValueError, match="sourced correspondence"):
        generalizer.probe_transfer(schedule, async_work, role_correspondence={"worker": "executor"})
    roles = {"worker": "executor", "capacity": "resource", "request": "process"}
    paired = generalizer.probe_transfer(schedule, async_work, trials=8,
                                        role_correspondence=roles,
                                        role_evidence="independent-role-analysis")
    altered = generalizer.probe_transfer(schedule, foil, trials=8,
                                         role_correspondence=roles,
                                         role_evidence="independent-role-analysis")
    assert paired.matched == len(schedule.relations)
    assert not paired.source_unmatched and not paired.target_unmatched
    assert paired.role_consistent
    assert altered.target_unmatched
    assert not altered.structural_candidate
    assert not paired.task_outcome_measured


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
