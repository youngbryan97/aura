"""When, one door for proposals, and a thing done twice becoming a thing known.

Cards 037, 038, 049, 091, 167, A12.16, A13.1, A13.5, A13.6, A13.8, A13.9,
A13.10, A13.11, A13.12, A13.13.
"""
from __future__ import annotations

import pytest

from core.cognition.action_hub import Proposal, Source, reset_action_hub_for_test
from core.cognition.cognitive_event import (
    Epistemic,
    Phase,
    ReadDependency,
    cycle,
    reads,
    reset_event_graph_for_test,
)
from core.cognition.procedure import Precondition, Signature, reset_procedure_registry_for_test
from core.cognition.trace_compiler import TraceCompiler
from core.knowledge.atomspace import AtomSpace
from core.knowledge.temporal import (
    Interval,
    TemporalRelation,
    TimedEvent,
    assert_temporal_rules,
    induce_temporal_rules,
    temporal_link,
)


# ── temporal ──────────────────────────────────────────────────────────────

def _stream(*, coin_flips: bool):
    events, t = [], 0.0
    for i in range(12):
        events += [TimedEvent("kettle_on", Interval(t)), TimedEvent("steam", Interval(t + 2.0))]
        offset = (-1.0 if i % 2 else 1.0) if coin_flips else 1.0
        events.append(TimedEvent("coin", Interval(t + offset)))
        t += 100.0
    return events


def test_a_consistent_ordering_becomes_a_rule():
    rules = induce_temporal_rules(_stream(coin_flips=False), window=10.0)
    orderings = {(r.antecedent, r.consequent) for r in rules if r.relation is TemporalRelation.BEFORE}
    assert ("kettle_on", "steam") in orderings


def test_two_events_whose_order_flips_produce_no_ordering_rule():
    rules = induce_temporal_rules(_stream(coin_flips=True), window=10.0)
    orderings = {(r.antecedent, r.consequent) for r in rules if r.relation is TemporalRelation.BEFORE}
    assert ("kettle_on", "coin") not in orderings
    assert ("coin", "kettle_on") not in orderings


def test_a_lag_whose_spread_matches_its_mean_is_not_a_lag():
    rules = induce_temporal_rules(_stream(coin_flips=False), window=10.0)
    kettle = next(r for r in rules if (r.antecedent, r.consequent) == ("kettle_on", "steam"))
    assert kettle.lag_is_meaningful and kettle.mean_lag == pytest.approx(2.0)


def test_recurrence_is_found_with_its_period():
    rules = induce_temporal_rules(_stream(coin_flips=False), window=10.0)
    recurring = {r.antecedent: r for r in rules if r.relation is TemporalRelation.RECURS}
    assert recurring["kettle_on"].period == pytest.approx(100.0)


def test_too_few_observations_produce_nothing():
    events = [TimedEvent("a", Interval(0.0)), TimedEvent("b", Interval(1.0))]
    assert induce_temporal_rules(events, window=10.0) == []


def test_a_window_stops_a_long_session_relating_everything_to_everything():
    events = [TimedEvent("a", Interval(0.0)), TimedEvent("b", Interval(10_000.0))]
    assert induce_temporal_rules(events, window=1.0) == []


def test_the_same_lag_and_a_different_lag_are_different_claims():
    fast = temporal_link(TemporalRelation.BEFORE, "a", "b", lag=1.0)
    slow = temporal_link(TemporalRelation.BEFORE, "a", "b", lag=3600.0)
    assert fast != slow


def test_re_running_the_induction_on_one_stream_is_one_observation():
    space = AtomSpace()
    rules = induce_temporal_rules(_stream(coin_flips=False), window=10.0)
    for _ in range(5):
        assert_temporal_rules(space, rules, source="induction:stream1")
    assert space.evidence_report()["duplicate_assertions_refused"] == 4 * len(rules)


def test_intervals_know_containment_from_overlap():
    outer, inner = Interval(0.0, 10.0), Interval(2.0, 5.0)
    straddle = Interval(8.0, 12.0)
    assert inner.during(outer)
    assert straddle.overlaps(outer)
    assert not inner.overlaps(outer)


# ── the action hub ────────────────────────────────────────────────────────

def _proposals():
    return [
        Proposal("click_save", Source.PROCEDURE, value=1.0, confidence=0.9),
        Proposal("ask_user", Source.CORTEX, value=1.0, confidence=0.5),
        Proposal("noop", Source.HABIT, value=0.1, confidence=0.1),
        Proposal(
            "dismiss_modal", Source.RULE, value=5.0, confidence=1.0,
            signature=Signature((Precondition("modal"),), ()),
        ),
    ]


def test_a_proposal_whose_preconditions_the_situation_does_not_meet_is_rejected():
    hub = reset_action_hub_for_test()
    decision = hub.decide(_proposals(), situation={"file_open": True})
    assert [p.action for p in decision.rejected_untyped_mismatch] == ["dismiss_modal"]


def test_an_untyped_proposal_is_still_considered():
    hub = reset_action_hub_for_test()
    decision = hub.decide(_proposals(), situation={})
    assert any(p.action == "click_save" for p in decision.considered)


def test_the_hub_reports_what_would_have_happened_without_each_source():
    hub = reset_action_hub_for_test()
    decision = hub.decide(_proposals(), situation={"file_open": True})
    assert decision.chosen.action == "click_save"
    assert decision.counterfactual["procedure"] == "ask_user"
    assert decision.counterfactual["cortex"] == "click_save"


def test_a_tie_is_reported_rather_than_settled_here():
    hub = reset_action_hub_for_test()
    decision = hub.decide(
        [
            Proposal("a", Source.HABIT, value=1.0, confidence=0.5),
            Proposal("b", Source.RULE, value=1.0, confidence=0.5),
        ],
        situation={},
    )
    assert decision.impasse.startswith("tie:")


def test_nothing_applicable_is_a_rejection_impasse_not_a_choice():
    hub = reset_action_hub_for_test()
    decision = hub.decide(
        [Proposal("x", Source.RULE, signature=Signature((Precondition("nope"),), ()))],
        situation={},
    )
    assert not decision.decided and decision.impasse.startswith("rejection:")


def test_a_source_that_proposes_constantly_and_never_matters_is_named():
    hub = reset_action_hub_for_test()
    for _ in range(12):
        hub.decide(_proposals(), situation={"file_open": True})
    attribution = hub.attribution()
    assert "cortex" in attribution["proposes_but_never_matters"]
    assert attribution["taken_by_source"] == {"procedure": 12}


def test_the_learned_fraction_of_actions_is_computable():
    hub = reset_action_hub_for_test()
    hub.decide([Proposal("a", Source.PROCEDURE, value=1.0, confidence=1.0)], situation={})
    hub.decide([Proposal("b", Source.CORTEX, value=1.0, confidence=1.0)], situation={})
    assert hub.attribution()["learned_fraction"] == pytest.approx(0.5)


# ── the trace compiler ────────────────────────────────────────────────────

def _run(graph, *, theme: str, task_seq: list[int]):
    with cycle("turn"):
        percept = graph.record(
            Phase.PERCEIVE, "perception", "frame",
            reads=reads([("editor", "vscode"), ("theme", theme)]), duration_s=0.4,
        )
        recall = graph.record(
            Phase.RETRIEVE, "memory", "recall", parents=[percept.seq],
            reads=(ReadDependency("never_checked", Epistemic.UNOBSERVED),), duration_s=0.9,
        )
        act = graph.record(
            Phase.APPLY, "actuator", "save", parents=[recall.seq],
            reads=reads([("file_open", True)]), duration_s=0.2,
        )
    task_seq.append(act.seq)
    return act.seq


def test_one_success_compiles_nothing():
    graph = reset_event_graph_for_test()
    compiler = TraceCompiler(reset_procedure_registry_for_test())
    seqs: list[int] = []
    compiler.observe(graph, "save", _run(graph, theme="dark", task_seq=seqs))
    result = compiler.compile("save")
    assert result.compiled is None and "anecdote" in result.reason


def test_three_successes_compile_a_procedure_over_their_shared_support():
    graph = reset_event_graph_for_test()
    registry = reset_procedure_registry_for_test()
    compiler = TraceCompiler(registry)
    seqs: list[int] = []
    for theme in ("light", "dark", "dark"):
        compiler.observe(graph, "save", _run(graph, theme=theme, task_seq=seqs))
    result = compiler.compile("save")
    assert result.compiled is not None
    assert set(result.shared_support) == {"editor", "theme", "file_open"}


def test_a_condition_nobody_ever_looked_at_never_reaches_the_procedure():
    graph = reset_event_graph_for_test()
    compiler = TraceCompiler(reset_procedure_registry_for_test())
    seqs: list[int] = []
    for theme in ("light", "dark", "dark"):
        compiler.observe(graph, "save", _run(graph, theme=theme, task_seq=seqs))
    result = compiler.compile("save")
    assert "never_checked" not in result.shared_support


def test_a_condition_that_never_varied_is_kept_and_marked():
    graph = reset_event_graph_for_test()
    compiler = TraceCompiler(reset_procedure_registry_for_test())
    seqs: list[int] = []
    for theme in ("light", "dark", "dark"):
        compiler.observe(graph, "save", _run(graph, theme=theme, task_seq=seqs))
    result = compiler.compile("save")
    assert set(result.unvaried) == {"editor", "file_open"}
    assert "theme" not in result.unvaried


def test_the_saving_is_measured_from_what_the_runs_actually_spent():
    graph = reset_event_graph_for_test()
    registry = reset_procedure_registry_for_test()
    compiler = TraceCompiler(registry)
    seqs: list[int] = []
    for theme in ("light", "dark", "dark"):
        compiler.observe(graph, "save", _run(graph, theme=theme, task_seq=seqs))
    procedure = compiler.compile("save").compiled
    assert procedure.value.value_when_it_works == pytest.approx(1.5, abs=0.01)


def test_a_task_whose_runs_share_no_support_is_refused():
    graph = reset_event_graph_for_test()
    compiler = TraceCompiler(reset_procedure_registry_for_test())
    for i in range(3):
        with cycle("turn"):
            act = graph.record(
                Phase.APPLY, "a", "x", reads=reads([(f"unique_{i}", i)]), duration_s=0.1
            )
        compiler.observe(graph, "flaky", act.seq)
    result = compiler.compile("flaky")
    assert result.compiled is None and "fires everywhere" in result.reason


def test_the_compiler_says_which_tasks_are_still_waiting_for_repetition():
    graph = reset_event_graph_for_test()
    compiler = TraceCompiler(reset_procedure_registry_for_test())
    seqs: list[int] = []
    compiler.observe(graph, "rare", _run(graph, theme="dark", task_seq=seqs))
    assert compiler.report()["awaiting_repetition"] == ["rare"]
