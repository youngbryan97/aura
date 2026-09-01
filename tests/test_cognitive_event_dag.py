"""One causal timeline, and the reads a compiled rule may rest on.

Cards 011, 018, 023, 027, 060, 061, 064, 069, 073, 176, A12.5, A12.8, A12.9,
A12.10, A12.13.

The distinction under test is the one Aura's learners never made: reading a
field and finding it false is a fact; not looking is not. A chunk compiled
from a run that never checked the sidebar must not carry "the sidebar was
absent" as a precondition, because it will then fire confidently in the case
it was never entitled to an opinion about.
"""
from __future__ import annotations

import pytest

from core.cognition.cognitive_event import (
    Epistemic,
    EventGraph,
    Phase,
    ReadDependency,
    current_cycle,
    cycle,
    reads,
    reset_event_graph_for_test,
)
from core.cognition.situation import (
    LearningBroadcast,
    SituationSnapshot,
    reset_coordinator_for_test,
    snapshot,
)


# ── the DAG ───────────────────────────────────────────────────────────────

def test_a_consequential_action_reconstructs_from_one_bundle():
    graph = reset_event_graph_for_test()
    with cycle("turn"):
        p = graph.record(Phase.PERCEIVE, "perception", "frame", loop="legacy_pipeline")
        r = graph.record(Phase.RETRIEVE, "memory", "recall", parents=[p.seq])
        s = graph.record(Phase.SELECT, "workspace", "win", parents=[r.seq])
        a = graph.record(Phase.APPLY, "actuator", "click", parents=[s.seq])
    bundle = graph.bundle(a.seq)
    assert bundle["found"]
    assert [e["organ"] for e in bundle["ancestors"]] == ["perception", "memory", "workspace"]
    assert bundle["cycle_id"] == p.cycle_id


def test_ids_are_monotonic_across_both_phase_loops():
    graph = reset_event_graph_for_test()
    a = graph.record(Phase.PERCEIVE, "o", "x", loop="legacy_pipeline")
    b = graph.record(Phase.PERCEIVE, "o", "x", loop="kernel_tick")
    c = graph.record(Phase.APPLY, "o", "x", loop="legacy_pipeline")
    assert a.seq < b.seq < c.seq


def test_an_unobserved_read_never_becomes_a_precondition():
    graph = reset_event_graph_for_test()
    e = graph.record(
        Phase.APPLY, "actuator", "click",
        reads=(
            ReadDependency("window", Epistemic.OBSERVED, "d1"),
            ReadDependency("sidebar", Epistemic.UNOBSERVED),
            ReadDependency("modal", Epistemic.INACCESSIBLE),
        ),
    )
    keys = [d.key for d in graph.minimal_support(e.seq)]
    assert keys == ["window"]
    assert {d["key"] for d in graph.bundle(e.seq)["unsupportable_reads"]} == {"sidebar", "modal"}


def test_observed_absent_is_a_fact_and_does_support_a_rule():
    graph = reset_event_graph_for_test()
    e = graph.record(Phase.APPLY, "a", "x", reads=reads([("sidebar", None)]))
    support = graph.minimal_support(e.seq)
    assert [d.key for d in support] == ["sidebar"]
    assert support[0].status is Epistemic.OBSERVED_ABSENT


def test_context_present_but_never_read_does_not_enter_the_support():
    """What makes a compiled rule fire under paraphrase."""
    graph = reset_event_graph_for_test()
    p = graph.record(Phase.PERCEIVE, "perception", "frame",
                     reads=reads([("button_label", "Save"), ("theme", "dark")]))
    unrelated = graph.record(Phase.PERCEIVE, "perception", "other",
                             reads=reads([("wallpaper", "blue")]))
    a = graph.record(Phase.APPLY, "actuator", "click", parents=[p.seq])
    keys = [d.key for d in graph.minimal_support(a.seq)]
    assert "wallpaper" not in keys
    assert unrelated.seq not in [e.seq for e in graph.ancestors(a.seq)]


def test_the_same_field_read_three_times_is_one_precondition():
    graph = reset_event_graph_for_test()
    a = graph.record(Phase.PERCEIVE, "o", "x", reads=reads([("k", 1)]))
    b = graph.record(Phase.RETRIEVE, "o", "x", parents=[a.seq], reads=reads([("k", 1)]))
    c = graph.record(Phase.APPLY, "o", "x", parents=[b.seq], reads=reads([("k", 1)]))
    assert len(graph.minimal_support(c.seq)) == 1


def test_a_read_that_was_observed_anywhere_outranks_an_absent_reading():
    graph = reset_event_graph_for_test()
    a = graph.record(Phase.PERCEIVE, "o", "x", reads=reads([("k", None)]))
    b = graph.record(Phase.APPLY, "o", "x", parents=[a.seq], reads=reads([("k", 5)]))
    assert graph.minimal_support(b.seq)[0].status is Epistemic.OBSERVED


def test_a_parent_that_does_not_exist_is_dropped_rather_than_faked():
    graph = reset_event_graph_for_test()
    e = graph.record(Phase.APPLY, "o", "x", parents=[999])
    assert e.parents == ()


def test_the_graph_is_bounded_and_says_what_it_dropped():
    graph = reset_event_graph_for_test(capacity=10)
    for _ in range(25):
        graph.record(Phase.PERCEIVE, "o", "x")
    report = graph.report()
    assert report["events"] == 10
    assert report["dropped"] == 15


def test_nested_cycles_keep_their_own_events():
    graph = reset_event_graph_for_test()
    with cycle("outer"):
        outer_id = current_cycle()
        a = graph.record(Phase.PERCEIVE, "o", "x")
        with cycle("substate"):
            inner_id = current_cycle()
            b = graph.record(Phase.IMPASSE, "o", "y")
        c = graph.record(Phase.APPLY, "o", "z")
    assert inner_id != outer_id
    assert a.cycle_id == c.cycle_id == outer_id
    assert b.cycle_id == inner_id
    assert [e.seq for e in graph.cycle_events(inner_id)] == [b.seq]


def test_cycle_id_is_zero_outside_a_cycle():
    graph = reset_event_graph_for_test()
    assert graph.record(Phase.PERCEIVE, "o", "x").cycle_id == 0


def test_time_localises_to_a_phase_and_to_a_loop():
    graph = reset_event_graph_for_test()
    graph.record(Phase.RETRIEVE, "memory", "slow", loop="legacy_pipeline", duration_s=2.0)
    graph.record(Phase.RETRIEVE, "memory", "fast", loop="kernel_tick", duration_s=0.1)
    graph.record(Phase.APPLY, "actuator", "click", loop="kernel_tick", duration_s=0.2)
    timing = graph.phase_timing()
    assert timing["by_phase"]["retrieve"]["max_s"] == pytest.approx(2.0)
    assert timing["by_loop"]["legacy_pipeline"]["total_s"] == pytest.approx(2.0)
    assert timing["by_loop"]["kernel_tick"]["count"] == 2


def test_causal_coverage_is_reported_so_orphan_events_are_visible():
    graph = reset_event_graph_for_test()
    a = graph.record(Phase.PERCEIVE, "o", "x")
    graph.record(Phase.APPLY, "o", "y", parents=[a.seq])
    assert graph.report()["causal_coverage"] == pytest.approx(0.5)


# ── the situation ─────────────────────────────────────────────────────────

def test_two_organs_can_prove_they_saw_the_same_world():
    a = SituationSnapshot(cycle_id=1, percepts={"x": 1}, goals=("save",))
    b = SituationSnapshot(cycle_id=1, percepts={"x": 1}, goals=("save",))
    c = SituationSnapshot(cycle_id=1, percepts={"x": 2}, goals=("save",))
    assert a.agrees_with(b)
    assert not a.agrees_with(c)


def test_the_snapshot_says_what_it_does_not_settle():
    snap = SituationSnapshot(cycle_id=1, percepts={"x": 1}, uncertainty={"window_focus": 0.6})
    assert snap.to_dict()["uncertainty"]["window_focus"] == 0.6


def test_a_snapshot_takes_the_cycle_it_was_built_in():
    with cycle("turn"):
        assert snapshot(percepts={}).cycle_id == current_cycle()


# ── the broadcast ─────────────────────────────────────────────────────────

def _broadcast(**kw):
    base = {"evidence_id": "e1", "content": {"k": 1}, "kind": "surprise", "prediction_error": 0.8}
    return LearningBroadcast(**{**base, **kw})


def test_one_event_updates_every_applicable_learner_under_one_id():
    coordinator = reset_coordinator_for_test()
    hits = []
    for name in ("episodic", "procedural", "perceptual", "attentional"):
        coordinator.subscribe(name, ["surprise"], lambda b, n=name: hits.append(n))
    coordinator.broadcast(_broadcast())
    assert sorted(hits) == ["attentional", "episodic", "perceptual", "procedural"]
    assert coordinator.learners_reached("e1") == frozenset(hits)
    assert coordinator.report()["events_that_taught_two_or_more"] == 1


def test_a_learner_only_hears_the_kinds_it_declared():
    coordinator = reset_coordinator_for_test()
    hits = []
    coordinator.subscribe("procedural", ["outcome"], lambda b: hits.append("procedural"))
    coordinator.subscribe("episodic", ["surprise"], lambda b: hits.append("episodic"))
    coordinator.broadcast(_broadcast(kind="surprise"))
    assert hits == ["episodic"]


def test_a_learner_that_declares_nothing_cannot_subscribe():
    coordinator = reset_coordinator_for_test()
    with pytest.raises(ValueError, match="nothing in particular"):
        coordinator.subscribe("greedy", [], lambda b: None)


def test_one_learner_failing_does_not_cost_the_others_their_update():
    coordinator = reset_coordinator_for_test()
    hits = []

    def broken(_b):
        raise RuntimeError("store is closed")

    coordinator.subscribe("broken", ["surprise"], broken)
    coordinator.subscribe("working", ["surprise"], lambda b: hits.append("working"))
    result = coordinator.broadcast(_broadcast())
    assert hits == ["working"]
    assert "broken" in result["errors"]
    assert result["reached"] == ["working"]


def test_an_event_with_nothing_at_stake_is_delivered_and_counted():
    coordinator = reset_coordinator_for_test()
    coordinator.subscribe("episodic", ["routine"], lambda b: None)
    result = coordinator.broadcast(_broadcast(kind="routine", prediction_error=0.0))
    assert result["reached"] == ["episodic"]
    assert not result["carried_a_signal"]
    assert coordinator.report()["signalless_broadcasts"] == 1


def test_the_delivery_ledger_is_bounded():
    coordinator = reset_coordinator_for_test()
    coordinator.subscribe("episodic", ["surprise"], lambda b: None)
    for i in range(5000):
        coordinator.broadcast(_broadcast(evidence_id=f"e{i}"))
    assert coordinator.report()["tracked_events"] <= 4096
