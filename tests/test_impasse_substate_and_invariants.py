"""Every deadlock becomes a smaller problem, and no update lands half-done.

Cards 016, 017, 019, 020, 024, 025, 029, 056, 175, 208, 209, A12.1, A12.2,
A12.3, A12.4, A12.11, A12.12, A12.17, A12.19.
"""
from __future__ import annotations

import asyncio

import pytest

from core.cognition.architecture_invariants import architecture_report
from core.cognition.impasse import Impasse, ImpasseType
from core.cognition.substate import (
    Resolution,
    SubstateBudget,
    SubstateOutcome,
    reset_impasse_bus_for_test,
)
from core.cognition.transaction import (
    InconsistentRollback,
    TransactionAborted,
    TransactionState,
    transaction,
)


def _tie(candidates=("a", "b")):
    return Impasse(ImpasseType.TIE, "sig", tuple(candidates))


# ── the bus ───────────────────────────────────────────────────────────────

def test_unrelated_organs_reach_the_same_mechanism():
    bus = reset_impasse_bus_for_test()
    seen = []
    bus.register(ImpasseType.TIE, "h", lambda ss: (seen.append(ss.organ), Resolution(SubstateOutcome.RESOLVED, choice="a"))[1])
    for organ in ("planner", "memory", "tool_chooser", "representation_search"):
        bus.raise_impasse(_tie(), organ=organ, goal="choose")
    assert sorted(seen) == ["memory", "planner", "representation_search", "tool_chooser"]
    assert bus.report()["organs_reporting"] == 4


def test_handlers_register_per_impasse_kind_not_per_organ():
    bus = reset_impasse_bus_for_test()
    bus.register(ImpasseType.TIE, "tie_handler", lambda ss: Resolution(SubstateOutcome.RESOLVED, choice="a"))
    resolved = bus.raise_impasse(_tie(), organ="planner", goal="g")
    unhandled = bus.raise_impasse(
        Impasse(ImpasseType.NO_CHANGE, "s", ("a",)), organ="planner", goal="g"
    )
    assert resolved.resolution.outcome is SubstateOutcome.RESOLVED
    assert unhandled.resolution.outcome is SubstateOutcome.UNHANDLED


def test_a_deadlock_is_recorded_even_when_nobody_handles_it():
    bus = reset_impasse_bus_for_test()
    substate = bus.raise_impasse(_tie(), organ="planner", goal="g")
    assert substate.resolution.outcome is SubstateOutcome.UNHANDLED
    assert bus.report()["by_organ"]["planner"] == 1


# ── recursion, with no organ-specific recursion code ──────────────────────

def test_three_unrelated_domains_nest_without_new_recursion_code():
    bus = reset_impasse_bus_for_test(default_budget=SubstateBudget(depth=3, seconds=5.0, work=100))

    def descend(ss):
        if ss.budget.depth <= 1:
            return Resolution(SubstateOutcome.RESOLVED, choice=ss.impasse.candidates[0])
        inner = bus.raise_impasse(
            _tie(("x", "y")), organ=f"{ss.organ}.sub", goal="narrower",
            parent_substate=ss.substate_id,
        )
        return Resolution(SubstateOutcome.RESOLVED, choice=inner.resolution.choice)

    bus.register(ImpasseType.TIE, "descend", descend)
    for organ in ("planner", "vision", "social"):
        outcome = bus.raise_impasse(_tie(), organ=organ, goal="g").resolution
        assert outcome.outcome is SubstateOutcome.RESOLVED
    assert bus.report()["nested"] >= 6


def test_a_nested_budget_is_strictly_smaller():
    parent = SubstateBudget(depth=3, seconds=5.0, work=100)
    child = parent.child(spent_seconds=1.0, spent_work=10)
    assert child.depth == 2 and child.seconds == 4.0 and child.work == 90


def test_recursion_terminates_on_the_budget_not_on_luck():
    bus = reset_impasse_bus_for_test(default_budget=SubstateBudget(depth=2, seconds=5.0, work=10))

    def always_descend(ss):
        inner = bus.raise_impasse(_tie(), organ="deep", goal="g", parent_substate=ss.substate_id)
        return Resolution(SubstateOutcome.RESOLVED, choice=inner.resolution.choice or "fell_through")

    bus.register(ImpasseType.TIE, "always", always_descend)
    top = bus.raise_impasse(_tie(), organ="planner", goal="g")
    assert top.resolution is not None
    assert bus.report()["by_outcome"].get(SubstateOutcome.EXHAUSTED.value, 0) >= 1


def test_an_exhausted_budget_is_reported_not_silently_defaulted():
    bus = reset_impasse_bus_for_test()
    substate = bus.raise_impasse(
        _tie(), organ="planner", goal="g", budget=SubstateBudget(depth=0, seconds=0.0, work=0)
    )
    assert substate.resolution.outcome is SubstateOutcome.EXHAUSTED


def test_a_handler_that_raises_is_a_datum_not_a_crash():
    bus = reset_impasse_bus_for_test()

    def broken(_ss):
        raise RuntimeError("solver is down")

    bus.register(ImpasseType.TIE, "broken", broken)
    bus.register(ImpasseType.TIE, "working", lambda ss: Resolution(SubstateOutcome.RESOLVED, choice="a"))
    substate = bus.raise_impasse(_tie(), organ="planner", goal="g")
    assert substate.resolution.outcome is SubstateOutcome.RESOLVED


def test_a_substate_resolution_can_be_a_preference_rather_than_a_choice():
    bus = reset_impasse_bus_for_test()
    bus.register(
        ImpasseType.TIE, "learn",
        lambda ss: Resolution(SubstateOutcome.LEARNED_PREFERENCE, preference=("a", "better_than", "b")),
    )
    substate = bus.raise_impasse(_tie(), organ="planner", goal="g")
    assert substate.resolution.decided
    assert substate.resolution.preference == ("a", "better_than", "b")


def test_the_substate_lands_on_the_event_dag_with_its_parent():
    from core.cognition.cognitive_event import Phase, reset_event_graph_for_test

    graph = reset_event_graph_for_test()
    bus = reset_impasse_bus_for_test()
    parent = graph.record(Phase.SELECT, "workspace", "tied")
    bus.register(ImpasseType.TIE, "h", lambda ss: Resolution(SubstateOutcome.RESOLVED, choice="a"))
    bus.raise_impasse(_tie(), organ="workspace", goal="g", parent_event=parent.seq)
    phases = [e.phase for e in graph]
    assert Phase.IMPASSE in phases and Phase.LEARN in phases


def test_substrate_disagreement_is_a_named_impasse_kind():
    """Card 056: implicit and explicit disagreeing is an occasion, not a blend."""
    bus = reset_impasse_bus_for_test()
    handled = []
    bus.register(
        ImpasseType.CONFLICT, "arbitrate",
        lambda ss: (handled.append(ss.context.get("substrates")), Resolution(SubstateOutcome.RESOLVED, choice="experiment"))[1],
    )
    substate = bus.raise_impasse(
        Impasse(ImpasseType.CONFLICT, "sig", ("neural_says_yes", "rule_says_no")),
        organ="dual_knowledge", goal="reconcile",
        context={"substrates": ["neural", "explicit"]},
    )
    assert substate.resolution.choice == "experiment"
    assert handled == [["neural", "explicit"]]


# ── transactions ──────────────────────────────────────────────────────────

def test_all_stores_commit_or_none_do():
    store: dict[str, str] = {}
    with transaction("e1") as txn:
        txn.join("memory", lambda: "ep", lambda v: store.__setitem__("memory", v), lambda v: store.pop("memory", None))
        txn.join("rules", lambda: "r", lambda v: store.__setitem__("rules", v), lambda v: store.pop("rules", None))
    assert store == {"memory": "ep", "rules": "r"}


def test_a_commit_failure_undoes_what_already_committed():
    store: dict[str, str] = {}

    def explode(_v):
        raise RuntimeError("disk full")

    with pytest.raises(TransactionAborted):
        with transaction("e2") as txn:
            txn.join("memory", lambda: "ep", lambda v: store.__setitem__("memory", v), lambda v: store.pop("memory", None))
            txn.join("rules", lambda: "r", explode, lambda v: None)
    assert store == {}


def test_a_prepare_failure_commits_nothing_at_all():
    store: dict[str, str] = {}

    def cannot(_=None):
        raise ValueError("no room")

    with pytest.raises(TransactionAborted, match="could not prepare"):
        with transaction("e3") as txn:
            txn.join("memory", lambda: "ep", lambda v: store.__setitem__("memory", v), lambda v: store.pop("memory", None))
            txn.join("rules", cannot, lambda v: None, lambda v: None)
    assert store == {}


def test_a_participant_with_no_rollback_is_refused():
    with pytest.raises(ValueError, match="makes the transaction a comment"):
        with transaction("e4") as txn:
            txn.join("memory", lambda: "ep", lambda v: None)


def test_a_failed_rollback_names_the_stranded_stores():
    store: dict[str, str] = {}

    def wont_undo(_v):
        raise RuntimeError("append-only")

    with pytest.raises(InconsistentRollback) as caught:
        with transaction("e5") as txn:
            txn.join("ledger", lambda: "x", lambda v: store.__setitem__("ledger", v), wont_undo)
            txn.join("rules", lambda: "r", lambda v: (_ for _ in ()).throw(RuntimeError("no")), lambda v: None)
    assert caught.value.stranded == ("ledger",)


def test_a_raising_body_commits_nothing():
    store: dict[str, str] = {}
    with pytest.raises(ZeroDivisionError):
        with transaction("e6") as txn:
            txn.join("memory", lambda: "ep", lambda v: store.__setitem__("memory", v), lambda v: None)
            1 / 0
    assert store == {}
    assert txn.state is TransactionState.ABORTED


# ── architecture invariants ───────────────────────────────────────────────

def test_the_architecture_invariants_run_and_report():
    report = architecture_report()
    assert report["checked"] >= 5
    assert isinstance(report["violations"], list)


def test_a_learner_fed_only_unverified_actions_trips_an_invariant():
    from core.cognition.action_receipt import (
        UnqualifiedTransition,
        qualified,
        reset_receipt_ledger_for_test,
        verify_transition,
    )

    reset_receipt_ledger_for_test()
    bad = verify_transition(action="a", target="t", authority="u", before={"x": 0}, after=None)
    for _ in range(25):
        with pytest.raises(UnqualifiedTransition):
            qualified(bad, learner="reckless")
    violations = [v["subject"] for v in architecture_report()["violations"]]
    assert "reckless" in violations
    reset_receipt_ledger_for_test()


def test_a_prohibited_candidate_cannot_win_a_decision():
    report = architecture_report()
    offenders = [v for v in report["violations"] if v["invariant"] == "cognition.one_authority_path"]
    assert not offenders, offenders


# ── the workspace actually reports through the bus ────────────────────────

def test_a_real_workspace_tie_reaches_the_impasse_bus():
    """Card 175's bar, against a live organ rather than a stub."""
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    bus = reset_impasse_bus_for_test()
    chosen = []

    def prefer_the_last_alphabetically(ss):
        chosen.append(ss.impasse.candidates)
        return Resolution(SubstateOutcome.RESOLVED, choice=sorted(ss.impasse.candidates)[-1])

    bus.register(ImpasseType.TIE, "alphabetical", prefer_the_last_alphabetically)

    workspace = GlobalWorkspace()
    for source in ("alpha", "beta", "gamma"):
        asyncio.run(workspace.submit(CognitiveCandidate(content=source, source=source, priority=0.7)))
    tied = tuple(sorted(c.source for c in workspace._candidates))
    winner = workspace._resolve_tie(tied)

    assert chosen and set(chosen[0]) == {"alpha", "beta", "gamma"}
    assert winner.source == "gamma"
    assert bus.report()["by_organ"]["global_workspace"] == 1


def test_the_workspace_still_decides_when_the_bus_resolves_nothing():
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    reset_impasse_bus_for_test()
    workspace = GlobalWorkspace()
    for source in ("alpha", "beta"):
        asyncio.run(workspace.submit(CognitiveCandidate(content=source, source=source, priority=0.7)))
    winner = workspace._resolve_tie(("alpha", "beta"))
    assert winner.source in {"alpha", "beta"}
