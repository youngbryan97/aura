"""tests/test_morphogenesis_topology.py

The topology layer: graph transactions, the governor's bounds, the substrate
contract, lineage, motifs, and the defects that emptied the live registry.

Every test here names the failure it prevents. A test whose name says what it
checks and not what would break without it is a test nobody reads when it goes
red.
"""
from __future__ import annotations

import json

import pytest

from core.morphogenesis import proposal as P
from core.morphogenesis.governor import MorphBounds, MorphGovernor
from core.morphogenesis.graph import (
    EdgeType,
    GraphIntegrityError,
    MorphEdge,
    MorphGraph,
)
from core.morphogenesis.lineage import Lineage, LineageCycleError
from core.morphogenesis.motifs import MotifLibrary, demand_fingerprint
from core.morphogenesis.proposal import Decision, RiskClass, TransitionKind
from core.morphogenesis.registry import MorphogenesisRegistry
from core.morphogenesis.substrate import (
    LocalRuntimeSubstrate,
    SimulationSubstrate,
    SubstrateAdapter,
    SubstratePhysics,
    TransitionOutcome,
)
from core.morphogenesis.types import CellManifest
from core.morphogenesis.workload import (
    RoutedWorkload,
    WorkerProfile,
    task_families,
)


# ── graph ───────────────────────────────────────────────────────────────

def _seeded_graph() -> MorphGraph:
    graph = MorphGraph()

    def build(scratch):
        for node in ("a", "b", "c"):
            scratch.add_node(node)
        scratch.add_edge(MorphEdge("a", "b", EdgeType.DATA, port="x"))
        scratch.add_edge(MorphEdge("b", "c", EdgeType.DATA, port="y"))

    graph.transaction(build, cause="test")
    return graph


def test_a_failed_transaction_leaves_the_graph_byte_identical():
    """Half-applied topology is a state no cell was designed for."""
    graph = _seeded_graph()
    before = graph.snapshot()
    with pytest.raises(GraphIntegrityError):
        graph.transaction(
            lambda scratch: scratch.add_edge(MorphEdge("a", "nowhere", port="x")),
            cause="bad",
        )
    after = graph.snapshot()
    assert after.digest() == before.digest()
    assert graph.version == before.version


def test_rollback_advances_the_version_while_restoring_content():
    """A reader holding v9 must never be handed a second, different v9."""
    graph = _seeded_graph()
    original = graph.snapshot()
    graph.transaction(lambda s: s.remove_node("c"), cause="cut")
    assert graph.node_count == 2
    graph.restore(original, cause="rollback")
    assert graph.node_count == 3
    assert graph.snapshot().digest() == original.digest()
    assert graph.version > original.version


def test_a_self_binding_is_refused():
    graph = _seeded_graph()
    with pytest.raises(GraphIntegrityError):
        graph.transaction(
            lambda s: s.add_edge(MorphEdge("a", "a", port="x")), cause="self"
        )


def test_the_port_contract_refuses_a_binding_neither_end_could_carry():
    graph = MorphGraph()
    contract = {
        "a": (frozenset({"x"}), frozenset()),
        "b": (frozenset(), frozenset({"y"})),
    }
    with pytest.raises(GraphIntegrityError):
        graph.transaction(
            lambda s: [s.add_node("a"), s.add_node("b"),
                       s.add_edge(MorphEdge("a", "b", port="x"))],
            cause="mismatch",
            port_contract=contract,
        )


def test_serialization_round_trips_and_is_deterministic():
    graph = _seeded_graph()
    twin = MorphGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert twin.snapshot().digest() == graph.snapshot().digest()
    assert twin.version == graph.version


def test_components_and_reachability_track_the_edges():
    graph = _seeded_graph()
    assert len(graph.components()) == 1
    assert graph.path_exists("a", "c")
    assert not graph.path_exists("c", "a")
    edge = next(e for e in graph.edges() if e.source == "b")
    graph.transaction(lambda s: s.remove_edge(edge.key), cause="cut")
    assert len(graph.components()) == 2
    assert not graph.path_exists("a", "c")


# ── topology is causal ──────────────────────────────────────────────────

def _routed(edges):
    graph = MorphGraph()
    workload = RoutedWorkload(graph, seed=1)
    for cell_id, capabilities in (
        ("c_in", ("ingest",)),
        ("c_ret", ("retrieve", "recall")),
        ("c_syn", ("synthesize",)),
        ("c_out", ("emit",)),
    ):
        workload.add_worker(WorkerProfile(cell_id, capabilities))

    def build(scratch):
        for cell_id in ("c_in", "c_ret", "c_syn", "c_out"):
            scratch.add_node(cell_id)
        for source, target, port in edges:
            scratch.add_edge(MorphEdge(source, target, EdgeType.DATA, port=port))

    graph.transaction(build, cause="wire", port_contract=workload.port_contract())
    workload.ingress = ["c_in"]
    for _ in range(20):
        workload.admit(task_families()["memory_heavy"])
    for _ in range(60):
        workload.step()
    return workload


def test_cutting_a_binding_changes_what_the_workload_computes():
    """If two shapes compute the same thing, the topology is decoration."""
    full = [
        ("c_in", "c_ret", "retrieve"), ("c_in", "c_ret", "recall"),
        ("c_ret", "c_syn", "synthesize"), ("c_syn", "c_out", "emit"),
    ]
    connected = _routed(full)
    severed = _routed([e for e in full if e != ("c_ret", "c_syn", "synthesize")])
    assert connected.metrics.completion_rate == 1.0
    assert severed.metrics.completion_rate == 0.0
    assert connected.signature() != severed.signature()


def test_a_longer_path_costs_more_than_a_shorter_one():
    short = _routed([
        ("c_in", "c_ret", "retrieve"), ("c_in", "c_ret", "recall"),
        ("c_ret", "c_syn", "synthesize"), ("c_syn", "c_out", "emit"),
    ])
    long_way = _routed([
        ("c_in", "c_ret", "retrieve"), ("c_in", "c_ret", "recall"),
        ("c_ret", "c_out", "emit"), ("c_out", "c_syn", "synthesize"),
        ("c_syn", "c_out", "emit"),
    ])
    assert long_way.metrics.mean_hops > short.metrics.mean_hops


def test_sojourn_counts_waiting_and_latency_does_not():
    """Hop cost alone scores congestion and free flow identically."""
    graph = MorphGraph()
    workload = RoutedWorkload(graph, seed=3)
    workload.add_worker(WorkerProfile("solo", ("ingest",), service_rate=1))
    graph.transaction(lambda s: s.add_node("solo"), cause="one")
    workload.ingress = ["solo"]
    for _ in range(6):
        workload.admit(("ingest",))
    for _ in range(12):
        workload.step()
    assert workload.metrics.completed == 6
    assert workload.metrics.mean_latency == 0.0
    assert workload.metrics.mean_sojourn > 0.0


# ── substrate ───────────────────────────────────────────────────────────

def test_both_substrates_satisfy_the_adapter_contract():
    assert isinstance(SimulationSubstrate(), SubstrateAdapter)
    assert isinstance(LocalRuntimeSubstrate(), SubstrateAdapter)


def test_a_migrating_cell_is_unreachable_and_cannot_be_bound():
    """Software written against instant, infallible binding does not survive
    contact with anything that has to physically move."""
    substrate = SimulationSubstrate(
        seed=1, physics=SubstratePhysics(migrate_ms=4000.0, migrate_blackout_s=4.0)
    )
    substrate.place("a")
    substrate.place("b")
    result = substrate.migrate("a", "bench_2")
    assert result.ok
    assert not substrate.reachable("a")
    refused = substrate.bind(MorphEdge("a", "b", port="x"))
    assert refused.outcome is TransitionOutcome.REFUSED
    substrate.advance(5.0)
    assert substrate.reachable("a")


def test_the_in_process_substrate_refuses_to_report_a_move_it_cannot_make():
    substrate = LocalRuntimeSubstrate(locus="aura_main")
    substrate.place("a")
    result = substrate.migrate("a", "another_host")
    assert result.outcome is TransitionOutcome.REFUSED
    assert "cannot leave this process" in result.detail


def test_a_partial_failure_leaves_the_world_changed():
    substrate = SimulationSubstrate(
        seed=5, physics=SubstratePhysics(failure_rate=1.0, partial_failure_share=1.0)
    )
    substrate.place("a")
    substrate.place("b")
    edge = MorphEdge("a", "b", port="x")
    result = substrate.bind(edge)
    assert not result.ok
    assert result.partial
    assert substrate.bound(edge), "a partial bind latched without handshaking"


# ── governor ────────────────────────────────────────────────────────────

def _governor(**bound_overrides):
    graph = _seeded_graph()
    substrate = SimulationSubstrate(seed=7)
    for node in graph.nodes():
        substrate.place(node)
    bounds = MorphBounds(cooldown_s=0.0, **bound_overrides)
    governor = MorphGovernor(
        graph, substrate, bounds=bounds, require_governance=True, emit_receipts=False
    )
    for node in graph.nodes():
        governor.lineage.seed(node)
    governor.credit("a", 50.0)
    return graph, substrate, governor


def test_a_critical_change_is_refused_without_governance():
    """The absence of a check is not a passed check."""
    _, _, governor = _governor()
    transaction = governor.adjudicate(P.retire("c", proposer="a"))
    assert transaction.decision is Decision.REJECTED
    assert "governance" in transaction.reason


def test_a_claimed_benefit_does_not_survive_measurement():
    _, _, governor = _governor()
    governor.shadow_evaluator = lambda graph, proposal=None: 0.9 - 0.2 * graph.node_count
    transaction = governor.adjudicate(
        P.spawn({"name": "w", "capabilities": ["solve"]}, proposer="a", parent="a", benefit=1.0)
    )
    assert transaction.decision is Decision.REJECTED
    assert "shadow" in transaction.reason
    assert transaction.shadow_score is not None


def test_an_unmeasurable_change_is_refused_rather_than_approved():
    _, _, governor = _governor()
    governor.shadow_evaluator = lambda graph, proposal=None: None
    transaction = governor.adjudicate(
        P.spawn({"name": "w", "capabilities": ["plan"]}, proposer="a", parent="a", benefit=1.0)
    )
    assert transaction.decision is Decision.REJECTED
    assert "could not measure" in transaction.reason


def test_a_change_that_would_sever_the_population_is_refused():
    graph, _, governor = _governor()
    governor.shadow_evaluator = lambda graph, proposal=None: 1.0
    edge = next(e for e in graph.edges() if e.source == "b")
    transaction = governor.adjudicate(P.unbind(edge, proposer="a", benefit=1.0))
    assert transaction.decision is Decision.REJECTED
    assert "pieces" in transaction.reason


def test_a_rate_limit_defers_rather_than_rejects():
    """DEFERRED is not REJECTED; a caller that conflates them discards work
    that was perfectly well formed."""
    _, _, governor = _governor(max_transitions_per_window=1)
    governor.adjudicate(P.bind("a", "c", "x", proposer="a", benefit=0.5))
    second = governor.adjudicate(P.bind("c", "a", "y", proposer="a", benefit=0.5))
    assert second.decision is Decision.DEFERRED
    assert "rate" in second.reason


def test_a_substrate_failure_rolls_back_the_graph_and_the_world():
    graph = _seeded_graph()
    substrate = SimulationSubstrate(
        seed=3, physics=SubstratePhysics(failure_rate=1.0, partial_failure_share=1.0)
    )
    for node in graph.nodes():
        substrate.place(node)
    governor = MorphGovernor(
        graph, substrate, bounds=MorphBounds(cooldown_s=0.0),
        require_governance=False, emit_receipts=False,
    )
    for node in graph.nodes():
        governor.lineage.seed(node)
    governor.credit("a", 10.0)
    before = graph.snapshot()
    proposal = P.bind("a", "c", "q", proposer="a")
    transaction = governor.adjudicate(proposal)
    assert transaction.decision is Decision.ROLLED_BACK
    assert graph.snapshot().digest() == before.digest()
    assert not substrate.bound(proposal.transitions[0].edge), "a latch nobody owns"


def test_a_transition_that_cannot_reverse_itself_is_refused_before_anything_else():
    proposal = P.MorphProposal(
        proposer="a",
        transitions=(P.MorphTransition(kind=TransitionKind.MIGRATE, subject="a"),),
    )
    assert "placement" in proposal.validate()


def test_specializing_is_measured_rather_than_waved_through():
    """It trades capability at one port against every other port."""
    proposal = P.specialize("a", "solve", proposer="a")
    assert proposal.risk is RiskClass.ELEVATED


def test_a_reversal_inside_the_window_is_refused():
    _, _, governor = _governor(reversal_window_s=999.0)
    applied = governor.adjudicate(P.bind("a", "c", "x", proposer="a", benefit=0.5))
    assert applied.decision is Decision.APPLIED
    edge = next(e for e in governor.graph.edges() if e.target == "c" and e.source == "a")
    undo = governor.adjudicate(P.unbind(edge, proposer="a", benefit=0.5))
    assert undo.decision is Decision.REJECTED
    assert "reversal" in undo.reason


def test_a_world_change_voids_the_reversal_bet():
    """Undoing a decision whose premise just died is repair, not thrash."""
    _, _, governor = _governor(reversal_window_s=999.0)
    governor.adjudicate(P.bind("a", "c", "x", proposer="a", benefit=0.5))
    governor.invalidate_reversal_history(cells=["a"], reason="test")
    edge = next(e for e in governor.graph.edges() if e.target == "c" and e.source == "a")
    undo = governor.adjudicate(P.unbind(edge, proposer="a", benefit=0.5))
    assert undo.decision is not Decision.REJECTED or "reversal" not in undo.reason


def test_the_population_cannot_pass_its_own_cap():
    _, _, governor = _governor(max_cells=3)
    transaction = governor.adjudicate(
        P.spawn({"name": "w", "capabilities": ["solve"]}, proposer="a", parent="a")
    )
    assert transaction.decision is Decision.REJECTED
    assert "population" in transaction.reason


def test_spawn_depth_is_bounded():
    _, _, governor = _governor(max_spawn_depth=1)
    governor.lineage.max_generation = 1
    governor.lineage.record_birth("child", parent_id="a")
    governor.credit("child", 20.0)
    transaction = governor.adjudicate(
        P.spawn({"name": "w", "capabilities": ["solve"]}, proposer="child", parent="child")
    )
    assert transaction.decision is Decision.REJECTED
    assert "depth" in transaction.reason


# ── lineage ─────────────────────────────────────────────────────────────

def test_a_cell_cannot_become_its_own_ancestor():
    lineage = Lineage()
    lineage.seed("a")
    lineage.record_birth("b", parent_id="a")
    with pytest.raises(LineageCycleError):
        lineage.record_birth("a", parent_id="b")
    assert lineage.acyclic()


# ── motifs ──────────────────────────────────────────────────────────────

def test_a_motif_earns_nothing_from_being_used():
    """Credit for use is how a library of guesses reads as experience."""
    graph = _seeded_graph()
    library = MotifLibrary()
    motif = library.learn(
        name="m", demand={"plan": 6, "solve": 4}, graph=graph,
        capabilities={"a": ["plan"], "b": ["solve"]},
    )
    for _ in range(10):
        library.note_application(motif.motif_id)
    assert motif.applications == 10
    assert motif.credit == 0.0
    assert not motif.credited
    assert library.select({"plan": 6, "solve": 4}) is None


def test_a_motif_earns_credit_only_by_beating_its_own_absence():
    graph = _seeded_graph()
    library = MotifLibrary()
    motif = library.learn(
        name="m", demand={"plan": 6, "solve": 4}, graph=graph,
        capabilities={"a": ["plan"], "b": ["solve"]},
    )
    library.record_trial(motif.motif_id, with_motif=0.9, without_motif=0.6)
    library.record_trial(motif.motif_id, with_motif=0.8, without_motif=0.7)
    assert motif.credited
    assert library.select({"plan": 6, "solve": 4}) is motif


def test_a_losing_motif_is_pruned():
    graph = _seeded_graph()
    library = MotifLibrary()
    motif = library.learn(
        name="bad", demand={"recall": 9}, graph=graph, capabilities={"a": ["recall"]}
    )
    for _ in range(4):
        library.record_trial(motif.motif_id, with_motif=0.3, without_motif=0.6)
    assert library.prune_uncredited() == [motif.motif_id]
    assert len(library) == 0


def test_a_motif_records_what_development_added_not_the_founding_kit():
    """Recording the whole final population makes every motif from one
    starting kit come out alike, and the library cannot be wrong about
    anything — nor right."""
    graph = _seeded_graph()
    library = MotifLibrary()
    founders = {"a": ["plan"], "b": ["solve"]}
    motif = library.learn(
        name="grown", demand={"plan": 4, "verify": 6}, graph=graph,
        capabilities={"a": ["plan"], "b": ["solve"], "c": ["verify"]},
        baseline_capabilities=founders,
    )
    assert "verify" in motif.seed_capabilities
    assert "solve" not in motif.seed_capabilities


def test_a_demand_fingerprint_ignores_scale():
    assert demand_fingerprint({"a": 6, "b": 4}) == demand_fingerprint({"a": 60, "b": 40})


# ── registry: the defects that emptied production ───────────────────────

def _registry(tmp_path, handler):
    registry = MorphogenesisRegistry(root=tmp_path)
    registry.register_cell(CellManifest(name="svc", subsystem="x"), handler=handler)
    return registry


def test_loading_keeps_the_handler_registered_before_the_load(tmp_path):
    """from_dict builds handler=None, so every cell went inert on boot two."""
    def handler(*_args, **_kwargs):
        return {"actions": []}

    first = _registry(tmp_path, handler)
    cell_id = next(iter(first.cells))
    first.cells[cell_id].state.activation_count = 42
    first.save()

    second = _registry(tmp_path, handler)
    assert second.load()
    assert second.cells[cell_id].handler is handler
    assert second.cells[cell_id].state.activation_count == 42


def test_an_empty_registry_is_not_written_over_a_populated_one(tmp_path):
    """Any bare runtime constructed anywhere used to overwrite the live
    population with {} on shutdown."""
    def handler(*_args, **_kwargs):
        return {"actions": []}

    _registry(tmp_path, handler).save()
    MorphogenesisRegistry(root=tmp_path).save()
    stored = json.loads((tmp_path / "morphogenesis_state.json").read_text())
    assert stored["payload"]["cells"], "an empty save clobbered a populated registry"


def test_a_corrupt_state_file_degrades_instead_of_raising(tmp_path):
    def handler(*_args, **_kwargs):
        return {"actions": []}

    registry = _registry(tmp_path, handler)
    registry.save()
    (tmp_path / "morphogenesis_state.json").write_text("{not json", encoding="utf-8")
    reloaded = _registry(tmp_path, handler)
    assert reloaded.load() is False
    assert len(reloaded.cells) == 1
