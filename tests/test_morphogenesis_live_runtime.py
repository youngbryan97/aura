"""tests/test_morphogenesis_live_runtime.py

The layer as the running instance actually uses it.

Every test here corresponds to something that was broken when the live path
was first wired and would have shipped silently: the offline suite passed
throughout. A layer whose sandbox is green and whose runtime is inert is the
failure mode this file exists to prevent.
"""
from __future__ import annotations

import asyncio

import pytest

from core.morphogenesis.governor import MorphBounds, MorphGovernor
from core.morphogenesis.graph import MorphGraph
from core.morphogenesis.proposal import Decision
from core.morphogenesis import proposal as P
from core.morphogenesis.integration import register_morphogenesis_services
from core.morphogenesis.live_policy import LiveCoverageEvaluator, LiveObserverPolicy
from core.morphogenesis.registry import MorphogenesisRegistry
from core.morphogenesis.runtime import MorphogeneticRuntime
from core.morphogenesis.substrate import LocalRuntimeSubstrate
from core.morphogenesis.types import (
    CellManifest,
    CellRole,
    MorphogenesisConfig,
    MorphogenSignal,
    SignalKind,
)


def _config(**overrides):
    base = dict(
        tick_interval_s=0.01,
        adaptive_immunity_bridge=False,
        propose_every_ticks=4,
        energy_credit_per_tick=0.08,
    )
    base.update(overrides)
    return MorphogenesisConfig(**base)


def _runtime(tmp_path, **overrides) -> MorphogeneticRuntime:
    config = _config(**overrides)
    return MorphogeneticRuntime(
        config=config, registry=MorphogenesisRegistry(root=tmp_path, config=config)
    )


async def _drive(runtime, *, ticks: int, subsystem: str = "", intensity: float = 1.0):
    for _ in range(ticks):
        if subsystem:
            runtime.emit_signal(MorphogenSignal(
                kind=SignalKind.ERROR, source="test", subsystem=subsystem,
                intensity=intensity, payload={"error": "injected"}, ttl_ticks=50,
            ))
        await runtime.tick()


def _sensor(name: str, subsystem: str) -> CellManifest:
    return CellManifest(
        name=name, role=CellRole.SENSOR, subsystem=subsystem,
        capabilities=[subsystem], consumes=["error", "danger", "repair"],
        emits=["repair", "error"],
    )


# ── the fail-open ───────────────────────────────────────────────────────

def test_a_governor_with_no_evaluator_refuses_every_non_routine_change():
    """It used to skip the shadow block entirely and commit them unmeasured.

    Not a missing feature — a hole that read as "no shadow configured" and
    applied every spawn, migration and specialization without measuring one.
    """
    graph = MorphGraph()
    substrate = LocalRuntimeSubstrate()
    graph.transaction(lambda s: [s.add_node("a"), s.add_node("b")], cause="seed")
    substrate.place("a")
    substrate.place("b")
    governor = MorphGovernor(
        graph, substrate, bounds=MorphBounds(cooldown_s=0.0),
        shadow_evaluator=None, require_governance=True, emit_receipts=False,
    )
    governor.lineage.seed("a")
    governor.credit("a", 50.0)

    for proposal in (
        P.spawn({"name": "x", "capabilities": ["q"]}, proposer="a", parent="a", benefit=1.0),
        P.specialize("b", "q", proposer="a", benefit=1.0),
        P.retire("b", proposer="a"),
    ):
        transaction = governor.adjudicate(proposal)
        assert transaction.decision is Decision.REJECTED
        assert "shadow evaluator" in transaction.reason
    assert graph.node_count == 2

    # A routine wiring change still goes through.
    assert governor.adjudicate(P.bind("a", "b", "q", proposer="a")).decision is Decision.APPLIED


def test_energy_is_capped_so_a_quiet_cell_cannot_bank_a_burst():
    graph = MorphGraph()
    governor = MorphGovernor(graph, LocalRuntimeSubstrate(), emit_receipts=False)
    for _ in range(1000):
        governor.credit("a", 1.0)
    assert governor.energy("a") == governor.max_banked_energy


# ── the live loop ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_boot_seeds_a_connected_topology(tmp_path):
    runtime = _runtime(tmp_path)
    register_morphogenesis_services(runtime)
    await _drive(runtime, ticks=3)
    assert runtime.graph.node_count >= 12
    assert runtime.graph.edge_count > 0
    assert len(runtime.graph.components()) == 1, "a healthy boot must not read as partitioned"
    assert runtime.substrate.health()["bindings"] == runtime.graph.edge_count


@pytest.mark.asyncio
async def test_the_population_covers_a_subsystem_nothing_could_reach(tmp_path):
    """The live policy had no callers, so the topology never changed at all."""
    runtime = _runtime(tmp_path, propose_every_ticks=5, energy_credit_per_tick=0.05)
    register_morphogenesis_services(runtime)
    runtime.registry.register_cell(_sensor("orphan_service", "orphanage"))
    await _drive(runtime, ticks=40, subsystem="orphanage")

    assert runtime.governor.stats.proposals_seen > 0, "the live policy never proposed"
    assert runtime.governor.stats.applied > 0
    assert len(runtime.graph.components()) == 1


@pytest.mark.asyncio
async def test_a_population_with_nothing_that_can_act_grows_one(tmp_path):
    """The ELEVATED path, measured. Baseline coverage is zero and has to be
    reported as zero rather than as unmeasurable, or the one state where
    growing is obviously right refuses every proposal."""
    runtime = _runtime(tmp_path)
    for name, subsystem in (("s1", "alpha"), ("s2", "beta")):
        runtime.registry.register_cell(_sensor(name, subsystem))
    await _drive(runtime, ticks=50, subsystem="alpha")

    grown = [c for c in runtime.registry.active_cells() if c.manifest.name.startswith("observer_")]
    assert grown, "nothing grew where nothing could act"
    record = runtime.lineage.get(grown[0].cell_id)
    assert record is not None and record.generation == 1
    assert record.parent_id

    applied = [t for t in runtime.governor.transactions if t.committed]
    measured = [t for t in applied if t.shadow_score is not None]
    assert measured, "an ELEVATED change committed without a measurement"
    for transaction in measured:
        assert transaction.shadow_score > transaction.baseline_score


@pytest.mark.asyncio
async def test_a_grown_cell_becomes_a_real_registry_cell_with_a_handler(tmp_path):
    """Without the spawn hook the governor committed a graph node and a birth
    while the registry never heard, and the next sync deleted it again."""
    runtime = _runtime(tmp_path)
    for name, subsystem in (("s1", "alpha"), ("s2", "beta")):
        runtime.registry.register_cell(_sensor(name, subsystem))
    await _drive(runtime, ticks=50, subsystem="alpha")

    grown = [c for c in runtime.registry.active_cells() if c.manifest.name.startswith("observer_")]
    assert grown
    cell = grown[0]
    assert cell.handler is not None, "a grown cell that cannot run is a graph node with a name"
    assert runtime.graph.has_node(cell.cell_id)
    assert runtime.substrate.placement(cell.cell_id) is not None

    # It survives the next sync rather than being deleted as unknown.
    before = runtime.graph.node_count
    await _drive(runtime, ticks=3)
    assert runtime.graph.node_count >= before


@pytest.mark.asyncio
async def test_an_arriving_cell_is_bound_to_what_it_is_related_to(tmp_path):
    """Every organ the stabilizer formalized used to arrive unbound, so a
    healthy runtime tripped the partition alarm."""
    runtime = _runtime(tmp_path)
    register_morphogenesis_services(runtime)
    await _drive(runtime, ticks=2)
    runtime.registry.register_cell(CellManifest(
        name="late_arrival", role=CellRole.SENSOR, subsystem="memory",
        capabilities=["memory"], consumes=["error", "repair"], emits=["repair"],
    ))
    await _drive(runtime, ticks=3)
    late = next(c for c in runtime.registry.active_cells() if c.manifest.name == "late_arrival")
    assert runtime.graph.in_edges(late.cell_id) or runtime.graph.out_edges(late.cell_id)
    assert len(runtime.graph.components()) == 1


@pytest.mark.asyncio
async def test_the_developed_shape_survives_a_restart(tmp_path):
    """Graph, lineage and motifs were memory-only, so every reboot threw away
    the shape the population had developed and who descended from whom."""
    first = _runtime(tmp_path)
    register_morphogenesis_services(first)
    await _drive(first, ticks=4)
    version, nodes, edges = first.graph.version, first.graph.node_count, first.graph.edge_count
    tracked = first.lineage.status()["tracked"]
    first.registry.save()

    second = _runtime(tmp_path)
    register_morphogenesis_services(second)
    second.registry.load()
    assert second.graph.node_count == nodes
    assert second.graph.edge_count == edges
    assert second.lineage.status()["tracked"] >= tracked
    assert second.graph.version > 0


@pytest.mark.asyncio
async def test_telemetry_and_invariants_are_live_after_a_tick(tmp_path):
    from core.container import ServiceContainer
    from core.fsw.telemetry_dictionary import channel_value
    from core.morphogenesis import telemetry
    import core.morphogenesis.invariants  # noqa: F401 — registers them
    from core.verify.invariants import verify

    runtime = _runtime(tmp_path, telemetry_every_ticks=1)
    register_morphogenesis_services(runtime)
    telemetry.declare()
    await _drive(runtime, ticks=3)

    assert channel_value(telemetry.CHANNEL_GRAPH_VERSION) is not None
    assert channel_value(telemetry.CHANNEL_CELLS) is not None

    ServiceContainer.register_instance("morphogenetic_runtime", runtime)
    try:
        report = verify("morphogenesis", record=False)
        assert report.checked == 9
        assert not report.violations, [str(v) for v in report.violations]
    finally:
        ServiceContainer.register_instance("morphogenetic_runtime", None)


@pytest.mark.asyncio
async def test_the_live_governor_keeps_governance_armed(tmp_path):
    runtime = _runtime(tmp_path)
    assert runtime.governor.require_governance is True
    assert runtime.governor.shadow_evaluator is not None
    assert isinstance(runtime.governor.shadow_evaluator, LiveCoverageEvaluator)


@pytest.mark.asyncio
async def test_a_quiet_system_is_not_a_reason_to_reorganise(tmp_path):
    """Coverage is undefined when nothing is troubled, and the evaluator
    refuses rather than returning a constant that would pass anything."""
    runtime = _runtime(tmp_path)
    register_morphogenesis_services(runtime)
    await _drive(runtime, ticks=3)
    evaluator = LiveCoverageEvaluator(runtime)
    assert evaluator(runtime.graph, None) is None


def test_the_live_policy_only_picks_a_port_both_ends_can_carry():
    """It assumed 'repair'; a binding satisfying one end is refused at commit,
    after the proposal has already cost a measurement."""
    source = CellManifest(name="a", emits=["repair", "growth"], capabilities=["x"])
    target = CellManifest(name="b", consumes=["danger", "repair"], capabilities=["y"])

    class _Cell:
        def __init__(self, manifest):
            self.manifest = manifest

    port = LiveObserverPolicy._port_between(_Cell(source), _Cell(target))
    assert port == "repair"

    mismatched = CellManifest(name="c", consumes=["nothing_shared"], capabilities=["z"])
    assert LiveObserverPolicy._port_between(_Cell(source), _Cell(mismatched)) == ""
