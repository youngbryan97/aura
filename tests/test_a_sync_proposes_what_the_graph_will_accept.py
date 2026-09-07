"""The population sync must propose a topology its own budget will admit.

Observed live on 2026-09-07, at 1.3 failures a second for the life of the
process: `_attachments_for` capped each arriving cell at four bindings but
added an uncapped reverse edge for each, so a peer several cells arrived
against reached out-degree 18 against a budget of 16. `MorphGraph._validate`
refused the transaction, which discarded the node additions and removals with
it, and the next tick proposed exactly the same thing. Nothing recovered and
nothing escalated; the runtime simply produced a fault forever.

A proposer that does not know the constraint that judges it cannot propose
something admissible, and a transaction that fails as a whole takes the
legitimate work down with the illegitimate part.
"""

from __future__ import annotations

import pytest

from core.morphogenesis.graph import GraphIntegrityError
from core.morphogenesis.runtime import MorphogeneticRuntime
from core.morphogenesis.types import CellManifest, MorphogenesisConfig


def _runtime_with_shared_peer(arrivals: int) -> tuple[MorphogeneticRuntime, set[str]]:
    """One peer that every arriving cell in the subsystem will attach to."""
    runtime = MorphogeneticRuntime(config=MorphogenesisConfig(enabled=False))
    runtime.registry.register_cell(CellManifest(name="peer", subsystem="shared"))
    names = {"peer"}
    for index in range(arrivals):
        name = f"arrival_{index:02d}"
        runtime.registry.register_cell(CellManifest(name=name, subsystem="shared"))
        names.add(name)
    live = {cell.cell_id for cell in runtime.registry.active_cells()}
    return runtime, live


def test_attachments_never_exceed_the_degree_budget() -> None:
    runtime, live = _runtime_with_shared_peer(arrivals=20)
    edges = runtime._attachments_for(set(live), live)

    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    for edge in edges:
        out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    over_out = {k: v for k, v in out_degree.items() if v > runtime.graph.max_out_degree}
    over_in = {k: v for k, v in in_degree.items() if v > runtime.graph.max_in_degree}
    assert not over_out, f"proposed an out-degree the graph refuses: {over_out}"
    assert not over_in, f"proposed an in-degree the graph refuses: {over_in}"


def test_the_transaction_the_sync_builds_actually_commits() -> None:
    """The behaviour, not the arithmetic: the graph must accept it."""
    runtime, live = _runtime_with_shared_peer(arrivals=20)
    arrived = set(live)
    edges = runtime._attachments_for(arrived, live)

    def sync(scratch):
        for cell_id in sorted(arrived):
            scratch.add_node(cell_id)
        for edge in edges:
            scratch.add_edge(edge)

    try:
        runtime.graph.transaction(sync, cause="test:population_sync")
    except GraphIntegrityError as error:  # pragma: no cover - the defect
        pytest.fail(f"the sync proposed a graph its own validator refuses: {error}")

    assert set(runtime.graph.nodes()) >= arrived


def test_a_second_sync_never_re_proposes_a_binding_the_graph_holds() -> None:
    """The tick that follows must add to the topology, never duplicate it."""
    runtime, live = _runtime_with_shared_peer(arrivals=20)
    arrived = set(live)
    first = runtime._attachments_for(arrived, live)

    def sync(scratch):
        for cell_id in sorted(arrived):
            scratch.add_node(cell_id)
        for edge in first:
            scratch.add_edge(edge)

    runtime.graph.transaction(sync, cause="test:first")
    held = {(edge.source, edge.target, edge.edge_type) for edge in runtime.graph.edges()}
    second = runtime._attachments_for(arrived, live)
    duplicated = [
        edge for edge in second
        if (edge.source, edge.target, edge.edge_type) in held
    ]
    assert not duplicated, f"re-proposed {len(duplicated)} bindings already held"

    def resync(scratch):
        for edge in second:
            scratch.add_edge(edge)

    runtime.graph.transaction(resync, cause="test:second")


def test_a_settled_population_proposes_nothing() -> None:
    """On a real tick nothing has arrived, so nothing is proposed."""
    runtime, live = _runtime_with_shared_peer(arrivals=20)
    assert runtime._attachments_for(set(), live) == []


def test_connectivity_survives_the_budget() -> None:
    """Clipping the reverse edge must not leave an arriving cell orphaned."""
    runtime, live = _runtime_with_shared_peer(arrivals=20)
    edges = runtime._attachments_for(set(live), live)
    touched = {edge.source for edge in edges} | {edge.target for edge in edges}
    assert live <= touched, f"unattached after the budget: {sorted(live - touched)}"
