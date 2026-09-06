"""core/connectome/beyond.py — three things a nervous system cannot do to itself.

Everything else in this package copies biology. This module is where copying
stops being the point, because there are things evolution cannot reach that are
trivial once the substrate is a file rather than a body.

**Delays can be solved for instead of grown into.** A brain makes signals arrive
together by adjusting how thickly an axon is myelinated, which is slow,
approximate, local, and cannot see the constraint it is solving. The same
problem — hold each message so that everything a cell is waiting for lands at
once — is a linear least-squares system, and it has an exact answer. What comes
out is a per-edge delay budget and the jitter it removes, measured against
leaving it alone and against random delays.

**A tangle can be asked what it does.** H01 found axon whorls in human cortex:
axons wrapped into knots, sometimes around other cells, function unknown, and
unknowable, because you cannot untie one in a person and see what stops working.
Aura's whorls are the tight recurrent tangles in her call graph, and every one of
them can be untied in a copy. The census here finds them and hands each to the
lesion measurement, so the structure that is a mystery in human tissue has an
answer here.

**An edit can be taken back.** Development prunes and that is final. A rewiring
proposed here carries the objective it was proposed for, the measurement that
justified it and the inverse edit, so a change that turns out to be wrong is
undone rather than argued about.

None of the three is claimed to make Aura better on its own. Each ships with the
measurement that would show it did, and the delay compiler ships with the null
that would show it did not.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .lesion import measure_effect
from .topology import DiGraphView
from .types import Connection, ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Beyond")

__all__ = [
    "DelaySchedule",
    "compile_delays",
    "Whorl",
    "whorl_census",
    "explain_whorls",
    "Rewiring",
    "propose_rewiring",
    "apply_rewiring",
]


# ---------------------------------------------------------------------------
# 1. The delay compiler
# ---------------------------------------------------------------------------


@dataclass
class DelaySchedule:
    """A held-message budget per edge, and what holding them bought."""

    added_delay: dict[tuple[str, str], float]
    ready_time: dict[str, float]
    jitter_before: float
    jitter_after: float
    jitter_random: float
    total_added: float
    convergence_cells: int
    edges: int

    @property
    def improvement(self) -> float:
        if self.jitter_before <= 0:
            return 0.0
        return (self.jitter_before - self.jitter_after) / self.jitter_before

    def as_json(self) -> dict[str, Any]:
        return {
            "edges": self.edges,
            "convergence_cells": self.convergence_cells,
            "jitter_before": round(self.jitter_before, 5),
            "jitter_after": round(self.jitter_after, 5),
            "jitter_random_null": round(self.jitter_random, 5),
            "improvement": round(self.improvement, 5),
            "total_added_delay": round(self.total_added, 3),
            "mean_added_delay": round(self.total_added / self.edges, 4) if self.edges else 0.0,
            "verdict": (
                "solving beats leaving it alone and beats random holds"
                if self.jitter_after < self.jitter_before
                and self.jitter_after < self.jitter_random
                else "solving did not beat its null"
            ),
        }


def _natural_ready(
    graph: DiGraphView,
    latency: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    """When each cell first becomes ready if nothing waits for anything.

    A cell that fires on its first input is the uncompiled schedule, and it is
    the right baseline: it is what the system does now. Sources start at zero
    and the relaxation is a shortest path, so a cell reached down a short branch
    is ready long before the same cell's other input arrives. That gap is the
    jitter the compiler exists to remove.
    """
    import heapq

    ready: dict[str, float] = {}
    heap: list[tuple[float, str]] = []
    for node in graph.nodes:
        if not graph.inbound.get(node):
            ready[node] = 0.0
            heap.append((0.0, node))
    if not heap:
        first = min(graph.nodes) if graph.nodes else None
        if first is None:
            return {}
        ready[first] = 0.0
        heap.append((0.0, first))
    heapq.heapify(heap)
    while heap:
        time_at, node = heapq.heappop(heap)
        if time_at > ready.get(node, float("inf")):
            continue
        for post in graph.out.get(node, ()):
            arrival = time_at + float(latency.get((node, post), 1.0))
            if arrival < ready.get(post, float("inf")):
                ready[post] = arrival
                heapq.heappush(heap, (arrival, post))
    unreached = [node for node in graph.nodes if node not in ready]
    fallback = max(ready.values(), default=0.0)
    for node in unreached:
        ready[node] = fallback
    return ready


def _jitter(
    graph: DiGraphView,
    latency: Mapping[tuple[str, str], float],
    ready: Mapping[str, float],
    extra: Mapping[tuple[str, str], float] | None = None,
) -> tuple[float, int]:
    """Mean spread of arrival times across each cell's inputs.

    Cells with one input contribute nothing: there is nothing for a lone signal
    to arrive together with, and including them would dilute the measurement
    with the majority of the graph.
    """
    total = 0.0
    counted = 0
    for node in graph.nodes:
        inputs = graph.inbound.get(node, ())
        if len(inputs) < 2:
            continue
        arrivals = [
            ready.get(pre, 0.0)
            + latency.get((pre, node), 1.0)
            + (extra.get((pre, node), 0.0) if extra else 0.0)
            for pre in inputs
        ]
        total += statistics.pstdev(arrivals)
        counted += 1
    return (total / counted if counted else 0.0), counted


def compile_delays(
    snapshot: ConnectomeSnapshot,
    *,
    latency: Mapping[tuple[str, str], float] | None = None,
    seed: int = 0,
) -> DelaySchedule:
    """Solve for the hold on each edge that makes convergent inputs coincide.

    The system is ``ready[post] - ready[pre] = latency[pre, post]`` over every
    edge, which is overdetermined and has a least-squares solution: the same
    shape as a trophic level, with the constant one replaced by each edge's own
    cost. The residual is what each edge has to be held by. Negative residuals
    are edges that cannot be made to fit without slowing the whole path, so they
    are clipped to zero and the remaining jitter is reported honestly rather
    than being absorbed.

    Three schedules are compared on the same graph: the uncompiled one, where
    every cell fires on its first input; the solved one; and holds of the same
    average size drawn at random, which is the null that catches a result coming
    from adding delay rather than from adding the right delay.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE)
    edges = graph.edges()
    if not edges:
        return DelaySchedule({}, {}, 0.0, 0.0, 0.0, 0.0, 0, 0)
    latency = latency or {}
    costs = [float(latency.get(edge, 1.0)) for edge in edges]
    latency_map = dict(zip(edges, costs, strict=True))

    nodes = list(graph.nodes)
    index = {uid: i for i, uid in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, (pre, post) in enumerate(edges):
        rows.extend((row, row))
        cols.extend((index[post], index[pre]))
        data.extend((1.0, -1.0))
    incidence = csr_matrix(
        (data, (rows, cols)), shape=(len(edges), len(nodes)), dtype=np.float64
    )
    solution = lsqr(incidence, np.asarray(costs, dtype=np.float64), atol=1e-10, btol=1e-10)[0]
    solution = np.asarray(solution, dtype=np.float64)
    solution -= solution.min()
    ready = {uid: float(solution[index[uid]]) for uid in nodes}

    added: dict[tuple[str, str], float] = {}
    for (pre, post), cost in zip(edges, costs, strict=True):
        slack = ready[post] - ready[pre] - cost
        if slack > 0:
            added[(pre, post)] = slack

    natural = _natural_ready(graph, latency_map)
    jitter_before, convergence = _jitter(graph, latency_map, natural)
    jitter_after, _ = _jitter(graph, latency_map, ready, added)

    rng = random.Random(seed)
    scale = statistics.fmean(added.values()) if added else 1.0
    random_holds = {edge: rng.random() * 2.0 * scale for edge in edges}
    jitter_random, _ = _jitter(graph, latency_map, natural, random_holds)

    return DelaySchedule(
        added_delay=added,
        ready_time=ready,
        jitter_before=jitter_before,
        jitter_after=jitter_after,
        jitter_random=jitter_random,
        total_added=sum(added.values()),
        convergence_cells=convergence,
        edges=len(edges),
    )


# ---------------------------------------------------------------------------
# 2. Whorls, and asking them what they do
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Whorl:
    """A tight recurrent tangle: a strongly connected component above size one."""

    members: tuple[str, ...]
    names: tuple[str, ...]
    internal_edges: int
    external_in: int
    external_out: int

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def density(self) -> float:
        possible = self.size * (self.size - 1)
        return self.internal_edges / possible if possible else 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "density": round(self.density, 4),
            "internal_edges": self.internal_edges,
            "external_in": self.external_in,
            "external_out": self.external_out,
            "members": list(self.names[:8]),
        }


def whorl_census(
    snapshot: ConnectomeSnapshot,
    *,
    min_size: int = 2,
    limit: int = 50,
) -> list[Whorl]:
    """Every recurrent tangle in the graph, largest first.

    A strongly connected component of size two or more is a set of cells each
    of which can reach every other and come back. In a call graph that is mutual
    recursion, or a cycle of modules that each need the other, and it is the
    closest structural analogue to what H01 photographed and could not explain.
    """
    import networkx as nx

    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE)
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes)
    for pre, targets in graph.out.items():
        for post in targets:
            digraph.add_edge(pre, post)
    whorls: list[Whorl] = []
    for component in nx.strongly_connected_components(digraph):
        if len(component) < min_size:
            continue
        internal = sum(
            1 for pre in component for post in graph.out.get(pre, ()) if post in component
        )
        external_in = sum(
            1 for post in component for pre in graph.inbound.get(post, ()) if pre not in component
        )
        external_out = sum(
            1 for pre in component for post in graph.out.get(pre, ()) if post not in component
        )
        members = tuple(sorted(component))
        whorls.append(
            Whorl(
                members=members,
                names=tuple(
                    snapshot.units[uid].name if uid in snapshot.units else uid for uid in members
                ),
                internal_edges=internal,
                external_in=external_in,
                external_out=external_out,
            )
        )
    whorls.sort(key=lambda w: (-w.size, -w.internal_edges, w.members[0]))
    return whorls[:limit]


def explain_whorls(
    snapshot: ConnectomeSnapshot,
    whorls: Sequence[Whorl],
    *,
    limit: int = 8,
    null_samples: int = 6,
) -> list[dict[str, Any]]:
    """Untie each whorl in a copy and report what stopped working.

    This is the measurement H01 could not make. Each tangle is removed whole
    and the damage compared against removing an equal number of degree-matched
    cells, so a whorl that carries nothing says so.
    """
    rows: list[dict[str, Any]] = []
    for index, whorl in enumerate(whorls[:limit]):
        effect = measure_effect(
            snapshot, list(whorl.members), null_samples=null_samples, seed=index
        )
        rows.append({"whorl": whorl.as_json(), "lesion": effect.as_json()})
    rows.sort(key=lambda row: -row["lesion"]["excess_reach_loss"])
    return rows


# ---------------------------------------------------------------------------
# 3. Rewiring that can be taken back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rewiring:
    """One reversible topology change, with the reason it was proposed."""

    add: tuple[tuple[str, str], ...]
    remove: tuple[tuple[str, str], ...]
    objective: str
    predicted_gain: float
    measured_gain: float = 0.0

    def inverse(self) -> Rewiring:
        return Rewiring(
            add=self.remove,
            remove=self.add,
            objective=f"undo: {self.objective}",
            predicted_gain=-self.predicted_gain,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "add": [f"{a}->{b}" for a, b in self.add],
            "remove": [f"{a}->{b}" for a, b in self.remove],
            "objective": self.objective,
            "predicted_gain": round(self.predicted_gain, 5),
            "measured_gain": round(self.measured_gain, 5),
        }


def propose_rewiring(
    snapshot: ConnectomeSnapshot,
    *,
    objective: str = "within_layer_recurrence",
    candidates: int = 200,
    seed: int = 0,
) -> list[Rewiring]:
    """Propose edges whose absence a measurement named.

    The one objective implemented is the one the cortical comparison found:
    Aura's within-region recurrence runs an order of magnitude below cortex's.
    Candidates are pairs inside one module that are two hops apart and not
    joined, which is where a local loop would close without inventing a
    dependency across a package boundary.
    """
    if objective != "within_layer_recurrence":
        raise ValueError(f"no proposer for objective: {objective}")
    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE)
    rng = random.Random(seed)
    proposals: list[Rewiring] = []
    nodes = [uid for uid in graph.nodes if graph.out.get(uid)]
    rng.shuffle(nodes)
    for uid in nodes:
        unit = snapshot.units.get(uid)
        if unit is None:
            continue
        for middle in graph.out.get(uid, ()):
            for far in graph.out.get(middle, ()):
                if far == uid or far in graph.out.get(uid, ()):
                    continue
                far_unit = snapshot.units.get(far)
                if far_unit is None or far_unit.neuropil != unit.neuropil:
                    continue
                proposals.append(
                    Rewiring(
                        add=((far, uid),),
                        remove=(),
                        objective=objective,
                        predicted_gain=1.0 / max(1, len(graph.out.get(uid, ()))),
                    )
                )
                break
            if len(proposals) >= candidates:
                break
        if len(proposals) >= candidates:
            break
    return proposals


def apply_rewiring(
    snapshot: ConnectomeSnapshot,
    rewiring: Rewiring,
) -> tuple[ConnectomeSnapshot, Rewiring]:
    """Apply a change and hand back the edit that undoes it.

    The inverse is returned rather than stored, so the caller holds it and the
    change cannot become permanent by nobody keeping the receipt.
    """
    connections = dict(snapshot.connections)
    for pre, post in rewiring.remove:
        connections.pop((pre, post, str(EdgeKind.DRIVE)), None)
    for pre, post in rewiring.add:
        key = (pre, post, str(EdgeKind.DRIVE))
        if key not in connections:
            connections[key] = Connection(pre=pre, post=post, contacts=1, sign=1, kind=EdgeKind.DRIVE)
    changed = ConnectomeSnapshot(
        version=snapshot.version + 1,
        units=snapshot.units,
        connections=connections,
        neuropils=snapshot.neuropils,
        built_at=snapshot.built_at,
        source=snapshot.source,
        attrs=dict(snapshot.attrs),
    )
    changed.attrs["rewiring"] = rewiring.as_json()
    return changed, rewiring.inverse()
