"""core/connectome/spine.py — from sense to action, and the narrow place in between.

The male CNS connectome is the first complete one that runs from eyes to legs:
brain, optic lobes and ventral nerve cord in a single graph with the neck
connective intact. The finding that matters for anything trying to act in the
world is what sits in that neck. The ascending and descending neurons are few,
and the paper's flow analysis shows they carry a disproportionate share of the
traffic and do not behave like wires. They converge on inputs and integrate
them. The brain does not reach the body directly; it reaches it through a
handful of cells that decide what the body hears.

That is a claim Aura can be measured against rather than decorated with. Both
ends of her axis are measurable without anyone declaring them:

* An **afferent** cell is one that calls out of the process to read the world.
  A screen grab, a camera frame, a socket receive.
* An **efferent** cell is one that calls out of the process to change it. A
  subprocess, a click, a file removed.

Everything between them is the nervous system, and the question is whether it
has a neck. Three things get measured:

**Utilisation.** How much of the sense-to-action traffic each cell carries,
counted over shortest paths from every afferent cell to every efferent one.

**Concentration.** How few cells carry most of it. A flat curve means every
path is its own private wire from perception to action, which is the shape that
lets one subsystem act without the rest of the system knowing.

**Integration.** Whether the cells at the top of the utilisation curve converge
on their inputs or merely pass one input along. A relay has in-degree near one.
An integrator has many, and the fly's neck is made of integrators.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .topology import DiGraphView
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Spine")

__all__ = [
    "SpineReport",
    "afferent_cells",
    "efferent_cells",
    "path_utilisation",
    "analyse_spine",
]


def afferent_cells(snapshot: ConnectomeSnapshot) -> list[str]:
    """Cells that read the world from outside the process."""
    return sorted(
        uid for uid, unit in snapshot.units.items() if int(unit.attrs.get("afferent", 0)) > 0
    )


def efferent_cells(snapshot: ConnectomeSnapshot) -> list[str]:
    """Cells that change the world outside the process."""
    return sorted(
        uid for uid, unit in snapshot.units.items() if int(unit.attrs.get("efferent", 0)) > 0
    )


def path_utilisation(
    graph: DiGraphView,
    sources: Sequence[str],
    sinks: Sequence[str],
    *,
    max_depth: int = 12,
) -> tuple[Counter[str], int, list[int]]:
    """Count how often each cell lies on a shortest sense-to-action path.

    One backward breadth-first search per sink builds the distance field, then
    each source walks down that field along every edge that decreases the
    distance. Every shortest path is counted rather than one per pair, which is
    what makes the result a flow measure instead of a sample.
    """
    sink_set = set(sinks)
    source_set = set(sources)
    utilisation: Counter[str] = Counter()
    reached = 0
    lengths: list[int] = []
    reverse: dict[str, set[str]] = {}
    for pre, targets in graph.out.items():
        for post in targets:
            reverse.setdefault(post, set()).add(pre)

    for sink in sink_set:
        distance: dict[str, int] = {sink: 0}
        queue = deque([sink])
        while queue:
            node = queue.popleft()
            depth = distance[node]
            if depth >= max_depth:
                continue
            for previous in reverse.get(node, ()):
                if previous not in distance:
                    distance[previous] = depth + 1
                    queue.append(previous)
        for source in source_set:
            start = distance.get(source)
            if start is None or start == 0:
                continue
            reached += 1
            lengths.append(start)
            frontier = {source}
            depth = start
            while depth > 0 and frontier:
                nxt: set[str] = set()
                for node in frontier:
                    for post in graph.out.get(node, ()):
                        if distance.get(post) == depth - 1:
                            nxt.add(post)
                for node in nxt:
                    if node not in sink_set:
                        utilisation[node] += 1
                frontier = nxt
                depth -= 1
    return utilisation, reached, lengths


@dataclass
class SpineReport:
    """What the sense-to-action axis looks like, with the neck test resolved."""

    afferent: int
    efferent: int
    reachable_pairs: int
    unreachable_pairs: int
    mean_path: float
    median_path: float
    top_cells: list[dict[str, Any]]
    concentration: dict[str, float]
    integrator_indegree: float
    graph_mean_indegree: float
    integration_ratio: float
    verdict: str
    evidence: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "afferent": self.afferent,
            "efferent": self.efferent,
            "reachable_pairs": self.reachable_pairs,
            "unreachable_pairs": self.unreachable_pairs,
            "mean_path": round(self.mean_path, 3),
            "median_path": self.median_path,
            "concentration": {k: round(v, 4) for k, v in self.concentration.items()},
            "integrator_indegree": round(self.integrator_indegree, 3),
            "graph_mean_indegree": round(self.graph_mean_indegree, 3),
            "integration_ratio": round(self.integration_ratio, 3),
            "verdict": self.verdict,
            "evidence": self.evidence,
            "top_cells": self.top_cells,
        }


def analyse_spine(
    snapshot: ConnectomeSnapshot,
    *,
    top: int = 25,
    max_depth: int = 12,
) -> SpineReport:
    """Measure the axis and settle whether Aura has a neck.

    The verdict is a decision rule stated before the numbers arrive. The fly's
    neck concentrates flow into few cells and those cells integrate; if the top
    twenty-five cells carry less than half the traffic, or if they converge no
    harder than an average cell does, then whatever Aura has between perception
    and action is not that.
    """
    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE)
    sources = [uid for uid in afferent_cells(snapshot) if uid in graph.out]
    sinks = [uid for uid in efferent_cells(snapshot) if uid in graph.inbound]
    utilisation, reached, lengths = path_utilisation(
        graph, sources, sinks, max_depth=max_depth
    )
    total_pairs = len(sources) * len(sinks)
    total_flow = sum(utilisation.values()) or 1
    ranked = utilisation.most_common(top)
    in_degrees = {node: len(graph.inbound.get(node, ())) for node in graph.nodes}
    graph_mean_indegree = statistics.fmean(in_degrees.values()) if in_degrees else 0.0
    integrator_indegree = (
        statistics.fmean([in_degrees.get(uid, 0) for uid, _ in ranked]) if ranked else 0.0
    )
    ordered = sorted(utilisation.values(), reverse=True)
    cumulative = 0
    shares: dict[str, float] = {}
    for cut in (10, 25, 50, 100, 250):
        shares[f"top_{cut}"] = sum(ordered[:cut]) / total_flow if ordered else 0.0
    cumulative = 0
    half = 0
    for index, value in enumerate(ordered, start=1):
        cumulative += value
        if cumulative >= total_flow / 2:
            half = index
            break
    shares["cells_carrying_half"] = float(half)

    ratio = integrator_indegree / graph_mean_indegree if graph_mean_indegree else 0.0
    # A static reconstruction cannot see a call made through a service lookup or
    # an event bus, and Aura routes a great deal that way. A pair with no path
    # here is a pair with no *statically visible* path, so the reachability
    # figure is a floor and the verdict is only worth stating once enough pairs
    # have one. Running this on a proofread snapshot, with the edges the
    # recorder saw firing joined in, is what makes the number mean more.
    evidence = (
        "sufficient"
        if reached >= 200
        else f"thin: only {reached} statically reachable pairs; proofread the snapshot first"
    )
    concentrated = shares.get("top_25", 0.0) >= 0.5
    integrates = ratio >= 2.0
    if concentrated and integrates:
        verdict = "neck present: flow concentrates and the carriers integrate"
    elif concentrated:
        verdict = "flow concentrates, but the carriers relay rather than integrate"
    elif integrates:
        verdict = "carriers integrate, but flow is not concentrated into a neck"
    else:
        verdict = "no neck: perception reaches action along private paths"

    top_cells: list[dict[str, Any]] = []
    for uid, flow in ranked:
        unit = snapshot.units.get(uid)
        top_cells.append(
            {
                "cell": unit.name if unit else uid,
                "flow_share": round(flow / total_flow, 5),
                "in_degree": in_degrees.get(uid, 0),
                "out_degree": len(graph.out.get(uid, ())),
                "cell_class": str(unit.cell_class) if unit else "",
                "region": unit.region if unit else "",
            }
        )

    return SpineReport(
        afferent=len(sources),
        efferent=len(sinks),
        reachable_pairs=reached,
        unreachable_pairs=max(0, total_pairs - reached),
        mean_path=statistics.fmean(lengths) if lengths else 0.0,
        median_path=statistics.median(lengths) if lengths else 0.0,
        top_cells=top_cells,
        concentration=shares,
        integrator_indegree=integrator_indegree,
        graph_mean_indegree=graph_mean_indegree,
        integration_ratio=ratio,
        verdict=verdict,
        evidence=evidence,
    )
