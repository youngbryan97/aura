"""core/connectome/coalition.py — does the loop close, or was it only ever drawn?

There is a diagram everyone who builds one of these draws:

    interoception → affect → workspace → higher order → self model →
    planning → action → interoception

It is drawn because it is the architecture people intend, and a module named
after each station is not evidence that the loop exists. The stronger question,
and the one this module asks, is whether recordings reconstruct it: when she is
working, does influence actually run around that ring, and does it run around it
more than around a ring drawn at random through the same stations?

Three things have to be true before any of it means anything, and each is
measured rather than assumed.

**The stations have to be identifiable.** They are assigned by module path, from
a table that is written down and can be replaced. That is the weakest link here
and it is stated: the claim is not that these modules implement interoception,
it is that if they do, this is what their influence looks like.

**The links have to survive a null.** A ring drawn through the same seven
stations in a shuffled order is the control. In a graph where everything reaches
everything, the intended ring closes and so does every other one, and only the
difference between them says anything.

**The loop has to recur.** One condition closing is an observation; the same
links closing across conditions that share no other machinery is a coalition.

Three things about how the links are read, all learned by getting them wrong
first.

A link between two subsystems is not a single call. These stations hold between
seventy-eight and six hundred and fourteen cells each, and asking for a direct
edge between two of them is asking for a monosynaptic connection between two
brain areas.

Reachability is not the measure either. Over the combined graph within four
hops, a ring drawn at random through the same stations closes about as often as
the intended one — the null said so, with a z of -0.9. What separates them is
not whether influence *can* arrive but whether it *preferentially* does: of the
influence leaving one station, what share lands in the next, against the share
its size alone would predict. That statistic stays discriminative when
everything reaches everything.

The share is conditioned on arriving at a station at all. Without that
condition most of every station's outflow drains into maintenance —
record_degradation alone receives from 3,760 cells — and every enrichment reads
low for the same reason, which is a fact about where errors are reported and not
about the circuit.

And the wired layer alone is the wrong graph. Subsystems here couple through the
container and the event bus far more than through direct calls, so a link is
read from the combined multilayer graph.

The lesion predictions are registered here rather than derived after the fact,
because a prediction written down after seeing the result is not one.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Coalition")

__all__ = [
    "Station",
    "STATION_TABLE",
    "COALITION_ORDER",
    "assign_stations",
    "ENRICHMENT_THRESHOLD",
    "flow_field",
    "LinkEvidence",
    "ClosureReport",
    "test_closure",
    "reproducibility",
    "LesionPrediction",
    "LESION_PREDICTIONS",
]


#: Which modules stand for which station. This table is the assumption the whole
#: analysis rests on, so it is here rather than inferred, and replacing it is an
#: edit to one dictionary. A pattern matches any module whose path contains it,
#: and :func:`assign_stations` reports every module each pattern pulled in, so a
#: pattern that reaches too far is visible rather than silent.
#:
#: The first version used exact module prefixes and missed most of each station —
#: core.consciousness.global_workspace was not in the workspace station, and
#: core.agi.hierarchical_planner was not in planning — which made links look
#: absent that were only unlooked-for.
STATION_TABLE: dict[str, tuple[str, ...]] = {
    "interoception": (
        "interocept",
        "somatic",
        "felt_state",
        "proprio",
        "homeostasis",
        "allostasis",
        "nociception",
        "body_schema",
    ),
    "affect": (
        "core.affect",
        "emotion",
        "mood",
        "neurochemical",
        "affective",
        "affect_state",
    ),
    "workspace": (
        "global_workspace",
        "workspace_ignition",
        "core.workspace.",
        "attention_field",
        "broadcast",
    ),
    "higher_order": (
        "higher_order",
        "metacog",
        "self_report",
        "introspect",
        "confidence_calibrat",
        "epistemic_calibration",
    ),
    "self_model": (
        "self_model",
        "self_object",
        "self_profile",
        "self_contract",
        "self_revision",
        "autobiograph",
    ),
    "planning": (
        "planner",
        "planning",
        "core.conation.",
        "goal_planner",
        "goal_pursuit",
        "intention",
        "deliberat",
    ),
    "action": (
        "actuat",
        "action_arbitrator",
        "effector",
        "motor_cortex",
        "core.executors.",
        "executors.",
    ),
}

#: The ring, in the order the architecture intends it.
COALITION_ORDER: tuple[str, ...] = (
    "interoception",
    "affect",
    "workspace",
    "higher_order",
    "self_model",
    "planning",
    "action",
)


@dataclass
class Station:
    """One stage of the ring, the cells standing for it, and where they came from."""

    name: str
    patterns: tuple[str, ...]
    cells: tuple[str, ...]
    modules: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "station": self.name,
            "patterns": list(self.patterns),
            "cells": len(self.cells),
            "modules": len(self.modules),
            "example_modules": list(self.modules[:8]),
        }


def assign_stations(
    snapshot: ConnectomeSnapshot,
    table: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Station]:
    """Put cells in stations by module path, from the written-down table.

    A cell belongs to at most one station: the first that claims it, in table
    order. Without that a module matching two patterns would be in two stations
    and every link between them would be guaranteed.
    """
    table = table or STATION_TABLE
    stations: dict[str, Station] = {}
    claimed: set[str] = set()
    for name, patterns in table.items():
        cells: list[str] = []
        modules: set[str] = set()
        for uid, unit in snapshot.units.items():
            if uid in claimed:
                continue
            if any(pattern in unit.neuropil for pattern in patterns):
                cells.append(uid)
                modules.add(unit.neuropil)
        claimed.update(cells)
        stations[name] = Station(
            name=name,
            patterns=tuple(patterns),
            cells=tuple(sorted(cells)),
            modules=tuple(sorted(modules)),
        )
    return stations


#: A link counts as carrying when the next station receives at least this much
#: more of the influence than its size predicts. One would be exactly its share.
ENRICHMENT_THRESHOLD: float = 1.5


@dataclass(frozen=True)
class LinkEvidence:
    """How much of one station's influence lands in the next."""

    source: str
    target: str
    structural_edges: int
    effective_edges: int
    strongest: float
    hops: int = 0
    share: float = 0.0
    expected: float = 0.0
    enrichment: float = 0.0

    @property
    def present(self) -> bool:
        return self.enrichment >= ENRICHMENT_THRESHOLD

    def as_json(self) -> dict[str, Any]:
        return {
            "link": f"{self.source} -> {self.target}",
            "structural_edges": self.structural_edges,
            "hops": self.hops,
            "effective_edges": self.effective_edges,
            "share": round(self.share, 5),
            "expected": round(self.expected, 5),
            "enrichment": round(self.enrichment, 3),
            "present": self.present,
        }


@dataclass
class ClosureReport:
    """Whether the ring closed, against rings drawn at random."""

    condition: str
    links: list[LinkEvidence]
    closed: bool
    links_present: int
    ring_enrichment: float = 0.0
    null_enrichment: float = 0.0
    enrichment_z: float = 0.0
    null_mean: float = 0.0
    null_spread: float = 0.0
    z: float = 0.0
    nulls: int = 0
    stations: dict[str, int] = field(default_factory=dict)
    skipped: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "stations": self.stations,
            "links": [link.as_json() for link in self.links],
            "links_present": self.links_present,
            "links_total": len(self.links),
            "closed": self.closed,
            "ring_enrichment": round(self.ring_enrichment, 4),
            "null_enrichment": round(self.null_enrichment, 4),
            "enrichment_z": round(self.enrichment_z, 3),
            "null_mean_links_present": round(self.null_mean, 3),
            "null_spread": round(self.null_spread, 3),
            "z": round(self.z, 3),
            "nulls": self.nulls,
            "skipped": self.skipped,
            "verdict": (
                self.skipped
                if self.skipped
                else
                "the ring closes and carries more than a ring drawn at random"
                if self.closed and self.enrichment_z >= 2.0
                else "the ring closes, and so does a ring drawn at random through the "
                "same stations"
                if self.closed
                else f"{len(self.links) - self.links_present} of {len(self.links)} links "
                f"carry nothing; the ring is no better than a random one "
                f"(enrichment z {self.enrichment_z:.2f})"
            ),
        }


def flow_field(
    snapshot: ConnectomeSnapshot,
    multilayer: Any = None,
    *,
    steps: int = 4,
    decay: float = 0.7,
) -> Any:
    """A propagation operator over every layer, row-normalised.

    Influence leaving a cell is split among the cells it reaches, so a hub does
    not hand each neighbour a full unit, and the decay makes a four-hop arrival
    worth less than a one-hop one. What comes back is the operator and the cell
    order; the seeding is the caller's.
    """
    import numpy as np
    from scipy import sparse

    nodes = sorted(snapshot.units)
    index = {uid: i for i, uid in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    adjacency = _adjacency(snapshot, multilayer)
    for pre, posts in adjacency.items():
        source = index.get(pre)
        if source is None:
            continue
        for post in posts:
            target = index.get(post)
            if target is not None and target != source:
                rows.append(target)
                cols.append(source)
    size = len(nodes)
    if not rows:
        return None, index, steps, decay
    data = np.ones(len(rows), dtype=np.float64)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    column_sums = np.asarray(matrix.sum(axis=0)).ravel()
    column_sums[column_sums == 0] = 1.0
    return (matrix @ sparse.diags(1.0 / column_sums)).tocsr(), index, steps, decay


def _propagate(field: Any, seeds: Sequence[str]) -> Any:
    """Accumulated influence over the whole graph, from a set of seed cells."""
    import numpy as np

    matrix, index, steps, decay = field
    size = len(index)
    total = np.zeros(size)
    if matrix is None:
        return total
    state = np.zeros(size)
    picked = [index[uid] for uid in seeds if uid in index]
    if not picked:
        return total
    state[picked] = 1.0 / len(picked)
    for _ in range(steps):
        state = decay * (matrix @ state)
        total = total + state
    return total


def _adjacency(
    snapshot: ConnectomeSnapshot,
    multilayer: Any = None,
) -> dict[str, set[str]]:
    """Outgoing neighbours, over the wired layer or over every layer.

    An undirected layer contributes both directions, because shared state
    couples both ways whatever the code intended.
    """
    out: dict[str, set[str]] = {}
    for connection in snapshot.edges(EdgeKind.DRIVE):
        out.setdefault(connection.pre, set()).add(connection.post)
    if multilayer is not None:
        from .layers import Layer

        for layer in Layer:
            if layer is Layer.WIRED:
                continue
            undirected = str(layer) in {"gap", "ipc"}
            for pre, post in multilayer.layer(layer):
                out.setdefault(pre, set()).add(post)
                if undirected:
                    out.setdefault(post, set()).add(pre)
    return out


def _hops_between(
    adjacency: Mapping[str, set[str]],
    sources: Sequence[str],
    targets: Sequence[str],
    max_hops: int,
) -> int:
    """Shortest number of hops from any source cell to any target cell."""
    from collections import deque

    target_set = set(targets)
    seen = set(sources)
    frontier: deque[tuple[str, int]] = deque((uid, 0) for uid in sources)
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for post in adjacency.get(node, ()):
            if post in target_set:
                return depth + 1
            if post not in seen:
                seen.add(post)
                frontier.append((post, depth + 1))
    return 0


def _link_evidence(
    source: Station,
    target: Station,
    snapshot: ConnectomeSnapshot,
    effective: Any,
    adjacency: Mapping[str, set[str]] | None = None,
    max_hops: int = 4,
    flow: Mapping[str, Any] | None = None,
) -> LinkEvidence:
    source_cells = set(source.cells)
    target_cells = set(target.cells)
    if adjacency is not None:
        structural = _hops_between(adjacency, source.cells, target.cells, max_hops)
    else:
        structural = sum(
            1
            for connection in snapshot.edges(EdgeKind.DRIVE)
            if connection.pre in source_cells and connection.post in target_cells
        )
    effective_edges = 0
    strongest = 0.0
    if effective is not None:
        for (pre, post), edge in effective.edges.items():
            if pre in source_cells and post in target_cells and edge.survives_null:
                effective_edges += 1
                strongest = max(strongest, edge.weight)
    share = expected = enrichment = 0.0
    if flow is not None:
        arrival = flow.get(source.name)
        if arrival is not None:
            index = flow["_index"]
            reached = flow[f"_reached:{source.name}"]
            sizes = flow[f"_sizes:{source.name}"]
            landed = sum(
                float(arrival[index[uid]]) for uid in target.cells if uid in index
            )
            share = landed / reached if reached > 0 else 0.0
            total_cells = sum(sizes.values())
            expected = (len(target.cells) / total_cells) if total_cells > 0 else 0.0
            enrichment = (share / expected) if expected > 0 else 0.0
    return LinkEvidence(
        source=source.name,
        target=target.name,
        structural_edges=structural,
        effective_edges=effective_edges,
        strongest=strongest,
        hops=structural if adjacency is not None else 0,
        share=share,
        expected=expected,
        enrichment=enrichment,
    )


def test_closure(
    snapshot: ConnectomeSnapshot,
    effective: Any,
    stations: Mapping[str, Station] | None = None,
    *,
    order: Sequence[str] = COALITION_ORDER,
    nulls: int = 200,
    seed: int = 0,
    use_effective: bool = True,
    multilayer: Any = None,
    max_hops: int = 4,
    recorded: Sequence[str] | None = None,
    min_recorded_per_station: int = 5,
) -> ClosureReport:
    """Walk the ring and count the links that carry, against shuffled rings.

    ``use_effective`` decides which graph the links are read from. With a
    recording the effective graph is the right one — the question is whether
    influence runs, not whether a call exists. Without one the structural graph
    answers a weaker question and the report says which was used.
    """
    stations = stations or assign_stations(snapshot)
    ordered = [stations[name] for name in order if name in stations]
    if use_effective and recorded is not None:
        fired = set(recorded)
        silent = [
            station.name
            for station in ordered
            if len(fired.intersection(station.cells)) < min_recorded_per_station
        ]
        if silent:
            # The recording cannot answer a question about stations that did not
            # fire in it. Reporting zero links here would read as a finding about
            # her and is a finding about the workload.
            report = ClosureReport(
                condition=getattr(effective, "condition", "structural"),
                links=[],
                closed=False,
                links_present=0,
                null_mean=0.0,
                null_spread=0.0,
                z=0.0,
                nulls=0,
                stations={name: len(st.cells) for name, st in stations.items()},
            )
            report.skipped = (
                "these stations did not fire in this recording: " + ", ".join(sorted(silent))
            )
            return report
    if len(ordered) < 3:
        return ClosureReport(
            condition=getattr(effective, "condition", "structural"),
            links=[],
            closed=False,
            links_present=0,
            null_mean=0.0,
            null_spread=0.0,
            z=0.0,
            nulls=0,
        )

    adjacency = None if use_effective else _adjacency(snapshot, multilayer)
    flow: dict[str, Any] | None = None
    if not use_effective:
        field = flow_field(snapshot, multilayer, steps=max_hops)
        matrix, index, _steps, _decay = field
        if matrix is not None:
            flow = {"_index": index, "_population": len(index)}
            for station in ordered:
                arrival = _propagate(field, station.cells)
                flow[station.name] = arrival
                sizes = {
                    other.name: len(other.cells)
                    for other in ordered
                    if other.name != station.name
                }
                reached = 0.0
                for other in ordered:
                    if other.name == station.name:
                        continue
                    reached += sum(
                        float(arrival[index[uid]]) for uid in other.cells if uid in index
                    )
                flow[f"_reached:{station.name}"] = reached
                flow[f"_sizes:{station.name}"] = sizes

    def _ring(sequence: Sequence[Station]) -> list[LinkEvidence]:
        return [
            _link_evidence(
                sequence[i],
                sequence[(i + 1) % len(sequence)],
                snapshot,
                effective if use_effective else None,
                adjacency,
                max_hops,
                flow,
            )
            for i in range(len(sequence))
        ]

    links = _ring(ordered)
    present = sum(1 for link in links if link.present)

    ring_enrichment = (
        statistics.fmean([link.enrichment for link in links]) if links else 0.0
    )
    rng = random.Random(seed)
    null_counts: list[int] = []
    null_enrichments: list[float] = []
    for _ in range(nulls):
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        if shuffled == ordered:
            continue
        null_links = _ring(shuffled)
        null_counts.append(sum(1 for link in null_links if link.present))
        null_enrichments.append(
            statistics.fmean([link.enrichment for link in null_links])
            if null_links
            else 0.0
        )
    null_mean = statistics.fmean(null_counts) if null_counts else 0.0
    null_spread = statistics.pstdev(null_counts) if len(null_counts) > 1 else 0.0
    z = (present - null_mean) / null_spread if null_spread > 0 else 0.0
    null_enrichment = statistics.fmean(null_enrichments) if null_enrichments else 0.0
    enrichment_spread = (
        statistics.pstdev(null_enrichments) if len(null_enrichments) > 1 else 0.0
    )
    enrichment_z = (
        (ring_enrichment - null_enrichment) / enrichment_spread
        if enrichment_spread > 0
        else 0.0
    )

    return ClosureReport(
        condition=getattr(effective, "condition", "structural"),
        links=links,
        closed=present == len(links),
        links_present=present,
        ring_enrichment=ring_enrichment,
        null_enrichment=null_enrichment,
        enrichment_z=enrichment_z,
        null_mean=null_mean,
        null_spread=null_spread,
        z=z,
        nulls=len(null_counts),
        stations={name: len(station.cells) for name, station in stations.items()},
    )


def reproducibility(reports: Sequence[ClosureReport]) -> dict[str, Any]:
    """How often each link carries, across conditions.

    One condition closing the ring is an observation. The same links closing
    across conditions that share no other machinery is a coalition, and the
    difference between those two is the whole point of recording more than one.
    """
    if not reports:
        return {"conditions": 0}
    counts: dict[str, int] = {}
    for report in reports:
        for link in report.links:
            key = f"{link.source} -> {link.target}"
            counts[key] = counts.get(key, 0) + (1 if link.present else 0)
    total = len(reports)
    always = [key for key, value in counts.items() if value == total]
    never = [key for key, value in counts.items() if value == 0]
    return {
        "conditions": total,
        "closed_in": sum(1 for report in reports if report.closed),
        "link_recurrence": {key: round(value / total, 4) for key, value in sorted(counts.items())},
        "links_in_every_condition": always,
        "links_in_none": never,
        "mean_z": round(statistics.fmean([r.z for r in reports]), 3),
        "verdict": (
            "the same ring recurs across conditions"
            if len(always) == len(counts) and counts
            else f"{len(always)} of {len(counts)} links carry in every condition"
        ),
    }


@dataclass(frozen=True)
class LesionPrediction:
    """A lesion, its readout, and what is expected to happen. Registered first."""

    name: str
    station: str
    predicted_intact: str
    predicted_lost: str
    readout: str
    readout_available: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "lesion": self.name,
            "station": self.station,
            "predicted_intact": self.predicted_intact,
            "predicted_lost": self.predicted_lost,
            "readout": self.readout,
            "readout_available": self.readout_available,
        }


#: The three lesions that would separate a real functional architecture from a
#: set of modules with the vocabulary. Each names what should survive and what
#: should not, and whether the readout that would show it exists yet. A
#: prediction with no readout is still worth writing down; it is the difference
#: between an experiment nobody has run and one nobody has designed.
LESION_PREDICTIONS: tuple[LesionPrediction, ...] = (
    LesionPrediction(
        name="higher_order_feedback",
        station="higher_order",
        predicted_intact="raw competence on tasks with a checkable answer",
        predicted_lost="calibrated self-report; confidence stops tracking correctness",
        readout="a calibration curve over tasks whose answers are known",
        readout_available=False,
    ),
    LesionPrediction(
        name="workspace_broadcast",
        station="workspace",
        predicted_intact="each local processor's own output",
        predicted_lost="cross-domain availability; one domain stops using another's result",
        readout="a task needing two domains, scored against each domain alone",
        readout_available=False,
    ),
    LesionPrediction(
        name="affective_loop",
        station="affect",
        predicted_intact="reasoning on a single presented problem",
        predicted_lost="state-dependent prioritisation; ordering stops moving with state",
        readout="the order chosen over a fixed set of tasks under two states",
        readout_available=False,
    ),
)
