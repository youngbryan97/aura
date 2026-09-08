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
    "PHASE_STATIONS",
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
    "RingReport",
    "measure_ring",
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
#: Which station each kernel phase stands for, and which stand for none.
#:
#: The first version of this table was written from module names, and it was
#: wrong in a way that took a recording to see. The modules named after the
#: stations — ``global_workspace``, ``action_arbitrator``, ``self_model`` —
#: exist, and a turn does not run them. What runs a turn is the kernel's phase
#: pipeline, and 0 of 129 workspace cells and 0 of 272 action cells fired in a
#: recording of 240 turns. Reporting "the ring does not close" from that would
#: have been reporting the table.
#:
#: Read off what each phase is, not chosen to make the ring close. Twelve of
#: the twenty-nine stand for none of the seven, and they are listed here with
#: the reason rather than left out, because a phase missing from a table and a
#: phase deliberately outside the ring look identical in the code.
PHASE_STATIONS: dict[str, str] = {
    "ProprioceptiveLoop": "interoception",
    "SensoryIngestionPhase": "interoception",
    "NativeMultimodalBridge": "interoception",
    "PerfectEmotionPhase": "affect",
    "AffectUpdatePhase": "affect",
    "CognitiveIntegrationPhase": "workspace",
    "UnityBindingPhase": "workspace",
    "PhiConsciousnessPhase": "higher_order",
    "ConsciousnessPhase": "higher_order",
    "SelfReviewPhase": "higher_order",
    "IdentityReflectionPhase": "self_model",
    "MotivationUpdatePhase": "planning",
    "InitiativeGenerationPhase": "planning",
    "ExecutiveClosurePhase": "planning",
    "CognitiveRoutingPhase": "action",
    "UnitaryResponsePhase": "action",
    "GodModeToolPhase": "action",
    # Outside the ring, and why.
    "SocialContextPhase": "",  # who she is talking to, not a stage of the ring
    "EternalMemoryPhase": "",  # storage
    "MemoryRetrievalPhase": "",  # storage
    "MemoryConsolidationPhase": "",  # storage
    "ShadowExecutionPhase": "",  # governance
    "EternalGrowthEngine": "",  # development, on a slower clock than a turn
    "TrueEvolutionPhase": "",  # development
    "InferencePhase": "",  # reasoning is not one of the seven stages
    "ConversationalDynamicsPhase": "",  # turn-taking
    "BondingPhase": "",  # relationship
    "RepairPhase": "",  # wording of an answer already decided
    "LearningPhase": "",  # what the turn leaves behind
}

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
        "phases.sensory_ingestion",
        "upgrades_10x:NativeMultimodalBridge",
    ),
    "affect": (
        "core.affect",
        "emotion",
        "mood",
        "neurochemical",
        "affective",
        "affect_state",
        "phases.affect_update",
        "upgrades_10x:PerfectEmotionPhase",
    ),
    "workspace": (
        "global_workspace",
        "workspace_ignition",
        "core.workspace.",
        "attention_field",
        "broadcast",
        "phases.cognitive_integration",
        "phases.unity_binding",
        "consciousness.integration",
        "consciousness.continuous_experience",
    ),
    "higher_order": (
        "higher_order",
        "metacog",
        "self_report",
        "introspect",
        "confidence_calibrat",
        "epistemic_calibration",
        "phases.phi_consciousness",
        "phases.consciousness_phase",
        "kernel.self_review",
        "consciousness.multiple_drafts",
        "consciousness.phenomenological_experiencer",
    ),
    "self_model": (
        "self_model",
        "self_object",
        "self_profile",
        "self_contract",
        "self_revision",
        "autobiograph",
        "phases.identity_reflection",
        "consciousness.minimal_selfhood",
        "consciousness.selfhood_tick",
    ),
    "planning": (
        "planner",
        "planning",
        "core.conation.",
        "goal_planner",
        "goal_pursuit",
        "intention",
        "deliberat",
        "phases.motivation_update",
        "phases.initiative_generation",
        "executive_closure",
        "runtime.watched_goal",
        "core.goals.",
    ),
    "action": (
        "actuat",
        "action_arbitrator",
        "effector",
        "motor_cortex",
        "core.executors.",
        "executors.",
        "phases.response_generation",
        "phases.response_contract",
        "phases.cognitive_routing",
        "upgrades_10x:GodModeToolPhase",
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
    """Put cells in stations by where they live, from the written-down table.

    A cell belongs to at most one station: the first that claims it, in table
    order. Without that a module matching two patterns would be in two stations
    and every link between them would be guaranteed.

    Matching runs against ``module:qualname`` rather than the module alone,
    because six of the kernel's phases share one module. ``upgrades_10x`` holds
    the multimodal bridge, the emotion phase and the tool phase, which stand for
    three different stations, and a module-level pattern has to put all three in
    one of them or none.
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
            # A reconstruction's unit.name already carries the module, so it is
            # module:qualname; a hand-built snapshot's may be the bare name.
            where = (
                unit.name
                if unit.name.startswith(unit.neuropil)
                else f"{unit.neuropil}:{unit.name}"
            )
            if any(pattern in where for pattern in patterns):
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
    #: Which measurement decided this link: "flow" over the structural graph, or
    #: "effective" from a recording. They are not interchangeable and the
    #: threshold for one is meaningless for the other.
    mode: str = "flow"

    @property
    def present(self) -> bool:
        if self.mode == "effective":
            return self.effective_edges > 0
        return self.enrichment >= ENRICHMENT_THRESHOLD

    def as_json(self) -> dict[str, Any]:
        return {
            "link": f"{self.source} -> {self.target}",
            "mode": self.mode,
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
        mode="flow" if flow is not None else "effective",
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
        readout_available=True,
    ),
    # Round two. The first three were written from the theory and two of them
    # named readouts that turned out not to be downstream of the station they
    # were meant to test — the control lesion moved them as far. What the first
    # run did find is below, registered as its own prediction and run again on
    # an independent seed, because a quantity noticed in a result and a quantity
    # written down before one are not the same kind of evidence and the only way
    # to convert the first into the second is another run.
    LesionPrediction(
        name="higher_order_writes_the_self_reading",
        station="higher_order",
        predicted_intact="the pipeline completes and working memory still fills",
        predicted_lost="the selfhood reading, and the variation in pending intents",
        readout="cognition.selfhood_reading, and the spread of pending_intents across objectives",
        readout_available=True,
    ),
    LesionPrediction(
        name="workspace_computes_fragmentation",
        station="workspace",
        predicted_intact="the pipeline completes and affective engagement is untouched",
        predicted_lost="the fragmentation score, and its movement across objectives",
        readout="cognition.fragmentation_score, its spread and its affective gap",
        readout_available=True,
    ),
    LesionPrediction(
        name="affect_regulates_what_is_injected",
        station="affect",
        predicted_intact="the pipeline completes and pending intents still form",
        predicted_lost=(
            "regulation: an injected valence difference of 1.2 survives to the end of "
            "the turn instead of being pulled back to 0.0004 of it"
        ),
        readout="the gap in affect.valence and affect.arousal between two injected states",
        readout_available=True,
    ),
)


@dataclass(frozen=True, slots=True)
class RingReport:
    """The ring measured against every other ring through the same stations.

    A link that beats its own rotations tells you the two stations are coupled.
    It does not tell you the ring is a ring: with seven stations there are 720
    directed cycles through them, and if influence were spread evenly the
    architecture's cycle would be an unremarkable one of the 720. So the ring is
    scored against all of them, exactly rather than by sampling, and the number
    that matters is where the real order falls in that list.
    """

    condition: str
    order: tuple[str, ...]
    links: tuple[dict[str, Any], ...]
    links_carrying: int
    ring_gain: float
    null_mean: float
    null_spread: float
    z: float
    percentile: float
    cycles: int
    pairs_measured: int = 0
    pairs_carrying: int = 0
    ring_ranks: tuple[int, ...] = ()
    strongest_pairs: tuple[tuple[str, float, bool], ...] = ()
    stations: dict[str, int] = field(default_factory=dict)
    skipped: str = ""

    @property
    def closed(self) -> bool:
        return not self.skipped and self.links_carrying == len(self.links)

    def as_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "order": list(self.order),
            "stations": self.stations,
            "links": list(self.links),
            "links_carrying": self.links_carrying,
            "links_total": len(self.links),
            "closed": self.closed,
            "ring_gain": round(self.ring_gain, 6),
            "null_mean": round(self.null_mean, 6),
            "null_spread": round(self.null_spread, 6),
            "z": round(self.z, 3),
            "percentile": round(self.percentile, 4),
            "cycles": self.cycles,
            "pairs_measured": self.pairs_measured,
            "pairs_carrying": self.pairs_carrying,
            "ring_ranks_by_gain": list(self.ring_ranks),
            "strongest_pairs": [
                {"link": link, "gain": round(gain, 5), "in_ring": in_ring}
                for link, gain, in_ring in self.strongest_pairs
            ],
            "skipped": self.skipped,
            "verdict": self._verdict(),
        }

    def _verdict(self) -> str:
        if self.skipped:
            return self.skipped
        if self.pairs_measured and self.pairs_carrying == self.pairs_measured:
            # Every ordered pair of stations carries, so "this link carries" has
            # stopped separating anything and the ranks are the finding. Saying
            # "all seven links carry" without this denominator would read as
            # support for the ring and is a statement about a pipeline over one
            # shared state object, in which every stage is coupled to every
            # other by construction.
            return (
                f"all {self.pairs_measured} ordered pairs of stations carry, so a "
                f"link carrying separates nothing. The ring's seven rank "
                f"{', '.join(str(rank) for rank in self.ring_ranks)} of "
                f"{self.pairs_measured} by strength, and the architecture's order is "
                f"stronger than {self.percentile:.1%} of the {self.cycles} cycles "
                "through the same stations"
            )
        if self.closed and self.percentile >= 0.95:
            return (
                f"every link carries and the architecture's order is stronger than "
                f"{self.percentile:.1%} of the {self.cycles} cycles through the same "
                "stations"
            )
        if self.closed:
            return (
                f"every link carries, and the architecture's order is not stronger than "
                f"a cycle drawn at random through the same stations "
                f"({self.percentile:.1%} of {self.cycles})"
            )
        missing = [
            link["link"] for link in self.links if not link.get("carries")
        ]
        return f"{len(missing)} of {len(self.links)} links do not carry: {', '.join(missing)}"


def measure_ring(
    trace: Any,
    condition: str,
    stations: Mapping[str, Station],
    *,
    order: Sequence[str] = COALITION_ORDER,
    lags: int = 3,
    rotations: int = 8,
    draws: int = 400,
    seed: int = 0,
    deconfound: bool = True,
) -> RingReport:
    """Measure every ordered pair of stations, then ask where the ring sits.

    Every pair, not only the seven the architecture names, because the seven
    mean nothing without the other thirty-five. The first version of this
    measured the ring alone and reported that all seven links carried, which was
    true and would have been true of almost any seven.
    """
    import itertools

    import numpy as np

    from core.connectome.effective import cross_influence

    names = [name for name in order if name in stations]
    if len(names) < 3:
        return RingReport(
            condition=condition,
            order=tuple(names),
            links=(),
            links_carrying=0,
            ring_gain=0.0,
            null_mean=0.0,
            null_spread=0.0,
            z=0.0,
            percentile=0.0,
            cycles=0,
            skipped=f"{len(names)} stations is not a ring",
        )

    gains: dict[tuple[str, str], float] = {}
    measured: dict[tuple[str, str], dict[str, Any]] = {}
    for source, target in itertools.permutations(names, 2):
        influence = cross_influence(
            trace,
            condition,
            stations[source].cells,
            stations[target].cells,
            source_station=source,
            target_station=target,
            lags=lags,
            rotations=rotations,
            draws=draws,
            seed=seed,
            deconfound=deconfound,
        )
        payload = influence.as_json()
        gains[(source, target)] = influence.median_gain if not influence.skipped else 0.0
        measured[(source, target)] = payload

    if all(value == 0.0 for value in gains.values()):
        return RingReport(
            condition=condition,
            order=tuple(names),
            links=(),
            links_carrying=0,
            ring_gain=0.0,
            null_mean=0.0,
            null_spread=0.0,
            z=0.0,
            percentile=0.0,
            cycles=0,
            stations={name: len(stations[name].cells) for name in names},
            skipped="no station pair could be measured in this condition",
        )

    ring_links = list(zip(names, names[1:] + names[:1], strict=True))
    ring_gain = float(np.mean([gains[pair] for pair in ring_links]))

    # Every directed cycle through the same stations. Fixing the first station
    # and permuting the rest enumerates each cycle once.
    head, *rest = names
    cycle_means = []
    for tail in itertools.permutations(rest):
        cycle = [head, *tail]
        pairs = list(zip(cycle, cycle[1:] + cycle[:1], strict=True))
        cycle_means.append(float(np.mean([gains[pair] for pair in pairs])))
    null = np.array(cycle_means, dtype=np.float64)
    null_mean = float(null.mean())
    null_spread = float(null.std())
    percentile = float((null < ring_gain).mean())
    z = (ring_gain - null_mean) / null_spread if null_spread > 0 else 0.0

    links = tuple(measured[pair] for pair in ring_links)
    carrying = sum(1 for link in links if link.get("carries"))
    # Where the ring's links sit among every ordered pair. Seven links that all
    # carry mean one thing when nothing else does and another when everything
    # does, and only the ranking says which.
    ranked = sorted(gains.items(), key=lambda item: -item[1])
    ring_set = set(ring_links)
    ranks = tuple(
        sorted(
            index
            for index, (pair, _gain) in enumerate(ranked, start=1)
            if pair in ring_set
        )
    )
    strongest = tuple(
        (f"{pre} -> {post}", gain, (pre, post) in ring_set)
        for (pre, post), gain in ranked[:8]
    )
    return RingReport(
        condition=condition,
        order=tuple(names),
        links=links,
        links_carrying=carrying,
        ring_gain=ring_gain,
        null_mean=null_mean,
        null_spread=null_spread,
        z=z,
        percentile=percentile,
        cycles=int(null.size),
        pairs_measured=len(measured),
        pairs_carrying=sum(1 for row in measured.values() if row.get("carries")),
        ring_ranks=ranks,
        strongest_pairs=strongest,
        stations={name: len(stations[name].cells) for name in names},
    )
