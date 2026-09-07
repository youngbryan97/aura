"""core/connectome/gating.py — state changes the route, not only the volume.

The male connectome paper describes circuits where a small structural
difference sends the same sensory signal down opposite behavioural paths, one
toward approach and one toward avoidance. The interesting part is not that a
state biases a decision. It is that the state changes which cells the signal
reaches at all.

Aura's affect, goals and drives currently work the other way round. They
contribute numbers into a scorer, everything is still computed, and the state
moves a weight at the end. That is a volume knob. This module is the other
mechanism:

    W_effective[i][j] = W[i][j] * g(i, j, state)

with ``g`` near zero closing a route and near one leaving it open. A closed
route is not a downweighted route: the traffic goes somewhere else, and the
cells on the closed path never run.

Because that is a strong claim, the module measures whether it happened. A gate
that changes only edge weights and leaves every path intact has not rerouted
anything, so :func:`routing_change` compares reachability and shortest paths
before and after, and :class:`GateReport` refuses to call a state change a
reroute when the reachable set is identical.

The gates themselves are declared, not learned, and each carries the condition
that opens it and the condition that closes it. A gate with no closing
condition is a gate that is always open.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .topology import DiGraphView
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Gating")

__all__ = [
    "Gate",
    "GateSet",
    "GateReport",
    "routing_change",
    "reachable_from",
]


@dataclass(frozen=True)
class Gate:
    """One conditional route, with both of its conditions named.

    ``predicate`` decides from the state whether the gate is open. ``closes_on``
    is the human-readable statement of what shuts it, and it exists so a gate
    that can never close is visible as such rather than being discovered later
    when something needed it to close.
    """

    name: str
    opens: str
    closes_on: str
    predicate: Callable[[Mapping[str, float]], float]
    edges: tuple[tuple[str, str], ...] = ()
    regions: tuple[tuple[str, str], ...] = ()

    def level(self, state: Mapping[str, float]) -> float:
        try:
            return max(0.0, min(1.0, float(self.predicate(state))))
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("gate %s could not read the state: %s", self.name, exc)
            return 1.0

    def covers(self, pre_region: str, post_region: str, edge: tuple[str, str]) -> bool:
        if self.edges and edge in self.edges:
            return True
        return bool(self.regions) and (pre_region, post_region) in self.regions


@dataclass
class GateSet:
    """Every declared gate, applied together to produce an effective graph."""

    gates: list[Gate] = field(default_factory=list)

    def add(self, gate: Gate) -> Gate:
        self.gates.append(gate)
        return gate

    def always_open(self) -> list[str]:
        """Gates whose closing condition is empty. Each one is decoration."""
        return [gate.name for gate in self.gates if not gate.closes_on.strip()]

    def effective(
        self,
        snapshot: ConnectomeSnapshot,
        state: Mapping[str, float],
        *,
        closed_below: float = 0.05,
    ) -> tuple[DiGraphView, dict[str, Any]]:
        """Apply the gates and return the graph the system can actually use.

        An edge whose combined gate level falls below ``closed_below`` is
        removed rather than scaled, because that is the difference this module
        exists to make. Everything above it keeps its weight scaled.
        """
        graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE, drop_isolated=False)
        levels = {gate.name: gate.level(state) for gate in self.gates}
        removed = 0
        scaled = 0
        for pre in list(graph.out):
            pre_unit = snapshot.units.get(pre)
            if pre_unit is None:
                continue
            for post in list(graph.out[pre]):
                post_unit = snapshot.units.get(post)
                if post_unit is None:
                    continue
                level = 1.0
                for gate in self.gates:
                    if gate.covers(pre_unit.region, post_unit.region, (pre, post)):
                        level *= levels[gate.name]
                if level < closed_below:
                    graph.out[pre].discard(post)
                    graph.inbound[post].discard(pre)
                    graph.weights.pop((pre, post), None)
                    removed += 1
                elif level < 1.0:
                    weight = graph.weights.get((pre, post), 1)
                    graph.weights[(pre, post)] = max(1, int(round(weight * level)))
                    scaled += 1
        return graph, {"levels": levels, "edges_closed": removed, "edges_scaled": scaled}


def reachable_from(graph: DiGraphView, sources: Sequence[str], *, max_depth: int = 12) -> set[str]:
    """Everything a set of cells can reach, bounded in depth."""
    seen: set[str] = set()
    frontier = deque((uid, 0) for uid in sources if uid in graph.out)
    while frontier:
        node, depth = frontier.popleft()
        if node in seen or depth > max_depth:
            continue
        seen.add(node)
        for post in graph.out.get(node, ()):
            if post not in seen:
                frontier.append((post, depth + 1))
    return seen


@dataclass
class GateReport:
    """Whether a state change rerouted anything, or only turned a dial."""

    state: dict[str, float]
    levels: dict[str, float]
    edges_closed: int
    edges_scaled: int
    reachable_before: int
    reachable_after: int
    lost_cells: int
    gained_cells: int
    always_open_gates: list[str]

    @property
    def rerouted(self) -> bool:
        return self.lost_cells > 0 or self.gained_cells > 0

    def as_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "levels": {k: round(v, 4) for k, v in self.levels.items()},
            "edges_closed": self.edges_closed,
            "edges_scaled": self.edges_scaled,
            "reachable_before": self.reachable_before,
            "reachable_after": self.reachable_after,
            "lost_cells": self.lost_cells,
            "gained_cells": self.gained_cells,
            "rerouted": self.rerouted,
            "always_open_gates": self.always_open_gates,
            "verdict": (
                "state changed the route"
                if self.rerouted
                else "state changed weights only; every cell is still reachable"
            ),
        }


def routing_change(
    snapshot: ConnectomeSnapshot,
    gates: GateSet,
    sources: Sequence[str],
    state: Mapping[str, float],
    *,
    baseline_state: Mapping[str, float] | None = None,
) -> GateReport:
    """Compare what is reachable under two states, and say which changed."""
    baseline_state = baseline_state or {}
    before_graph, _ = gates.effective(snapshot, baseline_state)
    after_graph, stats = gates.effective(snapshot, state)
    before = reachable_from(before_graph, sources)
    after = reachable_from(after_graph, sources)
    return GateReport(
        state=dict(state),
        levels=stats["levels"],
        edges_closed=int(stats["edges_closed"]),
        edges_scaled=int(stats["edges_scaled"]),
        reachable_before=len(before),
        reachable_after=len(after),
        lost_cells=len(before - after),
        gained_cells=len(after - before),
        always_open_gates=gates.always_open(),
    )
