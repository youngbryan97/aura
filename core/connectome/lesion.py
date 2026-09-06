"""core/connectome/lesion.py — the only way to tell a circuit from a coincidence.

Two models can fit the same recordings and disagree completely about what
causes what. Neuroscience settles that with damage: lesion the structure, stimulate
it, silence it, and see whether the function the model attributed to it goes
away. A model whose internal organisation resembles the brain survives the
perturbation; one that merely reproduces the outputs does not.

Aura can be lesioned exactly, repeatedly and reversibly, which no animal can be.
Removing a cell from a copy of the connectome costs nothing and the original is
untouched, so the experiment that neuroscience runs once per animal can be run
across every cell in the system.

Every lesion here is scored against a null, because the interesting question is
never whether removing a cell changes something. It always does. The question is
whether removing *this* cell changes more than removing a cell with the same
degree, and that comparison is what separates a circuit from a busy node.

Three effects are measured, and they disagree often enough to be worth keeping
apart:

reach
    How many cells the afferent population can still get to.
flow
    How much of the sense-to-action traffic survives.
fragmentation
    Whether the graph broke into pieces that can no longer talk.
"""

from __future__ import annotations

import logging
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .spine import afferent_cells, efferent_cells, path_utilisation
from .topology import DiGraphView
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Lesion")

__all__ = [
    "LesionEffect",
    "lesion_cells",
    "measure_effect",
    "rank_by_causal_importance",
    "degree_matched_null",
]


def _drop(graph: DiGraphView, removed: Iterable[str]) -> DiGraphView:
    gone = set(removed)
    work = graph.copy()
    for uid in gone:
        for post in work.out.get(uid, set()):
            work.inbound.get(post, set()).discard(uid)
        for pre in work.inbound.get(uid, set()):
            work.out.get(pre, set()).discard(uid)
        work.out[uid] = set()
        work.inbound[uid] = set()
    return work


def lesion_cells(snapshot: ConnectomeSnapshot, removed: Sequence[str]) -> DiGraphView:
    """A copy of the graph with those cells silenced. The original is untouched."""
    return _drop(DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE, drop_isolated=False), removed)


@dataclass
class LesionEffect:
    """What one lesion did, next to what a matched lesion would have done."""

    removed: tuple[str, ...]
    names: tuple[str, ...]
    reach_before: int
    reach_after: int
    flow_before: int
    flow_after: int
    components_before: int
    components_after: int
    null_reach_loss: float
    null_flow_loss: float
    null_samples: int

    @property
    def reach_loss(self) -> float:
        return (
            (self.reach_before - self.reach_after) / self.reach_before
            if self.reach_before
            else 0.0
        )

    @property
    def flow_loss(self) -> float:
        return (
            (self.flow_before - self.flow_after) / self.flow_before if self.flow_before else 0.0
        )

    @property
    def excess_reach_loss(self) -> float:
        return self.reach_loss - self.null_reach_loss

    def as_json(self) -> dict[str, Any]:
        return {
            "removed": list(self.names),
            "reach_before": self.reach_before,
            "reach_after": self.reach_after,
            "reach_loss": round(self.reach_loss, 5),
            "null_reach_loss": round(self.null_reach_loss, 5),
            "excess_reach_loss": round(self.excess_reach_loss, 5),
            "flow_loss": round(self.flow_loss, 5),
            "null_flow_loss": round(self.null_flow_loss, 5),
            "components_before": self.components_before,
            "components_after": self.components_after,
            "null_samples": self.null_samples,
            "verdict": (
                "this cell carries the circuit"
                if self.excess_reach_loss > 0.01
                else "no more than a cell of its degree would"
            ),
        }


def _components(graph: DiGraphView) -> int:
    adjacency = graph.undirected()
    seen: set[str] = set()
    count = 0
    for node in graph.nodes:
        if node in seen or not adjacency.get(node):
            continue
        count += 1
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, ()))
    return count


def _reach(graph: DiGraphView, sources: Sequence[str], *, max_depth: int = 10) -> int:
    from collections import deque

    seen: set[str] = set()
    frontier = deque((uid, 0) for uid in sources)
    while frontier:
        node, depth = frontier.popleft()
        if node in seen or depth > max_depth:
            continue
        seen.add(node)
        for post in graph.out.get(node, ()):
            if post not in seen:
                frontier.append((post, depth + 1))
    return len(seen)


def degree_matched_null(
    graph: DiGraphView,
    target: str,
    *,
    samples: int,
    tolerance: float = 0.2,
    seed: int = 0,
    exclude: Sequence[str] = (),
) -> list[str]:
    """Cells with about the same degree as the target, for the control lesion.

    Matching on degree is what makes the comparison mean something. A hub
    removed from any graph costs reach; the question is whether this hub costs
    more than the others of its size, and only a matched control answers it.

    ``exclude`` keeps the cells reach is measured *from* out of the pool.
    Removing a source destroys every path trivially, and a control that does
    that is measuring the experiment rather than the graph.
    """
    degree = graph.degree(target)
    if degree == 0:
        return []
    low = degree * (1.0 - tolerance)
    high = degree * (1.0 + tolerance)
    banned = set(exclude) | {target}
    pool = [node for node in graph.nodes if node not in banned and low <= graph.degree(node) <= high]
    if not pool:
        return []
    rng = random.Random(seed)
    if len(pool) <= samples:
        return sorted(pool)
    return sorted(rng.sample(pool, samples))


def measure_effect(
    snapshot: ConnectomeSnapshot,
    removed: Sequence[str],
    *,
    null_samples: int = 12,
    max_depth: int = 10,
    seed: int = 0,
) -> LesionEffect:
    """Silence some cells, measure the damage, and compare it to a matched control."""
    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE, drop_isolated=False)
    sources = [uid for uid in afferent_cells(snapshot) if uid in graph.out]
    sinks = [uid for uid in efferent_cells(snapshot) if uid in graph.inbound]
    if not sources:
        sources = sorted(
            (uid for uid in graph.nodes if not graph.inbound.get(uid) and graph.out.get(uid)),
        )[:200]

    reach_before = _reach(graph, sources, max_depth=max_depth)
    components_before = _components(graph)
    utilisation_before, _, _ = path_utilisation(graph, sources, sinks, max_depth=max_depth)
    flow_before = sum(utilisation_before.values())

    damaged = _drop(graph, removed)
    reach_after = _reach(damaged, sources, max_depth=max_depth)
    components_after = _components(damaged)
    utilisation_after, _, _ = path_utilisation(damaged, sources, sinks, max_depth=max_depth)
    flow_after = sum(utilisation_after.values())

    null_reach: list[float] = []
    null_flow: list[float] = []
    for index, target in enumerate(removed):
        for control in degree_matched_null(
            graph,
            target,
            samples=max(1, null_samples // max(1, len(removed))),
            seed=seed + index,
            exclude=[*sources, *sinks],
        ):
            control_graph = _drop(graph, [control])
            null_reach.append(
                (reach_before - _reach(control_graph, sources, max_depth=max_depth))
                / reach_before
                if reach_before
                else 0.0
            )
            control_flow, _, _ = path_utilisation(
                control_graph, sources, sinks, max_depth=max_depth
            )
            null_flow.append(
                (flow_before - sum(control_flow.values())) / flow_before if flow_before else 0.0
            )

    names = tuple(
        snapshot.units[uid].name if uid in snapshot.units else uid for uid in removed
    )
    return LesionEffect(
        removed=tuple(removed),
        names=names,
        reach_before=reach_before,
        reach_after=reach_after,
        flow_before=flow_before,
        flow_after=flow_after,
        components_before=components_before,
        components_after=components_after,
        null_reach_loss=statistics.fmean(null_reach) if null_reach else 0.0,
        null_flow_loss=statistics.fmean(null_flow) if null_flow else 0.0,
        null_samples=len(null_reach),
    )


def rank_by_causal_importance(
    snapshot: ConnectomeSnapshot,
    candidates: Sequence[str],
    *,
    null_samples: int = 6,
    max_depth: int = 8,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Lesion each candidate on its own and rank by damage beyond its degree."""
    rows: list[dict[str, Any]] = []
    for index, uid in enumerate(candidates):
        effect = measure_effect(
            snapshot, [uid], null_samples=null_samples, max_depth=max_depth, seed=index
        )
        rows.append(effect.as_json())
    rows.sort(key=lambda row: -row["excess_reach_loss"])
    return rows[:limit]
