"""core/connectome/topology.py — the statistics, each against a null that could beat it.

Network measurements are easy to report and easy to fool yourself with. Almost
every graph is clustered, almost every graph has hubs, and a number quoted with
no null model behind it says nothing about the graph it came from.

So every statistic here is paired with the same control the field uses: a
degree-preserving rewiring. Each cell keeps its in-degree and out-degree exactly
and everything else about the wiring is destroyed. A property that survives that
is a property of the degree sequence and nothing more. A property that collapses
is structure, and the z-score says how much.

What gets measured, and why each one earns its place:

reciprocity
    Whether A calling B predicts B calling A. In a call graph this is
    recursion and mutual dependency, and it is the first thing a rewiring
    destroys.
triad census
    The sixteen ways three cells can be wired. Motif analysis is triad counts
    against exactly this null, and the feed-forward loop is the motif that made
    the method worth having.
rich club
    Whether the busiest cells preferentially wire to each other. In cortex the
    rich club is the backbone that long-range traffic crosses.
small-worldness
    Clustering high, path length short, both relative to the rewiring.
modularity and participation
    Whether the graph falls into communities, and which cells hold them
    together. A connector cell with a high participation coefficient is a cell
    whose failure disconnects parts of the system that never touch otherwise.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Topology")

__all__ = [
    "DiGraphView",
    "degree_preserving_rewire",
    "reciprocity",
    "triad_census",
    "rich_club",
    "clustering_and_path",
    "small_worldness",
    "modularity_communities",
    "participation_coefficients",
    "power_law_fit",
    "TopologyReport",
    "analyse",
]


@dataclass
class DiGraphView:
    """A directed graph as plain dicts, so nothing depends on an import.

    networkx is used where it earns its place and avoided in the hot loops,
    because a rewiring null runs the whole measurement a hundred times and the
    cost of the wrapper starts to dominate the science.
    """

    nodes: tuple[str, ...]
    out: dict[str, set[str]]
    inbound: dict[str, set[str]]
    weights: dict[tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ConnectomeSnapshot,
        kind: EdgeKind | None = EdgeKind.DRIVE,
        *,
        drop_isolated: bool = True,
    ) -> DiGraphView:
        out: dict[str, set[str]] = {}
        inbound: dict[str, set[str]] = {}
        weights: dict[tuple[str, str], int] = {}
        for conn in snapshot.connections.values():
            if kind is not None and conn.kind is not kind:
                continue
            if conn.pre == conn.post:
                continue
            out.setdefault(conn.pre, set()).add(conn.post)
            inbound.setdefault(conn.post, set()).add(conn.pre)
            weights[(conn.pre, conn.post)] = conn.contacts
        if drop_isolated:
            nodes = tuple(sorted(set(out) | set(inbound)))
        else:
            nodes = tuple(sorted(snapshot.units))
        for node in nodes:
            out.setdefault(node, set())
            inbound.setdefault(node, set())
        return cls(nodes=nodes, out=out, inbound=inbound, weights=weights)

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def m(self) -> int:
        return sum(len(targets) for targets in self.out.values())

    def edges(self) -> list[tuple[str, str]]:
        return [(pre, post) for pre, targets in self.out.items() for post in sorted(targets)]

    def degree(self, node: str) -> int:
        return len(self.out.get(node, ())) + len(self.inbound.get(node, ()))

    def copy(self) -> DiGraphView:
        return DiGraphView(
            nodes=self.nodes,
            out={k: set(v) for k, v in self.out.items()},
            inbound={k: set(v) for k, v in self.inbound.items()},
            weights=dict(self.weights),
        )

    def undirected(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {node: set() for node in self.nodes}
        for pre, targets in self.out.items():
            for post in targets:
                adj[pre].add(post)
                adj[post].add(pre)
        return adj


def degree_preserving_rewire(
    graph: DiGraphView,
    *,
    swaps_per_edge: int = 10,
    seed: int = 0,
) -> DiGraphView:
    """Randomise the wiring while every in-degree and out-degree is untouched.

    The swap takes two edges ``a→b`` and ``c→d`` and replaces them with ``a→d``
    and ``c→b``. Rejecting a swap that would create a self-loop or a duplicate
    keeps the result a simple digraph, which is what the observed graph is, so
    the comparison is between like and like.
    """
    rng = random.Random(seed)
    work = graph.copy()
    edges = work.edges()
    if len(edges) < 4:
        return work
    target = swaps_per_edge * len(edges)
    attempts = 0
    done = 0
    ceiling = target * 4
    while done < target and attempts < ceiling:
        attempts += 1
        i = rng.randrange(len(edges))
        j = rng.randrange(len(edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if a == d or c == b or a == c or b == d:
            continue
        if d in work.out[a] or b in work.out[c]:
            continue
        work.out[a].discard(b)
        work.inbound[b].discard(a)
        work.out[c].discard(d)
        work.inbound[d].discard(c)
        work.out[a].add(d)
        work.inbound[d].add(a)
        work.out[c].add(b)
        work.inbound[b].add(c)
        edges[i] = (a, d)
        edges[j] = (c, b)
        done += 1
    return work


def reciprocity(graph: DiGraphView) -> float:
    """Fraction of edges whose reverse is also present."""
    total = graph.m
    if not total:
        return 0.0
    mutual = 0
    for pre, targets in graph.out.items():
        for post in targets:
            if pre in graph.out.get(post, ()):
                mutual += 1
    return mutual / total


#: The sixteen isomorphism classes of a three-node directed graph, in the
#: standard MAN ordering used by every motif paper since Milo 2002.
TRIAD_NAMES: tuple[str, ...] = (
    "003", "012", "102", "021D", "021U", "021C", "111D", "111U",
    "030T", "030C", "201", "120D", "120U", "120C", "210", "300",
)


def triad_census(graph: DiGraphView, *, sample: int = 20_000, seed: int = 0) -> dict[str, int]:
    """Count triad classes over a sample of connected triples.

    The full census over a graph this size is dominated by the empty class,
    which carries no information. Sampling connected triples concentrates the
    count where the motifs are, and the same sampler runs on the null, so the
    comparison stays fair.
    """
    import networkx as nx

    rng = random.Random(seed)
    edges = graph.edges()
    if not edges:
        return {name: 0 for name in TRIAD_NAMES}
    counts = {name: 0 for name in TRIAD_NAMES}
    undirected = graph.undirected()
    for _ in range(sample):
        a, b = edges[rng.randrange(len(edges))]
        neighbours = (undirected.get(a, set()) | undirected.get(b, set())) - {a, b}
        if not neighbours:
            continue
        c = rng.choice(sorted(neighbours))
        sub = nx.DiGraph()
        sub.add_nodes_from((a, b, c))
        for u in (a, b, c):
            for v in (a, b, c):
                if u != v and v in graph.out.get(u, ()):
                    sub.add_edge(u, v)
        census = nx.triadic_census(sub)
        for name, value in census.items():
            if value and name in counts:
                counts[name] += value
    return counts


def rich_club(graph: DiGraphView, *, degrees: Sequence[int] | None = None) -> dict[int, float]:
    """Density among the cells above each degree cut."""
    degree_of = {node: graph.degree(node) for node in graph.nodes}
    if degrees is None:
        values = sorted(degree_of.values())
        if not values:
            return {}
        top = values[-1]
        degrees = [k for k in (2, 4, 8, 16, 32, 64, 128, 256) if k < top]
    out: dict[int, float] = {}
    for k in degrees:
        members = {node for node, deg in degree_of.items() if deg > k}
        size = len(members)
        if size < 2:
            continue
        links = sum(1 for node in members for post in graph.out.get(node, ()) if post in members)
        out[k] = links / (size * (size - 1))
    return out


def clustering_and_path(
    graph: DiGraphView,
    *,
    sample: int = 600,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean local clustering and mean shortest path, both on a sample.

    Path length is measured from sampled sources over the undirected graph and
    only over pairs that are reachable, which is the standard treatment for a
    graph that is not strongly connected.
    """
    from collections import deque

    rng = random.Random(seed)
    adj = graph.undirected()
    nodes = [n for n in graph.nodes if adj.get(n)]
    if not nodes:
        return 0.0, 0.0
    picks = nodes if len(nodes) <= sample else rng.sample(nodes, sample)

    clustering: list[float] = []
    for node in picks:
        neighbours = adj[node]
        k = len(neighbours)
        if k < 2:
            clustering.append(0.0)
            continue
        links = 0
        neighbour_list = sorted(neighbours)
        neighbour_set = neighbours
        for i, u in enumerate(neighbour_list):
            for v in neighbour_list[i + 1 :]:
                if v in adj.get(u, ()) and u in neighbour_set:
                    links += 1
        clustering.append(2 * links / (k * (k - 1)))

    lengths: list[int] = []
    for source in picks[: max(1, sample // 6)]:
        seen = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            depth = seen[node]
            if depth >= 6:
                continue
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen[nxt] = depth + 1
                    queue.append(nxt)
        lengths.extend(v for v in seen.values() if v > 0)
    mean_path = statistics.fmean(lengths) if lengths else 0.0
    return (statistics.fmean(clustering) if clustering else 0.0), mean_path


def small_worldness(
    graph: DiGraphView,
    *,
    nulls: int = 4,
    sample: int = 600,
    seed: int = 0,
) -> dict[str, float]:
    """Sigma: clustering above the null divided by path length above the null.

    Sigma above one is the small-world signature. It is reported here with the
    two ratios it is made of, because a sigma that comes from a path-length
    collapse is a different finding from one that comes from clustering and the
    single number hides which happened.
    """
    c_obs, l_obs = clustering_and_path(graph, sample=sample, seed=seed)
    c_null: list[float] = []
    l_null: list[float] = []
    for i in range(nulls):
        rewired = degree_preserving_rewire(graph, swaps_per_edge=4, seed=seed + i + 1)
        clustered, path = clustering_and_path(rewired, sample=sample, seed=seed)
        c_null.append(clustered)
        l_null.append(path)
    c_bar = statistics.fmean(c_null) if c_null else 0.0
    l_bar = statistics.fmean(l_null) if l_null else 0.0
    gamma = (c_obs / c_bar) if c_bar > 0 else 0.0
    lam = (l_obs / l_bar) if l_bar > 0 else 0.0
    sigma = (gamma / lam) if lam > 0 else 0.0
    return {
        "clustering": c_obs,
        "clustering_null": c_bar,
        "path": l_obs,
        "path_null": l_bar,
        "gamma": gamma,
        "lambda": lam,
        "sigma": sigma,
    }


def modularity_communities(graph: DiGraphView, *, seed: int = 0) -> tuple[float, dict[str, int]]:
    """Greedy modularity communities over the undirected projection."""
    import networkx as nx

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for pre, targets in graph.out.items():
        for post in targets:
            undirected.add_edge(pre, post)
    if undirected.number_of_edges() == 0:
        return 0.0, {}
    communities = nx.community.louvain_communities(undirected, seed=seed)
    membership: dict[str, int] = {}
    for index, group in enumerate(communities):
        for node in group:
            membership[node] = index
    return nx.community.modularity(undirected, communities), membership


def participation_coefficients(
    graph: DiGraphView,
    membership: Mapping[str, int],
) -> dict[str, float]:
    """How evenly a cell's edges spread across communities.

    One is a cell whose neighbours are spread evenly across every community.
    Zero is a cell that only talks inside its own. The connector cells are the
    ones near one, and they are where a failure stops being local.
    """
    out: dict[str, float] = {}
    adj = graph.undirected()
    for node in graph.nodes:
        neighbours = adj.get(node, ())
        total = len(neighbours)
        if total == 0:
            out[node] = 0.0
            continue
        per: dict[int, int] = {}
        for neighbour in neighbours:
            group = membership.get(neighbour, -1)
            per[group] = per.get(group, 0) + 1
        out[node] = 1.0 - sum((count / total) ** 2 for count in per.values())
    return out


def power_law_fit(values: Sequence[int], *, xmin_candidates: int = 24) -> dict[str, float]:
    """Maximum-likelihood power law with the cut chosen the way Clauset does it.

    The exponent is the closed-form discrete MLE. ``xmin`` is not assumed: each
    candidate cut gets its own fit and the one with the smallest KS distance
    wins, which the closed-form exponent alone cannot give you. A distribution is called
    heavy tailed far more often than one is fitted, so ``ks`` and ``tail_n``
    are returned beside the exponent and a fit over a handful of points in the
    tail should be disbelieved however pretty the exponent looks.
    """
    data = sorted(v for v in values if v > 0)
    if len(data) < 16:
        return {"alpha": 0.0, "xmin": 0.0, "n": float(len(data)), "tail_n": 0.0, "ks": 1.0}
    cuts = sorted({v for v in data})[:xmin_candidates]
    best = {"alpha": 0.0, "xmin": 0.0, "n": float(len(data)), "tail_n": 0.0, "ks": 1.0}
    for xmin in cuts:
        tail = [v for v in data if v >= xmin]
        if len(tail) < 16:
            break
        denom = sum(math.log(v / (xmin - 0.5)) for v in tail)
        if denom <= 0:
            continue
        alpha = 1.0 + len(tail) / denom
        top = tail[-1]
        zeta = sum(k ** (-alpha) for k in range(xmin, top + 1))
        if zeta <= 0:
            continue
        ks = 0.0
        running = 0.0
        cursor = xmin
        n_tail = len(tail)
        for index, value in enumerate(tail, start=1):
            while cursor <= value:
                running += cursor ** (-alpha)
                cursor += 1
            ks = max(ks, abs(index / n_tail - running / zeta))
        if ks < best["ks"]:
            best = {
                "alpha": alpha,
                "xmin": float(xmin),
                "n": float(len(data)),
                "tail_n": float(n_tail),
                "ks": ks,
            }
    return best


@dataclass
class TopologyReport:
    """Everything measured, with the null beside each number."""

    nodes: int
    edges: int
    reciprocity: float
    reciprocity_null: float
    reciprocity_z: float
    small_world: dict[str, float]
    modularity: float
    communities: int
    rich_club: dict[int, float]
    rich_club_null: dict[int, float]
    triads: dict[str, int]
    triads_null: dict[str, float]
    triad_z: dict[str, float]
    in_degree_fit: dict[str, float]
    out_degree_fit: dict[str, float]
    triad_sample: int = 0
    rich_club_normalised: dict[int, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "reciprocity": round(self.reciprocity, 5),
            "reciprocity_null": round(self.reciprocity_null, 5),
            "reciprocity_z": round(self.reciprocity_z, 3),
            "small_world": {k: round(v, 4) for k, v in self.small_world.items()},
            "modularity": round(self.modularity, 4),
            "communities": self.communities,
            "rich_club": {str(k): round(v, 5) for k, v in self.rich_club.items()},
            "rich_club_null": {str(k): round(v, 5) for k, v in self.rich_club_null.items()},
            "rich_club_normalised": {
                str(k): round(v, 4) for k, v in self.rich_club_normalised.items()
            },
            "triad_sample": self.triad_sample,
            "triads_over_connected_triples": self.triads,
            "triad_z": {k: round(v, 3) for k, v in self.triad_z.items()},
            "in_degree_fit": {k: round(v, 4) for k, v in self.in_degree_fit.items()},
            "out_degree_fit": {k: round(v, 4) for k, v in self.out_degree_fit.items()},
        }

    def significant_motifs(self, threshold: float = 2.0) -> list[tuple[str, float]]:
        return sorted(
            ((name, z) for name, z in self.triad_z.items() if abs(z) >= threshold),
            key=lambda item: -abs(item[1]),
        )


def analyse(
    snapshot: ConnectomeSnapshot,
    *,
    kind: EdgeKind | None = EdgeKind.DRIVE,
    nulls: int = 8,
    triad_sample: int = 4_000,
    path_sample: int = 400,
    seed: int = 0,
) -> TopologyReport:
    """Run the whole battery, each statistic against the same rewiring null."""
    graph = DiGraphView.from_snapshot(snapshot, kind)
    observed_reciprocity = reciprocity(graph)
    observed_rich = rich_club(graph)
    observed_triads = triad_census(graph, sample=triad_sample, seed=seed)

    null_reciprocity: list[float] = []
    null_rich: dict[int, list[float]] = {k: [] for k in observed_rich}
    null_triads: dict[str, list[int]] = {name: [] for name in TRIAD_NAMES}
    for i in range(nulls):
        rewired = degree_preserving_rewire(graph, swaps_per_edge=4, seed=seed + 101 + i)
        null_reciprocity.append(reciprocity(rewired))
        for k, value in rich_club(rewired, degrees=sorted(observed_rich)).items():
            null_rich.setdefault(k, []).append(value)
        for name, value in triad_census(rewired, sample=triad_sample, seed=seed).items():
            null_triads[name].append(value)

    def _z(observed: float, samples: Sequence[float]) -> float:
        if len(samples) < 2:
            return 0.0
        spread = statistics.pstdev(samples)
        if spread <= 0:
            return 0.0
        return (observed - statistics.fmean(samples)) / spread

    modularity, membership = modularity_communities(graph, seed=seed)
    in_degrees = [len(graph.inbound.get(n, ())) for n in graph.nodes]
    out_degrees = [len(graph.out.get(n, ())) for n in graph.nodes]

    return TopologyReport(
        nodes=graph.n,
        edges=graph.m,
        reciprocity=observed_reciprocity,
        reciprocity_null=statistics.fmean(null_reciprocity) if null_reciprocity else 0.0,
        reciprocity_z=_z(observed_reciprocity, null_reciprocity),
        small_world=small_worldness(graph, nulls=max(2, nulls // 2), sample=path_sample, seed=seed),
        modularity=modularity,
        communities=len(set(membership.values())) if membership else 0,
        rich_club=observed_rich,
        rich_club_null={k: (statistics.fmean(v) if v else 0.0) for k, v in null_rich.items()},
        triads=observed_triads,
        triads_null={
            name: (statistics.fmean(values) if values else 0.0)
            for name, values in null_triads.items()
        },
        triad_z={
            name: _z(observed_triads.get(name, 0), null_triads.get(name, []))
            for name in TRIAD_NAMES
        },
        in_degree_fit=power_law_fit(in_degrees),
        out_degree_fit=power_law_fit(out_degrees),
        triad_sample=triad_sample,
        rich_club_normalised={
            k: (observed_rich[k] / statistics.fmean(null_rich[k]))
            if null_rich.get(k) and statistics.fmean(null_rich[k]) > 0
            else 0.0
            for k in observed_rich
        },
    )
