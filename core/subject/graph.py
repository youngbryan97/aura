"""What shape the surviving edges make.

A directed graph over ten domains is small enough that nothing here needs an
approximation. Strong connectivity, vertex connectivity and the full set of
simple cycles are all computed exactly by enumeration, which matters because
the interesting failures are one-node failures and an approximate answer would
hide exactly those.

The three questions, in the order they get harder to pass:

Can every domain reach every other one, eventually, by some path? That is
strong connectivity, and a star passes it, which is why it is only the first
question.

Does the graph survive losing a node? Vertex connectivity of at least two
means there is no single broker whose removal leaves the mind in pieces.
Brokered integration and reentrant integration look identical until this is
asked.

Does influence leave a domain and come back by another route? A cycle of
length two is a pair of variables updating each other. Reentry means a
perturbation travelled through at least two other domains and altered the one
it started in, which is the property a feed-forward pipeline with a return
edge cannot have.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GraphReport", "analyse_graph", "strongly_connected", "simple_cycles"]


def _adjacency(nodes: Sequence[str], edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        if source in out and target in out and source != target:
            out[source].add(target)
    return out


def strongly_connected(nodes: Sequence[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Tarjan, iterative, so a deep graph cannot exhaust the stack."""
    adjacency = _adjacency(nodes, edges)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    components: list[list[str]] = []

    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, list[str]]] = [(root, sorted(adjacency[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            if pending:
                nxt = pending.pop()
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, sorted(adjacency[nxt])))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    components.append(sorted(component))
    # Sorted by size, then by name, so two runs of the same graph print the
    # same report and a diff between runs means the graph changed.
    return sorted(components, key=lambda group: (-len(group), group))


def simple_cycles(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]], *, max_length: int = 10
) -> list[tuple[str, ...]]:
    """Every simple directed cycle, each reported once from its smallest node."""
    adjacency = _adjacency(nodes, edges)
    order = {node: position for position, node in enumerate(nodes)}
    found: list[tuple[str, ...]] = []

    def walk(start: str, node: str, path: list[str], seen: set[str]) -> None:
        if len(path) > max_length:
            return
        for nxt in sorted(adjacency[node]):
            if nxt == start and len(path) >= 2:
                found.append(tuple(path))
            elif nxt not in seen and order[nxt] > order[start]:
                seen.add(nxt)
                path.append(nxt)
                walk(start, nxt, path, seen)
                path.pop()
                seen.discard(nxt)

    for start in nodes:
        walk(start, start, [start], {start})
    return found


def _is_strong(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    if len(nodes) <= 1:
        return True
    components = strongly_connected(nodes, edges)
    return len(components) == 1 and len(components[0]) == len(nodes)


def vertex_connectivity(nodes: Sequence[str], edges: Sequence[tuple[str, str]], *, cap: int = 3) -> int:
    """Smallest number of nodes whose removal breaks strong connectivity.

    Reported up to ``cap``; a graph that survives every removal of that size
    is recorded as reaching the cap rather than as having a larger number
    nobody measured.
    """
    if not _is_strong(nodes, edges):
        return 0
    for size in range(1, min(cap, len(nodes) - 1) + 1):
        for gone in itertools.combinations(nodes, size):
            kept = [node for node in nodes if node not in gone]
            kept_edges = [
                (a, b) for a, b in edges if a not in gone and b not in gone
            ]
            if not _is_strong(kept, kept_edges):
                return size
    return cap


@dataclass
class GraphReport:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    components: list[list[str]]
    connectivity: int
    cycles: list[tuple[str, ...]] = field(default_factory=list)
    cycles_per_node: dict[str, int] = field(default_factory=dict)
    reentry_length: dict[str, int] = field(default_factory=dict)

    @property
    def one_component(self) -> bool:
        return len(self.components) == 1 and len(self.components[0]) == len(self.nodes)

    @property
    def every_node_recurs(self) -> bool:
        return bool(self.nodes) and all(
            self.cycles_per_node.get(node, 0) >= 2 for node in self.nodes
        )

    @property
    def every_node_reenters(self) -> bool:
        """Reentry needs a loop through at least two other domains."""
        return bool(self.nodes) and all(
            0 < self.reentry_length.get(node, 0) >= 3 for node in self.nodes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [f"{a}->{b}" for a, b in self.edges],
            "edge_count": len(self.edges),
            "components": self.components,
            "one_strongly_connected_component": self.one_component,
            "vertex_connectivity": self.connectivity,
            "cycle_count": len(self.cycles),
            "cycles_per_node": self.cycles_per_node,
            "shortest_reentry": self.reentry_length,
            "every_node_on_two_cycles": self.every_node_recurs,
            "every_node_reenters_through_two_others": self.every_node_reenters,
            "shortest_cycles": ["->".join(c) for c in sorted(self.cycles, key=len)[:12]],
        }


def analyse_graph(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> GraphReport:
    kept = tuple(dict.fromkeys(nodes))
    clean = tuple((a, b) for a, b in edges if a != b and a in kept and b in kept)
    cycles = simple_cycles(kept, clean)
    per_node: dict[str, int] = {node: 0 for node in kept}
    reentry: dict[str, int] = {node: 0 for node in kept}
    for cycle in cycles:
        for node in cycle:
            per_node[node] += 1
            if len(cycle) >= 3 and (reentry[node] == 0 or len(cycle) < reentry[node]):
                reentry[node] = len(cycle)
    return GraphReport(
        nodes=kept,
        edges=clean,
        components=strongly_connected(kept, clean),
        connectivity=vertex_connectivity(kept, clean),
        cycles=cycles,
        cycles_per_node=per_node,
        reentry_length=reentry,
    )
