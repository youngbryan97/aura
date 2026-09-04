"""core/morphogenesis/graph.py — the topology, as state.

The cell layer already had a population and a diffusive field. What it did
not have was ``E_t``: a set of bindings that says which cell can reach which
other cell. Without that, every cell talked to every other cell through one
global signal queue, so two different "shapes" of the system computed exactly
the same thing and the shape was decoration.

This module makes the shape load-bearing. A binding is a typed, directed edge
with a port contract, a weight, a latency and a capacity. Delivery in the
sandbox workload goes along edges and nowhere else, so cutting an edge removes
a path and the computation notices.

Three properties the rest of the layer depends on:

* **Version is monotonic.** Every committed change increments it. A snapshot
  carries the version it was taken at, so a stale reader can tell.
* **Commit is all-or-nothing.** A transition that fails validation leaves the
  authoritative graph byte-identical to what it was. There is no half-applied
  topology for a reader to observe.
* **Serialization is deterministic.** Sorted keys throughout, so two runs with
  the same seed produce the same bytes and a diff of two runs means something.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .types import clamp01, json_safe, stable_digest


class EdgeType(StrEnum):
    """What a binding carries.

    ``DATA`` moves work products. ``CONTROL`` moves requests and cancellations.
    ``REPAIR`` is the path a distress signal takes to something that can act on
    it. ``MEMORY`` reaches a store. ``OBSERVE`` is read-only and is the one
    edge type exempt from the port contract, because watching something needs
    no agreement from the thing being watched.
    """

    DATA = "data"
    CONTROL = "control"
    REPAIR = "repair"
    MEMORY = "memory"
    OBSERVE = "observe"


#: Edge types whose endpoints must agree on a port. OBSERVE is deliberately out.
PORT_CHECKED_EDGES = frozenset({EdgeType.DATA, EdgeType.CONTROL, EdgeType.REPAIR, EdgeType.MEMORY})


class GraphIntegrityError(RuntimeError):
    """A proposed graph state broke a structural rule and was not committed."""


@dataclass(frozen=True)
class MorphEdge:
    """One directed binding.

    ``port`` names the thing that flows. A node declares which ports it can be
    the source of and which it can be the target of, the way a docking face
    declares what it can send and what it will accept. ``latency_ms`` and
    ``capacity`` exist because a physical substrate cannot make a binding free,
    and the simulation refuses to pretend otherwise.
    """

    source: str
    target: str
    edge_type: EdgeType = EdgeType.DATA
    port: str = ""
    weight: float = 1.0
    latency_ms: float = 0.0
    capacity: int = 8
    created_at_version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Identity of a binding. Two cells may hold several edges of
        different types or ports, and each is its own binding."""
        return (self.source, self.target, str(self.edge_type), self.port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": str(self.edge_type),
            "port": self.port,
            "weight": round(float(self.weight), 6),
            "latency_ms": round(float(self.latency_ms), 4),
            "capacity": int(self.capacity),
            "created_at_version": int(self.created_at_version),
            "metadata": json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MorphEdge:
        payload = dict(data or {})
        raw_type = payload.get("edge_type", EdgeType.DATA)
        try:
            edge_type = EdgeType(str(raw_type))
        except ValueError:
            edge_type = EdgeType.DATA
        return cls(
            source=str(payload.get("source", "")),
            target=str(payload.get("target", "")),
            edge_type=edge_type,
            port=str(payload.get("port", "")),
            weight=float(payload.get("weight", 1.0)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            capacity=int(payload.get("capacity", 8)),
            created_at_version=int(payload.get("created_at_version", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class GraphSnapshot:
    """An immutable read of the graph at one version."""

    version: int
    nodes: tuple[str, ...]
    edges: tuple[MorphEdge, ...]
    taken_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges],
            "taken_at": self.taken_at,
            "digest": self.digest(),
        }

    def digest(self) -> str:
        """Content hash over nodes and edges. Two snapshots with the same
        digest describe the same topology, whatever version they carry."""
        parts = [*sorted(self.nodes)]
        parts.extend(sorted("|".join(str(v) for v in e.key) + f"@{e.weight:.6f}" for e in self.edges))
        return stable_digest(*parts, length=20)


@dataclass
class GraphDiff:
    """What changed between two snapshots."""

    added_nodes: tuple[str, ...] = ()
    removed_nodes: tuple[str, ...] = ()
    added_edges: tuple[MorphEdge, ...] = ()
    removed_edges: tuple[MorphEdge, ...] = ()
    from_version: int = 0
    to_version: int = 0

    @property
    def empty(self) -> bool:
        return not (self.added_nodes or self.removed_nodes or self.added_edges or self.removed_edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "added_nodes": list(self.added_nodes),
            "removed_nodes": list(self.removed_nodes),
            "added_edges": [e.to_dict() for e in self.added_edges],
            "removed_edges": [e.to_dict() for e in self.removed_edges],
            "empty": self.empty,
        }

    def summary(self) -> str:
        return (
            f"v{self.from_version}->v{self.to_version}: "
            f"+{len(self.added_nodes)}/-{len(self.removed_nodes)} nodes, "
            f"+{len(self.added_edges)}/-{len(self.removed_edges)} edges"
        )


class MorphGraph:
    """The authoritative topology.

    Mutation goes through :meth:`transaction`, which gives a scratch copy. The
    scratch copy becomes authoritative only when the body completes and the
    result passes :meth:`_validate`. Anything raising inside leaves the live
    graph untouched, which is what lets a substrate transition fail without
    stranding the graph in a state no cell was designed for.
    """

    #: A node may hold at most this many outbound edges. A cell that binds to
    #: everything is the topology equivalent of a global singleton, and the
    #: bound is what stops a policy rediscovering one.
    max_out_degree = 16
    max_in_degree = 16

    def __init__(self, *, max_nodes: int = 256, max_edges: int = 1024):
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self._nodes: set[str] = set()
        self._edges: dict[tuple[str, str, str, str], MorphEdge] = {}
        self._version = 0
        self._lock = threading.RLock()
        self._history: list[tuple[int, str]] = []

    # ── reads ───────────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def edge_count(self) -> int:
        with self._lock:
            return len(self._edges)

    def has_node(self, node: str) -> bool:
        with self._lock:
            return str(node) in self._nodes

    def nodes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._nodes))

    def edges(self) -> tuple[MorphEdge, ...]:
        with self._lock:
            return tuple(self._edges[k] for k in sorted(self._edges))

    def out_edges(self, node: str, *, edge_type: EdgeType | None = None) -> tuple[MorphEdge, ...]:
        node = str(node)
        with self._lock:
            found = [e for k, e in sorted(self._edges.items()) if k[0] == node]
        if edge_type is not None:
            found = [e for e in found if e.edge_type == edge_type]
        return tuple(found)

    def in_edges(self, node: str, *, edge_type: EdgeType | None = None) -> tuple[MorphEdge, ...]:
        node = str(node)
        with self._lock:
            found = [e for k, e in sorted(self._edges.items()) if k[1] == node]
        if edge_type is not None:
            found = [e for e in found if e.edge_type == edge_type]
        return tuple(found)

    def neighbours(self, node: str, *, radius: int = 1) -> set[str]:
        """Cells reachable within ``radius`` hops, ignoring direction.

        This is what a local policy is allowed to see. A policy that needs the
        whole graph is a central scheduler wearing a cell costume, and the
        ablation harness compares against one of those on purpose.
        """
        node = str(node)
        seen: set[str] = set()
        frontier = {node}
        for _ in range(max(0, int(radius))):
            nxt: set[str] = set()
            with self._lock:
                for edge in self._edges.values():
                    if edge.source in frontier:
                        nxt.add(edge.target)
                    if edge.target in frontier:
                        nxt.add(edge.source)
            nxt -= seen
            nxt.discard(node)
            seen |= nxt
            if not nxt:
                break
            frontier = nxt
        return seen

    def components(self) -> list[set[str]]:
        """Connected components, ignoring edge direction. A partition scenario
        is judged by how many of these exist and how big each one is."""
        with self._lock:
            adjacency: dict[str, set[str]] = {n: set() for n in self._nodes}
            for edge in self._edges.values():
                if edge.source in adjacency and edge.target in adjacency:
                    adjacency[edge.source].add(edge.target)
                    adjacency[edge.target].add(edge.source)
        seen: set[str] = set()
        out: list[set[str]] = []
        for node in sorted(adjacency):
            if node in seen:
                continue
            stack = [node]
            comp: set[str] = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                comp.add(current)
                stack.extend(adjacency[current] - seen)
            out.append(comp)
        return out

    def path_exists(self, source: str, target: str) -> bool:
        """Directed reachability. The workload uses this to decide whether a
        result can get back to whoever asked for it."""
        source, target = str(source), str(target)
        with self._lock:
            if source not in self._nodes or target not in self._nodes:
                return False
            adjacency: dict[str, list[str]] = {}
            for edge in self._edges.values():
                adjacency.setdefault(edge.source, []).append(edge.target)
        seen = {source}
        stack = [source]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            for nxt in adjacency.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def snapshot(self) -> GraphSnapshot:
        with self._lock:
            return GraphSnapshot(
                version=self._version,
                nodes=tuple(sorted(self._nodes)),
                edges=tuple(self._edges[k] for k in sorted(self._edges)),
            )

    @staticmethod
    def diff(before: GraphSnapshot, after: GraphSnapshot) -> GraphDiff:
        before_nodes, after_nodes = set(before.nodes), set(after.nodes)
        before_edges = {e.key: e for e in before.edges}
        after_edges = {e.key: e for e in after.edges}
        return GraphDiff(
            added_nodes=tuple(sorted(after_nodes - before_nodes)),
            removed_nodes=tuple(sorted(before_nodes - after_nodes)),
            added_edges=tuple(after_edges[k] for k in sorted(set(after_edges) - set(before_edges))),
            removed_edges=tuple(before_edges[k] for k in sorted(set(before_edges) - set(after_edges))),
            from_version=before.version,
            to_version=after.version,
        )

    # ── mutation ────────────────────────────────────────────────────────

    class _Scratch:
        """The mutable view handed to a transaction body."""

        def __init__(self, nodes: set[str], edges: dict[tuple[str, str, str, str], MorphEdge], version: int):
            self.nodes = nodes
            self.edges = edges
            self.version = version
            self.touched: list[str] = []

        def add_node(self, node: str) -> None:
            node = str(node)
            if node:
                self.nodes.add(node)
                self.touched.append(f"add_node:{node}")

        def remove_node(self, node: str) -> None:
            node = str(node)
            self.nodes.discard(node)
            for key in [k for k in self.edges if k[0] == node or k[1] == node]:
                self.edges.pop(key, None)
            self.touched.append(f"remove_node:{node}")

        def add_edge(self, edge: MorphEdge) -> None:
            self.edges[edge.key] = edge
            self.touched.append(f"add_edge:{'|'.join(str(v) for v in edge.key)}")

        def remove_edge(self, key: tuple[str, str, str, str]) -> None:
            self.edges.pop(key, None)
            self.touched.append(f"remove_edge:{'|'.join(str(v) for v in key)}")

    def transaction(
        self,
        body,
        *,
        cause: str = "",
        port_contract: Mapping[str, tuple[frozenset[str], frozenset[str]]] | None = None,
    ) -> GraphDiff:
        """Apply ``body`` to a scratch copy and commit it if it validates.

        ``body`` receives a :class:`_Scratch` and mutates it. ``port_contract``
        maps a node id to ``(out_ports, in_ports)``: what it may send and what
        it will accept. Supply it and a binding neither end could carry is
        refused before anything is committed.

        Raises :class:`GraphIntegrityError` when the result is invalid, having
        changed nothing.
        """
        with self._lock:
            before = self.snapshot()
            scratch = self._Scratch(set(self._nodes), dict(self._edges), self._version)
            try:
                body(scratch)
            except GraphIntegrityError:
                raise
            except Exception as exc:  # noqa: BLE001 — a failed body must not commit
                raise GraphIntegrityError(f"transaction body failed: {type(exc).__name__}: {exc}") from exc

            self._validate(scratch.nodes, scratch.edges, port_contract=port_contract)

            new_version = self._version + 1
            stamped = {
                key: (edge if edge.created_at_version else _restamp(edge, new_version))
                for key, edge in scratch.edges.items()
            }
            self._nodes = scratch.nodes
            self._edges = stamped
            self._version = new_version
            self._history.append((new_version, cause or "unspecified"))
            if len(self._history) > 512:
                del self._history[:-512]
            after = self.snapshot()
        return self.diff(before, after)

    def _validate(
        self,
        nodes: set[str],
        edges: dict[tuple[str, str, str, str], MorphEdge],
        *,
        port_contract: Mapping[str, tuple[frozenset[str], frozenset[str]]] | None = None,
    ) -> None:
        if len(nodes) > self.max_nodes:
            raise GraphIntegrityError(f"node budget exceeded: {len(nodes)} > {self.max_nodes}")
        if len(edges) > self.max_edges:
            raise GraphIntegrityError(f"edge budget exceeded: {len(edges)} > {self.max_edges}")

        out_degree: dict[str, int] = {}
        in_degree: dict[str, int] = {}
        for key, edge in edges.items():
            if edge.key != key:
                raise GraphIntegrityError(f"edge stored under a key it does not own: {key} vs {edge.key}")
            if edge.source not in nodes:
                raise GraphIntegrityError(f"dangling edge source: {edge.source}")
            if edge.target not in nodes:
                raise GraphIntegrityError(f"dangling edge target: {edge.target}")
            if edge.source == edge.target:
                raise GraphIntegrityError(f"self-binding is not a topology change: {edge.source}")
            out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        for node, degree in out_degree.items():
            if degree > self.max_out_degree:
                raise GraphIntegrityError(f"out-degree budget exceeded at {node}: {degree}")
        for node, degree in in_degree.items():
            if degree > self.max_in_degree:
                raise GraphIntegrityError(f"in-degree budget exceeded at {node}: {degree}")

        if port_contract is None:
            return
        for edge in edges.values():
            if edge.edge_type not in PORT_CHECKED_EDGES:
                continue
            out_ports = port_contract.get(edge.source, (frozenset(), frozenset()))[0]
            in_ports = port_contract.get(edge.target, (frozenset(), frozenset()))[1]
            if not edge.port:
                raise GraphIntegrityError(f"{edge.edge_type} edge without a port: {edge.key}")
            if edge.port not in out_ports:
                raise GraphIntegrityError(f"{edge.source} cannot send {edge.port!r}")
            if edge.port not in in_ports:
                raise GraphIntegrityError(f"{edge.target} does not accept {edge.port!r}")

    def restore(self, snapshot: GraphSnapshot, *, cause: str = "rollback") -> GraphDiff:
        """Put the graph back to a snapshot, bumping the version.

        Rollback moves forward in version and backward in content, so a reader
        holding v9 is never handed a second, different v9.
        """
        with self._lock:
            before = self.snapshot()
            self._nodes = set(snapshot.nodes)
            self._edges = {e.key: e for e in snapshot.edges}
            self._version += 1
            self._history.append((self._version, cause))
            after = self.snapshot()
        return self.diff(before, after)

    # ── persistence ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self._version,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
                "nodes": sorted(self._nodes),
                "edges": [self._edges[k].to_dict() for k in sorted(self._edges)],
                "history": [{"version": v, "cause": c} for v, c in self._history[-64:]],
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MorphGraph:
        payload = dict(data or {})
        graph = cls(
            max_nodes=int(payload.get("max_nodes", 256)),
            max_edges=int(payload.get("max_edges", 1024)),
        )
        graph._nodes = {str(n) for n in payload.get("nodes", [])}
        for raw in payload.get("edges", []):
            edge = MorphEdge.from_dict(raw)
            if edge.source in graph._nodes and edge.target in graph._nodes and edge.source != edge.target:
                graph._edges[edge.key] = edge
        graph._version = int(payload.get("version", 0))
        graph._history = [
            (int(h.get("version", 0)), str(h.get("cause", "")))
            for h in payload.get("history", [])
        ]
        return graph


def _restamp(edge: MorphEdge, version: int) -> MorphEdge:
    return MorphEdge(
        source=edge.source,
        target=edge.target,
        edge_type=edge.edge_type,
        port=edge.port,
        weight=clamp01(edge.weight) if edge.weight <= 1.0 else float(edge.weight),
        latency_ms=float(edge.latency_ms),
        capacity=int(edge.capacity),
        created_at_version=int(version),
        metadata=copy.deepcopy(edge.metadata),
    )


def iter_edge_keys(edges: Iterable[MorphEdge]) -> Iterator[tuple[str, str, str, str]]:
    for edge in edges:
        yield edge.key


__all__ = [
    "EdgeType",
    "GraphDiff",
    "GraphIntegrityError",
    "GraphSnapshot",
    "MorphEdge",
    "MorphGraph",
    "PORT_CHECKED_EDGES",
    "iter_edge_keys",
]
