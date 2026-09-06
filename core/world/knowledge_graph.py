"""core/world/knowledge_graph.py — Structured Entity-Relation Graph.

Handles entities, relationships, law/regulations, codebases, and target state domains.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

logger = logging.getLogger("Aura.KnowledgeGraph")


@dataclass
class GraphNode:
    node_id: str
    kind: str  # "person", "organization", "project", "technology", "codebase", "regulation"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relationship: str  # "depends_on", "owns", "author_of", "competes_with", "violates"
    attributes: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Manages the structured model entities and semantic links."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, set())

    def add_edge(self, edge: GraphEdge) -> None:
        # Enforce that both nodes exist in the graph
        if edge.source_id not in self.nodes:
            self.add_node(GraphNode(edge.source_id, "unknown"))
        if edge.target_id not in self.nodes:
            self.add_node(GraphNode(edge.target_id, "unknown"))

        self.edges.append(edge)
        self._adjacency.setdefault(edge.source_id, set()).add(edge.target_id)
        logger.debug("Added edge: %s --[%s]--> %s", edge.source_id, edge.relationship, edge.target_id)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> List[GraphEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming(self, node_id: str) -> List[GraphEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def find_path(self, start_id: str, end_id: str, visited: Set[str] | None = None) -> List[str] | None:
        """Finds a path between nodes using simple depth-first search."""
        if visited is None:
            visited = set()
        if start_id == end_id:
            return [start_id]
        if start_id not in self._adjacency:
            return None

        visited.add(start_id)
        for neighbor in self._adjacency[start_id]:
            if neighbor not in visited:
                path = self.find_path(neighbor, end_id, visited)
                if path:
                    return [start_id] + path
        return None


# Singleton
_graph_instance: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = KnowledgeGraph()
    return _graph_instance


# ---------------------------------------------------- the shared graph shape
#
# The adapter lives here rather than in core.knowledge because the interface
# may not reach this package, and opening the layering so an interface could
# describe the system would be describing a different system. A store adapts
# itself; core.knowledge.one_graph only says what the shape is.

from core.knowledge.one_graph import (  # noqa: E402
    ALink,
    ANode,
    a_store_that_is_a_graph,
    the_same,
)


class TheKnowledgeGraphAsAGraph:
    """core/world/knowledge_graph.py, under the shared shape."""

    def __init__(self, inner: Any = None) -> None:
        self._inner = inner if inner is not None else KnowledgeGraph()

    def put_node(self, node: ANode) -> str:
        self._inner.add_node(
            GraphNode(node.node_id, node.kind, {"name": node.name, **node.attributes})
        )
        return node.node_id

    def put_link(self, link: ALink) -> None:
        self._inner.add_edge(
            GraphEdge(link.source_id, link.target_id, link.kind, dict(link.attributes))
        )

    def node(self, node_id: str) -> ANode | None:
        found = self._inner.get_node(the_same(node_id))
        if found is None:
            return None
        attributes = dict(getattr(found, "attributes", {}) or {})
        return ANode(
            found.node_id, found.kind, str(attributes.pop("name", "")), attributes
        )

    def out_of(self, node_id: str) -> list[ALink]:
        return [
            ALink(one.relationship, one.source_id, one.target_id, dict(one.attributes))
            for one in self._inner.get_outgoing(the_same(node_id))
        ]

    def into(self, node_id: str) -> list[ALink]:
        return [
            ALink(one.relationship, one.source_id, one.target_id, dict(one.attributes))
            for one in self._inner.get_incoming(the_same(node_id))
        ]

    def all_nodes(self) -> list[ANode]:
        return [self.node(one) for one in list(self._inner.nodes)]


@a_store_that_is_a_graph("knowledge_graph")
def _the_knowledge_graph(*, live: bool = False):
    inner = None
    if live:
        from core.container import ServiceContainer
        from core.service_names import ServiceNames

        inner = ServiceContainer.get(ServiceNames.KNOWLEDGE_GRAPH, default=None)
        if inner is None:
            raise RuntimeError("no live knowledge graph")
    return TheKnowledgeGraphAsAGraph(inner)
