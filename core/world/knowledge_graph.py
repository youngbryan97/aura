"""core/world/knowledge_graph.py — Structured Entity-Relation Graph.

Handles entities, relationships, law/regulations, codebases, and target state domains.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.KnowledgeGraph")


@dataclass
class GraphNode:
    node_id: str
    kind: str  # "person", "organization", "project", "technology", "codebase", "regulation"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relationship: str  # "depends_on", "owns", "author_of", "competes_with", "violates"
    attributes: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Manages the structured model entities and semantic links."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._adjacency: dict[str, set[str]] = {}

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

    def get_outgoing(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def find_path(self, start_id: str, end_id: str, visited: set[str] | None = None) -> list[str] | None:
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
