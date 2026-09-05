from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Dict, Iterable, Optional

from .unity_state import BoundContent

_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _jaccard(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


@dataclass(frozen=True)
class CoPresenceNode:
    id: str
    kind: str
    source: str
    summary: str
    salience: float
    confidence: float
    timestamp: float
    ownership: str
    action_relevance: float
    affective_charge: float
    ttl_s: float = 4.0


@dataclass(frozen=True)
class CoPresenceEdge:
    src: str
    dst: str
    relation: str
    weight: float
    evidence: str | None = None


@dataclass(frozen=True)
class CoPresenceGraphSnapshot:
    cluster_id: str
    nodes: list[CoPresenceNode]
    edges: list[CoPresenceEdge]
    focus_id: str | None
    peripheral_ids: list[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class CoPresenceGraphBuilder:
    """Build an explicit graph of what is present together right now."""

    def _node_score(self, node: CoPresenceNode, focus_hint: str = "") -> float:
        hint_bonus = 0.25 * _jaccard(node.summary, focus_hint) if focus_hint else 0.0
        return (
            float(node.salience or 0.0) * 0.5
            + float(node.action_relevance or 0.0) * 0.35
            + float(node.confidence or 0.0) * 0.15
            + hint_bonus
        )

    def build(
        self,
        contents: Iterable[BoundContent],
        *,
        focus_hint: str = "",
        cluster_id: str = "",
    ) -> CoPresenceGraphSnapshot:
        nodes = [
            CoPresenceNode(
                id=item.content_id,
                kind=item.modality,
                source=item.source,
                summary=item.summary,
                salience=float(item.salience or 0.0),
                confidence=float(item.confidence or 0.0),
                timestamp=float(item.timestamp or 0.0),
                ownership=item.ownership,
                action_relevance=float(item.action_relevance or 0.0),
                affective_charge=float(item.affective_charge or 0.0),
            )
            for item in contents
        ]

        if not nodes:
            return CoPresenceGraphSnapshot(
                cluster_id=cluster_id or "cluster_empty",
                nodes=[],
                edges=[],
                focus_id=None,
                peripheral_ids=[],
                metrics={
                    "largest_connected_component_ratio": 0.0,
                    "focus_periphery_binding_strength": 0.0,
                    "conflict_density": 0.0,
                    "unowned_content_ratio": 1.0,
                    "cross_modal_edge_density": 0.0,
                    "action_relevant_cluster_strength": 0.0,
                },
            )

        edges: list[CoPresenceEdge] = []
        for idx, left in enumerate(nodes):
            for right in nodes[idx + 1 :]:
                overlap = _jaccard(left.summary, right.summary)
                same_owner = left.ownership == right.ownership
                modality_bridge = 1.0 if left.kind != right.kind else 0.0

                if overlap > 0.0:
                    edges.append(
                        CoPresenceEdge(
                            src=left.id,
                            dst=right.id,
                            relation="refers_to",
                            weight=round(min(1.0, 0.25 + overlap * 0.75), 4),
                            evidence="shared language",
                        )
                    )

                if same_owner:
                    relation = "belongs_to_self" if left.ownership == "self" else "belongs_to_world"
                    edges.append(
                        CoPresenceEdge(
                            src=left.id,
                            dst=right.id,
                            relation=relation,
                            weight=0.2,
                            evidence="shared ownership",
                        )
                    )

                if max(left.action_relevance, right.action_relevance) > 0.6:
                    edges.append(
                        CoPresenceEdge(
                            src=left.id,
                            dst=right.id,
                            relation="requires_action",
                            weight=round(0.25 + max(left.action_relevance, right.action_relevance) * 0.5, 4),
                            evidence="shared action pressure",
                        )
                    )

                charge_product = left.affective_charge * right.affective_charge
                if charge_product < -0.08:
                    edges.append(
                        CoPresenceEdge(
                            src=left.id,
                            dst=right.id,
                            relation="conflicts_with",
                            weight=round(min(1.0, abs(charge_product)), 4),
                            evidence="opposed affective charge",
                        )
                    )
                elif overlap > 0.15 or modality_bridge > 0.0:
                    edges.append(
                        CoPresenceEdge(
                            src=left.id,
                            dst=right.id,
                            relation="supports",
                            weight=round(min(1.0, 0.15 + overlap * 0.4 + modality_bridge * 0.2), 4),
                            evidence="co-present support",
                        )
                    )

        focus_node = max(nodes, key=lambda node: self._node_score(node, focus_hint=focus_hint))
        peripheral_nodes = [
            node
            for node in sorted(nodes, key=lambda node: self._node_score(node, focus_hint=focus_hint), reverse=True)
            if node.id != focus_node.id
        ][:6]

        adjacency: dict[str, set[str]] = {node.id: set() for node in nodes}
        node_map = {node.id: node for node in nodes}
        for edge in edges:
            adjacency.setdefault(edge.src, set()).add(edge.dst)
            adjacency.setdefault(edge.dst, set()).add(edge.src)

        visited: set[str] = set()
        largest_component = 0
        for node in nodes:
            if node.id in visited:
                continue
            stack = [node.id]
            size = 0
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                size += 1
                stack.extend(adjacency.get(current, set()) - visited)
            largest_component = max(largest_component, size)

        focus_edge_weights = [
            edge.weight
            for edge in edges
            if focus_node.id in {edge.src, edge.dst} and (
                edge.src in {node.id for node in peripheral_nodes}
                or edge.dst in {node.id for node in peripheral_nodes}
            )
        ]
        conflict_edges = [edge for edge in edges if edge.relation == "conflicts_with"]
        cross_modal_edges = [
            edge
            for edge in edges
            if edge.src in node_map
            and edge.dst in node_map
            and node_map[edge.src].kind != node_map[edge.dst].kind
        ]
        focus_component_nodes = {focus_node.id} | adjacency.get(focus_node.id, set())
        action_cluster_strength = sum(
            node.action_relevance for node in nodes if node.id in focus_component_nodes
        ) / max(1, len(focus_component_nodes))
        ambiguous_nodes = [node for node in nodes if node.ownership == "ambiguous"]

        metrics = {
            "largest_connected_component_ratio": round(largest_component / max(1, len(nodes)), 4),
            "focus_periphery_binding_strength": round(sum(focus_edge_weights) / max(1, len(focus_edge_weights)), 4),
            "conflict_density": round(len(conflict_edges) / max(1, len(edges)), 4),
            "unowned_content_ratio": round(len(ambiguous_nodes) / max(1, len(nodes)), 4),
            "cross_modal_edge_density": round(len(cross_modal_edges) / max(1, len(edges)), 4),
            "action_relevant_cluster_strength": round(action_cluster_strength, 4),
        }

        return CoPresenceGraphSnapshot(
            cluster_id=cluster_id or f"cluster_{focus_node.id}",
            nodes=nodes,
            edges=edges,
            focus_id=focus_node.id,
            peripheral_ids=[node.id for node in peripheral_nodes],
            metrics=metrics,
        )
