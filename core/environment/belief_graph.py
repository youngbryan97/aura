"""Persistent graph memory for bounded environments."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from .parsed_state import ParsedState


@dataclass
class BeliefNode:
    node_id: str
    kind: str
    label: str
    context_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    last_seen_seq: int = 0


@dataclass
class BeliefEdge:
    from_id: str
    to_id: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    last_confirmed_seq: int = 0


class EnvironmentBeliefGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, BeliefNode] = {}
        self.edges: list[BeliefEdge] = []
        self.context_stack: list[str] = []
        self.frontiers: set[str] = set()
        self.hazards: set[str] = set()
        self.contradictions: list[dict[str, Any]] = []

    def upsert_node(self, node: BeliefNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing and existing.kind != node.kind and node.confidence >= 0.7 and existing.confidence >= 0.7:
            self.contradictions.append(
                {
                    "node_id": node.node_id,
                    "old_kind": existing.kind,
                    "new_kind": node.kind,
                    "seq": node.last_seen_seq,
                }
            )
        if existing:
            existing.kind = node.kind or existing.kind
            existing.label = node.label or existing.label
            existing.context_id = node.context_id or existing.context_id
            existing.properties.update(node.properties)
            existing.confidence = max(existing.confidence * 0.9, node.confidence)
            existing.last_seen_seq = max(existing.last_seen_seq, node.last_seen_seq)
        else:
            self.nodes[node.node_id] = node

    def upsert_edge(self, edge: BeliefEdge) -> None:
        for existing in self.edges:
            if (
                existing.from_id == edge.from_id
                and existing.to_id == edge.to_id
                and existing.relation == edge.relation
            ):
                existing.properties.update(edge.properties)
                existing.confidence = max(existing.confidence * 0.9, edge.confidence)
                existing.last_confirmed_seq = max(existing.last_confirmed_seq, edge.last_confirmed_seq)
                return
        self.edges.append(edge)

    def mark_frontier(self, node_id: str) -> None:
        self.frontiers.add(node_id)

    def mark_hazard(self, node_id: str) -> None:
        self.hazards.add(node_id)

    def decay_unobserved(self, current_seq: int, *, half_life_steps: int = 200) -> None:
        half_life_steps = max(1, half_life_steps)
        for node in self.nodes.values():
            age = max(0, current_seq - int(node.last_seen_seq or 0))
            if age:
                node.confidence = max(0.05, node.confidence * (0.5 ** (age / half_life_steps)))
        for edge in self.edges:
            age = max(0, current_seq - int(edge.last_confirmed_seq or 0))
            if age:
                edge.confidence = max(0.05, edge.confidence * (0.5 ** (age / half_life_steps)))

    def update_from_parsed_state(self, state: ParsedState) -> None:
        context = state.context_id or "unknown"
        if not self.context_stack or self.context_stack[-1] != context:
            if self.context_stack:
                self.upsert_edge(
                    BeliefEdge(
                        from_id=f"context:{self.context_stack[-1]}",
                        to_id=f"context:{context}",
                        relation="transitioned_to",
                        confidence=0.8,
                        last_confirmed_seq=state.sequence_id,
                    )
                )
            self.context_stack.append(context)
        self.upsert_node(BeliefNode(f"context:{context}", "context", context, context, last_seen_seq=state.sequence_id))
        for entity in state.entities:
            self.upsert_node(
                BeliefNode(
                    entity.entity_id,
                    f"entity:{entity.kind}",
                    entity.label,
                    entity.context_id,
                    properties=entity.properties,
                    confidence=entity.confidence,
                    last_seen_seq=entity.last_seen_seq or state.sequence_id,
                )
            )
            self.upsert_edge(
                BeliefEdge(
                    f"context:{context}",
                    entity.entity_id,
                    "contains",
                    confidence=entity.confidence,
                    last_confirmed_seq=state.sequence_id,
                )
            )
            if entity.threat_score >= 0.5 or entity.kind == "hostile":
                self.mark_hazard(entity.entity_id)
        for obj in state.objects:
            self.upsert_node(
                BeliefNode(
                    obj.object_id,
                    f"object:{obj.kind}",
                    obj.label,
                    obj.context_id,
                    properties={**obj.properties, "affordances": obj.affordances, "risk_tags": obj.risk_tags},
                    confidence=obj.confidence,
                    last_seen_seq=obj.last_seen_seq or state.sequence_id,
                )
            )
            self.upsert_edge(
                BeliefEdge(f"context:{context}", obj.object_id, "contains", confidence=obj.confidence, last_confirmed_seq=state.sequence_id)
            )
            if obj.kind == "transition":
                self.mark_frontier(obj.object_id)
        for hazard in state.hazards:
            self.upsert_node(
                BeliefNode(
                    hazard.hazard_id,
                    f"hazard:{hazard.kind}",
                    hazard.label,
                    hazard.context_id,
                    properties={**hazard.properties, "severity": hazard.severity},
                    confidence=hazard.confidence,
                    last_seen_seq=hazard.last_seen_seq or state.sequence_id,
                )
            )
            self.mark_hazard(hazard.hazard_id)
        self.decay_unobserved(state.sequence_id)

    def record_blocked_edge(self, from_id: str, to_id: str, *, seq: int = 0, reason: str = "") -> None:
        self.upsert_edge(
            BeliefEdge(
                from_id=from_id,
                to_id=to_id,
                relation="blocks",
                properties={"reason": reason},
                confidence=0.9,
                last_confirmed_seq=seq,
            )
        )

    def shortest_safe_path(self, start_id: str, goal_id: str) -> list[str]:
        blocked = {(e.from_id, e.to_id) for e in self.edges if e.relation == "blocks" and e.confidence >= 0.5}
        adjacency: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.relation not in {"adjacent", "transitioned_to", "contains", "navigates_to"}:
                continue
            if (edge.from_id, edge.to_id) in blocked:
                continue
            if edge.to_id in self.hazards:
                continue
            adjacency.setdefault(edge.from_id, []).append(edge.to_id)
        queue: deque[tuple[str, list[str]]] = deque([(start_id, [start_id])])
        seen = {start_id}
        while queue:
            node, path = queue.popleft()
            if node == goal_id:
                return path
            for nxt in adjacency.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return []

    def stable_hash(self) -> str:
        payload = {
            "nodes": {k: asdict(v) for k, v in sorted(self.nodes.items())},
            "edges": [asdict(e) for e in sorted(self.edges, key=lambda e: (e.from_id, e.to_id, e.relation))],
            "frontiers": sorted(self.frontiers),
            "hazards": sorted(self.hazards),
            "context_stack": list(self.context_stack),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


__all__ = ["BeliefNode", "BeliefEdge", "EnvironmentBeliefGraph"]
