"""Persistent graph memory for bounded environments."""
from __future__ import annotations

import hashlib
import logging
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
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


class SpatialCell(dict):
    """Dict-backed spatial entry with legacy string equality.

    Older tests and callers treated ``belief.spatial[(ctx, x, y)]`` as a raw
    kind string. Newer infrastructure needs confidence, provenance, walkability,
    and node links. This keeps both contracts live.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.get("kind") == other
        return super().__eq__(other)


class EnvironmentBeliefGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, BeliefNode] = {}
        self.edges: list[BeliefEdge] = []
        self.context_stack: list[str] = []
        self.spatial: dict[tuple[str, int, int], dict[str, Any]] = {}
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

    def upsert_spatial(
        self,
        context_id: str,
        x: int,
        y: int,
        *,
        kind: str,
        confidence: float = 0.7,
        evidence_id: str = "",
        node_id: str = "",
        walkable: bool | None = None,
        properties: dict[str, Any] | None = None,
        sequence_id: int = 0,
    ) -> dict[str, Any]:
        """Merge a spatial observation into the canonical environment map.

        The map is conservative around hazards: a later low-confidence benign
        observation cannot erase a high-confidence hazard. This prevents
        split-brain map repair where perception momentarily loses sight of a
        trap, wall, unsafe UI region, or other risk-bearing location.
        """
        key = (str(context_id), int(x), int(y))
        incoming = SpatialCell({
            "kind": str(kind or "unknown"),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "evidence_id": evidence_id,
            "node_id": node_id,
            "walkable": walkable,
            "properties": dict(properties or {}),
            "last_seen_seq": int(sequence_id or 0),
        })
        existing = self.spatial.get(key)
        if existing:
            old_kind = str(existing.get("kind", "unknown"))
            old_conf = float(existing.get("confidence", 0.0) or 0.0)
            high_conf_hazard = old_kind in {"hazard", "trap", "damage", "hostile_entity"} and old_conf >= 0.7
            incoming_benign = incoming["kind"] in {"floor", "player", "unknown", "empty"}
            if high_conf_hazard and incoming_benign and incoming["confidence"] < old_conf:
                existing["last_seen_seq"] = max(int(existing.get("last_seen_seq", 0) or 0), incoming["last_seen_seq"])
                return existing
            if old_kind != incoming["kind"] and old_conf >= 0.7 and incoming["confidence"] >= 0.7:
                self.contradictions.append(
                    {
                        "spatial_key": key,
                        "old_kind": old_kind,
                        "new_kind": incoming["kind"],
                        "seq": incoming["last_seen_seq"],
                    }
                )
            existing.update(
                {
                    "kind": incoming["kind"] if incoming["confidence"] >= old_conf * 0.75 else old_kind,
                    "confidence": max(old_conf * 0.9, incoming["confidence"]),
                    "evidence_id": evidence_id or existing.get("evidence_id", ""),
                    "node_id": node_id or existing.get("node_id", ""),
                    "last_seen_seq": max(int(existing.get("last_seen_seq", 0) or 0), incoming["last_seen_seq"]),
                }
            )
            if walkable is not None:
                existing["walkable"] = walkable
            existing.setdefault("properties", {}).update(incoming["properties"])
            return existing
        self.spatial[key] = incoming
        return incoming

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
                        relation="transition",
                        confidence=0.8,
                        last_confirmed_seq=state.sequence_id,
                    )
                )
            self.context_stack.append(context)
        self.upsert_node(BeliefNode(f"context:{context}", "context", context, context, last_seen_seq=state.sequence_id))
        
        # Update spatial
        local_coords = state.self_state.get("local_coordinates")
        if isinstance(local_coords, (list, tuple)) and len(local_coords) >= 2:
            self.upsert_spatial(
                context,
                int(local_coords[0]),
                int(local_coords[1]),
                kind="player",
                confidence=0.95,
                evidence_id=state.raw_observation_ref,
                node_id=f"{state.environment_id}:self",
                walkable=True,
                properties={"self": True},
                sequence_id=state.sequence_id,
            )
            
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
            if entity.position is not None:
                self.upsert_spatial(
                    context,
                    int(entity.position[0]),
                    int(entity.position[1]),
                    kind="hostile_entity" if entity.entity_id in self.hazards else "entity",
                    confidence=entity.confidence,
                    evidence_id=entity.evidence_ref,
                    node_id=entity.entity_id,
                    walkable=False if entity.entity_id in self.hazards else None,
                    properties={"label": entity.label, "threat_score": entity.threat_score},
                    sequence_id=state.sequence_id,
                )
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
            # Inter-level edges for transitions (stairs, portals)
            if obj.kind == "transition":
                glyph = obj.properties.get("glyph", "")
                if glyph == ">" and state.self_state:
                    dlvl = state.self_state.get("dlvl")
                    if dlvl is not None:
                        target_ctx = f"dlvl_{int(dlvl) + 1}"
                        self.upsert_edge(
                            BeliefEdge(
                                from_id=f"context:{context}",
                                to_id=f"context:{target_ctx}",
                                relation="connects",
                                properties={"direction": "down", "via": obj.object_id},
                                confidence=0.9,
                                last_confirmed_seq=state.sequence_id,
                            )
                        )
                elif glyph == "<" and state.self_state:
                    dlvl = state.self_state.get("dlvl")
                    if dlvl is not None and int(dlvl) > 1:
                        target_ctx = f"dlvl_{int(dlvl) - 1}"
                        self.upsert_edge(
                            BeliefEdge(
                                from_id=f"context:{context}",
                                to_id=f"context:{target_ctx}",
                                relation="connects",
                                properties={"direction": "up", "via": obj.object_id},
                                confidence=0.9,
                                last_confirmed_seq=state.sequence_id,
                            )
                        )
                self.mark_frontier(obj.object_id)
            if obj.position is not None:
                spatial_kind = "transition" if obj.kind == "transition" else obj.kind
                self.upsert_spatial(
                    context,
                    int(obj.position[0]),
                    int(obj.position[1]),
                    kind=spatial_kind,
                    confidence=obj.confidence,
                    evidence_id=obj.evidence_ref,
                    node_id=obj.object_id,
                    walkable=True if obj.kind in {"transition", "item", "resource"} else None,
                    properties={
                        "label": obj.label,
                        "affordances": list(obj.affordances),
                        "risk_tags": list(obj.risk_tags),
                    },
                    sequence_id=state.sequence_id,
                )
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
            pos = hazard.properties.get("position") or hazard.properties.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                self.upsert_spatial(
                    context,
                    int(pos[0]),
                    int(pos[1]),
                    kind="hazard",
                    confidence=hazard.confidence,
                    evidence_id=hazard.evidence_ref,
                    node_id=hazard.hazard_id,
                    walkable=False,
                    properties={"label": hazard.label, "severity": hazard.severity},
                    sequence_id=state.sequence_id,
                )
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
            "spatial": {
                f"{ctx}:{x}:{y}": value
                for (ctx, x, y), value in sorted(self.spatial.items(), key=lambda item: item[0])
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def current_position(self, context_id: str | None = None) -> tuple[int, int] | None:
        """Return the most recent known self position in a context."""
        candidates: list[tuple[int, int, int]] = []
        for (ctx, x, y), entry in self.spatial.items():
            if context_id is not None and ctx != context_id:
                continue
            if entry.get("kind") == "player" or entry.get("properties", {}).get("self"):
                candidates.append((int(entry.get("last_seen_seq", 0) or 0), x, y))
        if not candidates:
            return None
        _, x, y = max(candidates)
        return (x, y)

    def nearest_spatial(
        self,
        *,
        context_id: str,
        kinds: set[str],
        origin: tuple[int, int] | None = None,
        min_confidence: float = 0.2,
    ) -> tuple[int, int, dict[str, Any]] | None:
        """Find the nearest known spatial entry matching one of ``kinds``."""
        origin = origin or self.current_position(context_id)
        best_distance: float | None = None
        best_entry: tuple[int, int, dict[str, Any]] | None = None
        for (ctx, x, y), entry in self.spatial.items():
            if ctx != context_id:
                continue
            if str(entry.get("kind")) not in kinds:
                continue
            if float(entry.get("confidence", 0.0) or 0.0) < min_confidence:
                continue
            distance = abs(x - origin[0]) + abs(y - origin[1]) if origin else 0
            if best_distance is None or float(distance) < best_distance:
                best_distance = float(distance)
                best_entry = (x, y, entry)
        if best_entry is None:
            return None
        return best_entry

    # ------------------------------------------------------------------
    # Frontier target prioritization
    # ------------------------------------------------------------------

    def get_frontier_targets(self) -> list[str]:
        """Return frontier node IDs sorted by descending confidence."""
        targets = []
        for fid in self.frontiers:
            node = self.nodes.get(fid)
            conf = node.confidence if node else 0.5
            targets.append((conf, fid))
        targets.sort(reverse=True)
        return [fid for _, fid in targets]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize belief graph to JSON file."""
        payload = {
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
            "frontiers": sorted(self.frontiers),
            "hazards": sorted(self.hazards),
            "context_stack": list(self.context_stack),
            "contradictions": list(self.contradictions),
            "spatial": [
                {"context_id": ctx, "x": x, "y": y, **value}
                for (ctx, x, y), value in self.spatial.items()
            ],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Deserialize belief graph from JSON file."""
        p = Path(path)
        if not p.exists():
            logging.getLogger(__name__).warning("belief_graph_load_missing: %s", p)
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.nodes.clear()
        self.edges.clear()
        self.frontiers.clear()
        self.hazards.clear()
        self.spatial.clear()
        self.context_stack.clear()
        self.contradictions.clear()
        for nid, ndata in data.get("nodes", {}).items():
            self.nodes[nid] = BeliefNode(**ndata)
        for edata in data.get("edges", []):
            self.edges.append(BeliefEdge(**edata))
        self.frontiers = set(data.get("frontiers", []))
        self.hazards = set(data.get("hazards", []))
        self.context_stack = list(data.get("context_stack", []))
        self.contradictions = list(data.get("contradictions", []))
        for item in data.get("spatial", []):
            ctx = str(item.pop("context_id"))
            x = int(item.pop("x"))
            y = int(item.pop("y"))
            self.spatial[(ctx, x, y)] = SpatialCell(item)


__all__ = ["BeliefNode", "BeliefEdge", "EnvironmentBeliefGraph"]
