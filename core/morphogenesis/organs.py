from __future__ import annotations

import itertools
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from .types import CellManifest, CellRole, json_safe, stable_digest


@dataclass
class Organ:
    organ_id: str
    name: str
    members: List[str]
    subsystem: str = "composite"
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    task_signatures: List[str] = field(default_factory=list)

    def to_manifest(self) -> CellManifest:
        return CellManifest(
            name=f"organ:{self.name}",
            role=CellRole.ORGAN,
            subsystem=self.subsystem,
            capabilities=["composite", "organ", *self.task_signatures[:6]],
            consumes=["task", "repair", "growth", "danger"],
            emits=["task", "repair", "growth"],
            dependencies=list(self.members),
            protected=False,
            criticality=min(0.9, 0.4 + self.confidence * 0.5),
            baseline_energy=0.45,
            metadata={"organ_id": self.organ_id, "members": list(self.members)},
        )

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "organ_id": self.organ_id,
            "name": self.name,
            "members": self.members,
            "subsystem": self.subsystem,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "activation_count": self.activation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "task_signatures": self.task_signatures,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Organ":
        return cls(**dict(data or {}))


class OrganStabilizer:
    """Detect stable co-activated cell coalitions without networkx.

    It maintains a bounded co-activation graph.  If a connected component is
    repeatedly co-active, successful, and large enough, it can be formalized
    as an Organ.
    """

    def __init__(
        self,
        *,
        min_coactivations: int = 6,
        min_members: int = 2,
        edge_threshold: float = 0.62,
        window: int = 512,
    ):
        self.min_coactivations = int(min_coactivations)
        self.min_members = int(min_members)
        self.edge_threshold = float(edge_threshold)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=window)
        self._pair_counts: Counter[Tuple[str, str]] = Counter()
        self._pair_success: Counter[Tuple[str, str]] = Counter()
        self._known_organs: Dict[str, Organ] = {}

    def observe_activation(
        self,
        cell_ids: Iterable[str],
        *,
        success: bool,
        task_signature: str = "",
        subsystem: str = "composite",
    ) -> None:
        ids = sorted({str(cid) for cid in cell_ids if cid})
        if len(ids) < 2:
            return
        event = {
            "cell_ids": ids,
            "success": bool(success),
            "task_signature": task_signature,
            "subsystem": subsystem,
            "timestamp": time.time(),
        }
        self._events.append(event)
        for a, b in itertools.combinations(ids, 2):
            pair = (a, b)
            self._pair_counts[pair] += 1
            if success:
                self._pair_success[pair] += 1

    def discover(self) -> List[Organ]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for pair, count in self._pair_counts.items():
            if count < self.min_coactivations:
                continue
            success_rate = self._pair_success[pair] / max(1, count)
            if success_rate < self.edge_threshold:
                continue
            a, b = pair
            graph[a].add(b)
            graph[b].add(a)

        components: List[Set[str]] = []
        seen: Set[str] = set()
        for node in list(graph):
            if node in seen:
                continue
            stack = [node]
            comp: Set[str] = set()
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                comp.add(n)
                stack.extend(graph[n] - seen)
            if len(comp) >= self.min_members:
                components.append(comp)

        organs: List[Organ] = []
        for comp in components:
            members = sorted(comp)
            oid = "organ_" + stable_digest(*members, length=12)
            if oid in self._known_organs:
                continue
            related_events = [e for e in self._events if set(e["cell_ids"]).issuperset(comp)]
            if not related_events:
                continue
            successes = sum(1 for e in related_events if e["success"])
            confidence = successes / max(1, len(related_events))
            task_counts = Counter(e.get("task_signature", "") for e in related_events if e.get("task_signature"))
            subsystem_counts = Counter(e.get("subsystem", "composite") for e in related_events)
            organ = Organ(
                organ_id=oid,
                name="-".join(m.split("_")[-1][:6] for m in members[:4]),
                members=members,
                subsystem=subsystem_counts.most_common(1)[0][0] if subsystem_counts else "composite",
                confidence=float(confidence),
                activation_count=len(related_events),
                success_count=successes,
                failure_count=len(related_events) - successes,
                task_signatures=[k for k, _ in task_counts.most_common(6)],
            )
            self._known_organs[oid] = organ
            organs.append(organ)
        return organs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_counts": {"|".join(k): v for k, v in self._pair_counts.items()},
            "pair_success": {"|".join(k): v for k, v in self._pair_success.items()},
            "known_organs": {oid: organ.to_dict() for oid, organ in self._known_organs.items()},
            "events": list(self._events)[-64:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "OrganStabilizer":
        inst = cls(**kwargs)
        for raw, v in dict(data.get("pair_counts", {})).items():
            parts = raw.split("|", 1)
            if len(parts) == 2:
                inst._pair_counts[(parts[0], parts[1])] = int(v)
        for raw, v in dict(data.get("pair_success", {})).items():
            parts = raw.split("|", 1)
            if len(parts) == 2:
                inst._pair_success[(parts[0], parts[1])] = int(v)
        inst._known_organs = {
            oid: Organ.from_dict(payload)
            for oid, payload in dict(data.get("known_organs", {})).items()
        }
        for e in list(data.get("events", []))[-64:]:
            inst._events.append(dict(e))
        return inst
