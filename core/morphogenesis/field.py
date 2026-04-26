from __future__ import annotations

import copy
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .types import MorphogenSignal, SignalKind, clamp01, json_safe

_FIELD_NAMES = (
    "danger",
    "inflammation",
    "damage",
    "repair",
    "growth",
    "curiosity",
    "novelty",
    "task_pressure",
    "resource_pressure",
    "memory_pressure",
    "social",
    "homeostasis",
    "inhibition",
)


@dataclass
class TissueNode:
    subsystem: str
    values: Dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in _FIELD_NAMES})
    last_updated: float = field(default_factory=time.time)

    def perturb(self, field_name: str, amount: float) -> None:
        if field_name not in self.values:
            self.values[field_name] = 0.0
        self.values[field_name] = clamp01(self.values[field_name] + amount)
        self.last_updated = time.time()

    def decay(self, decay: float) -> None:
        decay = clamp01(decay)
        for name in list(self.values):
            self.values[name] = clamp01(self.values[name] * (1.0 - decay))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "values": {k: round(float(v), 5) for k, v in self.values.items()},
            "last_updated": self.last_updated,
        }


class MorphogenField:
    """Diffusive field over Aura subsystems.

    This is deliberately close to AdaptiveImmunity.TissueField, but generalized:
    danger/damage still exist, while growth/novelty/task/resource fields let
    healthy cell coalitions self-organize rather than only heal failures.
    """

    def __init__(self, *, diffusion: float = 0.18, decay: float = 0.08):
        self.diffusion = clamp01(diffusion)
        self.decay_rate = clamp01(decay)
        self._nodes: Dict[str, TissueNode] = {}
        self._edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.RLock()

    def ensure_node(self, subsystem: str) -> TissueNode:
        subsystem = str(subsystem or "generic")
        node = self._nodes.get(subsystem)
        if node is None:
            node = TissueNode(subsystem=subsystem)
            self._nodes[subsystem] = node
        return node

    def register_edge(self, a: str, b: str, weight: float = 1.0) -> None:
        a, b = str(a or "generic"), str(b or "generic")
        weight = clamp01(weight)
        with self._lock:
            self.ensure_node(a)
            self.ensure_node(b)
            self._edges[a][b] = max(self._edges[a].get(b, 0.0), weight)
            self._edges[b][a] = max(self._edges[b].get(a, 0.0), weight)

    def perturb(self, subsystem: str, field_name: str, amount: float) -> None:
        with self._lock:
            self.ensure_node(subsystem).perturb(field_name, amount)

    def ingest_signal(self, signal: MorphogenSignal) -> None:
        kind = signal.kind.value if isinstance(signal.kind, SignalKind) else str(signal.kind)
        intensity = clamp01(signal.intensity)
        subsystem = signal.subsystem or "generic"

        mapping = {
            SignalKind.ERROR.value: ("danger", "damage", "repair"),
            SignalKind.EXCEPTION.value: ("danger", "damage", "repair"),
            SignalKind.DANGER.value: ("danger", "inflammation"),
            SignalKind.RESOURCE_PRESSURE.value: ("resource_pressure", "inhibition"),
            SignalKind.MEMORY_PRESSURE.value: ("memory_pressure", "repair"),
            SignalKind.TASK.value: ("task_pressure", "growth"),
            SignalKind.USER_NEED.value: ("task_pressure", "social"),
            SignalKind.NOVELTY.value: ("novelty", "curiosity"),
            SignalKind.CURIOSITY.value: ("curiosity", "growth"),
            SignalKind.REPAIR.value: ("repair",),
            SignalKind.GROWTH.value: ("growth",),
            SignalKind.SOCIAL.value: ("social",),
            SignalKind.INHIBITION.value: ("inhibition",),
            SignalKind.HOMEOSTASIS.value: ("homeostasis",),
            SignalKind.HEARTBEAT.value: ("homeostasis",),
        }
        fields = mapping.get(kind, ("task_pressure",))
        with self._lock:
            node = self.ensure_node(subsystem)
            for field_name in fields:
                node.perturb(field_name, intensity)
            if signal.target_cell_id:
                node.perturb("task_pressure", min(0.25, intensity * 0.35))

    def diffuse_step(self) -> None:
        with self._lock:
            # Snapshot first; update second to keep diffusion deterministic.
            current = {name: copy.deepcopy(node.values) for name, node in self._nodes.items()}
            increments: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

            for src, neighbours in self._edges.items():
                src_values = current.get(src)
                if not src_values:
                    continue
                for dst, weight in neighbours.items():
                    if dst not in self._nodes:
                        continue
                    for field_name, value in src_values.items():
                        increments[dst][field_name] += value * self.diffusion * weight

            for name, node in self._nodes.items():
                node.decay(self.decay_rate)
                for field_name, amount in increments.get(name, {}).items():
                    node.perturb(field_name, amount)

    def sample(self, subsystem: str) -> Dict[str, float]:
        with self._lock:
            node = self.ensure_node(subsystem)
            return {k: float(v) for k, v in node.values.items()}

    def need(self, subsystem: str) -> float:
        s = self.sample(subsystem)
        return clamp01(
            0.35 * s.get("danger", 0.0)
            + 0.25 * s.get("damage", 0.0)
            + 0.20 * s.get("task_pressure", 0.0)
            + 0.10 * s.get("growth", 0.0)
            + 0.10 * s.get("repair", 0.0)
        )

    def pressure(self, subsystem: str) -> float:
        s = self.sample(subsystem)
        return clamp01(
            0.35 * s.get("resource_pressure", 0.0)
            + 0.25 * s.get("inhibition", 0.0)
            + 0.20 * s.get("danger", 0.0)
            + 0.20 * s.get("memory_pressure", 0.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "diffusion": self.diffusion,
                "decay": self.decay_rate,
                "nodes": {name: node.to_dict() for name, node in self._nodes.items()},
                "edges": {a: dict(b) for a, b in self._edges.items()},
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MorphogenField":
        field = cls(
            diffusion=float(data.get("diffusion", 0.18)),
            decay=float(data.get("decay", 0.08)),
        )
        for name, node_data in dict(data.get("nodes", {})).items():
            node = TissueNode(subsystem=str(name))
            node.values.update({str(k): clamp01(float(v)) for k, v in dict(node_data.get("values", {})).items()})
            node.last_updated = float(node_data.get("last_updated", time.time()))
            field._nodes[str(name)] = node
        for a, neighbours in dict(data.get("edges", {})).items():
            for b, w in dict(neighbours).items():
                field.register_edge(a, b, float(w))
        return field
