"""Trainable predictive memory graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PredictiveMemoryGraph:
    dims: int = 16
    lr: float = 0.05
    node_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    relation_vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def add_node(self, node_id: str, embedding: np.ndarray | None = None) -> None:
        if embedding is None:
            rng = np.random.default_rng(abs(hash(node_id)) & 0xffffffff)
            embedding = rng.normal(0, 0.1, size=self.dims).astype(np.float32)
        self.node_vectors[node_id] = np.asarray(embedding, dtype=np.float32)[: self.dims]

    def add_relation(self, relation: str) -> None:
        if relation not in self.relation_vectors:
            rng = np.random.default_rng(abs(hash(relation)) & 0xffffffff)
            self.relation_vectors[relation] = rng.normal(0, 0.1, size=self.dims).astype(np.float32)

    def train_edge(self, src: str, relation: str, dst: str, *, positive: bool = True) -> float:
        self.add_node(src)
        self.add_node(dst)
        self.add_relation(relation)
        s = self.node_vectors[src]
        r = self.relation_vectors[relation]
        d = self.node_vectors[dst]
        score = self.score(src, relation, dst)
        target = 1.0 if positive else 0.0
        error = target - score
        grad = self.lr * error
        self.node_vectors[src] = s + grad * (d - r)
        self.relation_vectors[relation] = r + grad * (d - s)
        self.node_vectors[dst] = d + grad * (s + r)
        return float(abs(error))

    def score(self, src: str, relation: str, dst: str) -> float:
        self.add_node(src)
        self.add_node(dst)
        self.add_relation(relation)
        value = float(np.dot(self.node_vectors[src] + self.relation_vectors[relation], self.node_vectors[dst]))
        return 1.0 / (1.0 + float(np.exp(-value)))

    def predict_links(self, src: str, relation: str, candidates: list[str], *, top_k: int = 5) -> list[tuple[str, float]]:
        scored = [(candidate, self.score(src, relation, candidate)) for candidate in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dims": self.dims,
            "nodes": len(self.node_vectors),
            "relations": len(self.relation_vectors),
        }


__all__ = ["PredictiveMemoryGraph"]
