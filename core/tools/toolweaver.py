"""Hierarchical tool-code learning for scalable tool selection.

Inspired by ToolWeaver-style generative tool use: each tool receives a compact
hierarchical code, similar tools share prefixes, and agents can generate or
compare codes without scanning a giant flat catalog.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    capabilities: tuple[str, ...] = ()
    risk_tier: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCode:
    name: str
    code: tuple[int, ...]
    centroid: tuple[float, ...]
    risk_tier: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "code": list(self.code), "centroid": list(self.centroid), "risk_tier": self.risk_tier}


class ToolWeaverIndex:
    """Deterministic hierarchical vector-quantized tool index."""

    def __init__(self, *, depth: int = 4, branching: int = 16, dims: int = 32) -> None:
        self.depth = depth
        self.branching = branching
        self.dims = dims
        self._codes: dict[str, ToolCode] = {}
        self._specs: dict[str, ToolSpec] = {}

    def fit(self, tools: Iterable[ToolSpec]) -> dict[str, ToolCode]:
        self._codes.clear()
        self._specs.clear()
        for tool in tools:
            vector = self._embed(" ".join([tool.name, tool.description, *tool.capabilities]))
            code = self._quantize(vector)
            self._specs[tool.name] = tool
            self._codes[tool.name] = ToolCode(tool.name, code, tuple(round(x, 6) for x in vector), tool.risk_tier)
        return dict(self._codes)

    def add(self, tool: ToolSpec) -> ToolCode:
        return self.fit([*self._specs.values(), tool])[tool.name]

    def code_for(self, tool_name: str) -> ToolCode:
        return self._codes[tool_name]

    def retrieve(self, task: str, *, top_k: int = 5) -> list[ToolCode]:
        query_vec = self._embed(task)
        query_code = self._quantize(query_vec)
        scored = []
        for code in self._codes.values():
            prefix = self._common_prefix(query_code, code.code)
            distance = self._distance(query_vec, code.centroid)
            scored.append((prefix * 10.0 - distance, code))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [code for _, code in scored[:top_k]]

    def align_sequence(self, tool_names: Sequence[str]) -> list[list[int]]:
        return [list(self._codes[name].code) for name in tool_names if name in self._codes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "branching": self.branching,
            "dims": self.dims,
            "tools": {name: code.to_dict() for name, code in sorted(self._codes.items())},
        }

    def _embed(self, text: str) -> tuple[float, ...]:
        buckets = [0.0] * self.dims
        for token in text.lower().replace("/", " ").replace("_", " ").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(0, min(len(digest), self.dims)):
                buckets[idx] += (digest[idx] - 127.5) / 127.5
        norm = math.sqrt(sum(x * x for x in buckets)) or 1.0
        return tuple(x / norm for x in buckets)

    def _quantize(self, vector: Sequence[float]) -> tuple[int, ...]:
        code: list[int] = []
        for level in range(self.depth):
            start = (level * len(vector)) // self.depth
            end = ((level + 1) * len(vector)) // self.depth
            segment = vector[start:end]
            value = sum((idx + 1) * x for idx, x in enumerate(segment))
            bucket = int(abs(value * 9973)) % self.branching
            code.append(bucket)
        return tuple(code)

    @staticmethod
    def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
        count = 0
        for a, b in zip(left, right):
            if a != b:
                break
            count += 1
        return count

    @staticmethod
    def _distance(left: Sequence[float], right: Sequence[float]) -> float:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


__all__ = ["ToolSpec", "ToolCode", "ToolWeaverIndex"]
