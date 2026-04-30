"""Logarithmic temporal navigation for long-form media.

This is a clean-room, GPL-safe implementation of the useful idea behind
hierarchical video exploration: represent a long video as recursively
expandable temporal grids, keep positive evidence in a scratchpad, and mark
negative intervals as dead zones so follow-up perception does not waste budget.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TemporalCell:
    node_id: str
    cell_id: int
    start_s: float
    end_s: float
    depth: int
    dead: bool = False
    promising: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class EvidenceItem:
    start_s: float
    end_s: float
    description: str
    confidence: float = 1.0
    source: str = "aura"
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class TemporalNode:
    node_id: str
    start_s: float
    end_s: float
    depth: int
    parent_id: str | None
    cells: tuple[TemporalCell, ...]


class TemporalAtlas:
    """Recursive KxK temporal grid for bounded long-video exploration."""

    def __init__(
        self,
        duration_s: float,
        *,
        grid_size: int = 8,
        max_depth: int = 4,
        frame_sampler: Callable[[float, float], Any] | None = None,
    ) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self.duration_s = float(duration_s)
        self.grid_size = int(grid_size)
        self.cells_per_node = self.grid_size * self.grid_size
        self.max_depth = int(max_depth)
        self.frame_sampler = frame_sampler
        self._nodes: dict[str, TemporalNode] = {}
        self._dead_zones: list[tuple[float, float]] = []
        self._promising: set[tuple[str, int]] = set()
        self._evidence: list[EvidenceItem] = []
        self._nodes["root"] = self._make_node("root", 0.0, self.duration_s, 0, None)

    @property
    def root(self) -> TemporalNode:
        return self._nodes["root"]

    def node(self, node_id: str) -> TemporalNode:
        return self._nodes[node_id]

    def expand(self, node_id: str, cell_id: int) -> TemporalNode:
        parent = self.node(node_id)
        if parent.depth >= self.max_depth:
            raise ValueError("maximum atlas depth reached")
        cell = self._cell(parent, cell_id)
        child_id = f"{node_id}.{cell_id}"
        if child_id not in self._nodes:
            self._nodes[child_id] = self._make_node(
                child_id,
                cell.start_s,
                cell.end_s,
                parent.depth + 1,
                node_id,
            )
        return self._nodes[child_id]

    def zoom(self, node_id: str, cell_id: int) -> dict[str, Any]:
        cell = self._cell(self.node(node_id), cell_id)
        sample = self.frame_sampler(cell.start_s, cell.end_s) if self.frame_sampler else None
        return {"cell": asdict(cell), "sample": sample}

    def mark_dead(self, start_s: float, end_s: float) -> None:
        start, end = self._normalize_interval(start_s, end_s)
        self._dead_zones.append((start, end))
        self._refresh_nodes()

    def mark_promising(self, node_id: str, cell_ids: Iterable[int]) -> None:
        node = self.node(node_id)
        for cell_id in cell_ids:
            self._cell(node, int(cell_id))
            self._promising.add((node_id, int(cell_id)))
        self._refresh_nodes()

    def add_evidence(
        self,
        start_s: float,
        end_s: float,
        description: str,
        *,
        confidence: float = 1.0,
        source: str = "aura",
        payload: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        start, end = self._normalize_interval(start_s, end_s)
        item = EvidenceItem(
            start_s=start,
            end_s=end,
            description=description,
            confidence=max(0.0, min(1.0, float(confidence))),
            source=source,
            payload=dict(payload or {}),
        )
        self._evidence.append(item)
        return item

    def scratchpad(self) -> list[EvidenceItem]:
        return list(self._evidence)

    def dead_zones(self) -> list[tuple[float, float]]:
        return list(self._dead_zones)

    def coverage(self) -> dict[str, float]:
        dead = self._union_duration(self._dead_zones)
        evidence = self._union_duration((item.start_s, item.end_s) for item in self._evidence)
        return {
            "dead_zone_ratio": min(1.0, dead / self.duration_s),
            "evidence_ratio": min(1.0, evidence / self.duration_s),
            "unexplored_ratio": max(0.0, 1.0 - min(1.0, dead / self.duration_s)),
        }

    def next_frontier(self, *, limit: int = 8) -> list[TemporalCell]:
        cells: list[TemporalCell] = []
        for node in self._nodes.values():
            for cell in node.cells:
                if not cell.dead:
                    cells.append(cell)
        cells.sort(key=lambda c: (not c.promising, c.depth, c.start_s))
        return cells[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "grid_size": self.grid_size,
            "max_depth": self.max_depth,
            "nodes": {node_id: asdict(node) for node_id, node in sorted(self._nodes.items())},
            "dead_zones": self.dead_zones(),
            "evidence": [asdict(item) for item in self._evidence],
            "coverage": self.coverage(),
        }

    def _make_node(self, node_id: str, start_s: float, end_s: float, depth: int, parent_id: str | None) -> TemporalNode:
        width = (end_s - start_s) / self.cells_per_node
        cells = []
        for cell_id in range(self.cells_per_node):
            cell_start = start_s + width * cell_id
            cell_end = end_s if cell_id == self.cells_per_node - 1 else cell_start + width
            cells.append(
                TemporalCell(
                    node_id=node_id,
                    cell_id=cell_id,
                    start_s=cell_start,
                    end_s=cell_end,
                    depth=depth,
                    dead=self._is_dead(cell_start, cell_end),
                    promising=(node_id, cell_id) in self._promising,
                )
            )
        return TemporalNode(
            node_id=node_id,
            start_s=start_s,
            end_s=end_s,
            depth=depth,
            parent_id=parent_id,
            cells=tuple(cells),
        )

    def _refresh_nodes(self) -> None:
        existing = list(self._nodes.values())
        self._nodes = {
            node.node_id: self._make_node(
                node.node_id,
                node.start_s,
                node.end_s,
                node.depth,
                node.parent_id,
            )
            for node in existing
        }

    def _cell(self, node: TemporalNode, cell_id: int) -> TemporalCell:
        if cell_id < 0 or cell_id >= self.cells_per_node:
            raise IndexError(f"cell_id must be between 0 and {self.cells_per_node - 1}")
        return node.cells[cell_id]

    def _is_dead(self, start_s: float, end_s: float) -> bool:
        return any(start_s >= dead_start and end_s <= dead_end for dead_start, dead_end in self._dead_zones)

    def _normalize_interval(self, start_s: float, end_s: float) -> tuple[float, float]:
        start = max(0.0, min(self.duration_s, float(start_s)))
        end = max(0.0, min(self.duration_s, float(end_s)))
        if end <= start:
            raise ValueError("interval end must be greater than start")
        return start, end

    @staticmethod
    def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
        ordered = sorted((float(s), float(e)) for s, e in intervals if e > s)
        if not ordered:
            return 0.0
        total = 0.0
        cur_start, cur_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                total += cur_end - cur_start
                cur_start, cur_end = start, end
        total += cur_end - cur_start
        return total


__all__ = ["EvidenceItem", "TemporalAtlas", "TemporalCell", "TemporalNode"]
