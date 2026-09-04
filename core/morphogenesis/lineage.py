"""core/morphogenesis/lineage.py — where a cell came from.

The old cell state carried a ``lineage_id`` that was a digest of the cell's own
name and a ``generation`` that was always zero, because nothing ever spawned
anything. With SPAWN real, lineage becomes the record of development: which
cell produced which, under what pressure, and whether the child outlived the
condition that justified it.

The one hard rule is acyclicity. A cell cannot be its own ancestor, and a
lineage that admits a cycle would let a motif claim credit for producing
itself.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .types import json_safe


class LineageCycleError(RuntimeError):
    """A parent link would have made a cell its own ancestor."""


@dataclass
class LineageRecord:
    """One cell's origin."""

    cell_id: str
    parent_id: str = ""
    generation: int = 0
    born_at: float = field(default_factory=time.time)
    born_at_version: int = 0
    cause: str = ""
    motif_id: str = ""
    retired_at: float = 0.0
    retire_cause: str = ""

    @property
    def alive(self) -> bool:
        return self.retired_at <= 0.0

    @property
    def lifetime_s(self) -> float:
        end = self.retired_at if self.retired_at > 0 else time.time()
        return max(0.0, end - self.born_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "born_at": self.born_at,
            "born_at_version": self.born_at_version,
            "cause": self.cause,
            "motif_id": self.motif_id,
            "retired_at": self.retired_at,
            "retire_cause": self.retire_cause,
            "alive": self.alive,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LineageRecord:
        payload = dict(data or {})
        return cls(
            cell_id=str(payload.get("cell_id", "")),
            parent_id=str(payload.get("parent_id", "")),
            generation=int(payload.get("generation", 0)),
            born_at=float(payload.get("born_at", time.time())),
            born_at_version=int(payload.get("born_at_version", 0)),
            cause=str(payload.get("cause", "")),
            motif_id=str(payload.get("motif_id", "")),
            retired_at=float(payload.get("retired_at", 0.0)),
            retire_cause=str(payload.get("retire_cause", "")),
        )


class Lineage:
    """The developmental record for a population."""

    def __init__(self, *, max_generation: int = 6):
        self.max_generation = int(max_generation)
        self._records: dict[str, LineageRecord] = {}

    def seed(self, cell_id: str, *, cause: str = "seed") -> LineageRecord:
        """Register a founder — a cell nothing spawned."""
        record = self._records.get(cell_id)
        if record is None:
            record = LineageRecord(cell_id=cell_id, generation=0, cause=cause)
            self._records[cell_id] = record
        return record

    def record_birth(
        self,
        cell_id: str,
        *,
        parent_id: str,
        version: int = 0,
        cause: str = "",
        motif_id: str = "",
    ) -> LineageRecord:
        if cell_id == parent_id:
            raise LineageCycleError(f"{cell_id} cannot be its own parent")
        if parent_id and self.is_descendant(parent_id, cell_id):
            raise LineageCycleError(
                f"{parent_id} already descends from {cell_id}; this link would close a cycle"
            )
        parent = self._records.get(parent_id)
        generation = (parent.generation + 1) if parent is not None else 0
        record = LineageRecord(
            cell_id=cell_id,
            parent_id=parent_id,
            generation=generation,
            born_at_version=version,
            cause=cause,
            motif_id=motif_id,
        )
        self._records[cell_id] = record
        return record

    def record_retirement(self, cell_id: str, *, cause: str = "") -> None:
        record = self._records.get(cell_id)
        if record is not None and record.alive:
            record.retired_at = time.time()
            record.retire_cause = cause

    def get(self, cell_id: str) -> LineageRecord | None:
        return self._records.get(cell_id)

    def generation_of(self, cell_id: str) -> int:
        record = self._records.get(cell_id)
        return record.generation if record is not None else 0

    def would_exceed_depth(self, parent_id: str) -> bool:
        """Whether spawning from this parent would go past the depth bound.

        Spawn depth is one of the bounds that stops replication running away:
        a lineage that can always go one deeper is a population that can always
        grow.
        """
        return self.generation_of(parent_id) + 1 > self.max_generation

    def ancestors(self, cell_id: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = {cell_id}
        current = self._records.get(cell_id)
        while current is not None and current.parent_id:
            if current.parent_id in seen:
                break
            out.append(current.parent_id)
            seen.add(current.parent_id)
            current = self._records.get(current.parent_id)
        return out

    def is_descendant(self, candidate: str, ancestor: str) -> bool:
        return ancestor in self.ancestors(candidate)

    def children(self, cell_id: str) -> list[str]:
        return sorted(r.cell_id for r in self._records.values() if r.parent_id == cell_id)

    def living(self) -> list[str]:
        return sorted(r.cell_id for r in self._records.values() if r.alive)

    def acyclic(self) -> bool:
        """True when no cell reaches itself through parent links."""
        for cell_id in self._records:
            seen: set[str] = set()
            current = cell_id
            while True:
                record = self._records.get(current)
                if record is None or not record.parent_id:
                    break
                if record.parent_id in seen or record.parent_id == cell_id:
                    return False
                seen.add(record.parent_id)
                current = record.parent_id
        return True

    def status(self) -> dict[str, Any]:
        alive = [r for r in self._records.values() if r.alive]
        retired = [r for r in self._records.values() if not r.alive]
        by_generation: dict[int, int] = {}
        for record in alive:
            by_generation[record.generation] = by_generation.get(record.generation, 0) + 1
        return {
            "tracked": len(self._records),
            "alive": len(alive),
            "retired": len(retired),
            "max_generation": max((r.generation for r in self._records.values()), default=0),
            "generation_cap": self.max_generation,
            "by_generation": dict(sorted(by_generation.items())),
            "acyclic": self.acyclic(),
            "mean_retired_lifetime_s": (
                round(sum(r.lifetime_s for r in retired) / len(retired), 3) if retired else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_generation": self.max_generation,
            "records": {cid: r.to_dict() for cid, r in sorted(self._records.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lineage:
        payload = dict(data or {})
        lineage = cls(max_generation=int(payload.get("max_generation", 6)))
        for cell_id, raw in dict(payload.get("records", {})).items():
            lineage._records[str(cell_id)] = LineageRecord.from_dict(raw)
        return lineage

    def report(self) -> dict[str, Any]:
        return json_safe(self.status())


__all__ = ["Lineage", "LineageCycleError", "LineageRecord"]
