"""Lineage and heritable variation.

Addresses the strict artificial-life critique: a digital organism, in the ALife
sense, has self-replication with heritable variation under selection. Aura
cannot be that in the full Avida sense (her body is a large Python process),
but she can have a concrete lineage mechanism: forkable agent snapshots that
inherit configuration and state from a parent, introduce bounded mutation, and
track which offspring variants survive based on a selection signal.

This does not claim to be evolution. It is the minimum viable version of
"heritable variation + selection" — enough that the gap stops being
"completely absent" and becomes "bounded but auditable".
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class LineageSnapshot:
    snapshot_id: str
    parent_id: Optional[str]
    generation: int
    config: Dict[str, Any]
    trait_signature: str
    created_at: float
    selection_score: float = 0.0
    survived: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "config": dict(self.config),
            "trait_signature": self.trait_signature,
            "created_at": self.created_at,
            "selection_score": round(float(self.selection_score), 4),
            "survived": self.survived,
        }


class LineageManager:
    """Fork snapshots with bounded variation and track selection."""

    MUTATION_MAGNITUDE = 0.08
    MIN_SURVIVAL_SCORE = 0.25

    def __init__(self, db_path: Optional[str | Path] = None, *, seed: Optional[int] = None) -> None:
        self._lock = threading.RLock()
        self._rng = random.Random(seed)
        if db_path is None:
            try:
                from core.config import config
                db_path = Path(config.paths.data_dir) / "lineage.sqlite3"
            except Exception:
                db_path = Path.home() / ".aura" / "lineage.sqlite3"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lineage_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    generation INTEGER NOT NULL,
                    config TEXT NOT NULL,
                    trait_signature TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    selection_score REAL NOT NULL DEFAULT 0,
                    survived INTEGER NOT NULL DEFAULT 1
                )
                """
            )

    # ------------------------------------------------------------------
    # Fork + variation
    # ------------------------------------------------------------------
    def genesis(self, config: Mapping[str, Any]) -> LineageSnapshot:
        snap = self._make_snapshot(config=dict(config), parent_id=None, generation=0)
        self._persist(snap)
        return snap

    def fork(self, parent_id: str, *, mutation_mask: Mapping[str, float] | None = None) -> LineageSnapshot:
        parent = self.get(parent_id)
        if parent is None:
            raise KeyError(f"parent snapshot {parent_id} not found")
        child_config = self._mutate(parent.config, mutation_mask or {})
        child = self._make_snapshot(
            config=child_config,
            parent_id=parent_id,
            generation=parent.generation + 1,
        )
        self._persist(child)
        return child

    def _mutate(self, config: Mapping[str, Any], mask: Mapping[str, float]) -> Dict[str, Any]:
        mutated: Dict[str, Any] = {}
        for key, value in config.items():
            rate = float(mask.get(key, self.MUTATION_MAGNITUDE))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                drift = self._rng.gauss(0.0, rate)
                mutated[key] = type(value)(float(value) + drift * max(1.0, abs(float(value))))
            elif isinstance(value, list):
                mutated[key] = list(value)
                if value and rate > 0 and self._rng.random() < rate:
                    idx = self._rng.randrange(len(value))
                    mutated[key][idx] = value[idx]
            else:
                mutated[key] = value
        return mutated

    def _make_snapshot(self, *, config: Dict[str, Any], parent_id: Optional[str], generation: int) -> LineageSnapshot:
        created = time.time()
        payload = json.dumps(config, sort_keys=True, default=str)
        snapshot_id = hashlib.sha256(f"{parent_id}|{generation}|{created}|{payload}".encode("utf-8")).hexdigest()[:16]
        trait_signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return LineageSnapshot(
            snapshot_id=snapshot_id,
            parent_id=parent_id,
            generation=generation,
            config=dict(config),
            trait_signature=trait_signature,
            created_at=created,
        )

    def _persist(self, snap: LineageSnapshot) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lineage_snapshots
                    (snapshot_id, parent_id, generation, config, trait_signature, created_at, selection_score, survived)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap.snapshot_id,
                    snap.parent_id,
                    int(snap.generation),
                    json.dumps(snap.config, sort_keys=True, default=str),
                    snap.trait_signature,
                    float(snap.created_at),
                    float(snap.selection_score),
                    int(1 if snap.survived else 0),
                ),
            )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def record_score(self, snapshot_id: str, score: float) -> LineageSnapshot:
        snap = self.get(snapshot_id)
        if snap is None:
            raise KeyError(f"snapshot {snapshot_id} not found")
        survived = score >= self.MIN_SURVIVAL_SCORE
        updated = LineageSnapshot(
            snapshot_id=snap.snapshot_id,
            parent_id=snap.parent_id,
            generation=snap.generation,
            config=snap.config,
            trait_signature=snap.trait_signature,
            created_at=snap.created_at,
            selection_score=float(score),
            survived=survived,
        )
        self._persist(updated)
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, snapshot_id: str) -> Optional[LineageSnapshot]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM lineage_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if row is None:
            return None
        return LineageSnapshot(
            snapshot_id=row["snapshot_id"],
            parent_id=row["parent_id"],
            generation=int(row["generation"]),
            config=json.loads(row["config"]),
            trait_signature=row["trait_signature"],
            created_at=float(row["created_at"]),
            selection_score=float(row["selection_score"]),
            survived=bool(row["survived"]),
        )

    def descendants(self, snapshot_id: str) -> List[LineageSnapshot]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM lineage_snapshots WHERE parent_id = ? ORDER BY created_at ASC",
                (snapshot_id,),
            ).fetchall()
        return [
            LineageSnapshot(
                snapshot_id=r["snapshot_id"],
                parent_id=r["parent_id"],
                generation=int(r["generation"]),
                config=json.loads(r["config"]),
                trait_signature=r["trait_signature"],
                created_at=float(r["created_at"]),
                selection_score=float(r["selection_score"]),
                survived=bool(r["survived"]),
            )
            for r in rows
        ]

    def survivors(self) -> List[LineageSnapshot]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM lineage_snapshots WHERE survived = 1 ORDER BY selection_score DESC"
            ).fetchall()
        return [
            LineageSnapshot(
                snapshot_id=r["snapshot_id"],
                parent_id=r["parent_id"],
                generation=int(r["generation"]),
                config=json.loads(r["config"]),
                trait_signature=r["trait_signature"],
                created_at=float(r["created_at"]),
                selection_score=float(r["selection_score"]),
                survived=bool(r["survived"]),
            )
            for r in rows
        ]


_singleton: Optional[LineageManager] = None
_lock = threading.Lock()


def get_lineage_manager() -> LineageManager:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = LineageManager()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
