"""Causal genealogy graph for self-repair patches.

Every repair attempt is stored as a patch node with its trigger, risk tier,
pre/post metrics, validation evidence, and eventual outcome.  The graph makes
post-hoc questions answerable: which patch caused a regression, which fixes
improved latency, and which bug classes Aura should stop trying to auto-apply.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class PatchNode:
    patch_id: str
    trigger_id: str
    target_files: tuple[str, ...]
    bug_fingerprint: str
    risk_tier: str
    status: str
    pre_metrics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    post_metrics: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_files"] = list(self.target_files)
        return data


class PatchGenealogyGraph:
    """SQLite-backed graph for repair lineage and causal queries."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".aura" / "data" / "selfmod" / "patch_genealogy.sqlite3"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patch_nodes (
                    patch_id TEXT PRIMARY KEY,
                    trigger_id TEXT NOT NULL,
                    target_files TEXT NOT NULL,
                    bug_fingerprint TEXT NOT NULL,
                    risk_tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pre_metrics TEXT NOT NULL,
                    validation TEXT NOT NULL,
                    post_metrics TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patch_edges (
                    parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(parent_id, child_id, relation)
                )
                """
            )

    def make_patch_id(self, *, trigger_id: str, target_files: Iterable[str], diff: str = "") -> str:
        payload = json.dumps(
            {
                "trigger_id": trigger_id,
                "target_files": sorted(str(p) for p in target_files),
                "diff_hash": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
                "when": time.time(),
            },
            sort_keys=True,
        )
        return "patch_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def add_node(self, node: PatchNode) -> PatchNode:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO patch_nodes
                    (patch_id, trigger_id, target_files, bug_fingerprint, risk_tier, status,
                     pre_metrics, validation, post_metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.patch_id,
                    node.trigger_id,
                    json.dumps(list(node.target_files), sort_keys=True),
                    node.bug_fingerprint,
                    node.risk_tier,
                    node.status,
                    json.dumps(node.pre_metrics, sort_keys=True, default=str),
                    json.dumps(node.validation, sort_keys=True, default=str),
                    json.dumps(node.post_metrics, sort_keys=True, default=str),
                    float(node.created_at),
                ),
            )
        return node

    def add_edge(self, parent_id: str, child_id: str, relation: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO patch_edges(parent_id, child_id, relation, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (parent_id, child_id, relation, time.time()),
            )

    def update_status(
        self,
        patch_id: str,
        status: str,
        *,
        validation: Optional[dict[str, Any]] = None,
        post_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        assignments = ["status = ?"]
        values: list[Any] = [status]
        if validation is not None:
            assignments.append("validation = ?")
            values.append(json.dumps(validation, sort_keys=True, default=str))
        if post_metrics is not None:
            assignments.append("post_metrics = ?")
            values.append(json.dumps(post_metrics, sort_keys=True, default=str))
        values.append(patch_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE patch_nodes SET {', '.join(assignments)} WHERE patch_id = ?", values)

    def get_node(self, patch_id: str) -> PatchNode | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM patch_nodes WHERE patch_id = ?", (patch_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def query(
        self,
        *,
        status: str | None = None,
        target_contains: str | None = None,
        limit: int = 50,
    ) -> list[PatchNode]:
        sql = "SELECT * FROM patch_nodes"
        clauses: list[str] = []
        values: list[Any] = []
        if status:
            clauses.append("status = ?")
            values.append(status)
        if target_contains:
            clauses.append("target_files LIKE ?")
            values.append(f"%{target_contains}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        values.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, values).fetchall()
        return [self._row_to_node(row) for row in rows]

    def causal_chain(self, patch_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            edges = conn.execute(
                """
                WITH RECURSIVE chain(parent_id, child_id, relation, depth) AS (
                  SELECT parent_id, child_id, relation, 0 FROM patch_edges WHERE child_id = ?
                  UNION ALL
                  SELECT e.parent_id, e.child_id, e.relation, chain.depth + 1
                  FROM patch_edges e JOIN chain ON e.child_id = chain.parent_id
                )
                SELECT * FROM chain ORDER BY depth ASC
                """,
                (patch_id,),
            ).fetchall()
        return [dict(row) for row in edges]

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> PatchNode:
        return PatchNode(
            patch_id=row["patch_id"],
            trigger_id=row["trigger_id"],
            target_files=tuple(json.loads(row["target_files"])),
            bug_fingerprint=row["bug_fingerprint"],
            risk_tier=row["risk_tier"],
            status=row["status"],
            pre_metrics=json.loads(row["pre_metrics"]),
            validation=json.loads(row["validation"]),
            post_metrics=json.loads(row["post_metrics"]),
            created_at=float(row["created_at"]),
        )


_instance: PatchGenealogyGraph | None = None


def get_patch_genealogy() -> PatchGenealogyGraph:
    global _instance
    if _instance is None:
        _instance = PatchGenealogyGraph()
    return _instance


__all__ = ["PatchNode", "PatchGenealogyGraph", "get_patch_genealogy"]
