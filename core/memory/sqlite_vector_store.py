"""Local SQLite vector store with binary embeddings.

Aura's long-term memory should not keep high-dimensional vectors as JSON
float arrays. This module stores vectors as contiguous ``float32`` BLOBs and
streams rows during search, so legacy JSON is only read during explicit
migration.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Memory.SQLiteVectorStore")


@dataclass(frozen=True)
class VectorRecord:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float
    distance: float


class SQLiteVectorStore:
    """Embedded local vector store backed by SQLite BLOB columns.

    This is deliberately dependency-light. If an operator installs sqlite-vec
    later, this schema can be mirrored into a virtual table, but the baseline
    runtime already avoids JSON vector bloat and cloud vector databases.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path, *, collection_name: str = "default") -> None:
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA cache_size=-16000;")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_records (
                        id TEXT PRIMARY KEY,
                        collection TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        embedding BLOB NOT NULL,
                        dim INTEGER NOT NULL,
                        dtype TEXT NOT NULL DEFAULT 'float32',
                        timestamp REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vector_records_collection "
                    "ON vector_records(collection)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vector_records_time "
                    "ON vector_records(collection, timestamp DESC)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_store_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT OR REPLACE INTO vector_store_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION)),
                )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _as_float32(vector: Iterable[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("embedding vector is empty")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _metadata_json(metadata: Optional[dict[str, Any]]) -> str:
        return json.dumps(dict(metadata or {}), sort_keys=True, default=str)

    def upsert(
        self,
        record_id: str,
        content: str,
        embedding: Iterable[float] | np.ndarray,
        *,
        metadata: Optional[dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> None:
        vector = self._as_float32(embedding)
        now = time.time()
        meta = dict(metadata or {})
        timestamp = float(meta.get("timestamp") or now)
        coll = collection or self.collection_name
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vector_records
                    (id, collection, content, metadata, embedding, dim, dtype, timestamp, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'float32', ?, ?)
                    """,
                    (
                        str(record_id),
                        coll,
                        str(content or ""),
                        self._metadata_json(meta),
                        sqlite3.Binary(vector.tobytes(order="C")),
                        int(vector.size),
                        timestamp,
                        now,
                    ),
                )
        finally:
            conn.close()

    def upsert_many(
        self,
        records: Iterable[tuple[str, str, Iterable[float] | np.ndarray, dict[str, Any]]],
        *,
        collection: Optional[str] = None,
    ) -> int:
        rows: list[tuple[Any, ...]] = []
        coll = collection or self.collection_name
        now = time.time()
        for record_id, content, embedding, metadata in records:
            vector = self._as_float32(embedding)
            meta = dict(metadata or {})
            rows.append(
                (
                    str(record_id),
                    coll,
                    str(content or ""),
                    self._metadata_json(meta),
                    sqlite3.Binary(vector.tobytes(order="C")),
                    int(vector.size),
                    float(meta.get("timestamp") or now),
                    now,
                )
            )
        if not rows:
            return 0
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO vector_records
                    (id, collection, content, metadata, embedding, dim, dtype, timestamp, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'float32', ?, ?)
                    """,
                    rows,
                )
            return len(rows)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def count(self, *, collection: Optional[str] = None) -> int:
        coll = collection or self.collection_name
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM vector_records WHERE collection = ?",
                (coll,),
            ).fetchone()
            return int(row["n"] if row else 0)
        finally:
            conn.close()

    def iter_records(self, *, collection: Optional[str] = None, batch_size: int = 512) -> Iterator[dict[str, Any]]:
        coll = collection or self.collection_name
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT id, content, metadata, embedding, dim
                FROM vector_records
                WHERE collection = ?
                ORDER BY timestamp DESC
                """,
                (coll,),
            )
            for rows in iter(lambda: cursor.fetchmany(batch_size), []):
                for row in rows:
                    yield {
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": _safe_json(row["metadata"]),
                        "embedding": row["embedding"],
                        "dim": int(row["dim"]),
                    }
        finally:
            conn.close()

    def list_records(self, *, collection: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in self.iter_records(collection=collection):
            records.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": row["metadata"],
                }
            )
            if limit is not None and len(records) >= limit:
                break
        return records

    def query(
        self,
        embedding: Iterable[float] | np.ndarray,
        *,
        limit: int = 5,
        collection: Optional[str] = None,
        min_score: float = -1.0,
    ) -> list[VectorRecord]:
        query = self._as_float32(embedding)
        q_norm = float(np.linalg.norm(query))
        if q_norm <= 1e-12:
            return []

        scored: list[VectorRecord] = []
        for row in self.iter_records(collection=collection):
            if int(row["dim"]) != int(query.size):
                continue
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            denom = q_norm * float(np.linalg.norm(vector))
            if denom <= 1e-12:
                continue
            score = float(np.dot(query, vector) / denom)
            if score < min_score:
                continue
            scored.append(
                VectorRecord(
                    id=str(row["id"]),
                    content=str(row["content"]),
                    metadata=dict(row["metadata"] or {}),
                    score=score,
                    distance=1.0 - score,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(0, int(limit))]

    def clear(self, *, collection: Optional[str] = None) -> None:
        coll = collection or self.collection_name
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM vector_records WHERE collection = ?", (coll,))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def migrate_legacy_json(
        self,
        json_path: str | Path,
        *,
        collection: Optional[str] = None,
        remove_source: bool = False,
    ) -> int:
        """Migrate legacy ``[{id,text,vector,...}]`` JSON into BLOB storage."""
        path = Path(json_path)
        if not path.exists():
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("legacy vector JSON must be a list of records")
            migrated: list[tuple[str, str, np.ndarray, dict[str, Any]]] = []
            for idx, item in enumerate(raw):
                if not isinstance(item, dict) or "vector" not in item:
                    continue
                metadata = dict(item.get("metadata") or {})
                if "timestamp" in item and "timestamp" not in metadata:
                    metadata["timestamp"] = item["timestamp"]
                metadata["legacy_json_source"] = str(path)
                migrated.append(
                    (
                        str(item.get("id") or f"legacy-{idx}"),
                        str(item.get("text") or item.get("content") or ""),
                        self._as_float32(item["vector"]),
                        metadata,
                    )
                )
            count = self.upsert_many(migrated, collection=collection)
            if remove_source and count:
                path.unlink()
            return count
        except (OSError, json.JSONDecodeError, sqlite3.Error, TypeError, ValueError) as exc:
            record_degradation("sqlite_vector_store", exc)
            logger.error("Legacy vector JSON migration failed for %s: %s", path, exc)
            raise


def _safe_json(value: str | bytes | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return {}


__all__ = ["SQLiteVectorStore", "VectorRecord"]
