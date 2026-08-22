"""Bounded durable cold tier for memories that outlive hot indexes.

The memory facade has advertised a ``cold_store`` since its inventory became
measurable, but no service implemented that contract. This store is deliberately
small: SQLite durability, bounded lexical retrieval, and no model dependency.
It is the final archival fallback, not a second semantic authority.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

_MAX_CONTENT_BYTES = 2 * 1024 * 1024
_MAX_METADATA_BYTES = 256 * 1024
_MAX_QUERY_TERMS = 8
_WORD_RE = re.compile(r"[A-Za-z0-9_'-]{2,}")


class ColdMemoryStore:
    """Append-only archival memory with cheap bounded lexical recall."""

    backend_name = "sqlite-cold-memory"

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self._lock = checked_lock("core.memory.cold_store", reentrant=True)
        self._ready = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "cold_store.initialize",
            domain="file_write",
            constraints={"artifact": "cold_memory_database"},
        ):
            get_file_write_gateway().ensure_directory(
                self.db_path.parent,
                source="cold_store.initialize",
            )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cold_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_cold_memories_created "
                "ON cold_memories(created_at DESC)"
            )
            connection.commit()
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready and self.db_path.exists()

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM cold_memories"
            ).fetchone()
        return int(row["total"] if row is not None else 0)

    def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        text = str(content or "").strip()
        if not text:
            return False
        if len(text.encode("utf-8")) > _MAX_CONTENT_BYTES:
            raise ValueError("cold memory content exceeds bounded record size")
        encoded_metadata = json.dumps(
            dict(metadata or {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if len(encoded_metadata.encode("utf-8")) > _MAX_METADATA_BYTES:
            raise ValueError("cold memory metadata exceeds bounded record size")
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO cold_memories(id, content, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, text, encoded_metadata, time.time()),
            )
            connection.commit()
        return True

    remember = add_memory
    store = add_memory
    write = add_memory

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit or 5), 50))
        terms = tuple(dict.fromkeys(_WORD_RE.findall(str(query or "").lower())))
        terms = terms[:_MAX_QUERY_TERMS]
        with self._lock, self._connect() as connection:
            if terms:
                predicates = " OR ".join("lower(content) LIKE ?" for _ in terms)
                rows = connection.execute(
                    f"SELECT id, content, metadata_json, created_at "
                    f"FROM cold_memories WHERE {predicates} "
                    "ORDER BY created_at DESC LIMIT ?",
                    (*[f"%{term}%" for term in terms], bounded_limit * 8),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, content, metadata_json, created_at "
                    "FROM cold_memories ORDER BY created_at DESC LIMIT ?",
                    (bounded_limit,),
                ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            content = str(row["content"])
            lowered = content.lower()
            matched = sum(1 for term in terms if term in lowered)
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                metadata = {}
            results.append(
                {
                    "id": str(row["id"]),
                    "content": content,
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "created_at": float(row["created_at"]),
                    "score": (matched / len(terms)) if terms else 0.0,
                    "source": "cold_store",
                }
            )
        results.sort(
            key=lambda item: (float(item["score"]), float(item["created_at"])),
            reverse=True,
        )
        return results[:bounded_limit]

    retrieve = search
    get = search


__all__ = ["ColdMemoryStore"]
