"""
Dead Letter Queue for failed skill executions and autonomous tasks.
A persistent (SQLite-backed) queue of tasks that failed and weren't recovered.
"""
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Aura.DLQ")

_DB_PATH = config.paths.data_dir / "enterprise.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dlq (
    id           TEXT PRIMARY KEY,
    skill_name   TEXT NOT NULL,
    params       TEXT NOT NULL,
    error        TEXT NOT NULL,
    created_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dlq_skill ON dlq(skill_name);
CREATE INDEX IF NOT EXISTS idx_dlq_created ON dlq(created_at DESC);
"""


class DeadLetterQueue:
    def __init__(self, db_path: Optional[str] = None):
        self._db = str(db_path or _DB_PATH)
        Path(self._db).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init(self):
        with self._connect() as con:
            con.executescript(_SCHEMA)
            con.commit()

    def push(self, skill_name: str, params: Dict[str, Any], error: str) -> str:
        """Record a failed task."""
        entry_id = str(uuid.uuid4())[:8]
        with self._connect() as con:
            con.execute(
                "INSERT INTO dlq VALUES (?,?,?,?,?)",
                (entry_id, skill_name, json.dumps(params, default=str), error, time.time()),
            )
            con.commit()
        logger.warning(
            "💀 DLQ: '%s' failed and was quarantined. Error: %s", skill_name, error[:100]
        )
        return entry_id

    def get_failed(self, limit: int = 50) -> List[Dict]:
        """List failed tasks."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM dlq ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self, *, recent_window_s: float = 900.0, limit: int = 10) -> Dict[str, Any]:
        """Return queue pressure without requiring callers to inspect SQLite."""
        cutoff = time.time() - float(recent_window_s)
        with self._connect() as con:
            total = int(con.execute("SELECT COUNT(*) FROM dlq").fetchone()[0])
            recent_count = int(
                con.execute("SELECT COUNT(*) FROM dlq WHERE created_at >= ?", (cutoff,)).fetchone()[0]
            )
            rows = con.execute(
                "SELECT * FROM dlq ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return {
            "total": total,
            "recent_count": recent_count,
            "recent_window_s": float(recent_window_s),
            "recent": [dict(row) for row in rows],
        }

    def resolve(self, entry_id: str):
        """Remove a task from the DLQ (e.g. after manual retry)."""
        with self._connect() as con:
            con.execute("DELETE FROM dlq WHERE id = ?", (entry_id,))
            con.commit()
        logger.info("DLQ: Entry %s marked resolved/removed.", entry_id)


_dlq: Optional[DeadLetterQueue] = None


def get_dlq() -> DeadLetterQueue:
    global _dlq
    if _dlq is None:
        _dlq = DeadLetterQueue()
    return _dlq
