from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import config


class SQLitePersistentState:
    """Stdlib durable skill-execution audit log used when SQLAlchemy is absent."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path is not None else config.paths.data_dir / "zenith_state.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._lock = threading.RLock()
        self._init_schema()

    @contextlib.contextmanager
    def _connect(self):
        """Open a connection, commit or roll back, and CLOSE it.

        ``with sqlite3.connect(...) as conn`` does not close the connection —
        it only wraps a transaction — so every call here leaked a handle, and
        with journal_mode=WAL it leaked the -wal and -shm files with it. The
        symptom was a hermetic-leak failure reported against whichever test ran
        next, which is why it looked like a test-ordering problem rather than a
        connection this class never closed.
        """

        conn = sqlite3.connect(self.db_path, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=15000")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    params TEXT,
                    status TEXT,
                    duration_ms REAL,
                    result TEXT,
                    error TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_execution_logs_skill_name "
                "ON skill_execution_logs(skill_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_execution_logs_timestamp "
                "ON skill_execution_logs(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_skill_execution_logs_status "
                "ON skill_execution_logs(status)"
            )

    @staticmethod
    def _json_or_none(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def log_execution(
        self,
        skill_name: str,
        params: dict,
        status: str,
        duration_ms: float,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_execution_logs
                    (skill_name, timestamp, params, status, duration_ms, result, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(skill_name),
                    datetime.now(UTC).isoformat(),
                    self._json_or_none(params),
                    str(status),
                    float(duration_ms or 0.0),
                    self._json_or_none(result),
                    str(error) if error is not None else None,
                ),
            )
