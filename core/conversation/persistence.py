"""
Durable conversation state.
Persists every turn to SQLite so any crash can be recovered from.
"""
from core.runtime.errors import record_degradation
import json
import logging
import sqlite3
import time
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Aura.ConvPersistence")
SECONDS_PER_DAY = 86400
DEFAULT_CONVERSATION_RETENTION_DAYS = 30
DEFAULT_CONVERSATION_PRUNE_INTERVAL_S = 86400.0

_DB_PATH = config.paths.data_dir / "conversations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    last_active REAL NOT NULL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS turns (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,      -- 'user' | 'aura' | 'tool' | 'system'
    content     TEXT NOT NULL,
    origin      TEXT,               -- 'voice' | 'text' | 'autonomous' | etc.
    created_at  REAL NOT NULL,
    cid         TEXT,               -- correlation ID
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active DESC);
"""


class ConversationPersistence:

    def __init__(self, db_path: Optional[str] = None):
        self._db = str(db_path or _DB_PATH)
        Path(self._db).parent.mkdir(parents=True, exist_ok=True)
        self._init()
        self._current_session_id: Optional[str] = None
        self._retention_keep_days = DEFAULT_CONVERSATION_RETENTION_DAYS
        self._prune_interval_s = DEFAULT_CONVERSATION_PRUNE_INTERVAL_S
        self._maintenance_registered = False
        self._last_prune_at: float = 0.0

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self):
        with self._connect() as con:
            con.executescript(_SCHEMA)
            con.commit()

    def start_session(self, metadata: Optional[Dict] = None) -> str:
        session_id = str(uuid.uuid4())[:16]
        now = time.time()
        with self._connect() as con:
            con.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (session_id, now, now, json.dumps(metadata or {})),
            )
            con.commit()
        self._current_session_id = session_id
        logger.debug("Conversation session started: %s", session_id)
        return session_id

    def record_turn(
        self,
        role: str,
        content: str,
        origin: str = "",
        cid: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        sid = session_id or self._current_session_id
        if not sid:
            sid = self.start_session()

        turn_id = str(uuid.uuid4())[:12]
        now = time.time()
        with self._connect() as con:
            con.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?,?)",
                (turn_id, sid, role, content, origin, now, cid),
            )
            con.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?", (now, sid)
            )
            con.commit()
        
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish("turn_recorded", {"role": role, "content": content})
        except Exception:
            logger.debug("Turn recorded but event bus failed to publish.")
        return turn_id

    def get_session_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        sid = session_id or self._current_session_id
        if not sid:
            return []
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
                (sid, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT s.*, COUNT(t.id) as turn_count "
                "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id "
                "GROUP BY s.id ORDER BY s.last_active DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recover_last_session(self) -> Optional[str]:
        """Return the most recent session ID for crash recovery."""
        with self._connect() as con:
            row = con.execute(
                "SELECT id FROM sessions ORDER BY last_active DESC LIMIT 1"
            ).fetchone()
        if row:
            self._current_session_id = row["id"]
            logger.info("Recovered session: %s", row["id"])
            return row["id"]
        return None

    def prune_old_sessions(self, keep_days: int = DEFAULT_CONVERSATION_RETENTION_DAYS):
        """Remove sessions and their turns older than `keep_days` days."""
        cutoff = time.time() - (keep_days * SECONDS_PER_DAY)
        with self._connect() as con:
            # Manually cascade to be absolutely sure (Audit-33 fix)
            con.execute(
                "DELETE FROM turns WHERE session_id IN (SELECT id FROM sessions WHERE last_active < ?)", 
                (cutoff,)
            )
            deleted = con.execute(
                "DELETE FROM sessions WHERE last_active < ?", (cutoff,)
            ).rowcount
            con.commit()
        self._last_prune_at = time.time()
        if deleted:
            logger.info("Pruned %d old conversation sessions.", deleted)
        return deleted

    async def on_start_async(self) -> None:
        if self._maintenance_registered:
            return
        try:
            from core.scheduler import TaskSpec, scheduler

            await scheduler.register(TaskSpec(
                name="periodic_conversation_prune",
                coro=lambda: self.prune_old_sessions(self._retention_keep_days),
                tick_interval=self._prune_interval_s,
                metadata={"keep_days": self._retention_keep_days},
            ))
            self._maintenance_registered = True
        except Exception as exc:
            record_degradation('persistence', exc)
            logger.warning("ConversationPersistence maintenance registration failed: %s", exc)

    def get_retention_status(self) -> Dict[str, float]:
        return {
            "keep_days": float(self._retention_keep_days),
            "prune_interval_s": float(self._prune_interval_s),
            "last_prune_at": float(self._last_prune_at or 0.0),
        }


_persistence: Optional[ConversationPersistence] = None


def get_persistence() -> ConversationPersistence:
    global _persistence
    if _persistence is None:
        _persistence = ConversationPersistence()
    return _persistence
