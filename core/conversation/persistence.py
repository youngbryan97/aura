"""
Durable conversation state.
Persists every turn to SQLite so any crash can be recovered from.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ConvPersistence")
SECONDS_PER_DAY = 86400
DEFAULT_CONVERSATION_RETENTION_DAYS = 30
DEFAULT_CONVERSATION_PRUNE_INTERVAL_S = 86400.0
MAX_ROLE_CHARS = 32
MAX_ORIGIN_CHARS = 64
MAX_CID_CHARS = 128
MAX_CONTENT_CHARS = 2_000_000
MAX_QUERY_LIMIT = 1000

_PERSISTENCE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)

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


def _record_persistence_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "persistence",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "persistence",
                error,
                severity=severity,
                action=action or "conversation persistence degraded",
            )
        except TypeError:
            logger.warning(
                "Conversation persistence degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_text(value: object, *, default: str = "", max_chars: int = 4096) -> str:
    try:
        text = str(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        text = default
    text = text.replace("\x00", "")
    return text[:max_chars]


def _safe_limit(value: object, default: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        limit = default
    return max(1, min(MAX_QUERY_LIMIT, limit))


class ConversationPersistence:

    def __init__(self, db_path: str | Path | None = None):
        self._db = str(db_path or _DB_PATH)
        Path(self._db).parent.mkdir(parents=True, exist_ok=True)
        self._init()
        self._current_session_id: str | None = None
        self._retention_keep_days = DEFAULT_CONVERSATION_RETENTION_DAYS
        self._prune_interval_s = DEFAULT_CONVERSATION_PRUNE_INTERVAL_S
        self._maintenance_registered = False
        self._last_prune_at: float = 0.0
        self._last_persist_error_at: float = 0.0

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=10000")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self):
        with self._connect() as con:
            con.executescript(_SCHEMA)
            con.commit()

    def start_session(self, metadata: dict[str, Any] | None = None) -> str:
        session_id = str(uuid.uuid4())[:16]
        now = time.time()
        try:
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            _record_persistence_degradation(
                exc,
                action="started conversation session with sanitized metadata",
                severity="warning",
            )
            metadata_json = "{}"
        with self._connect() as con:
            con.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (session_id, now, now, metadata_json),
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
        cid: str | None = None,
        session_id: str | None = None,
    ) -> str:
        sid = _safe_text(session_id or self._current_session_id, max_chars=64)
        if not sid:
            sid = self.start_session()

        turn_id = str(uuid.uuid4())[:12]
        now = time.time()
        role = _safe_text(role, default="system", max_chars=MAX_ROLE_CHARS) or "system"
        content = _safe_text(content, max_chars=MAX_CONTENT_CHARS)
        origin = _safe_text(origin, max_chars=MAX_ORIGIN_CHARS)
        cid = _safe_text(cid, max_chars=MAX_CID_CHARS)
        with self._connect() as con:
            con.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?,?)",
                (turn_id, sid, role, content, origin, now, cid),
            )
            con.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?", (now, sid)
            )
            con.commit()

        self._publish_turn_recorded(
            role=role,
            content=content,
            origin=origin,
            cid=cid,
            session_id=sid,
            turn_id=turn_id,
        )
        return turn_id

    def _publish_turn_recorded(
        self,
        *,
        role: str,
        content: str,
        origin: str,
        cid: str,
        session_id: str,
        turn_id: str,
    ) -> None:
        try:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            payload = {
                "role": role,
                "content": content,
                "origin": origin,
                "cid": cid,
                "session_id": session_id,
                "turn_id": turn_id,
                "content_chars": len(content),
            }
            publish_threadsafe = getattr(bus, "publish_threadsafe", None)
            if callable(publish_threadsafe):
                publish_threadsafe("turn_recorded", payload)
                return
            publish_result = bus.publish("turn_recorded", payload)
            if asyncio.iscoroutine(publish_result):
                try:
                    get_task_tracker().create_task(
                        publish_result,
                        name="conversation.turn_recorded.publish",
                    )
                except _PERSISTENCE_ERRORS as schedule_exc:
                    publish_result.close()
                    raise schedule_exc
        except _PERSISTENCE_ERRORS as exc:
            self._last_persist_error_at = time.time()
            _record_persistence_degradation(
                exc,
                action="persisted turn while turn_recorded event publication failed",
                severity="warning",
            )
            logger.debug("Turn recorded but event bus failed to publish: %s", exc)

    def get_session_history(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sid = _safe_text(session_id or self._current_session_id, max_chars=64)
        if not sid:
            return []
        limit = _safe_limit(limit, 100)
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
                (sid, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        limit = _safe_limit(limit, 10)
        with self._connect() as con:
            rows = con.execute(
                "SELECT s.*, COUNT(t.id) as turn_count "
                "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id "
                "GROUP BY s.id ORDER BY s.last_active DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recover_last_session(self) -> str | None:
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
        try:
            keep_days = int(keep_days)
        except (TypeError, ValueError, OverflowError):
            keep_days = DEFAULT_CONVERSATION_RETENTION_DAYS
        keep_days = max(1, min(3650, keep_days))
        cutoff = time.time() - (keep_days * SECONDS_PER_DAY)
        with self._connect() as con:
            # Manually cascade to be absolutely sure (Audit-33 fix)
            con.execute(
                "DELETE FROM turns WHERE session_id IN (SELECT id FROM sessions WHERE last_active < ?)",
                (cutoff,),
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

            await scheduler.register(
                TaskSpec(
                    name="periodic_conversation_prune",
                    coro=lambda: self.prune_old_sessions(self._retention_keep_days),
                    tick_interval=self._prune_interval_s,
                    metadata={"keep_days": self._retention_keep_days},
                )
            )
            self._maintenance_registered = True
        except _PERSISTENCE_ERRORS as exc:
            self._last_persist_error_at = time.time()
            _record_persistence_degradation(
                exc,
                action="continued without scheduled conversation pruning after scheduler registration failed",
                severity="warning",
            )
            logger.warning(
                "ConversationPersistence maintenance registration failed: %s",
                exc,
            )

    def get_retention_status(self) -> dict[str, float]:
        return {
            "keep_days": float(self._retention_keep_days),
            "prune_interval_s": float(self._prune_interval_s),
            "last_prune_at": float(self._last_prune_at or 0.0),
            "last_persist_error_at": float(self._last_persist_error_at or 0.0),
        }


_persistence: ConversationPersistence | None = None


def get_persistence() -> ConversationPersistence:
    global _persistence
    if _persistence is None:
        _persistence = ConversationPersistence()
    return _persistence
