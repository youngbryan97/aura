"""
Durable conversation state.
Persists every turn to SQLite so any crash can be recovered from.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.sqlite_support import connecting
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
CONVERSATION_DB_BUSY_TIMEOUT_MS = 1000
MAX_PRINCIPAL_CHARS = 160
MAX_PRINCIPAL_SURFACE_CHARS = 32
MEMORY_LOG_OUTBOX_LEASE_S = 60.0
MEMORY_LOG_OUTBOX_MAX_ATTEMPTS = 8

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
CREATE INDEX IF NOT EXISTS idx_turns_session_cid ON turns(session_id, cid);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active DESC);

CREATE TABLE IF NOT EXISTS turn_revisions (
    turn_id         TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    revision        INTEGER NOT NULL CHECK (revision >= 1),
    content         TEXT NOT NULL,
    previous_content_sha256 TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    origin          TEXT NOT NULL,
    actor_principal_id TEXT NOT NULL,
    actor_principal_surface TEXT NOT NULL,
    updated_at      REAL NOT NULL,
    PRIMARY KEY (turn_id, revision)
);

CREATE TABLE IF NOT EXISTS conversation_memory_outbox (
    operation_id    TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    exchange_id     TEXT NOT NULL,
    revision        INTEGER NOT NULL CHECK (revision >= 1),
    user_content    TEXT NOT NULL,
    aura_content    TEXT NOT NULL,
    origin          TEXT NOT NULL,
    principal_id    TEXT NOT NULL,
    principal_surface TEXT NOT NULL,
    payload_sha256  TEXT NOT NULL,
    state           TEXT NOT NULL CHECK (
        state IN ('pending','processing','completed','rejected','failed')
    ),
    attempts        INTEGER NOT NULL DEFAULT 0,
    available_at    REAL NOT NULL,
    claimed_at      REAL,
    completed_at    REAL,
    episodic_logged INTEGER NOT NULL DEFAULT 0 CHECK (episodic_logged IN (0,1)),
    experience_recorded INTEGER NOT NULL DEFAULT 0 CHECK (experience_recorded IN (0,1)),
    consciousness_updated INTEGER NOT NULL DEFAULT 0 CHECK (consciousness_updated IN (0,1)),
    last_error      TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_outbox_ready
ON conversation_memory_outbox(state, available_at, created_at);
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


def _principal_binding(
    principal_id: object = "",
    principal_surface: object = "",
) -> tuple[str, str]:
    principal = " ".join(
        _safe_text(principal_id, max_chars=MAX_PRINCIPAL_CHARS).strip().split()
    )
    surface = _safe_text(
        principal_surface,
        max_chars=MAX_PRINCIPAL_SURFACE_CHARS,
    ).strip().casefold()
    if bool(principal) != bool(surface):
        raise ValueError("conversation principal binding requires id and surface")
    return principal, surface


def _session_metadata(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        decoded = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _metadata_principal_binding(metadata: object) -> tuple[str, str]:
    payload = _session_metadata(metadata)
    return _principal_binding(
        payload.get("principal_id") or "",
        payload.get("principal_surface") or "",
    )


def _validated_existing_turn_id(
    row: sqlite3.Row | None,
    *,
    expected_role: str,
    expected_content: str,
    cid: str,
) -> str | None:
    """Return an idempotent match, rejecting identity/content collisions."""

    if row is None:
        return None
    if (
        str(row["role"] or "") != expected_role
        or str(row["content"] or "") != expected_content
    ):
        raise ValueError(f"conversation turn cid conflict: {cid}")
    return str(row["id"])


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ConversationRevisionConflictError(RuntimeError):
    """The durable answer changed after regeneration selected its source."""


# Compatibility for callers compiled against CP358's pre-lint public name.
ConversationRevisionConflict = ConversationRevisionConflictError


class ConversationPersistence:

    def __init__(self, db_path: str | Path | None = None):
        self._db = str(db_path or _DB_PATH)
        Path(self._db).parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._init()
        self._current_session_id: str | None = None
        self._retention_keep_days = DEFAULT_CONVERSATION_RETENTION_DAYS
        self._prune_interval_s = DEFAULT_CONVERSATION_PRUNE_INTERVAL_S
        self._maintenance_registered = False
        self._last_prune_at: float = 0.0
        self._last_persist_error_at: float = 0.0

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(
            self._db,
            timeout=CONVERSATION_DB_BUSY_TIMEOUT_MS / 1000.0,
        )
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA busy_timeout={CONVERSATION_DB_BUSY_TIMEOUT_MS}")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self):
        with self._write_lock, connecting(self._connect()) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            existing_columns = {
                str(row[1])
                for row in con.execute("PRAGMA table_info(conversation_memory_outbox)")
            }
            for column in (
                "episodic_logged",
                "experience_recorded",
                "consciousness_updated",
            ):
                if column not in existing_columns:
                    con.execute(
                        f"ALTER TABLE conversation_memory_outbox ADD COLUMN {column} "
                        "INTEGER NOT NULL DEFAULT 0 CHECK ("
                        f"{column} IN (0,1))"
                    )
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
        with self._write_lock, connecting(self._connect()) as con:
            con.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (session_id, now, now, metadata_json),
            )
            con.commit()
        self._current_session_id = session_id
        logger.debug("Conversation session started: %s", session_id)
        return session_id

    @staticmethod
    def _ensure_session_row(
        con: sqlite3.Connection,
        session_id: str,
        now: float,
        *,
        principal_id: str = "",
        principal_surface: str = "",
    ) -> None:
        """Create an explicit UI session row before inserting turns.

        Desktop clients may send stable session ids that were not produced by
        ``start_session()``. The transcript store must treat those as real
        sessions instead of falling through to the boot singleton session or
        failing a foreign-key insert.
        """

        principal, surface = _principal_binding(principal_id, principal_surface)
        metadata = (
            {"principal_id": principal, "principal_surface": surface}
            if principal
            else {}
        )
        con.execute(
            "INSERT OR IGNORE INTO sessions VALUES (?,?,?,?)",
            (session_id, now, now, json.dumps(metadata, sort_keys=True)),
        )

        if not principal:
            return
        row = con.execute(
            "SELECT metadata FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        existing_principal, existing_surface = _metadata_principal_binding(
            row["metadata"] if row is not None else "{}"
        )
        if existing_principal:
            if (existing_principal, existing_surface) != (principal, surface):
                raise PermissionError("conversation session principal mismatch")
            return

        turn_count = int(
            con.execute(
                "SELECT COUNT(*) AS count FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()["count"]
        )
        if turn_count and surface != "owner":
            raise PermissionError(
                "unbound conversation history may only be adopted by the owner surface"
            )
        payload = _session_metadata(row["metadata"] if row is not None else "{}")
        payload.update({"principal_id": principal, "principal_surface": surface})
        con.execute(
            "UPDATE sessions SET metadata = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), session_id),
        )

    def record_turn(
        self,
        role: str,
        content: str,
        origin: str = "",
        cid: str | None = None,
        session_id: str | None = None,
        principal_id: str = "",
        principal_surface: str = "",
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
        inserted = False
        with self._write_lock, connecting(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            self._ensure_session_row(
                con,
                sid,
                now,
                principal_id=principal_id,
                principal_surface=principal_surface,
            )
            if cid:
                existing = con.execute(
                    "SELECT id, role, content FROM turns "
                    "WHERE session_id = ? AND cid = ? "
                    "ORDER BY created_at ASC, rowid ASC LIMIT 1",
                    (sid, cid),
                ).fetchone()
                if existing is not None:
                    existing_id = _validated_existing_turn_id(
                        existing,
                        expected_role=role,
                        expected_content=content,
                        cid=cid,
                    )
                    con.execute(
                        "UPDATE sessions SET last_active = ? WHERE id = ?",
                        (now, sid),
                    )
                    con.commit()
                    return str(existing_id)
            con.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?,?)",
                (turn_id, sid, role, content, origin, now, cid),
            )
            con.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?", (now, sid)
            )
            con.commit()
            inserted = True

        if inserted:
            self._publish_turn_recorded(
                role=role,
                content=content,
                origin=origin,
                cid=cid,
                session_id=sid,
                turn_id=turn_id,
            )
        return turn_id

    def record_exchange(
        self,
        user_content: str,
        aura_content: str,
        *,
        origin: str = "",
        cid: str | None = None,
        session_id: str | None = None,
        principal_id: str = "",
        principal_surface: str = "",
        enqueue_memory_log: bool = False,
    ) -> tuple[str, str]:
        """Atomically persist a completed user/Aura exchange."""

        sid = _safe_text(session_id or self._current_session_id, max_chars=64)
        if not sid:
            sid = self.start_session()

        user_turn_id = str(uuid.uuid4())[:12]
        aura_turn_id = str(uuid.uuid4())[:12]
        now = time.time()
        safe_user_content = _safe_text(user_content, max_chars=MAX_CONTENT_CHARS)
        safe_aura_content = _safe_text(aura_content, max_chars=MAX_CONTENT_CHARS)
        safe_origin = _safe_text(origin, max_chars=MAX_ORIGIN_CHARS)
        safe_cid = _safe_text(cid, max_chars=MAX_CID_CHARS)
        user_cid = (
            _safe_text(f"{safe_cid}:user", max_chars=MAX_CID_CHARS)
            if safe_cid
            else ""
        )
        aura_cid = (
            _safe_text(f"{safe_cid}:aura", max_chars=MAX_CID_CHARS)
            if safe_cid
            else ""
        )
        publish_user = False
        publish_aura = False

        with self._write_lock, connecting(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            self._ensure_session_row(
                con,
                sid,
                now,
                principal_id=principal_id,
                principal_surface=principal_surface,
            )
            existing_user = (
                con.execute(
                    "SELECT id, role, content FROM turns "
                    "WHERE session_id = ? AND cid = ? "
                    "ORDER BY created_at ASC, rowid ASC LIMIT 1",
                    (sid, user_cid),
                ).fetchone()
                if user_cid
                else None
            )
            if existing_user is not None:
                user_turn_id = str(
                    _validated_existing_turn_id(
                        existing_user,
                        expected_role="user",
                        expected_content=safe_user_content,
                        cid=user_cid,
                    )
                )
            else:
                con.execute(
                    "INSERT INTO turns VALUES (?,?,?,?,?,?,?)",
                    (
                        user_turn_id,
                        sid,
                        "user",
                        safe_user_content,
                        safe_origin,
                        now,
                        user_cid,
                    ),
                )
                publish_user = True

            existing_aura = (
                con.execute(
                    "SELECT id, role, content FROM turns "
                    "WHERE session_id = ? AND cid = ? "
                    "ORDER BY created_at ASC, rowid ASC LIMIT 1",
                    (sid, aura_cid),
                ).fetchone()
                if aura_cid
                else None
            )
            if existing_aura is not None:
                aura_turn_id = str(
                    _validated_existing_turn_id(
                        existing_aura,
                        expected_role="aura",
                        expected_content=safe_aura_content,
                        cid=aura_cid,
                    )
                )
            else:
                con.execute(
                    "INSERT INTO turns VALUES (?,?,?,?,?,?,?)",
                    (
                        aura_turn_id,
                        sid,
                        "aura",
                        safe_aura_content,
                        safe_origin,
                        now + 1e-6,
                        aura_cid,
                    ),
                )
                publish_aura = True
            con.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (now + 1e-6, sid),
            )
            if enqueue_memory_log:
                self._enqueue_memory_log_locked(
                    con,
                    session_id=sid,
                    exchange_id=safe_cid or aura_turn_id,
                    revision=1,
                    user_content=safe_user_content,
                    aura_content=safe_aura_content,
                    origin=safe_origin,
                    principal_id=principal_id,
                    principal_surface=principal_surface,
                    now=now + 1e-6,
                )
            con.commit()

        if publish_user:
            self._publish_turn_recorded(
                role="user",
                content=safe_user_content,
                origin=safe_origin,
                cid=user_cid,
                session_id=sid,
                turn_id=user_turn_id,
            )
        if publish_aura:
            self._publish_turn_recorded(
                role="aura",
                content=safe_aura_content,
                origin=safe_origin,
                cid=aura_cid,
                session_id=sid,
                turn_id=aura_turn_id,
            )
        return user_turn_id, aura_turn_id

    def replace_aura_turn(
        self,
        *,
        exchange_id: str,
        replacement_content: str,
        expected_revision: int,
        expected_content_sha256: str,
        session_id: str | None = None,
        principal_id: str = "",
        principal_surface: str = "",
        origin: str = "regenerate",
    ) -> dict[str, Any]:
        """Atomically replace one Aura turn if its selected revision is current.

        Original turns have logical revision 1 without requiring a migration
        rewrite. The first successful replacement creates the revision row at
        2; subsequent replacements advance it transactionally.
        """

        safe_exchange_id = _safe_text(exchange_id, max_chars=64).strip()
        if not safe_exchange_id:
            raise ValueError("conversation regeneration requires an exchange id")
        aura_cid = _safe_text(
            f"{safe_exchange_id}:aura",
            max_chars=MAX_CID_CHARS,
        )
        safe_session_id = _safe_text(session_id, max_chars=64).strip()
        safe_replacement = _safe_text(
            replacement_content,
            max_chars=MAX_CONTENT_CHARS,
        )
        if not safe_replacement:
            raise ValueError("conversation regeneration requires replacement content")
        try:
            expected_revision_value = int(expected_revision)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("conversation regeneration revision must be an integer") from exc
        if expected_revision_value < 1:
            raise ValueError("conversation regeneration revision must be positive")
        expected_sha = _safe_text(
            expected_content_sha256,
            max_chars=64,
        ).strip().lower()
        if len(expected_sha) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha
        ):
            raise ValueError("conversation regeneration requires a SHA-256 content identity")
        principal, surface = _principal_binding(principal_id, principal_surface)
        safe_origin = _safe_text(origin, max_chars=MAX_ORIGIN_CHARS)
        now = time.time()

        with self._write_lock, connecting(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            if safe_session_id:
                rows = con.execute(
                    "SELECT t.id, t.session_id, t.role, t.content, t.cid, t.created_at, "
                    "s.metadata, r.revision, r.content_sha256 "
                    "FROM turns t JOIN sessions s ON s.id = t.session_id "
                    "LEFT JOIN turn_revisions r ON r.turn_id = t.id AND "
                    "r.revision = (SELECT MAX(r2.revision) FROM turn_revisions r2 "
                    "WHERE r2.turn_id = t.id) "
                    "WHERE t.session_id = ? AND t.cid = ? "
                    "ORDER BY t.created_at ASC, t.rowid ASC",
                    (safe_session_id, aura_cid),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT t.id, t.session_id, t.role, t.content, t.cid, t.created_at, "
                    "s.metadata, r.revision, r.content_sha256 "
                    "FROM turns t JOIN sessions s ON s.id = t.session_id "
                    "LEFT JOIN turn_revisions r ON r.turn_id = t.id AND "
                    "r.revision = (SELECT MAX(r2.revision) FROM turn_revisions r2 "
                    "WHERE r2.turn_id = t.id) "
                    "WHERE t.cid = ? ORDER BY t.created_at ASC, t.rowid ASC",
                    (aura_cid,),
                ).fetchall()
            if len(rows) != 1:
                raise ConversationRevisionConflictError(
                    "conversation regeneration target is missing or ambiguous"
                )
            row = rows[0]
            if str(row["role"] or "") != "aura":
                raise ConversationRevisionConflictError(
                    "conversation regeneration target is not an Aura turn"
                )
            if principal:
                bound_principal, bound_surface = _metadata_principal_binding(
                    row["metadata"]
                )
                if bound_principal:
                    if (bound_principal, bound_surface) != (principal, surface):
                        raise PermissionError("conversation session principal mismatch")
                elif surface != "owner":
                    raise PermissionError(
                        "unbound conversation history may only be regenerated by the owner surface"
                    )

            current_content = str(row["content"] or "")
            current_sha = _content_sha256(current_content)
            current_revision = int(row["revision"] or 1)
            ledger_sha = str(row["content_sha256"] or "")
            if ledger_sha and not hmac.compare_digest(ledger_sha, current_sha):
                raise RuntimeError("conversation revision ledger/content drift")
            if current_revision != expected_revision_value or not hmac.compare_digest(
                current_sha,
                expected_sha,
            ):
                raise ConversationRevisionConflictError(
                    "conversation regeneration source revision changed"
                )

            replacement_sha = _content_sha256(safe_replacement)
            next_revision = current_revision + 1
            if current_revision == 1 and not ledger_sha:
                con.execute(
                    "INSERT INTO turn_revisions("
                    "turn_id, revision, content, previous_content_sha256, "
                    "content_sha256, origin, actor_principal_id, "
                    "actor_principal_surface, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        str(row["id"]),
                        1,
                        current_content,
                        "",
                        current_sha,
                        "initial_delivery",
                        "",
                        "",
                        float(row["created_at"]),
                    ),
                )
            updated = con.execute(
                "UPDATE turns SET content = ? WHERE id = ? AND content = ?",
                (safe_replacement, str(row["id"]), current_content),
            )
            if int(updated.rowcount or 0) != 1:
                raise ConversationRevisionConflictError(
                    "conversation regeneration lost its durable compare-and-swap"
                )
            con.execute(
                "INSERT INTO turn_revisions("
                "turn_id, revision, content, previous_content_sha256, "
                "content_sha256, origin, actor_principal_id, "
                "actor_principal_surface, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(row["id"]),
                    next_revision,
                    safe_replacement,
                    current_sha,
                    replacement_sha,
                    safe_origin,
                    principal,
                    surface,
                    now,
                ),
            )
            con.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (now, str(row["session_id"])),
            )
            self._enqueue_memory_log_locked(
                con,
                session_id=str(row["session_id"]),
                exchange_id=safe_exchange_id,
                revision=next_revision,
                user_content=self._exchange_user_content_locked(
                    con,
                    session_id=str(row["session_id"]),
                    exchange_id=safe_exchange_id,
                ),
                aura_content=safe_replacement,
                origin=safe_origin,
                principal_id=principal,
                principal_surface=surface,
                now=now,
            )
            con.commit()

        self._publish_turn_regenerated(
            exchange_id=safe_exchange_id,
            session_id=str(row["session_id"]),
            turn_id=str(row["id"]),
            previous_revision=current_revision,
            revision=next_revision,
            previous_content_sha256=current_sha,
            content_sha256=replacement_sha,
            origin=safe_origin,
            actor_principal_id=principal,
            actor_principal_surface=surface,
        )

        return {
            "exchange_id": safe_exchange_id,
            "session_id": str(row["session_id"]),
            "turn_id": str(row["id"]),
            "previous_revision": current_revision,
            "revision": next_revision,
            "previous_content_sha256": current_sha,
            "previous_content": current_content,
            "content_sha256": replacement_sha,
            "applied": True,
        }

    @staticmethod
    def _exchange_user_content_locked(
        con: sqlite3.Connection,
        *,
        session_id: str,
        exchange_id: str,
    ) -> str:
        row = con.execute(
            "SELECT content FROM turns WHERE session_id = ? AND cid = ? "
            "ORDER BY created_at ASC, rowid ASC LIMIT 1",
            (session_id, f"{exchange_id}:user"),
        ).fetchone()
        if row is None:
            raise ConversationRevisionConflictError(
                "conversation regeneration user turn is missing"
            )
        return str(row["content"] or "")

    @staticmethod
    def _enqueue_memory_log_locked(
        con: sqlite3.Connection,
        *,
        session_id: str,
        exchange_id: str,
        revision: int,
        user_content: str,
        aura_content: str,
        origin: str,
        principal_id: str,
        principal_surface: str,
        now: float,
    ) -> str:
        safe_exchange_id = _safe_text(exchange_id, max_chars=64)
        operation_id = f"{session_id}:{safe_exchange_id}:r{int(revision)}"
        payload = {
            "session_id": session_id,
            "exchange_id": safe_exchange_id,
            "revision": int(revision),
            "user_content": user_content,
            "aura_content": aura_content,
            "origin": origin,
            "principal_id": principal_id,
            "principal_surface": principal_surface,
        }
        payload_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        existing = con.execute(
            "SELECT payload_sha256 FROM conversation_memory_outbox "
            "WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if existing is not None:
            if not hmac.compare_digest(
                str(existing["payload_sha256"] or ""),
                payload_sha256,
            ):
                raise ValueError(
                    f"conversation memory outbox identity conflict: {operation_id}"
                )
            return operation_id
        con.execute(
            "INSERT INTO conversation_memory_outbox("
            "operation_id, session_id, exchange_id, revision, user_content, "
            "aura_content, origin, principal_id, principal_surface, payload_sha256, "
            "state, attempts, available_at, claimed_at, completed_at, last_error, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                operation_id,
                session_id,
                safe_exchange_id,
                int(revision),
                user_content,
                aura_content,
                origin,
                principal_id,
                principal_surface,
                payload_sha256,
                "pending",
                0,
                now,
                None,
                None,
                "",
                now,
                now,
            ),
        )
        return operation_id

    def claim_memory_log_batch(
        self,
        *,
        limit: int = 16,
        lease_s: float = MEMORY_LOG_OUTBOX_LEASE_S,
    ) -> list[dict[str, Any]]:
        """Lease durable memory-log work; expired claims become eligible again."""

        safe_limit = max(1, min(128, int(limit)))
        safe_lease_s = max(1.0, float(lease_s))
        now = time.time()
        with self._write_lock, connecting(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "UPDATE conversation_memory_outbox SET state='pending', "
                "claimed_at=NULL, updated_at=?, last_error=CASE "
                "WHEN last_error='' THEN 'expired_processing_lease' ELSE last_error END "
                "WHERE state='processing' AND claimed_at IS NOT NULL AND claimed_at < ?",
                (now, now - safe_lease_s),
            )
            rows = con.execute(
                "SELECT * FROM conversation_memory_outbox "
                "WHERE state='pending' AND available_at <= ? "
                "ORDER BY created_at ASC, operation_id ASC LIMIT ?",
                (now, safe_limit),
            ).fetchall()
            operation_ids = [str(row["operation_id"]) for row in rows]
            for operation_id in operation_ids:
                con.execute(
                    "UPDATE conversation_memory_outbox SET state='processing', "
                    "attempts=attempts+1, claimed_at=?, updated_at=? "
                    "WHERE operation_id=? AND state='pending'",
                    (now, now, operation_id),
                )
            con.commit()
        claimed = []
        for row in rows:
            item = dict(row)
            item["state"] = "processing"
            item["attempts"] = int(item.get("attempts") or 0) + 1
            item["claimed_at"] = now
            claimed.append(item)
        return claimed

    def settle_memory_log_item(
        self,
        operation_id: str,
        *,
        outcome: str,
        error: str = "",
        retry_delay_s: float = 0.0,
    ) -> str:
        """Complete, reject, or durably retry one leased outbox item."""

        safe_operation_id = _safe_text(operation_id, max_chars=160)
        safe_outcome = str(outcome or "").strip().casefold()
        if safe_outcome not in {"completed", "rejected", "retry", "failed"}:
            raise ValueError("invalid conversation memory outbox outcome")
        now = time.time()
        with self._write_lock, connecting(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT state, attempts FROM conversation_memory_outbox "
                "WHERE operation_id=?",
                (safe_operation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown conversation memory outbox item: {safe_operation_id}")
            if str(row["state"] or "") in {"completed", "rejected", "failed"}:
                con.commit()
                return str(row["state"])
            if str(row["state"] or "") != "processing":
                raise RuntimeError("conversation memory outbox item is not leased")
            attempts = int(row["attempts"] or 0)
            if safe_outcome == "retry" and attempts >= MEMORY_LOG_OUTBOX_MAX_ATTEMPTS:
                terminal_state = "failed"
            elif safe_outcome == "retry":
                terminal_state = "pending"
            else:
                terminal_state = safe_outcome
            completed_at = now if terminal_state in {"completed", "rejected", "failed"} else None
            available_at = (
                now + max(0.0, float(retry_delay_s))
                if terminal_state == "pending"
                else now
            )
            con.execute(
                "UPDATE conversation_memory_outbox SET state=?, available_at=?, "
                "claimed_at=NULL, completed_at=?, last_error=?, updated_at=? "
                "WHERE operation_id=?",
                (
                    terminal_state,
                    available_at,
                    completed_at,
                    _safe_text(error, max_chars=1000),
                    now,
                    safe_operation_id,
                ),
            )
            con.commit()
        return terminal_state

    def mark_memory_log_stage(self, operation_id: str, *, stage: str) -> bool:
        """Durably checkpoint one idempotent outbox stage before settlement.

        A leased item can be replayed after process death or a settlement
        failure. Persisting each completed stage keeps that replay from
        reapplying profile, memory, relationship, or consciousness effects.
        """

        safe_operation_id = _safe_text(operation_id, max_chars=160)
        column = {
            "episodic": "episodic_logged",
            "experience": "experience_recorded",
            "consciousness": "consciousness_updated",
        }.get(str(stage or "").strip().casefold())
        if column is None:
            raise ValueError("invalid conversation memory outbox stage")
        now = time.time()
        with self._write_lock, connecting(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                f"SELECT state, {column} FROM conversation_memory_outbox "
                "WHERE operation_id=?",
                (safe_operation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown conversation memory outbox item: {safe_operation_id}")
            if int(row[column] or 0) == 1:
                con.commit()
                return False
            if str(row["state"] or "") != "processing":
                raise RuntimeError("conversation memory outbox item is not leased")
            con.execute(
                f"UPDATE conversation_memory_outbox SET {column}=1, updated_at=? "
                "WHERE operation_id=? AND state='processing'",
                (now, safe_operation_id),
            )
            con.commit()
        return True

    def memory_log_outbox_status(self) -> dict[str, int]:
        with connecting(self._connect()) as con:
            rows = con.execute(
                "SELECT state, COUNT(*) AS count FROM conversation_memory_outbox "
                "GROUP BY state"
            ).fetchall()
        counts = {str(row["state"]): int(row["count"] or 0) for row in rows}
        return {
            state: counts.get(state, 0)
            for state in ("pending", "processing", "completed", "rejected", "failed")
        }

    def _publish_turn_regenerated(
        self,
        **payload: object,
    ) -> None:
        try:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            publish_threadsafe = getattr(bus, "publish_threadsafe", None)
            if callable(publish_threadsafe):
                publish_threadsafe("turn_regenerated", dict(payload))
                return
            publish_result = bus.publish("turn_regenerated", dict(payload))
            if asyncio.iscoroutine(publish_result):
                try:
                    get_task_tracker().create_task(
                        publish_result,
                        name="conversation.turn_regenerated.publish",
                    )
                except _PERSISTENCE_ERRORS as schedule_exc:
                    publish_result.close()
                    raise schedule_exc
        except _PERSISTENCE_ERRORS as exc:
            self._last_persist_error_at = time.time()
            _record_persistence_degradation(
                exc,
                action="persisted regeneration while event publication failed",
                severity="warning",
                extra={
                    "exchange_id": payload.get("exchange_id", ""),
                    "revision": payload.get("revision", 0),
                },
            )

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
        *,
        principal_id: str = "",
        principal_surface: str = "",
    ) -> list[dict[str, Any]]:
        sid = _safe_text(session_id or self._current_session_id, max_chars=64)
        if not sid:
            return []
        limit = _safe_limit(limit, 100)
        principal, surface = _principal_binding(principal_id, principal_surface)
        with connecting(self._connect()) as con:
            if principal:
                session = con.execute(
                    "SELECT metadata FROM sessions WHERE id = ?",
                    (sid,),
                ).fetchone()
                if session is None:
                    return []
                bound_principal, bound_surface = _metadata_principal_binding(
                    session["metadata"]
                )
                if bound_principal:
                    if (bound_principal, bound_surface) != (principal, surface):
                        return []
                elif surface != "owner":
                    return []
            rows = con.execute(
                "SELECT * FROM ("
                "SELECT t.*, r.revision AS revision, "
                "r.content_sha256 AS revision_content_sha256 "
                "FROM turns t LEFT JOIN turn_revisions r "
                "ON r.turn_id = t.id AND r.revision = ("
                "SELECT MAX(r2.revision) FROM turn_revisions r2 "
                "WHERE r2.turn_id = t.id) "
                "WHERE t.session_id = ? "
                "ORDER BY t.created_at DESC, t.rowid DESC LIMIT ?"
                ") ORDER BY created_at ASC",
                (sid, limit),
            ).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["revision"] = int(item.get("revision") or 1)
            item["content_sha256"] = str(
                item.pop("revision_content_sha256", "")
                or _content_sha256(str(item.get("content") or ""))
            )
            history.append(item)
        return history

    def get_recent_sessions(
        self,
        limit: int = 10,
        *,
        with_turns_only: bool = False,
        principal_id: str = "",
        principal_surface: str = "",
    ) -> list[dict[str, Any]]:
        """Most recently active sessions, newest first.

        ``with_turns_only`` drops sessions that hold no turns. Every boot opens
        a session row before anything is said in it, so the unfiltered list is
        mostly boot artifacts — see :meth:`recover_last_session` for what that
        cost in practice.
        """
        limit = _safe_limit(limit, 10)
        principal, surface = _principal_binding(principal_id, principal_surface)
        having = "HAVING COUNT(t.id) > 0 " if with_turns_only else ""
        with connecting(self._connect()) as con:
            rows = con.execute(
                "SELECT s.*, COUNT(t.id) as turn_count "
                "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id "
                f"GROUP BY s.id {having}ORDER BY s.last_active DESC LIMIT ?",
                (MAX_QUERY_LIMIT if principal else limit,),
            ).fetchall()
        sessions = [dict(r) for r in rows]
        if principal:
            authorized: list[dict[str, Any]] = []
            for session in sessions:
                bound_principal, bound_surface = _metadata_principal_binding(
                    session.get("metadata")
                )
                if bound_principal:
                    if (bound_principal, bound_surface) != (principal, surface):
                        continue
                elif surface != "owner":
                    continue
                authorized.append(session)
                if len(authorized) >= limit:
                    break
            return authorized
        return sessions[:limit]

    def recover_last_session(self) -> str | None:
        """Return the most recent session that actually holds turns.

        LIVE DEFECT, 2026-08-10. Asked "we talked earlier today and then I
        restarted you — do you remember what we were talking about?", she
        answered "my state was reset and I have no memory of it" while all 34
        turns of that conversation sat in this table.

        Every boot calls :meth:`start_session`, which writes a session row
        before a word is spoken in it. Ordering by ``last_active`` alone made
        the newest empty boot row the "last session", so recovery resumed a
        conversation with nothing in it, and the bounded scan that feeds
        durable recall spent its slots on empty rows — five were created on the
        day this was found. A session with no turns is a boot artifact, not a
        conversation to resume.
        """
        with connecting(self._connect()) as con:
            row = con.execute(
                "SELECT s.id FROM sessions s "
                "JOIN turns t ON t.session_id = s.id "
                "GROUP BY s.id ORDER BY s.last_active DESC LIMIT 1"
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
        with self._write_lock, connecting(self._connect()) as con:
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
