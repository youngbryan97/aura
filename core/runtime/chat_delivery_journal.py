"""Durable, fenced at-most-once coordination for HTTP chat turns.

The journal is deliberately independent of the chat implementation.  It owns
only admission, lease fencing, terminal response replay, and bounded retention;
the route remains responsible for authentication, governance, and cognition.
Every SQLite operation runs through an async ``to_thread`` facade so a busy or
damaged journal cannot block Aura's event loop. A process loss after effects
begin but before the terminal receipt is committed is recorded as ambiguous;
the journal never pretends it can safely replay an unreceipted external effect.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Never

from core.runtime.flags import FlagKind, declare
from core.runtime.state_ownership import state_root

_SCHEMA_VERSION = 2
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,240}$")
_PROGRESS_PHASE_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_TERMINAL_STATES = frozenset({"awaiting_approval", "completed", "failed", "ambiguous"})
_PENDING_STATES = frozenset({"queued", "running"})
_ALL_STATES = _TERMINAL_STATES | _PENDING_STATES
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_PROGRESS_BYTES = 64 * 1024
_CLOCK_SKEW_TOLERANCE_S = 1.0
_DB_PATH_FLAG = declare(
    "AURA_CHAT_DELIVERY_DB",
    kind=FlagKind.STRING,
    default="",
    description="Override path for the durable chat delivery journal",
    owner="core.runtime.chat_delivery_journal",
)


class ChatDeliveryJournalError(RuntimeError):
    """Base class for delivery-journal failures."""


class ChatDeliveryJournalUnavailable(ChatDeliveryJournalError):  # noqa: N818 - public API
    """The durable coordinator could not be reached safely."""


class ChatDeliveryJournalCorruption(ChatDeliveryJournalError):  # noqa: N818 - public API
    """The database or one of its response records failed validation."""


class ChatDeliveryFenceLost(ChatDeliveryJournalError):  # noqa: N818 - public API
    """A superseded execution attempted to publish a result."""


class AdmissionKind(StrEnum):
    EXECUTE = "execute"
    REPLAY = "replay"
    MISMATCH = "mismatch"
    PENDING = "pending"


class DeliveryState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class DeliveryIdentity:
    principal_digest: str
    session_id: str
    idempotency_key: str

    @classmethod
    def create(
        cls,
        *,
        principal: str,
        session_id: str,
        idempotency_key: str,
    ) -> DeliveryIdentity:
        normalized_principal = " ".join(str(principal or "").strip().split())
        normalized_session = str(session_id or "default").strip() or "default"
        normalized_key = str(idempotency_key or "").strip()
        if not normalized_principal:
            raise ValueError("chat delivery principal is required")
        if len(normalized_principal) > 240:
            raise ValueError("chat delivery principal exceeds 240 characters")
        if len(normalized_session) > 240 or any(ord(char) < 32 for char in normalized_session):
            raise ValueError("invalid chat delivery session")
        if not _IDEMPOTENCY_KEY_RE.fullmatch(normalized_key):
            raise ValueError("invalid chat idempotency key")
        return cls(
            principal_digest=hashlib.sha256(
                normalized_principal.encode("utf-8", errors="strict")
            ).hexdigest(),
            session_id=normalized_session,
            idempotency_key=normalized_key,
        )


@dataclass(frozen=True)
class DeliveryRecord:
    identity: DeliveryIdentity
    request_hash: str
    turn_id: str
    state: DeliveryState
    generation: int
    attempts: int
    created_at: float
    updated_at: float
    lease_expires_at: float
    terminal_at: float | None
    http_status: int | None
    response: dict[str, Any] | None
    progress_sequence: int
    progress_at: float | None
    progress: dict[str, Any] | None

    @property
    def terminal(self) -> bool:
        return self.state.value in _TERMINAL_STATES

    @property
    def approval_required(self) -> bool:
        return self.state is DeliveryState.AWAITING_APPROVAL

    def public_status(
        self,
        *,
        include_result: bool = True,
        request_matches: bool = True,
    ) -> dict[str, Any]:
        if not request_matches:
            return {
                "delivery_status": "mismatch",
                "state": "mismatch",
                "idempotency_key": self.identity.idempotency_key,
            }
        if self.approval_required:
            delivery_status = "approval"
        elif self.terminal:
            delivery_status = "terminal"
        else:
            delivery_status = "pending"
        payload: dict[str, Any] = {
            "delivery_status": delivery_status,
            "state": self.state.value,
            "idempotency_key": self.identity.idempotency_key,
            "turn_id": self.turn_id,
            "generation": self.generation,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "terminal": self.terminal,
            "approval_required": self.approval_required,
        }
        if self.http_status is not None:
            payload["http_status"] = self.http_status
        if not self.terminal:
            payload["retry_after_ms"] = 500
        if self.progress is not None:
            payload["progress"] = dict(self.progress)
        if include_result and self.terminal and self.response is not None:
            payload["result"] = self.response
        return payload


@dataclass(frozen=True)
class DeliveryAdmission:
    kind: AdmissionKind
    record: DeliveryRecord
    owner_token: str = ""

    @property
    def may_execute(self) -> bool:
        return self.kind is AdmissionKind.EXECUTE


def default_chat_delivery_db_path() -> Path:
    override = str(_DB_PATH_FLAG.value() or "").strip()
    if override:
        return Path(override).expanduser()
    test_root = str(os.environ.get("AURA_TEST_RUNTIME_ROOT") or "").strip()
    if test_root:
        return Path(test_root).expanduser() / "chat_delivery.sqlite3"
    return state_root() / "data" / "chat_delivery.sqlite3"


def canonical_request_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_response(payload: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("chat delivery response must be an object")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    raw = encoded.encode("utf-8", errors="strict")
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("chat delivery response exceeds durable replay limit")
    return encoded, hashlib.sha256(raw).hexdigest()


def _canonical_progress(
    *,
    phase: str,
    message: str,
    sequence: int,
    observed_at: float,
    details: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    normalized_phase = str(phase or "").strip().casefold()
    normalized_message = " ".join(str(message or "").strip().split())
    if not _PROGRESS_PHASE_RE.fullmatch(normalized_phase):
        raise ValueError("chat delivery progress phase is invalid")
    if not normalized_message or len(normalized_message) > 600:
        raise ValueError("chat delivery progress message must be 1-600 characters")
    if isinstance(sequence, bool) or int(sequence) < 1:
        raise ValueError("chat delivery progress sequence must be positive")
    if not math.isfinite(observed_at) or observed_at < 0:
        raise ValueError("chat delivery progress time must be finite")
    normalized_details = dict(details or {})
    payload = {
        "schema": "aura.chat.delivery.progress.v1",
        "sequence": int(sequence),
        "phase": normalized_phase,
        "message": normalized_message,
        "observed_at": float(observed_at),
        "details": normalized_details,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    raw = encoded.encode("utf-8", errors="strict")
    if len(raw) > _MAX_PROGRESS_BYTES:
        raise ValueError("chat delivery progress exceeds durable storage limit")
    return encoded, hashlib.sha256(raw).hexdigest(), payload


def _finite_positive(value: Any, *, name: str, minimum: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return normalized


def _finite_timestamp(value: Any, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ChatDeliveryJournalCorruption(f"invalid chat delivery {name}") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ChatDeliveryJournalCorruption(f"invalid chat delivery {name}")
    return normalized


class ChatDeliveryJournal:
    """SQLite-backed delivery admission and terminal replay authority."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        stale_after_s: float = 180.0,
        retention_s: float = 7 * 24 * 60 * 60.0,
        abandon_after_s: float = 24 * 60 * 60.0,
        max_rows: int = 10_000,
        poll_interval_s: float = 0.05,
        busy_timeout_s: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path or default_chat_delivery_db_path()).expanduser()
        self.stale_after_s = _finite_positive(
            stale_after_s,
            name="stale_after_s",
            minimum=0.05,
        )
        self.retention_s = max(
            self.stale_after_s,
            _finite_positive(retention_s, name="retention_s", minimum=0.05),
        )
        self.abandon_after_s = max(
            self.stale_after_s * 2.0,
            _finite_positive(
                abandon_after_s,
                name="abandon_after_s",
                minimum=0.1,
            ),
        )
        if isinstance(max_rows, bool):
            raise ValueError("max_rows must be an integer")
        try:
            normalized_max_rows = int(max_rows)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_rows must be an integer") from exc
        if normalized_max_rows < 10:
            raise ValueError("max_rows must be at least 10")
        self.max_rows = normalized_max_rows
        self.poll_interval_s = _finite_positive(
            poll_interval_s,
            name="poll_interval_s",
            minimum=0.01,
        )
        self.busy_timeout_s = _finite_positive(
            busy_timeout_s,
            name="busy_timeout_s",
            minimum=0.1,
        )
        self._initialize()

    def _raise_sqlite(self, exc: sqlite3.Error) -> Never:
        message = str(exc).casefold()
        if "locked" in message or "busy" in message:
            raise ChatDeliveryJournalUnavailable("chat delivery journal is busy") from exc
        raise ChatDeliveryJournalCorruption(
            "chat delivery journal failed integrity validation"
        ) from exc

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.busy_timeout_s,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_s * 1000)}")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    def _initialize(self) -> None:
        try:
            if self.db_path.is_symlink():
                raise ChatDeliveryJournalCorruption(
                    "chat delivery journal path must not be a symlink"
                )
            existed = self.db_path.exists() and self.db_path.stat().st_size > 0
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "chat_delivery_journal.initialize",
                domain="file_write",
            ):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(
                    self.db_path.parent,
                    source="core.runtime.chat_delivery_journal.initialize",
                )
                # SQLite must mutate its own file transactionally. Pre-open it
                # through the canonical gateway to enforce O_NOFOLLOW and 0600
                # before SQLite can create a database or WAL sidecar.
                with gateway.open_owned_binary(
                    self.db_path,
                    mode="a+b",
                    permissions=0o600,
                    source="core.runtime.chat_delivery_journal.initialize",
                ):
                    pass
            conn = self._connect()
            try:
                conn.execute("PRAGMA trusted_schema=OFF")
                check = conn.execute("PRAGMA quick_check(1)").fetchone()
                if check is None or str(check[0]).casefold() != "ok":
                    raise ChatDeliveryJournalCorruption("chat delivery journal quick_check failed")
                existing_tables = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                    if not str(row[0]).startswith("sqlite_")
                }
                if existed and existing_tables and "chat_delivery_meta" not in existing_tables:
                    raise ChatDeliveryJournalCorruption(
                        "existing chat delivery database has no schema identity"
                    )
                journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if journal_mode is None or str(journal_mode[0]).casefold() != "wal":
                    raise ChatDeliveryJournalUnavailable(
                        "chat delivery journal could not enable WAL durability"
                    )
                conn.execute("PRAGMA synchronous=FULL")
                synchronous = conn.execute("PRAGMA synchronous").fetchone()
                if synchronous is None or int(synchronous[0]) < 2:
                    raise ChatDeliveryJournalUnavailable(
                        "chat delivery journal could not enable full synchronization"
                    )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS chat_delivery_meta ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                version_row = conn.execute(
                    "SELECT value FROM chat_delivery_meta WHERE key='schema_version'"
                ).fetchone()
                if existed and existing_tables and version_row is None:
                    raise ChatDeliveryJournalCorruption(
                        "existing chat delivery database has no schema identity"
                    )
                schema_version = 0
                if version_row is not None:
                    try:
                        schema_version = int(version_row[0])
                    except (TypeError, ValueError) as exc:
                        raise ChatDeliveryJournalCorruption(
                            "invalid chat delivery journal schema identity"
                        ) from exc
                    if schema_version not in {1, _SCHEMA_VERSION}:
                        raise ChatDeliveryJournalCorruption(
                            "unsupported chat delivery journal schema"
                        )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_deliveries (
                        principal_digest TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        request_hash TEXT NOT NULL,
                        turn_id TEXT NOT NULL UNIQUE,
                        state TEXT NOT NULL CHECK (
                            state IN ('queued','running','awaiting_approval',
                                      'completed','failed','ambiguous')
                        ),
                        generation INTEGER NOT NULL CHECK (generation >= 0),
                        attempts INTEGER NOT NULL CHECK (attempts >= 0),
                        owner_token TEXT NOT NULL DEFAULT '',
                        lease_expires_at REAL NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        terminal_at REAL,
                        http_status INTEGER,
                        response_json TEXT,
                        response_hash TEXT,
                        progress_sequence INTEGER NOT NULL DEFAULT 0,
                        progress_at REAL,
                        progress_json TEXT,
                        progress_hash TEXT,
                        PRIMARY KEY (principal_digest, session_id, idempotency_key)
                    )
                    """
                )
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(chat_deliveries)").fetchall()
                }
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS chat_delivery_history ("
                    "turn_id TEXT PRIMARY KEY, capture_json TEXT NOT NULL, "
                    "capture_hash TEXT NOT NULL)"
                )
                progress_columns = {
                    "progress_sequence": "INTEGER NOT NULL DEFAULT 0",
                    "progress_at": "REAL",
                    "progress_json": "TEXT",
                    "progress_hash": "TEXT",
                }
                for column, declaration in progress_columns.items():
                    if column not in columns:
                        conn.execute(
                            f"ALTER TABLE chat_deliveries ADD COLUMN {column} {declaration}"
                        )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_deliveries_terminal "
                    "ON chat_deliveries(terminal_at, updated_at)"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO chat_delivery_meta(key, value) "
                    "VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                if schema_version == 1:
                    conn.execute(
                        "UPDATE chat_delivery_meta SET value=? WHERE key='schema_version'",
                        (str(_SCHEMA_VERSION),),
                    )
            finally:
                conn.close()
        except ChatDeliveryJournalError:
            raise
        except (OSError, ValueError) as exc:
            raise ChatDeliveryJournalUnavailable(
                "chat delivery journal could not initialize"
            ) from exc
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    @staticmethod
    def _validate_request_hash(request_hash: str) -> str:
        normalized = str(request_hash or "").strip().casefold()
        if not _HEX_64_RE.fullmatch(normalized):
            raise ValueError("request_hash must be lowercase SHA-256")
        return normalized

    @staticmethod
    def _row_identity(row: sqlite3.Row) -> DeliveryIdentity:
        return DeliveryIdentity(
            principal_digest=str(row["principal_digest"]),
            session_id=str(row["session_id"]),
            idempotency_key=str(row["idempotency_key"]),
        )

    def _decode_row(self, row: sqlite3.Row) -> DeliveryRecord:
        state = str(row["state"])
        if state not in _ALL_STATES:
            raise ChatDeliveryJournalCorruption("invalid chat delivery state")
        identity = self._row_identity(row)
        if not _HEX_64_RE.fullmatch(identity.principal_digest):
            raise ChatDeliveryJournalCorruption("invalid chat delivery principal digest")
        if (
            not identity.session_id
            or len(identity.session_id) > 240
            or any(ord(char) < 32 for char in identity.session_id)
            or not _IDEMPOTENCY_KEY_RE.fullmatch(identity.idempotency_key)
        ):
            raise ChatDeliveryJournalCorruption("invalid chat delivery identity")
        request_hash = str(row["request_hash"])
        turn_id = str(row["turn_id"])
        if not _HEX_64_RE.fullmatch(request_hash):
            raise ChatDeliveryJournalCorruption("invalid chat delivery request digest")
        if not re.fullmatch(r"[0-9a-f]{32}", turn_id):
            raise ChatDeliveryJournalCorruption("invalid chat delivery turn id")
        response: dict[str, Any] | None = None
        response_json = row["response_json"]
        response_hash = str(row["response_hash"] or "")
        if response_json is not None:
            raw = str(response_json).encode("utf-8", errors="strict")
            if not response_hash or not secrets.compare_digest(
                hashlib.sha256(raw).hexdigest(), response_hash
            ):
                raise ChatDeliveryJournalCorruption("chat delivery response digest mismatch")
            try:
                decoded = json.loads(raw)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ChatDeliveryJournalCorruption(
                    "chat delivery response is not valid JSON"
                ) from exc
            if not isinstance(decoded, dict):
                raise ChatDeliveryJournalCorruption("chat delivery response is not an object")
            response = decoded
        elif response_hash:
            raise ChatDeliveryJournalCorruption("chat delivery response digest has no payload")
        progress: dict[str, Any] | None = None
        progress_json = row["progress_json"]
        progress_hash = str(row["progress_hash"] or "")
        try:
            progress_sequence = int(row["progress_sequence"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ChatDeliveryJournalCorruption(
                "invalid chat delivery progress sequence"
            ) from exc
        progress_at_raw = row["progress_at"]
        progress_at = (
            _finite_timestamp(progress_at_raw, name="progress_at")
            if progress_at_raw is not None
            else None
        )
        if progress_json is not None:
            raw_progress = str(progress_json).encode("utf-8", errors="strict")
            if not progress_hash or not secrets.compare_digest(
                hashlib.sha256(raw_progress).hexdigest(), progress_hash
            ):
                raise ChatDeliveryJournalCorruption("chat delivery progress digest mismatch")
            try:
                decoded_progress = json.loads(raw_progress)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ChatDeliveryJournalCorruption(
                    "chat delivery progress is not valid JSON"
                ) from exc
            if not isinstance(decoded_progress, dict):
                raise ChatDeliveryJournalCorruption(
                    "chat delivery progress is not an object"
                )
            try:
                decoded_sequence = int(decoded_progress.get("sequence", -1))
                decoded_observed_at = float(decoded_progress.get("observed_at", -1))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ChatDeliveryJournalCorruption(
                    "chat delivery progress contains invalid numeric fields"
                ) from exc
            progress_message = str(decoded_progress.get("message") or "").strip()
            if (
                decoded_progress.get("schema") != "aura.chat.delivery.progress.v1"
                or decoded_sequence != progress_sequence
                or not _PROGRESS_PHASE_RE.fullmatch(
                    str(decoded_progress.get("phase") or "")
                )
                or not progress_message
                or len(progress_message) > 600
                or decoded_observed_at != progress_at
                or not isinstance(decoded_progress.get("details"), dict)
            ):
                raise ChatDeliveryJournalCorruption(
                    "chat delivery progress failed contract validation"
                )
            progress = decoded_progress
        elif progress_hash or progress_sequence != 0 or progress_at is not None:
            raise ChatDeliveryJournalCorruption(
                "chat delivery progress metadata has no payload"
            )
        try:
            generation = int(row["generation"])
            attempts = int(row["attempts"])
            http_status = int(row["http_status"]) if row["http_status"] is not None else None
        except (TypeError, ValueError, OverflowError) as exc:
            raise ChatDeliveryJournalCorruption("invalid chat delivery numeric field") from exc
        created_at = _finite_timestamp(row["created_at"], name="created_at")
        updated_at = _finite_timestamp(row["updated_at"], name="updated_at")
        lease_expires_at = _finite_timestamp(
            row["lease_expires_at"],
            name="lease_expires_at",
        )
        terminal_at_raw = row["terminal_at"]
        terminal_at = (
            _finite_timestamp(terminal_at_raw, name="terminal_at")
            if terminal_at_raw is not None
            else None
        )
        if generation < 0 or attempts < 0 or updated_at < created_at:
            raise ChatDeliveryJournalCorruption("invalid chat delivery chronology")
        if progress_at is not None and (
            progress_at < created_at or progress_at > updated_at + _CLOCK_SKEW_TOLERANCE_S
        ):
            raise ChatDeliveryJournalCorruption(
                "invalid chat delivery progress chronology"
            )
        terminal = state in _TERMINAL_STATES
        if terminal and (terminal_at is None or response is None):
            raise ChatDeliveryJournalCorruption("terminal chat delivery has no terminal receipt")
        if not terminal and (terminal_at is not None or response is not None):
            raise ChatDeliveryJournalCorruption("pending chat delivery contains terminal data")
        if terminal and (http_status is None or not 100 <= http_status <= 599):
            raise ChatDeliveryJournalCorruption("invalid chat delivery HTTP status")
        if not terminal and http_status is not None:
            raise ChatDeliveryJournalCorruption("pending chat delivery contains an HTTP status")
        owner_token = str(row["owner_token"] or "")
        if state == DeliveryState.RUNNING.value:
            if (
                generation < 1
                or attempts < 1
                or not _HEX_64_RE.fullmatch(owner_token)
                or lease_expires_at < updated_at
            ):
                raise ChatDeliveryJournalCorruption("invalid running chat delivery lease")
        elif owner_token or lease_expires_at != 0:
            raise ChatDeliveryJournalCorruption(
                "non-running chat delivery contains an execution lease"
            )
        if terminal:
            assert terminal_at is not None
            if terminal_at < created_at or terminal_at != updated_at:
                raise ChatDeliveryJournalCorruption("invalid terminal chat delivery chronology")
        return DeliveryRecord(
            identity=identity,
            request_hash=request_hash,
            turn_id=turn_id,
            state=DeliveryState(state),
            generation=generation,
            attempts=attempts,
            created_at=created_at,
            updated_at=updated_at,
            lease_expires_at=lease_expires_at,
            terminal_at=terminal_at,
            http_status=http_status,
            response=response,
            progress_sequence=progress_sequence,
            progress_at=progress_at,
            progress=progress,
        )

    def _compact_if_due_locked(
        self,
        conn: sqlite3.Connection,
        now: float,
    ) -> None:
        row = conn.execute(
            "SELECT value FROM chat_delivery_meta WHERE key='last_compaction_at'"
        ).fetchone()
        try:
            last_compaction = float(row[0]) if row is not None else 0.0
        except (TypeError, ValueError) as exc:
            raise ChatDeliveryJournalCorruption(
                "invalid chat delivery compaction checkpoint"
            ) from exc
        if not math.isfinite(last_compaction) or last_compaction < 0:
            raise ChatDeliveryJournalCorruption("invalid chat delivery compaction checkpoint")
        if last_compaction > now + _CLOCK_SKEW_TOLERANCE_S:
            raise ChatDeliveryJournalCorruption(
                "chat delivery compaction checkpoint is in the future"
            )
        if now - last_compaction < 60.0:
            return
        self._compact_locked(conn, now)
        conn.execute(
            "INSERT INTO chat_delivery_meta(key, value) VALUES('last_compaction_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (repr(now),),
        )

    @staticmethod
    def _select_row(
        conn: sqlite3.Connection,
        identity: DeliveryIdentity,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM chat_deliveries
            WHERE principal_digest=? AND session_id=? AND idempotency_key=?
            """,
            (
                identity.principal_digest,
                identity.session_id,
                identity.idempotency_key,
            ),
        ).fetchone()

    def _fence_stale_running_locked(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        record: DeliveryRecord,
        now: float,
    ) -> sqlite3.Row:
        if record.state is not DeliveryState.RUNNING:
            return row
        if record.lease_expires_at > now:
            return row
        ambiguous_payload, ambiguous_hash = _canonical_response(
            {
                "response": (
                    "The prior chat owner disappeared after execution began. "
                    "Its external effects cannot be proven absent, so automatic "
                    "re-execution is fenced."
                ),
                "status": "delivery_ambiguous",
                "response_confidence": "failed",
                "turn_id": record.turn_id,
                "idempotency_key": record.identity.idempotency_key,
                "delivery_state": DeliveryState.AMBIGUOUS.value,
                "delivery_generation": record.generation,
                "delivery_replayed": False,
            }
        )
        changed = conn.execute(
            """
            UPDATE chat_deliveries
            SET state='ambiguous', owner_token='', lease_expires_at=0,
                updated_at=?, terminal_at=?, http_status=409,
                response_json=?, response_hash=?
            WHERE principal_digest=? AND session_id=? AND idempotency_key=?
              AND request_hash=? AND state='running'
              AND generation=? AND owner_token=? AND lease_expires_at<=?
            """,
            (
                now,
                now,
                ambiguous_payload,
                ambiguous_hash,
                record.identity.principal_digest,
                record.identity.session_id,
                record.identity.idempotency_key,
                record.request_hash,
                record.generation,
                str(row["owner_token"] or ""),
                now,
            ),
        ).rowcount
        if changed != 1:
            raise ChatDeliveryJournalUnavailable("chat delivery stale-owner fencing raced")
        updated = self._select_row(conn, record.identity)
        if updated is None:
            raise ChatDeliveryJournalCorruption(
                "chat delivery disappeared during stale-owner fencing"
            )
        return updated

    def _compact_locked(self, conn: sqlite3.Connection, now: float) -> dict[str, int]:
        ambiguous_payload, ambiguous_hash = _canonical_response(
            {
                "response": (
                    "The prior chat execution ended without an authoritative terminal "
                    "receipt. Its idempotency key is fenced from automatic replay."
                ),
                "status": "delivery_ambiguous",
                "response_confidence": "failed",
            }
        )
        abandoned = conn.execute(
            """
            UPDATE chat_deliveries
            SET state='ambiguous', owner_token='', lease_expires_at=0,
                updated_at=?, terminal_at=?, http_status=409,
                response_json=?, response_hash=?
            WHERE (state='running' AND lease_expires_at <= ?)
               OR (state='queued' AND updated_at < ?)
            """,
            (
                now,
                now,
                ambiguous_payload,
                ambiguous_hash,
                now,
                now - self.abandon_after_s,
            ),
        ).rowcount
        expired = conn.execute(
            "DELETE FROM chat_deliveries WHERE terminal_at IS NOT NULL AND terminal_at < ? "
            "AND turn_id NOT IN (SELECT turn_id FROM chat_delivery_history)",
            (now - self.retention_s,),
        ).rowcount
        total = int(conn.execute("SELECT COUNT(*) FROM chat_deliveries").fetchone()[0])
        overflow = max(0, total - self.max_rows)
        evicted = 0
        if overflow:
            evicted = conn.execute(
                """
                DELETE FROM chat_deliveries WHERE rowid IN (
                    SELECT rowid FROM chat_deliveries
                    WHERE terminal_at IS NOT NULL
                      AND turn_id NOT IN (SELECT turn_id FROM chat_delivery_history)
                    ORDER BY terminal_at ASC, rowid ASC LIMIT ?
                )
                """,
                (overflow,),
            ).rowcount
        return {
            "abandoned": int(abandoned),
            "expired": int(expired),
            "evicted": int(evicted),
        }

    def _reserve_sync(
        self,
        identity: DeliveryIdentity,
        request_hash: str,
        *,
        now: float,
        approval_resume_token: str,
    ) -> DeliveryAdmission:
        request_hash = self._validate_request_hash(request_hash)
        if not math.isfinite(now) or now < 0:
            raise ValueError("chat delivery reservation time must be finite")
        owner_token = secrets.token_hex(32)
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                self._compact_if_due_locked(conn, now)
                row = self._select_row(conn, identity)
                if row is None:
                    row_count = int(
                        conn.execute("SELECT COUNT(*) FROM chat_deliveries").fetchone()[0]
                    )
                    if row_count >= self.max_rows:
                        needed = row_count - self.max_rows + 1
                        conn.execute(
                            """
                            DELETE FROM chat_deliveries WHERE rowid IN (
                                SELECT rowid FROM chat_deliveries
                                WHERE terminal_at IS NOT NULL
                                ORDER BY terminal_at ASC, rowid ASC LIMIT ?
                            )
                            """,
                            (needed,),
                        )
                        row_count = int(
                            conn.execute("SELECT COUNT(*) FROM chat_deliveries").fetchone()[0]
                        )
                    if row_count >= self.max_rows:
                        raise ChatDeliveryJournalUnavailable(
                            "chat delivery journal reached its active retention limit"
                        )
                    turn_id = uuid.uuid4().hex
                    conn.execute(
                        """
                        INSERT INTO chat_deliveries(
                            principal_digest, session_id, idempotency_key,
                            request_hash, turn_id, state, generation, attempts,
                            owner_token, lease_expires_at, created_at, updated_at
                        ) VALUES(?,?,?,?,?,'queued',0,0,'',0,?,?)
                        """,
                        (
                            identity.principal_digest,
                            identity.session_id,
                            identity.idempotency_key,
                            request_hash,
                            turn_id,
                            now,
                            now,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE chat_deliveries
                        SET state='running', generation=1, attempts=1,
                            owner_token=?, lease_expires_at=?, updated_at=?
                        WHERE principal_digest=? AND session_id=? AND idempotency_key=?
                        """,
                        (
                            owner_token,
                            now + self.stale_after_s,
                            now,
                            identity.principal_digest,
                            identity.session_id,
                            identity.idempotency_key,
                        ),
                    )
                    row = self._select_row(conn, identity)
                    conn.commit()
                    assert row is not None
                    return DeliveryAdmission(
                        AdmissionKind.EXECUTE,
                        self._decode_row(row),
                        owner_token,
                    )

                record = self._decode_row(row)
                if not secrets.compare_digest(record.request_hash, request_hash):
                    conn.commit()
                    return DeliveryAdmission(AdmissionKind.MISMATCH, record)

                approval_resume = (
                    record.state is DeliveryState.AWAITING_APPROVAL
                    and approval_resume_token
                    and secrets.compare_digest(
                        approval_resume_token,
                        record.turn_id,
                    )
                )
                stale_running = (
                    record.state in {DeliveryState.QUEUED, DeliveryState.RUNNING}
                    and record.lease_expires_at <= now
                )
                if stale_running and record.state is DeliveryState.RUNNING:
                    row = self._fence_stale_running_locked(
                        conn,
                        row,
                        record,
                        now,
                    )
                    conn.commit()
                    return DeliveryAdmission(
                        AdmissionKind.REPLAY,
                        self._decode_row(row),
                    )
                if record.state is DeliveryState.QUEUED or approval_resume:
                    next_generation = record.generation + 1
                    conn.execute(
                        """
                        UPDATE chat_deliveries
                        SET state='running', generation=?, attempts=attempts+1,
                            owner_token=?, lease_expires_at=?, updated_at=?,
                            terminal_at=NULL, http_status=NULL,
                            response_json=NULL, response_hash=NULL,
                            progress_sequence=0, progress_at=NULL,
                            progress_json=NULL, progress_hash=NULL
                        WHERE principal_digest=? AND session_id=? AND idempotency_key=?
                          AND generation=?
                        """,
                        (
                            next_generation,
                            owner_token,
                            now + self.stale_after_s,
                            now,
                            identity.principal_digest,
                            identity.session_id,
                            identity.idempotency_key,
                            record.generation,
                        ),
                    )
                    row = self._select_row(conn, identity)
                    conn.commit()
                    assert row is not None
                    return DeliveryAdmission(
                        AdmissionKind.EXECUTE,
                        self._decode_row(row),
                        owner_token,
                    )
                conn.commit()
                if record.terminal:
                    return DeliveryAdmission(AdmissionKind.REPLAY, record)
                return DeliveryAdmission(AdmissionKind.PENDING, record)
            except BaseException:  # noqa: BLE001 - transaction must roll back on interruption
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except ChatDeliveryJournalError:
            raise
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def reserve(
        self,
        identity: DeliveryIdentity,
        request_hash: str,
        *,
        wait_timeout_s: float = 120.0,
        approval_resume_token: str = "",
    ) -> DeliveryAdmission:
        """Reserve execution, wait for the owner, replay, or reject mismatch."""
        deadline = time.monotonic() + max(0.0, float(wait_timeout_s))
        admission = await asyncio.to_thread(
            self._reserve_sync,
            identity,
            request_hash,
            now=time.time(),
            approval_resume_token=str(approval_resume_token or ""),
        )
        while admission.kind is AdmissionKind.PENDING:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return admission
            await asyncio.sleep(min(self.poll_interval_s, remaining))
            admission = await asyncio.to_thread(
                self._reserve_sync,
                identity,
                request_hash,
                now=time.time(),
                approval_resume_token=str(approval_resume_token or ""),
            )
        return admission

    def _renew_sync(self, admission: DeliveryAdmission, now: float) -> bool:
        if not math.isfinite(now) or now < 0:
            raise ValueError("chat delivery renewal time must be finite")
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                changed = conn.execute(
                    """
                    UPDATE chat_deliveries
                    SET lease_expires_at=?, updated_at=?
                    WHERE principal_digest=? AND session_id=? AND idempotency_key=?
                      AND request_hash=? AND state='running'
                      AND generation=? AND owner_token=?
                    """,
                    (
                        now + self.stale_after_s,
                        now,
                        admission.record.identity.principal_digest,
                        admission.record.identity.session_id,
                        admission.record.identity.idempotency_key,
                        admission.record.request_hash,
                        admission.record.generation,
                        admission.owner_token,
                    ),
                ).rowcount
                conn.commit()
                return changed == 1
            finally:
                conn.close()
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def renew(self, admission: DeliveryAdmission) -> bool:
        if not admission.may_execute:
            return False
        return await asyncio.to_thread(self._renew_sync, admission, time.time())

    def _publish_progress_sync(
        self,
        admission: DeliveryAdmission,
        phase: str,
        message: str,
        details: dict[str, Any] | None,
        now: float,
    ) -> DeliveryRecord:
        if not admission.may_execute:
            raise ValueError("only an execution owner may publish delivery progress")
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = self._select_row(conn, admission.record.identity)
                if row is None:
                    conn.rollback()
                    raise ChatDeliveryFenceLost("chat delivery progress owner disappeared")
                current = self._decode_row(row)
                next_sequence = current.progress_sequence + 1
                progress_json, progress_hash, _ = _canonical_progress(
                    phase=phase,
                    message=message,
                    sequence=next_sequence,
                    observed_at=now,
                    details=details,
                )
                changed = conn.execute(
                    """
                    UPDATE chat_deliveries
                    SET progress_sequence=?, progress_at=?, progress_json=?,
                        progress_hash=?, updated_at=?, lease_expires_at=?
                    WHERE principal_digest=? AND session_id=? AND idempotency_key=?
                      AND request_hash=? AND state='running'
                      AND generation=? AND owner_token=?
                    """,
                    (
                        next_sequence,
                        now,
                        progress_json,
                        progress_hash,
                        now,
                        now + self.stale_after_s,
                        admission.record.identity.principal_digest,
                        admission.record.identity.session_id,
                        admission.record.identity.idempotency_key,
                        admission.record.request_hash,
                        admission.record.generation,
                        admission.owner_token,
                    ),
                ).rowcount
                updated = self._select_row(conn, admission.record.identity)
                if changed != 1 or updated is None:
                    conn.rollback()
                    raise ChatDeliveryFenceLost(
                        "chat delivery progress fence is no longer current"
                    )
                conn.commit()
                return self._decode_row(updated)
            except BaseException:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except ChatDeliveryJournalError:
            raise
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def publish_progress(
        self,
        admission: DeliveryAdmission,
        *,
        phase: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> DeliveryRecord:
        """Publish owner-fenced, durable progress and renew the execution lease."""

        return await asyncio.to_thread(
            self._publish_progress_sync,
            admission,
            phase,
            message,
            details,
            time.time(),
        )

    def _finalize_sync(
        self,
        admission: DeliveryAdmission,
        state: DeliveryState,
        http_status: int,
        response: dict[str, Any],
        now: float,
        history_capture: dict[str, Any] | None = None,
    ) -> DeliveryRecord:
        if not admission.may_execute:
            raise ValueError("only an execution owner may finalize a delivery")
        if state.value not in _TERMINAL_STATES:
            raise ValueError("chat delivery final state must be terminal")
        if isinstance(http_status, bool) or not 100 <= int(http_status) <= 599:
            raise ValueError("chat delivery HTTP status is invalid")
        if not math.isfinite(now) or now < 0:
            raise ValueError("chat delivery finalization time must be finite")
        response_json, response_hash = _canonical_response(response)
        capture = _canonical_response(history_capture) if history_capture else None
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                changed = conn.execute(
                    """
                    UPDATE chat_deliveries
                    SET state=?, owner_token='', lease_expires_at=0,
                        updated_at=?, terminal_at=?, http_status=?,
                        response_json=?, response_hash=?
                    WHERE principal_digest=? AND session_id=? AND idempotency_key=?
                      AND request_hash=? AND state='running'
                      AND generation=? AND owner_token=?
                    """,
                    (
                        state.value,
                        now,
                        now,
                        int(http_status),
                        response_json,
                        response_hash,
                        admission.record.identity.principal_digest,
                        admission.record.identity.session_id,
                        admission.record.identity.idempotency_key,
                        admission.record.request_hash,
                        admission.record.generation,
                        admission.owner_token,
                    ),
                ).rowcount
                row = self._select_row(conn, admission.record.identity)
                if changed != 1 or row is None:
                    conn.rollback()
                    raise ChatDeliveryFenceLost(
                        "chat delivery execution fence is no longer current"
                    )
                if capture is not None:
                    conn.execute(
                        "INSERT INTO chat_delivery_history(turn_id,capture_json,capture_hash) "
                        "VALUES(?,?,?)",
                        (admission.record.turn_id, *capture),
                    )
                conn.commit()
                return self._decode_row(row)
            except BaseException:  # noqa: BLE001 - transaction must roll back on interruption
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except ChatDeliveryJournalError:
            raise
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def finalize(
        self,
        admission: DeliveryAdmission,
        *,
        state: DeliveryState,
        http_status: int,
        response: dict[str, Any],
        history_capture: dict[str, Any] | None = None,
    ) -> DeliveryRecord:
        return await asyncio.to_thread(
            self._finalize_sync,
            admission,
            state,
            http_status,
            response,
            time.time(),
            history_capture,
        )

    def _pending_history_sync(self, limit: int) -> list[tuple[DeliveryRecord, dict[str, Any]]]:
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT d.*,h.capture_json,h.capture_hash FROM chat_deliveries d "
                    "JOIN chat_delivery_history h ON h.turn_id=d.turn_id "
                    "ORDER BY d.terminal_at,d.turn_id LIMIT ?",
                    (max(1, min(int(limit), 100)),),
                ).fetchall()
                pending = []
                for row in rows:
                    record = self._decode_row(row)
                    raw = str(row["capture_json"])
                    if not secrets.compare_digest(
                        hashlib.sha256(raw.encode("utf-8")).hexdigest(), row["capture_hash"]
                    ):
                        raise ChatDeliveryJournalCorruption("terminal history capture hash mismatch")
                    try:
                        capture = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ChatDeliveryJournalCorruption("invalid terminal history JSON") from exc
                    if not record.terminal or not isinstance(capture, dict):
                        raise ChatDeliveryJournalCorruption("invalid terminal history capture")
                    pending.append((record, capture))
                return pending
            finally:
                conn.close()
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def pending_history(self, *, limit: int = 20) -> list[tuple[DeliveryRecord, dict[str, Any]]]:
        """Read private transcript obligations without changing public receipts."""
        return await asyncio.to_thread(self._pending_history_sync, limit)

    def _acknowledge_history_sync(self, record: DeliveryRecord) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = self._select_row(conn, record.identity)
                current = self._decode_row(row) if row is not None else None
                if current is None or not current.terminal or current.turn_id != record.turn_id:
                    raise ChatDeliveryFenceLost("terminal history identity changed")
                if current.response != record.response:
                    raise ChatDeliveryFenceLost("terminal history response changed")
                conn.execute("DELETE FROM chat_delivery_history WHERE turn_id=?", (record.turn_id,))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def acknowledge_history(self, record: DeliveryRecord) -> None:
        """Retire an obligation only after its transcript commit succeeds."""
        await asyncio.to_thread(self._acknowledge_history_sync, record)

    def _get_sync(self, identity: DeliveryIdentity) -> DeliveryRecord | None:
        try:
            conn = self._connect()
            try:
                row = self._select_row(conn, identity)
                if row is None:
                    return None
                record = self._decode_row(row)
                now = time.time()
                if record.state is not DeliveryState.RUNNING or record.lease_expires_at > now:
                    return record
                conn.execute("BEGIN IMMEDIATE")
                current_row = self._select_row(conn, identity)
                if current_row is None:
                    conn.rollback()
                    raise ChatDeliveryJournalCorruption(
                        "chat delivery disappeared during status reconciliation"
                    )
                current = self._decode_row(current_row)
                current_row = self._fence_stale_running_locked(
                    conn,
                    current_row,
                    current,
                    now,
                )
                conn.commit()
                return self._decode_row(current_row)
            except BaseException:  # noqa: BLE001 - transaction must roll back on interruption
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except ChatDeliveryJournalError:
            raise
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def get(self, identity: DeliveryIdentity) -> DeliveryRecord | None:
        return await asyncio.to_thread(self._get_sync, identity)

    def _compact_sync(self, now: float) -> dict[str, int]:
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                result = self._compact_locked(conn, now)
                conn.commit()
                return result
            except BaseException:  # noqa: BLE001 - transaction must roll back on interruption
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except ChatDeliveryJournalError:
            raise
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)

    async def compact(self) -> dict[str, int]:
        return await asyncio.to_thread(self._compact_sync, time.time())


_JOURNAL_CACHE: dict[Path, ChatDeliveryJournal] = {}
_JOURNAL_CACHE_LOCK = threading.Lock()


def get_chat_delivery_journal() -> ChatDeliveryJournal:
    path = Path(os.path.abspath(default_chat_delivery_db_path()))
    with _JOURNAL_CACHE_LOCK:
        journal = _JOURNAL_CACHE.get(path)
        if journal is None:
            journal = ChatDeliveryJournal(path)
            _JOURNAL_CACHE[path] = journal
        return journal


def reset_chat_delivery_journals_for_test() -> None:
    with _JOURNAL_CACHE_LOCK:
        _JOURNAL_CACHE.clear()


__all__ = [
    "AdmissionKind",
    "ChatDeliveryFenceLost",
    "ChatDeliveryJournal",
    "ChatDeliveryJournalCorruption",
    "ChatDeliveryJournalError",
    "ChatDeliveryJournalUnavailable",
    "DeliveryAdmission",
    "DeliveryIdentity",
    "DeliveryRecord",
    "DeliveryState",
    "canonical_request_hash",
    "default_chat_delivery_db_path",
    "get_chat_delivery_journal",
    "reset_chat_delivery_journals_for_test",
]
