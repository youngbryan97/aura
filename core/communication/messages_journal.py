"""Durable, content-hiding delivery state for Aura's Messages transport."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.state_ownership import state_root

_SCHEMA_VERSION = 1
_ENDPOINT_RE = re.compile(r"^msg_[0-9a-f]{32}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,240}$")
_OUTBOUND_STATES = frozenset(
    {
        "queued",
        "sending",
        "accepted_unverified",
        "verified_local_history",
        "ambiguous",
        "failed_before_effect",
    }
)
_TERMINAL_OUTBOUND_STATES = frozenset(
    {
        "accepted_unverified",
        "verified_local_history",
        "ambiguous",
        "failed_before_effect",
    }
)
_INBOUND_STATES = frozenset({"processing", "retryable", "completed"})


class MessagesJournalError(RuntimeError):
    """The Messages delivery journal could not establish trustworthy state."""


class MessagesJournalCorruptionError(MessagesJournalError):
    """Existing journal state failed schema or integrity validation."""


@dataclass(frozen=True, slots=True)
class OutboundAdmission:
    idempotency_key: str
    endpoint_ref: str
    content_sha256: str
    state: str
    baseline_row_id: int | None
    observed_row_id: int | None
    attempts: int
    created_at: float
    updated_at: float
    error_code: str
    may_execute: bool

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_OUTBOUND_STATES

    def public_receipt(self) -> dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "endpoint_ref": self.endpoint_ref,
            "state": self.state,
            "attempts": self.attempts,
            "accepted": self.state
            in {"accepted_unverified", "verified_local_history"},
            "verified_local_history": self.state == "verified_local_history",
            "remote_delivery_verified": False,
            "remote_read_verified": False,
            "ambiguous": self.state == "ambiguous",
            "terminal": self.terminal,
            "updated_at": self.updated_at,
            **({"error_code": self.error_code} if self.error_code else {}),
        }


def default_messages_journal_path() -> Path:
    override = str(os.environ.get("AURA_MESSAGES_DELIVERY_DB") or "").strip()
    if override:
        return Path(override).expanduser()
    test_root = str(os.environ.get("AURA_TEST_RUNTIME_ROOT") or "").strip()
    if test_root:
        return Path(test_root).expanduser() / "messages_delivery.sqlite3"
    return state_root() / "data" / "messages_delivery.sqlite3"


def content_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="strict")).hexdigest()


def _validated_endpoint(value: Any) -> str:
    endpoint = str(value or "").strip()
    if not _ENDPOINT_RE.fullmatch(endpoint):
        raise ValueError("invalid Messages endpoint reference")
    return endpoint


def _validated_digest(value: Any, *, name: str) -> str:
    digest = str(value or "").strip().lower()
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError(f"invalid {name}")
    return digest


def _validated_idempotency_key(value: Any) -> str:
    key = str(value or "").strip()
    if not _IDEMPOTENCY_RE.fullmatch(key):
        raise ValueError("invalid Messages idempotency key")
    return key


class MessagesDeliveryJournal:
    """SQLite-backed at-most-once and replay authority for private messages."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_s: float = 3.0,
    ) -> None:
        self.db_path = Path(db_path or default_messages_journal_path()).expanduser()
        self._clock = clock
        self._busy_timeout_s = max(0.1, min(float(busy_timeout_s), 30.0))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self._busy_timeout_s,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_s * 1000)}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        return connection

    def _initialize(self) -> None:
        if self.db_path.is_symlink():
            raise MessagesJournalCorruptionError("Messages journal path must not be a symlink")
        existed = self.db_path.exists() and self.db_path.stat().st_size > 0
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "messages_delivery_journal.initialize",
            domain="file_write",
        ):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(
                self.db_path.parent,
                source="core.communication.messages_journal.initialize",
            )
            with gateway.open_owned_binary(
                self.db_path,
                mode="a+b",
                permissions=0o600,
                source="core.communication.messages_journal.initialize",
            ):
                pass
        stat = self.db_path.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise MessagesJournalCorruptionError(
                "Messages journal ownership or permissions are unsafe"
            )
        connection = self._connect()
        try:
            quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
            if quick_check is None or str(quick_check[0]).casefold() != "ok":
                raise MessagesJournalCorruptionError("Messages journal quick_check failed")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not str(row[0]).startswith("sqlite_")
            }
            if existed and tables and "messages_meta" not in tables:
                raise MessagesJournalCorruptionError(
                    "Existing Messages journal has no schema identity"
                )
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode is None or str(mode[0]).casefold() != "wal":
                raise MessagesJournalError("Messages journal could not enable WAL durability")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS messages_meta ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            version = connection.execute(
                "SELECT value FROM messages_meta WHERE key='schema_version'"
            ).fetchone()
            if version is not None and int(version[0]) != _SCHEMA_VERSION:
                raise MessagesJournalCorruptionError("Unsupported Messages journal schema")
            if existed and tables and version is None:
                raise MessagesJournalCorruptionError(
                    "Existing Messages journal has no schema version"
                )
            connection.execute(
                "INSERT OR IGNORE INTO messages_meta(key, value) VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages_cursors (
                    endpoint_ref TEXT PRIMARY KEY,
                    inbound_row_id INTEGER NOT NULL CHECK (inbound_row_id >= 0),
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages_inbound (
                    guid_sha256 TEXT PRIMARY KEY,
                    endpoint_ref TEXT NOT NULL,
                    source_row_id INTEGER NOT NULL CHECK (source_row_id > 0),
                    content_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('processing','retryable','completed')),
                    response_sha256 TEXT,
                    outcome_code TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(endpoint_ref, source_row_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages_outbound (
                    idempotency_key TEXT PRIMARY KEY,
                    endpoint_ref TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('queued','sending','accepted_unverified',
                                  'verified_local_history','ambiguous','failed_before_effect')
                    ),
                    baseline_row_id INTEGER,
                    observed_row_id INTEGER,
                    attempts INTEGER NOT NULL CHECK (attempts >= 0),
                    error_code TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_outbound_recent "
                "ON messages_outbound(endpoint_ref, updated_at)"
            )
            # A process loss after the effect was admitted cannot be replayed
            # safely. It becomes ambiguous until local history proves otherwise.
            now = float(self._clock())
            connection.execute(
                "UPDATE messages_outbound SET state='ambiguous', "
                "error_code='process_interrupted_after_effect_admission', updated_at=? "
                "WHERE state='sending'",
                (now,),
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise MessagesJournalCorruptionError(
                "Messages journal initialization failed"
            ) from exc
        finally:
            connection.close()

    async def prime_cursor(self, endpoint_ref: str, row_id: int) -> tuple[int, bool]:
        return await asyncio.to_thread(self._prime_cursor_sync, endpoint_ref, row_id)

    def _prime_cursor_sync(self, endpoint_ref: str, row_id: int) -> tuple[int, bool]:
        endpoint = _validated_endpoint(endpoint_ref)
        normalized_row = max(0, int(row_id))
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT inbound_row_id FROM messages_cursors WHERE endpoint_ref=?",
                (endpoint,),
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return int(existing[0]), False
            connection.execute(
                "INSERT INTO messages_cursors(endpoint_ref, inbound_row_id, updated_at) "
                "VALUES(?, ?, ?)",
                (endpoint, normalized_row, now),
            )
            connection.execute("COMMIT")
            return normalized_row, True
        except sqlite3.Error:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    async def cursor(self, endpoint_ref: str) -> int | None:
        return await asyncio.to_thread(self._cursor_sync, endpoint_ref)

    def _cursor_sync(self, endpoint_ref: str) -> int | None:
        endpoint = _validated_endpoint(endpoint_ref)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT inbound_row_id FROM messages_cursors WHERE endpoint_ref=?",
                (endpoint,),
            ).fetchone()
            return int(row[0]) if row is not None else None
        finally:
            connection.close()

    async def claim_inbound(
        self,
        *,
        endpoint_ref: str,
        source_row_id: int,
        guid_sha256: str,
        content_sha256: str,
    ) -> str:
        return await asyncio.to_thread(
            self._claim_inbound_sync,
            endpoint_ref,
            source_row_id,
            guid_sha256,
            content_sha256,
        )

    def _claim_inbound_sync(
        self,
        endpoint_ref: str,
        source_row_id: int,
        guid_sha256: str,
        content_sha256: str,
    ) -> str:
        endpoint = _validated_endpoint(endpoint_ref)
        guid_digest = _validated_digest(guid_sha256, name="inbound guid digest")
        body_digest = _validated_digest(content_sha256, name="inbound content digest")
        row_id = int(source_row_id)
        if row_id <= 0:
            raise ValueError("Messages source row id must be positive")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT endpoint_ref, source_row_id, content_sha256, state "
                "FROM messages_inbound WHERE guid_sha256=?",
                (guid_digest,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["endpoint_ref"]) != endpoint
                    or int(existing["source_row_id"]) != row_id
                    or str(existing["content_sha256"]) != body_digest
                ):
                    raise MessagesJournalCorruptionError(
                        "Inbound Messages identity changed after admission"
                    )
                state = str(existing["state"])
                if state == "retryable":
                    connection.execute(
                        "UPDATE messages_inbound SET state='processing', updated_at=? "
                        "WHERE guid_sha256=?",
                        (now, guid_digest),
                    )
                    state = "processing"
                connection.execute("COMMIT")
                return state
            connection.execute(
                "INSERT INTO messages_inbound("
                "guid_sha256, endpoint_ref, source_row_id, content_sha256, state, "
                "created_at, updated_at) VALUES(?, ?, ?, ?, 'processing', ?, ?)",
                (guid_digest, endpoint, row_id, body_digest, now, now),
            )
            connection.execute("COMMIT")
            return "processing"
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    async def mark_inbound_retryable(self, guid_sha256: str, outcome_code: str) -> None:
        await asyncio.to_thread(
            self._mark_inbound_sync,
            guid_sha256,
            "retryable",
            "",
            outcome_code,
            None,
            None,
        )

    async def complete_inbound(
        self,
        *,
        endpoint_ref: str,
        source_row_id: int,
        guid_sha256: str,
        response_sha256: str,
        outcome_code: str,
    ) -> None:
        await asyncio.to_thread(
            self._mark_inbound_sync,
            guid_sha256,
            "completed",
            response_sha256,
            outcome_code,
            endpoint_ref,
            source_row_id,
        )

    def _mark_inbound_sync(
        self,
        guid_sha256: str,
        state: str,
        response_sha256: str,
        outcome_code: str,
        endpoint_ref: str | None,
        source_row_id: int | None,
    ) -> None:
        guid_digest = _validated_digest(guid_sha256, name="inbound guid digest")
        if state not in _INBOUND_STATES:
            raise ValueError("invalid inbound Messages state")
        response_digest = (
            _validated_digest(response_sha256, name="response digest")
            if response_sha256
            else None
        )
        outcome = str(outcome_code or "unknown")[:120]
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE messages_inbound SET state=?, response_sha256=?, outcome_code=?, "
                "updated_at=? WHERE guid_sha256=?",
                (state, response_digest, outcome, now, guid_digest),
            )
            if updated.rowcount != 1:
                raise MessagesJournalCorruptionError("Inbound Messages receipt is missing")
            if state == "completed":
                endpoint = _validated_endpoint(endpoint_ref)
                row_id = int(source_row_id or 0)
                if row_id <= 0:
                    raise ValueError("completed inbound Messages row id must be positive")
                connection.execute(
                    "INSERT INTO messages_cursors(endpoint_ref, inbound_row_id, updated_at) "
                    "VALUES(?, ?, ?) ON CONFLICT(endpoint_ref) DO UPDATE SET "
                    "inbound_row_id=MAX(messages_cursors.inbound_row_id, excluded.inbound_row_id), "
                    "updated_at=excluded.updated_at",
                    (endpoint, row_id, now),
                )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    async def admit_outbound(
        self,
        *,
        idempotency_key: str,
        endpoint_ref: str,
        content_sha256: str,
        baseline_row_id: int | None,
    ) -> OutboundAdmission:
        return await asyncio.to_thread(
            self._admit_outbound_sync,
            idempotency_key,
            endpoint_ref,
            content_sha256,
            baseline_row_id,
        )

    def _admit_outbound_sync(
        self,
        idempotency_key: str,
        endpoint_ref: str,
        content_sha256: str,
        baseline_row_id: int | None,
    ) -> OutboundAdmission:
        key = _validated_idempotency_key(idempotency_key)
        endpoint = _validated_endpoint(endpoint_ref)
        body_digest = _validated_digest(content_sha256, name="outbound content digest")
        baseline = None if baseline_row_id is None else max(0, int(baseline_row_id))
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM messages_outbound WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            if row is not None:
                if (
                    str(row["endpoint_ref"]) != endpoint
                    or str(row["content_sha256"]) != body_digest
                ):
                    raise MessagesJournalCorruptionError(
                        "Messages idempotency key was reused for different content"
                    )
                admission = self._outbound_from_row(row, may_execute=str(row["state"]) == "queued")
                connection.execute("COMMIT")
                return admission
            connection.execute(
                "INSERT INTO messages_outbound("
                "idempotency_key, endpoint_ref, content_sha256, state, baseline_row_id, "
                "attempts, created_at, updated_at) VALUES(?, ?, ?, 'queued', ?, 0, ?, ?)",
                (key, endpoint, body_digest, baseline, now, now),
            )
            row = connection.execute(
                "SELECT * FROM messages_outbound WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise MessagesJournalCorruptionError(
                    "Messages outbound admission disappeared"
                )
            return self._outbound_from_row(row, may_execute=True)
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    async def lookup_outbound(
        self,
        *,
        idempotency_key: str,
        endpoint_ref: str,
        content_sha256: str,
    ) -> OutboundAdmission | None:
        return await asyncio.to_thread(
            self._lookup_outbound_sync,
            idempotency_key,
            endpoint_ref,
            content_sha256,
        )

    def _lookup_outbound_sync(
        self,
        idempotency_key: str,
        endpoint_ref: str,
        content_sha256: str,
    ) -> OutboundAdmission | None:
        key = _validated_idempotency_key(idempotency_key)
        endpoint = _validated_endpoint(endpoint_ref)
        body_digest = _validated_digest(content_sha256, name="outbound content digest")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM messages_outbound WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            if (
                str(row["endpoint_ref"]) != endpoint
                or str(row["content_sha256"]) != body_digest
            ):
                raise MessagesJournalCorruptionError(
                    "Messages idempotency key was reused for different content"
                )
            return self._outbound_from_row(
                row,
                may_execute=str(row["state"]) == "queued",
            )
        finally:
            connection.close()

    async def mark_outbound_sending(self, idempotency_key: str) -> OutboundAdmission:
        return await asyncio.to_thread(self._claim_outbound_sending_sync, idempotency_key)

    def _claim_outbound_sending_sync(self, idempotency_key: str) -> OutboundAdmission:
        key = _validated_idempotency_key(idempotency_key)
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM messages_outbound WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            if row is None:
                raise MessagesJournalCorruptionError("Messages outbound receipt is missing")
            if str(row["state"]) != "queued":
                connection.execute("COMMIT")
                return self._outbound_from_row(row, may_execute=False)
            updated = connection.execute(
                "UPDATE messages_outbound SET state='sending', attempts=attempts+1, "
                "error_code='', updated_at=? WHERE idempotency_key=? AND state='queued'",
                (now, key),
            )
            row = connection.execute(
                "SELECT * FROM messages_outbound WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise MessagesJournalCorruptionError(
                    "Messages outbound receipt disappeared"
                )
            return self._outbound_from_row(row, may_execute=updated.rowcount == 1)
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    async def mark_outbound_terminal(
        self,
        idempotency_key: str,
        *,
        state: str,
        observed_row_id: int | None = None,
        error_code: str = "",
    ) -> OutboundAdmission:
        if state not in _TERMINAL_OUTBOUND_STATES:
            raise ValueError("invalid terminal Messages outbound state")
        return await asyncio.to_thread(
            self._update_outbound_sync,
            idempotency_key,
            state,
            observed_row_id,
            error_code,
            False,
        )

    def _update_outbound_sync(
        self,
        idempotency_key: str,
        state: str,
        observed_row_id: int | None,
        error_code: str,
        increment_attempt: bool,
    ) -> OutboundAdmission:
        key = _validated_idempotency_key(idempotency_key)
        if state not in _OUTBOUND_STATES:
            raise ValueError("invalid Messages outbound state")
        observed = None if observed_row_id is None else max(0, int(observed_row_id))
        error = str(error_code or "")[:120]
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE messages_outbound SET state=?, observed_row_id=?, error_code=?, "
                "attempts=attempts+?, updated_at=? WHERE idempotency_key=?",
                (state, observed, error, 1 if increment_attempt else 0, now, key),
            )
            if updated.rowcount != 1:
                raise MessagesJournalCorruptionError("Messages outbound receipt is missing")
            row = connection.execute(
                "SELECT * FROM messages_outbound WHERE idempotency_key=?",
                (key,),
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise MessagesJournalCorruptionError(
                    "Messages outbound receipt disappeared"
                )
            return self._outbound_from_row(row, may_execute=False)
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    async def recent_outbound_attempts(self, endpoint_ref: str, *, since: float) -> int:
        return await asyncio.to_thread(self._recent_outbound_attempts_sync, endpoint_ref, since)

    def _recent_outbound_attempts_sync(self, endpoint_ref: str, since: float) -> int:
        endpoint = _validated_endpoint(endpoint_ref)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) FROM messages_outbound WHERE endpoint_ref=? AND updated_at>=? "
                "AND state IN ('sending','accepted_unverified','verified_local_history','ambiguous')",
                (endpoint, float(since)),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    @staticmethod
    def _outbound_from_row(row: sqlite3.Row, *, may_execute: bool) -> OutboundAdmission:
        state = str(row["state"])
        if state not in _OUTBOUND_STATES:
            raise MessagesJournalCorruptionError("Messages outbound state is invalid")
        return OutboundAdmission(
            idempotency_key=str(row["idempotency_key"]),
            endpoint_ref=str(row["endpoint_ref"]),
            content_sha256=str(row["content_sha256"]),
            state=state,
            baseline_row_id=(
                int(row["baseline_row_id"])
                if row["baseline_row_id"] is not None
                else None
            ),
            observed_row_id=(
                int(row["observed_row_id"])
                if row["observed_row_id"] is not None
                else None
            ),
            attempts=int(row["attempts"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            error_code=str(row["error_code"] or ""),
            may_execute=bool(may_execute),
        )


__all__ = [
    "MessagesDeliveryJournal",
    "MessagesJournalCorruptionError",
    "MessagesJournalError",
    "OutboundAdmission",
    "content_digest",
    "default_messages_journal_path",
]
