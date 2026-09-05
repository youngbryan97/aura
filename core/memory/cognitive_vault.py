"""core/memory/cognitive_vault.py - Aura 3.0: Cognitive Vault.

Unified memory persistence layer with a serialized SQLite write queue.
The vault never builds SQL from untrusted identifiers and never lets a failed
background write wedge shutdown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.task_tracker import get_task_tracker
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Aura.CognitiveVault")

_VAULT_RECOVERABLE_ERRORS = (
    AttributeError,
    FileNotFoundError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)

_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "memories": ("topic", "content", "metadata"),
    "audit_log": ("event", "details"),
}


def _record_vault_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "cognitive_vault",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


@dataclass(frozen=True)
class VaultTransaction:
    """A single atomic write targeting the cognitive vault."""

    table: str
    data: dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, "timestamp", time.time())


class CognitiveVault:
    """Unified memory persistence layer with a dedicated async write queue."""

    def __init__(self, db_path: str = "~/.aura/vault.db"):
        self.db_path = os.path.expanduser(db_path)
        self._queue: asyncio.Queue[VaultTransaction] = asyncio.Queue(maxsize=1024)
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False
        self._failed_writes = 0
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    async def on_start_async(self) -> bool:
        """Initializes the database schema and starts the write worker."""
        try:
            await asyncio.to_thread(self._initialize_schema)
        except _VAULT_RECOVERABLE_ERRORS as exc:
            _record_vault_degradation(
                exc,
                action="failed cognitive vault startup closed before accepting writes",
                severity="critical",
                extra={"db_path": self.db_path},
            )
            raise

        self._running = True
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = get_task_tracker().create_task(
                self._write_worker(),
                name="CognitiveVault.Worker",
            )
        logger.info("CognitiveVault ONLINE. Unified write pipeline active.")
        return True

    async def on_stop_async(self) -> bool:
        """Flushes the queue and closes the database."""
        self._running = False
        if self._worker_task:
            await self._queue.join()
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=2.0)
            except (asyncio.CancelledError, TimeoutError) as _exc:
                logger.debug("Suppressed %s in core.memory.cognitive_vault: %s", type(_exc).__name__, _exc)
            finally:
                self._worker_task = None
        logger.info("CognitiveVault SHUTDOWN.")
        return self._failed_writes == 0

    async def commit(self, table: str, data: dict[str, Any]) -> bool:
        """Queue persistent storage, falling back to direct write under pressure."""
        try:
            tx = self._build_transaction(table, data)
        except ValueError as exc:
            _record_vault_degradation(
                exc,
                action="rejected invalid cognitive vault transaction before SQL execution",
                extra={"table": str(table)[:80]},
            )
            return False

        if not self._running:
            return await self._execute_direct_fallback(tx, reason="vault worker is not running")

        try:
            self._queue.put_nowait(tx)
            return True
        except asyncio.QueueFull as exc:
            _record_vault_degradation(
                exc,
                action="applied bounded backpressure after cognitive vault queue filled",
                extra={"table": tx.table, "queue_size": self._queue.qsize()},
            )

        try:
            await asyncio.wait_for(self._queue.put(tx), timeout=0.5)
            return True
        except TimeoutError:
            return await self._execute_direct_fallback(tx, reason="queue backpressure timeout")

    def _initialize_schema(self) -> None:
        """Sets up WAL mode and core tables."""
        with connecting(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT,
                    details TEXT,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
            conn.commit()

    async def _write_worker(self) -> None:
        """Background coroutine that handles serialized database writes."""
        while self._running or not self._queue.empty():
            tx = None
            try:
                tx = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await asyncio.to_thread(self._execute_tx, tx)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except _VAULT_RECOVERABLE_ERRORS as exc:
                self._failed_writes += 1
                _record_vault_degradation(
                    exc,
                    action="kept vault worker alive after a transaction failed",
                    severity="degraded",
                    extra={"table": tx.table if tx else None},
                )
                logger.error("Vault worker error: %s", exc)
            finally:
                if tx is not None:
                    self._queue.task_done()

    def _execute_tx(self, tx: VaultTransaction) -> None:
        """Low-level SQLite execution using allowlisted table and column names."""
        columns = _TABLE_COLUMNS[tx.table]
        payload = self._coerce_payload(tx, columns)
        insert_columns = [column for column in columns if column in payload]
        insert_columns.append("timestamp")
        placeholders = ", ".join("?" for _ in insert_columns)
        quoted_columns = ", ".join(insert_columns)
        values = [payload[column] for column in insert_columns if column != "timestamp"]
        values.append(tx.timestamp)
        query = f"INSERT INTO {tx.table} ({quoted_columns}) VALUES ({placeholders})"
        with connecting(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            conn.execute(query, values)
            conn.commit()

    async def _execute_direct_fallback(self, tx: VaultTransaction, *, reason: str) -> bool:
        try:
            await asyncio.to_thread(self._execute_tx, tx)
            _record_vault_degradation(
                RuntimeError(reason),
                action="persisted cognitive vault transaction through direct bounded fallback",
                extra={"table": tx.table},
            )
            return True
        except _VAULT_RECOVERABLE_ERRORS as exc:
            self._failed_writes += 1
            _record_vault_degradation(
                exc,
                action="reported cognitive vault write failure after queue and direct fallback failed",
                severity="degraded",
                extra={"table": tx.table, "reason": reason},
            )
            return False

    def _build_transaction(self, table: str, data: dict[str, Any]) -> VaultTransaction:
        normalized_table = str(table or "").strip()
        if normalized_table not in _TABLE_COLUMNS:
            raise ValueError(f"unsupported cognitive vault table: {normalized_table!r}")
        if not isinstance(data, dict) or not data:
            raise ValueError("cognitive vault transaction data must be a non-empty mapping")
        allowed = set(_TABLE_COLUMNS[normalized_table])
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unsupported cognitive vault columns: {', '.join(unknown)}")
        return VaultTransaction(table=normalized_table, data=dict(data))

    def _coerce_payload(self, tx: VaultTransaction, columns: tuple[str, ...]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for column in columns:
            if column not in tx.data:
                continue
            value = tx.data[column]
            if isinstance(value, (dict, list, tuple, set)):
                payload[column] = json.dumps(value, sort_keys=True, default=str)
            elif value is None:
                payload[column] = None
            else:
                payload[column] = str(value)
        if not payload:
            raise ValueError("cognitive vault transaction has no writable columns")
        return payload
