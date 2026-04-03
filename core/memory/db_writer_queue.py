"""core/memory/db_writer_queue.py — Serialized SQLite Writer

Prevents 'database is locked' contention by routing ALL SQLite writes
through a single background thread with a queue. Reads remain direct.

Usage:
    from core.memory.db_writer_queue import get_db_writer
    writer = get_db_writer()
    writer.execute("INSERT INTO knowledge ...", (val1, val2))
    rows = writer.fetchall("SELECT * FROM knowledge WHERE ...", (val,))
"""
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.DBWriter")

_SENTINEL = object()  # Poison pill for shutdown


class _WriteRequest:
    """A queued write operation."""
    __slots__ = ("db_path", "sql", "params", "future")

    def __init__(self, db_path: str, sql: str, params: tuple, future: asyncio.Future):
        self.db_path = db_path
        self.sql = sql
        self.params = params
        self.future = future


class SerializedDBWriter:
    """Single-threaded SQLite writer with async-safe interface.

    All writes go through one thread to prevent 'database is locked'.
    Reads bypass the queue for speed but use shared connection pool.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        # ISSUE 27 fix: Use thread-local for writer connection and reader connections
        self._local = threading.local()
        self._writer_conns: Dict[str, sqlite3.Connection] = {}
        self._lock = threading.Lock()
        self._checkpoint_every = 500
        self._writes_since_checkpoint: Dict[str, int] = {}
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="db-writer")
        self._thread.start()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        logger.info("📝 SerializedDBWriter started")

    def _get_conn(self, db_path: str) -> sqlite3.Connection:
        """ISSUE 27 fix: Thread-local connection fetching to avoid cross-thread collision."""
        if not hasattr(self._local, "connections"):
            self._local.connections = {}
        
        if db_path not in self._local.connections:
            from core.memory.db_config import configure_connection
            conn = configure_connection(db_path)
            conn.row_factory = sqlite3.Row
            self._local.connections[db_path] = conn
        return self._local.connections[db_path]

    def _get_writer_conn(self, db_path: str) -> sqlite3.Connection:
        with self._lock:
            conn = self._writer_conns.get(db_path)
            if conn is None:
                from core.memory.db_config import configure_connection

                conn = configure_connection(db_path)
                self._writer_conns[db_path] = conn
                self._writes_since_checkpoint[db_path] = 0
            return conn

    def _drain_batch(self, first_item: Any, *, max_items: int = 50, window_s: float = 0.05) -> Tuple[List["_WriteRequest"], bool]:
        batch: List[_WriteRequest] = []
        stop_requested = False

        if first_item is _SENTINEL:
            return batch, True

        batch.append(first_item)
        deadline = time.monotonic() + window_s
        while len(batch) < max_items:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout <= 0.0:
                break
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if item is _SENTINEL:
                stop_requested = True
                break
            batch.append(item)
        return batch, stop_requested

    def _maybe_checkpoint(self, db_path: str, conn: sqlite3.Connection) -> None:
        writes = int(self._writes_since_checkpoint.get(db_path, 0) or 0)
        if writes < self._checkpoint_every:
            return
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.commit()
            self._writes_since_checkpoint[db_path] = 0
        except Exception as exc:
            logger.warning("DBWriter checkpoint failed for %s: %s", db_path, exc)

    def _writer_loop(self):
        """Background thread: processes queued writes in short micro-batches."""
        while True:
            try:
                item = self._queue.get()
                batch, stop_requested = self._drain_batch(item)
                if not batch and stop_requested:
                    break

                grouped: Dict[str, List[_WriteRequest]] = {}
                for req in batch:
                    grouped.setdefault(req.db_path, []).append(req)

                for db_path, requests in grouped.items():
                    conn = self._get_writer_conn(db_path)
                    results: List[Tuple[_WriteRequest, Dict[str, Any]]] = []
                    try:
                        with conn:
                            for req in requests:
                                cursor = conn.execute(req.sql, req.params)
                                results.append(
                                    (
                                        req,
                                        {"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid},
                                    )
                                )
                                self._writes_since_checkpoint[db_path] = int(
                                    self._writes_since_checkpoint.get(db_path, 0) or 0
                                ) + 1

                        self._maybe_checkpoint(db_path, conn)

                        try:
                            from core.container import ServiceContainer

                            mycelium = ServiceContainer.get("mycelial_network", default=None)
                            if mycelium:
                                h = mycelium.get_hypha("core", "memory")
                                if h:
                                    h.pulse(success=True)
                        except Exception as e:
                            capture_and_log(e, {'module': __name__})

                        for req, result in results:
                            if self._loop and not req.future.done():
                                self._loop.call_soon_threadsafe(req.future.set_result, result)
                    except Exception as e:
                        logger.error(
                            "DBWriter error on %s sql=%s params=%s: %s",
                            db_path,
                            requests[0].sql if requests else "",
                            requests[0].params if requests else (),
                            e,
                        )
                        for req in requests:
                            if self._loop and not req.future.done():
                                self._loop.call_soon_threadsafe(req.future.set_exception, e)

                if stop_requested:
                    break
            except Exception as e:
                logger.error("DBWriter loop error: %s", e)

    async def execute(self, db_path: str, sql: str, params: tuple = ()) -> Dict[str, Any]:
        """Queue a write and await its completion."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        future = self._loop.create_future()
        self._queue.put(_WriteRequest(str(db_path), sql, params, future))
        return await future

    def execute_sync(self, db_path: str, sql: str, params: tuple = ()) -> Dict[str, Any]:
        """Synchronous write (for use from non-async contexts)."""
        conn = self._get_conn(str(db_path))
        cursor = conn.execute(sql, params)
        conn.commit()
        return {"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid}

    def fetchall(self, db_path: str, sql: str, params: tuple = ()) -> List[Any]:
        """Direct read (bypasses queue for speed)."""
        conn = self._get_conn(str(db_path))
        cursor = conn.execute(sql, params)
        return cursor.fetchall()

    def fetchone(self, db_path: str, sql: str, params: tuple = ()) -> Optional[Any]:
        """Direct single-row read."""
        conn = self._get_conn(str(db_path))
        cursor = conn.execute(sql, params)
        return cursor.fetchone()

    def shutdown(self):
        """Gracefully stop the writer thread."""
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=5)
        with self._lock:
            for conn in self._writer_conns.values():
                conn.close()
            self._writer_conns.clear()
        logger.info("📝 SerializedDBWriter shut down")

    def flush_and_checkpoint(self) -> None:
        """Force WAL checkpoint across all writer connections."""
        with self._lock:
            for db_path, conn in self._writer_conns.items():
                try:
                    conn.execute("PRAGMA wal_checkpoint(FULL)")
                    conn.commit()
                    self._writes_since_checkpoint[db_path] = 0
                except Exception as exc:
                    logger.warning("DBWriter forced checkpoint failed for %s: %s", db_path, exc)


# ── Singleton ──
_instance: Optional[SerializedDBWriter] = None
_instance_lock = threading.Lock()


def get_db_writer() -> SerializedDBWriter:
    """Get the global serialized DB writer."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SerializedDBWriter()
    return _instance
