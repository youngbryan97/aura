"""Privacy-safe evidence of what live recall actually ranked and returned.

Every competitive recall contributes activation, rank, and candidate count.
Those are sufficient to fit a retrieval curve and contain no memory content,
keys, episode ids, query text, or user identifiers.

The hot path is an in-memory bounded append. Persistence is micro-batched by
one coalesced worker in Aura's shutdown-owned I/O pool, so SQLite never blocks
the event loop, a recall, or a caller thread. The durable store is independently
bounded and is what the standalone fitter reads; a process-local ring cannot
be evidence for a tool launched in another process.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import sqlite3
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Aura.Memory.RecallObservations")

__all__ = [
    "RecallObservation",
    "RecallObservationRing",
    "get_recall_observations",
    "peek_recall_observations",
    "record_ranking",
]

_RING_CAPACITY = 20_000
_PERSIST_BATCH_LIMIT = 2_048


def _default_db_path() -> Path:
    from core.config import config

    return Path(config.paths.memory_dir) / "recall_observations.db"


@dataclass(frozen=True, slots=True)
class RecallObservation:
    """One anonymous candidate's fate in one ranked recall."""

    activation: float
    rank: int
    candidates: int
    returned: bool
    recorded_at: float = 0.0


class RecallObservationRing:
    """Bounded live ring plus a bounded, process-readable evidence store."""

    def __init__(
        self,
        capacity: int = _RING_CAPACITY,
        *,
        db_path: str | Path | None = None,
        persistence: bool = True,
        background_persistence: bool = True,
    ) -> None:
        self._lock = checked_lock(f"recall_observations.state.{id(self):x}", rank=LockRank.LEAF)
        self._ring: deque[RecallObservation] = deque(maxlen=max(1, int(capacity)))
        self._pending: deque[RecallObservation] = deque(maxlen=max(1, int(capacity)))
        self._rankings = 0
        self._dropped_pending = 0
        self._flush_failures = 0
        self._schedule_failures = 0
        self._persisted_observations = 0
        self._inflight_observations = 0
        self._last_flush_at = 0.0
        self._flush_future: concurrent.futures.Future[int] | None = None
        self._flush_scheduled = False
        self._retry_not_before = 0.0
        self._persistence = bool(persistence)
        self._background_persistence = bool(background_persistence)
        self._db_path = Path(db_path) if db_path is not None else _default_db_path()

    def record(self, activations: Sequence[float], *, returned_count: int) -> int:
        """Append one ranked recall in memory; return valid observations added."""
        total = len(activations)
        if total < 2:
            return 0
        returned = max(0, min(total, int(returned_count)))
        now = time.time()
        observations: list[RecallObservation] = []
        for rank, activation in enumerate(activations):
            try:
                numeric = float(activation)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(numeric):
                observations.append(
                    RecallObservation(
                        numeric,
                        rank,
                        total,
                        rank < returned,
                        recorded_at=now,
                    )
                )
        if not observations:
            return 0
        with self._lock:
            self._rankings += 1
            self._ring.extend(observations)
            if self._persistence:
                overflow = max(0, len(self._pending) + len(observations) - self._pending.maxlen)
                self._dropped_pending += overflow
                self._pending.extend(observations)
        if self._persistence and self._background_persistence:
            self._schedule_flush()
        return len(observations)

    def _schedule_flush(self) -> None:
        """Own one coalesced flush in Aura's shutdown-owned I/O pool."""
        try:
            from core.runtime.executors import submit_blocking_io
        except (ImportError, AttributeError) as exc:
            self._note_schedule_failure(exc)
            return

        with self._lock:
            current = self._flush_future
            if self._flush_scheduled or (current is not None and not current.done()):
                return
            if time.monotonic() < self._retry_not_before:
                return
            # Reserve ownership before leaving the critical section. Without
            # this flag, two caller threads can both observe no Future and
            # enqueue two SQLite workers.
            self._flush_scheduled = True
        try:
            future = submit_blocking_io(
                self._flush_worker,
                label="recall-observations-flush",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            with self._lock:
                self._flush_scheduled = False
            self._note_schedule_failure(exc)
            return
        with self._lock:
            self._flush_future = future
        future.add_done_callback(self._flush_done)

    def _flush_worker(self) -> int:
        persisted = 0
        while True:
            with self._lock:
                batch = [
                    self._pending.popleft()
                    for _ in range(min(len(self._pending), _PERSIST_BATCH_LIMIT))
                ]
                self._inflight_observations = len(batch)
            if not batch:
                return persisted
            try:
                written = self._persist_batch(batch)
            except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
                self._restore_failed_batch(batch)
                self._note_flush_failure(exc)
                return persisted
            with self._lock:
                self._persisted_observations = max(self._persisted_observations, written)
                self._inflight_observations = 0
                self._last_flush_at = time.time()
                self._retry_not_before = 0.0
            persisted = written

    def _flush_done(self, future: concurrent.futures.Future[int]) -> None:
        cancelled = future.cancelled()
        try:
            exc = future.exception()
            if exc is not None:
                self._note_flush_failure(exc)
        except concurrent.futures.CancelledError:
            cancelled = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            self._note_flush_failure(exc)
        finally:
            with self._lock:
                if self._flush_future is future:
                    self._flush_future = None
                self._flush_scheduled = False
                pending = bool(self._pending)
            try:
                from core.runtime.task_ownership import runtime_shutdown_requested

                shutting_down = runtime_shutdown_requested()
            except (ImportError, AttributeError, RuntimeError):
                shutting_down = False
            if pending and not cancelled and not shutting_down:
                self._schedule_flush()

    def flush(self) -> int:
        """Synchronously flush queued observations; intended for tools/tests."""
        if not self._background_persistence:
            return self._flush_worker()

        deadline = time.monotonic() + 10.0
        self._schedule_flush()
        while time.monotonic() < deadline:
            with self._lock:
                current = self._flush_future
                pending = bool(self._pending)
                persisted = self._persisted_observations
                scheduled = self._flush_scheduled
            if not pending and not scheduled:
                return persisted
            if current is None:
                self._schedule_flush()
                time.sleep(0.005)
                continue
            try:
                current.result(timeout=max(0.01, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError:
                break
            except concurrent.futures.CancelledError:
                break
        with self._lock:
            return self._persisted_observations

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = f"file:{quote(str(self._db_path), safe='/')}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            return conn
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _persist_batch(self, batch: Sequence[RecallObservation]) -> int:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("memory_recall_observations"):
            get_file_write_gateway().ensure_directory(
                self._db_path.parent,
                source="memory.recall_observations.persist",
            )
            with connecting(self._connect()) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS recall_observations (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        activation REAL NOT NULL,
                        rank INTEGER NOT NULL,
                        candidates INTEGER NOT NULL,
                        returned INTEGER NOT NULL,
                        recorded_at REAL NOT NULL
                    )"""
                )
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(recall_observations)").fetchall()
                }
                if "returned" not in columns:
                    # Rows written by the pre-label prototype remain unknown,
                    # not silently false. load_persisted excludes their NULLs.
                    conn.execute("ALTER TABLE recall_observations ADD COLUMN returned INTEGER")
                conn.executemany(
                    "INSERT INTO recall_observations "
                    "(activation, rank, candidates, returned, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            obs.activation,
                            obs.rank,
                            obs.candidates,
                            int(obs.returned),
                            obs.recorded_at,
                        )
                        for obs in batch
                    ],
                )
                conn.execute(
                    "DELETE FROM recall_observations WHERE sequence <= COALESCE(("
                    "SELECT sequence FROM recall_observations ORDER BY sequence DESC "
                    "LIMIT 1 OFFSET ?), 0)",
                    (int(self._ring.maxlen),),
                )
                count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM recall_observations WHERE returned IN (0, 1)"
                    ).fetchone()[0]
                )
        return count

    def load_persisted(self, *, replace: bool = True) -> int:
        """Load the durable tail. Safe for a standalone fitter process."""
        if not self._db_path.exists():
            return 0
        with connecting(self._connect(readonly=True)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recall_observations'"
            ).fetchone()
            if not exists:
                return 0
            rows = conn.execute(
                "SELECT activation, rank, candidates, returned, recorded_at "
                "FROM recall_observations WHERE returned IN (0, 1) "
                "ORDER BY sequence DESC LIMIT ?",
                (int(self._ring.maxlen),),
            ).fetchall()
        loaded = [
            RecallObservation(
                float(a),
                int(rank),
                int(candidates),
                bool(returned),
                float(recorded_at),
            )
            for a, rank, candidates, returned, recorded_at in reversed(rows)
        ]
        with self._lock:
            if replace:
                self._ring.clear()
                self._rankings = 0
            self._ring.extend(loaded)
            self._rankings += sum(1 for obs in loaded if obs.rank == 0)
            self._persisted_observations = len(loaded)
        return len(loaded)

    def _restore_failed_batch(self, batch: Sequence[RecallObservation]) -> None:
        with self._lock:
            available = max(0, self._pending.maxlen - len(self._pending))
            restored = list(batch[-available:]) if available else []
            self._dropped_pending += len(batch) - len(restored)
            self._pending.extendleft(reversed(restored))
            self._inflight_observations = 0

    def _note_flush_failure(self, exc: BaseException) -> None:
        with self._lock:
            self._flush_failures += 1
            count = self._flush_failures
            self._retry_not_before = time.monotonic() + min(60.0, float(2 ** min(count, 6)))
        if count & (count - 1) == 0:
            logger.warning(
                "Recall observation persistence failed %d time(s); live recall "
                "continues and %d observations remain buffered: %s",
                count,
                self.stats()["pending_persistence"],
                exc,
            )

    def _note_schedule_failure(self, exc: BaseException) -> None:
        with self._lock:
            self._schedule_failures += 1
            count = self._schedule_failures
        if count & (count - 1) == 0:
            logger.debug(
                "Recall observation flush scheduling unavailable %d time(s); "
                "evidence remains buffered: %s",
                count,
                exc,
            )

    def observations(self) -> list[RecallObservation]:
        with self._lock:
            return list(self._ring)

    def samples(self) -> list[tuple[float, int]]:
        return [(obs.activation, int(obs.returned)) for obs in self.observations()]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "observations": len(self._ring),
                "rankings": self._rankings,
                "capacity": self._ring.maxlen,
                "saturated": len(self._ring) == self._ring.maxlen,
                "pending_persistence": len(self._pending) + self._inflight_observations,
                "queued_persistence": len(self._pending),
                "inflight_persistence": self._inflight_observations,
                "persisted_observations": self._persisted_observations,
                "dropped_pending": self._dropped_pending,
                "flush_failures": self._flush_failures,
                "schedule_failures": self._schedule_failures,
                "flush_active": bool(
                    self._flush_scheduled or (self._flush_future and not self._flush_future.done())
                ),
                "retry_in_s": max(0.0, self._retry_not_before - time.monotonic()),
                "last_flush_at": self._last_flush_at or None,
                "store": str(self._db_path),
                "content_fields_stored": [],
            }

    def clear(self, *, persisted: bool = False) -> None:
        with self._lock:
            self._ring.clear()
            self._pending.clear()
            self._inflight_observations = 0
            self._rankings = 0
            self._persisted_observations = 0
        if persisted and self._db_path.exists():
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope("memory_recall_observations"):
                with connecting(self._connect()) as conn:
                    exists = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='recall_observations'"
                    ).fetchone()
                    if exists:
                        conn.execute("DELETE FROM recall_observations")


_ring: RecallObservationRing | None = None
_ring_lock = checked_lock("recall_observations.singleton", rank=LockRank.REGISTRY)
_recording_failures = 0


def peek_recall_observations() -> RecallObservationRing | None:
    """The ring if one exists, and never a reason to build one.

    A read-only consumer — telemetry, a state probe, a health block — must not
    be the thing that constructs the store. Construction resolves the memory
    directory, which imports the configuration, which touches disk; doing that
    on a request path because something wanted to look is how a read turns
    into a stall.
    """
    return _ring


def get_recall_observations() -> RecallObservationRing:
    """The ring, building one if this is the first caller.

    Built OUTSIDE the singleton lock and published under it. Construction
    resolves the store path, which imports the configuration and can touch
    disk, and lockdep saw exactly that: an fsync attempted while holding
    ``recall_observations.singleton``. A racing second caller builds a ring
    that is discarded unread, which costs nothing and holds no lock.
    """
    global _ring
    if _ring is not None:
        return _ring
    candidate = RecallObservationRing()
    with _ring_lock:
        if _ring is None:
            _ring = candidate
        return _ring


def record_ranking(activations: Iterable[float], *, returned_count: int) -> None:
    """Best-effort record of one ranked recall. Never raises into recall."""
    global _recording_failures
    try:
        get_recall_observations().record(list(activations), returned_count=returned_count)
    except (RuntimeError, ValueError, TypeError, MemoryError, OSError) as exc:
        with _ring_lock:
            _recording_failures += 1
            count = _recording_failures
        if count & (count - 1) == 0:
            logger.warning(
                "Recall observation recording failed %d time(s); recall itself "
                "continues without this evidence: %s",
                count,
                exc,
            )
