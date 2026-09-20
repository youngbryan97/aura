"""One process-wide embedding model with explicit subsystem ownership.

Vector memory, semantic RAG, and evidence routing all use the same encoder.
Historically each surface constructed its own ``EmbeddingEngine`` and the
first foreground turn could therefore load the same checkpoint three times.
This runtime gives every consumer an independently closable lease while
keeping exactly one underlying engine alive until the final owner releases it.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.EmbeddingRuntime")


class SharedEmbeddingLease:
    """A transparent, independently closable view of the shared engine."""

    __slots__ = ("_closed", "_runtime", "_token")

    def __init__(self, runtime: SharedEmbeddingRuntime, token: str) -> None:
        self._runtime = runtime
        self._token = token
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime.resolve(self._token), name)

    def __enter__(self) -> SharedEmbeddingLease:
        self._runtime.resolve(self._token)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime.release(self._token)


class SharedEmbeddingRuntime:
    """Reference-counted owner of one lazily loaded embedding engine."""

    def __init__(self, engine_factory: Callable[[], Any]) -> None:
        self._engine_factory = engine_factory
        self._lock = checked_lock("embedding_runtime.lifecycle", reentrant=True)
        self._engine: Any | None = None
        self._owners: dict[str, str] = {}
        #: How long the last build took, and the timer that will close the
        #: engine that long after the final release.
        #:
        #: The main runtime holds a process-owned lease from
        #: `prewarm_shared_embedding_runtime`, so its engine stays resident.
        #: The MLX worker has no such lease and every consumer there acquires
        #: and releases around one call, so the engine was built and torn
        #: down per use: live on 2026-09-20 the worker loaded
        #: Qwen3-Embedding-0.6B five times in six seconds, 03:55:27 to
        #: 03:55:33, each load followed within a second by "closed engine
        #: after final owner release".
        #:
        #: Holding it for exactly as long as building it took is the trade
        #: stated in its own terms: at worst the memory is held for the time
        #: it would have cost to get it back, and at best a burst of acquires
        #: reuses one engine. A process that genuinely stops embedding still
        #: gives the memory back.
        self._build_seconds = 0.0
        self._closing_timer: threading.Timer | None = None

    def acquire(self, owner: str) -> SharedEmbeddingLease:
        owner_name = str(owner or "unknown").strip() or "unknown"
        with self._lock:
            timer, self._closing_timer = self._closing_timer, None
            if timer is not None:
                timer.cancel()
            if self._engine is None:
                began = time.monotonic()
                self._engine = self._engine_factory()
                self._build_seconds = max(0.0, time.monotonic() - began)
                logger.info(
                    "Embedding runtime created shared engine in %.2fs",
                    self._build_seconds,
                )
            token = uuid.uuid4().hex
            self._owners[token] = owner_name
        return SharedEmbeddingLease(self, token)

    def resolve(self, token: str) -> Any:
        with self._lock:
            if token not in self._owners or self._engine is None:
                raise RuntimeError("shared_embedding_lease_released")
            return self._engine

    def release(self, token: str) -> None:
        with self._lock:
            if self._owners.pop(token, None) is None:
                return
            if self._owners or self._engine is None:
                return
            # One path, always the timer. A build that took no measurable
            # time schedules a zero delay and closes at once, which is the
            # same answer a special case would give and one fewer branch to
            # be wrong about.
            timer = threading.Timer(self._build_seconds, self._close_if_still_idle)
            timer.daemon = True
            self._closing_timer = timer
            timer.start()

    def _close_if_still_idle(self) -> None:
        with self._lock:
            self._closing_timer = None
            if self._owners or self._engine is None:
                return
            self._close_engine_locked(
                f"after {self._build_seconds:.2f}s idle, which is what building it cost"
            )

    def _close_engine_locked(self, because: str) -> None:
        engine, self._engine = self._engine, None
        if engine is None:
            return
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        logger.info("Embedding runtime closed engine %s", because)

    def close(self) -> None:
        """Invalidate every lease and close the engine exactly once."""
        with self._lock:
            timer, self._closing_timer = self._closing_timer, None
            if timer is not None:
                timer.cancel()
            self._owners.clear()
            engine, self._engine = self._engine, None
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine_live": self._engine is not None,
                "lease_count": len(self._owners),
                "owners": tuple(sorted(self._owners.values())),
            }


def _new_embedding_engine() -> Any:
    # Lazy import avoids a module cycle while VectorMemoryEngine itself acquires
    # a lease during construction.
    from core.memory.vector_memory_engine import EmbeddingEngine

    return EmbeddingEngine()


_RUNTIME = SharedEmbeddingRuntime(_new_embedding_engine)
_PREWARM_LOCK = checked_lock("embedding_runtime.prewarm", reentrant=True)
_PREWARM_LEASE: SharedEmbeddingLease | None = None


def acquire_shared_embedding_engine(owner: str) -> SharedEmbeddingLease:
    return _RUNTIME.acquire(owner)


def shared_embedding_runtime_snapshot() -> dict[str, Any]:
    return _RUNTIME.snapshot()


def prewarm_shared_embedding_runtime() -> dict[str, Any]:
    """Load the shared encoder once outside the first foreground request.

    The retained lease is process-owned.  Consumers still acquire and release
    independent leases, while the encoder remains resident until runtime
    shutdown instead of being closed between sparse recalls.
    """

    global _PREWARM_LEASE
    with _PREWARM_LOCK:
        lease = _PREWARM_LEASE
        if lease is None:
            lease = _RUNTIME.acquire("runtime-prewarm")
            _PREWARM_LEASE = lease
    try:
        vector = lease.embed("Aura semantic memory readiness probe")
    except Exception:
        with _PREWARM_LOCK:
            if _PREWARM_LEASE is lease:
                _PREWARM_LEASE = None
        lease.close()
        raise
    snapshot = _RUNTIME.snapshot()
    snapshot["prewarmed"] = True
    snapshot["vector_dimensions"] = int(getattr(vector, "size", len(vector)))
    return snapshot


def close_shared_embedding_runtime() -> None:
    global _PREWARM_LEASE
    with _PREWARM_LOCK:
        lease, _PREWARM_LEASE = _PREWARM_LEASE, None
    if lease is not None:
        lease.close()
    _RUNTIME.close()


atexit.register(close_shared_embedding_runtime)
