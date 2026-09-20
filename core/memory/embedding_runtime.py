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


_SHORTEST_HOLD_S: float | None = None


def _shortest_hold_s() -> float:
    """How long a timer takes to fire on this host, measured once.

    A hold shorter than that cannot be a hold: the timer would fire after
    the caller has already moved on, on another thread. Measured rather
    than chosen, so the threshold is the mechanism's own resolution.
    """
    global _SHORTEST_HOLD_S
    if _SHORTEST_HOLD_S is None:
        fired = threading.Event()
        began = time.monotonic()
        timer = threading.Timer(0.0, fired.set)
        timer.daemon = True
        timer.start()
        fired.wait(timeout=1.0)
        _SHORTEST_HOLD_S = max(0.0, time.monotonic() - began)
    return _SHORTEST_HOLD_S


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
        #: Set while one acquirer builds the engine OUTSIDE the lock. Loading
        #: the embedding model reads and fsyncs its cache, and lockdep named
        #: the fsync under this lock (2026-09-20); a build under the lock
        #: also held every release and snapshot for the whole load.
        self._building = threading.Event()
        self._building.set()  # nobody is building

    def acquire(self, owner: str) -> SharedEmbeddingLease:
        owner_name = str(owner or "unknown").strip() or "unknown"
        while True:
            with self._lock:
                timer, self._closing_timer = self._closing_timer, None
                if timer is not None:
                    timer.cancel()
                if self._engine is not None:
                    token = uuid.uuid4().hex
                    self._owners[token] = owner_name
                    return SharedEmbeddingLease(self, token)
                if self._building.is_set():
                    self._building.clear()  # this caller builds
                    break
            # Another caller is building: wait for it off the lock, then
            # look again. Bounded so a builder that died cannot hold the
            # rest for ever; the next look sees no engine and builds.
            self._building.wait(timeout=120.0)
        began = time.monotonic()
        try:
            engine = self._engine_factory()
        finally:
            self._building.set()
        with self._lock:
            if self._engine is None:
                self._engine = engine
                self._build_seconds = max(0.0, time.monotonic() - began)
                logger.info(
                    "Embedding runtime created shared engine in %.2fs",
                    self._build_seconds,
                )
            else:
                # Lost a race to a second builder; this one is surplus.
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
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
            if self._build_seconds <= _shortest_hold_s():
                # Nothing to hold for: an engine that cost less to build
                # than a timer takes to fire is closed now, on this thread.
                # A zero-delay timer is not "at once" — it is another
                # thread, later — and a caller that releases the last lease
                # and then reads the engine's state saw it still open.
                engine, self._engine = self._engine, None
            else:
                engine = None
                timer = threading.Timer(self._build_seconds, self._close_if_still_idle)
                timer.daemon = True
                self._closing_timer = timer
                timer.start()
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
            logger.info("Embedding runtime closed engine after final owner release")

    def _close_if_still_idle(self) -> None:
        with self._lock:
            self._closing_timer = None
            if self._owners or self._engine is None:
                return
            engine, self._engine = self._engine, None
        # Closing may persist a cache; outside the lock, as `close` does.
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        logger.info(
            "Embedding runtime closed engine after %.2fs idle, which is what building it cost",
            self._build_seconds,
        )

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
