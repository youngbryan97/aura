from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import os
import sqlite3
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import aiosqlite

from core.bus.actor_bus import BusDegraded
from core.memory.retention_policy import state_log_retention_policy
from core.runtime.background_policy import is_user_facing_origin
from core.runtime.effect_boundary import effect_sink
from core.runtime.errors import record_degradation
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.task_tracker import get_task_tracker

from ..bus.shared_mem_bus import SharedMemoryTransport
from ..container import ServiceContainer

if TYPE_CHECKING:
    from .aura_state import (
        AuraState,
    )

logger = logging.getLogger(__name__)
_STATE_LOG_RETENTION_POLICY = state_log_retention_policy()

_STATE_SUBSYSTEM = "state_repository"
_STATE_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
    asyncio.InvalidStateError,
)
_REBASEABLE_ISOLATION_CAUSES = frozenset(
    {
        "task_isolation_reset",
        "dnu_kernel_task_isolation",
    }
)
def _state_proxy_commit_timeout_seconds() -> float:
    raw = os.getenv("AURA_STATE_PROXY_COMMIT_TIMEOUT_S", "15.0")
    try:
        return max(5.0, float(raw or "15.0"))
    except (TypeError, ValueError):
        logger.warning("Invalid AURA_STATE_PROXY_COMMIT_TIMEOUT_S override: %r", raw)
        return 15.0


def _is_rebaseable_isolation_commit(cause: str) -> bool:
    normalized = str(cause or "")
    return (
        normalized in _REBASEABLE_ISOLATION_CAUSES
        or normalized.startswith("agency_task_isolation")
        or normalized.startswith("agency_kernel_task_isolation")
    )


def _record_state_degradation(
    error: BaseException,
    *,
    action: str = "state repository operation degraded and isolated",
    severity: str = "degraded",
) -> None:
    record_degradation(_STATE_SUBSYSTEM, error, severity=severity, action=action)


def _record_proxy_transport_degradation(
    error: BaseException,
    *,
    action: str = "state proxy transport deferred commit to durable outbox",
    severity: str = "degraded",
) -> None:
    """Record proxy transport issues without converting durable deferral into data loss.

    The canonical state repository remains fail-closed for real persistence and
    coherence failures. A proxy-to-vault transport timeout is different when the
    commit payload is retained in the local durable outbox for replay; escalating
    that as a state_repository failure aborts foreground work before the repair
    path can do its job.
    """
    record_degradation(
        "state_repository_proxy_transport",
        error,
        severity=severity,
        action=action,
    )


def _shutdown_requested() -> bool:
    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        return bool(is_shutdown_requested())
    except _STATE_BOUNDARY_ERRORS:
        return False


def _is_shutdown_commit_payload(payload: dict[str, Any]) -> bool:
    return str(payload.get("cause") or "").lower() == "shutdown" or _shutdown_requested()


def _close_if_possible(awaitable: Any) -> None:
    try:
        close = awaitable.close
    except AttributeError:
        return
    try:
        close()
    except _STATE_BOUNDARY_ERRORS as exc:
        _record_state_degradation(exc, action="unscheduled state awaitable close failed")


def _schedule_state_task(awaitable: Any, *, name: str, tracker: Any = None) -> asyncio.Task | None:
    try:
        task_owner = tracker if tracker is not None else get_task_tracker()
        try:
            schedule = task_owner.create_task
        except AttributeError:
            schedule = task_owner.track
        return schedule(awaitable, name=name)
    except RuntimeError as exc:
        _close_if_possible(awaitable)
        logger.debug(
            "StateRepository background task %s deferred outside an event loop: %s", name, exc
        )
        return None
    except _STATE_BOUNDARY_ERRORS as exc:
        _close_if_possible(awaitable)
        _record_state_degradation(exc, action=f"state background task {name} was not scheduled")
        logger.debug("StateRepository background task %s scheduling failed: %s", name, exc)
        return None


def get_state_shm_size_bytes() -> int:
    """Scale the state SHM segment to the host instead of pinning it at 2MB."""
    override = os.getenv("AURA_STATE_SHM_BYTES")
    if override:
        try:
            return max(2 * 1024 * 1024, int(override))
        except ValueError:
            logger.warning("Invalid AURA_STATE_SHM_BYTES override: %r", override)

    try:
        from core.runtime import resource_psutil as psutil

        total_gb = psutil.virtual_memory().total / float(1024**3)
    except _STATE_BOUNDARY_ERRORS:
        total_gb = 0.0

    if total_gb >= 48:
        return 16 * 1024 * 1024
    if total_gb >= 24:
        return 8 * 1024 * 1024
    return 2 * 1024 * 1024


def _sign_state_lineage(state: Any) -> None:
    """Extend the entity's signed chain with this state's link to its parent.

    The signature is attached to the state so a committed record carries the
    evidence with it rather than requiring a side lookup, and the chain head
    lives in the entity identity so a tampered historic link breaks every
    signature after it.
    """

    try:
        from core.identity.entity_key import entity_identity

        continuity_hash = ""
        if hasattr(state, "get_continuity_hash"):
            continuity_hash = str(state.get_continuity_hash() or "")
        link = entity_identity().sign_state_link(
            state_id=str(getattr(state, "state_id", "") or ""),
            version=int(getattr(state, "version", 0) or 0),
            parent_state_id=str(getattr(state, "parent_state_id", "") or ""),
            continuity_hash=continuity_hash,
        )
        state.lineage_signature = link.signature
        state.lineage_entity_id = link.entity_id
    except _STATE_BOUNDARY_ERRORS as exc:
        _record_state_degradation(
            exc,
            action="committed state without a lineage signature",
            severity="warning",
        )


class StateVersionConflictError(Exception):
    """Raised when a state commit is rejected due to version stagnation or backtrack."""

    def __init__(self, current_v: int, rejected_v: int, cause: str):
        self.current_v = current_v
        self.rejected_v = rejected_v
        self.cause = cause
        super().__init__(
            f"State version conflict: current={current_v}, rejected={rejected_v} (cause={cause})"
        )


def _is_user_facing_origin(origin: Any) -> bool:
    return is_user_facing_origin(origin)


class StateRepository:
    """
    Persists and retrieves AuraState.
    The 'continuity' is here — not in the LLM context window.

    Uses an append-only log so the full history of Aura's
    state transitions is recoverable. This IS the long-term memory
    of experience (episodic), separate from semantic memory (vector store).
    """

    # ── Long-Run Stability Config ──────────────────────────────────────────
    STATE_LOG_MAX_ROWS = _STATE_LOG_RETENTION_POLICY.max_items
    STATE_LOG_RETENTION_BASIS = _STATE_LOG_RETENTION_POLICY.basis
    STATE_LOG_PRUNE_EVERY = 100  # Prune check interval (commits)
    STATE_LOG_VACUUM_EVERY = 1000  # VACUUM interval (commits)
    DB_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024
    TRANSPORT_SNAPSHOT_MAX_ITEMS = 64
    TRANSPORT_SNAPSHOT_MAX_TEXT = 4096
    TRANSPORT_WORKING_MEMORY_LIMIT = 36
    TRANSPORT_LONG_TERM_MEMORY_LIMIT = 12
    TRANSPORT_GOAL_LIMIT = 12
    TRANSPORT_PERCEPT_LIMIT = 48
    PROXY_OUTBOX_SLOT = "latest"

    def __init__(self, db_path: str = "data/aura_state.db", is_vault_owner: bool = False):
        self.db_path = db_path
        self.is_vault_owner = is_vault_owner
        self._current: AuraState | None = None
        self._lock: asyncio.Lock | None = None
        self._commit_transaction_lock: asyncio.Lock | None = None
        self._mutation_queue_maxsize = 32
        self._mutation_queue: asyncio.Queue = asyncio.Queue(maxsize=self._mutation_queue_maxsize)
        self._is_processing = False
        self._consumer_task: asyncio.Task | None = None
        self._buffer: dict[str, list] = {}  # Per-trace buffer for causal ordering
        self._shm: SharedMemoryTransport | None = None
        self._db: aiosqlite.Connection | None = None
        self._transport: Any = None
        self._dropped_commit_count = 0
        self._commit_counter = 0  # Tracks commits for prune/VACUUM scheduling
        self._last_commit_at = 0.0
        self._last_commit_duration_ms = 0.0
        self._failed_commit_count = 0
        self._last_commit_failure_at = 0.0
        self._last_commit_error = ""
        self._deferred_commit_count = 0
        self._last_commit_deferred_at = 0.0
        self._last_commit_deferred_reason = ""
        self._last_serialization_ms = 0.0
        self._last_consumer_activity_at = 0.0
        self._repair_count = 0
        self._last_shm_write_mode = "idle"
        self._last_shm_overflow_bytes = 0
        self._pending_proxy_commit_payload: dict[str, Any] | None = None
        self._pending_proxy_commit_count = 0
        self._last_proxy_commit_error = ""

    @property
    def lock(self) -> Any:
        if self._lock is None:
            from core.utils.concurrency import get_robust_lock

            self._lock = get_robust_lock(
                f"StateRepository:{'Owner' if self.is_vault_owner else 'Proxy'}"
            )
        return self._lock

    @property
    def commit_transaction_lock(self) -> Any:
        """Serialize admission, persistence, and publication as one commit."""

        if self._commit_transaction_lock is None:
            from core.utils.concurrency import get_robust_lock

            self._commit_transaction_lock = get_robust_lock(
                f"StateRepositoryCommit:{'Owner' if self.is_vault_owner else 'Proxy'}"
            )
        return self._commit_transaction_lock

    async def _ensure_db(self) -> aiosqlite.Connection:
        """Return the repository-owned connection, opening it once when needed.

        ``aiosqlite>=0.20`` dispatches each result to the calling future's loop;
        its connection no longer carries a legacy ``_loop`` binding. Treating
        that absent private field as a mismatch reopened a worker thread for
        every repository access and left the final worker alive at process exit.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("Database access attempted outside of event loop") from exc

        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")

        return self._db

    def _sanitize_restored_state(self) -> bool:
        """Apply migration cleanup only to the owner's boot snapshot."""

        restore_sanitizer = getattr(
            getattr(self._current, "cognition", None),
            "sanitize_restored_autonomy_state",
            None,
        )
        if not callable(restore_sanitizer):
            return False
        restore_sanitizer()
        return True

    async def initialize(self) -> None:
        from .aura_state import AuraState

        serialized_current: str | None = None
        boot_governance_decision = SimpleNamespace(
            receipt_id="state_repository_bootstrap",
            domain="state_mutation",
            source="state_repository.initialize",
            constraints={"boot_phase": "initialize"},
        )
        if self.is_vault_owner:
            self._ensure_db_parent_directory()
            db = await self._ensure_db()
            await db.execute("""
                CREATE TABLE IF NOT EXISTS state_log (
                    state_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    parent_state_id TEXT,
                    transition_cause TEXT,
                    state_json TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_version ON state_log(version)")
            await db.commit()
            # Load latest from DB
            await self._load_latest_state()
            self._sanitize_restored_state()

            # Ensure we have a default state if DB is empty
            if self._current is None:
                from core.governance_context import governed_scope

                from .aura_state import AuraState

                self._current = AuraState()
                logger.info("🆕 [STATE] No state found in DB. Initialized default AuraState.")
                # Synchronously commit genesis so it's ready in DB
                serialized_current = self._serialize(self._current)
                async with governed_scope(boot_governance_decision):
                    await self._commit_to_db(self._current, serialized_current)

            # Setup SHM for writing
            self._shm = SharedMemoryTransport(
                name="aura_state_shm", size=get_state_shm_size_bytes()
            )
            if self._shm:
                try:
                    await self._shm.create()
                except PermissionError as e:
                    logger.warning(
                        "⚠️ [STATE] Shared memory unavailable in this runtime. Continuing without SHM: %s",
                        e,
                    )
                    self._shm = None
                except OSError as e:
                    logger.warning(
                        "⚠️ [STATE] Shared memory initialization failed. Continuing without SHM: %s",
                        e,
                    )
                    self._shm = None

                # Synchronously write to SHM so it's ready before MindTick boots
                if self._shm:
                    try:
                        if serialized_current is None:
                            serialized_current = await asyncio.to_thread(
                                self._serialize, self._current
                            )
                        from core.governance_context import governed_scope

                        async with governed_scope(boot_governance_decision):
                            shm_mode = await self._sync_to_shm(self._current, serialized_current)
                        if shm_mode == "full":
                            logger.info("✓ [STATE] Genesis state pushed to SHM.")
                        elif shm_mode == "marker":
                            logger.info("✓ [STATE] Genesis overflow marker pushed to SHM.")
                    except _STATE_BOUNDARY_ERRORS as e:
                        _record_state_degradation(e)
                        logger.warning("⚠️ [STATE] Initial SHM write failed: %s", e)

            logger.info("✓ [STATE] Vault Owner Initialized with SHM for writing.")

            # Start consumer
            self._is_processing = True
            if self._start_consumer_task() is None:
                logger.warning(
                    "⚠️ [STATE] Mutation consumer scheduling deferred; runtime repair will retry."
                )
        else:
            self._transport = self._resolve_transport()
            await self._load_pending_proxy_commit()
            # Proxy Mode: Attach to SHM for reading
            self._shm = SharedMemoryTransport(
                name="aura_state_shm", size=get_state_shm_size_bytes()
            )
            if self._shm:
                try:
                    await asyncio.wait_for(self._shm.attach(), timeout=2.5)
                    # Increased retry/wait for initial sync
                    # If this is genesis, the owner might be fractions of a second behind
                    for _attempt in range(5):
                        data = await self.get_state()
                        if data:
                            logger.info("✓ [STATE] Proxy Attached and Synced from Shared Memory")
                            break
                        await asyncio.sleep(0.2)

                    if not self._current:
                        logger.warning("⚠️ [STATE] Proxy attached but SHM is empty (Wait possible)")
                except _STATE_BOUNDARY_ERRORS as e:
                    _record_state_degradation(e)
                    logger.warning(
                        "⚠️ [STATE] Failed to attach to SHM, falling back to boot state: %s", e
                    )
                    self._shm = None
            if not self._current:
                await self._fetch_state_from_vault()
            if not self._current:
                from .aura_state import AuraState

                self._current = AuraState()
                logger.warning(
                    "⚠️ [STATE] Proxy could not hydrate from SHM or Vault. Using local boot snapshot."
                )

    def _resolve_transport(self) -> Any:
        """Resolve the latest ActorBus instance from the container when needed."""
        if self._transport is not None:
            return self._transport
        self._transport = ServiceContainer.get("actor_bus", default=None)
        return self._transport

    def _transport_has_vault(self) -> bool:
        transport = self._resolve_transport()
        if not transport:
            return False
        is_actor_usable = getattr(transport, "is_actor_usable", None)
        if callable(is_actor_usable):
            return bool(is_actor_usable("state_vault"))
        has_actor = getattr(transport, "has_actor", None)
        if callable(has_actor):
            return bool(has_actor("state_vault"))
        return "state_vault" in getattr(transport, "_transports", {})

    async def _fetch_state_from_vault(self) -> AuraState | None:
        """Fallback path when SHM is not yet readable: request the canonical state from the vault actor."""
        transport = self._resolve_transport()
        if not transport:
            logger.debug("🔄 [STATE] ActorBus unavailable; cannot fetch state from Vault yet.")
            return self._current
        if not self._transport_has_vault():
            logger.debug(
                "🔄 [STATE] ActorBus present but state_vault transport not registered yet."
            )
            return self._current

        try:
            logger.info("🔄 [STATE] SHM empty. Requesting full state fetch from Vault...")
            res = await transport.request("state_vault", "get_state", {"full": True})
            if isinstance(res, dict) and res.get("state"):
                self._current = self._deserialize(json.dumps(res["state"]))
                logger.info("✓ [STATE] Full state fetched from Vault via Bus.")
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.error("❌ [STATE] Full fetch failed: %s", e)

        return self._current

    async def commit(
        self, new_state: AuraState, cause: str, trace_id: str | None = None
    ) -> AuraState:
        """Queue a state transition for atomic owner-side processing."""
        trace_id = trace_id or f"trace_{int(time.time() * 1000)}"
        if self.is_vault_owner:
            await self._enqueue_owner_commit(
                {
                    "type": "commit",
                    "state": new_state,
                    "cause": cause,
                    "trace_id": trace_id,
                    "ts": time.time(),
                }
            )
            return new_state  # Processor keeps the 'live' reference

        # Proxy to owner via shared bus
        transport = self._resolve_transport()
        if transport:
            if self._should_use_bounded_db_snapshot(new_state, cause):
                serialized_state = await asyncio.to_thread(
                    self._serialize_transport_snapshot, new_state
                )
                state_dict = json.loads(serialized_state)
            else:
                state_dict = await asyncio.to_thread(self._circular_safe_asdict, new_state)
            payload = {
                "state": state_dict,
                "cause": cause,
                "trace_id": trace_id,
            }
            pending_payload = self._pending_proxy_commit_payload
            if pending_payload is not None:
                pending_ok, transport, pending_error = await self._send_proxy_commit_request(
                    transport,
                    pending_payload,
                )
                if pending_ok:
                    self._pending_proxy_commit_payload = None
                    await self._clear_pending_proxy_commit()
                    logger.info("✅ [STATE] Replayed deferred proxy commit before current commit.")
                else:
                    await self._defer_proxy_commit(payload, pending_error)
                    return new_state

            ok, transport, error = await self._send_proxy_commit_request(transport, payload)
            if ok:
                return new_state
            if _is_shutdown_commit_payload(payload):
                direct_ok = await self._commit_shutdown_direct_snapshot(new_state, error)
                if direct_ok:
                    return new_state
            await self._defer_proxy_commit(payload, error)
            return new_state
        else:
            if os.environ.get("AURA_STRICT_RUNTIME") == "1":
                if self._should_use_bounded_db_snapshot(new_state, cause):
                    serialized_state = await asyncio.to_thread(
                        self._serialize_transport_snapshot, new_state
                    )
                    state_dict = json.loads(serialized_state)
                else:
                    state_dict = await asyncio.to_thread(self._circular_safe_asdict, new_state)
                await self._defer_proxy_commit(
                    {
                        "state": state_dict,
                        "cause": cause,
                        "trace_id": trace_id,
                    },
                    RuntimeError("state_vault transport unavailable"),
                )
                return new_state
            logger.warning(
                "⚠️ [STATE] ActorBus/Transport missing in Proxy Mode (Standalone/Test runtime). Falling back to direct database persistence."
            )
            new_state.transition_cause = cause
            new_state.updated_at = time.time()
            self._current = new_state

            try:
                # Reconstruct/create schema just in case tables do not exist
                db = await self._ensure_db()
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS state_log (
                        state_id TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        parent_state_id TEXT,
                        transition_cause TEXT,
                        state_json TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
                await db.execute("CREATE INDEX IF NOT EXISTS idx_version ON state_log(version)")
                await db.commit()
            except _STATE_BOUNDARY_ERRORS as schema_err:
                logger.error("⚠️ [STATE] Standalone fallback schema setup failed: %s", schema_err)

            try:
                serialized_data = await asyncio.to_thread(self._serialize, new_state)
                db = await self._ensure_db()
                for attempt in range(3):
                    try:
                        async with db.execute("BEGIN IMMEDIATE"):
                            await db.execute(
                                """INSERT OR REPLACE INTO state_log 
                                   (state_id, version, parent_state_id, transition_cause, state_json, timestamp)
                                   VALUES (?, ?, ?, ?, ?, ?)""",
                                (
                                    new_state.state_id,
                                    new_state.version,
                                    new_state.parent_state_id,
                                    new_state.transition_cause,
                                    serialized_data,
                                    new_state.updated_at,
                                ),
                            )
                            await db.commit()
                        logger.debug("💾 Standalone fallback: State v%s committed to local DB.", new_state.version)
                        break
                    except aiosqlite.OperationalError as e:
                        if "database is locked" in str(e) and attempt < 2:
                            await asyncio.sleep(0.1 * (attempt + 1))
                            continue
                        raise
            except _STATE_BOUNDARY_ERRORS as db_err:
                logger.error("🛑 [STATE] Standalone fallback direct database commit failed: %s", db_err)

        return new_state

    async def _send_proxy_commit_request(
        self,
        transport: Any,
        payload: dict[str, Any],
    ) -> tuple[bool, Any, BaseException | None]:
        """Send one proxy commit to the vault with bounded retries."""

        last_error: BaseException | None = None
        for attempt in range(2):
            if transport is None:
                transport = self._resolve_transport()
            if transport is None:
                # Keep the original pipe/bus failure when a retry could not
                # re-resolve a transport — callers classify on that error.
                last_error = last_error or RuntimeError("state_vault transport unavailable")
                if _is_shutdown_commit_payload(payload):
                    logger.info(
                        "🔌 [STATE] Vault transport unavailable during shutdown; attempting shutdown snapshot fallback."
                    )
                    return False, None, last_error
                _record_proxy_transport_degradation(
                    last_error,
                    action="state proxy commit deferred because vault transport was unavailable",
                )
                if attempt == 0:
                    await asyncio.sleep(0.2)
                    continue
                return False, None, last_error
            try:
                response = await transport.request(
                    "state_vault",
                    "commit",
                    payload,
                    timeout=_state_proxy_commit_timeout_seconds(),
                )
                if isinstance(response, dict) and (
                    response.get("failed") is True
                    or response.get("ok") is False
                    or response.get("error")
                ):
                    raise RuntimeError(
                        f"state_vault commit failed: {response.get('error') or response}"
                    )
                return True, transport, None
            except (BrokenPipeError, BusDegraded, ConnectionError) as exc:
                last_error = exc
                if _is_shutdown_commit_payload(payload):
                    logger.debug(
                        "🔌 [STATE] Vault transport closed during shutdown; direct snapshot fallback will be used.",
                    )
                    return False, None, exc
                else:
                    _record_proxy_transport_degradation(exc)
                    logger.warning(
                        "⚠️ [STATE] Vault pipe broken (attempt %d/2): %s — commit deferred for replay.",
                        attempt + 1,
                        type(exc).__name__,
                    )
                self._transport = None
                transport = self._resolve_transport()
                if attempt == 0:
                    await asyncio.sleep(0.3)
                    continue
            except _STATE_BOUNDARY_ERRORS as exc:
                _record_proxy_transport_degradation(exc)
                last_error = exc
                logger.warning(
                    "❌ [STATE] Proxy Commit Request FAILED (attempt %d/2): %s", attempt + 1, exc
                )
                if not isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                    self._transport = None
                    transport = self._resolve_transport()
                if attempt == 0:
                    await asyncio.sleep(0.2)
                    continue
        return False, transport, last_error

    async def _ensure_proxy_outbox_schema(self, db: aiosqlite.Connection) -> None:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS proxy_commit_outbox (
                slot TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)
        await db.commit()

    async def _load_pending_proxy_commit(self) -> None:
        """Load the deferred proxy commit outbox after a process restart."""

        try:
            db = await self._ensure_db()
            await self._ensure_proxy_outbox_schema(db)
            async with db.execute(
                "SELECT payload_json, error, attempts FROM proxy_commit_outbox WHERE slot = ?",
                (self.PROXY_OUTBOX_SLOT,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                raise ValueError("proxy_commit_outbox payload was not a dictionary")
            self._pending_proxy_commit_payload = payload
            self._last_proxy_commit_error = str(row[1] or "loaded_from_outbox")
            self._pending_proxy_commit_count = max(
                self._pending_proxy_commit_count,
                int(row[2] or 1),
            )
            if _is_shutdown_commit_payload(payload):
                logger.info(
                    "🔁 [STATE] Loaded graceful-shutdown state commit for boot replay "
                    "(attempts=%d, cause=%s).",
                    self._pending_proxy_commit_count,
                    payload.get("cause"),
                )
            else:
                logger.warning(
                    "⚠️ [STATE] Loaded deferred proxy commit from durable outbox "
                    "(attempts=%d, cause=%s).",
                    self._pending_proxy_commit_count,
                    payload.get("cause"),
                )
        except _STATE_BOUNDARY_ERRORS as exc:
            _record_state_degradation(
                exc,
                action="deferred proxy commit outbox load failed",
            )
            logger.warning("⚠️ [STATE] Deferred proxy commit outbox load failed: %s", exc)

    async def _persist_pending_proxy_commit(self, payload: dict[str, Any]) -> None:
        try:
            db = await self._ensure_db()
            await self._ensure_proxy_outbox_schema(db)
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            for attempt in range(3):
                try:
                    async with db.execute("BEGIN IMMEDIATE"):
                        await db.execute(
                            """INSERT OR REPLACE INTO proxy_commit_outbox
                               (slot, payload_json, error, attempts, updated_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.PROXY_OUTBOX_SLOT,
                                payload_json,
                                self._last_proxy_commit_error,
                                int(self._pending_proxy_commit_count),
                                time.time(),
                            ),
                        )
                        await db.commit()
                    return
                except aiosqlite.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < 2:
                        await asyncio.sleep(0.1 * (attempt + 1))
                        continue
                    raise
        except _STATE_BOUNDARY_ERRORS as exc:
            _record_state_degradation(
                exc,
                action="deferred proxy commit outbox persist failed",
            )
            logger.error("🛑 [STATE] Deferred proxy commit outbox persist failed: %s", exc)

    async def _clear_pending_proxy_commit(self) -> None:
        try:
            db = await self._ensure_db()
            await self._ensure_proxy_outbox_schema(db)
            await db.execute(
                "DELETE FROM proxy_commit_outbox WHERE slot = ?",
                (self.PROXY_OUTBOX_SLOT,),
            )
            await db.commit()
            self._last_proxy_commit_error = ""
        except _STATE_BOUNDARY_ERRORS as exc:
            _record_state_degradation(
                exc,
                action="deferred proxy commit outbox clear failed",
            )
            logger.warning("⚠️ [STATE] Deferred proxy commit outbox clear failed: %s", exc)

    async def _defer_proxy_commit(
        self,
        payload: dict[str, Any],
        error: BaseException | None,
    ) -> None:
        """Keep the latest failed proxy commit so a later healthy bus can replay it."""

        self._pending_proxy_commit_payload = payload
        self._pending_proxy_commit_count += 1
        self._last_proxy_commit_error = (
            f"{type(error).__name__}: {error}" if error is not None else "unknown"
        )
        await self._persist_pending_proxy_commit(payload)
        if _is_shutdown_commit_payload(payload):
            logger.info(
                "🔌 [STATE] Graceful-shutdown state commit stored for boot replay "
                "(pending_count=%d, cause=%s).",
                self._pending_proxy_commit_count,
                payload.get("cause"),
            )
            logger.debug(
                "Graceful-shutdown state replay source: %s",
                self._last_proxy_commit_error,
            )
        else:
            logger.warning(
                "⚠️ [STATE] Deferred proxy commit for replay "
                "(pending_count=%d, cause=%s, error=%s).",
                self._pending_proxy_commit_count,
                payload.get("cause"),
                self._last_proxy_commit_error,
            )

    async def _commit_shutdown_direct_snapshot(
        self,
        state: AuraState,
        error: BaseException | None,
    ) -> bool:
        """Persist the final shutdown snapshot if the vault already exited.

        This path is deliberately limited to lifecycle shutdown commits. It
        prevents a clean shutdown from manufacturing a boot-replay item after
        the supervisor has already stopped the vault actor, while keeping normal
        foreground state mutations on the canonical StateVault path.
        """

        try:
            await run_sync_shutdown_callable(
                lambda: self._write_shutdown_snapshot_with_governance(state),
                timeout_s=3.0,
                name="state-repository-shutdown-snapshot",
            )
            self._current = state
            self._pending_proxy_commit_payload = None
            self._pending_proxy_commit_count = 0
            self._last_proxy_commit_error = ""
            if isinstance(error, (BrokenPipeError, BusDegraded, ConnectionError)):
                source = "vault_transport_closed"
            elif error is not None:
                source = type(error).__name__
            else:
                source = "transport_unavailable"
            logger.info(
                "✅ [STATE] Shutdown state committed via direct snapshot (cause=%s, source=%s).",
                getattr(state, "transition_cause", "shutdown"),
                source,
            )
            return True
        except _STATE_BOUNDARY_ERRORS as exc:
            _record_state_degradation(
                exc,
                action="shutdown direct state snapshot failed; falling back to boot replay",
            )
            logger.warning(
                "⚠️ [STATE] Shutdown direct state snapshot failed; boot replay will be used: %s",
                exc,
            )
            return False

    def _write_shutdown_snapshot_with_governance(self, state: AuraState) -> None:
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope(
            "state_repository.shutdown_direct_snapshot",
            domain="state_mutation",
            constraints={
                "cause": getattr(state, "transition_cause", "shutdown"),
                "fallback": "vault_transport_closed",
                "bounded_shutdown_writer": True,
            },
        ):
            self._commit_shutdown_snapshot_sync(state)

    @effect_sink(
        "state.shutdown_direct_snapshot",
        allowed_domains=("state_mutation",),
    )
    def _commit_shutdown_snapshot_sync(self, state: AuraState) -> None:
        """Persist final state without creating an executor-owned DB worker."""

        target = Path(self.db_path)
        self._ensure_db_parent_directory()
        serialized_data = (
            self._serialize_transport_snapshot(state)
            if self._should_use_bounded_db_snapshot(state, "shutdown")
            else self._serialize(state)
        )
        payload_bytes = len(serialized_data.encode("utf-8"))
        if payload_bytes > self.DB_PAYLOAD_MAX_BYTES:
            serialized_data = self._serialize_transport_snapshot(state)

        connection = sqlite3.connect(target, timeout=2.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=2000")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS state_log (
                    state_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    parent_state_id TEXT,
                    transition_cause TEXT,
                    state_json TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_version ON state_log(version)")
            connection.execute(
                """INSERT OR REPLACE INTO state_log
                   (state_id, version, parent_state_id, transition_cause, state_json, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    state.state_id,
                    state.version,
                    state.parent_state_id,
                    state.transition_cause,
                    serialized_data,
                    state.updated_at,
                ),
            )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS proxy_commit_outbox (
                    slot TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
            """)
            connection.execute(
                "DELETE FROM proxy_commit_outbox WHERE slot = ?",
                (self.PROXY_OUTBOX_SLOT,),
            )
            connection.commit()
        finally:
            connection.close()

    def _ensure_db_parent_directory(self) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "state_repository.db_directory",
            domain="state_mutation",
            constraints={"operation": "ensure_state_db_parent"},
        ):
            get_file_write_gateway().ensure_directory(
                Path(self.db_path).parent,
                source="state_repository.db_directory",
            )

    async def _enqueue_owner_commit(self, payload: dict[str, Any]) -> None:
        """
        Queue owner-side state transitions with overload coalescing.

        The owner path intentionally avoids deepcopying the full AuraState on the
        foreground request path. State transitions are versioned handoffs created
        through `derive()`, and deep-copying the full organism here was causing
        user-facing chat timeouts after long uptimes.
        """
        if not self._mutation_queue.full():
            await self._mutation_queue.put(payload)
            return

        dropped = self._coalesce_pending_mutations(keep_latest=False)
        self._dropped_commit_count += dropped
        logger.warning(
            "⚠️ [STATE] Mutation queue saturated. Dropped %d stale pending commit(s) before enqueueing the latest state.",
            dropped,
        )
        await self._mutation_queue.put(payload)

    def _coalesce_pending_mutations(self, *, keep_latest: bool) -> int:
        drained = []
        drain_budget = self._mutation_queue.qsize()
        for _ in range(drain_budget):
            try:
                item = self._mutation_queue.get_nowait()
                self._mutation_queue.task_done()
                drained.append(item)
            except asyncio.QueueEmpty:
                break

        if keep_latest and drained:
            latest = drained[-1]
            self._mutation_queue.put_nowait(latest)
            return max(0, len(drained) - 1)
        return len(drained)

    async def get_current(self) -> AuraState | None:
        """Async-compatible alias for get_state (Internal API Standardization)."""
        return await self.get_state()

    async def get_state(self) -> AuraState | None:
        """
        Retrieve the latest state.
        In Proxy mode, this reads from Shared Memory for zero-latency access.
        """
        if self.is_vault_owner:
            return self._current

        if self._current is not None and self._shm is None:
            return self._current

        if self._shm and self.is_vault_owner is False:
            try:
                data = await self._shm.read()
                if data:
                    if isinstance(data, dict) and data.get("_state_overflow"):
                        marker_version = int(data.get("version", 0) or 0)
                        current_version = int(getattr(self._current, "version", 0) or 0)
                        if self._current is None or marker_version > current_version:
                            await self._fetch_state_from_vault()
                    else:
                        try:
                            self._current = self._deserialize(json.dumps(data))
                        except _STATE_BOUNDARY_ERRORS as e:
                            _record_state_degradation(e)
                            logger.error("Failed to auto-sync from SHM: %s", e)
            except _STATE_BOUNDARY_ERRORS as e:
                _record_state_degradation(e)
                logger.warning("⚠️ [STATE] SHM read failed: %s", e)

        if not self._current and self.is_vault_owner is False:
            await self._fetch_state_from_vault()

        return self._current

    async def close(self) -> None:
        """Release durable resources before the event loop shuts down."""
        self._is_processing = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError as _exc:
                logger.debug("StateRepository consumer cancelled during shutdown: %s", _exc)
            except _STATE_BOUNDARY_ERRORS as e:
                _record_state_degradation(e)
                logger.debug("StateRepository consumer shutdown issue: %s", e)
            finally:
                self._consumer_task = None

        if self._shm:
            try:
                self._shm.close()
            except _STATE_BOUNDARY_ERRORS as e:
                _record_state_degradation(e)
                logger.debug("StateRepository SHM close issue: %s", e)
            finally:
                self._shm = None

        if self._db is not None:
            try:
                await self._db.close()
            except _STATE_BOUNDARY_ERRORS as e:
                _record_state_degradation(e)
                logger.debug("StateRepository DB close issue: %s", e)
            finally:
                self._db = None

    def _start_consumer_task(self) -> asyncio.Task | None:
        task = _schedule_state_task(
            self._mutation_consumer_loop(),
            name="vault_mutation_consumer",
        )
        if task is None:
            return None
        self._consumer_task = task
        return task

    async def _mutation_consumer_loop(self):
        """Atomic Mutation Consumer — The Heart of the Actor-Kernel."""
        logger.info("🧠 State Mutation Consumer active.")
        try:
            while self._is_processing:
                try:
                    msg = await self._mutation_queue.get()
                    self._last_consumer_activity_at = time.time()
                    if msg is None:
                        continue
                    try:
                        if msg.get("type") == "commit":
                            state = msg.get("state")
                            cause = msg.get("cause", "unknown")
                            if state:
                                await self._process_commit(state, cause)
                    finally:
                        self._mutation_queue.task_done()
                        self._last_consumer_activity_at = time.time()
                except asyncio.CancelledError:
                    logger.info("[STATE] mutation consumer cancelled")
                    break
                except _STATE_BOUNDARY_ERRORS as e:
                    _record_state_degradation(e)
                    logger.error("🛑 Error in mutation consumer: %s", e)
                    # small backoff to avoid hot-loop on repeated failure
                    await asyncio.sleep(0.1)
        finally:
            logger.info("🧠 Mutation consumer exiting.")

    async def _process_commit(self, new_state: AuraState, cause: str) -> bool:
        """Admit, persist, and publish one state transition atomically."""

        async with self.commit_transaction_lock:
            return await self._process_commit_transaction(new_state, cause)

    async def _process_commit_transaction(self, new_state: AuraState, cause: str) -> bool:
        """Run one serialized owner-side commit without exposing partial state."""

        commit_started = time.perf_counter()
        governance_decision = None
        try:
            from core.constitution import (
                ProposalOutcome,
                get_constitutional_core,
                unpack_governance_result,
            )

            new_state.health = copy.deepcopy(getattr(new_state, "health", {}) or {})
            if hasattr(new_state, "compact"):
                try:
                    new_state.compact()
                except _STATE_BOUNDARY_ERRORS as exc:
                    _record_state_degradation(exc)
                    logger.debug("State compaction skipped during commit: %s", exc)

            approved, reason, governance_decision = unpack_governance_result(
                await get_constitutional_core().approve_state_mutation(
                    getattr(new_state, "transition_origin", "system"),
                    cause,
                    state=new_state,
                    return_decision=True,
                )
            )
            if not approved:
                if getattr(governance_decision, "outcome", None) == ProposalOutcome.DEFERRED:
                    self._record_commit_deferral(reason or "unspecified")
                    logger.info(
                        "[STATE] ConstitutionalCore deferred state mutation "
                        "(origin=%s cause=%s reason=%s)",
                        getattr(new_state, "transition_origin", "system"),
                        cause,
                        reason,
                    )
                    return False
                self._record_commit_failure(f"governance_denied:{reason or 'unspecified'}")
                logger.warning(
                    "🚫 [STATE] ConstitutionalCore blocked state mutation (origin=%s cause=%s reason=%s)",
                    getattr(new_state, "transition_origin", "system"),
                    cause,
                    reason,
                )
                return False
        except _STATE_BOUNDARY_ERRORS as exc:
            self._record_commit_failure(
                f"governance_unavailable:{type(exc).__name__}: {exc}"
            )
            _record_state_degradation(
                exc,
                action="state commit refused because constitutional admission was unavailable",
                severity="critical",
            )
            logger.error("Constitutional state gate unavailable; state commit refused: %s", exc)
            return False

        async with self.lock:
            current = self._current
            previous_pending = (
                list(getattr(getattr(current, "cognition", None), "pending_initiatives", []) or [])
                if current
                else []
            )
            # Atomic Version Guard
            if current and new_state.version <= current.version:
                if _is_rebaseable_isolation_commit(cause):
                    logger.debug(
                        "[STATE] Rebased isolation commit: Version %d <= current %d (Cause: %s)",
                        new_state.version,
                        self._current.version,
                        cause,
                    )
                    new_state.version = current.version + 1
                    new_state.parent_state_id = current.state_id
                elif cause != "bootstrap":
                    logger.debug(
                        "[STATE] Atomic Guard Reject: Version %d <= current %d (Cause: %s)",
                        new_state.version,
                        self._current.version,
                        cause,
                    )
                    return False

            new_state.transition_cause = cause
            new_state.updated_at = time.time()

        # Sign this state's link to its parent with the durable entity key.
        #
        # Lineage was already strong — parent_state_id, monotonic version,
        # serialized admission, a continuity hash over the non-volatile fields
        # — and signed to nothing, because the identity anchor was itself a
        # function of state_id. Signing here, after rebasing, means the
        # signature covers the version and parent the commit actually used.
        #
        # Best-effort by construction: a signing failure must not refuse a
        # state commit. It is recorded, and the chain shows the gap.
        _sign_state_lineage(new_state)

        # Serialize only after version rebasing and transition metadata are final,
        # so the durable JSON agrees with the indexed database columns.
        try:
            start_ser = time.perf_counter()
            if self._should_use_bounded_db_snapshot(new_state, cause):
                serialized_data = await asyncio.to_thread(
                    self._serialize_transport_snapshot, new_state
                )
            else:
                serialized_data = await asyncio.to_thread(self._serialize, new_state)
                payload_bytes = len(serialized_data.encode("utf-8"))
                if payload_bytes > self.DB_PAYLOAD_MAX_BYTES:
                    logger.warning(
                        "⚠️ [STATE] Full DB payload overflow: %d bytes exceeds budget %d. "
                        "Persisting bounded hot snapshot instead.",
                        payload_bytes,
                        self.DB_PAYLOAD_MAX_BYTES,
                    )
                    serialized_data = await asyncio.to_thread(
                        self._serialize_transport_snapshot, new_state
                    )
            ser_ms = (time.perf_counter() - start_ser) * 1000
            self._last_serialization_ms = ser_ms
            if ser_ms > 20:
                logger.warning("📉 [STATE] Heavy Serialization Detected: %.2fms", ser_ms)
        except _STATE_BOUNDARY_ERRORS as e:
            self._record_commit_failure(f"serialization_failed:{type(e).__name__}: {e}")
            _record_state_degradation(e)
            logger.error("🛑 [STATE] Serialization failed: %s", e)
            return False

        # Persist DB + publish SHM inline within the single
        # consumer instead of spawning unbounded write tasks. The queue already
        # gives us async decoupling from foreground chat, and inline writes keep
        # long uptimes from degenerating into thousands of pending DB/SHM tasks.
        try:
            if governance_decision is not None:
                from core.governance_context import governed_scope

                async with governed_scope(governance_decision):
                    await self._commit_to_db(new_state, serialized_data)
                    if self._shm:
                        try:
                            await self._sync_to_shm(new_state, serialized_data)
                        except _STATE_BOUNDARY_ERRORS as exc:
                            _record_state_degradation(exc)
                            logger.warning("⚠️ [STATE] SHM propagation failed: %s", exc)
            else:
                # The same commit, so the same scope.
                #
                # The database write went through and its mirror into shared
                # memory did not, because only the mirror is a declared
                # effect sink. Every reader of the shared state then held a
                # version older than the one on disk, and the only sign was
                # "SHM Update Failed: sink:state.sync_to_shm called outside
                # governed context" — eighteen of them in the live log.
                #
                # A commit with no decision behind it is her own state
                # keeping up with itself, which is maintenance. What must not
                # differ is what the two lanes are allowed to do.
                from core.governance_context import (  # noqa: PLC0415
                    local_internal_governed_scope,
                )

                with local_internal_governed_scope(
                    "state_repository.commit", domain="state_mutation"
                ):
                    await self._commit_to_db(new_state, serialized_data)
                    if self._shm:
                        try:
                            await self._sync_to_shm(new_state, serialized_data)
                        except _STATE_BOUNDARY_ERRORS as exc:
                            _record_state_degradation(exc)
                            logger.warning(
                                "⚠️ [STATE] SHM propagation failed: %s", exc
                            )
        except _STATE_BOUNDARY_ERRORS as exc:
            self._record_commit_failure(f"persistence_failed:{type(exc).__name__}: {exc}")
            _record_state_degradation(exc)
            logger.error("🛑 [STATE] Vault persistence failed: %s", exc)
            raise  # Fail closed on critical state mutation failure
        finally:
            self._last_commit_duration_ms = (time.perf_counter() - commit_started) * 1000.0

        async with self.lock:
            self._current = new_state
            self._last_commit_at = time.time()
            self._last_commit_error = ""
            logger.debug(
                "💾 [STATE] Durable state v%d published to memory.", new_state.version
            )

        try:
            from core.constitution import ProposalKind, get_constitutional_core

            current_pending = list(
                getattr(getattr(new_state, "cognition", None), "pending_initiatives", []) or []
            )
            previous_keys = {
                json.dumps(item, sort_keys=True, default=str)
                for item in previous_pending
                if isinstance(item, dict)
            }
            constitution = get_constitutional_core()
            for item in current_pending:
                if not isinstance(item, dict):
                    continue
                item_key = json.dumps(item, sort_keys=True, default=str)
                if item_key in previous_keys:
                    continue
                constitution.record_external_decision(
                    kind=ProposalKind.INITIATIVE,
                    source=str(
                        item.get("source") or getattr(new_state, "transition_origin", "system")
                    ),
                    summary=str(item.get("goal") or item.get("type") or "initiative"),
                    outcome="recorded",
                    reason=f"state_commit:{cause}",
                    target="pending_initiatives",
                    payload=item,
                    state=new_state,
                )
        except _STATE_BOUNDARY_ERRORS as exc:
            _record_state_degradation(exc)
            logger.debug("Initiative proposal audit skipped: %s", exc)
        return True

    def _record_commit_failure(self, reason: str) -> None:
        self._failed_commit_count += 1
        self._last_commit_failure_at = time.time()
        self._last_commit_error = str(reason or "unknown")[:512]

    def _record_commit_deferral(self, reason: str) -> None:
        self._deferred_commit_count += 1
        self._last_commit_deferred_at = time.time()
        self._last_commit_deferred_reason = str(reason or "unspecified")[:512]


    def _should_use_bounded_db_snapshot(self, state: AuraState, cause: str) -> bool:
        origin = getattr(state, "transition_origin", "") or getattr(
            getattr(state, "cognition", None), "current_origin", ""
        )
        if not _is_user_facing_origin(origin):
            return True
        cause_lower = str(cause or "").strip().lower()
        return any(
            marker in cause_lower
            for marker in (
                "background",
                "baseline",
                "dream",
                "research",
                "consolidation",
                "identity_refresh",
                "autonomous",
                "idle",
            )
        )

    def get_runtime_status(self) -> dict[str, Any]:
        local_consumer_alive = bool(self._consumer_task and not self._consumer_task.done())
        shm_attached = bool(self._shm is not None)
        state_available = self._current is not None
        vault_transport_available = False
        if not self.is_vault_owner:
            try:
                vault_transport_available = bool(self._transport_has_vault())
            except _STATE_BOUNDARY_ERRORS:
                vault_transport_available = False

        consumer_alive = local_consumer_alive
        if not self.is_vault_owner:
            # Proxy repositories do not own a local mutation consumer. They are healthy
            # when they are hydrated and still attached to the vault/SHM path.
            consumer_alive = bool(state_available and (shm_attached or vault_transport_available))

        return {
            "is_vault_owner": bool(self.is_vault_owner),
            "queue_depth": int(self._mutation_queue.qsize()),
            "queue_maxsize": int(self._mutation_queue_maxsize),
            "dropped_commit_count": int(self._dropped_commit_count),
            "consumer_alive": consumer_alive,
            "local_consumer_alive": local_consumer_alive,
            "consumer_done": bool(self._consumer_task.done()) if self._consumer_task else False,
            "db_connected": self._db is not None,
            "current_version": int(getattr(self._current, "version", 0) or 0),
            "last_commit_at": float(self._last_commit_at or 0.0),
            "last_commit_duration_ms": float(self._last_commit_duration_ms or 0.0),
            "failed_commit_count": int(self._failed_commit_count),
            "last_commit_failure_at": float(self._last_commit_failure_at or 0.0),
            "last_commit_error": str(self._last_commit_error),
            "deferred_commit_count": int(self._deferred_commit_count),
            "last_commit_deferred_at": float(self._last_commit_deferred_at or 0.0),
            "last_commit_deferred_reason": str(self._last_commit_deferred_reason),
            "last_serialization_ms": float(self._last_serialization_ms or 0.0),
            "last_consumer_activity_at": float(self._last_consumer_activity_at or 0.0),
            "repair_count": int(self._repair_count),
            "last_shm_write_mode": str(self._last_shm_write_mode),
            "last_shm_overflow_bytes": int(self._last_shm_overflow_bytes),
            "pending_proxy_commit": self._pending_proxy_commit_payload is not None,
            "pending_proxy_commit_count": int(self._pending_proxy_commit_count),
            "last_proxy_commit_error": str(self._last_proxy_commit_error),
            "shm_attached": shm_attached,
            "state_available": state_available,
            "vault_transport_available": vault_transport_available,
        }

    def is_initialized(self) -> bool:
        """Return True only when continuity state and transport are usable."""
        status = self.get_runtime_status()
        if not bool(status["state_available"]):
            return False
        if self.is_vault_owner:
            # db_connected is the hard gate. The mutation consumer may take a
            # moment to schedule its first iteration after boot — that is a
            # liveness concern handled by repair_runtime, not an initialization
            # prerequisite. Blocking boot health on consumer_alive causes a
            # race where the health contract evaluates before the event loop
            # has had a chance to run the consumer coroutine.
            return bool(status["db_connected"])
        # For proxy/client repositories, we check if the state is available AND we have at least one usable transport.
        return bool(status["shm_attached"] or status["vault_transport_available"])

    async def repair_runtime(self) -> dict[str, Any]:
        actions: list[str] = []

        if (
            self.is_vault_owner
            and self._is_processing
            and (self._consumer_task is None or self._consumer_task.done())
        ):
            if self._start_consumer_task() is not None:
                self._repair_count += 1
                actions.append("restarted_consumer")
            else:
                actions.append("consumer_restart_deferred")

        if self.is_vault_owner and self._db is None:
            await self._ensure_db()
            self._repair_count += 1
            actions.append("reconnected_db")

        if not self.is_vault_owner and self._current is None:
            try:
                await self._fetch_state_from_vault()
            except _STATE_BOUNDARY_ERRORS as exc:
                _record_state_degradation(exc, action="proxy rehydrate repair failed")
            if self._current is not None:
                self._repair_count += 1
                actions.append("rehydrated_proxy")

        if not self.is_vault_owner and self._pending_proxy_commit_payload is not None:
            transport = self._resolve_transport()
            if transport is not None:
                ok, _transport, error = await self._send_proxy_commit_request(
                    transport,
                    self._pending_proxy_commit_payload,
                )
                if ok:
                    self._pending_proxy_commit_payload = None
                    await self._clear_pending_proxy_commit()
                    self._repair_count += 1
                    actions.append("flushed_pending_proxy_commit")
                elif error is not None:
                    self._last_proxy_commit_error = f"{type(error).__name__}: {error}"

        queue_depth = self._mutation_queue.qsize()
        if queue_depth >= max(1, int(self._mutation_queue_maxsize * 0.75)):
            dropped = self._coalesce_pending_mutations(keep_latest=True)
            if dropped > 0:
                self._dropped_commit_count += dropped
                self._repair_count += 1
                actions.append(f"coalesced_queue:{dropped}")

        return {
            "actions": actions,
            "status": self.get_runtime_status(),
        }

    #: Progressively tighter budgets for the bounded hot snapshot, tried in
    #: order until one fits the transport. Text shrinks faster than list
    #: lengths: a shorter narrative costs a reader far less than a lost goal.
    _TRANSPORT_FIT_LADDER: tuple[tuple[int, int], ...] = (
        (1024, 12),
        (512, 8),
        (256, 4),
        (128, 2),
        (64, 1),
    )
    #: Set only for the duration of a fit attempt. A ceiling, never a floor —
    #: it can tighten a call site's own limit and never loosen it, so no
    #: individual bound can be widened by turning this on.
    _transport_budget: tuple[int, int] | None = None

    def _transport_text_ceiling(self, limit: int | None) -> int:
        base = int(limit or self.TRANSPORT_SNAPSHOT_MAX_TEXT)
        budget = self._transport_budget
        return min(base, budget[0]) if budget else base

    def _transport_item_ceiling(self, limit: int | None) -> int:
        base = int(limit or self.TRANSPORT_SNAPSHOT_MAX_ITEMS)
        budget = self._transport_budget
        return min(base, budget[1]) if budget else base

    def _fit_transport_snapshot(self, state: AuraState, capacity: int) -> bytes | None:
        """Shrink the hot snapshot until it fits ``capacity``, or give up.

        The fixed TRANSPORT_* limits produce a snapshot that is small, not one
        that is small ENOUGH — they know nothing about the transport it has to
        cross. Measured 2026-08-06: 4098 bytes against a 4096-byte capacity.
        Over by two, and the caller fell back to an overflow marker, which
        carries a state id, a version and no state at all. Every reader on the
        other side goes blind for a reason that has nothing to do with how much
        state there actually is.

        Returns None when even the tightest budget will not fit — which is the
        honest case for the marker, and the only one.
        """
        previous = self._transport_budget
        try:
            for budget in self._TRANSPORT_FIT_LADDER:
                self._transport_budget = budget
                try:
                    candidate = self._serialize_transport_snapshot(state).encode("utf-8")
                except _STATE_BOUNDARY_ERRORS as exc:
                    _record_state_degradation(exc)
                    return None
                if len(candidate) <= capacity:
                    return candidate
            return None
        finally:
            self._transport_budget = previous

    def _truncate_transport_text(self, value: Any, *, limit: int | None = None) -> Any:
        if not isinstance(value, str):
            return value
        max_len = self._transport_text_ceiling(limit)
        if len(value) <= max_len:
            return value
        return value[: max(0, max_len - 3)] + "..."

    def _bounded_transport_value(
        self,
        value: Any,
        *,
        max_items: int | None = None,
        max_text: int | None = None,
        depth: int = 0,
        prefer_tail: bool = False,
    ) -> Any:
        item_limit = self._transport_item_ceiling(max_items)
        text_limit = self._transport_text_ceiling(max_text)

        if depth >= 6:
            return f"<TRUNCATED:{type(value).__name__}>"
        if isinstance(value, str):
            return self._truncate_transport_text(value, limit=text_limit)
        if isinstance(value, list):
            items = list(value)
            if len(items) > item_limit:
                items = items[-item_limit:] if prefer_tail else items[:item_limit]
            return [
                self._bounded_transport_value(
                    item,
                    max_items=item_limit,
                    max_text=text_limit,
                    depth=depth + 1,
                    prefer_tail=prefer_tail,
                )
                for item in items
            ]
        if isinstance(value, dict):
            items = list(value.items())
            if len(items) > item_limit:
                items = items[:item_limit]
            return {
                str(key): self._bounded_transport_value(
                    item,
                    max_items=item_limit,
                    max_text=text_limit,
                    depth=depth + 1,
                )
                for key, item in items
            }
        return value

    def _serialize_transport_snapshot(self, state: AuraState) -> str:
        snapshot = self._circular_safe_asdict(state.snapshot_hot())
        if not isinstance(snapshot, dict):
            raise TypeError("State hot snapshot did not serialize to a dict")

        snapshot["_transport_snapshot_kind"] = "hot"

        identity = snapshot.get("identity")
        if isinstance(identity, dict):
            identity["current_narrative"] = self._truncate_transport_text(
                identity.get("current_narrative"),
                limit=2048,
            )
            identity["concept_graph"] = self._bounded_transport_value(
                identity.get("concept_graph", {})
            )

        cognition = snapshot.get("cognition")
        if isinstance(cognition, dict):
            cognition["working_memory"] = self._bounded_transport_value(
                list(cognition.get("working_memory", []) or [])[
                    -self.TRANSPORT_WORKING_MEMORY_LIMIT :
                ],
                max_items=self.TRANSPORT_WORKING_MEMORY_LIMIT,
                prefer_tail=True,
            )
            cognition["long_term_memory"] = self._bounded_transport_value(
                list(cognition.get("long_term_memory", []) or [])[
                    -self.TRANSPORT_LONG_TERM_MEMORY_LIMIT :
                ],
                max_items=self.TRANSPORT_LONG_TERM_MEMORY_LIMIT,
                prefer_tail=True,
            )
            cognition["active_goals"] = self._bounded_transport_value(
                list(cognition.get("active_goals", []) or [])[-self.TRANSPORT_GOAL_LIMIT :],
                max_items=self.TRANSPORT_GOAL_LIMIT,
                prefer_tail=True,
            )
            cognition["pending_initiatives"] = self._bounded_transport_value(
                list(cognition.get("pending_initiatives", []) or [])[-self.TRANSPORT_GOAL_LIMIT :],
                max_items=self.TRANSPORT_GOAL_LIMIT,
                prefer_tail=True,
            )
            cognition["pending_intents"] = self._bounded_transport_value(
                list(cognition.get("pending_intents", []) or [])[-self.TRANSPORT_GOAL_LIMIT :],
                max_items=self.TRANSPORT_GOAL_LIMIT,
                prefer_tail=True,
            )
            cognition["rolling_summary"] = self._truncate_transport_text(
                cognition.get("rolling_summary")
            )
            cognition["last_response"] = self._truncate_transport_text(
                cognition.get("last_response"), limit=2048
            )
            cognition["modifiers"] = self._bounded_transport_value(
                cognition.get("modifiers", {}), max_items=48
            )

        world = snapshot.get("world")
        if isinstance(world, dict):
            world["known_entities"] = self._bounded_transport_value(
                world.get("known_entities", {}), max_items=96
            )
            world["relationship_graph"] = self._bounded_transport_value(
                world.get("relationship_graph", {}), max_items=96
            )
            world["recent_percepts"] = self._bounded_transport_value(
                list(world.get("recent_percepts", []) or [])[-self.TRANSPORT_PERCEPT_LIMIT :],
                max_items=self.TRANSPORT_PERCEPT_LIMIT,
                prefer_tail=True,
            )
            world["spatial_context"] = self._bounded_transport_value(
                world.get("spatial_context", {}), max_items=24
            )

        affect = snapshot.get("affect")
        if isinstance(affect, dict):
            emotions = dict(affect.get("emotions", {}) or {})
            baselines = dict(affect.get("mood_baselines", {}) or {})
            top_emotions = {
                str(key): value
                for key, value in sorted(
                    emotions.items(),
                    key=lambda item: float(item[1] or 0.0),
                    reverse=True,
                )[:8]
            }
            affect["emotions"] = self._bounded_transport_value(top_emotions, max_items=8)
            affect["mood_baselines"] = {
                key: baselines[key]
                for key in top_emotions
                if key in baselines
            }
            affect["markers"] = self._bounded_transport_value(
                affect.get("markers", {}), max_items=12, max_text=512
            )
            affect["resonance"] = self._bounded_transport_value(
                affect.get("resonance", {}), max_items=8
            )

        snapshot["health"] = self._bounded_transport_value(snapshot.get("health", {}), max_items=64)
        snapshot["response_modifiers"] = self._bounded_transport_value(
            snapshot.get("response_modifiers", {}), max_items=48
        )

        return json.dumps(snapshot, ensure_ascii=False)

    @effect_sink("state.sync_to_shm", allowed_domains=("state_mutation",))
    async def _sync_to_shm(self, state: AuraState, serialized_state: str) -> str:
        """Push serialized state into SHM without re-walking the object graph."""
        shm = self._shm
        if shm is None:
            self._last_shm_write_mode = "disabled"
            return "disabled"

        payload = (
            serialized_state.encode("utf-8")
            if isinstance(serialized_state, str)
            else bytes(serialized_state)
        )
        if len(payload) > shm.payload_capacity:
            hot_snapshot_payload: bytes | None = None
            try:
                hot_snapshot_payload = self._serialize_transport_snapshot(state).encode("utf-8")
            except _STATE_BOUNDARY_ERRORS as exc:
                _record_state_degradation(exc)
                logger.warning("⚠️ [STATE] Failed to build bounded SHM hot snapshot: %s", exc)

            # Bounded is not the same as bounded ENOUGH. If the fixed limits
            # still overflow the transport, tighten until it fits rather than
            # dropping to a marker that carries no state.
            if hot_snapshot_payload and len(hot_snapshot_payload) > shm.payload_capacity:
                hot_snapshot_payload = self._fit_transport_snapshot(
                    state, shm.payload_capacity
                )

            if hot_snapshot_payload and len(hot_snapshot_payload) <= shm.payload_capacity:
                if self._last_shm_write_mode != "hot":
                    logger.warning(
                        "⚠️ [STATE] Full SHM snapshot overflow: %d bytes exceeds capacity %d bytes. "
                        "Publishing bounded hot snapshot instead (%d bytes).",
                        len(payload),
                        shm.payload_capacity,
                        len(hot_snapshot_payload),
                    )
                self._last_shm_write_mode = "hot"
                self._last_shm_overflow_bytes = len(payload)
                await asyncio.to_thread(shm.write_serialized, hot_snapshot_payload)
                return "hot"

            overflow_marker = {
                "_state_overflow": True,
                "state_id": getattr(state, "state_id", None),
                "version": int(getattr(state, "version", 0) or 0),
                "updated_at": float(getattr(state, "updated_at", time.time()) or time.time()),
            }
            marker_payload = json.dumps(overflow_marker).encode("utf-8")
            if len(marker_payload) > shm.payload_capacity:
                raise ValueError(
                    "Shared memory overflow marker exceeds SHM capacity "
                    f"({len(marker_payload)} bytes > {shm.payload_capacity} bytes)"
                )
            if self._last_shm_write_mode != "marker":
                logger.warning(
                    "⚠️ [STATE] SHM payload overflow: %d bytes exceeds capacity %d bytes. "
                    "Publishing overflow marker instead.",
                    len(payload),
                    shm.payload_capacity,
                )
            self._last_shm_write_mode = "marker"
            self._last_shm_overflow_bytes = len(payload)
            await asyncio.to_thread(shm.write_serialized, marker_payload)
            return "marker"

        if self._last_shm_write_mode in {"marker", "hot"}:
            logger.info(
                "✓ [STATE] SHM payload back within capacity (%d bytes <= %d bytes). Restoring full snapshot sync.",
                len(payload),
                shm.payload_capacity,
            )
        self._last_shm_write_mode = "full"
        self._last_shm_overflow_bytes = 0
        await asyncio.to_thread(shm.write_serialized, payload)
        return "full"

    async def get_history(self, limit: int = 100) -> list[AuraState]:
        """Read recent persisted state snapshots from the shared DB connection."""
        db = await self._ensure_db()

        try:
            async with db.execute(
                "SELECT state_json FROM state_log ORDER BY version DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
            return [self._deserialize(row[0]) for row in rows]
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.error("❌ [STATE] History retrieval failed: %s", e)
            return []

    async def _load_latest_state(self) -> None:
        """Hydrate the latest persisted state from the shared DB connection."""
        db = await self._ensure_db()

        try:
            async with db.execute(
                "SELECT state_json FROM state_log ORDER BY version DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                self._current = self._deserialize(row[0])
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.error("❌ [STATE] Failed to load latest: %s", e)

    async def _has_been_persisted(self) -> bool:
        db = await self._ensure_db()
        try:
            async with db.execute("SELECT COUNT(*) FROM state_log") as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0] > 0
        except _STATE_BOUNDARY_ERRORS as _e:
            _record_state_degradation(_e, action="state log persistence probe skipped")
            logger.debug("StateRepository persistence probe skipped: %s", _e)
        return False

    async def rollback(
        self, reason: str = "Unknown", *, expected_version: int | None = None
    ) -> AuraState | None:
        """Rollback to the last stable state in the log.

        ``expected_version`` is a compare-and-swap precondition: the caller
        says which version it believes is current, and a rollback is refused
        when the repository has moved past it. Without it a failed turn could
        revert a state some OTHER turn had committed in the meantime — the
        caller had no way to prove it authored what it was undoing.
        """
        async with self.lock:
            logger.warning("🚨 [STATE] Initiating Rollback. Reason: %s", reason)
            if expected_version is not None and self._current is not None:
                current_version = int(getattr(self._current, "version", 0) or 0)
                if current_version != int(expected_version):
                    logger.error(
                        "🛑 [STATE] Rollback refused: current version %d is not the "
                        "caller's expected %d; another turn has committed since.",
                        current_version,
                        int(expected_version),
                    )
                    return self._current
            history = await self.get_history(limit=2)
            if len(history) < 2:
                logger.error("🛑 [STATE] Rollback failed: Insufficient history.")
                return self._current

            # Revert to the state BEFORE the current one
            previous_state = history[1]
            if previous_state is None or (
                self._current and previous_state.version >= self._current.version
            ):
                logger.error(
                    "🛑 [STATE] Rollback failed: Previous state is not older than current."
                )
                return self._current
            # Derive a new 'stabilized' state from the previous one
            stabilized_state = await previous_state.derive_async(f"rollback: {reason}")

            # Commit the stabilized state
            try:
                serialized = self._serialize(stabilized_state)
                from core.governance_context import is_governed, local_internal_governed_scope

                if is_governed():
                    await self._commit_to_db(stabilized_state, serialized)
                else:
                    with local_internal_governed_scope(
                        "state_repository.rollback",
                        domain="state_mutation",
                        constraints={
                            "reason": str(reason or "unknown")[:160],
                            "operation": "rollback_persistence",
                        },
                    ):
                        await self._commit_to_db(stabilized_state, serialized)
                self._current = stabilized_state
                logger.info(
                    "✅ [STATE] Rollback complete. Restored to version %d", stabilized_state.version
                )
            except _STATE_BOUNDARY_ERRORS as e:
                _record_state_degradation(e)
                logger.error("🛑 [STATE] Rollback persistence failed: %s", e)

            return self._current

    @effect_sink("state.commit_to_db", allowed_domains=("state_mutation",))
    async def _commit_to_db(self, state: AuraState, serialized_data: str):
        """Persist a serialized state snapshot using the shared DB connection."""
        db = await self._ensure_db()
        for attempt in range(3):
            try:
                async with db.execute("BEGIN IMMEDIATE"):
                    await db.execute(
                        """INSERT OR REPLACE INTO state_log 
                           (state_id, version, parent_state_id, transition_cause, state_json, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            state.state_id,
                            state.version,
                            state.parent_state_id,
                            state.transition_cause,
                            serialized_data,
                            state.updated_at,
                        ),
                    )
                    await db.commit()
                logger.debug("💾 State v%s committed to Vault DB.", state.version)
                break
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e) and attempt < 2:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                raise

        # ── Long-Run Stability: scheduled pruning & VACUUM ────────────────
        self._commit_counter += 1
        if self._commit_counter % self.STATE_LOG_PRUNE_EVERY == 0:
            try:
                await self._prune_state_log(db)
            except _STATE_BOUNDARY_ERRORS as prune_err:
                _record_state_degradation(prune_err)
                logger.warning("⚠️ [STATE] State log pruning failed: %s", prune_err)
        if self._commit_counter % self.STATE_LOG_VACUUM_EVERY == 0:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._vacuum_sync)
            except _STATE_BOUNDARY_ERRORS as vacuum_err:
                _record_state_degradation(vacuum_err)
                logger.warning("⚠️ [STATE] VACUUM failed: %s", vacuum_err)

    async def _prune_state_log(self, db: aiosqlite.Connection) -> None:
        """Remove old state log rows, keeping only the most recent STATE_LOG_MAX_ROWS.

        Rows whose transition_cause contains 'checkpoint' or 'evolution' are always
        kept as historical anchors. This prevents the append-only log from growing
        unboundedly on long-running systems.
        """
        try:
            async with db.execute("SELECT COUNT(*) FROM state_log") as cursor:
                row = await cursor.fetchone()
                total = row[0] if row else 0
            if total <= self.STATE_LOG_MAX_ROWS:
                return

            excess = total - self.STATE_LOG_MAX_ROWS
            # Delete oldest rows that are NOT checkpoints or evolution markers
            await db.execute(
                """DELETE FROM state_log WHERE state_id IN (
                     SELECT state_id FROM state_log
                     WHERE transition_cause NOT LIKE '%checkpoint%'
                       AND transition_cause NOT LIKE '%evolution%'
                     ORDER BY version ASC
                     LIMIT ?
                   )""",
                (excess,),
            )
            await db.commit()
            logger.info(
                "🧹 [STATE] Pruned state log: removed up to %d of %d rows (keeping %d).",
                excess,
                total,
                self.STATE_LOG_MAX_ROWS,
            )
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.error("🛑 [STATE] Prune query failed: %s", e)

    def _vacuum_sync(self) -> None:
        """Run VACUUM synchronously in a thread to reclaim disk space."""
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
            logger.info("🧹 [STATE] VACUUM completed on %s.", self.db_path)
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.warning("⚠️ [STATE] VACUUM sync error: %s", e)

    def _circular_safe_asdict(self, obj, memo=None, depth=0) -> Any:
        """Recursive conversion to dict with cycle detection and depth guards."""
        if memo is None:
            memo = set()
            # Ensure recursion limit is sufficient for deep state trees
            import sys

            if sys.getrecursionlimit() < 2000:
                sys.setrecursionlimit(2000)

        # 1. Depth Guard
        if depth > 80:
            logger.error(
                "🚨 [STATE] Serialization depth limit exceeded (>80). CRITICAL RECURSION RISK."
            )
            return f"<DEPTH_LIMIT_REACHED: {type(obj).__name__}>"

        # 2. Cycle Detection
        obj_id = id(obj)
        # We only track complexity/containers for cycles
        is_container = dataclasses.is_dataclass(obj) or isinstance(obj, (dict, list, tuple))

        if is_container:
            if obj_id in memo:
                logger.warning(
                    "♻️ [STATE] Circular reference detected: %s (id=%d)", type(obj).__name__, obj_id
                )
                return f"<CircularReference: {type(obj).__name__}>"
            memo.add(obj_id)

        try:
            # 3. Type Dispatch
            if dataclasses.is_dataclass(obj):
                result = {}
                for f in dataclasses.fields(obj):
                    # Skip private fields if they leaked in
                    if f.name.startswith("_"):
                        continue
                    value = getattr(obj, f.name)
                    result[f.name] = self._circular_safe_asdict(value, memo, depth + 1)
                return result

            elif isinstance(obj, dict):
                return {
                    str(k): self._circular_safe_asdict(v, memo, depth + 1) for k, v in obj.items()
                }

            elif isinstance(obj, (list, tuple)):
                return [self._circular_safe_asdict(i, memo, depth + 1) for i in obj]

            elif isinstance(obj, Enum):
                return obj.value

            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj

            else:
                # Prevent recursion via __str__/repr on unknown objects
                type_name = type(obj).__name__
                if depth > 40:
                    return f"<{type_name} @ depth {depth}>"
                return str(obj)
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.error("🛑 [STATE] Item serialization error: %s", e)
            return f"<ERROR: {type(obj).__name__}>"
        finally:
            if is_container:
                memo.remove(obj_id)

    def _serialize(self, state: AuraState) -> str:
        """Harden serialization to prevent infinite recursion from state pollution."""
        if state is None:
            raise ValueError("Cannot serialize None state")
        if not state.state_id:
            logger.warning("[STATE] Serializing state with missing state_id. Assigning default.")
            state.state_id = f"st_{int(time.time() * 1000)}"

        try:
            d = self._circular_safe_asdict(state)
            return json.dumps(d, ensure_ascii=False)
        except _STATE_BOUNDARY_ERRORS as e:
            _record_state_degradation(e)
            logger.error("🛑 [STATE] Hard serialization failure: %s", e)
            raise

    def _deserialize(self, json_str: str) -> AuraState:
        from core.unity.unity_state import UnityState

        from .aura_state import (
            AffectVector,
            AuraState,
            CognitiveContext,
            CognitiveMode,
            ColdStore,
            CurriculumItem,
            IdentityKernel,
            MotivationState,
            PhenomenalField,
            SomaState,
            WorldModel,
        )

        data = json.loads(json_str)
        # Reconstruct nested dataclasses with safety defaults
        data["identity"] = IdentityKernel(**data.get("identity", {}))
        data["affect"] = AffectVector(**data.get("affect", {}))

        cog = data.get("cognition", {})
        legacy_pending_intents = data.pop("pending_intents", None)
        if "current_mode" in cog:
            cog["current_mode"] = CognitiveMode(cog["current_mode"])
        phenomenal = cog.get("phenomenal_state")
        if isinstance(phenomenal, dict):
            cog["phenomenal_state"] = PhenomenalField(**phenomenal)
        unity_state = cog.get("unity_state")
        if isinstance(unity_state, dict):
            cog["unity_state"] = UnityState.from_dict(unity_state)
        if legacy_pending_intents and "pending_intents" not in cog:
            cog["pending_intents"] = legacy_pending_intents
        data["cognition"] = CognitiveContext(**cog)

        data["world"] = WorldModel(**data.get("world", {}))
        data["soma"] = SomaState(**data.get("soma", {}))
        data["motivation"] = MotivationState(**data.get("motivation", {}))

        # ColdStore hydration (including CurriculumItems)
        cold_data = data.get("cold", {})
        curriculum_data = cold_data.get("training_curriculum", [])
        cold_data["training_curriculum"] = [CurriculumItem(**item) for item in curriculum_data]
        data["cold"] = ColdStore(**cold_data)

        # Health field reconstruction
        data["health"] = data.get(
            "health", {"circuits": {}, "capabilities": {}, "watchdog_timestamp": time.time()}
        )

        # Remove transport-injected fields
        data.pop("_bus_id", None)
        data.pop("_transport_snapshot_kind", None)

        return AuraState(**data)
