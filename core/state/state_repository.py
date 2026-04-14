from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiosqlite

from core.runtime.effect_boundary import effect_sink
from core.runtime.background_policy import is_user_facing_origin

from ..bus.shared_mem_bus import SharedMemoryTransport
from ..container import ServiceContainer

if TYPE_CHECKING:
    from .aura_state import (
        AuraState,
    )

logger = logging.getLogger(__name__)


def get_state_shm_size_bytes() -> int:
    """Scale the state SHM segment to the host instead of pinning it at 2MB."""
    override = os.getenv("AURA_STATE_SHM_BYTES")
    if override:
        try:
            return max(2 * 1024 * 1024, int(override))
        except ValueError:
            logger.warning("Invalid AURA_STATE_SHM_BYTES override: %r", override)

    try:
        import psutil

        total_gb = psutil.virtual_memory().total / float(1024 ** 3)
    except Exception:
        total_gb = 0.0

    if total_gb >= 48:
        return 16 * 1024 * 1024
    if total_gb >= 24:
        return 8 * 1024 * 1024
    return 2 * 1024 * 1024

class StateVersionConflictError(Exception):
    """Raised when a state commit is rejected due to version stagnation or backtrack."""
    def __init__(self, current_v: int, rejected_v: int, cause: str):
        self.current_v = current_v
        self.rejected_v = rejected_v
        self.cause = cause
        super().__init__(f"State version conflict: current={current_v}, rejected={rejected_v} (cause={cause})")

class StateRepository:
    """
    Persists and retrieves AuraState.
    The 'continuity' is here — not in the LLM context window.
    
    Uses an append-only log so the full history of Aura's
    state transitions is recoverable. This IS the long-term memory
    of experience (episodic), separate from semantic memory (vector store).
    """

    # ── Long-Run Stability Config ──────────────────────────────────────────
    STATE_LOG_MAX_ROWS = 500           # Keep last N state versions
    STATE_LOG_PRUNE_EVERY = 100        # Prune check interval (commits)
    STATE_LOG_VACUUM_EVERY = 1000      # VACUUM interval (commits)
    DB_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024
    TRANSPORT_SNAPSHOT_MAX_ITEMS = 64
    TRANSPORT_SNAPSHOT_MAX_TEXT = 4096
    TRANSPORT_WORKING_MEMORY_LIMIT = 36
    TRANSPORT_LONG_TERM_MEMORY_LIMIT = 12
    TRANSPORT_GOAL_LIMIT = 12
    TRANSPORT_PERCEPT_LIMIT = 48

    def __init__(self, db_path: str = "data/aura_state.db", is_vault_owner: bool = False):
        self.db_path = db_path
        self.is_vault_owner = is_vault_owner
        self._current: Optional[AuraState] = None
        self._lock: Optional[asyncio.Lock] = None
        self._mutation_queue_maxsize = 32
        self._mutation_queue: asyncio.Queue = asyncio.Queue(maxsize=self._mutation_queue_maxsize)
        self._is_processing = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._buffer: Dict[str, list] = {} # Per-trace buffer for causal ordering
        self._shm: Optional[SharedMemoryTransport] = None
        self._db: Optional[aiosqlite.Connection] = None
        self._transport: Any = None
        self._dropped_commit_count = 0
        self._commit_counter = 0       # Tracks commits for prune/VACUUM scheduling
        self._last_commit_at = 0.0
        self._last_commit_duration_ms = 0.0
        self._last_serialization_ms = 0.0
        self._last_consumer_activity_at = 0.0
        self._repair_count = 0
        self._last_shm_write_mode = "idle"
        self._last_shm_overflow_bytes = 0


    @property
    def lock(self) -> Any:
        if self._lock is None:
            from core.utils.concurrency import get_robust_lock
            self._lock = get_robust_lock(f"StateRepository:{'Owner' if self.is_vault_owner else 'Proxy'}")
        return self._lock

    async def _ensure_db(self) -> aiosqlite.Connection:
        """[CF-5] Ensures the DB connection is alive and bound to the current loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("Database access attempted outside of event loop")

        if self._db is not None:
            # Check for loop mismatch
            bound_loop = getattr(self._db, '_get_loop', lambda: getattr(self._db, '_loop', None))()
            if bound_loop != current_loop:
                logger.debug("Loop mismatch detected in StateRepository DB connection. Reconnecting.")
                try:
                    if self._db is not None:
                        await self._db.close()
                except Exception as _e:
                    logger.debug('Ignored Exception in state_repository.py: %s', _e)
                self._db = None

        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")
            
        return self._db

    async def initialize(self) -> None:
        from .aura_state import AuraState
        serialized_current: Optional[str] = None
        boot_governance_decision = SimpleNamespace(
            receipt_id="state_repository_bootstrap",
            domain="state_mutation",
            source="state_repository.initialize",
            constraints={"boot_phase": "initialize"},
        )
        if self.is_vault_owner:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
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
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_version ON state_log(version)"
            )
            await db.commit()
            # Load latest from DB
            await self._load_latest_state()
            
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
            self._shm = SharedMemoryTransport(name="aura_state_shm", size=get_state_shm_size_bytes())
            if self._shm:
                try:
                    await self._shm.create()
                except PermissionError as e:
                    logger.warning("⚠️ [STATE] Shared memory unavailable in this runtime. Continuing without SHM: %s", e)
                    self._shm = None
                except OSError as e:
                    logger.warning("⚠️ [STATE] Shared memory initialization failed. Continuing without SHM: %s", e)
                    self._shm = None

                # Synchronously write to SHM so it's ready before MindTick boots
                if self._shm:
                    try:
                        if serialized_current is None:
                            serialized_current = await asyncio.to_thread(self._serialize, self._current)
                        from core.governance_context import governed_scope

                        async with governed_scope(boot_governance_decision):
                            shm_mode = await self._sync_to_shm(self._current, serialized_current)
                        if shm_mode == "full":
                            logger.info("✓ [STATE] Genesis state pushed to SHM.")
                        elif shm_mode == "marker":
                            logger.info("✓ [STATE] Genesis overflow marker pushed to SHM.")
                    except Exception as e:
                        logger.warning(f"⚠️ [STATE] Initial SHM write failed: {e}")

            logger.info("✓ [STATE] Vault Owner Initialized with SHM for writing.")
            
            # Start consumer
            self._is_processing = True
            self._consumer_task = asyncio.create_task(
                self._mutation_consumer_loop(),
                name="vault_mutation_consumer"
            )
        else:
            self._transport = self._resolve_transport()
            # Proxy Mode: Attach to SHM for reading
            self._shm = SharedMemoryTransport(name="aura_state_shm", size=get_state_shm_size_bytes())
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
                except Exception as e:
                    logger.warning(f"⚠️ [STATE] Failed to attach to SHM, falling back to boot state: {e}")
                    self._shm = None
            if not self._current:
                await self._fetch_state_from_vault()
            if not self._current:
                from .aura_state import AuraState
                self._current = AuraState()
                logger.warning("⚠️ [STATE] Proxy could not hydrate from SHM or Vault. Using local boot snapshot.")

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
        has_actor = getattr(transport, "has_actor", None)
        if callable(has_actor):
            return bool(has_actor("state_vault"))
        return "state_vault" in getattr(transport, "_transports", {})

    async def _fetch_state_from_vault(self) -> Optional["AuraState"]:
        """Fallback path when SHM is not yet readable: request the canonical state from the vault actor."""
        transport = self._resolve_transport()
        if not transport:
            logger.debug("🔄 [STATE] ActorBus unavailable; cannot fetch state from Vault yet.")
            return self._current
        if not self._transport_has_vault():
            logger.debug("🔄 [STATE] ActorBus present but state_vault transport not registered yet.")
            return self._current

        try:
            logger.info("🔄 [STATE] SHM empty. Requesting full state fetch from Vault...")
            res = await transport.request("state_vault", "get_state", {"full": True})
            if isinstance(res, dict) and res.get("state"):
                self._current = self._deserialize(json.dumps(res["state"]))
                logger.info("✓ [STATE] Full state fetched from Vault via Bus.")
        except Exception as e:
            logger.error(f"❌ [STATE] Full fetch failed: {e}")

        return self._current


    async def commit(self, new_state: AuraState, cause: str, trace_id: Optional[str] = None) -> AuraState:
        """
        Strangler Fig: Transition to a new state.
        Now enqueues a mutation command for atomic processing.
        """
        trace_id = trace_id or f"trace_{int(time.time() * 1000)}"
        if self.is_vault_owner:
            await self._enqueue_owner_commit({
                "type": "commit",
                "state": new_state,
                "cause": cause,
                "trace_id": trace_id,
                "ts": time.time()
            })
            return new_state  # Processor keeps the 'live' reference
        
        # Proxy to owner via shared bus
        transport = self._resolve_transport()
        if transport:
            state_dict = await asyncio.to_thread(self._circular_safe_asdict, new_state)
            last_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    await transport.request(
                        "state_vault",
                        "commit",
                        {
                            "state": state_dict,
                            "cause": cause,
                            "trace_id": trace_id,
                        },
                    )
                    return new_state
                except Exception as e:
                    last_error = e
                    logger.warning("❌ [STATE] Proxy Commit Request FAILED (attempt %d/2): %s", attempt + 1, e)
                    self._transport = None
                    transport = self._resolve_transport()
                    if attempt == 0:
                        await asyncio.sleep(0.2)
                        continue
            if last_error:
                raise last_error
        else:
            logger.error("🛑 [STATE] Commit failed: ActorBus/Transport missing in Proxy Mode")
        
        return new_state

    async def _enqueue_owner_commit(self, payload: Dict[str, Any]) -> None:
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
        while True:
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

    async def get_current(self) -> Optional[AuraState]:
        """Async-compatible alias for get_state (Internal API Standardization)."""
        return await self.get_state()

    async def get_state(self) -> Optional[AuraState]:
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
                        except Exception as e:
                            logger.error(f"Failed to auto-sync from SHM: {e}")
            except Exception as e:
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
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
            except Exception as e:
                logger.debug("StateRepository consumer shutdown issue: %s", e)
            finally:
                self._consumer_task = None

        if self._shm:
            try:
                self._shm.close()
            except Exception as e:
                logger.debug("StateRepository SHM close issue: %s", e)
            finally:
                self._shm = None

        if self._db is not None:
            try:
                await self._db.close()
            except Exception as e:
                logger.debug("StateRepository DB close issue: %s", e)
            finally:
                self._db = None

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
                except Exception as e:
                    logger.error("🛑 Error in mutation consumer: %s", e)
                    # small backoff to avoid hot-loop on repeated failure
                    await asyncio.sleep(0.1)
        finally:
            logger.info("🧠 Mutation consumer exiting.")

    async def _process_commit(self, new_state: AuraState, cause: str):
        """Internal atomic processing of a commit. - [UNIFICATION OPTIMIZED]"""
        commit_started = time.perf_counter()
        governance_decision = None
        try:
            from core.constitution import get_constitutional_core, unpack_governance_result

            new_state.health = copy.deepcopy(getattr(new_state, "health", {}) or {})
            if hasattr(new_state, "compact"):
                try:
                    new_state.compact()
                except Exception as exc:
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
                logger.warning(
                    "🚫 [STATE] ConstitutionalCore blocked state mutation (origin=%s cause=%s reason=%s)",
                    getattr(new_state, "transition_origin", "system"),
                    cause,
                    reason,
                )
                return
        except Exception as exc:
            logger.debug("Constitutional state gate unavailable: %s", exc)

        # 1. Serialize OUTSIDE the lock (O(n) walk) - Offload to thread
        try:
            start_ser = time.perf_counter()
            if self._should_use_bounded_db_snapshot(new_state, cause):
                serialized_data = await asyncio.to_thread(self._serialize_transport_snapshot, new_state)
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
                    serialized_data = await asyncio.to_thread(self._serialize_transport_snapshot, new_state)
            ser_ms = (time.perf_counter() - start_ser) * 1000
            self._last_serialization_ms = ser_ms
            if ser_ms > 20:
                logger.warning("📉 [STATE] Heavy Serialization Detected: %.2fms", ser_ms)
        except Exception as e:
            logger.error("🛑 [STATE] Serialization failed: %s", e)
            return

        async with self.lock:
            current = self._current
            previous_pending = list(getattr(getattr(current, "cognition", None), "pending_initiatives", []) or []) if current else []
            # Atomic Version Guard
            if current and new_state.version <= current.version:
                if cause != "bootstrap":
                    logger.debug(
                        "[STATE] Atomic Guard Reject: Version %d <= current %d (Cause: %s)",
                        new_state.version, self._current.version, cause
                    )
                    return

            new_state.transition_cause = cause
            new_state.updated_at = time.time()
            
            # --- ATOMIC MEMORY UPDATE ---
            self._current = new_state
            logger.debug("💾 [STATE] Memory state updated to v%d. Releasing lock for IO.", new_state.version)

        try:
            from core.constitution import ProposalKind, get_constitutional_core

            current_pending = list(getattr(getattr(new_state, "cognition", None), "pending_initiatives", []) or [])
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
                    source=str(item.get("source") or getattr(new_state, "transition_origin", "system")),
                    summary=str(item.get("goal") or item.get("type") or "initiative"),
                    outcome="recorded",
                    reason=f"state_commit:{cause}",
                    target="pending_initiatives",
                    payload=item,
                    state=new_state,
                )
        except Exception as exc:
            logger.debug("Initiative proposal audit skipped: %s", exc)

        # 2. PROCEED OUTSIDE LOCK: publish SHM + DB inline within the single
        # consumer instead of spawning unbounded write tasks. The queue already
        # gives us async decoupling from foreground chat, and inline writes keep
        # long uptimes from degenerating into thousands of pending DB/SHM tasks.
        try:
            if governance_decision is not None:
                from core.governance_context import governed_scope

                async with governed_scope(governance_decision):
                    if self._shm:
                        try:
                            await self._sync_to_shm(new_state, serialized_data)
                        except Exception as exc:
                            logger.warning("⚠️ [STATE] SHM propagation failed: %s", exc)
                    await self._commit_to_db(new_state, serialized_data)
            else:
                if self._shm:
                    try:
                        await self._sync_to_shm(new_state, serialized_data)
                    except Exception as exc:
                        logger.warning("⚠️ [STATE] SHM propagation failed: %s", exc)
                await self._commit_to_db(new_state, serialized_data)
        except Exception as exc:
            logger.error("🛑 [STATE] Vault persistence failed: %s", exc)
        finally:
            self._last_commit_at = time.time()
            self._last_commit_duration_ms = (time.perf_counter() - commit_started) * 1000.0

    @staticmethod
    def _is_user_facing_origin(origin: Any) -> bool:
        return is_user_facing_origin(origin)

    def _should_use_bounded_db_snapshot(self, state: AuraState, cause: str) -> bool:
        origin = getattr(state, "transition_origin", "") or getattr(getattr(state, "cognition", None), "current_origin", "")
        if not self._is_user_facing_origin(origin):
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

    def get_runtime_status(self) -> Dict[str, Any]:
        local_consumer_alive = bool(self._consumer_task and not self._consumer_task.done())
        shm_attached = bool(self._shm is not None)
        state_available = self._current is not None
        vault_transport_available = False
        if not self.is_vault_owner:
            try:
                vault_transport_available = bool(self._transport_has_vault())
            except Exception:
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
            "last_serialization_ms": float(self._last_serialization_ms or 0.0),
            "last_consumer_activity_at": float(self._last_consumer_activity_at or 0.0),
            "repair_count": int(self._repair_count),
            "last_shm_write_mode": str(self._last_shm_write_mode),
            "last_shm_overflow_bytes": int(self._last_shm_overflow_bytes),
            "shm_attached": shm_attached,
            "state_available": state_available,
            "vault_transport_available": vault_transport_available,
        }

    async def repair_runtime(self) -> Dict[str, Any]:
        actions: List[str] = []

        if self.is_vault_owner and self._is_processing and (self._consumer_task is None or self._consumer_task.done()):
            self._consumer_task = asyncio.create_task(
                self._mutation_consumer_loop(),
                name="vault_mutation_consumer",
            )
            self._repair_count += 1
            actions.append("restarted_consumer")

        if self.is_vault_owner and self._db is None:
            await self._ensure_db()
            self._repair_count += 1
            actions.append("reconnected_db")

        if not self.is_vault_owner and self._current is None:
            try:
                await self._fetch_state_from_vault()
            except Exception:
                pass
            if self._current is not None:
                self._repair_count += 1
                actions.append("rehydrated_proxy")

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

    def _truncate_transport_text(self, value: Any, *, limit: int | None = None) -> Any:
        if not isinstance(value, str):
            return value
        max_len = int(limit or self.TRANSPORT_SNAPSHOT_MAX_TEXT)
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
        item_limit = int(max_items or self.TRANSPORT_SNAPSHOT_MAX_ITEMS)
        text_limit = int(max_text or self.TRANSPORT_SNAPSHOT_MAX_TEXT)

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
            identity["concept_graph"] = self._bounded_transport_value(identity.get("concept_graph", {}))

        cognition = snapshot.get("cognition")
        if isinstance(cognition, dict):
            cognition["working_memory"] = self._bounded_transport_value(
                list(cognition.get("working_memory", []) or [])[-self.TRANSPORT_WORKING_MEMORY_LIMIT:],
                max_items=self.TRANSPORT_WORKING_MEMORY_LIMIT,
                prefer_tail=True,
            )
            cognition["long_term_memory"] = self._bounded_transport_value(
                list(cognition.get("long_term_memory", []) or [])[-self.TRANSPORT_LONG_TERM_MEMORY_LIMIT:],
                max_items=self.TRANSPORT_LONG_TERM_MEMORY_LIMIT,
                prefer_tail=True,
            )
            cognition["active_goals"] = self._bounded_transport_value(
                list(cognition.get("active_goals", []) or [])[-self.TRANSPORT_GOAL_LIMIT:],
                max_items=self.TRANSPORT_GOAL_LIMIT,
                prefer_tail=True,
            )
            cognition["pending_initiatives"] = self._bounded_transport_value(
                list(cognition.get("pending_initiatives", []) or [])[-self.TRANSPORT_GOAL_LIMIT:],
                max_items=self.TRANSPORT_GOAL_LIMIT,
                prefer_tail=True,
            )
            cognition["pending_intents"] = self._bounded_transport_value(
                list(cognition.get("pending_intents", []) or [])[-self.TRANSPORT_GOAL_LIMIT:],
                max_items=self.TRANSPORT_GOAL_LIMIT,
                prefer_tail=True,
            )
            cognition["rolling_summary"] = self._truncate_transport_text(cognition.get("rolling_summary"))
            cognition["last_response"] = self._truncate_transport_text(cognition.get("last_response"), limit=2048)
            cognition["modifiers"] = self._bounded_transport_value(cognition.get("modifiers", {}), max_items=48)

        world = snapshot.get("world")
        if isinstance(world, dict):
            world["known_entities"] = self._bounded_transport_value(world.get("known_entities", {}), max_items=96)
            world["relationship_graph"] = self._bounded_transport_value(world.get("relationship_graph", {}), max_items=96)
            world["recent_percepts"] = self._bounded_transport_value(
                list(world.get("recent_percepts", []) or [])[-self.TRANSPORT_PERCEPT_LIMIT:],
                max_items=self.TRANSPORT_PERCEPT_LIMIT,
                prefer_tail=True,
            )
            world["spatial_context"] = self._bounded_transport_value(world.get("spatial_context", {}), max_items=24)

        snapshot["health"] = self._bounded_transport_value(snapshot.get("health", {}), max_items=64)
        snapshot["response_modifiers"] = self._bounded_transport_value(snapshot.get("response_modifiers", {}), max_items=48)

        return json.dumps(snapshot, ensure_ascii=False)

    @effect_sink("state.sync_to_shm", allowed_domains=("state_mutation",))
    async def _sync_to_shm(self, state: AuraState, serialized_state: str) -> str:
        """Push serialized state into SHM without re-walking the object graph."""
        shm = self._shm
        if shm is None:
            self._last_shm_write_mode = "disabled"
            return "disabled"

        payload = serialized_state.encode("utf-8") if isinstance(serialized_state, str) else bytes(serialized_state)
        if len(payload) > shm.payload_capacity:
            hot_snapshot_payload: bytes | None = None
            try:
                hot_snapshot_payload = self._serialize_transport_snapshot(state).encode("utf-8")
            except Exception as exc:
                logger.warning("⚠️ [STATE] Failed to build bounded SHM hot snapshot: %s", exc)

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

    async def get_history(self, limit: int = 100) -> List[AuraState]:
        """[CF] Reusing self._db for history retrieval."""
        db = await self._ensure_db()
            
        try:
            async with db.execute(
                "SELECT state_json FROM state_log ORDER BY version DESC LIMIT ?",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
            return [self._deserialize(row[0]) for row in rows]
        except Exception as e:
            logger.error("❌ [STATE] History retrieval failed: %s", e)
            return []

    async def _load_latest_state(self) -> None:
        """[CF] Reusing self._db for reads."""
        db = await self._ensure_db()
            
        try:
            async with db.execute(
                "SELECT state_json FROM state_log ORDER BY version DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                self._current = self._deserialize(row[0])
        except Exception as e:
            logger.error("❌ [STATE] Failed to load latest: %s", e)

    async def _has_been_persisted(self) -> bool:
        db = await self._ensure_db()
        try:
            async with db.execute("SELECT COUNT(*) FROM state_log") as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0] > 0
        except Exception as _e:
            logger.debug('Ignored Exception in state_repository.py: %s', _e)
        return False
        
    async def rollback(self, reason: str = "Unknown") -> Optional[AuraState]:
        """Rollback to the last stable state in the log."""
        async with self.lock:
            logger.warning("🚨 [STATE] Initiating Rollback. Reason: %s", reason)
            history = await self.get_history(limit=2)
            if len(history) < 2:
                logger.error("🛑 [STATE] Rollback failed: Insufficient history.")
                return self._current
            
            # Revert to the state BEFORE the current one
            previous_state = history[1]
            if previous_state is None or (self._current and previous_state.version >= self._current.version):
                logger.error("🛑 [STATE] Rollback failed: Previous state is not older than current.")
                return self._current
            # Derive a new 'stabilized' state from the previous one
            stabilized_state = await previous_state.derive_async(f"rollback: {reason}")
            
            # Commit the stabilized state
            try:
                serialized = self._serialize(stabilized_state)
                await self._commit_to_db(stabilized_state, serialized)
                self._current = stabilized_state
                logger.info("✅ [STATE] Rollback complete. Restored to version %d", stabilized_state.version)
            except Exception as e:
                logger.error("🛑 [STATE] Rollback persistence failed: %s", e)
            
            return self._current

    @effect_sink("state.commit_to_db", allowed_domains=("state_mutation",))
    async def _commit_to_db(self, state: AuraState, serialized_data: str):
        """[CF] Using self._db instead of opening a new connection per write."""
        db = await self._ensure_db()
        for attempt in range(3):
            try:
                async with db.execute("BEGIN IMMEDIATE"):
                    await db.execute(
                        """INSERT OR REPLACE INTO state_log 
                           (state_id, version, parent_state_id, transition_cause, state_json, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (state.state_id, state.version, state.parent_state_id, 
                         state.transition_cause, serialized_data, state.updated_at)
                    )
                    await db.commit()
                logger.debug(f"💾 State v{state.version} committed to Vault DB.")
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
            except Exception as prune_err:
                logger.warning("⚠️ [STATE] State log pruning failed: %s", prune_err)
        if self._commit_counter % self.STATE_LOG_VACUUM_EVERY == 0:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._vacuum_sync)
            except Exception as vacuum_err:
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
                excess, total, self.STATE_LOG_MAX_ROWS,
            )
        except Exception as e:
            logger.error("🛑 [STATE] Prune query failed: %s", e)

    def _vacuum_sync(self) -> None:
        """Run VACUUM synchronously in a thread to reclaim disk space."""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
            logger.info("🧹 [STATE] VACUUM completed on %s.", self.db_path)
        except Exception as e:
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
            logger.error("🚨 [STATE] Serialization depth limit exceeded (>80). CRITICAL RECURSION RISK.")
            return f"<DEPTH_LIMIT_REACHED: {type(obj).__name__}>"

        # 2. Cycle Detection
        obj_id = id(obj)
        # We only track complexity/containers for cycles
        is_container = dataclasses.is_dataclass(obj) or isinstance(obj, (dict, list, tuple))
        
        if is_container:
            if obj_id in memo:
                logger.warning("♻️ [STATE] Circular reference detected: %s (id=%d)", type(obj).__name__, obj_id)
                return f"<CircularReference: {type(obj).__name__}>"
            memo.add(obj_id)

        try:
            # 3. Type Dispatch
            if dataclasses.is_dataclass(obj):
                result = {}
                for f in dataclasses.fields(obj):
                    # Skip private fields if they leaked in
                    if f.name.startswith("_"): continue
                    value = getattr(obj, f.name)
                    result[f.name] = self._circular_safe_asdict(value, memo, depth + 1)
                return result
            
            elif isinstance(obj, dict):
                return {str(k): self._circular_safe_asdict(v, memo, depth + 1) for k, v in obj.items()}
            
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
        except Exception as e:
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
            state.state_id = f"st_{int(time.time()*1000)}"
            
        try:
            # Zenith-v6.3 Fix: Replace dataclasses.asdict with cycle-safe version
            d = self._circular_safe_asdict(state)
            return json.dumps(d, ensure_ascii=False)
        except Exception as e:
            logger.error("🛑 [STATE] Hard serialization failure: %s", e)
            raise

    def _deserialize(self, json_str: str) -> AuraState:
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
        data['identity'] = IdentityKernel(**data.get('identity', {}))
        data['affect'] = AffectVector(**data.get('affect', {}))
        
        cog = data.get('cognition', {})
        legacy_pending_intents = data.pop('pending_intents', None)
        if 'current_mode' in cog:
            cog['current_mode'] = CognitiveMode(cog['current_mode'])
        phenomenal = cog.get("phenomenal_state")
        if isinstance(phenomenal, dict):
            cog["phenomenal_state"] = PhenomenalField(**phenomenal)
        if legacy_pending_intents and "pending_intents" not in cog:
            cog["pending_intents"] = legacy_pending_intents
        data['cognition'] = CognitiveContext(**cog)
        
        data['world'] = WorldModel(**data.get('world', {}))
        data['soma'] = SomaState(**data.get('soma', {}))
        data['motivation'] = MotivationState(**data.get('motivation', {}))
        
        # ColdStore hydration (including CurriculumItems)
        cold_data = data.get('cold', {})
        curriculum_data = cold_data.get('training_curriculum', [])
        cold_data['training_curriculum'] = [CurriculumItem(**item) for item in curriculum_data]
        data['cold'] = ColdStore(**cold_data)
        
        # Health field reconstruction
        data['health'] = data.get('health', {
            "circuits": {},
            "capabilities": {},
            "watchdog_timestamp": time.time()
        })
        
        # Remove transport-injected fields
        data.pop('_bus_id', None)
        data.pop('_transport_snapshot_kind', None)
        
        return AuraState(**data)
