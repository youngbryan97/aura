import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from core.governance_context import (
    get_active_governance,
    governance_runtime_active,
    require_governance,
)
from core.memory import embedding_model
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SemanticMemory")
_MEMORY_CONTROL_DOMAINS = ("memory_write", "state_mutation")
_HIDDEN_MEMORY_TAGS = ("contested", "false", "deleted")


class SemanticMemory:
    """Hybrid Semantic Memory System.

    Boots instantly in 'Lite Mode' (JSON keyword search), then upgrades
    to 'Vector Mode' (FAISS + SentenceTransformers) in a background thread
    when ML libraries are available.

    Thread-safe: all metadata mutations are protected by a lock.
    """

    def __init__(self, memory_dir: str = "memory_storage"):
        logger.info("🧠 Booting Semantic Memory (Hybrid Mode)...")

        self.memory_dir = memory_dir
        self.metadata_path = os.path.join(self.memory_dir, "aura_metadata.json")
        self.index_path = os.path.join(self.memory_dir, "aura_memory.index")

        # Thread safety
        self._lock = threading.Lock()
        self._vector_lock = threading.RLock()
        self._init_lock = threading.Lock()
        self._closing = False
        self._lane_lease: Any | None = None
        self._model_loading = False

        # State flags
        self.is_vector_ready = False
        self._init_error: str | None = None
        self.encoder = None
        self.index = None
        self.vector_dimension = 384

        os.makedirs(self.memory_dir, exist_ok=True)

        # 1. Immediate Lite Mode Init
        self.metadata: list[dict[str, Any]] = []
        self._load_metadata()

        # 2. Manual Upgrade (Deferred)
        logger.info("SemanticMemory initialized in Lite mode. Call await initialize() for vector upgrade.")

    async def initialize(self):
        """Perform async background startup tasks."""
        if not self.is_vector_ready and not self._closing:
            await self._async_background_start()

    async def _evict_vector_model(self, _owner: Any, reason: str) -> bool:
        if self._model_loading:
            return False

        def _release_if_idle() -> bool:
            if not self._vector_lock.acquire(blocking=False):
                return False
            try:
                return self._release_vector_model(
                    reason=f"semantic_memory_lane_eviction:{reason}"
                )
            finally:
                self._vector_lock.release()

        return await asyncio.to_thread(_release_if_idle)

    async def _compensate_vector_model(self, _owner: Any, reason: str) -> bool:
        if self._closing:
            return False
        logger.info("Restoring semantic embedding model after failed candidate: %s", reason)
        await self._async_background_start()
        return bool(self.is_vector_ready and self.encoder is not None)

    def _release_vector_model(self, *, reason: str) -> bool:
        with self._vector_lock:
            self.encoder = None
            self.is_vector_ready = False
            lease, self._lane_lease = self._lane_lease, None
            if lease is not None:
                lease.release(reason=reason)
            return self.encoder is None

    def on_stop(self) -> None:
        self._closing = True
        self._release_vector_model(reason="semantic_memory_stopped")

    # ── Status & Telemetry ──────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return current status for telemetry dashboards."""
        with self._lock:
            return {
                "mode": "vector" if self.is_vector_ready else "lite",
                "memory_count": len(self.metadata),
                "vector_ready": self.is_vector_ready,
                "init_error": self._init_error,
            }

    @property
    def memory_count(self) -> int:
        with self._lock:
            return len(self.metadata)

    # ── Persistence ─────────────────────────────────────────────────

    def _load_metadata(self):
        """Load metadata from disk (called once at init, no lock needed yet)."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.metadata = data
                    logger.info("Loaded %d memories (Lite Mode).", len(self.metadata))
                else:
                    logger.warning("Metadata file had unexpected format; starting fresh.")
                    self.metadata = []
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error loading metadata: %s", e)
            self.metadata = []

    def _save_metadata(self):
        """Persist metadata to disk.  Caller MUST hold self._lock."""
        try:
            atomic_write_text(
                self.metadata_path,
                json.dumps(self.metadata, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("Failed to save metadata: %s", e)

    # ── Background Vector Upgrade ───────────────────────────────────

    async def _async_background_start(self):
        """Wait for startup then offload heavy init to thread."""
        await asyncio.sleep(2)  # Non-blocking wait
        await asyncio.to_thread(self._background_init)

    def _background_init(self):
        """Load heavy ML libraries. No time.sleep here."""
        lane_lease = None
        self._init_lock.acquire()
        self._model_loading = True
        try:
            if self._closing:
                return
            logger.info("🧠 Starting Background Vector Engine Init...")

            # --- FAISS ---
            try:
                import faiss as _faiss
            except ImportError:
                logger.info("faiss-cpu not installed. Staying in Lite Mode.")
                self._init_error = "faiss not installed"
                return

            from core.runtime.model_lane_control import (
                ModelLaneControlError,
                acquire_synchronous_in_process_model_lane,
            )

            try:
                lane_lease = acquire_synchronous_in_process_model_lane(
                    owner_id=f"semantic-memory-encoder:{id(self)}",
                    model_path=embedding_model.REPO_ID,
                    purpose="serve",
                    request_gb=embedding_model.FOOTPRINT_GB,
                    priority=40,
                    preemptible=False,
                    evict=self._evict_vector_model,
                    compensate=self._compensate_vector_model,
                    metadata={
                        "engine": "semantic_memory",
                        "model_role": "embedding",
                        "activation_state": "loading",
                    },
                )
            except ModelLaneControlError as exc:
                self._init_error = f"model_lane_refused:{exc}"
                logger.warning("Semantic vector upgrade deferred by model lane: %s", exc)
                return

            # Load encoder (can take 5-10s on first run)
            logger.info("Loading Embedding Model (%s)...", embedding_model.REPO_ID)
            encoder = embedding_model.load_encoder(
                model_lane_lease=lane_lease,
            )

            # Build or load FAISS index
            if os.path.exists(self.index_path):
                logger.info("Loading FAISS Index from disk...")
                index = _faiss.read_index(self.index_path)
            else:
                logger.info("Creating fresh FAISS Index...")
                index = _faiss.IndexFlatL2(self.vector_dimension)
                # Re-index existing metadata
                with self._lock:
                    texts = [m["text"] for m in self.metadata if m.get("text")]
                if texts:
                    logger.info("Re-indexing %d existing memories...", len(texts))
                    embeddings = encoder.encode(texts, show_progress_bar=False)
                    _faiss.normalize_L2(embeddings)
                    index.add(embeddings.astype("float32"))
                    _faiss.write_index(index, self.index_path)

            # Commit — atomic swap
            with self._vector_lock:
                if self._closing:
                    lane_lease.release(reason="semantic_memory_closed_during_load")
                    return
                self.encoder = encoder
                self.index = index
                if not lane_lease.set_preemptible(True):
                    self.encoder = None
                    self.index = None
                    lane_lease.release(
                        reason="semantic_memory_activation_fence_lost"
                    )
                    raise RuntimeError("semantic_memory_activation_fence_lost")
                self._lane_lease = lane_lease
                lane_lease = None
                self.is_vector_ready = True
                self._init_error = None
            logger.info("🧠 Semantic Memory Upgraded: VECTOR MODE READY ⚡")

        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as e:
            if lane_lease is not None:
                lane_lease.release(reason="semantic_memory_model_load_failed")
            record_degradation('semantic_memory', e)
            logger.error("Background Vector Init Failed: %s", e, exc_info=True)
            self._init_error = str(e)
        finally:
            self._model_loading = False
            self._init_lock.release()
            logger.info("Continuing in Lite Mode (Keyword Search).")

    # ── Write ───────────────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async wrapper for add_memory."""
        await asyncio.to_thread(self.add_memory, content, context_tags=metadata)

    def add_memory(
        self,
        text: str,
        context_tags: dict[str, Any] | None = None,
    ) -> None:
        """Add a memory entry.  Thread-safe."""
        if not text or not text.strip():
            return

        try:
            with self._lock:
                memory_entry = {
                    "id": str(uuid.uuid4()),
                    "text": text.strip(),
                    "tags": context_tags or {},
                    "timestamp": time.time(),
                }
                self.metadata.append(memory_entry)
                self._save_metadata()

            # Vector index update (outside the metadata lock)
            try:
                with self._vector_lock:
                    encoder = self.encoder
                    index = self.index
                    if self.is_vector_ready and encoder and index:
                        import faiss as _faiss

                        vector = encoder.encode([text.strip()], show_progress_bar=False)
                        _faiss.normalize_L2(vector)
                        index.add(vector.astype("float32"))
                        _faiss.write_index(index, self.index_path)
            except (ImportError, AttributeError, RuntimeError) as ve:
                record_degradation('semantic_memory', ve)
                logger.warning("Vector add failed (data saved to JSON): %s", ve)

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('semantic_memory', e)
            logger.error("Failed to add memory: %s", e)

    # ── Read ────────────────────────────────────────────────────────

    def search_memories(self, query: str, top_k: int = 3) -> list[dict]:
        """Search memories. Uses vector search if available, else keyword fallback."""
        if not query or not query.strip():
            return []

        # Vector search path
        with self._vector_lock:
            encoder = self.encoder
            index = self.index
            vector_ready = self.is_vector_ready
        if vector_ready and encoder and index:
            try:
                import faiss as _faiss
                with self._vector_lock:
                    if encoder is not self.encoder or index is not self.index:
                        raise RuntimeError("semantic_vector_owner_changed_before_search")
                    query_vector = encoder.encode([query], show_progress_bar=False)
                    _faiss.normalize_L2(query_vector)
                    distances, indices = index.search(
                        query_vector.astype("float32"),
                        top_k,
                    )

                results = []
                with self._lock:
                    for dist, idx in zip(distances[0], indices[0], strict=True):
                        if idx != -1 and idx < len(self.metadata) and dist < 1.0:
                            entry = self.metadata[idx]
                            if self._is_hidden_memory(entry):
                                continue
                            results.append(entry)
                return results
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('semantic_memory', e)
                logger.error("Vector search error, falling back: %s", e)

        # Keyword fallback
        q_lower = query.lower()
        with self._lock:
            matches = []
            for m in self.metadata:
                if q_lower in m.get("text", "").lower():
                    if self._is_hidden_memory(m):
                        continue
                    matches.append(m)
        return matches[-top_k:]

    def edit_memory(self, record_id: str, new_text: str) -> bool:
        """Edit a memory's text content. Thread-safe."""
        text = str(new_text or "").strip()
        if not text:
            return False
        return self._mutate_memory_entry(record_id, "edit", {"text": text})

    def delete_memory(self, record_id: str) -> bool:
        """Mark a memory as deleted. Thread-safe."""
        return self._mutate_memory_entry(record_id, "delete", {"deleted": True})

    def freeze_memory(self, record_id: str, frozen: bool = True) -> bool:
        """Freeze or unfreeze a memory. Thread-safe."""
        return self._mutate_memory_entry(record_id, "freeze", {"frozen": bool(frozen)})

    def contest_memory(self, record_id: str, contested: bool = True) -> bool:
        """Flag a memory as contested. Thread-safe."""
        return self._mutate_memory_entry(record_id, "contest", {"contested": bool(contested)})

    def mark_false(self, record_id: str, is_false: bool = True) -> bool:
        """Flag a memory as false. Thread-safe."""
        return self._mutate_memory_entry(record_id, "mark_false", {"false": bool(is_false)})

    def get_provenance(self, record_id: str) -> dict[str, Any]:
        """Get the provenance chain/receipts for a memory."""
        with self._lock:
            for entry in self.metadata:
                if entry.get("id") == record_id:
                    tags = dict(entry.get("tags", {}) or {})
                    will_receipt_id = tags.get("will_receipt_id") or tags.get("receipt_id")
                    response = {
                        "id": record_id,
                        "text": entry.get("text"),
                        "timestamp": entry.get("timestamp"),
                        "tags": tags,
                        "will_receipt_id": will_receipt_id,
                        "receipts": [],
                    }
                    break
            else:
                return {}
        if will_receipt_id:
            from core.runtime.post_action_receipt import get_post_action_receipt_store

            post_store = get_post_action_receipt_store()
            post_receipts = post_store.get_by_will_id(will_receipt_id)
            for pr in post_receipts:
                response["receipts"].append(
                    {
                        "type": "post_action",
                        "receipt_id": pr.receipt_id,
                        "executor": pr.executor_name,
                        "outcome": pr.actual_outcome,
                        "welfare_transaction_id": pr.welfare_transaction_id,
                        "body_delta": pr.body_delta,
                        "timestamp": pr.timestamp,
                    }
                )
        return response

    def list_memory_records(self) -> list[dict[str, Any]]:
        """Return a metadata snapshot without exposing the internal lock."""
        with self._lock:
            return [dict(entry) for entry in self.metadata]

    @staticmethod
    def _is_hidden_memory(entry: dict[str, Any]) -> bool:
        tags = entry.get("tags", {}) or {}
        return bool(entry.get("deleted") or any(tags.get(tag) for tag in _HIDDEN_MEMORY_TAGS))

    def _require_memory_control_governance(self, operation: str) -> None:
        if governance_runtime_active():
            require_governance(
                f"semantic_memory.{operation}",
                strict=True,
                allowed_domains=_MEMORY_CONTROL_DOMAINS,
            )

    def _mutate_memory_entry(
        self,
        record_id: str,
        operation: str,
        updates: dict[str, Any],
    ) -> bool:
        record_id = str(record_id or "").strip()
        if not record_id:
            return False
        self._require_memory_control_governance(operation)
        token = get_active_governance()
        with self._lock:
            for entry in self.metadata:
                if entry.get("id") != record_id:
                    continue
                tags = entry.setdefault("tags", {})
                if tags.get("frozen") and operation != "freeze":
                    logger.warning("Attempted to %s frozen memory: %s", operation, record_id)
                    return False
                if "text" in updates:
                    entry["text"] = str(updates["text"]).strip()
                for tag in ("deleted", "frozen", "contested", "false"):
                    if tag in updates:
                        tags[tag] = bool(updates[tag])
                        if tag == "deleted":
                            entry["deleted"] = bool(updates[tag])
                tags["last_control_operation"] = operation
                tags["last_control_timestamp"] = time.time()
                if token is not None:
                    tags["last_control_receipt_id"] = token.receipt_id
                    tags["last_control_source"] = token.source
                self._save_metadata()
                if operation == "edit":
                    self._invalidate_vector_index("semantic_memory_edit")
                logger.info("Semantic memory %s applied to %s", operation, record_id)
                return True
        return False

    def _invalidate_vector_index(self, reason: str) -> None:
        if self.is_vector_ready:
            self.is_vector_ready = False
            self.index = None
            self._init_error = f"{reason}:vector_rebuild_required"

    # ── Consolidation ───────────────────────────────────────────────

    async def consolidate_from_history(self, history: list[dict[str, str]], cognitive_engine):
        """Summarise recent history into a long-term memory entry."""
        if not history or not cognitive_engine:
            return

        try:
            from core.brain.types import ThinkingMode
            text_to_summarize = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history[-10:]
            )
            prompt = (
                "Summarize key facts (under 30 words).\n"
                f"CONVERSATION:\n{text_to_summarize}\n\nSUMMARY:"
            )
            summary_thought = await cognitive_engine.think(
                prompt,
                mode=ThinkingMode.FAST,
                origin="semantic_memory_consolidation",
                is_background=True,
            )
            # Phase 34 FIX: Handle both dict and object returns
            if hasattr(summary_thought, "content"):
                summary = summary_thought.content
            elif isinstance(summary_thought, dict):
                summary = summary_thought.get("content", str(summary_thought))
            else:
                summary = str(summary_thought)
            if summary and "CONVERSATION:" not in summary:
                logger.info("Consolidating Memory: %s", summary[:80])
                self.add_memory(summary, context_tags={"source": "consolidation"})
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('semantic_memory', e)
            logger.error("Memory consolidation failed: %s", e)
