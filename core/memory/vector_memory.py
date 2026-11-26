import asyncio
import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Memory.Vector")

_CHROMA_IMPORT_ERRORS = (ImportError, OSError, RuntimeError)
_SQLITE_FALLBACK_ERRORS = (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError)
_UPSERT_QUEUE_STOP = object()


# ---------------------------------------------------------------------------
# Try to import ChromaDB — if missing, set a flag for sovereign fallback
# ---------------------------------------------------------------------------
_CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except _CHROMA_IMPORT_ERRORS as e:
    logger.warning("VectorMemory: ChromaDB unavailable. Falling back to Sovereign Mode. Error: %s", e)
    _CHROMA_AVAILABLE = False


class AuraEmbeddingFunction:
    """Consolidated embedding function using Aura's internal LLMs."""

    VERSION = "aura-embedding-function-v1"
    METRIC = "cosine"

    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            from core.container import ServiceContainer
            adapter = ServiceContainer.get("api_adapter")
            if not adapter:
                # Return deterministic pseudo-embeddings if adapter missing
                return [self._pseudo_embed(text) for text in input]
            
            embeddings = []
            for text in input:
                embeddings.append(adapter.embed_sync(text))
            return embeddings
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('vector_memory', e)
            logger.error("AuraEmbeddingFunction failed: %s", e)
            return [self._pseudo_embed(text) for text in input]

    def _pseudo_embed(self, text: str) -> list[float]:
        """Bag-of-words hashing embedding that preserves semantic similarity.
        Texts sharing words will have proportional cosine similarity.
        """
        import hashlib

        import numpy as np
        dim = 768
        vec = np.zeros(dim, dtype=np.float64)
        words = text.lower().split()
        if not words:
            return vec.tolist()
        for word in words:
            clean = ''.join(c for c in word if c.isalnum())
            if not clean:
                continue
            weight = 1.0 + min(len(clean), 12) * 0.15
            for salt in (b"a", b"b", b"c"):
                h = hashlib.md5(salt + clean.encode()).digest()
                idx = int.from_bytes(h[:2], "big") % dim
                sign = 1.0 if h[2] & 1 else -1.0
                vec[idx] += sign * weight
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec.tolist()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VectorMemory:
    """Semantic vector store backed by ChromaDB + Internal Aura embeddings.
    Fails over to local SQLite BLOB persistence if ChromaDB is unavailable.
    """

    def __init__(
        self,
        collection_name: str = "aura_memories",
        persist_directory: str | None = None,
    ):
        self.collection_name = collection_name
        self._fallback_mode = False
        self._mutation_lock = checked_lock("vector_memory", reentrant=True)
        self.single_principal_collection = True
        self.embedding_metric = AuraEmbeddingFunction.METRIC
        self.embedding_version = AuraEmbeddingFunction.VERSION

        if persist_directory is None:
            from core.utils.paths import DATA_DIR
            persist_directory = str(DATA_DIR / "vector_store")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.fallback_file = self.persist_directory / f"{collection_name}_fallback.json"
        self._upsert_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._upsert_stop = threading.Event()
        self._upsert_thread: threading.Thread | None = None
        self._closed = False

        from core.utils.core_db import get_core_db
        self.db = get_core_db()
        self._sqlite_vectors = None

        if _CHROMA_AVAILABLE:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._embed_fn = AuraEmbeddingFunction()
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self._embed_fn,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "VectorMemory ONLINE — collection '%s' (%d vectors), persist=%s",
                    collection_name, self._collection.count(), persist_directory
                )
                self._upsert_thread = threading.Thread(
                    target=self._async_upsert_loop,
                    daemon=True,
                    name="ChromaUpsert",
                )
                self._upsert_thread.start()
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('vector_memory', e)
                logger.error("ChromaDB init failed, falling back to Sovereign Persistence: %s", e)
                self._fallback_mode = True
        else:
            self._fallback_mode = True

        if self._fallback_mode:
            from core.memory.sqlite_vector_store import SQLiteVectorStore
            self._sqlite_vectors = SQLiteVectorStore(
                self.persist_directory / "vectors.sqlite3",
                collection_name=self.collection_name,
            )
            self._store = self._load_fallback()
            logger.info("VectorMemory: Sovereign Fallback Active (records: %d)", len(self._store))

    def _load_fallback(self) -> list[dict[str, Any]]:
        """Load memories from local SQLite BLOB store with legacy JSON migration."""
        memories: list[dict[str, Any]] = []
        if self._sqlite_vectors is not None:
            try:
                memories = self._sqlite_vectors.list_records()
            except _SQLITE_FALLBACK_ERRORS as e:
                record_degradation('vector_memory', e)
                logger.error("Failed to load fallback vectors from SQLite: %s", e)

        # Phase 8 Migration: Check if legacy JSON file exists
        if not memories and self.fallback_file.exists():
            try:
                logger.info("Migrating '%s' memories from JSON to SQLite...", self.collection_name)
                with open(self.fallback_file, encoding="utf-8") as f:
                    legacy_store = json.load(f)
                
                # Bulk insert into binary-vector SQLite storage.
                self._upsert_fallback_batch(legacy_store)
                return self._sqlite_vectors.list_records() if self._sqlite_vectors is not None else []
            except (OSError, json.JSONDecodeError, sqlite3.Error, RuntimeError, TypeError, ValueError) as e:
                record_degradation('vector_memory', e)
                logger.error("Failed to migrate legacy memory file: %s", e)

        return memories

    def _upsert_fallback_batch(self, memories: list[dict[str, Any]]) -> bool:
        """Persist a batch of entries to SQLite fallback efficiently."""
        if not memories:
            return True
        if self._sqlite_vectors is None:
            return False
        try:
            records = []
            embedder = AuraEmbeddingFunction()
            for memory in memories:
                content = str(memory.get("content") or memory.get("text") or "")
                metadata = dict(memory.get("metadata") or {})
                vector = memory.get("vector")
                if vector is None:
                    vector = embedder._pseudo_embed(content)
                records.append((memory["id"], content, vector, metadata))
            self._sqlite_vectors.upsert_many(records)
            return True
        except _SQLITE_FALLBACK_ERRORS as e:
            record_degradation('vector_memory', e)
            logger.error("Failed to batch upsert fallback memories to DB: %s", e)
            return False

    def _upsert_fallback(self, doc_id: str, content: str, metadata: dict[str, Any]) -> bool:
        """Persist a single entry to SQLite fallback."""
        if self._sqlite_vectors is None:
            return False
        try:
            vector = AuraEmbeddingFunction()._pseudo_embed(content)
            self._sqlite_vectors.upsert(doc_id, content, vector, metadata=metadata)
            return True
        except _SQLITE_FALLBACK_ERRORS as e:
            record_degradation('vector_memory', e)
            logger.error("Failed to upsert fallback memory to DB: %s", e)
            return False

    def _fallback_keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Compatibility path for tests that inject ``_store`` directly."""
        query_words = set(query.lower().split())
        if not query_words:
            return []

        scored_memories = []
        for m in self._store:
            content_low = str(m.get('content', '')).lower()
            metadata = dict(m.get("metadata") or {})
            matches = sum(1 for w in query_words if w in content_low)
            if matches > 0:
                overlap = matches / len(query_words)
                recency = max(0, 1.0 - (time.time() - metadata.get('timestamp', 0)) / 86400) * 0.1
                scored_memories.append({**m, "score": overlap + recency})

        scored_memories.sort(key=lambda x: x['score'], reverse=True)
        return scored_memories[:limit]

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def _async_upsert_loop(self) -> None:
        """Background thread for ChromaDB upserts to avoid event loop blocking."""
        while not self._upsert_stop.is_set():
            try:
                batch = []
                item = self._upsert_queue.get(timeout=0.2)
                if item is _UPSERT_QUEUE_STOP:
                    break
                batch.append(item)
                while not self._upsert_stop.is_set() and len(batch) < 100:
                    try:
                        item = self._upsert_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _UPSERT_QUEUE_STOP:
                        self._upsert_stop.set()
                        break
                    batch.append(item)
                if batch:
                    ids = [i[0] for i in batch]
                    documents = [i[1] for i in batch]
                    metadatas = [i[2] for i in batch]
                    try:
                        with self._mutation_lock:
                            self._collection.upsert(
                                ids=ids,
                                documents=documents,
                                metadatas=metadatas,
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        record_degradation('vector_memory', e)
                        logger.error("Async Chroma upsert failed: %s", e)
            except queue.Empty:
                continue
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.debug("Async Chroma writer error: %s", e)
                time.sleep(1.0)

    def close(self, timeout_s: float = 2.0) -> None:
        """Stop the Chroma upsert worker deterministically."""
        if self._closed:
            return
        self._closed = True
        self._upsert_stop.set()
        if self._upsert_thread is None:
            return
        try:
            self._upsert_queue.put_nowait(_UPSERT_QUEUE_STOP)
        except queue.Full:
            try:
                self._upsert_queue.get_nowait()
                self._upsert_queue.put_nowait(_UPSERT_QUEUE_STOP)
            except queue.Empty:
                logger.debug("VectorMemory upsert queue drained before stop sentinel")
        if threading.current_thread() is self._upsert_thread:
            return
        if self._upsert_thread.is_alive():
            self._upsert_thread.join(timeout=max(0.0, float(timeout_s)))
            if self._upsert_thread.is_alive():
                logger.warning("VectorMemory upsert writer did not stop within %.2fs", timeout_s)

    def on_stop(self) -> None:
        self.close()

    def cleanup(self) -> None:
        self.close()

    def add_memory(
        self,
        content: str,
        metadata: dict | None = None,
        _id: str | None = None,
    ) -> bool:
        """Persist a text memory with optional metadata and emotional state."""
        if not content:
            return False

        doc_id = _id or hashlib.sha256(content.encode()).hexdigest()[:16] + uuid.uuid4().hex[:8]
        meta = dict(metadata or {})
        meta.setdefault("timestamp", time.time())
        meta.setdefault("memory_namespace", self.collection_name)
        meta.setdefault("embedding_metric", self.embedding_metric)
        meta.setdefault("embedding_version", self.embedding_version)
        
        # ── Pillar 4: Emotional Salience (Stamping) ──
        try:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect:
                # Synchronous-compatible access if possible, or skip for now
                # H-28 FIX: Use safe getattr to avoid AttributeError on V2 engine
                markers = getattr(affect, 'markers', None)
                if markers and hasattr(markers, 'get_wheel'):
                    wheel = markers.get_wheel()
                    if isinstance(wheel, dict) and "primary" in wheel:
                        w = wheel["primary"]
                        if isinstance(w, dict):
                            pos = w.get("joy", 0) + w.get("trust", 0)
                            neg = w.get("fear", 0) + w.get("sadness", 0) + w.get("anger", 0)
                            meta["valence"] = float(pos - neg)
                            meta["arousal"] = float(max(w.values()) if w else 0.0)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('vector_memory', e)
            logger.debug("Emotional salience stamping failed: %s", e)

        if self._fallback_mode:
            with self._mutation_lock:
                if not self._upsert_fallback(doc_id, content, meta):
                    return False
                self._store.append({"id": doc_id, "content": content, "metadata": meta})
            logger.debug("VectorMemory: Saved to fallback: %s...", content[:60])
            return True
        
        # Ensure last_accessed is set
        meta.setdefault("last_accessed", time.time())
        if self._closed or self._upsert_stop.is_set():
            logger.warning("VectorMemory add rejected because upsert writer is closed: %s", doc_id)
            return False

        try:
            self._upsert_queue.put_nowait((doc_id, content, meta))
            logger.debug("VectorMemory.add_memory (queued): %s...", content[:60])
            return True
        except queue.Full:
            logger.warning("VectorMemory upsert queue full, dropping memory: %s", doc_id)
            return False
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.add_memory enqueue failed: %s", e)
            return False

    def search_similar(self, query: str, limit: int = 5, **kwargs) -> list[dict]:
        """Return semantically similar memories, biased by Emotional Salience."""
        if not query:
            return []

        if self._fallback_mode:
            if self._sqlite_vectors is not None:
                try:
                    embedder = AuraEmbeddingFunction()
                    query_vector = embedder._pseudo_embed(query)
                    with self._mutation_lock:
                        results = self._sqlite_vectors.query(query_vector, limit=limit)
                        if results:
                            now = time.time()
                            out = []
                            for record in results:
                                metadata = dict(record.metadata)
                                metadata["last_accessed"] = now
                                self._sqlite_vectors.upsert(
                                    record.id,
                                    record.content,
                                    embedder._pseudo_embed(record.content),
                                    metadata=metadata,
                                )
                                out.append(
                                    {
                                        "id": record.id,
                                        "content": record.content,
                                        "metadata": metadata,
                                        "score": record.score,
                                        "distance": record.distance,
                                    }
                                )
                            self._store = self._sqlite_vectors.list_records()
                            return out
                except _SQLITE_FALLBACK_ERRORS as e:
                    record_degradation('vector_memory', e)
                    logger.error("SQLite vector fallback search failed: %s", e)
            return self._fallback_keyword_search(query, limit)

        try:
            # --- Pillar 4: Emotional Salience (Re-ranking) ---
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect_engine", default=None)
            current_valence = 0.0
            current_arousal = 0.0
            if affect:
                # H-28 FIX: Use safe getattr to avoid AttributeError on V2 engine
                markers = getattr(affect, 'markers', None)
                if markers and hasattr(markers, 'get_wheel'):
                    w = markers.get_wheel()["primary"]
                    current_valence = float((w.get("joy", 0) + w.get("trust", 0)) - (w.get("fear", 0) + w.get("sadness", 0) + w.get("anger", 0)))
                    current_arousal = float(max(w.values()) if w else 0.0)

            # Issue 111: Emotional Coloring - Arousal-based K-expansion
            # High arousal (panic/excitement) leads to broader, less precise associative leaps
            arousal_boost = int(current_arousal * limit * 2)
            internal_limit = (limit + arousal_boost) * 3
            
            results = self._collection.query(
                query_texts=[query],
                n_results=min(internal_limit, max(self._collection.count(), 1)),
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            # Already retrieved above due to k-expansion needs

            import numpy as np
            
            # Vectorized scoring
            num_res = len(docs)
            dist_arr = np.array(dists[:num_res]) if dists else np.zeros(num_res)
            # Ensure lengths match
            metas_clean = metas[:num_res] if metas else [{}] * num_res
            valence_arr = np.array([m.get("valence", 0.0) for m in metas_clean])
            
            # Semantic score = 1.0 - distance
            semantic_scores = 1.0 - dist_arr
            
            # Emotional alignment = 1.0 - (abs(diff) / 2.0)
            emotional_alignments = 1.0 - (np.abs(current_valence - valence_arr) / 2.0)
            
            # Final weighted score
            final_scores = (semantic_scores * 0.7) + (emotional_alignments * 0.3)
            
            scored_results = []
            for i in range(num_res):
                scored_results.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas_clean[i],
                    "score": float(final_scores[i]),
                    "distance": float(dists[i]) if i < len(dists) else None,
                })

            # Sort by final score descending
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            # --- Update last_accessed for the top results ---
            top_ids = [r["id"] for r in scored_results[:limit]]
            if not self._fallback_mode and top_ids:
                try:
                    # ChromaDB doesn't allow bulk metadata update by ID in the same way as SQL
                    # We have to fetch and update or use a loop for small sets
                    with self._mutation_lock:
                        for tid in top_ids:
                            idx = ids.index(tid)
                            m = metas[idx]
                            m["last_accessed"] = time.time()
                            self._collection.update(ids=[tid], metadatas=[m])
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('vector_memory', e)
                    logger.debug("Failed to update last_accessed: %s", e)
            elif self._fallback_mode:
                with self._mutation_lock:
                    for tid in top_ids:
                        for m in self._store:
                            if m["id"] == tid:
                                m["metadata"]["last_accessed"] = time.time()
                                self._upsert_fallback(m["id"], m["content"], m["metadata"])

            return scored_results[:limit]

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.search_similar failed: %s", e)
            return []

    def get_stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        if self._fallback_mode:
            total = self._sqlite_vectors.count() if self._sqlite_vectors is not None else len(self._store)
            return {"total_vectors": total, "engine": "sqlite_vector", "status": "active_local"}
        try:
            count = self._collection.count()
            return {"total_vectors": count, "engine": "chromadb", "status": "active"}
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('vector_memory', e)
            logger.debug("ChromaDB count failed: %s", e)
            return {"total_vectors": -1, "engine": "chromadb", "status": "error"}

    # ------------------------------------------------------------------
    # Aliases — compatibility
    # ------------------------------------------------------------------

    def add(self, content: str, metadata: dict | None = None, **kwargs) -> bool:
        return self.add_memory(content, metadata=metadata, _id=kwargs.get("_id"))

    async def index(self, content: str, metadata: dict | None = None, **kwargs) -> bool:
        """Async shim for MemoryManager compatibility."""
        return await asyncio.to_thread(self.add_memory, content, metadata=metadata, **kwargs)

    def search(self, query: str = "", limit: int = 5, k: int = 0, **kwargs) -> list[dict]:
        effective_limit = k if k > 0 else limit
        return self.search_similar(query or "", limit=effective_limit)

    def clear(self):
        """Delete all vectors in the collection."""
        if self._fallback_mode:
            with self._mutation_lock:
                if self._sqlite_vectors is not None:
                    self._sqlite_vectors.clear()
                self._store.clear()
            return
        try:
            with self._mutation_lock:
                ids = self._collection.get()["ids"]
                if ids:
                    self._collection.delete(ids=ids)
            logger.info("VectorMemory: cleared collection '%s'", self.collection_name)
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.clear failed: %s", e)

    def prune_low_salience(self, threshold_days: int = 30, min_salience: float = -0.2) -> int:
        """Removes memories that are old, unaccessed, and have low emotional salience.
        
        This implements 'Strategic Forgetting'.
        """
        logger.info("Pruning low-salience memories (threshold=%s days)...", threshold_days)
        now = time.time()
        expiry_seconds = threshold_days * 86400
        neutral_expiry_seconds = max(expiry_seconds, min(45 * 86400, expiry_seconds * 2))
        ids_to_prune = []

        def _is_stale(meta: dict) -> bool:
            """The forgetting rule, in one place.

            Two clocks: something actively unpleasant (valence below
            min_salience) expires at the normal threshold, while something
            merely neutral gets roughly twice as long before it goes. Defined
            once because it is applied by three different storage paths, and a
            rule that drifts between them means memories that survive in one
            configuration and vanish in another.
            """
            last_access = meta.get("last_accessed", meta.get("timestamp", 0)) or 0
            valence = meta.get("valence", 0.0) or 0.0
            age = now - float(last_access)
            return (age > expiry_seconds and float(valence) < min_salience) or (
                age > neutral_expiry_seconds and float(valence) <= 0.05
            )

        if self._fallback_mode:
            if self._sqlite_vectors is None:
                # No durable fallback store means the in-memory list IS the
                # store. Returning 0 here reported "nothing needed pruning"
                # for a configuration that could never prune anything, so this
                # VectorMemory grew without bound and said it was fine.
                with self._mutation_lock:
                    before = len(self._store)
                    self._store = [
                        item
                        for item in self._store
                        if not _is_stale(dict(item.get("metadata") or {}))
                    ]
                    return before - len(self._store)
            try:
                with self._mutation_lock:
                    rows = list(self._sqlite_vectors.iter_records())
                    expected_revisions: dict[str, float] = {}
                    for row in rows:
                        if _is_stale(dict(row.get("metadata") or {})):
                            expected_revisions[str(row["id"])] = float(row["updated_at"])
                    if expected_revisions:
                        self._sqlite_vectors.delete_records_atomic(
                            record_ids=expected_revisions,
                            expected_updated_at=expected_revisions,
                        )
                    self._store = self._sqlite_vectors.list_records()
                return len(expected_revisions)
            except _SQLITE_FALLBACK_ERRORS as e:
                record_degradation('vector_memory', e)
                logger.error("SQLite vector fallback prune failed: %s", e)
                return 0

        try:
            # We can't easily query by metadata logic 'AND' in all ChromaDB versions
            # So we fetch all metadatas and filter client-side (safe for < 100k records)
            with self._mutation_lock:
                results = self._collection.get(include=["metadatas"])
                ids = results.get("ids", [])
                metas = results.get("metadatas", [])

                for _id, meta in zip(ids, metas, strict=False):
                    last_access = meta.get("last_accessed", meta.get("timestamp", 0))
                    valence = meta.get("valence", 0.0)

                    if (
                        ((now - last_access > expiry_seconds) and (valence < min_salience))
                        or ((now - last_access > neutral_expiry_seconds) and (valence <= 0.05))
                    ):
                        ids_to_prune.append(_id)

                if ids_to_prune:
                    for i in range(0, len(ids_to_prune), 100):
                        self._collection.delete(ids=ids_to_prune[i:i+100])
            if ids_to_prune:
                logger.info("Pruned %s low-salience vectors.", len(ids_to_prune))
            
            return len(ids_to_prune)
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.prune_low_salience failed: %s", e)
            return 0

# Alias for compatibility
VectorStorage = VectorMemory
