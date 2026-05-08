from core.runtime.errors import record_degradation
import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Memory.Vector")

_SQLITE_FALLBACK_ERRORS = (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError)


# ---------------------------------------------------------------------------
# Try to import ChromaDB — if missing, set a flag for sovereign fallback
# ---------------------------------------------------------------------------
_CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning("VectorMemory: ChromaDB unavailable. Falling back to Sovereign Mode. Error: %s", e)
    _CHROMA_AVAILABLE = False


class AuraEmbeddingFunction:
    """Consolidated embedding function using Aura's internal LLMs."""
    def __call__(self, input: List[str]) -> List[List[float]]:
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
        except Exception as e:
            record_degradation('vector_memory', e)
            logger.error("AuraEmbeddingFunction failed: %s", e)
            return [self._pseudo_embed(text) for text in input]

    def _pseudo_embed(self, text: str) -> List[float]:
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
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self._fallback_mode = False

        if persist_directory is None:
            from core.common.paths import DATA_DIR
            persist_directory = str(DATA_DIR / "vector_store")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.fallback_file = self.persist_directory / f"{collection_name}_fallback.json"
        
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
            except Exception as e:
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

    def _load_fallback(self) -> List[Dict[str, Any]]:
        """Load memories from local SQLite BLOB store with legacy JSON migration."""
        memories: List[Dict[str, Any]] = []
        if self._sqlite_vectors is not None:
            try:
                memories = self._sqlite_vectors.list_records()
            except _SQLITE_FALLBACK_ERRORS as e:
                record_degradation('vector_memory', e)
                logger.error("Failed to load fallback vectors from SQLite: %s", e)

        # Phase 8 Migration: Check if legacy JSON file exists
        if not memories and self.fallback_file.exists():
            try:
                logger.info("📦 Migrating '%s' memories from JSON to SQLite...", self.collection_name)
                with open(self.fallback_file, 'r') as f:
                    legacy_store = json.load(f)
                
                # Bulk insert into binary-vector SQLite storage.
                self._upsert_fallback_batch(legacy_store)
                return self._sqlite_vectors.list_records() if self._sqlite_vectors is not None else []
            except (OSError, json.JSONDecodeError, sqlite3.Error, RuntimeError, TypeError, ValueError) as e:
                record_degradation('vector_memory', e)
                logger.error("Failed to migrate legacy memory file: %s", e)

        return memories

    def _upsert_fallback_batch(self, memories: List[Dict[str, Any]]):
        """Persist a batch of entries to SQLite fallback efficiently."""
        if not memories or self._sqlite_vectors is None:
            return
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
        except _SQLITE_FALLBACK_ERRORS as e:
            record_degradation('vector_memory', e)
            logger.error("Failed to batch upsert fallback memories to DB: %s", e)

    def _upsert_fallback(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Persist a single entry to SQLite fallback."""
        if self._sqlite_vectors is None:
            return
        try:
            vector = AuraEmbeddingFunction()._pseudo_embed(content)
            self._sqlite_vectors.upsert(doc_id, content, vector, metadata=metadata)
        except _SQLITE_FALLBACK_ERRORS as e:
            record_degradation('vector_memory', e)
            logger.error("Failed to upsert fallback memory to DB: %s", e)

    def _fallback_keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
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

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        _id: Optional[str] = None,
    ) -> bool:
        """Persist a text memory with optional metadata and emotional state."""
        if not content:
            return False

        doc_id = _id or hashlib.sha256(content.encode()).hexdigest()[:16] + uuid.uuid4().hex[:8]
        meta = metadata or {}
        meta.setdefault("timestamp", time.time())
        
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
        except Exception as e:
            record_degradation('vector_memory', e)
            logger.debug("Emotional salience stamping failed: %s", e)

        if self._fallback_mode:
            self._store.append({"id": doc_id, "content": content, "metadata": meta})
            self._upsert_fallback(doc_id, content, meta)
            logger.debug("VectorMemory: Saved to fallback: %s...", content[:60])
            return True
        
        # Ensure last_accessed is set
        meta.setdefault("last_accessed", time.time())

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[meta],
            )
            logger.debug("VectorMemory.add_memory: %s...", content[:60])
            return True
        except Exception as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.add_memory failed: %s", e)
            return False

    def search_similar(self, query: str, limit: int = 5, **kwargs) -> List[Dict]:
        """Return semantically similar memories, biased by Emotional Salience."""
        if not query:
            return []

        if self._fallback_mode:
            if self._sqlite_vectors is not None:
                try:
                    embedder = AuraEmbeddingFunction()
                    query_vector = embedder._pseudo_embed(query)
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
                    for tid in top_ids:
                        idx = ids.index(tid)
                        m = metas[idx]
                        m["last_accessed"] = time.time()
                        self._collection.update(ids=[tid], metadatas=[m])
                except Exception as e:
                    record_degradation('vector_memory', e)
                    logger.debug("Failed to update last_accessed: %s", e)
            elif self._fallback_mode:
                for tid in top_ids:
                    for m in self._store:
                        if m["id"] == tid:
                            m["metadata"]["last_accessed"] = time.time()
                            self._upsert_fallback(m["id"], m["content"], m["metadata"])

            return scored_results[:limit]

        except Exception as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.search_similar failed: %s", e)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        if self._fallback_mode:
            total = self._sqlite_vectors.count() if self._sqlite_vectors is not None else len(self._store)
            return {"total_vectors": total, "engine": "sqlite_vector", "status": "active_local"}
        try:
            count = self._collection.count()
            return {"total_vectors": count, "engine": "chromadb", "status": "active"}
        except Exception as e:
            record_degradation('vector_memory', e)
            logger.debug("ChromaDB count failed: %s", e)
            return {"total_vectors": -1, "engine": "chromadb", "status": "error"}

    # ------------------------------------------------------------------
    # Aliases — compatibility
    # ------------------------------------------------------------------

    def add(self, content: str, metadata: Optional[Dict] = None, **kwargs) -> bool:
        return self.add_memory(content, metadata=metadata, _id=kwargs.get("_id"))

    async def index(self, content: str, metadata: Optional[Dict] = None, **kwargs) -> bool:
        """Async shim for MemoryManager compatibility."""
        return await asyncio.to_thread(self.add_memory, content, metadata=metadata, **kwargs)

    def search(self, query: str = "", limit: int = 5, k: int = 0, **kwargs) -> List[Dict]:
        effective_limit = k if k > 0 else limit
        return self.search_similar(query or "", limit=effective_limit)

    def clear(self):
        """Delete all vectors in the collection."""
        if self._fallback_mode:
            self._store.clear()
            if self._sqlite_vectors is not None:
                self._sqlite_vectors.clear()
            return
        try:
            ids = self._collection.get()["ids"]
            if ids:
                self._collection.delete(ids=ids)
            logger.info("VectorMemory: cleared collection '%s'", self.collection_name)
        except Exception as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.clear failed: %s", e)

    def prune_low_salience(self, threshold_days: int = 30, min_salience: float = -0.2) -> int:
        """Removes memories that are old, unaccessed, and have low emotional salience.
        
        This implements 'Strategic Forgetting'.
        """
        logger.info("🧹 Pruning low-salience memories (threshold=%s days)...", threshold_days)
        now = time.time()
        expiry_seconds = threshold_days * 86400
        neutral_expiry_seconds = max(expiry_seconds, min(45 * 86400, expiry_seconds * 2))
        ids_to_prune = []

        if self._fallback_mode:
            initial_count = len(self._store)
            self._store = [
                m for m in self._store
                if not (
                    (
                        (now - m["metadata"].get("last_accessed", m["metadata"].get("timestamp", 0)) > expiry_seconds)
                        and (m["metadata"].get("valence", 0.0) < min_salience)
                    )
                    or (
                        (now - m["metadata"].get("last_accessed", m["metadata"].get("timestamp", 0)) > neutral_expiry_seconds)
                        and (m["metadata"].get("valence", 0.0) <= 0.05)
                    )
                )
            ]
            final_count = len(self._store)
            if initial_count != final_count:
                if self._sqlite_vectors is not None:
                    self._sqlite_vectors.clear()
                    self._upsert_fallback_batch(self._store)
            return initial_count - final_count

        try:
            # We can't easily query by metadata logic 'AND' in all ChromaDB versions
            # So we fetch all metadatas and filter client-side (safe for < 100k records)
            results = self._collection.get(include=["metadatas"])
            ids = results.get("ids", [])
            metas = results.get("metadatas", [])

            for _id, meta in zip(ids, metas):
                last_access = meta.get("last_accessed", meta.get("timestamp", 0))
                valence = meta.get("valence", 0.0)
                
                # Expiry condition: Old unaccessed AND low salience
                if (
                    ((now - last_access > expiry_seconds) and (valence < min_salience))
                    or ((now - last_access > neutral_expiry_seconds) and (valence <= 0.05))
                ):
                    ids_to_prune.append(_id)

            if ids_to_prune:
                # Delete in chunks to avoid overwhelming the DB
                for i in range(0, len(ids_to_prune), 100):
                    self._collection.delete(ids=ids_to_prune[i:i+100])
                logger.info("✅ Pruned %s low-salience vectors.", len(ids_to_prune))
            
            return len(ids_to_prune)
        except Exception as e:
            record_degradation('vector_memory', e)
            logger.error("VectorMemory.prune_low_salience failed: %s", e)
            return 0

# Alias for compatibility
VectorStorage = VectorMemory
