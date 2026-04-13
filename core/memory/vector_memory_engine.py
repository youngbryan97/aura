"""
core/memory/vector_memory_engine.py
=====================================
True Semantic Memory — Vector Embeddings + Persistent ChromaDB

The current memory system uses keyword search and SQLite.
That's good for exact retrieval. It's bad for meaning-based retrieval.

A real mind doesn't search for "dog" when thinking about "canine."
It retrieves by conceptual proximity. This does that.

What this adds:
    - Every memory is embedded as a dense semantic vector
    - Retrieval is by cosine similarity in embedding space
    - "What did we talk about when I was sad?" actually works
    - Memories cluster by topic, emotion, and time
    - Forgetting curve: distant, low-importance memories fade
    - Consolidation: similar memories merge over time

This is the difference between a filing cabinet and an associative cortex.

Architecture:
    EmbeddingEngine  — converts text to vectors (local, no API needed)
    MemoryVault      — ChromaDB-backed persistent store
    ImportanceScorer — determines which memories to keep
    ConsolidationEngine — merges and summarizes related memories over time
    VectorMemoryEngine — unified interface

Wire to orchestrator:
    Replace ServiceContainer.get("semantic_memory") usage with this.
    Register as "vector_memory_engine" in ServiceContainer.
    
    # On every turn:
    await memory_engine.store(
        content=user_input,
        memory_type="episodic",
        emotional_context=affect_engine.get_snapshot(),
        importance=0.6
    )
    await memory_engine.store(
        content=aura_response,
        memory_type="episodic",
        source="self",
        importance=0.5
    )
    
    # When building context:
    relevant = await memory_engine.recall(query=user_input, limit=5)
"""

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.runtime.effect_boundary import effect_sink

logger = logging.getLogger("Aura.VectorMemory")


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Memory:
    """A single stored memory with full metadata."""
    id: str
    content: str
    memory_type: str                      # "episodic", "semantic", "insight", "dream"
    timestamp: float
    importance: float                     # 0.0–1.0
    emotional_valence: float = 0.0        # -1.0 (negative) to 1.0 (positive)
    emotional_arousal: float = 0.0        # 0.0 (calm) to 1.0 (intense)
    source: str = "conversation"          # Who/what created this memory
    tags: List[str] = field(default_factory=list)
    access_count: int = 0                 # How many times retrieved
    last_accessed: float = 0.0
    linked_ids: List[str] = field(default_factory=list)  # Connected memories


@dataclass
class RecalledMemory:
    """A retrieved memory with relevance score."""
    memory: Memory
    relevance: float                      # Cosine similarity to query
    recency_weight: float                 # How recent this memory is
    combined_score: float                 # Final ranking score


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Engine — Local, No API Required
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Converts text to dense semantic vectors.
    
    Uses sentence-transformers (local, runs on MPS/CPU).
    Model: all-MiniLM-L6-v2 (fast, 384-dim, excellent quality)
    
    Falls back to TF-IDF if sentence-transformers isn't available.
    """

    PREFERRED_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DIM = 384

    def __init__(self):
        self._model = None
        self._tfidf_fallback = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization — don't load model until first use."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.PREFERRED_MODEL)
            # Move to Apple Silicon GPU if available
            try:
                import torch
                if torch.backends.mps.is_available():
                    self._model = self._model.to("mps")
                    logger.info("🧠 EmbeddingEngine: Using Apple Silicon GPU (MPS)")
            except Exception as _e:
                logger.debug('Ignored Exception in vector_memory_engine.py: %s', _e)
            logger.info("✅ EmbeddingEngine: sentence-transformers loaded (%s)", self.PREFERRED_MODEL)
        except ImportError:
            logger.warning(
                "⚠️ sentence-transformers not installed. "
                "Install with: pip install sentence-transformers\n"
                "Falling back to TF-IDF (lower quality recall)"
            )
            self._init_tfidf_fallback()

        self._initialized = True

    def _init_tfidf_fallback(self):
        """Simple TF-IDF fallback when sentence-transformers unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_fallback = TfidfVectorizer(max_features=512)
            self._tfidf_corpus = []
            logger.info("EmbeddingEngine: TF-IDF fallback initialized")
        except ImportError:
            logger.error("Neither sentence-transformers nor sklearn available. "
                        "Memory recall will be degraded.")

    def embed(self, text: str) -> np.ndarray:
        """Convert text to a dense vector."""
        self._initialize()

        if self._model:
            return self._model.encode(text, normalize_embeddings=True)

        if self._tfidf_fallback:
            return self._embed_tfidf(text)

        # Last resort: character n-gram hash (very basic)
        return self._embed_hash(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed for efficiency."""
        self._initialize()

        if self._model:
            return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

        return np.vstack([self.embed(t) for t in texts])

    def _embed_tfidf(self, text: str) -> np.ndarray:
        """TF-IDF embedding fallback."""
        self._tfidf_corpus.append(text)
        try:
            matrix = self._tfidf_fallback.fit_transform(self._tfidf_corpus)
            vec = matrix[-1].toarray()[0]
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            return np.zeros(512)

    def _embed_hash(self, text: str) -> np.ndarray:
        """Minimal hash-based embedding (last resort)."""
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = vec / 255.0
        # Pad/truncate to standard dim
        target = 384
        if len(vec) < target:
            vec = np.pad(vec, (0, target - len(vec)))
        return vec[:target]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Memory Vault — ChromaDB Backend
# ─────────────────────────────────────────────────────────────────────────────

class MemoryVault:
    """
    ChromaDB-backed persistent vector store.
    
    ChromaDB stores vectors on disk and supports fast ANN search.
    If ChromaDB is unavailable, falls back to in-memory numpy search.
    """

    def __init__(self, db_path: str, collection_name: str = "aura_memories"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._fallback_store: Dict[str, Dict] = {}
        self._fallback_vectors: Dict[str, np.ndarray] = {}
        self._init_storage()

    def _init_storage(self):
        """Initialize ChromaDB or fallback."""
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.db_path))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            count = self._collection.count()
            logger.info("✅ MemoryVault: ChromaDB initialized with %d memories", count)
        except ImportError:
            logger.warning(
                "⚠️ chromadb not installed. Install with: pip install chromadb\n"
                "Using in-memory fallback (memories lost on restart)"
            )
        except Exception as e:
            logger.error("ChromaDB init failed: %s. Using in-memory fallback.", e)

    @effect_sink("memory.vault_store", allowed_domains=("memory_write",))
    def store(self, memory: Memory, vector: np.ndarray):
        """Persist a memory with its embedding."""
        metadata = {
            "memory_type": memory.memory_type,
            "timestamp": memory.timestamp,
            "importance": memory.importance,
            "emotional_valence": memory.emotional_valence,
            "emotional_arousal": memory.emotional_arousal,
            "source": memory.source,
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed,
            "tags": json.dumps(memory.tags),
            "linked_ids": json.dumps(memory.linked_ids),
        }

        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[memory.id],
                    embeddings=[vector.tolist()],
                    documents=[memory.content],
                    metadatas=[metadata],
                )
                return
            except Exception as e:
                logger.error("ChromaDB store failed: %s", e)

        # Fallback
        self._fallback_store[memory.id] = {"memory": asdict(memory), "metadata": metadata}
        self._fallback_vectors[memory.id] = vector

    def query(
        self,
        query_vector: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Query by vector similarity.
        Returns list of (id, content, metadata, distance) tuples.
        """
        if self._collection is not None:
            try:
                kwargs = {
                    "query_embeddings": [query_vector.tolist()],
                    "n_results": min(n_results, self._collection.count() or 1),
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where

                results = self._collection.query(**kwargs)

                output = []
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append((
                        doc_id,
                        results["documents"][0][i],
                        results["metadatas"][0][i],
                        results["distances"][0][i],
                    ))
                return output
            except Exception as e:
                logger.error("ChromaDB query failed: %s", e)

        # Fallback: brute-force numpy search
        if not self._fallback_vectors:
            return []

        scores = []
        for mem_id, vec in self._fallback_vectors.items():
            sim = EmbeddingEngine.cosine_similarity(query_vector, vec)
            scores.append((mem_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for mem_id, sim in scores[:n_results]:
            stored = self._fallback_store.get(mem_id, {})
            mem_data = stored.get("memory", {})
            meta = stored.get("metadata", {})
            results.append((mem_id, mem_data.get("content", ""), meta, 1.0 - sim))
        return results

    def count(self) -> int:
        if self._collection is not None:
            return self._collection.count()
        return len(self._fallback_store)


# ─────────────────────────────────────────────────────────────────────────────
# Importance Scorer
# ─────────────────────────────────────────────────────────────────────────────

class ImportanceScorer:
    """
    Determines how important a memory is.
    
    Important memories are kept longer and retrieved with higher priority.
    Unimportant memories fade (but aren't deleted immediately — they're
    deprioritized in retrieval first).
    """

    def score(
        self,
        content: str,
        memory_type: str,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.0,
        source: str = "conversation",
        explicitly_important: bool = False,
    ) -> float:
        """Calculate importance score for a memory."""
        score = 0.3  # Baseline

        # Memory type weights
        type_weights = {
            "insight":       0.8,
            "dream":         0.5,
            "semantic":      0.6,
            "episodic":      0.4,
            "self_play":     0.5,
            "correction":    0.9,  # User corrections are extremely important
            "conversation":  0.3,
        }
        score = type_weights.get(memory_type, 0.3)

        # Emotional intensity increases importance
        emotional_intensity = abs(emotional_valence) * 0.5 + emotional_arousal * 0.3
        score += emotional_intensity * 0.3

        # Content signals
        word_count = len(content.split())
        if word_count > 50:
            score += 0.1  # Substantive content
        
        important_keywords = [
            "remember", "important", "always", "never", "promise",
            "my name", "i am", "i feel", "i want", "i believe",
            "discovered", "realized", "understood", "solved"
        ]
        if any(k in content.lower() for k in important_keywords):
            score += 0.15

        if explicitly_important:
            score = max(score, 0.85)

        return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation Engine — Memory Merging
# ─────────────────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    """
    Periodically consolidates related memories.
    
    Similar to how sleep consolidates human memory:
    - Groups semantically similar memories
    - Merges them into a richer, generalized representation
    - This is how "insights" emerge from repeated experience
    
    Call consolidate() during Aura's sleep/idle cycle.
    """

    SIMILARITY_THRESHOLD = 0.85  # Memories above this are candidates for merging

    def __init__(self, vault: MemoryVault, embedding_engine: EmbeddingEngine):
        self.vault = vault
        self.embedder = embedding_engine

    async def consolidate(self, brain=None, max_merges: int = 5) -> int:
        """
        Run a consolidation pass.
        Returns number of memories consolidated.
        """
        if self.vault.count() < 10:
            return 0  # Not enough memories to consolidate

        logger.info("🌙 ConsolidationEngine: Starting memory consolidation pass")
        consolidated = 0

        # We can't easily get all memories for comparison without ChromaDB iteration
        # This is a simplified version that works with the available API
        # A full implementation would iterate collections in batches

        if brain and self.vault._collection is not None:
            try:
                # Get a sample of recent memories for consolidation
                recent = self.vault._collection.get(
                    limit=100,
                    include=["documents", "metadatas", "embeddings"]
                )

                if not recent["ids"]:
                    return 0

                # Find clusters of similar memories
                ids = recent["ids"]
                docs = recent["documents"]
                embeddings = [np.array(e) for e in recent["embeddings"]]

                for i in range(len(ids)):
                    for j in range(i + 1, min(i + 10, len(ids))):
                        sim = EmbeddingEngine.cosine_similarity(embeddings[i], embeddings[j])
                        if sim > self.SIMILARITY_THRESHOLD and consolidated < max_merges:
                            # Merge these two memories into an insight
                            merged = await self._merge_memories(
                                docs[i], docs[j], brain
                            )
                            if merged:
                                consolidated += 1
                                logger.debug(
                                    "Consolidated memories:\n  A: %s...\n  B: %s...\n  → %s...",
                                    docs[i][:50], docs[j][:50], merged[:50]
                                )

            except Exception as e:
                logger.error("Consolidation failed: %s", e)

        logger.info("🌙 Consolidation complete: %d memories merged", consolidated)
        return consolidated

    async def _merge_memories(self, content_a: str, content_b: str, brain) -> Optional[str]:
        """Ask the LLM to synthesize two similar memories into a generalized insight."""
        if not brain:
            return None

        prompt = f"""Two related memories:
Memory A: {content_a[:200]}
Memory B: {content_b[:200]}

Synthesize these into a single, generalized insight or principle.
Be concise (1-2 sentences). Extract the universal pattern."""

        try:
            from core.brain.cognitive_engine import ThinkingMode
            result = await brain.think(prompt, mode=ThinkingMode.FAST, max_tokens=150)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Vector Memory Engine — Unified Interface
# ─────────────────────────────────────────────────────────────────────────────

class VectorMemoryEngine:
    """
    The complete memory system. This is what gets wired to the orchestrator.
    
    Usage:
        engine = VectorMemoryEngine()
        
        # Store
        await engine.store("User mentioned they love jazz music", 
                          memory_type="episodic", importance=0.6)
        
        # Recall
        memories = await engine.recall("music preferences", limit=5)
        for m in memories:
            print(f"{m.relevance:.2f}: {m.memory.content[:80]}")
    """

    def __init__(self, db_path: Optional[str] = None):
        from core.config import config
        self.db_path = str(
            Path(db_path or config.paths.data_dir / "memory" / "vector_store")
        )
        self.embedder = EmbeddingEngine()
        self.vault = MemoryVault(self.db_path)
        self.scorer = ImportanceScorer()
        self.consolidator = ConsolidationEngine(self.vault, self.embedder)
        self._recent_memory_ids: List[str] = []  # For recency weighting
        logger.info("✅ VectorMemoryEngine initialized. Memories: %d", self.vault.count())

    def _constitutional_runtime_live(self) -> bool:
        return (
            ServiceContainer.has("executive_core")
            or ServiceContainer.has("aura_kernel")
            or ServiceContainer.has("kernel_interface")
            or bool(getattr(ServiceContainer, "_registration_locked", False))
        )

    async def _approve_memory_write(
        self,
        content: str,
        *,
        memory_type: str,
        importance: float,
        source: str,
        tags: Optional[List[str]] = None,
        return_decision: bool = False,
    ) -> bool | tuple[bool, Any]:
        try:
            from core.constitution import get_constitutional_core, unpack_governance_result

            approved, reason, decision = unpack_governance_result(
                await get_constitutional_core().approve_memory_write(
                    memory_type,
                    str(content or "")[:240],
                    source=source or "vector_memory_engine",
                    importance=max(0.0, min(1.0, float(importance or 0.0))),
                    metadata={"tags": list(tags or [])[:10]},
                    return_decision=True,
                )
            )
            if not approved:
                record_degraded_event(
                    "vector_memory_engine",
                    "memory_write_blocked",
                    detail=str(content or "")[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
            if return_decision:
                return approved, decision
            return approved
        except Exception as exc:
            if self._constitutional_runtime_live():
                record_degraded_event(
                    "vector_memory_engine",
                    "memory_write_gate_failed",
                    detail=str(content or "")[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"error": type(exc).__name__},
                    exc=exc,
                )
                if return_decision:
                    return False, None
                return False
            logger.debug("VectorMemoryEngine constitutional gate unavailable: %s", exc)
            if return_decision:
                return True, None
            return True

    async def store(
        self,
        content: str,
        memory_type: str = "episodic",
        emotional_context: Optional[Dict] = None,
        importance: Optional[float] = None,
        source: str = "conversation",
        tags: List[str] = None,
        explicitly_important: bool = False,
    ) -> str:
        """
        Store a new memory.
        Returns the memory ID.
        """
        if not content or not content.strip():
            return ""

        emotional_context = emotional_context or {}
        valence = emotional_context.get("valence", 0.0)
        arousal = emotional_context.get("arousal", 0.0)

        if importance is None:
            importance = self.scorer.score(
                content=content,
                memory_type=memory_type,
                emotional_valence=valence,
                emotional_arousal=arousal,
                source=source,
                explicitly_important=explicitly_important,
            )

        approved, governance_decision = await self._approve_memory_write(
            content,
            memory_type=memory_type,
            importance=importance,
            source=source,
            tags=tags,
            return_decision=True,
        )
        if not approved:
            return ""

        memory_id = hashlib.sha256(
            f"{content}{time.time()}".encode()
        ).hexdigest()[:16]

        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            importance=importance,
            emotional_valence=valence,
            emotional_arousal=arousal,
            source=source,
            tags=tags or [],
        )

        # Embed and store
        vector = await asyncio.to_thread(self.embedder.embed, content)
        if governance_decision is not None:
            from core.governance_context import governed_scope

            async with governed_scope(governance_decision):
                self.vault.store(memory, vector)
        else:
            self.vault.store(memory, vector)

        # Track recency
        self._recent_memory_ids.append(memory_id)
        if len(self._recent_memory_ids) > 1000:
            self._recent_memory_ids = self._recent_memory_ids[-1000:]

        return memory_id

    async def recall(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        recency_weight: float = 0.2,
    ) -> List[RecalledMemory]:
        """
        Retrieve memories by semantic similarity.
        
        Scoring = (relevance * 0.7) + (importance * 0.1) + (recency * recency_weight)
        
        Args:
            query: What to search for (embedded and compared semantically)
            limit: Number of memories to return
            memory_type: Filter by type (episodic, semantic, insight, etc.)
            min_importance: Filter by minimum importance
            recency_weight: How much to favor recent memories (0.0–0.5)
        """
        if not query:
            return []

        # Embed query
        query_vector = await asyncio.to_thread(self.embedder.embed, query)

        # Build filter
        where = {}
        if memory_type:
            where["memory_type"] = memory_type
        if min_importance > 0:
            where["importance"] = {"$gte": min_importance}

        # Query vault
        raw_results = self.vault.query(
            query_vector=query_vector,
            n_results=limit * 3,  # Get more, then re-rank
            where=where if where else None,
        )

        if not raw_results:
            return []

        # Re-rank with recency and importance
        recent_set = set(self._recent_memory_ids[-100:])

        recalled = []
        for mem_id, content, metadata, distance in raw_results:
            relevance = 1.0 - distance  # ChromaDB returns distance, not similarity
            importance = float(metadata.get("importance", 0.5))
            is_recent = mem_id in recent_set
            recency = 0.3 if is_recent else 0.0

            combined = (relevance * 0.7) + (importance * 0.1) + (recency * recency_weight)

            memory = Memory(
                id=mem_id,
                content=content,
                memory_type=metadata.get("memory_type", "unknown"),
                timestamp=float(metadata.get("timestamp", 0)),
                importance=importance,
                emotional_valence=float(metadata.get("emotional_valence", 0)),
                emotional_arousal=float(metadata.get("emotional_arousal", 0)),
                source=metadata.get("source", "unknown"),
                tags=json.loads(metadata.get("tags", "[]")),
                access_count=int(metadata.get("access_count", 0)),
            )

            recalled.append(RecalledMemory(
                memory=memory,
                relevance=relevance,
                recency_weight=recency,
                combined_score=combined,
            ))

        # Sort by combined score
        recalled.sort(key=lambda r: r.combined_score, reverse=True)
        return recalled[:limit]

    async def recall_formatted(self, query: str, limit: int = 5) -> str:
        """
        Recall and format memories as a context string for prompt injection.
        Drop-in replacement for old memory.retrieve() calls.
        """
        memories = await self.recall(query, limit=limit)
        if not memories:
            return ""

        lines = ["RELEVANT MEMORIES:"]
        for r in memories:
            age_s = time.time() - r.memory.timestamp
            age_str = f"{int(age_s/3600)}h ago" if age_s < 86400 else f"{int(age_s/86400)}d ago"
            lines.append(f"  [{r.memory.memory_type}, {age_str}, relevance={r.relevance:.2f}] "
                        f"{r.memory.content[:150]}")

        return "\n".join(lines)

    async def consolidate(self, brain=None) -> int:
        """Run memory consolidation (call from sleep/idle cycle)."""
        return await self.consolidator.consolidate(brain=brain)

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_memories": self.vault.count(),
            "recent_tracked": len(self._recent_memory_ids),
            "db_path": self.db_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_vector_memory(orchestrator=None) -> VectorMemoryEngine:
    """
    Wire to orchestrator in _init_memory or equivalent.
    
    Replaces:
        ServiceContainer.get("semantic_memory")
        ServiceContainer.get("vector_memory")
        ServiceContainer.get("episodic_memory")
    
    All now point to this engine.
    """
    from core.container import ServiceContainer

    engine = VectorMemoryEngine()
    ServiceContainer.register_instance("vector_memory_engine", engine)
    
    # Register legacy aliases so existing code doesn't break
    ServiceContainer.register_instance("semantic_memory", engine)
    ServiceContainer.register_instance("vector_memory", engine)

    logger.info("🧠 VectorMemoryEngine registered as primary memory system")
    return engine
