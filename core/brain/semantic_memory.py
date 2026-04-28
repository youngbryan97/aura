from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.SemanticMemory")


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

        # State flags
        self.is_vector_ready = False
        self._init_error: Optional[str] = None
        self.encoder = None
        self.index = None
        self.vector_dimension = 384

        os.makedirs(self.memory_dir, exist_ok=True)

        # 1. Immediate Lite Mode Init
        self.metadata: List[Dict[str, Any]] = []
        self._load_metadata()

        # 2. Manual Upgrade (Deferred)
        logger.info("SemanticMemory initialized in Lite mode. Call await initialize() for vector upgrade.")

    async def initialize(self):
        """Perform async background startup tasks."""
        if not self.is_vector_ready:
            await self._async_background_start()

    # ── Status & Telemetry ──────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
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
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.metadata = data
                    logger.info("Loaded %d memories (Lite Mode).", len(self.metadata))
                else:
                    logger.warning("Metadata file had unexpected format; starting fresh.")
                    self.metadata = []
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Error loading metadata: %s", e)
            self.metadata = []

    def _save_metadata(self):
        """Persist metadata to disk.  Caller MUST hold self._lock."""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            logger.error("Failed to save metadata: %s", e)

    # ── Background Vector Upgrade ───────────────────────────────────

    async def _async_background_start(self):
        """Wait for startup then offload heavy init to thread."""
        await asyncio.sleep(2)  # Non-blocking wait
        await asyncio.to_thread(self._background_init)

    def _background_init(self):
        """Load heavy ML libraries. No time.sleep here."""
        try:
            logger.info("🧠 Starting Background Vector Engine Init...")

            # --- FAISS ---
            try:
                import faiss as _faiss
            except ImportError:
                logger.info("faiss-cpu not installed. Staying in Lite Mode.")
                self._init_error = "faiss not installed"
                return

            # --- SentenceTransformers ---
            try:
                from sentence_transformers import SentenceTransformer as _ST
            except ImportError:
                logger.info("sentence-transformers not installed. Staying in Lite Mode.")
                self._init_error = "sentence-transformers not installed"
                return

            # Load encoder (can take 5-10s on first run)
            logger.info("Loading Embedding Model (all-MiniLM-L6-v2)...")
            encoder = _ST("all-MiniLM-L6-v2")

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
                    embeddings = encoder.encode(texts)
                    _faiss.normalize_L2(embeddings)
                    index.add(embeddings.astype("float32"))
                    _faiss.write_index(index, self.index_path)

            # Commit — atomic swap
            self.encoder = encoder
            self.index = index
            self.is_vector_ready = True
            self._init_error = None
            logger.info("🧠 Semantic Memory Upgraded: VECTOR MODE READY ⚡")

        except Exception as e:
            record_degradation('semantic_memory', e)
            logger.error("Background Vector Init Failed: %s", e, exc_info=True)
            self._init_error = str(e)
            logger.info("Continuing in Lite Mode (Keyword Search).")

    # ── Write ───────────────────────────────────────────────────────

    async def remember(self, content: str, metadata: Dict[str, Any] = None):
        """Async wrapper for add_memory."""
        await asyncio.to_thread(self.add_memory, content, context_tags=metadata)

    def add_memory(self, text: str, context_tags: Dict[str, Any] = None):
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
            if self.is_vector_ready and self.encoder and self.index:
                try:
                    import faiss as _faiss
                    vector = self.encoder.encode([text.strip()])
                    _faiss.normalize_L2(vector)
                    self.index.add(vector.astype("float32"))
                    _faiss.write_index(self.index, self.index_path)
                except Exception as ve:
                    record_degradation('semantic_memory', ve)
                    logger.warning("Vector add failed (data saved to JSON): %s", ve)

        except Exception as e:
            record_degradation('semantic_memory', e)
            logger.error("Failed to add memory: %s", e)

    # ── Read ────────────────────────────────────────────────────────

    def search_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search memories. Uses vector search if available, else keyword fallback."""
        if not query or not query.strip():
            return []

        # Vector search path
        if self.is_vector_ready and self.encoder and self.index:
            try:
                import faiss as _faiss
                query_vector = self.encoder.encode([query])
                _faiss.normalize_L2(query_vector)
                distances, indices = self.index.search(query_vector.astype("float32"), top_k)

                results = []
                with self._lock:
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx != -1 and idx < len(self.metadata) and dist < 1.0:
                            results.append(self.metadata[idx])
                return results
            except Exception as e:
                record_degradation('semantic_memory', e)
                logger.error("Vector search error, falling back: %s", e)

        # Keyword fallback
        q_lower = query.lower()
        with self._lock:
            matches = [m for m in self.metadata if q_lower in m.get("text", "").lower()]
        return matches[-top_k:]

    # ── Consolidation ───────────────────────────────────────────────

    async def consolidate_from_history(self, history: List[Dict[str, str]], cognitive_engine):
        """Summarise recent history into a long-term memory entry."""
        if not history or not cognitive_engine:
            return

        try:
            from core.brain.cognitive_engine import ThinkingMode
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
        except Exception as e:
            record_degradation('semantic_memory', e)
            logger.error("Memory consolidation failed: %s", e)
