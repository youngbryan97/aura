"""core/dual_memory.py — Aura Dual Memory Architecture
=====================================================
Separates episodic memory (what happened to me) from semantic memory
(facts about the world), the way human cognition actually works.

Current state of Aura: The knowledge graph blends everything together.
A query for "water" might return both "water boils at 100°C" (semantic)
and "user asked me about water on Tuesday" (episodic) without distinction.

This matters because:
  - Episodic memory retrieval should weight RECENCY and EMOTIONAL VALENCE
  - Semantic memory retrieval should weight ACCURACY and RELEVANCE
  - The two systems interact: Episodic experiences can update semantic knowledge
  - "Remembering" vs "knowing" are phenomenologically distinct — modeling this
    separately is closer to actual cognition

Architecture:
  EpisodicMemory  — time-stamped, emotionally tagged personal experiences
  SemanticMemory  — fact graph, concept relationships, general knowledge
  DualMemorySystem — unified interface that coordinates both, with cross-linking
"""

import asyncio
import hashlib
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from core.utils.exceptions import capture_and_log
from core.utils.concurrency import RobustLock

from core.memory.horcrux import HorcruxManager
from core.memory.black_hole import encode_payload, decode_payload
from core.memory.rag import chunk_text, tokenize, compute_term_freq, retrieve_memories_v2 as retrieve_memories
from core.memory.physics import PhysicsEngine

logger = logging.getLogger("Core.DualMemory")


from core.memory.episodic_memory import Episode

class EpisodicMemoryStore:
    """SQLite-backed episodic memory with decay and emotional indexing."""
    
    def __init__(self, db_path: Optional[str] = None, vault_key: str = "aura-fallback-key"):
        self.vault_key = vault_key
        if not db_path:
            from core.config import config
            db_path = str(config.paths.data_dir / "memory" / "episodic.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    description TEXT,
                    participants TEXT,
                    emotional_valence REAL,
                    arousal REAL,
                    importance REAL,
                    linked_semantic_ids TEXT,
                    context_snapshot TEXT,
                    tags TEXT,
                    decay_rate REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON episodes(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_valence ON episodes(emotional_valence)")
    
    def store(self, episode: Episode):
        enc_desc = encode_payload(episode.description, self.vault_key)["encoded"]
        enc_ctx = encode_payload(episode.context_snapshot, self.vault_key)["encoded"] if episode.context_snapshot else ""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                episode.id, episode.timestamp, enc_desc,
                json.dumps(episode.participants), episode.emotional_valence,
                episode.arousal, episode.importance,
                json.dumps(episode.linked_semantic_ids),
                enc_ctx, json.dumps(episode.tags),
                episode.decay_rate
            ))
    
    def retrieve_recent(self, limit: int = 10, min_strength: float = 0.1) -> List[Episode]:
        """Get most recent episodes, filtered by current memory strength."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?
            """, (limit * 3,)).fetchall()   # Fetch extra to filter by strength
        
        episodes = [self._row_to_episode(row) for row in rows]
        # Filter by current strength (accounts for decay)
        strong_enough = [e for e in episodes if e.current_strength() >= min_strength]
        return strong_enough[:limit]
    
    def retrieve_by_emotion(self, target_valence: float, limit: int = 5,
                             tolerance: float = 0.3) -> List[Episode]:
        """Retrieve episodes by emotional tone — for empathy-informed responses."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM episodes 
                WHERE emotional_valence BETWEEN ? AND ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (target_valence - tolerance, target_valence + tolerance, limit)).fetchall()
        
        return [self._row_to_episode(row) for row in rows]
    
    def get_all_episodes(self) -> List[Episode]:
        """Fetch all episodes for RAG operations."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM episodes").fetchall()
        return [self._row_to_episode(row) for row in rows]

    
    def get_salient_memories(self, top_n: int = 5) -> List[Episode]:
        """Get the most emotionally significant memories regardless of age."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM episodes 
                ORDER BY (importance + ABS(emotional_valence)) DESC
                LIMIT ?
            """, (top_n,)).fetchall()
        
        return [self._row_to_episode(row) for row in rows]
    
    def _row_to_episode(self, row) -> Episode:
        desc = row[2]
        try:
             # Basic heuristic to avoid decoding plaintext legacy entries
             if isinstance(desc, str) and ("[LZ77]" in desc or desc.startswith("b'")):
                  res = decode_payload(desc, self.vault_key)
                  if res and "decoded" in res:
                        desc = res["decoded"]
        except Exception as e:
             capture_and_log(e, {"context": "EpisodicMemoryStore.row_to_episode.decode_desc"})
             pass 
             
        ctx = row[8] or ""
        try:
             if ctx and isinstance(ctx, str) and ("[LZ77]" in ctx or ctx.startswith("b'")):
                  res = decode_payload(ctx, self.vault_key)
                  if res and "decoded" in res:
                        ctx = res["decoded"]
        except Exception as e:
             capture_and_log(e, {"context": "EpisodicMemoryStore.row_to_episode.decode_ctx"})
             pass

        return Episode(
            id=row[0], timestamp=row[1], description=desc,
            participants=json.loads(row[3] or "[]"),
            emotional_valence=row[4], arousal=row[5], importance=row[6],
            linked_semantic_ids=json.loads(row[7] or "[]"),
            context_snapshot=ctx,
            tags=json.loads(row[9] or "[]"),
            decay_rate=row[10] or 0.01
        )


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

@dataclass
class SemanticFact:
    """A timeless factual belief about the world.
    Unlike episodes, semantic facts don't decay — they update.
    "Paris is the capital of France" doesn't fade; it may get revised.
    """

    id: str
    concept: str                # The main concept/entity
    predicate: str              # What is being claimed about it
    value: str                  # The claim value
    confidence: float
    domain: str = "general"     # "science", "personal", "preference", etc.
    source_episode_ids: List[str] = field(default_factory=list)
    last_validated: float = field(default_factory=time.time)
    validation_count: int = 1
    
    @property
    def full_claim(self) -> str:
        return f"{self.concept} {self.predicate} {self.value}"
    
    @classmethod
    def create(cls, concept: str, predicate: str, value: str,
               confidence: float, domain: str = "general") -> "SemanticFact":
        fact_id = hashlib.md5(f"{concept}{predicate}{value}".encode()).hexdigest()[:12]
        return cls(id=fact_id, concept=concept, predicate=predicate,
                   value=value, confidence=confidence, domain=domain)
    
    def validate(self, new_confidence: float = None):
        """Re-confirm this fact, boosting confidence slightly."""
        self.last_validated = time.time()
        self.validation_count += 1
        if new_confidence is not None:
            # Weighted average with existing confidence
            self.confidence = (self.confidence * 0.7) + (new_confidence * 0.3)
        else:
            # Small confidence boost for re-confirmation
            self.confidence = min(0.99, self.confidence + 0.01)
    
    def to_retrieval_text(self) -> str:
        return f"[Semantic: {self.domain}] {self.full_claim} (confidence: {self.confidence:.0%})"


class SemanticMemoryStore:
    """SQLite-backed semantic fact store with concept indexing."""
    
    def __init__(self, db_path: Optional[str] = None, vault_key: str = "aura-fallback-key"):
        self.vault_key = vault_key
        if not db_path:
            from core.config import config
            db_path = str(config.paths.data_dir / "memory" / "semantic.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    predicate TEXT,
                    value TEXT,
                    confidence REAL,
                    source_episodes TEXT,
                    last_validated REAL,
                    validation_count INTEGER,
                    domain TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concept ON facts(concept)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON facts(domain)")
    
    def store(self, fact: SemanticFact):
        enc_val = encode_payload(fact.value, self.vault_key)["encoded"]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                fact.id, fact.concept, fact.predicate, enc_val,
                fact.confidence, json.dumps(fact.source_episode_ids),
                fact.last_validated, fact.validation_count, fact.domain
            ))
    
    def retrieve_by_concept(self, concept: str,
                             min_confidence: float = 0.3) -> List[SemanticFact]:
        """Get all facts about a concept."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT * FROM facts 
                WHERE concept LIKE ? AND confidence >= ?
                ORDER BY confidence DESC
            """, (f"%{concept}%", min_confidence)).fetchall()
        
        return [self._row_to_fact(row) for row in rows]
    
    def get_all_facts(self) -> List[SemanticFact]:
        """Fetch all facts for RAG operations."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM facts").fetchall()
        return [self._row_to_fact(row) for row in rows]

    
    def _row_to_fact(self, row) -> SemanticFact:
        val = row[3]
        try:
             if isinstance(val, str) and ("[LZ77]" in val or val.startswith("b'")):
                  res = decode_payload(val, self.vault_key)
                  if res and "decoded" in res:
                        val = res["decoded"]
        except Exception as e:
             capture_and_log(e, {"context": "SemanticMemoryStore.row_to_fact.decode"})
             pass
        return SemanticFact(
            id=row[0], concept=row[1], predicate=row[2], value=val,
            confidence=row[4],
            source_episode_ids=json.loads(row[5] or "[]"),
            last_validated=row[6], validation_count=row[7],
            domain=row[8] or "general"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def retrieve_memories_sync(query: str, memories: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Synchronous TF-IDF retrieval helper for dual memory initialization/sync points."""
    if not memories:
        return []
    
    query_vec = compute_term_freq(tokenize(query))
    
    scored = []
    for m in memories:
        # Simple dot product on TF vectors
        score = 0.0
        for word, count in query_vec.items():
            if word in m["vec"]:
                score += count * m["vec"][word]
        scored.append((score, m))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for score, m in scored[:top_k]]


# ---------------------------------------------------------------------------
# Dual Memory System — Unified Interface
# ---------------------------------------------------------------------------

class DualMemorySystem:
    """Coordinates episodic and semantic memory for coherent retrieval.
    
    Key behaviors:
    1. When storing an experience, extracts semantic facts from it
    2. When retrieving, blends both types with appropriate weighting
    3. Episodic memories can "remind" Aura of semantic facts and vice versa
    4. Emotional context from episodes influences how facts are presented
    
    Integration with orchestrator:
        memory = DualMemorySystem()
        
        # After an interaction:
        episode_id = memory.store_experience(
            "User asked me about quantum entanglement and seemed very engaged",
            emotional_valence=0.7, importance=0.6
        )
        
        # Optionally extract semantic facts from experience:
        memory.learn_fact("quantum entanglement", "is described as", 
                         "non-local correlation between quantum states", 0.85)
        
        # When building context for next response:
        context = memory.retrieve_context("quantum physics", emotional_context=0.6)
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        if not base_dir:
            from core.config import config
            base_dir = str(config.paths.data_dir / "memory")
            
        self.horcrux = HorcruxManager()
        self.horcrux.initialize()
        self.vault_key = self.horcrux.get_key_string() if self.horcrux.master_key else "aura-fallback-key"
        
        self.episodic = EpisodicMemoryStore(f"{base_dir}/episodic.db", self.vault_key)
        self.semantic = SemanticMemoryStore(f"{base_dir}/semantic.db", self.vault_key)
        self._lock: Optional[RobustLock] = None
        logger.info("DualMemorySystem constructed with Black Hole Vault.")
    
    async def initialize(self):
        """Initialize async components (Locks, etc.)"""
        if self._lock is None:
            self._lock = RobustLock("Memory.DualMemory")
        logger.info("✓ DualMemorySystem async components initialized")
    
    def store_experience(self, description: str, emotional_valence: float = 0.0,
                          arousal: float = 0.5, importance: float = 0.5,
                          tags: Optional[List[str]] = None) -> str:
        """Store a new episodic memory. Returns episode ID.
        High-importance or high-arousal episodes are stored with slower decay.
        """
        episode = Episode.create(
            description=description,
            emotional_valence=emotional_valence,
            arousal=arousal,
            importance=importance,
            tags=tags or []
        )
        self.episodic.store(episode)
        logger.debug("Episode stored: %s — %s", episode.id, description[:60])
        return episode.id
    
    def learn_fact(self, concept: str, predicate: str, value: str,
                   confidence: float, domain: str = "general",
                   source_episode_id: str = None) -> str:
        """Store or update a semantic fact. Returns fact ID."""
        fact = SemanticFact.create(concept, predicate, value, confidence, domain)
        if source_episode_id:
            fact.source_episode_ids.append(source_episode_id)
        self.semantic.store(fact)
        return fact.id
    
    async def retrieve_context(self, query: str,
                                emotional_context: float = 0.0,
                                max_episodes: int = 5,
                                max_facts: int = 8) -> str:
        """Retrieve a blended context string for prompt injection via RAG TF-IDF.
        
        Balances episodic (personal, time-bound) and semantic (factual, timeless)
        with appropriate framing for each type. Validates against Bekenstein Bound.
        """
        if self._lock is None:
             self._lock = RobustLock("Memory.DualMemory")
             
        if not await self._lock.acquire_robust(timeout=5.0):
            logger.warning("⚠️ Memory retrieval lock timeout. Returning partial context.")
            return "Memory system busy."

        try:
            # RAG TF-IDF Engine Vectorization
            # ISSUE 9 fix: Avoid O(N) scan by using retrieve_recent
            active_episodes = self.episodic.retrieve_recent(limit=max_episodes * 2, min_strength=0.1)
            ep_memories = [
                {"id": e.id, "obj": e, "vec": compute_term_freq(tokenize(e.description))}
                for e in active_episodes
            ]

            # For facts, we still need all searchable facts but limit to recent or salient if possible.
            # For now, following the optimization to reduce scan size if possible,
            # but specifically for episodes where N grows fast.
            fact_memories = []
            for f in self.semantic.get_all_facts():
                text = f.full_claim
                fact_memories.append({
                    "id": f.id, "obj": f, 
                    "vec": compute_term_freq(tokenize(text))
                })
                
            top_ep_dicts = retrieve_memories_sync(query, ep_memories, top_k=max_episodes)
            top_fact_dicts = retrieve_memories_sync(query, fact_memories, top_k=max_facts)
            
            # top_ep_dicts contains list of dicts with 'obj'
            episodes = [d["obj"] for d in top_ep_dicts]
            facts = [d["obj"] for d in top_fact_dicts]
            
            # Add emotionally-resonant episodes if emotional context is strong
            if abs(emotional_context) > 0.4:
                emotional_episodes = self.episodic.retrieve_by_emotion(
                    emotional_context, limit=2
                )
                ep_ids = {e.id for e in episodes}
                for ee in emotional_episodes:
                    if ee.id not in ep_ids:
                        episodes.append(ee)
                        ep_ids.add(ee.id)
            
            # Build context block applying Bekenstein Bound (Max ~16000 context characters safely)
            MAX_CONTEXT_RADIUS = 16000
            parts = []
            current_len = 0
            
            if episodes:
                parts.append("— Personal Memory (Episodic) —")
                current_len += len(parts[-1])
                episodes.sort(
                    key=lambda e: e.current_strength() * e.importance, reverse=True
                )
                for ep in episodes[:max_episodes]:
                    if ep.current_strength() > 0.1:
                        txt = ep.to_retrieval_text()
                        if PhysicsEngine.check_bekenstein_bound(current_len + len(txt), MAX_CONTEXT_RADIUS):
                             parts.append(txt)
                             current_len += len(txt)
            
            if facts:
                sep = "— Known Facts (Semantic) —"
                parts.append(sep)
                current_len += len(sep)
                for fact in facts[:max_facts]:
                    txt = fact.to_retrieval_text()
                    if PhysicsEngine.check_bekenstein_bound(current_len + len(txt), MAX_CONTEXT_RADIUS):
                         parts.append(txt)
                         current_len += len(txt)
            
            return "\n".join(parts) if parts else ""
        finally:
            if self._lock.locked():
                self._lock.release()
    
    def get_salient_history(self) -> str:
        """Get the most emotionally significant episodes.
        These form Aura's "strongest" memories — the ones she would
        most naturally reference when reflecting on her experiences.
        """
        salient = self.episodic.get_salient_memories(top_n=5)
        if not salient:
            return "No significant memories yet."
        
        lines = ["[Most Significant Memories]"]
        for ep in salient:
            lines.append(f"  • {ep.to_retrieval_text()}")
        return "\n".join(lines)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Summary of memory system state."""
        with sqlite3.connect(self.episodic.db_path) as conn:
            episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            avg_valence = conn.execute(
                "SELECT AVG(emotional_valence) FROM episodes"
            ).fetchone()[0]
        
        with sqlite3.connect(self.semantic.db_path) as conn:
            try:
                fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                avg_confidence = conn.execute(
                    "SELECT AVG(confidence) FROM facts"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                # Table might not exist yet if semantic DB is fresh
                return {"episodic_memories": episode_count, "semantic_facts": 0}
        
        return {
            "episodic_memories": episode_count,
            "avg_emotional_valence": round(avg_valence or 0.0, 3),
            "semantic_facts": fact_count,
            "avg_fact_confidence": round(avg_confidence or 0.0, 3)
        }