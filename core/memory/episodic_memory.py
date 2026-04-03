"""Episodic Memory v5.0 — Autobiographical event records for Aura.

Unlike SQLiteMemory (structured operational logs) and VectorMemory (semantic search),
EpisodicMemory stores rich narratives of *episodes* — context + action + outcome +
emotional valence — and supports both recency-based and relevance-based retrieval.

Integrates with:
  - VectorMemory: for semantic similarity search across episodes
  - ReliabilityTracker: records tool outcomes alongside episodes
  - BeliefGraph: episodes can update beliefs
"""
from core.utils.exceptions import capture_and_log
import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.config import config
from core.health.degraded_events import record_degraded_event
from core.resilience.state_manager import _SafeEncoder

logger = logging.getLogger("Memory.Episodic")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    episode_id: str = Field(alias="id") # Support both 'id' and 'episode_id'
    timestamp: float
    context: str = ""              # What was happening / user request
    action: str = ""               # What Aura did
    outcome: str = ""              # What happened
    description: Optional[str] = None # Legacy flat description
    success: bool = True
    emotional_valence: float = 0.0  # -1.0 (distressing) to +1.0 (rewarding)
    arousal: float = 0.5      # 0.0 (calm) to 1.0 (intense)
    importance: float = 0.5   # 0.0–1.0, controls retention priority
    
    participants: List[str] = Field(default_factory=lambda: ["user", "aura"])
    tools_used: List[str] = Field(default_factory=list)
    lessons: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    linked_semantic_ids: List[str] = Field(default_factory=list)
    
    access_count: int = 0
    last_accessed: float = 0.0
    decay_rate: float = 0.01
    qualia_snapshot: Dict[str, Any] = Field(default_factory=dict, alias="context_snapshot")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    # ISSUE 30 fix: Removed @property def id to avoid Pydantic v2 alias conflict

    @property
    def full_description(self) -> str:
        if self.description:
            return self.description
        return f"{self.context} | {self.action} | {self.outcome}"

    def current_strength(self) -> float:
        """Memory strength at current time, accounting for decay.
        Uses Ebbinghaus forgetting curve logic.
        """
        import math
        elapsed_hours = (time.time() - self.timestamp) / 3600
        # Increased importance = higher stability = slower decay
        stability = (1.0 / self.decay_rate) * (1 + self.importance)
        raw_strength = math.exp(-elapsed_hours / stability)
        
        # Emotional salience boosts retention
        emotional_boost = abs(self.emotional_valence) * 0.2
        return min(1.0, raw_strength + emotional_boost)

    def to_retrieval_text(self) -> str:
        """Format for injection into prompt context."""
        age_hours = (time.time() - self.timestamp) / 3600
        if age_hours < 1:
            time_desc = f"{int(age_hours * 60)} minutes ago"
        elif age_hours < 24:
            time_desc = f"{int(age_hours)} hours ago"
        else:
            time_desc = f"{int(age_hours / 24)} days ago"
        
        valence_desc = "positively" if self.emotional_valence > 0.2 else \
                      "negatively" if self.emotional_valence < -0.2 else "neutrally"
        
        return (
            f"[Episodic Memory — {time_desc}] "
            f"Context: {self.context} | Action: {self.action} | Outcome: {self.outcome} "
            f"(experienced {valence_desc}, importance: {self.importance:.0%})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Persistent autobiographical memory with importance-weighted retention.
    """

    MAX_EPISODES = 10_000  # Hard cap — after this, prune low-importance episodes
    _RECORD_COOLDOWN = 0.5  # Minimum seconds between recordings (rate limit)
    
    # Retention Policy Constants
    DEFAULT_IMPORTANCE = 0.5
    FAILURE_IMPORTANCE_BOOST = 0.7
    EMOTIONAL_IMPORTANCE_BOOST = 0.8
    EMOTIONAL_THRESHOLD = 0.7 # Corrected from malformed input
    KEYWORD_SEARCH_SCAN_LIMIT = 600

    def __init__(self, db_path: str = None, vector_memory=None):

        self._db_path = db_path or str(config.paths.home_dir / "episodic.db")
        self._vector_memory = vector_memory
        self._lock = threading.Lock()
        self._last_record_time = 0.0
        self._init_db()

    # ---- Database -----------------------------------------------------------

    def _init_db(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")  # 30s — match db_config.py
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    timestamp  REAL NOT NULL,
                    context    TEXT NOT NULL,
                    action     TEXT NOT NULL,
                    outcome    TEXT NOT NULL,
                    success    INTEGER NOT NULL,
                    emotional_valence REAL DEFAULT 0.0,
                    arousal    REAL DEFAULT 0.5,
                    importance REAL DEFAULT 0.5,
                    participants TEXT DEFAULT '["user", "aura"]',
                    tools_used TEXT DEFAULT '[]',
                    lessons    TEXT DEFAULT '[]',
                    tags       TEXT DEFAULT '[]',
                    linked_semantic_ids TEXT DEFAULT '[]',
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0.0,
                    decay_rate REAL DEFAULT 0.01,
                    qualia_snapshot TEXT DEFAULT '{}',
                    next_decay_eval REAL DEFAULT 0.0
                )
            """)
            conn.commit()
            
            # Migration: Ensure all columns exist
            columns = [
                ("emotional_valence", "REAL DEFAULT 0.0"),
                ("arousal", "REAL DEFAULT 0.5"),
                ("importance", "REAL DEFAULT 0.5"),
                ("participants", "TEXT DEFAULT '[\"user\", \"aura\"]'"),
                ("tools_used", "TEXT DEFAULT '[]'"),
                ("lessons", "TEXT DEFAULT '[]'"),
                ("tags", "TEXT DEFAULT '[]'"),
                ("linked_semantic_ids", "TEXT DEFAULT '[]'"),
                ("access_count", "INTEGER DEFAULT 0"),
                ("last_accessed", "REAL DEFAULT 0.0"),
                ("decay_rate", "REAL DEFAULT 0.01"),
                ("qualia_snapshot", "TEXT DEFAULT '{}'"),
                ("next_decay_eval", "REAL DEFAULT 0.0"),
            ]
            # Add all missing columns before creating indexes that depend on them.
            cursor = conn.execute("PRAGMA table_info(episodes)")
            existing_columns = set()
            for row in cursor.fetchall():
                try:
                    existing_columns.add(row["name"])
                except Exception:
                    existing_columns.add(row[1])
            
            for col_name, col_def in columns:
                if col_name not in existing_columns:
                    conn.execute(f"ALTER TABLE episodes ADD COLUMN {col_name} {col_def}")
                    conn.commit()
                    logger.info("📝 Schema migration: added %s column", col_name)
                    existing_columns.add(col_name)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_next_decay ON episodes (next_decay_eval)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_timestamp ON episodes (timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes (importance DESC)")
            conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                conn.commit()
            except Exception as checkpoint_exc:
                logger.debug("EpisodicMemory WAL checkpoint skipped after init: %s", checkpoint_exc)

    def _get_conn(self) -> sqlite3.Connection:
        from core.memory import db_config
        conn = db_config.configure_connection(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- Async Wrappers -----------------------------------------------------

    async def record_episode_async(
        self,
        context: str,
        action: str,
        outcome: str,
        success: bool,
        emotional_valence: float = 0.0,
        tools_used: Optional[List[str]] = None,
        lessons: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> str:
        return await asyncio.to_thread(
            self.record_episode,
            context,
            action,
            outcome,
            success,
            emotional_valence,
            tools_used,
            lessons,
            importance
        )

    async def recall_recent_async(self, limit: int = 10) -> List[Episode]:
        return await asyncio.to_thread(self.recall_recent, limit)

    async def recall_similar_async(self, query: str, limit: int = 5) -> List[Episode]:
        return await asyncio.to_thread(self.recall_similar, query, limit)

    async def recall_failures_async(self, limit: int = 10) -> List[Episode]:
        return await asyncio.to_thread(self.recall_failures, limit)

    async def recall_by_tool_async(self, tool_name: str, limit: int = 10) -> List[Episode]:
        return await asyncio.to_thread(self.recall_by_tool, tool_name, limit)

    async def add_lesson_async(self, episode_id: str, lesson: str):
        return await asyncio.to_thread(self.add_lesson, episode_id, lesson)

    async def delete_episodes_async(self, episode_ids: List[str]):
        """Async wrapper for delete_episodes."""
        return await asyncio.to_thread(self.delete_episodes, episode_ids)

    def _constitutional_runtime_live(self) -> bool:
        try:
            from core.container import ServiceContainer

            return (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
        except Exception:
            return False

    def _approve_memory_write(
        self,
        context: str,
        action: str,
        outcome: str,
        importance: float,
    ) -> bool:
        preview = f"{context} | {action} | {outcome}".strip()[:240]
        try:
            from core.constitution import get_constitutional_core

            approved, reason = get_constitutional_core().approve_memory_write_sync(
                "episodic_episode",
                preview,
                source="episodic_memory",
                importance=max(0.0, min(1.0, float(importance or 0.0))),
                metadata={"context": str(context or "")[:120], "action": str(action or "")[:120]},
            )
            if not approved:
                record_degraded_event(
                    "episodic_memory",
                    "memory_write_blocked",
                    detail=preview[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
            return approved
        except Exception as exc:
            if self._constitutional_runtime_live():
                record_degraded_event(
                    "episodic_memory",
                    "memory_write_gate_failed",
                    detail=preview[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"error": type(exc).__name__},
                    exc=exc,
                )
                return False
            logger.debug("EpisodicMemory constitutional gate unavailable: %s", exc)
            return True

    # ---- Core API -----------------------------------------------------------

    def record_episode(
        self,
        context: str,
        action: str,
        outcome: str,
        success: bool,
        emotional_valence: float = 0.0,
        tools_used: Optional[List[str]] = None,
        lessons: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> str:
        """Record a new episode. Returns the episode_id.
        Importance is auto-boosted for failures (we learn more from mistakes).
        Automatically captures current qualia snapshot for mood-congruent recall.
        """
        import uuid
        if not context and not action and not outcome:
            return ""
        if not self._approve_memory_write(context, action, outcome, importance):
            return ""
        episode_id = str(uuid.uuid4())[:12]

        # Rate limiting — prevent flood during rapid tool loops
        # ISSUE 31 fix: Capture constant timestamp for storage consistency
        now_mono = time.monotonic()
        if now_mono - self._last_record_time < self._RECORD_COOLDOWN:
            return episode_id  # Silently skip
            
        # Deduplication — check against last episode content
        last_episode = self.recall_recent(limit=1)
        if last_episode:
            le = last_episode[0]
            if le.context == context and le.action == action and le.outcome == outcome:
                return le.episode_id
                
        self._last_record_time = now_mono
        now = time.time()  # Epoch timestamp for DB storage

        # Failures are inherently more important to remember
        if not success:
            importance = max(importance, self.FAILURE_IMPORTANCE_BOOST)
        # Emotionally extreme events are more memorable
        if abs(emotional_valence) > self.EMOTIONAL_THRESHOLD:
            importance = max(importance, self.EMOTIONAL_IMPORTANCE_BOOST)

        # Capture current qualia snapshot for mood-congruent recall
        qualia_snapshot = {}
        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if qualia:
                qualia_snapshot = qualia.get_qualia_for_memory()
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        tools = tools_used or []
        lesson_list = lessons or []

        with self._lock:
            # Retry-on-locked: handle concurrent writes from metabolic background tasks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self._get_conn() as conn:
                        conn.execute(
                            """INSERT INTO episodes
                               (episode_id, timestamp, context, action, outcome, success,
                                emotional_valence, arousal, importance, participants, 
                                tools_used, lessons, tags, linked_semantic_ids, decay_rate, qualia_snapshot, next_decay_eval)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                episode_id, now, context, action, outcome,
                                int(success), emotional_valence, 0.5, importance,
                                json.dumps(["user", "aura"]),
                                json.dumps(tools), json.dumps(lesson_list), 
                                json.dumps([]), json.dumps([]), 0.01,
                                json.dumps(qualia_snapshot, cls=_SafeEncoder),
                                now + 21600, # First evaluation in 6 hours
                            ),
                        )
                    break  # Success
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < max_retries - 1:
                        logger.debug("Episode write locked (attempt %d/%d), retrying...", attempt + 1, max_retries)
                        time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                    else:
                        raise

            # Also index in vector memory for semantic retrieval
            if self._vector_memory:
                try:
                    text = f"{context} | {action} | {outcome}"
                    self._vector_memory.add_memory(
                        text,
                        metadata={
                            "type": "episode",
                            "episode_id": episode_id,
                            "success": success,
                            "importance": importance,
                            "qualia_norm": qualia_snapshot.get("q_norm", 0.0),
                        },
                    )
                except Exception as e:
                    logger.warning("Failed to index episode in vector memory: %s", e)

            self._maybe_prune()

        logger.info("📝 Episode recorded: %s (success=%s, importance=%.2f, q=%.2f)",
                    episode_id, success, importance, qualia_snapshot.get("q_norm", 0.0))
        return episode_id

    # ---- Compatibility Shims ------------------------------------------------
    
    async def add(self, content: str, **kwargs):
        """Shim for MemoryManager compatibility."""
        return await self.record_episode_async(
            context=str(content), 
            action="logged", 
            outcome="stored_via_manager", 
            success=True, 
            **kwargs
        )

    async def consolidate(self):
        """Memory consolidation — prone decayed memories, boost rehearsed ones.
        
        Aura Hardening: Transition to async-to-thread + indexed scan.
        """
        return await asyncio.to_thread(self._consolidate_sync)

    def _consolidate_sync(self):
        """Synchronous consolidation logic run in a worker thread."""
        pruned = 0
        boosted = 0
        now = time.time()
        
        try:
            with self._lock:
                with self._get_conn() as conn:
                    # Only calculate decay for records due for evaluation
                    rows = conn.execute(
                        "SELECT * FROM episodes WHERE next_decay_eval < ? ORDER BY next_decay_eval ASC LIMIT 500",
                        (now,)
                    ).fetchall()
                    
                    if not rows:
                        return {"pruned": 0, "boosted": 0}

                    prune_ids = []
                    
                    for row in rows:
                        episode = self._row_to_episode(row)
                        strength = episode.current_strength()
                        episode_id = episode.episode_id
                        
                        # Set next evaluation: sooner if low strength, later if strong
                        # Baseline: every 6 hours
                        next_eval = now + 21600 
                        
                        # Prune fully decayed, unimportant memories
                        if strength < 0.05 and episode.importance < 0.7:
                            prune_ids.append(episode_id)
                        
                        # Rehearsal boost: frequently accessed memories get slower decay
                        elif episode.access_count > 3 and episode.decay_rate > 0.005:
                            new_decay = max(0.005, episode.decay_rate * 0.85)
                            conn.execute(
                                "UPDATE episodes SET decay_rate = ?, next_decay_eval = ? WHERE episode_id = ?",
                                (new_decay, next_eval, episode_id)
                            )
                            boosted += 1
                        else:
                            # Just update the timer
                            conn.execute(
                                "UPDATE episodes SET next_decay_eval = ? WHERE episode_id = ?",
                                (next_eval, episode_id)
                            )
                    
                    # Prune decayed episodes
                    if prune_ids:
                        placeholders = ",".join("?" for _ in prune_ids)
                        conn.execute(
                            f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
                            prune_ids
                        )
                        pruned = len(prune_ids)
                    
                    conn.commit()
            
            if pruned or boosted:
                logger.info(
                    "🧠 Memory consolidation: pruned %d, boosted %d",
                    pruned, boosted
                )
                
                # Emit visibility event (skip if emitter not async-safe from here)
                # In 2026 Aura, we assume event bus is non-blocking or we spawn task
                    
        except Exception as e:
            logger.error("Consolidation failed: %s", e)
        
        return {"pruned": pruned, "boosted": boosted}

    def recall_recent(self, limit: int = 10) -> List[Episode]:
        """Retrieve the most recent episodes, ranked by memory strength.
        
        Applies Ebbinghaus decay: old, unimportant, unrehearsed memories
        are ranked lower. Fully decayed memories (strength < 0.05) are
        excluded entirely.
        """
        with self._get_conn() as conn:
            # Fetch more than needed so we can filter out decayed ones
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", 
                (limit * 3,)
            ).fetchall()
        
        episodes = [self._row_to_episode(r) for r in rows]
        
        # Filter out fully decayed memories
        alive = [e for e in episodes if e.current_strength() >= 0.05]
        
        # Sort by strength (recency + importance + emotional salience)
        alive.sort(key=lambda e: e.current_strength(), reverse=True)
        
        return alive[:limit]

    def recall_similar(self, query: str, limit: int = 5) -> List[Episode]:
        """Hybrid search: combines vector similarity with keyword matching.

        Mood-Congruent Recall: Episodes formed in a similar qualia state
        to the current state are boosted in ranking.
        """
        seen_ids: set = set()
        combined: List[Episode] = []

        # 1. Vector search (semantic similarity)
        if self._vector_memory:
            try:
                results = self._vector_memory.search_similar(
                    query=query,
                    k=limit * 2,
                    filter_metadata={"type": "episode"},
                )
                episode_ids = [r.get("metadata", {}).get("episode_id") for r in results if r.get("metadata")]
                episode_ids = [eid for eid in episode_ids if eid]
                if episode_ids:
                    episodes = self._fetch_by_ids(episode_ids)
                    episodes = self._apply_qualia_boost(episodes)
                    for ep in episodes:
                        if ep.episode_id not in seen_ids:
                            seen_ids.add(ep.episode_id)
                            combined.append(ep)
            except Exception as e:
                logger.debug("Vector recall failed: %s", e)

        # 2. Keyword search only when vector recall is insufficient or the user
        # is clearly asking for exact wording.
        if len(combined) < limit or self._query_needs_keyword_fallback(query):
            try:
                keyword_results = self._keyword_search(query, limit)
                for ep in keyword_results:
                    if ep.episode_id not in seen_ids:
                        seen_ids.add(ep.episode_id)
                        combined.append(ep)
            except Exception as e:
                logger.debug("Keyword recall failed: %s", e)

        # 3. Sort by importance + recency blend
        combined.sort(
            key=lambda ep: (ep.importance * 0.6) + (min(1.0, max(0, ep.timestamp - 1774000000) / 2000000) * 0.4),
            reverse=True,
        )
        return combined[:limit]

    def _apply_qualia_boost(self, episodes: List[Episode]) -> List[Episode]:
        """Re-rank episodes by qualia congruence with current phenomenal state."""
        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if not qualia or qualia.q_norm < 0.1:
                return episodes  # No qualia data — skip boosting

            current = qualia.get_qualia_for_memory()
            current_norm = current.get("q_norm", 0.0)
            current_dim = current.get("dominant_dim", "")

            def congruence_score(ep: Episode) -> float:
                qs = ep.qualia_snapshot
                if not qs:
                    return ep.importance
                # Similarity: norm proximity + dimension match
                norm_sim = 1.0 - min(1.0, abs(qs.get("q_norm", 0) - current_norm))
                dim_bonus = 0.2 if qs.get("dominant_dim") == current_dim else 0.0
                return ep.importance + (norm_sim * 0.3) + dim_bonus

            episodes.sort(key=congruence_score, reverse=True)
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        return episodes

    def recall_failures(self, limit: int = 10) -> List[Episode]:
        """Retrieve recent failures — the best learning opportunities."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE success = 0 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def recall_by_tool(self, tool_name: str, limit: int = 10) -> List[Episode]:
        """Retrieve episodes involving a specific tool."""
        with self._get_conn() as conn:
            # tools_used is a JSON array; use LIKE for simple matching
            rows = conn.execute(
                "SELECT * FROM episodes WHERE tools_used LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f'%"{tool_name}"%', limit),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_summary(self) -> Dict[str, Any]:
        """Introspection summary for self-model."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            successes = conn.execute("SELECT COUNT(*) FROM episodes WHERE success = 1").fetchone()[0]
            failures = total - successes
            avg_valence = conn.execute("SELECT AVG(emotional_valence) FROM episodes").fetchone()[0] or 0.0
            important = conn.execute("SELECT COUNT(*) FROM episodes WHERE importance > 0.7").fetchone()[0]
        return {
            "total_episodes": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(1, total),
            "avg_emotional_valence": round(avg_valence, 3),
            "important_memories": important,
        }

    def add_lesson(self, episode_id: str, lesson: str):
        """Append a lesson to an existing episode (post-hoc reflection)."""
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT lessons FROM episodes WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                if row:
                    lessons = json.loads(row[0]) if row[0] else []
                    lessons.append(lesson)
                    conn.execute(
                        "UPDATE episodes SET lessons = ? WHERE episode_id = ?",
                        (json.dumps(lessons), episode_id),
                    )
                    conn.commit()

    # ---- Internal -----------------------------------------------------------

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a sqlite row (Row object) to an Episode."""
        def load_json(val, default):
            if not val: return default
            try:
                return json.loads(val)
            except Exception:
                return default

        # sqlite3.Row doesn't have .get() — use keys() check for optional cols
        row_keys = row.keys()
        def safe_get(key, default):
            return row[key] if key in row_keys else default

        return Episode(
            episode_id=row["episode_id"],
            timestamp=row["timestamp"],
            context=row["context"],
            action=row["action"],
            outcome=row["outcome"],
            success=bool(row["success"]),
            emotional_valence=row["emotional_valence"],
            arousal=row["arousal"] if "arousal" in row_keys else 0.5,
            importance=safe_get("importance", 0.5),
            participants=load_json(row["participants"], ["user", "aura"]),
            tools_used=load_json(row["tools_used"], []),
            lessons=load_json(row["lessons"], []),
            tags=load_json(row["tags"], []),
            linked_semantic_ids=load_json(row["linked_semantic_ids"], []),
            access_count=safe_get("access_count", 0),
            last_accessed=safe_get("last_accessed", 0.0),
            decay_rate=safe_get("decay_rate", 0.01),
            qualia_snapshot=load_json(safe_get("qualia_snapshot", "{}"), {}),
        )

    def _fetch_by_ids(self, episode_ids: List[str]) -> List[Episode]:
        """Fetch episodes by ID list and mark as accessed."""
        now = time.time()
        episodes = []
        with self._get_conn() as conn:
            placeholders = ",".join("?" for _ in episode_ids)
            rows = conn.execute(
                f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
                episode_ids,
            ).fetchall()
            # Update access stats
            for eid in episode_ids:
                conn.execute(
                    "UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE episode_id = ?",
                    (now, eid),
                )
            conn.commit()
        return [self._row_to_episode(r) for r in rows]

    def _keyword_search(self, query: str, limit: int) -> List[Episode]:
        """Simple keyword search across context + action + outcome."""
        words = query.lower().split()[:5]  # Limit search terms
        if not words:
            return []
        conditions = " AND ".join(
            "(LOWER(context) LIKE ? OR LOWER(action) LIKE ? OR LOWER(outcome) LIKE ?)"
            for _ in words
        )
        params = []
        for w in words:
            pattern = f"%{w}%"
            params.extend([pattern, pattern, pattern])
        recent_scan_limit = max(self.KEYWORD_SEARCH_SCAN_LIMIT, limit * 60)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM (
                    SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?
                )
                WHERE {conditions}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [recent_scan_limit, *params, limit],
            ).fetchall()
            if not rows and self._query_needs_keyword_fallback(query):
                rows = conn.execute(
                    f"SELECT * FROM episodes WHERE {conditions} ORDER BY timestamp DESC LIMIT ?",
                    [*params, limit],
                ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    @staticmethod
    def _query_needs_keyword_fallback(query: str) -> bool:
        lowered = " ".join(str(query or "").lower().split())
        if not lowered:
            return False
        return (
            '"' in lowered
            or "'" in lowered
            or "exact phrase" in lowered
            or "exact words" in lowered
            or "exact wording" in lowered
            or "what did i tell you" in lowered
            or "what do you remember" in lowered
            or "remember forever" in lowered
        )

    def _maybe_prune(self):
        """Remove lowest-importance episodes if we exceed MAX_EPISODES."""
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            if count > self.MAX_EPISODES:
                excess = count - self.MAX_EPISODES
                conn.execute(
                    """DELETE FROM episodes WHERE episode_id IN (
                        SELECT episode_id FROM episodes
                        ORDER BY importance ASC, access_count ASC, timestamp ASC
                        LIMIT ?
                    )""",
                    (excess,),
                )
                conn.commit()
                logger.info("Pruned %s low-importance episodes", excess)

    def delete_episodes(self, episode_ids: List[str]):
        """Hard delete specific episodes (e.g., after consolidation)."""
        if not episode_ids:
            return
        with self._lock:
            try:
                with self._get_conn() as conn:
                    placeholders = ",".join("?" for _ in episode_ids)
                    conn.execute(f"DELETE FROM episodes WHERE episode_id IN ({placeholders})", episode_ids)
                    conn.commit()
                
                # Also remove from vector memory if possible
                if self._vector_memory:
                    try:
                        self._vector_memory.delete_memories(filter_metadata={"episode_id": episode_ids})
                    except Exception as e:
                        logger.debug("Vector deletion failed during episode prune: %s", e)
                
                logger.info("🗑️ Deleted %d episodes from storage.", len(episode_ids))
            except Exception as e:
                logger.error("Failed to delete episodes: %s", e)


# ---------------------------------------------------------------------------
# Global Instance (lazy — only set up when imported)
# ---------------------------------------------------------------------------
_instance: Optional[EpisodicMemory] = None


def get_episodic_memory(vector_memory=None) -> EpisodicMemory:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = EpisodicMemory(vector_memory=vector_memory)
    return _instance
