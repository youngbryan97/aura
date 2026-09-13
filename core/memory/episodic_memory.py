"""Episodic Memory v5.0 — Autobiographical event records for Aura.

Unlike SQLiteMemory (structured operational logs) and VectorMemory (semantic search),
EpisodicMemory stores rich narratives of *episodes* — context + action + outcome +
emotional valence — and supports both recency-based and relevance-based retrieval.

Integrates with:
  - VectorMemory: for semantic similarity search across episodes
  - ReliabilityTracker: records tool outcomes alongside episodes
  - BeliefGraph: episodes can update beliefs
"""
import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import config
from core.health.degraded_events import record_degraded_event
from core.memory.a_deferral_is_not_a_refusal import DeferredWrites, is_a_deferral
from core.memory.engram_association import (
    get_engram_association_field,
    is_engram_association_enabled,
)
from core.memory.hippocampus import HippocampalIndex
from core.memory.reconsolidation import ReconsolidationEngine, ReconsolidationOutcome
from core.memory.retention_policy import episodic_retention_policy
from core.resilience.state_manager import _SafeEncoder
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log

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
    description: str | None = None # Legacy flat description
    success: bool = True
    emotional_valence: float = 0.0  # -1.0 (distressing) to +1.0 (rewarding)
    arousal: float = 0.5      # 0.0 (calm) to 1.0 (intense)
    importance: float = 0.5   # 0.0–1.0, controls retention priority
    
    participants: list[str] = Field(default_factory=lambda: ["user", "aura"])
    tools_used: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    linked_semantic_ids: list[str] = Field(default_factory=list)
    
    access_count: int = 0
    last_accessed: float = 0.0
    decay_rate: float = 0.01
    qualia_snapshot: dict[str, Any] = Field(default_factory=dict, alias="context_snapshot")

    # --- Engram dynamics (reconsolidation / fidelity) ---
    # A memory is not a static recording. Each time it is recalled under the
    # spotlight of attention it can soften and be rewritten by the present
    # context. These fields track that lifecycle.
    fidelity: float = 1.0            # 1.0 = faithful to encoding; drops as the trace is rewritten on recall
    original_valence: float | None = None  # emotional tone at encoding (drift reference)
    reconsolidation_count: int = 0  # times the trace re-entered a labile state and was updated
    last_reconsolidated: float = 0.0
    novelty: float = 0.5            # prediction error at encoding — drives consolidation strength
    source: str = "episodic_memory"
    source_ref: str = ""
    source_revision: int = 0
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    source_session_id: str = ""
    source_exchange_id: str = ""
    source_authoritative: bool = True
    superseded_at: float = 0.0
    superseded_by: str = ""

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    # ISSUE 30 fix: Removed @property def id to avoid Pydantic v2 alias conflict

    @property
    def full_description(self) -> str:
        if self.description:
            return self.description
        # A conversation episode holds two voices: `context` is what the
        # person said, `outcome` is what Aura said back. Flattened to
        # "a | conversation_reply | b" both sentences lose their speaker, and
        # a first-person sentence with no speaker is not neutral data — the
        # reader supplies one. On 2026-07-28 the reader was Aura, recalling
        # Bryan's "I was trying to get you to write about yourself" as a
        # thing she had wanted, and she spent six turns acting on it.
        #
        # Naming the speakers costs eight characters and removes the
        # ambiguity at the source, so nothing downstream has to guess.
        if str(self.action or "").strip().lower().startswith("conversation"):
            said = str(self.context or "").strip()
            replied = str(self.outcome or "").strip()
            parts = []
            if said:
                parts.append(f"User: {said}")
            if replied:
                parts.append(f"Aura: {replied}")
            if parts:
                return "\n".join(parts)
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

    def current_fidelity(self) -> float:
        """How faithful this trace still is to its original encoding (1.0 = pristine).

        Distinct from ``current_strength`` (vividness/retrievability): a memory can
        be vivid yet inaccurate. Repeated reconsolidation lowers fidelity even as
        rehearsal raises strength — vividness is not accuracy.
        """
        return max(0.0, min(1.0, self.fidelity))

    def valence_drift(self) -> float:
        """Signed drift of emotional tone away from the original encoding."""
        if self.original_valence is None:
            return 0.0
        return round(self.emotional_valence - self.original_valence, 4)

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

        # Surface vividness≠accuracy: a heavily re-recalled, drifted trace is
        # flagged so downstream reasoning treats it as reshaped, not verbatim.
        fidelity_note = ""
        if self.reconsolidation_count >= 3 and self.current_fidelity() < 0.7:
            fidelity_note = (
                f" [recalled {self.reconsolidation_count}× — vivid but likely reshaped, "
                f"fidelity {self.current_fidelity():.0%}]"
            )

        return (
            f"[Episodic Memory — {time_desc}] "
            f"Context: {self.context} | Action: {self.action} | Outcome: {self.outcome} "
            f"(experienced {valence_desc}, importance: {self.importance:.0%}){fidelity_note}"
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

from .episodic_ranking import _RanksWhatToRecall


class EpisodicMemory(_RanksWhatToRecall):
    """Persistent autobiographical memory with importance-weighted retention.
    """

    MAX_EPISODES = episodic_retention_policy().max_items  # Hard cap, scaled by machine policy.
    RETENTION_POLICY = episodic_retention_policy()
    _RECORD_COOLDOWN = 0.5  # Minimum seconds between recordings (rate limit)
    
    # Retention Policy Constants
    DEFAULT_IMPORTANCE = 0.5
    FAILURE_IMPORTANCE_BOOST = 0.7
    EMOTIONAL_IMPORTANCE_BOOST = 0.8
    EMOTIONAL_THRESHOLD = 0.7 # Corrected from malformed input
    KEYWORD_SEARCH_SCAN_LIMIT = 600
    
    # Relational/bonding conversation markers (for automatic preservation)
    RELATIONAL_KEYWORDS = {
        "bonding", "understand", "know each other", "friend", "trust", "care about",
        "connection", "relationship", "mutual", "together", "promise", "co-pilot",
        "building", "shared", "we have", "understand me", "know me",
        "secret", "real", "honest", "genuine", "authentically", "truly",
        "heartfelt", "sincere", "vulnerable", "deeper", "intimate",
        "travel", "ship", "adventure", "explore", "discover", "future",
        "journey", "quest", "mission", "starship",
        "dream", "wish", "hope", "aspiration", "goal", "wish for",
        "meaningful", "significant", "important", "matters",
    }
    RELATIONAL_IMPORTANCE_BOOST = 0.85  # Prevent loss of bonding conversations

    def __init__(self, db_path: str = None, vector_memory=None):

        self._db_path = db_path or str(config.paths.home_dir / "episodic.db")
        self._vector_memory = vector_memory
        self._lock = threading.Lock()
        self._last_record_time = 0.0
        self._init_db()
        # Hippocampal cue index (engram binding + pattern completion) and the
        # reconsolidation engine that rewrites traces as the present seeps in.
        self._hippocampus = HippocampalIndex(self._get_conn)
        self._reconsolidation = ReconsolidationEngine()
        # Competitive recall weights from the most recent plasticity-resolved
        # recall, keyed by episode_id. Consumed once by _register_recall to apply
        # bounded LTP consolidation to the engrams that won competition.
        self._last_competition_weights: dict[str, float] = {}
        # What the governor put off, kept until it can be made. See
        # core/memory/a_deferral_is_not_a_refusal.py for why this is not the
        # same thing as a refusal.
        self._last_refusal_reason = ""
        self._deferred_episodes: DeferredWrites[dict[str, Any]] = DeferredWrites(
            "episodic_memory", self._write_a_held_episode,
            identity=lambda held: held["idempotency_key"],
        )

    def _hold_if_deferred(
        self,
        episode_id: str,
        context: str,
        action: str,
        outcome: str,
        success: bool,
        emotional_valence: float,
        tools_used: list[str] | None,
        lessons: list[str] | None,
        importance: float,
        source: str,
        metadata: dict[str, Any] | None,
        stable_key: str,
    ) -> None:
        """Keep an episode the governor deferred; drop one it refused."""

        reason = self._last_refusal_reason
        if not is_a_deferral(reason):
            return
        self._deferred_episodes.hold(
            {
                "context": context,
                "action": action,
                "outcome": outcome,
                "success": success,
                "emotional_valence": emotional_valence,
                "tools_used": list(tools_used or []) or None,
                "lessons": list(lessons or []) or None,
                "importance": importance,
                "source": source,
                "metadata": dict(metadata or {}) or None,
                # Held episodes replay through the same idempotency key, so a
                # replay that races the original writes one episode and not
                # two.
                "idempotency_key": stable_key or f"deferred:{episode_id}",
            },
            reason,
        )

    def _write_a_held_episode(self, held: dict[str, Any]) -> bool:
        """Retry one held episode. True when it landed."""

        return bool(self.record_episode(**held))

    def deferred_state(self) -> dict[str, Any]:
        """What is waiting on the governor, for the health surface."""

        return self._deferred_episodes.state()

    def _detect_relational_significance(self, context: str, action: str, outcome: str) -> bool:
        """Detect if this conversation is relational/bonding and should be preserved.
        
        Returns: True if conversation contains relational markers (e.g., bonding, promises, dreams)
        """
        combined = f"{context} {action} {outcome}".lower()
        
        # Count relational keywords
        keyword_count = 0
        for keyword in self.RELATIONAL_KEYWORDS:
            if keyword.lower() in combined:
                keyword_count += 1
        
        # Threshold: at least 2 relational keywords indicates bonding conversation
        if keyword_count >= 2:
            logger.debug(f"🤝 Relational conversation detected ({keyword_count} markers)")
            return True
        
        # Also detect strong bonding patterns
        bonding_patterns = [
            ("understand", "you", "me"),  # Mutual understanding
            ("promise", "together"),      # Commitments
            ("trust", "care"),            # Deep emotions
            ("dream", "future", "together"),  # Shared aspirations
        ]
        
        for pattern in bonding_patterns:
            if all(p.lower() in combined for p in pattern):
                logger.debug(f"🤝 Relational pattern detected: {pattern}")
                return True
        
        return False

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
                    next_decay_eval REAL DEFAULT 0.0,
                    fidelity REAL DEFAULT 1.0,
                    original_valence REAL,
                    reconsolidation_count INTEGER DEFAULT 0,
                    last_reconsolidated REAL DEFAULT 0.0,
                    novelty REAL DEFAULT 0.5,
                    source TEXT DEFAULT 'episodic_memory',
                    source_ref TEXT DEFAULT '',
                    source_revision INTEGER DEFAULT 0,
                    source_metadata TEXT DEFAULT '{}',
                    source_session_id TEXT DEFAULT '',
                    source_exchange_id TEXT DEFAULT '',
                    source_authoritative INTEGER NOT NULL DEFAULT 1,
                    superseded_at REAL DEFAULT 0.0,
                    superseded_by TEXT DEFAULT ''
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
                ("fidelity", "REAL DEFAULT 1.0"),
                ("original_valence", "REAL"),
                ("reconsolidation_count", "INTEGER DEFAULT 0"),
                ("last_reconsolidated", "REAL DEFAULT 0.0"),
                ("novelty", "REAL DEFAULT 0.5"),
                ("source", "TEXT DEFAULT 'episodic_memory'"),
                ("source_ref", "TEXT DEFAULT ''"),
                ("source_revision", "INTEGER DEFAULT 0"),
                ("source_metadata", "TEXT DEFAULT '{}'"),
                ("source_session_id", "TEXT DEFAULT ''"),
                ("source_exchange_id", "TEXT DEFAULT ''"),
                ("source_authoritative", "INTEGER NOT NULL DEFAULT 1"),
                ("superseded_at", "REAL DEFAULT 0.0"),
                ("superseded_by", "TEXT DEFAULT ''"),
            ]
            # Add all missing columns before creating indexes that depend on them.
            cursor = conn.execute("PRAGMA table_info(episodes)")
            existing_columns = set()
            for row in cursor.fetchall():
                try:
                    existing_columns.add(row["name"])
                except (IndexError, KeyError, TypeError):
                    existing_columns.add(row[1])
            
            for col_name, col_def in columns:
                if col_name not in existing_columns:
                    conn.execute(f"ALTER TABLE episodes ADD COLUMN {col_name} {col_def}")
                    conn.commit()
                    logger.info("📝 Schema migration: added %s column", col_name)
                    existing_columns.add(col_name)

            self._migrate_conversation_revision_authority(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_next_decay ON episodes (next_decay_eval)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_timestamp ON episodes (timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes (importance DESC)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_source_revision ON episodes "
                "(source, source_session_id, source_exchange_id, source_revision DESC)"
            )
            conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                conn.commit()
            except (sqlite3.Error, OSError) as checkpoint_exc:
                record_degradation('episodic_memory', checkpoint_exc)
                logger.debug("EpisodicMemory WAL checkpoint skipped after init: %s", checkpoint_exc)

    def _get_conn(self) -> sqlite3.Connection:
        from core.memory import db_config
        conn = db_config.configure_connection(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _migrate_conversation_revision_authority(conn: sqlite3.Connection) -> None:
        """Backfill CP359 chat rows and normalize one winner per exchange."""

        rows = conn.execute(
            "SELECT episode_id, timestamp, source_ref, source_revision, "
            "source_metadata, source_session_id, source_exchange_id "
            "FROM episodes WHERE source='chat_turn_logger'"
        ).fetchall()
        groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in rows:
            session_id = str(row["source_session_id"] or "")[:64]
            exchange_id = str(row["source_exchange_id"] or "")[:128]
            try:
                metadata = json.loads(str(row["source_metadata"] or "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                metadata = {}
            if not session_id:
                session_id = str(metadata.get("session_id") or "")[:64]
            if not exchange_id:
                exchange_id = str(
                    metadata.get("conversation_exchange_id") or ""
                )[:128]
            revision = int(row["source_revision"] or 0)
            source_ref = str(row["source_ref"] or "")
            suffix = f":r{revision}"
            prefix = f"{session_id}:" if session_id else ""
            if (
                not exchange_id
                and prefix
                and source_ref.startswith(prefix)
                and source_ref.endswith(suffix)
            ):
                exchange_id = source_ref[len(prefix) : -len(suffix)][:128]
            if not session_id or not exchange_id or revision <= 0:
                continue
            conn.execute(
                "UPDATE episodes SET source_session_id=?, source_exchange_id=? "
                "WHERE episode_id=?",
                (session_id, exchange_id, str(row["episode_id"])),
            )
            groups.setdefault((session_id, exchange_id), []).append(row)

        for (session_id, exchange_id), revisions in groups.items():
            winner = max(
                revisions,
                key=lambda item: (
                    int(item["source_revision"] or 0),
                    float(item["timestamp"] or 0.0),
                    str(item["episode_id"]),
                ),
            )
            winner_id = str(winner["episode_id"])
            winner_at = float(winner["timestamp"] or 0.0)
            conn.execute(
                "UPDATE episodes SET source_authoritative=CASE WHEN episode_id=? "
                "THEN 1 ELSE 0 END, superseded_at=CASE WHEN episode_id=? THEN 0.0 "
                "ELSE ? END, superseded_by=CASE WHEN episode_id=? THEN '' ELSE ? END "
                "WHERE source='chat_turn_logger' AND source_session_id=? "
                "AND source_exchange_id=?",
                (
                    winner_id,
                    winner_id,
                    winner_at,
                    winner_id,
                    winner_id,
                    session_id,
                    exchange_id,
                ),
            )
        conn.commit()

    def close(self) -> None:
        """Commit and release every connection owned by this durable store."""

        from core.memory.db_config import close_connections_for_path

        report = close_connections_for_path(self._db_path)
        if not report["clean"]:
            raise RuntimeError(f"EpisodicMemory close failed: {report['failures']}")

    # ---- Async Wrappers -----------------------------------------------------

    async def record_episode_async(
        self,
        context: str,
        action: str,
        outcome: str,
        success: bool,
        emotional_valence: float = 0.0,
        tools_used: list[str] | None = None,
        lessons: list[str] | None = None,
        importance: float = 0.5,
        source: str = "episodic_memory",
        metadata: dict[str, Any] | None = None,
        idempotency_key: str = "",
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
            importance,
            source,
            metadata,
            idempotency_key,
        )

    async def recall_recent_async(self, limit: int = 10) -> list[Episode]:
        return await asyncio.to_thread(self.recall_recent, limit)

    async def recall_similar_async(self, query: str, limit: int = 5) -> list[Episode]:
        return await asyncio.to_thread(self.recall_similar, query, limit)

    async def recall_failures_async(self, limit: int = 10) -> list[Episode]:
        return await asyncio.to_thread(self.recall_failures, limit)

    async def recall_by_tool_async(self, tool_name: str, limit: int = 10) -> list[Episode]:
        return await asyncio.to_thread(self.recall_by_tool, tool_name, limit)

    async def add_lesson_async(self, episode_id: str, lesson: str):
        return await asyncio.to_thread(self.add_lesson, episode_id, lesson)

    async def delete_episodes_async(self, episode_ids: list[str]):
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
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("episodic_memory", exc)
            logger.debug("Constitutional runtime liveness check failed: %s", exc)
            return False

    def _approve_memory_write(
        self,
        context: str,
        action: str,
        outcome: str,
        importance: float,
        *,
        source: str = "episodic_memory",
        metadata: dict[str, Any] | None = None,
        return_decision: bool = False,
    ) -> bool | tuple[bool, Any]:
        preview = f"{context} | {action} | {outcome}".strip()[:240]
        try:
            from core.constitution import get_constitutional_core, unpack_governance_result

            approved, reason, decision = unpack_governance_result(
                get_constitutional_core().approve_memory_write_sync(
                    "episodic_episode",
                    preview,
                    source=source or "episodic_memory",
                    importance=max(0.0, min(1.0, float(importance or 0.0))),
                    metadata={
                        "context": str(context or "")[:120],
                        "action": str(action or "")[:120],
                        **dict(metadata or {}),
                    },
                    return_decision=True,
                )
            )
            if not approved:
                self._last_refusal_reason = str(reason or "")
                logger.info("EpisodicMemory: deferring episode write: %s", reason)
            else:
                self._last_refusal_reason = ""
            if return_decision:
                return approved, decision
            return approved
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('episodic_memory', exc)
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
                if return_decision:
                    return False, None
                return False
            logger.debug("EpisodicMemory constitutional gate unavailable: %s", exc)
            if return_decision:
                return True, None
            return True

    # ---- Core API -----------------------------------------------------------

    def record_episode(
        self,
        context: str,
        action: str,
        outcome: str,
        success: bool,
        emotional_valence: float = 0.0,
        tools_used: list[str] | None = None,
        lessons: list[str] | None = None,
        importance: float = 0.5,
        source: str = "episodic_memory",
        metadata: dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> str:
        """Record a new episode. Returns the episode_id.
        Importance is auto-boosted for failures (we learn more from mistakes).
        Automatically captures current qualia snapshot for mood-congruent recall.
        """
        if not context and not action and not outcome:
            return ""
        episode_metadata = dict(metadata or {})
        stable_key = str(
            idempotency_key
            or episode_metadata.get("memory_log_operation_id")
            or ""
        ).strip()
        if stable_key:
            episode_id = "idem-" + hashlib.sha256(
                stable_key.encode("utf-8")
            ).hexdigest()[:24]
            with self._get_conn() as conn:
                existing = conn.execute(
                    "SELECT context, action, outcome, success, source_authoritative "
                    "FROM episodes "
                    "WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
            if existing is not None:
                if (
                    str(existing["context"] or "") != context
                    or str(existing["action"] or "") != action
                    or str(existing["outcome"] or "") != outcome
                    or bool(existing["success"]) != bool(success)
                ):
                    raise ValueError("episodic idempotency identity collision")
                return episode_id
        else:
            import uuid

            episode_id = str(uuid.uuid4())[:12]
        # Before the check, not after it.
        #
        # The replay sat below the deferral branch, so it ran only on a write
        # the governor had just approved — and the queue only ever fills when
        # the governor is deferring. LIVE, 2026-09-08: "holding deferred
        # writes (50 queued, 50 held so far, 0 landed)". Nothing had landed in
        # that process because nothing could: the retry was behind the failure
        # it retries.
        #
        # Here it runs on every attempt, so the first write after the governor
        # relents drains what was held, and so does the first write of a turn
        # that is itself about to be deferred.
        self._deferred_episodes.replay()
        approved, governance_decision = self._approve_memory_write(
            context,
            action,
            outcome,
            importance,
            source=source,
            metadata=metadata,
            return_decision=True,
        )
        if not approved:
            # A deferral is not a refusal.
            #
            # The ontogeny organ EXPLORES with "deferred" at the executive
            # admission control point, and the comment beside that choice
            # justifies it by saying a deferral only costs time. It costs
            # time only where somebody comes back; here it destroyed the
            # episode, so the exploration's own premise was false at this
            # call site and the loss was systematic — the organ is a policy
            # and defers again under the same conditions.
            #
            # LIVE, 2026-09-07: every episode write on a fresh boot came back
            # "sync_approved|ontogeny:deferred".
            self._hold_if_deferred(
                episode_id,
                context,
                action,
                outcome,
                success,
                emotional_valence,
                tools_used,
                lessons,
                importance,
                source,
                metadata,
                stable_key,
            )
            return ""
        # Rate limiting — prevent flood during rapid tool loops
        # ISSUE 31 fix: Capture constant timestamp for storage consistency
        now_mono = time.monotonic()
        if not stable_key and now_mono - self._last_record_time < self._RECORD_COOLDOWN:
            return ""
            
        # Deduplication — check against last episode content (read-only peek)
        last_episode = self._peek_recent(limit=1)
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
        # Relational/bonding conversations should be preserved to prevent memory loss
        if self._detect_relational_significance(context, action, outcome):
            importance = max(importance, self.RELATIONAL_IMPORTANCE_BOOST)

        # Capture current qualia snapshot for mood-congruent recall
        qualia_snapshot = {}
        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if qualia:
                qualia_snapshot = qualia.get_qualia_for_memory()
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('episodic_memory', e)
            capture_and_log(e, {'module': __name__})

        tools = tools_used or []
        lesson_list = lessons or []

        # Novelty (prediction error) drives consolidation strength: a surprising
        # episode is more likely to be retained (Duszkiewicz 2019; Frank & Kafkas
        # 2021). Sourced from the predictive engine's surprise signal when
        # available, else estimated from dissimilarity to recent episodes.
        novelty = self._estimate_novelty(context, action, outcome)
        if novelty >= 0.7:
            importance = max(importance, min(0.9, importance + 0.15))
        original_valence = emotional_valence

        inserted = False
        authoritative = True

        def _persist() -> None:
            nonlocal inserted, authoritative
            inserted, authoritative = self._insert_and_index(
                episode_id=episode_id,
                now=now,
                context=context,
                action=action,
                outcome=outcome,
                success=success,
                emotional_valence=emotional_valence,
                original_valence=original_valence,
                importance=importance,
                tools=tools,
                lesson_list=lesson_list,
                qualia_snapshot=qualia_snapshot,
                novelty=novelty,
                source=source,
                source_ref=stable_key,
                source_revision=int(
                    episode_metadata.get("conversation_revision") or 0
                ),
                source_metadata=episode_metadata,
            )

        if governance_decision is not None:
            from core.governance_context import governed_scope_sync

            with governed_scope_sync(governance_decision):
                _persist()
        else:
            _persist()

        if not inserted:
            return episode_id
        if not authoritative:
            logger.info(
                "Recorded superseded conversation revision for audit only: %s",
                episode_id,
            )
            return episode_id

        # Causal encode signals — novelty feeds the neuromodulatory system and the
        # encoding is broadcast so self-model / dreaming / identity can react.
        self._on_encode_signals(episode_id, novelty, importance, emotional_valence, success)

        logger.info("📝 Episode recorded: %s (success=%s, importance=%.2f, q=%.2f, novelty=%.2f)",
                    episode_id, success, importance, qualia_snapshot.get("q_norm", 0.0), novelty)
        return episode_id

    def _insert_and_index(
        self,
        *,
        episode_id: str,
        now: float,
        context: str,
        action: str,
        outcome: str,
        success: bool,
        emotional_valence: float,
        original_valence: float,
        importance: float,
        tools: list[str],
        lesson_list: list[str],
        qualia_snapshot: dict[str, Any],
        novelty: float,
        source: str,
        source_ref: str,
        source_revision: int,
        source_metadata: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Persist one episode row, index it for semantic + associative recall,
        and bind its engram cues in the hippocampal index.

        Shared by the governed and ungoverned encode paths so the write logic
        lives in exactly one place.
        """
        with self._lock:
            with self._get_conn() as conn:
                existing = conn.execute(
                    "SELECT context, action, outcome, success, source_authoritative "
                    "FROM episodes "
                    "WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
            if existing is not None:
                if (
                    str(existing["context"] or "") != context
                    or str(existing["action"] or "") != action
                    or str(existing["outcome"] or "") != outcome
                    or bool(existing["success"]) != bool(success)
                ):
                    raise ValueError("episodic episode identity collision")
                return False, bool(existing["source_authoritative"])
            max_retries = 3
            inserted_here = False
            source_session_id = str(source_metadata.get("session_id") or "")[:64]
            source_exchange_id = str(
                source_metadata.get("conversation_exchange_id") or ""
            )[:128]
            authoritative = True
            for attempt in range(max_retries):
                try:
                    with self._get_conn() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        if (
                            str(source or "") == "chat_turn_logger"
                            and source_session_id
                            and source_exchange_id
                            and source_revision > 0
                        ):
                            latest = conn.execute(
                                "SELECT episode_id, source_revision FROM episodes "
                                "WHERE source='chat_turn_logger' "
                                "AND source_session_id=? AND source_exchange_id=? "
                                "ORDER BY source_revision DESC, timestamp DESC LIMIT 1",
                                (source_session_id, source_exchange_id),
                            ).fetchone()
                            if latest is not None:
                                latest_revision = int(latest["source_revision"] or 0)
                                if latest_revision > source_revision:
                                    authoritative = False
                                elif latest_revision == source_revision:
                                    raise ValueError(
                                        "conversation episode revision identity collision"
                                    )
                                else:
                                    conn.execute(
                                        "UPDATE episodes SET source_authoritative=0, "
                                        "superseded_at=?, superseded_by=? "
                                        "WHERE source='chat_turn_logger' "
                                        "AND source_session_id=? AND source_exchange_id=? "
                                        "AND source_revision < ? AND source_authoritative=1",
                                        (
                                            now,
                                            episode_id,
                                            source_session_id,
                                            source_exchange_id,
                                            source_revision,
                                        ),
                                    )
                        cursor = conn.execute(
                            """INSERT OR IGNORE INTO episodes
                               (episode_id, timestamp, context, action, outcome, success,
                                emotional_valence, arousal, importance, participants,
                                tools_used, lessons, tags, linked_semantic_ids, decay_rate,
                                qualia_snapshot, next_decay_eval,
                                fidelity, original_valence, reconsolidation_count,
                                last_reconsolidated, novelty, source, source_ref,
                                source_revision, source_metadata, source_session_id,
                                source_exchange_id, source_authoritative, superseded_at,
                                superseded_by)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                episode_id, now, context, action, outcome,
                                int(success), emotional_valence, 0.5, importance,
                                json.dumps(["user", "aura"]),
                                json.dumps(tools), json.dumps(lesson_list),
                                json.dumps([]), json.dumps([]), 0.01,
                                json.dumps(qualia_snapshot, cls=_SafeEncoder),
                                now + 21600,  # First decay evaluation in 6 hours
                                1.0, original_valence, 0, 0.0, novelty,
                                str(source or "episodic_memory")[:64],
                                str(source_ref or "")[:160],
                                int(source_revision),
                                json.dumps(source_metadata, cls=_SafeEncoder),
                                source_session_id,
                                source_exchange_id,
                                int(authoritative),
                                0.0 if authoritative else now,
                                "" if authoritative else str(latest["episode_id"]),
                            ),
                        )
                        inserted_here = cursor.rowcount == 1
                        if not inserted_here:
                            existing = conn.execute(
                                "SELECT context, action, outcome, success FROM episodes "
                                "WHERE episode_id = ?",
                                (episode_id,),
                            ).fetchone()
                            if existing is None:
                                raise RuntimeError(
                                    "episodic idempotent insert was ignored without an existing row"
                                )
                            if (
                                str(existing["context"] or "") != context
                                or str(existing["action"] or "") != action
                                or str(existing["outcome"] or "") != outcome
                                or bool(existing["success"]) != bool(success)
                            ):
                                raise ValueError("episodic episode identity collision")
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < max_retries - 1:
                        logger.debug("Episode write locked (attempt %d/%d), retrying...", attempt + 1, max_retries)
                        time.sleep(0.5 * (2 ** attempt))
                    else:
                        raise

            if not inserted_here:
                return False, authoritative

            if not authoritative:
                return True, False

            # Index in vector memory for semantic retrieval
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
                except (OSError, ConnectionError, TimeoutError) as e:
                    record_degradation('episodic_memory', e)
                    logger.warning("Failed to index episode in vector memory: %s", e)

            # Bind the engram's associative cues for pattern completion.
            try:
                cues = self._hippocampus.extract_cues(
                    context, action, outcome,
                    tools=tools, qualia_snapshot=qualia_snapshot,
                )
                self._hippocampus.bind(episode_id, cues)
            except (sqlite3.Error, AttributeError, TypeError, ValueError) as e:
                record_degradation('episodic_memory', e)
                logger.debug("Hippocampal cue binding skipped for %s: %s", episode_id, e)

            self._maybe_prune()
            return True, True

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
        """Synchronous consolidation logic run in a worker thread.

        Beyond pruning decayed traces, this is where sleep-style *replay* happens:
        salient engrams are restabilised (made more solid and slower to decay),
        and distressing, high-arousal, repeatedly-reactivated memories become
        candidates for gentle sleep-time emotional processing — a governed,
        bounded therapeutic reconsolidation that mirrors how sleep helps the
        brain metabolise emotional memories.
        """
        pruned = 0
        boosted = 0
        replayed = 0
        softened = 0
        soften_candidates: list[str] = []
        now = time.time()

        try:
            with self._lock:
                with self._get_conn() as conn:
                    # Only calculate decay for records due for evaluation
                    rows = conn.execute(
                        "SELECT * FROM episodes WHERE source_authoritative=1 "
                        "AND next_decay_eval < ? ORDER BY next_decay_eval ASC LIMIT 500",
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

                        salient = episode.importance >= 0.7 or abs(episode.emotional_valence) >= 0.6

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
                        # Sleep replay: salient engrams are restabilised even
                        # without recent access — replay makes them more solid.
                        elif salient and episode.decay_rate > 0.005:
                            new_decay = max(0.004, episode.decay_rate * 0.9)
                            conn.execute(
                                "UPDATE episodes SET decay_rate = ?, next_decay_eval = ? WHERE episode_id = ?",
                                (new_decay, next_eval, episode_id)
                            )
                            replayed += 1
                        else:
                            # Just update the timer
                            conn.execute(
                                "UPDATE episodes SET next_decay_eval = ? WHERE episode_id = ?",
                                (next_eval, episode_id)
                            )

                        # Distressing, high-arousal memories that keep being
                        # reactivated are candidates for sleep-time softening.
                        if (episode.emotional_valence <= -0.4
                                and episode.arousal >= 0.6
                                and episode.reconsolidation_count >= 2):
                            soften_candidates.append(episode_id)
                    
                    # Prune decayed episodes (and their engram cues, same txn —
                    # forgetting a memory must also retire its associative index).
                    if prune_ids:
                        placeholders = ",".join("?" for _ in prune_ids)
                        conn.execute(
                            f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
                            prune_ids
                        )
                        conn.execute(
                            f"DELETE FROM engram_cues WHERE episode_id IN ({placeholders})",
                            prune_ids
                        )
                        pruned = len(prune_ids)

                    conn.commit()

            if pruned or boosted or replayed:
                logger.info(
                    "🧠 Memory consolidation: pruned %d, boosted %d, replayed %d",
                    pruned, boosted, replayed
                )
                self._emit_event("memory.consolidated", {
                    "pruned": pruned, "boosted": boosted, "replayed": replayed,
                })

            # Sleep-time emotional processing — runs OUTSIDE the lock because
            # therapeutic reconsolidation re-acquires it. Bounded per cycle and
            # governed by the constitutional memory-write gate.
            for episode_id in soften_candidates[:3]:
                try:
                    if self.reconsolidate_memory_in_context(
                        episode_id, target_valence=-0.1, intensity=0.25
                    ):
                        softened += 1
                except (sqlite3.Error, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('episodic_memory', exc)
                    logger.debug("Sleep-time softening skipped for %s: %s", episode_id, exc)
            if softened:
                logger.info("🌙 Sleep-time emotional processing softened %d memories", softened)

        except (sqlite3.Error, OSError) as e:
            record_degradation('episodic_memory', e)
            logger.error("Consolidation failed: %s", e)

        return {"pruned": pruned, "boosted": boosted, "replayed": replayed, "softened": softened}

    async def compact_to_semantic(self, batch_size: int = 20) -> dict:
        """Compress old low-strength episodes into semantic summaries.

        This is the episodic→semantic bridge for Zero-Touch memory management.
        Instead of just deleting weak episodes, we extract their essential insight
        and store it in vector memory, then delete the originals.

        Called by AutonomicCore during substrate defrag (every 30 min or at 85% RAM).
        """
        compacted = 0
        try:
            with self._lock:
                with self._get_conn() as conn:
                    # Find episodes that are near-death but haven't been compacted yet
                    rows = conn.execute(
                        """SELECT * FROM episodes
                           WHERE source_authoritative=1 AND importance < 0.5
                           ORDER BY timestamp ASC
                           LIMIT ?""",
                        (batch_size,)
                    ).fetchall()

                    if not rows:
                        return {"compacted": 0}

                    # Build batch summary from the episodes
                    episodes = [self._row_to_episode(r) for r in rows]
                    weak_episodes = [e for e in episodes if e.current_strength() < 0.15]

                    if not weak_episodes:
                        return {"compacted": 0}

                    # Create a single consolidated summary (no LLM needed — just structured text)
                    summaries = []
                    delete_ids = []
                    for ep in weak_episodes:
                        ctx = (ep.context or "")[:100]
                        act = (ep.action or "")[:100]
                        out = (ep.outcome or "")[:100]
                        success_str = "succeeded" if ep.success else "failed"
                        summaries.append(f"- {ctx}: {act} ({success_str}: {out})")
                        delete_ids.append(ep.episode_id)

                    if not delete_ids:
                        return {"compacted": 0}

                    # Store consolidated summary in vector memory
                    try:
                        from core.container import ServiceContainer
                        dual_memory = ServiceContainer.get("dual_memory", default=None)
                        if dual_memory and hasattr(dual_memory, 'store'):
                            consolidated_text = (
                                f"[COMPACTED] {len(delete_ids)} episodic memories consolidated:\n"
                                + "\n".join(summaries[:20])
                            )
                            await dual_memory.store(
                                consolidated_text,
                                metadata={
                                    "type": "episodic_compaction",
                                    "source_count": len(delete_ids),
                                    "timestamp": time.time(),
                                }
                            )
                    except (ImportError, AttributeError, RuntimeError) as store_err:
                        record_degradation('episodic_memory', store_err)
                        logger.debug("Episodic compaction: vector store skipped: %s", store_err)

                    # Delete compacted episodes
                    placeholders = ",".join("?" for _ in delete_ids)
                    conn.execute(
                        f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
                        delete_ids
                    )
                    conn.commit()
                    compacted = len(delete_ids)

            if compacted:
                logger.info("Episodic compaction: %d weak episodes compressed to semantic summary.", compacted)

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('episodic_memory', e)
            logger.error("Episodic compaction failed: %s", e)

        return {"compacted": compacted}

    def recall_recent(self, limit: int = 10) -> list[Episode]:
        """Retrieve the most recent episodes, ranked by memory strength.
        
        Applies Ebbinghaus decay: old, unimportant, unrehearsed memories
        are ranked lower. Fully decayed memories (strength < 0.05) are
        excluded entirely.
        """
        with self._get_conn() as conn:
            # Fetch more than needed so we can filter out decayed ones
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit * 3,)
            ).fetchall()
        
        episodes = [self._row_to_episode(r) for r in rows]
        
        # Filter out fully decayed memories
        alive = [e for e in episodes if e.current_strength() >= 0.05]
        
        # Sort by strength (recency + importance + emotional salience)
        alive.sort(key=lambda e: e.current_strength(), reverse=True)

        self._observe_ranked_recall(alive, returned_count=limit)
        return self._register_recall(alive[:limit])

    def recall_similar(self, query: str, limit: int = 5) -> list[Episode]:
        """Hybrid search: combines vector similarity with keyword matching.

        Mood-Congruent Recall: Episodes formed in a similar qualia state
        to the current state are boosted in ranking.
        """
        seen_ids: set = set()
        combined: list[Episode] = []

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
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('episodic_memory', e)
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
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('episodic_memory', e)
                logger.debug("Keyword recall failed: %s", e)

        # 2b. Associative pattern completion — re-present the query's cues to the
        # hippocampal index to surface engrams that share them (partial cue →
        # whole memory), the way a smell or a word can summon a full episode.
        if len(combined) < limit:
            try:
                cues = HippocampalIndex.extract_cues(query)
                pc = self._hippocampus.pattern_complete(cues, limit=limit, exclude_ids=seen_ids)
                if pc:
                    for ep in self._fetch_by_ids([eid for eid, _ in pc]):
                        if ep.episode_id not in seen_ids:
                            seen_ids.add(ep.episode_id)
                            combined.append(ep)
            except (sqlite3.Error, AttributeError, TypeError, ValueError) as e:
                record_degradation('episodic_memory', e)
                logger.debug("Pattern-completion recall path failed: %s", e)

        # 3. Resolve the ranking by plasticity competition: candidate engrams
        # drive a transient voltage-dependent field so the best-matching trace
        # wins, weakly-relevant ones are gated out below threshold, and the
        # homeostatic bound stops one over-strong trace from swamping recall
        # (anti-confabulation). Falls back to the static importance+recency blend.
        ranked = self._competitive_rank(combined, query)
        self._observe_ranked_recall(ranked, returned_count=limit)
        return self._register_recall(ranked[:limit])







    def recall_failures(self, limit: int = 10) -> list[Episode]:
        """Retrieve recent failures — the best learning opportunities."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 AND success = 0 "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def recall_by_tool(self, tool_name: str, limit: int = 10) -> list[Episode]:
        """Retrieve episodes involving a specific tool."""
        with self._get_conn() as conn:
            # tools_used is a JSON array; use LIKE for simple matching
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 "
                "AND tools_used LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f'%"{tool_name}"%', limit),
            ).fetchall()
        return self._register_recall([self._row_to_episode(r) for r in rows])

    def superseded_source_episode_ids(
        self,
        *,
        source: str,
        session_id: str,
        exchange_id: str,
    ) -> tuple[str, ...]:
        """Return audit rows displaced by the current source revision."""

        safe_source = str(source or "")[:64]
        safe_session = str(session_id or "")[:64]
        safe_exchange = str(exchange_id or "")[:128]
        if not safe_source or not safe_session or not safe_exchange:
            return ()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT episode_id FROM episodes WHERE source=? "
                "AND source_session_id=? AND source_exchange_id=? "
                "AND source_authoritative=0 ORDER BY source_revision ASC",
                (safe_source, safe_session, safe_exchange),
            ).fetchall()
        return tuple(str(row["episode_id"]) for row in rows)

    def get_summary_cached(self, max_age_seconds: float = 30.0) -> dict[str, Any]:
        """TTL-cached summary for telemetry hot paths (health endpoint, panels).

        get_summary() runs eight aggregate queries over the episodes table —
        observed live as a 5.1s event-loop stall inside /health under DB
        contention. Telemetry readers must use this accessor (and call it off
        the event loop); the fresh path stays available for introspection that
        genuinely needs point-in-time numbers.
        """
        now = time.monotonic()
        cached = getattr(self, "_summary_cache", None)
        cached_at = getattr(self, "_summary_cache_at", 0.0)
        if cached is not None and now - cached_at < max_age_seconds:
            return cached
        summary = self.get_summary()
        self._summary_cache = summary
        self._summary_cache_at = now
        return summary

    def get_summary(self) -> dict[str, Any]:
        """Introspection summary for self-model."""
        with self._get_conn() as conn:
            stored_rows = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            superseded = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE source_authoritative = 0"
            ).fetchone()[0]
            total = stored_rows - superseded
            successes = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE source_authoritative = 1 "
                "AND success = 1"
            ).fetchone()[0]
            failures = total - successes
            avg_valence = conn.execute(
                "SELECT AVG(emotional_valence) FROM episodes "
                "WHERE source_authoritative = 1"
            ).fetchone()[0] or 0.0
            important = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE source_authoritative = 1 "
                "AND importance > 0.7"
            ).fetchone()[0]
            avg_fidelity = conn.execute(
                "SELECT AVG(fidelity) FROM episodes WHERE source_authoritative = 1"
            ).fetchone()[0]
            reshaped = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE source_authoritative = 1 "
                "AND reconsolidation_count > 0"
            ).fetchone()[0]
            total_reconsolidations = conn.execute(
                "SELECT COALESCE(SUM(reconsolidation_count), 0) FROM episodes "
                "WHERE source_authoritative = 1"
            ).fetchone()[0]
            avg_novelty = conn.execute(
                "SELECT AVG(novelty) FROM episodes WHERE source_authoritative = 1"
            ).fetchone()[0]
        engram_stats = {}
        try:
            engram_stats = self._hippocampus.stats()
        except (sqlite3.Error, AttributeError):
            pass
        return {
            "total_episodes": total,
            "stored_episode_rows": stored_rows,
            "superseded_conversation_revisions": superseded,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / max(1, total),
            "avg_emotional_valence": round(avg_valence, 3),
            "important_memories": important,
            # Engram dynamics — vividness vs accuracy at the population level.
            "avg_fidelity": round(avg_fidelity if avg_fidelity is not None else 1.0, 3),
            "reshaped_memories": reshaped,
            "total_reconsolidations": int(total_reconsolidations or 0),
            "avg_novelty": round(avg_novelty if avg_novelty is not None else 0.5, 3),
            "indexed_engrams": engram_stats.get("indexed_engrams", 0),
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

    # ---- Engram dynamics: neuromodulation, reconsolidation, recall ----------

    def _service(self, name: str):
        """Best-effort lookup of a runtime service; never raises."""
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get(name, default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('episodic_memory', exc)
            return None

    def _current_qualia(self) -> dict[str, Any]:
        """Present phenomenal context — the 'context that seeps in' on recall."""
        qualia = self._service("qualia_synthesizer")
        if not qualia:
            return {}
        try:
            return qualia.get_qualia_for_memory() or {}
        except (AttributeError, RuntimeError, ValueError) as exc:
            record_degradation('episodic_memory', exc)
            return {}

    def _plasticity_gain(self) -> float:
        """Neuromodulatory lability gain (ACh/dopamine raise it, cortisol impairs).

        This is the model's 'chemicals that make the neurons able to change':
        how moldable memories are *right now*. Baseline ~1.0.
        """
        ncs = self._service("neurochemical_system")
        if not ncs or not hasattr(ncs, "get_mesh_modulation"):
            return 1.0
        try:
            _gain, plasticity, _noise = ncs.get_mesh_modulation()
            return max(0.3, min(2.0, float(plasticity)))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation('episodic_memory', exc)
            return 1.0

    def _emit_event(self, topic: str, payload: dict[str, Any]) -> None:
        """Broadcast a memory event so other subsystems can react. Sync-safe."""
        bus = self._service("event_bus")
        if not bus:
            return
        try:
            if hasattr(bus, "publish_threadsafe"):
                bus.publish_threadsafe(topic, payload)
            elif hasattr(bus, "publish"):
                bus.publish({"type": topic, **payload})
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('episodic_memory', exc)

    def _estimate_novelty(self, context: str, action: str, outcome: str) -> float:
        """Prediction error for a new episode in [0, 1].

        Prefers the predictive subsystem's world-model surprise (a genuine
        cross-subsystem input). Falls back to lexical dissimilarity from recent
        episodes so novelty still works before the predictive engine is online.
        """
        for svc_name in ("predictive_engine", "self_prediction"):
            svc = self._service(svc_name)
            if svc and hasattr(svc, "get_surprise_signal"):
                try:
                    s = float(svc.get_surprise_signal())
                    if s == s:  # reject NaN
                        return max(0.0, min(1.0, s))
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    pass
        try:
            new_tokens = set(HippocampalIndex.extract_cues(context, action, outcome))
            if not new_tokens:
                return 0.5
            best = 0.0
            for ep in self._peek_recent(limit=5):
                old_tokens = set(HippocampalIndex.extract_cues(ep.context, ep.action, ep.outcome))
                if not old_tokens:
                    continue
                overlap = len(new_tokens & old_tokens) / len(new_tokens | old_tokens)
                best = max(best, overlap)
            return round(max(0.0, min(1.0, 1.0 - best)), 4)
        except (sqlite3.Error, AttributeError, TypeError, ValueError) as exc:
            record_degradation('episodic_memory', exc)
            return 0.5

    def _on_encode_signals(
        self, episode_id: str, novelty: float, importance: float,
        emotional_valence: float, success: bool,
    ) -> None:
        """Causal outputs at encoding: novelty → neuromodulation; broadcast."""
        if novelty >= 0.55:
            ncs = self._service("neurochemical_system")
            if ncs and hasattr(ncs, "on_novelty"):
                try:
                    ncs.on_novelty(min(0.6, novelty * 0.6))
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    record_degradation('episodic_memory', exc)
        self._emit_event("memory.encoded", {
            "episode_id": episode_id,
            "novelty": round(float(novelty), 4),
            "importance": round(float(importance), 4),
            "valence": round(float(emotional_valence), 4),
            "success": bool(success),
        })

    def _register_recall(self, episodes: list[Episode]) -> list[Episode]:
        """Recall is not passive read-out.

        Bringing a memory to mind bumps its vividness (access) and can return the
        trace to a labile state where the present rewrites it (reconsolidation).
        Applied once per memory per public recall:
          * neuromodulatory plasticity gain controls lability (impacted-by),
          * content rewrite (drift) passes the constitutional gate (governed),
          * resulting prediction error feeds the neurochemical system (impacts).
        """
        if not episodes:
            return episodes
        now = time.time()
        current_qualia = self._current_qualia()
        lability = self._plasticity_gain()
        drift_events: list[tuple[Episode, ReconsolidationOutcome]] = []

        try:
            with self._lock:
                with self._get_conn() as conn:
                    for ep in episodes:
                        outcome = self._reconsolidation.reconsolidate(
                            now=now,
                            timestamp=ep.timestamp,
                            emotional_valence=ep.emotional_valence,
                            original_valence=ep.original_valence,
                            importance=ep.importance,
                            decay_rate=ep.decay_rate,
                            fidelity=ep.fidelity,
                            reconsolidation_count=ep.reconsolidation_count,
                            last_reconsolidated=ep.last_reconsolidated,
                            current_strength=ep.current_strength(),
                            qualia_snapshot=ep.qualia_snapshot,
                            current_qualia=current_qualia,
                            lability=lability,
                        )
                        # Content rewrite is a governed memory mutation.
                        if outcome.drifted and not self._approve_reconsolidation(ep, outcome):
                            outcome = self._rehearsal_only(ep, outcome, now)

                        # Voltage-gated LTP consolidation: an engram that WON the
                        # plasticity competition for this recall is strengthened,
                        # scaled by its competitive weight and the neuromodulatory
                        # lability gain — and capped (homeostatic bound) so no
                        # single trace can be rehearsed into runaway dominance.
                        consolidated_importance = outcome.importance
                        ltp_weight = self._last_competition_weights.get(ep.episode_id, 0.0)
                        if ltp_weight > 0.0:
                            ltp_gain = 0.03 * ltp_weight * min(1.5, max(0.0, lability))
                            consolidated_importance = min(0.98, outcome.importance + ltp_gain)

                        new_access = ep.access_count + 1
                        conn.execute(
                            """UPDATE episodes SET
                                 access_count = ?, last_accessed = ?,
                                 importance = ?, decay_rate = ?,
                                 emotional_valence = ?, qualia_snapshot = ?,
                                 fidelity = ?, reconsolidation_count = ?,
                                 last_reconsolidated = ?
                               WHERE episode_id = ?""",
                            (
                                new_access, now,
                                consolidated_importance, outcome.decay_rate,
                                outcome.emotional_valence,
                                json.dumps(outcome.qualia_snapshot, cls=_SafeEncoder),
                                outcome.fidelity, outcome.reconsolidation_count,
                                outcome.last_reconsolidated, ep.episode_id,
                            ),
                        )
                        # Reflect the rewrite on the live object handed to callers.
                        ep.access_count = new_access
                        ep.last_accessed = now
                        ep.importance = consolidated_importance
                        ep.decay_rate = outcome.decay_rate
                        ep.emotional_valence = outcome.emotional_valence
                        ep.qualia_snapshot = outcome.qualia_snapshot
                        ep.fidelity = outcome.fidelity
                        ep.reconsolidation_count = outcome.reconsolidation_count
                        ep.last_reconsolidated = outcome.last_reconsolidated
                        if outcome.drifted:
                            drift_events.append((ep, outcome))
                    conn.commit()
        except (sqlite3.Error, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('episodic_memory', exc)
            logger.debug("Recall registration (reconsolidation) degraded: %s", exc)
            return episodes

        for ep, outcome in drift_events:
            self._on_reconsolidation_signals(ep, outcome)
        # Associative learning: engrams recalled together wire together. Drive
        # their slots through the persistent voltage-STDP weight field so the
        # learned association strengthens (and is saved across sessions).
        if len(episodes) >= 2 and is_engram_association_enabled():
            try:
                cue_groups = [HippocampalIndex.extract_cues(ep.full_description) for ep in episodes]
                get_engram_association_field().learn(cue_groups)
            except (AttributeError, ValueError, TypeError) as exc:
                record_degradation("episodic_memory", exc)
        # Competitive LTP weights are consumed once per recall — clear so they
        # cannot leak consolidation into a later recall on a different path.
        self._last_competition_weights = {}
        return episodes

    @staticmethod
    def _rehearsal_only(ep: Episode, outcome: ReconsolidationOutcome, now: float) -> ReconsolidationOutcome:
        """Keep only the benign rehearsal strengthening when governance vetoes the
        content rewrite — the memory is reinforced but not rewritten."""
        return ReconsolidationOutcome(
            fired=outcome.fired,
            drifted=False,
            emotional_valence=ep.emotional_valence,
            qualia_snapshot=ep.qualia_snapshot,
            importance=outcome.importance,
            decay_rate=outcome.decay_rate,
            fidelity=ep.fidelity,
            reconsolidation_count=ep.reconsolidation_count,
            last_reconsolidated=now,
            prediction_error=outcome.prediction_error,
            note="governance_vetoed_drift",
        )

    def _approve_reconsolidation(self, ep: Episode, outcome: ReconsolidationOutcome) -> bool:
        """Constitutional gate for rewriting an existing memory's content.

        Rewriting one's own past is exactly the kind of act that belongs under
        governance, so spontaneous and therapeutic drift both pass through the
        same constitutional memory-write approval used for new episodes.
        """
        try:
            from core.constitution import get_constitutional_core, unpack_governance_result
            delta = outcome.emotional_valence - ep.emotional_valence
            preview = (
                f"reconsolidate {ep.episode_id}: Δvalence={delta:+.3f} "
                f"fidelity={outcome.fidelity:.2f}"
            )
            approved, reason, _decision = unpack_governance_result(
                get_constitutional_core().approve_memory_write_sync(
                    "episodic_reconsolidation",
                    preview,
                    source="reconsolidation",
                    importance=max(0.0, min(1.0, float(ep.importance or 0.0))),
                    metadata={
                        "episode_id": ep.episode_id,
                        "prediction_error": round(float(outcome.prediction_error), 4),
                        "reconsolidation_count": int(outcome.reconsolidation_count),
                    },
                    return_decision=True,
                )
            )
            if not approved:
                logger.debug("Reconsolidation vetoed for %s: %s", ep.episode_id, reason)
            return bool(approved)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation('episodic_memory', exc)
            # When the constitutional runtime is live but the gate errored, fail
            # closed (no rewrite). Otherwise (no runtime, e.g. tests) allow it.
            return not self._constitutional_runtime_live()

    def _on_reconsolidation_signals(self, ep: Episode, outcome: ReconsolidationOutcome) -> None:
        """Causal outputs of a drift: prediction error → neuromodulation; broadcast."""
        pe = float(outcome.prediction_error)
        if pe > 0.05:
            ncs = self._service("neurochemical_system")
            if ncs and hasattr(ncs, "on_prediction_error"):
                try:
                    ncs.on_prediction_error(min(0.6, pe))
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    record_degradation('episodic_memory', exc)
        self._emit_event("memory.reconsolidated", {
            "episode_id": ep.episode_id,
            "prediction_error": round(pe, 4),
            "fidelity": round(float(ep.fidelity), 4),
            "valence": round(float(ep.emotional_valence), 4),
            "valence_drift": ep.valence_drift(),
            "reconsolidation_count": int(ep.reconsolidation_count),
            "note": outcome.note,
        })

    def pattern_complete(self, cue: "str | list[str]", limit: int = 5) -> list[Episode]:
        """Associative recall: complete an episode from a partial cue set.

        Re-presenting part of an engram reinstates the whole (hippocampal pattern
        completion) — complementary to vector similarity and keyword search.
        ``cue`` may be free text or an explicit list of cue tokens.
        """
        if isinstance(cue, str):
            cues = HippocampalIndex.extract_cues(cue)
        else:
            cues = [str(c).lower() for c in (cue or [])]
        if not cues:
            return []
        try:
            matches = self._hippocampus.pattern_complete(cues, limit=limit)
        except (sqlite3.Error, AttributeError, TypeError, ValueError) as exc:
            record_degradation('episodic_memory', exc)
            return []
        if not matches:
            return []
        episodes = self._fetch_by_ids([eid for eid, _ in matches])
        order = {eid: i for i, (eid, _score) in enumerate(matches)}
        episodes.sort(key=lambda e: order.get(e.episode_id, 999))
        return self._register_recall(episodes[:limit])

    def reconsolidate_memory_in_context(
        self,
        episode_id: str,
        target_valence: float,
        intensity: float = 0.5,
        safe_context: dict[str, Any] | None = None,
    ) -> bool:
        """Therapeutic reconsolidation: deliberately revisit a memory in a safe
        context so its emotional tone updates toward ``target_valence``.

        Models reframing a hurtful memory in safety (Speer et al. 2021): the
        memory restabilises a little less aversive, at a small cost to fidelity.
        Governed by the constitutional memory-write gate.
        """
        rows = self._fetch_by_ids([episode_id])
        if not rows:
            return False
        ep = rows[0]
        now = time.time()
        outcome = self._reconsolidation.reconsolidate_in_context(
            now=now,
            emotional_valence=ep.emotional_valence,
            qualia_snapshot=ep.qualia_snapshot,
            importance=ep.importance,
            fidelity=ep.fidelity,
            reconsolidation_count=ep.reconsolidation_count,
            target_valence=float(target_valence),
            intensity=float(intensity),
            safe_context=safe_context or self._current_qualia(),
        )
        if not self._approve_reconsolidation(ep, outcome):
            logger.info("Therapeutic reconsolidation vetoed for %s", episode_id)
            return False
        try:
            with self._lock:
                with self._get_conn() as conn:
                    conn.execute(
                        """UPDATE episodes SET emotional_valence = ?, qualia_snapshot = ?,
                             fidelity = ?, reconsolidation_count = ?, last_reconsolidated = ?
                           WHERE episode_id = ?""",
                        (
                            outcome.emotional_valence,
                            json.dumps(outcome.qualia_snapshot, cls=_SafeEncoder),
                            outcome.fidelity, outcome.reconsolidation_count, now, episode_id,
                        ),
                    )
                    conn.commit()
        except (sqlite3.Error, OSError) as exc:
            record_degradation('episodic_memory', exc)
            return False
        ep.emotional_valence = outcome.emotional_valence
        ep.qualia_snapshot = outcome.qualia_snapshot
        ep.fidelity = outcome.fidelity
        ep.reconsolidation_count = outcome.reconsolidation_count
        ep.last_reconsolidated = now
        self._on_reconsolidation_signals(ep, outcome)
        logger.info("🛋️ Therapeutic reconsolidation of %s → valence %.2f (fidelity %.2f)",
                    episode_id, outcome.emotional_valence, outcome.fidelity)
        return True

    # ---- Internal -----------------------------------------------------------

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a sqlite row (Row object) to an Episode."""
        def load_json(val, default):
            if not val:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                record_degradation("episodic_memory", exc)
                logger.debug("Episode JSON field decode failed: %s", exc)
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
            fidelity=safe_get("fidelity", 1.0),
            original_valence=safe_get("original_valence", None),
            reconsolidation_count=safe_get("reconsolidation_count", 0),
            last_reconsolidated=safe_get("last_reconsolidated", 0.0),
            novelty=safe_get("novelty", 0.5),
            source=safe_get("source", "episodic_memory"),
            source_ref=safe_get("source_ref", ""),
            source_revision=safe_get("source_revision", 0),
            source_metadata=load_json(safe_get("source_metadata", "{}"), {}),
            source_session_id=safe_get("source_session_id", ""),
            source_exchange_id=safe_get("source_exchange_id", ""),
            source_authoritative=bool(safe_get("source_authoritative", 1)),
            superseded_at=safe_get("superseded_at", 0.0),
            superseded_by=safe_get("superseded_by", ""),
        )

    def _fetch_by_ids(self, episode_ids: list[str]) -> list[Episode]:
        """Fetch episodes by ID list (read-only).

        Access stats and reconsolidation are applied by :meth:`_register_recall`
        on the memories actually returned from a public recall, so this stays a
        pure read used by the recall paths and by direct id lookups.
        """
        if not episode_ids:
            return []
        with self._get_conn() as conn:
            placeholders = ",".join("?" for _ in episode_ids)
            rows = conn.execute(
                f"SELECT * FROM episodes WHERE source_authoritative=1 "
                f"AND episode_id IN ({placeholders})",
                episode_ids,
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _peek_recent(self, limit: int = 5) -> list[Episode]:
        """Read-only recent episodes — no access bump, no reconsolidation.

        Used by internal bookkeeping (dedup, novelty estimation) that merely
        glances at memory rather than constituting a deliberate recall, so it
        must not trigger the labile-window dynamics.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _keyword_search(self, query: str, limit: int) -> list[Episode]:
        """Keyword search across context + action + outcome.

        Strategy (progressive relaxation so natural-language LLM queries can
        still recall partial matches without flooding the caller):

        1. Strict AND match across recent episodes — highest precision.
        2. If the strict path finds nothing, retry AND across ALL episodes
           (used to be gated behind a narrow "exact phrase" heuristic; now
           always attempted because the strict path already narrowed scan).
        3. If AND still finds nothing, run an OR match and rank by the number
           of distinct query words that matched. Episodes that match more
           words rank higher; ties break on recency.
        """
        # Normalize and deduplicate words; drop very short tokens that
        # produce false-positive `LIKE '%a%'` matches.
        raw_words = query.lower().split()[:8]
        words = []
        seen = set()
        for w in raw_words:
            w = w.strip(".,;:?!\"'()[]{}")
            if len(w) < 3 or w in seen:
                continue
            seen.add(w)
            words.append(w)
        words = words[:5]
        if not words:
            return []

        and_conditions = " AND ".join(
            "(LOWER(context) LIKE ? OR LOWER(action) LIKE ? OR LOWER(outcome) LIKE ?)"
            for _ in words
        )
        params: list[str] = []
        for w in words:
            pattern = f"%{w}%"
            params.extend([pattern, pattern, pattern])
        recent_scan_limit = max(self.KEYWORD_SEARCH_SCAN_LIMIT, limit * 60)
        with self._get_conn() as conn:
            # 1. Strict AND on recent slice
            rows = conn.execute(
                f"""
                SELECT * FROM (
                    SELECT * FROM episodes WHERE source_authoritative=1
                    ORDER BY timestamp DESC LIMIT ?
                )
                WHERE {and_conditions}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [recent_scan_limit, *params, limit],
            ).fetchall()
            # 2. Strict AND across the full table
            if not rows:
                rows = conn.execute(
                    f"SELECT * FROM episodes WHERE source_authoritative=1 "
                    f"AND {and_conditions} ORDER BY timestamp DESC LIMIT ?",
                    [*params, limit],
                ).fetchall()
            if rows:
                return [self._row_to_episode(r) for r in rows]

            # 3. OR fallback ranked by match count.
            or_conditions = " OR ".join(
                "(LOWER(context) LIKE ? OR LOWER(action) LIKE ? OR LOWER(outcome) LIKE ?)"
                for _ in words
            )
            or_rows = conn.execute(
                f"""
                SELECT * FROM (
                    SELECT * FROM episodes WHERE source_authoritative=1
                    ORDER BY timestamp DESC LIMIT ?
                )
                WHERE {or_conditions}
                """,
                [recent_scan_limit, *params],
            ).fetchall()

        def _match_score(row) -> int:
            haystack = " ".join(
                str(row[k] or "").lower() for k in ("context", "action", "outcome")
            )
            return sum(1 for w in words if w in haystack)

        scored = [(row, _match_score(row)) for row in or_rows]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (item[1], item[0]["timestamp"]), reverse=True)
        return [self._row_to_episode(row) for row, _ in scored[:limit]]

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
                keep_count = self.RETENTION_POLICY.keep_count(count)
                excess = count - keep_count
                victims = [
                    r[0] for r in conn.execute(
                        """SELECT episode_id FROM episodes
                           ORDER BY source_authoritative ASC, importance ASC,
                                    access_count ASC, timestamp ASC
                           LIMIT ?""",
                        (excess,),
                    ).fetchall()
                ]
                if victims:
                    placeholders = ",".join("?" for _ in victims)
                    conn.execute(f"DELETE FROM episodes WHERE episode_id IN ({placeholders})", victims)
                    conn.execute(f"DELETE FROM engram_cues WHERE episode_id IN ({placeholders})", victims)
                    conn.commit()
                    logger.info(
                        "Pruned %s low-importance episodes using %s policy",
                        len(victims),
                        self.RETENTION_POLICY.basis,
                    )

    def delete_episodes(self, episode_ids: list[str]):
        """Hard delete specific episodes (e.g., after consolidation)."""
        if not episode_ids:
            return
        with self._lock:
            try:
                with self._get_conn() as conn:
                    placeholders = ",".join("?" for _ in episode_ids)
                    conn.execute(f"DELETE FROM episodes WHERE episode_id IN ({placeholders})", episode_ids)
                    conn.execute(f"DELETE FROM engram_cues WHERE episode_id IN ({placeholders})", episode_ids)
                    conn.commit()

                # Also remove from vector memory if possible
                if self._vector_memory:
                    try:
                        self._vector_memory.delete_memories(filter_metadata={"episode_id": episode_ids})
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        record_degradation('episodic_memory', e)
                        logger.debug("Vector deletion failed during episode prune: %s", e)
                
                logger.info("🗑️ Deleted %d episodes from storage.", len(episode_ids))
            except (sqlite3.Error, OSError) as e:
                record_degradation('episodic_memory', e)
                logger.error("Failed to delete episodes: %s", e)


# ---------------------------------------------------------------------------
# Global Instance (lazy — only set up when imported)
# ---------------------------------------------------------------------------
_instance: EpisodicMemory | None = None


def get_episodic_memory(vector_memory=None) -> EpisodicMemory:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = EpisodicMemory(vector_memory=vector_memory)
    return _instance
