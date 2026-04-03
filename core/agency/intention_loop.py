"""core/agency/intention_loop.py
Intention Loop — Say / Do / Observe / Revise
==============================================
Closed-loop intentional action cycle.  Every autonomous action Aura
takes passes through this loop:

  1. **Say**   (intend)   — declare what will be done and why.
  2. **Do**    (record_action) — execute and record each tool invocation.
  3. **Observe** (observe) — compare actual outcome against expectation.
  4. **Revise**  (revise)  — update beliefs and self-model from surprise.

The loop provides:
  - Full provenance of every autonomous action.
  - Surprise-based belief revision triggers.
  - Tension creation / resolution tracking.
  - Persistent SQLite storage (WAL mode, matching CognitiveLedger pattern).
  - Registration as ``intention_loop`` in ServiceContainer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.IntentionLoop")


# ── Enums ───────────────────────────────────────────────────────────────────

class IntentionStatus(str, Enum):
    INTENDED = "intended"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ActionRecord:
    """One tool invocation within an intention."""
    tool_name: str
    args: dict
    result: Any
    success: bool
    executed_at: float
    duration_ms: float


@dataclass
class BeliefUpdate:
    """A single belief change triggered by observation."""
    belief: str
    old_confidence: float
    new_confidence: float
    reason: str


@dataclass
class IntentionRecord:
    """Full lifecycle record for one Say-Do-Observe-Revise cycle."""
    id: str
    intention: str                              # "I intend to ..."
    drive: str                                  # Which drive / motivation
    intended_at: float                          # Timestamp
    plan: Optional[List[str]] = None            # Planned steps (natural language)
    actions_taken: List[ActionRecord] = field(default_factory=list)
    observation: Optional[str] = None           # Result summary
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    surprise: float = 0.0                       # Divergence 0-1
    belief_updates: List[BeliefUpdate] = field(default_factory=list)
    self_model_updates: List[str] = field(default_factory=list)
    tension_created: Optional[str] = None
    tension_resolved: Optional[str] = None
    status: IntentionStatus = IntentionStatus.INTENDED
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ── SQLite schema ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS intentions (
    id TEXT PRIMARY KEY,
    intention TEXT NOT NULL,
    drive TEXT NOT NULL,
    intended_at REAL NOT NULL,
    plan_json TEXT DEFAULT '[]',
    actions_json TEXT DEFAULT '[]',
    observation TEXT DEFAULT '',
    expected_outcome TEXT DEFAULT '',
    actual_outcome TEXT DEFAULT '',
    surprise REAL DEFAULT 0.0,
    belief_updates_json TEXT DEFAULT '[]',
    self_model_updates_json TEXT DEFAULT '[]',
    tension_created TEXT DEFAULT '',
    tension_resolved TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'intended',
    completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_intentions_status ON intentions(status);
CREATE INDEX IF NOT EXISTS idx_intentions_intended_at ON intentions(intended_at);
CREATE INDEX IF NOT EXISTS idx_intentions_surprise ON intentions(surprise);
"""


# ── Intention Loop ──────────────────────────────────────────────────────────

class IntentionLoop:
    """Manages the full Say-Do-Observe-Revise cycle.

    Thread-safe.  Persists to SQLite in WAL mode (same pattern as
    CognitiveLedger).  Keeps the last 100 completed intentions in memory
    for fast introspective queries.
    """

    COMPLETED_HISTORY = 100

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self._db_path = Path(db_path)
        else:
            from core.config import config
            self._db_path = config.paths.data_dir / "memory" / "intention_loop.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._active_intentions: Dict[str, IntentionRecord] = {}
        self._completed_intentions: deque[IntentionRecord] = deque(maxlen=self.COMPLETED_HISTORY)

        # Lazy service references (resolved on first use)
        self._ledger = None
        self._belief_engine = None

        self._initialize()

    # ── Initialization ──────────────────────────────────────────────────

    def _initialize(self) -> None:
        try:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._hydrate_active()
            logger.info(
                "IntentionLoop online — %d active, %d completed in history. DB: %s",
                len(self._active_intentions),
                len(self._completed_intentions),
                self._db_path,
            )
        except Exception as e:
            logger.error("IntentionLoop initialization failed: %s", e)
            self._conn = None

    def _hydrate_active(self) -> None:
        """Reload in-progress intentions from disk on restart."""
        if self._conn is None:
            return
        try:
            cur = self._conn.execute(
                "SELECT * FROM intentions WHERE status IN ('intended', 'in_progress') "
                "ORDER BY intended_at ASC"
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                raw = dict(zip(cols, row))
                rec = self._row_to_record(raw)
                self._active_intentions[rec.id] = rec

            # Also load recent completed for history
            cur = self._conn.execute(
                "SELECT * FROM intentions WHERE status IN ('completed', 'failed', 'abandoned') "
                "ORDER BY completed_at DESC LIMIT ?",
                (self.COMPLETED_HISTORY,),
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                raw = dict(zip(cols, row))
                rec = self._row_to_record(raw)
                self._completed_intentions.appendleft(rec)
        except Exception as e:
            logger.error("IntentionLoop hydration failed: %s", e)

    # ── Service accessors (lazy) ────────────────────────────────────────

    def _get_ledger(self):
        if self._ledger is None:
            try:
                from core.container import ServiceContainer
                self._ledger = ServiceContainer.get("cognitive_ledger", default=None)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        return self._ledger

    def _get_belief_engine(self):
        if self._belief_engine is None:
            try:
                from core.container import ServiceContainer
                self._belief_engine = ServiceContainer.get("belief_revision_engine", default=None)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        return self._belief_engine

    # ── SAY: Declare intention ──────────────────────────────────────────

    def intend(
        self,
        intention: str,
        drive: str,
        expected_outcome: Optional[str] = None,
        plan: Optional[List[str]] = None,
    ) -> str:
        """Declare an intention. Returns the intention_id."""
        intention_id = uuid.uuid4().hex[:12]
        rec = IntentionRecord(
            id=intention_id,
            intention=intention,
            drive=drive,
            intended_at=time.time(),
            plan=plan,
            expected_outcome=expected_outcome,
            status=IntentionStatus.INTENDED,
        )

        with self._lock:
            self._active_intentions[intention_id] = rec

        self._persist(rec)

        # Record to CognitiveLedger
        ledger = self._get_ledger()
        if ledger:
            from core.resilience.cognitive_ledger import Transition, TransitionType
            t = Transition.create(
                ttype=TransitionType.ACTION_PROPOSE,
                subsystem="intention_loop",
                cause=f"intend: {intention[:80]}",
                payload={
                    "intention_id": intention_id,
                    "intention": intention,
                    "drive": drive,
                    "expected_outcome": expected_outcome or "",
                    "plan": plan or [],
                },
                prior_hash=self._state_hash(rec),
            )
            ledger.append(t)

        logger.info("Intention declared [%s]: %s (drive=%s)", intention_id, intention[:60], drive)
        return intention_id

    # ── DO: Record action execution ─────────────────────────────────────

    def record_action(
        self,
        intention_id: str,
        tool_name: str,
        args: dict,
        result: Any,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a tool invocation against an intention."""
        with self._lock:
            rec = self._active_intentions.get(intention_id)
            if rec is None:
                logger.warning("record_action: unknown intention_id %s", intention_id)
                return

            action = ActionRecord(
                tool_name=tool_name,
                args=args,
                result=result,
                success=success,
                executed_at=time.time(),
                duration_ms=duration_ms,
            )
            rec.actions_taken.append(action)
            rec.status = IntentionStatus.IN_PROGRESS

        self._persist(rec)

        # Record to CognitiveLedger
        ledger = self._get_ledger()
        if ledger:
            from core.resilience.cognitive_ledger import Transition, TransitionType
            t = Transition.create(
                ttype=TransitionType.ACTION_COMMIT,
                subsystem="intention_loop",
                cause=f"action: {tool_name} for intention {intention_id}",
                payload={
                    "intention_id": intention_id,
                    "tool_name": tool_name,
                    "args": _safe_serialize(args),
                    "success": success,
                    "duration_ms": duration_ms,
                },
                prior_hash=self._state_hash(rec),
            )
            ledger.append(t)

        logger.debug(
            "Action recorded [%s]: %s success=%s (%.1fms)",
            intention_id, tool_name, success, duration_ms,
        )

    # ── OBSERVE: Compare expected vs actual ─────────────────────────────

    def observe(
        self,
        intention_id: str,
        observation: str,
        actual_outcome: str,
    ) -> float:
        """Record observation and calculate surprise. Returns surprise score."""
        with self._lock:
            rec = self._active_intentions.get(intention_id)
            if rec is None:
                logger.warning("observe: unknown intention_id %s", intention_id)
                return 0.0

            rec.observation = observation
            rec.actual_outcome = actual_outcome

            # Calculate surprise as semantic distance between expected and actual
            rec.surprise = self._calculate_surprise(rec.expected_outcome, actual_outcome)

        self._persist(rec)

        # Record to CognitiveLedger
        ledger = self._get_ledger()
        if ledger:
            from core.resilience.cognitive_ledger import Transition, TransitionType
            t = Transition.create(
                ttype=TransitionType.PERCEPT,
                subsystem="intention_loop",
                cause=f"observe: intention {intention_id} surprise={rec.surprise:.2f}",
                payload={
                    "intention_id": intention_id,
                    "observation": observation[:500],
                    "actual_outcome": actual_outcome[:500],
                    "expected_outcome": (rec.expected_outcome or "")[:500],
                    "surprise": rec.surprise,
                },
                prior_hash=self._state_hash(rec),
            )
            ledger.append(t)

        if rec.surprise > 0.5:
            logger.info(
                "HIGH SURPRISE [%s]: expected=%s actual=%s (surprise=%.2f)",
                intention_id,
                (rec.expected_outcome or "?")[:40],
                actual_outcome[:40],
                rec.surprise,
            )

        return rec.surprise

    # ── REVISE: Update beliefs and self-model ───────────────────────────

    def revise(
        self,
        intention_id: str,
        belief_updates: Optional[List[BeliefUpdate]] = None,
        self_model_updates: Optional[List[str]] = None,
        tension_created: Optional[str] = None,
        tension_resolved: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Close the loop: record revisions and finalize the intention."""
        with self._lock:
            rec = self._active_intentions.get(intention_id)
            if rec is None:
                logger.warning("revise: unknown intention_id %s", intention_id)
                return

            rec.belief_updates = belief_updates or []
            rec.self_model_updates = self_model_updates or []
            rec.tension_created = tension_created
            rec.tension_resolved = tension_resolved
            rec.status = IntentionStatus.COMPLETED if success else IntentionStatus.FAILED
            rec.completed_at = time.time()

            # Move from active to completed
            del self._active_intentions[intention_id]
            self._completed_intentions.append(rec)

        self._persist(rec)

        # Push belief updates to BeliefRevisionEngine
        belief_engine = self._get_belief_engine()
        if belief_engine and rec.belief_updates:
            for bu in rec.belief_updates:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(
                            belief_engine.process_new_claim(
                                claim=bu.belief,
                                domain="self",
                                source="intention_loop",
                                confidence=bu.new_confidence,
                            )
                        )
                    else:
                        asyncio.run_coroutine_threadsafe(
                            belief_engine.process_new_claim(
                                claim=bu.belief,
                                domain="self",
                                source="intention_loop",
                                confidence=bu.new_confidence,
                            ),
                            loop,
                        )
                except Exception as e:
                    logger.debug("Belief push deferred: %s", e)

        # Record to CognitiveLedger
        ledger = self._get_ledger()
        if ledger:
            from core.resilience.cognitive_ledger import Transition, TransitionType
            t = Transition.create(
                ttype=TransitionType.SELF_MODEL_UPDATE,
                subsystem="intention_loop",
                cause=f"revise: intention {intention_id} status={rec.status.value}",
                payload={
                    "intention_id": intention_id,
                    "belief_updates": [asdict(bu) for bu in rec.belief_updates],
                    "self_model_updates": rec.self_model_updates,
                    "tension_created": tension_created or "",
                    "tension_resolved": tension_resolved or "",
                    "final_status": rec.status.value,
                },
                prior_hash=self._state_hash(rec),
            )
            ledger.append(t)

        logger.info(
            "Intention %s [%s]: %d belief updates, %d self-model updates",
            rec.status.value, intention_id,
            len(rec.belief_updates), len(rec.self_model_updates),
        )

    # ── ABANDON ─────────────────────────────────────────────────────────

    def abandon(self, intention_id: str, reason: str) -> None:
        """Mark an intention as abandoned."""
        with self._lock:
            rec = self._active_intentions.get(intention_id)
            if rec is None:
                logger.warning("abandon: unknown intention_id %s", intention_id)
                return

            rec.status = IntentionStatus.ABANDONED
            rec.completed_at = time.time()
            rec.observation = f"ABANDONED: {reason}"

            del self._active_intentions[intention_id]
            self._completed_intentions.append(rec)

        self._persist(rec)

        # Record to CognitiveLedger
        ledger = self._get_ledger()
        if ledger:
            from core.resilience.cognitive_ledger import Transition, TransitionType
            t = Transition.create(
                ttype=TransitionType.ACTION_COMMIT,
                subsystem="intention_loop",
                cause=f"abandon: intention {intention_id} — {reason[:80]}",
                payload={
                    "intention_id": intention_id,
                    "reason": reason,
                    "final_status": IntentionStatus.ABANDONED.value,
                },
                prior_hash=self._state_hash(rec),
            )
            ledger.append(t)

        logger.info("Intention ABANDONED [%s]: %s", intention_id, reason[:80])

    # ── Query methods ───────────────────────────────────────────────────

    def get_open_intentions(self) -> List[IntentionRecord]:
        """Return all active (intended or in-progress) intentions."""
        with self._lock:
            return list(self._active_intentions.values())

    def get_recent_surprises(self, threshold: float = 0.5) -> List[IntentionRecord]:
        """Return completed intentions with surprise above threshold."""
        return [
            rec for rec in self._completed_intentions
            if rec.surprise >= threshold
        ]

    def get_revision_history(self) -> List[IntentionRecord]:
        """Return completed intentions that produced belief or self-model updates."""
        return [
            rec for rec in self._completed_intentions
            if rec.belief_updates or rec.self_model_updates
        ]

    def get_intention(self, intention_id: str) -> Optional[IntentionRecord]:
        """Look up a specific intention by ID (active or completed)."""
        with self._lock:
            rec = self._active_intentions.get(intention_id)
            if rec:
                return rec
        for rec in self._completed_intentions:
            if rec.id == intention_id:
                return rec
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return loop statistics."""
        with self._lock:
            active_count = len(self._active_intentions)
        completed_count = len(self._completed_intentions)
        high_surprise = len(self.get_recent_surprises())
        revision_count = len(self.get_revision_history())

        total_from_db = 0
        if self._conn:
            try:
                cur = self._conn.execute("SELECT COUNT(*) FROM intentions")
                total_from_db = cur.fetchone()[0]
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        return {
            "active_intentions": active_count,
            "completed_in_memory": completed_count,
            "total_persisted": total_from_db,
            "high_surprise_recent": high_surprise,
            "revisions_recent": revision_count,
            "db_path": str(self._db_path),
        }

    # ── Surprise calculation ────────────────────────────────────────────

    def _calculate_surprise(
        self,
        expected: Optional[str],
        actual: str,
    ) -> float:
        """Compute surprise as semantic distance between expected and actual.

        Uses a lightweight token-overlap heuristic.  When an embedding
        service is available in the container, it falls back to cosine
        distance for richer comparison.
        """
        if not expected:
            return 0.3  # Mild surprise when no expectation was set

        # Try embedding-based comparison first
        try:
            from core.container import ServiceContainer
            embedder = ServiceContainer.get("embedding_engine", default=None)
            if embedder and hasattr(embedder, "similarity"):
                sim = embedder.similarity(expected, actual)
                return round(max(0.0, min(1.0, 1.0 - sim)), 3)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # Fallback: normalised token overlap (Jaccard distance)
        expected_tokens = set(expected.lower().split())
        actual_tokens = set(actual.lower().split())
        if not expected_tokens and not actual_tokens:
            return 0.0
        union = expected_tokens | actual_tokens
        intersection = expected_tokens & actual_tokens
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        return round(max(0.0, min(1.0, 1.0 - jaccard_sim)), 3)

    # ── Persistence ─────────────────────────────────────────────────────

    def _persist(self, rec: IntentionRecord) -> bool:
        """Upsert an IntentionRecord to SQLite."""
        if self._conn is None:
            return False
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO intentions
                    (id, intention, drive, intended_at, plan_json, actions_json,
                     observation, expected_outcome, actual_outcome, surprise,
                     belief_updates_json, self_model_updates_json,
                     tension_created, tension_resolved, status, completed_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec.id,
                        rec.intention,
                        rec.drive,
                        rec.intended_at,
                        json.dumps(rec.plan or [], default=str),
                        json.dumps(
                            [asdict(a) for a in rec.actions_taken],
                            default=str,
                        ),
                        rec.observation or "",
                        rec.expected_outcome or "",
                        rec.actual_outcome or "",
                        rec.surprise,
                        json.dumps(
                            [asdict(bu) for bu in rec.belief_updates],
                            default=str,
                        ),
                        json.dumps(rec.self_model_updates, default=str),
                        rec.tension_created or "",
                        rec.tension_resolved or "",
                        rec.status.value,
                        rec.completed_at,
                    ),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("IntentionLoop persist failed: %s", e)
                return False

    def _row_to_record(self, raw: Dict[str, Any]) -> IntentionRecord:
        """Deserialize a database row into an IntentionRecord."""
        actions_data = json.loads(raw.get("actions_json", "[]"))
        actions = [
            ActionRecord(
                tool_name=a["tool_name"],
                args=a.get("args", {}),
                result=a.get("result"),
                success=a.get("success", False),
                executed_at=a.get("executed_at", 0.0),
                duration_ms=a.get("duration_ms", 0.0),
            )
            for a in actions_data
        ]

        bu_data = json.loads(raw.get("belief_updates_json", "[]"))
        belief_updates = [
            BeliefUpdate(
                belief=b["belief"],
                old_confidence=b.get("old_confidence", 0.0),
                new_confidence=b.get("new_confidence", 0.0),
                reason=b.get("reason", ""),
            )
            for b in bu_data
        ]

        plan_data = json.loads(raw.get("plan_json", "[]"))

        return IntentionRecord(
            id=raw["id"],
            intention=raw["intention"],
            drive=raw["drive"],
            intended_at=raw["intended_at"],
            plan=plan_data if plan_data else None,
            actions_taken=actions,
            observation=raw.get("observation") or None,
            expected_outcome=raw.get("expected_outcome") or None,
            actual_outcome=raw.get("actual_outcome") or None,
            surprise=raw.get("surprise", 0.0),
            belief_updates=belief_updates,
            self_model_updates=json.loads(raw.get("self_model_updates_json", "[]")),
            tension_created=raw.get("tension_created") or None,
            tension_resolved=raw.get("tension_resolved") or None,
            status=IntentionStatus(raw.get("status", "intended")),
            completed_at=raw.get("completed_at"),
        )

    # ── Maintenance ─────────────────────────────────────────────────────

    def prune_old(self, max_age_days: int = 30) -> int:
        """Delete intentions older than max_age_days. Returns rows deleted."""
        if self._conn is None:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM intentions WHERE completed_at IS NOT NULL AND completed_at < ?",
                    (cutoff,),
                )
                deleted = cur.rowcount
                self._conn.commit()
                if deleted:
                    logger.info(
                        "IntentionLoop: pruned %d intentions older than %d days.",
                        deleted, max_age_days,
                    )
                return deleted
            except Exception as e:
                logger.error("IntentionLoop prune failed: %s", e)
                return 0

    def close(self) -> None:
        """Shut down the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _state_hash(rec: IntentionRecord) -> str:
        """Quick hash for ledger prior_state_hash."""
        blob = f"{rec.id}:{rec.status.value}:{len(rec.actions_taken)}:{rec.surprise}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-safe for ledger payloads."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[IntentionLoop] = None


def get_intention_loop() -> IntentionLoop:
    """Return the global IntentionLoop singleton."""
    global _instance
    if _instance is None:
        _instance = IntentionLoop()
        # Register with ServiceContainer
        try:
            from core.container import ServiceContainer
            ServiceContainer.register_instance("intention_loop", _instance)
        except Exception as e:
            logger.debug("IntentionLoop: ServiceContainer registration deferred: %s", e)
    return _instance
