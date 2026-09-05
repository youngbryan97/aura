"""core/resilience/cognitive_ledger.py
Cognitive Ledger — Structured Transition Journal
=================================================
Upgrades the simple WAL into a full append-only, replayable, branchable
state transition ledger.  Every meaningful organismic state change is
recorded as a ``Transition`` with causal provenance, prior/next state
hashes, side-effect manifests, and rollback eligibility.

The ledger gives Aura:
  - **Replayability**: replay transitions into a clean runtime to verify
    deterministic identity re-emergence.
  - **Forensic debugging**: answer exactly when continuity drift began.
  - **Safe self-modification**: test patches against recent real transition
    history before promotion.
  - **Identity continuity proofs**: compare state signatures across
    restarts and rollbacks.
  - **Counterfactual branching**: fork from any transition and test
    alternate policy conditions.

Implementation: SQLite WAL-mode database with append-only semantics.
The old CognitiveWAL (wal.jsonl) is preserved as a lightweight fallback;
this ledger is the canonical record.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import open_tracked

logger = logging.getLogger("Aura.CognitiveLedger")

_SQLITE_CORRUPTION_MARKERS = (
    "database disk image is malformed",
    "file is not a database",
    "database schema is corrupt",
    "malformed database",
)


def _is_sqlite_corruption(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _SQLITE_CORRUPTION_MARKERS)


class TransitionType(str, Enum):
    PERCEPT = "percept"
    WORKSPACE_WIN = "workspace_win"
    MEMORY_WRITE = "memory_write"
    GOAL_CREATE = "goal_create"
    GOAL_UPDATE = "goal_update"
    POLICY_UPDATE = "policy_update"
    ACTION_PROPOSE = "action_propose"
    ACTION_COMMIT = "action_commit"
    SELF_MODEL_UPDATE = "self_model_update"
    PATCH_APPLY = "patch_apply"
    PATCH_ROLLBACK = "patch_rollback"
    MODE_CHANGE = "mode_change"
    AFFECT_SHIFT = "affect_shift"
    TICK_COMPLETE = "tick_complete"
    BOOT = "boot"
    SHUTDOWN = "shutdown"
    # SDOR (Say→Do→Observe→Revise) chain types
    INTENTION_CREATE = "intention_create"
    INTENTION_EXECUTE = "intention_execute"
    OBSERVATION = "observation"
    BELIEF_REVISION = "belief_revision"
    INTENTION_ABANDON = "intention_abandon"
    # Convergence
    COHERENCE_REPORT = "coherence_report"
    TENSION_CREATE = "tension_create"
    TENSION_RESOLVE = "tension_resolve"
    SELF_EVOLUTION = "self_evolution"


@dataclass
class Transition:
    """One atomic state change in the organism's history."""
    id: str
    parent_id: str | None
    ts: float
    ttype: TransitionType
    subsystem: str
    cause: str
    payload: dict[str, Any]
    prior_state_hash: str
    next_state_hash: str = ""
    confidence: float = 0.5
    uncertainty: float = 0.5
    side_effects: list[dict[str, Any]] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    rollback_eligible: bool = True

    @staticmethod
    def create(
        ttype: TransitionType,
        subsystem: str,
        cause: str,
        payload: dict[str, Any],
        prior_hash: str,
        parent_id: str | None = None,
        confidence: float = 0.5,
        uncertainty: float = 0.5,
    ) -> Transition:
        return Transition(
            id=uuid.uuid4().hex[:12],
            parent_id=parent_id,
            ts=time.time(),
            ttype=ttype,
            subsystem=subsystem,
            cause=cause,
            payload=payload,
            prior_state_hash=prior_hash,
            confidence=confidence,
            uncertainty=uncertainty,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ttype"] = self.ttype.value
        return d


_SCHEMA = """
CREATE TABLE IF NOT EXISTS transitions (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    ts REAL NOT NULL,
    ttype TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    cause TEXT NOT NULL,
    prior_state_hash TEXT NOT NULL,
    next_state_hash TEXT NOT NULL DEFAULT '',
    confidence REAL DEFAULT 0.5,
    uncertainty REAL DEFAULT 0.5,
    payload_json TEXT NOT NULL DEFAULT '{}',
    side_effects_json TEXT DEFAULT '[]',
    memory_writes_json TEXT DEFAULT '[]',
    rollback_eligible INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_ts ON transitions(ts);
CREATE INDEX IF NOT EXISTS idx_ttype ON transitions(ttype);
CREATE INDEX IF NOT EXISTS idx_subsystem ON transitions(subsystem);

CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    transition_id TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    state_json TEXT NOT NULL
);
"""


class CognitiveLedger:
    """Append-only, replayable state transition journal.

    Thread-safe.  Uses SQLite in WAL mode for concurrent reads during
    writes.  All mutations go through ``append()``.
    """

    MAX_PAYLOAD_SIZE = 32_000  # Truncate oversized payloads

    def __init__(self, db_path: str | None = None):
        if db_path:
            self._db_path = Path(db_path)
        else:
            from core.config import config
            self._db_path = config.paths.data_dir / "memory" / "cognitive_ledger.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._transition_count = 0
        self._last_transition_id: str | None = None
        self._initialize()

    def _initialize(self):
        try:
            self._conn = open_tracked(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            # Count existing transitions
            cur = self._conn.execute("SELECT COUNT(*) FROM transitions")
            self._transition_count = cur.fetchone()[0]
            # Get last transition
            cur = self._conn.execute("SELECT id FROM transitions ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
            self._last_transition_id = row[0] if row else None
            logger.info(
                "CognitiveLedger online — %d transitions loaded. DB: %s",
                self._transition_count, self._db_path,
            )
        except (sqlite3.Error, OSError) as e:
            if _is_sqlite_corruption(e) and self.recover_storage(e):
                return
            record_degradation('cognitive_ledger', e)
            logger.error("CognitiveLedger initialization failed: %s", e)
            self._conn = None

    def close(self) -> None:
        """Release the sqlite handle.

        The ledger is a process-global singleton and its connection outlives
        every caller, which is correct for the live runtime and a leak anywhere
        the process is meant to be torn down and rebuilt. Idempotent, and it
        never raises: closing is cleanup, and cleanup that can fail turns one
        problem into two.
        """
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                conn.close()
            except sqlite3.Error as exc:
                logger.debug("CognitiveLedger: close failed: %s", exc)

    def recover_storage(self, reason: BaseException | str) -> bool:
        """Quarantine a corrupt ledger database and rebuild an empty ledger.

        The ledger is append-only evidence. If SQLite reports corruption, the
        safe production behavior is to preserve the damaged files for forensics
        and bring a fresh ledger online instead of crashing supervision.
        """
        timestamp = int(time.time())
        quarantine_dir = self._db_path.parent / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        reason_label = type(reason).__name__ if isinstance(reason, BaseException) else "reason"

        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error as exc:
                    logger.debug("CognitiveLedger: close during recovery failed: %s", exc)
                self._conn = None

            moved = []
            for source in (
                self._db_path,
                Path(f"{self._db_path}-wal"),
                Path(f"{self._db_path}-shm"),
            ):
                if not source.exists():
                    continue
                target = quarantine_dir / f"{source.name}.corrupt.{timestamp}"
                try:
                    source.replace(target)
                    moved.append(str(target))
                except OSError as exc:
                    logger.warning("CognitiveLedger: could not quarantine corrupt file %s: %s", source, exc)

            self._transition_count = 0
            self._last_transition_id = None

        logger.warning(
            "CognitiveLedger recovered from corrupt storage (%s). Quarantined files: %s",
            reason_label,
            moved or "none",
        )
        try:
            self._initialize()
        except (sqlite3.Error, OSError) as exc:
            record_degradation("cognitive_ledger", exc)
            logger.error("CognitiveLedger recovery reinitialize failed: %s", exc)
            self._conn = None
        return self._conn is not None

    # ── Core operations ──────────────────────────────────────────────────

    def append(self, t: Transition) -> bool:
        """Append a transition to the ledger. Returns True on success."""
        if self._conn is None:
            return False
        with self._lock:
            try:
                payload_json = json.dumps(t.payload, default=str)
                if len(payload_json) > self.MAX_PAYLOAD_SIZE:
                    payload_json = json.dumps({"_truncated": True, "cause": t.cause})
                self._conn.execute(
                    """INSERT INTO transitions
                    (id, parent_id, ts, ttype, subsystem, cause,
                     prior_state_hash, next_state_hash,
                     confidence, uncertainty,
                     payload_json, side_effects_json, memory_writes_json,
                     rollback_eligible)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        t.id, t.parent_id, t.ts, t.ttype.value,
                        t.subsystem, t.cause,
                        t.prior_state_hash, t.next_state_hash,
                        t.confidence, t.uncertainty,
                        payload_json,
                        json.dumps(t.side_effects, default=str),
                        json.dumps(t.memory_writes, default=str),
                        int(t.rollback_eligible),
                    ),
                )
                self._conn.commit()
                self._transition_count += 1
                self._last_transition_id = t.id
                return True
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger append failed: %s", e)
                return False

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent transitions as dicts."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT * FROM transitions ORDER BY ts DESC LIMIT ?", (limit,)
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger get_recent failed: %s", e)
                return []

    def get_by_subsystem(self, subsystem: str, limit: int = 50) -> list[dict]:
        """Filter transitions by originating subsystem."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT * FROM transitions WHERE subsystem=? ORDER BY ts DESC LIMIT ?",
                    (subsystem, limit),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger get_by_subsystem failed: %s", e)
                return []

    def get_since(self, since_ts: float) -> list[dict]:
        """Return all transitions since a given timestamp."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT * FROM transitions WHERE ts >= ? ORDER BY ts ASC",
                    (since_ts,),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger get_since failed: %s", e)
                return []

    # ── Snapshots ────────────────────────────────────────────────────────

    def save_snapshot(self, state_json: str, state_hash: str) -> bool:
        """Save a periodic state snapshot for fast recovery."""
        if self._conn is None:
            return False
        with self._lock:
            try:
                snap_id = uuid.uuid4().hex[:12]
                self._conn.execute(
                    "INSERT INTO snapshots (id, ts, transition_id, state_hash, state_json) "
                    "VALUES (?,?,?,?,?)",
                    (snap_id, time.time(), self._last_transition_id or "", state_hash, state_json),
                )
                self._conn.commit()
                logger.debug("CognitiveLedger: snapshot saved (hash=%s)", state_hash[:12])
                return True
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger snapshot failed: %s", e)
                return False

    def get_latest_snapshot(self) -> dict | None:
        """Return the most recent snapshot."""
        if self._conn is None:
            return None
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT * FROM snapshots ORDER BY ts DESC LIMIT 1"
                )
                cols = [d[0] for d in cur.description]
                row = cur.fetchone()
                return dict(zip(cols, row)) if row else None
            except (sqlite3.Error, OSError):
                return None

    # ── Replay support ───────────────────────────────────────────────────

    def replay_since_snapshot(self, snapshot_ts: float) -> list[dict]:
        """Return all transitions after a snapshot for replay."""
        return self.get_since(snapshot_ts)

    # ── Maintenance ──────────────────────────────────────────────────────

    def compact(self) -> bool:
        """Run WAL checkpoint to reclaim disk space."""
        if self._conn is None:
            return False
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.debug("CognitiveLedger: WAL checkpoint complete.")
                return True
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger compact failed: %s", e)
                return False

    def prune_old(self, max_age_days: int = 7) -> int:
        """Delete transitions older than max_age_days, keeping snapshots intact.

        Returns number of rows deleted.
        """
        if self._conn is None:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM transitions WHERE ts < ? AND ttype NOT IN ('boot', 'shutdown')",
                    (cutoff,),
                )
                deleted = cur.rowcount
                self._conn.commit()
                if deleted:
                    self._transition_count = max(0, self._transition_count - deleted)
                    logger.info(
                        "CognitiveLedger: pruned %d transitions older than %d days.",
                        deleted, max_age_days,
                    )
                return deleted
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger prune failed: %s", e)
                return 0

    # ── SDOR Chain Queries ────────────────────────────────────────────────

    def get_intention_chain(self, intention_id: str) -> list[dict]:
        """Get all transitions in an intention's SDOR chain (linked by parent_id or payload.intention_id)."""
        if self._conn is None:
            return []
        sdor_types = ("intention_create", "intention_execute", "observation",
                      "belief_revision", "intention_abandon", "action_propose", "action_commit")
        with self._lock:
            try:
                # Search both by parent chain and by payload reference
                cur = self._conn.execute(
                    """SELECT * FROM transitions
                       WHERE (parent_id = ? OR payload_json LIKE ?)
                       AND ttype IN ({})
                       ORDER BY ts ASC""".format(",".join("?" for _ in sdor_types)),
                    (intention_id, f'%"{intention_id}"%', *sdor_types),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger get_intention_chain failed: %s", e)
                return []

    def get_belief_revisions(self, since_ts: float = 0, limit: int = 50) -> list[dict]:
        """Get all belief revision transitions since a timestamp."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT * FROM transitions WHERE ttype = 'belief_revision' AND ts >= ? ORDER BY ts DESC LIMIT ?",
                    (since_ts, limit),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger get_belief_revisions failed: %s", e)
                return []

    def get_autobiographical_summary(self, hours: float = 24, limit: int = 200) -> dict[str, Any]:
        """Generate an autobiographical summary of the last N hours.

        Returns counts by type, key events, self-model changes, and
        intention completion rate.
        """
        if self._conn is None:
            return {}
        since = time.time() - (hours * 3600)
        with self._lock:
            try:
                # Counts by type
                cur = self._conn.execute(
                    "SELECT ttype, COUNT(*) FROM transitions WHERE ts >= ? GROUP BY ttype",
                    (since,),
                )
                by_type = {row[0]: row[1] for row in cur.fetchall()}

                # Self-model updates
                cur = self._conn.execute(
                    "SELECT cause, payload_json FROM transitions WHERE ttype IN ('self_model_update', 'self_evolution') AND ts >= ? ORDER BY ts DESC LIMIT 20",
                    (since,),
                )
                self_changes = [{"cause": r[0], "payload": r[1][:200]} for r in cur.fetchall()]

                # Intention stats
                created = by_type.get("intention_create", 0)
                executed = by_type.get("intention_execute", 0)
                abandoned = by_type.get("intention_abandon", 0)
                revised = by_type.get("belief_revision", 0)

                # Key events (high-confidence transitions)
                cur = self._conn.execute(
                    "SELECT ttype, subsystem, cause, ts FROM transitions WHERE ts >= ? AND confidence >= 0.7 ORDER BY confidence DESC LIMIT 20",
                    (since,),
                )
                key_events = [{"type": r[0], "subsystem": r[1], "cause": r[2], "ts": r[3]} for r in cur.fetchall()]

                return {
                    "period_hours": hours,
                    "total_transitions": sum(by_type.values()),
                    "by_type": by_type,
                    "self_changes": self_changes,
                    "intentions_created": created,
                    "intentions_executed": executed,
                    "intentions_abandoned": abandoned,
                    "belief_revisions": revised,
                    "intention_completion_rate": executed / max(1, created),
                    "key_events": key_events,
                }
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger autobiographical summary failed: %s", e)
                return {}

    def get_coherence_history(self, limit: int = 50) -> list[dict]:
        """Get recent coherence reports from the ledger."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                cur = self._conn.execute(
                    "SELECT ts, payload_json FROM transitions WHERE ttype = 'coherence_report' ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
                results = []
                for row in cur.fetchall():
                    try:
                        payload = json.loads(row[1])
                        payload["ts"] = row[0]
                        results.append(payload)
                    except json.JSONDecodeError as _exc:
                        logger.debug("Suppressed json.JSONDecodeError: %s", _exc)
                return results
            except (sqlite3.Error, OSError) as e:
                record_degradation('cognitive_ledger', e)
                logger.error("CognitiveLedger get_coherence_history failed: %s", e)
                return []

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._transition_count

    @property
    def last_id(self) -> str | None:
        return self._last_transition_id

    def get_stats(self) -> dict[str, Any]:
        """Return ledger statistics."""
        stats = {
            "total_transitions": self._transition_count,
            "last_transition_id": self._last_transition_id,
            "db_path": str(self._db_path),
        }
        if self._conn:
            try:
                cur = self._conn.execute(
                    "SELECT ttype, COUNT(*) FROM transitions GROUP BY ttype"
                )
                stats["by_type"] = {row[0]: row[1] for row in cur.fetchall()}
                cur = self._conn.execute("SELECT COUNT(*) FROM snapshots")
                stats["snapshot_count"] = cur.fetchone()[0]
            except (sqlite3.Error, OSError) as _exc:
                record_degradation('cognitive_ledger', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
        return stats

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_state_hash(state: Any) -> str:
    """Quick hash of key state fields for change detection."""
    try:
        blob = json.dumps({
            "phi": getattr(state, "phi", 0),
            "valence": getattr(getattr(state, "affect", None), "valence", 0),
            "arousal": getattr(getattr(state, "affect", None), "arousal", 0),
            "mode": getattr(getattr(state, "cognition", None), "current_mode", ""),
            "version": getattr(state, "version", 0),
        }, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
    except (json.JSONDecodeError, TypeError, ValueError):
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]


# ── Singleton ────────────────────────────────────────────────────────────

_ledger: CognitiveLedger | None = None


def get_cognitive_ledger() -> CognitiveLedger:
    global _ledger
    if _ledger is None:
        _ledger = CognitiveLedger()
    return _ledger


def reset_cognitive_ledger_for_test() -> None:
    """Close and drop the process-global ledger.

    Anything that records a transition — the transition model, the resource
    governor, the intention loop — brings this singleton online lazily, so the
    test that happens to be first to reach one of those paths acquires a sqlite
    handle for the rest of the process and gets blamed for the leak. Closing it
    between tests makes the blame land where the work happened.
    """
    global _ledger
    ledger, _ledger = _ledger, None
    if ledger is not None:
        ledger.close()
