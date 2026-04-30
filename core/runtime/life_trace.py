"""LifeTrace ledger — auditable life history.

Every significant autonomous event records:

  - drive state before and after
  - memory context at the moment of decision
  - counterfactuals considered
  - will decision + receipt id
  - action taken + outcome
  - memory update produced
  - any future-policy change

The ledger is hash-chained so the audit trail is tamper-evident, and it
can produce a daily summary in the shape the reviewer asked for
(self-generated vs user-requested counts, deferred-by-will count,
blocked-by-resource count, policy-changed count).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EVENT_TYPES = (
    "initiative_proposed",
    "initiative_scored",
    "initiative_selected",
    "initiative_deferred",
    "initiative_blocked",
    "action_executed",
    "outcome_assessed",
    "policy_changed",
    "memory_written",
    "self_generated",
    "user_requested",
    "repair_applied",
)


@dataclass(frozen=True)
class LifeTraceEvent:
    event_id: str
    event_type: str
    origin: str
    user_requested: bool
    drive_state_before: Dict[str, Any]
    drive_state_after: Dict[str, Any]
    memory_context: List[Any]
    counterfactuals_considered: List[Any]
    will_decision: Dict[str, Any]
    action_taken: Dict[str, Any]
    result: Dict[str, Any]
    memory_update: Dict[str, Any]
    future_policy_change: Dict[str, Any]
    timestamp: float
    prev_hash: str
    hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "origin": self.origin,
            "user_requested": self.user_requested,
            "drive_state_before": self.drive_state_before,
            "drive_state_after": self.drive_state_after,
            "memory_context": list(self.memory_context),
            "counterfactuals_considered": list(self.counterfactuals_considered),
            "will_decision": self.will_decision,
            "action_taken": self.action_taken,
            "result": self.result,
            "memory_update": self.memory_update,
            "future_policy_change": self.future_policy_change,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


class LifeTraceLedger:
    """Hash-chained append-only ledger with daily summaries."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self._lock = threading.RLock()
        if db_path is None:
            try:
                from core.config import config
                db_path = Path(config.paths.data_dir) / "life_trace.sqlite3"
            except Exception:
                db_path = Path.home() / ".aura" / "life_trace.sqlite3"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS life_trace (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    user_requested INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_life_trace_ts ON life_trace(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_life_trace_type ON life_trace(event_type)")

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------
    def record(
        self,
        event_type: str,
        *,
        origin: str,
        user_requested: bool = False,
        drive_state_before: Optional[Dict[str, Any]] = None,
        drive_state_after: Optional[Dict[str, Any]] = None,
        memory_context: Optional[Iterable[Any]] = None,
        counterfactuals_considered: Optional[Iterable[Any]] = None,
        will_decision: Optional[Dict[str, Any]] = None,
        action_taken: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        memory_update: Optional[Dict[str, Any]] = None,
        future_policy_change: Optional[Dict[str, Any]] = None,
    ) -> LifeTraceEvent:
        if event_type not in EVENT_TYPES:
            # Allow new types but keep the set visible to reviewers
            pass  # no-op: intentional
        ts = time.time()
        payload = {
            "event_type": event_type,
            "origin": origin,
            "user_requested": bool(user_requested),
            "drive_state_before": drive_state_before or {},
            "drive_state_after": drive_state_after or {},
            "memory_context": list(memory_context or []),
            "counterfactuals_considered": list(counterfactuals_considered or []),
            "will_decision": will_decision or {},
            "action_taken": action_taken or {},
            "result": result or {},
            "memory_update": memory_update or {},
            "future_policy_change": future_policy_change or {},
            "timestamp": ts,
        }
        event_id = hashlib.sha256(
            f"{ts}|{event_type}|{origin}|{json.dumps(payload['action_taken'], sort_keys=True, default=str)}".encode("utf-8")
        ).hexdigest()[:16]
        with self._lock:
            prev_hash = self._tail_hash()
            payload["prev_hash"] = prev_hash
            payload["event_id"] = event_id
            hash_ = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO life_trace
                        (event_id, event_type, origin, user_requested, payload, timestamp, prev_hash, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        event_type,
                        origin,
                        int(bool(user_requested)),
                        json.dumps(payload, sort_keys=True, default=str),
                        ts,
                        prev_hash,
                        hash_,
                    ),
                )
        return LifeTraceEvent(
            event_id=event_id,
            event_type=event_type,
            origin=origin,
            user_requested=bool(user_requested),
            drive_state_before=payload["drive_state_before"],
            drive_state_after=payload["drive_state_after"],
            memory_context=payload["memory_context"],
            counterfactuals_considered=payload["counterfactuals_considered"],
            will_decision=payload["will_decision"],
            action_taken=payload["action_taken"],
            result=payload["result"],
            memory_update=payload["memory_update"],
            future_policy_change=payload["future_policy_change"],
            timestamp=ts,
            prev_hash=prev_hash,
            hash=hash_,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM life_trace ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def since(self, since_ts: float) -> List[Dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM life_trace WHERE timestamp >= ? ORDER BY timestamp ASC",
                (float(since_ts),),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        try:
            payload = json.loads(row["payload"])
        except Exception:
            payload = {}
        payload["event_id"] = row["event_id"]
        payload["prev_hash"] = row["prev_hash"]
        payload["hash"] = row["hash"]
        return payload

    def _tail_hash(self) -> str:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT hash FROM life_trace ORDER BY timestamp DESC, event_id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else "GENESIS"

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def verify_chain(self) -> bool:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM life_trace ORDER BY timestamp ASC, event_id ASC"
            ).fetchall()
        prev = "GENESIS"
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except Exception:
                return False
            if row["prev_hash"] != prev:
                return False
            expected = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            if expected != row["hash"]:
                return False
            prev = row["hash"]
        return True

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------
    def daily_summary(self, *, window_hours: float = 24.0) -> Dict[str, Any]:
        cutoff = time.time() - float(window_hours) * 3600.0
        events = self.since(cutoff)
        total = len(events)
        user_requested = sum(1 for e in events if e.get("user_requested"))
        self_generated = total - user_requested
        by_type: Dict[str, int] = defaultdict(int)
        for e in events:
            by_type[str(e.get("event_type") or "")] += 1
        deferred = int(by_type.get("initiative_deferred", 0))
        blocked = int(by_type.get("initiative_blocked", 0))
        executed = int(by_type.get("action_executed", 0))
        policy_changed = int(by_type.get("policy_changed", 0))
        repairs = int(by_type.get("repair_applied", 0))
        return {
            "window_hours": float(window_hours),
            "total_events": total,
            "user_requested": user_requested,
            "self_generated": self_generated,
            "actions_executed": executed,
            "deferred_by_will": deferred,
            "blocked_by_resource": blocked,
            "policy_changes": policy_changed,
            "repairs_applied": repairs,
            "event_counts": dict(by_type),
            "chain_intact": self.verify_chain(),
        }


_singleton: Optional[LifeTraceLedger] = None
_lock = threading.Lock()


def get_life_trace() -> LifeTraceLedger:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = LifeTraceLedger()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
