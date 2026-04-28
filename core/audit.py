"""
Immutable autonomous action audit trail.
Append-only SQLite table — no UPDATE or DELETE ever runs on audit records.
"""
from core.runtime.errors import record_degradation
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Aura.Audit")

_DB_PATH = config.paths.data_dir / "audit.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           TEXT PRIMARY KEY,
    action_type  TEXT NOT NULL,     -- 'skill_call' | 'autonomous_goal' | 'self_repair' | 'file_write' | etc.
    description  TEXT NOT NULL,
    actor        TEXT NOT NULL,     -- 'user' | 'autonomous' | 'terminal_monitor' | 'hephaestus'
    skill_name   TEXT,
    params       TEXT,              -- JSON, redacted
    result_ok    INTEGER,           -- 1=success, 0=failure, NULL=unknown
    cid          TEXT,              -- correlation ID
    session_id   TEXT,
    created_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(action_type);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor);
"""


class AuditLog:
    """
    Append-only audit log. Once written, records are never modified.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or _DB_PATH)
        self._con: Optional[sqlite3.Connection] = None
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
            self._con.row_factory = sqlite3.Row
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
        return self._con

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    def _init(self):
        con = self._connect()
        con.executescript(_SCHEMA)
        con.commit()

    def record(
        self,
        action_type: str,
        description: str,
        actor: str = "autonomous",
        skill_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        result_ok: Optional[bool] = None,
        cid: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        entry_id = str(uuid.uuid4())[:12]
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    entry_id, action_type, description, actor,
                    skill_name,
                    json.dumps(params, default=str) if params else None,
                    1 if result_ok is True else (0 if result_ok is False else None),
                    cid, session_id,
                    time.time(),
                ),
            )
            con.commit()
        except Exception as e:
            record_degradation('audit', e)
            logger.error("Failed to record audit entry: %s", e)
        return entry_id

    def get_recent(
        self,
        limit: int = 100,
        action_type: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM audit_log"
        args = []
        conditions = []
        if action_type:
            conditions.append("action_type = ?")
            args.append(action_type)
        if actor:
            conditions.append("actor = ?")
            args.append(actor)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._connect() as con:
            rows = con.execute(query, args).fetchall()
        return [dict(r) for r in rows]

    def get_autonomous_summary(self, since_hours: float = 24.0) -> Dict[str, Any]:
        """Summary of autonomous actions in the last N hours."""
        cutoff = time.time() - (since_hours * 3600)
        with self._connect() as con:
            total = con.execute(
                "SELECT COUNT(*) FROM audit_log WHERE actor != 'user' AND created_at > ?",
                (cutoff,),
            ).fetchone()[0]
            by_type = con.execute(
                "SELECT action_type, COUNT(*) as c FROM audit_log "
                "WHERE actor != 'user' AND created_at > ? GROUP BY action_type",
                (cutoff,),
            ).fetchall()
            failures = con.execute(
                "SELECT COUNT(*) FROM audit_log "
                "WHERE actor != 'user' AND result_ok = 0 AND created_at > ?",
                (cutoff,),
            ).fetchone()[0]
        return {
            "period_hours": since_hours,
            "total_autonomous_actions": total,
            "failures": failures,
            "by_type": {r["action_type"]: r["c"] for r in by_type},
        }

    def get_skill_performance_stats(self, since_hours: float = 24.0) -> List[Dict[str, Any]]:
        """Calculates performance statistics for each skill in the last N hours."""
        cutoff = time.time() - (since_hours * 3600)
        query = """
            SELECT 
                skill_name,
                COUNT(*) as calls,
                SUM(result_ok) as successes,
                AVG(CASE WHEN result_ok IS NOT NULL THEN 1.0 ELSE 0.0 END) as reporting_rate
            FROM audit_log
            WHERE action_type = 'skill_call' AND created_at > ? AND skill_name IS NOT NULL
            GROUP BY skill_name
        """
        with self._connect() as con:
            rows = con.execute(query, (cutoff,)).fetchall()
        
        stats = []
        for r in rows:
            calls = r["calls"]
            successes = r["successes"] or 0
            success_rate = (successes / calls) if calls > 0 else 1.0
            stats.append({
                "skill_name": r["skill_name"],
                "calls": calls,
                "successes": successes,
                "success_rate": success_rate,
                "reporting_rate": r["reporting_rate"]
            })
        return sorted(stats, key=lambda x: x["success_rate"])


_audit: Optional[AuditLog] = None


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        _audit = AuditLog()
    return _audit
