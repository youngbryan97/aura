"""Audit Trail — Immutable log of all consequential Aura actions

Every action Aura takes that modifies state (file writes, skill executions,
shell commands, memory writes, goal changes) is recorded here with:
  - What happened
  - Who/what requested it
  - The outcome
  - A reversibility flag

This is both for safety (post-incident review) and transparency
(showing the user exactly what Aura did and when).
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Security.AuditTrail")


class AuditTrail:
    """Immutable, append-only audit log backed by SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(config.paths.home_dir / "data/audit_trail.db")
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("✓ Audit Trail initialized: %s", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self):
        with self._get_conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                target TEXT,
                params TEXT,
                outcome TEXT NOT NULL,
                outcome_detail TEXT,
                reversible BOOLEAN DEFAULT 0,
                reversed BOOLEAN DEFAULT 0,
                reverse_id TEXT,
                session_id TEXT
            )""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                           ON audit_log(timestamp DESC)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_audit_category 
                           ON audit_log(category)""")
            conn.commit()

    def log_action(
        self,
        category: str,
        action: str,
        actor: str = "aura",
        target: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        outcome_detail: str = "",
        reversible: bool = False,
        session_id: Optional[str] = None,
    ) -> str:
        """Record an action in the audit trail.
        
        Args:
            category: "skill", "file", "memory", "goal", "security", "config"
            action: What was done (e.g., "file_write", "skill_execute", "memory_store")
            actor: Who/what triggered it ("aura", "user", "autonomy", "react_loop")
            target: What was affected (file path, memory ID, goal ID)
            params: Action parameters (sanitized — no secrets!)
            outcome: "success", "failure", "blocked", "partial"
            outcome_detail: Human-readable explanation
            reversible: Whether this action can be undone
            session_id: Current conversation session
            
        Returns:
            Audit entry ID
        """
        entry_id = str(uuid.uuid4())[:12]

        # Sanitize params — strip anything that looks like credentials
        safe_params = None
        if params:
            safe_params = {}
            for k, v in params.items():
                k_lower = k.lower()
                if any(s in k_lower for s in ("password", "secret", "token", "key", "credential")):
                    safe_params[k] = "[REDACTED]"
                elif isinstance(v, str) and len(v) > 500:
                    safe_params[k] = str(v)[:500] + "...[truncated]"
                else:
                    safe_params[k] = v

        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        """INSERT INTO audit_log 
                           (id, timestamp, category, action, actor, target, params, 
                            outcome, outcome_detail, reversible, session_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry_id,
                            time.time(),
                            category,
                            action,
                            actor,
                            target,
                            json.dumps(safe_params) if safe_params else None,
                            outcome,
                            outcome_detail,
                            reversible,
                            session_id,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error("Audit trail write failed: %s", e)
                return ""

        logger.debug(
            "📋 Audit: [%s] %s → %s (%s) by %s",
            category, action, target or "—", outcome, actor
        )
        return entry_id

    def get_recent(self, limit: int = 50, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent audit entries."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            if category:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_by_target(self, target: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all audit entries for a specific target (e.g., a file path)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE target = ? ORDER BY timestamp DESC LIMIT ?",
                (target, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_reversed(self, entry_id: str, reverse_entry_id: str):
        """Mark an action as reversed/rolled back."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE audit_log SET reversed = 1, reverse_id = ? WHERE id = ?",
                    (reverse_entry_id, entry_id),
                )
                conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Audit trail statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            by_category = {}
            for row in conn.execute(
                "SELECT category, COUNT(*) as cnt FROM audit_log GROUP BY category"
            ).fetchall():
                by_category[row[0]] = row[1]
            by_outcome = {}
            for row in conn.execute(
                "SELECT outcome, COUNT(*) as cnt FROM audit_log GROUP BY outcome"
            ).fetchall():
                by_outcome[row[0]] = row[1]
            return {
                "total_entries": total,
                "by_category": by_category,
                "by_outcome": by_outcome,
            }

    def count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    global _instance
    if _instance is None:
        _instance = AuditTrail()
    return _instance
