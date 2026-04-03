"""Outcome-Based Learning — Track what works and improve over time

Aura learns from the outcomes of her actions:
  - Did the user accept/reject the response?
  - Did the tool call succeed or fail?
  - Was the goal completed or abandoned?
  - How long did the user engage afterward?

These outcomes are fed back into:
  - Skill proficiency ratings (in the knowledge graph)
  - Belief confidence adjustments
  - Strategy selection preferences
  - Response style calibration
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Learning.Outcomes")


class OutcomeLearner:
    """Tracks outcomes of Aura's actions and learns from them."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(config.paths.home_dir / "data/outcomes.db")
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        # In-memory short-term stats
        self._session_successes = 0
        self._session_failures = 0
        self._session_start = time.time()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._get_conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                action TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                confidence_before REAL,
                confidence_after REAL,
                user_feedback TEXT,
                duration_ms REAL,
                context TEXT,
                lesson TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS strategy_scores (
                strategy TEXT PRIMARY KEY,
                total_uses INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                avg_duration_ms REAL DEFAULT 0,
                last_used REAL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                window_minutes INTEGER DEFAULT 60
            )""")
            conn.commit()
        logger.info("✓ Outcome Learner initialized")

    def record_outcome(
        self,
        category: str,
        action: str,
        success: bool,
        confidence_before: Optional[float] = None,
        user_feedback: Optional[str] = None,
        duration_ms: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        lesson: Optional[str] = None,
    ):
        """Record the outcome of an action for learning.
        
        Args:
            category: "skill", "reasoning", "memory", "conversation", "tool"
            action: What was attempted
            success: Whether it achieved the desired outcome
            confidence_before: Aura's confidence before acting
            user_feedback: Explicit user feedback ("good", "bad", "retry", etc.)
            duration_ms: How long the action took
            context: Additional context for the outcome
            lesson: Extracted lesson from the outcome
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        """INSERT INTO outcomes 
                           (timestamp, category, action, success, confidence_before,
                            confidence_after, user_feedback, duration_ms, context, lesson)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            time.time(),
                            category,
                            action,
                            success,
                            confidence_before,
                            None,  # confidence_after updated later
                            user_feedback,
                            duration_ms,
                            json.dumps(context) if context else None,
                            lesson,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.debug("Outcome recording failed: %s", e)

        if success:
            self._session_successes += 1
        else:
            self._session_failures += 1

        logger.debug(
            "📊 Outcome: %s/%s → %s (%.0fms)",
            category, action, "✓" if success else "✗", duration_ms or 0
        )

    def record_strategy_usage(
        self, strategy: str, success: bool, duration_ms: float
    ):
        """Track which reasoning strategies work best."""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    row = conn.execute(
                        "SELECT total_uses, successes, avg_duration_ms FROM strategy_scores WHERE strategy = ?",
                        (strategy,),
                    ).fetchone()

                    if row:
                        total, wins, avg_dur = row
                        new_total = total + 1
                        new_wins = wins + (1 if success else 0)
                        # Running average
                        new_avg = (avg_dur * total + duration_ms) / new_total
                        conn.execute(
                            """UPDATE strategy_scores 
                               SET total_uses = ?, successes = ?, avg_duration_ms = ?, last_used = ?
                               WHERE strategy = ?""",
                            (new_total, new_wins, new_avg, time.time(), strategy),
                        )
                    else:
                        conn.execute(
                            """INSERT INTO strategy_scores 
                               (strategy, total_uses, successes, avg_duration_ms, last_used)
                               VALUES (?, 1, ?, ?, ?)""",
                            (strategy, 1 if success else 0, duration_ms, time.time()),
                        )
                    conn.commit()
            except Exception as e:
                logger.debug("Strategy score update failed: %s", e)

    def record_metric(self, name: str, value: float, window_minutes: int = 60):
        """Record a performance metric data point."""
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        """INSERT INTO performance_metrics (timestamp, metric_name, metric_value, window_minutes)
                           VALUES (?, ?, ?, ?)""",
                        (time.time(), name, value, window_minutes),
                    )
                    # Prune old metrics (keep last 7 days)
                    cutoff = time.time() - (7 * 86400)
                    conn.execute(
                        "DELETE FROM performance_metrics WHERE timestamp < ?",
                        (cutoff,),
                    )
                    conn.commit()
            except Exception as e:
                logger.debug("Metric recording failed: %s", e)

    def get_success_rate(self, category: Optional[str] = None, hours: int = 24) -> float:
        """Get success rate over a time window."""
        cutoff = time.time() - (hours * 3600)
        with self._get_conn() as conn:
            if category:
                row = conn.execute(
                    """SELECT COUNT(*) as total, SUM(success) as wins 
                       FROM outcomes WHERE category = ? AND timestamp > ?""",
                    (category, cutoff),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT COUNT(*) as total, SUM(success) as wins 
                       FROM outcomes WHERE timestamp > ?""",
                    (cutoff,),
                ).fetchone()
            total, wins = row
            if not total:
                return 1.0  # No data — assume good
            return (wins or 0) / total

    def get_best_strategy(self) -> Optional[str]:
        """Get the highest-performing reasoning strategy."""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT strategy, 
                          CAST(successes AS REAL) / MAX(1, total_uses) as win_rate
                   FROM strategy_scores 
                   WHERE total_uses >= 3
                   ORDER BY win_rate DESC, avg_duration_ms ASC
                   LIMIT 1"""
            ).fetchone()
            return row[0] if row else None

    def get_session_stats(self) -> Dict[str, Any]:
        """Current session performance."""
        total = self._session_successes + self._session_failures
        uptime = time.time() - self._session_start
        return {
            "session_successes": self._session_successes,
            "session_failures": self._session_failures,
            "session_success_rate": (
                f"{self._session_successes / max(1, total):.0%}"
            ),
            "session_uptime_minutes": float(f"{uptime / 60.0:.1f}"),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Overall learning statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            wins = conn.execute("SELECT SUM(success) FROM outcomes").fetchone()[0] or 0
            strategies = conn.execute(
                "SELECT strategy, total_uses, successes FROM strategy_scores ORDER BY total_uses DESC"
            ).fetchall()

        return {
            "total_outcomes": total,
            "overall_success_rate": f"{wins / max(1, total):.0%}",
            "24h_success_rate": f"{self.get_success_rate(hours=24):.0%}",
            "strategy_rankings": [
                {"strategy": s[0], "uses": s[1], "wins": s[2]} for s in strategies
            ],
            **self.get_session_stats(),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[OutcomeLearner] = None


def get_outcome_learner() -> OutcomeLearner:
    global _instance
    if _instance is None:
        _instance = OutcomeLearner()
    return _instance
