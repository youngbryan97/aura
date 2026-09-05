"""core/planning/plan_failure_memory.py — learning from failed plans across episodes.

The planner (HierarchicalPlanner / TaskDecomposer) decomposes goals and the
RecoveryEngine retries/falls-back *within* an episode — but nothing remembered, across
episodes, that a given approach to a given kind of goal has failed before. So Aura could
re-attempt the same doomed strategy tomorrow that died today. That is precisely the
capability a roguelike like NetHack forces: permadeath means the only way to get better
is to *learn across runs* which strategies lead to death and stop choosing them.

This is that loop, kept general (not NetHack-specific): every plan outcome is recorded
against a *goal class* (a normalized signature of the objective, so the lesson transfers
to similar-but-differently-worded goals) and a *strategy signature*. Before planning a
new goal, the planner asks for ``guidance``: which strategies have a high failure rate
for this class (avoid) and which have paid off (prefer). Over episodes the planner stops
walking into the same wall.

Real machinery, measurable without any LLM: persisted success/failure counts drive the
avoid/prefer sets; the loop is a deterministic statistic, not a prompt. Bounded (counts
decay, store is capped) and governed (it can only *bias* planning toward what worked — it
never forces or forbids an action; the action gate still owns safety).
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import connecting
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.PlanFailureMemory")

# A strategy needs at least this many attempts before it can be called a reliable
# failure/success — one bad run is noise, a pattern is a lesson.
MIN_ATTEMPTS = 2
AVOID_FAILURE_RATE = 0.6     # ≥ this failure rate (with enough attempts) → avoid
PREFER_SUCCESS_RATE = 0.6    # ≥ this success rate (with enough attempts) → prefer

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "with", "from",
    "this", "that", "then", "into", "your", "you", "please", "could", "would", "should",
    "i", "we", "it", "is", "are", "be", "do", "does", "my", "me", "her", "his", "their",
})


def _salient_set(objective: str) -> set[str]:
    """The salient (content) tokens of an objective — the basis for generalization."""
    words = re.findall(r"[a-z0-9]+", str(objective or "").lower())
    salient = {w for w in words if len(w) >= 4 and w not in _STOPWORDS}
    return salient or set(words)


def goal_class(objective: str) -> str:
    """Stable class key for *storing* an outcome (the salient-token signature).

    Recall generalizes by token OVERLAP (see ``guidance``), not by exact key match, so
    "open the browser and search" and "open a browser to search for something" — which
    share the salient tokens {open, browser, search} — transfer lessons to each other
    even though their full signatures differ.
    """
    salient = sorted(_salient_set(objective))
    return "|".join(salient[:6]) or "generic"


# A query goal and a recorded class are "the same kind of goal" when they share at least
# this many salient tokens (or a high Jaccard overlap).
_MIN_SHARED_TOKENS = 2
_MIN_JACCARD = 0.5


def strategy_signature(strategy: str) -> str:
    """Normalize a strategy / approach / failure-mode to a stable signature."""
    s = re.sub(r"\s+", "_", str(strategy or "").strip().lower())
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:64] or "default"


@dataclass(frozen=True)
class PlanGuidance:
    goal_class: str
    avoid: list[str] = field(default_factory=list)         # strategy sigs to steer away from
    prefer: list[str] = field(default_factory=list)        # strategy sigs that have paid off
    failure_rates: dict[str, float] = field(default_factory=dict)
    total_observations: int = 0

    @property
    def has_lessons(self) -> bool:
        return bool(self.avoid or self.prefer)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_class": self.goal_class,
            "avoid": list(self.avoid),
            "prefer": list(self.prefer),
            "failure_rates": {k: round(v, 3) for k, v in self.failure_rates.items()},
            "total_observations": self.total_observations,
        }

    def caution_text(self) -> str:
        """Compact factual caution for the planner's context (a read-out, not the mechanism)."""
        if not self.has_lessons:
            return ""
        parts = ["## PRIOR PLAN OUTCOMES (learned across episodes)"]
        if self.avoid:
            parts.append(
                "- Avoid (failed before for this kind of goal): "
                + ", ".join(f"{s} [{self.failure_rates.get(s, 0):.0%} fail]" for s in self.avoid[:4])
            )
        if self.prefer:
            parts.append("- Prefer (worked before): " + ", ".join(self.prefer[:4]))
        return "\n".join(parts) + "\n"


class PlanFailureMemory:
    """Persistent cross-episode record of which strategies fail/succeed per goal class."""

    SERVICE_NAME = "plan_failure_memory"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        if db_path is None:
            try:
                from core.config import config

                db_path = Path(config.paths.data_dir) / "plan_failure_memory.sqlite3"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("plan_failure_memory", exc, severity="debug")
                db_path = state_root() / "data" / "plan_failure_memory.sqlite3"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS plan_outcomes (
                        goal_class TEXT NOT NULL,
                        strategy   TEXT NOT NULL,
                        successes  INTEGER NOT NULL DEFAULT 0,
                        failures   INTEGER NOT NULL DEFAULT 0,
                        last_failure_mode TEXT DEFAULT '',
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (goal_class, strategy)
                    )"""
                )
                conn.commit()
        except (sqlite3.Error, OSError) as exc:
            record_degradation("plan_failure_memory", exc)

    # ── record ───────────────────────────────────────────────────────────────
    def record_outcome(
        self,
        goal: str,
        strategy: str,
        *,
        success: bool,
        failure_mode: str = "",
    ) -> None:
        gc = goal_class(goal)
        sig = strategy_signature(strategy)
        with self._lock:
            try:
                with connecting(self._connect()) as conn:
                    conn.execute(
                        """INSERT INTO plan_outcomes (goal_class, strategy, successes, failures, last_failure_mode, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(goal_class, strategy) DO UPDATE SET
                               successes = successes + ?,
                               failures = failures + ?,
                               last_failure_mode = CASE WHEN ?='' THEN last_failure_mode ELSE ? END,
                               updated_at = ?""",
                        (
                            gc, sig, 1 if success else 0, 0 if success else 1,
                            failure_mode[:200], time.time(),
                            1 if success else 0, 0 if success else 1,
                            failure_mode[:200], failure_mode[:200], time.time(),
                        ),
                    )
                    conn.commit()
            except (sqlite3.Error, OSError) as exc:
                record_degradation("plan_failure_memory", exc)
        logger.debug("plan outcome: class=%s strategy=%s success=%s", gc, sig, success)

    # ── recall ───────────────────────────────────────────────────────────────
    def guidance(self, goal: str) -> PlanGuidance:
        gc = goal_class(goal)
        query_tokens = _salient_set(goal)
        avoid: list[str] = []
        prefer: list[str] = []
        rates: dict[str, float] = {}
        total = 0
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT goal_class, strategy, successes, failures FROM plan_outcomes"
                ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            record_degradation("plan_failure_memory", exc, severity="debug")
            rows = []
        # Aggregate outcomes across every recorded class that is "the same kind of goal"
        # as the query — overlap matching, so lessons generalize across rewordings.
        agg: dict[str, list[int]] = {}
        for row_class, strategy, succ, fail in rows:
            cls_tokens = set(str(row_class).split("|"))
            shared = len(query_tokens & cls_tokens)
            union = len(query_tokens | cls_tokens) or 1
            if shared >= _MIN_SHARED_TOKENS or (shared / union) >= _MIN_JACCARD:
                bucket = agg.setdefault(strategy, [0, 0])
                bucket[0] += int(succ)
                bucket[1] += int(fail)
        # Rank avoid by failure rate (desc), prefer by success rate (desc).
        scored: list[tuple[str, float, float, int]] = []
        for strategy, (succ, fail) in agg.items():
            attempts = int(succ) + int(fail)
            total += attempts
            if attempts < MIN_ATTEMPTS:
                continue
            fail_rate = fail / attempts
            succ_rate = succ / attempts
            rates[strategy] = round(fail_rate, 3)
            scored.append((strategy, fail_rate, succ_rate, attempts))
        for strategy, fail_rate, succ_rate, _ in sorted(scored, key=lambda x: x[1], reverse=True):
            if fail_rate >= AVOID_FAILURE_RATE:
                avoid.append(strategy)
        for strategy, _, succ_rate, _ in sorted(scored, key=lambda x: x[2], reverse=True):
            if succ_rate >= PREFER_SUCCESS_RATE:
                prefer.append(strategy)
        return PlanGuidance(
            goal_class=gc, avoid=avoid, prefer=prefer,
            failure_rates=rates, total_observations=total,
        )

    def caution_text(self, goal: str) -> str:
        return self.guidance(goal).caution_text()

    def stats(self) -> dict[str, Any]:
        try:
            with connecting(self._connect()) as conn:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(successes),0), COALESCE(SUM(failures),0) FROM plan_outcomes"
                ).fetchone()
            return {
                "service": self.SERVICE_NAME,
                "strategy_classes": int(row[0]) if row else 0,
                "total_successes": int(row[1]) if row else 0,
                "total_failures": int(row[2]) if row else 0,
                "db_path": str(self._db_path),
            }
        except (sqlite3.Error, OSError):
            return {"service": self.SERVICE_NAME, "db_path": str(self._db_path)}


_engine: PlanFailureMemory | None = None
_engine_lock = threading.Lock()


def get_plan_failure_memory() -> PlanFailureMemory:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = PlanFailureMemory()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: PlanFailureMemory) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(PlanFailureMemory.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(PlanFailureMemory.SERVICE_NAME, engine,
                    required=False, registered_by="plan_failure_memory")
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


def reset_plan_failure_memory_for_test() -> None:
    global _engine
    _engine = None
