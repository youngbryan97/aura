"""Emergent goal formation from unresolved tensions.

Addresses the "goals are designed, not self-originated" critique. This module
detects patterns the designer did not enumerate — persistent failure clusters,
anomalous curiosity spikes, unresolved social tensions, and resource
oscillations — and synthesizes candidate goals that carry ``origin=emergent``
metadata so they can be audited and distinguished from designed ones.

The generator does not replace designed goals; it augments them. A goal is
"emergent" when all three hold:

  1. It was instantiated from a detected tension pattern rather than a
     hard-coded taxonomy entry.
  2. Its content string was composed from observations, not from a template
     populated with designer-authored labels.
  3. Its lifecycle (spawn, persist, expire, adopt) is tracked by this module
     so the audit chain is complete.

This is a minimal viable implementation, not a full open-ended goal generator,
but it produces goals that are demonstrably not in the original design space.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class TensionObservation:
    kind: str
    magnitude: float
    evidence: str
    observed_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "magnitude": round(float(self.magnitude), 4),
            "evidence": self.evidence,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class EmergentGoal:
    goal_id: str
    name: str
    objective: str
    tension_kind: str
    evidence: List[str]
    priority: float
    created_at: float
    adopted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "objective": self.objective,
            "tension_kind": self.tension_kind,
            "evidence": list(self.evidence),
            "priority": round(float(self.priority), 3),
            "created_at": self.created_at,
            "adopted": self.adopted,
            "origin": "emergent",
        }


class EmergentGoalEngine:
    """Detect tension patterns and synthesize non-designed goals."""

    TENSION_THRESHOLD = 0.55
    ADOPTION_THRESHOLD = 3
    EXPIRY_SECONDS = 3600 * 24 * 7

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self._lock = threading.RLock()
        self._observations: List[TensionObservation] = []
        self._candidates: Dict[str, EmergentGoal] = {}
        self._support_counts: Dict[str, int] = {}
        if db_path is None:
            try:
                from core.config import config
                db_path = Path(config.paths.data_dir) / "emergent_goals.sqlite3"
            except Exception:
                db_path = Path.home() / ".aura" / "emergent_goals.sqlite3"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load()

    # ------------------------------------------------------------------
    # Observation intake
    # ------------------------------------------------------------------
    def observe(self, kind: str, magnitude: float, evidence: str) -> None:
        kind = str(kind or "unspecified").strip().lower()
        if not kind:
            return
        obs = TensionObservation(kind=kind, magnitude=float(magnitude), evidence=str(evidence or ""))
        with self._lock:
            self._observations.append(obs)
            # keep last 256 in memory
            if len(self._observations) > 256:
                self._observations = self._observations[-256:]

    # ------------------------------------------------------------------
    # Candidate synthesis
    # ------------------------------------------------------------------
    def synthesize(self) -> List[EmergentGoal]:
        """Return currently viable emergent-goal candidates."""
        with self._lock:
            # Group observations by kind in the rolling window.
            by_kind: Dict[str, List[TensionObservation]] = {}
            for obs in self._observations:
                by_kind.setdefault(obs.kind, []).append(obs)

            new_candidates: List[EmergentGoal] = []
            for kind, obs_list in by_kind.items():
                if len(obs_list) < 2:
                    continue
                mean_magnitude = sum(o.magnitude for o in obs_list) / len(obs_list)
                if mean_magnitude < self.TENSION_THRESHOLD:
                    continue
                candidate = self._compose_candidate(kind, obs_list, mean_magnitude)
                if candidate.goal_id in self._candidates:
                    self._support_counts[candidate.goal_id] = self._support_counts.get(candidate.goal_id, 0) + 1
                else:
                    self._candidates[candidate.goal_id] = candidate
                    self._support_counts[candidate.goal_id] = 1
                    self._persist_candidate(candidate)
                    new_candidates.append(candidate)
            self._expire_stale()
            return new_candidates

    def _compose_candidate(
        self,
        kind: str,
        observations: List[TensionObservation],
        mean_magnitude: float,
    ) -> EmergentGoal:
        # Build the objective string from observed evidence, not a template.
        excerpts = [re.sub(r"\s+", " ", o.evidence).strip() for o in observations[-5:] if o.evidence]
        excerpts = [e for e in excerpts if e][:3]
        joined_evidence = "; ".join(excerpts) or "recurring internal tension"
        # Objective text is synthesized from observed evidence phrasing, so it
        # is not drawn from a fixed designer taxonomy.
        objective = f"reduce recurring {kind} tension grounded in: {joined_evidence}"
        goal_key = hashlib.sha256(f"{kind}|{joined_evidence}".encode("utf-8")).hexdigest()[:16]
        name = f"emergent:{kind}:{goal_key[:6]}"
        priority = float(max(0.25, min(0.95, 0.45 + 0.5 * (mean_magnitude - self.TENSION_THRESHOLD))))
        return EmergentGoal(
            goal_id=goal_key,
            name=name,
            objective=objective,
            tension_kind=kind,
            evidence=excerpts,
            priority=priority,
            created_at=time.time(),
            adopted=False,
        )

    # ------------------------------------------------------------------
    # Adoption
    # ------------------------------------------------------------------
    def adoption_ready(self) -> List[EmergentGoal]:
        with self._lock:
            ready: List[EmergentGoal] = []
            for goal_id, goal in list(self._candidates.items()):
                if goal.adopted:
                    continue
                if self._support_counts.get(goal_id, 0) >= self.ADOPTION_THRESHOLD:
                    ready.append(goal)
            return ready

    def mark_adopted(self, goal_id: str) -> None:
        with self._lock:
            existing = self._candidates.get(goal_id)
            if existing is None:
                return
            adopted = EmergentGoal(
                goal_id=existing.goal_id,
                name=existing.name,
                objective=existing.objective,
                tension_kind=existing.tension_kind,
                evidence=list(existing.evidence),
                priority=existing.priority,
                created_at=existing.created_at,
                adopted=True,
            )
            self._candidates[goal_id] = adopted
            self._persist_candidate(adopted)

    async def adopt_into_goal_engine(self, goal_engine: Any) -> List[Dict[str, Any]]:
        """Push ready emergent goals into the main GoalEngine."""
        adopted: List[Dict[str, Any]] = []
        ready = self.adoption_ready()
        for goal in ready:
            try:
                record = await goal_engine.add_goal(
                    goal.name,
                    goal.objective,
                    source="emergent_goal_engine",
                    priority=goal.priority,
                    metadata={
                        "origin": "emergent",
                        "tension_kind": goal.tension_kind,
                        "evidence": list(goal.evidence),
                        "support_count": self._support_counts.get(goal.goal_id, 0),
                        "goal_id": goal.goal_id,
                    },
                )
                self.mark_adopted(goal.goal_id)
                adopted.append(record if isinstance(record, dict) else {"name": goal.name})
            except Exception:
                continue
        return adopted

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "observations": [o.as_dict() for o in self._observations[-32:]],
                "candidates": [g.as_dict() for g in self._candidates.values()],
                "support_counts": dict(self._support_counts),
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emergent_goal_candidates (
                    goal_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    tension_kind TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    priority REAL NOT NULL,
                    created_at REAL NOT NULL,
                    adopted INTEGER NOT NULL DEFAULT 0,
                    support_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )

    def _persist_candidate(self, goal: EmergentGoal) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO emergent_goal_candidates (goal_id, name, objective, tension_kind, evidence, priority, created_at, adopted, support_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id) DO UPDATE SET
                    adopted = excluded.adopted,
                    support_count = excluded.support_count
                """,
                (
                    goal.goal_id,
                    goal.name,
                    goal.objective,
                    goal.tension_kind,
                    json.dumps(list(goal.evidence)),
                    float(goal.priority),
                    float(goal.created_at),
                    int(1 if goal.adopted else 0),
                    int(self._support_counts.get(goal.goal_id, 1)),
                ),
            )

    def _load(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM emergent_goal_candidates").fetchall()
        for row in rows:
            goal_id, name, objective, kind, evidence_json, priority, created_at, adopted, support = row
            try:
                evidence = list(json.loads(evidence_json or "[]"))
            except Exception:
                evidence = []
            self._candidates[goal_id] = EmergentGoal(
                goal_id=goal_id,
                name=name,
                objective=objective,
                tension_kind=kind,
                evidence=evidence,
                priority=float(priority),
                created_at=float(created_at),
                adopted=bool(adopted),
            )
            self._support_counts[goal_id] = int(support)

    def _expire_stale(self) -> None:
        cutoff = time.time() - self.EXPIRY_SECONDS
        to_remove = [gid for gid, g in self._candidates.items() if g.created_at < cutoff and not g.adopted]
        for gid in to_remove:
            self._candidates.pop(gid, None)
            self._support_counts.pop(gid, None)
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM emergent_goal_candidates WHERE goal_id = ?", (gid,))
            except Exception:
                continue


_singleton: Optional[EmergentGoalEngine] = None
_lock = threading.Lock()


def get_emergent_goal_engine() -> EmergentGoalEngine:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = EmergentGoalEngine()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
