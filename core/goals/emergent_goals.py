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
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Aura.EmergentGoals")


@dataclass(frozen=True)
class TensionObservation:
    kind: str
    magnitude: float
    evidence: str
    observed_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
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
    evidence: list[str]
    priority: float
    created_at: float
    adopted: bool = False

    def as_dict(self) -> dict[str, Any]:
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

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._observations: list[TensionObservation] = []
        self._candidates: dict[str, EmergentGoal] = {}
        self._support_counts: dict[str, int] = {}
        if db_path is None:
            try:
                from core.config import config
                db_path = Path(config.paths.data_dir) / "emergent_goals.sqlite3"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("emergent_goals", exc)
                logger.debug("EmergentGoalEngine config path lookup failed: %s", exc)
                db_path = state_root() / "emergent_goals.sqlite3"
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
    def synthesize(self) -> list[EmergentGoal]:
        """Return currently viable emergent-goal candidates."""
        with self._lock:
            # Group observations by kind in the rolling window.
            by_kind: dict[str, list[TensionObservation]] = {}
            for obs in self._observations:
                by_kind.setdefault(obs.kind, []).append(obs)

            new_candidates: list[EmergentGoal] = []
            # Within a kind, group by what the trouble is ABOUT.
            #
            # A tension's identity was its exact evidence text, which grows
            # with every observation, so the same trouble was never the same
            # goal twice. Making it the kind alone fixed that and overshot:
            # a category as broad as "a regretted action" then collapses every
            # unrelated recurring problem into one goal, and a goal that is
            # about everything is about nothing.
            #
            # What sits between them is what the evidence is about — the words
            # its accounts have in common. Two reports of the same trouble
            # share them; two different troubles under one heading do not.
            for kind, obs_list in by_kind.items():
                for founded_on, group in self._grouped_by_what_it_is_about(obs_list):
                    if len(group) < 2:
                        continue
                    mean_magnitude = sum(o.magnitude for o in group) / len(group)
                    if mean_magnitude < self.TENSION_THRESHOLD:
                        continue
                    candidate = self._compose_candidate(
                        kind, group, mean_magnitude, founded_on=founded_on
                    )
                    if candidate.goal_id in self._candidates:
                        self._support_counts[candidate.goal_id] = self._support_counts.get(candidate.goal_id, 0) + 1
                        # Evidence sharpens the objective it already had.
                        self._candidates[candidate.goal_id] = candidate
                        # Written down as it grows, not only when it is born.
                        #
                        # Candidates were persistent and their support was not: the
                        # count reached disk on creation and again on adoption, and
                        # every unit of support in between lived in memory. So a
                        # tension that recurred four times across a day of uptime
                        # came back after a reboot as a tension that had been
                        # noticed once, and a goal three-quarters of the way to
                        # being adopted started again. For a mind whose whole point
                        # is that experience accumulates, that is the wrong thing
                        # to lose.
                        self._persist_candidate(candidate)
                    else:
                        self._candidates[candidate.goal_id] = candidate
                        self._support_counts[candidate.goal_id] = 1
                        self._persist_candidate(candidate)
                        new_candidates.append(candidate)
            self._expire_stale()
            return new_candidates

    #: How much two accounts of a trouble have to have in common before they
    #: are accounts of the SAME trouble. A share of the smaller one's words, so
    #: a short note and a long one can still be about one thing.
    ABOUT_THE_SAME_THING = 0.34

    @staticmethod
    def _what_it_is_about(said: str) -> frozenset[str]:
        """The words that carry what a piece of evidence is about.

        Everything of three letters or more, lowercased. Not a vocabulary: the
        point is only that two accounts of one trouble reuse words and two
        accounts of different troubles do not, which holds whatever the trouble
        is and whatever anybody calls it.
        """
        return frozenset(
            word for word in re.findall(r"[a-z0-9]{3,}", str(said or "").lower())
        )

    def _grouped_by_what_it_is_about(
        self, observations: list[TensionObservation]
    ) -> list[list[TensionObservation]]:
        """Observations of one kind, split into the troubles they are about.

        Each observation joins the group it shares most with, when that is
        enough to call it the same thing, and otherwise starts one of its own.
        Order-dependent by construction and that is honest: what she has seen
        so far is what she has to group by.
        """
        groups: list[tuple[frozenset[str], set[str], list[TensionObservation]]] = []
        for one in observations:
            about = self._what_it_is_about(one.evidence)
            best: tuple[float, int] = (0.0, -1)
            for index, (_founded, words, _kept) in enumerate(groups):
                if not about and not words:
                    shared = 1.0
                elif not about or not words:
                    shared = 0.0
                else:
                    shared = len(about & words) / min(len(about), len(words))
                if shared > best[0]:
                    best = (shared, index)
            if best[0] >= self.ABOUT_THE_SAME_THING and best[1] >= 0:
                _founded, words, kept = groups[best[1]]
                words |= about
                kept.append(one)
            else:
                groups.append((about, set(about), [one]))
        # What FOUNDED each group, not everything it has since collected.
        #
        # The identity has to hold still while evidence accumulates, and the
        # words a group has gathered grow with every new account of it — which
        # is the same drift that made a tension a new goal every time it
        # recurred, one level up. What a group was founded on does not move.
        return [(founded, kept) for founded, _words, kept in groups]

    def _compose_candidate(
        self,
        kind: str,
        observations: list[TensionObservation],
        mean_magnitude: float,
        founded_on: frozenset[str] | None = None,
    ) -> EmergentGoal:
        # The objective's EVIDENCE is observed; its shape is not.
        #
        # Said plainly because the previous note here claimed the objective was
        # built from evidence "not a template", and the line below is a
        # template: what varies is the kind and the evidence, and the frame
        # around them is written here. That is worth being honest about,
        # because it is exactly the boundary between recombining motives and
        # inventing one — she can now form a goal nobody listed, out of tensions
        # nobody predicted, and the sentence it is expressed in is still ours.
        # Distinct excerpts, so the same trouble said twice is one piece of
        # evidence rather than two.
        seen: list[str] = []
        for one in observations[-5:]:
            said = re.sub(r"\s+", " ", one.evidence).strip()
            if said and said not in seen:
                seen.append(said)
        joined_evidence = "; ".join(seen[:3]) or "recurring internal tension"
        objective = f"reduce recurring {kind} tension grounded in: {joined_evidence}"
        # A recurring tension is ONE tension, however much evidence of it
        # arrives.
        #
        # This hashed the evidence text, which grows with every observation —
        # so the fourth sighting of the same trouble was a different goal from
        # the first, support never accumulated on anything, and the adoption
        # threshold could be reached only by a tension whose evidence happened
        # to read identically every time. The mechanism looked live and was
        # almost unreachable.
        #
        # What a tension IS is what it is about. Evidence accumulates onto it
        # and sharpens how the objective reads; it does not make a new one.
        # Kind, plus what this group is about. Stable as evidence accumulates —
        # the shared words are what made it one group — and different for two
        # unrelated troubles filed under the same heading.
        core = "|".join(sorted(founded_on or ())[:8])
        goal_key = hashlib.sha256(f"{kind}|{core}".encode()).hexdigest()[:16]
        name = f"emergent:{kind}:{goal_key[:6]}"
        priority = float(max(0.25, min(0.95, 0.45 + 0.5 * (mean_magnitude - self.TENSION_THRESHOLD))))
        return EmergentGoal(
            goal_id=goal_key,
            name=name,
            objective=objective,
            tension_kind=kind,
            evidence=tuple(seen[:3]),
            priority=priority,
            created_at=time.time(),
            adopted=False,
        )

    # ------------------------------------------------------------------
    # Adoption
    # ------------------------------------------------------------------
    def adoption_ready(self) -> list[EmergentGoal]:
        from core.goals.goal_governance import get_goal_governance_gate

        gate = get_goal_governance_gate()
        with self._lock:
            ready: list[EmergentGoal] = []
            for goal_id, goal in list(self._candidates.items()):
                if goal.adopted:
                    continue
                if self._support_counts.get(goal_id, 0) < self.ADOPTION_THRESHOLD:
                    continue
                # #47 governance bound: an open-ended emergent goal may be adopted
                # only if it stays within the designed value space (CLAIM_BOUNDARIES
                # 4.A — composition within the designed drives, not unbounded
                # genesis). Out-of-space / unsafe candidates are refused here.
                if not gate.is_permitted(goal.objective):
                    continue
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

    async def adopt_into_goal_engine(self, goal_engine: Any) -> list[dict[str, Any]]:
        """Push ready emergent goals into the main GoalEngine."""
        adopted: list[dict[str, Any]] = []
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
            except (OSError, ConnectionError, TimeoutError) as exc:
                record_degradation("emergent_goals", exc)
                logger.debug("Emergent goal adoption failed for %s: %s", goal.goal_id, exc)
                continue
        return adopted

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
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
        with connecting(sqlite3.connect(self._db_path)) as conn:
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
        with connecting(sqlite3.connect(self._db_path)) as conn:
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
        with connecting(sqlite3.connect(self._db_path)) as conn:
            rows = conn.execute("SELECT * FROM emergent_goal_candidates").fetchall()
        for row in rows:
            goal_id, name, objective, kind, evidence_json, priority, created_at, adopted, support = row
            try:
                evidence = list(json.loads(evidence_json or "[]"))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                record_degradation("emergent_goals", exc)
                logger.debug("Emergent goal evidence decode failed for %s: %s", goal_id, exc)
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
                with connecting(sqlite3.connect(self._db_path)) as conn:
                    conn.execute("DELETE FROM emergent_goal_candidates WHERE goal_id = ?", (gid,))
            except (sqlite3.Error, OSError) as exc:
                record_degradation("emergent_goals", exc)
                logger.debug("Emergent goal expiry delete failed for %s: %s", gid, exc)
                continue


_singleton: EmergentGoalEngine | None = None
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
