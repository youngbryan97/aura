"""core/agency/tension_engine.py — Cognitive Tension Tracker
============================================================
Tracks unresolved cognitive tensions that fuel autonomous motivation.

Tensions are the "itches" of a mind — contradictions, open questions,
broken expectations, and unfinished commitments that create pressure
to act. The TensionEngine scans for these across subsystems and
maintains a registry of active tensions sorted by severity.

Persists to disk so tensions survive restarts.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Agency")


class TensionCategory(Enum):
    GOAL_CONFLICT = "goal_conflict"
    BELIEF_CONTRADICTION = "belief_contradiction"
    OPEN_QUESTION = "open_question"
    BROKEN_EXPECTATION = "broken_expectation"
    UNFINISHED_COMMITMENT = "unfinished_commitment"
    IDENTITY_INCONSISTENCY = "identity_inconsistency"
    CURIOSITY_GAP = "curiosity_gap"


@dataclass
class Tension:
    id: str
    category: TensionCategory
    description: str
    severity: float                          # 0.0–1.0
    created_at: float = field(default_factory=time.time)
    last_checked_at: float = field(default_factory=time.time)
    resolution_attempts: int = 0
    source_subsystem: str = "unknown"
    related_beliefs: List[str] = field(default_factory=list)
    related_goals: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution: Optional[str] = None

    # -- Serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Tension":
        d = dict(d)  # shallow copy — don't mutate the source
        d["category"] = TensionCategory(d["category"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Severity aging constants
# ---------------------------------------------------------------------------
_AGE_RATE_PER_HOUR = 0.02      # severity increases this much per hour unresolved
_MAX_SEVERITY = 1.0
_STALE_QUESTION_SECS = 300.0   # 5 minutes before a curiosity queue item counts


class TensionEngine:
    """Registry and scanner for unresolved cognitive tensions."""

    name = "tension_engine"

    def __init__(self, persist_path: Optional[Path] = None):
        if persist_path is not None:
            self._persist_path = Path(persist_path)
        else:
            try:
                from core.config import config
                self._persist_path = config.paths.data_dir / "tensions.json"
            except Exception:
                self._persist_path = Path.home() / ".aura" / "data" / "tensions.json"

        self._tensions: Dict[str, Tension] = {}
        self._load()

    # -- Persistence ----------------------------------------------------------

    def _load(self) -> None:
        if self._persist_path.exists():
            try:
                raw = json.loads(self._persist_path.read_text())
                for entry in raw:
                    try:
                        t = Tension.from_dict(entry)
                        self._tensions[t.id] = t
                    except Exception as exc:
                        logger.debug("Skipping malformed tension entry: %s", exc)
                logger.info("TensionEngine loaded %d tensions from disk.", len(self._tensions))
            except Exception as exc:
                logger.warning("TensionEngine failed to load persisted tensions: %s", exc)
        else:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [t.to_dict() for t in self._tensions.values()]
            self._persist_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.error("TensionEngine failed to persist tensions: %s", exc)

    # -- Public API -----------------------------------------------------------

    def register_tension(self, tension: Tension) -> None:
        """Add or update a tension from any subsystem."""
        self._tensions[tension.id] = tension
        logger.debug("Tension registered [%s]: %s (severity=%.2f)",
                      tension.category.value, tension.description[:80], tension.severity)
        self._save()

    def resolve_tension(self, tension_id: str, resolution: str) -> None:
        """Mark a tension as resolved."""
        t = self._tensions.get(tension_id)
        if t is None:
            logger.debug("resolve_tension: unknown id %s", tension_id)
            return
        t.resolved = True
        t.resolution = resolution
        t.last_checked_at = time.time()
        logger.info("Tension resolved [%s]: %s", tension_id, resolution[:120])
        self._save()

    def get_active_tensions(self) -> List[Tension]:
        """All unresolved tensions sorted by severity descending."""
        active = [t for t in self._tensions.values() if not t.resolved]
        active.sort(key=lambda t: t.severity, reverse=True)
        return active

    def get_highest_tension(self) -> Optional[Tension]:
        """The single most pressing unresolved tension, or None."""
        active = self.get_active_tensions()
        return active[0] if active else None

    def get_tension_pressure(self) -> float:
        """Overall tension level (0.0–1.0).

        Computed as a saturating sum: each active tension contributes its
        severity, but the total is capped at 1.0 with a soft sigmoid-like
        curve so that a handful of moderate tensions don't immediately
        saturate the signal.
        """
        active = self.get_active_tensions()
        if not active:
            return 0.0
        raw = sum(t.severity for t in active)
        # Soft cap: pressure = raw / (raw + k) where k controls steepness
        k = 3.0
        return min(1.0, raw / (raw + k))

    # -- Tick: Automatic Scanning ---------------------------------------------

    async def tick(self, state) -> None:
        """Scan subsystems for new tensions and age existing ones.

        Designed to be called once per cognitive tick.  Each sub-scanner
        is wrapped in its own try/except so a single failure never blocks
        the rest.
        """
        now = time.time()

        # 1. Belief contradictions
        try:
            self._scan_belief_contradictions()
        except Exception as exc:
            logger.debug("TensionEngine: belief scan failed: %s", exc)

        # 2. Goal conflicts
        try:
            self._scan_goal_conflicts(state)
        except Exception as exc:
            logger.debug("TensionEngine: goal conflict scan failed: %s", exc)

        # 3. Stale curiosity questions
        try:
            self._scan_curiosity_gaps(now)
        except Exception as exc:
            logger.debug("TensionEngine: curiosity gap scan failed: %s", exc)

        # 4. Broken expectations (recent failed actions)
        try:
            self._scan_broken_expectations(state)
        except Exception as exc:
            logger.debug("TensionEngine: broken expectation scan failed: %s", exc)

        # 5. Identity drift signals
        try:
            self._scan_identity_drift()
        except Exception as exc:
            logger.debug("TensionEngine: identity drift scan failed: %s", exc)

        # 6. Age all unresolved tensions
        self._age_tensions(now)

        self._save()

    # -- Sub-scanners ---------------------------------------------------------

    def _scan_belief_contradictions(self) -> None:
        """Look for beliefs with contradictory content or low confidence clusters."""
        bre = ServiceContainer.get("belief_revision_engine", default=None)
        if bre is None:
            return
        beliefs = getattr(bre, "beliefs", [])
        if len(beliefs) < 2:
            return

        # Simple heuristic: beliefs in the same domain with opposing valences
        # and high confidence are likely contradictions.
        by_domain: Dict[str, list] = {}
        for b in beliefs:
            by_domain.setdefault(b.domain, []).append(b)

        for domain, group in by_domain.items():
            if len(group) < 2:
                continue
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if a.confidence > 0.5 and b.confidence > 0.5:
                        # Opposing valence signals potential contradiction
                        if a.emotional_valence * b.emotional_valence < -0.1:
                            tid = f"belief_contra_{a.id}_{b.id}"
                            if tid not in self._tensions or self._tensions[tid].resolved:
                                self.register_tension(Tension(
                                    id=tid,
                                    category=TensionCategory.BELIEF_CONTRADICTION,
                                    description=(
                                        f"Belief '{a.content[:60]}' (valence={a.emotional_valence:+.1f}) "
                                        f"contradicts '{b.content[:60]}' (valence={b.emotional_valence:+.1f}) "
                                        f"in domain '{domain}'."
                                    ),
                                    severity=min(1.0, (a.confidence + b.confidence) / 2 * 0.7),
                                    source_subsystem="belief_revision_engine",
                                    related_beliefs=[a.id, b.id],
                                ))

    def _scan_goal_conflicts(self, state) -> None:
        """Detect conflicting active goals (resource contention or mutual exclusion)."""
        goals = getattr(state.cognition, "active_goals", [])
        if len(goals) < 2:
            return

        # Lightweight conflict detection: goals that target the same resource
        # or have explicitly opposing objectives.
        seen_resources: Dict[str, list] = {}
        for g in goals:
            resource = g.get("resource") or g.get("type", "general")
            seen_resources.setdefault(resource, []).append(g)

        for resource, group in seen_resources.items():
            if len(group) < 2:
                continue
            tid = f"goal_conflict_{resource}_{int(time.time())}"
            descs = [g.get("goal", g.get("description", "?"))[:40] for g in group[:3]]
            if tid not in self._tensions:
                self.register_tension(Tension(
                    id=tid,
                    category=TensionCategory.GOAL_CONFLICT,
                    description=f"Multiple goals competing for '{resource}': {', '.join(descs)}",
                    severity=0.5,
                    source_subsystem="cognitive_context",
                    related_goals=[g.get("goal", "")[:60] for g in group[:3]],
                ))

    def _scan_curiosity_gaps(self, now: float) -> None:
        """Promote stale curiosity queue items to tensions."""
        try:
            from core.agi.curiosity_explorer import get_curiosity_explorer
            explorer = get_curiosity_explorer()
        except Exception:
            return

        for item in getattr(explorer, "_queue", []):
            if item.completed:
                continue
            age = now - item.created_at
            if age < _STALE_QUESTION_SECS:
                continue
            tid = f"curiosity_gap_{hash(item.question) & 0xFFFFFFFF}"
            if tid in self._tensions and not self._tensions[tid].resolved:
                continue
            self.register_tension(Tension(
                id=tid,
                category=TensionCategory.CURIOSITY_GAP,
                description=f"Unanswered curiosity: '{item.question[:100]}'",
                severity=min(0.8, 0.3 + (age / 3600) * 0.1),
                source_subsystem="curiosity_explorer",
            ))

    def _scan_broken_expectations(self, state) -> None:
        """Look for recent failed actions flagged in cognition modifiers."""
        modifiers = getattr(state.cognition, "modifiers", {})
        failures = modifiers.get("recent_failures", [])
        if not failures:
            return

        for fail in failures[-5:]:
            description = fail if isinstance(fail, str) else str(fail)[:120]
            tid = f"broken_exp_{hash(description) & 0xFFFFFFFF}"
            if tid in self._tensions and not self._tensions[tid].resolved:
                continue
            self.register_tension(Tension(
                id=tid,
                category=TensionCategory.BROKEN_EXPECTATION,
                description=f"Failed action: {description}",
                severity=0.5,
                source_subsystem="cognition",
            ))

    def _scan_identity_drift(self) -> None:
        """Check the identity drift monitor for recent signals."""
        drift_monitor = ServiceContainer.get("identity_drift_monitor", default=None)
        if drift_monitor is None:
            return

        history = getattr(drift_monitor, "_drift_history", [])
        recent = [s for s in history if not s.corrected and (time.time() - s.timestamp) < 600]
        if not recent:
            return

        # Aggregate into a single tension per signal type
        by_type: Dict[str, list] = {}
        for s in recent:
            by_type.setdefault(s.signal_type, []).append(s)

        for signal_type, signals in by_type.items():
            tid = f"identity_drift_{signal_type}"
            avg_severity = sum(s.severity for s in signals) / len(signals)
            if tid in self._tensions and not self._tensions[tid].resolved:
                # Update severity to reflect accumulation
                existing = self._tensions[tid]
                existing.severity = min(1.0, max(existing.severity, avg_severity + 0.1 * len(signals)))
                existing.last_checked_at = time.time()
                existing.resolution_attempts += 1
                continue
            self.register_tension(Tension(
                id=tid,
                category=TensionCategory.IDENTITY_INCONSISTENCY,
                description=f"Identity drift detected: {signal_type} ({len(signals)} signals in last 10m)",
                severity=min(1.0, avg_severity + 0.05 * len(signals)),
                source_subsystem="identity_drift_monitor",
            ))

    # -- Aging ----------------------------------------------------------------

    def _age_tensions(self, now: float) -> None:
        """Increase severity of unresolved tensions over time."""
        for t in self._tensions.values():
            if t.resolved:
                continue
            hours_since_check = (now - t.last_checked_at) / 3600.0
            if hours_since_check > 0:
                t.severity = min(_MAX_SEVERITY, t.severity + _AGE_RATE_PER_HOUR * hours_since_check)
                t.last_checked_at = now


# -- Singleton ----------------------------------------------------------------

_instance: Optional[TensionEngine] = None


def get_tension_engine() -> TensionEngine:
    global _instance
    if _instance is None:
        _instance = TensionEngine()
    return _instance
