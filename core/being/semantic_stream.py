"""core/being/semantic_stream.py — Always-On Semantic State Process.

A small, persistent semantic model that evolves BETWEEN LLM calls.
Not a giant 32B running constantly — a typed state graph + short
latent summaries that bridge episodic LLM cognition.

Maintains:
  - current_situation: what is happening now
  - unresolved_tensions: active conflicts/uncertainties
  - active_goals: what Aura is trying to achieve
  - predicted_needs: what will be needed next
  - memory_uncertainties: what is unreliable
  - relationship_state: context with interaction partner
  - self_state: compact welfare/body/affect summary
  - environmental_affordances: what actions are available

The stream evolves while idle. Not with prose monologue — with
structured state updates.

Design:
  - Compact typed state graph (no LLM needed for updates)
  - Evolves via rule-based transitions + welfare feedback
  - The big model reads the stream when invoked
  - Produces internally generated state evolution between user turns
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.SemanticStream")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class StreamGoal:
    """A goal in the semantic stream."""
    goal_id: str
    description: str
    priority: float = 0.5       # 0-1
    progress: float = 0.0       # 0-1
    blocked: bool = False
    blocked_reason: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    last_updated: float = field(default_factory=lambda: time.time())

    def is_stale(self, max_age_s: float = 3600.0) -> bool:
        return (time.time() - self.last_updated) > max_age_s


@dataclass
class StreamTension:
    """An unresolved tension or conflict."""
    tension_id: str
    description: str
    severity: float = 0.3       # 0-1
    source: str = ""            # where did this come from
    created_at: float = field(default_factory=lambda: time.time())
    resolved: bool = False


@dataclass
class StreamMemoryUncertainty:
    """A specific memory that is uncertain or unreliable."""
    topic: str
    confidence: float = 0.5     # 0-1, how reliable
    last_verified: float = 0.0
    contradiction_count: int = 0


@dataclass
class SemanticState:
    """The full semantic stream state at a point in time."""
    # Situation awareness
    current_situation: str = "idle"
    situation_since: float = field(default_factory=lambda: time.time())

    # Goals
    active_goals: list[StreamGoal] = field(default_factory=list)
    completed_goals: int = 0
    blocked_goals: int = 0

    # Tensions
    unresolved_tensions: list[StreamTension] = field(default_factory=list)

    # Predicted needs
    predicted_next_needs: list[str] = field(default_factory=list)

    # Memory state
    memory_uncertainties: list[StreamMemoryUncertainty] = field(default_factory=list)
    memory_coherence_estimate: float = 1.0

    # Relationship
    relationship_trust: float = 1.0
    interaction_recency_s: float = 0.0
    interaction_count: int = 0

    # Self state (compact welfare/body summary)
    welfare_score: float = 0.5
    body_health: float = 1.0
    distress_level: float = 0.0
    fatigue_level: float = 0.0
    recovery_drive: float = 0.0

    # Environment
    available_tools: int = 0
    pending_tasks: int = 0
    environment_stable: bool = True

    # Stream metadata
    tick: int = 0
    last_evolution: float = field(default_factory=lambda: time.time())
    evolutions_since_interaction: int = 0

    def is_idle(self) -> bool:
        return self.current_situation == "idle"

    def has_active_work(self) -> bool:
        return bool(self.active_goals) or self.pending_tasks > 0

    def to_prompt_block(self) -> str:
        """Compact representation for the LLM to read."""
        goals_str = ", ".join(
            f"{g.goal_id}({g.priority:.1f})" for g in self.active_goals[:5]
        ) or "none"
        tensions_str = ", ".join(
            t.description[:40] for t in self.unresolved_tensions if not t.resolved
        )[:100] or "none"
        uncertainties_str = ", ".join(
            f"{u.topic}({u.confidence:.1f})" for u in self.memory_uncertainties[:3]
        ) or "none"

        return (
            f"## SEMANTIC STREAM (tick={self.tick})\n"
            f"- Situation: {self.current_situation}\n"
            f"- Goals: {goals_str}\n"
            f"- Tensions: {tensions_str}\n"
            f"- Uncertainties: {uncertainties_str}\n"
            f"- Welfare: {self.welfare_score:.2f}, Distress: {self.distress_level:.2f}, "
            f"Fatigue: {self.fatigue_level:.2f}\n"
            f"- Body: {self.body_health:.2f}, Recovery: {self.recovery_drive:.2f}\n"
            f"- Idle evolutions: {self.evolutions_since_interaction}\n"
        )


class SemanticStream:
    """Always-on semantic state that evolves between LLM calls.

    Usage:
        stream = SemanticStream.get()
        stream.update_welfare(welfare_score=0.7, distress=0.1, ...)
        stream.add_goal("resolve_user_issue", "Fix the failing test", priority=0.8)
        stream.evolve()  # called by heartbeat/timer, not by user
        state = stream.state  # read by LLM when invoked
    """

    _instance: SemanticStream | None = None

    def __init__(self) -> None:
        self._state = SemanticState()
        self._history: deque[dict[str, Any]] = deque(maxlen=200)
        self._lesioned = False

    @classmethod
    def get(cls) -> SemanticStream:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @property
    def state(self) -> SemanticState:
        return self._state

    def update_welfare(
        self,
        *,
        welfare_score: float = 0.5,
        distress: float = 0.0,
        fatigue: float = 0.0,
        recovery_drive: float = 0.0,
        body_health: float = 1.0,
    ) -> None:
        """Update welfare/body dimensions from WelfareState/BodyStateService."""
        self._state.welfare_score = _clip(welfare_score)
        self._state.distress_level = _clip(distress)
        self._state.fatigue_level = _clip(fatigue)
        self._state.recovery_drive = _clip(recovery_drive)
        self._state.body_health = _clip(body_health)

    def update_situation(self, situation: str) -> None:
        if situation != self._state.current_situation:
            self._state.current_situation = situation
            self._state.situation_since = time.time()

    def add_goal(self, goal_id: str, description: str, priority: float = 0.5) -> None:
        # Don't duplicate
        for g in self._state.active_goals:
            if g.goal_id == goal_id:
                g.priority = priority
                g.last_updated = time.time()
                return
        self._state.active_goals.append(StreamGoal(
            goal_id=goal_id, description=description[:200], priority=_clip(priority),
        ))

    def complete_goal(self, goal_id: str) -> None:
        self._state.active_goals = [
            g for g in self._state.active_goals if g.goal_id != goal_id
        ]
        self._state.completed_goals += 1

    def block_goal(self, goal_id: str, reason: str) -> None:
        for g in self._state.active_goals:
            if g.goal_id == goal_id:
                g.blocked = True
                g.blocked_reason = reason[:100]
                g.last_updated = time.time()

    def add_tension(self, tension_id: str, description: str, severity: float = 0.3, source: str = "") -> None:
        for t in self._state.unresolved_tensions:
            if t.tension_id == tension_id:
                t.severity = _clip(severity)
                return
        self._state.unresolved_tensions.append(StreamTension(
            tension_id=tension_id, description=description[:200],
            severity=_clip(severity), source=source,
        ))

    def resolve_tension(self, tension_id: str) -> None:
        for t in self._state.unresolved_tensions:
            if t.tension_id == tension_id:
                t.resolved = True

    def add_memory_uncertainty(self, topic: str, confidence: float = 0.5) -> None:
        for u in self._state.memory_uncertainties:
            if u.topic == topic:
                u.confidence = _clip(confidence)
                return
        self._state.memory_uncertainties.append(StreamMemoryUncertainty(
            topic=topic[:100], confidence=_clip(confidence),
        ))

    def record_interaction(self) -> None:
        """Mark that a user interaction occurred."""
        self._state.interaction_count += 1
        self._state.interaction_recency_s = 0.0
        self._state.evolutions_since_interaction = 0

    def evolve(self) -> dict[str, Any]:
        """Evolve the semantic state one step. Called by heartbeat, not by user.

        Returns a summary of what changed.
        """
        if self._lesioned:
            return {"evolved": False, "reason": "lesioned"}

        changes: dict[str, Any] = {}
        self._state.tick += 1
        self._state.last_evolution = time.time()
        self._state.evolutions_since_interaction += 1

        # ── Time-based decay ──
        # Interaction recency grows
        self._state.interaction_recency_s += 1.0  # approximate

        # ── Goal maintenance ──
        # Prune stale goals
        stale_goals = [g for g in self._state.active_goals if g.is_stale(7200)]
        if stale_goals:
            self._state.active_goals = [
                g for g in self._state.active_goals if not g.is_stale(7200)
            ]
            changes["pruned_stale_goals"] = len(stale_goals)

        # Count blocked goals
        self._state.blocked_goals = sum(1 for g in self._state.active_goals if g.blocked)

        # ── Tension evolution ──
        # Unresolved tensions accumulate severity over time
        for t in self._state.unresolved_tensions:
            if not t.resolved:
                t.severity = _clip(t.severity + 0.005)  # slow escalation

        # Prune resolved tensions
        resolved = [t for t in self._state.unresolved_tensions if t.resolved]
        if resolved:
            self._state.unresolved_tensions = [
                t for t in self._state.unresolved_tensions if not t.resolved
            ]
            changes["resolved_tensions"] = len(resolved)

        # ── Memory uncertainty decay ──
        # Uncertainties grow slightly over time without verification
        for u in self._state.memory_uncertainties:
            u.confidence = _clip(u.confidence - 0.002)  # slow degradation

        if self._state.memory_uncertainties:
            uncertainty_pressure = max(
                1.0 - u.confidence for u in self._state.memory_uncertainties
            )
            average_pressure = sum(
                1.0 - u.confidence for u in self._state.memory_uncertainties
            ) / len(self._state.memory_uncertainties)
            self._state.memory_coherence_estimate = _clip(
                1.0 - max(uncertainty_pressure * 0.65, average_pressure * 0.5)
            )
        else:
            self._state.memory_coherence_estimate = 1.0

        # ── Welfare-driven state transitions ──
        if self._state.distress_level > 0.6 and self._state.current_situation != "recovery":
            self.update_situation("recovery_needed")
            changes["situation_transition"] = "recovery_needed"

        if self._state.recovery_drive > 0.5 and not self._state.has_active_work():
            self.update_situation("idle_recovery")
            changes["situation_transition"] = "idle_recovery"

        # ── Predicted needs ──
        needs = []
        if self._state.distress_level > 0.4:
            needs.append("stabilization")
        if self._state.fatigue_level > 0.5:
            needs.append("rest")
        if self._state.blocked_goals > 0:
            needs.append("goal_unblocking")
        if self._state.memory_coherence_estimate < 0.7:
            needs.append("memory_verification")
        if not self._state.active_goals and self._state.current_situation == "idle":
            needs.append("goal_formation")
        self._state.predicted_next_needs = needs
        if needs:
            changes["predicted_needs"] = needs

        # ── Record evolution ──
        if changes:
            self._history.append({
                "tick": self._state.tick,
                "timestamp": time.time(),
                "changes": changes,
            })

        return changes

    def evolution_history(self, n: int = 50) -> list[dict[str, Any]]:
        return list(self._history)[-n:]

    def has_evolved_since_interaction(self) -> bool:
        return self._state.evolutions_since_interaction > 0

    def lesion(self) -> None:
        self._lesioned = True

    def restore(self) -> None:
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned
