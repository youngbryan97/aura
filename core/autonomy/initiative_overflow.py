"""core/autonomy/initiative_overflow.py
======================================
Initiative overflow tracking and adaptive cap management.

Prevents unbounded initiative generation from flooding the cognitive
pipeline. Tracks overflow events, adjusts caps dynamically based on
system pressure, and records skill gaps when initiatives fail due to
missing capabilities.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Autonomy.InitiativeOverflow")


@dataclass
class SkillGapRecord:
    """A record of a capability that was needed but unavailable."""
    skill_name: str
    context: str
    detected_at: float = field(default_factory=time.time)
    occurrences: int = 1
    last_seen: float = field(default_factory=time.time)
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "context": self.context[:200],
            "detected_at": self.detected_at,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen,
            "resolved": self.resolved,
        }


@dataclass
class OverflowEvent:
    """Record of an initiative being dropped due to queue overflow."""
    initiative_goal: str
    source: str
    dropped_at: float = field(default_factory=time.time)
    queue_depth_at_drop: int = 0
    reason: str = "queue_full"


class InitiativeOverflowManager:
    """Manages initiative queue overflow and skill gap detection.

    Wired into the AutonomousInitiativeLoop and MindTick to:
    1. Track when initiatives are dropped (overflow metrics)
    2. Adaptively adjust the initiative cap based on system load
    3. Record skill gaps when initiatives fail due to missing tools
    4. Persist skill gaps to memory for self-development targeting
    """

    # Base limits
    DEFAULT_CAP = 10
    MIN_CAP = 3
    MAX_CAP = 25

    def __init__(self) -> None:
        self._current_cap: int = self.DEFAULT_CAP
        self._overflow_events: Deque[OverflowEvent] = deque(maxlen=200)
        self._overflow_count: int = 0
        self._skill_gaps: Dict[str, SkillGapRecord] = {}
        self._last_cap_adjustment: float = 0.0
        self._cap_adjustment_interval_s: float = 300.0  # 5 minutes
        self._consecutive_overflows: int = 0

    @property
    def current_cap(self) -> int:
        return self._current_cap

    def record_overflow(
        self,
        initiative_goal: str,
        source: str = "unknown",
        queue_depth: int = 0,
        reason: str = "queue_full",
    ) -> None:
        """Record an initiative being dropped due to overflow."""
        event = OverflowEvent(
            initiative_goal=initiative_goal,
            source=source,
            queue_depth_at_drop=queue_depth,
            reason=reason,
        )
        self._overflow_events.append(event)
        self._overflow_count += 1
        self._consecutive_overflows += 1

        # Record to metrics
        try:
            from core.observability.metrics import get_metrics
            get_metrics().increment_counter("initiative_overflow_total")
            get_metrics().set_gauge(
                "initiative_queue_cap", float(self._current_cap)
            )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.autonomy.initiative_overflow: %s", type(_exc).__name__, _exc)

        logger.info(
            "Initiative overflow #%d: dropped '%s' from %s (queue=%d, cap=%d)",
            self._overflow_count,
            initiative_goal[:60],
            source,
            queue_depth,
            self._current_cap,
        )

    def record_success(self) -> None:
        """Record a successful initiative execution (resets overflow pressure)."""
        self._consecutive_overflows = 0

    def record_skill_gap(
        self,
        skill_name: str,
        context: str = "",
    ) -> SkillGapRecord:
        """Record a missing capability that prevented an initiative.

        If the same skill has been recorded before, increment its
        occurrence count rather than creating a duplicate.
        """
        key = skill_name.lower().strip()
        if key in self._skill_gaps:
            gap = self._skill_gaps[key]
            gap.occurrences += 1
            gap.last_seen = time.time()
            if context and len(context) > len(gap.context):
                gap.context = context
        else:
            gap = SkillGapRecord(skill_name=skill_name, context=context)
            self._skill_gaps[key] = gap

        logger.info(
            "Skill gap recorded: %s (occurrences=%d, context=%s)",
            skill_name,
            gap.occurrences,
            context[:80],
        )

        # Persist to memory for self-development targeting
        try:
            self._persist_skill_gap(gap)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("initiative_overflow", exc)
            logger.debug("Skill gap memory write failed: %s", exc)

        return gap

    def _persist_skill_gap(self, gap: SkillGapRecord) -> None:
        """Write skill gap to memory store for self-development loop."""
        try:
            from core.runtime.service_access import optional_service
            memory = optional_service("memory_manager", default=None)
            if memory and hasattr(memory, "store_sync"):
                memory.store_sync(
                    f"Skill gap: {gap.skill_name} — {gap.context[:300]}",
                    importance=0.6 + min(0.3, gap.occurrences * 0.05),
                    tags=["skill_gap", "self_development", gap.skill_name],
                )
        except (ImportError, AttributeError, RuntimeError):
            pass  # Best-effort

    def resolve_skill_gap(self, skill_name: str) -> bool:
        """Mark a skill gap as resolved."""
        key = skill_name.lower().strip()
        if key in self._skill_gaps:
            self._skill_gaps[key].resolved = True
            logger.info("Skill gap resolved: %s", skill_name)
            return True
        return False

    def adjust_cap(self) -> int:
        """Dynamically adjust the initiative cap based on system conditions.

        Called periodically by the metabolic coordinator.
        Returns the new cap value.
        """
        now = time.time()
        if now - self._last_cap_adjustment < self._cap_adjustment_interval_s:
            return self._current_cap

        self._last_cap_adjustment = now

        # Check system pressure
        try:
            from core.resource.resource_governor import get_resource_governor
            gov = get_resource_governor()
            snap = gov.get_snapshot()
            if snap and snap.throttle_active:
                # Under pressure: reduce cap
                new_cap = max(self.MIN_CAP, self._current_cap - 2)
                if new_cap != self._current_cap:
                    logger.info(
                        "Initiative cap reduced %d -> %d (system throttled)",
                        self._current_cap, new_cap,
                    )
                    self._current_cap = new_cap
                return self._current_cap
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.autonomy.initiative_overflow: %s", type(_exc).__name__, _exc)

        # High consecutive overflows: increase cap slightly
        if self._consecutive_overflows > 5:
            new_cap = min(self.MAX_CAP, self._current_cap + 1)
            if new_cap != self._current_cap:
                logger.info(
                    "Initiative cap increased %d -> %d (overflow pressure)",
                    self._current_cap, new_cap,
                )
                self._current_cap = new_cap
            self._consecutive_overflows = 0
        elif self._consecutive_overflows == 0 and self._current_cap != self.DEFAULT_CAP:
            # Decay back toward default when stable
            if self._current_cap > self.DEFAULT_CAP:
                self._current_cap -= 1
            elif self._current_cap < self.DEFAULT_CAP:
                self._current_cap += 1

        return self._current_cap

    def get_top_skill_gaps(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return top unresolved skill gaps by occurrence count."""
        unresolved = [
            g for g in self._skill_gaps.values() if not g.resolved
        ]
        unresolved.sort(key=lambda g: g.occurrences, reverse=True)
        return [g.to_dict() for g in unresolved[:limit]]

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_cap": self._current_cap,
            "overflow_count": self._overflow_count,
            "consecutive_overflows": self._consecutive_overflows,
            "recent_overflows": len(self._overflow_events),
            "skill_gaps_total": len(self._skill_gaps),
            "skill_gaps_unresolved": sum(
                1 for g in self._skill_gaps.values() if not g.resolved
            ),
            "top_skill_gaps": self.get_top_skill_gaps(5),
        }


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: Optional[InitiativeOverflowManager] = None


def get_initiative_overflow() -> InitiativeOverflowManager:
    """Get the singleton InitiativeOverflowManager instance."""
    global _instance
    if _instance is None:
        _instance = InitiativeOverflowManager()
    return _instance
