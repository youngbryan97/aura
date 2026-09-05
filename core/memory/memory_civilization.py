"""core/memory/memory_civilization.py — Memory Civilization.

Manages durable memories, episodic storage lanes, and post-mission
narrative compression to distill logs into core operational lessons:
  what happened, what changed, what failed, what was learned,
  what should be remembered, what should be forgotten, what should be retried.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.MemoryCivilization")


@dataclass
class DurableLesson:
    lesson_id: str
    mission_id: str
    what_happened: str
    what_changed: str
    what_failed: str
    what_was_learned: str
    remember_targets: List[str]
    forget_targets: List[str]
    retry_recommendation: str
    timestamp: float = field(default_factory=time.time)


class MemoryCivilization:
    """Manages structural lessons and log compression to avoid context bloating."""

    def __init__(self) -> None:
        self.lessons: Dict[str, DurableLesson] = {}
        self.raw_logs: List[Dict[str, Any]] = []

    def record_raw_event(self, event_type: str, details: str) -> None:
        self.raw_logs.append({
            "type": event_type,
            "details": details,
            "timestamp": time.time(),
        })

    def compress_mission_logs(
        self,
        mission_id: str,
        outcome_ok: bool,
        failed_steps: List[str],
    ) -> DurableLesson:
        """Post-mission narrative compression: distills logs into structured lessons."""
        logger.info("📦 MemoryCivilization: compressing logs for mission %s...", mission_id)

        # Ingest recent raw logs related to the mission
        mission_raw = [r for r in self.raw_logs if mission_id in r["details"]]
        self.raw_logs = [r for r in self.raw_logs if mission_id not in r["details"]]  # Prune/forget raw logs

        what_happened = f"Mission {mission_id} finished. Success={outcome_ok}."
        what_changed = f"Cleaned {len(mission_raw)} raw log lines. Persisted as compressed lesson."
        what_failed = "None"
        retry_recommendation = "Maintain current approach"

        if not outcome_ok:
            what_failed = f"Failed steps: {', '.join(failed_steps)}" if failed_steps else "Unexpected termination"
            retry_recommendation = "Re-validate plan steps and insert rollback safeguards"

        lesson = DurableLesson(
            lesson_id=f"lesson_{int(time.time())}",
            mission_id=mission_id,
            what_happened=what_happened,
            what_changed=what_changed,
            what_failed=what_failed,
            what_was_learned="Systematic validation prevents execution drift.",
            remember_targets=[f"Outcome of {mission_id} was {outcome_ok}"],
            forget_targets=[f"Raw logging traces for {mission_id}"],
            retry_recommendation=retry_recommendation,
        )

        self.lessons[lesson.lesson_id] = lesson
        logger.info("💾 Compressed %d raw events into lesson %s", len(mission_raw), lesson.lesson_id)
        return lesson

    def get_lessons(self) -> List[DurableLesson]:
        return list(self.lessons.values())

    async def retrieve_context(self, objective: str) -> Dict[str, Any]:
        """ episodic contextual recall based on keyword matching."""
        matches = [l for l in self.lessons.values() if any(k in l.what_happened.lower() for k in objective.lower().split()[:3])]
        return {
            "relevant_lessons_count": len(matches),
            "lessons": [l.what_was_learned for l in matches[:3]],
        }

    async def record_mission_outcome(self, objective: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Commit mission outcome to the lessons list."""
        self.compress_mission_logs(
            mission_id=objective[:80],
            outcome_ok=result.get("ok", False),
            failed_steps=[] if result.get("ok", False) else ["execution_step"],
        )
        return {"committed": True}


# ── Singleton ───────────────────────────────────────────────────────────
_memory_civilization_instance: MemoryCivilization | None = None


def get_memory_civilization() -> MemoryCivilization:
    global _memory_civilization_instance
    if _memory_civilization_instance is None:
        _memory_civilization_instance = MemoryCivilization()
    return _memory_civilization_instance
