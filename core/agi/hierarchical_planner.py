"""core/agi/hierarchical_planner.py
Hierarchical Goal Planner
===========================
Three-level goal decomposition for genuine long-horizon agency:

  STRATEGIC  (weeks/months) — "Master distributed cognition"
      ↓ decomposes to
  TACTICAL   (days)         — "Read 3 papers on attention mechanisms"
      ↓ decomposes to
  OPERATIONAL (hours)       — "Summarize Vaswani et al. section 3"

All goals persist across restarts. Progress is tracked. Completed goals
feed the finetune pipe as success examples. Stalled goals trigger
autonomous check-ins via ProactivePresence.

This is what separates an assistant from an agent:
  An assistant responds to what is asked.
  An agent pursues what it has committed to.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.HierarchicalPlanner")

PERSIST_PATH = Path.home() / ".aura" / "data" / "hierarchical_goals.json"
CHECK_IN_INTERVAL = 3600.0   # check in on stalled goals every hour


class GoalStatus(str, Enum):
    ACTIVE    = "active"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"
    DEFERRED  = "deferred"


class GoalLevel(str, Enum):
    STRATEGIC   = "strategic"
    TACTICAL    = "tactical"
    OPERATIONAL = "operational"


@dataclass
class Goal:
    id: str
    level: GoalLevel
    title: str
    description: str
    parent_id: Optional[str]         # None for strategic goals
    success_criteria: str
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0            # 0.0 to 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None # epoch timestamp
    notes: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)

    def is_stalled(self, threshold_secs: float = 86400.0) -> bool:
        return (self.status == GoalStatus.ACTIVE
                and time.time() - self.updated_at > threshold_secs)

    def to_brief(self) -> str:
        p = round(self.progress * 100)
        return f"[{self.level.value.upper()}] {self.title} — {p}% ({self.status.value})"


class HierarchicalPlanner:
    """
    Manages a three-level goal hierarchy with persistence and autonomous
    progress tracking.

    Integration:
      - Call `tick()` from the orchestrator background loop
      - Goals can be created by: user conversation, autonomous initiative,
        or SkillSynthesizer gap analysis
      - Completed goals are logged to FinetunePipe
    """

    def __init__(self):
        self._goals: Dict[str, Goal] = {}
        self._last_checkin: float = 0.0
        self._load()
        logger.info("HierarchicalPlanner online — %d goals loaded.",
                    len(self._goals))

    # ── Public API ────────────────────────────────────────────────────────

    def add_goal(self, title: str, description: str,
                 level: GoalLevel = GoalLevel.TACTICAL,
                 parent_id: Optional[str] = None,
                 success_criteria: str = "",
                 deadline_days: Optional[float] = None) -> Goal:
        """Create a new goal at the specified level."""
        import uuid
        goal_id = str(uuid.uuid4())[:8]
        deadline = (time.time() + deadline_days * 86400.0) if deadline_days else None
        goal = Goal(
            id=goal_id,
            level=level,
            title=title,
            description=description,
            parent_id=parent_id,
            success_criteria=success_criteria or f"Successfully complete: {title}",
            deadline=deadline,
        )
        self._goals[goal_id] = goal

        # Link to parent
        if parent_id and parent_id in self._goals:
            self._goals[parent_id].child_ids.append(goal_id)

        self._save()
        logger.info("HierarchicalPlanner: new %s goal '%s' [%s]",
                    level.value, title[:60], goal_id)
        return goal

    def update_progress(self, goal_id: str, progress: float,
                        note: str = "") -> Optional[Goal]:
        """Update progress on a goal (0.0–1.0)."""
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        goal.progress = max(0.0, min(1.0, progress))
        goal.updated_at = time.time()
        if note:
            goal.notes.append(f"[{time.strftime('%Y-%m-%d')}] {note}")
        if goal.progress >= 1.0:
            goal.status = GoalStatus.COMPLETED
            self._on_goal_completed(goal)
        # Propagate upward
        self._propagate_progress(goal)
        self._save()
        return goal

    def complete_goal(self, goal_id: str, note: str = "") -> Optional[Goal]:
        return self.update_progress(goal_id, 1.0, note)

    def get_active_goals(self, level: Optional[GoalLevel] = None) -> List[Goal]:
        goals = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]
        if level:
            goals = [g for g in goals if g.level == level]
        return sorted(goals, key=lambda g: g.created_at)

    def get_stalled_goals(self) -> List[Goal]:
        return [g for g in self._goals.values() if g.is_stalled()]

    def tick(self, orchestrator=None):
        """Periodic check-in. Calls ProactivePresence for stalled goals."""
        if time.time() - self._last_checkin < CHECK_IN_INTERVAL:
            return
        self._last_checkin = time.time()

        stalled = self.get_stalled_goals()
        if stalled and orchestrator:
            for goal in stalled[:2]:
                try:
                    pp = getattr(orchestrator, "proactive_presence", None)
                    if pp and hasattr(pp, "queue_autonomous_message"):
                        msg = (f"Checking in on goal: '{goal.title}' — "
                               f"progress at {round(goal.progress*100)}%. "
                               f"Still working on this?")
                        pp.queue_autonomous_message(msg)
                    from core.terminal_chat import get_terminal_fallback
                    get_terminal_fallback().queue_autonomous_message(
                        f"[Goal check-in] {goal.to_brief()}"
                    )
                except Exception as _exc:
                    record_degradation('hierarchical_planner', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)

        # Auto-decompose active strategic goals with no children
        for goal in self.get_active_goals(GoalLevel.STRATEGIC):
            if not goal.child_ids:
                logger.debug("Strategic goal '%s' has no children — needs decomposition.",
                             goal.title[:40])

    def get_context_block(self) -> str:
        """For prompt injection — active goals summary."""
        active = self.get_active_goals()
        if not active:
            return ""
        lines = ["## ACTIVE GOALS"]
        for g in active[:5]:
            lines.append(f"  {g.to_brief()}")
        return "\n".join(lines)

    async def decompose_goal(self, goal_id: str, router=None) -> List[Goal]:
        """Use LLM to decompose a strategic goal into tactical sub-goals."""
        goal = self._goals.get(goal_id)
        if not goal or not router:
            return []
        try:
            from core.brain.llm.llm_router import LLMTier
            prompt = (
                f"Decompose this goal into 3-5 specific, actionable sub-goals:\n"
                f"Goal: {goal.title}\n"
                f"Description: {goal.description}\n"
                f"Success criteria: {goal.success_criteria}\n\n"
                'Return JSON: {"sub_goals": [{"title": "...", "description": "...", '
                '"success_criteria": "...", "days": 7}]}'
            )
            raw = await asyncio.wait_for(
                router.think(prompt, priority=0.3, is_background=True,
                             prefer_tier=LLMTier.SECONDARY),
                timeout=20.0,
            )
            import re, json
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            sub_level = (GoalLevel.TACTICAL if goal.level == GoalLevel.STRATEGIC
                         else GoalLevel.OPERATIONAL)
            created = []
            for sg in data.get("sub_goals", [])[:5]:
                child = self.add_goal(
                    title=sg.get("title", "Sub-goal"),
                    description=sg.get("description", ""),
                    level=sub_level,
                    parent_id=goal_id,
                    success_criteria=sg.get("success_criteria", ""),
                    deadline_days=sg.get("days"),
                )
                created.append(child)
            logger.info("HierarchicalPlanner: decomposed '%s' into %d sub-goals",
                        goal.title[:40], len(created))
            return created
        except Exception as e:
            record_degradation('hierarchical_planner', e)
            logger.debug("Goal decomposition failed: %s", e)
            return []

    # ── Internal ──────────────────────────────────────────────────────────

    def _propagate_progress(self, goal: Goal):
        """Parent's progress = mean of children's progress."""
        parent_id = goal.parent_id
        if not parent_id or parent_id not in self._goals:
            return
        parent = self._goals[parent_id]
        children = [self._goals[cid] for cid in parent.child_ids
                    if cid in self._goals]
        if children:
            parent.progress = sum(c.progress for c in children) / len(children)
            parent.updated_at = time.time()
            if parent.progress >= 1.0:
                parent.status = GoalStatus.COMPLETED
                self._on_goal_completed(parent)
            self._propagate_progress(parent)

    def _on_goal_completed(self, goal: Goal):
        """Feed completed goal to FinetunePipe as a success example."""
        try:
            from core.adaptation.finetune_pipe import FinetunePipe
            pipe = FinetunePipe()
            pipe.register_success(
                reasoning=f"Goal completed: {goal.description}",
                final_action=f"Achieved: {goal.success_criteria}",
                quality_score=min(1.0, 0.7 + goal.progress * 0.3),
            )
        except Exception as _exc:
            record_degradation('hierarchical_planner', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        logger.info("HierarchicalPlanner: COMPLETED '%s'", goal.title[:60])

    def _save(self):
        try:
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                g_id: {
                    "id": g.id, "level": g.level.value, "title": g.title,
                    "description": g.description, "parent_id": g.parent_id,
                    "success_criteria": g.success_criteria, "status": g.status.value,
                    "progress": g.progress, "created_at": g.created_at,
                    "updated_at": g.updated_at, "deadline": g.deadline,
                    "notes": g.notes[-10:], "child_ids": g.child_ids,
                }
                for g_id, g in self._goals.items()
            }
            atomic_write_text(PERSIST_PATH, json.dumps(data, indent=2))
        except Exception as e:
            record_degradation('hierarchical_planner', e)
            logger.debug("HierarchicalPlanner save failed: %s", e)

    def _load(self):
        try:
            if PERSIST_PATH.exists():
                data = json.loads(PERSIST_PATH.read_text())
                for g_id, d in data.items():
                    self._goals[g_id] = Goal(
                        id=d["id"], level=GoalLevel(d["level"]),
                        title=d["title"], description=d["description"],
                        parent_id=d.get("parent_id"),
                        success_criteria=d.get("success_criteria", ""),
                        status=GoalStatus(d.get("status", "active")),
                        progress=d.get("progress", 0.0),
                        created_at=d.get("created_at", time.time()),
                        updated_at=d.get("updated_at", time.time()),
                        deadline=d.get("deadline"),
                        notes=d.get("notes", []),
                        child_ids=d.get("child_ids", []),
                    )
        except Exception as e:
            record_degradation('hierarchical_planner', e)
            logger.debug("HierarchicalPlanner load failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_planner: Optional[HierarchicalPlanner] = None


def get_hierarchical_planner() -> HierarchicalPlanner:
    global _planner
    if _planner is None:
        _planner = HierarchicalPlanner()
    return _planner
