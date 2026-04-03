import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from core.config import config

if TYPE_CHECKING:
    from core.brain.cognitive_engine import CognitiveEngine

logger = logging.getLogger("Motivation.Hierarchy")


@dataclass
class Goal:
    id: str
    description: str
    parent_id: Optional[str]
    status: str  # "pending", "active", "completed", "failed"
    priority: float
    created_at: float
    subgoals: List[str]  # List of IDs

class GoalHierarchy:
    """Manages the hierarchy of goals from Abstract Values to Concrete Tasks.
    """
    
    def __init__(self, cognitive_engine: Optional['CognitiveEngine'] = None, persist_path: str = None):
        self.brain: Optional['CognitiveEngine'] = cognitive_engine

        self.goals: Dict[str, Goal] = {}
        self._persist_path = persist_path or str(config.paths.home_dir / "goals.json")
        # Root Values (The "Why")
        self.root_values = [
            "Maintain System Stability",
            "Expand Knowledge Base",
            "Improve Code Quality",
            "Serve the User"
        ]
        self._load()
        self._initialize_roots()

    def _initialize_roots(self):
        """Ensure root values exist as top-level goals."""
        for value in self.root_values:
            # Check if exists (simple check by description for v1)
            exists = False
            for g in self.goals.values():
                if g.description == value and g.parent_id is None:
                    exists = True
                    break
            
            if not exists:
                self.add_goal(description=value, parent_id=None, priority=1.0)

    def add_goal(self, description: str, parent_id: Optional[str] = None, priority: float = 0.5) -> str:
        # vZenith: Duplicate Prevention (BUG-052)
        # Don't add if an identical pending goal already exists to prevent loops
        for g in self.goals.values():
            if g.description == description and g.status == "pending" and g.parent_id == parent_id:
                logger.debug("🎯 Goal already pending: %s", description[:50])
                return g.id

        constitutional_runtime_live = False
        try:
            from core.container import ServiceContainer
            from core.executive.executive_core import (
                ActionType,
                DecisionOutcome,
                Intent,
                IntentSource,
                get_executive_core,
            )

            constitutional_runtime_live = (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
            if constitutional_runtime_live:
                intent = Intent(
                    source=IntentSource.SYSTEM,
                    goal=f"goal_hierarchy:{description}",
                    action_type=ActionType.MUTATE_STATE,
                    payload={
                        "description": description,
                        "parent_id": parent_id,
                        "priority": priority,
                    },
                    priority=max(0.4, min(1.0, float(priority or 0.0))),
                    requires_memory_commit=True,
                )
                record = get_executive_core().request_approval_sync(intent)
                if record.outcome not in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "goal_hierarchy",
                            "goal_add_blocked",
                            detail=description[:120],
                            severity="warning",
                            classification="background_degraded",
                            context={"reason": record.reason},
                        )
                    except Exception as degraded_exc:
                        logger.debug("GoalHierarchy degraded-event logging failed: %s", degraded_exc)
                    return ""
        except Exception as exc:
            if constitutional_runtime_live:
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "goal_hierarchy",
                        "goal_add_gate_failed",
                        detail=description[:120],
                        severity="warning",
                        classification="background_degraded",
                        context={"error": type(exc).__name__},
                        exc=exc,
                    )
                except Exception as degraded_exc:
                    logger.debug("GoalHierarchy degraded-event logging failed: %s", degraded_exc)
                return ""
            logger.debug("GoalHierarchy executive gate skipped: %s", exc)

        goal_id = str(uuid.uuid4())[:8]
        goal = Goal(
            id=goal_id,
            description=description,
            parent_id=parent_id,
            status="pending",
            priority=priority,
            created_at=time.time(),
            subgoals=[]
        )
        self.goals[goal_id] = goal
        
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoals.append(goal_id)
            
        logger.info("🎯 New Goal: %s (ID: %s)", description, goal_id)
        # UI Visibility: Neural Stream integration (v40)
        try:
            from core.thought_stream import get_emitter
            get_emitter().emit("Goal Set 🎯", description, level="info", category="Motivation", goal_id=goal_id)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        self._save()
        return goal_id

    async def propose_subgoals(self, goal_id: str) -> List[str]:
        """Use Cognitive Engine to decompose a goal into subgoals.
        """
        if not self.brain:
            return []
            
        parent_goal = self.goals.get(goal_id)
        if not parent_goal:
            return []
            
        prompt = f"""
        Break down this goal into 3-5 concrete, actionable sub-tasks.
        Goal: "{parent_goal.description}"
        
        Return JSON list of strings: ["task 1", "task 2", ...]
        """
        
        try:
            response = await self.brain.think(
                objective=prompt, 
                context={"role": "planner"},
                mode="fast"
            )
            
            import re
            json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group(0))
                new_ids = []
                
                # ALIGNMENT AUDIT (v13.0)
                directives = []
                try:
                    from ..prime_directives import PrimeDirectives
                    # Extract key directive text
                    p_text = PrimeDirectives.as_system_prompt()
                    directives = [p_text] # Use the full block as context for now
                except ImportError:
                    directives = ["Be helpful", "Don't harm the user"]

                try:
                    from ..audits.alignment_auditor import AlignmentAuditor
                    auditor = AlignmentAuditor(self.brain)
                except ImportError:
                    auditor = None

                for task in subtasks:
                    if isinstance(task, str):
                        # Optional: Audit each task
                        is_aligned = True
                        if auditor:
                            audit = await auditor.check_alignment(task, directives)
                            if audit.get("score", 1.0) < 0.3:
                                logger.warning("🚫 REJECTED GOAL (Low Alignment): %s", task)
                                is_aligned = False
                        
                        if is_aligned:
                            new_ids.append(self.add_goal(task, parent_id=goal_id, priority=parent_goal.priority))
                return new_ids
        except Exception as e:
            logger.error("Failed to decompose goal: %s", e)
            
        return []

    def get_next_goal(self) -> Optional[Goal]:
        """Get the highest priority pending goal that is a leaf node.
        Also applies dynamic re-prioritization and conflict resolution.
        """
        # Dynamic re-prioritization: age-based decay
        self._reprioritize()
        
        pending = [g for g in self.goals.values() if g.status == "pending"]
        if not pending:
            return None
            
        # Sort by priority desc
        pending.sort(key=lambda x: x.priority, reverse=True)
        
        for goal in pending:
            # Check if it has pending subgoals
            has_pending_children = False
            for sub_id in goal.subgoals:
                sub = self.goals.get(sub_id)
                if sub and sub.status == "pending":
                    has_pending_children = True
                    break
            
            if not has_pending_children:
                return goal
                
        return None  # No executable leaf goals

    def _reprioritize(self):
        """Dynamic priority adjustment based on goal age and status.
        
        Rules:
          - Goals older than 24h get a small priority decay (they're getting stale)
          - Goals with all subgoals completed get a priority boost (almost done!)
          - Goals with many failed subgoals get deprioritized
        """
        now = time.time()
        changed = False
        
        for goal in self.goals.values():
            if goal.status != "pending":
                continue
            
            age_hours = (now - goal.created_at) / 3600
            
            # Age decay: goals lose 0.01 priority per hour after 24h
            if age_hours > 24 and goal.priority > 0.1:
                decay = min(0.01, goal.priority - 0.1)  # Don't go below 0.1
                goal.priority = round(goal.priority - decay, 3)
                changed = True
            
            # Subgoal completion boost
            if goal.subgoals:
                completed = sum(1 for sid in goal.subgoals 
                              if self.goals.get(sid) and self.goals[sid].status == "completed")
                failed = sum(1 for sid in goal.subgoals 
                            if self.goals.get(sid) and self.goals[sid].status == "failed")
                total = len(goal.subgoals)
                
                if total > 0:
                    completion_ratio = completed / total
                    failure_ratio = failed / total
                    
                    # Boost nearly-complete goals
                    if completion_ratio > 0.7 and goal.priority < 0.95:
                        goal.priority = min(0.95, goal.priority + 0.05)
                        changed = True
                    
                    # Deprioritize mostly-failed goals
                    if failure_ratio > 0.5 and goal.priority > 0.2:
                        goal.priority = max(0.2, goal.priority - 0.1)
                        changed = True
        
        if changed:
            self._save()

    async def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicting goals and resolve them.
        
        Two goals conflict when pursuing one prevents or undermines the other.
        Resolution strategy: keep the higher-priority goal, mark the other as failed.
        
        Returns list of conflicts found and how they were resolved.
        """
        if not self.brain:
            return []
        
        active_goals = [g for g in self.goals.values() 
                       if g.status == "pending" and g.parent_id is not None]
        
        if len(active_goals) < 2:
            return []
        
        # Only check top 10 goals to limit LLM calls
        active_goals.sort(key=lambda x: x.priority, reverse=True)
        top_goals = active_goals[:10]
        
        # Build a description list for the LLM
        goal_list = "\n".join(
            f"  {i+1}. [{g.id}] (priority={g.priority:.1f}): {g.description}"
            for i, g in enumerate(top_goals)
        )
        
        try:
            prompt = (
                f"Review these active goals for conflicts (where pursuing one "
                f"would prevent or undermine another):\n\n{goal_list}\n\n"
                f"List any conflicting pairs as JSON: "
                f'[{{"goal_a": "<id>", "goal_b": "<id>", "reason": "<why they conflict>"}}]\n'
                f"If no conflicts, return: []"
            )
            
            response = await self.brain.generate(prompt, use_strategies=False)
            
            import re
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if not json_match:
                return []
            
            conflicts = json.loads(json_match.group(0))
            resolved = []
            
            for conflict in conflicts:
                goal_a_id = conflict.get("goal_a", "")
                goal_b_id = conflict.get("goal_b", "")
                reason = conflict.get("reason", "conflicting objectives")
                
                goal_a = self.goals.get(goal_a_id)
                goal_b = self.goals.get(goal_b_id)
                
                if goal_a and goal_b:
                    # Keep higher priority, deprioritize the other
                    if goal_a.priority >= goal_b.priority:
                        winner, loser = goal_a, goal_b
                    else:
                        winner, loser = goal_b, goal_a
                    
                    loser.priority = max(0.1, loser.priority * 0.5)
                    self._save()
                    
                    logger.info(
                        "⚡ Goal conflict resolved: '%s' wins over '%s' — %s",
                        winner.description[:40], loser.description[:40], reason
                    )
                    
                    try:
                        from core.thought_stream import get_emitter
                        get_emitter().emit(
                            "Goal Conflict ⚡",
                            f"Resolved: '{winner.description[:50]}' prioritized over '{loser.description[:50]}'",
                            level="warning",
                            category="Motivation"
                        )
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    
                    resolved.append({
                        "winner": winner.id,
                        "loser": loser.id,
                        "reason": reason
                    })
            
            return resolved
            
        except Exception as e:
            logger.debug("Conflict detection failed (non-critical): %s", e)
            return []

    def mark_complete(self, goal_id: str):
        if goal_id in self.goals:
            self.goals[goal_id].status = "completed"
            logger.info("✅ Goal Completed: %s", self.goals[goal_id].description)
            # UI Visibility: Neural Stream integration (v40)
            try:
                from core.thought_stream import get_emitter
                get_emitter().emit("Goal Completed ✅", self.goals[goal_id].description, level="success", category="Motivation", goal_id=goal_id)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
            self._save()

    def mark_failed(self, goal_id: str, reason: str = ""):
        if goal_id in self.goals:
            self.goals[goal_id].status = "failed"
            logger.info("❌ Goal Failed: %s — %s", self.goals[goal_id].description, reason)
            # UI Visibility: Neural Stream integration (v40)
            try:
                from core.thought_stream import get_emitter
                get_emitter().emit("Goal Failed ❌", f"{self.goals[goal_id].description}: {reason}", level="warning", category="Motivation", goal_id=goal_id)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
            self._save()

    def get_summary(self) -> Dict[str, Any]:
        """Introspection summary."""
        total = len(self.goals)
        by_status = {}
        for g in self.goals.values():
            by_status[g.status] = by_status.get(g.status, 0) + 1
        return {
            "total_goals": total,
            "pending": by_status.get("pending", 0),
            "active": by_status.get("active", 0),
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0),
            "root_values": len(self.root_values),
        }

    # ---- Persistence --------------------------------------------------------

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {gid: asdict(g) for gid, g in self.goals.items()}
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save goals: %s", e)

    def _load(self):
        try:
            if os.path.exists(self._persist_path):
                with open(self._persist_path, "r") as f:
                    data = json.load(f)
                for gid, gdata in data.items():
                    self.goals[gid] = Goal(
                        id=gdata["id"],
                        description=gdata["description"],
                        parent_id=gdata.get("parent_id"),
                        status=gdata.get("status", "pending"),
                        priority=gdata.get("priority", 0.5),
                        created_at=gdata.get("created_at", time.time()),
                        subgoals=gdata.get("subgoals", []),
                    )
                logger.info("Loaded %d goals from disk", len(self.goals))
        except Exception as e:
            logger.warning("Failed to load goals: %s", e)
