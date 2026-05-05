"""core/planning/hierarchical_planner.py -- Hierarchical Task Network Planner
=============================================================================
Aura's higher-order planning layer.  Moves beyond atomic impulses by
decomposing high-level goals into a sequence of executable subgoals.

This module provides the "Planning Horizon" missing from the basic
reactive loop.  It maintains a GoalStack and uses the WorldState +
Neural Mesh readouts to determine when a subgoal is complete.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Planning")

@dataclass
class Subgoal:
    """A single executable step in a plan."""
    description: str
    id: str
    priority: float = 0.5
    status: str = "pending"  # pending, active, completed, failed
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class HierarchicalPlan:
    """A collection of subgoals targeting a high-level objective."""
    objective: str
    id: str
    stack: List[Subgoal] = field(default_factory=list)
    priority: float = 0.5
    horizon: str = "short"  # short, medium, long
    created_at: float = field(default_factory=time.time)

class HierarchicalPlanner:
    """Manages multi-step plans and provides subgoals to the Synthesizer."""

    def __init__(self) -> None:
        self._active_plans: List[HierarchicalPlan] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("hierarchical_planner", self, required=False)
        self._started = True
        logger.info("HierarchicalPlanner ONLINE -- multi-step reasoning active")

    def create_plan(self, objective: str, subgoals: List[str], priority: float = 0.5) -> str:
        """Create a new hierarchical plan from a list of strings."""
        plan_id = f"plan_{int(time.time())}_{len(self._active_plans)}"
        plan = HierarchicalPlan(
            objective=objective,
            id=plan_id,
            priority=priority,
            stack=[
                Subgoal(description=s, id=f"{plan_id}_sub_{i}", priority=priority)
                for i, s in enumerate(subgoals)
            ]
        )
        self._active_plans.append(plan)
        logger.info("Created plan [%s]: %s (%d subgoals)", plan_id, objective, len(subgoals))
        return plan_id

    def get_current_subgoal(self) -> Optional[Subgoal]:
        """Return the next actionable subgoal from the highest priority plan."""
        if not self._active_plans:
            return None
        
        # Sort by priority
        self._active_plans.sort(key=lambda p: p.priority, reverse=True)
        
        for plan in self._active_plans:
            for subgoal in plan.stack:
                if subgoal.status in ("pending", "active"):
                    return subgoal
        return None

    def mark_subgoal_complete(self, subgoal_id: str):
        """Mark a subgoal as completed and advance the plan."""
        for plan in self._active_plans:
            for subgoal in plan.stack:
                if subgoal.id == subgoal_id:
                    subgoal.status = "completed"
                    logger.info("Subgoal completed: %s", subgoal.description)
                    self._check_plan_completion(plan)
                    return

    def _check_plan_completion(self, plan: HierarchicalPlan):
        if all(s.status == "completed" for s in plan.stack):
            logger.info("Plan fully completed: %s", plan.objective)
            self._active_plans.remove(plan)

    def update(self, state: Any):
        """Review active plans against current world state.
        
        This is where 'Grounding' happens. If the state indicates the 
        environment has changed such that a subgoal is now impossible 
        or already finished, we adjust.
        """
        world_state = ServiceContainer.get("world_state", default=None)
        if not world_state:
            return

        # Simple heuristic: if we see an error event related to a plan, mark it failed
        events = world_state.get_salient_events(limit=5)
        for event in events:
            desc = event.get("description", "").lower()
            if "failed" in desc or "error" in desc:
                subgoal = self.get_current_subgoal()
                if subgoal and any(word in desc for word in subgoal.description.lower().split()):
                    logger.warning("Current subgoal likely failed due to environment event: %s", desc)
                    subgoal.status = "failed"

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_plans": len(self._active_plans),
            "current_objective": self._active_plans[0].objective if self._active_plans else "None",
            "subgoal_count": sum(len(p.stack) for p in self._active_plans)
        }

# Singleton
_planner_instance: Optional[HierarchicalPlanner] = None

def get_hierarchical_planner() -> HierarchicalPlanner:
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = HierarchicalPlanner()
    return _planner_instance
