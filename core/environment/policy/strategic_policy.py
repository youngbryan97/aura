"""Strategic policy for long-horizon planning.

Now integrates with HTN planner for:
- Emergency override conditions
- Active task-driven candidate prioritization  
- Do-not-progress gates
- Phase-specific objective selection
"""
from __future__ import annotations

from core.environment.command import ActionIntent
from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.homeostasis import Homeostasis
from core.environment.strategy.htn_planner import HTNPlanner

from .tactical_policy import TacticalPolicy


class StrategicPolicy:
    """Manages high-level goals and delegates to the tactical policy."""

    def __init__(self):
        self.tactical = TacticalPolicy()
        self.planner = HTNPlanner()

    def select_action(
        self,
        parsed_state: ParsedState,
        belief: EnvironmentBeliefGraph,
        homeostasis: Homeostasis,
        recent_frames: list | None = None,
    ) -> ActionIntent | None:
        """Selects an action based on strategic goals and tactical viability."""
        # 1. Check emergency overrides first (e.g., critical HP)
        emergency = self.planner.check_emergencies(parsed_state, belief)
        if emergency:
            return ActionIntent(
                name=emergency.name,
                risk="caution",
                expected_effect="emergency_resolved",
                tags={"emergency"},
            )

        # 2. Update the strategic plan based on new beliefs
        self.planner.update(parsed_state, belief)

        # 3. Get active tasks from HTN
        active_tasks = self.planner.get_active_tasks()

        # 4. If HTN has an active task, bias tactical selection toward it
        task_hint = None
        if active_tasks:
            top_task = active_tasks[0]
            task_hint = top_task.name  # e.g., "stabilize_health", "explore_frontier"

        # 5. Delegate to tactical with strategic hint
        return self.tactical.select_action(
            parsed_state, belief, homeostasis,
            recent_frames=recent_frames,
            strategic_hint=task_hint,
        )

__all__ = ["StrategicPolicy"]

