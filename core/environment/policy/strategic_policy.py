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
            if self._emergency_action_repeatedly_failed(emergency.name, parsed_state.context_id, recent_frames):
                return ActionIntent(
                    name="retreat_to_safety",
                    risk="caution",
                    expected_effect="emergency_repositioned",
                    tags={"emergency", "recovery"},
                )
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

    @staticmethod
    def _emergency_action_repeatedly_failed(action_name: str, context_id: str | None, recent_frames: list | None) -> bool:
        if not recent_frames:
            return False
        failures = 0
        for frame in recent_frames[-5:]:
            if not getattr(frame, "action_intent", None) or not getattr(frame, "outcome_assessment", None):
                continue
            if frame.action_intent.name != action_name:
                continue
            parsed = getattr(frame, "post_parsed_state", None) or getattr(frame, "parsed_state", None)
            if context_id and parsed is not None and getattr(parsed, "context_id", None) != context_id:
                continue
            if frame.outcome_assessment.success_score < 0.3:
                failures += 1
        return failures >= 2

__all__ = ["StrategicPolicy"]
