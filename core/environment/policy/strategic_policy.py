"""Strategic policy for long-horizon planning."""
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
    ) -> ActionIntent | None:
        """Selects an action based on strategic goals and tactical viability."""
        # 1. Update the strategic plan based on new beliefs
        self.planner.update(parsed_state, belief)

        # 2. Extract current active sub-goal
        # current_goal = self.planner.get_active_goal()
        
        # 3. For now, rely heavily on tactical policy until HTN is fully integrated
        return self.tactical.select_action(parsed_state, belief, homeostasis)

__all__ = ["StrategicPolicy"]
