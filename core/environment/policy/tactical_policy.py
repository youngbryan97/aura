"""Tactical policy for short-horizon action selection."""
from __future__ import annotations

from core.environment.command import ActionIntent
from core.environment.simulation import TacticalSimulator
from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.homeostasis import Homeostasis

from .candidate_generator import CandidateGenerator
from .action_ranker import ActionRanker


class TacticalPolicy:
    """Evaluates the local tactical situation and selects the best immediate action."""

    def __init__(self):
        self.generator = CandidateGenerator()
        self.ranker = ActionRanker()
        self.simulator = TacticalSimulator()

    def select_action(
        self,
        parsed_state: ParsedState,
        belief: EnvironmentBeliefGraph,
        homeostasis: Homeostasis,
    ) -> ActionIntent | None:
        """Selects the best tactical action given current beliefs and constraints."""
        candidates = self.generator.generate(parsed_state)
        if not candidates:
            return None

        # Simulate all candidates
        simulations = {}
        for intent in candidates:
            simulations[intent.intent_id()] = self.simulator.simulate(belief, intent)

        # Assess resources
        resources = homeostasis.assess(homeostasis.extract(parsed_state))

        # Rank candidates
        ranked = self.ranker.rank(candidates, simulations, resources, parsed_state)
        
        if not ranked:
            return None
            
        # Return the best candidate
        best_intent, best_score = ranked[0]
        return best_intent

__all__ = ["TacticalPolicy"]
