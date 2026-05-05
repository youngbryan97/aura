"""Central policy orchestrator that drives the full decision pipeline.

This is the canonical entry point for autonomous action selection:
CandidateGenerator → TacticalSimulator → ActionRanker → StrategicPolicy/HTN → intent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.environment.command import ActionIntent
from core.environment.policy.candidate_generator import CandidateGenerator
from core.environment.policy.action_ranker import ActionRanker
from core.environment.policy.strategic_policy import StrategicPolicy
from core.environment.policy.tactical_policy import TacticalPolicy
from core.environment.simulation import TacticalSimulator
from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.homeostasis import Homeostasis
from core.environment.episode_manager import EpisodeManager


class PolicyOrchestrator:
    """Orchestrates the multi-tier policy stack to choose the best intent."""

    def __init__(self):
        self.candidate_generator = CandidateGenerator()
        self.action_ranker = ActionRanker()
        self.strategic_policy = StrategicPolicy()
        self.tactical_policy = TacticalPolicy()
        self.simulator = TacticalSimulator()

    def select_action(
        self,
        *,
        parsed_state: ParsedState,
        belief: EnvironmentBeliefGraph,
        homeostasis: Homeostasis,
        episode: EpisodeManager | None,
        recent_frames: list[Any],
        **kwargs
    ) -> ActionIntent:
        """Determines the best action to take given the current cognitive state.
        
        Flow:
        1. Strategic policy checks emergencies and HTN goals
        2. If strategic returns an intent (emergency/goal-driven), use it
        3. Otherwise, generate candidates → simulate → rank → select
        """
        # 1. Let strategic policy check for emergencies and HTN-driven actions
        strategic_intent = self.strategic_policy.select_action(
            parsed_state, belief, homeostasis, recent_frames=recent_frames,
        )
        if strategic_intent:
            return strategic_intent

        # 2. Full candidate pipeline as fallback
        candidates = self.candidate_generator.generate(
            parsed_state, belief=belief, recent_frames=recent_frames,
        )

        # 3. Extract homeostasis resources
        resources = homeostasis.assess(homeostasis.extract(parsed_state))

        # 4. Simulate outcomes for candidates
        simulations = {}
        for candidate in candidates:
            sim = self.simulator.simulate(belief, candidate)
            simulations[candidate.intent_id()] = sim

        # 5. Rank candidates
        ranked = self.action_ranker.rank(
            candidates=candidates,
            simulations=simulations,
            resources=resources,
            parsed_state=parsed_state,
        )

        if ranked:
            return ranked[0][0]

        # Fallback
        return ActionIntent(name="observe", expected_effect="observe_more", risk="safe")

