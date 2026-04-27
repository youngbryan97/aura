"""Deep Theory of Mind contract.

The current IntersubjectivityEngine is heuristic. This module adds a
discrete user-mental-state model that can be updated, queried, and
ablated. The audit's required behaviors:

  - false-belief test: Aura knows fact A; user does not
  - perspective divergence: Aura predicts user's interpretation
  - trust update: user corrects Aura -> belief model updates

The contract here is small enough to be exercised by tests but real
enough to drive explanation strategy.
"""
from __future__ import annotations


import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class UserBeliefState:
    beliefs: Dict[str, Any] = field(default_factory=dict)
    knowledge_gaps: Set[str] = field(default_factory=set)


@dataclass
class UserGoalState:
    active_goals: List[str] = field(default_factory=list)
    finished_goals: List[str] = field(default_factory=list)


@dataclass
class TrustState:
    trust: float = 0.5
    recent_corrections: int = 0
    last_correction_at: Optional[float] = None


class FalseBeliefSimulator:
    """Lightweight false-belief tracker.

    Aura records facts she knows and the user's confirmed beliefs. When
    she's about to make an explanation, she checks whether her fact
    contradicts the user's belief — if so, she explicitly corrects rather
    than overwriting.
    """

    def __init__(self):
        self._aura_facts: Dict[str, Any] = {}
        self._user_beliefs: Dict[str, Any] = {}

    def aura_knows(self, key: str, value: Any) -> None:
        self._aura_facts[key] = value

    def user_believes(self, key: str, value: Any) -> None:
        self._user_beliefs[key] = value

    def divergence(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._aura_facts:
            return None
        if key not in self._user_beliefs:
            # User has no belief yet — they don't know A.
            return {"key": key, "kind": "user_knowledge_gap", "aura_fact": self._aura_facts[key]}
        if self._aura_facts[key] != self._user_beliefs[key]:
            return {
                "key": key,
                "kind": "false_belief",
                "aura_fact": self._aura_facts[key],
                "user_belief": self._user_beliefs[key],
            }
        return None


class TheoryOfMindEngine:
    def __init__(self):
        self.belief = UserBeliefState()
        self.goals = UserGoalState()
        self.trust = TrustState()
        self.simulator = FalseBeliefSimulator()
        self.history: List[Dict[str, Any]] = []

    def observe_user_message(self, *, text: str, declared_belief: Optional[Dict[str, Any]] = None) -> None:
        if declared_belief:
            for k, v in declared_belief.items():
                self.belief.beliefs[k] = v
                self.simulator.user_believes(k, v)
        self.history.append({"text": text, "at": time.time()})

    def record_correction(self, *, key: str, correct_value: Any) -> None:
        self.belief.beliefs[key] = correct_value
        self.simulator.user_believes(key, correct_value)
        self.trust.recent_corrections += 1
        self.trust.last_correction_at = time.time()
        # Tiny trust adjustment.
        self.trust.trust = max(0.0, self.trust.trust - 0.05)

    def explanation_strategy(self, key: str) -> str:
        div = self.simulator.divergence(key)
        if div is None:
            return "share_directly"
        if div["kind"] == "user_knowledge_gap":
            return "explain_from_first_principles"
        return "respectfully_correct_false_belief"
