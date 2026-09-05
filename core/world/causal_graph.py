"""core/world/causal_graph.py
Models cause-and-effect relationship predictions from actions to outcomes.
"""
from typing import Dict, Any, List


class CausalGraph:
    """Tracks action execution histories to calculate probability of success/failure."""

    def __init__(self):
        # Maps action_channel -> {outcome -> frequency}
        self._frequencies: Dict[str, Dict[str, int]] = {}

    def record_causal_link(self, action: str, outcome: str) -> None:
        if action not in self._frequencies:
            self._frequencies[action] = {}
        
        self._frequencies[action][outcome] = self._frequencies[action].get(outcome, 0) + 1

    def predict_outcome_probability(self, action: str, outcome: str) -> float:
        """Returns estimated success/failure probability."""
        if action not in self._frequencies or not self._frequencies[action]:
            return 0.5
        
        outcomes = self._frequencies[action]
        total = sum(outcomes.values())
        return outcomes.get(outcome, 0) / total
