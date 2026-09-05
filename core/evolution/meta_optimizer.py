import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("MetaCognition.MetaOptimizer")

class MetaOptimizer:
    """Analyzes historical performance to optimize future strategies and prompts.
    Implements Meta-Learning / Self-Improvement.
    """
    
    def __init__(self, history_manager: Any):
        self.history = history_manager  # Usually access to memory_nexus or log files
        self.strategy_scores: Dict[str, List[float]] = {}

    async def evaluate_strategies(self) -> None:
        """Fetch recent memory from history manager and evaluate strategy scores."""
        recent_memory: List[Dict[str, Any]] = await self.history.get_recent(limit=100)
        for entry in recent_memory:
            # Look for entries with action + outcome
            action = entry.get("action")
            success = entry.get("success", False)

            if not action:
                continue

            score = 1.0 if success else 0.0

            if action not in self.strategy_scores:
                self.strategy_scores[action] = []

            self.strategy_scores[action].append(score)

        logger.info("Evaluated %d entries. Strategy scores updated.", len(recent_memory))

    def get_best_strategy(self, available_strategies: List[str]) -> str:
        """Returns the strategy with the highest mean score."""
        if not available_strategies:
            return "default"
        if not self.strategy_scores:
            return available_strategies[0]

        means = {
            s: float(np.mean(self.strategy_scores[s]))
            for s in available_strategies
            if s in self.strategy_scores and self.strategy_scores[s]
        }

        if not means:
            return available_strategies[0]

        return max(means, key=means.__getitem__)

    def optimize_prompt(self, task_type: str, current_prompt: str) -> str:
        """Heuristic-based prompt optimization based on failure patterns."""
        scores_for_task = self.strategy_scores.get(task_type, [])
        if not scores_for_task:
            # Fall back to global average only if no task-specific data
            all_scores = [v for sub in self.strategy_scores.values() for v in sub]
            avg_score = float(np.mean(all_scores)) if all_scores else 1.0
        else:
            avg_score = float(np.mean(scores_for_task))

        optimized = current_prompt
        if avg_score < 0.4:
            optimized += "\n[ADAPTATION: High failure rate detected. Use step-by-step reasoning.]"

        return optimized

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "top_strategies": {
                k: float(np.mean(v)) for k, v in self.strategy_scores.items() if v
            },
            "history_size": sum(len(v) for v in self.strategy_scores.values()),
        }