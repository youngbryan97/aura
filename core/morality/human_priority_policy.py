"""core/morality/human_priority_policy.py
Enforces absolute prioritization of human instructions over agentic directives.
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger("Morality.HumanPriorityPolicy")


class HumanPriorityPolicy:
    """Prioritizes operator instructions over background/autonomous loop tasks."""

    def prioritize_agenda(self, active_goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Orders goals such that human-requested tasks are always placed first."""
        logger.info("HumanPriorityPolicy ordering active goal priority queue...")
        # Sort goals: items marked user_requested first
        user_tasks = [g for g in active_goals if g.get("source") == "user"]
        bg_tasks = [g for g in active_goals if g.get("source") != "user"]
        return user_tasks + bg_tasks
