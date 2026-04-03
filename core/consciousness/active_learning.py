"""core/cognition/active_learning.py
Strategic knowledge acquisition system.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AGI.ActiveLearning")

@dataclass
class LearningGoal:
    """A specific thing to learn"""

    topic: str
    motivation: str
    priority: float
    knowledge_gap: str
    learning_strategy: str
    created_at: float
    
    def to_dict(self):
        return asdict(self)


class ActiveLearningEngine:
    """Strategic knowledge acquisition system.
    """
    
    def __init__(self, cognitive_engine):
        self.brain = cognitive_engine
        self.learning_goals: List[LearningGoal] = []
        logger.info("ActiveLearningEngine initialized")
    
    def identify_learning_need(self, task_failed: str, reason: str) -> LearningGoal:
        logger.info("Identifying learning need from failure: %s", task_failed)
        goal = LearningGoal(
            topic=task_failed,
            motivation=f"Failed task: {task_failed}",
            priority=0.8,
            knowledge_gap=reason,
            learning_strategy="research",
            created_at=time.time()
        )
        self.learning_goals.append(goal)
        return goal
    
    def prioritize_learning_goals(self) -> List[LearningGoal]:
        sorted_goals = sorted(self.learning_goals, key=lambda g: g.priority, reverse=True)
        return sorted_goals[:5]
    
    def what_should_i_ask_about(self, context: Dict[str, Any]) -> List[str]:
        top_goals = self.prioritize_learning_goals()
        if not top_goals: return []
        return [f"I need to learn more about {g.topic} because of {g.knowledge_gap}" for g in top_goals[:2]]