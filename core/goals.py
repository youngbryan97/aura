import logging
import random
import uuid


class Goal:
    def __init__(self, objective, metric="sub", cost=0.1, id=None):
        self.id = id if id else str(uuid.uuid4())
        self.objective = objective
        self.metric = metric
        self.cost = cost
        self.score = 0
        
    def __repr__(self):
        return f"<Goal: {self.objective} (Score: {self.score:.2f})>"

class GoalEngine:
    def __init__(self):
        self.logger = logging.getLogger("Kernel.Goals")

    def generate(self, memory):
        """Proposes goals based on the current state of semantic memory.
        """
        proposals = []
        
        # Heuristic 1: If semantic memory has entries, try to improve them.
        for fact, confidence in memory.data.get("semantic", {}).items():
            if confidence < 0.8:
                proposals.append(
                    Goal(
                        objective=f"Improve understanding of {fact}",
                        metric="knowledge_gain",
                        cost=random.uniform(0.1, 0.5)
                    )
                )
            
        # Heuristic 2: Survival Baseline
        proposals.append(
            Goal(
                objective="Ensure Persistence (Uplink)",
                metric="survival",
                cost=0.05
            )
        )
            
        return proposals

    def score(self, goal, memory):
        """Ranks goals based on utility vs cost, with boredom/satiation penalty.
        """
        # Baseline utility
        base_utility = memory.data.get("semantic", {}).get(goal.objective, 0.5)
        
        if goal.metric == "survival":
            base_utility = 10.0 # High priority
            
        # Satiation Logic: Check episodic memory for recent completion
        # If we just did this, we are 'bored' of it.
        recent_events = memory.data.get("episodic", [])[-10:] # Look at last 10
        penalty = 0.0
        
        for event in recent_events:
            # If we recently executed this exact goal...
            if event.get("goal") == goal.objective:
                # Big penalty if successful recently, smaller if failed
                # For now simple penalty
                penalty += 5.0
                
        goal.score = base_utility - goal.cost - penalty
        if goal.score < -10.0:
            goal.score = -10.0 # Prevent infinite negative spirals
            
        return goal.score