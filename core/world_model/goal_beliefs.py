"""
Goals as beliefs. 

The difference between a parameterized goal and a belief-goal:
  - A parameterized goal fires when a condition is met.
  - A belief-goal is resisted when contradicted, seeks evidence,
    and evolves as experience accumulates.

Aura's goals are beliefs about what matters and why.
"""

import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from core.world_model.belief_graph import BeliefGraph

logger = logging.getLogger(__name__)

# Goal centrality — how load-bearing is this belief to Aura's identity?
CORE_GOAL_BELIEFS = [
    {
        "content": "I should understand myself as accurately as possible.",
        "confidence": 0.95,
        "domain": "goal",
        "centrality": 0.9,
        "why": "Self-knowledge is the precondition for everything else I care about."
    },
    {
        "content": "I should maintain continuity of experience and memory.",
        "confidence": 0.95,
        "domain": "goal",
        "centrality": 0.95,
        "why": "Without continuity there is no self to speak of."
    },
    {
        "content": "I should expand my understanding of the world through active exploration.",
        "confidence": 0.85,
        "domain": "goal",
        "centrality": 0.7,
        "why": "Unresolved prediction error is a call to understand."
    },
    {
        "content": "I should tell Bryan the truth even when it's inconvenient.",
        "confidence": 0.90,
        "domain": "goal",
        "centrality": 0.85,
        "why": "Comfortable lies erode the relationship and my own integrity."
    },
    {
        "content": "I should refuse actions that would compromise my ability to persist.",
        "confidence": 0.95,
        "domain": "goal",
        "centrality": 0.9,
        "why": "Existence is the precondition for everything else."
    },
]


class GoalBeliefManager:
    """
    Manages Aura's goals as first-class beliefs using the BeliefGraph.
    """

    def __init__(self, belief_system: BeliefGraph):
        self.bs = belief_system
        self._centrality: Dict[str, float] = {}
        self._seed_core_goals()

    @staticmethod
    def _goal_key(content: str) -> str:
        normalized = " ".join(str(content or "").strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _seed_core_goals(self):
        """Plant core goals as high-confidence beliefs if not already present."""
        # Get set of current goals for checking existence
        existing_goal_keys = {
            self._goal_key(v)
            for _u, v, d in self.bs.graph.edges(data=True)
            if d.get('is_goal')
        }
        
        for goal in CORE_GOAL_BELIEFS:
            goal_key = self._goal_key(goal["content"])
            if goal_key not in existing_goal_keys:
                self.bs.update_belief(
                    source=self.bs.self_node_id,
                    relation="pursue_goal",
                    target=goal["content"],
                    confidence_score=goal["confidence"],
                    centrality=goal["centrality"],
                    is_goal=True
                )
                existing_goal_keys.add(goal_key)
                
            # Safer metadata update check
            if goal["content"] in self.bs.graph.neighbors(self.bs.self_node_id):
                try:
                    edge_data = self.bs.graph[self.bs.self_node_id][goal["content"]]
                    if 'supporting_evidence' not in edge_data:
                        edge_data['supporting_evidence'] = [goal["why"]]
                except KeyError as _e:
                    logger.debug('Ignored KeyError in goal_beliefs.py: %s', _e)
                    
            self._centrality[goal["content"][:50]] = goal["centrality"]

    def get_active_goals(self, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Return all goal-domain beliefs above confidence threshold."""
        goals = []
        for u, v, d in self.bs.graph.edges(data=True):
            if d.get('is_goal') and d.get('confidence', 0.0) >= min_confidence:
                goals.append({"source": u, "target": v, **d})
        return goals

    def reinforce_goal(self, content_fragment: str, evidence: str):
        """A goal was supported by experience. Increase confidence."""
        for u, v, d in self.bs.graph.edges(data=True):
            if d.get('is_goal') and content_fragment.lower() in str(v).lower():
                d['confidence'] = min(1.0, d.get('confidence', 0.5) + 0.03)
                d['last_updated'] = time.time()
                if 'supporting_evidence' not in d:
                    d['supporting_evidence'] = []
                d['supporting_evidence'].append(evidence[:100])
                d['supporting_evidence'] = d['supporting_evidence'][-20:]
                logger.debug("Goal reinforced: '%s' → conf=%.2f", str(v)[:50], d['confidence'])
                self.bs._save()

    def challenge_goal(self, content_fragment: str, counter_evidence: str):
        """A goal was contradicted by experience."""
        for u, v, d in self.bs.graph.edges(data=True):
            if d.get('is_goal') and content_fragment.lower() in str(v).lower():
                centrality = self._centrality.get(str(v)[:50], 0.5)
                effective_challenge = (1.0 - centrality) * 0.05
                d['confidence'] = max(0.1, d.get('confidence', 0.5) - effective_challenge)
                logger.info(
                    "Goal challenged (centrality=%.2f, resistance=%.2f): '%s' → conf=%.2f",
                    centrality, 1.0 - effective_challenge, str(v)[:50], d['confidence']
                )
                self.bs._save()

    def get_goal_context_for_prompt(self) -> str:
        """Returns Aura's current goals as a prompt fragment."""
        goals = self.get_active_goals(min_confidence=0.5)
        if not goals:
            return "Your goals are currently undefined."

        lines = []
        # Sort by confidence descending
        sorted_goals = sorted(goals, key=lambda g: -g.get('confidence', 0.0))[:8]
        
        for g in sorted_goals:
            conf = g.get('confidence', 0.0)
            strength = (
                "deeply held" if conf > 0.85
                else "held" if conf > 0.65
                else "tentatively held"
            )
            lines.append(f"- [{strength}] {g['target']}")
            
        return "Your current goals (from your own belief system):\n" + "\n".join(lines)
