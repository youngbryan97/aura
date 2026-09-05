"""core/agency/initiative_selector.py
Initiative selector initiating autonomous background tasks based on boredom/drives.
"""
from typing import List, Dict, Any


class InitiativeSelector:
    """Proposes tasks to run when the agent is idle."""

    def select_initiative(self, boredom: float, active_goals: List[Any]) -> List[Dict[str, Any]]:
        """Proposes background research or maintenance tasks."""
        initiatives = []
        
        # High boredom triggers a curiosity/research initiative
        if boredom > 60.0 and not active_goals:
            initiatives.append({
                "channel": "gesture",
                "params": {"gesture": "trigger_curiosity_exploration"}
            })
            
        return initiatives
