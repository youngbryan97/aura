"""core/brain/reflex_engine.py — Autonomic Reflex Engine (System 1).

Provides fast-path, low-latency action mapping for predictable environmental states.
Bypasses the heavy LLM cognitive loop for routine tasks like navigation and basic combat.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("Aura.Brain.ReflexEngine")

class ReflexEngine:
    """The 'Spinal Cord' of Aura. Handles System 1 reflexes."""
    
    def __init__(self):
        # Pre-compiled heuristic mappings or tiny model hooks would go here.
        self._action_priority = 0.8
        
    def decide(self, state: Dict[str, Any]) -> Optional[str]:
        """Decide on a reflex action based on structured environment state."""
        # Example reflex: Move towards stairs if goal is to descend and path is clear.
        # Example reflex: Attack adjacent hostile monster.
        
        vitals = state.get("vitals", {})
        if vitals.get("hp_percent", 1.0) < 0.2:
            return "pray" # Panic reflex
            
        local_monsters = state.get("local_monsters", [])
        if local_monsters:
            # Attack nearest monster if adjacent
            for m in local_monsters:
                dist = m.get("distance", 999)
                if dist <= 1.5: # Adjacent or diagonal
                    return f"attack_{m['direction']}"
                    
        # Basic navigation reflex
        target_path = state.get("target_path", [])
        if target_path and len(target_path) > 0:
            next_step = target_path[0]
            return f"move_{next_step}"
            
        return None # No reflex triggered, pass to System 2

_INSTANCE: Optional[ReflexEngine] = None

def get_reflex_engine() -> ReflexEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ReflexEngine()
    return _INSTANCE
