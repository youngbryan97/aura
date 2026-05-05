"""core/brain/reflex_engine.py — Autonomic Reflex Engine (System 1).

Provides fast-path, low-latency action mapping for predictable environmental states.
Bypasses the heavy LLM cognitive loop for routine tasks like navigation and basic combat.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("Aura.Brain.ReflexEngine")

class ReflexEngine:
    """The 'Spinal Cord' of Aura. Handles System 1 reflexes."""
    
    def __init__(self):
        self._action_priority = 0.8
        self._last_pos: Optional[Tuple[int, int]] = None
        self._stuck_count = 0
        self._target_path: List[str] = []
        
    def decide(self, state: Dict[str, Any]) -> Optional[str]:
        """Decide on a reflex action based on structured environment state."""
        
        # 1. Prompt Handling (High Priority)
        # Check raw text for common prompts
        raw_text = state.get("raw_text", "")
        if "--More--" in raw_text:
            return " "
        if "(y/n)" in raw_text.lower():
            return "n"
        if "Shall I pick up" in raw_text:
            return "n"
            
        vitals = state.get("vitals", {})
        if vitals.get("hp_percent", 1.0) < 0.2:
            return "pray" # Panic reflex
            
        # Stuck Detection
        curr_pos = state.get("player_pos", (0, 0))
        if curr_pos == self._last_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
            self._last_pos = curr_pos
            
        if self._stuck_count > 5:
            self._stuck_count = 0
            self._target_path = [] # Clear path if stuck
            return random.choice(["move_n", "move_s", "move_e", "move_w"])

        # 2. Combat Reflexes
        local_monsters = state.get("local_monsters", [])
        if local_monsters:
            for m in local_monsters:
                dist = m.get("distance", 999)
                if dist <= 1.5:
                    return f"attack_{m['direction']}"
                    
        # 3. Path Execution
        if self._target_path:
            return f"move_{self._target_path.pop(0)}"
            
        # 4. Exploration (Atlas-Aware)
        from core.memory.spatial_atlas import get_spatial_atlas
        atlas = get_spatial_atlas()
        dlvl = vitals.get("dlvl", 1)
        
        # Priority: Stairs down
        stairs_down = atlas.find_nearest("stairs_down", dlvl, curr_pos[0], curr_pos[1])
        if stairs_down:
            path = atlas.get_path(dlvl, curr_pos, (stairs_down[1], stairs_down[2]))
            if path:
                self._target_path = path
                return f"move_{self._target_path.pop(0)}"
            elif curr_pos == (stairs_down[1], stairs_down[2]):
                return "descend"

        # Search for frontiers (unexplored areas)
        frontier = atlas.find_frontier(dlvl, curr_pos[0], curr_pos[1])
        if frontier:
            path = atlas.get_path(dlvl, curr_pos, frontier)
            if path:
                self._target_path = path
                return f"move_{self._target_path.pop(0)}"
            
        # Fallback: Random walk with momentum
        directions = ["move_n", "move_s", "move_e", "move_w", "move_ne", "move_nw", "move_se", "move_sw"]
        return random.choice(directions)

_INSTANCE: Optional[ReflexEngine] = None

def get_reflex_engine() -> ReflexEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ReflexEngine()
    return _INSTANCE
