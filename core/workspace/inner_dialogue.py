"""core/workspace/inner_dialogue.py
Generates internal dialogue thoughts during ticks.
"""
from typing import Dict, Any
import time


class InnerDialogueGenerator:
    """Compiles internal monologue strings representing active reasoning steps."""

    def generate_monologue(self, state: Any) -> str:
        attention = state.cognition.active_attention
        welfare_idx = state.welfare.welfare_index
        
        # Build logical narrative structure
        monologue = (
            f"[{time.strftime('%H:%M:%S')}] Attention focus is set to '{attention}'. "
            f"Unified welfare is {welfare_idx:.2f}. "
            f"Evaluating plan constraints..."
        )
        return monologue
