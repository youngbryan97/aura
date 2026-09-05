"""core/executive/attention_controller.py
Attention Controller directing focal focus of cognitive ticks.
"""
from core.organism.life_state import LifeState


class AttentionController:
    """Calculates active attention tags based on current loop priority."""

    async def focus_attention(self, state: LifeState) -> str:
        """Determines if focus is on sleep, user, active goal, or idle curiosity."""
        if state.body.is_sleeping:
            return "sleep_consolidation"
            
        if state.cognition.current_goals:
            # Focus on highest priority goal
            return f"goal_focus:{state.cognition.current_goals[0].get('id')}"
            
        # Default fallback is environment observation
        return "ambient_perception"
