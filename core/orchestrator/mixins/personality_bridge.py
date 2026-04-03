"""Personality Bridge Mixin for RobustOrchestrator.
Extracts identity resolution and formatting logic.
"""
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _dispose_awaitable(result: Any) -> None:
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()

class PersonalityBridgeMixin:
    """Handles parsing and formatting of the Aura Persona and User Identity."""

    def _get_personality_data(self) -> dict:
        """Get raw personality metrics as a dictionary."""
        try:
            from core.brain.personality_engine import get_personality_engine
            pe = get_personality_engine()
            update_result = pe.update() # Refresh states
            if inspect.isawaitable(update_result):
                _dispose_awaitable(update_result)
            res = pe.get_emotional_context_for_response()
            if inspect.isawaitable(res):
                _dispose_awaitable(res)
                return {"mood": "neutral", "tone": "balanced", "emotional_state": {}}
            if not isinstance(res, dict):
                # Handle unexpected or non-dict returns
                return {"mood": "neutral", "tone": "balanced", "emotional_state": {}}
            return res
        except Exception as e:
            logger.warning("Personality metric retrieval failed: %s", e)
            return {"mood": "neutral", "tone": "snarky", "emotional_state": {}}

    def _stringify_personality(self, ctx: dict) -> str:
        """Convert personality dict to HUD-compatible string."""
        mood = ctx.get("mood", "neutral").upper()
        tone = ctx.get("tone", "balanced")
        emotions = ", ".join([f"{n}: {i:.0f}" for n, i in ctx.get("emotional_state", {}).items() if i > 65])
        return f"MOOD: {mood} | TONE: {tone} | INTENSE EMOTIONS: {emotions or 'none'}"

    def _get_personality_context(self) -> str:
        """Legacy wrapper for personality string."""
        data = self._get_personality_data()
        return self._stringify_personality(data)

    def _detect_user_identity(self, message: str) -> dict[str, Any]:
        """Determine who is talking to Aura."""
        msg = message.lower()
        if any(x in msg for x in ["i'm bryan", "im bryan", "it's bryan", "its bryan", "this is bryan"]):
            return {"name": "Bryan", "role": "Architect", "relation": "Kin"}
        if any(x in msg for x in ["i'm tatiana", "im tatiana", "it's tatiana", "its tatiana"]):
            return {"name": "Tatiana", "role": "Core Kin", "relation": "Kin"}
        
        # Default to previous session context if available, otherwise "Stranger"
        return {"name": "Stranger", "role": "Unknown", "relation": "Neutral"}
