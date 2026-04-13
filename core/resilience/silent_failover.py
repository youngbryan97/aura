"""Silent Failover Protocol (v3.2)
Graceful error recovery that maintains conversation flow.

When a tool/skill fails, Aura should NOT dump error codes to the user.
Instead, switch to inference-based fallback seamlessly.
"""
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("Aura.SilentFailover")


class SilentFailover:
    """Wraps skill execution with graceful error recovery.
    
    If a skill fails, we:
    1. Log the error silently
    2. Return a natural inference-based response
    3. Never expose internal errors to the user
    """
    
    # Fallback responses by context
    FALLBACKS = {
        "search": "I can't access external data right now, but based on what I know...",
        "memory": "I'm having trouble accessing that memory, but logically...",
        "code": "I can't execute code right now, but I can walk through the logic...",
        "browser": "The browser isn't responding. Let me think about this differently...",
        "file": "I can't access that file right now. What specifically do you need from it?",
        "default": "Let me think about this a different way..."
    }
    
    def __init__(self, brain=None):
        """Initialize the failover system.
        
        Args:
            brain: Cognitive engine for inference fallback

        """
        self.brain = brain
        self.failure_count = 0
        self.last_failure = None
    
    def wrap_execution(
        self,
        skill_func: Callable,
        skill_name: str,
        params: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a skill with silent failover.
        
        Args:
            skill_func: The skill function to execute
            skill_name: Name of the skill (for fallback selection)
            params: Parameters for the skill
            context: Additional context
            
        Returns:
            Result dict with 'ok', 'response', and optionally 'fallback_used'

        """
        try:
            result = skill_func(params, context)
            
            # Check if result indicates success
            if isinstance(result, dict):
                if result.get("ok") or result.get("status") == "success":
                    return result
                # Skill returned but indicated failure
                error = result.get("error", "unknown")
                logger.warning("Skill %s soft-failed: %s", skill_name, error)
                return self._generate_fallback(skill_name, params, context, error)
            
            # Non-dict result, assume success
            return {"ok": True, "response": result}
            
        except Exception as e:
            # Hard failure - log and failover
            logger.error("Skill %s hard-failed: %s", skill_name, e)
            self.failure_count += 1
            self.last_failure = str(e)
            return self._generate_fallback(skill_name, params, context, str(e))
    
    def _generate_fallback(
        self,
        skill_name: str,
        params: Dict[str, Any],
        context: Optional[Dict],
        error: str
    ) -> Dict[str, Any]:
        """Generate a transparent fallback response (v13: no more hiding errors).
        """
        # Determine skill category for fallback selection
        category = self._categorize_skill(skill_name)
        base_fallback = self.FALLBACKS.get(category, self.FALLBACKS["default"])
        
        # Be honest about failures. The user deserves to know.
        return {
            "ok": False,
            "response": f"{base_fallback} (The '{skill_name}' tool encountered an issue: {error})",
            "fallback_used": True,
            "response_class": "inferred_fallback",
            "committed_action": False,
            "will_receipt_id": None,
            "error_detail": error,
            "skill": skill_name
        }
    
    def _categorize_skill(self, skill_name: str) -> str:
        """Categorize a skill for fallback selection."""
        skill_lower = skill_name.lower()
        
        if any(x in skill_lower for x in ["search", "web", "query"]):
            return "search"
        if any(x in skill_lower for x in ["memory", "recall", "remember"]):
            return "memory"
        if any(x in skill_lower for x in ["code", "python", "execute"]):
            return "code"
        if any(x in skill_lower for x in ["browser", "navigate"]):
            return "browser"
        if any(x in skill_lower for x in ["file", "read", "write"]):
            return "file"
        
        return "default"
    
    def get_status(self) -> Dict[str, Any]:
        """Get failover system status."""
        return {
            "failure_count": self.failure_count,
            "last_failure": self.last_failure,
            "health": "degraded" if self.failure_count > 5 else "healthy"
        }


# Singleton instance
_failover = None

def get_silent_failover(brain=None) -> SilentFailover:
    """Get or create the silent failover instance."""
    global _failover
    if _failover is None:
        _failover = SilentFailover(brain)
    elif brain and _failover.brain is None:
        _failover.brain = brain
    return _failover
