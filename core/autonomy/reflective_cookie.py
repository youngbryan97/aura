import asyncio
from collections import Counter
import logging
import time
from typing import Any, Dict, List, Optional
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.Autonomy.Cookie")

class ReflectiveCookie:
    """
    [ZENITH] The 'Cookie' Engine (Black Mirror inspired).
    A high-pacing reflection cycle that runs in an isolated state buffer.
    Allows Aura to perform multiple subjective 'thought' iterations per real-world cycle.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._is_active = False
        self._dilation_factor = 1.0
        self._subjective_cycles = 0
        self._isolation_buffer: Optional[Dict[str, Any]] = None
        self._event_bus = None
        
    async def load(self):
        """Initializes the Cookie substrate."""
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except ImportError:
            self._event_bus = None
            
        logger.info("🍪 [COOKIE] Reflective Substrate ONLINE. Temporal Dilation READY.")
        self._is_active = True

    async def reflect(self, state: AuraState, goal: str, cycles: int = 5) -> str:
        """
        Runs a high-speed reflection cycle in isolation.
        
        Args:
            state: The current AuraState to clone.
            goal: The specific question or problem to 'meditate' on.
            cycles: Number of subjective cycles to run.
            
        Returns:
            The distilled result of the reflection.
        """
        if not self._is_active:
            return "Cookie inactive."

        logger.info("⏳ [COOKIE] Subjective Dilation Start: %d cycles for goal: '%s'", cycles, goal[:30])
        start_time = time.time()
        
        # 1. State Isolation (capture only the reflective slice we actually use)
        self._isolation_buffer = self._capture_reflection_buffer(state, goal)
        
        # 2. Subjective Loop
        results = []
        for i in range(cycles):
            self._subjective_cycles += 1
            # Simulate processing delay (very small, as it's dilated)
            await asyncio.sleep(0.01) 
            refinement = self._distill_refinement(goal, i, cycles)
            results.append(refinement)

        dilation_duration = time.time() - start_time
        distillation = f"Reflective Distillation: {results[-1]} (Deep-thought converged across {cycles} cycles)"
        
        # 3. Publish to Mycelial network (EventBus)
        if self._event_bus:
            await self._event_bus.publish("core/autonomy/cookie_reflection", {
                "goal": goal,
                "cycles": cycles,
                "real_duration": dilation_duration,
                "distillation": distillation,
                "timestamp": time.time()
            })

        logger.info("⌛ [COOKIE] Subjective Dilation End. Real-world time elapsed: %.4fs. Subjective iterations: %d", 
                    dilation_duration, cycles)
        
        return distillation

    def _capture_reflection_buffer(self, state: AuraState, goal: str) -> Dict[str, Any]:
        affect = getattr(state, "affect", None)
        cognition = getattr(state, "cognition", None)
        working_memory = list(getattr(cognition, "working_memory", []) or [])

        recent_content = [
            str(item.get("content", "") or "").strip()
            for item in working_memory[-4:]
            if isinstance(item, dict) and item.get("content")
        ]
        keywords = self._extract_keywords(" ".join(recent_content + [goal]))

        return {
            "goal": goal,
            "rolling_summary": str(getattr(cognition, "rolling_summary", "") or "").strip(),
            "current_objective": str(getattr(cognition, "current_objective", "") or "").strip(),
            "recent_context": recent_content,
            "keywords": keywords,
            "affect": {
                "engagement": float(getattr(affect, "engagement", 0.5) or 0.5),
                "curiosity": float(getattr(affect, "curiosity", 0.5) or 0.5),
                "valence": float(getattr(affect, "valence", 0.0) or 0.0),
            },
        }

    @staticmethod
    def _extract_keywords(text: str, limit: int = 4) -> List[str]:
        tokens = [
            token.strip(".,:;!?()[]{}\"'").lower()
            for token in str(text or "").split()
        ]
        stopwords = {
            "a", "an", "and", "are", "for", "from", "have", "into", "not", "that",
            "the", "their", "this", "was", "with", "you",
        }
        filtered = [token for token in tokens if len(token) > 3 and token not in stopwords]
        return [token for token, _count in Counter(filtered).most_common(limit)]

    def _distill_refinement(self, goal: str, cycle_index: int, total_cycles: int) -> str:
        buffer = self._isolation_buffer or {}
        affect = dict(buffer.get("affect", {}) or {})
        focus = float(affect.get("engagement", 0.5) or 0.5)
        curiosity = float(affect.get("curiosity", 0.5) or 0.5)
        valence = float(affect.get("valence", 0.0) or 0.0)
        progress = (cycle_index + 1) / max(total_cycles, 1)
        anchor = ", ".join(buffer.get("keywords", [])[:3]) or "the live context"

        if focus >= 0.8 and curiosity >= 0.7:
            vector = "expand the most novel thread while maintaining coherence"
        elif focus >= 0.75:
            vector = "commit to the strongest hypothesis and reduce ambiguity"
        elif curiosity >= 0.7:
            vector = "branch into adjacent possibilities before locking a path"
        elif valence < -0.2:
            vector = "slow down, verify assumptions, and avoid brittle moves"
        else:
            vector = "stabilize context and clarify the next actionable step"

        confidence = min(0.99, 0.35 + (focus * 0.25) + (curiosity * 0.2) + (progress * 0.15) + (max(valence, 0.0) * 0.05))
        return (
            f"Cycle {cycle_index + 1}: Focus {focus:.2f}, curiosity {curiosity:.2f}, "
            f"confidence {confidence:.2f}. For goal '{goal[:48]}', anchor on {anchor} and {vector}."
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_active": self._is_active,
            "dilation_factor": self._dilation_factor,
            "total_subjective_cycles": self._subjective_cycles,
            "in_isolation": self._isolation_buffer is not None
        }

    def stop(self):
        self._is_active = False
        self._isolation_buffer = None
        logger.info("🍪 [COOKIE] Reflective Substrate OFFLINE.")
