from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict, List

from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.CognitiveManager")

class CognitiveManager:
    """Manages the lifecycle and health of the cognitive engine.
    Extracted from RobustOrchestrator to reduce complexity.
    """

    def __init__(self):
        self.engine = None
        self.router = None
        self.health_monitor = None
        self.initialized = False

    async def on_start_async(self):
        """Initialize the cognitive infrastructure."""
        logger.info("Initializing Cognitive Manager...")
        
        try:
            self.engine = get_runtime_service("cognitive_engine", default=None)
            self.router = get_runtime_service("skill_router", default=None)
            if self.engine is None or self.router is None:
                raise RuntimeError("cognitive_engine or skill_router unavailable")
            # Assume health monitor is available via container later or init here
            self.initialized = True
            logger.info("Cognitive Manager online.")
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('cognitive_manager', e)
            logger.error("Cognitive initialization failed: %s", e)
            raise

    async def generate_autonomous_thought(self, clean_msg: str, history: List[Dict[str, Any]]) -> Any:
        """Handle internal cognitive impulses (boredom, curiosity, reflection).
        """
        if not self.initialized:
            raise RuntimeError("CognitiveManager not initialized")
            
        from core.brain.cognitive_engine import ThinkingMode
        try:
            from core.thought_stream import get_emitter
        except ImportError:
            # Fallback for headless environments
            class MockEmitter:
                def emit(self, *args, **kwargs): 
                    logger.debug("MockEmitter: %s %s", args, kwargs)
            get_emitter = lambda: MockEmitter()

        get_emitter().emit("Thought 💭", f"Thinking about: {clean_msg}", level="info")
        
        context = {
            "role": "system",
            "mode": "autonomous_reflection",
            "recent_history": history[-3:]
        }
        
        thought = await self.engine.think(
            objective=f"Internal Reflection: {clean_msg}\n\nAnalyze this impulse. If it requires external action (search, etc.), formulate a plan. If it's a realization, record it.",
            context=context,
            mode=ThinkingMode.DEEP
        )
        
        logger.info("🧠 Autonomous Thought: %s...", thought.content[:100])
        get_emitter().emit("Reflection 🧠", thought.content[:200], level="info")
        return thought

    def get_status(self) -> Dict[str, Any]:
        """Return the health and status of the cognitive core."""
        return {
            "initialized": self.initialized,
            "engine_status": "active" if self.engine else "inactive",
            "router_ready": self.router is not None
        }
