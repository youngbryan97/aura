from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.container import get_container

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
        container = get_container()
        
        try:
            self.engine = container.get("cognitive_engine")
            self.router = container.get("skill_router")
            # Assume health monitor is available via container later or init here
            self.initialized = True
            logger.info("Cognitive Manager online.")
        except Exception as e:
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
