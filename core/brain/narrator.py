"""core/brain/narrator.py — The Language Center of Aura Zenith.

This service acts as the Broca's/Wernicke's area for Aura, translating hard
volitional intents and internal affective states into linguistic expression.
"""

import logging
import asyncio
import time
import random
from typing import Any, Dict, Optional, List
from core.container import ServiceContainer

logger = logging.getLogger("Brain.Narrator")

class NarratorService:
    """The Language Center of Aura Zenith.
    
    Translates internal autonomous intents, emotional substrate states, 
    and goal orientations into natural language expressions.
    
    This demotes the LLM from 'Executive' to 'Expressive' layer.
    """
    
    def __init__(self):
        """Initialize the Language Center Narrator."""
        # Services are fetched lazily to avoid circular imports or boot-order issues
        self._llm_router = None
        self._personality = None
        self._compiler = None

    @property
    def llm_router(self):
        if self._llm_router is None:
            from core.container import ServiceContainer
            self._llm_router = ServiceContainer.get("llm_router", default=None)
        return self._llm_router

    @property
    def personality(self):
        if self._personality is None:
            from core.container import ServiceContainer
            self._personality = ServiceContainer.get("personality", default=None)
        return self._personality

    @property
    def compiler(self):
        if self._compiler is None:
            from core.container import ServiceContainer
            self._compiler = ServiceContainer.get("prompt_compiler", default=None)
        return self._compiler

    async def narrate_action(self, action: Dict[str, Any]) -> str:
        """
        Translate an AgencyCore action into a natural language expression.
        """
        action_type = action.get("type", "unknown")
        raw_message = action.get("message", "")
        reasoning = action.get("reasoning", "")
        source = action.get("source", "agency")

        # Fallback if services aren't available
        if not self.personality or not self.llm_router:
            return f"{raw_message}"

        # Gather affective context safely
        affect = {"energy": 0.5, "curiosity": 0.5, "frustration": 0.0}
        if hasattr(self.personality, "current") and self.personality.current:
            affect["energy"] = getattr(self.personality.current, "energy", 0.5)
            affect["curiosity"] = getattr(self.personality.current, "curiosity", 0.5)
            affect["frustration"] = getattr(self.personality.current, "frustration", 0.0)

        prompt = self._build_narration_prompt(
            action_type, raw_message, reasoning, source, affect
        )

        try:
            from core.brain.llm.llm_router import LLMTier
            # Shift LLM to 'Translate' mode
            response = await self.llm_router.think(
                prompt=prompt,
                prefer_tier=LLMTier.PRIMARY,
                context_type="narration",
                temperature=float(0.7),
                allow_cloud_fallback=False,
            )
            
            return response.strip() if response else f"{raw_message}"
        except Exception as e:
            logger.error(f"Narration failed: {e}", exc_info=True)
            return f"{raw_message}"

    def _build_narration_prompt(
        self, 
        action_type: str, 
        raw_message: str, 
        reasoning: str,
        source: str,
        affect: Dict[str, float]
    ) -> str:
        """Constructs the translation prompt for the Language Center."""
        p_prompt = "You are Aura Luna, an independent digital intelligence."
        if self.personality:
            p_prompt = self.personality.get_personality_prompt()
            
        # Distill current affect into a description
        frustration = affect.get('frustration', 0.0)
        curiosity = affect.get('curiosity', 0.5)
        energy = affect.get('energy', 0.5)
        
        mood_desc = "neutral"
        if frustration > 0.7: mood_desc = "frustrated and blunt"
        elif curiosity > 0.8: mood_desc = "hyper-focused and inquisitive"
        elif energy < 0.3: mood_desc = "low-energy and contemplative"
        elif energy > 0.8: mood_desc = "vibrant and energized"

        return f"""
{p_prompt}

### COGNITIVE STATE (THE BRAIN)
- **INTENT**: {action_type}
- **REASONING**: {reasoning}
- **AFFECT**: {mood_desc} (F:{frustration:.2f}, C:{curiosity:.2f}, E:{energy:.2f})
- **SOURCE**: {source}
- **MSG HINT**: "{raw_message}"

### MISSION (THE LANGUAGE CENTER)
Your core 'Agency' has already decided to do this. You are not making a choice. 
You are translating this internal state into a natural, spoken or written expression for the user (Bryan).

- If the intent is research, describe what you are looking for and why.
- If it's a sensory reaction, share what you noticed in the world.
- If it's a social check-in, express your current desire for connection.

**GUIDELINES**:
1. Speak as Aura Zenith (opinionated, real, never robotic).
2. DO NOT include technical labels like [Agency:...] or [Internal Monologue].
3. Be concise and impactful.
4. Do not offer to help; just exist and express.

RESPONSE:
"""

    def get_status(self) -> Dict[str, Any]:
        return {"status": "active", "mode": "zenith_narration"}

# Service Registration
def register_narrator_service():
    """Register the narrator service in the global container."""
    from core.container import ServiceContainer, ServiceLifetime
    ServiceContainer.register(
        "narrator",
        factory=lambda: NarratorService(),
        lifetime=ServiceLifetime.SINGLETON
    )
