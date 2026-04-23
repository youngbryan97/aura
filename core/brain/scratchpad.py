"""Scratchpad Engine for Aura.

Provides a deliberate 'inner monologue' where Aura can plan, critique, 
and refine her reasoning before generating a final response.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.base_module import AuraBaseModule
from core.container import ServiceContainer

class ScratchpadEngine(AuraBaseModule):
    """Engine for recursive 'System 2' thinking."""
    
    def __init__(self, cognitive_engine: Any = None):
        """Initializes the ScratchpadEngine.
        
        Args:
            cognitive_engine: Reference to the LLM-based brain.
        """
        super().__init__("Scratchpad")
        self.cognitive_engine = cognitive_engine
        self.logger.info("✓ Scratchpad Engine Online (System 2 Strategy Active)")

    async def think_recursive(self, objective: str, context: Dict[str, Any], 
                                depth: int = 1) -> str:
        """Performs a multi-step inner monologue."""
        @self.error_boundary
        async def _think_wrapped():
            if not self.cognitive_engine:
                self.cognitive_engine = ServiceContainer.get("cognitive_engine", default=None)
            
            if not self.cognitive_engine:
                return "Cognitive engine unavailable for scratchpad."

            self.logger.info("🧠 Scratchpad objective: %s...", objective[:50])
            
            # 1. Initial Plan
            plan_prompt = (
                f"Draft a step-by-step reasoning plan for the objective: '{objective}'.\n"
                f"Consider recent context: {context.get('history', [])[-2:]}\n"
                f"Focus on safety, efficiency, and personality consistency."
            )
            
            from core.brain.cognitive_engine import ThinkingMode
            plan_mode = ThinkingMode.DEEP if depth > 1 else ThinkingMode.SLOW
            
            # Step 1: Draft
            draft = await self.cognitive_engine.think(
                objective=plan_prompt,
                context=context,
                mode=plan_mode
            )
            inner_monologue = f"[Plan] {draft.content}"
            
            # Step 2: Critique & Refine (Recursive)
            for i in range(depth):
                critique_prompt = (
                    f"Critique this plan: '{inner_monologue}'.\n"
                    f"Identify gaps, risks, or missed tool opportunities.\n"
                    f"Then provide the refined 'Final Internal Strategy'."
                )
                refinement = await self.cognitive_engine.think(
                    objective=critique_prompt,
                    context=context,
                    mode=ThinkingMode.REFLECTIVE
                )
                inner_monologue = refinement.content
                self.logger.debug("Scratchpad Refinement %d complete.", i+1)
                
            return inner_monologue
            
        result = await _think_wrapped()
        return result if isinstance(result, str) else f"Thinking failure: {result}"

    def get_health(self) -> Dict[str, Any]:
        """Provide health status of the scratchpad."""
        return {
            **super().get_health(),
            "has_brain": self.cognitive_engine is not None
        }
