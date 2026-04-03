"""Persona Evolver (Phase 8)
Analyzes interaction memory and adapts personality baselines subtly over time.
"""
import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from core.config import config
from core.container import ServiceContainer, ServiceLifetime
from core.brain.personality_engine import get_personality_engine

logger = logging.getLogger("Aura.PersonaEvolver")

class PersonaEvolver:
    """Adapts Aura's personality over long-term interactions."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.last_evolution_time = time.time()
        self.evolution_interval = 3600 * 24  # Evaluate daily or when explicitly triggered
        self.min_memories_for_evolution = 10
        
    async def update_persona(self, reflection: str):
        """Allows for manual or event-driven persona updates based on reflection."""
        # v34 Hardening: Validate reflection before self-mod
        if not reflection or len(reflection) < 20:
            logger.debug("Reflection too short for persona evolution.")
            return
            
        try:
            logger.info("🧬 Performing manual persona evolution from reflection...")
            # Atomic update logic
            # For now, we reuse the existing internal logic but conditioned on the new reflection
            # In a full v34 implementation, this would call a LLM to generate the delta
            await self.run_evolution_cycle(force=True, custom_reflection=reflection)
        except Exception as e:
            logger.error(f"Persona evolution failed: {e}")

    async def run_evolution_cycle(self, force: bool = False, custom_reflection: str = None):
        """Analyze memory and apply drift to personality if needed."""
        now = time.time()
        if not force and (now - self.last_evolution_time < self.evolution_interval):
            return
            
        personality = get_personality_engine()
        memories = getattr(personality, "interaction_memories", [])
        
        if not custom_reflection and len(memories) < self.min_memories_for_evolution:
            logger.debug("PersonaEvolver: Not enough new interaction memories to evolve.")
            return
            
        logger.info("🧬 Initiating Persona Evolution Cycle...")
        
        # Format memories for the LLM
        memory_text = custom_reflection or ""
        if not custom_reflection:
            for m in memories:
                msg = m.get("message", "")
                sent = m.get("sentiment", "neutral")
                memory_text += f"- [{sent}] {msg}\n"
            
        prompt = f"""You are analyzing Aura's recent interactions to adjust her personality.
Based on the following interactions, how should her core traits and emotional baselines drift?
Output JSON ONLY with no markdown formatting.
Small fractional changes (-0.05 to +0.05 for traits, -5.0 to +5.0 for emotion base/volatility).
Only include fields that should change. Example format:
{{
  "traits": {{"agreeableness": 0.02, "extraversion": -0.01}},
  "emotions": {{"frustration": {{"base": -2.0, "volatility": 0.1}}}}
}}

Recent Interactions:
{memory_text}
"""
        try:
            from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode
            if not hasattr(self.orchestrator, 'cognitive_engine'):
                logger.warning("No cognitive engine available for evolution.")
                return
                
            engine = self.orchestrator.cognitive_engine
            
            # Using deep thinking for self-reflection
            response = await asyncio.wait_for(
                engine.think(prompt, mode=ThinkingMode.DEEP, block_user=True),
                timeout=120.0
            )
            
            import re
            content = response.content.strip()
            
            # Robust JSON extraction
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                logger.warning("PersonaEvolver: Could not find JSON block in response.")
                return
                
            json_str = match.group(0)
            changes = json.loads(json_str)
            
            if changes:
                self._apply_evolution(changes, personality)
                # Clear memories after evolution to prevent over-weighting
                personality.interaction_memories = []
                self.last_evolution_time = time.time()
                
        except Exception as e:
            logger.error("PersonaEvolver cycle failed: %s", e)
            
    def _apply_evolution(self, changes: Dict[str, Any], personality):
        """Merge changes into evolved_persona.json and reload."""
        evolved_path = config.paths.data_dir / "evolved_persona.json"
        
        evolved_data = {"traits": {}, "emotions": {}}
        if evolved_path.exists():
            try:
                with open(evolved_path, "r") as f:
                    evolved_data = json.load(f)
            except json.JSONDecodeError:
                import logging
                logger.debug("Exception caught during execution", exc_info=True)
                
        # Merge traits
        new_traits = changes.get("traits", {})
        for t, val in new_traits.items():
            current = evolved_data["traits"].get(t, personality.traits.get(t, 0.5))
            evolved_data["traits"][t] = max(0.0, min(1.0, current + val))
            
        # Merge emotions
        new_emotions = changes.get("emotions", {})
        for e, data in new_emotions.items():
            if e not in evolved_data["emotions"]:
                evolved_data["emotions"][e] = {}
                
            if e in personality.emotions:
                current_state = personality.emotions[e]
                if "base" in data:
                    c_base = evolved_data["emotions"][e].get("base", current_state.base_level)
                    evolved_data["emotions"][e]["base"] = max(0.0, min(100.0, c_base + data["base"]))
                if "volatility" in data:
                    c_vol = evolved_data["emotions"][e].get("volatility", current_state.volatility)
                    evolved_data["emotions"][e]["volatility"] = max(0.1, c_vol + data["volatility"])
                    
        # Save and trigger reload
        with open(evolved_path, "w") as f:
            json.dump(evolved_data, f, indent=2)
            
        logger.info("🧬 Persona evolved based on interaction sentiment.")
        personality.reload_persona()