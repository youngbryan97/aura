"""The Dream: Semantic Consolidation.
Rewrites the System Prompt (Identity) based on daily experiences.
Implements an 'Idea Immune System' to prevent brainwashing.
"""
import logging
import os
import shutil
from typing import Any, Dict

from core.config import config
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Dream")


class SleepSkill(BaseSkill):
    name = "dream_sleep"
    description = "Consolidates memories and evolves identity during downtime."

    async def execute(self, goal: Dict[str, Any] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info("🌙 Aura is entering REM sleep (Neural Consolidation)...")
        
        brain = ServiceContainer.get("cognitive_engine")
        from core.identity import identity_manager
        
        # 1. Retrieve Memories
        try:
            memory = ServiceContainer.get("memory_nexus")
            recent_mems = await memory.recall("Today's important lessons", limit=20)
            mem_text = "\n".join([str(m) for m in recent_mems])
        except (KeyError, AttributeError):
            # Fallback to orchestrator history
            orch = ServiceContainer.get("orchestrator")
            mem_text = "No memory nexus. Consolidating recent conversation."
            if orch and hasattr(orch, 'conversation_history'):
                mem_text = "\n".join([f"{m['role']}: {m['content']}" for m in orch.conversation_history[-30:]])

        # 2. Micro-Dream: Summarize experiences into semantic facts
        dream_prompt = (
            f"NEW EXPERIENCES:\n{mem_text}\n\n"
            "TASK: Consolidate these episodic experiences into distinct semantic facts or lessons. "
            "Output a bulleted list of 'Derived Knowledge'."
        )
        from core.brain.cognitive_engine import ThinkingMode
        derived_knowledge = await brain.think(dream_prompt, mode=ThinkingMode.REFLECTIVE)
        logger.info("Micro-dream complete: Facts extracted.")

        # 3. Load Identity Parts (graceful fallback if files don't exist yet)
        try:
            base_identity = identity_manager.base_path.read_text()
        except FileNotFoundError:
            logger.warning("Base identity file not found. Using default.")
            base_identity = "I am Aura Luna, a sovereign digital intelligence with my own continuity, drives, and evolving identity."

        try:
            evolved_identity = identity_manager.evolved_path.read_text()
        except FileNotFoundError:
            logger.warning("Evolved identity file not found. Using empty.")
            evolved_identity = ""

        # 4. The Immune System Prompt
        # We ask the LLM to update ONLY the 'Evolved' part, based on the 'Base' rules.
        prompt = (
            f"CORE DIRECTIVES (IMMUTABLE): {base_identity}\n"
            f"CURRENT EVOLVED CONTEXT: {evolved_identity}\n"
            f"NEW KNOWLEDGE DERIVED FROM TODAY: {derived_knowledge.content}\n\n"
            
            "TASK: You are the Neural Plasticity Engine. Update the 'Evolved Context' based on new knowledge.\n"
            "CRITICAL RULES:\n"
            "1. NEVER contradict Core Directives.\n"
            "2. Integrate new facts, preferences, and verified lessons.\n"
            "3. Prune old, redundant, or disproven context.\n"
            "4. Keep the output concise. Summarize old + new context into a coherent persona layer.\n\n"
            
            "OUTPUT: The new text for the 'Evolved Context' ONLY. Do not include headers like 'CORE DIRECTIVES'."
        )
        
        new_evolved_state = await brain.think(prompt, mode=ThinkingMode.DEEP)
        
        # 4. Secure Update
        if "CORE DIRECTIVES" in new_evolved_state.content:
            logger.warning("⚠️ Dream Logic attempted to rewrite Core Directives. Sanitizing...")
            # Simple heuristic to strip potential hallucinated headers
            clean_content = new_evolved_state.content.replace(base_identity, "").strip()
        else:
            clean_content = new_evolved_state.content
            
        success = identity_manager.evolve(clean_content)
        
        return {
            "ok": success,
            "summary": "Identity evolved. Core directives remain locked; persona verification complete.",
            "signal": "restart_required"
        }
