"""core/utils/context_assembler.py

Unifies context gathering for LLM prompts. 
Consolidates logic from orchestrator, conversation_loop, and strategic_planner.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.ContextAssembler")

class ContextAssembler:
    """Centralized utility for gathering high-fidelity system context."""

    def __init__(self, orchestrator=None):
        self.orch = orchestrator

    async def gather_full_context(self, objective: str = "", include_memory: bool = True) -> Dict[str, Any]:
        """Gathers a comprehensive snapshot of system state."""
        context = {
            "timestamp": time.time(),
            "objective": objective,
            "internal_state": {},
            "cognitive_modifiers": {},
            "world_model": {},
            "social_model": {},
            "hardware_state": {}
        }

        # 1. Internal State (Affect/Drives)
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis:
                context["internal_state"]["drives"] = homeostasis.get_status()
            
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect:
                context["internal_state"]["affect"] = await affect.get() if hasattr(affect, 'get') else {}
            
            coupling = ServiceContainer.get("homeostatic_coupling", default=None)
            if coupling:
                # HomeostaticCoupling provides prompt injections and modifiers
                context["cognitive_modifiers"] = coupling.get_metadata() if hasattr(coupling, 'get_metadata') else {}
                context["prompt_injection"] = coupling.get_prompt_injection()
        except Exception as e:
            logger.debug("Failed to gather internal state: %s", e)

        # 2. Consciousness State (Global Workspace / Substrate)
        try:
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate:
                context["internal_state"]["substrate"] = await substrate.get_state_summary()
            
            gw = ServiceContainer.get("global_workspace", default=None)
            if gw:
                context["consciousness_snapshot"] = gw.get_snapshot()
        except Exception as e:
            logger.debug("Failed to gather consciousness state: %s", e)

        # 3. World Model & Beliefs
        try:
            bg = ServiceContainer.get("belief_graph", default=None)
            if bg:
                context["world_model"]["beliefs"] = bg.get_beliefs() if hasattr(bg, 'get_beliefs') else {}
        except Exception as e:
            logger.debug("Failed to gather world model: %s", e)

        # 4. Theory of Mind
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom:
                # Assume default_user for now or get from orch
                user_id = "default_user"
                if self.orch and hasattr(self.orch, 'current_user_id'):
                    user_id = self.orch.current_user_id
                
                if hasattr(tom, 'known_selves') and user_id in tom.known_selves:
                    context["social_model"]["user"] = tom.known_selves[user_id].to_dict()
        except Exception as e:
            logger.debug("Failed to gather ToM: %s", e)

        # 5. Strategic Context
        try:
            planner = ServiceContainer.get("strategic_planner", default=None)
            if planner:
                context["strategic_state"] = planner.get_status() if hasattr(planner, 'get_status') else {}
        except Exception as e:
            logger.debug("Failed to gather strategic state: %s", e)

        # 6. Memory (Optional/Large)
        if include_memory and self.orch:
            try:
                # Get memory from orch's manager
                mem = self.orch.memory
                if mem:
                    # Summary of relevant memories based on objective
                    # (This is usually done in the cognitive cycle, but we can provide a hook)
                    pass
            except Exception as e:
                logger.debug("Failed to gather memory: %s", e)

        return context

    def format_as_prompt(self, context: Dict[str, Any]) -> str:
        """Converts the context dict into a prompt-ready string."""
        parts = []
        
        # 1. Prompt Injection (Affective Tone etc.)
        if "prompt_injection" in context:
            parts.append(context["prompt_injection"])
            
        # 2. Internal State Summary
        d = context.get("internal_state", {}).get("drives", {})
        if d:
            drive_str = ", ".join([f"{k}: {v:.2f}" for k, v in d.items()])
            parts.append(f"[DRIVES: {drive_str}]")
            
        # 3. Social Context
        u = context.get("social_model", {}).get("user", {})
        if u:
            parts.append(f"[USER STATE: mood={u.get('emotional_state')}, rapport={u.get('rapport', 0.5):.2f}]")
            
        # 4. Strategic Intent
        s = context.get("strategic_state", {})
        if s.get("active_objective"):
             parts.append(f"[ACTIVE MISSION: {s['active_objective']}]")

        return "\n".join(parts)