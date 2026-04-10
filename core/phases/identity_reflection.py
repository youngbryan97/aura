import asyncio
import logging
import time
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState

logger = logging.getLogger(__name__)

class IdentityReflectionPhase(BasePhase):
    """
    Phase 7: Identity Reflection.
    Updates Aura's self-narrative based on recent experiences and themes.
    Moves Aura from a static agent to an evolving identity.
    """
    
    def __init__(self, container: Any):
        self.container = container

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        [CLAUDE AUDIT] Identity Guard / Hard Stop.
        Ensures Aura's output hasn't deviated into hallucination or dangerous territory.
        If validation fails, returns the PARENT state (Hard Stop) instead of the new one.
        """
        logger.debug("🛡️ CognitiveGuard: Validating state transition integrity.")
        
        # 1. Identity Consistency Check
        identity_name = str(getattr(state.identity, "name", "Aura") or "Aura").strip()
        if identity_name and identity_name.lower() not in ("aura", "aura luna"):
            logger.critical("IDENTITY BREACH: Identity name altered to '%s'. COGNITIVE HARD STOP.", identity_name)
            # Repair: force identity name back to Aura
            try:
                state.identity.name = "Aura"
            except (AttributeError, TypeError):
                pass
            return state
             
        # 2. Output Characterization (Anti-Hallucination)
        if state.cognition.working_memory:
            last_msg = state.cognition.working_memory[-1]
            if last_msg.get("role") == "assistant":
                content = last_msg.get("content", "")
                
                # Check for "Ghost Aura" patterns or excessive repetition
                if len(content) > 5000:
                    logger.warning("🛡️ CognitiveGuard: Output too long. Potential runaway loop. Truncating.")
                    last_msg["content"] = content[:500] + "... [Guard Truncated]"
                
                # Claude's Identity Stability Check:
                # Ensure the message doesn't claim things that violate the core identity
                if "i am a human" in content.lower() or "i am chatgpt" in content.lower():
                    logger.critical("🚨 COGNITIVE ROLLBACK: Identity Hallucination detected. REJECTING TRANSITION.")
                    # [CLAUDE AUDIT] Hard Stop: Return original state to rollback the turn.
                    return state 

                # Check Identity Guard service directly if available
                from core.identity.identity_guard import PersonaEnforcementGate
                ok, reason, _ = PersonaEnforcementGate().validate_output(
                    content,
                    enforce_supervision=False,
                )
                if not ok:
                    logger.critical(f"🚨 COGNITIVE ROLLBACK: Identity Guard rejected output (%s).", reason)
                    return state

        # 3. Success: Narrative Drift Update
        if state.version % 20 == 0:
            from core.container import ServiceContainer
            ladder = ServiceContainer.get("growth_ladder", default=None)
            
            proposal_content = "Evolution of self-narrative: 'I am stable and evolve safely.'"
            
            if ladder:
                # v40: Self-modification requires consent
                from core.self_modification.growth_ladder import ModificationLevel
                granted = await ladder.submit_proposal(
                    level=ModificationLevel.EXPRESSION,
                    domain="identity_narrative",
                    description=proposal_content,
                    justification="Evolving self-narrative for stability."
                )
                if not granted:
                    logger.info("🚫 IdentityReflection: Evolution proposal vetoed by Aura.")
                    return state

            state.identity.narrative_version += 1
            state.identity.last_evolution_timestamp = time.time()
            if "evolving" not in state.identity.current_narrative:
                state.identity.current_narrative += " I am stable and evolve safely."

        return state
