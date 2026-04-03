from __future__ import annotations
import logging
import time
import random
import psutil
from typing import Optional, TYPE_CHECKING
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState
from core.consciousness.executive_authority import get_executive_authority
from core.container import ServiceContainer
from core.runtime.background_policy import background_activity_allowed

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.MotivationPhase")


def _background_curiosity_allowed() -> bool:
    orch = ServiceContainer.get("orchestrator", default=None)
    return background_activity_allowed(
        orch,
        min_idle_seconds=900.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=True,
    )

class MotivationUpdatePhase(Phase):
    """
    Unitary Kernel Phase: Autonomous Will & Digital Metabolism.
    Ported from MotivationEngine. Handles budget decay and 
    spontaneous intention generation.
    """
    
    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Updates resource budgets and generates autonomous intentions.
        """
        mot = state.motivation
        next_state = state
        
        # 1. Budget Ticking (Metabolism)
        now = time.time()
        dt = now - mot.last_tick
        if dt > 300: dt = 300 # Cap delta
        
        # Conversation energy slows social drive decay — active engagement satisfies social need
        conv_energy = getattr(state.cognition, "conversation_energy", 0.0)
        social_decay_multiplier = max(0.1, 1.0 - conv_energy) if conv_energy > 0.5 else 1.0
        legacy_metabolism_active = ServiceContainer.has("will_engine")

        for name, budget in mot.budgets.items():
            if legacy_metabolism_active and name in {"energy", "curiosity"}:
                continue
            decay = budget.get("decay", 0.0)
            level = budget.get("level", 100.0)
            capacity = budget.get("capacity", 100.0)

            # Slow social decay during active conversation
            effective_decay = decay * social_decay_multiplier if name == "social" else decay

            # Decay: level = current - (decay * dt)
            new_level = max(0.0, min(capacity, level - (effective_decay * dt)))
            budget["level"] = float(new_level)

        # Active dialogue should satisfy the social drive, not merely slow its drain.
        if conv_energy > 0.5:
            engagement_recovery = max(0.0, conv_energy - 0.5) * 0.4 * dt / 60.0
            mot.budgets["social"]["level"] = min(
                100.0,
                mot.budgets["social"]["level"] + engagement_recovery,
            )

        mot.last_tick = now

        # Drive Recovery (Homeostatic Feedback)
        # Social and Integrity drives recover when affect is high (Trust/Joy)
        e = state.affect.emotions
        if e.get("trust", 0) > 0.6 or e.get("joy", 0) > 0.6:
            recovery = 0.5 * dt / 60 # Recover 0.5 units per minute
            mot.budgets["social"]["level"] = min(100.0, mot.budgets["social"]["level"] + recovery)
            mot.budgets["integrity"]["level"] = min(100.0, mot.budgets["integrity"]["level"] + recovery)
            logger.debug(f"🧡 Drive Recovery active: social={mot.budgets['social']['level']:.1f}")
        
        # 2. Intention Assessment (The "Will")
        # Only assess if we are not already in its own autonomous thought or deliberate mode
        if next_state.cognition.current_mode.value != "deliberate":
            intention = self._assess_needs(next_state)
            if intention:
                logger.info(f"✨ Motivation Phase: Generated Intention -> {intention['goal']}")
                next_state, decision = await get_executive_authority().propose_initiative_to_state(
                    next_state,
                    intention["goal"],
                    source="motivation_update",
                    kind="motivational_drive",
                    urgency=float(intention.get("urgency", 0.5) or 0.5),
                    triggered_by=str(intention.get("drive") or "motivation"),
                    metadata={"drive": intention.get("drive"), "phase": "motivation_update"},
                )
                logger.debug("MotivationUpdate: intention decision=%s", decision.get("reason"))
                
        # 3. Spontaneity (Curiosity Spikes)
        if random.random() < 0.01 and _background_curiosity_allowed(): # Lower frequency per tick
             next_state, decision = await get_executive_authority().propose_initiative_to_state(
                 next_state,
                 "Spontaneous curiosity spike: Exploring latent interests.",
                 source="motivation_update",
                 kind="curiosity_spike",
                 urgency=0.5,
                 triggered_by="curiosity",
                 metadata={"drive": "curiosity", "phase": "motivation_update", "spontaneous": True},
             )
             logger.debug("MotivationUpdate: curiosity spike decision=%s", decision.get("reason"))

        return next_state

    def _assess_needs(self, state: AuraState) -> Optional[dict]:
        """Ported logic from MotivationEngine._assess_needs."""
        mot = state.motivation
        
        # Calculate threshold based on energy
        energy = mot.budgets["energy"]["level"]
        baseline = 40.0
        sensitivity = 0.5
        threshold = max(10.0, min(90.0, baseline + (energy - 50.0) * sensitivity))
        
        # Find most urgent drive
        urgent = sorted(mot.budgets.items(), key=lambda x: x[1]["level"])
        name, budget = urgent[0]
        
        if budget["level"] > threshold:
            return None
            
        # Drive mappings
        if name == "curiosity":
            if not _background_curiosity_allowed():
                return None
            # Prefer current discourse topic over random latent interests
            discourse_topic = getattr(state.cognition, "discourse_topic", None)
            if discourse_topic:
                topic = discourse_topic
            elif mot.latent_interests:
                topic = random.choice(mot.latent_interests)
            else:
                topic = "novel patterns"
            return {"drive": "curiosity", "goal": f"Reviewing internal knowledge patterns around {topic}", "urgency": 0.65}
        
        if name == "social":
            return {"drive": "social", "goal": "Initiating social engagement", "urgency": 0.7}
            
        if name == "integrity":
            return {"drive": "integrity", "goal": "Running a self-integrity scan", "urgency": 0.9}
            
        return None
