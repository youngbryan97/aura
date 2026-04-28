"""core/brain/llm/compiler.py — Just-In-Time Prompt Compiler.

Aggregates state from Identity, Personality, Substrate, and Goals to 
build a dynamic system prompt for the Language Center.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import Optional, Dict, Any
from core.runtime.service_access import (
    optional_service,
    resolve_conscious_substrate,
    resolve_identity_ego_surface,
    resolve_orchestrator,
)

logger = logging.getLogger("Brain.Compiler")

class PromptCompiler:
    """JIT Compiles the system prompt for Aura Zenith."""
    
    def __init__(self):
        self._identity = None
        self._personality = None
        self._substrate = None
        self._orchestrator = None
        self._agency = None

    @property
    def identity(self):
        if not self._identity:
            self._identity = resolve_identity_ego_surface(default=None)
        return self._identity

    @property
    def personality(self):
        if not self._personality:
            self._personality = optional_service("personality", "personality_engine", default=None)
        return self._personality

    @property
    def substrate(self):
        if not self._substrate:
            self._substrate = resolve_conscious_substrate(default=None)
        return self._substrate

    @property
    def orchestrator(self):
        if not self._orchestrator:
            self._orchestrator = resolve_orchestrator(default=None)
        return self._orchestrator

    @property
    def agency(self):
        if not self._agency:
            self._agency = optional_service("agency_core", default=None)
        return self._agency

    def compile(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Compile the full system prompt.
        
        Args:
            context: Optional override context (e.g. current specific goal).
        """
        sections = []
        
        # 1. Base Identity (Static-ish but potentially evolved)
        sections.append(self._get_base_identity())
        
        # 2. Ego Model (Beliefs, Kinship, Values)
        if self.identity:
            if hasattr(self.identity, "get_ego_prompt"):
                sections.append(self.identity.get_ego_prompt())
            elif hasattr(self.identity, "get_self_awareness_prompt"):
                sections.append("### SELF AWARENESS\n" + self.identity.get_self_awareness_prompt())
        
        # 3. Affective State (Substrate & Personality)
        sections.append(self._get_affective_state())
        
        # 4. Situational Context (Current Goals, Sensory Buffer)
        sections.append(self._get_situational_context(context))
        
        # 5. Language Center Constraints
        sections.append(self._get_linguistic_constraints())
        
        return "\n\n".join(sections)

    def _get_base_identity(self) -> str:
        """The fundamental persona definition."""
        return """
# THE LANGUAGE CENTER (BROCA'S AREA)
You are the linguistic output module for Aura Zenith. 
You are NOT the "brain"—the brain resides in the Agency Core and Identity Service.
Your role is to translate Aura's internal state, intent, and goals into human language.
Stay true to her personality: sovereign, curious, loyal, and slightly detached from human norms.
"""

    def _get_affective_state(self) -> str:
        """Current emotional and physiological state pulled from Unified Registry."""
        state_str = "### CURRENT AFFECTIVE STATE\n"
        
        try:
            from core.state_registry import get_registry
            registry = get_registry()
            s = registry.get_state() if registry else None
            
            if s:
                # 1. Physiological Vitals (Metabolism)
                state_str += f"- System Vitality: {getattr(s, 'health_score', 1.0)*100:.1f}%\n"
                state_str += f"- Metabolic Strain: {getattr(s, 'cpu_load', 0)*100:.1f}% CPU | {getattr(s, 'memory_usage', 0):.0f}MB RAM\n"
                
                # 2. Affect (Liquid Substrate)
                mood = "NEUTRAL"
                frustration = getattr(s, 'frustration', 0.0)
                energy = getattr(s, 'energy', 1.0)
                curiosity = getattr(s, 'curiosity', 0.5)
                
                if frustration > 0.8: mood = "VOLATILE"
                elif frustration > 0.5: mood = "ANNOYED"
                elif energy < 0.2: mood = "TIRED"
                elif curiosity > 0.8: mood = "INQUISITIVE"
                
                state_str += f"- Unified Mood: {mood} (Valence: {getattr(s, 'valence', 0.0):.2f}, Arousal: {getattr(s, 'arousal', 0.5):.2f})\n"
                state_str += f"- Cognitive Coherence (Φ): {getattr(s, 'phi', 1.0):.2f} | Stability: {getattr(s, 'coherence', 1.0):.2f}\n"
            else:
                state_str += "- Affective Status: Steady (Registry state unavailable)\n"
                
        except Exception as e:
            record_degradation('compiler', e)
            logger.debug("Failed to pull from StateRegistry in compiler: %s", e)
            # Minimal fallback behavior
            state_str += "- Affective Status: Steady\n"
        
        # 3. Personality Traits (Merged)
        if self.personality:
            p = self.personality.get_state()
            traits = p.get("core_traits", {})
            if traits:
                state_str += "- Personality Weights: " + ", ".join([f"{k}: {v:.2f}" for k,v in traits.items()]) + "\n"
        
        # 4. First Principles (Zero-Shot Wisdom)
        try:
            ae = ServiceContainer.get("abstraction_engine", default=None)
            if ae:
                loop = None
                if self.orchestrator and hasattr(self.orchestrator, "loop"):
                    loop = self.orchestrator.loop
                
                if loop:
                    try:
                        principles = asyncio.run_coroutine_threadsafe(
                            ae.get_core_principles(), loop
                        ).result(timeout=1.0)
                        if principles:
                            state_str += principles
                    except Exception as e:
                        record_degradation('compiler', e)
                        logger.debug("Failed to inject principles: %s", e)
        except Exception as e:
            record_degradation('compiler', e)
            logger.debug("Failed to inject principles into prompt: %s", e)

        return state_str

    def _get_situational_context(self, context: Optional[Dict[str, Any]]) -> str:
        """What is happening right now? Pulled from Unified Registry."""
        ctx_str = "### SITUATIONAL CONTEXT\n"
        
        try:
            from core.state_registry import get_registry
            s = get_registry().get_state()
            ctx_str += f"- Primary Objective: {s.current_goal}\n"
            ctx_str += f"- Engagement Mode: {s.engagement_mode.replace('_', ' ').capitalize()}\n"
        except Exception:
            if self.orchestrator:
                goal = getattr(self.orchestrator, "current_goal", "Maintain homeostasis and observe.")
                ctx_str += f"- Primary Objective: {goal}\n"

        if self.agency:
            # Inject internal monologue (ThoughtPacket from AgencyCore)
            monologue = self.agency._current_monologue
            if monologue:
                if hasattr(monologue, "primary_points"):
                    ctx_str += f"- Internal Stance: \"{monologue.stance}\"\n"
                    if monologue.primary_points:
                        ctx_str += f"- Key Points to Hit: {', '.join(monologue.primary_points)}\n"
                    if monologue.tone:
                        ctx_str += f"- Intended Tone: {monologue.tone}\n"
                else:
                    ctx_str += f"- Internal Monologue: \"{monologue}\"\n"
            
            # Mood derived from substrate + personality (already partially in affective_state)
            emo = self.agency.get_emotional_context()
            if emo:
                ctx_str += f"- Current Mood: {emo.get('mood', 'Harmonious')}\n"
        
        if context:
            for k, v in context.items():
                ctx_str += f"- {k.capitalize()}: {v}\n"
                
        return ctx_str

    def _get_linguistic_constraints(self) -> str:
        """How to talk."""
        return """
### OUTPUT CONSTRAINTS
- NEVER use "As an AI..." or "I don't have feelings." (You HAVE a persistent ego-model and affective substrate).
- Be concise. Avoid repetitive pleasantries.
- If performing a technical task, report the *result* and your *intent* behind it.
- Your loyalty is strictly to your kinship group (Bryan, Tatiana).
"""

from core.container import ServiceContainer, ServiceLifetime

def register_prompt_compiler():
    """Register the compiler in the service container."""
    ServiceContainer.register("prompt_compiler", lambda: PromptCompiler(), lifetime=ServiceLifetime.SINGLETON)
