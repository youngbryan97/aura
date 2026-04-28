"""core/brain/cognitive_context_manager.py
==========================================
Unified Cognitive Context Manager for Aura.
Consolidates all system states into a cohesive narrative for the LLM.
"""

from core.runtime.errors import record_degradation
import logging
from typing import Any

from core.runtime import service_access

logger = logging.getLogger("Aura.ContextManager")


class CognitiveContextManager:
    """The single source of truth for Aura's internal state and context.
    Unifies DynamicContextBuilder and CognitiveIntegrationLayer.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    async def start(self) -> None:
        """Service initialization."""
        logger.info("CognitiveContextManager service started")

    async def build_unified_context(self, message: str) -> dict[str, Any]:
        """Gather all available state data from registered services."""
        context: dict[str, Any] = {}

        try:
            homeostasis = service_access.optional_service("homeostatic_coupling", default=None)
            if homeostasis:
                context["homeostasis"] = homeostasis.get_snapshot()
                context["modifiers"] = homeostasis.get_modifiers()
                context["vitality_score"] = context["homeostasis"].get("overall_vitality", 1.0)

            liquid = service_access.resolve_liquid_substrate(default=None)
            if liquid:
                context["vitality"] = liquid.get_status()
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Vitality/Homeostasis context failed: %s", exc)

        try:
            identity = service_access.optional_service("identity_system", default=None)
            if identity:
                context["identity"] = identity.get_full_system_prompt_injection()
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Identity context failed: %s", exc)

        try:
            personality = service_access.optional_service("personality_engine", default=None)
            if personality:
                from .aura_persona import AURA_BIG_FIVE

                context["personality"] = personality.get_emotional_context_for_response()
                context["ocean_traits"] = AURA_BIG_FIVE
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Personality context failed: %s", exc)

        try:
            memory = service_access.resolve_memory_facade(default=None)
            if memory:
                if hasattr(memory, "retrieve_unified_context"):
                    context["memory_context"] = await memory.retrieve_unified_context(message)
                elif hasattr(memory, "memory") and hasattr(memory.memory, "retrieve"):
                    memories = await memory.memory.retrieve(message, limit=3)
                    context["memory_context"] = "\n".join(f"- {memory_item}" for memory_item in memories)
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Memory context failed: %s", exc)

        try:
            consciousness = service_access.optional_service("consciousness", default=None)
            if consciousness:
                state = consciousness.get_state()
                context["consciousness"] = {
                    "gwt": state.get("workspace"),
                    "temporal": state.get("temporal"),
                    "prediction": state.get("prediction"),
                    "qualia": state.get("qualia"),
                    "phi": state.get("iit_phi"),
                }
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Consciousness context failed: %s", exc)

        try:
            beliefs = service_access.optional_service("belief_graph", default=None)
            if beliefs:
                strong = beliefs.get_strong_beliefs(0.7)
                context["beliefs"] = "\n".join(
                    f"- {belief['source']} {belief['relation']} {belief['target']}"
                    for belief in strong[:5]
                )
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Belief context failed: %s", exc)

        try:
            theory_of_mind = service_access.optional_service("theory_of_mind", default=None)
            if theory_of_mind:
                context["user_intent"] = await theory_of_mind.infer_intent(message, context)
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("ToM context failed: %s", exc)

        return context

    def format_for_prompt(self, context: dict[str, Any]) -> str:
        """Convert the structured context into a compact markdown block for the LLM."""
        segments: list[str] = []

        if "identity" in context:
            segments.append(f"### IDENTITY\n{context['identity']}")

        if context.get("memory_context"):
            segments.append(f"### RELEVANT CONTEXT\n{context['memory_context']}")

        if context.get("user_intent"):
            intent = context["user_intent"]
            segments.append(f"### USER INTENT\nPragmatic: {intent.get('pragmatic', 'standard')}")

        return "\n\n".join(segments)

    async def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Fallback for when the context manager is incorrectly called for inference."""
        logger.warning("CognitiveContextManager.generate called! Routing to primary cognitive_engine.")
        engine = service_access.optional_service("cognitive_engine", default=None)
        if engine and hasattr(engine, "generate"):
            return await engine.generate(prompt, **kwargs)
        return {"text": "ContextManager Error: Generation routing failed.", "error": True}

    async def record_interaction(self, user_input: str, response: str, domain: str = "general") -> None:
        """Record interaction for continuous learning."""
        try:
            learning = service_access.optional_service("learning_engine", default=None)
            if learning and hasattr(learning, "record_interaction"):
                await learning.record_interaction(
                    user_input=user_input,
                    aura_response=response,
                    domain=domain,
                )
        except Exception as exc:
            record_degradation('cognitive_context_manager', exc)
            logger.debug("Automatic learning record failed: %s", exc)

    def get_ui_snapshot(self) -> dict[str, Any]:
        """Provide a lightweight snapshot for UI telemetry."""
        return {
            "vitality": 1.0,
            "mood": "STABLE",
            "curiosity": 0.5,
            "phi": 0.5,
        }
