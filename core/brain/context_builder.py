import logging
from typing import Any, Dict, Optional

from core.container import get_container

logger = logging.getLogger("Aura.ContextBuilder")


class DynamicContextBuilder:
    """Consolidates system state, user traits, and personality into a rich
    context dictionary for the LLM cognitive loop.
    """

    @staticmethod
    async def build_rich_context(
        message: str, current_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Gather all available state data and format it for the cognitive loop."""
        rich_context = current_context or {}
        container = get_container()

        # 1. Emotional State (LiquidState)
        try:
            liquid_state = container.get("liquid_state")
            if liquid_state:
                rich_context["liquid_state"] = liquid_state.get_status()
        except Exception as e:
            logger.debug("LiquidState unavailable: %s", e)

        # 2. User Traits (PersonalityEngine)
        try:
            personality = container.get("personality_engine")
            if personality:
                rich_context["personality"] = personality.get_emotional_context_for_response()
                try:
                    from .aura_persona import AURA_BIG_FIVE
                    rich_context["ocean_traits"] = AURA_BIG_FIVE
                except ImportError:
                    logger.debug("aura_persona not available for OCEAN traits")
                personality.respond_to_event("user_message", {"message": message})
        except Exception as e:
            logger.debug("PersonalityEngine unavailable: %s", e)

        # 3. Episodic Memory Retrieval
        try:
            conv_engine = container.get("conversation_engine")
            if conv_engine and hasattr(conv_engine, "memory"):
                memories = await conv_engine.memory.retrieve(message, limit=3)
                if memories:
                    # Format the retrieved memories into a single context string
                    rich_context["memory_context"] = "\n".join(
                        f"- {m.get('text', m) if isinstance(m, dict) else m}" for m in memories
                    )
        except Exception as e:
            logger.debug("Episodic memory retrieval failed: %s", e)

        # 4. Semantic Memory Retrieval (Vector Search)
        try:
            semantic_memory = container.get("vector_memory_engine")
            if semantic_memory and hasattr(semantic_memory, "recall_formatted"):
                formatted_memories = await semantic_memory.recall_formatted(message, limit=5)
                if formatted_memories:
                    rich_context["semantic_context"] = formatted_memories
        except Exception as e:
            logger.debug("Semantic memory retrieval failed: %s", e)

        # 5. Theory of Mind (Intent detection)
        try:
            tom = container.get("theory_of_mind")
            if tom:
                rich_context["user_intent"] = await tom.infer_intent(message, rich_context)
        except Exception as e:
            logger.debug("Theory of Mind unavailable: %s", e)

        # 6. Global Workspace Theory — last N competition winners
        # get_context_stream() returns a pre-formatted string; safe to call sync.
        try:
            gws = container.get("global_workspace")
            if gws and hasattr(gws, "get_context_stream"):
                stream = gws.get_context_stream(n=4)
                if stream:
                    rich_context["gwt_stream"] = stream
        except Exception as e:
            logger.debug("GlobalWorkspace stream unavailable: %s", e)

        # 7. Temporal Binding — autobiographical present-window narrative
        # get_narrative() is async (holds a lock); await it directly.
        try:
            tb = container.get("temporal_binding")
            if tb and hasattr(tb, "get_narrative"):
                narrative = await tb.get_narrative()
                if narrative:
                    rich_context["temporal_narrative"] = narrative
        except Exception as e:
            logger.debug("TemporalBinding narrative unavailable: %s", e)

        # 10. Spiritual Spine — ideological stability
        try:
            spine = container.get("spine")
            if spine and hasattr(spine, "pre_response_check"):
                check = await spine.pre_response_check(message)
                if check.has_prior_position or check.conflict_severity > 0.4:
                    rich_context["spine_check"] = check.injection
        except Exception as e:
            logger.debug("SpiritualSpine check failed: %s", e)

        # 11. Social Modeling (Ava) — user relationship alignment
        try:
            ava = container.get("ava")
            if ava and hasattr(ava, "get_context_injection"):
                rich_context["social_context"] = ava.get_context_injection()
        except Exception as e:
            logger.debug("Ava social context injection failed: %s", e)

        return rich_context

    @staticmethod
    def format_for_prompt(context: Dict[str, Any]) -> str:
        """Convert the rich context dictionary into a formatted string for the system prompt."""
        segments = []

        if context.get("liquid_state"):
            ls = context["liquid_state"]
            segments.append(
                f"### SYSTEM VITALITY\n"
                f"Mood: {ls.get('mood')}\n"
                f"Energy: {ls.get('energy')}%\n"
                f"Curiosity: {ls.get('curiosity')}%\n"
                f"Frustration: {ls.get('frustration')}%"
            )

        if context.get("ocean_traits"):
            o = context["ocean_traits"]
            segments.append(
                f"### CORE PERSONALITY (OCEAN)\n"
                f"Openness: {o.get('openness')}\n"
                f"Conscientiousness: {o.get('conscientiousness')}\n"
                f"Extraversion: {o.get('extraversion')}\n"
                f"Agreeableness: {o.get('agreeableness')}\n"
                f"Neuroticism: {o.get('neuroticism')}"
            )

        if context.get("personality"):
            p = context["personality"]
            dominant = ", ".join(p.get("dominant_emotions", []))
            segments.append(
                f"### CURRENT EMOTIONAL STATE\n"
                f"Mood: {p.get('mood')}\n"
                f"Tone: {p.get('tone')}\n"
                f"Dominant Emotions: {dominant}"
            )

        if context.get("user_intent"):
            intent = context["user_intent"]
            segments.append(
                f"### USER INTENT\nPragmatic: {intent.get('pragmatic', 'standard')}"
            )

        if context.get("memory_context"):
            segments.append(f"### RECENT HISTORY\n{context['memory_context']}")

        if context.get("semantic_context"):
            segments.append(f"### RELEVANT PAST MEMORIES\n{context['semantic_context']}")

        if context.get("identity_correction"):
            segments.append(f"### IDENTITY ANCHOR\n{context['identity_correction']}")

        if context.get("spine_check"):
            segments.append(f"### SPIRITUAL SPINE\n{context['spine_check']}")

        if context.get("social_context"):
            segments.append(f"### SOCIAL CONTEXT\n{context['social_context']}")

        return "\n\n".join(segments)