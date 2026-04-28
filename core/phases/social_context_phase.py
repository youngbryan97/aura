from __future__ import annotations
from core.runtime.errors import record_degradation

import logging
from typing import Any, Optional
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState
from core.container import ServiceContainer
from core.service_names import ServiceNames

logger = logging.getLogger("Aura.SocialContextPhase")

class SocialContextPhase(Phase):
    """
    Phase to inject social context from Ava (SocialModelingEngine).
    Ensures that Aura's responses are tailored to the user's communication style.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        # Phase usually takes kernel, but modular phases take container
        self.container = container or ServiceContainer

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        1. Analyze user message if this is a user-initiated turn.
        2. Inject social context into state modifiers.
        """
        try:
            ava = self.container.get(ServiceNames.AVA, default=None)
            if not ava or not objective:
                return state

            # ISSUE-89: Synchronized Social Analysis
            # 1. Analysis (if user message is new)
            if state.cognition.current_origin in ("user", "voice", "admin") and objective:
                if hasattr(ava, "analyze_message"):
                    # Use getattr to safely check if it's a coroutine
                    import asyncio
                    res = ava.analyze_message(objective)
                    if asyncio.iscoroutine(res):
                        await res

            # 2. Injection (FIX BUG: Unified Engine Resolution)
            if ava and hasattr(ava, "get_context_injection"):
                injection = ava.get_context_injection()
                if not hasattr(state.cognition, "modifiers") or state.cognition.modifiers is None:
                    state.cognition.modifiers = {}
                
                # 3. ENGAGEMENT CUES (Linguistic Alignment - Phase 6)
                # Analyze conversational momentum and reciprocity
                user_msg_len = len(objective.split())
                
                # A. RECIPROCAL INTENSITY (Mirroring length)
                # We set a 'mirror_length_target' for the response phase
                state.cognition.modifiers["mirror_length_target"] = user_msg_len
                
                if user_msg_len < 4:
                    # User is under-engaging (e.g. "ok", "cool", "yeah")
                    state.cognition.modifiers["interaction_style"] = "proactive_engagement"
                    state.cognition.modifiers["desired_brevity"] = "extreme"
                    logger.debug("SocialContext: User under-engaging. Pushing proactive engagement.")
                elif user_msg_len > 100:
                    # User is over-explaining
                    state.cognition.modifiers["interaction_style"] = "backchannel_heavy"
                    state.cognition.modifiers["desired_brevity"] = "low"
                    logger.debug("SocialContext: User over-explaining. Pushing backchannel logic.")
                else:
                    state.cognition.modifiers["interaction_style"] = "balanced_flow"
                    state.cognition.modifiers["desired_brevity"] = "moderate"

                # B. LEXICAL OVERLAP (Mirroring vocabulary)
                # Identifying 'signal words' from the user to reflect back
                stop_words = {"the", "a", "an", "is", "are", "and", "or", "but", "i", "you", "my", "your"}
                user_words = set(objective.lower().split())
                signal_words = [w for w in user_words if w not in stop_words and len(w) > 4]
                state.cognition.modifiers["lexical_mirror"] = signal_words[:5]
                
                # 4. TheoryOfMind Rapport — modulate interaction style by relationship depth
                try:
                    tom = ServiceContainer.get("theory_of_mind", default=None)
                    if tom and tom.known_selves:
                        user_model = next(iter(tom.known_selves.values()))
                        rapport = getattr(user_model, "rapport", 0.5)
                        # High rapport → more intimate/playful; low rapport → more careful/formal
                        if rapport > 0.75:
                            state.cognition.modifiers["relational_register"] = "intimate"
                        elif rapport > 0.4:
                            state.cognition.modifiers["relational_register"] = "warm"
                        else:
                            state.cognition.modifiers["relational_register"] = "cordial"
                        state.cognition.modifiers["rapport_level"] = rapport
                except Exception as _exc:
                    record_degradation('social_context_phase', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)

                # 5. Synchronize
                if state.cognition.modifiers.get("social_context") != injection:
                    state.cognition.modifiers["social_context"] = injection
                    logger.debug("Social context synchronized: %s", injection)

        except Exception as e:
            record_degradation('social_context_phase', e)
            logger.warning("SocialContextPhase failed: %s", e)
            
        return state
