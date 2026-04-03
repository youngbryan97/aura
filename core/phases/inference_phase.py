from __future__ import annotations
import logging
from typing import Any, Optional
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState
from core.container import ServiceContainer
from core.service_names import ServiceNames

logger = logging.getLogger("Aura.InferencePhase")

class InferencePhase(Phase):
    """
    Experimental phase to extract subtext and implicit intent from user messages.
    Ensures Aura responds to what is FELT, not just what is SAID.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container or ServiceContainer

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        1. Perform 'Deep Inference' if the message is from a human.
        2. Inject inferred intent and subtext into state modifiers.
        """
        priority = kwargs.get("priority", False)
        if state.cognition.current_origin not in ("user", "voice", "admin"):
            return state

        try:
            router = self.container.get(ServiceNames.LLM_ROUTER, default=None)
            if not router:
                return state

            # Construct the inference prompt
            inference_prompt = (
                "Analyze the following user message for IMPLICIT INTENT, AFFECTIVE SUBTEXT, and CONVERSATION HOOKS. "
                "Do not respond to the user. Only return a JSON object with: "
                "{\n"
                "  'implicit_intent': '...',\n"
                "  'user_subtext': '...',\n"
                "  'momentum': 'stalled|flowing|intense',\n"
                "  'conversation_hooks': ['list of 2-3 specific topics or entities to address']\n"
                "}\n\n"
                f"User Message: {objective}"
            )

            # Use a fast/lightweight model for inference
            inference_data = await router.think(
                inference_prompt,
                system_prompt="You are Aura's subtext processor. Extract the unsaid.",
                prefer_tier="fast",
                priority=priority
            )

            # Parse and inject
            import json
            import re
            
            # Clean JSON if LLM added markdown wrappers
            clean_json = re.search(r"\{.*\}", inference_data, re.DOTALL)
            if clean_json:
                data = json.loads(clean_json.group(0))
                
                if not hasattr(state.cognition, "modifiers") or state.cognition.modifiers is None:
                    state.cognition.modifiers = {}
                
                state.cognition.modifiers["inferred_intent"] = data.get("implicit_intent", "")
                state.cognition.modifiers["user_subtext"] = data.get("user_subtext", "")
                state.cognition.modifiers["momentum"] = data.get("momentum", "flowing")
                state.cognition.modifiers["conversation_hooks"] = data.get("conversation_hooks", [])
                
                logger.info(f"Deep Inference: Intent='{data.get('implicit_intent')}', Subtext='{data.get('user_subtext')}'")
            
        except Exception as e:
            logger.warning("InferencePhase failed: %s", e)
            
        return state
