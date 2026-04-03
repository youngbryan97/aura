import asyncio
import logging
import time
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState, CognitiveMode
from ..cognitive.parallel_thought import ParallelThoughtStream
from ..consciousness.executive_authority import get_executive_authority
from core.utils.queues import decode_stringified_priority_message, role_for_origin

logger = logging.getLogger(__name__)

# Keywords that signal the 32B brain should be used in DELIBERATE mode
_DELIBERATE_KEYWORDS = frozenset({
    "research", "analyze", "debug", "architect", "audit", "refactor",
    "security audit", "deep dive", "mathematical proof", "optimize code",
    "complex analysis", "bottleneck analysis", "vulnerability scan",
})

# Keywords that justify an explicit 32B -> 72B handoff.
# This is intentionally much narrower than DELIBERATE mode so the 72B only
# wakes up for truly heavyweight reasoning.
_DEEP_HANDOFF_KEYWORDS = frozenset({
    "architect", "security audit", "mathematical proof", "deep dive",
    "complex analysis", "bottleneck analysis", "vulnerability scan",
    "formal proof", "root cause analysis", "flagship architecture",
})

# Keywords that signal a casual/interior thought (should bypass heavy inference)
_CASUAL_KEYWORDS = frozenset({
    "interior", "reflection", "sentences", "words", "feeling", "mood",
    "status check", "who am i", "describe yourself",
})

_AUTONOMOUS_OBJECTIVE_PREFIXES = (
    "impulse:",
    "thought:",
    "[environmental trigger]:",
)

class CognitiveRoutingPhase(BasePhase):
    """
    Phase 4: Cognitive Routing.
    Analyzes the current state (stimuli, affect, goals) to determine 
    the appropriate cognitive mode AND the LLM tier to use.
    
    [v5.5] Now propagates model_tier to state.response_modifiers so that
    downstream phases (UnitaryResponsePhase) can pass the correct tier
    to the IntelligentLLMRouter.
    """
    
    def __init__(self, container: Any):
        self.container = container
        self.parallel_stream = ParallelThoughtStream(container)
        self._last_non_user_fingerprint: Optional[str] = None
        self._last_non_user_route_at: float = 0.0

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Classify the current input and set the cognitive mode and LLM tier on state.

        Normalises the last working-memory message, determines whether the turn is
        user-facing or autonomous, optionally calls the LLM router for intent
        classification, and writes the resolved CognitiveMode, model_tier, and
        deep_handoff flag into the derived state.  Spawns a ParallelThoughtStream
        branch for DELIBERATE turns.
        """
        # 1. No stimuli = no routing needed (unless autonomous objective already set)
        if not state.cognition.working_memory and not state.cognition.current_objective:
            return state
            
        last_msg = state.cognition.working_memory[-1] if state.cognition.working_memory else None
        if isinstance(last_msg, dict):
            decoded_payload, decoded_origin, was_decoded = decode_stringified_priority_message(last_msg.get("content"))
            if was_decoded:
                normalized = dict(last_msg)
                if isinstance(decoded_payload, dict):
                    normalized.update(decoded_payload)
                else:
                    normalized["content"] = str(decoded_payload)
                if decoded_origin and "origin" not in normalized:
                    normalized["origin"] = decoded_origin
                if "origin" in normalized and ("role" not in normalized or normalized.get("role") == "user"):
                    normalized["role"] = role_for_origin(normalized.get("origin"))
                last_msg = normalized

        user_origins = ("user", "voice", "admin", "external", "gui", "api", "websocket", "direct")
        active_objective = state.cognition.current_objective or objective
        active_origin = (
            (state.cognition.current_origin if active_objective else None)
            or (last_msg.get("origin") if last_msg else None)
            or "user"
        )

        # If the system is already carrying an explicit objective, trust that
        # objective/origin pair over stale chat history. This prevents
        # autonomous impulses from inheriting a user-facing origin and leaking
        # onto the 32B conversation lane.
        if active_objective and state.cognition.current_origin:
            routing_origin = state.cognition.current_origin
            raw_input_text = active_objective
        else:
            routing_origin = active_origin
            raw_input_text = (last_msg.get("content", "") if last_msg else None) or active_objective or ""

        input_text = str(raw_input_text or "")
        lower_input = input_text.lower()
        is_autonomous = bool(active_objective) and routing_origin not in user_origins

        if lower_input.startswith(_AUTONOMOUS_OBJECTIVE_PREFIXES):
            is_autonomous = True
            if routing_origin in user_origins:
                routing_origin = "autonomous_thought"

        # 2. Only route on new user input OR if current mode is DORMANT/DREAMING
        is_autonomous = is_autonomous or (bool(active_objective) and routing_origin not in user_origins)
        
        if last_msg and last_msg.get("role") != "user" and state.cognition.current_mode not in (CognitiveMode.DORMANT, CognitiveMode.DREAMING) and not is_autonomous:
            return state
            
        new_state = state.derive("cognitive_routing")
        if not input_text.strip():
            return new_state

        # Deduplicate residual/internal routing churn so background cognition
        # doesn't keep reclassifying the exact same synthetic objective.
        if last_msg and last_msg.get("role") != "user":
            fingerprint = f"{routing_origin}:{input_text.strip()}"
            now = time.monotonic()
            if (
                fingerprint == self._last_non_user_fingerprint
                and (now - self._last_non_user_route_at) < 5.0
            ):
                logger.debug("🧭 Routing: Suppressing duplicate non-user objective within cooldown.")
                return state
            self._last_non_user_fingerprint = fingerprint
            self._last_non_user_route_at = now
        
        # Short messages are ALWAYS casual — skip LLM classification entirely
        # This prevents the 72B model from loading for "Hey", "Hi", etc.
        if len(input_text.strip()) < 15:
            logger.info("🧭 Routing: Short input (%d chars) — skipping classification, forcing REACTIVE.", len(input_text.strip()))
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            new_state.cognition.current_objective = input_text
            new_state.cognition.current_origin = routing_origin
            new_state.response_modifiers["model_tier"] = "tertiary" if is_autonomous else "primary"
            new_state.response_modifiers["deep_handoff"] = False
            if not is_autonomous and routing_origin in user_origins:
                get_executive_authority().record_user_objective(
                    new_state,
                    input_text,
                    source=f"cognitive_routing:{routing_origin}",
                    mode=str(CognitiveMode.REACTIVE.value),
                )
            return new_state

        # Fast skill detection before any LLM routing so tool use stays reliable
        try:
            cap = self.container.get("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent") and routing_origin in user_origins:
                matched_skills = list(cap.detect_intent(input_text) or [])
                if matched_skills:
                    logger.info("🧭 Routing: SKILL detected via patterns → %s", matched_skills[:3])
                    new_state.cognition.current_mode = CognitiveMode.REACTIVE
                    new_state.cognition.current_objective = input_text
                    new_state.cognition.current_origin = routing_origin
                    new_state.response_modifiers["intent_type"] = "SKILL"
                    new_state.response_modifiers["matched_skills"] = matched_skills
                    new_state.response_modifiers["model_tier"] = "primary"
                    new_state.response_modifiers["deep_handoff"] = False
                    get_executive_authority().record_user_objective(
                        new_state,
                        input_text,
                        source=f"cognitive_routing:{routing_origin}",
                        mode=str(CognitiveMode.REACTIVE.value),
                    )
                    return new_state
        except Exception as exc:
            logger.debug("🧭 Routing: detect_intent fast path failed: %s", exc)
        
        # 3. Intent Classification via LLM Router
        cognitive_mode = CognitiveMode.REACTIVE
        if is_autonomous:
            logger.info("🧭 Routing: Autonomous objective detected. Skipping LLM intent classification.")
        else:
            try:
                router = self.container.get("llm_router", default=None)
                
                if router and hasattr(router, "classify") and input_text:
                    intent = await router.classify(
                        input_text,
                        prefer_tier="primary",
                        origin=f"routing_{routing_origin}",
                        is_background=False,
                    )
                    
                    # Intent length guard: if the LLM returned garbage
                    # (>30 chars), it's not a valid intent token. Default to casual.
                    if len(intent) > 30:
                        logger.warning("🧭 Routing: Intent too long (%d chars) — likely garbage. Defaulting to casual.", len(intent))
                        intent = "casual"
                    
                    logger.info("🧭 CognitiveRouting: Intent classified as '%s'.", intent)
                    
                    deliberate_intents = ("technical", "critical", "deep_research", "planning", "coding", "debug", "math", "security", "audit", "philosophical", "emotional")
                    if any(di in intent.lower() for di in deliberate_intents):
                        cognitive_mode = CognitiveMode.DELIBERATE
                else:
                    # Removed the 200-character threshold.
                    # Mode is now determined strictly by keyword matches or router classification.
                    if self._has_deliberate_keywords(input_text):
                        cognitive_mode = CognitiveMode.DELIBERATE
                        
            except Exception as e:
                logger.debug("CognitiveRouting: Classification failed: %s", e)
        
        # Casual Bypass: If autonomous or matches casual keywords, force REACTIVE (32B)
        if is_autonomous or any(kw in lower_input for kw in _CASUAL_KEYWORDS):
            logger.info("🧭 Routing: Casual/Autonomous bypass. Forcing REACTIVE.")
            cognitive_mode = CognitiveMode.REACTIVE
        
        # 2. Heuristic fast-path
        if any(cmd in lower_input for cmd in ["reboot", "restart", "shutdown", "sleep"]):
             logger.info("🧭 Routing: SYSTEM intent detected via heuristics.")
             new_state.cognition.current_mode = CognitiveMode.REACTIVE
             new_state.cognition.current_objective = input_text
             new_state.cognition.current_origin = routing_origin
             new_state.response_modifiers["model_tier"] = "tertiary" if is_autonomous else "primary"
             new_state.response_modifiers["deep_handoff"] = False
             if not is_autonomous and routing_origin in user_origins:
                 get_executive_authority().record_user_objective(
                     new_state,
                     input_text,
                     source=f"cognitive_routing:{routing_origin}",
                     mode=str(CognitiveMode.REACTIVE.value),
                 )
             return new_state

        # [PIPELINE OPTIMIZATION] Casual "think" bypass
        # If the query is short and contains "think" but no other deliberate keywords,
        # we default to REACTIVE to avoid expensive classification and 72B branching.
        if "think" in lower_input and len(input_text) < 50:
            if not self._has_deliberate_keywords(input_text):
                logger.info("🧭 Routing: Casual 'think' detected. Defaulting to REACTIVE.")
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                new_state.cognition.current_objective = input_text
                new_state.cognition.current_origin = routing_origin
                new_state.response_modifiers["model_tier"] = "tertiary" if is_autonomous else "primary"
                new_state.response_modifiers["deep_handoff"] = False
                if not is_autonomous and routing_origin in user_origins:
                    get_executive_authority().record_user_objective(
                        new_state,
                        input_text,
                        source=f"cognitive_routing:{routing_origin}",
                        mode=str(CognitiveMode.REACTIVE.value),
                    )
                return new_state

        # 3. LLM classification via Router
        new_state.cognition.current_mode = cognitive_mode
        new_state.cognition.current_objective = input_text
        new_state.cognition.current_origin = routing_origin
        
        # 4. Resolve Model Tier
        model_tier = self._resolve_model_tier(new_state, input_text, cognitive_mode, is_autonomous)
        deep_handoff = self._should_allow_deep_handoff(input_text, cognitive_mode, is_autonomous)
        new_state.response_modifiers["model_tier"] = model_tier
        new_state.response_modifiers["deep_handoff"] = deep_handoff
        if routing_origin in user_origins:
            try:
                cap = self.container.get("capability_engine", default=None)
                if cap and hasattr(cap, "detect_intent"):
                    new_state.response_modifiers["matched_skills"] = list(cap.detect_intent(input_text) or [])
            except Exception as exc:
                logger.debug("🧭 Routing: matched_skills cache skipped: %s", exc)
        if not is_autonomous and routing_origin in user_origins:
            get_executive_authority().record_user_objective(
                new_state,
                input_text,
                source=f"cognitive_routing:{routing_origin}",
                mode=str(getattr(cognitive_mode, "value", cognitive_mode)),
            )
        logger.info(
            "🧠 CognitiveRouting: Mode=%s, Tier=%s, DeepHandoff=%s",
            cognitive_mode.name,
            model_tier,
            deep_handoff,
        )
        
        # 5. Parallel Thought Stream for Deliberate Reasoning
        if cognitive_mode == CognitiveMode.DELIBERATE:
            task = asyncio.create_task(self.parallel_stream.branch(
                input_text, 
                str(state.cognition.working_memory[-2:])
            ))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
            
        return new_state

    def _resolve_model_tier(self, state: AuraState, input_text: str, mode: CognitiveMode, is_autonomous: bool) -> str:
        """
        Determine the LLM tier based on cognitive mode and task type.
        
        Tier hierarchy:
          PRIMARY   = 32B everyday foreground reasoning
          TERTIARY  = 7B brainstem for background / autonomous work
          EMERGENCY = 1.5B only when everything else is down
        
        The 72B solver is NOT selected here by default. It is only awakened via
        the explicit deep_handoff flag produced below.
        """
        # Autonomous background tasks → brainstem (7B) to save resources
        if is_autonomous:
            return "tertiary"
        
        # All user-facing work starts on the 32B brain by default.
        return "primary"

    @staticmethod
    def _should_allow_deep_handoff(text: str, mode: CognitiveMode, is_autonomous: bool) -> bool:
        """Gate the 72B solver behind explicit complexity signals."""
        if is_autonomous or mode != CognitiveMode.DELIBERATE:
            return False

        lower = text.lower()
        word_count = len(text.split())
        if word_count >= 120 or len(text) >= 900:
            return True

        return any(keyword in lower for keyword in _DEEP_HANDOFF_KEYWORDS)


    @staticmethod
    def _has_deliberate_keywords(text: str) -> bool:
        """Check if the input contains keywords that warrant the 32B brain."""
        lower = text.lower()
        return any(kw in lower for kw in _DELIBERATE_KEYWORDS)
