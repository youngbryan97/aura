from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import re
import time
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState, CognitiveMode
from ..cognitive.parallel_thought import ParallelThoughtStream
from ..consciousness.executive_authority import get_executive_authority
from core.runtime.skill_task_bridge import looks_like_execution_report, looks_like_multi_step_skill_request
from core.runtime.turn_analysis import analyze_turn, looks_like_deep_mind_probe
from core.utils.queues import decode_stringified_priority_message, role_for_origin

# Regex to detect URLs in user input for auto-browser invocation
_URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\')\]]+',
    re.IGNORECASE,
)

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
        is_execution_report = looks_like_execution_report(input_text)
        is_deep_mind_probe = looks_like_deep_mind_probe(input_text)

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
        new_state.response_modifiers.pop("auto_browse_urls", None)

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
        
        # ── SYSTEM / ENVIRONMENTAL SENSORY FEED FAST-PATH ──
        # General improvement: If the input is explicitly a system directive or sensory
        # feed injected by a background process (e.g. embodied environments, terminals),
        # we bypass eager tool execution and treat it strictly as REACTIVE/CHAT.
        if (
            input_text.startswith("CORE DIRECTIVE:") 
            or "[environmental context" in lower_input 
            or "[embodied control contract]" in lower_input
            or "sensory update" in lower_input
            or "[sensory feed" in lower_input
        ):
            logger.info("🧭 Routing: SENSORY FEED / CORE DIRECTIVE detected. Routing as CHAT to avoid eager tool commitments.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            new_state.cognition.current_objective = input_text
            new_state.cognition.current_origin = routing_origin
            new_state.response_modifiers["intent_type"] = "CHAT"
            new_state.response_modifiers["semantic_intent"] = "casual"
            new_state.response_modifiers["model_tier"] = "primary"
            new_state.response_modifiers["deep_handoff"] = False
            return new_state

        # Fast skill detection before any LLM routing so tool use stays reliable
        matched_skills: list[str] = []
        try:
            cap = self.container.get("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent") and routing_origin in user_origins and not is_deep_mind_probe:
                matched_skills = list(cap.detect_intent(input_text) or [])
                if matched_skills:
                    if is_execution_report:
                        logger.info(
                            "🧭 Routing: execution report detected; ignoring skill fast-path candidates %s",
                            matched_skills[:3],
                        )
                    elif looks_like_multi_step_skill_request(input_text, matched_skills):
                        new_state.response_modifiers["matched_skills"] = matched_skills
                        logger.info(
                            "🧭 Routing: multi-step skill-backed task detected → TASK via %s",
                            matched_skills[:3],
                        )
                    else:
                        new_state.response_modifiers["matched_skills"] = matched_skills
                        logger.info("🧭 Routing: SKILL detected via patterns → %s", matched_skills[:3])
                        new_state.cognition.current_mode = CognitiveMode.REACTIVE
                        new_state.cognition.current_objective = input_text
                        new_state.cognition.current_origin = routing_origin
                        new_state.response_modifiers["intent_type"] = "SKILL"
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
            record_degradation('cognitive_routing', exc)
            logger.debug("🧭 Routing: detect_intent fast path failed: %s", exc)

        # ── URL Auto-Detection ────────────────────────────────────────
        # When the user pastes a URL, auto-invoke sovereign_browser to FETCH
        # the page content. Without this, URLs are treated as plain text and
        # Aura hallucinates about content she never accessed.
        if routing_origin in user_origins and not matched_skills:
            url_matches = _URL_PATTERN.findall(input_text)
            if url_matches:
                logger.info("🧭 Routing: URL detected in user input → auto-matching sovereign_browser: %s", url_matches[0][:80])
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                new_state.cognition.current_objective = input_text
                new_state.cognition.current_origin = routing_origin
                new_state.response_modifiers["intent_type"] = "SKILL"
                new_state.response_modifiers["matched_skills"] = ["sovereign_browser"]
                new_state.response_modifiers["model_tier"] = "primary"
                new_state.response_modifiers["deep_handoff"] = False
                new_state.response_modifiers["auto_browse_urls"] = url_matches[:3]
                get_executive_authority().record_user_objective(
                    new_state,
                    input_text,
                    source=f"cognitive_routing:{routing_origin}",
                    mode=str(CognitiveMode.REACTIVE.value),
                )
                return new_state

        analysis = analyze_turn(input_text, matched_skills=matched_skills)
        cognitive_mode = CognitiveMode.REACTIVE
        new_state.response_modifiers["intent_type"] = analysis.intent_type
        new_state.response_modifiers["semantic_intent"] = analysis.semantic_mode
        logger.info(
            "🧭 CognitiveRouting: deterministic intent=%s semantic=%s live_voice=%s",
            analysis.intent_type,
            analysis.semantic_mode,
            analysis.requires_live_aura_voice,
        )
        if (
            not analysis.is_execution_report
            and (
                analysis.intent_type == "TASK"
                or analysis.suggests_deliberate_mode
                or self._has_deliberate_keywords(input_text)
            )
        ):
            cognitive_mode = CognitiveMode.DELIBERATE

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
        deep_handoff = self._should_allow_deep_handoff(
            input_text,
            cognitive_mode,
            is_autonomous,
            analysis=analysis,
        )
        new_state.response_modifiers["model_tier"] = model_tier
        new_state.response_modifiers["deep_handoff"] = deep_handoff
        if routing_origin in user_origins and not analysis.is_execution_report and not is_deep_mind_probe:
            try:
                cap = self.container.get("capability_engine", default=None)
                if cap and hasattr(cap, "detect_intent"):
                    new_state.response_modifiers["matched_skills"] = list(cap.detect_intent(input_text) or [])
            except Exception as exc:
                record_degradation('cognitive_routing', exc)
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
            task = get_task_tracker().create_task(self.parallel_stream.branch(
                input_text, 
                str(state.cognition.working_memory[-2:])
            ))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
            
        return new_state

    def _resolve_model_tier(self, state: AuraState, input_text: str, mode: CognitiveMode, is_autonomous: bool) -> str:
        """
        Determine the LLM tier based on cognitive mode, task type, AND substrate state.

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

    def _should_allow_deep_handoff(
        self,
        text: str,
        mode: CognitiveMode,
        is_autonomous: bool,
        *,
        analysis: Any | None = None,
    ) -> bool:
        """Gate the 72B solver using SUBSTRATE STATE, not just keywords.

        The substrate decides whether a thought is complex enough for the
        deep solver. High unified field coherence + high phi = the system
        is integrated and can handle complexity. Low coherence = fragmented,
        stay reactive. High free energy = high surprise, needs deep processing.

        Keywords are kept as a fallback but the substrate is the primary signal.
        """
        if is_autonomous or mode != CognitiveMode.DELIBERATE:
            return False

        lower = text.lower()
        word_count = len(text.split())
        current_analysis = analysis or analyze_turn(text)
        if getattr(current_analysis, "is_execution_report", False) or looks_like_execution_report(text):
            return False
        semantic_mode = str(getattr(current_analysis, "semantic_mode", "") or "").lower()
        explicit_deep_request = any(keyword in lower for keyword in _DEEP_HANDOFF_KEYWORDS)
        looks_technical = semantic_mode == "technical" or any(
            marker in lower
            for marker in (
                "pytest",
                "traceback",
                "stack trace",
                "failing test",
                "exception",
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                "core/",
                "interface/",
                "root cause",
                "debug",
                "refactor",
                "architecture",
            )
        )
        if not explicit_deep_request and not looks_technical:
            # Deliberate emotional/philosophical conversation should stay on the
            # stable Cortex lane unless the user explicitly asks for heavyweight
            # deep reasoning. This preserves conversational continuity.
            return False

        # ── Substrate-driven handoff decision ─────────────────────────
        substrate_score = 0.0
        try:
            from core.voice.substrate_voice_engine import _extract_unified_field, _extract_neurochemicals
            uf = _extract_unified_field()
            nc = _extract_neurochemicals()

            coherence = uf.get("coherence", 0.7)
            phi = uf.get("phi", 0.5)
            field_complexity = uf.get("field_complexity", 0.5)
            acetylcholine = nc.get("acetylcholine", 0.5)
            norepinephrine = nc.get("norepinephrine", 0.5)

            # High coherence = system is integrated, CAN go deep
            if coherence > 0.6:
                substrate_score += 0.25
            # High phi = genuine integrated information, complex thought warranted
            if phi > 0.3:
                substrate_score += 0.2
            # High field complexity = rich internal state, deep processing natural
            if field_complexity > 0.6:
                substrate_score += 0.15
            # High acetylcholine = sharp attention, ready for focused work
            if acetylcholine > 0.6:
                substrate_score += 0.15
            # High norepinephrine = alert, can handle demanding task
            if norepinephrine > 0.5:
                substrate_score += 0.1

            # Require an explicit deep-reasoning request OR a substantial technical
            # prompt for substrate-approved handoff. Borderline philosophical
            # questions ("describe one moment …") were spuriously promoted to
            # the 72B lane, hot-swapping the warm 32B and forcing 12s cooldowns
            # mid-conversation. Keep them on the primary lane unless the user
            # clearly signaled heavy work.
            if substrate_score >= 0.5 and (
                explicit_deep_request
                or (looks_technical and word_count >= 60)
            ):
                logger.info(
                    "🧠 CognitiveRouting: SUBSTRATE approves deep handoff (score=%.2f, "
                    "coherence=%.2f, phi=%.2f, complexity=%.2f)",
                    substrate_score, coherence, phi, field_complexity,
                )
                return True

        except Exception as exc:
            record_degradation('cognitive_routing', exc)
            logger.debug("Substrate routing check failed, falling back to keywords: %s", exc)

        # ── Keyword fallback (still useful for explicit requests) ──────
        if looks_technical and (word_count >= 120 or len(text) >= 900):
            return True
        return explicit_deep_request

    @staticmethod
    def _has_deliberate_keywords(text: str) -> bool:
        """Check if the input contains keywords that warrant the 32B brain."""
        lower = text.lower()
        return any(kw in lower for kw in _DELIBERATE_KEYWORDS)
