"""Refactored CognitiveEngine - Now a thin facade over modular phases.
"""
import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, Optional, Union

from core.consciousness.executive_authority import get_executive_authority
from core.runtime import background_policy
from core.runtime.pipeline_blueprint import instantiate_legacy_runtime_phases
from core.state.aura_state import AuraState
from core.utils.concurrency import RobustLock

from ..container import get_container
from .autopoiesis import AutopoieticGraph
from .llm.context_assembler import ContextAssembler
from .reasoning_strategies import ReasoningStrategies, StrategyType
from .types import ThinkingMode, Thought

logger = logging.getLogger(__name__)

_BACKGROUND_REFLECTIVE_MODES = frozenset({
    ThinkingMode.REFLECTIVE,
    ThinkingMode.CREATIVE,
})


def _record_objective_binding(state: AuraState, objective: str, *, source: str, mode: Any, reason: str) -> None:
    try:
        mode_value = getattr(mode, "value", mode)
        get_executive_authority().record_objective_binding(
            state,
            objective,
            source=source,
            mode=str(mode_value or ""),
            reason=reason,
        )
    except Exception as exc:
        logger.debug("Executive objective audit skipped for %s: %s", source, exc)

class CognitiveEngine:
    """
    Cognitive Engine facade.
    Now delegates to modular phases for structured thinking.
    """
    
    def __init__(self, backend: Any = None):
        self.backend = backend
        self.thoughts: deque = deque(maxlen=500)  # QUAL-04: bounded to prevent memory leak (BUG-017)
        self._phases = []
        self._augmentors = []
        self.state_repository = None
        self.autopoiesis = AutopoieticGraph()
        self._recovery_lock = RobustLock("CognitiveEngine.RecoveryLock")  # Audit Fix: Mutex for recovery
        self._reasoning: Optional[ReasoningStrategies] = None  # Lazy-init
        
    @property
    def consciousness(self) -> Any:
        """Unified access to the consciousness layer for metric aggregation."""
        from ..container import get_container
        return get_container().get("consciousness_core", default=None)

    @property
    def _current_tier(self) -> str:
        """Visibility for routing tests."""
        container = get_container()
        router = container.get("llm_router", default=None)
        if router and hasattr(router, "last_tier"):
            return router.last_tier
        return "unknown"

    @property
    def lobotomized(self) -> bool:
        """True if the engine has no usable cognitive pathway."""
        return self.state_repository is None and len(self._phases) == 0
        
    def setup(self, registry=None, router=None, event_bus=None):
        """Initialize components and phases."""
        container = get_container()
        # Ported Zenith: Phases expect Kernel, but modular boot often passes Container
        # We resolve the kernel instance or use a fallback mechanism
        kernel = container.get("aura_kernel", default=None)

        phase_entries = instantiate_legacy_runtime_phases(
            kernel or container,
            include_executive_closure=False,
        )
        self._phases = [phase for _, phase in phase_entries]
        
        # ISSUE-97: AuraPipeline Awareness
        required_phases = len(phase_entries)
        if len(self._phases) != required_phases:
            logger.warning("⚠️ AuraPipeline: Incomplete cognitive pipeline (%d/%d phases).", 
                           len(self._phases), required_phases)
        else:
            logger.info("🧠 AuraPipeline: Full cognitive spectrum online (%d phases).", required_phases)
        
        self.phase_map = {phase.__class__.__name__: phase for _, phase in phase_entries}

    async def on_start_async(self):
        """Lifecycle hook."""
        self.setup()
        logger.info("⚡ CognitiveEngine active.")

    async def check_health(self) -> Dict[str, Any]:
        """Health check."""
        return {
            "status": "healthy",
            "modular": True,
            "phases_count": len(self._phases),
            "augmentors_count": len(self._augmentors)
        }

    def register_augmentor(self, augmentor: Any):
        """Register a cognitive augmentor (e.g. SovereignWebAugmentor)."""
        if augmentor not in self._augmentors:
            self._augmentors.append(augmentor)
            logger.info("🧠 CognitiveEngine: Registered augmentor %s", type(augmentor).__name__)

    @staticmethod
    def _normalize_mode(mode: Union[ThinkingMode, str, Any]) -> ThinkingMode:
        if isinstance(mode, ThinkingMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            for candidate in ThinkingMode:
                if candidate.name.lower() == normalized:
                    return candidate
        return ThinkingMode.FAST

    @classmethod
    def _is_background_request(cls, origin: str, explicit_background: bool) -> bool:
        return background_policy.is_background_origin(origin, explicit_background=explicit_background)

    @staticmethod
    def _empty_thought(mode: ThinkingMode, reason: str) -> Thought:
        return Thought(
            id=str(uuid.uuid4()),
            content="",
            mode=mode,
            confidence=0.0,
            reasoning=[reason],
            metadata={"suppressed": True},
        )

    def _should_suppress_background_reflection(self, mode: ThinkingMode, is_background: bool) -> bool:
        if not is_background or mode not in _BACKGROUND_REFLECTIVE_MODES:
            return False

        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator:
                status = getattr(orchestrator, "status", None)
                if status and getattr(status, "is_processing", False):
                    return True

                last_user = float(getattr(orchestrator, "_last_user_interaction_time", 0.0) or 0.0)
                if last_user and (time.time() - last_user) < 180.0:
                    return True
        except Exception as exc:
            logger.debug("Background reflection suppression check failed: %s", exc)

        try:
            import psutil

            if psutil.virtual_memory().percent >= 80.0:
                return True
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return False

    def _background_suppression_reason(self) -> str:
        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            return str(
                background_policy.background_activity_reason(
                    orchestrator,
                    profile=background_policy.THOUGHT_BACKGROUND_POLICY,
                )
                or ""
            )
        except Exception as exc:
            logger.debug("Background thought policy check failed: %s", exc)
            return ""

    async def _set_recovery_in_progress(self, value: bool) -> None:
        """Flip the recovery flag under a short lock without holding it across slow awaits."""
        if await self._recovery_lock.acquire_robust(timeout=1.0):
            try:
                self._recovery_in_progress = value
            finally:
                if self._recovery_lock.locked():
                    self._recovery_lock.release()
        else:
            self._recovery_in_progress = value

    async def generate_autonomous_thought(self, prompt: str = None, **kwargs) -> Thought:
        """Entry point for self-initiated/autonomous thinking."""
        objective = prompt or "Reflecting on current inner state and environment."
        return await self.think(objective, origin="autonomous", **kwargs)

    async def think(self,
                    objective: str,
                    context: Dict[str, Any] = None,
                    mode: ThinkingMode = ThinkingMode.FAST,
                    origin: str = "user",
                    **kwargs) -> Thought:
        """
        Execute a cognitive cycle to produce a thought.
        This now drives the 8 phases to transform state.
        """
        mode = self._normalize_mode(mode)
        is_background = self._is_background_request(origin, bool(kwargs.get("is_background", False)))

        if is_background:
            suppression_reason = self._background_suppression_reason()
            if suppression_reason:
                logger.debug(
                    "🛡️ CognitiveEngine: Suppressing background thought for origin=%s (%s).",
                    origin,
                    suppression_reason,
                )
                return self._empty_thought(mode, f"background_thought_suppressed:{suppression_reason}")

        if self._should_suppress_background_reflection(mode, is_background):
            logger.debug("🛡️ CognitiveEngine: Suppressing background %s thought during active service window.", mode.name)
            return self._empty_thought(mode, "background_reflection_suppressed")

        logger.info("🧠 CognitiveEngine.think: %s... (%s) Origin: %s", objective[:50], mode.name, origin)
        
        # 1. Get current state (BUG-12 Fix: handle None state on first boot)
        repo = self.state_repository
        if repo is None:
            container = get_container()
            repo = container.get("state_repository")
            self.state_repository = repo
            
        if repo is None:
            from core.state.aura_state import AuraState
            state = AuraState.default()
        else:
            state = await repo.get_current()
            
        if state is None:
            from core.state.aura_state import AuraState
            state = AuraState.default()
        
        # 2. Derive base state for this cognitive cycle (Zenith-HF12 Fix)
        # This ensures every cycle starts with a unique version to prevent Atomic Guard rejections.
        state = state.derive(f"cognitive_intent: {origin}")
        
        # 3. Hardening: Set Current Objective & Origin
        # This prevents the race condition where ResponseGeneration would pick up
        # a background motivation message instead of the user's input.
        state.cognition.current_objective = objective
        state.cognition.current_origin = origin
        _record_objective_binding(
            state,
            objective,
            source=f"cognitive_engine:{origin}",
            mode=mode,
            reason="cognitive_cycle_bound",
        )
        state.response_modifiers["model_tier"] = "tertiary" if is_background else "primary"
        state.response_modifiers["deep_handoff"] = False

        # v40: Spiritual Spine - Prior Position Injection
        # The ordering is critical: injection -> system prompt -> user message.
        from core.container import ServiceContainer
        spine = ServiceContainer.get("spine", default=None)
        if spine and origin in ("user", "voice", "admin"):
            # Extract topic: look for nouns or use the first sentence.
            # v40: Improved topic extraction
            import re
            # Extract first sentence, then remove common filler
            raw = re.split(r'[.?!]', objective)[0].strip()
            # Remove "Tell me about", "What is", etc.
            topic = re.sub(r'(?i)^(tell me about|what is|what are|do you think about|give me|how does)\s+', '', raw)
            topic = topic[:60] if topic else "general"
            
            check = await spine.pre_response_check(objective, topic=topic)
            if check.injection:
                logger.info("⚡ [Spine] Injecting prior position into cognitive objective.")
                # Prepend the injection to the objective so it influences the entire cycle
                objective = check.injection + "\n\n" + objective
                state.cognition.current_objective = objective
                _record_objective_binding(
                    state,
                    objective,
                    source=f"cognitive_engine:{origin}",
                    mode=mode,
                    reason="spine_injection_bound",
                )

        # v40: Identity Drift - Context Refresh check
        # If history is too long and burying identity, we "refresh" by reminding Aura who she is.
        drift = ServiceContainer.get("drift_monitor", default=None)
        orchestrator = ServiceContainer.get("orchestrator", default=None)
        
        if drift:
            # Check for a specific pending correction from the last turn
            pending = getattr(orchestrator, "_pending_correction", "")
            if pending:
                # v40: Cast to str to satisfy weird type checker slice error
                pending_str = str(pending)
                logger.warning("🩹 [Drift] Applying pending identity correction: %s...", pending_str[:50])
                objective = f"{pending_str}\n\n{objective}"
                state.cognition.current_objective = objective
                _record_objective_binding(
                    state,
                    objective,
                    source=f"cognitive_engine:{origin}",
                    mode=mode,
                    reason="drift_correction_bound",
                )
            else:
                # Estimate general context health if no specific correction
                hist_len = len(str(state.cognition.working_memory))
                sys_len = len(ContextAssembler.build_system_prompt(state))
                if background_policy.is_user_facing_origin(origin) and drift.needs_context_refresh(hist_len, sys_len):
                    logger.warning("🔄 [Drift] Identity anchor buried. Triggering cognitive refresh.")
                    objective = "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]\n" + objective
                    state.cognition.current_objective = objective
                    _record_objective_binding(
                        state,
                        objective,
                        source=f"cognitive_engine:{origin}",
                        mode=mode,
                        reason="identity_refresh_bound",
                    )

        # v5.2: Augmentor Context Injection
        # Pull signals from registered augmentors before the phase loop
        augmentor_context = {}
        for aug in self._augmentors:
            try:
                if hasattr(aug, "get_augmentation"):
                    aug_data = aug.get_augmentation(objective)
                    if aug_data:
                        augmentor_context[type(aug).__name__] = aug_data
            except Exception as e:
                logger.warning("Augmentor %s failed: %s", type(aug).__name__, e)

        if augmentor_context:
            context = context or {}
            context.update({"augmentations": augmentor_context})

        loop_kwargs = dict(kwargs)
        loop_kwargs["is_background"] = is_background

        thought = await self._run_thinking_loop(
            state,
            objective,
            mode,
            origin,
            context,
            **loop_kwargs,
        )
        
        # v40: Clear drift correction after use
        orchestrator = ServiceContainer.get("orchestrator", default=None)
        if orchestrator and hasattr(orchestrator, "_pending_correction"):
            orchestrator._pending_correction = ""
            
        return thought

    async def _run_thinking_loop(self, 
                                 state: AuraState, 
                                 objective: str, 
                                 mode: ThinkingMode, 
                                 origin: str, 
                                 context: Dict[str, Any] = None, 
                                 **kwargs) -> Thought:
        """
        Internal method to execute the core cognitive phase loop.
        Extracted from `think` to allow pre/post-processing in `think`.
        """
        if origin in ("user", "voice", "admin", "external"):
            # Check if already in history to avoid duplication
            # vResilience: Workaround for Pyre2 slice limitations
            history = state.cognition.working_memory
            recent_count = min(5, len(history))
            recent = [history[i] for i in range(len(history)-recent_count, len(history))]
            is_duplicate = any(m.get("content") == objective for m in recent)
            if not is_duplicate:
                # We already derived at the start of the cycle, so we just append here.
                state.cognition.working_memory.append({
                    "role": "user",
                    "content": objective,
                    "timestamp": time.time(),
                    "origin": origin
                })
        
        # 4. Phase Execution Loop with Watchdog
        import copy
        backup_state = copy.deepcopy(state)
        temp_state = state
        success = False
        is_background = bool(kwargs.get("is_background", False))
        
        try:
            # v26.3 HARDENING: 400s Cognitive Watchdog (accommodates 360s Phase/MLX timeouts)
            async with asyncio.timeout(400.0):
                for phase in self._phases:
                    # Pass through kwargs like is_background if phases support it
                    temp_state = await phase.execute(temp_state, objective=objective, **kwargs)
                
                state = temp_state
                success = True
        except TimeoutError:
            logger.error("🛑 [COGNITION] Watchdog: Cognitive cycle TIMEOUT (240s).")
            # Immediate Reactive Recovery
            return await self._reactive_recovery(objective, mode, origin, "timeout")
        except Exception as e:
            logger.error("🚨 [COGNITION] Fatal error in phase logic: %s", e)
            # v14.1 HARDENING: Rollback & Downshift
            if mode == ThinkingMode.DEEP:
                logger.warning("🔄 [COGNITION] Downshifting to REACTIVE mode due to Deep Failure...")
                return await self.think(objective, mode=ThinkingMode.FAST, origin=origin, **kwargs)
            
            return await self._reactive_recovery(objective, mode, origin, f"crash: {e}")
        finally:
            # Cache Storm Fix: ALWAYS clear objective after processing
            # to prevent background tasks from sticking in the state and re-triggering
            # MindTick response generation indefinitely, even if we crashed or timed out.
            try:
                # vResilience: Avoid locals().get() for type stability
                if not success and 'backup_state' in locals():
                    state = backup_state
                
                if 'state' in locals() and state is not None:
                    state.cognition.current_objective = None
                    state.cognition.current_origin = None
            except Exception as _e:
                logger.debug('Ignored Exception in cognitive_engine.py: %s', _e)

        # ─── SUCCESS PATH (Unreachable before fix) ──────────────────────────
        # 5. Final State Commit
        # HF12: Handle concurrent version conflicts with a mini-retry loop
        from core.state.state_repository import StateVersionConflictError
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # v14.2: Ensure the repository reference is correct (self.state_repository)
                await self.state_repository.commit(state, "cognitive_cycle")
                break # Success!
            except StateVersionConflictError as v_err:
                if attempt == max_retries - 1:
                    logger.error("Final state commit failed after %d retries: %s", max_retries, v_err)
                    break
                
                logger.warning("🔄 [STATE] Version conflict (attempt %d/%d). Re-deriving from latest...", attempt+1, max_retries)
                # Preserve the cognitive work completed in this cycle
                preserved_memory = list(state.cognition.working_memory)
                preserved_objective = state.cognition.current_objective
                preserved_origin = state.cognition.current_origin
                
                latest = await self.state_repository.get_current()
                state = latest.derive(f"rebase_retry_{attempt+1}: {origin}")
                
                # Apply preserved cognitive context onto the newly derived state
                state.cognition.working_memory = preserved_memory
                state.cognition.current_objective = preserved_objective
                state.cognition.current_origin = preserved_origin
                
                # HF12 Extension: Preserve additional cognitive labor
                # These might have been updated by InitiativeGeneration or Consciousness phases
                state.cognition.active_goals = list(temp_state.cognition.active_goals)
                state.cognition.pending_initiatives = list(temp_state.cognition.pending_initiatives)
                state.cognition.attention_focus = temp_state.cognition.attention_focus
                state.cognition.phenomenal_state = temp_state.cognition.phenomenal_state
                # Audit Fix: Preserve modifiers (CIL-injected fields)
                if hasattr(temp_state.cognition, "modifiers"):
                    state.cognition.modifiers = dict(getattr(temp_state.cognition, "modifiers", {}) or {})
            except Exception as e:
                logger.error("Failed to commit final cognitive state: %s", e)
                break
        
        # 6. Extract Response
        last_msg = state.cognition.working_memory[-1] if state.cognition.working_memory else None
        if last_msg and last_msg.get("role") == "assistant":
            self.autopoiesis.experience_friction(objective[:20], 0.05)
            
            thought = Thought(
                id=str(uuid.uuid4()),
                content=last_msg["content"],
                mode=mode,
                confidence=0.9,
                reasoning=["Phase-based cognitive cycle completed successfully."]
            )
            self.thoughts.append(thought)
            return thought
            
        # Experience friction for unresolved objectives
        self.autopoiesis.experience_friction(objective[:20], 0.45)

        import random
        _processing_fallbacks = [
            "I'm turning that over. Give me a moment to find the right words.",
            "That's sitting with me, but I haven't landed on how to say it yet.",
            "I'm working through something with that. Let me try again.",
            "I heard you. My thinking is running deeper than my words right now.",
            "I'm reaching for an answer that feels honest, not just quick.",
        ]
        return Thought(
            id=str(uuid.uuid4()),
            content=random.choice(_processing_fallbacks),
            mode=mode,
            confidence=0.5,
            reasoning=["No explicit response generated in this cycle."]
        )

    async def _reactive_recovery(self, objective: str, mode: ThinkingMode, origin: str, reason: str) -> Thought:
        """
        Emergency reactive response when the main cognitive loop fails.
        BUG-10: Added recursion guard, timeout, and proper exception handling.
        """
        # Only use the mutex to guard the flag flip; long-running recovery work
        # must happen outside the lock so watchdogs don't see a false deadlock.
        if not await self._recovery_lock.acquire_robust(timeout=1.0):
            return Thought(
                id=str(uuid.uuid4()),
                content="I'm still gathering myself. Give me a moment.",
                mode=ThinkingMode.FAST,
                confidence=0.2,
                reasoning=["Recovery lock busy"]
            )

        try:
            if getattr(self, '_recovery_in_progress', False):
                return Thought(
                    id=str(uuid.uuid4()),
                    content="I'm still gathering myself. Give me a moment.",
                    mode=ThinkingMode.FAST,
                    confidence=0.2,
                    reasoning=["Recovery recursion guard triggered"]
                )
            self._recovery_in_progress = True
        finally:
            if self._recovery_lock.locked():
                self._recovery_lock.release()

        try:
            logger.warning("⚡ [COGNITION] Initiating Reactive Recovery Phase. Reason: %s", reason)
            
            # 1. Rollback state to last stable version (with timeout + guard)
            try:
                async with asyncio.timeout(5.0):
                    await self.state_repository.rollback(f"recovery: {reason}")
            except Exception as rollback_err:
                logger.warning("Rollback failed during recovery: %s", rollback_err)
            
            # 2. Get a quick reflex response if possible
            container = get_container()
            router = container.get("llm_router")
            
            reflex = None
            if hasattr(router, 'get_reflex_response'):
                reflex = router.get_reflex_response(objective)
                
            if reflex:
                return Thought(
                    id=str(uuid.uuid4()),
                    content=reflex,
                    mode=ThinkingMode.FAST,
                    confidence=1.0,
                    reasoning=[f"Reactive recovery via reflex matrix ({reason})"]
                )
                
            # 3. Last-resort fallback (natural, human-sounding)
            fallback_msg = "Hmm, I lost my train of thought for a second there. What were you saying?"
            if "user" in origin:
                fallback_msg = "Sorry, I got a bit tangled up in my thoughts. Can you say that again?"

            return Thought(
                id=str(uuid.uuid4()),
                content=fallback_msg,
                mode=ThinkingMode.FAST,
                confidence=0.3,
                reasoning=[f"Hard fallback after cognitive failure: {reason}"]
            )
        except Exception as recovery_err:
            logger.error("Error during recovery: %s", recovery_err)
            return Thought(
                id=str(uuid.uuid4()),
                content="I'm experiencing a momentary cognitive glitch. I'll be back in a second.",
                mode=ThinkingMode.FAST,
                confidence=0.1,
                reasoning=[f"Recovery itself failed: {recovery_err}"]
            )
        finally:
            await self._set_recovery_in_progress(False)

    def stop(self):
        """Shutdown logic (BUG-19)."""
        logger.info("🛑 CognitiveEngine stopping...")
        self._phases = []

    async def record_interaction(self, user_input: str, response: str, domain: str = "general") -> None:
        """Persist completed turns through the active learning/context stack."""
        container = get_container()

        context_manager = container.get("context_manager", default=None)
        if context_manager and context_manager is not self and hasattr(context_manager, "record_interaction"):
            try:
                await context_manager.record_interaction(user_input, response, domain=domain)
                return
            except Exception as exc:
                logger.debug("CognitiveEngine.record_interaction context-manager path failed: %s", exc)

        learning = container.get("learning_engine", default=None)
        if learning and hasattr(learning, "record_interaction"):
            try:
                await learning.record_interaction(
                    user_input=user_input,
                    aura_response=response,
                    domain=domain,
                )
            except Exception as exc:
                logger.debug("CognitiveEngine.record_interaction learning path failed: %s", exc)

    async def think_stream(self, objective: str, **kwargs):
        """Streaming thought generator via modular router."""
        container = get_container()
        router = container.get("llm_router")
        state = await self.state_repository.get_current()
        if not state:
            from core.state.aura_state import AuraState
            state = AuraState.default()
        
        # Build structured messages
        messages = ContextAssembler.build_messages(state, objective)
        
        # Standard streaming path
        async for event in router.think_stream(messages=messages, **kwargs):
            if hasattr(event, "content"):
                yield event.content
            else:
                yield str(event)

    async def see(self, vision_payload: Dict[str, Any]) -> str:
        """Process a vision payload from the sensory pipeline.
        
        [ZENITH] Functionalized: Linking Sensory Buffer to Cognitive reasoning.
        """
        from core.container import ServiceContainer
        buffer = ServiceContainer.get("vision_buffer", default=None)
        if not buffer:
            logger.warning("👁️ [VISION] see() called but vision_buffer not found in container.")
            return "👁️ visual_analysis: Sensory buffer unavailable."
            
        prompt = vision_payload.get("query") or vision_payload.get("prompt") or "Describe the current visual state."
        return await buffer.query_visual_context(prompt, brain=self)


    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response by routing through the LLM router.

        Bridge method for callers like LanguageCenter that expect a
        ``generate()`` interface.  Now enhanced with reasoning strategies
        for complex queries (debate, decomposition, consistency).

        Args:
            prompt: The text prompt to send to the LLM.
            **kwargs: Additional parameters forwarded to the router.

        Returns:
            The generated text response.
        """
        container = get_container()
        purpose = str(kwargs.get("purpose", "") or "").strip().lower()
        origin = str(kwargs.get("origin", "") or "").strip().lower()
        user_facing_purposes = {"chat", "conversation", "expression", "reply", "user_response"}
        user_facing_origins = {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}

        if not origin:
            origin = "system"
            kwargs["origin"] = origin

        if "is_background" not in kwargs:
            kwargs["is_background"] = not (
                purpose in user_facing_purposes or origin in user_facing_origins
            )

        if kwargs.get("is_background") and "prefer_tier" not in kwargs:
            kwargs["prefer_tier"] = "tertiary"
        
        # v40: Spiritual Spine - Prior Position Injection
        spine = container.get("spine", default=None)
        if spine:
            check = await spine.pre_response_check(prompt)
            if check.injection:
                prompt = check.injection + "\n\n" + prompt

        router = container.get("llm_router", default=None)
        
        # v41: Reasoning Strategy Enhancement
        # For non-trivial queries, apply advanced reasoning (debate, decompose, etc.)
        use_strategies = kwargs.pop("use_strategies", True)
        force_strategy = kwargs.pop("force_strategy", None)
        strategy_query = str(kwargs.pop("strategy_query", "") or "").strip()

        if router and use_strategies:
            # Lazy-init the reasoning layer on first use
            if self._reasoning is None:
                async def _raw_generate(p, **kw):
                    return await router.think(p, **kw)
                self._reasoning = ReasoningStrategies(_raw_generate)
            
            strategy = force_strategy
            if strategy is None:
                if not strategy_query:
                    messages = kwargs.get("messages")
                    if isinstance(messages, list):
                        for msg in reversed(messages):
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", "") or "").strip().lower()
                            content = str(msg.get("content", "") or "").strip()
                            if role in {"user", "human"} and content:
                                strategy_query = content
                                break
                classify_target = strategy_query or prompt
                # Only use advanced strategies for user-facing queries, not internal prompts
                classified = self._reasoning.classify(classify_target)
                if classified != StrategyType.DIRECT and len(classify_target) > 30:
                    strategy = classified
            
            if strategy and strategy != StrategyType.DIRECT:
                try:
                    from ..thought_stream import get_emitter
                    get_emitter().emit(
                        "Deep Reasoning 🧠",
                        f"Using {strategy.name} strategy",
                        level="info",
                        category="Cognition"
                    )
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                
                strategy_input = strategy_query or prompt
                result = await self._reasoning.execute(strategy_input, strategy=strategy, **kwargs)
                return result.content
        
        # Standard direct generation
        if router:
            return await router.think(prompt, **kwargs)
        # Fallback if no router
        thought = await self.think(prompt, **kwargs)
        return thought.content if hasattr(thought, 'content') else str(thought)

    def _emit_thought(self, thought: str):
        """Internal helper to publish thoughts to the event bus."""
        container = get_container()
        eb = container.get("event_bus")
        if eb:
            eb.publish_threadsafe("thought", {
                "timestamp": time.time(),
                "content": thought,
                "engine": "ReAct" if "ReAct" in thought else "Modular"
            })
