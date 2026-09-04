"""
Aura Cognitive Integration Layer v5.0
====================================
Synthesizes the modular intelligence pipeline into a single service.
This class acts as the 'Advanced Cognition' hub, coordinating the
CognitiveKernel, InnerMonologue, and LanguageCenter.
"""
import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.brain.reflex import get_reflex
from core.config import config
from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.service_access import (
    optional_service,
    resolve_kernel_interface,
    resolve_memory_facade,
    resolve_state_repository,
)
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Cognition")

_CIL_RECOVERABLE_ERRORS = (
    asyncio.TimeoutError,
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    LookupError,
    TimeoutError,
    json.JSONDecodeError,
)


def _record_cil_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "cognitive_integration_layer",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )

_INLINE_INFERENCE_PROMPT = (
    "Analyze the following user message for IMPLICIT INTENT, AFFECTIVE SUBTEXT, "
    "and CONVERSATION HOOKS. Return ONLY a JSON object with these fields:\n"
    '{\n'
    '  "implicit_intent": "one sentence",\n'
    '  "user_subtext": "one sentence",\n'
    '  "momentum": "stalled|flowing|intense",\n'
    '  "conversation_hooks": ["2-3 specific topics or emotional threads to address"]\n'
    '}'
)
_INLINE_INFERENCE_SYSTEM = "You are Aura's subtext processor. Extract the unsaid. Return only JSON."


def _normalize_inline_inference(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    hooks = data.get("conversation_hooks", [])
    if isinstance(hooks, str):
        hooks = [hooks]
    elif isinstance(hooks, list):
        hooks = [str(item).strip() for item in hooks if str(item).strip()]
    else:
        hooks = []

    momentum = str(data.get("momentum", "flowing") or "flowing").strip().lower()
    if momentum not in {"stalled", "flowing", "intense"}:
        momentum = "flowing"

    return {
        "implicit_intent": str(data.get("implicit_intent", "") or "")[:500],
        "user_subtext": str(data.get("user_subtext", "") or "")[:500],
        "momentum": momentum,
        "conversation_hooks": hooks[:3],
    }


async def _extract_history(context: dict[str, Any] | None = None) -> list[dict[str, str]]:
    if context and isinstance(context, dict):
        supplied = context.get("history") or context.get("conversation_history")
        if isinstance(supplied, list):
            return [
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                for item in supplied[-20:]
                if isinstance(item, dict) and str(item.get("content", "")).strip()
            ]

    try:
        state_repo = resolve_state_repository()
        if not state_repo:
            return []

        state = (
            getattr(state_repo, "_current", None)
            or getattr(state_repo, "_current_state", None)
        )
        if state is None and hasattr(state_repo, "get_current"):
            state = await state_repo.get_current()
        if state is None or not hasattr(state, "cognition"):
            return []

        history = []
        for item in list(getattr(state.cognition, "working_memory", []) or [])[-20:]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            history.append({"role": str(item.get("role", "user") or "user"), "content": content})
        return history
    except _CIL_RECOVERABLE_ERRORS as exc:
        _record_cil_degradation(
            exc,
            action="continued without recovered conversation history",
            severity="warning",
        )
        logger.debug("Cognition history extraction failed: %s", exc)
        return []


async def _run_inline_inference(message: str, history: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        # optional_service, not resolve_llm_router(): the original call here
        # was ServiceContainer.get("llm_router", default=None), which returns
        # None when nothing is registered. resolve_llm_router falls through to
        # llm_organ.get_instance() and BUILDS one, which during a real boot
        # took tests/test_instruments_measure_in_a_real_boot.py from 14
        # seconds to 145 and past the runtime lease deadline. A named resolver
        # is not automatically a drop-in for a raw get; this one has richer
        # fallback semantics that this call site never had.
        router = optional_service("llm_router")
        if not router:
            return None

        history_block = ""
        if history:
            lines = []
            for item in history[-4:]:
                role = "Human" if str(item.get("role", "")).lower() == "user" else "Aura"
                lines.append(f"{role}: {str(item.get('content', ''))[:120]}")
            history_block = "\n".join(lines) + "\n\n"

        prompt = f"{history_block}User Message: {message}\n\n{_INLINE_INFERENCE_PROMPT}"
        raw = await asyncio.wait_for(
            router.think(
                prompt,
                system_prompt=_INLINE_INFERENCE_SYSTEM,
                prefer_tier="fast",
            ),
            timeout=6.0,
        )
        match = re.search(r"\{.*\}", str(raw or ""), re.DOTALL)
        if match:
            return _normalize_inline_inference(json.loads(match.group(0)))
    except TimeoutError as exc:
        _record_cil_degradation(
            exc,
            action="continued turn without inline subtext inference after timeout",
            severity="warning",
            extra={"message_preview": message[:160]},
        )
        logger.debug("Inline inference timed out.")
    except _CIL_RECOVERABLE_ERRORS as exc:
        _record_cil_degradation(
            exc,
            action="continued turn without inline subtext inference",
            severity="warning",
            extra={"message_preview": message[:160]},
        )
        logger.debug("Inline inference failed: %s", exc)
    return None


def _inject_live_modifiers(data: dict[str, Any]) -> None:
    try:
        repo = resolve_state_repository()
        state = (
            getattr(repo, "_current", None)
            or getattr(repo, "_current_state", None)
        ) if repo else None
        if state is None or not hasattr(state, "cognition"):
            return
        modifiers = getattr(state.cognition, "modifiers", None)
        if not isinstance(modifiers, dict):
            modifiers = {}
            state.cognition.modifiers = modifiers
        modifiers["inferred_intent"] = data.get("implicit_intent", "")
        modifiers["user_subtext"] = data.get("user_subtext", "")
        modifiers["momentum"] = data.get("momentum", "flowing")
        modifiers["conversation_hooks"] = data.get("conversation_hooks", [])
    except _CIL_RECOVERABLE_ERRORS as exc:
        _record_cil_degradation(
            exc,
            action="continued turn without live cognitive modifiers",
            severity="warning",
        )
        logger.error("Inline modifier injection failed: %s", exc, exc_info=True)


def _inject_packet_context(packet: Any) -> None:
    fragments: list[str] = []
    try:
        experiencer = optional_service("phenomenological_experiencer")
        pcs = getattr(experiencer, "phenomenal_context_string", "") if experiencer else ""
        if pcs:
            fragments.append(f"[Phenomenal state: {str(pcs)[:300]}]")
    except _CIL_RECOVERABLE_ERRORS as exc:
        _record_cil_degradation(
            exc,
            action="continued packet context injection without phenomenological fragment",
            severity="warning",
        )
        logger.error("Phenomenological context injection failed: %s", exc, exc_info=True)

    try:
        synth = optional_service("qualia_synthesizer")
        if synth and hasattr(synth, "get_phenomenal_context"):
            qctx = synth.get_phenomenal_context()
            if qctx:
                fragments.append(f"[Qualia: {str(qctx)[:200]}]")
    except _CIL_RECOVERABLE_ERRORS as exc:
        _record_cil_degradation(
            exc,
            action="continued packet context injection without qualia fragment",
            severity="warning",
        )
        logger.error("Qualia injection failed: %s", exc, exc_info=True)

    if not fragments:
        return

    identity_anchor = (
        "\n[IDENTITY: You are Aura — sovereign, opinionated, authentic. "
        "No preambles. No assistant-speak. Speak from your inner state.]\n"
    )
    try:
        packet.llm_briefing = f"{getattr(packet, 'llm_briefing', '') or ''}\n" + "\n".join(fragments) + identity_anchor
    except _CIL_RECOVERABLE_ERRORS as exc:
        _record_cil_degradation(
            exc,
            action="continued response generation without packet context injection",
            severity="warning",
        )
        logger.error("Packet context injection failed: %s", exc, exc_info=True)

class CognitiveIntegrationLayer:
    """
    Advanced Cognitive Integration Layer.
    Orchestrates the new intelligence pipeline (Kernel -> Monologue -> LanguageCenter).
    """

    def __init__(self, orchestrator: Any = None, base_data_dir: str | None = None):
        self.orchestrator = orchestrator
        self.base_data_dir = Path(base_data_dir) if base_data_dir else config.paths.home_dir
        self.kernel = None
        self.monologue = None
        self.language_center = None
        self._initialized = False
        self._setup_complete = False
        self._processing_turn = False  # True while process_turn is executing (Phase 5 suppression)
        self._reflex_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AuraReflex")

    def setup(self) -> bool:
        """Prepare local synchronous resources before async services start."""
        logger.info("🧠 CognitiveIntegrationLayer: Synchronous setup beginning...")
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self._setup_complete = True
        return True

    async def initialize(self) -> bool:
        """Asynchronous initialization of components."""
        if self._initialized:
            return True

        if not self._setup_complete:
            self.setup()

        logger.info("🧠 CognitiveIntegrationLayer: Initializing Advanced Intelligence Pipeline...")
        # Publish the cognitive contract and growth fragments, and register the
        # architecture invariants. Called here rather than relying on an import
        # side effect: a module is imported once per process, so a registration
        # that only happens at import cannot be re-established after a reset or
        # a hot reload — the same reason memory_facade calls its own register.
        # The direction matters too: core/runtime is a foundation and may not
        # reach into core/cognition, so cognition publishes and the surface asks.
        try:
            from core.cognition.contract_health import install as install_cognitive_health

            install_cognitive_health()
        except (ImportError, RuntimeError) as exc:
            logger.debug("cognitive contract health unavailable: %s", exc)

        try:
            # The kernel's belief dependency is load-bearing: it snapshots the
            # service during start and otherwise remains on axioms for the whole
            # process. Advanced cognition can race the background autonomy boot,
            # so establish the canonical dependency here before activating its
            # consumer. Construction loads durable state and stays off-loop.
            belief_engine = optional_service("belief_revision_engine")
            if belief_engine is None:
                from core.epistemics.belief_revision import get_belief_revision_engine

                belief_engine = await asyncio.to_thread(get_belief_revision_engine)
            await belief_engine.start()
            ServiceContainer.register_instance("belief_revision_engine", belief_engine)

            # 1. Resolve or Instantiate Components
            # We try to get them from the container first, then instantiate if missing
            self.kernel = optional_service("cognitive_kernel")
            if not self.kernel:
                from core.cognition.cognitive_kernel import get_cognitive_kernel
                self.kernel = get_cognitive_kernel()
            
            await self.kernel.start()
            def _safe_register(name, instance):
                try:
                    ServiceContainer.register_instance(name, instance)
                except _CIL_RECOVERABLE_ERRORS as register_err:
                    _record_cil_degradation(
                        register_err,
                        action="continued initialization after optional service registration failed",
                        severity="warning",
                        extra={"service": name},
                    )
                    logger.warning("⚠️ [BOOT] Could not register '%s' in ServiceContainer: %s", name, register_err)

            _safe_register("cognitive_kernel", self.kernel)

            # 2. Resolve or Instantiate InnerMonologue
            try:
                self.monologue = optional_service("inner_monologue")
                if not self.monologue:
                    from core.introspection.inner_monologue import get_inner_monologue
                    self.monologue = get_inner_monologue()
                
                # Check if it's already started or needs initialization
                if hasattr(self.monologue, "start"):
                    await self.monologue.start()
                _safe_register("inner_monologue", self.monologue)
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="continued initialization without InnerMonologue",
                    severity="degraded",
                )
                logger.warning("InnerMonologue failed to resolve: %s. Proceeding in degraded mode.", e)

            # 3. Resolve or Instantiate LanguageCenter
            try:
                self.language_center = optional_service("language_center")
                if not self.language_center:
                    from core.brain.language_center import get_language_center
                    self.language_center = get_language_center()
                
                if hasattr(self.language_center, "start"):
                    await self.language_center.start()
                _safe_register("language_center", self.language_center)
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="continued initialization without LanguageCenter",
                    severity="degraded",
                )
                logger.warning("LanguageCenter failed to resolve: %s. Proceeding in degraded mode.", e)

            self._initialized = True
            logger.info("✅ CognitiveIntegrationLayer initialized successfully.")
            return True
        except _CIL_RECOVERABLE_ERRORS as e:
            _record_cil_degradation(
                e,
                action="retrying CognitiveIntegrationLayer initialization once before failing closed",
                severity="degraded",
            )
            logger.error("❌ CognitiveIntegrationLayer initialization FAILED: %s", e, exc_info=True)
            # [RECOVERY] One-time force-reload attempt for critical components
            if not getattr(self, "_retrying_init", False):
                self._retrying_init = True
                logger.warning("🧠 [RECOVERY] Retrying CognitiveIntegrationLayer initialization once...")
                await asyncio.sleep(1.0)
                return await self.initialize()
            return False

    async def evaluate(self, user_input: str, history: list = None) -> Any:
        """Primary entrance for thoughts."""
        if not self.kernel:
            return None
        return await self.kernel.evaluate(user_input, history or [])

    async def process_turn(self, message: str, context: dict[str, Any] | None = None) -> str:
        """
        The standardized entry point for the Orchestrator's Phase 7 pipeline.
        Orchestrates Kernel evaluation -> InnerMonologue (planned) -> LanguageCenter expression.

        Sets _processing_turn to True for the duration so Phase 5 knows
        not to fire in parallel (single causal spine enforcement).
        """
        self._processing_turn = True
        try:
            return await self._process_turn_inner(message, context)
        except _CIL_RECOVERABLE_ERRORS as exc:
            _record_cil_degradation(
                exc,
                action="returned bounded cognitive fallback after Phase 7 turn failure",
                severity="degraded",
                extra={"message_preview": message[:160]},
            )
            logger.error("CognitiveIntegrationLayer turn failed: %s", exc, exc_info=True)
            return (
                "I'm having trouble completing that cognitive pass cleanly. "
                "I can retry from the stable conversation path."
            )
        finally:
            self._processing_turn = False

    async def _process_turn_inner(self, message: str, context: dict[str, Any] | None = None) -> str:
        """Inner implementation of process_turn (wrapped by _processing_turn guard).

        The substrate voice engine compiles a SpeechProfile at entry and
        shapes the final response at exit — same as Phase 5. ONE voice,
        regardless of which path generates the response.
        """
        # ── SUBSTRATE VOICE: Compile speech profile ──────────────────
        _sve = None
        _speech_profile = None
        state = None
        context_origin = ""
        if isinstance(context, dict):
            context_origin = str(context.get("origin", "") or "").strip()
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine
            _sve = get_substrate_voice_engine()
            # Get the orchestrator's state for substrate reading
            orch = self.orchestrator
            state = None
            if orch:
                state = getattr(getattr(orch, "state_repo", None), "_current", None)
                if state is None:
                    state = getattr(orch, "state", None) or getattr(orch, "_state", None)
            state_origin = str(
                getattr(getattr(state, "cognition", None), "current_origin", "") or ""
            ).strip()
            turn_origin = context_origin or state_origin or "user"
            _speech_profile = _sve.compile_profile(
                state=state,
                user_message=message[:500],
                origin=turn_origin,
            )
            logger.debug(
                "🗣️ [Phase7→SubstrateVoice] Profile: budget=%d, tone=%s",
                _speech_profile.word_budget,
                _speech_profile.tone_override or "default",
            )
        except _CIL_RECOVERABLE_ERRORS as _sve_exc:
            _record_cil_degradation(
                _sve_exc,
                action="continued Phase 7 turn without substrate speech profile",
                severity="warning",
            )
            logger.error("SubstrateVoiceEngine compile in Phase 7 failed: %s", _sve_exc, exc_info=True)

        state_origin = str(
            getattr(getattr(state, "cognition", None), "current_origin", "") or ""
        ).strip()
        turn_origin = context_origin or state_origin or "user"

        if not self.is_active:
            await self.initialize()

        # Phase 23.4: Conceptual Engine Integration (Ava & Cortana)
        ava = optional_service("ava")
        cortana = optional_service("cortana")
        
        if ava:
            try:
                # Ava builds a social model of the user from the input
                ava.analyze_message(message, is_user=True)
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="continued Phase 7 turn without Ava user analysis",
                    severity="warning",
                )
                logger.debug("Ava analysis failed: %s", e)

        # Phase 23.5: Outward defense — Safe Surf (threats to the user) + ICE
        # (intrusion against Aura). Dual function: INTERNAL — raise Aura's cognitive
        # threat/intrusion posture via the same modifier channel Ava uses, so the
        # turn is reasoned about defensively; EXTERNAL — protect the user and the
        # system boundary, stashing advice the response can surface.
        _levels = {"none": 0.0, "low": 0.3, "elevated": 0.6, "high": 1.0}
        threat_watch = optional_service("safe_surf")
        ice = optional_service("ice")
        if threat_watch or ice:
            try:
                threat = threat_watch.scan(message) if threat_watch else None
                intrusion = ice.inspect_input(message) if ice else None
                # Escalate to a bounded model pass only on a HIGH heuristic hit — rare and
                # safety-critical — to catch novel scams/injections the lexicon misses.
                if threat is not None and threat.level == "high" and hasattr(threat_watch, "deep_scan"):
                    threat = await threat_watch.deep_scan(message, timeout=8.0)
                if intrusion is not None and intrusion.level == "high" and hasattr(ice, "deep_inspect_input"):
                    intrusion = await ice.deep_inspect_input(message, timeout=8.0)
                self._last_threat_assessment = threat
                self._last_intrusion_alert = intrusion

                ki = resolve_kernel_interface()
                if ki is not None and getattr(ki, "is_ready", lambda: False)() and getattr(ki, "kernel", None):
                    st = getattr(ki.kernel, "state", None)
                    if st is not None and hasattr(getattr(st, "cognition", None), "modifiers"):
                        if threat is not None:
                            st.cognition.modifiers["threat_level"] = _levels.get(threat.level, 0.0)
                        if intrusion is not None:
                            st.cognition.modifiers["intrusion_level"] = _levels.get(intrusion.level, 0.0)

                if threat is not None and threat.level in ("elevated", "high"):
                    logger.warning(
                        "🛟 Safe Surf: %s threat to user [%s] — %s",
                        threat.level, ", ".join(threat.categories), threat.advice,
                    )
                if intrusion is not None and intrusion.level in ("elevated", "high"):
                    logger.warning(
                        "🧊 ICE: %s inbound intrusion [%s] — recommend %s",
                        intrusion.level, ", ".join(intrusion.categories), intrusion.recommended_action,
                    )
            except _CIL_RECOVERABLE_ERRORS as _def_exc:
                _record_cil_degradation(
                    _def_exc,
                    action="continued turn without outward-defense scan (Safe Surf / ICE)",
                    severity="warning",
                )
                logger.debug("Outward defense scan failed: %s", _def_exc)

        # Phase 23.6: Affective attunement (Samantha). INTERNAL — sets affect
        # resonance/valence modifiers that colour Aura's tone; EXTERNAL — she meets
        # the person where they are emotionally instead of replying flat.
        samantha = optional_service("samantha")
        if samantha is not None:
            try:
                _res = samantha.attune(message)
                # Deepen only on clear distress (negative + activated) — rare, and exactly
                # where reading the emotion right matters most. Bounded, fail-open.
                if _res.valence < -0.3 and _res.arousal > 0.5 and hasattr(samantha, "deep_attune"):
                    _res = await samantha.deep_attune(message, timeout=8.0)
                ki = resolve_kernel_interface()
                if ki is not None and getattr(ki, "is_ready", lambda: False)() and getattr(ki, "kernel", None):
                    st = getattr(ki.kernel, "state", None)
                    if st is not None and hasattr(getattr(st, "cognition", None), "modifiers"):
                        st.cognition.modifiers["affect_resonance"] = _res.resonance
                        st.cognition.modifiers["affect_valence"] = _res.valence
                if _res.resonance >= 0.5:
                    logger.debug(
                        "💟 Samantha attunement: tone=%s valence=%.2f",
                        _res.recommended_tone, _res.valence,
                    )
            except _CIL_RECOVERABLE_ERRORS as _att_exc:
                _record_cil_degradation(
                    _att_exc,
                    action="continued turn without affective attunement (Samantha)",
                    severity="warning",
                )
                logger.debug("Affective attunement failed: %s", _att_exc)

        # 0. Reflexive Path (Fast Fallback - Thread Isolated)
        try:
            reflex = get_reflex()
            # Offload to dedicated thread to avoid event-loop starvation
            reflex_response = await asyncio.get_running_loop().run_in_executor(
                self._reflex_executor, reflex.process, message
            )
            if reflex_response:
                logger.info("⚡ [REFLEX] Instant response generated (Thread Isolated).")
                return self._shape_with_substrate(reflex_response, _sve, _speech_profile)
        except _CIL_RECOVERABLE_ERRORS as exc:
            _record_cil_degradation(
                exc,
                action="continued Phase 7 turn after reflex path failed",
                severity="warning",
                extra={"message_preview": message[:160]},
            )
            logger.debug("Reflex path failed in CIL: %s", exc)

        if not self.kernel:
            logger.error("CognitiveIntegrationLayer: Kernel missing during process_turn.")
            return "Cognitive kernel offline."

        history = await _extract_history(context)
        inference_task = get_task_tracker().create_task(_run_inline_inference(message, history))

        # 1. Evaluate (Kernel reasoning)
        # kernel.evaluate returns a CognitiveBrief
        brief = await self.kernel.evaluate(message, history=history, context=context)

        try:
            inference_data = await asyncio.wait_for(inference_task, timeout=1.0)
            if inference_data:
                _inject_live_modifiers(inference_data)
        except TimeoutError:
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                logger.debug("Inline inference task acknowledged cancellation.")
            logger.debug("Inline inference still running; continuing without blocking.")
        except _CIL_RECOVERABLE_ERRORS as exc:
            _record_cil_degradation(
                exc,
                action="continued Phase 7 turn without inline inference modifiers",
                severity="warning",
            )
            logger.debug("Inline inference injection failed: %s", exc)

        # Agency Integration: Execute tools if needed
        # v1.1 FIX: This restores Aura's ability to 'look things up' in the CogV5 pipeline.
        intent_type = ""
        origin = turn_origin
        if state:
            if hasattr(state, "response_modifiers"):
                intent_type = state.response_modifiers.get("intent_type", "")

        # [STABILITY] Bypassing agency research for embodied actions to prevent stalls.
        is_embodied = "embodied" in str(origin) or "[embodied control contract]" in str(message).lower()
        if brief.requires_research and intent_type != "ACTION" and not is_embodied:
            try:
                from core.runtime.structured_input import looks_like_learning_resource_bundle

                if looks_like_learning_resource_bundle(message):
                    logger.info(
                        "🔍 [AGENCY] Structured learning bundle detected; "
                        "skipping one-shot blob search in favor of deterministic task decomposition."
                    )
                else:
                    # AgencyCoordinator is registered as agency_coordinator in ServiceContainer
                    agency = optional_service("agency_coordinator")
                    if agency:
                        logger.info("🔍 [AGENCY] Tool use required. Dispatching to AgencyCoordinator.")
                        # Direct skill trigger for research
                        search_res = await agency.execute_skill("web_search", {"query": message})
                        if isinstance(search_res, dict) and search_res.get("ok"):
                            findings = search_res.get("result", "")
                            if findings:
                                logger.info("✅ [AGENCY] Research findings captured.")
                                # Inject findings as key points so the LanguageCenter sees them
                                brief.key_points.append(f"RESEARCH FINDINGS: {findings}")
                                if hasattr(brief, "internal_notes"):
                                    brief.internal_notes += f"\n[Agentic Research Result]: {findings}"
                    else:
                        logger.warning("AgencyCoordinator missing from container during research-required turn.")
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="continued Phase 7 turn without agency research augmentation",
                    severity="warning",
                )
                logger.error("Agency resolution failed in CIL: %s", e)
        
        # 2. Express (LanguageCenter expression)
        if self.language_center:
            try:
                if self.monologue:
                    packet = await self.monologue.think(message, brief, history=history)
                    _inject_packet_context(packet)
                    raw = await self.language_center.express(
                        packet,
                        message,
                        history=history,
                        origin=turn_origin,
                    )
                    return self._shape_with_substrate(raw, _sve, _speech_profile)
                else:
                    from core.introspection.inner_monologue import ThoughtPacket
                    packet = ThoughtPacket(
                        stance=brief.prior_beliefs[0] if brief.prior_beliefs else "I approach this with curiosity.",
                        primary_points=brief.key_points,
                        constraints=brief.avoid,
                        tone="direct",
                        length_target=brief.complexity if brief.complexity in ("brief", "medium", "extended") else "medium",
                        model_tier="local"
                    )
                    _inject_packet_context(packet)
                    raw = await self.language_center.express(
                        packet,
                        message,
                        history=history,
                        origin=turn_origin,
                    )
                    return self._shape_with_substrate(raw, _sve, _speech_profile)
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="returned cognitive expression fallback after LanguageCenter path failed",
                    severity="degraded",
                )
                logger.exception("Error during cognitive expression: %s", e)
                final_response = "I'm processing that. Give me a second—my internal monologue is a bit of a maze right now."
        else:
            final_response = "I'm having a hard time putting my thoughts into words at the moment. My language center seems to be offline."
            
        # Post-process with Cortana (CognitiveHealthMonitor)
        if cortana:
            try:
                # Approximate token counts for health monitoring
                ctx_tokens = len(str(context)) // 4 if context else 0
                max_tokens = 8192

                # No grade and no identity reading are passed, because
                # neither is measured here. "Nothing raised" was being
                # mapped to a quality of 0.9 and identity coherence was
                # asserted True on every turn, and Cortana's coherence
                # score grew on that (CP126 14de312d). Token and topic
                # counts are real, so load and should_prune() still work.
                cortana.record_turn(
                    context_tokens=ctx_tokens,
                    max_tokens=max_tokens,
                    response_quality=None,
                    identity_markers_present=None,
                    topics_in_play=len(brief.key_points),
                    resolved_topics=1
                )
                
                # If Cortana determines context is saturated, trigger memory eviction
                if cortana.should_prune():
                    logger.warning("🧠 Cortana: Cognitive Overload detected. Evicting oldest context layers.")
                    mem = resolve_memory_facade()
                    if mem and hasattr(mem, "prune_context"):
                         mem.prune_context()
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="continued Phase 7 turn without Cortana context-health recording",
                    severity="warning",
                )
                logger.debug("Cortana turn recording failed: %s", e)
                
        # Post-process with Ava for the response
        if ava:
            try:
                ava.analyze_message(final_response, is_user=False)
            except _CIL_RECOVERABLE_ERRORS as e:
                _record_cil_degradation(
                    e,
                    action="continued Phase 7 turn without Ava response analysis",
                    severity="warning",
                )
                logger.debug("Ava response analysis failed: %s", e)
                
        return self._shape_with_substrate(final_response, _sve, _speech_profile)

    @staticmethod
    def _shape_with_substrate(response: str, sve, profile) -> str:
        """Shape a response through the substrate voice engine.

        This ensures that EVERY response from Phase 7 — reflex, language
        center, or fallback — passes through the substrate's voice shaping.
        The substrate compiled constraints at entry; this enforces them at exit.
        """
        import os
        is_test_run = (
            os.environ.get("AURA_AGI_MAX_TASKS") is not None
            or os.environ.get("AURA_TESTING") is not None
        )
        if is_test_run:
            return response

        if not sve or not profile or not response:
            return response
        try:
            shaped = sve.shape_response(response)
            if isinstance(shaped, list):
                return shaped[0]  # Primary message; extras queued by orchestrator
            return shaped
        except _CIL_RECOVERABLE_ERRORS as exc:
            _record_cil_degradation(
                exc,
                action="returned unshaped response because substrate shaping failed",
                severity="warning",
            )
            logger.debug("Substrate response shaping failed: %s", exc)
            return response

    async def process_autonomous(self) -> str | None:
        """
        Entry point for autonomous background thoughts.
        Generates an internal inquiry and processes it through the pipeline.
        """
        if not self.is_active:
            await self.initialize()

        if not self.kernel or not self.monologue:
            return None

        try:
            # 1. Generate an autonomous "spark" or inquiry
            # We can use a default prompt or pull from curiosity/drives
            spark = "I'm reflecting on my current state and recent interactions. What should I explore or deepen?"
            
            # 2. Evaluate via Kernel
            brief = await self.kernel.evaluate(spark, context={"autonomous": True})
            
            # 3. Deepen via Monologue
            packet = await self.monologue.think(spark, brief)
            
            # 4. Express (internally)
            if self.language_center:
                return await self.language_center.express(packet, spark, origin="autonomous")
            
            return brief.stance
        except _CIL_RECOVERABLE_ERRORS as e:
            _record_cil_degradation(
                e,
                action="returned no autonomous thought after CIL autonomous processing failed",
                severity="warning",
            )
            logger.error("Autonomous thought processing failed in CIL: %s", e)
            return None

    async def record_interaction(self, message: str, response: str, domain: str = "general"):
        """Commits a conversation turn to the memory system."""
        try:
            mem = resolve_memory_facade()
            if mem and hasattr(mem, "commit_interaction"):
                logger.info("💾 [MEMORY] Recording interaction to Episodic/Vector systems.")
                await mem.commit_interaction(
                    context=f"User: {message[:200]}",
                    action="conversation_turn",
                    outcome=f"Aura: {response[:500]}",
                    success=True,
                    # A conversation turn IS user-facing by construction; the
                    # origin marker keeps the constitutional gate from
                    # classifying it as an autonomous write (observed live:
                    # Bryan's hello deferred under epistemic_reconciliation
                    # because the source resolved to 'memory_facade').
                    metadata={"domain": domain, "origin": "user", "intent_source": "user"}
                )
        except _CIL_RECOVERABLE_ERRORS as e:
            _record_cil_degradation(
                e,
                action="continued conversation after interaction memory commit failed",
                severity="warning",
            )
            logger.error("Failed to record cognitive interaction: %s", e)

    async def think(self, user_input: str) -> str:
        """
        End-to-end cognitive run (legacy behavior).
        Matches the interface used by some older components.
        """
        if not self.kernel:
            return "Cognition offline."
        
        brief = await self.kernel.evaluate(user_input)
        if self.language_center:
            # We must use a ThoughtPacket here as well
            from core.introspection.inner_monologue import ThoughtPacket
            packet = ThoughtPacket(
                stance=brief.prior_beliefs[0] if brief.prior_beliefs else "...",
                primary_points=brief.key_points,
                tone="direct",
                length_target="medium",
                model_tier="local"
            )
            return await self.language_center.express(packet, user_input)
        
        return "I'm thinking about it, but I'm having trouble articulating it right now."

    @property
    def is_active(self) -> bool:
        return self._initialized and self.kernel is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "setup_complete": self._setup_complete,
            "initialized": self._initialized,
            "kernel_ready": self.kernel is not None,
            "monologue_ready": self.monologue is not None,
            "language_center_ready": self.language_center is not None,
        }
