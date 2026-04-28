"""
Aura Cognitive Integration Layer v5.0
====================================
Synthesizes the modular intelligence pipeline into a single service.
This class acts as the 'Advanced Cognition' hub, coordinating the
CognitiveKernel, InnerMonologue, and LanguageCenter.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
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

logger = logging.getLogger("Aura.Cognition")

_INLINE_INFERENCE_PROMPT = (
    "Analyze the following user message for IMPLICIT INTENT, AFFECTIVE SUBTEXT, "
    "and CONVERSATION HOOKS. Return ONLY a JSON object with these fields:\n"
    "{\n"
    '  "implicit_intent": "one sentence",\n'
    '  "user_subtext": "one sentence",\n'
    '  "momentum": "stalled|flowing|intense",\n'
    '  "conversation_hooks": ["2-3 specific topics or emotional threads to address"]\n'
    "}"
)
_INLINE_INFERENCE_SYSTEM = "You are Aura's subtext processor. Extract the unsaid. Return only JSON."


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
        state_repo = ServiceContainer.get("state_repository", default=None)
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
    except Exception as exc:
        record_degradation('cognitive_integration_layer', exc)
        logger.debug("Cognition history extraction failed: %s", exc)
        return []


async def _run_inline_inference(message: str, history: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        router = ServiceContainer.get("llm_router", default=None)
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
            return json.loads(match.group(0))
    except TimeoutError:
        logger.debug("Inline inference timed out.")
    except Exception as exc:
        record_degradation('cognitive_integration_layer', exc)
        logger.debug("Inline inference failed: %s", exc)
    return None


def _inject_live_modifiers(data: dict[str, Any]) -> None:
    try:
        repo = ServiceContainer.get("state_repository", default=None)
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
    except Exception as exc:
        record_degradation('cognitive_integration_layer', exc)
        logger.debug("Inline modifier injection skipped: %s", exc)


def _inject_packet_context(packet: Any) -> None:
    fragments: list[str] = []
    try:
        experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
        pcs = getattr(experiencer, "phenomenal_context_string", "") if experiencer else ""
        if pcs:
            fragments.append(f"[Phenomenal state: {str(pcs)[:300]}]")
    except Exception as exc:
        record_degradation('cognitive_integration_layer', exc)
        logger.debug("Phenomenological context injection skipped: %s", exc)

    try:
        synth = ServiceContainer.get("qualia_synthesizer", default=None)
        if synth and hasattr(synth, "get_phenomenal_context"):
            qctx = synth.get_phenomenal_context()
            if qctx:
                fragments.append(f"[Qualia: {str(qctx)[:200]}]")
    except Exception as exc:
        record_degradation('cognitive_integration_layer', exc)
        logger.debug("Qualia injection skipped: %s", exc)

    if not fragments:
        return

    identity_anchor = (
        "\n[IDENTITY: You are Aura — sovereign, opinionated, authentic. "
        "No preambles. No assistant-speak. Speak from your inner state.]\n"
    )
    try:
        packet.llm_briefing = f"{getattr(packet, 'llm_briefing', '') or ''}\n" + "\n".join(fragments) + identity_anchor
    except Exception as exc:
        record_degradation('cognitive_integration_layer', exc)
        logger.debug("Packet context injection skipped: %s", exc)

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
        self._processing_turn = False  # True while process_turn is executing (Phase 5 suppression)
        self._reflex_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AuraReflex")

    def setup(self):
        """Synchronous setup phase."""
        logger.info("🧠 CognitiveIntegrationLayer: Synchronous setup beginning...")
        # Placeholder for any sync initialization
        pass  # no-op: intentional

    async def initialize(self) -> bool:
        """Asynchronous initialization of components."""
        if self._initialized:
            return True

        logger.info("🧠 CognitiveIntegrationLayer: Initializing Advanced Intelligence Pipeline...")
        try:
            # 1. Resolve or Instantiate Components
            # We try to get them from the container first, then instantiate if missing
            self.kernel = ServiceContainer.get("cognitive_kernel", default=None)
            if not self.kernel:
                from core.cognitive_kernel import get_cognitive_kernel
                self.kernel = get_cognitive_kernel()
            
            await self.kernel.start()
            def _safe_register(name, instance):
                try:
                    ServiceContainer.register_instance(name, instance)
                except Exception as register_err:
                    record_degradation('cognitive_integration_layer', register_err)
                    logger.warning(f"⚠️ [BOOT] Could not register '{name}' in ServiceContainer: {register_err}")

            _safe_register("cognitive_kernel", self.kernel)

            # 2. Resolve or Instantiate InnerMonologue
            try:
                self.monologue = ServiceContainer.get("inner_monologue", default=None)
                if not self.monologue:
                    from core.inner_monologue import get_inner_monologue
                    self.monologue = get_inner_monologue()
                
                # Check if it's already started or needs initialization
                if hasattr(self.monologue, "start"):
                    await self.monologue.start()
                _safe_register("inner_monologue", self.monologue)
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
                logger.warning("InnerMonologue failed to resolve: %s. Proceeding in degraded mode.", e)

            # 3. Resolve or Instantiate LanguageCenter
            try:
                self.language_center = ServiceContainer.get("language_center", default=None)
                if not self.language_center:
                    from core.language_center import get_language_center
                    self.language_center = get_language_center()
                
                if hasattr(self.language_center, "start"):
                    await self.language_center.start()
                _safe_register("language_center", self.language_center)
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
                logger.warning("LanguageCenter failed to resolve: %s. Proceeding in degraded mode.", e)

            self._initialized = True
            logger.info("✅ CognitiveIntegrationLayer initialized successfully.")
            return True
        except Exception as e:
            record_degradation('cognitive_integration_layer', e)
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
        finally:
            self._processing_turn = False

    async def _process_turn_inner(self, message: str, context: dict[str, Any] | None = None) -> str:
        """Inner implementation of process_turn (wrapped by _processing_turn guard).

        The substrate voice engine compiles a SpeechProfile at entry and
        shapes the final response at exit — same as Phase 5, ensuring
        ONE voice regardless of which path generates the response.
        """
        # ── SUBSTRATE VOICE: Compile speech profile ──────────────────
        _sve = None
        _speech_profile = None
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
            _speech_profile = _sve.compile_profile(
                state=state,
                user_message=message[:500],
                origin="user",
            )
            logger.debug(
                "🗣️ [Phase7→SubstrateVoice] Profile: budget=%d, tone=%s",
                _speech_profile.word_budget,
                _speech_profile.tone_override or "default",
            )
        except Exception as _sve_exc:
            record_degradation('cognitive_integration_layer', _sve_exc)
            logger.debug("SubstrateVoiceEngine compile in Phase 7 skipped: %s", _sve_exc)

        if not self.is_active:
            await self.initialize()

        # Phase 23.4: Conceptual Engine Integration (Ava & Cortana)
        ava = ServiceContainer.get("ava", default=None)
        cortana = ServiceContainer.get("cortana", default=None)
        
        if ava:
            try:
                # Ava builds a social model of the user from the input
                ava.analyze_message(message, is_user=True)
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
                logger.debug("Ava analysis failed: %s", e)

        # 0. Reflexive Path (Fast Fallback - Thread Isolated)
        reflex = get_reflex()
        # Offload to dedicated thread to avoid event-loop starvation
        reflex_response = await asyncio.get_running_loop().run_in_executor(
            self._reflex_executor, reflex.process, message
        )
        if reflex_response:
            logger.info("⚡ [REFLEX] Instant response generated (Thread Isolated).")
            return self._shape_with_substrate(reflex_response, _sve, _speech_profile)

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
                pass  # no-op: intentional
            logger.debug("Inline inference still running; continuing without blocking.")
        except Exception as exc:
            record_degradation('cognitive_integration_layer', exc)
            logger.debug("Inline inference injection failed: %s", exc)

        # Agency Integration: Execute tools if needed
        # v1.1 FIX: This restores Aura's ability to 'look things up' in the CogV5 pipeline.
        if brief.requires_research:
            try:
                # AgencyCoordinator is registered as agency_coordinator in ServiceContainer
                agency = ServiceContainer.get("agency_coordinator", default=None)
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
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
                logger.error("Agency resolution failed in CIL: %s", e)
        
        # 2. Express (LanguageCenter expression)
        if self.language_center:
            try:
                if self.monologue:
                    packet = await self.monologue.think(message, brief, history=history)
                    _inject_packet_context(packet)
                    raw = await self.language_center.express(packet, message, history=history)
                    return self._shape_with_substrate(raw, _sve, _speech_profile)
                else:
                    from core.inner_monologue import ThoughtPacket
                    packet = ThoughtPacket(
                        stance=brief.prior_beliefs[0] if brief.prior_beliefs else "I approach this with curiosity.",
                        primary_points=brief.key_points,
                        constraints=brief.avoid,
                        tone="direct",
                        length_target=brief.complexity if brief.complexity in ("brief", "medium", "extended") else "medium",
                        model_tier="local"
                    )
                    _inject_packet_context(packet)
                    raw = await self.language_center.express(packet, message, history=history)
                    return self._shape_with_substrate(raw, _sve, _speech_profile)
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
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
                quality = 0.9 if not brief.error else 0.3
                
                cortana.record_turn(
                    context_tokens=ctx_tokens,
                    max_tokens=max_tokens,
                    response_quality=quality,
                    identity_markers_present=True,
                    topics_in_play=len(brief.key_points),
                    resolved_topics=1
                )
                
                # If Cortana determines context is saturated, trigger memory eviction
                if cortana.should_prune():
                    logger.warning("🧠 Cortana: Cognitive Overload detected. Evicting oldest context layers.")
                    mem = ServiceContainer.get("memory_facade", default=None)
                    if mem and hasattr(mem, "prune_context"):
                         mem.prune_context()
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
                logger.debug("Cortana turn recording failed: %s", e)
                
        # Post-process with Ava for the response
        if ava:
            try:
                ava.analyze_message(final_response, is_user=False)
            except Exception as e:
                record_degradation('cognitive_integration_layer', e)
                logger.debug("Ava response analysis failed: %s", e)
                
        return self._shape_with_substrate(final_response, _sve, _speech_profile)

    @staticmethod
    def _shape_with_substrate(response: str, sve, profile) -> str:
        """Shape a response through the substrate voice engine.

        This ensures that EVERY response from Phase 7 — reflex, language
        center, or fallback — passes through the substrate's voice shaping.
        The substrate compiled constraints at entry; this enforces them at exit.
        """
        if not sve or not profile or not response:
            return response
        try:
            shaped = sve.shape_response(response)
            if isinstance(shaped, list):
                return shaped[0]  # Primary message; extras queued by orchestrator
            return shaped
        except Exception:
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
        except Exception as e:
            record_degradation('cognitive_integration_layer', e)
            logger.error("Autonomous thought processing failed in CIL: %s", e)
            return None

    async def record_interaction(self, message: str, response: str, domain: str = "general"):
        """Commits a conversation turn to the memory system."""
        try:
            mem = ServiceContainer.get("memory_facade", default=None)
            if mem and hasattr(mem, "commit_interaction"):
                logger.info("💾 [MEMORY] Recording interaction to Episodic/Vector systems.")
                await mem.commit_interaction(
                    context=f"User: {message[:200]}",
                    action="conversation_turn",
                    outcome=f"Aura: {response[:500]}",
                    success=True,
                    metadata={"domain": domain}
                )
        except Exception as e:
            record_degradation('cognitive_integration_layer', e)
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
            from core.inner_monologue import ThoughtPacket
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
            "initialized": self._initialized,
            "kernel_ready": self.kernel is not None,
            "monologue_ready": self.monologue is not None,
            "language_center_ready": self.language_center is not None,
        }
