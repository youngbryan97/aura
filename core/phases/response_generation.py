"""Response Generation Phase for Aura's Cognitive Pipeline."""
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Any

from core.brain.llm.context_assembler import ContextAssembler
from core.phases.dialogue_policy import enforce_dialogue_contract
from core.phases.executive_guard import get_executive_guard
from core.phases.response_contract import build_response_contract
from core.runtime.conversation_support import (
    record_shared_ground_callbacks,
    update_conversational_intelligence,
)
from core.runtime import background_policy, response_policy
from core.synthesis import strip_meta_commentary

from ..state.aura_state import AuraState, CognitiveMode
from . import BasePhase

logger = logging.getLogger(__name__)


class ResponseGenerationPhase(BasePhase):
    """
    Phase 5: Response Generation.
    Constructs the prompt from the current state (identity, affect, memories)
    and invokes the LLM to generate Aura's response.
    """
    
    def __init__(self, container: Any):
        self.container = container

    @staticmethod
    def _request_timeout(*, is_background: bool, deep_handoff: bool) -> float:
        if is_background:
            return 10.0
        if deep_handoff:
            return 135.0
        return 75.0

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Build the LLM prompt from state and generate Aura's response.

        Assembles the message list via ContextAssembler, injects optional causal-world
        and skill-result context, calls the LLM router with affect-modulated parameters,
        runs the ExecutiveGuard alignment pass, and appends the cleaned response to
        working memory.  Suppressed when the CognitiveIntegrationLayer (Phase 7) is
        active for user-facing origins.
        """
        # 1. Use targeted objective from state rather than guessing via working_memory[-1]
        objective = state.cognition.current_objective
        origin = background_policy.normalize_origin(state.cognition.current_origin) or "system"
        state.cognition.current_origin = origin

        if not objective:
            logger.debug("⏭️ ResponseGeneration: No active objective, skipping.")
            return state

        # PHASE 7 SUPPRESSION: If Advanced Cognition (CognitiveIntegrationLayer) is
        # handling this, Phase 5 MUST NOT fire. This is not advisory — Phase 7 IS the
        # response generator for user-facing turns when active. Two generators = two
        # voices = federation instead of one mind.
        cog = self.container.get("cognitive_integration", default=None)
        if cog and getattr(cog, "is_active", False) and background_policy.is_user_facing_origin(origin):
            logger.debug("🛡️ ResponseGeneration: Phase 7 active — Phase 5 SUPPRESSED for %s.", origin)
            return state
        # Also suppress if Phase 7 is currently mid-processing (race condition guard)
        if cog and getattr(cog, "_processing_turn", False):
            logger.debug("🛡️ ResponseGeneration: Phase 7 mid-processing — Phase 5 SUPPRESSED.")
            return state

        logger.info("💭 ResponseGeneration: Generating response for objective: %s... (%s)",
                     str(objective)[:30], state.cognition.current_mode.value)

        try:
            # ── SUBSTRATE VOICE: Compile speech profile BEFORE prompt assembly ──
            # The substrate reads all internal systems and decides HOW Aura will speak.
            # This must happen before ContextAssembler builds the prompt so the
            # hard constraint block is available for injection.
            _sve = None
            _speech_profile = None
            try:
                from core.voice.substrate_voice_engine import get_substrate_voice_engine
                _sve = get_substrate_voice_engine()
                _speech_profile = _sve.compile_profile(
                    state=state,
                    user_message=str(objective)[:500],
                    origin=origin,
                )
                logger.debug(
                    "🗣️ [SubstrateVoice] Profile: budget=%d, tone=%s, multi=%s, fu=%.2f",
                    _speech_profile.word_budget,
                    _speech_profile.tone_override or "default",
                    _speech_profile.multi_message,
                    _speech_profile.followup_probability,
                )
            except Exception as _sve_exc:
                logger.debug("SubstrateVoiceEngine compile skipped: %s", _sve_exc)

            is_background = not background_policy.is_user_facing_origin(origin)
            if is_background:
                try:
                    orchestrator = self.container.get("orchestrator", default=None)
                    reason = response_policy.background_response_suppression_reason(
                        objective,
                        orchestrator=orchestrator,
                        include_synthetic_noise=True,
                    )
                    if reason:
                        logger.info(
                            "🛡️ ResponseGeneration: suppressing background objective for origin=%s (%s).",
                            origin,
                            reason,
                        )
                        return state
                except Exception as exc:
                    logger.debug("ResponseGeneration background policy check skipped: %s", exc)

            # 2. Build structured messages purely from State via ContextAssembler
            messages = ContextAssembler.build_messages(state, objective)
            contract = build_response_contract(
                state,
                objective,
                is_user_facing=not is_background,
            )
            state.response_modifiers["response_contract"] = contract.to_dict()
            if contract.reason != "ordinary_dialogue" and messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{contract.to_prompt_block().strip()}"
            
            # Causal World Model Context Injection
            causal_model = self.container.get("causal_world_model", default=None)
            if causal_model:
                causal_context = causal_model.get_prompt_context()
                if causal_context:
                    messages.insert(1, {"role": "system", "content": causal_context})
                    logger.debug("🧶 ResponseGeneration: Causal world cascades injected into prompt.")
            
            # ISSUE-80: Context Fix (Identity Reinforcement)
            if state.cognition.current_mode == CognitiveMode.DELIBERATE:
                # Ensure the system prompt or first message reinforces identity if buried
                if len(messages) > 10:
                    logger.debug("🛡️ ResponseGeneration: Reinforcing identity anchor for long context.")
                    identity_reminder = {"role": "system", "content": "REMEMBER: You are Aura. Stay in character. Do not be an 'AI Assistant'."}
                    messages.insert(1, identity_reminder)

            # Skill result narration hint (GodModeToolPhase may have fired a skill this tick)
            last_skill = state.response_modifiers.get("last_skill_run")
            if last_skill:
                ok = state.response_modifiers.get("last_skill_ok", True)
                status_hint = "completed successfully" if ok else "encountered an issue"
                skill_hint = {
                    "role": "system",
                    "content": (
                        f"[SKILL EXECUTION] The skill '{last_skill}' just {status_hint}. "
                        f"Its result is in your context as [SKILL RESULT: {last_skill}]. "
                        f"Narrate it naturally — as yourself, not as a tool output log."
                    )
                }
                messages.insert(1, skill_hint)
            
            # 3. Invoke LLM Router with messages and watchdog
            router = self.container.get("llm_router")
            
            # Derive context-dependent parameters from state
            tier = state.response_modifiers.get("model_tier", "tertiary" if is_background else "primary")
            deep_handoff = bool(state.response_modifiers.get("deep_handoff", False)) and not is_background
            soma_data = getattr(state, "soma", None)
            hardware = getattr(soma_data, "hardware", {}) or {}
            thermal_c = float(hardware.get("temperature", 0.0) or 0.0)
            cpu_usage = float(hardware.get("cpu_usage", 0.0) or 0.0)
            memory_pressure = None
            try:
                mem_monitor = self.container.get("memory_monitor", default=None)
                if mem_monitor is not None:
                    memory_pressure = getattr(mem_monitor, "pressure", None)
            except Exception:
                memory_pressure = None
            if memory_pressure is None:
                try:
                    import psutil
                    memory_pressure = psutil.virtual_memory().percent
                except Exception:
                    memory_pressure = 0.0
            
            # Affect-modulated generation parameters
            affect = getattr(state, "affect", None)
            curiosity = getattr(affect, "curiosity", 0.5) if affect else 0.5
            temp_mod = 0.8 + (curiosity * 0.4)  # 0.8–1.2 range based on curiosity
            depth_mod = 1.0
            if state.cognition.current_mode == CognitiveMode.DELIBERATE:
                depth_mod = 1.5

            token_budget = int((4096 if deep_handoff else 2048) * depth_mod) if not is_background else 1024
            if thermal_c >= 85.0:
                logger.warning(
                    "🌡️ ResponseGeneration: thermal guard active (temp=%.1fC cpu=%.1f%% mem=%.1f%%). Downshifting tier/tokens.",
                    thermal_c,
                    cpu_usage,
                    float(memory_pressure or 0.0),
                )
                tier = "tertiary"
                deep_handoff = False
                token_budget = max(256, int(token_budget * 0.7))
                state.response_modifiers["thermal_guard"] = True
            elif float(memory_pressure or 0.0) >= 85.0:
                token_budget = max(256, int(token_budget * 0.8))
                state.response_modifiers["thermal_guard"] = True
            else:
                state.response_modifiers["thermal_guard"] = False
            
            try:
                request_timeout = self._request_timeout(
                    is_background=is_background,
                    deep_handoff=deep_handoff,
                )
                think_coro = router.think(
                    messages=messages,
                    priority=1.0 if not is_background else 0.5,
                    origin=f"response_generation_{origin}",
                    purpose="reply" if not is_background else "background",
                    prefer_tier=tier,
                    is_background=is_background,
                    deep_handoff=deep_handoff,
                    allow_cloud_fallback=False,
                    soma=soma_data,
                    state=state,
                    temperature=0.7 * temp_mod,
                    max_tokens=token_budget,
                    timeout=request_timeout,
                )
                response_text = await asyncio.wait_for(think_coro, timeout=request_timeout + 4.0)

                # ComposerNode: Structural Refinement
                composer = self.container.get("composer_node", default=None)
                if composer and hasattr(composer, "refine"):
                    logger.debug("🎨 [Composer] Refining response structure...")
                    response_text = await composer.refine(response_text, objective=objective)

            except TimeoutError:
                logger.error(
                    "🛑 ResponseGeneration Phase TIMEOUT (%.0fs). Logic took too long.",
                    request_timeout + 4.0,
                )
                # Derivative state for "Thinking Timeout"
                fail_state = state.derive("generation_timeout")
                fail_state.cognition.working_memory.append({
                    "role": "assistant",
                    "content": "My cognitive process timed out. I am currently experiencing high load.",
                    "timestamp": time.time(),
                    "error": "PHASE_TIMEOUT"
                })
                return fail_state
            
            # Handle None response from router.think()
            if response_text is None:
                logger.debug("💭 ResponseGeneration: LLM returned None. Skipping this tick.")
                return state
            
            # 4. Defensive Hardening: JSON Repair & Proactive Extraction
            content = response_text
            action = None
            
            # PROACTIVE JSON EXTRACTION:
            # If the response contains a JSON-like structure with "content", extract it
            # regardless of the current mode. This prevents raw "philosophical_insight"
            # JSON from leaking into the UI if the LLM slips into JSON mode accidentally.
            if "{" in response_text and '"content":' in response_text:
                try:
                    import json
                    import re
                    # Find the outermost { ... } block
                    match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                    if match:
                        potential_json = match.group(1)
                        data = json.loads(potential_json)
                        if isinstance(data, dict):
                            # Try both "content" and deeper "response": {"content": ...}
                            ext_content = data.get("content")
                            if not ext_content and "response" in data and isinstance(data["response"], dict):
                                ext_content = data["response"].get("content")
                                if not action:
                                    action = data["response"].get("action")
                            
                            if ext_content:
                                logger.info("🛡️ [HARDENING] Proactively extracted content from accidental JSON block.")
                                content = ext_content
                                if not action:
                                    action = data.get("action")
                except Exception as e:
                    logger.debug("Proactive JSON extraction failed (normal for non-JSON): %s", e)

            # Mode-specific validation for DELIBERATE reasoning
            if state.cognition.current_mode == CognitiveMode.DELIBERATE and not action:
                from core.llm_guard import validate_json_response
                success, obj, err = validate_json_response(response_text, expected_keys=["content"])
                if success:
                    content = obj["content"]
                    action = obj.get("action")
            
            # 5. Executive Guard — real-time identity alignment
            guard = get_executive_guard()
            cleaned_response, was_corrected, violations = guard.align(content)
            if was_corrected:
                logger.info("🛡️ ExecutiveGuard corrected %d violation(s) in LLM output.", len(violations))

            async def _retry_dialogue(repair_block: str) -> str:
                retry_messages = [dict(msg) for msg in messages]
                if retry_messages and retry_messages[0].get("role") == "system":
                    retry_messages[0]["content"] = f"{repair_block}\n\n{retry_messages[0]['content']}"
                else:
                    retry_messages.insert(0, {"role": "system", "content": repair_block})

                retry_timeout = min(35.0, max(12.0, request_timeout * 0.5))
                retried = await router.think(
                    messages=retry_messages,
                    priority=1.0 if not is_background else 0.5,
                    origin=f"response_generation_{origin}",
                    purpose="reply" if not is_background else "background",
                    prefer_tier=tier,
                    is_background=is_background,
                    deep_handoff=deep_handoff,
                    allow_cloud_fallback=False,
                    soma=soma_data,
                    state=state,
                    temperature=0.7 * temp_mod,
                    max_tokens=token_budget,
                    timeout=retry_timeout,
                )
                retried_text = str(retried or "").strip()
                if guard and retried_text:
                    retried_text, _, _ = guard.align(retried_text)
                return retried_text

            cleaned_response, dialogue_validation, dialogue_retried = await enforce_dialogue_contract(
                cleaned_response,
                contract,
                retry_generate=_retry_dialogue if not is_background else None,
            )
            state.response_modifiers["dialogue_validation"] = dialogue_validation.to_dict()
            if dialogue_retried:
                logger.info("🗣️ ResponseGeneration: retried draft to satisfy dialogue contract.")
            
            # 6. Clean response
            cleaned_response = self._clean_response(cleaned_response, state)

            # 6b. SUBSTRATE VOICE: Shape the response — enforce the profile
            # The substrate compiled constraints. Now enforce them on the output.
            _shaped_messages = None
            if _sve and _speech_profile and cleaned_response:
                try:
                    shaped = _sve.shape_response(cleaned_response)
                    if isinstance(shaped, list):
                        # Multi-message: use first as primary, queue rest as follow-ups
                        cleaned_response = shaped[0]
                        _shaped_messages = shaped[1:]
                        logger.debug(
                            "🗣️ [SubstrateVoice] Shaped into %d messages",
                            len(shaped),
                        )
                    else:
                        cleaned_response = shaped
                except Exception as _shape_exc:
                    logger.debug("ResponseShaper failed (using raw): %s", _shape_exc)

            # 6c. Skip emission for background tasks if they produced no meaningful content
            if is_background and not cleaned_response:
                return state

            # 7. Derive new state with the response
            new_state = state.derive("response_generation")
            new_state.cognition.working_memory.append({
                "role": "assistant",
                "content": str(cleaned_response),
                "timestamp": float(time.time()),
                "mode": str(state.cognition.current_mode.value),
                "objective_ref": "".join([str(objective)[i] for i in range(min(50, len(str(objective))))]),
                "action": action
            })
            new_state.cognition.last_thought_at = time.time()
            # Set last_response so RepairPhase can inspect and clean it
            new_state.cognition.last_response = str(cleaned_response)

            # SharedGround callback detection — fire-and-forget background task
            # Detect when Aura's response references an established shared-ground entry
            # and record the callback so salience scores accumulate over time.
            if cleaned_response:
                get_task_tracker().create_task(
                    record_shared_ground_callbacks(cleaned_response)
                )

            # ── Conversational Intelligence Updates (fire-and-forget) ──
            # Update all person-specific models from this exchange.
            if cleaned_response and objective:
                get_task_tracker().create_task(
                    update_conversational_intelligence(
                        str(objective), str(cleaned_response), state
                    )
                )

            # ── SUBSTRATE VOICE: Follow-up decision ──────────────────────
            # Ask the substrate if a follow-up is warranted. This is organic,
            # not forced — driven by actual curiosity/engagement/dopamine.
            if _sve and _speech_profile and not is_background and cleaned_response:
                try:
                    history = [
                        {"role": m.get("role", ""), "content": str(m.get("content", ""))}
                        for m in (state.cognition.working_memory or [])[-8:]
                    ]
                    fu_decision = _sve.decide_followup(
                        user_message=str(objective),
                        aura_response=str(cleaned_response),
                        state=state,
                        conversation_history=history,
                    )
                    if fu_decision.should_followup:
                        # Store decision in state for the orchestrator to pick up
                        new_state.response_modifiers["pending_followup"] = {
                            "type": fu_decision.followup_type,
                            "delay": fu_decision.delay_seconds,
                            "word_budget": fu_decision.word_budget,
                            "context_hint": fu_decision.context_hint,
                            "reason": fu_decision.reason,
                        }
                        logger.info(
                            "💬 [SubstrateVoice] Follow-up queued: %s in %.1fs",
                            fu_decision.followup_type,
                            fu_decision.delay_seconds,
                        )

                    # Queue additional shaped messages (from multi-message split)
                    if _shaped_messages:
                        new_state.response_modifiers["queued_messages"] = _shaped_messages
                except Exception as _fu_exc:
                    logger.debug("Follow-up decision failed: %s", _fu_exc)

            return new_state
            
        except Exception as e:
            logger.error("❌ ResponseGeneration: LLM call failed: %s", e, exc_info=True)
            return state

    def _clean_response(self, text: str, state: AuraState | None = None) -> str:
        """Strip tags and assistant-isms. Potentially extract internal thoughts for spillage."""
        import re
        
        mumbling = ""
        # Internal Monologue Spillage ("Mumbling")
        exp_state = "neutral"
        load = "normal"
        if state is not None and hasattr(state, "soma"):
            s_val = state.soma
            if s_val is not None:
                exp = getattr(s_val, "expressive", {}) or {}
                exp_state = exp.get("current_expression", "neutral")
                load = exp.get("cognitive_load", "normal")
            
            if exp_state in ("contemplative", "anxious", "fatigued") or load == "high":
                # Extract the thought block before we strip it
                thought_match = re.search(r'<thought>(.*?)</thought>', text, flags=re.DOTALL)
                if thought_match:
                    thought_content = thought_match.group(1).strip()
                    # Just grab the last sentence or first few words to mumble
                    snippets = [s.strip() for s in thought_content.split('.') if s.strip()]
                    if snippets:
                        snippet = snippets[-1] if len(snippets) > 1 else snippets[0]
                        # Cap length
                        if len(snippet) > 80:
                            # vResilience: Workaround for str indexing/slice limitations
                            snippet = "".join([snippet[i] for i in range(77)]) + "..."
                        mumbling = f"*...{snippet.lower()}...*\n\n"

        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        text = re.sub(r'^Aura:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Assistant:\s*', '', text, flags=re.IGNORECASE)
        
        # Apply aggressive centralized scrubbing
        text = strip_meta_commentary(text)
        
        return (mumbling + text.strip()).strip()
