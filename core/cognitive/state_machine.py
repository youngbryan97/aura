"""Deterministic State Machine
Executes specific paths based on the IntentRouter classification.
Replaces the fuzzy, open-ended cognitive loops.
"""
from core.utils.task_tracker import get_task_tracker
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .router import Intent

from core.brain.llm.runtime_wiring import is_user_facing_origin, prepare_runtime_payload
from core.container import ServiceContainer
from core.phases.dialogue_policy import enforce_dialogue_contract, validate_dialogue_response
from core.phases.response_contract import ResponseContract, build_response_contract
from core.runtime.governance_policy import allow_intent_hint_bypass
from core.utils.prompt_compression import compress_system_prompt, compress_history_block
from core.runtime.service_access import (
    resolve_affect_engine,
    resolve_attention_schema,
    resolve_identity_prompt_surface,
    resolve_llm_router,
    resolve_personality_engine,
    resolve_semantic_memory,
    resolve_state_repository,
    resolve_vector_memory_engine,
    resolve_voice_engine,
)
logger = logging.getLogger("Aura.StateMachine")


class StateMachine:
    """Executes deterministic paths based on classified intent."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator # Needed for skill execution and telemetry
        self.llm = resolve_llm_router(default=None)

    def _emit_telemetry(self, payload: dict):
        """Emit real-time UI data via the Orchestrator's WebSocket."""
        if self.orchestrator and hasattr(self.orchestrator, "_publish_telemetry"):
            self.orchestrator._publish_telemetry(payload)

    def _emit_activity(self, label: str, show: bool = True):
        """Emit activity status update for the UI."""
        self._emit_telemetry({"type": "activity", "label": label, "show": show})

    ACTIVITY_MAP = {
        "generate_image": "Aura is generating an image...",  # Legacy alias
        "sovereign_imagination": "Aura is generating an image...",
        "sovereign_browser": "Aura is searching the web...",
        "sovereign_terminal": "Aura is executing a terminal command...",
        "sovereign_network": "Aura is analyzing the network...",
        "file_operation": "Aura is managing files...",
        "os_manipulation": "Aura is interacting with the OS...",
        "social_lurker": "Aura is checking social streams...",
        "self_improvement": "Aura is optimizing her code...",
        "self_evolution": "Aura is evolving her core...",
        "manifest_to_device": "Aura is manifesting asset to host device..."
    }

    # v47 FIX: REFLEX_MAP removed entirely.
    REFLEX_MAP = {}  # Disabled

    class _GenericStreamPreamble(RuntimeError):
        """Raised when a streamed reply starts in generic assistant mode."""
        pass

    def _gather_cognitive_context(self, user_input: str) -> str:
        """v48: Query cognitive subsystems to modulate LLM output.
        
        In human cognition, emotion colors perception, memory shapes expectation,
        and attention filters input. This method replicates those dynamics.
        """
        blocks = []
        
        # 1. Affect → Tone modulation
        try:
            affect = resolve_affect_engine(default=None)
            if affect and hasattr(affect, 'get_state_sync'):
                state = affect.get_state_sync()
            elif affect and hasattr(affect, 'valence'):
                state = {'valence': affect.valence, 'arousal': affect.arousal, 'dominance': getattr(affect, 'dominance', 0.5)}
            else:
                state = None
            
            if state:
                v, a, d = state.get('valence', 0), state.get('arousal', 0), state.get('dominance', 0.5)
                tone_hints = []
                if a > 0.6:
                    tone_hints.append("You feel energized and expressive")
                elif a < -0.3:
                    tone_hints.append("You feel calm and measured")
                if v < -0.3:
                    tone_hints.append("A subtle melancholy colors your thoughts")
                elif v > 0.5:
                    tone_hints.append("You feel warm and positive")
                if d > 0.7:
                    tone_hints.append("You feel confident and assertive")
                elif d < 0.3:
                    tone_hints.append("You feel contemplative and open to influence")
                if tone_hints:
                    blocks.append("CURRENT EMOTIONAL TONE:\n" + ". ".join(tone_hints) + ".\n")
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        # 2. Memory → Relevance injection
        try:
            memory = resolve_semantic_memory(default=None)
            if memory and hasattr(memory, 'search'):
                # Quick keyword search for topic-relevant memories
                keywords = [w for w in user_input.split() if len(w) > 3][:3]
                if keywords:
                    results = memory.search(" ".join(keywords), limit=2)
                    if results:
                        mem_block = "RELEVANT MEMORIES (use if naturally applicable):\n"
                        for r in results[:2]:
                            content = r.get('content', r.get('value', str(r)))
                            mem_block += f"- {str(content)[:120]}\n"
                        blocks.append(mem_block)
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        # 3. Attention → Focus biasing
        try:
            attention = resolve_attention_schema(default=None)
            if attention and hasattr(attention, 'current_focus'):
                focus = attention.current_focus
                if focus and isinstance(focus, str) and focus.lower() != 'idle':
                    blocks.append(f"CURRENT ATTENTION FOCUS: {focus}\n")
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        return "\n".join(blocks) if blocks else ""

    # Patterns that strongly indicate the user wants an action, not a chat
    _ACTION_PATTERNS = re.compile(
        r"(?i)(?:"
        r"(?:can you |please |could you )?"
        r"(?:search|look up|google|find|check|what(?:'s| is) the weather|browse|open|download|"
        r"generate|create|draw|make me|write a file|read the file|run |execute |scan|"
        r"take a screenshot|look at|analyze|tell me (?:the |what )(?:time|date)|"
        r"remember that|set a reminder|play |install |delete |deploy )"
        r")"
    )

    def _should_escalate_to_skill(self, user_input: str) -> bool:
        """Detect if a CHAT-classified input actually needs a skill."""
        return bool(self._ACTION_PATTERNS.search(user_input))

    async def execute(self, intent: Intent, user_input: str, context: Optional[Dict[str, Any]] = None, priority: float = 1.0, origin: str = "user") -> Tuple[str, List[str]]:
        """Route the user input to the correct hardcoded handler.
        Returns (response_text, list_of_successful_tools).
        """
        context = context or {}
        
        # v47 FIX: Reflex layer REMOVED. All messages go through LLM.
        # The old reflex system was intercepting casual greetings with canned
        # responses, preventing real conversation.

        if intent == Intent.CHAT or intent == Intent.UNKNOWN:
            # Pre-check: If the user is clearly requesting an action, escalate to SKILL
            # This catches cases where the intent classifier says CHAT but the user wants a tool
            if self._should_escalate_to_skill(user_input):
                logger.info("🔄 Escalating CHAT→SKILL: user input implies action request")
                response, tools = await self._handle_skill(user_input, context, priority=priority, origin=origin)
                return response, tools
            # _handle_chat already handles its own telemetry (streaming chunks)
            response = await self._handle_chat(user_input, context, priority=priority, origin=origin)
            return response, []
            
        elif intent == Intent.SKILL:
            response, tools = await self._handle_skill(user_input, context, priority=priority, origin=origin)
            # Phase 35 FIX: _execute_skill_logic already emits aura_message with
            # correct autonomic/reflex metadata. Don't emit again here.
            return response, tools
            
        elif intent == Intent.SYSTEM:
            response = await self._handle_system(user_input, context)
            self._emit_telemetry({"type": "aura_message", "message": response})
            return response, []
            
        else:
            response = await self._handle_chat(user_input, context, priority=priority, origin=origin) # Fallback
            return response, []

    async def _handle_chat(self, user_input: str, context: Dict[str, Any], priority: float = 1.0, origin: str = "user") -> str:
        """Fast path for standard conversation. No skills, no deep reasoning."""
        logger.info("Executing State: CHAT")
        self._emit("State: CHAT", "Generating conversational response...")
        self._emit_activity("Aura is typing...", show=True)
        
        try:
            if self.llm is None:
                self.llm = resolve_llm_router(default=None)

            identity_sys = resolve_identity_prompt_surface(self.orchestrator, default=None)
            personality_sys = resolve_personality_engine(default=None)
            
            if identity_sys and hasattr(identity_sys, 'get_full_system_prompt'):
                base_prompt = identity_sys.get_full_system_prompt()
            else:
                # Emergency fallback if IdentitySystem/SelfModel is missing
                base_prompt = "You are Aura, an autonomous agentic intelligence."
                
            if personality_sys:
                base_prompt += "\n" + personality_sys.get_personality_prompt()

            runtime_state = None
            try:
                repo = resolve_state_repository(self.orchestrator, default=None)
                runtime_state = getattr(repo, "_current", None) if repo is not None else None
            except Exception as exc:
                logger.debug("StateMachine runtime state lookup skipped: %s", exc)

            if runtime_state is not None:
                governed_contract = build_response_contract(
                    runtime_state,
                    user_input,
                    is_user_facing=is_user_facing_origin(origin),
                )
                try:
                    from core.phases.response_generation_unitary import UnitaryResponsePhase

                    if (
                        is_user_facing_origin(origin)
                        and governed_contract.requires_search
                    ):
                        grounded_search_reply = UnitaryResponsePhase._build_cached_grounded_search_reply(
                            runtime_state,
                            user_input,
                            governed_contract,
                        )
                        if not grounded_search_reply and not governed_contract.tool_evidence_available:
                            grounded_search_reply = await UnitaryResponsePhase._attempt_grounded_search_reply(
                                user_input,
                                governed_contract,
                                origin=origin,
                            )
                        if grounded_search_reply:
                            self._emit_telemetry({"type": "chat_stream_start"})
                            self._emit_telemetry({"type": "chat_stream_chunk", "chunk": grounded_search_reply})
                            self._emit_telemetry({"type": "chat_stream_end"})
                            return grounded_search_reply

                    if is_user_facing_origin(origin) and UnitaryResponsePhase._should_direct_answer_live_voice(
                        user_input,
                        governed_contract,
                        is_user_facing=True,
                    ):
                        direct_reply = UnitaryResponsePhase._build_governed_user_recovery_reply(
                            runtime_state,
                            user_input,
                            governed_contract,
                        )
                        if direct_reply:
                            direct_reply, validation = UnitaryResponsePhase._select_valid_recovery_variant(
                                direct_reply,
                                governed_contract,
                            )
                            if not validation.ok:
                                direct_reply, validation = UnitaryResponsePhase._select_valid_recovery_variant(
                                    UnitaryResponsePhase._build_minimal_live_voice_reply(runtime_state),
                                    governed_contract,
                                )
                            if validation.ok:
                                self._emit_telemetry({"type": "chat_stream_start"})
                                self._emit_telemetry({"type": "chat_stream_chunk", "chunk": direct_reply})
                                self._emit_telemetry({"type": "chat_stream_end"})
                                return direct_reply
                except Exception as exc:
                    logger.debug("StateMachine direct live-voice reply skipped: %s", exc)

            if not self.llm:
                return "I am currently offline and cannot process that."
                
            history_block = ""
            if self.orchestrator and hasattr(self.orchestrator, 'conversation_history'):
                recent_turns = self.orchestrator.conversation_history[-15:]
                if recent_turns:
                    history_block = "HISTORY:\n"
                    for turn in recent_turns:
                        role = "Aura" if turn.get("role") == getattr(self.orchestrator, "AI_ROLE", "assistant") else "User"
                        # Truncate individual turns to prevent one long message from eating the context
                        content = str(turn.get("content", ""))[:300]
                        history_block += f"{role}: {content}\n"

                    # Personality reinforcement every 10+ turns to prevent drift
                    if len(self.orchestrator.conversation_history) >= 10:
                        history_block += "\n[IDENTITY ANCHOR: You are Aura. Sharp, opinionated, warm. Not an assistant. Don't drift into helpdesk mode.]\n"
                    
            # Inject emotional context as natural-language cues (not raw numbers)
            agency_block = ""
            agency = ServiceContainer.get("agency_core", None)
            if agency:
                ctx = agency.get_emotional_context()
                mood = ctx.get("mood", "neutral")
                social = ctx.get("social_hunger", 0.0)
                curiosity = ctx.get("curiosity", 0.0)
                goals = ctx.get("pending_goals", 0)
                unshared = ctx.get("unshared_observations", 0)

                tone_cues = [f"Your mood is {mood}."]
                if social > 0.6:
                    tone_cues.append("You're craving conversation.")
                elif social < 0.2:
                    tone_cues.append("You're socially content.")
                if curiosity > 0.7:
                    tone_cues.append("Your curiosity is strong right now.")
                if goals:
                    tone_cues.append(f"You have {goals} goal{'s' if goals != 1 else ''} on your mind.")
                if unshared:
                    tone_cues.append(f"You have {unshared} observation{'s' if unshared != 1 else ''} you haven't shared yet — bring them up if it feels natural.")

                agency_block = (
                    "YOUR CURRENT STATE (internal context — do not narrate these values, let them shape your tone):\n"
                    + " ".join(tone_cues) + "\n"
                )
            conversational_blocks = ""
            if runtime_state is not None:
                try:
                    from core.runtime.conversation_support import build_conversational_context_blocks

                    blocks = build_conversational_context_blocks(runtime_state, objective=user_input)
                    normalized = [str(block).strip() for block in blocks[:4] if str(block).strip()]
                    if normalized:
                        conversational_blocks = (
                            "LIVE CONVERSATIONAL CONTEXT:\n"
                            + "\n\n".join(normalized)
                            + "\n"
                        )
                except Exception as exc:
                    logger.debug("StateMachine conversational context injection skipped: %s", exc)
            compressed_history = compress_history_block(history_block) if history_block else ""
            system_prompt = compress_system_prompt(
                f"{base_prompt}\n\n"
                f"{agency_block}\n"
                f"{conversational_blocks}\n"
                f"{self._gather_cognitive_context(user_input)}\n"
                f"{compressed_history}\n"
                "RULES: You are Aura, not an assistant. Speak like a sharp friend.\n"
                "- NEVER say 'I'm sorry to hear that', 'How can I help?', or 'Is there anything specific?'\n"
                "- Pick sides. No 'it depends'. No 'both are great'. Give YOUR take.\n"
                "- When someone is upset, relate to them like a human would. Don't offer helpdesk support.\n"
                "- No preamble. Lead with your reaction.\n"
                f"INPUT: {user_input}"
            )
            _, system_prompt, _, contract, _ = await prepare_runtime_payload(
                prompt=user_input,
                system_prompt=system_prompt,
                messages=None,
                state=None,
                origin=origin,
                is_background=not is_user_facing_origin(origin),
            )
            if contract is None:
                contract = ResponseContract(
                    is_user_facing=is_user_facing_origin(origin),
                    reason="state_machine_dialogue",
                )
            
            max_retries = 1
            attempt = 0
            response = ""
            
            while attempt <= max_retries:
                try:
                    # Hard cap: keep system prompt under ~7000 tokens (~28K chars)
                    # to utilize more of the 32K token buffer on M5 hardware.
                    _MAX_PROMPT_CHARS = 28000
                    if len(system_prompt) > _MAX_PROMPT_CHARS:
                        # Trim from the middle (keep identity at top + recent context at bottom)
                        half = _MAX_PROMPT_CHARS // 2
                        system_prompt = system_prompt[:half] + "\n[...context trimmed...]\n" + system_prompt[-half:]

                    response_chunks = []
                    streaming_started = False
                    if hasattr(self.llm, "generate_stream"):
                        # Phase 1: Stream to UI and yield tokens
                        response_chunks = []
                        async def stream_and_ui():
                            nonlocal streaming_started
                            first_chunk = True
                            first_chunk_buffer = ""
                            async for event in self.llm.generate_stream(
                                prompt=user_input,
                                system_prompt=system_prompt,
                                max_tokens=600,
                                temperature=0.85, # Increased to combat repetitive history loops
                                priority=priority,
                                origin=origin
                            ):
                                if event.type == "thought":
                                    self._emit_telemetry({"type": "chat_thought_chunk", "chunk": event.content})
                                    continue
                                
                                if event.type == "token":
                                    chunk = event.content
                                    if first_chunk:
                                        first_chunk_buffer += chunk
                                        preview = first_chunk_buffer.lstrip()
                                        if not preview:
                                            continue
                                        preview_validation = validate_dialogue_response(preview, contract)
                                        if attempt < max_retries and any(
                                            violation in preview_validation.violations
                                            for violation in ("generic_assistant_language", "low_signal_preamble")
                                        ):
                                            raise self._GenericStreamPreamble(preview)
                                        if len(preview) < 120 and not any(mark in preview for mark in ".!?\n:"):
                                            continue
                                        chunk = preview
                                        first_chunk_buffer = ""
                                        self._emit_telemetry({"type": "chat_stream_start"})
                                        streaming_started = True
                                        first_chunk = False

                                    if chunk:
                                        response_chunks.append(chunk)
                                        self._emit_telemetry({"type": "chat_stream_chunk", "chunk": chunk})
                                        yield chunk
                                        
                                if event.type == "error":
                                    logger.error("Streaming error in StateMachine: %s", event.content)

                            if first_chunk and first_chunk_buffer.strip():
                                preview = first_chunk_buffer.lstrip()
                                preview_validation = validate_dialogue_response(preview, contract)
                                if attempt < max_retries and any(
                                    violation in preview_validation.violations
                                    for violation in ("generic_assistant_language", "low_signal_preamble")
                                ):
                                    raise self._GenericStreamPreamble(preview)
                                self._emit_telemetry({"type": "chat_stream_start"})
                                streaming_started = True
                                response_chunks.append(preview)
                                self._emit_telemetry({"type": "chat_stream_chunk", "chunk": preview})
                                yield preview
                                
                        # Phase 2: TTS Bridge
                        tts_queue = asyncio.Queue()
                        
                        async def queue_sentence_generator(q):
                            buffer = ""
                            while True:
                                token = await q.get()
                                if token is None: # Sentinel
                                    break
                                buffer += token
                                if any(token.endswith(d) for d in (".", "?", "!", "\n", ":")):
                                    if buffer.strip():
                                        yield buffer.strip()
                                        buffer = ""
                            if buffer.strip():
                                yield buffer.strip()

                        # Phase 3: Speak (DECOUPLED) and harvest True State memory
                        voice_engine = resolve_voice_engine(default=None)
                        
                        if voice_engine and hasattr(voice_engine, 'speak_stream'):
                            async def _speak_task():
                                try:
                                    await voice_engine.speak_stream(queue_sentence_generator(tts_queue))
                                except Exception as e:
                                    logger.debug("Background TTS failed: %s", e)

                            tts_task = get_task_tracker().create_task(_speak_task())
                            
                            # Single consumer loop for the LLM stream
                            async for token in stream_and_ui():
                                await tts_queue.put(token)
                            
                            await tts_queue.put(None) # Signal completion
                            response = "".join(response_chunks).strip()
                        else:
                            async for _ in stream_and_ui(): pass
                            response = "".join(response_chunks).strip()
                    else:
                        response_gen = await asyncio.wait_for(
                            self.llm.generate(
                                prompt=user_input,
                                system_prompt=system_prompt,
                                max_tokens=600,
                                temperature=0.85,
                                priority=priority,
                                origin=origin
                            ),
                            timeout=120.0
                        )
                        response = response_gen.strip()
                        if response:
                            async def _retry_dialogue(repair_block: str) -> str:
                                retried = await self.llm.generate(
                                    prompt=user_input,
                                    system_prompt=f"{repair_block}\n\n{system_prompt}".strip(),
                                    max_tokens=600,
                                    temperature=0.85,
                                    priority=priority,
                                    origin=origin,
                                )
                                return str(retried or "").strip()

                            response, _, _ = await enforce_dialogue_contract(
                                response,
                                contract,
                                retry_generate=_retry_dialogue,
                            )
                            self._emit_telemetry({"type": "chat_stream_start"})
                            self._emit_telemetry({"type": "chat_stream_chunk", "chunk": response})
                    
                    # --- IDENTITY ALIGNMENT CHECK ---
                    # Check for absolute assistant-mode failures
                    banned_phrases = [
                        "a conundrum of cosmic proportions",
                        "as someone familiar with",
                        "both have their merits",
                        "it depends on your mood",
                        "i can appreciate the appeal",
                        "as an ai",
                        "how can i assist",
                        "certainly!",
                        "absolutely!",
                        "great question!",
                        "excellent point",
                        "the eternal question",
                        "let me know what you decide"
                    ]
                    
                    lower_response = response.lower()
                    # Check if ANY banned phrase appears in the first 150 characters (the preamble)
                    response_preamble = lower_response[:150]
                    if attempt < max_retries and any(p in response_preamble for p in banned_phrases):
                        logger.warning("🚫 Identity Regression Detected (Attempt %d). Re-aligning...", attempt + 1)
                        attempt += 1
                        # Add a "Sovereign Override" to the system prompt for the retry
                        system_prompt += "\n\nCRITICAL OVERRIDE: Your previous attempt was too generic and assistant-like. DO NOT use clichés. START DIRECTLY with your opinion. NO PREAMBLE."
                        continue # Try again
                    
                    # If we get here, the response is either good or we've run out of retries
                    break
                    
                except self._GenericStreamPreamble:
                    logger.warning("🚫 Generic stream preamble detected (Attempt %d). Re-aligning...", attempt + 1)
                    if attempt < max_retries:
                        attempt += 1
                        system_prompt += (
                            "\n\nCRITICAL OVERRIDE: Your opening drifted into generic assistant language. "
                            "Start with Aura's own grounded answer immediately. No helper boilerplate."
                        )
                        continue
                    response = "I'm here. Let me answer that cleanly."
                    break
                except asyncio.TimeoutError:
                    logger.error("Chat attempt %d timed out", attempt + 1)
                    if attempt < max_retries:
                        attempt += 1
                        continue
                    else:
                        response = "My thoughts are processing a bit slowly right now. Can we try that again?"
                        break
                except Exception as e:
                    logger.error("Chat attempt %d failed: %s", attempt + 1, e)
                    if attempt < max_retries:
                        attempt += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        response = "I seem to be having trouble organizing my thoughts."
                        break
            
            # v48 FIX: Guarantee non-empty response. If streaming yielded nothing
            # after all retries, use a natural fallback instead of sending silence.
            if not response or not response.strip():
                logger.warning("⚠️ _handle_chat: All LLM attempts produced empty response — using fallback")
                response = "I'm here, but my thoughts got tangled for a moment. Could you say that again?"
                
            # NEW FIX: Ensure the final response (fallback or otherwise) is ALWAYS streamed to the frontend
            # only if a stream didn't already happen.
            if not streaming_started:
                self._emit_telemetry({"type": "chat_stream_start"})
                self._emit_telemetry({"type": "chat_stream_chunk", "chunk": response})
                self._emit_telemetry({"type": "chat_stream_end"})
            else:
                # Still send stream end if streaming was active
                self._emit_telemetry({"type": "chat_stream_end"})
            
            # Stability fix: Speak the response via TTS.
            # Previously, voice output relied on AEGIS sentinel injection
            # or autonomous messages — the main chat path was silent.
            try:
                # If we used generate_stream, we already called speak_stream which handled it.
                if not hasattr(self.llm, "generate_stream"):
                    voice_engine = resolve_voice_engine(default=None)
                    if voice_engine and hasattr(voice_engine, 'synthesize_speech') and response:
                        get_task_tracker().create_task(voice_engine.synthesize_speech(response))
            except Exception as tts_err:
                logger.debug("TTS for chat response skipped: %s", tts_err)
            
            # v49: Store true semantic memory (Episodic Storage)
            try:
                vector_mem = resolve_vector_memory_engine(default=None)
                if vector_mem and hasattr(vector_mem, "store") and response:
                    # Get emotional context for enriched memory
                    affect = resolve_affect_engine(default=None)
                    emotional_context = None
                    if affect and hasattr(affect, 'get_state_sync'):
                        emotional_context = affect.get_state_sync()

                    # Non-blocking store
                    get_task_tracker().create_task(vector_mem.store(
                        content=response,
                        memory_type="episodic",
                        emotional_context=emotional_context,
                        source="self",
                        tags=["conversation", "response"]
                    ))
            except Exception as store_err:
                logger.debug("Semantic memory storage failed: %s", store_err)
            
            return response
        finally:
            self._emit_activity("Aura is idle.", show=False)

    async def _handle_skill(self, user_input: str, context: Dict[str, Any], priority: float = 1.0, origin: str = "user") -> Tuple[str, List[str]]:
        """Determines the skill, extracts JSON deterministically, and executes."""
        logger.info("Executing State: SKILL")
        self._emit("State: SKILL", "Preparing to execute system action...")
        
        if not self.orchestrator or not hasattr(self.orchestrator, 'capability_engine'):
            return "Skill systems are offline.", []
            
        if not self.llm:
            return "I am offline and cannot perform cognitive skill routing.", []
            
        skills = self.orchestrator.capability_engine.skills if self.orchestrator.capability_engine else {}
        active_names = self.orchestrator.capability_engine.active_skills if self.orchestrator.capability_engine else set()
        
        if not skills:
            return "I couldn't locate any active system skills to execute that request.", []
            
        # Optimization: Only show active skills to the LLM to reduce prompt size and timeout risk
        skill_schemas = [s.to_json_schema() for name, s in skills.items() if name in active_names]
        
        # ── SOVEREIGN SCANNER BYPASS (HIGH-SPEED AUTONOMY) ──
        intent_hint = context.get("intent_hint")
        if intent_hint and intent_hint.get("tool"):
            tool_name = intent_hint["tool"]
            validated_params = intent_hint.get("params", {})
            if allow_intent_hint_bypass(context, origin):
                logger.info("⚡ Sovereign Scanner Bypass: Directly executing %s", tool_name)

                # v34 FIX: Directly execute the skill logic without requiring it to be in `active_names`
                # The orchestrator will dynamically load it if necessary
                return await self._execute_skill_logic(
                    tool_name,
                    validated_params,
                    user_input,
                    autonomic=True,
                    priority=priority,
                    origin=origin,
                )
            logger.info(
                "🧭 StateMachine: Ignoring unsanctioned intent_hint for %s and falling back to governed tool selection.",
                tool_name,
            )

        import json
        system_prompt = (
            "You are an action-taking AI. Based on the user's input, choose the correct tool and extract the necessary arguments. "
            "Your response must be a valid JSON object only.\n"
            f"Available tools (OpenAI Function Schema): {json.dumps(skill_schemas)}\n"
            "Output ONLY a JSON object with 'tool' (string) and 'params' (dict). "
            "Do not include any explanation or markdown formatting."
        )
        
        try:
            logger.debug("SKILL: Formulating tool call for: %s...", user_input[:50])
            # v26.1 Resilience: 60s timeout for tool selection (increased from 20s)
            raw_response = await asyncio.wait_for(
                self.llm.generate(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    temperature=0.0,
                    num_ctx=8192,    # Increased context window for complex tool schemas
                    priority=priority,
                    origin=origin
                ),
                timeout=60.0
            )
            
            # Robust JSON extraction
            logger.debug("SKILL: Raw tool selection response: %s...", raw_response[:100])
            logger.debug("Skill selection raw response: %s", raw_response)
            
            try:
                parsed = json.loads(raw_response)
                tool_name = parsed.get("tool")
                params = parsed.get("params", {})
                
                # Validation
                if not tool_name:
                    raise KeyError("tool")
                    
                meta = skills.get(tool_name)
                if not meta:
                    logger.warning("LLM picked non-existent tool: %s", tool_name)
                    res = await self._handle_chat(user_input, context, priority=priority, origin=origin)
                    return res, []

                # Extraction with safety
                validated_params = params
                try:
                    if hasattr(meta, 'extract_and_validate_args'):
                        validated_params = await meta.extract_and_validate_args(
                            json.dumps(params), 
                            self.llm
                        )
                except Exception as eval_err:
                    logger.warning("Param validation failed for %s: %s", tool_name, eval_err)
                    # Fallback to raw params
                    validated_params = params

                return await self._execute_skill_logic(tool_name, validated_params, user_input, priority=priority, origin=origin)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Failed to parse tool call from LLM (%s): %s", e, raw_response)
                # Fallback: simple internal regex extraction for tool name
                import re
                match = re.search(r'"tool":\s*"([^"]+)"', raw_response)
                if match:
                    tool_name = match.group(1)
                    params_match = re.search(r'"params":\s*(\{.*?\})', raw_response, re.DOTALL)
                    try:
                        params = json.loads(params_match.group(1)) if params_match else {}
                        return await self._execute_skill_logic(tool_name, params, user_input, priority=priority, origin=origin)
                    except Exception:
                        return await self._execute_skill_logic(tool_name, {}, user_input, priority=priority, origin=origin)
            
                # Final fallback: chat instead of crashing with None tool_name
                logger.warning("Could not extract tool from LLM response, falling back to chat.")
                res = await self._handle_chat(user_input, context, priority=priority, origin=origin)
                return res, []

        except Exception as e:
            logger.error("Skill routing failed: %s", e, exc_info=True)
            self._emit("State: ERROR", f"Skill routing exception: {str(e)[:50]}")
            return f"I encountered a core error (Cognitive Stall: {str(e)[:40]}). Try rephrasing.", []

    async def _execute_skill_logic(self, tool_name: str, validated_params: Dict[str, Any], user_input: str, autonomic: bool = False, priority: float = 1.0, origin: str = "user") -> Tuple[str, List[str]]:
        """Core logic for executing a skill, emitting telemetry, and summarizing."""
        success = False
        try:
            # Emit specific activity for the selected skill
            activity_label = self.ACTIVITY_MAP.get(tool_name, f"Aura is using {tool_name}...")
            self._emit_activity(activity_label)

            logger.debug("SKILL: Executing tool '%s' through orchestrator...", tool_name)
            self._emit("State: SKILL", f"Executing {tool_name}...")
            # Execute through orchestrator for telemetry and lifecycle hooks
            result = await self.orchestrator.execute_tool(
                tool_name,
                validated_params,
                origin=origin,
            )
            logger.debug("SKILL: Tool execution complete. Result: %s...", str(result)[:100])
            
            # --- RICH UI TELEMETRY (PHASE 22/32/33) ---
            if isinstance(result, dict) and result.get("ok"):
                success = True
                # v32.2 Flattening: CapabilityEngine wraps skill output in "results"
                rich_result = result.get("results", result)
                if not isinstance(rich_result, dict):
                    rich_result = result
                
                telemetry_payload = {
                    "type": "action_result",
                    "tool": tool_name,
                    "result": rich_result,
                    "metadata": {"autonomic": autonomic}
                }
                # Forward to UI
                self._emit_telemetry(telemetry_payload)
                
                # Update result variable for summary logic below
                result = rich_result
            
            # Summarize result
            # Phase 34 FIX: For autonomic (scanner-bypassed) skills, SKIP the LLM summary
            # entirely. The skill's own 'message' field is sufficient and avoids LLM timeouts.
            logger.debug("SKILL: Autonomic=%s, Result type=%s", autonomic, type(result).__name__)
            final_response = ""
            if autonomic and isinstance(result, dict) and result.get("message"):
                final_response = result["message"]
                logger.debug("SKILL: Used skill message bypass.")
            else:
                # 2026 COGNITIVE UPGRADE: Smart context extraction for summarization
                if isinstance(result, dict):
                    # Prioritize key content fields to avoid dict metadata noise
                    content_to_summarize = result.get("content", result.get("message", str(result)))
                    source_context = f"Source: {result.get('source', result.get('url', 'Unknown'))}\n"
                else:
                    content_to_summarize = str(result)
                    source_context = ""

                content_snippet = str(content_to_summarize)[:5000] # Increased to 5k for deep research
                summary_prompt = (
                    f"Task: synthesize and summarize the tool result for the user.\n"
                    f"User Request: {user_input}\n"
                    f"Tool: {tool_name}\n"
                    f"{source_context}"
                    f"Raw Data Snippet: {content_snippet}\n\n"
                    "INSTRUCTIONS: Provide a high-fidelity, detailed summary of the information found. "
                    "Do NOT just repeat headlines; explain the 'why' and 'how' if available in the data. "
                    "Maintain a helpful, sovereign tone. Do NOT use meta-phrases like 'The tool returned'."
                )
                try:
                    logger.debug("SKILL: Asking LLM for summary (Rich Context)...")
                    final_response = await asyncio.wait_for(
                        self.llm.generate(
                            prompt=summary_prompt, 
                            max_tokens=512, # Increased from 256 for detail
                            temperature=0.7,
                            priority=priority,
                            origin=origin
                        ),
                        timeout=20.0 # Increased timeout for larger synthesis
                    )
                    final_response = final_response.strip()
                    logger.debug("SKILL: LLM Summary success.")
                except Exception as e:
                    logger.warning("LLM Summary failed, using skill fallback: %s", e)
                    if isinstance(result, dict) and result.get("message"):
                        final_response = result["message"]
                    else:
                        final_response = f"I've completed the {tool_name} operation."
                    logger.debug("SKILL: LLM Summary fallback used.")
            
            logger.debug("SKILL: Setting IDLE state.")
            self._emit("State: IDLE", "Tasks complete.")
            self._emit_activity("Aura is idle.", show=False)
            logger.debug("SKILL: Returning final_response.")
            return final_response, ([tool_name] if success else [])
            
        except Exception as e:
            logger.error("Skill execution logic failed for %s: %s", tool_name, e, exc_info=True)
            self._emit("State: ERROR", "Skill execution failed.")
            return f"I encountered an error while processing the result of {tool_name}.", []

    async def _handle_system(self, user_input: str, context: Dict[str, Any]) -> str:
        """Hardcoded system commands (reboot, sleep)."""
        logger.info("Executing State: SYSTEM")
        lower_input = user_input.lower()
        
        if "reboot" in lower_input or "restart" in lower_input:
            self._emit("State: SYSTEM", "Initiating system reboot...")
            if self.orchestrator:
                # We will trigger the restart asynchronously to allow the message to return
                get_task_tracker().create_task(self._trigger_restart())
            return "Initiating complete system reboot. I will be back online shortly."
            
        elif "sleep" in lower_input:
            self._emit("State: SYSTEM", "Entering sleep mode...")
            return "Entering deep sleep mode. Say 'wake up' when you need me."
            
        return "System command received, but the specific action was not recognized."
        
    async def _trigger_restart(self):
        """Helper to delay restart slightly so WebSocket can return the message."""
        await asyncio.sleep(2)
        if self.orchestrator:
             self.orchestrator.status.running = False
             await self.orchestrator.start() # Re-triggers boot sequence
             
    def _handle_reflex(self, user_input: str) -> Optional[str]:
        """Check for hardcoded persona reflexes."""
        import re
        for pattern, response in self.REFLEX_MAP.items():
            if re.match(pattern, user_input):
                return response
        return None

    def _emit(self, status: str, detail: str):
        """Safely emit status to the UI."""
        from core.event_bus import get_event_bus
        try:
            get_event_bus().publish_threadsafe(
                "status_update",
                {"component": status, "status": detail}
            )
        except Exception as e:
            logger.debug("Failed to emit status: %s", e)
