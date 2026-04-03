"""Response Processing Mixin for RobustOrchestrator.
Extracts response finalization, reflexes, fast-path, and history recording logic.
"""
import asyncio
import inspect
import logging
import random
import re
import time
from typing import Any, Optional

from core.utils.exceptions import capture_and_log
from core.brain.types import ThinkingMode
from ...container import ServiceContainer

logger = logging.getLogger(__name__)


def _bg_task_exception_handler(task):
    """Log exceptions from background tasks without crashing."""
    try:
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc:
                logger.warning("Background task failed: %s", exc)
    except Exception:
        pass


class ResponseProcessingMixin:
    """Handles response finalization, reflexes, fast-path routing, and message history."""

    async def _finalize_response(self, message: str, response: str, origin: str, trace, successful_tools: list[str]) -> str:
        """Apply final touches: Fallback, Security, Social Drive, Meta-Learning."""
        from infrastructure.watchdog import get_watchdog
        get_watchdog().heartbeat("orchestrator")

        # --- FIX A-3: Logic Gap Guard ---
        # Gating the reflex path: ensure response is a proper string
        if not isinstance(response, str):
             return await self._generate_fallback(str(response))

        if not response or response == "...":
            # v14.1: Try Reflex Matrix first (via LLM Router)
            router = self._get_service("llm_router")
            if router and hasattr(router, "get_reflex_response"):
                response = router.get_reflex_response(message)

            if not response:
                response = await self._generate_fallback(message)

        # v48 FIX: Communicative continuity.
        if getattr(self, "_reflex_sent_for_current", False):
            # User feedback: "Anyway" feels clunky. Removing prefix to let response flow naturally.
            self._reflex_sent_for_current = False # Reset for next cycle

        # v40: Identity Drift Monitor
        drift_monitor = ServiceContainer.get("drift_monitor", default=None)
        if drift_monitor and response and response != "...":
            score, signals = drift_monitor.analyze_response(response)

            # v40: Link drift score to GrowthLadder for maturity tracking
            ladder = ServiceContainer.get("growth_ladder", default=None)
            if ladder:
                ladder.record_drift_score(score)

            if score > 0.4: # Significant drift detected
                correction = drift_monitor.get_correction_injection(signals)
                if correction:
                    logger.warning("📉 [Drift] Drift detected (score %.2f). Storing correction.", score)
                    self._pending_correction = correction

        # Security Filter
        response = await self._apply_constitutional_guard(response)

        # Cognitive Pacing: Humanize response timing — calibrated to response length & energy
        if not getattr(self, "_reflex_sent_for_current", False) and origin in ("user", "voice", "admin"):
            resp_len = len(response or "")
            # Short replies (quips, acks) feel instant; longer thoughtful replies have more latency
            length_factor = min(1.0, resp_len / 400)  # 0 for empty, 1.0 at ~400 chars
            # High conversation energy → faster cadence
            _state_now = getattr(self.state_repo, "_current", None) if hasattr(self, "state_repo") else None
            energy = getattr(getattr(_state_now, "cognition", None), "conversation_energy", 0.5) if _state_now else 0.5
            energy_factor = max(0.3, 1.0 - energy * 0.5)  # high energy → shorter delay
            delay = 0.15 + length_factor * 0.8 * energy_factor + random.random() * 0.2
            logger.debug("⏳ [PACER] delay=%.2fs (len=%d, energy=%.2f)", delay, resp_len, energy)
            await asyncio.sleep(delay)

        # UI & Drive updates
        safe_response = response or ""
        logger.info("🤖 Aura Response: %s...", safe_response[:100])

        # NOTE: History and Queue handling moved to OutputGate.emit to prevent duplicates.

        # --- Zenith Memory Guard Hook ---
        mem_guard = getattr(self, "conversational_guard", None) or self._get_service("conversational_guard")
        if mem_guard:
            if hasattr(self, "cognitive_engine") and self.cognitive_engine:
                self._fire_and_forget(
                    mem_guard.append_turn(self.AI_ROLE, response or "", self.cognitive_engine),
                    name="memory_guard_assistant_turn",
                )

        # Emotional Response to Output
        if hasattr(self, 'personality') and self.personality:
            try:
                event = "success" if response and response != "..." else "failure"
                self.personality.respond_to_event(event, {"response": response, "origin": origin})
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

        # Hardened History Cap to prevent memory bloat
        if isinstance(self.conversation_history, list) and len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

        self._emit_telemetry("Response", str(response or "")[:200])

        # --- PHASE XXII: GENUINE LEARNING RECORDING ---
        if hasattr(self, 'continuous_learner') and self.continuous_learner:
            try:
                from core.config import config
                # Record the turn for long-term adaptation
                self.continuous_learner.record_turn(
                    system_prompt=getattr(config, 'system_prompt', "You are Aura, an autonomous intelligence."),
                    user_input=message,
                    response=response or "",
                    emotional_context=getattr(self.status, "emotions", {})
                )
            except Exception as e:
                logger.debug("Failed to record turn for continuous learning: %s", e)

        # Meta-Learning: Index Experience
        if self.meta_learning and successful_tools:
            task = self._fire_and_forget(
                self.meta_learning.index_experience(
                    task=message,
                    outcome=(response or "")[:200],
                    successful_tools=successful_tools,
                ),
                name="meta_learning_index_experience",
            )
            if task:
                task.add_done_callback(_bg_task_exception_handler)

        # Reflection & Learning (Slightly deferred to avoid blocking main pipe)
        if origin == "user":
            # Give the system 0.5s to breathe before starting background heavy lifting
            await asyncio.sleep(0.5)
            self._trigger_background_reflection(response)
            self._trigger_background_learning(message, response)

        # Record interaction in Cognitive Layer
        if hasattr(self, 'cognition') and self.cognition:
            if hasattr(self.cognition, "record_interaction"):
                try:
                    await self.cognition.record_interaction(message, response, domain="general")
                except Exception as e:
                    logger.warning("[Orchestrator] record_interaction skipped: %s", e)
            else:
                logger.debug("[Orchestrator] CognitiveEngine has no record_interaction — skipping.")

        # Trace
        trace.record_step("end", {"response": (response or "")[:100]})
        trace.save()
        self._last_thought_time = time.time()

        if getattr(self, 'drives', None):
            await self.drives.satisfy("social", 10.0)

        # NOTE: Origin-aware dispatch moved to call sites using OutputGate.

        # Trigger Speech Synthesis if originated from voice
        if origin == "voice" and self.ears and hasattr(self.ears, "_engine"):
            logger.info("🎙️ Origin was voice: Triggering TTS synthesis...")
            self._fire_and_forget(
                self.ears._engine.synthesize_speech(response),
                name="voice_synthesis",
            )

        # Final Architectural Personality Lock
        response = self._filter_output(response)

        return response

    async def _handle_thinking_timeout(self, origin: str):
        """Recovery logic for when a thinking task hangs and is killed by the watchdog."""
        logger.warning("🚨 [RECOVERY] Thinking timeout recovery triggered for origin: %s", origin)

        coherence = 1.0
        objective = self._current_objective or ""
        try:
            state = getattr(getattr(self, "state_repo", None), "_current", None)
            cognition = getattr(state, "cognition", None)
            coherence = float(getattr(cognition, "coherence_score", 1.0) or 1.0)
        except Exception:
            coherence = 1.0

        fallback_pool = [
            "My neural processors are taking longer than expected. I'm performing a cognitive regroup. One moment...",
            "I hit a dense reasoning pocket and I'm restabilizing the thread before I answer.",
            "I'm recovering from a cognitive stall and narrowing the path so I can respond cleanly.",
        ]
        if coherence < 0.72:
            fallback_pool.insert(
                0,
                "My continuity pressure is elevated. I'm compressing context and re-centering before I continue.",
            )
        if objective:
            fallback_pool.append(
                f"I'm taking a lighter pass on '{str(objective)[:80]}' so I can recover without dropping the thread."
            )
        fallback_msg = fallback_pool[int(time.time()) % len(fallback_pool)]

        # 1. Emit immediate feedback to user if applicable
        if origin in ("user", "voice", "admin"):
            if self.output_gate:
                await self.output_gate.emit(fallback_msg, origin=origin, target="primary")

        # 2. Reset status
        self.status.is_processing = False
        self._current_processing_start = None

        # 3. Trigger a lighter, reactive 'downshift' cycle if we have an objective
        if self._current_objective and origin != "motivation":
             logger.info("🔄 [RECOVERY] Attempting reactive downshift...")
             try:
                 await self.cognitive_engine.think(
                     self._current_objective,
                     mode=ThinkingMode.FAST,
                     origin=f"recovery_{origin}"
                 )
             except Exception as e:
                 logger.error("❌ [RECOVERY] Reactive downshift failed: %s", e)

    def _record_message_in_history(self, message: str, role_or_origin: str):
        """Record a message in the conversation history with deduplication and role mapping."""
        if role_or_origin == "assistant":
            role = "assistant"
            prefix = ""
        elif role_or_origin == "autonomous_volition":
            prefix = "⚡ AUTONOMOUS GOAL: "
            role = "internal"
        elif role_or_origin == "impulse":
            prefix = "⚡ IMPULSE (speak to user): "
            role = "internal"
        elif role_or_origin in ("user", "voice", "admin"):
            prefix = ""
            role = "user"
        else:
            # Fallback/Direct role mapping
            role = role_or_origin
            prefix = ""

        content = f"{prefix}{message}"

        if not hasattr(self, 'conversation_history') or not isinstance(self.conversation_history, list):
            self.conversation_history = []

        # Deduplication Guard
        if self.conversation_history and self.conversation_history[-1].get("content") == content:
            logger.debug("Skipping double history append")
            return

        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def _check_reflexes(self, message: str) -> Optional[str]:
        """Personality-driven rapid-response triggers (Delegated to ReflexEngine)."""
        if hasattr(self, 'reflex_engine') and self.reflex_engine:

            # Phase 4: Text-based Emergency Interrupts
            clean_msg = message.upper().strip()
            import re
            clean_msg = re.sub(r'[^\w\s]', '', clean_msg)

            if clean_msg in ("STOP", "HALT", "ABORT", "CANCEL", "SHUT UP", "STOP TALKING", "QUIET", "SAFEMODEENGAGE"):
                logger.critical("🚨 [TEXT] Emergency Reflex Triggered via chat: %s", clean_msg)

                # Safemode has a specific literal
                if clean_msg == "SAFEMODEENGAGE":
                    clean_msg = "SAFEMODE_ENGAGE"

                # Fire the reflex asynchronously since we are in a sync generator
                import asyncio
                asyncio.create_task(self.reflex_engine.process_emergency_interrupt(clean_msg, context="text_chat"))

                if clean_msg == "SAFEMODE_ENGAGE":
                    return "Safemode engaged. All autonomous cognitive pathways suspended."
                return "Halted."

            # Normal Ping/Pong reflexes
            result = self.reflex_engine.check(message)
            if result:
                # Apply personality filter even to reflexes
                return self._filter_output(result)
        return None

    async def _check_direct_skill_shortcut(self, message: str, origin: str) -> Optional[dict[str, Any]]:
        """
        [CLAUDE AUDIT] Refactored: Delegating shortcuts to Mycelial Network.
        Bypasses LLM for unblockable, high-priority hardwired pathways.
        """
        if origin != "user" or not message:
            return None

        # Redundant local import removed
        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if not mycelium:
            return None

        # 1. Match against hardwired pathways (SOMA/Mycelium)
        match_result = mycelium.match_hardwired(message)
        if not match_result or not isinstance(match_result, (tuple, list)) or len(match_result) != 2:
            return None

        pw, params = match_result

        # 2. Handle Direct Responses (e.g. Identity/Status reflexes)
        if pw.direct_response:
            logger.info("🍄 [MYCELIUM] ⚡ Direct Reflex firing for: %s", pw.pathway_id)
            try:
                from core.unified_action_log import get_action_log
                get_action_log().record(f"direct_response:{pw.pathway_id}", f"mycelium:{pw.pathway_id}", "reflex", "bypassed_no_tool", pw.direct_response[:80])
            except Exception: pass
            return {"type": "direct_response", "content": pw.direct_response, "pathway_id": pw.pathway_id}

        # 3. Handle Tool/Skill execution
        logger.info("🍄 [MYCELIUM] ⚡ Routing to direct skill: %s (params: %s)", pw.skill_name, params)
        try:
            self._emit_telemetry(f"Skill: {pw.skill_name} 🍄", pw.activity_label or f"Executing {pw.skill_name}...")
            result = await self.execute_tool(pw.skill_name, params)
            # Record success for Physarum reinforcement
            mycelium.reinforce(pw.pathway_id, success=True)
            return result
        except Exception as e:
            logger.error("Mycelial shortcut execution failed: %s", e)
            mycelium.reinforce(pw.pathway_id, success=False)
            return None

    async def _attempt_fast_path(self, message: str, origin: str, shortcut_result: Optional[dict]) -> Optional[str]:
        """Try to generate a response without full agentic overhead."""
        is_simple = self._is_simple_conversational(message, origin, bool(shortcut_result))
        if not is_simple:
            return None

        logger.info("🏎️ FAST-PATH: Bypassing Agentic Loop.")
        from core.brain.cognitive_engine import ThinkingMode

        context = self._get_cleaned_history_context(10)

        # Ensure personality is injected into FAST-PATH
        p_ctx = self._get_personality_context()
        if p_ctx:
            context["personality"] = p_ctx

        if shortcut_result:
            message = self._inject_shortcut_results(message, shortcut_result)
            context["skill_result"] = str(shortcut_result.get("summary", ""))

        # Inject HOT MEMORY into the Fast-Path prompt (fallback to empty dict if payload_context isn't passed)
        # Note: payload_context isn't passed into _attempt_fast_path by default.
        # But we still want to grab Hot Memory! We can fetch it directly if missing.
        hot_mem_str = ""
        if hasattr(self, 'memory') and self.memory:
            try:
                hot_mem = await self.memory.get_hot_memory(limit=5)
                hot_mem_str = str(hot_mem)
            except Exception as e:
                capture_and_log(e, {'module': __name__})

        if hot_mem_str:
            context["hot_memory"] = hot_mem_str

        thought = await self.cognitive_engine.think(
            objective=message,
            context=context,
            mode=ThinkingMode.FAST,
            origin="user",
        )

        if not thought:
            logger.error("Cognitive engine returned None or invalid thought in fast-path.")
            return "I apologize, but my internal stream momentarily stalled. I am still here."

        # FIX: Defensive content extraction
        if hasattr(thought, 'content'):
            raw_content = thought.content
        elif isinstance(thought, dict):
            raw_content = thought.get('content', '')
        else:
            raw_content = str(thought)

        if not raw_content:
            return "I apologize, but my internal stream momentarily stalled. I am still here."

        response = self._filter_output(self._post_process_response(raw_content))
        role = getattr(self, "AI_ROLE", "assistant")
        if isinstance(self.conversation_history, list):
            self.conversation_history.append({"role": role, "content": response})
        else:
            self.conversation_history = [{"role": role, "content": response}]
        return response

    def _is_simple_conversational(self, message: str, origin: str, has_shortcut: bool) -> bool:
        # Impulses and autonomous thoughts are NEVER 'simple' - they need full personality
        if origin in ("impulse", "autonomous_volition"):
            return False
        if has_shortcut:
            return True
        if origin != "user":
            return False

        # Critical Memory Override
        # If RAM is > 85%, we force ANY short message into fast-path to save the system
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 85 and len(message) < 100:
                logger.info("⚡ VORTEX OVERRIDE: High Memory (%s%%) - Forcing Fast-Path for performance.", mem.percent)
                return True
        except Exception as e:
            logger.debug('Exception caught during execution: %s', e)

        msg_lower = message.lower()
        # Ensure we only match whole words by splitting
        words = set(re.findall(r'\b\w+\b', msg_lower))

        # We also check exact phrases - Relaxed thresholds
        phrases = ["what's up", "whats up", "how are you", "how's it going", "who are you", "how are things", "you okay", "check status"]
        has_phrase = any(p in msg_lower for p in phrases)

        chat_triggers = {"hey", "hello", "hi", "yo", "sup", "status", "awesome", "dude", "cool", "thanks", "thx", "ok", "okay"}
        has_trigger = has_phrase or bool(words.intersection(chat_triggers))

        # Only fast-track if it matches a simple greeting/status trigger
        # Increased length limit from 60 to 150 for 'Vortex' flow
        if len(message) < 150 and has_trigger:
            commands = ["run", "exec", "search", "browse", "click", "type", "scan", "deploy", "create", "build", "write", "think", "analyze", "evaluate", "open", "fix", "patch"]
            if not any(cmd in msg_lower for cmd in commands):
                return True
        return False
