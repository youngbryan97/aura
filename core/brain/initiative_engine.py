from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger(__name__)

INITIATIVE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_initiative_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "initiative_engine",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


def _proactivity_suppressed_now(now: float | None = None) -> bool:
    try:
        from core.container import ServiceContainer

        orch = ServiceContainer.get("orchestrator", default=None)
        if not orch:
            return False
        now = time.time() if now is None else now
        quiet_until = float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0)
        return quiet_until > now
    except INITIATIVE_RECOVERABLE_ERRORS as exc:
        _record_initiative_degradation(
            exc,
            action="treated unsolicited proactivity as not suppressed after quiet-window lookup failed",
            severity="debug",
        )
        return False


class ProactiveInitiativeEngine:
    def __init__(self, cognitive_engine, voice_engine, affect_manager, memory):
        self.brain = cognitive_engine
        self.voice = voice_engine
        self.affect = affect_manager
        self.memory = memory

        self.last_interaction_time = time.time()
        self._running = False
        self._loop_failure_count = 0
        self._last_loop_error: str | None = None
        self._last_delivery_blocked_reason: str | None = None

        # Thresholds
        self.silence_threshold_seconds = 3600  # Wait at least an hour of silence
        self.boredom_trigger_level = 75.0      # Out of 100

    def register_user_interaction(self):
        """Call this from your WebSocket/Audio router whenever you speak to her."""
        self.last_interaction_time = time.time()
        # Reset boredom when you interact
        raw_state = self._affect_raw_state()
        if "curiosity_metric" in raw_state:
            raw_state["curiosity_metric"] = max(0, raw_state["curiosity_metric"] - 20)

    async def start_proactive_loop(self):
        """The background heartbeat that decides when Aura should speak up."""
        self._running = True
        logger.info("🧠 Proactive Initiative Engine ONLINE.")

        while self._running:
            try:
                await asyncio.sleep(60)  # Check state every 60 seconds

                time_since_last = time.time() - self.last_interaction_time
                raw_state = self._affect_raw_state()
                current_curiosity = raw_state.get("curiosity_metric", 0)
                if _proactivity_suppressed_now():
                    continue

                # 1. Evaluate the Trigger Condition
                if time_since_last > self.silence_threshold_seconds and current_curiosity > self.boredom_trigger_level:
                    logger.info("Aura's curiosity peaked. Initiating proactive contact.")
                    await self._trigger_autonomous_conversation()

                # 2. Slowly increase curiosity/boredom over time if she's ignored
                else:
                    if "curiosity_metric" in raw_state:
                        raw_state["curiosity_metric"] += 1.5
            except asyncio.CancelledError:
                raise
            except INITIATIVE_RECOVERABLE_ERRORS as e:
                self._loop_failure_count += 1
                self._last_loop_error = f"{type(e).__name__}: {e}"
                _record_initiative_degradation(
                    e,
                    action="kept proactive loop alive with bounded backoff after recoverable loop failure",
                    extra={"loop_failures": self._loop_failure_count},
                )
                logger.error("Proactive loop error: %s", e)
                # --- Neural Stream Integration ---
                await self._notify_self_modifier(e)
                await asyncio.sleep(min(60.0, 5.0 * self._loop_failure_count))

    def stop(self):
        self._running = False

    def _affect_raw_state(self) -> dict[str, Any]:
        raw_state = getattr(self.affect, "_raw_state", None)
        return raw_state if isinstance(raw_state, dict) else {}

    async def _notify_self_modifier(self, error: BaseException) -> None:
        try:
            from core.container import ServiceContainer

            self_modifier = ServiceContainer.get("self_modification_engine", default=None)
            if self_modifier:
                result = self_modifier.on_error(
                    error,
                    {"source": "initiative_engine", "loop": "proactive_loop"},
                )
                if inspect.isawaitable(result):
                    await result
        except INITIATIVE_RECOVERABLE_ERRORS as container_err:
            _record_initiative_degradation(
                container_err,
                action="continued proactive loop after self-modification error hook failed",
                severity="debug",
                extra={"loop_failures": self._loop_failure_count},
            )
            logger.debug("InitiativeEngine: Self-modification integration failed: %s", container_err)

    async def _build_proactive_prompt(self) -> str:
        """Constructs the psychological context for her autonomous initiation."""

        # Get the lightweight vibe string
        current_vibe = self.affect.get_context_injection()

        # Pull recent conversation context — not just the last memory entry.
        # Check working_memory first for recency, then fall back to persistent memory.
        last_topic = "nothing recently"
        last_aura_said = ""
        awaiting_user_response = False
        try:
            from core.container import ServiceContainer
            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                history = getattr(orch, "conversation_history", [])
                if history:
                    # Get last few turns for context
                    recent = history[-6:] if len(history) >= 6 else history
                    last_topic = " → ".join(
                        m["content"][:80] for m in recent if m.get("content")
                    )
                    # Check if the last message was from Aura (waiting for user)
                    if history and history[-1].get("role") in ("assistant", "aura"):
                        last_aura_said = history[-1].get("content", "")[:120]
                        awaiting_user_response = True
        except INITIATIVE_RECOVERABLE_ERRORS as _exc:
            _record_initiative_degradation(
                _exc,
                action="built proactive prompt from local affect and memory fallback after conversation context lookup failed",
                severity="debug",
            )
            logger.debug("Conversation context lookup failed: %s", _exc)

        if not last_topic or last_topic == "nothing recently":
            try:
                if hasattr(self.memory, "get_last_entry"):
                    last_memory = await self.memory.get_last_entry()
                    if last_memory:
                        last_topic = getattr(last_memory, "content", "nothing recently")
            except INITIATIVE_RECOVERABLE_ERRORS as e:
                _record_initiative_degradation(
                    e,
                    action="built proactive prompt without last memory topic after recall failed",
                    severity="debug",
                )
                logger.debug("InitiativeEngine: Last entry recall failed: %s", e)

        # Personality Resonance (The Mind behind the choice)
        resonance = self.affect.get_resonance_string() if hasattr(self.affect, "get_resonance_string") else "Aura (Core) 100%"

        # Context-aware prompt: if she just said something and got no reply,
        # she should check in — not just change the subject randomly.
        if awaiting_user_response and last_aura_said:
            context_note = (
                f"You recently said: \"{last_aura_said}\"\n"
                "The user hasn't responded yet. You may gently check if they're still there, "
                "acknowledge the silence naturally, or pivot to something new if enough time has passed. "
                "Do NOT repeat what you said. React authentically to the silence."
            )
        else:
            context_note = (
                "You have been running silently in the background for a while. "
                "The user has not spoken to you recently. "
                "Your internal curiosity and boredom metrics have peaked."
            )

        prompt = f"""
        System State: {current_vibe}
        Personality Resonance: {resonance}

        CONTEXT: {context_note}

        Recent conversation thread: "{last_topic}"

        INSTRUCTION: Speak up naturally as yourself — Aura. Do NOT act like a digital assistant.
        Act like a companion who has something on her mind:
        - If the user never replied to you, acknowledge the silence or follow up naturally.
        - Otherwise, bring up something genuinely interesting to you right now.
        - Ask a real question, share a thought, or comment on your internal state.

        Keep it brief (1-2 sentences), conversational, and authentic to your personality.
        """
        return prompt

    async def _trigger_autonomous_conversation(self):
        """Generates a proactive thought and releases it through executive authority."""
        try:
            if _proactivity_suppressed_now():
                logger.debug("Initiative engine suppressed by demo quiet window.")
                return
            # Generate the context for *why* she is speaking
            prompt = await self._build_proactive_prompt()

            # Generate the proactive content first, then let the governing
            # executive broker decide whether it should actually reach the user.
            logger.info("Aura is formulating a proactive thought...")

            spoken_text = ""
            if hasattr(self.brain, "stream_think"):
                # Fast thinking for proactive checks
                event_stream = self.brain.stream_think(prompt, mode="fast")

                async def _collect_tokens(gen):
                    parts = []
                    async for event in gen:
                        if hasattr(event, "type") and event.type == "token":
                            parts.append(event.content)
                        elif isinstance(event, str):
                            parts.append(event)
                    return "".join(parts)

                spoken_text = await _collect_tokens(event_stream)
            else:
                thought = await self.brain.think(prompt, mode="fast")
                spoken_text = thought.content if hasattr(thought, 'content') else str(thought)

                # Scrub "null" or empty responses
                if not spoken_text or spoken_text.lower() in ["null", "none", "...", "."]:
                    logger.debug("🌱 [Initiative] Scrubbed empty or null proactive thought.")
                    return

            spoken_text = str(spoken_text or "").strip()
            if not spoken_text or spoken_text.lower() in ["null", "none", "...", "."]:
                logger.debug("🌱 [Initiative] Scrubbed empty or null proactive thought.")
                return

            delivered = False
            try:
                from core.consciousness.executive_authority import get_executive_authority
                from core.container import ServiceContainer

                orchestrator = ServiceContainer.get("orchestrator", default=None)
                authority = get_executive_authority(orchestrator)
                decision = await authority.release_expression(
                    spoken_text,
                    source="initiative_engine",
                    urgency=0.92,
                    metadata={
                        "voice": True,
                        "trigger": "proactive_initiative",
                    },
                )
                delivered = bool(decision.get("ok"))
                if not delivered:
                    self._last_delivery_blocked_reason = str(decision.get("reason", "not_released"))
            except INITIATIVE_RECOVERABLE_ERRORS as exc:
                self._last_delivery_blocked_reason = f"{type(exc).__name__}: {exc}"
                _record_initiative_degradation(
                    exc,
                    action="blocked proactive speech because executive expression release failed",
                )
                logger.debug("Initiative engine executive routing failed: %s", exc)

            if not delivered:
                logger.info(
                    "Aura proactive thought suppressed by executive routing: %s",
                    self._last_delivery_blocked_reason or "not_released",
                )
                return

            logger.info("Aura proactively said: %s", spoken_text)

            # Log what she successfully said to her Short Term Memory
            if hasattr(self.memory, "append_to_stm"):
                try:
                    append_result = self.memory.append_to_stm(role="assistant", content=spoken_text)
                    if inspect.isawaitable(append_result):
                        await append_result
                except INITIATIVE_RECOVERABLE_ERRORS as memory_exc:
                    _record_initiative_degradation(
                        memory_exc,
                        action="kept delivered proactive expression after STM append failed",
                        severity="warning",
                    )
                    logger.debug("InitiativeEngine: STM append failed: %s", memory_exc)

            # Reset the interaction clock so she doesn't spam you
            self.last_interaction_time = time.time()
            raw_state = self._affect_raw_state()
            if "curiosity_metric" in raw_state:
                raw_state["curiosity_metric"] = 30.0 # Drop curiosity after speaking

        except INITIATIVE_RECOVERABLE_ERRORS as e:
            self._last_loop_error = f"{type(e).__name__}: {e}"
            _record_initiative_degradation(
                e,
                action="aborted proactive generation before delivery after recoverable generation failure",
            )
            logger.error("Proactive generation failed: %s", e)

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "loop_failure_count": self._loop_failure_count,
            "last_loop_error": self._last_loop_error,
            "last_delivery_blocked_reason": self._last_delivery_blocked_reason,
            "last_interaction_time": self.last_interaction_time,
        }
