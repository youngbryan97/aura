import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.desktop_task_contract import (
    DESKTOP_TASK_ALLOWED_ACTIONS,
    desktop_task_planning_schema,
)
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.VoiceBridge")


_WAKE_PREFIX_RE = re.compile(r"^\s*(?:hey|hi|okay|ok)?\s*aura[\s,;:.-]*", re.IGNORECASE)


class VoiceConversationBridge:
    """
    The neural bridge between the Voice Pipeline (Senses) and the 
    Conversation Engine (Cognition).
    
    Handles low-latency streaming of thoughts and speech directly
    to the voice output buffer.
    """
    def __init__(self, orchestrator, conversation_engine):
        self._orch = orchestrator
        self._engine = conversation_engine
        self._bus = get_event_bus()
        self._active_utterance_task: asyncio.Task | None = None

    @staticmethod
    def _normalize_voice_text(text: str) -> str:
        cleaned = _WAKE_PREFIX_RE.sub("", str(text or "").strip()).strip()
        return cleaned or str(text or "").strip()

    @staticmethod
    def _thought_to_text(thought: Any) -> str:
        content = getattr(thought, "content", None)
        if content is None and isinstance(thought, dict):
            content = thought.get("content") or thought.get("response")
        return str(content if content is not None else thought or "").strip()

    @staticmethod
    def _looks_like_desktop_objective(text: str) -> bool:
        return looks_like_desktop_objective(text)

    async def _run_cognitive_engine(self, text: str) -> str | None:
        try:
            engine = ServiceContainer.get("cognitive_engine", default=None)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_bridge",
                exc,
                severity="warning",
                action="used the injected conversation engine if available; otherwise returned no cognitive reply",
            )
            logger.warning("Voice Bridge CognitiveEngine lookup failed: %s", exc)
            engine = None
        if engine is None and hasattr(self._engine, "think"):
            engine = self._engine
        if engine is None or not hasattr(engine, "think"):
            return None
        try:
            from core.brain.types import ThinkingMode

            desktop_execution_contract = self._looks_like_desktop_objective(text)
            context = {
                "route": "voice_desktop",
                "source": "voice",
                "origin": "voice",
                "foreground_request": True,
                "user_facing": True,
            }
            if desktop_execution_contract:
                context.update(
                    {
                        "desktop_execution_contract": True,
                        "allow_heuristic_desktop_plan": True,
                        "desktop_task_planning_schema": desktop_task_planning_schema(),
                        "desktop_task_allowed_actions": DESKTOP_TASK_ALLOWED_ACTIONS,
                        "max_tokens": 1024,
                        "num_predict": 1024,
                        "deep_handoff": False,
                        "allow_deep_handoff": False,
                    }
                )
            thought = await engine.think(
                text,
                context=context,
                mode=(
                    ThinkingMode.SLOW
                    if desktop_execution_contract
                    else ThinkingMode.FAST
                ),
                origin="voice",
                foreground_request=True,
                is_background=False,
                priority=True,
            )
            reply = self._thought_to_text(thought)
            return reply or None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
            record_degradation("voice_bridge", exc)
            logger.warning("Voice Bridge CognitiveEngine path failed: %s", exc)
            return None

    async def _execute_desktop_objective(self, text: str, cognitive_reply: str) -> dict[str, Any] | None:
        if not self._looks_like_desktop_objective(text):
            return None
        try:
            engine = ServiceContainer.get("capability_engine", default=None)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_bridge",
                exc,
                severity="degraded",
                action="failed the spoken desktop objective explicitly without attempting an ungoverned fallback",
            )
            logger.warning("Voice desktop capability lookup failed: %s", exc)
            engine = None
        if engine is None or not hasattr(engine, "execute"):
            return {
                "ok": False,
                "status": "capability_engine_unavailable",
                "error": "Voice desktop objective requires the governed capability engine.",
            }
        context = {
            "origin": "voice",
            "source": "voice",
            "route": "voice.desktop_objective",
            "objective": text[:500],
            "message": text[:500],
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_requested_action": True,
            "desktop_execution_contract": True,
            "allow_heuristic_desktop_plan": True,
            "disable_outer_skill_retry": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
            "verification_required": True,
            "desktop_task_document_body": cognitive_reply,
            "cognitive_reply": cognitive_reply,
        }
        result = await engine.execute(
            "desktop_task",
            {
                "objective": text,
                "steps": [],
                "disable_outer_skill_retry": True,
                "desktop_execution_contract": True,
                "allow_heuristic_desktop_plan": True,
            },
            context=context,
        )
        if isinstance(result, dict):
            return result
        return {"ok": bool(result), "result": result}

    async def _process_voice_input(self, text: str) -> str | None:
        normalized = self._normalize_voice_text(text)
        reply = await self._run_cognitive_engine(normalized)
        if reply:
            desktop_result = await self._execute_desktop_objective(normalized, reply)
            if desktop_result is not None:
                completed = int(desktop_result.get("steps_completed") or 0)
                requested = int(desktop_result.get("steps_requested") or 0)
                if desktop_result.get("ok"):
                    summary = str(desktop_result.get("summary") or "").strip()
                    return (
                        summary
                        or f"I completed the spoken desktop task through governed desktop control ({completed}/{requested} steps)."
                    )
                status = str(desktop_result.get("status") or desktop_result.get("error") or "desktop_task_failed")
                return (
                    "I routed that spoken request through CognitiveEngine and governed desktop control, "
                    f"but it did not complete: {status}. Completed {completed}/{requested} steps."
                )
            return reply

        if self._looks_like_desktop_objective(normalized):
            return (
                "The spoken desktop request required CognitiveEngine before governed desktop control, "
                "but the live cognitive path did not produce an acceptable reply. I am not using the "
                "legacy voice fallback to claim or attempt desktop work."
            )

        if self._orch and hasattr(self._orch, "process_user_input"):
            response = await self._orch.process_user_input(normalized, origin="voice")
            return str(response or "").strip() or None
        return None
        
    async def process_voice_input(self, text: str):
        """
        Routes transcribed voice input into the cognitive engine.
        Supports real-time interruption of current speaking tasks.
        """
        logger.info("🎙️ Voice Bridge: Routing utterance -> %s...", f"{text[:50]}")
        
        # 1. Interrupt any current TTS or thinking
        if self._active_utterance_task and not self._active_utterance_task.done():
            self._active_utterance_task.cancel()
            
        # 2. Process through the full cognitive pipeline
        self._active_utterance_task = get_task_tracker().create_task(
            self._process_voice_input(text),
            name="voice_bridge.process_voice_input",
        )
        
        try:
            response = await self._active_utterance_task
            return response
        except asyncio.CancelledError:
            logger.debug("Voice Bridge: Task cancelled due to barge-in/new input")
            return None
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('voice_bridge', e)
            logger.error("Voice Bridge process input error: %s", e)
            return None

    async def stream_response_to_voice(self, stream: AsyncGenerator[str, None]):
        """
        Feeds tokens/chunks from the LLM directly into the voice pipeline's
        streaming buffer for low-latency response.
        Chunks tokens into speakable clauses before sending to the engine.
        """
        voice_engine = ServiceContainer.get("voice_presence", default=None)
        if not voice_engine:
            logger.debug("VoiceBridge: No voice presence engine found. Dropping stream.")
            async for _ in stream:
                pass  # no-op: intentional
            return

        buffer: str = ""
        # Delimiters that constitute a "speakable clause"
        delimiters = re.compile(r'([.!?\n]+|,\s)')
        
        logger.info("🎙️ Voice Bridge: Streaming response chunks to voice engine...")
        try:
            async for token in stream:
                if self._active_utterance_task and self._active_utterance_task.cancelled():
                    logger.debug("VoiceBridge: Stream aborted due to barge-in.")
                    break
                    
                buffer = str(buffer) + str(token)
                
                # If we encounter a clause-ending delimiter, ship it to the TTS engine
                match = delimiters.search(buffer)
                if match:
                    split_idx = match.end()
                    chunk = buffer[:split_idx].strip()
                    buffer = buffer[split_idx:]
                    
                    if chunk:
                        # Fallback for old TTSEngine vs DecoupledVoiceEngine
                        if hasattr(voice_engine, "speak_nonblocking"):
                            voice_engine.speak_nonblocking(chunk)
                        elif hasattr(voice_engine, "speak"):
                            # If it's pure async we schedule it
                            get_task_tracker().create_task(voice_engine.speak(chunk))
                            
            # Flush any remaining text in the buffer
            if buffer.strip():
                if hasattr(voice_engine, "speak_nonblocking"):
                    voice_engine.speak_nonblocking(buffer.strip())
                elif hasattr(voice_engine, "speak"):
                    get_task_tracker().create_task(voice_engine.speak(buffer.strip()))
                    
        except (RuntimeError, AttributeError, TypeError) as e:
            record_degradation('voice_bridge', e)
            logger.error("VoiceBridge Stream Error: %s", e)
