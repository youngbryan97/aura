# skills/native_chat.py
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.NativeChat")

_NATIVE_CHAT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    TimeoutError,
)
_NATIVE_CHAT_THINK_TIMEOUT_SECONDS = 30.0


def _record_native_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "native_chat",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )


def _load_emitter() -> Any | None:
    try:
        from core.thought_stream import get_emitter

        return get_emitter()
    except ImportError:
        return None
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_native_degradation(
            exc,
            stage="thought_stream_emitter_load",
            action="started native chat without ThoughtStream emitter; direct skill response remains available",
        )
        return None


emitter = _load_emitter()


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _close_awaitable(awaitable: Any) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()
        return
    cancel = getattr(awaitable, "cancel", None)
    if callable(cancel):
        cancel()


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def _schedule_background_task(awaitable: Any, *, name: str) -> bool:
    if awaitable is None:
        return True
    if not (asyncio.iscoroutine(awaitable) or isinstance(awaitable, asyncio.Future)):
        return True

    try:
        from core.utils.task_tracker import get_task_tracker

        get_task_tracker().create_task(awaitable, name=name)
        return True
    except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
        _record_native_degradation(
            exc,
            stage="background_task_tracker",
            action=f"task tracker could not schedule {name}; falling back to the active event loop",
            extra={"task_name": name},
        )

    try:
        asyncio.get_running_loop().create_task(awaitable, name=name)
        return True
    except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
        _close_awaitable(awaitable)
        _record_native_degradation(
            exc,
            stage="background_task_loop_fallback",
            action=f"closed {name} coroutine because both task tracker and event loop scheduling failed",
            severity="degraded",
            extra={"task_name": name},
        )
        logger.debug("NativeChat background task %s could not be scheduled: %s", name, exc)
        return False


def _response_text(thought: Any) -> str:
    content = getattr(thought, "content", thought)
    if content is None:
        return ""
    return str(content).strip()


class NativeChatSkill(BaseSkill):
    name = "native_chat"
    description = "Conversational engine with robust dependency resolution."
    aliases = ["chat", "talk"]
    inputs = {"message": "User input to respond to."}

    def __init__(self, brain: CognitiveEngine | None = None):
        self.brain = brain

    def _resolve_brain(self, context: Mapping[str, Any]) -> Any | None:
        """Resolve the cognitive engine from injection, context, container, or global fallback."""
        if self.brain:
            return self.brain

        for key in ("brain", "cognitive_engine"):
            brain = context.get(key)
            if brain:
                self.brain = brain
                return brain

        orchestrator = context.get("orchestrator")
        for attr in ("brain", "cognitive_engine", "_cognitive_engine"):
            brain = getattr(orchestrator, attr, None)
            if brain:
                self.brain = brain
                return brain

        try:
            from core.container import ServiceContainer

            brain = ServiceContainer.get("cognitive_engine", default=None)
            if brain:
                self.brain = brain
                return brain
        except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
            _record_native_degradation(
                exc,
                stage="brain_container_lookup",
                action="continued native chat brain resolution after ServiceContainer lookup failed",
            )

        try:
            from core.brain.cognitive_engine import cognitive_engine

            if cognitive_engine:
                self.brain = cognitive_engine
                return cognitive_engine
        except ImportError as exc:
            _record_native_degradation(
                exc,
                stage="brain_global_lookup",
                action="global cognitive_engine fallback unavailable during native chat brain resolution",
            )

        return None

    async def _build_legacy_context(
        self,
        msg_str: str,
        context: dict[str, Any],
        intent_context: dict[str, Any],
    ) -> dict[str, Any]:
        memory_context = ""
        mem_sys = context.get("memory")
        if mem_sys and hasattr(mem_sys, "retrieve_context"):
            try:
                memory_context = await _maybe_await(mem_sys.retrieve_context(msg_str))
            except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
                _record_native_degradation(
                    exc,
                    stage="legacy_memory_context",
                    action="continued with empty memory context after legacy retrieval failed",
                )

        personality_context: dict[str, Any] = {}
        try:
            from core.brain.personality_engine import get_personality_engine

            personality = get_personality_engine()
            personality.respond_to_event("user_message", {"message": msg_str})
            personality_context = personality.get_emotional_context_for_response()
        except ImportError:
            personality_context = {}
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_native_degradation(
                exc,
                stage="legacy_personality_context",
                action="continued with neutral personality context after personality engine failed",
            )

        return {
            **context,
            "user_intent": intent_context,
            "memory_context": memory_context,
            "personality": personality_context,
        }

    async def _build_rich_context(
        self,
        msg_str: str,
        context: dict[str, Any],
        intent_context: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            from core.brain.context_builder import DynamicContextBuilder

            rich_context = await DynamicContextBuilder.build_rich_context(msg_str, context)
            personality_context = rich_context.get("personality", {})
            logger.info(
                "Personality State (v5.5): %s (%s)",
                personality_context.get("mood"),
                personality_context.get("tone"),
            )
            rich_context["prompt_segment"] = DynamicContextBuilder.format_for_prompt(rich_context)
            return rich_context
        except ImportError as exc:
            _record_native_degradation(
                exc,
                stage="dynamic_context_import",
                action="used legacy context builder because DynamicContextBuilder is unavailable",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_native_degradation(
                exc,
                stage="dynamic_context_build",
                action="used legacy context builder after dynamic context assembly failed",
            )
        return await self._build_legacy_context(msg_str, context, intent_context)

    async def _think(self, brain: Any, final_llm_input: str, rich_context: dict[str, Any]) -> Any:
        think = getattr(brain, "think", None)
        if callable(think):
            return await asyncio.wait_for(
                think(final_llm_input, context=rich_context, mode=ThinkingMode.CREATIVE),
                timeout=_NATIVE_CHAT_THINK_TIMEOUT_SECONDS,
            )

        generate = getattr(brain, "generate", None)
        if callable(generate):
            return await asyncio.wait_for(
                generate(final_llm_input, context=rich_context),
                timeout=_NATIVE_CHAT_THINK_TIMEOUT_SECONDS,
            )

        raise AttributeError("Resolved brain exposes neither think() nor generate().")

    async def execute(
        self, goal: Mapping[str, Any] | None, context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Execute chat with explicit recovery paths and no silent fallbacks."""
        goal_data = _mapping_or_empty(goal)
        context_data = _mapping_or_empty(context)
        degraded_stages: list[str] = []

        brain = self._resolve_brain(context_data)
        if not brain:
            msg = "Brain not found in constructor, context, ServiceContainer, or global scope."
            exc = RuntimeError(msg)
            _record_native_degradation(
                exc,
                stage="brain_resolution",
                action="failed closed without fabricating a chat response because no cognitive engine was available",
                severity="critical",
            )
            logger.critical(msg)
            return {"ok": False, "error": msg, "stage": "brain_resolution"}

        params = _mapping_or_empty(goal_data.get("params", {}))
        msg = params.get("message") or goal_data.get("message") or goal_data.get("objective")
        if msg is None or str(msg).strip() == "":
            return {"ok": False, "error": "No message provided.", "stage": "input_validation"}

        msg_str = str(msg)
        logger.info("Processing chat message (Type: %s): %s...", type(msg_str), msg_str[:50])

        try:
            from core.container import ServiceContainer

            cme = ServiceContainer.get("conversational_momentum_engine", default=None)
            if cme:
                scheduled = _schedule_background_task(
                    cme.on_new_user_message(msg_str),
                    name="native_chat.momentum",
                )
                if not scheduled:
                    degraded_stages.append("momentum")
        except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
            degraded_stages.append("momentum")
            _record_native_degradation(
                exc,
                stage="conversational_momentum",
                action="continued chat turn without momentum update",
            )
            logger.debug("NativeChat momentum update failed: %s", exc)

        intent_context: dict[str, Any] = {}
        tom = context_data.get("theory_of_mind")
        if tom:
            try:
                intent_context = dict(tom.infer_intent(msg_str, context_data) or {})
                logger.info("ToM Intent: %s", intent_context.get("pragmatic", "standard"))
            except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
                degraded_stages.append("theory_of_mind")
                _record_native_degradation(
                    exc,
                    stage="theory_of_mind",
                    action="continued chat turn with standard intent after ToM inference failed",
                )

        rich_context = await self._build_rich_context(msg_str, context_data, intent_context)

        try:
            if emitter:
                emitter.emit(
                    "Cognition",
                    f"Intent: {intent_context.get('pragmatic', 'standard')} | {msg_str[:30]}...",
                    level="info",
                )
        except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
            degraded_stages.append("thought_stream_intent_emit")
            _record_native_degradation(
                exc,
                stage="thought_stream_intent_emit",
                action="continued chat generation after cognition visibility emit failed",
            )

        final_llm_input = msg_str
        if "prompt_segment" in rich_context:
            final_llm_input = f"{rich_context['prompt_segment']}\n\nUser Input: {msg_str}"

        try:
            thought = await self._think(brain, final_llm_input, rich_context)
            response = _response_text(thought)
            if not response:
                raise RuntimeError("Cognitive engine returned an empty response.")
        except asyncio.CancelledError:
            raise
        except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
            _record_native_degradation(
                exc,
                stage="cognitive_generation",
                action="failed closed with explicit chat error; no fabricated response emitted",
                severity="critical",
            )
            logger.error("Cognitive failure: %s", exc, exc_info=True)
            return {
                "ok": False,
                "error": f"Cognitive failure: {exc}",
                "stage": "cognitive_generation",
            }

        if emitter:
            try:
                logger.info("Emitting chat response to ThoughtStream: %s...", response[:50])
                emitter.emit("AURA", response, level="chat")
            except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
                degraded_stages.append("thought_stream_response_emit")
                _record_native_degradation(
                    exc,
                    stage="thought_stream_response_emit",
                    action="returned chat response directly after ThoughtStream response emit failed",
                )
                logger.error("Failed to emit chat response: %s", exc)
        else:
            logger.warning("No ThoughtStream emitter found in NativeChatSkill.")

        mem_sys = context_data.get("memory")
        if mem_sys:
            try:
                user_write = mem_sys.remember(
                    msg_str,
                    metadata={"role": "user", "intent": intent_context.get("pragmatic")},
                )
                if not _schedule_background_task(user_write, name="native_chat.remember_user"):
                    degraded_stages.append("memory_user_write")
            except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
                degraded_stages.append("memory_user_write")
                _record_native_degradation(
                    exc,
                    stage="memory_user_write",
                    action="returned chat response after user memory write could not be queued",
                )

            try:
                aura_write = mem_sys.remember(
                    response,
                    metadata={"role": "aura", "mode": "chat"},
                )
                if not _schedule_background_task(aura_write, name="native_chat.remember_aura"):
                    degraded_stages.append("memory_aura_write")
            except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
                degraded_stages.append("memory_aura_write")
                _record_native_degradation(
                    exc,
                    stage="memory_aura_write",
                    action="returned chat response after Aura memory write could not be queued",
                )

        try:
            orchestrator = context_data.get("orchestrator")
            if orchestrator and hasattr(orchestrator, "biorhythm"):
                orchestrator.biorhythm.mark_interaction()
        except _NATIVE_CHAT_RECOVERABLE_ERRORS as exc:
            degraded_stages.append("biorhythm")
            _record_native_degradation(
                exc,
                stage="biorhythm",
                action="returned chat response after biorhythm interaction mark failed",
            )

        result: dict[str, Any] = {
            "ok": True,
            "response": response,
            "summary": "Replied to user.",
        }
        if degraded_stages:
            result["degraded"] = True
            result["degraded_stages"] = sorted(set(degraded_stages))
        return result
