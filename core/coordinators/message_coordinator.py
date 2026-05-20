"""Message Coordinator — message queue, dispatch, conversation history,
streaming, and impulse handling.

Extracted from orchestrator.py as part of the God Object decomposition.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.tagged_reply_queue import reply_delivery_scope
from core.utils.queues import unpack_priority_message
from core.utils.task_tracker import get_task_tracker, task_tracker

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 60_000
MAX_ORIGIN_CHARS = 80
MAX_HISTORY_ENTRIES = 300
DISPATCH_CONCURRENCY = 10

_COORDINATOR_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    ConnectionError,
)


def _emit_message_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "message_coordinator",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("message_coordinator", error)


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _fallback_reply(stage: str) -> str:
    return (
        "I hit a recoverable message-routing fault before I could complete that reply. "
        f"The failure was captured at {stage}, and I avoided inventing an answer."
    )


def _trim_history(history: Any) -> None:
    if isinstance(history, list) and len(history) > MAX_HISTORY_ENTRIES:
        del history[:-MAX_HISTORY_ENTRIES]


class MessageCoordinator:
    """Handles message intake, dispatch, history recording, and streaming."""

    def __init__(self, orch):
        self.orch = orch

    # ------------------------------------------------------------------
    # Queue Management
    # ------------------------------------------------------------------

    async def acquire_next_message(self) -> str | None:
        """Get next message from queue. Returns None if queue is empty."""
        orch = self.orch
        try:
            raw = orch.message_queue.get_nowait()
            msg, _origin = unpack_priority_message(raw)
            msg = _safe_text(msg, max_chars=MAX_MESSAGE_CHARS)

            logger.info("Processing queued message: %s", str(msg)[:100])
            if hasattr(orch, "liquid_state") and orch.liquid_state:
                update = orch.liquid_state.update(delta_curiosity=0.2, delta_frustration=-0.1)
                try:
                    get_task_tracker().create_task(
                        update,
                        name="message_coordinator.liquid_state_update",
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    close = getattr(update, "close", None)
                    if callable(close):
                        close()
                    _emit_message_fault(
                        exc,
                        action="continued message processing without liquid-state side update",
                        severity="warning",
                        stage="acquire_next_message.liquid_state",
                    )
            orch._last_thought_time = time.time()
            return msg
        except asyncio.QueueEmpty:
            return None
        except _COORDINATOR_RECOVERABLE_ERRORS as e:
            _emit_message_fault(
                e,
                action="dropped malformed queued message before dispatch",
                severity="degraded",
                stage="acquire_next_message",
            )
            logger.error("Error acquiring message: %s", e)
            return None

    def enqueue_message(self, message: Any, priority: int = 20):
        """Standard interface for injecting messages into the core loop."""
        self.orch.enqueue_message(message, priority=priority)

    def enqueue_from_thread(self, message: Any, origin: str = "user", priority: int = 10):
        """Safely enqueue a message from a synchronous thread to the async loop."""
        self.orch.enqueue_from_thread(message, origin=origin, priority=priority)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch_message(self, message: str, origin: str = "user"):
        """Dispatch message to the async handler with bounded concurrency."""
        from core.orchestrator.types import _bg_task_exception_handler

        orch = self.orch
        if not hasattr(orch, "_dispatch_semaphore"):
            orch._dispatch_semaphore = asyncio.Semaphore(DISPATCH_CONCURRENCY)

        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        origin = _safe_text(origin, "user", max_chars=MAX_ORIGIN_CHARS)

        async def _bounded_handler():
            async with orch._dispatch_semaphore:
                await self.handle_incoming_message(message, origin=origin)

        handler = _bounded_handler()
        try:
            task_tracker.create_task(
                handler,
                name="message_coordinator.dispatch",
            ).add_done_callback(_bg_task_exception_handler)
        except (RuntimeError, TypeError, ValueError) as exc:
            with contextlib.suppress(RuntimeError):
                handler.close()
            _emit_message_fault(
                exc,
                action="failed closed before dispatch because task tracker rejected handler",
                severity="degraded",
                stage="dispatch_message.task_tracker",
                extra={"origin": origin},
            )
            return
        self.emit_dispatch_telemetry(message)

    def emit_dispatch_telemetry(self, message: str):
        """Log dispatch event to thought stream."""
        try:
            from core.thought_stream import get_emitter

            if message.startswith("Impulse:"):
                label = "Impulse ⚡"
            elif message.startswith("Thought:"):
                label = "Thought 💭"
            else:
                label = "User"
            get_emitter().emit(f"Input ({label})", message[:120], level="info")
        except _COORDINATOR_RECOVERABLE_ERRORS as exc:
            _emit_message_fault(
                exc,
                action="continued dispatch without thought-stream telemetry",
                severity="warning",
                stage="dispatch.telemetry",
            )
            logger.error("Dispatch telemetry failure: %s", exc)

    # ------------------------------------------------------------------
    # Message Processing
    # ------------------------------------------------------------------

    async def process_message(self, message: str) -> dict[str, Any]:
        """Backward compatibility for main.py. Processes message and returns response."""
        orch = self.orch
        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        with reply_delivery_scope("user") as session_id:
            await self.handle_incoming_message(message, origin="user")
            try:
                if hasattr(orch.reply_queue, "get_for_origin"):
                    reply = await orch.reply_queue.get_for_origin(
                        "user",
                        session_id=session_id,
                        timeout=30.0,
                    )
                    if reply is None:
                        return {"ok": False, "error": "Thinking timeout (30s)"}
                else:
                    reply = await asyncio.wait_for(orch.reply_queue.get(), timeout=30)
                return {"ok": True, "response": reply}
            except (OSError, ConnectionError, TimeoutError) as e:
                _emit_message_fault(
                    e,
                    action="returned timeout response after reply queue wait failed",
                    severity="degraded",
                    stage="process_message.reply_wait",
                )
                logger.error("Timed out waiting for reply to: %s", message[:50])
                return {"ok": False, "error": f"Response timeout: {str(e)}"}

    async def process_user_input(self, message: str, origin: str = "user") -> str | None:
        """Public API for injecting user/voice input.
        Returns the generated reply after processing.
        Bypasses the message queue for immediate priority processing.
        """
        orch = self.orch
        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        origin = _safe_text(origin, "user", max_chars=MAX_ORIGIN_CHARS)
        if (
            origin in ("user", "voice")
            and orch._current_thought_task
            and not orch._current_thought_task.done()
        ):
            logger.info("Interruption: user input detected. Cancelling autonomous thought.")
            orch._current_thought_task.cancel()
            try:
                await orch._current_thought_task
            except asyncio.CancelledError:
                logger.debug("Previous autonomous thought cancelled.")
            except _COORDINATOR_RECOVERABLE_ERRORS as exc:
                _emit_message_fault(
                    exc,
                    action="continued direct user processing after cancelling prior task failed",
                    severity="warning",
                    stage="process_user_input.cancel_prior",
                )
        try:
            logger.info("DIRECT processing user message: %s...", message[:80])
            with reply_delivery_scope(origin) as session_id:
                await self.handle_incoming_message(message, origin=origin)
                if origin in ("user", "voice", "admin"):
                    try:
                        if hasattr(orch.reply_queue, "get_for_origin"):
                            reply = await orch.reply_queue.get_for_origin(
                                origin,
                                session_id=session_id,
                                timeout=240.0,
                            )
                        else:
                            reply = await asyncio.wait_for(orch.reply_queue.get(), timeout=240.0)

                        # The frontend relies on 'chat_stream_chunk' from `state_machine.py`
                        # We NO LONGER emit 'chat_response' here to prevent duplicate UI rendering.
                        if reply is not None:
                            return reply
                    except TimeoutError as _exc:
                        _emit_message_fault(
                            _exc,
                            action="moved direct user turn to background after reply wait timeout",
                            severity="degraded",
                            stage="process_user_input.reply_wait",
                            extra={"origin": origin},
                        )

                    logger.warning("Timed out waiting for cognitive reply after 240s.")
                    if orch._current_thought_task and not orch._current_thought_task.done():
                        logger.info("Cognitive task is still RUNNING. Moving to background.")
                        return "The cognitive task is still running in the background; the live reply timed out and was logged as degraded."
                    return "The cognitive loop timed out before producing a coherent reply; the failure was logged for recovery."
            return None
        except asyncio.QueueFull:
            logger.warning("Message queue full. Input dropped.")
            return "The processing queue is overloaded, so this input was not accepted into the live reply path."
        except _COORDINATOR_RECOVERABLE_ERRORS as exc:
            _emit_message_fault(
                exc,
                action="returned bounded direct-input fallback after processing failure",
                severity="degraded",
                stage="process_user_input",
                extra={"origin": origin},
            )
            return _fallback_reply("direct input")

    async def handle_incoming_message(self, message: Any, origin: str = "user"):
        """Route an incoming message through the deterministic State Machine pipeline."""
        orch = self.orch
        payload_context = {}
        if isinstance(message, tuple):
            while isinstance(message, tuple):
                message = message[-1]

        if isinstance(message, dict):
            payload_context = message.get("context", {})
            origin = message.get("origin", origin)
            message = message.get("content", str(message))
        if not isinstance(payload_context, dict):
            payload_context = {}
        if isinstance(message, str):
            if origin == "user" and message.startswith("Impulse:"):
                origin = "impulse"
                message = message.replace("Impulse:", "").strip()
            elif message.startswith("Thought:"):
                origin = "autonomous_volition"
                message = message.replace("Thought:", "").strip()
            elif message.startswith("[VOICE]"):
                origin = "voice"
                message = message.replace("[VOICE]", "").strip()
            elif message.startswith("[ADMIN]"):
                origin = "admin"
                message = message.replace("[ADMIN]", "").strip()
        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        origin = _safe_text(origin, "user", max_chars=MAX_ORIGIN_CHARS)
        logger.info("Processing message (%s): %s...", origin, message[:100])
        orch.status.is_processing = True
        try:
            await orch.hooks.trigger("on_message", message=message, origin=origin)
            if orch._current_thought_task is not None and not orch._current_thought_task.done():
                if origin == "user":
                    logger.info("🛑 Interrupting previous task for user...")
                    orch._current_thought_task.cancel()
                    try:
                        await orch._current_thought_task
                    except asyncio.CancelledError:
                        logger.debug("Previous task cancelled successfully.")

            async def _execute_and_reply():
                try:
                    intent = await orch.intent_router.classify(message, payload_context)
                    final_response = await orch.state_machine.execute(
                        intent, message, payload_context
                    )
                    self.record_message_in_history(message, origin)
                    orch.conversation_history.append(
                        {"role": orch.AI_ROLE, "content": final_response}
                    )
                    _trim_history(orch.conversation_history)
                    if origin in ("user", "voice", "admin") and orch.reply_queue:
                        try:
                            orch.reply_queue.put_nowait(final_response)
                        except asyncio.QueueFull:
                            _emit_message_fault(
                                asyncio.QueueFull(),
                                action="dropped completed reply because reply queue was full",
                                severity="degraded",
                                stage="handle_incoming_message.reply_queue_full",
                                extra={"origin": origin},
                            )
                except asyncio.CancelledError:
                    raise
                except _COORDINATOR_RECOVERABLE_ERRORS as e:
                    fallback = _fallback_reply("state_machine")
                    _emit_message_fault(
                        e,
                        action="returned bounded fallback after state-machine execution failed",
                        severity="degraded",
                        stage="handle_incoming_message.execute",
                        extra={"origin": origin},
                    )
                    logger.error("State machine execution failed: %s", e)
                    if origin in ("user", "voice", "admin") and orch.reply_queue:
                        with contextlib.suppress(asyncio.QueueFull):
                            orch.reply_queue.put_nowait(fallback)
                finally:
                    orch.status.is_processing = False

            runner = _execute_and_reply()
            try:
                orch._current_thought_task = task_tracker.create_task(
                    runner,
                    name="message_coordinator.execute_and_reply",
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                with contextlib.suppress(RuntimeError):
                    runner.close()
                _emit_message_fault(
                    exc,
                    action="failed closed before state-machine task could be scheduled",
                    severity="degraded",
                    stage="handle_incoming_message.task_tracker",
                    extra={"origin": origin},
                )
                orch.status.is_processing = False
                if origin in ("user", "voice", "admin") and orch.reply_queue:
                    with contextlib.suppress(asyncio.QueueFull):
                        orch.reply_queue.put_nowait(_fallback_reply("task scheduling"))
        except _COORDINATOR_RECOVERABLE_ERRORS as e:
            _emit_message_fault(
                e,
                action="failed closed before message could enter state machine",
                severity="degraded",
                stage="handle_incoming_message.pre_execute",
                extra={"origin": origin},
            )
            logger.error("Error in handle_incoming_message: %s", e)
            orch.status.is_processing = False
            if origin in ("user", "voice", "admin") and orch.reply_queue:
                with contextlib.suppress(asyncio.QueueFull):
                    orch.reply_queue.put_nowait(_fallback_reply("pre-execution"))

    def record_message_in_history(self, message: str, origin: str):
        """Record the incoming message with appropriate role/prefix."""
        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        if origin == "autonomous_volition":
            prefix = "AUTONOMOUS GOAL: "
            role = "internal"
        elif origin == "impulse":
            prefix = "IMPULSE (speak to user): "
            role = "internal"
        else:
            prefix = ""
            role = "user"
        self.orch.conversation_history.append({"role": role, "content": f"{prefix}{message}"})
        _trim_history(self.orch.conversation_history)

    # ------------------------------------------------------------------
    # Impulses
    # ------------------------------------------------------------------

    async def handle_impulse(self, impulse: str):
        """Handle an autonomous impulse from the Consciousness Core."""
        impulse = _safe_text(impulse, max_chars=MAX_MESSAGE_CHARS)
        logger.info("Processing impulse: %s", impulse)
        directives = {
            "explore_knowledge": "I'm curious about something in my knowledge base. I should explore it.",
            "seek_novelty": "I'm feeling a bit idle. I think I'll look for something new to learn or do.",
            "deep_reflection": "I'm going to take a moment for deep reflection on my recent experiences.",
        }
        message = directives.get(impulse, f"I have an internal impulse: {impulse}")
        await self.process_user_input(message, origin="impulse")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def chat_stream(self, message: str):
        """v5.2: Stream tokens from the cognitive engine.
        Bypasses wait-loops and queues for maximum speed.
        """
        from core.brain.cognitive_engine import ThinkingMode

        orch = self.orch
        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        orch.status.is_processing = True
        try:
            reflex = orch._check_reflexes(message)
            if reflex:
                yield reflex
                orch.conversation_history.append({"role": "user", "content": message})
                orch.conversation_history.append({"role": orch.AI_ROLE, "content": reflex})
                return
            tier = "light"
            try:
                from core.ops.thinking_mode import ModeRouter

                tier = ModeRouter(orch.reflex_engine).route(message).value
            except _COORDINATOR_RECOVERABLE_ERRORS as exc:
                _emit_message_fault(
                    exc,
                    action="continued stream with light tier after mode routing failed",
                    severity="warning",
                    stage="chat_stream.mode_route",
                )
                logger.debug("Suppressed: %s", exc)
            context = orch._get_cleaned_history_context(8)
            try:
                from core.container import get_container

                container = get_container()
                ls = container.get("liquid_state")
                context["liquid_state"] = ls.get_status()
                logger.debug("TOOL EXECUTION: Injected liquid_state: %s", context["liquid_state"])
            except _COORDINATOR_RECOVERABLE_ERRORS as e:
                _emit_message_fault(
                    e,
                    action="continued stream without liquid-state context injection",
                    severity="warning",
                    stage="chat_stream.liquid_state",
                )
                logger.warning("TOOL EXECUTION: LiquidState injection failed: %s", e)
            token_buffer = ""
            if hasattr(orch.cognitive_engine, "think_stream"):
                async for token in orch.cognitive_engine.think_stream(
                    message, context=context, tier=tier
                ):
                    token_buffer += token
                    yield token
            else:
                thought = await orch.cognitive_engine.think(
                    message, context=context, mode=ThinkingMode.DEEP
                )
                token_buffer = thought.content
                yield orch._filter_output(token_buffer)
            orch.conversation_history.append({"role": "user", "content": message})
            orch.conversation_history.append({"role": orch.AI_ROLE, "content": token_buffer})
            _trim_history(orch.conversation_history)
            if hasattr(orch, "drives"):
                await orch.drives.satisfy("social", 5.0)
        except _COORDINATOR_RECOVERABLE_ERRORS as e:
            _emit_message_fault(
                e,
                action="yielded bounded stream error marker after chat stream failure",
                severity="degraded",
                stage="chat_stream",
            )
            logger.error("Chat stream failed: %s", e)
            yield " [The stream hit a recoverable routing fault.] "
        finally:
            orch.status.is_processing = False

    async def sentence_stream_generator(self, message: str):
        """v5.2: Yields complete sentences as they are generated.
        Perfect for TTS pipe ingestion.
        """
        sentence_delimiters = (".", "?", "!", "\n", ":")
        buffer = ""
        async for token in self.chat_stream(message):
            buffer += token
            if any(token.endswith(d) for d in sentence_delimiters):
                if buffer.strip():
                    yield buffer.strip()
                    buffer = ""
        if buffer.strip():
            yield self.orch._filter_output(buffer.strip())
