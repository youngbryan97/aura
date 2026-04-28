"""Message Coordinator — message queue, dispatch, conversation history,
streaming, and impulse handling.

Extracted from orchestrator.py as part of the God Object decomposition.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import queue
import time
from typing import Any, Dict, List, Optional

from core.tagged_reply_queue import reply_delivery_scope
from core.utils.task_tracker import get_task_tracker, task_tracker
from core.utils.queues import unpack_priority_message

logger = logging.getLogger(__name__)


class MessageCoordinator:
    """Handles message intake, dispatch, history recording, and streaming."""

    def __init__(self, orch):
        self.orch = orch

    # ------------------------------------------------------------------
    # Queue Management
    # ------------------------------------------------------------------

    async def acquire_next_message(self) -> Optional[str]:
        """Get next message from queue. Returns None if queue is empty."""
        orch = self.orch
        try:
            raw = orch.message_queue.get_nowait()
            msg, _origin = unpack_priority_message(raw)
                
            logger.info("Processing queued message: %s", str(msg)[:100])
            if hasattr(orch, 'liquid_state') and orch.liquid_state:
                try:
                    get_task_tracker().create_task(
                        orch.liquid_state.update(delta_curiosity=0.2, delta_frustration=-0.1),
                        name="message_coordinator.liquid_state_update",
                    )
                except RuntimeError as _e:
                    logger.debug('Ignored RuntimeError in message_coordinator.py: %s', _e)
            orch._last_thought_time = time.time()
            return msg
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            record_degradation('message_coordinator', e)
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
            orch._dispatch_semaphore = asyncio.Semaphore(10)
        async def _bounded_handler():
            async with orch._dispatch_semaphore:
                await self.handle_incoming_message(message, origin=origin)
        task_tracker.create_task(
            _bounded_handler(),
            name="message_coordinator.dispatch",
        ).add_done_callback(_bg_task_exception_handler)
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
        except Exception as exc:
            record_degradation('message_coordinator', exc)
            logger.error("Dispatch telemetry failure: %s", exc)

    # ------------------------------------------------------------------
    # Message Processing
    # ------------------------------------------------------------------

    async def process_message(self, message: str) -> Dict[str, Any]:
        """Backward compatibility for main.py. Processes message and returns response."""
        orch = self.orch
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
                        reply = {"ok": False, "error": "Thinking timeout (30s)"}
                else:
                    reply = await asyncio.wait_for(orch.reply_queue.get(), timeout=30)
                return {"ok": True, "response": reply}
            except Exception as e:
                record_degradation('message_coordinator', e)
                logger.error("Timed out waiting for reply to: %s", message[:50])
                return {"ok": False, "error": f"Response timeout: {str(e)}"}

    async def process_user_input(self, message: str, origin: str = "user") -> Optional[str]:
        """Public API for injecting user/voice input.
        Returns the generated reply after processing.
        Bypasses the message queue for immediate priority processing.
        """
        orch = self.orch
        if origin in ("user", "voice") and orch._current_thought_task and not orch._current_thought_task.done():
            logger.info("🛑 Interruption: User input detected. Cancelling autonomous thought.")
            orch._current_thought_task.cancel()
            try:
                await orch._current_thought_task
            except (asyncio.CancelledError, Exception) as _exc:
                import logging
                logger.debug("Exception caught during execution", exc_info=True)
        try:
            logger.info("📩 DIRECT Processing user message: %s...", message[:80])
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
                    except asyncio.TimeoutError as _exc:
                        logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)

                    logger.warning("Timed out waiting for cognitive reply after 240s.")
                    if orch._current_thought_task and not orch._current_thought_task.done():
                        logger.info("Cognitive task is still RUNNING. Moving to background.")
                        return "I'm still processing that deep thought... check the Neural Feed for progress."
                    return "I'm sorry, my cognitive loop timed out. Please try again or check my status."
            return None
        except asyncio.QueueFull:
            logger.warning("Message queue full. Input dropped.")
            return "My processing queue is currently overloaded. One moment..."

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
        logger.info("📩 Processing message (%s): %s...", origin, message[:100])
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
                    final_response = await orch.state_machine.execute(intent, message, payload_context)
                    self.record_message_in_history(message, origin)
                    orch.conversation_history.append({"role": orch.AI_ROLE, "content": final_response})
                    if origin in ("user", "voice", "admin") and orch.reply_queue:
                        try:
                            orch.reply_queue.put_nowait(final_response)
                        except asyncio.QueueFull:
                            import logging
                            logger.debug("Exception caught during execution", exc_info=True)
                except Exception as e:
                    record_degradation('message_coordinator', e)
                    logger.error("State machine execution failed: %s", e)
                finally:
                    orch.status.is_processing = False
            orch._current_thought_task = task_tracker.create_task(
                _execute_and_reply(),
                name="message_coordinator.execute_and_reply",
            )
        except Exception as e:
            record_degradation('message_coordinator', e)
            logger.error("Error in handle_incoming_message: %s", e)
            orch.status.is_processing = False
        finally:
            orch.status.is_processing = False

    def record_message_in_history(self, message: str, origin: str):
        """Record the incoming message with appropriate role/prefix."""
        if origin == "autonomous_volition":
            prefix = "⚡ AUTONOMOUS GOAL: "
            role = "internal"
        elif origin == "impulse":
            prefix = "⚡ IMPULSE (speak to user): "
            role = "internal"
        else:
            prefix = ""
            role = "user"
        self.orch.conversation_history.append({"role": role, "content": f"{prefix}{message}"})

    # ------------------------------------------------------------------
    # Impulses
    # ------------------------------------------------------------------

    async def handle_impulse(self, impulse: str):
        """Handle an autonomous impulse from the Consciousness Core."""
        logger.info("⚡ Processing Impulse: %s", impulse)
        directives = {
            "explore_knowledge": "I'm curious about something in my knowledge base. I should explore it.",
            "seek_novelty": "I'm feeling a bit idle. I think I'll look for something new to learn or do.",
            "deep_reflection": "I'm going to take a moment for deep reflection on my recent experiences."
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
            except Exception as exc:
                record_degradation('message_coordinator', exc)
                logger.debug("Suppressed: %s", exc)
            context = orch._get_cleaned_history_context(8)
            try:
                from core.container import get_container
                container = get_container()
                ls = container.get('liquid_state')
                context['liquid_state'] = ls.get_status()
                logger.debug("TOOL EXECUTION: Injected liquid_state: %s", context['liquid_state'])
            except Exception as e:
                record_degradation('message_coordinator', e)
                logger.warning("TOOL EXECUTION: LiquidState injection failed: %s", e)
            token_buffer = ""
            if hasattr(orch.cognitive_engine, "think_stream"):
                async for token in orch.cognitive_engine.think_stream(message, context=context, tier=tier):
                    token_buffer += token
                    yield token
            else:
                thought = await orch.cognitive_engine.think(message, context=context, mode=ThinkingMode.DEEP)
                token_buffer = thought.content
                yield orch._filter_output(token_buffer)
            orch.conversation_history.append({"role": "user", "content": message})
            orch.conversation_history.append({"role": orch.AI_ROLE, "content": token_buffer})
            if hasattr(orch, 'drives'): await orch.drives.satisfy("social", 5.0)
        except Exception as e:
            record_degradation('message_coordinator', e)
            logger.error("Chat stream failed: %s", e)
            yield f" [Error: {e}] "
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
