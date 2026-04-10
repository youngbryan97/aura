"""Message Handling Mixin for RobustOrchestrator.
Extracts message acquisition, enqueueing, dispatch, and user input processing logic.
"""
import asyncio
import inspect
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageHandlingMixin:
    """Handles message queue operations, dispatch, and user input processing."""

    async def _acquire_next_message(self) -> Optional[str]:
        """Get next message from queue. Returns None if queue is empty."""
        try:
            item = self.message_queue.get_nowait()

            from core.utils.queues import unpack_priority_message

            message, _origin = unpack_priority_message(item)

            # v48 RECURSION GUARD: If the payload itself is a stringified tuple
            # (the "Russian Doll" bug), recursively unpack it to reach the raw content.
            recursion_depth = 0
            while isinstance(message, str) and message.startswith("(") and message.endswith(")") and recursion_depth < 5:
                try:
                    import ast

                    possible_tuple = ast.literal_eval(message)
                    if isinstance(possible_tuple, tuple):
                        message, nested_origin = unpack_priority_message(possible_tuple)
                        recursion_depth = recursion_depth + 1
                        logger.info("🪆 Russian Doll: Decoded nested tuple at depth %d", recursion_depth)
                        if nested_origin and isinstance(message, dict) and "origin" not in message:
                            message = {**message, "origin": nested_origin}
                    else:
                        break
                except Exception:
                    break

            logger.info("📦 Decoded message from queue: %s", str(message)[:60])

            # Pacing stim (optional boost during high activity)
            if hasattr(self, 'liquid_state') and self.liquid_state:
                if hasattr(self.liquid_state, 'update'):
                    self._fire_and_forget(
                        self.liquid_state.update(delta_curiosity=0.2),
                        name="liquid_pacing_acquire"
                    )

            return message

        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            logger.error("Error acquiring message from queue: %s", e)
            return None

    async def _defer_enqueue_message(
        self,
        message: Any,
        priority: int,
        origin: str,
        delay: float,
        *,
        _authority_checked: bool = False,
    ):
        await asyncio.sleep(delay)
        self.enqueue_message(
            message,
            priority=priority,
            origin=origin,
            _flow_checked=True,
            _authority_checked=_authority_checked,
        )

    def _background_enqueue_summary(self, message: Any, origin: str) -> str:
        payload = message
        tool_hint = ""
        if isinstance(payload, dict):
            context = dict(payload.get("context") or {})
            hint = dict(context.get("intent_hint") or {})
            tool = str(hint.get("tool") or "").strip()
            if tool:
                tool_hint = f"{tool}:"
            payload = (
                payload.get("content")
                or payload.get("message")
                or payload.get("objective")
                or payload.get("thought")
                or payload
            )
        return f"enqueue:{origin}:{tool_hint}{str(payload or '')[:160]}"

    def _authorize_background_enqueue_sync(self, message: Any, origin: str, priority: int) -> bool:
        summary = self._background_enqueue_summary(message, origin)
        current_state = getattr(getattr(self, "state_repo", None), "_current", None)
        try:
            from core.constitution import get_constitutional_core

            approved, reason = get_constitutional_core(self).approve_initiative_sync(
                summary,
                source=origin,
                urgency=max(0.05, min(1.0, float(priority) / 100.0)),
                state=current_state,
            )
        except Exception as exc:
            approved = False
            reason = f"background_enqueue_gate_failed:{type(exc).__name__}"

        if approved:
            return True

        try:
            from core.health.degraded_events import record_degraded_event

            event_reason = "background_enqueue_blocked"
            if any(
                marker in str(reason or "")
                for marker in ("gate_failed", "required", "unavailable")
            ):
                event_reason = "background_enqueue_gate_failed"
            record_degraded_event(
                "message_queue",
                event_reason,
                detail=summary[:160],
                severity="warning",
                classification="background_degraded",
                context={"origin": origin, "priority": priority, "reason": reason},
            )
        except Exception as exc:
            logger.debug("Background enqueue degraded-event logging failed: %s", exc)
        return False

    def enqueue_message(
        self,
        message: Any,
        priority: int = 20,
        origin: str = "background",
        _flow_checked: bool = False,
        _authority_checked: bool = False,
    ):
        """Standard interface for injecting messages into the core loop."""
        if not _flow_checked and hasattr(self, "_flow_controller") and self._flow_controller:
            decision = self._flow_controller.admit(self, origin=origin, priority=priority)
            if not decision.allow:
                logger.info(
                    "🧯 FlowControl: Dropped message from %s (%s).",
                    origin,
                    decision.reason,
                )
                return
            priority = decision.priority
            if decision.defer_seconds > 0:
                try:
                    asyncio.create_task(
                        self._defer_enqueue_message(
                            message,
                            priority=priority,
                            origin=origin,
                            delay=decision.defer_seconds,
                            _authority_checked=_authority_checked,
                        ),
                        name="DeferredEnqueue",
                    )
                    logger.debug(
                        "⏳ FlowControl: Deferred message from %s for %.2fs.",
                        origin,
                        decision.defer_seconds,
                    )
                    return
                except RuntimeError as _exc:
                    logger.debug("Suppressed RuntimeError: %s", _exc)

        logger.info("📩 Incoming Input [%s]: %s...", origin, str(message)[:100])
        if self._is_user_facing_origin(origin):
            # User-facing work should be dispatched once, not also enqueued for later re-processing.
            self._dispatch_message(message, origin=origin)
            self._last_user_interaction_time = time.time()
            return

        if not _authority_checked and not self._authorize_background_enqueue_sync(message, origin, priority):
            logger.info("🛡️ Background enqueue blocked for %s.", origin)
            return False

        try:
            # Zenith v47 Hardening: Deeply sanitize all messages before they enter
            # the queue to prevent circular references in AuraState.
            message = self._deep_circular_safe_sanitize(message)

            self._message_counter += 1
            # v61: Include origin in the payload to prevent double-processing in run() loop
            item = (priority, time.monotonic(), self._message_counter, message, origin)
            self.message_queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("⚠️ Message queue full. Dropped %s message from %s: %s",
                           "SYSTEM" if priority <= 20 else "AUTONOMOUS", origin,
                           str(message)[:120])
            return False

    def enqueue_from_thread(
        self,
        message: Any,
        origin: str = "user",
        priority: int = 10,
        _authority_checked: bool = False,
    ):
        """Safely enqueue a message from a synchronous thread to the async loop."""
        if hasattr(self, "_flow_controller") and self._flow_controller:
            decision = self._flow_controller.admit(self, origin=origin, priority=priority)
            if not decision.allow:
                logger.info(
                    "🧯 FlowControl: Dropped threaded message from %s (%s).",
                    origin,
                    decision.reason,
                )
                return
            priority = decision.priority

        if not self._is_user_facing_origin(origin) and not _authority_checked:
            if not self._authorize_background_enqueue_sync(message, origin, priority):
                logger.info("🛡️ Background threaded enqueue blocked for %s.", origin)
                return

        # Zenith v47 Hardening: Deeply sanitize all messages before they enter the queue
        message = self._deep_circular_safe_sanitize(message)

        if isinstance(message, str):
            message = {"content": message, "origin": origin}
        elif isinstance(message, dict):
            if "origin" not in message:
                message["origin"] = origin
        else:
            # Fallback for weird edge cases
            message = {"content": str(message), "origin": origin}

        self._message_counter += 1
        # Standard format for PriorityQueue: (priority, timestamp, counter, payload, origin)
        # v61: Added origin to tuple for loop-drain transparency
        item = (priority, time.monotonic(), self._message_counter, message, origin)

        # FIX: Never use self.loop. Resolve the running loop at the call site.
        # If we're already inside an async context, use call_soon_threadsafe.
        # If there is no running loop, we have a genuine boot-ordering bug —
        # log it loudly instead of silently dropping the message.
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self.message_queue.put_nowait, item)
        except RuntimeError:
            # No running loop in this thread — fall back to the orchestrator's
            # primary loop reference before probing loop-bound asyncio primitives.
            try:
                primary_loop = getattr(self, "loop", None)
                if primary_loop and primary_loop.is_running():
                    primary_loop.call_soon_threadsafe(self.message_queue.put_nowait, item)
                else:
                    stop_loop = getattr(self._stop_event, "_loop", None)
                    if stop_loop and stop_loop.is_running():
                        stop_loop.call_soon_threadsafe(self.message_queue.put_nowait, item)
                    else:
                        logger.error(
                            "enqueue_from_thread: No running event loop found. "
                            "Message dropped. This is a boot-ordering bug."
                        )
            except Exception as inner_err:
                logger.error("enqueue_from_thread: Loop resolution failed: %s", inner_err)

    def _deep_circular_safe_sanitize(self, obj: Any, memo: Optional[set] = None) -> Any:
        """Recursively sanitizes an object to be JSON-serializable and cycle-free.

        Zenith v47: Stays deep to prevent service instances (Orchestrator, Engine)
        from leaking into the AuraState, which causes asdict() failures.
        """
        if memo is None:
            memo = set()

        # Standard primitives
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        if id(obj) in memo:
            return f"<Cycle detected: {type(obj).__name__}>"

        memo.add(id(obj))

        try:
            # Handle Dataclasses (Manual expansion to avoid asdict() deepcopy crashes)
            from dataclasses import is_dataclass, fields
            if is_dataclass(obj):
                res = {}
                for f in fields(obj):
                    val = getattr(obj, f.name)
                    res[f.name] = self._deep_circular_safe_sanitize(val, memo)
                return res

            # Handle Pydantic
            if hasattr(obj, "model_dump"):
                try:
                    return self._deep_circular_safe_sanitize(obj.model_dump(), memo)
                except Exception:
                    return f"<Pydantic Serialization Failed: {type(obj).__name__}>"

            # Handle Dicts
            if isinstance(obj, dict):
                return {str(k): self._deep_circular_safe_sanitize(v, memo) for k, v in obj.items()}

            # Handle Lists/Tuples
            if isinstance(obj, (list, tuple)):
                return [self._deep_circular_safe_sanitize(i, memo) for i in obj]

            # Default: Stringify anything else (prevents deepcopy crashes in asdict)
            return str(obj)
        finally:
            memo.remove(id(obj))

    def _normalize_to_dict(self, obj: Any) -> Any:
        """Utility to convert Pydantic models and Dataclasses to dicts for legacy compat."""
        # Zenith v47 Hardening: Use deep_circular_safe_sanitize instead of standard asdict
        # to prevent system references from leaking into the serializable state.
        res = self._deep_circular_safe_sanitize(obj)

        if isinstance(res, dict):
            # Zenith-v6.3 Alias: deliberator returns 'action', orchestrator expects 'decision'
            if "action" in res and "decision" not in res:
                res["decision"] = res["action"]
        return res

    def _dispatch_message(self, message: str, origin: str = "user"):
        """Dispatch message to the async handler with bounded concurrency."""
        if not hasattr(self, "_dispatch_semaphore"):
            # v31.1: Reduce concurrency to 1 for Unitary Organism stability.
            # Allowing multiple concurrent ticks causes StateLock contention.
            self._dispatch_semaphore = asyncio.Semaphore(1)

        async def _bounded_handler():
            async with self._dispatch_semaphore:
                await self._handle_incoming_message(message, origin=origin)

        from core.utils.task_tracker import get_task_tracker

        def _bg_task_exception_handler(task: asyncio.Task) -> None:
            try:
                if not task.cancelled():
                    exc = task.exception()
                    if exc:
                        logging.getLogger("Aura.BgTasks").error(f"Background task failed: {repr(exc)}")
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError: %s', _e)
            except Exception as e:
                logging.getLogger("Aura.BgTasks").debug(f"Task exception handler itself failed: {e}")

        get_task_tracker().track_task(asyncio.create_task(_bounded_handler())).add_done_callback(_bg_task_exception_handler)
        self._emit_dispatch_telemetry(message)

    def _emit_dispatch_telemetry(self, message: Any):
        """Log dispatch event to thought stream."""
        try:
            from ...thought_stream import get_emitter
            safe_msg = str(message)[:1000] # Cap message size for telemetry
            get_emitter().emit("dispatch", f"Processing message: {safe_msg}")

            if safe_msg.startswith("Impulse:"):
                label = "Impulse ⚡"
            elif safe_msg.startswith("Thought:"):
                label = "Thought 💭"
            else:
                label = "User"
            get_emitter().emit(f"Input ({label})", safe_msg[:120], level="info", category="Perception")
        except Exception as exc:
            logger.error("Dispatch telemetry failure: %s", exc)

    async def process_user_input_priority(self, message: str, origin: str = "user", timeout_sec: float = 300.0) -> Optional[str]:
        """Bypasses the message queue for immediate priority processing and AWAITS the result."""
        # Use Semaphore, not Global Lock
        async with self._user_input_semaphore:
            try:
                # Use a specific timeout to prevent indefinite hangs
                async with asyncio.timeout(timeout_sec):
                    if self._is_user_facing_origin(origin):
                        self._last_user_interaction_time = time.time()
                    current_task = asyncio.current_task()
                    in_flight = getattr(self, "_current_thought_task", None)
                    if (
                        self._is_user_facing_origin(origin)
                        and in_flight is not None
                        and in_flight != current_task
                        and not in_flight.done()
                    ):
                        replaceable = (
                            getattr(self, "_current_task_is_autonomous", True)
                            or getattr(self, "_current_origin", "") == "voice"
                        )
                        if replaceable:
                            logger.info(
                                "🛑 Cancelling stale %s task for direct user input...",
                                getattr(self, "_current_origin", "background"),
                            )
                            in_flight.cancel()
                            try:
                                await in_flight
                            except asyncio.CancelledError:
                                logger.debug("Autonomous task cancelled successfully.")
                    return await self._process_user_input_core(message, origin)
            except asyncio.TimeoutError:
                logger.error("⌛ Priority processing TIMEOUT for: %s...", message[:50])
                return "I was deep in thought and took too long. Please try again."
            except Exception as e:
                logger.error("❌ Priority processing FAILED: %s", e)
                return f"A cognitive fault occurred: {str(e)}"

    async def _process_user_input_unlocked(
        self,
        message: str,
        origin: str = "user",
        timeout_sec: Optional[float] = 300.0,
    ) -> Optional[str]:
        """Public entry point — applies timeout, then delegates to core logic."""
        if timeout_sec is not None:
            try:
                async with asyncio.timeout(timeout_sec):
                    return await self._process_user_input_core(message, origin)
            except TimeoutError:
                logger.warning("⌛ Input processing timed out after %ss", timeout_sec)
                return None
        return await self._process_user_input_core(message, origin)

    async def _process_user_input_core(self, message: str, origin: str = "user") -> Optional[str]:
        """Actual processing logic — never calls itself, never recurses."""
        from ...container import ServiceContainer

        normalized_message = str(message or "").strip()
        if not normalized_message:
            if self._is_user_facing_origin(origin):
                logger.info("🫥 Ignoring empty foreground message from origin=%s.", origin)
            else:
                logger.debug("🫥 Dropping empty internal message from origin=%s.", origin)
            return None
        message = normalized_message

        # Deduplication Guard (FINGERPRINTING)
        msg_hash = self._get_fingerprint(f"{message}_{origin}")
        if msg_hash == self._last_emitted_fingerprint:
            logger.info("♻️ Deduplication: Same fingerprint. Skipping.")
            return None
        self._last_emitted_fingerprint = msg_hash

        # ZENITH BYPASS: ALL user-origin messages go through InferenceGate. NO EXCEPTIONS.
        # This completely decouples user requests from the Legacy Pipeline (CognitiveEngine →
        # HealthRouter → contended MLX worker path) which caused starvation.
        if self._is_user_facing_origin(origin):
            # INVARIANT: _inference_gate must never be None for user messages.
            # Boot.py sets it, but if something went catastrophically wrong, lazy-init here.
            if not self._inference_gate:
                await self._ensure_inference_gate_ready(context="user_message")

            # [HARDENING] Final None guard: if InferenceGate is STILL None after lazy-init,
            # return a safe fallback instead of crashing on NoneType.generate().
            if self._inference_gate is None:
                logger.error("🛑 InferenceGate is None after ensure_ready(). Cannot process user message.")
                return "I'm still waking up — inference engine isn't ready yet."

            current_task = asyncio.current_task()
            self._current_thought_task = current_task
            self._current_origin = origin
            self._current_task_is_autonomous = False
            self.status.is_processing = True
            self._last_user_interaction_time = time.time()
            self._extend_foreground_quiet_window(5.0)
            self._publish_telemetry({"event": "thinking", "origin": origin, "interim": True})
            self._publish_telemetry({"type": "status", "is_processing": True, "is_idle": False})

            # Notify ConversationalMomentumEngine of new user turn (fire-and-forget)
            try:
                cme = ServiceContainer.get("conversational_momentum_engine", default=None)
                if cme:
                    asyncio.ensure_future(cme.on_new_user_message(message))
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

            try:
                # ── CONSTITUTIONAL CLOSURE: Route through Kernel first ──
                # KernelInterface.process() runs the full phase pipeline (27 phases)
                # including affect, identity, bonding, consciousness, tools, repair.
                # InferenceGate is the fallback when kernel isn't ready.
                from core.kernel.kernel_interface import KernelInterface
                ki = KernelInterface.get_instance()
                if ki.is_ready():
                    logger.info("🧠 Constitutional: Routing user message through KernelInterface...")
                    response = await ki.process(message, origin=origin, priority=True)
                    if response:
                        # KI worked — skip InferenceGate entirely
                        # Record in history anyway
                        async with self._lock:
                            self._record_message_in_history(message, origin)
                            self._record_message_in_history(response, "assistant")
                        return response

                # ── FALLBACK: Direct InferenceGate (kernel not ready) ──
                # 1. State Capture (Limited Lock)
                async with self._lock:
                    history = self.conversation_history[-15:] if hasattr(self, 'conversation_history') else []
                    # Inject social brief/personality context if available
                    brief = "Normal turn."
                    try:
                        kernel = ServiceContainer.get("cognitive_kernel", default=None)
                        if kernel:
                            brief = await kernel.evaluate(message, history=history)
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)

                # 2. GENERATION (Unlocked Phase)
                # This allows the rest of the system (WebSockets, Bus) to continue pulsing.
                logger.info("🧠 Fallback: Calling InferenceGate directly (kernel not ready)...")
                _brief_text = brief.to_briefing_text() if hasattr(brief, 'to_briefing_text') else str(brief)
                response = await self._inference_gate.generate(
                    message,
                    context={
                        "history": history,
                        "brief": _brief_text,
                        "origin": origin,
                        "is_background": False,
                        "prefer_tier": "primary",
                    },
                )

                # DR-3: InferenceGate.generate() ALWAYS returns a string (error message at worst).
                # But even if somehow it returns None/empty, we catch it here. NO FALLTHROUGH.
                if not response:
                    logger.error("InferenceGate returned None/empty. Attempting emergency recovery.")
                    # STABILITY FIX: Instead of a generic error, try one more time with
                    # a forced cortex recovery, then give a real response
                    try:
                        gate = self._get_service("inference_gate")
                        if gate and hasattr(gate, "_respawn_cortex_if_needed"):
                            await gate._respawn_cortex_if_needed()
                            # Wait briefly for cortex to come up
                            await asyncio.sleep(3.0)
                            # Retry once
                            response = await gate.generate(
                                message,
                                context={"history": history, "origin": origin, "is_background": False},
                            )
                    except Exception as retry_err:
                        logger.debug("Emergency retry failed: %s", retry_err)

                    if not response:
                        # Last resort: acknowledge the user naturally
                        response = (
                            "I heard you, but my thinking engine is restarting right now. "
                            "Give me a moment and try again — I'll be back."
                        )

                # ── Silence Protocol ──────────────────────────────────────────
                # If the model chose to emit <|SILENCE|>, the InferenceGate
                # returns the sentinel string. We honour the choice: record the
                # fact internally but send nothing to the user.
                from core.brain.inference_gate import InferenceGate as _IG
                if response == _IG.SILENCE_SENTINEL:
                    logger.info("🤫 Silence Protocol honoured — suppressing output.")
                    try:
                        from core.thought_stream import get_emitter
                        get_emitter().emit(
                            "Silence",
                            "Chose not to respond — output suppressed.",
                            level="info",
                            category="SilenceProtocol",
                        )
                        from core.affect.heartstone_values import get_heartstone_values
                        get_heartstone_values().on_silence_chosen()
                        from core.affect.affective_circumplex import get_circumplex
                        get_circumplex().apply_event(+0.04, -0.08)  # calm, settled
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    # Record to history so she knows she stayed silent
                    async with self._lock:
                        self._record_message_in_history(message, origin)
                    return None   # Server receives None → sends no message

                # 3. State COMMIT (Limited Lock)
                async with self._lock:
                    self._record_message_in_history(message, origin)
                    self._record_message_in_history(response, "assistant")

                # ── Heartstone outcome signals ─────────────────────────────
                # Detect positive user tone to evolve Empathy/Curiosity weights
                try:
                    import re as _re
                    _positive_pat = _re.compile(
                        r'\b(thanks?|thank\s+you|great|perfect|awesome|love\s+it|'
                        r'nice|good\s+job|well\s+done|exactly|brilliant|yes[!.]*$)\b',
                        _re.IGNORECASE
                    )
                    if _positive_pat.search(message):
                        from core.affect.heartstone_values import get_heartstone_values as _ghsv
                        _ghsv().on_positive_interaction()
                        from core.affect.affective_circumplex import get_circumplex as _gc
                        _gc().apply_event(+0.06, +0.04)
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)

                # ── Epistemic Filter: run user messages through for belief retention ──
                # Long user messages may contain claims worth persisting
                try:
                    if len(message) > 80:
                        from core.world_model.epistemic_filter import get_epistemic_filter as _gef
                        _gef().ingest(message, source_type="conversation",
                                      source_label="user", emit_thoughts=False)
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)

                return response
            finally:
                self.status.is_processing = False
                self._extend_foreground_quiet_window(3.0)
                if getattr(self, "_current_thought_task", None) is current_task:
                    self._current_thought_task = None
                    self._current_origin = ""
                    self._current_task_is_autonomous = False
                # Fast-Path Telemetry (NO output_gate.emit — the REST handler already
                # returns this response to the client. Emitting here ALSO broadcasts
                # via EventBus → WebSocket → frontend, causing DUPLICATE messages.)
                self._publish_telemetry({"type": "status", "is_processing": False, "is_idle": True})

        # NON-USER origins (background, system, terminal_monitor, sensory_motor, etc.)
        # These still use the internal cognitive pipeline — they don't need InferenceGate.
        block_reason = self._background_message_block_reason(origin)
        if block_reason:
            logger.info(
                "🛡️ Deferring internal message [%s] while runtime is protected (%s).",
                origin,
                block_reason,
            )
            return None
        logger.info("📩 Processing internal message [%s]: %s...", origin, message[:80])
        await self._handle_incoming_message(message, origin=origin)
        return None

    async def _process_message(self, message: str, metadata: dict = None):
        """Primary cognitive entry point."""
        import uuid
        from core.tagged_reply_queue import reply_delivery_scope
        trace_id = str(uuid.uuid4())[:8]
        if os.environ.get("AURA_TRACE_MODE") == "1":
            logger.info("🧠 [TRACE] [%s] Orchestrator received: %s", trace_id, message[:50])

        # This method is used by the synchronous-style CLI loop in main.py
        # We need to wait for the response to appear in the reply_queue
        # or capture it from the handler.

        with reply_delivery_scope("user", trace_id) as session_id:
            response = await self._handle_incoming_message(message, origin="user")
            if isinstance(response, str):
                return {"ok": True, "response": response}

            try:
                reply_queue_type = type(self.reply_queue)
                origin_getter = getattr(reply_queue_type, "get_for_origin", None)
                empty_fn = getattr(self.reply_queue, "empty", None)
                queue_is_empty = False
                if callable(empty_fn):
                    try:
                        queue_is_empty = bool(empty_fn())
                    except Exception:
                        queue_is_empty = False
                queue_timeout = 2.0 if queue_is_empty else 125.0
                if inspect.iscoroutinefunction(origin_getter):
                    if queue_is_empty and not self.is_busy:
                        error_message = "No reply produced"
                        logger.warning(
                            "No reply queued after processing message: %s",
                            message[:50],
                        )
                        return {
                            "ok": False,
                            "error": error_message,
                            "response": {"error": error_message},
                        }
                    reply = await self.reply_queue.get_for_origin(
                        "user",
                        session_id=session_id,
                        timeout=queue_timeout,
                    )
                    if reply is None:
                        error_message = "No reply produced" if queue_is_empty else f"Thinking timeout ({int(queue_timeout)}s)"
                        reply = {
                            "ok": False,
                            "error": error_message,
                        }
                else:
                    reply_coro = self.reply_queue.get()
                    try:
                        reply = await asyncio.wait_for(reply_coro, timeout=queue_timeout)
                    finally:
                        if inspect.iscoroutine(reply_coro):
                            state = inspect.getcoroutinestate(reply_coro)
                            if state == inspect.CORO_CREATED:
                                reply_coro.close()
                return {"ok": True, "response": reply}
            except Exception as e:
                error_detail = str(e).strip()
                error_message = f"Response timeout: {error_detail}" if error_detail else "Response timeout"
                logger.error("Timed out waiting for reply to: %s", message[:50])
                return {
                    "ok": False,
                    "error": error_message,
                    "response": {"error": error_message},
                }
