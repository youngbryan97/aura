"""Message Handling Mixin for RobustOrchestrator.
Extracts message acquisition, enqueueing, dispatch, and user input processing logic.
"""

import asyncio
import collections
import hashlib
import inspect
import logging
import os
import sys
import time
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

_MESSAGE_HANDLING_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    asyncio.QueueEmpty,
    asyncio.QueueFull,
    Exception,
)


def _record_message_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "message_handling",
        error,
        severity=severity,
        action=action,
    )


async def _resolve_generation_result(result: Any) -> Any:
    """Resolve sync, async, or nested-awaitable generation adapter results."""
    depth = 0
    while inspect.isawaitable(result):
        result = await result
        depth += 1
        if depth >= 5:
            raise RuntimeError("generation adapter returned too many nested awaitables")
    return result


# ── Response Repetition Detection ────────────────────────────────────────
# General-purpose mechanism that detects when Aura is stuck in a cognitive
# loop producing near-identical responses. When detected, injects a
# metacognitive warning into conversation history so the next inference
# cycle is aware of the loop and can break out of it.
_REPETITION_RING_SIZE = 5
_REPETITION_SIMILARITY_THRESHOLD = 0.8


def _response_fingerprint(text: str) -> str:
    """Normalize and hash a response for similarity comparison."""
    normalized = " ".join(str(text or "").lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def _responses_are_similar(a: str, b: str) -> bool:
    """Check if two responses are near-identical (>80% token overlap)."""
    if not a or not b:
        return False
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return False
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union)
    return jaccard >= _REPETITION_SIMILARITY_THRESHOLD


class MessageHandlingMixin:
    """Handles message queue operations, dispatch, and user input processing."""

    # Ring buffer of recent response fingerprints for repetition detection
    _recent_response_ring: collections.deque  # initialized in __init__ or lazily

    async def _acquire_next_message(self) -> str | None:
        """Get next message from queue. Returns None if queue is empty."""
        try:
            item = self.message_queue.get_nowait()

            from core.utils.queues import unpack_priority_message

            message, _origin = unpack_priority_message(item)

            # Legacy tuple formatting is removed in Zenith. We expect strictly typed IPCMessage.
            # Any remaining tuple formats are caught by unpack_priority_message.

            logger.info("📦 Decoded message from queue: %s", str(message)[:60])

            # Pacing stim (optional boost during high activity)
            if hasattr(self, "liquid_state") and self.liquid_state:
                if hasattr(self.liquid_state, "update"):
                    self._fire_and_forget(
                        self.liquid_state.update(delta_curiosity=0.2), name="liquid_pacing_acquire"
                    )

            return message

        except asyncio.QueueEmpty:
            return None
        except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as e:
            _record_message_degradation(
                e,
                action="returned no queue message after queue decode failure",
                severity="error",
            )
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

    def _background_enqueue_defer_reason(self, origin: str, priority: int) -> str:
        if priority < 20 or self._is_user_facing_origin(origin):
            return ""
        try:
            from core.runtime.background_policy import background_activity_reason

            return background_activity_reason(
                self,
                min_idle_seconds=180.0,
                max_memory_percent=78.0,
                max_failure_pressure=0.25,
                require_conversation_ready=False,
            )
        except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as exc:
            _record_message_degradation(
                exc,
                action="allowed background enqueue policy to fall through to authority gate",
            )
            logger.debug("Background enqueue policy probe failed: %s", exc)
            return ""

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
        except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as exc:
            _record_message_degradation(
                exc,
                action="blocked background enqueue because authority gate failed closed",
                severity="error",
            )
            approved = False
            reason = f"background_enqueue_gate_failed:{type(exc).__name__}"

        if approved:
            return True

        # Trusted internal volition sources should not be silently dropped —
        # that produced the "sensory_motor triggered volition → blocked →
        # nothing happens" pattern Bryan observed. Admit them at reduced
        # priority when approval failed for infrastructure reasons (executive
        # core / authority gateway unavailable / low_priority_initiative),
        # NOT for safety vetoes (somatic veto, constitutional principle).
        trusted_internal = {
            "sensory_motor",
            "drive_engine",
            "volition_engine",
            "emergent_goal_engine",
            "agency_core",
            "agency_facade",
            "internal_volition",
        }
        hard_safety_markers = (
            "somatic_veto",
            "constitutional_violation",
            "identity_violation",
            "safety_veto",
            "forbidden_action",
        )
        reason_str = str(reason or "")
        safety_vetoed = any(m in reason_str for m in hard_safety_markers)
        if (origin or "").strip().lower() in trusted_internal and not safety_vetoed:
            logger.info(
                "🛡️ Internal volition admitted at reduced priority: %s (gate reason: %s)",
                origin,
                reason_str or "unspecified",
            )
            return True

        try:
            from core.health.degraded_events import record_degraded_event

            event_reason = "background_enqueue_blocked"
            if any(
                marker in str(reason or "") for marker in ("gate_failed", "required", "unavailable")
            ):
                event_reason = "background_enqueue_gate_failed"
            record_degraded_event(
                "message_queue",
                event_reason,
                detail=summary[:160],
                severity="info",  # Reduced from warning — background enqueue blocks are normal during idle
                classification="background_degraded",
                context={"origin": origin, "priority": priority, "reason": reason},
            )
        except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as exc:
            _record_message_degradation(
                exc,
                action="kept background enqueue blocked after degraded-event logging failed",
            )
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
                    from core.utils.task_tracker import get_task_tracker

                    get_task_tracker().create_task(
                        self._defer_enqueue_message(
                            message,
                            priority=priority,
                            origin=origin,
                            delay=decision.defer_seconds,
                            _authority_checked=_authority_checked,
                        ),
                        name="message_handling.deferred_enqueue",
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

        if not _authority_checked and not self._authorize_background_enqueue_sync(
            message, origin, priority
        ):
            logger.info("🛡️ Background enqueue blocked for %s.", origin)
            return False

        defer_reason = (
            ""
            if (_flow_checked and _authority_checked)
            else self._background_enqueue_defer_reason(origin, priority)
        )
        if defer_reason:
            logger.debug(
                "🛡️ Background enqueue deferred for %s: %s",
                origin,
                defer_reason,
            )
            return False

        # Zenith v47 Hardening: Deeply sanitize all messages before they enter
        # the queue to prevent circular references in AuraState.
        message = self._deep_circular_safe_sanitize(message)

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is None:
            self.enqueue_from_thread(
                message, origin=origin, priority=priority, _authority_checked=_authority_checked
            )
            return

        try:
            # Zenith v47 Hardening: Deeply sanitize all messages before they enter
            # the queue to prevent circular references in AuraState.
            message = self._deep_circular_safe_sanitize(message)

            self._message_counter += 1
            from core.schemas import IPCMessage

            item = IPCMessage(
                priority=priority,
                timestamp=time.monotonic(),
                sequence=self._message_counter,
                payload=message,
                origin=origin,
            )
            self.message_queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning(
                "⚠️ Message queue full. Dropped %s message from %s: %s",
                "SYSTEM" if priority <= 20 else "AUTONOMOUS",
                origin,
                str(message)[:120],
            )
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
            defer_reason = self._background_enqueue_defer_reason(origin, priority)
            if defer_reason:
                logger.debug(
                    "🛡️ Background threaded enqueue deferred for %s: %s",
                    origin,
                    defer_reason,
                )
                return
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
        from core.schemas import IPCMessage

        item = IPCMessage(
            priority=priority,
            timestamp=time.monotonic(),
            sequence=self._message_counter,
            payload=message,
            origin=origin,
        )

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
            except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as inner_err:
                _record_message_degradation(
                    inner_err,
                    action="dropped threaded enqueue after all loop targets failed",
                    severity="error",
                )
                logger.error("enqueue_from_thread: Loop resolution failed: %s", inner_err)

    def _deep_circular_safe_sanitize(self, obj: Any, memo: set | None = None) -> Any:
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
            from dataclasses import fields, is_dataclass

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
                except _MESSAGE_HANDLING_RECOVERABLE_ERRORS:
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
                        logging.getLogger("Aura.BgTasks").error(
                            f"Background task failed: {repr(exc)}"
                        )
            except asyncio.CancelledError as _e:
                logger.debug("Ignored asyncio.CancelledError: %s", _e)
            except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as e:
                _record_message_degradation(
                    e,
                    action="kept dispatch task result isolated after callback failure",
                )
                logging.getLogger("Aura.BgTasks").debug(
                    f"Task exception handler itself failed: {e}"
                )

        get_task_tracker().create_task(
            _bounded_handler(),
            name="message_handling.bounded_dispatch",
        ).add_done_callback(_bg_task_exception_handler)
        self._emit_dispatch_telemetry(message)

    def _emit_dispatch_telemetry(self, message: Any):
        """Log dispatch event to thought stream."""
        try:
            from ...thought_stream import get_emitter

            safe_msg = str(message)[:1000]  # Cap message size for telemetry
            get_emitter().emit("dispatch", f"Processing message: {safe_msg}")

            if safe_msg.startswith("Impulse:"):
                label = "Impulse ⚡"
            elif safe_msg.startswith("Thought:"):
                label = "Thought 💭"
            else:
                label = "User"
            get_emitter().emit(
                f"Input ({label})", safe_msg[:120], level="info", category="Perception"
            )
        except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as exc:
            _record_message_degradation(
                exc,
                action="continued dispatch after telemetry emission failed",
            )
            logger.error("Dispatch telemetry failure: %s", exc)

    async def process_user_input_priority(
        self, message: str, origin: str = "user", timeout_sec: float = 300.0
    ) -> str | None:
        """Bypasses the message queue for immediate priority processing and AWAITS the result."""

        # [REFLEX BYPASS] Somatic motor reflexes MUST NEVER block on the user input semaphore.
        # This prevents deadlocks when a heavy cognitive task is stalled.
        print(
            f"DEBUG: Priority input origin={origin}, contract={('[EMBODIED CONTROL CONTRACT]' in message)}"
        )
        sys.stdout.flush()
        if origin == "embodied_motor_reflex" or "[EMBODIED CONTROL CONTRACT]" in message:
            # Check for deterministic somatic reflexes first
            if "[EMBODIED CONTROL CONTRACT]" in message:
                res = await self._check_embodied_reflexes(message)
                if res:
                    return res

            return await self._process_message_pipeline(message, origin=origin)

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
            except TimeoutError:
                logger.error("⌛ Priority processing TIMEOUT for: %s...", message[:50])
                # [STABILITY v55] Return empty so chat.py can retry/escalate.
                return ""
            except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as e:
                _record_message_degradation(
                    e,
                    action="returned empty priority response so caller can retry or escalate",
                    severity="error",
                )
                logger.error("❌ Priority processing FAILED: %s", e)
                # [STABILITY v55] Return empty so chat.py can retry/escalate.
                return ""

    async def _process_user_input_unlocked(
        self,
        message: str,
        origin: str = "user",
        timeout_sec: float | None = 300.0,
    ) -> str | None:
        """Public entry point — applies timeout, then delegates to core logic."""
        if timeout_sec is not None:
            try:
                async with asyncio.timeout(timeout_sec):
                    return await self._process_user_input_core(message, origin)
            except TimeoutError:
                logger.warning("⌛ Input processing timed out after %ss", timeout_sec)
                return None
        return await self._process_user_input_core(message, origin)

    async def _process_user_input_core(self, message: str, origin: str = "user") -> str | None:
        """Actual processing logic — never calls itself, never recurses."""
        print(f"\n--- ORCHESTRATOR INPUT: origin={origin}, len={len(message or '')} ---")
        sys.stdout.flush()
        from ...container import ServiceContainer

        normalized_message = str(message or "").strip()
        if not normalized_message:
            if self._is_user_facing_origin(origin):
                logger.info("🫥 Ignoring empty foreground message from origin=%s.", origin)
            else:
                logger.debug("🫥 Dropping empty internal message from origin=%s.", origin)
            return None
        message = normalized_message

        # Somatic Reflex Bypass (Zero-Latency)
        # Bypasses the Unified Will and InferenceGate for deterministic UI prompts
        # when an active embodiment contract is detected.
        # [ARCHITECTURE] Moved ABOVE deduplication to ensure reflexes fire even if
        # the screen hasn't advanced since the last sensor tick.
        has_contract = "[EMBODIED CONTROL CONTRACT]" in message
        logger.debug(
            "Input core: has_contract=%s, origin=%s, message_len=%s",
            has_contract,
            origin,
            len(message),
        )

        if has_contract:
            somatic_response = await self._check_embodied_reflexes(message)
            if somatic_response:
                return somatic_response

        # Deduplication Guard (FINGERPRINTING)
        msg_hash = self._get_fingerprint(f"{message}_{origin}")
        if msg_hash == self._last_emitted_fingerprint:
            logger.info("♻️ Deduplication: Same fingerprint. Skipping.")
            return None
        self._last_emitted_fingerprint = msg_hash

        # ── UNIFIED WILL GATE ──────────────────────────────────────────
        # ALL processing — user-facing or internal — must pass through the
        # Unified Will. This is THE architectural invariant that makes
        # Aura a unified intelligence rather than a federation.
        try:
            from core.will import ActionDomain, get_will

            will = get_will()
            if will._started:
                domain = (
                    ActionDomain.RESPONSE
                    if self._is_user_facing_origin(origin)
                    else ActionDomain.REFLECTION
                )
                will_decision = will.decide(
                    content=message[:200],
                    source=f"message_handler:{origin}",
                    domain=domain,
                    priority=0.8 if self._is_user_facing_origin(origin) else 0.3,
                    context={"origin": origin, "message_length": len(message)},
                )
                if not will_decision.is_approved():
                    logger.info(
                        "🛡️ Unified Will %s message from %s: %s",
                        will_decision.outcome.value,
                        origin,
                        will_decision.reason,
                    )
                    if self._is_user_facing_origin(origin):
                        # User messages are always processed but constraints are applied
                        if will_decision.constraints:
                            logger.info("Will constraints applied: %s", will_decision.constraints)
                    else:
                        return None  # Internal messages can be refused
        except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _will_err:
            _record_message_degradation(
                _will_err,
                action="continued user-input processing with degraded Will gate",
                severity="error",
            )
            logger.warning("Unified Will gate failed (degraded): %s", _will_err, exc_info=True)

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
                logger.error(
                    "🛑 InferenceGate is None after ensure_ready(). Cannot process user message."
                )
                # [STABILITY v55] Return empty instead of robot message
                return ""

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
                    from core.utils.task_tracker import get_task_tracker

                    get_task_tracker().track(
                        cme.on_new_user_message(message), name="on_new_user_message"
                    )
            except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _exc:
                _record_message_degradation(
                    _exc,
                    action="continued user turn after conversational momentum notification failed",
                )
                logger.debug("Conversational momentum notification skipped: %s", _exc)

            try:
                # ── CONSTITUTIONAL CLOSURE: Route through Kernel first ──
                # KernelInterface.process() runs the full phase pipeline (27 phases)
                # including affect, identity, bonding, consciousness, tools, repair.
                # InferenceGate is the fallback when kernel isn't ready.
                from core.kernel.kernel_interface import KernelInterface

                ki = KernelInterface.get_instance()
                if ki.is_ready():
                    logger.info(
                        "🧠 Constitutional: Routing user message through KernelInterface..."
                    )
                    response = await ki.process(message, origin=origin, priority=True)
                    if response:
                        # KI worked — skip InferenceGate entirely
                        # Record in history anyway
                        async with self._lock:
                            self._record_message_in_history(message, origin)
                            self._record_message_in_history(response, "assistant")

                        # JARVIS activity telemetry
                        jarvis = ServiceContainer.get("jarvis", default=None)
                        if jarvis:
                            jarvis.record_activity(user_input=message, response=response or "")
                            self._fire_and_forget(jarvis.run_cycle(), name="orchestrator.jarvis.run_cycle")

                        return response

                # ── FALLBACK: Direct InferenceGate (kernel not ready) ──
                # 1. State Capture (Limited Lock)
                async with self._lock:
                    history = (
                        self.conversation_history[-15:]
                        if hasattr(self, "conversation_history")
                        else []
                    )
                    # Inject social brief/personality context if available
                    brief = "Normal turn."
                    try:
                        kernel = ServiceContainer.get("cognitive_kernel", default=None)
                        if kernel:
                            brief = await kernel.evaluate(message, history=history)
                    except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _exc:
                        _record_message_degradation(
                            _exc,
                            action="continued direct InferenceGate fallback with default cognitive brief",
                        )
                        logger.debug("Cognitive kernel brief skipped: %s", _exc)

                # 2. GENERATION (Unlocked Phase)
                # This allows the rest of the system (WebSockets, Bus) to continue pulsing.
                logger.info("🧠 Fallback: Calling InferenceGate directly (kernel not ready)...")
                _brief_text = (
                    brief.to_briefing_text() if hasattr(brief, "to_briefing_text") else str(brief)
                )
                response = await _resolve_generation_result(
                    self._inference_gate.generate(
                        message,
                        context={
                            "history": history,
                            "brief": _brief_text,
                            "origin": origin,
                            "is_background": False,
                            "prefer_tier": "primary",
                            "protected_foreground_lane": True,
                        },
                    )
                )

                # DR-3: InferenceGate.generate() ALWAYS returns a string (error message at worst).
                # But even if somehow it returns None/empty, we catch it here. NO FALLTHROUGH.
                if not response:
                    logger.error(
                        "InferenceGate returned None/empty. Attempting emergency recovery."
                    )
                    # STABILITY FIX: Instead of a generic error, try one more time with
                    # a forced cortex recovery, then give a real response
                    try:
                        gate = self._get_service("inference_gate")
                        if gate and hasattr(gate, "_respawn_cortex_if_needed"):
                            await gate._respawn_cortex_if_needed()
                            # Wait briefly for cortex to come up
                            await asyncio.sleep(3.0)
                            response = await _resolve_generation_result(
                                gate.generate(
                                    message,
                                    context={
                                        "history": history,
                                        "origin": origin,
                                        "is_background": False,
                                        "protected_foreground_lane": True,
                                    },
                                )
                            )
                    except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as retry_err:
                        _record_message_degradation(
                            retry_err,
                            action="returned empty foreground response after emergency retry failed",
                            severity="error",
                        )
                        logger.debug("Emergency retry failed: %s", retry_err)

                    if not response:
                        # Allow the empty string to propagate back to the caller
                        # so that it can trigger cloud fallback instead of masking
                        # the failure with a robotic reflex.
                        response = ""

                # ── Auto-Continuation Reflex ──────────────────────────────────
                # If the response ends abruptly without punctuation, it likely
                # hit a max_tokens cap. Automatically prompt for continuation
                # and concatenate to form a seamless thought.
                continuation_count = 0
                while continuation_count < 3 and response and len(response) > 200:
                    last_char = response.strip()[-1] if response.strip() else ""
                    if last_char not in ".!?\"'”’*)\\]}>~`\\n":
                        logger.info(
                            "⚡ Auto-Continuation triggered! Response appears truncated (ended with '%s').",
                            last_char,
                        )
                        continuation_count += 1

                        temp_history = list(history)
                        temp_history.append({"role": "user", "content": message})
                        temp_history.append({"role": "assistant", "content": response})

                        continue_msg = "[SYSTEM: You hit your output token limit. Continue your exact previous thought seamlessly from where you left off.]"
                        next_part = await _resolve_generation_result(
                            self._inference_gate.generate(
                                continue_msg,
                                context={
                                    "history": temp_history,
                                    "brief": _brief_text,
                                    "origin": origin,
                                    "is_background": False,
                                    "prefer_tier": "primary",
                                    "protected_foreground_lane": True,
                                },
                            )
                        )
                        if next_part and next_part != "<|SILENCE|>":
                            # Add a space if it doesn't start with punctuation or space
                            if next_part[0].isalnum() and response[-1].isalnum():
                                response += ""  # The model usually continues the exact word if cut off mid-word, or we could add a space. Usually no space is safer.
                            response += next_part
                        else:
                            break
                    else:
                        break

                # ── Silence Protocol ──────────────────────────────────────────
                # If the model chose to emit <|SILENCE|>, the InferenceGate
                # returns the sentinel string. We honour the choice: record the
                # fact internally but send nothing to the user.
                from core.brain.inference_gate import InferenceGate

                if response == InferenceGate.SILENCE_SENTINEL:
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
                    except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _exc:
                        _record_message_degradation(
                            _exc,
                            action="preserved silence response after silence telemetry failed",
                        )
                        logger.debug("Silence protocol side-effect skipped: %s", _exc)
                    # Record to history so she knows she stayed silent
                    async with self._lock:
                        self._record_message_in_history(message, origin)
                    return None  # Server receives None → sends no message

                # 3. State COMMIT (Limited Lock)
                async with self._lock:
                    if response:
                        self._record_message_in_history(message, origin)
                        self._record_message_in_history(response, "assistant")

                # ── Response Repetition Detection ─────────────────────────
                # General-purpose: if she's producing near-identical
                # responses, inject a metacognitive interrupt so her
                # next reasoning cycle knows to try something different.
                try:
                    ring = getattr(self, "_recent_response_ring", None)
                    if ring is None:
                        ring = collections.deque(maxlen=_REPETITION_RING_SIZE)
                        self._recent_response_ring = ring
                    fp = _response_fingerprint(response)
                    consecutive_dupes = sum(1 for prev_fp in ring if prev_fp == fp)
                    ring.append(fp)
                    if consecutive_dupes >= 2:
                        logger.warning(
                            "🔄 Response Repetition Detected: %d consecutive near-identical responses.",
                            consecutive_dupes + 1,
                        )
                        metacognitive_warning = (
                            "[METACOGNITIVE INTERRUPT] You have produced the same response "
                            f"{consecutive_dupes + 1} times in a row. Your current strategy is "
                            "NOT working — the environment is not changing in response to your "
                            "actions. You MUST try a completely different approach. Do not repeat "
                            "your previous plan. Analyze what went wrong and adapt."
                        )
                        async with self._lock:
                            self._record_message_in_history(metacognitive_warning, "system")
                except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _rep_exc:
                    _record_message_degradation(
                        _rep_exc,
                        action="continued response delivery without repetition-ring update",
                    )
                    logger.debug("Repetition detection error: %s", _rep_exc)

                # ── Heartstone outcome signals ─────────────────────────────
                # Detect positive user tone to evolve Empathy/Curiosity weights
                try:
                    import re as _re

                    _positive_pat = _re.compile(
                        r"\b(thanks?|thank\s+you|great|perfect|awesome|love\s+it|"
                        r"nice|good\s+job|well\s+done|exactly|brilliant|yes[!.]*$)\b",
                        _re.IGNORECASE,
                    )
                    if _positive_pat.search(message):
                        from core.affect.heartstone_values import get_heartstone_values as _ghsv

                        _ghsv().on_positive_interaction()
                        from core.affect.affective_circumplex import get_circumplex as _gc

                        _gc().apply_event(+0.06, +0.04)
                except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _exc:
                    _record_message_degradation(
                        _exc,
                        action="continued response delivery without positive-interaction affect update",
                    )
                    logger.debug("Positive interaction affect update skipped: %s", _exc)

                # ── Epistemic Filter: run user messages through for belief retention ──
                # Long user messages may contain claims worth persisting
                try:
                    if len(message) > 80:
                        from core.world_model.epistemic_filter import get_epistemic_filter as _gef

                        _gef().ingest(
                            message,
                            source_type="conversation",
                            source_label="user",
                            emit_thoughts=False,
                        )
                except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as _exc:
                    _record_message_degradation(
                        _exc,
                        action="continued response delivery without epistemic filter ingest",
                    )
                    logger.debug("Epistemic filter ingest skipped: %s", _exc)

                # JARVIS activity telemetry
                jarvis = ServiceContainer.get("jarvis", default=None)
                if jarvis:
                    jarvis.record_activity(user_input=message, response=response or "")
                    self._fire_and_forget(jarvis.run_cycle(), name="orchestrator.jarvis.run_cycle")

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
                    except _MESSAGE_HANDLING_RECOVERABLE_ERRORS:
                        queue_is_empty = False
                queue_timeout = 240.0
                if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AURA_TEST_MODE") == "1":
                    queue_timeout = float(os.environ.get("AURA_TEST_REPLY_QUEUE_TIMEOUT_S", "0.5"))
                if inspect.iscoroutinefunction(origin_getter):
                    reply = await self.reply_queue.get_for_origin(
                        "user",
                        session_id=session_id,
                        timeout=queue_timeout,
                    )
                    if reply is None:
                        error_message = (
                            "No reply produced"
                            if queue_is_empty
                            else f"Thinking timeout ({int(queue_timeout)}s)"
                        )
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
            except _MESSAGE_HANDLING_RECOVERABLE_ERRORS as e:
                _record_message_degradation(
                    e,
                    action="returned structured response-timeout error to caller",
                    severity="error",
                )
                error_detail = str(e).strip()
                error_message = (
                    f"Response timeout: {error_detail}" if error_detail else "Response timeout"
                )
                logger.error("Timed out waiting for reply to: %s", message[:50])
                return {
                    "ok": False,
                    "error": error_message,
                    "response": {"error": error_message},
                }

    async def _run_cognitive_loop(self, message: str, origin: str = "user") -> str | None:
        """Legacy compatibility shim returning assistant text for one turn."""
        if self._is_user_facing_origin(origin):
            result = await self._process_message(message, metadata={"origin": origin})
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                response = result.get("response")
                if isinstance(response, str):
                    return response
                if isinstance(response, dict):
                    for key in ("content", "response", "message", "error"):
                        value = response.get(key)
                        if isinstance(value, str) and value:
                            return value
                error = result.get("error")
                if isinstance(error, str) and error:
                    return error
            return None

        history_len = (
            len(self.conversation_history)
            if isinstance(getattr(self, "conversation_history", None), list)
            else 0
        )
        await self._handle_incoming_message(message, origin=origin)
        if not isinstance(getattr(self, "conversation_history", None), list):
            return None
        for item in reversed(self.conversation_history[history_len:]):
            if item.get("role") in ("assistant", "aura", "model"):
                content = item.get("content")
                if isinstance(content, str) and content:
                    return content
        return None
