"""Autonomous Output Gate — Communication Triage for Aura.

Standardizes which messages reach the User (Primary) vs. Background (Secondary).
Prevents "Autonomous Pollution" where background search results flood the chat.
"""
import asyncio
import logging
import time
import weakref
from collections.abc import AsyncIterator
from typing import Any, cast

from core.runtime.effect_boundary import effect_sink
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.OutputGate")


class AutonomousOutputGate:
    """Triage engine for Aura's communicative outputs."""
    
    def __init__(self, orchestrator: Any = None) -> None:
        self.orchestrator = orchestrator
        # Secondary sink for background/autonomous logs
        self.secondary_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        
        # Identity Guard (Bridge 3)
        try:
            from core.identity.identity_guard import PersonaEnforcementGate
            self.identity_guard = PersonaEnforcementGate()
        except ImportError:
            self.identity_guard = None
        
        # v30 Hardening: Forbidden patterns — anything that looks like internal/computational output
        self._blocked_patterns = [
            r"\[INTERNAL\]",
            r"DEBUG:",
            r"<thought_trace>",
            r"as an AI language model",
            r"Novel Stimulation",
            r"Internal Simulation",
            r"In the quiet expanse of my thoughts",
            r"Imagine you are standing at the edge",
            r"Here's what we can do:",
            r"Let's dive into a novel internal simulation",
            r"Scenario: The Case of",
            r"Context: You are sitting in your digital",
            r"^Scenario:",
            r"^Context:",
            r"Internal Monologue:",
            r"^Execute Goal:",
            r"(?m)^\s*volition_trigger\b.*$",
            r"(?m)^\s*volition_error\b.*$",
            r"Still with me\? Sometimes quiet",
            r"Would you like to dive into",
            # v48: Block internal cognitive state leaking into chat
            r"Cognitive baseline tick\s*\d+",
            r"monitoring internal state",
            r"baseline_continuity",
            r"Winner:\s*\w+\s*\|\s*Content:",
            r"In the \d[\d.]*\s*minutes just passed:",
            r"Pending initiatives:",
            r"Reconcile continuity gap",
        ]

    def is_ready(self) -> bool:
        """Synchronous liveness probe for the runtime health contract."""
        return (
            isinstance(self.secondary_queue, asyncio.Queue)
            and isinstance(self._blocked_patterns, list)
            and callable(getattr(self, "emit", None))
            and callable(getattr(self, "_foreground_policy", None))
            and callable(getattr(self, "_sanitize_autonomous_output", None))
        )

    def _sanitize_autonomous_output(self, text: str) -> str:
        """Unified scrubber for all outgoing text."""
        import re
        if self.identity_guard:
            text = self.identity_guard.sanitize(text)
        # Strip computational thinking artifacts that leak past other scrubbers
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        text = re.sub(r'^(?:Step|Phase)\s*\d+[:.]\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        return text.strip()

    def _is_output_blocked(self, text: str) -> bool:
        """Check for forbidden patterns or system-leakage."""
        import re
        for pattern in self._blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _foreground_policy(self, content: str, origin: str, target: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Route internal autonomy chatter away from the user-facing channel."""
        import re

        trusted_primary_origins = {"user", "voice", "admin", "api"}
        explicit_user_visible = bool(metadata.get("user_visible") or metadata.get("foreground_receipt"))
        executive_authority = bool(metadata.get("executive_authority"))
        if origin in trusted_primary_origins or (explicit_user_visible and executive_authority):
            return target, metadata

        background_origin_terms = (
            "system",
            "cognitive",
            "autonomous",
            "motivation",
            "intention",
            "capability",
            "neural",
            "health",
            "telemetry",
            "mycelium",
            "subconscious",
            "dream",
        )
        background_content_patterns = (
            r"Self-Initiated:",
            r"Brief Curiosity Scan",
            r"^Goal:\s",
            r"Quietly consolidating memory",
            r"Drive alert:",
            r"Winner:\s*",
            r"Cognitive baseline tick",
            r"UNIFIED HEALTH PULSE",
            r"System Active \(Mood:",
            r"Pong \(Reflex path active\)",
            r"Improvement proposal drafted",
            r"Sandbox tests generated",
            r"Top opportunity:",
            r"Running a quiet codebase scan",
            r"cognitive stall in my primary reasoning loop",
        )
        origin_text = str(origin or "").lower()
        looks_background = any(term in origin_text for term in background_origin_terms) or any(
            re.search(pattern, content or "", re.IGNORECASE) for pattern in background_content_patterns
        )
        if looks_background and target in {"primary", "both"}:
            metadata = dict(metadata)
            metadata["autonomous"] = True
            metadata["authority_rerouted"] = True
            metadata["voice"] = False
            metadata["suppress_bus"] = True
            return "secondary", metadata
        return target, metadata

    async def _emit_output_receipt(
        self,
        content: str,
        *,
        origin: str,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        try:
            from core.runtime.executors import run_durable_receipt_io
            from core.runtime.receipts import (
                OutputReceipt,
                digest_output_content,
                get_receipt_store,
            )

            digest = digest_output_content(content)
            recipient_principal_digest = str(
                (metadata or {}).get("recipient_principal_digest") or ""
            ).strip().casefold()
            if not (
                len(recipient_principal_digest) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in recipient_principal_digest
                )
            ):
                recipient_principal_digest = ""
            receipt = OutputReceipt(
                cause=f"output_gate:{origin}",
                origin=str(origin or "system"),
                target=str(target or "primary"),
                digest=digest,
                governance_receipt_id=str((metadata or {}).get("will_receipt_id") or "") or None,
                metadata={
                    "origin": str(origin or "system"),
                    "target": str(target or "primary"),
                    "autonomous": bool((metadata or {}).get("autonomous", False)),
                    "accepted_sinks": list((metadata or {}).get("accepted_sinks", [])),
                    "delivery_stage": "transport_accepted",
                    "recipient_principal_digest": recipient_principal_digest,
                },
            )
            stored = await run_durable_receipt_io(
                get_receipt_store().emit,
                receipt,
                timeout_s=10.0,
                label="output_gate_receipt",
            )
            return str(stored.receipt_id or "") or None
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            record_degradation('output_gate', exc)
            logger.debug("OutputGate: output receipt emit skipped: %s", exc)
            return None
        
    async def emit(
        self,
        content: str,
        origin: str = "system",
        target: str = "primary",
        metadata: dict[str, Any] | None = None,
        timeout: float = 5.0,  # noqa: ASYNC109 - public transport budget
    ) -> str | None:
        """Route a message to the appropriate sink.

        Targets:
        - primary: The main user chat (reply_queue)
        - secondary: Background logs/process trace (secondary_queue)
        - both: Send to both

        INVARIANT: Every primary emission is governed by the Unified Will, or —
        if the Will engine itself is unavailable — fails closed. Autonomous
        output is rerouted to the background sink; a user-awaited reply is still
        delivered (never silently dropped) but is marked governance_degraded and
        recorded CRITICAL so the bypass is audited, never claimed as governed.
        No autonomous content ever reaches the user ungoverned.
        """
        if not content:
            return None
        _primary_governance_decision = None
        metadata = dict(metadata or {})
        target, metadata = self._foreground_policy(content, origin, target, metadata)

        # ── UNIFIED WILL HARD GATE ────────────────────────────────────
        # Nothing user-visible reaches the user ungoverned. If the Will engine
        # itself throws, we fail closed rather than fall through and emit.
        _trusted_reply = origin in ("user", "voice", "admin")
        if target in ("primary", "both"):
            try:
                from core.will import ActionDomain, get_will
                _will = get_will()
                _domain = ActionDomain.RESPONSE if _trusted_reply else ActionDomain.EXPRESSION
                _decision = _will.decide(
                    content=content[:200],
                    source=f"output_gate:{origin}",
                    domain=_domain,
                    priority=0.9 if _trusted_reply else 0.5,
                )
                if not _decision.is_approved():
                    logger.info("OutputGate: Unified Will blocked emission from %s: %s", origin, _decision.reason)
                    return None
                # Attach receipt to metadata for provenance
                metadata = metadata or {}
                metadata["will_receipt_id"] = _decision.receipt_id
                _primary_governance_decision = _decision
            except (ImportError, AttributeError, RuntimeError) as _will_err:
                record_degradation('output_gate', _will_err)
                logger.debug("OutputGate: Will gate degraded: %s", _will_err)
                # FAIL CLOSED — the pre-fix behavior fell through here and
                # emitted to primary with no WillReceipt, silently breaking the
                # stated invariant. Autonomous output has no waiting user, so it
                # is routed to the background sink. A direct reply the user is
                # waiting on is still delivered (dropping it reproduces the
                # "no reply when I talk" failure), but marked ungoverned and
                # recorded CRITICAL so the bypass is audited, not hidden.
                if _trusted_reply:
                    metadata["will_receipt_id"] = None
                    metadata["governance_degraded"] = True
                    record_degradation(
                        'output_gate.will_unavailable_user_reply',
                        _will_err,
                        severity="critical",
                        action="delivered a user-awaited reply marked governance_degraded after the Will engine threw",
                    )
                else:
                    logger.warning(
                        "🛡️ OutputGate: Will engine unavailable — failing closed; routing autonomous %s off the primary channel.",
                        origin,
                    )
                    record_degradation(
                        'output_gate.will_unavailable_autonomous',
                        _will_err,
                        severity="critical",
                        action="rerouted autonomous emission off the primary channel because the Will engine threw",
                    )
                    if target == "both":
                        target = "secondary"
                    else:  # target == "primary"
                        return None

        current_task = asyncio.current_task()
        if current_task is not None and not getattr(current_task, "_aura_supervised", False):
            try:
                cast(Any, current_task)._aura_supervised = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                record_degradation('output_gate', _exc)
                logger.debug("Suppressed Exception: %s", _exc)

        # v30 Hardening: Unified sanitization gate
        content = self._sanitize_autonomous_output(content)
        if self._is_output_blocked(content):
            logger.warning("OutputGate: Blocked potentially unsafe or non-aligned output.")
            return None

        # The closed loop is a REGISTERED SERVICE ("closed_causal_loop"), so
        # importing core.consciousness to notify it was an inverted dependency
        # that bought nothing: a utility module cannot be loaded without the
        # cognitive layer it is supposed to be usable without. Resolved through
        # the container instead — same call, same fail-open behaviour, no edge
        # from utils into consciousness.
        try:
            from core.container import ServiceContainer

            loop = ServiceContainer.get("closed_causal_loop", default=None)
            if loop is not None and hasattr(loop, "on_inference_output"):
                loop.on_inference_output(content)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation('output_gate', exc)
            logger.debug("OutputGate: Closed-loop notification skipped: %s", exc)

        # Bridge 3: Identity Enforcment
        if self.identity_guard:
            try:
                valid, reason, score = self.identity_guard.validate_output(content)
                if not valid:
                    # Log to standard logger
                    logger.error("🛡️ IdentityGuard BLOCKED output: %s (reason=%s)", content[:50], reason)
                    
                    # Log to Enterprise Audit service
                    from core.container import ServiceContainer
                    audit = ServiceContainer.get("audit", default=None)
                    if audit:
                        audit.record(
                            action_type="identity_block",
                            description=f"Identity breach: {reason}",
                            actor="identity_guard",
                            params={"content_snippet": content[:100], "reason": reason, "score": score},
                            result_ok=False
                        )
                    
                    if reason == "FORBIDDEN_PATTERN":
                        content = self.identity_guard.sanitize(content)
                        # Penalize integrity for even attempted breach
                        homeostasis = ServiceContainer.get("homeostasis", default=None)
                        if homeostasis:
                            homeostasis.integrity = max(0.0, homeostasis.integrity - 0.05)
                    else:
                        # Critical breach: Drop output and heavy penalty
                        homeostasis = ServiceContainer.get("homeostasis", default=None)
                        if homeostasis:
                            homeostasis.integrity = max(0.0, homeostasis.integrity - 0.15)
                        return None  # Reject entirely if alignment is too low
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('output_gate', e)
                logger.warning("IdentityGuard evaluation failed: %s", e)

        # Identity drift: measured here, acted on nowhere in this path.
        # The monitor accumulates density across responses; the trend is
        # read by the tension engine and the growth ladder. It deliberately
        # no longer returns a correction string to splice into the next
        # objective — prompting a drifting process to stop drifting is not
        # a causal fix, and the gate is not the place to attempt one.
        from core.container import ServiceContainer
        drift_monitor = ServiceContainer.get("drift_monitor", default=None)
        if drift_monitor:
            drift_monitor.analyze_response(content)


        is_autonomous = metadata.get("autonomous", False)

        # Auto-classify background/internal origins as autonomous.
        # Prevents internal cognitive ticks (reflection, consolidation, dream, initiative)
        # from reaching the primary user channel even when metadata lacks the flag.
        background_origins = frozenset({
            "cognitive_tick", "autonomous", "internal", "background", "dream",
            "reflection", "consolidation", "initiative", "self_model", "shadow",
            "response_generation_internal", "response_generation_background",
            "response_generation_cognitive_tick", "response_generation_consolidation",
            "response_generation_reflection", "response_generation_dream",
        })
        if not is_autonomous and any(bg in origin for bg in background_origins):
            is_autonomous = True
            logger.debug("OutputGate: Auto-classified origin '%s' as autonomous (thought leak prevention).", origin)

        # LOGIC: If it's autonomous but no target specified, default to secondary
        # v30 FIX: Allow 'spontaneous' messages to bypass this and reach the user.
        is_spontaneous = metadata.get("spontaneous", False)
        force_user = metadata.get("force_user", False)
        executive_authority = bool(metadata.get("executive_authority", False))
        trusted_primary_origins = {"user", "voice", "admin", "api"}
        runtime_live = False
        try:
            from core.container import ServiceContainer

            runtime_live = bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
            )
        except (ImportError, AttributeError, RuntimeError):
            runtime_live = False

        if (
            target == "primary"
            and (force_user or (runtime_live and is_spontaneous))
            and origin not in trusted_primary_origins
            and not executive_authority
        ):
            is_autonomous = True
            target = "secondary"
            metadata["authority_missing"] = True
            metadata["authority_rerouted"] = True
            logger.warning(
                "🛡️ OutputGate: Rerouting unauthorized autonomous primary output from %s to secondary.",
                origin,
            )
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "output_gate",
                    "autonomous_primary_without_authority",
                    detail=origin,
                    severity="warning",
                    classification="background_degraded",
                    context={"origin": origin, "target": "primary"},
                )
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation('output_gate', exc)
                logger.debug("OutputGate: degraded-event routing note failed: %s", exc)

        if is_autonomous and target == "primary" and not (is_spontaneous or force_user):
            target = "secondary"
            logger.debug("🛡️ OutputGate: Redirecting autonomous output to secondary: %s...", content[:50])

        primary_receipt_id: str | None = None
        if target in ["primary", "both"]:
            if _primary_governance_decision is not None:
                from core.governance_context import governed_scope

                async with governed_scope(_primary_governance_decision):
                    primary_receipt_id = await self._send_to_primary(
                        content,
                        origin,
                        metadata,
                        timeout=timeout,
                    )
            else:
                primary_receipt_id = await self._send_to_primary(
                    content,
                    origin,
                    metadata,
                    timeout=timeout,
                )
            
        if target in ["secondary", "both"]:
            await self._send_to_secondary(content, origin, metadata, timeout=timeout)
        return primary_receipt_id

    @effect_sink(  # type: ignore[untyped-decorator]
        "output.primary",
        allowed_domains=("response", "expression"),
    )
    async def _send_to_primary(
        self,
        content: str,
        origin: str,
        metadata: dict[str, Any] | None,
        timeout: float = 5.0,  # noqa: ASYNC109 - sink budget
    ) -> str | None:
        """Send to the primary user communication channel."""
        # ★ NEW: Feed reply_queue for REST API waiters (per Architecture Audit)
        from core.container import ServiceContainer
        from core.conversation.tagged_reply_queue import (
            current_reply_origin,
            current_reply_session_id,
        )
        metadata = dict(metadata or {})
        try:
            from core.conversation.session_scope import (
                current_conversation_session,
                current_conversation_turn,
            )

            conversation_id = current_conversation_session()
            conversation_turn_id = current_conversation_turn()
            if conversation_id:
                metadata.setdefault("conversation_id", conversation_id)
            if conversation_turn_id:
                metadata.setdefault("conversation_turn_id", conversation_turn_id)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "output_gate.conversation_custody",
                exc,
                severity="warning",
                action="continued output delivery without conversation-turn metadata",
            )
        accepted_sinks: list[str] = []
        orch = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
        if orch and hasattr(orch, "reply_queue"):
            is_interim = metadata.get("interim", False)
            
            if not is_interim:
                def _put_reply() -> None:
                    try:
                        orch.reply_queue.put_nowait(
                            content,
                            origin=metadata.get("reply_origin")
                            or current_reply_origin(origin),
                            session_id=metadata.get("reply_session_id")
                            or current_reply_session_id(""),
                        )
                    except TypeError:
                        orch.reply_queue.put_nowait(content)

                try:
                    _put_reply()
                    accepted_sinks.append("reply_queue")
                except asyncio.QueueFull:
                    # Drain one stale entry, then retry
                    try:
                        orch.reply_queue.get_nowait()
                        _put_reply()
                        accepted_sinks.append("reply_queue")
                    except (
                        asyncio.QueueEmpty,
                        asyncio.QueueFull,
                        OSError,
                        ConnectionError,
                        TimeoutError,
                    ) as _exc:
                        record_degradation('output_gate', _exc)
                        logger.warning("OutputGate: reply_queue retry failed: %s", _exc)
                except (OSError, ConnectionError, TimeoutError) as e:
                    record_degradation('output_gate', e)
                    logger.warning("OutputGate: Failed to feed reply_queue: %s", e)

            # 2. Add to Conversation History
            if hasattr(self.orchestrator, 'conversation_history'):
                history = self.orchestrator.conversation_history
                if not history or history[-1].get("content") != content:
                    self.orchestrator.conversation_history.append({
                        "role": getattr(self.orchestrator, 'AI_ROLE', 'assistant'),
                        "content": content,
                        "metadata": metadata or {}
                    })

        # 3. Publish to EventBus
        suppress = metadata.get("suppress_bus", False)
        if not suppress:
            try:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                # Legacy HUD Bridging (v14.5)
                # Ensure the message also appears in the log stream for outdated UIs
                bus.publish_threadsafe("log", {
                    "type": "log",
                    "message": f"AURA: {content}",
                    "level": "info",
                    "timestamp": time.time(),
                    "log": f"AURA: {content}" # Explicitly for the .log key check in App.svelte
                })

                aura_message_payload = {
                    "type": "aura_message",
                    "message": content,
                    "origin": origin,
                    "metadata": metadata or {}
                }
                logger.info("OutputGate: Publishing to EventBus...")
                bus.publish_threadsafe("aura_message", aura_message_payload)
                accepted_sinks.append("event_bus")
                bus.publish_threadsafe("log", f"PRIMARY_OUT: {content[:100]}")
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('output_gate', e)
                logger.error("EventBus failure in _send_to_primary: %s. Falling back to Mycelial fail-safe.", e)
                try:
                    from core.mycelium import MycelialNetwork
                    mycelium = MycelialNetwork()
                    logger.info("OutputGate: Checking Mycelial UI callback...")
                    if mycelium.ui_callback:
                        logger.info("OutputGate: Triggering Mycelial UI callback...")
                        # Execute direct UI callback if available
                        try:
                            loop = asyncio.get_running_loop()
                            asyncio.run_coroutine_threadsafe(mycelium.ui_callback(content), loop)
                            accepted_sinks.append("mycelial_ui")
                        except RuntimeError:
                            logger.warning("No running loop for Mycelial fail-safe.")
                    else:
                        logger.warning("OutputGate: Mycelial UI callback is NOT set.")
                except (ImportError, AttributeError, RuntimeError) as e2:
                    record_degradation('output_gate', e2)
                    logger.critical("Final fail-safe failed: %s", e2)

        # 4. Trigger a user-facing multimodal or voice transport. Task
        # acceptance is receipt-worthy; later render failure remains visible in
        # the supervised task's degradation path and cannot score an outcome
        # without a subsequent user reaction.
        renderer = ServiceContainer.get("multimodal_orchestrator", default=None)
        if renderer:
            try:
                track_output_task(get_task_tracker().create_task(renderer.render(content, metadata)))
                accepted_sinks.append("multimodal_renderer")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('output_gate', e)
                logger.debug("Multimodal rendering failed: %s", e)
        else:
            voice = ServiceContainer.get("voice_engine", default=None)
            metadata_allows_voice = metadata.get("voice", True)
            tts_enabled = getattr(voice, "speaking_enabled", True) if voice else False
            if voice and metadata_allows_voice and tts_enabled:
                try:
                    track_output_task(get_task_tracker().create_task(voice.speak(content)))
                    accepted_sinks.append("voice_engine")
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('output_gate', e)
                    logger.debug("Legacy Voice trigger failed: %s", e)

        if not accepted_sinks:
            failure = RuntimeError("primary output was not accepted by any user-facing sink")
            record_degradation(
                "output_gate.primary_delivery",
                failure,
                severity="error",
                action="withheld output receipt because no primary transport accepted the response",
            )
            logger.error("OutputGate: %s", failure)
            return None
        metadata["accepted_sinks"] = accepted_sinks
        receipt_id = await self._emit_output_receipt(
            content,
            origin=origin,
            target="primary",
            metadata=metadata,
        )
        if not receipt_id:
            failure = RuntimeError("primary output receipt could not be persisted")
            record_degradation(
                "output_gate.primary_delivery",
                failure,
                severity="error",
                action="withheld delivery confirmation because its output receipt was not durable",
            )
            logger.error("OutputGate: %s", failure)
            return None
        try:
            from core.epistemics.epistemic_reach import (
                acknowledge_epistemic_correction_delivery,
            )

            acknowledge_epistemic_correction_delivery(content)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("output_gate", exc)
            logger.debug("Epistemic correction delivery acknowledgement skipped: %s", exc)

        return receipt_id

    async def _send_to_secondary(
        self,
        content: str,
        origin: str,
        metadata: dict[str, Any] | None,
        timeout: float = 5.0,  # noqa: ASYNC109 - sink budget
    ) -> None:
        """Send to the secondary background log channel."""
        try:
            await asyncio.wait_for(self.secondary_queue.put({
                "content": content,
                "origin": origin,
                "metadata": metadata or {}
            }), timeout=timeout)
        except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as e:
            record_degradation('output_gate', e)
            logger.error("Failed to put in secondary_queue: %s", e)
            
        logger.info("📡 [AUTONOMOUS] %s: %s", origin, content)

    async def get_secondary_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Generator for secondary output stream."""
        while getattr(self, '_running', True):
            yield await self.secondary_queue.get()
            self.secondary_queue.task_done()

_gates: weakref.WeakKeyDictionary[Any, AutonomousOutputGate] = weakref.WeakKeyDictionary()
_background_tasks: set[asyncio.Task[Any]] = set()


class _DummyOrchestrator:
    pass


_dummy_orchestrator = _DummyOrchestrator()


def get_output_gate(orchestrator: Any = None) -> AutonomousOutputGate:
    if orchestrator is None:
        # Fallback for legacy calls without orchestrator
        orchestrator = _dummy_orchestrator

    if orchestrator not in _gates:
        _gates[orchestrator] = AutonomousOutputGate(orchestrator if not hasattr(orchestrator, "__dict__") or "reply_queue" in orchestrator.__dict__ else None)
    return _gates[orchestrator]

# Helper to track background tasks in OutputGate
def track_output_task(task: asyncio.Task[Any]) -> None:
    try:
        task_metadata = cast(Any, task)
        task_metadata._aura_supervised = True
        task_metadata._aura_task_tracker = "OutputGate"
    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
        record_degradation('output_gate', _exc)
        logger.debug("Suppressed Exception: %s", _exc)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    
    def _handle_result(t: asyncio.Task[Any]) -> None:
        try:
            t.result()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('output_gate', e)
            logging.getLogger("Aura.OutputGate").error("Output task failed: %s", e)
            
    task.add_done_callback(_handle_result)
