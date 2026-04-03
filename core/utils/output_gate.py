"""Autonomous Output Gate — Communication Triage for Aura.

Standardizes which messages reach the User (Primary) vs. Background (Secondary).
Prevents "Autonomous Pollution" where background search results flood the chat.
"""
import logging
import asyncio
import time
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("Aura.OutputGate")

class AutonomousOutputGate:
    """Triage engine for Aura's communicative outputs."""
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        # Secondary sink for background/autonomous logs
        self.secondary_queue = asyncio.Queue()
        
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
            r"Still with me\? Sometimes quiet",
            r"Would you like to dive into",
        ]

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
        
    async def emit(self, content: str, 
             origin: str = "system", 
             target: str = "primary",
             metadata: Optional[Dict[str, Any]] = None,
             timeout: float = 5.0):
        """Route a message to the appropriate sink.
        
        Targets:
        - primary: The main user chat (reply_queue)
        - secondary: Background logs/process trace (secondary_queue)
        - both: Send to both
        """
        if not content:
            return

        current_task = asyncio.current_task()
        if current_task is not None and not getattr(current_task, "_aura_supervised", False):
            try:
                setattr(current_task, "_aura_supervised", True)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        # v30 Hardening: Unified sanitization gate
        content = self._sanitize_autonomous_output(content)
        if self._is_output_blocked(content):
            logger.warning("OutputGate: Blocked potentially unsafe or non-aligned output.")
            return

        try:
            from core.consciousness.closed_loop import notify_closed_loop_output

            notify_closed_loop_output(content)
        except Exception as exc:
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
                        return # Reject entirely if alignment is too low
            except Exception as e:
                logger.warning("IdentityGuard evaluation failed: %s", e)

        # v40: Identity Drift Monitor
        from core.container import ServiceContainer
        drift_monitor = ServiceContainer.get("drift_monitor", default=None)
        if drift_monitor:
            score, signals = drift_monitor.analyze_response(content)
            if signals:
                correction = drift_monitor.get_correction_injection(signals)
                # Store correction for next generation context if needed
                # For now, we record it in the orchestrator if available
                if self.orchestrator and hasattr(self.orchestrator, "_pending_correction"):
                    self.orchestrator._pending_correction = correction
            
            # Check context window health
            # This logic will be handled in the ContextAssembler or CognitiveEngine think loop
            pass


        metadata = dict(metadata or {})
        is_autonomous = metadata.get("autonomous", False)

        # Auto-classify background/internal origins as autonomous.
        # Prevents internal cognitive ticks (reflection, consolidation, dream, initiative)
        # from reaching the primary user channel even when metadata lacks the flag.
        _BACKGROUND_ORIGINS = frozenset({
            "cognitive_tick", "autonomous", "internal", "background", "dream",
            "reflection", "consolidation", "initiative", "self_model", "shadow",
            "response_generation_internal", "response_generation_background",
            "response_generation_cognitive_tick", "response_generation_consolidation",
            "response_generation_reflection", "response_generation_dream",
        })
        if not is_autonomous and any(bg in origin for bg in _BACKGROUND_ORIGINS):
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
        except Exception:
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
            except Exception as exc:
                logger.debug("OutputGate: degraded-event routing note failed: %s", exc)

        if is_autonomous and target == "primary" and not (is_spontaneous or force_user):
            target = "secondary"
            logger.debug("🛡️ OutputGate: Redirecting autonomous output to secondary: %s...", content[:50])

        if target in ["primary", "both"]:
            await self._send_to_primary(content, origin, metadata, timeout=timeout)
            
        if target in ["secondary", "both"]:
            await self._send_to_secondary(content, origin, metadata, timeout=timeout)

    async def _send_to_primary(self, content: str, origin: str, metadata: Optional[Dict[str, Any]], timeout: float = 5.0):
        """Send to the primary user communication channel."""
        # ★ NEW: Feed reply_queue for REST API waiters (per Architecture Audit)
        from core.container import ServiceContainer
        from core.tagged_reply_queue import current_reply_origin, current_reply_session_id
        orch = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
        if orch and hasattr(orch, "reply_queue"):
            metadata = metadata or {}
            is_interim = metadata.get("interim", False)
            
            if not is_interim:
                try:
                    # Using put_nowait for non-blocking REST response feeding
                    orch.reply_queue.put_nowait(
                        content,
                        origin=metadata.get("reply_origin") or current_reply_origin(origin),
                        session_id=metadata.get("reply_session_id") or current_reply_session_id(""),
                    )
                except TypeError:
                    orch.reply_queue.put_nowait(content)
                except asyncio.QueueFull:
                    # Drain one stale entry, then retry
                    try:
                        orch.reply_queue.get_nowait()
                        try:
                            orch.reply_queue.put_nowait(
                                content,
                                origin=metadata.get("reply_origin") or current_reply_origin(origin),
                                session_id=metadata.get("reply_session_id") or current_reply_session_id(""),
                            )
                        except TypeError:
                            orch.reply_queue.put_nowait(content)
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                except Exception as e:
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
                bus.publish_threadsafe("log", f"PRIMARY_OUT: {content[:100]}")
            except Exception as e:
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
                        except RuntimeError:
                            logger.warning("No running loop for Mycelial fail-safe.")
                    else:
                        logger.warning("OutputGate: Mycelial UI callback is NOT set.")
                except Exception as e2:
                    logger.critical("Final fail-safe failed: %s", e2)

        # 4. Trigger High-Fidelity Multimodal Manifestation
        from core.container import ServiceContainer
        renderer = ServiceContainer.get("multimodal_orchestrator", default=None)
        if renderer:
            try:
                track_output_task(asyncio.create_task(renderer.render(content, metadata)))
            except Exception as e:
                logger.debug("Multimodal rendering failed: %s", e)
        else:
            voice = ServiceContainer.get("voice_engine", default=None)
            # Check both: metadata voice flag AND voice engine's speaking_enabled (TTS mute state)
            metadata_allows_voice = metadata.get("voice", True) if metadata else True
            tts_enabled = getattr(voice, "speaking_enabled", True) if voice else False
            if voice and metadata_allows_voice and tts_enabled:
                try:
                    track_output_task(asyncio.create_task(voice.speak(content)))
                except Exception as e:
                    logger.debug("Legacy Voice trigger failed: %s", e)

    async def _send_to_secondary(self, content: str, origin: str, metadata: Optional[Dict[str, Any]], timeout: float = 5.0):
        """Send to the secondary background log channel."""
        try:
            await asyncio.wait_for(self.secondary_queue.put({
                "content": content,
                "origin": origin,
                "metadata": metadata or {}
            }), timeout=timeout)
        except Exception as e:
            logger.error("Failed to put in secondary_queue: %s", e)
            
        logger.info("📡 [AUTONOMOUS] %s: %s", origin, content)

    async def get_secondary_stream(self):
        """Generator for secondary output stream."""
        while True:
            yield await self.secondary_queue.get()
            self.secondary_queue.task_done()

import weakref
from typing import Set

_gates = weakref.WeakKeyDictionary()
_background_tasks: Set[asyncio.Task] = set()

def get_output_gate(orchestrator=None):
    if orchestrator is None:
        # Fallback for legacy calls without orchestrator
        # We use a static dummy object as a key
        class Dummy: pass
        if not hasattr(get_output_gate, "_dummy"):
            get_output_gate._dummy = Dummy()
        orchestrator = get_output_gate._dummy

    if orchestrator not in _gates:
        _gates[orchestrator] = AutonomousOutputGate(orchestrator if not hasattr(orchestrator, "__dict__") or "reply_queue" in orchestrator.__dict__ else None)
    return _gates[orchestrator]

# Helper to track background tasks in OutputGate
def track_output_task(task: asyncio.Task):
    try:
        setattr(task, "_aura_supervised", True)
        setattr(task, "_aura_task_tracker", "OutputGate")
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    
    def _handle_result(t: asyncio.Task):
        try:
            t.result()
        except Exception as e:
            logging.getLogger("Aura.OutputGate").error("Output task failed: %s", e)
            
    task.add_done_callback(_handle_result)
