"""Incoming Logic Mixin for RobustOrchestrator.
Extracts the incoming message handling pipeline, routing, and filesystem checks.
"""
import asyncio
import logging
import os
import re
import time
from typing import Any

from core.runtime.governance_policy import allow_direct_user_shortcut

logger = logging.getLogger(__name__)


class IncomingLogicMixin:
    """Handles incoming message routing, pipeline dispatch, and the core logic handler."""

    async def _route_prefixed_message(self, message: str, prefix: str, origin: str) -> Any:
        # Implementation of legacy prefix routing (e.g. stripping tag and updating origin)
        content = message[len(prefix):].strip()
        new_origin = origin
        if prefix == "[VOICE]": new_origin = "voice"
        elif prefix == "[ADMIN]": new_origin = "admin"
        return await self._process_message_pipeline(content, origin=new_origin)

    async def _process_message_pipeline(self, message: Any, origin: str = "user", **kwargs) -> None:
        # This is the original _handle_incoming_message logic, moved for testability
        return await self._original_handle_incoming_logic(message, origin, **kwargs)

    async def _handle_incoming_message(self, message: Any, origin: str = "user", **kwargs) -> None:
        from core.utils.task_tracker import get_task_tracker  # lazy, inside body
        tracker = get_task_tracker()

        # Strip legacy prefix tags and route
        if isinstance(message, str):
            for prefix in ("[VOICE]", "[ADMIN]", "Impulse:", "Thought:"):
                if message.startswith(prefix):
                    coro = self._route_prefixed_message(message, prefix, origin)
                    # v31.1: Await the task to ensure serialization via the dispatch semaphore.
                    self._current_thought_task = tracker.track_task(asyncio.create_task(coro))
                    await self._current_thought_task
                    return

        # Default path
        coro = self._process_message_pipeline(message, origin=origin, **kwargs)
        self._current_thought_task = tracker.track_task(asyncio.create_task(coro))
        await self._current_thought_task
        return self._current_thought_task

    async def _handle_filesystem_reality_check(self, message: str, origin: str) -> bool:
        """Fast-path explicit file existence checks without invoking deep cognition."""
        if not isinstance(message, str):
            return False

        match = re.search(
            r"(?:(?:check|see|verify|test)\s+(?:if\s+)?|does\s+)(.+?)\s+exist(?:s)?(?:\.|!|\?|$)",
            message.strip(),
            re.IGNORECASE,
        )
        if not match:
            return False

        raw_path = match.group(1).strip().strip("'\"")
        if not raw_path:
            return False

        resolved_path = os.path.expanduser(raw_path)
        exists = await asyncio.to_thread(os.path.exists, resolved_path)
        is_dir = await asyncio.to_thread(os.path.isdir, resolved_path) if exists else False
        summary = f"{raw_path} exists." if exists else f"{raw_path} does not exist."
        result = {
            "ok": True,
            "path": raw_path,
            "exists": exists,
            "kind": "directory" if exists and is_dir else ("file" if exists else None),
            "state": "present" if exists else "missing",
            "summary": summary,
        }

        trace = None
        try:
            trace = self._init_cognitive_trace(message, origin)
            trace.record_step("filesystem_reality_check", result)
        except Exception as trace_err:
            logger.debug("Filesystem reality trace init skipped: %s", trace_err)

        try:
            from core.world_model.expectation_engine import ExpectationEngine

            ee = ExpectationEngine(getattr(self, "cognitive_engine", None))
            await ee.update_beliefs_from_result("file_exists_check", result)
        except Exception as belief_err:
            logger.debug("Filesystem reality belief update skipped: %s", belief_err)

        try:
            self._record_action_in_history("file_exists_check", result)
        except Exception as history_err:
            logger.debug("Filesystem reality action history skipped: %s", history_err)

        self._record_message_in_history(message, origin)
        self._record_message_in_history(summary, "assistant")

        if origin in ("user", "voice", "admin"):
            await self.output_gate.emit(summary, origin=origin, target="primary")

        if trace is not None:
            trace.record_step("end", {"response": summary[:100]})
            trace.save()

        self._last_thought_time = time.time()
        return True

    async def _original_handle_incoming_logic(self, message: Any, origin: str = "user", suppress_ui: bool = False):
        """Route an incoming message through the deterministic State Machine pipeline."""
        from ...container import ServiceContainer
        from ...config import config
        from core.autonomy_guardian import AutonomyGuardian
        from core.supervisor.registry import get_task_registry, TaskStatus
        from core.health.degraded_events import record_degraded_event

        payload_context: dict[str, Any] = {}
        if isinstance(message, dict):
            payload_context = message.get("context", {})
            origin = message.get("origin", origin)
            message = message.get("content", str(message))

        # v49 [STRICTURE] Fix: Extract actual string if passed a tuple from the priority queue
        # Unpack nested tuples (common in PriorityQueues)
        while isinstance(message, tuple):
            message = message[-1]

        if not isinstance(message, str):
            message = str(message)

        # Detect Origin from message prefixes (Legacy support)
        safe_msg = message

        # Ignoring Logic: Skip background noise if user is active
        now = time.time()
        _last_user = getattr(self, '_last_user_interaction_time', 0)
        if origin == "voice" and (now - _last_user) < 30.0:
            # If user just typed something, ignore background audio for 30s
            logger.info("🛡️ IGNORE: Skipping [VOICE] input during active user session.")
            return None

        # Publish status with interim=True (per audit)
        self._publish_status({
            "event": "thinking",
            "origin": origin,
            "message": safe_msg[:50],
            "interim": True
        })

        if await self._handle_filesystem_reality_check(safe_msg, origin):
            return None

        # Phase 0: Social Reflexes (Zero-Latency)
        if origin in ("user", "voice", "admin") and not config.skeletal_mode:
            reflex_response = await self._check_social_reflexes(safe_msg)
            if reflex_response:
                logger.info("⚡ [REFLEX] Social ritual matched: '%s' -> '%s'...", safe_msg[:20], reflex_response[:30])
                await self.output_gate.emit(reflex_response, origin=origin, target="primary")
                # Record in history anyway so subsequent turns have context
                self._record_message_in_history(safe_msg, origin)
                self._record_message_in_history(reflex_response, "assistant")
                return None

        # Notify AgencyCore of user interaction
        if origin in ("user", "voice", "admin"):
            agency = getattr(self, '_agency_core', None)
            if agency:
                agency.on_user_message()

            # WorldState: Track user activity for environment awareness
            try:
                from core.world_state import get_world_state
                get_world_state().on_user_message(message=safe_msg if isinstance(safe_msg, str) else "")
            except Exception:
                pass

            # DriveEngine: Satisfy social drive on user contact + relieve boredom
            try:
                drive = ServiceContainer.get("drive_engine", default=None)
                if drive:
                    self._fire_and_forget(drive.satisfy("social", 15.0), name="drive_social_satisfy")
                    if hasattr(drive, "relieve_boredom"):
                        drive.relieve_boredom("user_interaction")
            except Exception:
                pass

            # NeurochemicalSystem: user interaction triggers novelty + social
            try:
                ncs = ServiceContainer.get("neurochemical_system", default=None)
                if ncs:
                    ncs.on_social_connection(0.2)
                    ncs.on_novelty(0.15)
            except Exception:
                pass

            # Zenith Hardening: Reset boredom and learning cooldowns
            if hasattr(self, 'volition') and self.volition:
                self.volition.notify_activity()
            if hasattr(self, 'continuous_learner') and self.continuous_learner:
                self.continuous_learner.scheduler.notify_activity()

        # Admin Commands (Snapshot Freezing/Thawing)
        if origin == "admin":
            if safe_msg.strip() == "/snapshot":
                try:
                    from core.resilience.snapshot_manager import SnapshotManager
                    mgr = SnapshotManager(self)
                    success = mgr.freeze()
                    await self.output_gate.emit(f"✅ Cognitive State Snapshot {'successful' if success else 'failed'}.", origin="admin")
                except Exception as e:
                    logger.error("Snapshot command failed: %s", e)
                return None
            elif safe_msg.strip() == "/thaw":
                try:
                    from core.resilience.snapshot_manager import SnapshotManager
                    mgr = SnapshotManager(self)
                    success = mgr.thaw()
                    await self.output_gate.emit(f"🔥 Cognitive State Thaw {'successful' if success else 'failed'}.", origin="admin")
                except Exception as e:
                    logger.error("Thaw command failed: %s", e)
                return None

        # Broadcast for autonomous monitoring
        try:
            from core.event_bus import EventPriority, get_event_bus
            priority_tag = EventPriority.USER if origin in ("user", "voice", "admin") else EventPriority.BACKGROUND
            get_event_bus().publish_threadsafe("chat_input", {"text": message, "origin": origin, "context": payload_context}, priority=priority_tag)
        except Exception as e:
            logger.debug("Failed to publish chat_input for scanner: %s", e)

        self.status.is_processing = True
        self._current_processing_start = time.monotonic()
        self._reflex_sent_for_current = False
        # v47: Track user interaction time for dream cooldown
        if origin in ("user", "voice", "admin"):
            self._last_user_interaction_time = time.time()
            # v50: Reset idle model swap flag so AutonomicCore knows to
            # re-warm the 32B cortex if it was hibernated.
            try:
                autonomic = ServiceContainer.get("autonomic_core", default=None)
                if autonomic and hasattr(autonomic, '_reset_idle_swap'):
                    autonomic._reset_idle_swap()
            except Exception:
                pass
            # Standardize: Always append to conversation history early for traceability
            if not hasattr(self, 'conversation_history'):
                self.conversation_history = []
            self.conversation_history.append({"role": "user", "content": message, "timestamp": time.time()})

            # v49: Store true semantic memory (Episodic Storage)
            try:
                vector_mem = ServiceContainer.get("vector_memory_engine", default=None)
                if vector_mem and hasattr(vector_mem, "store"):
                    # Get emotional context for enriched memory
                    affect = ServiceContainer.get("affect_engine", None)
                    emotional_context = None
                    if affect and hasattr(affect, 'get_state_sync'):
                        emotional_context = affect.get_state_sync()

                    # Non-blocking store — gated by Unified Will
                    _mem_allowed = True
                    try:
                        from core.will import ActionDomain, get_will
                        _mem_decision = get_will().decide(
                            content=message[:80], source="vector_memory",
                            domain=ActionDomain.MEMORY_WRITE, priority=0.3,
                        )
                        if not _mem_decision.is_approved():
                            _mem_allowed = False
                            logger.debug("Vector memory store blocked by Unified Will: %s", _mem_decision.reason)
                    except Exception:
                        pass  # fail-open for safety

                    if _mem_allowed:
                        self._fire_and_forget(vector_mem.store(
                            content=message,
                            memory_type="episodic",
                            emotional_context=emotional_context,
                            source="user",
                            tags=["conversation", "user_input"]
                        ), name="vector_memory_store")
            except Exception as store_err:
                logger.debug("Semantic memory storage failed: %s", store_err)

            # --- Zenith Memory Guard & Initiative Hooks ---
            mem_guard = getattr(self, "conversational_guard", None) or self._get_service("conversational_guard")
            if mem_guard:
                if hasattr(self, "cognitive_engine") and self.cognitive_engine:
                    self._fire_and_forget(
                        mem_guard.append_turn("user", message, self.cognitive_engine),
                        name="memory_guard_user_turn",
                    )

            ini_engine = getattr(self, "initiative_engine", None) or self._get_service("initiative_engine")
            if ini_engine:
                ini_engine.register_user_interaction()

            # Resolve current AuraState (sync, no await needed — state_repo._current is the live object)
            _live_state = getattr(self.state_repo, "_current", None) if hasattr(self, "state_repo") else None

            # Update discourse state (topic thread, user emotional trend, conversation energy)
            # Gated by Unified Will — internal model updates are STATE_MUTATION
            _internal_update_allowed = True
            try:
                from core.will import ActionDomain, get_will
                _state_decision = get_will().decide(
                    content="internal_model_update", source="cognitive_background",
                    domain=ActionDomain.STATE_MUTATION, priority=0.2,
                )
                if not _state_decision.is_approved():
                    _internal_update_allowed = False
                    logger.debug("Background cognitive updates blocked by Unified Will: %s", _state_decision.reason)
            except Exception:
                pass  # fail-open

            if _internal_update_allowed:
                try:
                    discourse_tracker = ServiceContainer.get("discourse_tracker", default=None)
                    if discourse_tracker and _live_state is not None:
                        self._fire_and_forget(
                            discourse_tracker.update(_live_state, message),
                            name="discourse_tracker_update",
                        )
                except Exception as _dt_err:
                    logger.debug("DiscourseTracker update skipped: %s", _dt_err)

                # Update Theory of Mind user model (rapport, trust, emotional state)
                try:
                    tom = ServiceContainer.get("theory_of_mind", default=None)
                    if tom:
                        user_id = (getattr(self, "user_identity", {}) or {}).get("name", "bryan")
                        self._fire_and_forget(
                            tom.understand_user(user_id, message),
                            name="theory_of_mind_update",
                        )
                except Exception as _tom_err:
                    logger.debug("TheoryOfMind update skipped: %s", _tom_err)

        # Initialize AutonomyGuardian if not present
        if not hasattr(self, '_autonomy_guardian'):
            self._autonomy_guardian = AutonomyGuardian(orchestrator=self)

        try:
            # 1. System Hooks (Personality, Moral Awareness, etc.)
            await self.hooks.trigger("on_message", message=message, origin=origin)

            # 2. Cancel / wait for previous thought task
            # v47 FIX: ONLY cancel AUTONOMOUS/DREAM tasks for user messages.
            # NEVER cancel an in-flight user-message response — that throws
            # away the response the user is waiting for.
            current_task = asyncio.current_task()
            if (task := self._current_thought_task) is not None and task != current_task and not task.done():
                is_user_origin = (origin == "user") # Keyboard user is highest priority
                current_is_replaceable = getattr(self, '_current_task_is_autonomous', True) or getattr(self, '_current_origin', "") == "voice"

                if is_user_origin and current_is_replaceable:
                    # Cancel autonomous/dream OR VOICE task to make way for user
                    logger.info("🛑 Cancelling stale %s task for direct user input...", getattr(self, '_current_origin', "background"))
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.debug("Autonomous task cancelled successfully.")
                elif is_user_origin and not current_is_replaceable:
                    in_flight_age_s = max(
                        0.0,
                        time.monotonic() - float(getattr(self, "_current_processing_start", 0.0) or 0.0),
                    )
                    supersede_after_s = 20.0
                    grace_wait_s = 8.0 if in_flight_age_s < supersede_after_s else 0.0

                    if grace_wait_s > 0.0:
                        logger.info(
                            "⏳ Previous user response in flight — giving it %.0fs to finish (age %.1fs)...",
                            grace_wait_s,
                            in_flight_age_s,
                        )
                        try:
                            await asyncio.wait_for(asyncio.shield(task), timeout=grace_wait_s)
                        except (TimeoutError, asyncio.CancelledError, asyncio.exceptions.TimeoutError):
                            pass

                    if task.done():
                        try:
                            await task
                        except asyncio.CancelledError:
                            logger.debug("Previous user task finished via cancellation before supersede.")
                    else:
                        logger.warning(
                            "🛑 Superseding stalled user response after %.1fs to honor the latest user turn.",
                            max(
                                in_flight_age_s,
                                time.monotonic() - float(getattr(self, "_current_processing_start", 0.0) or 0.0),
                            ),
                        )
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=2.0)
                        except (TimeoutError, asyncio.CancelledError, asyncio.exceptions.TimeoutError):
                            logger.debug("Previous user task did not finish cleanly after supersede cancel.")
                else:
                    logger.info("🛡️ Guardian: Preserving in-flight task (origin=%s is not user)", origin)
                    try:
                        # Increase to 60s for background tasks
                        await asyncio.wait_for(task, timeout=60.0)
                    except (TimeoutError, asyncio.CancelledError, asyncio.exceptions.TimeoutError):
                        logger.warning("🛡️ Guardian: Previous task timed out (60s), proceeding anyway")

            # 3. Commit current origin and message state
            self._current_origin = origin
            self._current_user_message = message

            # 3. Inject HOT MEMORY before cognitive processing
            hot_memory = {}
            if self.memory:
                try:
                    hot_memory = await self.memory.get_hot_memory(limit=5)
                    payload_context["hot_memory"] = hot_memory
                except Exception as e:
                    logger.debug("Failed to fetch Hot Memory: %s", e)

            # WIRE-04: Inject Cognitive Brief
            try:
                kernel = ServiceContainer.get("cognitive_kernel", default=None)
                if kernel:
                    brief = await kernel.evaluate(message, history=getattr(self, 'conversation_history', []))
                    payload_context["cognitive_brief"] = brief
            except Exception as e:
                logger.debug("Failed to fetch Cognitive Brief: %s", e)

            # 3b. Inject Meta-Cognition Correction Shards
            if hasattr(self, '_correction_shards') and self._correction_shards:
                unapplied_shards = [s for s in self._correction_shards if not s.get("applied", False)]
                if unapplied_shards:
                    shards_content = [s["content"] for s in unapplied_shards]
                    payload_context["correction_shards"] = shards_content
                    for s in unapplied_shards:
                        s["applied"] = True
                    logger.info("🎯 [META] Injected %d correction shards into context.", len(shards_content))

            # 4. Deterministic Execution
            # Initialize the cognitive trace before deferring execution
            trace = None  # Sentinel for safety
            try:
                trace = self._init_cognitive_trace(message, origin)
            except Exception as trace_err:
                logger.debug("Failed to initialize cognitive trace: %s", trace_err)

            # [HARDENING] If trace init failed, create a minimal fallback so downstream
            # code never sees None / 'unknown' for trace_id.
            if trace is None:
                import uuid as _uuid
                _fallback_id = f"fallback_{_uuid.uuid4().hex[:12]}"
                logger.warning("Cognitive trace unavailable — using fallback trace_id=%s", _fallback_id)
                trace = type("FallbackTrace", (), {"trace_id": _fallback_id, "record_step": lambda *a, **k: None})()

            # We defer execution to the State Machine so we don't block the dispatch loop
            async def _execute_and_reply():
                successful_tools = []
                priority = 1.0 if origin in ("user", "voice", "admin") else 0.1

                # ══════════════════════════════════════════════════════
                # UNIFIED WILL GATE — Every response passes through here
                # ══════════════════════════════════════════════════════
                try:
                    from core.will import ActionDomain, get_will
                    _will = get_will()
                    _will_decision = _will.decide(
                        content=message[:200] if isinstance(message, str) else str(message)[:200],
                        source=origin,
                        domain=ActionDomain.RESPONSE,
                        priority=priority,
                        context=payload_context,
                    )
                    # Store the will decision for downstream use
                    payload_context["will_decision"] = _will_decision
                    payload_context["will_receipt_id"] = _will_decision.receipt_id
                    if not _will_decision.is_approved():
                        logger.warning("Unified Will REFUSED response: %s", _will_decision.reason)
                        if origin in ("user", "voice", "admin"):
                            await self.output_gate.emit(
                                "I need a moment to collect myself before I can respond properly.",
                                origin=origin, target="primary",
                            )
                        return
                    if _will_decision.constraints:
                        payload_context["will_constraints"] = _will_decision.constraints
                except Exception as _will_err:
                    logger.debug("Unified Will gate degraded: %s", _will_err)

                # Task Registry Integration
                task_id = self._task_registry.register_task(
                    owner=f"Orchestrator_{origin}",
                    description=f"Thinking: {message[:100]}...",
                    metadata={"origin": origin, "trace_id": getattr(trace, 'trace_id', 'unknown')}
                )
                self._task_registry.update_task(task_id, status=TaskStatus.RUNNING)

                # Use mycelial context only if available (Enterprise v3.0 path)
                class AsyncNullContext:
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args): pass

                flow_ctx = self.mycelium.rooted_flow("cognition", "response", f"Process: {message[:20]}", priority=priority) if self.mycelium and hasattr(self.mycelium, 'rooted_flow') else AsyncNullContext()

                async with flow_ctx:
                    try:
                        # ══════════════════════════════════════════════════════
                        # MYCELIAL HARDWIRED PATHWAY BYPASS (Enterprise v3.0)
                        # ══════════════════════════════════════════════════════
                        hardwired_result = self.mycelium.match_hardwired(message) if self.mycelium and hasattr(self.mycelium, 'match_hardwired') else None
                        if hardwired_result:
                            pathway, extracted_params = hardwired_result

                            logger.info(
                                "🍄 [MYCELIUM] ⚡ HARDWIRED MATCH: '%s' → %s",
                                pathway.pathway_id, pathway.skill_name
                            )

                            if not allow_direct_user_shortcut(origin):
                                logger.info(
                                    "🧭 [MYCELIUM] Yielding hardwired user-facing pathway '%s' to the governed runtime path",
                                    pathway.pathway_id,
                                )
                                hardwired_result = None
                            elif pathway.direct_response:
                                try:
                                    from core.constitution import get_constitutional_core

                                    _emit_ok, _emit_reason, _authority_decision = await get_constitutional_core(self).approve_expression(
                                        pathway.direct_response,
                                        source=f"mycelium:{pathway.pathway_id}",
                                        urgency=max(0.1, min(1.0, float(priority or 0.0))),
                                    )
                                    if not _emit_ok:
                                        logger.info(
                                            "🛡️ [AUTHORITY] Hardwired direct response '%s' blocked: %s",
                                            pathway.pathway_id,
                                            _emit_reason,
                                        )
                                        try:
                                            from core.unified_action_log import get_action_log
                                            get_action_log().record(
                                                pathway.skill_name or "direct_response",
                                                f"mycelium:{pathway.pathway_id}",
                                                "reflex",
                                                "blocked",
                                                str(_emit_reason),
                                            )
                                        except Exception as _exc:
                                            logger.debug("Suppressed Exception: %s", _exc)
                                        hardwired_result = None
                                except Exception as _exec_err:
                                    logger.debug("Hardwired direct-response emission approval skipped: %s", _exec_err)

                        if hardwired_result:
                            try:
                                self._publish_telemetry({
                                    "type": "activity",
                                    "label": pathway.activity_label,
                                    "show": True,
                                })

                                # Phase 5.1: Zero-Latency Direct Response Bypass
                                if pathway.direct_response:
                                    logger.info("⚡ [REFLEX] Direct response triggered for '%s'", pathway.pathway_id)
                                    try:
                                        from core.constitution import ProposalKind, get_constitutional_core

                                        get_constitutional_core(self).record_external_decision(
                                            kind=ProposalKind.EXPRESSION,
                                            source=origin or "user",
                                            summary=pathway.direct_response[:220],
                                            outcome="approved",
                                            reason=f"hardwired_direct_response:{pathway.pathway_id}",
                                            target="primary",
                                            payload={"pathway_id": pathway.pathway_id},
                                        )
                                    except Exception as audit_exc:
                                        logger.debug("Hardwired direct-response audit skipped: %s", audit_exc)
                                    res = {"ok": True, "response": pathway.direct_response}
                                else:
                                    if pathway.skill_name in ("generate_image", "sovereign_imagination") and "prompt" not in extracted_params:
                                        extracted_params["prompt"] = message.strip()

                                    res = await self.execute_tool(
                                        pathway.skill_name,
                                        extracted_params,
                                        origin=origin,
                                        payload_context=payload_context,
                                    )
                            finally:
                                self._publish_telemetry({
                                    "type": "activity",
                                    "label": "Aura is idle.",
                                    "show": False,
                                })

                            exec_success = isinstance(res, dict) and res.get("ok", False)
                            if exec_success:
                                successful_tools.append(pathway.skill_name)

                            if self.mycelium and hasattr(self.mycelium, 'reinforce'):
                                self.mycelium.reinforce(pathway.pathway_id, success=exec_success)

                            rich_res = res.get("results", res) if isinstance(res, dict) else res

                            try:
                                self._record_action_in_history(pathway.skill_name, rich_res)
                            except Exception as history_err:
                                logger.debug("Hardwired action history skipped: %s", history_err)

                            try:
                                from core.world_model.expectation_engine import ExpectationEngine

                                ee = ExpectationEngine(getattr(self, "cognitive_engine", None))
                                await ee.update_beliefs_from_result(pathway.skill_name, rich_res)
                            except Exception as belief_err:
                                logger.debug("Hardwired belief update skipped: %s", belief_err)

                            self._publish_telemetry({
                                "type": "action_result",
                                "tool": pathway.skill_name,
                            })

                            # ══════════════════════════════════════════════════════
                            # SCANNER / FILTER LAYER (Pre-Intent)
                            # ══════════════════════════════════════════════════════
                            from core.container import ServiceNotFoundError, get_container
                            scanner = None
                            try:
                                scanner = get_container().get("sovereign_scanner")
                            except ServiceNotFoundError:
                                # Fallback to intent_router if sovereign_scanner is not found
                                # This ensures some form of filtering/scanning is always available
                                logger.debug("Sovereign Scanner not found, falling back to IntentRouter for scanning.")
                                scanner = self.intent_router

                            if scanner and not getattr(scanner, "called", False) and not config.skeletal_mode:
                                try:
                                    scan_res = scanner.scan(message) if hasattr(scanner, "scan") else None
                                    if scan_res and scan_res.get("blocked"):
                                        final_response = scan_res.get("reason", "I cannot process this request.")
                                        if origin in ("user", "voice", "admin"):
                                            await self.output_gate.emit(final_response, origin=origin, target="primary")
                                        return
                                except Exception as e:
                                    logger.warning("Error during scanner execution: %s", e)
                                    # Continue processing if scanner fails, don't block the whole flow

                            # ══════════════════════════════════════════════════════
                            # STANDARD COGNITIVE PIPELINE (Fallback)
                            # No hardwired pathway matched. Fall through to the
                            # Sovereign Scanner → IntentRouter → StateMachine chain.

                            # v42.1 FIX: Use a more natural fallback message if 'message' is missing
                            # this prevents robotic "I have executed the speak protocol"
                            fallback_msg = f"I've updated my internal state regarding {pathway.skill_name}." if "speak" not in pathway.skill_name else ""
                            if isinstance(rich_res, dict):
                                final_response = (
                                    rich_res.get("message")
                                    or rich_res.get("summary")
                                    or rich_res.get("content")
                                    or rich_res.get("error")
                                    or fallback_msg
                                )
                            else:
                                final_response = str(rich_res or fallback_msg)

                            # If we have no message at all (e.g. passive skill without a report), don't emit anything
                            if not final_response:
                                return

                            # Finalize via standardized pipe
                            final_response = await self._finalize_response(
                                message, final_response, origin, trace, successful_tools
                            )

                            if origin in ("user", "voice", "admin"):
                                await self.output_gate.emit(final_response, origin=origin, target="primary")
                            return

                        # ══════════════════════════════════════════════════════
                        # STANDARD COGNITIVE PIPELINE (Fallback)
                        # No hardwired pathway matched. Fall through to the
                        # Sovereign Scanner → IntentRouter → StateMachine chain.
                        # ══════════════════════════════════════════════════════

                        # ══════════════════════════════════════════════════════
                        # PHASE 7: COGNITIVE INVERSION PIPELINE
                        # ══════════════════════════════════════════════════════
                        # ══════════════════════════════════════════════════════
                        # PHASE XXII: REACT LOOP INTEGRATION (Reasoning)
                        # ══════════════════════════════════════════════════════
                        react_engaged = False
                        if hasattr(self, 'react_loop') and self.react_loop and origin in ("user", "admin") and not config.skeletal_mode:
                            # Heuristic for complex reasoning: longer queries, factual/lookup questions,
                            # or any signals that Aura would need external knowledge to answer correctly.
                            _factual_triggers = [
                                "search", "find", "why", "how", "analyze", "deep", "autonomy",
                                "look up", "lookup", "what is", "what are", "who is", "who are",
                                "when did", "when was", "where is", "where are", "tell me about",
                                "explain", "define", "calculate", "equation", "percentage", "probability",
                                "latest", "recent", "current", "news", "today", "statistic", "data",
                                "research", "study", "evidence", "source", "fact", "according to",
                                "drake equation", "scientific", "formula", "rate of", "estimate",
                            ]
                            is_complex = (
                                len(message.split()) > 8
                                or "?" in message
                                or any(t in message.lower() for t in _factual_triggers)
                            )
                            if is_complex:
                                self._emit_thought_stream("🧠 Engaging ReAct reasoning loop...")

                                # H-52: "Cognitive Bridge" - immediate user reassurance for complex logic
                                if origin in ("user", "voice", "admin") and not getattr(self, "_reflex_sent_for_current", False):
                                    await self.output_gate.emit("I'll think about that for a moment. This requires a bit of reasoning.", origin=origin, target="primary")
                                    self._reflex_sent_for_current = True

                                try:
                                    # H-52: Proactive Heartbeat for slow reasoning (Claude feedback)
                                    stream_active = True
                                    async def _reasoning_heartbeat():
                                        while stream_active:
                                            await asyncio.sleep(25)
                                            if stream_active:
                                                self._emit_thought_stream("⏳ Still thinking... exploring deep neural pathways.")
                                                if origin in ("user", "voice", "admin"):
                                                    await self.output_gate.emit("I'm still processing your request. My logic is deep, but I'm with you.", origin=origin, target="primary")

                                    heartbeat_task = asyncio.create_task(_reasoning_heartbeat())
                                    priority = origin in ("user", "voice", "admin")
                                    try:
                                        async for event in self.react_loop.run_stream(message, priority=priority):
                                            if event["type"] == "thought":
                                                self._emit_thought_stream(f"💭 {event['content']}")
                                            elif event["type"] == "action":
                                                self._emit_thought_stream(f"🛠️ Executing: {event['action']}")
                                            elif event["type"] == "observation":
                                                logger.debug("ReAct Observation: %s", event.get("content", "")[:50])
                                            elif event["type"] == "final":
                                                final_response = event["content"]
                                                trace_obj = event.get("trace")
                                                successful_tools = []
                                                react_engaged = True
                                                logger.info("✅ ReActLoop reasoning completed.")
                                            elif event["type"] == "error":
                                                self._emit_thought_stream(f"⚠️ Reasoning friction: {event['content']}")
                                                # Zenith-H52: If severe friction, break loop to trigger immediate fallback
                                                if any(term in event['content'].lower() for term in ["failed", "refused", "timeout", "critical", "error"]):
                                                    logger.warning("ReAct error detected, triggering early termination.")
                                                    break
                                    finally:
                                        stream_active = False
                                        heartbeat_task.cancel()
                                        try:
                                            await heartbeat_task
                                        except asyncio.CancelledError:
                                            logger.debug("Direct match result handled.")

                                except Exception as e:
                                    logger.error("ReActLoop failed: %s", e)
                                    # Ensure we don't leave the user hanging if the loop crashes
                                    final_response = None

                        if self.kernel_interface and not config.skeletal_mode and origin != "kernel":
                            self._emit_thought_stream("🧠 [ZENITH] Unitary Kernel Pipeline engaged...")
                            # ══════════════════════════════════════════════════════
                            # UNITARY KERNEL BYPASS (Phase 10 Hardening)
                            # ══════════════════════════════════════════════════════
                            # Capture final response directly from process()
                            final_response = await self.kernel_interface.process(
                                message,
                                origin=origin,
                                priority=origin in ("user", "voice", "admin"),
                            )
                            successful_tools = []
                        elif origin == "kernel":
                            logger.debug("🧠 Legacy bridge request detected; bypassing KernelInterface to avoid recursive re-entry.")
                            if (cog := ServiceContainer.get("cognitive_integration", default=None)) and cog.is_active:
                                self._emit_thought_stream("🧠 Phase 7: Cognitive Inversion Pipeline engaged...")
                                final_response = await cog.process_turn(message, payload_context)
                                successful_tools = []
                            else:
                                logger.warning("Kernel-origin request arrived while CognitiveIntegration is offline; suppressing recursive fallback.")
                                final_response = None
                        elif (cog := ServiceContainer.get("cognitive_integration", default=None)) and cog.is_active:
                            self._emit_thought_stream("🧠 Phase 7: Cognitive Inversion Pipeline engaged...")
                            # The new pipeline handles classification, strategy, reasoning, and drafting
                            final_response = await cog.process_turn(message, payload_context)
                            successful_tools = [] # CognitiveIntegration manages tools internally
                        else:
                            # ══════════════════════════════════════════════════════
                            # LEGACY COGNITIVE PIPELINE (Fallback)
                            # Phase 50 FIX: When CIL is offline, use the LLM router
                            # directly instead of the broken state_machine pipeline
                            # which returns "I'm processing that..." default string.
                            # ══════════════════════════════════════════════════════
                            logger.warning("CognitiveIntegration offline - using direct LLM fallback")
                            record_degraded_event(
                                "cognitive_integration",
                                "offline_fallback",
                                detail="using direct_llm_fallback",
                                severity="warning",
                                classification="non_critical_fallback",
                                context={"origin": origin},
                            )

                            # Try CIL lazy re-initialization first
                            if cog and not cog.is_active:
                                try:
                                    await cog.initialize()
                                    if cog.is_active:
                                        logger.info("✅ CIL recovered via lazy re-init!")
                                        final_response = await cog.process_turn(message, payload_context)
                                        successful_tools = []
                                    else:
                                        final_response = None  # Fall through to direct LLM
                                except Exception as e:
                                    logger.warning("CIL lazy re-init failed: %s", e)
                                    record_degraded_event(
                                        "cognitive_integration",
                                        "lazy_reinit_failed",
                                        detail=f"{type(e).__name__}: {e}",
                                        severity="warning",
                                        classification="non_critical_fallback",
                                        context={"origin": origin},
                                        exc=e,
                                    )
                                    final_response = None
                            else:
                                final_response = None

                            # Governed StateMachine fallback if CIL still not available
                            if not final_response:
                                try:
                                    record_degraded_event(
                                        "cognitive_integration",
                                        "governed_state_machine_fallback",
                                        detail="state_machine_execute",
                                        severity="warning",
                                        classification="non_critical_fallback",
                                        context={"origin": origin},
                                    )
                                    intent = await self.intent_router.classify(message, payload_context)
                                    res = await self.state_machine.execute(
                                        intent,
                                        message,
                                        payload_context,
                                        priority=priority,
                                        origin=origin,
                                    )
                                    final_response = res[0] if isinstance(res, (tuple, list)) else res
                                    successful_tools = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else []
                                except Exception as e:
                                    logger.warning("Governed StateMachine fallback failed: %s", e)
                                    record_degraded_event(
                                        "cognitive_integration",
                                        "governed_state_machine_fallback_failed",
                                        detail=f"{type(e).__name__}: {e}",
                                        severity="warning",
                                        classification="non_critical_fallback",
                                        context={"origin": origin},
                                        exc=e,
                                    )
                                    final_response = None

                            # Raw direct LLM fallback only as the final degraded lane
                            if not final_response:
                                final_response = await self._generate_fallback(message)

                        # Phase Transcendental: Standardized Finalization
                        self._record_message_in_history(message, origin)
                        final_response = await self._finalize_response(
                            message, final_response, origin, trace, successful_tools
                        )

                        # ══════════════════════════════════════════════════════
                        # KNOWLEDGE GAP DETECTION + AUTO WEB SEARCH
                        # If Aura's response signals ignorance of a fact, she
                        # searches the web and replaces the hallucinated answer
                        # with grounded information before speaking.
                        # ══════════════════════════════════════════════════════
                        if origin in ("user", "voice", "admin") and final_response:
                            _gap_markers = [
                                "i don't have", "i don't know", "i'm not sure",
                                "i cannot access", "my training", "knowledge cutoff",
                                "unable to access the internet", "i cannot browse",
                                "i'm not certain", "i don't have access",
                                "i can't look that up", "i cannot search",
                                "beyond my knowledge", "i lack real-time",
                                "without internet access", "i'm unable to verify",
                                "my information may be outdated", "i can't provide real-time",
                                "i have no information about", "not in my training",
                            ]
                            _resp_lower = final_response.lower()
                            _has_gap = any(m in _resp_lower for m in _gap_markers)
                            if _has_gap and hasattr(self, 'agency') and self.agency:
                                logger.info("🔍 [KNOWLEDGE GAP] Uncertainty in response — auto-searching: %s", message[:80])
                                try:
                                    _search_result = await asyncio.wait_for(
                                        self.execute_tool(
                                            "sovereign_browser",
                                            {"query": message.strip(), "mode": "search", "deep": False},
                                            origin="knowledge_gap_auto_search",
                                            payload_context=payload_context,
                                        ),
                                        timeout=25.0
                                    )
                                    if isinstance(_search_result, dict) and _search_result.get("ok") and _search_result.get("message"):
                                        _snippet = _search_result["message"]
                                        _src = _search_result.get("source", "web")
                                        final_response = f"{_snippet}\n\n*(sourced from web: {_src})*"
                                        logger.info("✅ [KNOWLEDGE GAP] Response grounded via web search.")
                                    else:
                                        logger.warning("🔍 [KNOWLEDGE GAP] Web search returned no usable result.")
                                except asyncio.TimeoutError:
                                    logger.warning("🔍 [KNOWLEDGE GAP] Web search timed out (25s).")
                                except Exception as _gap_err:
                                    logger.error("🔍 [KNOWLEDGE GAP] Web search failed: %s", _gap_err)

                        if origin in ("user", "voice", "admin"):
                            if self.output_gate:
                                await self.output_gate.emit(
                                    final_response,
                                    origin=origin,
                                    target="primary",
                                    metadata={"voice": True}
                                )
                        else:
                            # Spontaneous emission from internal thought or perception
                            await self.emit_spontaneous_message(final_response, modality="both", origin=origin)

                        # : Final telemetry pulse BEFORE marking as idle.
                        # This ensures the frontend receives the message before the typing indicator hides.
                        await asyncio.sleep(0.05)
                        self._publish_telemetry({"type": "status", "is_processing": False, "is_idle": True})

                        # Reset processing tracking (Hardening)
                        self.status.is_processing = False

                        return final_response
                    except Exception as e:
                        logger.error("State machine execution failed: %s", e)
                        if self._autonomy_guardian and hasattr(self._autonomy_guardian, 'ensure_delivery'):
                            self._autonomy_guardian.ensure_delivery(
                                f"I encountered an internal error: {str(e)[:100]}", origin
                            )
                        raise  # Mycelium will catch this and trigger bypass if critical
                    finally:
                        self.status.is_processing = False

            self._current_task_is_autonomous = origin not in ("user", "voice", "admin")  # v47

            # v14.1 HARDENING: Thinking Watchdog (120s)
            # Wrap the task in a wait_for to prevent infinite hangs.
            async def _watchdog_wrapper():
                try:
                    return await asyncio.wait_for(_execute_and_reply(), timeout=300.0)
                except asyncio.TimeoutError:
                    logger.error("🛑 [WATCHDOG] Thinking task exceeded 300s limit. Force terminating.")
                    await self._handle_thinking_timeout(origin)
                    return "Cognitive process timed out."
                except Exception as e:
                    logger.error("❌ [WATCHDOG] Thinking task failed: %s", e)
                    # GROK ZENITH HARDENING: Ensure the user is notified if cognition fails entirely.
                    if origin in ("user", "voice", "admin"):
                        error_msg = f"My cognitive process encountered a fatal interruption: {str(e)[:100]}. I am recovering my state."
                        if hasattr(self, 'output_gate') and self.output_gate:
                             await self.output_gate.emit(error_msg, origin=origin, target="primary")
                        self.status.is_processing = False
                    return f"Cognitive failure: {str(e)}"

            from core.utils.task_tracker import get_task_tracker
            task = get_task_tracker().track_task(asyncio.create_task(_watchdog_wrapper()))
            self._current_thought_task = task

            # Await inline — guard against cross-loop Future errors that can occur when
            # Starlette middleware or a background context holds a reference to this task.
            try:
                return await task
            except RuntimeError as _loop_err:
                if "attached to a different loop" in str(_loop_err):
                    logger.warning(
                        "⚠️ [Watchdog] Cross-loop Future detected (Starlette/uvicorn isolation). "
                        "Task still scheduled — result: %s",
                        "done" if task.done() else "pending",
                    )
                    if task.done() and not task.cancelled():
                        return task.result()
                    return None
                raise

        except Exception as e:
            logger.error("Error in handle_incoming_message: %s", e)
            self.status.is_processing = False
        finally:
            self.status.is_processing = False
