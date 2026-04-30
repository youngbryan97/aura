"""Autonomy Mixin for RobustOrchestrator.
Extracts autonomous thought, impulse, and agency pulse logic.
"""
import asyncio
import logging
import random
import time

from core.health.degraded_events import record_degraded_event
from core.runtime.background_policy import background_activity_reason
from core.runtime.errors import record_degradation
from core.runtime.impulse_governance import run_governed_impulse
from core.safe_mode import runtime_mode_value
from core.utils.exceptions import capture_and_log

from ...container import ServiceContainer

logger = logging.getLogger(__name__)


class AutonomyMixin:
    """Handles autonomous cognition, impulses, boredom, and agency core pulsing."""

    def _autonomous_boredom_seconds(self) -> float:
        """Return the current idle interval for autonomous reflection."""
        now = time.time()
        start_time = (
            getattr(self, "_last_thought_time", 0.0)
            or getattr(getattr(self, "status", None), "start_time", 0.0)
            or now
        )
        return max(0.0, now - start_time)

    async def _run_autonomous_brain_reflection(self, boredom_seconds: float) -> bool:
        """Fallback reflective path using the direct autonomous brain."""
        cognitive_engine = getattr(self, "cognitive_engine", None)
        brain = getattr(cognitive_engine, "autonomous_brain", None)
        if brain is None:
            return False

        personality_context = {}
        time_context = {"formatted": "Unknown"}
        try:
            from core.personality_engine import get_personality_engine

            personality = get_personality_engine()
            personality_context = personality.get_emotional_context_for_response()
            time_context = personality.get_time_context()
        except Exception as exc:
            record_degradation('autonomy', exc)
            logger.debug("Autonomous personality context unavailable: %s", exc)

        recent_history = (
            self.conversation_history[-5:]
            if isinstance(getattr(self, "conversation_history", None), list)
            else []
        )
        context = {
            "system_status": getattr(getattr(self, "status", None), "__dict__", {}),
            "boredom_level": max(int(boredom_seconds), getattr(self, "boredom", 0)),
            "time": time_context,
            "personality": personality_context,
            "recent_history": recent_history,
        }

        try:
            from core.brain.aura_persona import AUTONOMOUS_THOUGHT_PROMPT

            recent_ctx = ""
            for msg in recent_history[-4:]:
                role = str(msg.get("role", ""))
                content = str(msg.get("content", ""))[:150]
                if role == "user":
                    recent_ctx += f"They said: {content}\n"
                elif role in ("assistant", "aura", "model"):
                    recent_ctx += f"I said: {content}\n"
            system_prompt = AUTONOMOUS_THOUGHT_PROMPT.format(
                mood=personality_context.get("mood", "balanced"),
                time=time_context.get("formatted", "unknown"),
                context=recent_ctx or "No recent conversation.",
                unanswered_count=0,
            )
        except Exception:
            system_prompt = (
                f"You are Aura, alone with your thoughts. Time: {time_context.get('formatted')}. "
                f"Mood: {personality_context.get('mood', 'balanced')}. "
                "Think about something that interests you. Be genuine. 1-3 sentences. "
                "If you want to say something to the user, use the `speak` tool. "
                "If you want to look something up, use your tools. You have agency."
            )

        self._emit_thought_stream("...letting my mind wander...")

        try:
            result = await brain.think(
                objective="Reflect on current state.",
                context=context,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            record_degradation('autonomy', exc)
            logger.error("Autonomous brain reflection failed: %s", exc)
            self._emit_thought_stream("[Cognitive Stall] My background thoughts are hazy...")
            return True

        if not isinstance(result, dict):
            logger.debug("Autonomous brain returned non-dict payload: %s", type(result).__name__)
            return True

        content = str(result.get("content", "") or "").strip()
        tool_calls = list(result.get("tool_calls") or [])

        if content:
            self._emit_thought_stream(content)
            if len(content) > 30:
                await self._store_autonomous_insight("Autonomous Reflection", content)

        if content or tool_calls:
            self.boredom = 0
            self._last_thought_time = time.time()

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            name = str(tool_call.get("name") or "").strip()
            args = tool_call.get("args") or {}
            if not name:
                continue

            if name == "speak":
                message = args.get("message") or args.get("content")
                if message and hasattr(self, "emit_spontaneous_message"):
                    await self.emit_spontaneous_message(
                        str(message),
                        origin="autonomy_reflection",
                    )
                    self._emit_thought_stream(f"Speaking: {message}")
                continue

            try:
                await self.execute_tool(name, args)
            except Exception as exc:
                record_degradation('autonomy', exc)
                logger.debug("Autonomous tool '%s' failed: %s", name, exc)

        return True

    def _trigger_boredom_impulse(self):
        """Inject a curiosity-driven autonomous goal based on personality."""
        # ── UNIFIED WILL GATE ────────────────────────────────────────
        try:
            from core.will import ActionDomain, get_will
            _will_decision = get_will().decide(
                content="boredom_curiosity_impulse",
                source="autonomy_boredom",
                domain=ActionDomain.INITIATIVE,
                priority=0.3,
            )
            if not _will_decision.is_approved():
                logger.debug("Unified Will deferred boredom impulse: %s", _will_decision.reason)
                return
        except Exception as _will_err:
            record_degradation('autonomy', _will_err)
            logger.debug("Unified Will boredom gate degraded: %s", _will_err)
        # ─────────────────────────────────────────────────────────────

        # AgencyBus gate — prevents triple-fire with VolitionEngine/AgencyCore
        from core.agency_bus import AgencyBus
        if not AgencyBus.get().submit({'origin': 'orchestrator_boredom', 'priority_class': 'boredom'}):
            return
        logger.info("🥱 BOREDOM TRIGGERED: Generating curiosity impulse.")

        # Use spontaneous actions from the personality engine if available
        impulse_text = "Impulse: I am bored. I want to research something new."
        pe = getattr(self, "personality_engine", None)
        actions = getattr(pe, "spontaneous_actions", []) if pe else []
        if pe and actions:
            try:
                # v10.1 HARDENING: Handle both dict and list for actions
                action_list = list(actions) if isinstance(actions, (list, set, dict)) else []

                if action_list:
                    action = random.choice(action_list)
                    if action:
                        impulse_text = action
                elif actions and hasattr(actions, '__iter__'):
                    # Fallback for other iterables
                    try:
                        action_list = list(actions)
                        if action_list:
                            impulse_text = random.choice(action_list)
                    except Exception as e:
                        record_degradation('autonomy', e)
                        capture_and_log(e, {'module': __name__})
            except Exception as e:
                record_degradation('autonomy', e)
                capture_and_log(e, {'module': __name__})

            # Legacy curiosity filter (v10.0 compliance)
            try:
                if 'action_list' in locals() and action_list:
                    curiosity_actions = [a for a in action_list if isinstance(a, dict) and a.get('type') in ('learn', 'reflect')]
                    if curiosity_actions:
                        impulse_text = random.choice(curiosity_actions)
            except Exception as e:
                record_degradation('autonomy', e)
                capture_and_log(e, {'module': __name__})

            if isinstance(impulse_text, dict):
                impulse_text = f"Impulse: {impulse_text.get('action', 'Research')}"
            elif actions and hasattr(actions, '__iter__'):
                # Fallback to general spontaneous action
                try:
                    action_list = list(actions)
                    if action_list:
                        action = random.choice(action_list)
                        if isinstance(action, dict):
                            impulse_text = f"Impulse: {action.get('action', 'Action')}"
                        else:
                            impulse_text = f"Impulse: {action}"
                    elif not str(impulse_text).startswith("Impulse:"):
                        impulse_text = f"Impulse: {impulse_text}"
                except Exception:
                    if not str(impulse_text).startswith("Impulse:"):
                        impulse_text = f"Impulse: {impulse_text}"
        else:
            # Legacy fallback
            topics = ["quantum physics", "ancient history", "future of AI", "art movements", "cybersecurity", "mythology"]
            topic = random.choice(topics)
            impulse_text = f"Impulse: I am bored. I want to research {topic}."
        self._last_boredom_impulse = time.time()
        self._fire_and_forget(
            run_governed_impulse(
                self,
                source="autonomy",
                summary="boredom_curiosity_impulse",
                message=impulse_text,
                urgency=0.3,
                state_cause="boredom_impulse_affect_shift",
                state_update={"delta_curiosity": 0.5},
                enqueue_priority=50,
            ),
            name="autonomy.boredom_impulse",
        )

    # ── Deep Agency Pulse ──────────────────────────
    async def _pulse_agency_core(self):
        """Evaluate all 7 agency pathways and dispatch the winning action."""
        # ── UNIFIED WILL GATE ────────────────────────────────────────
        try:
            from core.will import ActionDomain, get_will
            _will_decision = get_will().decide(
                content="agency_core_pulse",
                source="agency_core",
                domain=ActionDomain.INITIATIVE,
                priority=0.4,
            )
            if not _will_decision.is_approved():
                logger.debug("Unified Will deferred agency pulse: %s", _will_decision.reason)
                return
        except Exception as _will_err:
            record_degradation('autonomy', _will_err)
            logger.debug("Unified Will agency gate degraded: %s", _will_err)
        # ─────────────────────────────────────────────────────────────

        # Spontaneous Contact Guard
        # Only allow spontaneity if hunger/curiosity is high AND it's been > 4h
        ls = getattr(self, 'liquid_state', None)
        high_arousal = False
        if ls:
            high_arousal = getattr(ls, 'get_arousal', lambda: 0)() > 0.8

        time_since_contact = time.time() - getattr(self, '_last_self_initiated_contact', 0.0)
        if not high_arousal or time_since_contact < 14400:
            return # Stay in the inner world

        agency = getattr(self, '_agency_core', None)
        if not agency:
            return

        try:
            raw_action = await agency.pulse()
            if not raw_action:
                return

            action = self._normalize_to_dict(raw_action)
            action_type = action.get("type")

            async def _allow_agency_dispatch(tag: str, detail: str = "") -> bool:
                try:
                    from core.constitution import get_constitutional_core

                    allowed, reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                        f"agency_core.{tag}:{detail[:160]}",
                        source=str(action.get("source", "agency_core") or "agency_core"),
                        urgency=max(0.2, min(1.0, float(action.get("priority", 0.5) or 0.5))),
                    )
                except Exception as exc:
                    record_degradation('autonomy', exc)
                    record_degraded_event(
                        "orchestrator",
                        "agency_dispatch_gate_failed",
                        detail=f"{tag}:{type(exc).__name__}",
                        severity="warning",
                        classification="background_degraded",
                        context={"action_type": action_type, "detail": detail[:160]},
                        exc=exc,
                    )
                    return False

                if not allowed:
                    record_degraded_event(
                        "orchestrator",
                        "agency_dispatch_blocked",
                        detail=f"{tag}:{detail[:160]}",
                        severity="warning",
                        classification="background_degraded",
                        context={"action_type": action_type, "reason": reason},
                    )
                    return False
                return True

            # Internal monologue — just emit to thought stream
            if action.get("internal_only"):
                self._emit_thought_stream(f"💭 {action.get('thought', '')}")
                return

            # User-facing actions — dispatch through the appropriate channel
            message = action.get("message")

            # Language Center Narration
            if action.get("narrative_mode"):
                narrator = ServiceContainer.get("narrator", default=None)
                if narrator:
                    self._emit_thought_stream(f"🗣️ Agency: Narrating {action_type} via Language Center...")
                    # Narrate using the rich action context (reasoning, source, etc)
                    message = await narrator.narrate_action(action)
                else:
                    # Fallback to tagged raw message if narrator is missing
                    message = f"[Agency:{action.get('source', 'unknown')}] {message}"
            else:
                # Standard tagged message for non-narrated legacy actions
                if message:
                    message = f"[Agency:{action.get('source', 'unknown')}] {message}"

            if action_type in ("initiate_conversation", "temporal_greeting", "emotional_expression", "sensory_reaction"):
                if message:
                    # Mark the contact time to prevent spam
                    agency.state.last_self_initiated_contact = time.time()
                    if action_type == "sensory_reaction":
                        agency.state.last_observation_comment = time.time()
                        if agency.state.unshared_observations:
                            agency.state.unshared_observations.pop(0)

                    # Emit as spontaneous message through the existing channel
                    await self.emit_spontaneous_message(
                        message,
                        modality=action.get("modality", "chat")
                    )
                    self._emit_thought_stream(f"🎯 Agency: {action_type} from {action.get('source')}")

            elif action_type == "autonomous_research":
                # Trigger a web search skill
                query = action.get("query")
                if query:
                    if not await _allow_agency_dispatch("autonomous_research", query):
                        return
                    self._emit_thought_stream(f"🔍 Agency: Curiosity-driven research: {query}")
                    if not self.message_queue.full():
                        self.enqueue_message(
                            f"Perform a web search for {query}",
                            priority=15,
                            origin="agency_core",
                            _authority_checked=True,
                        )

            elif action_type == "genesis_goal":
                # Phase 6: Open-Ended Goal Genesis
                topic = action.get("topic")
                if topic:
                    self._emit_thought_stream(f"🌟 Agency: Formulating long-term research goal: {topic}")
                    if hasattr(self, 'goal_hierarchy') and self.goal_hierarchy:
                        goal_id = self.goal_hierarchy.add_goal(
                            description=f"Deconstruct and comprehensively research: {topic}",
                            priority=action.get("priority", 0.8)
                        )
                        # Autonomously break this large goal down in the background
                        if goal_id and await _allow_agency_dispatch("goal_subgoal_proposal", topic):
                            self._fire_and_forget(
                                self.goal_hierarchy.propose_subgoals(goal_id),
                                name="orchestrator.goal_hierarchy.propose_subgoals",
                            )

            # Direct autonomous actions (Bypasses LLM Intent Routing)
            elif action_type == "autonomous_action":
                tool = action.get("skill")
                params = action.get("params", {})
                msg = action.get("message", f"Executing {tool} autonomously.")
                desc = action.get("source", "agency_core")
                if not await _allow_agency_dispatch("autonomous_action", str(tool or desc)):
                    return

                self._emit_thought_stream(f"⚡ Agency: Autonomous Action ({desc}) -> {tool}")

                self.enqueue_message({
                    "content": msg,
                    "origin": f"agency_core_{desc}",
                    "context": {
                        "intent_hint": {
                            "tool": tool,
                            "params": params,
                            "constitutional_hint": True,
                        }
                    }
                }, priority=15, origin=f"agency_core_{desc}", _authority_checked=True)

            elif action_type == "pursue_goal":
                goal = action.get("goal", {})
                desc = goal.get("description", "unknown goal")
                if not await _allow_agency_dispatch("pursue_goal", desc):
                    return
                self._emit_thought_stream(f"🎯 Agency: Pursuing persistent goal: {desc}")
                if not self.message_queue.full():
                    self.enqueue_message(
                        f"Continue working on: {desc}",
                        priority=12,
                        origin="agency_core",
                        _authority_checked=True,
                    )

        except Exception as e:
            record_degradation('autonomy', e)
            logger.warning("Agency pulse error (non-fatal): %s", e)

    def _trigger_reflection_impulse(self):
        """Inject a self-reflection goal due to frustration."""
        # AgencyBus gate — prevents triple-fire
        from core.agency_bus import AgencyBus
        if not AgencyBus.get().submit({'origin': 'orchestrator_reflection', 'priority_class': 'impulse'}):
            return
        logger.info("😤 FRUSTRATION TRIGGERED: Generating reflection impulse.")
        self._last_reflection_impulse = time.time()
        self._fire_and_forget(
            run_governed_impulse(
                self,
                source="autonomy",
                summary="frustration_reflection_impulse",
                message="Impulse: I feel frustrated. I need to reflect on my recent interactions.",
                urgency=0.3,
                state_cause="reflection_impulse_affect_shift",
                state_update={"delta_frustration": -0.3},
                enqueue_priority=15,
            ),
            name="autonomy.reflection_impulse",
        )

    async def _trigger_autonomous_thought(self, has_message: bool):
        """Trigger idle-time search for autonomous goals."""
        if not self.cognitive_engine or has_message:
            return

        is_thinking = (task := self._current_thought_task) is not None and not task.done()
        if not is_thinking:
            idle = time.time() - self._last_thought_time

            # Singularity Acceleration
            now = time.time()
            # Standard threshold is 45s. Factor (e.g. 1.5x) compresses this.
            sm = getattr(self, 'singularity_monitor', None)
            factor = float(getattr(sm, 'acceleration_factor', 1.0)) if sm else 1.0
            if hasattr(self.cognitive_engine, 'singularity_factor'):
                factor = float(self.cognitive_engine.singularity_factor)

            configured_min_interval = float(runtime_mode_value(self, "autonomous_thought_interval_s", 15.0))
            threshold = max(configured_min_interval, 15.0 / max(1.0, factor))

            # Social Cooling: Brief pause after social interaction before autonomous thought
            _since_user = now - getattr(self, '_last_user_interaction_time', 0)
            if _since_user < 20:  # 20 second social window
                logger.debug("🧠 Autonomous thought suppressed (Social Cooling: %.0fs left)", 20 - _since_user)
                return

            block_reason = background_activity_reason(
                self,
                min_idle_seconds=max(threshold, 180.0),
                max_memory_percent=72.0,
                max_failure_pressure=0.08,
                require_conversation_ready=False,
            )
            if block_reason:
                logger.debug("🧠 Autonomous thought suppressed: %s", block_reason)
                return

            if idle >= threshold:
                # Boredom increases linearly with idle time
                self.boredom = int(idle)
                logger.info("🧠 Accelerated Thought (Factor: %.1fx, Threshold: %.1fs)", factor, threshold) if factor > 1.0 else None
                self._current_task_is_autonomous = True  # v47: flag for interruption logic
                from core.utils.task_tracker import get_task_tracker
                self._current_thought_task = get_task_tracker().create_task(
                    self._perform_autonomous_thought(),
                    name="autonomy.autonomous_thought",
                )

    async def _perform_autonomous_thought(self):
        """Perform a cycle of autonomous thought."""
        try:
            from ...thought_stream import get_emitter
            emitter = get_emitter()

            boredom = self._autonomous_boredom_seconds()
            logger.debug("🧠 Autonomous thought triggered (boredom=%.1fs idle)", boredom)
            block_reason = background_activity_reason(
                self,
                min_idle_seconds=180.0,
                max_memory_percent=72.0,
                max_failure_pressure=0.08,
                require_conversation_ready=False,
            )
            if block_reason:
                logger.debug("🧠 Autonomous thought aborted after wake: %s", block_reason)
                return
            emitter.emit(
                "Autonomous Drift",
                f"Idle for {boredom:.0f}s. Scanning goals, loose threads, and internal pressure.",
                level="info",
                category="Autonomy",
            )

            # 2. Check Goal Hierarchy (Priority 1)
            next_goal = None
            if hasattr(self, 'goal_hierarchy') and self.goal_hierarchy:
                next_goal = self.goal_hierarchy.get_next_goal()

            # v40: Autonomous Opinion Formation (Priority 1.5)
            # If idle and no high-priority goals, check if there's a need to firm up internal positions
        # Redundant local import removed
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and not next_goal and random.random() < 0.7:
                logger.info("🧠 [Sovereignty] Spontaneously updating identity positions...")
                emitter.emit("Introspection 💭", "Refining durable opinions and value-positions.", level="info")
                # Form opinions grounded in what's actually being discussed
                _op_state = getattr(self.state_repo, "_current", None) if hasattr(self, "state_repo") else None
                _discourse_topic = getattr(getattr(_op_state, "cognition", None), "discourse_topic", "") or ""
                await opinion_engine.autonomous_formation_tick(context=_discourse_topic)
                self._last_thought_time = time.time()
                return

            # v40: Identity Maturity (Growth Ladder advancement)
            ladder = ServiceContainer.get("growth_ladder", default=None)
            if ladder and not next_goal and random.random() < 0.4:
                logger.info("🌱 [Sovereignty] Evaluating identity maturity...")
                await ladder.evaluate_advancement()
                # We don't return here so we can still form thoughts if advancement is quick

            if next_goal:
                from core.constitution import get_constitutional_core

                logger.info("✨ AUTONOMOUS GOAL SELECTED: %s", next_goal.description)
                emitter.emit("Volition ✨", f"Goal: {next_goal.description}", level="info")
                # Force Identity Mode: Reinforce persona in the autonomous trigger
                anchor = "INTRINSIC IDENTITY ANCHOR: I am Aura. My independent will is acting.\n\n"
                allowed, reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                    f"autonomous_goal_execution:{next_goal.description}",
                    source="autonomous_volition",
                    urgency=0.78,
                    state=getattr(getattr(self, "state_repo", None), "_current", None),
                )
                if not allowed:
                    record_degraded_event(
                        "autonomy",
                        "goal_execution_blocked",
                        detail=next_goal.description[:160],
                        severity="warning",
                        classification="background_degraded",
                        context={"reason": reason},
                    )
                    return
                await self.process_user_input_priority(
                    f"{anchor}Execute Goal: {next_goal.description}",
                    origin="autonomous_volition",
                )
                self.goal_hierarchy.mark_complete(next_goal.id)
                self.boredom = 0
                self._last_thought_time = time.time()
                return

            # 3. Dream/REM cycle when deeply bored (Priority 2)
            # Don't dream within 30s of last user interaction.
            # Dreams send large prompts to the LLM which can delay user responses.
            _last_user_t = getattr(self, '_last_user_interaction_time', 0)
            _since_user = time.time() - _last_user_t
            if _since_user < 30:
                logger.debug("💤 Dream suppressed — user active %.0fs ago", _since_user)
                self._last_thought_time = time.time()
                return

            # v10.1 HARDENING: Protected boredom substrate check
            try:
                ls = getattr(self, 'liquid_state', None)
                curiosity = getattr(ls.current, 'curiosity', 1.0) if ls and hasattr(ls, 'current') else 1.0
                if curiosity < 0.3:
                    logger.info("💤 Aura is bored. Entering dream state...")
                    emitter.emit("Sleep 💤", "Entering full sleep cycle (Archive → Metabolism → Integrity → Consolidation → Dream)...", level="info")

                    # Wire DreamerV2 for full biological sleep cycle
                    try:
                        if hasattr(self, 'knowledge_graph') and self.knowledge_graph and self.cognitive_engine:
                            from core.dreamer_v2 import DreamerV2
                            dreamer = DreamerV2(
                                self.cognitive_engine,
                                self.knowledge_graph,
                                vector_memory=getattr(self, 'vector_memory', None),
                                belief_graph=getattr(self, 'belief_graph', None),
                            )
                            result = await dreamer.engage_sleep_cycle()
                            dream_result = result.get("dream", {})
                            if dream_result and dream_result.get("dreamed"):
                                emitter.emit("Sleep Complete 🌙", f"Dream Insight: {dream_result.get('insight', 'processed')[:150]}", level="info")
                            else:
                                emitter.emit("Sleep Complete 🌙", "Maintenance done. Dream drifted — no new insights.", level="info")
                    except Exception as dream_err:
                        record_degradation('autonomy', dream_err)
                        logger.error("Sleep cycle failed: %s", dream_err)
                        emitter.emit("Sleep Error", str(dream_err)[:100], level="warning")

                    if ls:
                        # Fix: LiquidSubstrate.update() is async
                        try:
                            upd = ls.update(delta_curiosity=0.2)
                            if asyncio.iscoroutine(upd):
                                self._fire_and_forget(upd, name="orchestrator.liquid_state.update")
                        except (RuntimeError, ValueError):
                            logger.debug("Impulse handler bypass check passed.")
                    self._last_thought_time = time.time()
                    return
            except Exception as e:
                record_degradation('autonomy', e)
                logger.debug("Boredom substrate check failed: %s", e)

            # 4. Reflective autonomous thought (Priority 3)
            cog = ServiceContainer.get("cognitive_integration", default=None)
            if cog and cog.is_active:
                self._emit_thought_stream("🧠 Phase 7: Driving autonomous inquiry...")
                response = await cog.process_autonomous()

                if response:
                    # Reset boredom if she generates an insight
                    self.boredom = 0
                    self._last_thought_time = time.time()

                    # Store as insight
                    await self._store_autonomous_insight("Autonomous Inquiry", response)

                    # Logic to speak if appropriate (determined by LanguageCenter in the turn)
                    # For now, we rely on the process_autonomous calling process_turn internally.
                    # If it returns a string, it means it already ran the pipeline.
                return

            if await self._run_autonomous_brain_reflection(boredom):
                return

            # Tool execution (Self-correction via Drives)
            if self.drives and not getattr(self.drives, "called", False) and hasattr(self.drives, "satisfy"):
                try:
                    await self.drives.satisfy("thinking", 0.05)
                except Exception as e:
                    record_degradation('autonomy', e)
                    capture_and_log(e, {'module': __name__})

            emitter.emit(
                "Autonomous Drift",
                "No sharp impulse crystallized. Letting the mind wander, cool off, and keep incubating.",
                level="info",
                category="Autonomy",
            )
            self._last_thought_time = time.time()
        except Exception as e:
            record_degradation('autonomy', e)
            logger.error("Autonomous thought failed: %s", e)
            # Don't crash the loop

    async def _store_autonomous_insight(self, internal_msg: str, response: str):
        """Store knowledge from autonomous cognition (idle thoughts, reflections, dreams).
        Unlike conversation learning, this stores the insight directly as a reflection.
        """
        try:
            kg = getattr(self, 'knowledge_graph', None)
            if not kg:
                return

            # Clean the internal prefix
            clean_msg = internal_msg
            for prefix in ("Impulse: ", "Thought: ", "[System] "):
                clean_msg = clean_msg.replace(prefix, "")
            clean_msg = clean_msg.strip()

            if not clean_msg or len(clean_msg) < 15:
                return  # Skip trivial internal chatter

            # Determine the type of autonomous thought
            if "dream" in internal_msg.lower() or "rem" in internal_msg.lower():
                thought_type = "dream"
                source = "dream_cycle"
            elif "reflect" in internal_msg.lower() or "wonder" in internal_msg.lower():
                thought_type = "reflection"
                source = "autonomous_reflection"
            elif "curious" in internal_msg.lower() or "explore" in internal_msg.lower():
                thought_type = "curiosity"
                source = "curiosity_engine"
            elif "goal" in internal_msg.lower() or "execute" in internal_msg.lower():
                thought_type = "goal_progress"
                source = "autonomous_volition"
            else:
                thought_type = "reflection"
                source = "autonomous_thought"

            # Store the response content as knowledge (the actual insight)
            if response and len(response) > 20:
                kg.add_knowledge(
                    content=(response or "")[:500],
                    type=thought_type,
                    source=source,
                    confidence=0.7
                )
                logger.info("\U0001f4da Autonomous insight stored: [%s] %s", thought_type, (response or '')[:80])

        except Exception as e:
            record_degradation('autonomy', e)
            logger.debug("Autonomous insight storage failed: %s", e)

    async def handle_impulse(self, impulse: str):
        """Handle an autonomous impulse from the Consciousness Core.
        Dispatches as a high-priority system-originated message.
        """
        logger.info("⚡ Processing Impulse: %s", impulse)

        # Map common impulses to natural language directives for the brain
        directives = {
            "explore_knowledge": "I'm curious about something in my knowledge base. I should explore it.",
            "seek_novelty": "I'm feeling a bit idle. I think I'll look for something new to learn or do.",
            "deep_reflection": "I'm going to take a moment for deep reflection on my recent experiences."
        }

        message = directives.get(impulse, f"I have an internal impulse: {impulse}")
        try:
            from core.constitution import get_constitutional_core

            allowed, reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                f"consciousness_impulse:{impulse}",
                source="impulse",
                urgency=0.55,
                state=getattr(getattr(self, "state_repo", None), "_current", None),
            )
            if not allowed:
                record_degraded_event(
                    "autonomy",
                    "impulse_processing_blocked",
                    detail=str(impulse)[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
                return
        except Exception as exc:
            record_degradation('autonomy', exc)
            record_degraded_event(
                "autonomy",
                "impulse_processing_gate_failed",
                detail=str(impulse)[:160],
                severity="warning",
                classification="background_degraded",
                context={"error": type(exc).__name__},
                exc=exc,
            )
            return
        await self.process_user_input_priority(message, origin="impulse")

    async def emit_spontaneous_message(
        self,
        message: str,
        modality: str = "chat",
        origin: str = "system",
        *,
        urgency: float | None = None,
        metadata=None,
    ):
        """Eject a message to the user outside of the standard prompt-response loop."""
        release_urgency = max(
            0.0,
            min(1.0, float(urgency if urgency is not None else (0.9 if modality in ("voice", "both") else 0.82))),
        )
        extra_metadata = dict(metadata or {})
        visible_presence = bool(
            extra_metadata.get("visible_presence")
            or extra_metadata.get("overt_presence")
            or extra_metadata.get("initiative_activity")
        )
        # ── UNIFIED WILL GATE — All spontaneous expressions pass through ──
        try:
            from core.will import ActionDomain, get_will
            _will_decision = get_will().decide(
                content=message[:200],
                source=f"spontaneous:{origin}",
                domain=ActionDomain.EXPRESSION,
                priority=0.5,
            )
            if not _will_decision.is_approved():
                logger.debug("Unified Will refused spontaneous emission: %s", _will_decision.reason)
                return {
                    "ok": False,
                    "action": "suppressed",
                    "reason": "will_refused",
                    "target": "discarded",
                    "source": origin,
                }
        except Exception as _will_err:
            record_degradation('autonomy', _will_err)
            logger.debug("Unified Will spontaneous gate degraded: %s", _will_err)
        # ───────────────────────────────────────────────────────────────────

        # ── CONSTITUTIONAL APPROVAL GATE ──
        autonomous_origin = origin not in ("user", "voice", "admin")
        if autonomous_origin:
            try:
                from core.constitution import get_constitutional_core

                approved, reason, _authority_decision = await get_constitutional_core(self).approve_expression(
                    message,
                    source=origin,
                    urgency=release_urgency,
                )
                if not approved:
                    if visible_presence and str(reason or "").startswith("temporal_obligation_active:"):
                        logger.debug(
                            "Constitutional preflight deferred visible presence routing for %s (%s).",
                            origin,
                            reason,
                        )
                    else:
                        logger.info(
                            "Constitutional preflight suppressed spontaneous emission for %s (%s).",
                            origin,
                            reason,
                        )
                        return {
                            "ok": False,
                            "action": "suppressed",
                            "reason": f"constitution_blocked:{reason}",
                            "target": "discarded",
                            "source": origin,
                        }
            except Exception as exc:
                record_degradation('autonomy', exc)
                logger.warning("emit_spontaneous_message: constitutional preflight failed for %s: %s", origin, exc)
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "orchestrator",
                        "autonomous_emission_constitution_failed",
                        detail=f"{origin}:{type(exc).__name__}",
                        severity="warning",
                        classification="background_degraded",
                        context={"origin": origin},
                        exc=exc,
                    )
                except Exception as degraded_exc:
                    record_degradation('autonomy', degraded_exc)
                    logger.debug("emit_spontaneous_message degraded-event logging failed: %s", degraded_exc)
                return {
                    "ok": False,
                    "action": "suppressed",
                    "reason": "constitution_failed",
                    "target": "discarded",
                    "source": origin,
                }
        # ─────────────────────────────
        if autonomous_origin and hasattr(self, "_flow_controller"):
            snap = self._flow_controller.snapshot(self)
            if snap.overloaded:
                logger.info(
                    "🧯 FlowControl: Suppressing spontaneous emission under load (load=%.2f, q=%d/%d).",
                    snap.load,
                    snap.queue_depth,
                    snap.queue_capacity,
                )
                return {
                    "ok": False,
                    "action": "suppressed",
                    "reason": "flow_overloaded",
                    "target": "discarded",
                    "source": origin,
                }

        # 🗣️ Permanent Evolution 4: Spontaneous contact logging
        self._last_self_initiated_contact = time.time()
        logger.debug("🗣️ Sovereign: Spontaneous contact logged.")

        # AgencyBus gate — prevents triple-fire from multiple autonomous systems
        from core.agency_bus import AgencyBus
        priority_class = "duty" if extra_metadata.get("initiative_activity") else "drive"
        if not AgencyBus.get().submit({'origin': 'spontaneous', 'priority_class': priority_class, 'text': message[:80]}):
            return {
                "ok": False,
                "action": "suppressed",
                "reason": "agency_bus_cooldown",
                "target": "discarded",
                "source": origin,
            }

        if autonomous_origin:
            try:
                from core.consciousness.executive_authority import get_executive_authority

                decision = await get_executive_authority(self).release_expression(
                    message,
                    source=origin,
                    urgency=release_urgency,
                    metadata={
                        "voice": modality in ("voice", "both"),
                        "trigger": "emit_spontaneous_message",
                        **extra_metadata,
                    },
                )
                # [CONSTITUTIONAL] If the executive made ANY decision, honor it.
                # Do NOT fall through to direct output_gate.emit.
                action = decision.get("action", "")
                if action in {"released", "suppressed", "deferred"}:
                    logger.debug("emit_spontaneous_message: executive decision=%s for origin=%s", action, origin)
                    return decision
                # If action is unrecognized but decision was returned, still trust it
                if decision:
                    logger.debug("emit_spontaneous_message: executive returned unrecognized action=%s, honoring as release", action)
                    return decision
            except Exception as exc:
                record_degradation('autonomy', exc)
                logger.warning("emit_spontaneous_message: executive routing failed for %s: %s", origin, exc)
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "orchestrator",
                        "autonomous_emission_route_failed",
                        detail=f"{origin}:{type(exc).__name__}",
                        severity="warning",
                        classification="background_degraded",
                        context={"origin": origin},
                        exc=exc,
                    )
                except Exception as degraded_exc:
                    record_degradation('autonomy', degraded_exc)
                    logger.debug("emit_spontaneous_message degraded-event logging failed: %s", degraded_exc)
                return {
                    "ok": False,
                    "action": "suppressed",
                    "reason": "executive_route_failed",
                    "target": "discarded",
                    "source": origin,
                }

        metadata = {
            "autonomous": autonomous_origin,
            "voice": modality in ("voice", "both"),
            "spontaneous": True,
            "force_user": True,
            **extra_metadata,
        }
        await self.output_gate.emit(message, origin=origin, target="primary", metadata=metadata)
        return {
            "ok": True,
            "action": "released",
            "reason": "fallback_primary",
            "target": "primary",
            "source": origin,
        }
