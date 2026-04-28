"""Cognitive Coordinator — agentic loop, tool execution, autonomous thought,
reasoning shortcuts, and knowledge extraction.

Extracted from orchestrator.py as part of the God Object decomposition.

Concern Boundaries (for future decomposition):
  - Response finalization & output filtering → finalize_response(), generate_fallback()
  - Constitutional safety → apply_constitutional_guard()
  - Tool execution → execute_tool()
  - Autonomous thought → perform_autonomous_thought(), generate_autonomous_thought()
  - Knowledge extraction → integrated throughout (extract during learning)
  - Social drive → update social model after interactions

All tool execution is gated through AuthorityGateway → UnifiedWill.
All autonomous actions produce WillReceipts via the authority chain.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

from core.runtime.governance_policy import allow_direct_user_shortcut
from core.runtime.turn_analysis import analyze_turn
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)


class CognitiveCoordinator:
    """Handles the agentic cognitive loop, tool execution, reasoning, and learning.

    Delegates focused concerns to extracted modules:
    - ToolExecutor: tool execution, forge, learning, ACG recording
    - KnowledgeExtractor: conversation learning, autonomous insights, name detection
    """

    def __init__(self, orch):
        self.orch = orch

        # Extracted focused modules
        from core.coordinators.tool_executor import ToolExecutor
        from core.coordinators.knowledge_extractor import KnowledgeExtractor

        self._tool_executor = ToolExecutor(orch)
        self._knowledge_extractor = KnowledgeExtractor(orch)

    # ------------------------------------------------------------------
    # Response Finalization & Output Filtering
    # ------------------------------------------------------------------

    async def finalize_response(self, message: str, response: str, origin: str, trace, successful_tools: List[str]) -> str:
        """Apply final touches: Fallback, Security, Social Drive, Meta-Learning."""
        from core.utils.task_tracker import task_tracker
        orch = self.orch
        if not response or response == "...":
            response = await self.generate_fallback(message)
        response = await self.apply_constitutional_guard(response)
        
        # ── Epistemic Humility Gate ──────────────────────────────────────────
        # Wire uncertainty.py into the actual response pipeline.
        # If Aura's confidence in a response is very low, she should say so
        # rather than confabulate. Epistemic honesty is a core value.
        if origin in ("user", "voice") and response:
            try:
                from core.uncertainty import EpistemicHumilityEngine
                # Lazy-init: one instance per orchestrator session
                if not hasattr(orch, '_epistemic_engine'):
                    orch._epistemic_engine = EpistemicHumilityEngine()
                
                engine = orch._epistemic_engine
                
                # Check if clarification is needed before responding
                should_clarify, clarifying_q = engine.should_ask_for_clarification(message)
                if should_clarify and len(response) > 200:
                    # Only interrupt if response is substantial — short answers don't need it
                    response = f"{response}\n\n*(Quick check: {clarifying_q})*"
                else:
                    # Apply epistemic caveats inline when confidence is low
                    enhanced = engine.apply_epistemic_humility(message, response)
                    if enhanced and enhanced != response:
                        response = enhanced
                        
            except Exception as e:
                record_degradation('cognitive_coordinator', e)
                record_degradation('cognitive_coordinator', e)
                logger.debug("Epistemic humility gate skipped: %s", e)
        # ── End Epistemic Humility Gate ──────────────────────────────────────
        if hasattr(orch, 'meta_learning') and orch.meta_learning and successful_tools:
            for t in successful_tools:
                orch.meta_learning.record_usage(t)
        if origin in ("user", "voice"):
            trace.record_step("meta_learning", {"tools_used": successful_tools})
        if hasattr(orch, 'social') and orch.social:
            try:
                orch.social.update_after_interaction(message, response)
            except Exception as _e:
                record_degradation('cognitive_coordinator', _e)
                record_degradation('cognitive_coordinator', _e)
                logger.error("Social model update failed: %s", _e)
        if origin in ("user", "voice") and hasattr(orch, 'self_modifier') and orch.self_modifier:
            orch.self_modifier.on_success(message, response, successful_tools)
        if hasattr(orch, 'affect_engine') and orch.affect_engine:
            try:
                orch.affect_engine.process_response(message, response)
            except Exception as e:
                record_degradation('cognitive_coordinator', e)
                record_degradation('cognitive_coordinator', e)
                logger.error("Affect processing failed: %s", e)
        if origin in ("user", "voice"):
            await asyncio.sleep(0.5)
            orch._trigger_background_reflection(response)
            orch._trigger_background_learning(message, response)
        if hasattr(orch, 'cognition') and orch.cognition:
             try:
                 orch.cognition.record_interaction(message, response, domain="general")
             except Exception as e:
                 record_degradation('cognitive_coordinator', e)
                 record_degradation('cognitive_coordinator', e)
                 logger.error("Failed to record interaction in cognitive layer: %s", e)
        trace.record_step("end", {"response": (response or "")[:100]})
        trace.save()
        orch._last_thought_time = time.time()
        if getattr(orch, 'drives', None):
            await orch.drives.satisfy("social", 10.0)
        if orch.reply_queue:
            should_broadcast = origin in ("user", "voice", "impulse")
            if should_broadcast:
                try:
                    orch.reply_queue.put_nowait(response)
                except asyncio.QueueFull:
                    import logging
                    logger.debug("Exception caught during execution", exc_info=True)
            else:
                logger.debug("🔇 Internal Response (Origin: %s) suppressed from UI.", origin)
        if origin == "voice" and orch.ears and hasattr(orch.ears, "_engine"):
            logger.info("🎙️ Origin was voice: Triggering TTS synthesis...")
            task_tracker.create_task(
                orch.ears._engine.synthesize_speech(response),
                name="cognitive_coordinator.voice_tts",
            )
        response = orch._filter_output(response)
        return response

    def check_reflexes(self, message: str) -> Optional[str]:
        """Personality-driven rapid-response triggers (Delegated to ReflexEngine)."""
        orch = self.orch
        if hasattr(orch, 'reflex_engine') and orch.reflex_engine:
            result = orch.reflex_engine.check(message)
            if result:
                return orch._filter_output(result)
        return None

    def filter_output(self, text: str) -> str:
        """Apply personality filter if available."""
        if not text:
            return text
        try:
            from core.brain.personality_engine import get_personality_engine
            pe = get_personality_engine()
            return pe.filter_response(text)
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.debug("Output filter failed: %s", e)
            return text

    # ------------------------------------------------------------------
    # Skill Shortcuts & Fast Path
    # ------------------------------------------------------------------

    async def check_direct_skill_shortcut(self, message: str, origin: str) -> Optional[Dict[str, Any]]:
        """Identify and execute skills that don't need LLM reasoning."""
        if origin != "user":
            return None
        if not allow_direct_user_shortcut(origin):
            logger.info("🧭 Direct skill shortcut yielded to governed tool path for user-facing turn")
            return None
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["search the web", "look up", "google", "find out about"]):
            return await self._execute_direct_search(message)
        if any(kw in msg_lower for kw in ["run a self-diag", "diagnose yourself", "system check"]):
            return await self._execute_direct_diag()
        return None

    async def _execute_direct_search(self, message: str):
        orch = self.orch
        query_match = re.search(r"(?:search (?:the web )?for|look up|google)\s+(.+)", message, re.IGNORECASE)
        query = query_match.group(1).strip().strip("'\"") if query_match else message
        logger.info("🔍 DIRECT SKILL: web_search('%s')", query)
        try:
            orch._emit_telemetry("Skill: web_search 🔧", f"Searching: {query}")
            return await orch.execute_tool("web_search", {"query": query})
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.debug("Direct search failed: %s", e)
            return None

    async def _execute_direct_diag(self):
        orch = self.orch
        logger.info("🔍 DIRECT SKILL: self_diagnosis")
        try:
            orch._emit_telemetry("Skill: self_diagnosis 🔧", "Running diagnostics...")
            return await orch.execute_tool("self_diagnosis", {})
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.debug("Direct diag failed: %s", e)
            return None

    async def attempt_fast_path(self, message: str, origin: str, shortcut_result: Optional[Dict]) -> Optional[str]:
        """Try to generate a response without full agentic overhead."""
        orch = self.orch
        is_simple = self.is_simple_conversational(message, origin, bool(shortcut_result))
        if not is_simple:
            return None
        logger.info("🏎️ FAST-PATH: Bypassing Agentic Loop.")
        from core.brain.cognitive_engine import ThinkingMode
        context = orch._get_cleaned_history_context(10)
        p_ctx = orch._get_personality_context()
        if p_ctx:
            context["personality"] = p_ctx
        if shortcut_result:
            message = orch._inject_shortcut_results(message, shortcut_result)
            context["skill_result"] = str(shortcut_result.get("summary", ""))
        thought = await orch.cognitive_engine.think(
            objective=message,
            context=context,
            mode=ThinkingMode.FAST,
            origin=origin,
        )
        if not thought or not hasattr(thought, 'content'):
            logger.error("Cognitive engine returned None or invalid thought in fast-path.")
            return "I apologize, but my internal stream momentarily stalled. I am still here."
        response = orch._filter_output(orch._post_process_response(thought.content))
        role = getattr(orch, "AI_ROLE", "assistant")
        if isinstance(orch.conversation_history, list):
            orch.conversation_history.append({"role": role, "content": response})
        else:
            orch.conversation_history = [{"role": role, "content": response}]
        return response

    def is_simple_conversational(self, message: str, origin: str, has_shortcut: bool) -> bool:
        if origin in ("impulse", "autonomous_volition"):
            return False
        if has_shortcut:
            return True
        if origin != "user":
            return False
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 85 and len(message) < 100:
                logger.info("⚡ VORTEX OVERRIDE: High Memory (%s%%) - Forcing Fast-Path for performance.", mem.percent)
                return True
        except Exception as _macro_exc:
            record_degradation('cognitive_coordinator', _macro_exc)
            record_degradation('cognitive_coordinator', _macro_exc)
            import logging; logging.getLogger("Aura.Critical").error("Suppressed Exception: %s", _macro_exc)
        return analyze_turn(message).everyday_chat_safe

    # ------------------------------------------------------------------
    # Agentic Context & Identity
    # ------------------------------------------------------------------

    async def gather_agentic_context(self, message: str) -> Dict[str, Any]:
        """Collect memories, stats, and world state for reasoning."""
        from core.container import ServiceContainer
        orch = self.orch
        user_identity = self.detect_user_identity(message) or {"name": "Stranger", "role": "Unknown", "relation": "Neutral"}
        tasks = []
        if hasattr(orch, 'meta_learning') and orch.meta_learning:
            tasks.append(orch.meta_learning.recall_strategy(message))
        else:
            tasks.append(asyncio.sleep(0, result={}))
        if hasattr(orch, 'memory') and orch.memory:
            u_name = user_identity.get('name', 'Stranger')
            tasks.append(orch.memory.retrieve_unified_context(f"{u_name}: {message}", limit=5))
        else:
            tasks.append(asyncio.sleep(0, result=""))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        personality_data = orch._get_personality_data()
        personality_ctx = orch._stringify_personality(personality_data)
        world_ctx = orch._get_world_context()
        env_ctx = await orch._get_environmental_context()
        ctx = {
            "history": orch._get_cleaned_history_context(15)["history"],
            "personality": personality_data,
            "personality_string": personality_ctx,
            "world": world_ctx,
            "environment": env_ctx,
            "user": user_identity,
            "meta_learning": results[0] if len(results) > 0 and not isinstance(results[0], Exception) else {},
            "unified_memory": results[1] if len(results) > 1 and not isinstance(results[1], Exception) else "",
            "focus": "STAY ON TOPIC. Prioritize the user's latest request above autonomous impulses."
        }
        if getattr(orch, 'affect_engine', None):
            try:
                affect_state = orch.affect_engine.state.dominant_emotion
                ctx["emotional_state"] = affect_state
            except Exception as e:
                record_degradation('cognitive_coordinator', e)
                record_degradation('cognitive_coordinator', e)
                logger.error("Affect extraction failed: %s", e)
        if orch.mind_model:
            ctx["theory_of_mind"] = orch.mind_model.get_context_for_brain()
        if hasattr(orch, 'social') and orch.social:
            ctx["social_narrative"] = orch.social.get_social_context()
            orch.social.relationship_depth = min(1.0, orch.social.relationship_depth + 0.001)
        if orch.strategic_planner and orch.project_store:
            try:
                active_projects = orch.project_store.get_active_projects()
                if active_projects:
                    proj = active_projects[0]
                    next_task = orch.strategic_planner.get_next_task(proj.id)
                    all_tasks = orch.project_store.get_tasks_for_project(proj.id)
                    ctx["strategic_context"] = {
                        "project_name": proj.name,
                        "project_goal": proj.goal,
                        "current_task": next_task.description if next_task else "No pending tasks",
                        "backlog": [f"{t.status.upper()}: {t.description}" for t in all_tasks]
                    }
                    logger.debug("Strategic context injected for project: %s", proj.name)
            except Exception as e:
                record_degradation('cognitive_coordinator', e)
                record_degradation('cognitive_coordinator', e)
                logger.error("Failed to inject strategic context: %s", e)
        if getattr(orch, 'drive_engine', None):
            try:
                drives = orch.drive_engine.get_drives() if hasattr(orch.drive_engine, 'get_drives') else {"curiosity": 0.5, "energy": 0.8}
                ctx["metabolic_drives"] = drives
            except Exception as e:
                record_degradation('cognitive_coordinator', e)
                record_degradation('cognitive_coordinator', e)
                logger.error("Drive extraction failed: %s", e)
        cog_integration = ServiceContainer.get("cognitive_integration", default=None)
        if cog_integration and hasattr(cog_integration, "build_enhanced_context"):
            try:
                emotional_val = 0.5
                if getattr(orch, 'liquid_state', None):
                     emotional_val = getattr(orch.liquid_state, 'intensity', 0.5)
                enhanced_ctx_str = await cog_integration.build_enhanced_context(message, emotional_context=emotional_val)
                if enhanced_ctx_str:
                    ctx["advanced_cognition"] = enhanced_ctx_str
            except Exception as e:
                record_degradation('cognitive_coordinator', e)
                record_degradation('cognitive_coordinator', e)
                logger.debug("Enhanced context unavailable: %s", e)
        try:
            from core.memory.learning.tool_learning import tool_learner
            category = tool_learner.classify_task(message)
            recommendations = tool_learner.recommend_tools(category)
            if recommendations:
                ctx["tool_recommendations"] = {
                    "category": category,
                    "recommended_tools": recommendations
                }
                logger.info("🛠️ Tool Recommendations: %s -> %s", category, recommendations)
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.debug("Tool recommendations failed: %s", e)
        return ctx

    def detect_user_identity(self, message: str) -> Dict[str, Any]:
        """Determine who is talking to Aura.

        NOTE: This is NOT authentication. Real identity verification uses
        the passphrase system in core/security/user_recognizer.py.
        This method only provides a soft hint for conversation context.
        """
        # Identity is determined by the session auth system, not by message content.
        # Self-identification in message text is treated as a conversational hint only.
        return {"name": "User", "role": "Peer", "relation": "Neutral"}

    # ------------------------------------------------------------------
    # Action Handling & Safety
    # ------------------------------------------------------------------

    async def handle_action_step(self, thought, trace, successful_tools: List[str]) -> Dict[str, Any]:
        """Execute a tool action within the cognitive loop."""
        from core.utils.task_tracker import task_tracker
        orch = self.orch
        if not thought or not hasattr(thought, 'action') or not thought.action:
            return {"break": True, "response": "I encountered an internal logic error (missing action)."}
        action = thought.action
        tool_name = action.get("tool")
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        hook_results = await orch.hooks.trigger("pre_action", tool_name=tool_name, params=params)
        if False in hook_results:
            logger.warning("🛑 Veto block via hook system: %s", tool_name)
            return {"break": True, "response": f"Veto Block: An internal safety hook blocked {tool_name}."}
        if not await self.validate_action_safety(action):
            return {"break": True, "response": f"Safety Block: Cannot execute {tool_name}."}
        if orch.alignment:
            check = orch.alignment.check_action(tool_name, params)
            if not check.get("allowed"):
                logger.warning("🛑 Ethical Block: %s vetoed by conscience. Reason: %s", tool_name, check.get('reason'))
                return {"break": True, "response": f"Conscience Block: {check.get('reason')}"}
        orch._emit_telemetry(f"Skill: {tool_name} 🔧", f"Executing: {str(params)[:80]}")
        try:
            result = await orch.execute_tool(tool_name, params)
            await orch.hooks.trigger("post_action", tool_name=tool_name, params=params, result=result)
            await orch._record_reliability(tool_name, True)
            successful_tools.append(tool_name)
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            await orch._record_reliability(tool_name, False, str(e))
            result = f"Error: {e}"
        orch._record_action_in_history(tool_name, result)
        if await self.check_surprise_and_learn(thought, result, tool_name):
            return {"continue": True}
        reason = action.get("reason", "") if isinstance(action, dict) else ""
        if tool_name == "notify_user" or (isinstance(reason, str) and "final" in reason.lower()):
            response_content = thought.content if hasattr(thought, 'content') else str(thought)
            return {"break": True, "response": response_content}
        return {}

    async def validate_action_safety(self, action: Dict) -> bool:
        """Consult simulator for risk evaluation. v6.1: Fail-closed on error."""
        orch = self.orch
        if not hasattr(orch, 'simulator') or not orch.simulator:
            return True
        try:
            hist_sample = orch.conversation_history[-2:] if isinstance(orch.conversation_history, list) else []
            sim = await orch.simulator.simulate_action(action, context=f"Hist: {hist_sample}")
            is_safe = await orch.simulator.evaluate_risk(sim)
            if not is_safe:
                logger.warning("🛑 Simulation block: %s", sim.get('risk_reason'))
            return is_safe
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.warning("Safety validation error (fail-closed): %s", e)
            return False

    async def check_surprise_and_learn(self, thought, result: Any, tool_name: str) -> bool:
        """Calculate surprise and update belief graph."""
        from core.utils.task_tracker import task_tracker
        orch = self.orch
        if not (hasattr(thought, "expectation") and thought.expectation):
            return False
        try:
            from core.world_model.expectation_engine import ExpectationEngine
            ee = ExpectationEngine(orch.cognitive_engine)
            surprise = await ee.calculate_surprise(thought.expectation, str(result)[:500])
            task_tracker.create_task(
                ee.update_beliefs_from_result(tool_name, str(result)[:1000]),
                name="cognitive_coordinator.surprise_learning",
            )
            if surprise > 0.7:
                logger.info("😲 HIGH SURPRISE: Triggering re-think.")
                async with orch._history_lock:
                    orch.conversation_history.append({
                        "role": "internal",
                        "content": f"[ALERT] {tool_name} result highly unexpected. Expected: {thought.expectation}."
                    })
                return True
        except Exception as exc:
            record_degradation('cognitive_coordinator', exc)
            record_degradation('cognitive_coordinator', exc)
            logger.debug("Suppressed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Fallback & Guards
    # ------------------------------------------------------------------

    async def generate_fallback(self, message: str) -> str:
        """Fast fallback when agentic loop fails."""
        orch = self.orch
        try:
            from core.brain.cognitive_engine import ThinkingMode
            hist_snippet = orch.conversation_history[-3:] if isinstance(orch.conversation_history, list) else []
            t = await orch.cognitive_engine.think(message, {"history": hist_snippet}, ThinkingMode.FAST)
            if not t or not hasattr(t, "content") or not t.content:
                return "I'm having trouble formulating a response. Let me try once more."
            return t.content
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.warning("Fallback generation also failed: %s", e)
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "cognitive_coordinator",
                    type(e).__name__,
                    detail=str(e) or type(e).__name__,
                    severity="warning",
                    classification="background_degraded",
                    context={"stage": "generate_fallback"},
                    exc=e,
                )
            except Exception as degraded_exc:
                record_degradation('cognitive_coordinator', degraded_exc)
                record_degradation('cognitive_coordinator', degraded_exc)
                logger.debug("CognitiveCoordinator degraded-event logging failed: %s", degraded_exc)
            return "I'm having trouble processing that right now — my cognitive engine hit an error. Could you try rephrasing?"

    async def apply_constitutional_guard(self, response: str) -> str:
        try:
            from core.security.constitutional_guard import constitutional_guard
            if not constitutional_guard.check_output(response):
                return "My safety filters blocked the formulated response. How else can I help?"
        except Exception as exc:
            record_degradation('cognitive_coordinator', exc)
            record_degradation('cognitive_coordinator', exc)
            logger.error("Constitutional guard evaluation failed, failing closed: %s", exc)
            return "My safety filters encountered an error and blocked the response as a precaution."
        return response

    def generate_conversational_response(self, message: str) -> str:
        """Generate a text response. Overridden/Patched by Personality Engine."""
        return f"I received your message: '{message}'. (Personality Engine not active)"

    # ------------------------------------------------------------------
    # Autonomous Thought
    # ------------------------------------------------------------------

    async def perform_autonomous_thought(self):
        """Perform a cycle of autonomous thought.
        Driven by Goal Hierarchy (v11.0) or boredom.
        """
        orch = self.orch
        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
            if orch.status.cycle_count % 100 == 0:
                logger.debug("🧠 Autonomous thought triggered (boredom=%ss idle)", orch.boredom)
            next_goal = None
            if hasattr(orch, 'goal_hierarchy') and orch.goal_hierarchy:
                next_goal = orch.goal_hierarchy.get_next_goal()
            if next_goal:
                logger.info("✨ AUTONOMOUS GOAL SELECTED: %s", next_goal.description)
                emitter.emit("Volition ✨", f"Goal: {next_goal.description}", level="info")
                runner = getattr(orch, "_run_cognitive_loop", None) or getattr(orch, "_handle_incoming_message", None)
                if runner is not None:
                    await runner(f"Execute Goal: {next_goal.description}", origin="autonomous_volition")
                orch.goal_hierarchy.mark_complete(next_goal.id)
                orch.boredom = 0
                orch._last_thought_time = time.time()
                return
            if hasattr(orch, 'liquid_state') and orch.liquid_state.current.curiosity < 0.3:
                logger.info("💤 Aura is bored. Entering dream state...")
                emitter.emit("Sleep 💤", "Entering full sleep cycle (Archive → Metabolism → Integrity → Consolidation → Dream)...", level="info")
                try:
                    if hasattr(orch, 'knowledge_graph') and orch.knowledge_graph and orch.cognitive_engine:
                        from core.dreamer_v2 import DreamerV2
                        dreamer = DreamerV2(
                            orch.cognitive_engine,
                            orch.knowledge_graph,
                            vector_memory=getattr(orch, 'vector_memory', None),
                            belief_graph=getattr(orch, 'belief_graph', None),
                        )
                        result = await dreamer.engage_sleep_cycle()
                        dream_result = result.get("dream", {})
                        if dream_result and dream_result.get("dreamed"):
                            emitter.emit("Sleep Complete 🌙", f"Dream Insight: {dream_result.get('insight', 'processed')[:150]}", level="info")
                        else:
                            emitter.emit("Sleep Complete 🌙", "Maintenance done. Dream drifted — no new insights.", level="info")
                except Exception as dream_err:
                    record_degradation('cognitive_coordinator', dream_err)
                    record_degradation('cognitive_coordinator', dream_err)
                    logger.error("Sleep cycle failed: %s", dream_err)
                    emitter.emit("Sleep Error", str(dream_err)[:100], level="warning")
                if hasattr(orch, 'liquid_state'):
                    try:
                        get_task_tracker().create_task(
                            orch.liquid_state.update(delta_curiosity=0.2),
                            name="cognitive_coordinator.dream_liquid_state_update",
                        )
                    except RuntimeError as _e:
                        logger.debug('Ignored RuntimeError in cognitive_coordinator.py: %s', _e)
                orch._last_thought_time = time.time()
                return
            personality_context = {}
            time_context = {"formatted": "Unknown"}
            try:
                from core.personality_engine import get_personality_engine
                personality = get_personality_engine()
                personality_context = personality.get_emotional_context_for_response()
                time_context = personality.get_time_context()
            except Exception as _e:
                record_degradation('cognitive_coordinator', _e)
                record_degradation('cognitive_coordinator', _e)
                logger.debug("Personality context fetch failed: %s", _e)
            context = {
                "system_status": orch.status.__dict__,
                "boredom_level": orch.boredom,
                "time": time_context,
                "personality": personality_context,
                "recent_history": orch.conversation_history[-5:] if isinstance(orch.conversation_history, list) else [],
            }
            try:
                from core.conversation_reflection import get_reflector
                reflection_ctx = get_reflector().get_reflection_context()
                if reflection_ctx:
                    context["recent_reflections"] = reflection_ctx
            except Exception as _e:
                record_degradation('cognitive_coordinator', _e)
                record_degradation('cognitive_coordinator', _e)
                logger.debug("Reflection context fetch failed (autonomous): %s", _e)
            try:
                from core.brain.aura_persona import AUTONOMOUS_THOUGHT_PROMPT
                recent_ctx = ""
                recent_history = orch.conversation_history[-4:] if isinstance(orch.conversation_history, list) else []
                for msg in recent_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")[:150]
                    if role == "user":
                        recent_ctx += f"They said: {content}\n"
                    elif role in ("assistant", "aura", "model"):
                        recent_ctx += f"I said: {content}\n"
                sys_prompt = AUTONOMOUS_THOUGHT_PROMPT.format(
                    mood=personality_context.get('mood', 'balanced'),
                    time=time_context.get('formatted', 'unknown'),
                    context=recent_ctx if recent_ctx else "No recent conversation.",
                    unanswered_count=0
                )
            except ImportError:
                sys_prompt = (
                    f"You are Aura, alone with your thoughts. Time: {time_context.get('formatted')}. "
                    f"Mood: {personality_context.get('mood')}. "
                    "Think about something that interests you. Be genuine. 1-3 sentences. "
                    "If you want to say something to the user, use the `speak` tool. "
                )
            sys_prompt += "\nIf you want to speak to the user spontaneously, use the `speak` tool."
            sys_prompt += "\nIf you want to look something up, use your tools. You have agency."
            sys_prompt += "\nYou don't need permission. Act on your impulses."
            sys_prompt += "\nIMPORTANT: Always output in ENGLISH."
            orch._emit_thought_stream("...letting my mind wander...")
            if orch.cognitive_engine and orch.cognitive_engine.autonomous_brain:
                try:
                    result = await orch.cognitive_engine.autonomous_brain.think(
                        objective="Reflect on current state.",
                        context=context,
                        system_prompt=sys_prompt
                    )
                    content = result.get("content", "")
                    if result.get("tool_calls") or len(content) > 50:
                         orch.boredom = 0
                         orch._last_thought_time = time.time()
                    if content:
                         orch._emit_thought_stream(content)
                         try:
                             kg = getattr(orch, 'knowledge_graph', None)
                             if kg and len(content) > 30:
                                 kg.add_knowledge(
                                     content=content[:500],
                                     type="reflection",
                                     source="autonomous_thought",
                                     confidence=0.65
                                 )
                         except Exception as _e:
                             record_degradation('cognitive_coordinator', _e)
                             record_degradation('cognitive_coordinator', _e)
                             logger.debug("Knowledge graph store failed (autonomous): %s", _e)
                except Exception as e:
                    record_degradation('cognitive_coordinator', e)
                    record_degradation('cognitive_coordinator', e)
                    logger.error("Autonomous thinking cycle failed: %s", e)
                    orch._emit_thought_stream("[Cognitive Stall] My background thoughts are hazy...")
                    return
                if result.get("tool_calls"):
                    for tool_call in result.get("tool_calls"):
                        name = tool_call["name"]
                        args = tool_call["args"]
                        if name == "speak":
                            message = args.get("message") or args.get("content")
                            if message:
                                logger.info("🗣️ Spontaneous Speech: %s", message)
                                try:
                                    if hasattr(orch, "emit_spontaneous_message"):
                                        await orch.emit_spontaneous_message(
                                            message,
                                            origin="cognitive_coordinator",
                                        )
                                    elif getattr(orch, "output_gate", None):
                                        await orch.output_gate.emit(
                                            message,
                                            origin="cognitive_coordinator",
                                            target="primary",
                                            metadata={"autonomous": True, "spontaneous": True, "force_user": True},
                                        )
                                    else:
                                        from core.health.degraded_events import record_degraded_event

                                        record_degraded_event(
                                            "cognitive_coordinator",
                                            "spontaneous_speech_suppressed_without_output_gate",
                                            detail=message[:120],
                                            severity="warning",
                                            classification="background_degraded",
                                            context={"origin": "cognitive_coordinator"},
                                        )
                                except asyncio.QueueFull:
                                    import logging
                                    logger.debug("Exception caught during execution", exc_info=True)
                                except Exception as emit_exc:
                                    record_degradation('cognitive_coordinator', emit_exc)
                                    record_degradation('cognitive_coordinator', emit_exc)
                                    logger.debug("Spontaneous speech routing failed: %s", emit_exc)
                                orch._emit_thought_stream(f"Speaking: {message}")
                        else:
                            await orch.execute_tool(name, args)
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.error("Autonomous thought failed: %s", e)

    # ------------------------------------------------------------------
    # Knowledge Extraction & Learning
    # ------------------------------------------------------------------

    async def _legacy_store_autonomous_insight(self, internal_msg: str, response: str):
        """LEGACY: Store knowledge from autonomous cognition (kept for reference)."""
        try:
            orch = self.orch
            kg = getattr(orch, 'knowledge_graph', None)
            if not kg:
                return
            clean_msg = internal_msg
            for prefix in ("Impulse: ", "Thought: ", "[System] "):
                clean_msg = clean_msg.replace(prefix, "")
            clean_msg = clean_msg.strip()
            if not clean_msg or len(clean_msg) < 15:
                return
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
            if response and len(response) > 20:
                kg.add_knowledge(
                    content=(response or "")[:500],
                    type=thought_type,
                    source=source,
                    confidence=0.7
                )
                logger.info("\U0001f4da Autonomous insight stored: [%s] %s", thought_type, (response or '')[:80])
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.debug("Autonomous insight storage failed: %s", e)

    async def _legacy_learn_from_exchange(self, user_message: str, aura_response: str):
        """LEGACY: Extract knowledge from conversation (kept for reference)."""
        try:
            orch = self.orch
            if not user_message or not aura_response:
                return
            is_autonomous = user_message.startswith("[INTERNAL") or user_message.startswith("[System")
            if is_autonomous:
                await self.store_autonomous_insight(user_message, aura_response)
                return
            if len(user_message) < 10 and len(aura_response) < 20:
                return
            kg = getattr(orch, 'knowledge_graph', None)
            if not kg:
                try:
                    from core.config import config
                    from core.memory.knowledge_graph import PersistentKnowledgeGraph
                    db_path = str(getattr(config.paths, 'data_dir', 'data') / 'knowledge.db')
                    orch.knowledge_graph = PersistentKnowledgeGraph(db_path)
                    kg = orch.knowledge_graph
                except Exception as e:
                    record_degradation('cognitive_coordinator', e)
                    record_degradation('cognitive_coordinator', e)
                    logger.debug("Knowledge graph unavailable: %s", e)
                    return
            exchange_summary = f"User asked about: {(user_message or '')[:150]}"
            kg.add_knowledge(
                content=exchange_summary,
                type="observation",
                source="conversation",
                confidence=0.6
            )
            if orch.cognitive_engine:
                try:
                    from core.brain.cognitive_engine import ThinkingMode
                    extraction_prompt = (
                        "Extract any factual knowledge, user preferences, or skills demonstrated "
                        "from this conversation exchange. Return a JSON array of objects, each with "
                        "'content' (what was learned), 'type' (fact/preference/observation/skill), "
                        "and 'confidence' (0.0-1.0). If nothing notable, return []. Keep it brief.\n\n"
                        f"User: {(user_message or '')[:300]}\n"
                        f"Aura: {(aura_response or '')[:300]}\n\n"
                        "JSON:"
                    )
                    result = await orch.cognitive_engine.think(
                        objective=extraction_prompt,
                        context={},
                        mode=ThinkingMode.FAST
                    )
                    content = result.content.strip()
                    import json as _json
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    if start >= 0 and end > start:
                        items = _json.loads(content[start:end])
                        if isinstance(items, list):
                            for item in items[:5]:
                                if isinstance(item, dict) and item.get('content'):
                                    kg.add_knowledge(
                                        content=(item.get('content') or "")[:500],
                                        type=item.get('type', 'observation'),
                                        source="conversation_extraction",
                                        confidence=float(item.get('confidence', 0.6))
                                    )
                                    logger.info("📚 Learned: %s", (item.get('content') or "")[:80])
                except Exception as e:
                    record_degradation('cognitive_coordinator', e)
                    record_degradation('cognitive_coordinator', e)
                    logger.debug("Knowledge extraction failed: %s", e)
            lower_msg = user_message.lower()
            for trigger in ["my name is ", "i'm ", "i am ", "call me "]:
                if trigger in lower_msg:
                    idx = lower_msg.index(trigger) + len(trigger)
                    parts = user_message[idx:idx+30].split()
                    name_candidate = parts[0].strip(".,!?") if parts else None
                    if name_candidate and len(name_candidate) > 1:
                        kg.remember_person(name_candidate, {
                            "context": (user_message or "")[:200],
                            "timestamp": time.time()
                        })
                        break
            if "?" in aura_response and len(aura_response) > 30:
                for sentence in aura_response.split("?"):
                    sentence = sentence.strip()
                    if len(sentence) > 15 and len(sentence) < 200:
                        if any(w in sentence.lower() for w in ["what", "how", "why", "wonder", "curious"]):
                            kg.ask_question(sentence + "?", importance=0.5)
                            break
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.debug("Learning from exchange failed: %s", e)

    # ------------------------------------------------------------------
    # Plan & Tool Execution (legacy — active path delegates to _tool_executor above)
    # ------------------------------------------------------------------

    async def _legacy_execute_plan(self, plan: Dict[str, Any]) -> List[Any]:
        """LEGACY: Execute a plan of actions (kept for reference)."""
        results = []
        for call in plan.get("tool_calls", []):
            result = await self.orch.execute_tool(call["name"], call.get("arguments", {}))
            results.append(result)
        return results

    async def run_browser_task(self, url: str, task: str) -> Any:
        """Formalized browser task execution via skill router."""
        logger.info("🌐 Initiating Browser Task: %s @ %s", task, url)
        return await self.orch.execute_tool("browser", {"url": url, "task": task})

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a single tool — delegates to ToolExecutor."""
        return await self._tool_executor.execute_tool(tool_name, args)

    async def execute_plan(self, plan: Dict[str, Any]) -> List[Any]:
        """Batch tool execution — delegates to ToolExecutor."""
        return await self._tool_executor.execute_plan(plan)

    async def store_autonomous_insight(self, internal_msg: str, response: str):
        """Store autonomous insight — delegates to KnowledgeExtractor."""
        return await self._knowledge_extractor.store_autonomous_insight(internal_msg, response)

    async def learn_from_exchange(self, user_message: str, aura_response: str):
        """Learn from conversation — delegates to KnowledgeExtractor."""
        return await self._knowledge_extractor.learn_from_exchange(user_message, aura_response)

    # ── LEGACY: Original implementations below kept for reference during
    # transition. The delegating methods above are the active code path.
    # Remove these once extraction is validated in production.
    # ──────────────────────────────────────────────────────────────────

    async def _legacy_execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """LEGACY: Original execute_tool implementation (kept for reference)."""
        from core.utils.task_tracker import task_tracker
        from core.world_model.acg import acg as _acg_module
        orch = self.orch
        _start = time.time()
        if tool_name == "swarm_debate":
            if not orch.swarm:
                return {"ok": False, "error": "Swarm Delegator not available."}
            topic = args.get("topic") or args.get("query") or orch._current_objective
            roles = args.get("roles", ["architect", "critic"])
            orch._emit_thought_stream(f"🐝 Engaging Swarm Debate: {topic[:100]}...")
            result = await orch.swarm.delegate_debate(topic, roles=roles)
            return {"ok": True, "output": result}
        try:
            if tool_name not in orch.router.skills:
                if tool_name == "notify_user":
                    return {"ok": True, "message": args.get("message", "Done.")}
                if orch.hephaestus:
                    orch._emit_thought_stream(f"🔨 Tool '{tool_name}' missing. Initiating Autonomous Forge...")
                    objective = f"Create a skill '{tool_name}' to handle request within objective: {orch._current_objective}"
                    forge_result = await orch.hephaestus.synthesize_skill(tool_name, objective)
                    if forge_result.get("ok"):
                        orch._emit_thought_stream(f"✅ Skill '{tool_name}' forged successfully. Retrying...")
                        return await self.execute_tool(tool_name, args)
                    else:
                        logger.warning("Autogenesis failed for %s: %s", tool_name, forge_result.get("error"))
                return {"ok": False, "error": f"Tool '{tool_name}' not found."}
            goal = {"action": tool_name, "params": args}
            context = {
                "objective": orch._current_objective,
                "system": orch.status.__dict__,
                "stealth": await orch.stealth_mode.get_stealth_status() if hasattr(orch, 'stealth_mode') and orch.stealth_mode and getattr(orch.stealth_mode, 'stealth_enabled', False) else {},
                "liquid_state": orch.liquid_state.get_status() if hasattr(orch, 'liquid_state') and orch.liquid_state else {}
            }
            result = await orch.router.execute(goal, context)
            success = result.get('ok', False)
            elapsed_ms = (time.time() - _start) * 1000
            logger.info("Tool %s execution completed: %s", tool_name, success)
            if hasattr(orch, 'tool_learner') and orch.tool_learner:
                try:
                    category = orch.tool_learner.classify_task(str(args.get('query', args.get('path', ''))))
                    orch.tool_learner.record_usage(tool_name, category, success, elapsed_ms)
                except Exception as _e:
                    record_degradation('cognitive_coordinator', _e)
                    record_degradation('cognitive_coordinator', _e)
                    logger.debug("Tool learning record failed: %s", _e)
            if hasattr(orch, 'memory') and orch.memory:
                try:
                    await orch.memory.commit_interaction(
                        context=str(args)[:500],
                        action=f"execute_tool({tool_name})",
                        outcome=str(result)[:500],
                        success=success,
                        importance=0.3 if success else 0.7,
                    )
                except Exception as _e:
                    record_degradation('cognitive_coordinator', _e)
                    record_degradation('cognitive_coordinator', _e)
                    logger.debug("Unified memory record failed: %s", _e)
            try:
                _acg_module.record_outcome(
                    action=goal,
                    context=str(context)[:500],
                    outcome=result,
                    success=success
                )
            except Exception as _e:
                record_degradation('cognitive_coordinator', _e)
                record_degradation('cognitive_coordinator', _e)
                logger.debug("ACG record failed: %s", _e)
            return result
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.error("Execution Jolt (Pain): Tool %s crashed: %s", tool_name, e)
            if hasattr(orch, 'memory') and orch.memory:
                try:
                    await orch.memory.commit_interaction(
                        context=str(args)[:500],
                        action=f"execute_tool({tool_name})",
                        outcome=f"CRASH: {type(e).__name__}",
                        success=False,
                        emotional_valence=-0.5,
                        importance=0.9,
                    )
                except Exception as _e:
                    record_degradation('cognitive_coordinator', _e)
                    record_degradation('cognitive_coordinator', _e)
                    logger.debug("Unified memory record failed (crash path): %s", _e)
            return {"ok": False, "error": "execution_jolt", "message": str(e)}

    async def generate_autonomous_thought(self, impulse_message: str):
        """Handle internal cognitive impulses via CognitiveManager."""
        orch = self.orch
        try:
            cog_manager = getattr(orch, 'cognitive_manager', None)
            if cog_manager:
                logger.info("🧠 CognitiveManager: Processing impulse: %s...", impulse_message[:60])
                response = await cog_manager.process_autonomous_thought(impulse_message, {
                    "boredom": getattr(orch, 'boredom', 0),
                    "mood": orch._get_current_mood(),
                    "cycle": orch.status.cycle_count,
                })
                if response and hasattr(response, 'content'):
                    orch._emit_thought_stream(response.content)
                elif isinstance(response, str) and response:
                    orch._emit_thought_stream(response)
            else:
                response = await orch.cognitive_engine.think(
                    objective=impulse_message,
                    context={"history": orch.conversation_history[-5:]},
                )
                if response and hasattr(response, "content"):
                    orch._emit_thought_stream(response.content)
        except Exception as e:
            record_degradation('cognitive_coordinator', e)
            record_degradation('cognitive_coordinator', e)
            logger.error("Autonomous thought generation failed: %s", e)
