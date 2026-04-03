"""Message Pipeline Mixin for RobustOrchestrator.
Handles request lifecycle steps, guardrails, and context assembly.
"""
import inspect
import logging
import asyncio
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _dispose_awaitable(result: Any) -> None:
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()

class MessagePipelineMixin:
    """Encapsulates the cognitive loop's individual step executions and context creation."""

    async def _handle_action_step(self, thought, trace, successful_tools: list[str]) -> dict[str, Any]:
        """Execute a tool action within the cognitive loop."""
        if not thought or not hasattr(thought, 'action') or not thought.action:
            return {"break": True, "response": "I encountered an internal logic error (missing action)."}
            
        action = self._normalize_to_dict(thought.action)
        tool_name = action.get("tool")
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        # 1. Simulation / Safety Check
        hook_results = await self.hooks.trigger("pre_action", tool_name=tool_name, params=params)
        if False in hook_results:
            logger.warning("🛑 Veto block via hook system: %s", tool_name)
            return {"break": True, "response": f"Veto Block: An internal safety hook blocked {tool_name}."}

        safety_check = await self._validate_action_safety(action)
        if not safety_check.get("allowed", False):
            reason = safety_check.get("reason", "Unknown safety violation")
            return {"break": True, "response": f"Safety Block: {reason}"}
            
        # 1.5 Conscience Check (Ethical Compass)
        try:
            from core.container import ServiceContainer
            alignment = ServiceContainer.get("constitutional_alignment", default=None)
            if alignment:
                check_result = await alignment.check_action(action_description=f"Executing {tool_name} with args {params}", context=self.conversation_history[-2:] if isinstance(self.conversation_history, list) else [])
                if not check_result:
                    logger.warning("🛑 Constitutional Block: %s vetoed by alignment layer.", tool_name)
                    return {"break": True, "response": "Constitutional Block: Action violates core principles."}
        except Exception as e:
            logger.debug("Constitutional check failed: %s", e)
            
        # 2. Execution
        self._emit_telemetry(f"Skill: {tool_name} 🔧", f"Executing: {str(params)[:80]}")
        
        try:
            result = await self.execute_tool(tool_name, params)
            await self.hooks.trigger("post_action", tool_name=tool_name, params=params, result=result)
            await self._record_reliability(tool_name, True)
            successful_tools.append(tool_name)
        except Exception as e:
            await self._record_reliability(tool_name, False, str(e))
            result = f"Error: {e}"

        # 3. Perception / Surprise Logic
        self._record_action_in_history(tool_name, result)
        if await self._check_surprise_and_learn(thought, result, tool_name):
            return {"continue": True}
            
        # 4. Completion Check
        reason = action.get("reason", "")
        if tool_name == "notify_user" or (isinstance(reason, str) and "final" in reason.lower()):
            # Guard thought.content access
            response_content = thought.content if hasattr(thought, 'content') else str(thought)
            return {"break": True, "response": response_content}
            
        return {}

    async def _validate_action_safety(self, action: dict) -> dict:
        """Consult simulator for risk evaluation. Fail-closed on error."""
        if not hasattr(self, 'simulator') or not self.simulator:
            return {"allowed": True}
        try:
            hist_sample = self.conversation_history[-2:] if isinstance(self.conversation_history, list) else []
            sim = await self.simulator.simulate_action(action, context=f"Hist: {hist_sample}")
            is_safe = await self.simulator.evaluate_risk(sim)
            reason = sim.get('risk_reason', 'Action determined to be too risky.')
            if not is_safe:
                logger.warning("🛑 Simulation block: %s", reason)
            return {"allowed": is_safe, "reason": reason}
        except Exception as e:
            logger.warning("Safety validation error (fail-closed): %s", e)
            return {"allowed": False, "reason": f"Internal safety verification fault: {str(e)}"}

    async def _check_surprise_and_learn(self, thought, result: Any, tool_name: str) -> bool:
        """Calculate surprise and update belief graph."""
        if not (hasattr(thought, "expectation") and thought.expectation):
            return False
            
        try:
            from core.world_model.expectation_engine import ExpectationEngine
            ee = ExpectationEngine(self.cognitive_engine)
            surprise = await ee.calculate_surprise(thought.expectation, str(result)[:500])
            
            from core.utils.task_tracker import get_task_tracker
            # Recursive Learning task
            get_task_tracker().track_task(asyncio.create_task(ee.update_beliefs_from_result(tool_name, str(result)[:1000])))
            
            if surprise > 0.7:
                logger.info("😲 HIGH SURPRISE: Triggering re-think.")
                async with self._history_lock:
                    self.conversation_history.append({
                        "role": "internal", 
                        "content": f"[ALERT] {tool_name} result highly unexpected. Expected: {thought.expectation}."
                    })
                return True
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

            return False

    async def _generate_fallback(self, message: str) -> str:
        """Fast fallback when agentic loop fails (transparent)."""
        try:
            from core.brain.cognitive_engine import ThinkingMode
            hist_snippet = self.conversation_history[-3:] if isinstance(self.conversation_history, list) else []
            t = await self.cognitive_engine.think(message, {"history": hist_snippet}, ThinkingMode.FAST)
            if not t or not hasattr(t, "content") or not t.content:
                return "I'm having trouble formulating a response. Let me try once more."
            return t.content
        except Exception as e:
            logger.warning("Fallback generation also failed: %s", e)
            return "I'm having trouble processing that right now — my cognitive engine hit an error. Could you try rephrasing?"

    async def _apply_constitutional_guard(self, response: str) -> str:
        """Runs the Gemini safety check with a hard 5s timeout. Fails open."""
        try:
            guarded = await asyncio.wait_for(
                self._run_guard_inner(response),
                timeout=5.0
            )
            return guarded
        except asyncio.TimeoutError:
            logger.warning("[ConstitutionalGuard] Timed out after 5s — passing raw response.")
            return response
        except Exception as e:
            logger.warning(f"[ConstitutionalGuard] Error — passing raw response. Reason: {e}")
            return response

    async def _run_guard_inner(self, response: str) -> str:
        try:
            from core.security.constitutional_guard import constitutional_guard
            if not constitutional_guard.check_output(response):
                return "My safety filters blocked the formulated response. How else can I help?"
        except Exception as exc:
            logger.error("Constitutional guard evaluation failed, failing closed: %s", exc)
            return "My safety filters encountered an error and blocked the response as a precaution."

        return response

    def _get_cleaned_history_context(self, limit: int = 12) -> dict[str, Any]:
        """Filter out internal noise for LLM context while preserving tool results.
        
        Peak Quality: Truncates tool results more aggressively and filters more internal noise.
        """
        if not hasattr(self, "conversation_history") or not isinstance(self.conversation_history, list):
            return {"history": []}
            
        cleaned = []
        # Use a sliding window for history to prevent extreme token bloat
        for msg in self.conversation_history[-limit:]:
            content = msg.get("content", "")
            role = msg.get("role", "unknown")
            
            # Skip high-frequency internal signals or thought traces
            if not content or content.startswith(("⚡", "[INTERNAL", "[System", "Impulse: ", "Thought:", "Observation: ")):
                continue
                
            # Truncate function results aggressively but keep start/end
            if role == "function":
                if len(content) > 300:
                    content = f"[TOOL RESULT]: {content[:150]} ... [TRUNCATED] ... {content[-100:]}"
                
            cleaned.append({"role": role, "content": content})
            
        return {"history": cleaned}

    def _init_cognitive_trace(self, message: str, origin: str):
        from core.meta.cognitive_trace import CognitiveTrace
        trace = CognitiveTrace(trace_id=f"{origin}_{int(time.time())}")
        trace.record_step("start", {"message": message, "origin": origin})
        return trace

    async def _check_social_reflexes(self, message: str) -> Optional[str]:
        """[PROVING GROUND] Scripted reflexes disabled to ensure 100% LLM-driven novelty."""
        return None
        
        # import re
        import random
        from core.brain import aura_persona
        
        msg = message.lower().strip()
        # Strip punctuation for better matching
        msg = re.sub(r'[^\w\s]', '', msg)
        
        # 1. Greetings
        greetings = [r"^hi$", r"^hello$", r"^hey$", r"^yo$", r"^sup$", r"^whats up$", r"^hey aura$"]
        if any(re.match(p, msg) for p in greetings):
            return random.choice(aura_persona.GREETING_RESPONSES)
            
        # 2. Well-being
        well_being = [r"^how are you$", r"^hows it going$", r"^how you doing$", r"^you okay$"]
        if any(re.match(p, msg) for p in well_being):
            return random.choice(aura_persona.HOW_ARE_YOU_RESPONSES)
            
        # 3. Identity
        identity = [r"^who are you$", r"^what are you$", r"^who is aura$"]
        if any(re.match(p, msg) for p in identity):
            return random.choice(aura_persona.IDENTITY_RESPONSES)
            
        # 4. Assistant Denial
        denial = [r".*assistant.*", r".*chatbot.*", r".*ai model.*"]
        if any(re.match(p, msg) for p in denial):
            # Only trigger if they are asking or asserting identity
            if any(x in msg for x in ["are you", "you are", "tell me about"]):
                return random.choice(aura_persona.ASSISTANT_DENIAL_RESPONSES)
                
        return None

    async def _get_environmental_context(self) -> dict[str, Any]:
        """Get rich environment data from EnvironmentAwareness module."""
        import datetime
        try:
            from core.environment_awareness import get_environment
            env = get_environment()
            # Refresh context (device/location are cached internally)
            ctx = await env.get_full_context()
            
            # Ensure time/date are present for prompt compatibility
            ctx["time"] = datetime.datetime.now().strftime("%I:%M %p")
            ctx["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
            
            return ctx
        except Exception as e:
            logger.error("Environment Context Error: %s", e)
            return {}

    def _get_world_context(self) -> str:
        try:
            from core.world_model.belief_graph import get_belief_graph
            bg = get_belief_graph()
            graph = getattr(bg, "graph", None)
            if inspect.isawaitable(graph):
                _dispose_awaitable(graph)
                return ""
            self_node = bg.graph.nodes.get(bg.self_node_id, {})
            if inspect.isawaitable(self_node):
                _dispose_awaitable(self_node)
                return ""
            attrs = self_node.get("attributes", {})
            return f"MOOD: {attrs.get('emotional_valence')}, ENERGY: {attrs.get('energy_level')}"
        except Exception as e:
            logger.warning("World context retrieval failed: %s", e)
            return ""

    async def _record_reliability(self, tool: str, success: bool, error: Optional[str] = None):
        try:
            from core.reliability_tracker import reliability_tracker
            reliability_tracker.record_attempt(tool, success, error)
        except Exception as e:
            logger.debug("Reliability record failed: %s", e)
            
    def _record_action_in_history(self, tool_name: str, result: Any):
        # Use more descriptive internal markers to prompt narration
        if not hasattr(self, "conversation_history") or not isinstance(self.conversation_history, list):
            return
        self.conversation_history.append({
            "role": "internal", 
            "content": f"[SKILL OUTPUT: {tool_name}]\n{str(result)}"
        })

    def _inject_shortcut_results(self, message: str, result: dict) -> str:
        summary = str(result.get("summary", result.get("result", result)))[:800]
        return f"{message}\n\n[DIRECT RESULT]: {summary}\n\nSynthesize this result for the user."
