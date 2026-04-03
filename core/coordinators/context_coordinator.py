"""Context Coordinator — gathers personality, environment, world state,
and history context for the cognitive loop.

Extracted from orchestrator.py as part of the God Object decomposition.
"""
import datetime
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextCoordinator:
    """Pure data-gathering coordinator. No side-effects on system state."""

    def __init__(self, orch):
        self.orch = orch

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_cleaned_history_context(self, limit: int = 12) -> Dict[str, Any]:
        """Filter out internal noise for LLM context while preserving tool results."""
        history = self.orch.conversation_history
        if not isinstance(history, list):
            return {"history": []}
        cleaned: List[Dict[str, str]] = []
        for msg in history[-limit:]:
            content = msg.get("content", "")
            role = msg.get("role", "unknown")
            if not content or content.startswith(
                ("⚡", "[INTERNAL", "[System", "Impulse: ", "Thought:", "Observation: ")
            ):
                continue
            if role == "function" and len(content) > 300:
                content = f"[TOOL RESULT]: {content[:150]} ... [TRUNCATED] ... {content[-100:]}"
            cleaned.append({"role": role, "content": content})
        return {"history": cleaned}

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def emit_telemetry(self, flow: str, text: str):
        """Send updates to the Thought Stream UI."""
        try:
            from core.thought_stream import get_emitter
            cycle = self.orch.status.cycle_count if hasattr(self.orch, "status") else 0
            get_emitter().emit(flow, text, level="info", cycle=cycle)
        except Exception as e:
            logger.debug("Telemetry emit failed: %s", e)

    def emit_thought_stream(self, thought):
        """Emit autonomous thoughts / monologues to UI."""
        try:
            from core.thought_stream import get_emitter
            get_emitter().emit("thought", str(thought))
        except Exception as _exc:
            import logging
            logger.debug("Exception caught during execution", exc_info=True)

    # ------------------------------------------------------------------
    # Cognitive Trace
    # ------------------------------------------------------------------

    def init_cognitive_trace(self, message: str, origin: str):
        from core.meta.cognitive_trace import CognitiveTrace
        trace = CognitiveTrace(trace_id=f"{origin}_{int(time.time())}")
        trace.record_step("start", {"message": message, "origin": origin})
        return trace

    # ------------------------------------------------------------------
    # Personality & Mood
    # ------------------------------------------------------------------

    def get_personality_data(self) -> dict:
        """Get raw personality metrics as a dictionary."""
        try:
            from core.brain.personality_engine import get_personality_engine
            pe = get_personality_engine()
            pe.update()
            return pe.get_emotional_context_for_response()
        except Exception as e:
            logger.warning("Personality metric retrieval failed: %s", e)
            return {"mood": "neutral", "tone": "snarky", "emotional_state": {}}

    def stringify_personality(self, ctx: dict) -> str:
        """Convert personality dict to HUD-compatible string."""
        mood = ctx.get("mood", "neutral").upper()
        tone = ctx.get("tone", "balanced")
        emotions = ", ".join(
            [f"{n}: {i:.0f}" for n, i in ctx.get("emotional_state", {}).items() if i > 65]
        )
        return f"MOOD: {mood} | TONE: {tone} | INTENSE EMOTIONS: {emotions or 'none'}"

    def get_personality_context(self) -> str:
        """Legacy wrapper for personality string."""
        data = self.get_personality_data()
        return self.stringify_personality(data)

    def get_current_mood(self) -> str:
        """Get current mood from personality engine (safe helper)."""
        try:
            from core.brain.personality_engine import get_personality_engine
            return get_personality_engine().current_mood
        except Exception:
            return "balanced"

    def get_current_time_str(self) -> str:
        """Get current time string (safe helper)."""
        try:
            from core.brain.personality_engine import get_personality_engine
            return get_personality_engine().get_time_context().get("formatted", "")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Environment & World
    # ------------------------------------------------------------------

    async def get_environmental_context(self) -> Dict[str, Any]:
        """Get rich environment data from EnvironmentAwareness module."""
        try:
            from core.environment_awareness import get_environment
            env = get_environment()
            ctx = await env.get_full_context()
            ctx["time"] = datetime.datetime.now().strftime("%I:%M %p")
            ctx["date"] = datetime.datetime.now().strftime("%Y-%m-%d")
            return ctx
        except Exception as e:
            logger.error("Environment Context Error: %s", e)
            return {}

    def get_world_context(self) -> str:
        try:
            from core.world_model.belief_graph import get_belief_graph
            bg = get_belief_graph()
            self_node = bg.graph.nodes.get(bg.self_node_id, {})
            attrs = self_node.get("attributes", {})
            return f"MOOD: {attrs.get('emotional_valence')}, ENERGY: {attrs.get('energy_level')}"
        except Exception as e:
            logger.warning("World context retrieval failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    async def record_reliability(self, tool: str, success: bool, error: Optional[str] = None):
        try:
            from core.reliability_tracker import reliability_tracker
            reliability_tracker.record_attempt(tool, success, error)
        except Exception as e:
            logger.debug("Reliability record failed: %s", e)

    def record_action_in_history(self, tool_name: str, result: Any):
        self.orch.conversation_history.append({
            "role": "internal",
            "content": f"[SKILL OUTPUT: {tool_name}]\n{str(result)}"
        })

    def inject_shortcut_results(self, message: str, result: Dict) -> str:
        summary = str(result.get("summary", result.get("result", result)))[:800]
        return f"{message}\n\n[DIRECT RESULT]: {summary}\n\nSynthesize this result for the user."

    def post_process_response(self, text: str) -> str:
        return text.strip()