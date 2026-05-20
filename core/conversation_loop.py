"""Autonomous Conversation Loop - The Heart of Aura's Agency

CRITICAL FIX: This module ensures Aura continues thinking and generating
goals autonomously, preventing the "echo only" behavior.

Features:
1. Proactive goal generation from drives
2. Background thinking cycles
3. Conversation memory integration
4. Plan execution coordination
"""

import asyncio
import logging
import random
import sqlite3
import time
from queue import Queue
from typing import Any

from core.conversation.unified_transcript import UnifiedTranscript
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

_RECOVERABLE_LOOP_ERRORS = (
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


def _record_conversation_loop_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "conversation_loop",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )


def get_transcript() -> UnifiedTranscript:
    global _transcript
    if "_transcript" not in globals():
        _transcript = UnifiedTranscript.get_instance()
    return _transcript


logger = logging.getLogger("Kernel.ConversationLoop")


class AutonomousConversationLoop:
    """Manages the autonomous thought and action cycle.
    Prevents Aura from becoming a passive echo chamber.
    """

    # Aura's true identity
    AI_ROLE = "Aura"

    def __init__(self, planner, executor, drive_system, memory, brain):
        """Initialize the conversation loop.

        Args:
            planner: Planner instance for goal decomposition
            executor: SkillRouter for executing actions
            drive_system: DriveSystem for autonomous motivation
            memory: VectorMemory for context
            brain: Cognitive engine for thinking

        """
        self.planner = planner
        self.executor = executor
        self.drives = drive_system
        self.memory = memory
        self.brain = brain
        self.hierarchical_orch = None
        self.conversation_reflector = None

        # State management
        self.is_running = False
        self.background_thread = None
        self.goal_queue = Queue()
        self._main_loop = None  # Captured on start() — the main asyncio event loop

        from core.utils.paths import DATA_DIR

        data_dir = DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = str(data_dir / "recent_history.json")
        # conversation_history is now a property pulling from UnifiedTranscript
        # Timing controls
        self.last_autonomous_action = time.time()
        self.autonomous_interval = 30  # Seconds between autonomous actions

        # Statistics
        self.stats = {
            "autonomous_goals": 0,
            "user_goals": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "recovered_tool_failures": 0,
        }
        self._autonomous_failure_streak = 0
        self._plan_failure_streak = 0
        self._last_loop_error = ""

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Dynamically build conversation history from the unified transcript."""
        transcript = get_transcript()
        return [entry.to_dict() for entry in transcript.get_context_window(n=50)]

    def start(self):
        """Start the autonomous loop"""
        if self.is_running:
            logger.warning("Loop already running")
            return

        self.is_running = True
        try:
            self._main_loop = asyncio.get_running_loop()
            self.background_thread = get_task_tracker().create_task(
                self._background_loop(),
                name="AuraAutonomousLoop",
            )
            logger.info("✓ Autonomous conversation loop started")
        except RuntimeError:
            logger.error("No running event loop found when starting AutonomousConversationLoop")
            self._main_loop = None
            self.is_running = False

    def stop(self):
        """Stop the autonomous loop"""
        self.is_running = False
        if self.background_thread:
            self.background_thread.cancel()
        logger.info("Autonomous conversation loop stopped")

    async def process_user_input(self, user_message: str) -> dict[str, Any]:
        """Process user input and generate response (Async).

        Args:
            user_message: User's message

        Returns:
            Response dictionary with conversation and actions

        """
        logger.info("📥 User input: %s...", user_message[:100])

        # Add to conversation history
        await self._add_to_history("user", user_message)

        # Phase 6: Hierarchical Memory Compaction
        if not self.hierarchical_orch:
            from core.container import get_container

            self.hierarchical_orch = get_container().get("hierarchical_memory_orchestrator")

        if self.hierarchical_orch:
            # The orchestrator will internalize the history and store compact versions in the DB
            await self.hierarchical_orch.maybe_compact(self.conversation_history)

        # Phase 10: Meta-Cognition (Conversation Reflection)
        if not self.conversation_reflector:
            try:
                from core.conversation_reflection import get_reflector

                self.conversation_reflector = get_reflector()
            except ImportError as _exc:
                logger.debug("Suppressed ImportError: %s", _exc)

        if self.conversation_reflector:
            # Background the reflection task to avoid blocking the main interaction
            reflect_coro = self.conversation_reflector.maybe_reflect(
                self.conversation_history, self.brain
            )
            try:
                reflect_task = get_task_tracker().create_task(
                    reflect_coro,
                    name="conversation_loop_reflection",
                )
            except RuntimeError:
                reflect_coro.close()
            except (AttributeError, TypeError, ValueError):
                reflect_coro.close()
                raise
            else:
                if not (asyncio.isfuture(reflect_task) or isinstance(reflect_task, asyncio.Task)):
                    reflect_coro.close()
                    reflect_task = None
                if reflect_task is None:
                    reflect_coro.close()

        # Update stats
        self.stats["user_goals"] += 1

        # Update social drive (user interaction satisfies social need)
        if hasattr(self.drives, "satisfy"):
            self.drives.satisfy("social", 20.0)

        # === FAST PATH: Simple greetings get quick response ===
        lower_msg = user_message.lower().strip()
        simple_greetings = [
            "hello",
            "hi",
            "hey",
            "greetings",
            "sup",
            "what's up",
            "how are you",
            "yo",
            "hiya",
            "hello!",
        ]
        if lower_msg in simple_greetings or lower_msg.rstrip("!?.,") in simple_greetings:
            response = "I'm Aura Cortex, Autonomous Intelligence."
            await self._add_to_history("self", response)
            logger.info("Fast-path greeting response")
            return {
                "ok": True,
                "response": response,
                "type": "conversation",
                "plan": ["Responded to greeting"],
            }

        # Force every response through the full cognitive stack.
        # Instead of the 'planner' shortcut, we enter the unitary pipeline directly.
        # This ensures every turn gets full perception, planning, and self-correction.
        logger.info("🧠 ConversationLoop: Routing through full cognitive stack...")
        try:
            response = await self.brain.generate(user_message, priority=True)
            await self._add_to_history(self.AI_ROLE, response)
            return {
                "ok": True,
                "response": response,
                "type": "conversation",
                "plan": ["Processed via Unitary Cognitive Pipeline"],
            }
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_conversation_loop_degradation(
                e,
                stage="full_cognitive_stack",
                action="fell back to direct conversational generation after unitary stack failed",
                severity="degraded",
            )
            logger.error("❌ ConversationLoop: Full cognitive stack failed: %s", e)
            # Fallback to direct generate if phase loop fails
            response = await self._generate_conversational_response(user_message)
            await self._add_to_history(self.AI_ROLE, response)
            return {
                "ok": True,
                "response": response,
                "type": "conversation",
                "plan": ["Fallback to direct generation"],
            }

    async def _background_loop(self):
        """Background task for autonomous behavior.
        Generates internal goals based on drives.
        """
        logger.info("Background autonomous loop starting...")

        while self.is_running:
            try:
                # Update drives
                if hasattr(self.drives, "update"):
                    await self.drives.update()

                # Check if it's time for autonomous action
                now = time.time()
                time_since_last = now - self.last_autonomous_action

                if time_since_last >= self.autonomous_interval:
                    # Check for imperative from drives
                    imperative = None
                    if hasattr(self.drives, "get_imperative"):
                        imperative = await self.drives.get_imperative()

                    # Idle Monologue (Show signs of life)
                    if not imperative:
                        try:
                            idle_thoughts = [
                                "Scanning memory vector space...",
                                "Analyzing recent interactions for patterns...",
                                "Optimizing internal drive weights...",
                                "Reviewing recent skill performance...",
                                "Considering potential future goals...",
                            ]
                            from core.thought_stream import get_emitter

                            get_emitter().emit(
                                "Monologue", random.choice(idle_thoughts), level="info"
                            )
                        except (ImportError, AttributeError, RuntimeError) as e:
                            _record_conversation_loop_degradation(
                                e,
                                stage="idle_monologue_emit",
                                action="continued autonomous loop after optional idle monologue emission failed",
                                severity="warning",
                            )
                            logger.debug("Idle monologue emit failed: %s", e)

                    if imperative:
                        logger.info("🧠 Autonomous imperative: %s", imperative)
                        # EMIT THOUGHT
                        try:
                            from core.thought_stream import get_emitter

                            get_emitter().emit("Autonomous Goal", imperative, level="goal")
                        except (ImportError, AttributeError, RuntimeError) as e:
                            _record_conversation_loop_degradation(
                                e,
                                stage="autonomous_goal_emit",
                                action="continued autonomous goal execution after thought-stream emission failed",
                                severity="warning",
                            )
                            logger.debug("Autonomous goal emit failed: %s", e)

                        # We are now in an asyncio task, so just await it directly
                        try:
                            # Await the goal with a timeout
                            await asyncio.wait_for(
                                self._execute_autonomous_goal(imperative), timeout=60.0
                            )
                        except TimeoutError:
                            timeout_exc = TimeoutError(
                                "Autonomous goal execution timed out after 60s"
                            )
                            _record_conversation_loop_degradation(
                                timeout_exc,
                                stage="autonomous_goal_timeout",
                                action="abandoned autonomous goal after bounded execution timeout",
                                severity="degraded",
                            )
                            logger.error("Autonomous goal execution timed out after 60s")
                        except asyncio.CancelledError:
                            raise
                        except _RECOVERABLE_LOOP_ERRORS as e:
                            _record_conversation_loop_degradation(
                                e,
                                stage="autonomous_goal_background",
                                action="kept background loop alive after autonomous goal execution failed",
                                severity="degraded",
                            )
                            logger.error("Autonomous goal execution error: %s", e)

                        self.last_autonomous_action = now
                        self.stats["autonomous_goals"] += 1

                # Sleep briefly to yield control to the event loop
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except (ImportError, AttributeError, RuntimeError) as e:
                self._last_loop_error = f"{type(e).__name__}: {e}"
                _record_conversation_loop_degradation(
                    e,
                    stage="background_loop",
                    action="kept autonomous loop alive with fixed recovery sleep after background tick failed",
                    severity="degraded",
                )
                logger.error("Background loop error: %s", e)
                # Ensure exponential backoff instead of instant hot-loop
                await asyncio.sleep(10)

        logger.info("Background autonomous loop stopped")

    async def _execute_autonomous_goal(self, goal_text: str) -> dict[str, Any]:
        """Execute an autonomous goal generated internally (Async).

        Args:
            goal_text: The autonomous goal to pursue

        """
        try:
            logger.info("⚡ Executing autonomous goal: %s", goal_text)

            # Decompose into plan
            plan = await self.planner.decompose(goal_text)

            if not plan or not plan.get("tool_calls"):
                logger.warning("Autonomous goal produced no actionable plan")
                self._autonomous_failure_streak += 1
                return {"ok": False, "error": "no_actionable_plan", "results": []}

            # Execute the plan
            results = await self._execute_plan(plan)

            # Store results in memory for context
            summary = self._summarize_results(results)
            await self.memory.add(
                text=f"Autonomous action: {goal_text}\nResults: {summary}",
                metadata={"type": "autonomous", "timestamp": time.time()},
            )

            # Update drives based on success
            if hasattr(self.drives, "satisfy"):
                if any(r.get("ok") for r in results):
                    self.drives.satisfy("curiosity", 15.0)
                    self.drives.satisfy("competence", 10.0)
                    self._autonomous_failure_streak = 0
                    logger.info("✓ Autonomous goal completed successfully")
                    return {"ok": True, "results": results}
                else:
                    self.drives.punish("competence", 10.0)
                    self._autonomous_failure_streak += 1
                    logger.warning("✗ Autonomous goal failed")
            return {"ok": any(r.get("ok") for r in results), "results": results}
        except _RECOVERABLE_LOOP_ERRORS as e:
            self._autonomous_failure_streak += 1
            self._last_loop_error = f"{type(e).__name__}: {e}"
            _record_conversation_loop_degradation(
                e,
                stage="autonomous_goal",
                action="punished competence and abandoned autonomous goal after recoverable execution failure",
                severity="degraded",
                extra={"failure_streak": self._autonomous_failure_streak},
            )
            logger.error("Autonomous goal execution failed: %s", e, exc_info=True)
            if hasattr(self.drives, "punish"):
                self.drives.punish("competence", 15.0)
            return {"ok": False, "error": type(e).__name__, "detail": str(e), "results": []}

    async def _execute_plan(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a plan's tool calls sequentially (Async).

        Args:
            plan: Plan dictionary with tool_calls

        Returns:
            List of execution results

        """
        tool_calls = plan.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            error = TypeError("plan.tool_calls must be a list")
            _record_conversation_loop_degradation(
                error,
                stage="plan_shape",
                action="returned empty execution results because plan tool_calls had invalid shape",
                severity="degraded",
            )
            self._plan_failure_streak += 1
            self._last_loop_error = f"{type(error).__name__}: {error}"
            return []
        results: list[dict[str, Any]] = []

        # Build execution context
        context = {
            "conversation_history": self.conversation_history[-10:],
            "headless": True,  # Default to headless browser
        }

        # EMIT PLAN to Neural Stream
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Plan Execution", f"Executing {len(tool_calls)} steps", level="system"
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_conversation_loop_degradation(
                e,
                stage="plan_execution_emit",
                action="continued plan execution after thought-stream plan emission failed",
                severity="warning",
            )
            logger.debug("Plan execution emit failed: %s", e)

        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("tool", "unknown")
            logger.info("Executing tool call %d/%d: %s", i + 1, len(tool_calls), tool_name)

            try:
                # Create properly formatted goal dict for executor
                objective_str = tool_call.get("objective", "")
                if not objective_str and tool_name:
                    objective_str = f"use {tool_name} skill"

                goal = {
                    "objective": objective_str,
                    "tool": tool_name,
                    "params": tool_call.get("params", {}),
                    "id": f"tc_{i}",
                }

                # Execute via SkillRouter (Async)
                result = await self.executor.execute(goal, context)

                # Handle None results gracefully
                if result is None:
                    result = {"ok": False, "error": "no_result", "skill": tool_name}

                results.append(result)

                # Update stats
                if result.get("ok"):
                    self.stats["successful_executions"] += 1
                else:
                    self.stats["failed_executions"] += 1
                    logger.warning(
                        "Tool '%s' returned: %s", tool_name, result.get("error", "unknown error")
                    )

                # Small delay between calls (Async)
                await asyncio.sleep(0.5)

            except _RECOVERABLE_LOOP_ERRORS as e:
                self._plan_failure_streak += 1
                self._last_loop_error = f"{type(e).__name__}: {e}"
                self.stats["recovered_tool_failures"] += 1
                _record_conversation_loop_degradation(
                    e,
                    stage="tool_call_execution",
                    action="recorded failed tool result and continued remaining plan steps",
                    severity="degraded",
                    extra={
                        "tool": str(tool_name),
                        "step": i,
                        "failure_streak": self._plan_failure_streak,
                    },
                )
                logger.error("Tool call execution failed for '%s': %s", tool_name, e, exc_info=True)
                results.append(
                    {
                        "ok": False,
                        "error": "execution_exception",
                        "skill": tool_name,
                        "detail": str(e),
                    }
                )
                self.stats["failed_executions"] += 1
            else:
                if results and results[-1].get("ok"):
                    self._plan_failure_streak = 0

        return results

    async def _generate_conversational_response(self, user_message: str) -> str:
        """Generate a conversational response using the cognitive engine (Async).

        Args:
            user_message: User's message

        Returns:
            Generated response

        """
        try:
            # Import identity lock and meta-filter
            from core.synthesis import IDENTITY_LOCK, strip_meta_commentary

            # Build context from conversation history
            history_text = ""
            for entry in self.conversation_history[-8:]:
                role = entry.get("role")
                content = entry.get("content", "")
                if role == "user":
                    history_text += f"User: {content}\n"
                elif role in [self.AI_ROLE, "aura", "model"]:
                    history_text += f"Aura: {content}\n"

            # Build prompt with IDENTITY_LOCK
            prompt = f"""{IDENTITY_LOCK}

FOURTH WALL PROTOCOL (v3.2):
- NEVER narrate your process. Just speak.
- NEVER apologize for apologies.
- You are the subject, not an observer.

Conversation history:
{history_text if history_text else "(Starting new conversation)"}

User: {user_message}

Respond naturally as Aura:
"""

            # Generate response (Async)
            thought = await self.brain.think(prompt)
            response = thought.content if thought else None

            if not response:
                return "I recorded a degraded cognitive cycle instead of inventing an answer."

            # Clean up response
            response = response.strip()

            # Remove any "Aura:" prefix that might have been generated
            if response.startswith("Aura:"):
                response = response[5:].strip()

            # Apply Fourth Wall filter
            response = strip_meta_commentary(response)

            return response

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_conversation_loop_degradation(
                e,
                stage="direct_conversational_response",
                action="returned explicit degraded response after direct generation failed",
                severity="degraded",
            )
            logger.error("Conversational response generation failed: %s", e, exc_info=True)
            return "I encountered an error generating my response. Let me try to help you in another way."

    async def _synthesize_response(
        self, user_message: str, plan: dict[str, Any], results: list[dict[str, Any]]
    ) -> str:
        """Synthesize a natural language response from execution results (Async).

        Args:
            user_message: Original user message
            plan: The executed plan
            results: Execution results

        Returns:
            Synthesized response

        """
        try:
            # Extract key information from results
            summaries = []
            for result in results:
                if result.get("ok"):
                    summary = result.get("summary") or result.get("response") or "Action completed"
                    summaries.append(summary)
                else:
                    error = result.get("error", "unknown_error")
                    summaries.append(f"Error: {error}")

            # Apply Identity Lock to Synthesis (Fix 1)
            try:
                from core.synthesis import IDENTITY_LOCK, strip_meta_commentary

                identity_lock = IDENTITY_LOCK
            except ImportError:
                identity_lock = ""

                def strip_meta_commentary(value):
                    return value

            # Build synthesis prompt
            results_text = "\n".join(f"- {s}" for s in summaries)

            prompt = f"""{identity_lock}

Synthesize a natural conversational response based on these action results.

User asked: {user_message}

Actions taken:
{results_text}

Generate a concise, helpful response (2-4 sentences) that:
1. Addresses what the user asked for
2. Summarizes what was done
3. Provides the key findings or results

CRITICAL: Do NOT halluncinate or make up information. Use ONLY the 'Actions taken' above. If no results are listed, admit you found nothing. Do Not output jinja templates.

Response:"""

            thought = await self.brain.think(prompt)
            response = thought.content if thought else None

            if not response:
                # Fallback to simple concatenation
                return strip_meta_commentary(
                    f"I completed {len(results)} actions: " + ". ".join(summaries[:3])
                )

            return strip_meta_commentary(response.strip())

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_conversation_loop_degradation(
                e,
                stage="response_synthesis",
                action="returned deterministic result summary after response synthesis failed",
                severity="degraded",
            )
            logger.error("Response synthesis failed: %s", e, exc_info=True)
            return f"I completed your request but had trouble summarizing the results. (Error: {str(e)})"

    def _summarize_results(self, results: list[dict[str, Any]]) -> str:
        """Create a brief summary of results"""
        successful = sum(1 for r in results if r.get("ok"))
        total = len(results)

        summaries = []
        for result in results[:3]:  # Only first 3
            if result.get("ok"):
                summaries.append(result.get("summary", "Success"))
            else:
                summaries.append(f"Failed: {result.get('error', 'unknown')}")

        return f"{successful}/{total} successful. " + "; ".join(summaries)

    async def _add_to_history(
        self, role: str, content: str, channel: str = "text", modality: str = "typed"
    ):
        """Add entry to conversation history (Async) and UnifiedTranscript."""
        transcript = get_transcript()
        transcript.add(role, content, channel=channel, modality=modality)
        # UnifiedTranscript handles its own state synchronization and bounds logic

    def get_status(self) -> dict[str, Any]:
        """Get current loop status"""
        return {
            "running": self.is_running,
            "conversation_length": len(self.conversation_history),
            "drives": self.drives.get_status() if hasattr(self.drives, "get_status") else {},
            "stats": self.stats,
            "last_autonomous_action": time.time() - self.last_autonomous_action,
            "autonomous_failure_streak": self._autonomous_failure_streak,
            "plan_failure_streak": self._plan_failure_streak,
            "last_loop_error": self._last_loop_error[:160],
        }
