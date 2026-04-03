"""Context Streaming Mixin for RobustOrchestrator.
Extracts context gathering, chat streaming, and history management logic.
"""
import asyncio
import inspect
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _dispose_awaitable(result: Any) -> None:
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()


class ContextStreamingMixin:
    """Handles agentic context gathering, chat streaming, and history ops."""

    async def _gather_agentic_context(self, message: str) -> dict[str, Any]:
        """Collect memories, stats, and world state for reasoning."""
        # 0. User Identity Detection
        user_identity = self._detect_user_identity(message) or {"name": "Stranger", "role": "Unknown", "relation": "Neutral"}

        # 1. Meta-Recall & Memory Query (Parallel)
        tasks = []
        if hasattr(self, 'meta_learning') and self.meta_learning:
            recall_result = self.meta_learning.recall_strategy(message)
            if inspect.isawaitable(recall_result):
                tasks.append(recall_result)
            else:
                tasks.append(asyncio.sleep(0, result=recall_result or {}))
        else:
            tasks.append(asyncio.sleep(0, result={}))

        # Add Cold/Deep Memory Context (Vector/Graph)
        if hasattr(self, 'memory') and self.memory:
            u_name = user_identity.get('name', 'Stranger')
            cold_memory_result = self.memory.get_cold_memory_context(f"{u_name}: {message}", limit=5)
            if inspect.isawaitable(cold_memory_result):
                tasks.append(asyncio.create_task(cold_memory_result))
            else:
                tasks.append(asyncio.sleep(0, result=cold_memory_result or ""))
        else:
            tasks.append(asyncio.sleep(0, result=""))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 2. Personality & World state
        personality_data = self._get_personality_data()
        personality_ctx = self._stringify_personality(personality_data)
        world_ctx = self._get_world_context()
        env_ctx = await self._get_environmental_context()

        ctx = {
            "history": self._get_cleaned_history_context(15)["history"],
            "personality": personality_data,
            "personality_string": personality_ctx,
            "world": world_ctx,
            "environment": env_ctx,
            "user": user_identity,
            "meta_learning": results[0] if len(results) > 0 and not isinstance(results[0], Exception) else {},
            "unified_memory": f"[INTERNAL RECALL]: {results[1]}" if results[1] and not isinstance(results[1], Exception) else "",
            "inner_monologue": "",
            "focus": "STAY ON TOPIC. Prioritize the user's latest request above autonomous impulses."
        }
        if hasattr(self, "substrate"):
            try:
                inner_monologue = self.substrate.get_latest_monologue()
                if inspect.isawaitable(inner_monologue):
                    _dispose_awaitable(inner_monologue)
                else:
                    ctx["inner_monologue"] = inner_monologue or ""
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        # Sentient Context Injection: Affect & Drives
        ctx["emotional_state"] = "Stable"
        if getattr(self, 'affect_engine', None):
            try:
                if hasattr(self.affect_engine, 'state'):
                    ctx["emotional_state"] = self.affect_engine.state.dominant_emotion
                elif hasattr(self.affect_engine, 'get_mood'):
                    mood = self.affect_engine.get_mood()
                    if inspect.isawaitable(mood):
                        _dispose_awaitable(mood)
                    else:
                        ctx["emotional_state"] = mood
            except Exception as e:
                logger.debug("Affect extraction skipped: %s", e)

        # Theory of Mind Projection
        if self.mind_model:
            try:
                theory_of_mind = self.mind_model.get_context_for_brain()
                if inspect.isawaitable(theory_of_mind):
                    _dispose_awaitable(theory_of_mind)
                    theory_of_mind = {}
                ctx["theory_of_mind"] = theory_of_mind
            except Exception:
                ctx["theory_of_mind"] = {}
        else:
            ctx["theory_of_mind"] = {}

        # Social Relationship context
        if hasattr(self, 'social') and self.social:
            try:
                social_context = self.social.get_social_context()
                if inspect.isawaitable(social_context):
                    _dispose_awaitable(social_context)
                    social_context = ""
                ctx["social_narrative"] = social_context
            except Exception:
                ctx["social_narrative"] = ""

            # Passive depth increase with robust guard for None
            if hasattr(self.social, "relationship_depth") and self.social.relationship_depth is not None:
                try:
                    # Capture current value safely
                    curr = self.social.relationship_depth
                    if isinstance(curr, (int, float)):
                        self.social.relationship_depth = min(1.0, float(curr) + 0.001)
                except (TypeError, ValueError, AttributeError):
                    logger.debug("Fast-path tool call skipped or failed.")

        # Strategic Project Context
        if self.strategic_planner and self.project_store:
            try:
                active_projects = self.project_store.get_active_projects()
                if inspect.isawaitable(active_projects):
                    _dispose_awaitable(active_projects)
                    active_projects = []
                if active_projects:
                    proj = active_projects[0]
                    next_task = self.strategic_planner.get_next_task(proj.id)
                    if inspect.isawaitable(next_task):
                        _dispose_awaitable(next_task)
                        next_task = None
                    all_tasks = self.project_store.get_tasks_for_project(proj.id)
                    if inspect.isawaitable(all_tasks):
                        _dispose_awaitable(all_tasks)
                        all_tasks = []
                    ctx["strategic_context"] = {
                        "project_name": proj.name,
                        "project_goal": proj.goal,
                        "current_task": next_task.description if next_task else "No pending tasks",
                        "backlog": [f"{t.status.upper()}: {t.description}" for t in all_tasks]
                    }
                    logger.debug("Strategic context injected for project: %s", proj.name)
            except Exception as e:
                logger.error("Failed to inject strategic context: %s", e)

        # Core Engines
        from ..container import ServiceContainer
        from core.utils.concurrency import RobustLock

        self._cognitive_engine = ServiceContainer.get("cognitive_engine", default=None)
        self._memory = ServiceContainer.get("memory_facade", default=None)
        if self._memory and hasattr(self._memory, 'setup'):
            setup_result = self._memory.setup()
            if inspect.isawaitable(setup_result):
                _dispose_awaitable(setup_result)

        self._capability_engine = ServiceContainer.get("capability_engine", default=None)
        self._scratchpad_engine = ServiceContainer.get("scratchpad_engine", default=None)

        self._cognition = ServiceContainer.get("cognitive_integration", default=None)
        if self._cognition and hasattr(self._cognition, 'setup'):
            setup_result = self._cognition.setup()
            if inspect.isawaitable(setup_result):
                _dispose_awaitable(setup_result)

        self.output_gate = ServiceContainer.get("output_gate", default=None)

        # Z-10 Fix: Moved cache and lock initialization to boot/setup
        # to prevent resetting deduplication state on every message.
        if not hasattr(self, "_input_hash_cache"):
            import collections
            self._input_hash_cache = collections.deque(maxlen=5)
        if not hasattr(self, "_input_lock"):
            self._input_lock = RobustLock("Orchestrator.InputLock")

        self._agency_core = ServiceContainer.get("agency_core", default=None)
        if getattr(self, 'drive_engine', None):
            try:
                drives = self.drive_engine.get_drives() if hasattr(self.drive_engine, 'get_drives') else {"curiosity": 0.5, "energy": 0.8}
                if inspect.isawaitable(drives):
                    _dispose_awaitable(drives)
                    drives = {"curiosity": 0.5, "energy": 0.8}
                ctx["metabolic_drives"] = drives
            except Exception as e:
                logger.error("Drive extraction failed: %s", e)

        # Legacy CognitiveIntegration Integration
        cog_integration = ServiceContainer.get("cognitive_integration", default=None)
        if cog_integration and hasattr(cog_integration, "build_enhanced_context"):
            try:
                emotional_val = 0.5
                if getattr(self, 'liquid_state', None):
                     emotional_val = getattr(self.liquid_state, 'intensity', 0.5)

                enhanced_ctx_str = await cog_integration.build_enhanced_context(message, emotional_context=emotional_val)
                if enhanced_ctx_str:
                    ctx["advanced_cognition"] = enhanced_ctx_str
            except Exception as e:
                logger.debug("Enhanced context unavailable: %s", e)

        # Tool Recommendation Injection
        try:
            from core.memory.learning.tool_learning import tool_learner
            category = tool_learner.classify_task(message)
            if inspect.isawaitable(category):
                _dispose_awaitable(category)
                category = None
            recommendations = tool_learner.recommend_tools(category)
            if inspect.isawaitable(recommendations):
                _dispose_awaitable(recommendations)
                recommendations = []
            if recommendations:
                ctx["tool_recommendations"] = {
                    "category": category,
                    "recommended_tools": recommendations
                }
                logger.info("🛠️ Tool Recommendations: %s -> %s", category, recommendations)
        except Exception as e:
            logger.debug("Tool recommendations failed: %s", e)

        return ctx

    async def chat_stream(self, message: str):
        """Stream tokens from the cognitive engine.
        Bypasses wait-loops and queues for maximum speed.
        """
        from core.brain.types import ThinkingMode

        self.status.is_processing = True
        try:
            # Check reflexes first
            reflex = self._check_reflexes(message)
            if reflex:
                yield reflex
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": self.AI_ROLE, "content": reflex})
                return

            # Determine thinking tier
            tier = "light"
            try:
                from core.ops.thinking_mode import ModeRouter
                tier = ModeRouter(self.reflex_engine).route(message).value
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
            # Build objective
            # Cleaned history for streaming chat speed (Fix: No leaks)
            context = self._get_cleaned_history_context(8)
            context_injection_error = None

            # Inject LiquidState into Tool Execution Context
            try:
                from core.container import get_container
                container = get_container()
                ls = container.get('liquid_state')
                context['liquid_state'] = ls.get_status()
                logger.debug("TOOL EXECUTION: Injected liquid_state: %s", context['liquid_state'])
            except Exception as e:
                context_injection_error = e
                logger.warning("TOOL EXECUTION: LiquidState injection failed: %s", e)

            # Start Stream
            token_buffer = ""
            payload_context = {}
            if hasattr(self.cognitive_engine, "think_stream"):
                async for token in self.cognitive_engine.think_stream(message, context=context, tier=tier):
                    if not payload_context.get("stream_started"):
                        payload_context["stream_started"] = True
                    token_buffer += token
                    yield token
            else:
                # Fallback for legacy/broken instances
                if context_injection_error is not None:
                    raise context_injection_error
                thought = await self.cognitive_engine.think(message, context=context, mode=ThinkingMode.DEEP)
                # FIX: Defensive content extraction
                if hasattr(thought, 'content'):
                    token_buffer = thought.content
                elif isinstance(thought, dict):
                    token_buffer = thought.get('content', '')
                else:
                    token_buffer = str(thought)

                if token_buffer:
                    payload_context["stream_started"] = True
                yield self._filter_output(token_buffer)

            # Cleanup: Update history and memory after stream finishes
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": self.AI_ROLE, "content": token_buffer})

            # Satisfy drive — robustly
            if hasattr(self, 'drives') and self.drives and hasattr(self.drives, 'satisfy'):
                try:
                    await self.drives.satisfy("social", 5.0)
                except Exception as _de:
                    logger.debug("Drive satisfaction failed in stream: %s", _de)

        except Exception as e:
            logger.error("Chat stream failed: %s", e)
            # v10.0 Zenith: Silent failure in stream if header already sent
            # but provide clean error formatting for dev
            yield f"\n\n[System Maintenance: {type(e).__name__}]"
        finally:
            self.status.is_processing = False

    async def sentence_stream_generator(self, message: str):
        # Yields complete sentences as they are generated.
        # Perfect for TTS pipe ingestion.
        sentence_delimiters = (".", "?", "!", "\n", ":")
        buffer = ""
        async for token in self.chat_stream(message):
            buffer += token
            # Yield if we find a sentence boundary
            if any(token.endswith(d) for d in sentence_delimiters):
                if buffer.strip():
                    yield buffer.strip()
                    buffer = ""
        # Final flush
        if buffer.strip():
            yield self._filter_output(buffer.strip())

    def _deduplicate_history(self):
        """Remove consecutive identical messages."""
        if not self.conversation_history:
            return

        # Primary safeguard against empty history
        first_msg = self.conversation_history[0] if self.conversation_history else None
        if not first_msg:
            return

        deduped = [first_msg]
        for msg in self.conversation_history[1:]:
            if msg.get("content") != deduped[-1].get("content"):
                deduped.append(msg)
        self.conversation_history = deduped

    async def _prune_history_async(self):
        """Asynchronously prune history via context pruner."""
        try:
            from core.memory.context_pruner import context_pruner
            self.conversation_history = await context_pruner.prune_history(
                self.conversation_history, self.cognitive_engine
            )
        except Exception as e:
            logger.debug("History pruning failed: %s", e)
            if isinstance(self.conversation_history, list) and len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]

    async def _consolidate_long_term_memory(self):
        """Summarize and move important session highlights to long-term vector memory."""
        try:
            # Only consolidate every 15-20 messages to avoid spam
            if len(self.conversation_history) % 15 != 0:
                return

            logger.info("🧠 Consolidating session highlights to long-term memory...")

            # 1. Gather recent dialogue (last 20 messages)
            recent = self.conversation_history[-20:] if isinstance(self.conversation_history, list) else []
            if not recent:
                return
            chat_text = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in recent])

            # 2. Ask the brain to summarize key takeaways/facts
            from core.brain.cognitive_engine import ThinkingMode
            summary_prompt = (
                "Review this recent conversation fragment and extract 3-5 key 'long-term' facts "
                "or user preferences learned. Format as single-sentence declarations. "
                "Focus on what's important for future context, ignoring fluff.\n\n"
                f"Conversation:\n{chat_text}"
            )

            summary_thought = await self.cognitive_engine.think(
                objective=summary_prompt,
                context={"history": []}, # Clean slate for summary
                mode=ThinkingMode.FAST,
                origin="memory_consolidation",
                is_background=True,
            )

            # FIX: Handle both dict and object returns from cognitive engine
            if summary_thought:
                if hasattr(summary_thought, 'content'):
                    highlights = summary_thought.content
                elif isinstance(summary_thought, dict):
                    highlights = summary_thought.get('content', '')
                else:
                    highlights = str(summary_thought)
            else:
                highlights = ''

            if highlights:
                logger.info("✨ Key Highlights Extracted: %s", (highlights or "")[:100])

                # 3. Store in Vector Memory
                if self.memory_manager:
                    await self.memory_manager.log_event(
                        "session_consolidation",
                        highlights,
                        metadata={"type": "summary", "session_start": self.status.start_time}
                    )
                    self._emit_telemetry("Memory", "Session highlights consolidated to long-term storage.")

                # 4. Sentient Unity: Metabolic Archival Compression
                # Periodically compress raw SQLite and text logs so the engine doesn't bloat over infinite horizons
                from ..container import ServiceContainer
                archive_eng = ServiceContainer.get("archive_engine", default=None)
                if archive_eng and hasattr(archive_eng, 'archive_vital_logs'):
                    logger.info("📦 Deep Sleep Cycle: Triggering Metabolic Archival Compression...")
                    await archive_eng.archive_vital_logs()

        except Exception as e:
            logger.error("Memory consolidation failed: %s", e)
