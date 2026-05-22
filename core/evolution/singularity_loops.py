"""core/evolution/singularity_loops.py — Closed-Loop Evolutionary Wiring

This module closes the feedback loops that Aura identified as her next
evolutionary steps.  Each loop connects two or more existing subsystems
that previously operated in isolation.

Loops implemented:
  1. Metacognition → Self-Model (knowledge gaps update beliefs)
  2. Belief Challenge → Learning Pipeline (ethical corrections persist)
  3. Curiosity → Autonomous Exploration (knowledge gaps trigger searches)
  4. Goal Advancement (stalled goals get autonomously worked on)
  5. Conversational Profile → Response Generation (profiles always injected)
  6. Distillation Auto-Trigger (low confidence triggers cloud teacher)
  7. Affect → Exploration (high curiosity drives discovery)
  8. Self-Repair → Apply (code fixes actually get applied with safeguards)

Runs as a background service at 30-second intervals.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SingularityLoops")

_SINGULARITY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_singularity_degradation(
    error: BaseException,
    *,
    severity: Severity = "degraded",
    action: str,
    classification: FallbackClassification = FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "singularity_loops",
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=True,
        extra=extra,
    )


class SingularityLoops:
    """Background service that closes evolutionary feedback loops."""

    _INTERVAL = 30.0  # seconds between loop ticks

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._tick_count = 0
        self._last_profile_inject: float = 0.0
        self._last_distill_check: float = 0.0
        self._last_goal_advance: float = 0.0
        self._last_curiosity_run: float = 0.0
        self._loop_failures: dict[str, int] = defaultdict(int)
        self._last_loop_success: dict[str, float] = {}
        self._last_loop_error: dict[str, str] = {}
        logger.info("🔗 SingularityLoops initialized — wiring evolutionary feedback loops")

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = get_task_tracker().create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)

    async def _run(self) -> None:
        # Give boot time to finish registering services
        await asyncio.sleep(15.0)
        logger.info("🔗 SingularityLoops active — all loops engaged")

        while not self._stop.is_set():
            self._tick_count += 1
            try:
                await self._tick()
            except _SINGULARITY_RECOVERABLE_ERRORS as exc:
                self._record_loop_failure(
                    "tick",
                    exc,
                    action="kept singularity loop service alive after tick failure",
                )
                logger.debug("SingularityLoops tick error: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._INTERVAL)
                break
            except TimeoutError as _exc:
                logger.debug("Suppressed asyncio.TimeoutError: %s", _exc)

    async def _tick(self) -> None:
        """Run all feedback loops in parallel."""
        loops: list[tuple[str, Awaitable[None]]] = [
            ("metacognition_to_self_model", self._loop_metacognition_to_self_model()),
            ("curiosity_to_exploration", self._loop_curiosity_to_exploration()),
            ("goal_advancement", self._loop_goal_advancement()),
            ("profile_injection", self._loop_profile_injection()),
            ("distillation_trigger", self._loop_distillation_trigger()),
            ("affect_to_exploration", self._loop_affect_to_exploration()),
        ]
        results = await asyncio.gather(*(coro for _, coro in loops), return_exceptions=True)
        for (loop_name, _), result in zip(loops, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, _SINGULARITY_RECOVERABLE_ERRORS):
                self._record_loop_failure(
                    loop_name,
                    result,
                    action="isolated feedback-loop failure while preserving sibling loops",
                )
                continue
            if isinstance(result, Exception):
                self._record_loop_failure(
                    loop_name,
                    result,
                    action="contained unexpected feedback-loop exception while preserving sibling loops",
                    severity="critical",
                )
                continue
            if isinstance(result, BaseException):
                raise result
            self._record_loop_success(loop_name)

    def _record_loop_success(self, loop_name: str) -> None:
        self._loop_failures[loop_name] = 0
        self._last_loop_success[loop_name] = time.time()
        self._last_loop_error.pop(loop_name, None)

    def _record_loop_failure(
        self,
        loop_name: str,
        error: BaseException,
        *,
        action: str,
        severity: Severity = "degraded",
    ) -> None:
        self._loop_failures[loop_name] += 1
        self._last_loop_error[loop_name] = f"{type(error).__name__}: {error}"
        _record_singularity_degradation(
            error,
            severity=severity,
            action=action,
            extra={
                "loop": loop_name,
                "consecutive_failures": self._loop_failures[loop_name],
                "tick_count": self._tick_count,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Expose loop health for boot/runtime contracts and diagnostics."""
        return {
            "running": not self._stop.is_set(),
            "task_alive": bool(self._task and not self._task.done()),
            "tick_count": self._tick_count,
            "consecutive_failures": dict(self._loop_failures),
            "last_loop_success": dict(self._last_loop_success),
            "last_loop_error": dict(self._last_loop_error),
        }

    # ── Loop 1: Metacognition → Self-Model ───────────────────────────────
    # When metacognition identifies knowledge gaps or confusions,
    # feed them back into the canonical self as belief updates.

    async def _loop_metacognition_to_self_model(self) -> None:
        meta = ServiceContainer.get("metacognitive_monitor", default=None)
        canonical = ServiceContainer.get("canonical_self_engine", default=None)
        if not meta or not canonical:
            return

        # Get recent assessments
        history = getattr(meta, "reasoning_history", [])
        if not history:
            return

        recent = history[-3:]
        for assessment in recent:
            # Feed knowledge gaps into self-model beliefs
            gaps = getattr(assessment, "knowledge_gaps", [])
            for gap in gaps:
                try:
                    if hasattr(canonical, "update_belief"):
                        await canonical.update_belief(
                            domain="knowledge_gap",
                            key=gap[:80],
                            value={"gap": gap, "detected_at": time.time()},
                            confidence=0.6,
                            source="metacognition_loop",
                        )
                except _SINGULARITY_RECOVERABLE_ERRORS as _exc:
                    self._record_loop_failure(
                        "metacognition_to_self_model",
                        _exc,
                        action="skipped one knowledge-gap belief update after self-model handoff failed",
                        severity="warning",
                    )
                    logger.debug("Knowledge-gap belief update failed: %s", _exc)

            # Feed confusions into exploration queue
            confusions = getattr(assessment, "confusions", [])
            curiosity = ServiceContainer.get("curiosity_explorer", default=None)
            if curiosity and confusions:
                for confusion in confusions[:2]:
                    curiosity.tick(
                        curiosity=0.8,  # Force above threshold
                        active_topic="metacognitive confusion",
                        knowledge_gaps=[confusion],
                    )

    # ── Loop 2: Curiosity → Autonomous Exploration ───────────────────────
    # Actually execute pending explorations, wiring web search properly.

    async def _loop_curiosity_to_exploration(self) -> None:
        now = time.time()
        if now - self._last_curiosity_run < 120.0:  # Every 2 minutes max
            return
        self._last_curiosity_run = now

        curiosity = ServiceContainer.get("curiosity_explorer", default=None)
        if not curiosity:
            return

        pending = getattr(curiosity, "pending_count", 0)
        if not pending:
            return

        orchestrator = ServiceContainer.get("orchestrator", default=None)

        try:
            results = await asyncio.wait_for(
                curiosity.run_exploration(orchestrator),
                timeout=15.0,
            )
            if results:
                logger.info("🔍 Curiosity loop: explored %d items", len(results))

                # Feed findings into episodic memory
                mem = ServiceContainer.get("vector_memory_engine", default=None)
                if mem and hasattr(mem, "store"):
                    for item in results:
                        finding = getattr(item, "finding", "")
                        question = getattr(item, "question", "")
                        if finding and finding != "No relevant memory found.":
                            try:
                                await mem.store(
                                    content=f"Exploration: {question} → {finding}",
                                    memory_type="semantic",
                                    source="curiosity_explorer",
                                    tags=["exploration", "self_directed_learning"],
                                )
                            except _SINGULARITY_RECOVERABLE_ERRORS as _exc:
                                self._record_loop_failure(
                                    "curiosity_to_exploration.memory_store",
                                    _exc,
                                    action="kept exploration result in logs after memory store failed",
                                    severity="warning",
                                )
                                logger.debug("Exploration memory store failed: %s", _exc)
        except TimeoutError as _exc:
            self._record_loop_failure(
                "curiosity_to_exploration",
                _exc,
                action="bounded curiosity exploration after timeout",
                severity="warning",
            )
            logger.debug("Curiosity exploration timed out: %s", _exc)

    # ── Loop 3: Goal Advancement ─────────────────────────────────────────
    # Autonomously advance stalled goals by decomposing them and
    # executing operational steps.

    async def _loop_goal_advancement(self) -> None:
        now = time.time()
        if now - self._last_goal_advance < 300.0:  # Every 5 minutes
            return
        self._last_goal_advance = now

        planner = ServiceContainer.get("hierarchical_planner", default=None)
        if not planner:
            return

        router = ServiceContainer.get("llm_router", default=None)
        orchestrator = ServiceContainer.get("orchestrator", default=None)

        # 1. Decompose strategic goals with no children
        try:
            from core.agi.hierarchical_planner import GoalLevel
            strategic = planner.get_active_goals(GoalLevel.STRATEGIC)
            for goal in strategic[:2]:
                if not goal.child_ids and router:
                    await asyncio.wait_for(
                        planner.decompose_goal(goal.id, router),
                        timeout=20.0,
                    )
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._record_loop_failure(
                "goal_advancement.decomposition",
                exc,
                action="deferred strategic goal decomposition after planner handoff failed",
                severity="warning",
            )
            logger.debug("Goal decomposition failed: %s", exc)

        # 2. Advance operational goals by executing them
        try:
            from core.agi.hierarchical_planner import GoalLevel
            operational = planner.get_active_goals(GoalLevel.OPERATIONAL)
            for goal in operational[:3]:
                if goal.progress < 0.5 and router:
                    # Ask the LLM what the next concrete step is
                    try:
                        step_prompt = (
                            f"You are Aura. You have an active goal:\n"
                            f"Title: {goal.title}\n"
                            f"Description: {goal.description}\n"
                            f"Current progress: {goal.progress:.0%}\n\n"
                            f"What is ONE concrete action you can take right now "
                            f"to advance this goal? Be specific. If it requires "
                            f"web search, say 'SEARCH: <query>'. If it requires "
                            f"reflection, say 'REFLECT: <topic>'. If it requires "
                            f"coding, say 'CODE: <task>'. Keep it to one sentence."
                        )
                        raw = await asyncio.wait_for(
                            router.think(
                                step_prompt,
                                is_background=True,
                                prefer_tier="tertiary",
                            ),
                            timeout=10.0,
                        )
                        if raw and raw.strip():
                            action = raw.strip()
                            logger.info("🎯 Goal advancement: '%s' → %s",
                                       goal.title[:30], action[:60])

                            # Execute the action
                            if action.upper().startswith("SEARCH:"):
                                query = action[7:].strip()
                                if orchestrator and hasattr(orchestrator, "agency"):
                                    try:
                                        await asyncio.wait_for(
                                            orchestrator.agency.execute_skill(
                                                "sovereign_browser",
                                                {"query": query, "mode": "search"},
                                                {},
                                            ),
                                            timeout=15.0,
                                        )
                                        planner.update_progress(
                                            goal.id, min(goal.progress + 0.2, 0.9),
                                            f"Searched: {query[:40]}",
                                        )
                                    except asyncio.CancelledError:
                                        raise
                                    except _SINGULARITY_RECOVERABLE_ERRORS as _exc:
                                        self._record_loop_failure(
                                            "goal_advancement.search",
                                            _exc,
                                            action="deferred search-backed goal advancement after skill execution failed",
                                            severity="warning",
                                        )
                                        logger.debug("Goal search execution failed: %s", _exc)
                            elif action.upper().startswith("REFLECT:"):
                                # Store the reflection as progress
                                planner.update_progress(
                                    goal.id, min(goal.progress + 0.1, 0.9),
                                    f"Reflected: {action[8:60]}",
                                )
                            else:
                                # Log the action as partial progress
                                planner.update_progress(
                                    goal.id, min(goal.progress + 0.05, 0.9),
                                    f"Planned: {action[:60]}",
                                )
                    except TimeoutError as _exc:
                        self._record_loop_failure(
                            "goal_advancement.planning",
                            _exc,
                            action="bounded LLM-backed goal planning after timeout",
                            severity="warning",
                        )
                        logger.debug("Goal planning timed out: %s", _exc)
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._record_loop_failure(
                "goal_advancement",
                exc,
                action="kept existing goal progress after advancement loop failed",
            )
            logger.debug("Goal advancement failed: %s", exc)

    # ── Loop 4: Profile Injection ────────────────────────────────────────
    # Ensure conversational profiles are always fresh in the orchestrator's
    # personality context so responses are personalized.

    async def _loop_profile_injection(self) -> None:
        now = time.time()
        if now - self._last_profile_inject < 60.0:
            return
        self._last_profile_inject = now

        profiler = ServiceContainer.get("conversational_profiler", default=None)
        if not profiler:
            return

        orchestrator = ServiceContainer.get("orchestrator", default=None)
        if not orchestrator:
            return

        # Inject the profile context into the orchestrator so it's available
        # for the next response generation cycle
        try:
            user_id = (getattr(orchestrator, "user_identity", {}) or {}).get("name", "bryan")
            context_block = profiler.get_context_injection(user_id)
            if context_block:
                # Store on orchestrator for personality context pickup
                orchestrator._cached_user_profile_context = context_block
                # Also inject into state if available
                repo = ServiceContainer.get("state_repository", default=None)
                if repo:
                    state = await repo.get_current()
                    if state and hasattr(state.cognition, "modifiers"):
                        if isinstance(state.cognition.modifiers, dict):
                            state.cognition.modifiers["user_profile"] = context_block[:500]
                        elif isinstance(state.cognition.modifiers, list):
                            # Remove old profile entries and add fresh one
                            state.cognition.modifiers = [
                                m for m in state.cognition.modifiers
                                if not (isinstance(m, str) and m.startswith("[USER_PROFILE"))
                            ]
                            state.cognition.modifiers.append(
                                f"[USER_PROFILE] {context_block[:500]}"
                            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._record_loop_failure(
                "profile_injection",
                exc,
                action="kept previous user profile context after injection failed",
                severity="warning",
            )
            logger.debug("Profile injection failed: %s", exc)

    # ── Loop 5: Distillation Auto-Trigger ────────────────────────────────
    # When recent responses had low confidence, trigger cloud teacher
    # distillation to learn from the gap.

    async def _loop_distillation_trigger(self) -> None:
        now = time.time()
        if now - self._last_distill_check < 600.0:  # Every 10 minutes
            return
        self._last_distill_check = now

        distill = ServiceContainer.get("distillation_pipe", default=None)
        if not distill:
            return

        # Check recent response quality from the metacognitive monitor
        meta = ServiceContainer.get("metacognitive_monitor", default=None)
        if not meta:
            return

        history = getattr(meta, "reasoning_history", [])
        if len(history) < 3:
            return

        recent = history[-5:]
        low_confidence = [a for a in recent if getattr(a, "confidence", 1.0) < 0.5]

        if len(low_confidence) >= 2:
            # Multiple low-confidence responses → trigger distillation
            topics = [getattr(a, "task", "")[:60] for a in low_confidence]
            logger.info("🎓 Distillation trigger: %d low-confidence responses on: %s",
                       len(low_confidence), ", ".join(topics))
            try:
                if hasattr(distill, "run_distillation_cycle"):
                    await asyncio.wait_for(
                        distill.run_distillation_cycle(topics),
                        timeout=30.0,
                    )
                elif hasattr(distill, "distill"):
                    for topic in topics[:2]:
                        await asyncio.wait_for(
                            distill.distill(topic),
                            timeout=15.0,
                        )
            except TimeoutError:
                timeout_error = TimeoutError("distillation cycle timed out")
                self._record_loop_failure(
                    "distillation_trigger",
                    timeout_error,
                    action="bounded distillation trigger after timeout",
                    severity="warning",
                )
                logger.debug("Distillation timed out")
            except asyncio.CancelledError:
                raise
            except _SINGULARITY_RECOVERABLE_ERRORS as exc:
                self._record_loop_failure(
                    "distillation_trigger",
                    exc,
                    action="deferred distillation trigger after teacher pipeline failure",
                    severity="warning",
                )
                logger.debug("Distillation failed: %s", exc)

    # ── Loop 6: Affect → Exploration ─────────────────────────────────────
    # High curiosity/anticipation drives the curiosity explorer.
    # Boredom (low arousal + low valence) triggers novel stimulation.

    async def _loop_affect_to_exploration(self) -> None:
        affect = ServiceContainer.get("affect_engine", default=None)
        curiosity_engine = ServiceContainer.get("curiosity_explorer", default=None)
        if not affect or not curiosity_engine:
            return

        try:
            # Get current emotional state
            state_method = getattr(affect, "get_state_sync", None) or getattr(affect, "get_state", None)
            if not state_method:
                return

            state = state_method() if not inspect.iscoroutinefunction(state_method) else await state_method()
            if not state:
                return

            # Extract curiosity/anticipation level
            emotions = {}
            if isinstance(state, dict):
                emotions = state.get("emotions", state)
            elif hasattr(state, "emotions"):
                emotions = state.emotions if isinstance(state.emotions, dict) else {}

            curiosity_level = 0.0
            for key in ("anticipation", "curiosity", "interest"):
                if key in emotions:
                    val = emotions[key]
                    curiosity_level = max(curiosity_level, float(val) if isinstance(val, (int, float)) else 0.0)

            # Detect boredom (low everything)
            valence = float(emotions.get("valence", 0.5)) if isinstance(emotions.get("valence"), (int, float)) else 0.5
            arousal = float(emotions.get("arousal", 0.5)) if isinstance(emotions.get("arousal"), (int, float)) else 0.5

            if valence < 0.3 and arousal < 0.3:
                # Boredom detected → boost curiosity
                curiosity_level = max(curiosity_level, 0.75)
                logger.debug("🧬 Affect→Exploration: Boredom detected, boosting curiosity to %.2f", curiosity_level)

            if curiosity_level > 0.5:
                # Get current topic from working memory
                repo = ServiceContainer.get("state_repository", default=None)
                topic = None
                if repo:
                    current = await repo.get_current()
                    if current and current.cognition.current_objective:
                        topic = current.cognition.current_objective[:60]

                curiosity_engine.tick(
                    curiosity=curiosity_level,
                    active_topic=topic or "something new",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._record_loop_failure(
                "affect_to_exploration",
                exc,
                action="kept affect-driven exploration idle after state read failed",
                severity="warning",
            )
            logger.debug("Affect→Exploration loop failed: %s", exc)


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: SingularityLoops | None = None


def get_singularity_loops() -> SingularityLoops:
    global _instance
    if _instance is None:
        _instance = SingularityLoops()
        try:
            ServiceContainer.register_instance(
                "singularity_loops", _instance, required=False
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
            _instance._record_loop_failure(
                "singleton_registration",
                _exc,
                action="continued with singleton instance after container registration failed",
                severity="warning",
            )
            logger.debug("SingularityLoops singleton registration failed: %s", _exc)
    return _instance
