"""core/agency/task_commitment_verifier.py
Task Commitment Verifier
========================
Ensures Aura never verbally commits to a task she cannot execute,
and that her completion/progress reports are grounded in actual
execution state rather than LLM inference.

The gap this closes
-------------------
Without this module, Aura's LLM could say "I'll research X for you" and
inject that text into the conversation even if no skill or task engine
had been triggered.  The user sees a promise; the system never acts.
Worse, a follow-up "are you done?" could produce a hallucinated "yes"
because the LLM has no signal saying the task never ran.

How it works
------------
1. GodModeToolPhase calls `verify_and_dispatch(objective, state)` for any
   request that looks task-like (multi-step, long-horizon, not a single
   reflex skill).

2. The verifier runs a capability pre-check against CapabilityEngine:
   - If matching skills exist → execute immediately (short tasks) or
     launch async and register a Commitment (long tasks).
   - If no match → return a CAPABILITY_GAP result so the LLM tells the
     user honestly what it cannot do today, and what it *can* offer.

3. The result (COMPLETED, STARTED, CAPABILITY_GAP, or DENIED) is injected
   into working_memory as a [TASK_RESULT] system message *before* the
   LLM generates its response.  The response is therefore always grounded
   in the actual execution outcome.

4. Active task statuses are tracked so "are you done with X?" queries can
   be answered accurately from stored state rather than guessed.
"""
from __future__ import annotations

from core.utils.task_tracker import get_task_tracker

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from threading import RLock
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.config import config
from core.runtime.skill_task_bridge import looks_like_multi_step_skill_request
from core.utils.file_utils import atomic_write_json

logger = logging.getLogger("Aura.TaskCommitmentVerifier")
_STATUS_FOLLOWUP_MARKERS = (
    "keep going",
    "keep it going",
    "continue",
    "resume",
    "let's do it",
    "lets do it",
    "do it",
    "carry on",
    "pick it back up",
    "go ahead",
    "finish it",
    "fix it",
    "patch it",
    "are you done",
    "did you finish",
    "still running",
    "progress",
    "status",
    "what happened",
    "follow up",
    "task",
)
_CONTINUATION_MARKERS = (
    "keep going",
    "keep it going",
    "continue",
    "resume",
    "let's do it",
    "lets do it",
    "do it",
    "carry on",
    "pick it back up",
    "go ahead",
    "finish it",
    "fix it",
    "patch it",
    "try again",
)
_CONTINUATION_FILLER_TOKENS = {
    "a", "an", "the", "it", "that", "this", "task", "please", "now",
    "just", "again", "up", "back", "to", "on",
}
_STATUS_QUERY_MARKERS = (
    "are you done",
    "did you finish",
    "still running",
    "progress",
    "status",
    "what happened",
    "follow up",
    "did it work",
    "how did it go",
    "where are you on",
)
_TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled", "capability_gap", "denied"}
_RUNNING_TASK_STATUSES = {"running", "running_async"}
_ACTIVE_TASK_STATUSES = _RUNNING_TASK_STATUSES | {"interrupted"}
_USER_FACING_ORIGINS = {
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "websocket",
    "direct",
    "external",
    "frontend",
    "ui",
}


class DispatchOutcome(str, Enum):
    COMPLETED    = "completed"     # Task ran and verified successful
    STARTED      = "started"       # Long task launched async; commitment registered
    FAILED       = "failed"        # Task ran but execution failed
    CAPABILITY_GAP = "capability_gap"  # No skill/tool to fulfil this request
    DENIED       = "denied"        # Requires approval; pending human sign-off


@dataclass
class CapabilityAssessment:
    """Result of checking whether Aura can fulfil a given request."""
    can_fulfil: bool
    matched_skills: List[str] = field(default_factory=list)
    matched_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0       # 0–1 confidence that matched skills cover the request
    gap_description: str = ""     # Human-readable explanation of what is missing
    alternatives: List[str] = field(default_factory=list)  # What CAN be offered instead


@dataclass
class TaskAcceptance:
    """Structured result returned to GodModeToolPhase for working_memory injection."""
    outcome: DispatchOutcome
    task_id: Optional[str] = None
    commitment_id: Optional[str] = None
    objective: str = ""
    requested_objective: str = ""
    summary: str = ""
    result_data: Any = None
    elapsed_ms: float = 0.0

    def to_working_memory_message(self) -> str:
        """Format for injection into state.cognition.working_memory."""
        icon = {
            DispatchOutcome.COMPLETED:      "✅",
            DispatchOutcome.STARTED:        "🔄",
            DispatchOutcome.FAILED:         "⚠️",
            DispatchOutcome.CAPABILITY_GAP: "❌",
            DispatchOutcome.DENIED:         "🔒",
        }.get(self.outcome, "ℹ️")

        parts = [f"[TASK_RESULT:{self.outcome.value}] {icon} {self.summary}"]
        if self.task_id:
            parts.append(f"task_id={self.task_id}")
        if self.commitment_id:
            parts.append(f"commitment={self.commitment_id}")
        return "  ".join(parts)


class TaskCommitmentVerifier:
    """
    Pre-execution capability gating and result injection for task-class requests.

    Integration point: called from GodModeToolPhase when intent_type is TASK
    (or from a new routing path that upgrades SKILL intents that look multi-step).

    Usage::

        verifier = TaskCommitmentVerifier(kernel)
        acceptance = await verifier.verify_and_dispatch(objective, state)
        # Caller injects acceptance.to_working_memory_message() into state
    """

    # Tasks with ≤ this many expected steps are run inline (response waits for result).
    # Longer tasks are launched async with a Commitment registered.
    INLINE_STEP_THRESHOLD = 4

    # Maximum seconds to wait for an inline task before treating it as async-started.
    INLINE_TIMEOUT_S = 25.0

    def __init__(self, kernel: Any, persist_path: Path | None = None):
        self.kernel = kernel
        base_dir = config.paths.data_dir / "runtime"
        self.persist_path = Path(persist_path or (base_dir / "task_commitment_state.json"))
        self._lock = RLock()
        self._active_tasks: Dict[str, Dict] = {}   # task_id → status record
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._updated_at: float = 0.0
        self._load()

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _coerce_execution_origin(cls, origin: Any) -> str:
        normalized = cls._normalize_origin(origin)
        if not normalized:
            return ""
        if normalized in _USER_FACING_ORIGINS:
            return normalized
        tokens = {token for token in normalized.split("_") if token}
        for candidate in ("user", "api", "voice", "admin", "gui", "websocket", "ws", "direct", "external"):
            if candidate in tokens:
                return candidate
        return normalized

    def _resolve_execution_origin(self, state: Any) -> str:
        response_modifiers = getattr(state, "response_modifiers", {}) if state is not None else {}
        cognition = getattr(state, "cognition", None) if state is not None else None
        candidates = [
            getattr(cognition, "current_origin", ""),
            response_modifiers.get("origin") if isinstance(response_modifiers, dict) else "",
            response_modifiers.get("task_origin") if isinstance(response_modifiers, dict) else "",
            getattr(state, "transition_origin", "") if state is not None else "",
        ]
        for candidate in candidates:
            resolved = self._coerce_execution_origin(candidate)
            if resolved:
                return resolved
        return "commitment_verifier"

    # ── Public API ────────────────────────────────────────────────────────────

    async def verify_and_dispatch(
        self,
        objective: str,
        state: Any,
        force_async: bool = False,
    ) -> TaskAcceptance:
        """Check capability, then execute or register commitment.

        Args:
            objective: The natural-language task request.
            state:     Current AuraState (read for phi, context; not mutated here).
            force_async: If True, always launch async and register a commitment,
                         even for short tasks.

        Returns:
            TaskAcceptance — always. Never raises. Errors become FAILED outcomes.
        """
        t0 = time.monotonic()
        task_id = str(uuid.uuid4())[:8]
        requested_objective = str(objective or "").strip()
        normalized_objective = " ".join(requested_objective.split()).strip()
        objective = requested_objective or normalized_objective
        followup_entry = self._resolve_relevant_entry(normalized_objective)
        resume_plan_id = str((followup_entry or {}).get("plan_id", "") or "")

        if self.is_continuation_request(normalized_objective):
            if followup_entry is None:
                elapsed = (time.monotonic() - t0) * 1000
                return TaskAcceptance(
                    outcome=DispatchOutcome.FAILED,
                    task_id=task_id,
                    objective=requested_objective,
                    requested_objective=requested_objective,
                    summary=(
                        "I don't have a tracked task to continue from that alone. "
                        "I need the actual task, not just 'keep going'."
                    ),
                    elapsed_ms=elapsed,
                )

            reuse_acceptance = self._build_continuation_reuse_acceptance(followup_entry, t0=t0)
            if reuse_acceptance is not None:
                return reuse_acceptance

            objective = str(followup_entry.get("objective", "") or requested_objective).strip() or requested_objective
            normalized_objective = " ".join(objective.split()).strip()

        # 1. Capability pre-check
        assessment = self._assess_capability(normalized_objective or objective)

        if not assessment.can_fulfil:
            elapsed = (time.monotonic() - t0) * 1000
            gap_msg = (
                f"I don't have a way to do that directly. {assessment.gap_description}"
                f"{' I can offer: ' + ', '.join(assessment.alternatives) if assessment.alternatives else ''}"
            )
            logger.info(
                "TaskCommitmentVerifier: CAPABILITY_GAP for '%s'. Gap: %s",
                objective[:60], assessment.gap_description,
            )
            return TaskAcceptance(
                outcome=DispatchOutcome.CAPABILITY_GAP,
                task_id=task_id,
                objective=objective,
                requested_objective=requested_objective,
                summary=gap_msg,
                elapsed_ms=elapsed,
            )

        # 2. Get the task engine
        task_engine = self._get_task_engine()
        if task_engine is None:
            elapsed = (time.monotonic() - t0) * 1000
            return TaskAcceptance(
                outcome=DispatchOutcome.FAILED,
                task_id=task_id,
                objective=objective,
                requested_objective=requested_objective,
                summary="Task engine unavailable. Cannot execute autonomous tasks right now.",
                elapsed_ms=elapsed,
            )

        # 3. Estimate plan size to decide inline vs. async
        estimated_steps = self._estimate_steps(objective, assessment)
        is_long = force_async or estimated_steps > self.INLINE_STEP_THRESHOLD

        if is_long:
            return await self._dispatch_async(
                task_id,
                objective,
                state,
                task_engine,
                t0,
                matched_skills=assessment.matched_skills,
                matched_tools=assessment.matched_tools,
                requested_objective=requested_objective,
                continued_from_task_id=str((followup_entry or {}).get("task_id", "") or ""),
                resume_plan_id=resume_plan_id,
            )
        else:
            return await self._dispatch_inline(
                task_id,
                objective,
                state,
                task_engine,
                t0,
                matched_skills=assessment.matched_skills,
                matched_tools=assessment.matched_tools,
                requested_objective=requested_objective,
                continued_from_task_id=str((followup_entry or {}).get("task_id", "") or ""),
                resume_plan_id=resume_plan_id,
            )

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Return stored status for a previously dispatched task.

        Used by the response generator to answer "are you done with X?" accurately.
        """
        with self._lock:
            entry = self._active_tasks.get(task_id)
            return dict(entry) if isinstance(entry, dict) else entry

    def get_all_active(self) -> List[Dict]:
        """Return all non-terminal task records."""
        with self._lock:
            return [
                dict(t)
                for t in self._active_tasks.values()
                if str(t.get("status", "") or "") not in _TERMINAL_TASK_STATUSES
            ]

    @classmethod
    def is_continuation_request(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        if not lowered:
            return False
        if not any(marker in lowered for marker in _CONTINUATION_MARKERS):
            return False

        stripped = lowered
        for marker in _CONTINUATION_MARKERS:
            stripped = stripped.replace(marker, " ")
        residual_tokens = [
            token
            for token in re.findall(r"[a-z0-9_./-]+", stripped)
            if token not in _CONTINUATION_FILLER_TOKENS
        ]
        return len(residual_tokens) <= 1

    @classmethod
    def is_status_followup_request(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        if not lowered:
            return False
        return any(marker in lowered for marker in _STATUS_QUERY_MARKERS)

    def build_status_reply(
        self,
        objective: str,
        *,
        last_result_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.is_status_followup_request(objective):
            return ""

        entry = self._resolve_relevant_entry(objective)
        if entry:
            reply = self._render_entry_status_reply(entry, objective)
            if reply:
                return reply

        if isinstance(last_result_payload, dict):
            return self._render_payload_status_reply(last_result_payload, objective)
        return ""

    def get_context_block(self, objective: str = "", limit: int = 4) -> str:
        """Compact prompt-facing task continuity block."""
        entries = self._relevant_entries(objective, limit=limit)
        if not entries:
            return ""

        lines = ["## TASK CONTINUITY"]
        for entry in entries:
            status = str(entry.get("status", "unknown") or "unknown")
            objective_text = str(entry.get("objective", "") or "").strip()
            summary = str(entry.get("summary", "") or entry.get("error", "") or "").strip()
            task_id = str(entry.get("task_id", "") or "").strip()
            progress = ""
            steps_total = int(entry.get("steps_total", 0) or 0)
            steps_completed = int(entry.get("steps_completed", 0) or 0)
            if steps_total > 0:
                progress = f" — progress: {steps_completed}/{steps_total}"
            rendered = f"- [{task_id}] {objective_text[:90]} — status: {status}{progress}"
            if summary:
                rendered += f" — {summary[:140]}"
            lines.append(rendered)

        lines.append("Use task state to answer progress/follow-up questions from real execution status, not guesses.")
        return "\n".join(lines)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _assess_capability(self, objective: str) -> CapabilityAssessment:
        """Check CapabilityEngine and AutonomousTaskEngine tool registry for coverage."""
        cap_engine = self._get_cap_engine()
        if cap_engine is None:
            return CapabilityAssessment(
                can_fulfil=True,   # Assume capable if engine unavailable — fail-safe not fail-silent
                confidence=0.3,
                gap_description="CapabilityEngine unavailable; proceeding with best effort.",
            )

        # Pattern-match the objective against registered skills
        matched: List[str] = []
        if hasattr(cap_engine, "detect_intent"):
            matched = cap_engine.detect_intent(objective) or []

        # Fall back to checking skill list by keyword overlap
        if not matched and hasattr(cap_engine, "list_skills"):
            skill_names: List[str] = cap_engine.list_skills() or []
            obj_lower = objective.lower()
            matched = [s for s in skill_names if any(w in obj_lower for w in s.split("_"))]

        if matched:
            # Filter out skills that require_approval (caller handles gating separately)
            executable = []
            for sk in matched:
                meta = cap_engine.get(sk) if hasattr(cap_engine, "get") else None
                instance = getattr(meta, "instance", None) if meta else None
                needs_approval = getattr(instance, "requires_approval", False)
                if not needs_approval:
                    executable.append(sk)
                else:
                    logger.debug(
                        "TaskCommitmentVerifier: skill '%s' requires approval — excluded from auto-execute", sk
                    )

            if executable:
                return CapabilityAssessment(
                    can_fulfil=True,
                    matched_skills=executable,
                    confidence=min(1.0, 0.5 + 0.1 * len(executable)),
                )

        # Check AutonomousTaskEngine's built-in tool registry
        task_engine = self._get_task_engine()
        if task_engine is not None:
            builtin_tools = list(getattr(task_engine, "_tool_registry", {}).keys())
            if builtin_tools:
                # TaskEngine has built-in tools (think, web_search, etc.) that can
                # handle generic goals even without a specific named skill.
                return CapabilityAssessment(
                    can_fulfil=True,
                    matched_tools=builtin_tools,
                    confidence=0.6,
                )

        # Genuinely no coverage
        all_skills = cap_engine.list_skills() if hasattr(cap_engine, "list_skills") else []
        return CapabilityAssessment(
            can_fulfil=False,
            confidence=0.0,
            gap_description=(
                "No registered skill or built-in tool matches this request."
            ),
            alternatives=all_skills[:5],  # offer a sample of what IS available
        )

    def _estimate_steps(self, objective: str, assessment: CapabilityAssessment) -> int:
        """Heuristic step-count estimate — avoids an LLM call just for routing."""
        if looks_like_multi_step_skill_request(objective, assessment.matched_skills or assessment.matched_tools):
            return max(5, len(assessment.matched_skills) + 2)
        # If a single specific skill matched, it's one step
        if len(assessment.matched_skills) == 1:
            return 1
        # Word-count heuristic: longer, more complex objectives tend to need more steps
        word_count = len(objective.split())
        if word_count > 40:
            return 6
        if word_count > 20:
            return 3
        return 2

    async def _dispatch_inline(
        self,
        task_id: str,
        objective: str,
        state: Any,
        task_engine: Any,
        t0: float,
        *,
        matched_skills: Optional[List[str]] = None,
        matched_tools: Optional[List[str]] = None,
        requested_objective: str = "",
        continued_from_task_id: str = "",
        resume_plan_id: str = "",
    ) -> TaskAcceptance:
        """Run task synchronously and wait for result (short tasks only)."""
        execution_origin = self._resolve_execution_origin(state)
        self._store_task_entry(
            task_id,
            {
                "task_id": task_id,
                "objective": objective[:120],
                "requested_objective": requested_objective[:120] if requested_objective and requested_objective != objective else "",
                "continued_from_task_id": continued_from_task_id,
                "status": "running",
                "started_at": time.time(),
            },
        )
        await self._track_goal_dispatch(
            objective,
            task_id=task_id,
            source="commitment_verifier_inline",
            quick_win=True,
        )
        execution_task = get_task_tracker().create_task(
            task_engine.execute(
                goal=objective,
                context={
                    "task_id": task_id,
                    "source": "commitment_verifier",
                    "origin": execution_origin,
                    "intent_source": execution_origin,
                    "request_origin": execution_origin,
                    "quick_win": True,
                    "attention_policy": "interruptible",
                    "priority": 0.9,
                    "horizon": "short_term",
                    "resume_plan_id": resume_plan_id,
                    "matched_skills": list(matched_skills or []),
                    "matched_tools": list(matched_tools or []),
                },
            )
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(execution_task),
                timeout=self.INLINE_TIMEOUT_S,
            )
            succeeded = self._result_counts_as_success(result)
            summary = getattr(result, "summary", str(result))
            outcome = DispatchOutcome.COMPLETED if succeeded else DispatchOutcome.FAILED
            tracking_updates, evidence = self._extract_result_tracking_fields(result)
            self._update_task_entry(
                task_id,
                status=outcome.value,
                completed_at=time.time(),
                summary=summary,
                **tracking_updates,
            )
            await self._update_goal_dispatch(
                task_id=task_id,
                status=outcome.value,
                summary=summary,
                error="" if succeeded else summary,
                evidence=evidence,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "TaskCommitmentVerifier: inline task %s → %s in %.0fms: %s",
                task_id, outcome.value, elapsed, summary[:80],
            )
            return TaskAcceptance(
                outcome=outcome,
                task_id=task_id,
                objective=objective,
                requested_objective=requested_objective,
                summary=summary,
                result_data=result,
                elapsed_ms=elapsed,
            )
        except asyncio.TimeoutError:
            commitment_id = self._register_commitment(objective)
            updates: Dict[str, Any] = {"status": "running_async"}
            if commitment_id:
                updates["commitment_id"] = commitment_id
            self._update_task_entry(task_id, **updates)
            self._background_tasks[task_id] = execution_task
            execution_task.add_done_callback(
                lambda fut, _task_id=task_id, _objective=objective, _commitment_id=commitment_id:
                get_task_tracker().create_task(
                    self._finalize_background_task(
                        task_id=_task_id,
                        objective=_objective,
                        future=fut,
                        commitment_id=_commitment_id,
                    )
                )
            )
            await self._track_goal_dispatch(
                objective,
                task_id=task_id,
                source="commitment_verifier_timeout",
                commitment_id=commitment_id,
                quick_win=False,
            )
            elapsed = (time.monotonic() - t0) * 1000
            return TaskAcceptance(
                outcome=DispatchOutcome.STARTED,
                task_id=task_id,
                commitment_id=commitment_id,
                objective=objective,
                requested_objective=requested_objective,
                summary=(
                    f"This task is still running in the background (took >={self.INLINE_TIMEOUT_S:.0f}s). "
                    "I'll keep tracking it until it completes."
                    + (f" Commitment {commitment_id} is attached." if commitment_id else "")
                ),
                elapsed_ms=elapsed,
            )
        except Exception as e:
            self._update_task_entry(task_id, status="failed", error=str(e))
            await self._update_goal_dispatch(
                task_id=task_id,
                status=DispatchOutcome.FAILED.value,
                summary=f"Task execution failed: {e}",
                error=str(e),
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.warning("TaskCommitmentVerifier: inline task %s failed: %s", task_id, e)
            return TaskAcceptance(
                outcome=DispatchOutcome.FAILED,
                task_id=task_id,
                objective=objective,
                requested_objective=requested_objective,
                summary=f"Task execution failed: {e}",
                elapsed_ms=elapsed,
            )

    async def _dispatch_async(
        self,
        task_id: str,
        objective: str,
        state: Any,
        task_engine: Any,
        t0: float,
        *,
        matched_skills: Optional[List[str]] = None,
        matched_tools: Optional[List[str]] = None,
        requested_objective: str = "",
        continued_from_task_id: str = "",
        resume_plan_id: str = "",
    ) -> TaskAcceptance:
        """Launch task in the background and register a CommitmentEngine entry."""
        execution_origin = self._resolve_execution_origin(state)
        # Register with CommitmentEngine for cross-session tracking
        commitment_id = self._register_commitment(objective)

        self._store_task_entry(
            task_id,
            {
                "task_id": task_id,
                "objective": objective[:120],
                "requested_objective": requested_objective[:120] if requested_objective and requested_objective != objective else "",
                "continued_from_task_id": continued_from_task_id,
                "commitment_id": commitment_id,
                "status": "running_async",
                "started_at": time.time(),
            },
        )
        await self._track_goal_dispatch(
            objective,
            task_id=task_id,
            source="commitment_verifier_async",
            commitment_id=commitment_id,
            quick_win=False,
        )
        execution_task = get_task_tracker().create_task(
            task_engine.execute(
                goal=objective,
                context={
                    "task_id": task_id,
                    "source": "commitment_verifier_async",
                    "origin": execution_origin,
                    "intent_source": execution_origin,
                    "request_origin": execution_origin,
                    "commitment_id": commitment_id,
                    "quick_win": False,
                    "attention_policy": "sustained",
                    "priority": 0.8,
                    "horizon": "short_term",
                    "resume_plan_id": resume_plan_id,
                    "matched_skills": list(matched_skills or []),
                    "matched_tools": list(matched_tools or []),
                },
            )
        )
        self._background_tasks[task_id] = execution_task
        execution_task.add_done_callback(
            lambda fut, _task_id=task_id, _objective=objective, _commitment_id=commitment_id:
            get_task_tracker().create_task(
                self._finalize_background_task(
                    task_id=_task_id,
                    objective=_objective,
                    future=fut,
                    commitment_id=_commitment_id,
                )
            )
        )

        elapsed = (time.monotonic() - t0) * 1000
        return TaskAcceptance(
            outcome=DispatchOutcome.STARTED,
            task_id=task_id,
            commitment_id=commitment_id,
            objective=objective,
            requested_objective=requested_objective,
            summary=(
                f"I've started this task (id={task_id}). "
                "I'll follow up when it's done."
                + (f" Tracking commitment {commitment_id}." if commitment_id else "")
            ),
            elapsed_ms=elapsed,
        )

    def _get_cap_engine(self) -> Optional[Any]:
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("capability_engine", default=None)
        except Exception:
            return None

    def _get_task_engine(self) -> Optional[Any]:
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("task_engine", default=None)
        except Exception:
            return None

    def _get_goal_engine(self) -> Optional[Any]:
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("goal_engine", default=None)
        except Exception:
            return None

    def _register_commitment(self, objective: str) -> Optional[str]:
        try:
            from core.agency.commitment_engine import CommitmentType, get_commitment_engine

            commitment = get_commitment_engine().commit(
                description=objective[:120],
                outcome="Task completed successfully with verified result.",
                deadline_hours=4.0,
                commitment_type=CommitmentType.AUTONOMOUS,
            )
            return commitment.id
        except Exception as ex:
            logger.debug("TaskCommitmentVerifier: CommitmentEngine registration failed: %s", ex)
            return None

    @staticmethod
    def _extract_result_tracking_fields(result: Any) -> tuple[Dict[str, Any], List[str]]:
        updates: Dict[str, Any] = {}
        evidence: List[str] = []
        for key in ("plan_id", "trace_id", "steps_completed", "steps_total", "duration_s"):
            value = getattr(result, key, None)
            if value not in (None, ""):
                updates[key] = value
        for item in list(getattr(result, "evidence", []) or [])[:4]:
            text = str(item or "").strip()
            if text:
                evidence.append(text)
        if evidence:
            updates["evidence"] = evidence
        return updates, evidence

    @staticmethod
    def _result_counts_as_success(result: Any) -> bool:
        if not getattr(result, "succeeded", bool(result)):
            return False

        summary = str(getattr(result, "summary", "") or "").lower()
        if any(
            marker in summary
            for marker in (
                "attempted",
                "tried",
                "meant to",
                "intended to",
                "couldn't",
                "could not",
                "did not complete",
                "still running",
            )
        ):
            return False

        steps_total = int(getattr(result, "steps_total", 0) or 0)
        steps_completed = int(getattr(result, "steps_completed", 0) or 0)
        if steps_total > 0 and steps_completed < steps_total:
            return False

        return True

    async def _track_goal_dispatch(
        self,
        objective: str,
        *,
        task_id: str,
        source: str,
        commitment_id: Optional[str] = None,
        quick_win: bool,
    ) -> None:
        goal_engine = self._get_goal_engine()
        if goal_engine and hasattr(goal_engine, "track_dispatch"):
            try:
                await goal_engine.track_dispatch(
                    objective,
                    task_id=task_id,
                    source=source,
                    commitment_id=commitment_id,
                    quick_win=quick_win,
                )
            except Exception as exc:
                logger.debug("TaskCommitmentVerifier: goal dispatch tracking failed: %s", exc)

    async def _update_goal_dispatch(
        self,
        *,
        task_id: str,
        status: str,
        summary: str = "",
        error: str = "",
        evidence: Optional[List[str]] = None,
    ) -> None:
        goal_engine = self._get_goal_engine()
        if goal_engine and hasattr(goal_engine, "update_task_lifecycle"):
            try:
                await goal_engine.update_task_lifecycle(
                    task_id=task_id,
                    status=status,
                    summary=summary,
                    error=error,
                    evidence=evidence,
                )
            except Exception as exc:
                logger.debug("TaskCommitmentVerifier: goal lifecycle update failed: %s", exc)

    async def _finalize_background_task(
        self,
        *,
        task_id: str,
        objective: str,
        future: asyncio.Future,
        commitment_id: Optional[str],
    ) -> None:
        self._background_tasks.pop(task_id, None)
        try:
            if future.cancelled():
                raise asyncio.CancelledError(f"Background task {task_id} was cancelled")
            result = future.result()
            succeeded = self._result_counts_as_success(result)
            summary = getattr(result, "summary", str(result))
            tracking_updates, evidence = self._extract_result_tracking_fields(result)
            self._update_task_entry(
                task_id,
                status="completed" if succeeded else "failed",
                completed_at=time.time(),
                summary=summary,
                **tracking_updates,
            )
            try:
                await self._update_goal_dispatch(
                    task_id=task_id,
                    status=DispatchOutcome.COMPLETED.value if succeeded else DispatchOutcome.FAILED.value,
                    summary=summary,
                    error="" if succeeded else summary,
                    evidence=evidence,
                )
            except Exception as goal_exc:
                logger.error("TaskCommitmentVerifier: goal dispatch failed for %s: %s", task_id, goal_exc)
            if commitment_id:
                try:
                    from core.agency.commitment_engine import get_commitment_engine

                    if succeeded:
                        get_commitment_engine().fulfill(commitment_id, note=summary[:200])
                    else:
                        get_commitment_engine().break_commitment(
                            commitment_id,
                            note=f"Task failed: {summary[:160]}",
                            progress=0.0,
                        )
                except Exception as exc:
                    logger.debug("TaskCommitmentVerifier: commitment settlement failed: %s", exc)
            logger.info(
                "TaskCommitmentVerifier: async task %s %s: %s",
                task_id,
                "completed" if succeeded else "failed",
                summary[:80],
            )
        except asyncio.CancelledError:
            self._update_task_entry(task_id, status="cancelled", completed_at=time.time())
            try:
                await self._update_goal_dispatch(
                    task_id=task_id,
                    status=DispatchOutcome.FAILED.value,
                    summary="Task was cancelled",
                    error="cancelled",
                )
            except Exception:
                pass
            if commitment_id:
                try:
                    from core.agency.commitment_engine import get_commitment_engine

                    get_commitment_engine().break_commitment(
                        commitment_id,
                        note="Task was cancelled before completion.",
                        progress=0.0,
                    )
                except Exception as exc:
                    logger.debug("TaskCommitmentVerifier: cancelled commitment settlement failed: %s", exc)
            logger.warning("TaskCommitmentVerifier: async task %s was cancelled", task_id)
        except Exception as e:
            self._update_task_entry(
                task_id,
                status="failed",
                error=str(e),
                completed_at=time.time(),
            )
            try:
                await self._update_goal_dispatch(
                    task_id=task_id,
                    status=DispatchOutcome.FAILED.value,
                    summary=f"Task execution failed: {e}",
                    error=str(e),
                )
            except Exception as goal_exc:
                logger.error("TaskCommitmentVerifier: goal dispatch failed during error handling for %s: %s", task_id, goal_exc)
            if commitment_id:
                try:
                    from core.agency.commitment_engine import get_commitment_engine

                    get_commitment_engine().break_commitment(
                        commitment_id,
                        note=f"Task execution raised: {str(e)[:160]}",
                        progress=0.0,
                    )
                except Exception as exc:
                    logger.debug("TaskCommitmentVerifier: exception commitment settlement failed: %s", exc)
            logger.warning("TaskCommitmentVerifier: async task %s failed: %s", task_id, e)
        finally:
            # Prevent unbounded _active_tasks growth: clean up terminal entries
            # after a grace period so callers can still query recent results.
            entry = self.get_task_status(task_id)
            if entry and entry.get("status") in {"completed", "failed", "cancelled"}:
                self._update_task_entry(task_id, cleanup_at=time.time() + 300)
            self._prune_terminal_tasks()

    def _prune_terminal_tasks(self, max_age: float = 300.0) -> None:
        """Remove completed/failed/cancelled tasks older than max_age seconds."""
        now = time.time()
        with self._lock:
            to_remove = [
                tid
                for tid, entry in self._active_tasks.items()
                if str(entry.get("status", "") or "") in {"completed", "failed", "cancelled"}
                and now >= float(entry.get("cleanup_at", now + 1))
            ]
            for tid in to_remove:
                self._active_tasks.pop(tid, None)
            if to_remove:
                self._updated_at = now
                self._save_locked()

    def _task_engine_entries(self) -> List[Dict[str, Any]]:
        task_engine = self._get_task_engine()
        if task_engine is None or not hasattr(task_engine, "get_active_plans"):
            return []
        try:
            snapshot = list(task_engine.get_active_plans() or [])
        except Exception as exc:
            logger.debug("TaskCommitmentVerifier: task engine snapshot skipped: %s", exc)
            return []

        entries: List[Dict[str, Any]] = []
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            plan_id = str(item.get("plan_id", "") or "").strip()
            objective = str(item.get("goal", "") or "").strip()
            if not plan_id or not objective:
                continue
            entries.append(
                {
                    "task_id": str(item.get("task_id", "") or plan_id),
                    "plan_id": plan_id,
                    "objective": objective,
                    "status": str(item.get("status", "") or "running"),
                    "summary": str(item.get("summary", "") or "").strip(),
                    "steps_completed": int(item.get("steps_completed", 0) or 0),
                    "steps_total": int(item.get("steps_total", 0) or 0),
                    "updated_at": time.time(),
                }
            )
        return entries

    @staticmethod
    def _token_overlap_score(text: str, objective: str) -> float:
        text_tokens = set(re.findall(r"[a-z0-9_]+", str(text or "").lower()))
        objective_tokens = set(re.findall(r"[a-z0-9_]+", str(objective or "").lower()))
        if not text_tokens or not objective_tokens:
            return 0.0
        overlap = text_tokens & objective_tokens
        return len(overlap) / max(1, len(objective_tokens))

    def _relevant_entries(self, objective: str, *, limit: int) -> List[Dict]:
        with self._lock:
            records = [dict(item) for item in self._active_tasks.values()]
        if not records:
            records = []
        external_records = self._task_engine_entries()

        lowered = str(objective or "").lower()
        followup_query = any(marker in lowered for marker in _STATUS_FOLLOWUP_MARKERS)

        def _score(entry: Dict[str, Any]) -> tuple[float, float]:
            status = str(entry.get("status", "") or "")
            overlap = self._token_overlap_score(entry.get("objective", ""), objective)
            terminal = status in {"completed", "failed", "cancelled"}
            recency = float(entry.get("completed_at") or entry.get("started_at") or 0.0)
            score = overlap
            if status == "running_async":
                score += 0.45
            elif status == "running":
                score += 0.4
            elif followup_query and terminal:
                score += 0.25
            if overlap > 0:
                score += 0.2
            if not objective and not terminal:
                score += 0.1
            return (score, recency)

        ordered = sorted(records, key=_score, reverse=True)
        if objective:
            filtered = [entry for entry in ordered if _score(entry)[0] > 0.0]
            if filtered:
                return filtered[:limit]
        if ordered and not objective:
            return ordered[:limit]

        if not external_records:
            return ordered[:limit]

        ordered_external = sorted(external_records, key=_score, reverse=True)
        if objective:
            filtered_external = [entry for entry in ordered_external if _score(entry)[0] > 0.0]
            if filtered_external:
                return filtered_external[:limit]
        return (ordered or ordered_external)[:limit]

    def _resolve_relevant_entry(self, objective: str) -> Optional[Dict[str, Any]]:
        entries = self._relevant_entries(objective, limit=1)
        if entries:
            return dict(entries[0])
        return None

    def _build_continuation_reuse_acceptance(
        self,
        entry: Dict[str, Any],
        *,
        t0: float,
    ) -> Optional[TaskAcceptance]:
        status = str(entry.get("status", "") or "")
        task_id = str(entry.get("task_id", "") or "") or None
        commitment_id = str(entry.get("commitment_id", "") or "") or None
        objective = str(entry.get("objective", "") or "").strip()
        summary = str(entry.get("summary", "") or entry.get("error", "") or "").strip()
        elapsed = (time.monotonic() - t0) * 1000

        if status in _RUNNING_TASK_STATUSES:
            msg = f"I'm already working on {objective}." if objective else "I'm already working on that."
            if summary:
                msg += f" Latest state: {summary}"
            return TaskAcceptance(
                outcome=DispatchOutcome.STARTED,
                task_id=task_id,
                commitment_id=commitment_id,
                objective=objective,
                requested_objective=objective,
                summary=msg,
                elapsed_ms=elapsed,
            )

        if status == "completed":
            msg = f"That task already finished: {objective}." if objective else "That task already finished."
            if summary:
                msg += f" {summary}"
            return TaskAcceptance(
                outcome=DispatchOutcome.COMPLETED,
                task_id=task_id,
                commitment_id=commitment_id,
                objective=objective,
                requested_objective=objective,
                summary=msg,
                elapsed_ms=elapsed,
            )
        return None

    def _render_entry_status_reply(self, entry: Dict[str, Any], objective: str) -> str:
        status = str(entry.get("status", "") or "unknown")
        task_objective = str(entry.get("objective", "") or "that task").strip()
        summary = str(entry.get("summary", "") or "").strip()
        error = str(entry.get("error", "") or "").strip()
        lowered = str(objective or "").lower()
        progress = ""
        steps_total = int(entry.get("steps_total", 0) or 0)
        steps_completed = int(entry.get("steps_completed", 0) or 0)
        if steps_total > 0:
            progress = f" ({steps_completed}/{steps_total} steps)"

        if status in _RUNNING_TASK_STATUSES:
            base = f"No. It's still running{progress}: {task_objective}."
            if "progress" in lowered or "status" in lowered:
                base = f"It's still in progress{progress}: {task_objective}."
            if summary:
                base += f" Latest state: {summary}"
            return base

        if status == "interrupted":
            base = (
                f"Not yet. I was working on {task_objective}{progress}, but that run was interrupted before it finished."
            )
            detail = summary or "I need to resume it rather than pretend it completed."
            return f"{base} {detail}".strip()

        if status == "waiting_for_approval":
            detail = summary or "It needs explicit approval before the next execution step can continue."
            return f"It is paused pending approval{progress}: {task_objective}. {detail}".strip()

        if status == "completed":
            base = f"Yes. I finished {task_objective}{progress}."
            if "what happened" in lowered or "follow up" in lowered:
                base = f"That task finished{progress}: {task_objective}."
            if summary:
                base += f" {summary}"
            return base

        if status == "failed":
            detail = error or summary or "It failed without a clean summary."
            return f"It didn't finish cleanly. {task_objective}{progress} failed. {detail}".strip()

        if status == "cancelled":
            return f"I stopped that run before it finished{progress}: {task_objective}."

        return ""

    def _render_payload_status_reply(self, payload: Dict[str, Any], objective: str) -> str:
        status = str(payload.get("status", "") or "")
        task_objective = str(payload.get("objective", "") or "that task").strip()
        summary = str(payload.get("summary", "") or payload.get("error", "") or "").strip()
        progress = ""
        steps_total = int(payload.get("steps_total", 0) or 0)
        steps_completed = int(payload.get("steps_completed", 0) or 0)
        if steps_total > 0:
            progress = f" ({steps_completed}/{steps_total} steps)"

        if status == DispatchOutcome.STARTED.value:
            base = "It's still in flight."
            if task_objective:
                base = f"It's still in flight{progress}: {task_objective}."
            if summary:
                base += f" {summary}"
            return base
        if status == DispatchOutcome.COMPLETED.value:
            base = "Yes. That task finished."
            if task_objective:
                base = f"Yes. {task_objective}{progress} finished."
            if summary:
                base += f" {summary}"
            return base
        if status == DispatchOutcome.FAILED.value:
            base = "It didn't finish cleanly."
            if task_objective:
                base = f"It didn't finish cleanly{progress}: {task_objective}."
            if summary:
                base += f" {summary}"
            return base
        return ""

    def _store_task_entry(self, task_id: str, entry: Dict[str, Any]) -> None:
        with self._lock:
            payload = dict(entry)
            payload.setdefault("task_id", task_id)
            payload.setdefault("updated_at", time.time())
            self._active_tasks[task_id] = payload
            self._updated_at = time.time()
            self._save_locked()

    def _update_task_entry(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            entry = self._active_tasks.get(task_id)
            if not entry:
                return
            entry.update(updates)
            entry["updated_at"] = time.time()
            self._updated_at = time.time()
            self._save_locked()

    def _save_locked(self) -> None:
        payload = {
            "updated_at": self._updated_at or time.time(),
            "active_tasks": list(self._active_tasks.values()),
        }
        atomic_write_json(str(self.persist_path), payload)

    def _load(self) -> None:
        try:
            if not self.persist_path.exists():
                return
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("TaskCommitmentVerifier: state load skipped: %s", exc)
            return

        now = time.time()
        loaded: Dict[str, Dict[str, Any]] = {}
        needs_save = False
        for item in list(raw.get("active_tasks") or []):
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", "") or "").strip()
            if not task_id:
                continue
            entry = dict(item)
            status = str(entry.get("status", "") or "")
            if status in _RUNNING_TASK_STATUSES:
                entry["status"] = "interrupted"
                entry.setdefault(
                    "summary",
                    "The last run was interrupted before it could finish, so it needs an explicit resume.",
                )
                entry["interrupted_at"] = now
                entry["updated_at"] = now
                needs_save = True
            loaded[task_id] = entry

        with self._lock:
            self._active_tasks = loaded
            self._updated_at = float(raw.get("updated_at") or now)
            self._prune_terminal_tasks(max_age=86400.0)
            if needs_save:
                self._updated_at = now
                self._save_locked()


# ── Singleton ─────────────────────────────────────────────────────────────────

_verifier: Optional[TaskCommitmentVerifier] = None


def get_task_commitment_verifier(kernel: Any = None) -> TaskCommitmentVerifier:
    global _verifier
    if _verifier is None:
        _verifier = TaskCommitmentVerifier(kernel)
    return _verifier
