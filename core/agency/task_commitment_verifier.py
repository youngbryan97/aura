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

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.TaskCommitmentVerifier")


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

    def __init__(self, kernel: Any):
        self.kernel = kernel
        self._active_tasks: Dict[str, Dict] = {}   # task_id → status record

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

        # 1. Capability pre-check
        assessment = self._assess_capability(objective)

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
                summary="Task engine unavailable. Cannot execute autonomous tasks right now.",
                elapsed_ms=elapsed,
            )

        # 3. Estimate plan size to decide inline vs. async
        estimated_steps = self._estimate_steps(objective, assessment)
        is_long = force_async or estimated_steps > self.INLINE_STEP_THRESHOLD

        if is_long:
            return await self._dispatch_async(task_id, objective, state, task_engine, t0)
        else:
            return await self._dispatch_inline(task_id, objective, state, task_engine, t0)

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Return stored status for a previously dispatched task.

        Used by the response generator to answer "are you done with X?" accurately.
        """
        return self._active_tasks.get(task_id)

    def get_all_active(self) -> List[Dict]:
        """Return all non-terminal task records."""
        return [
            t for t in self._active_tasks.values()
            if t.get("status") not in ("completed", "failed", "capability_gap")
        ]

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
    ) -> TaskAcceptance:
        """Run task synchronously and wait for result (short tasks only)."""
        self._active_tasks[task_id] = {
            "task_id": task_id, "objective": objective[:120],
            "status": "running", "started_at": time.time(),
        }
        try:
            result = await asyncio.wait_for(
                task_engine.execute(
                    goal=objective,
                    context={"task_id": task_id, "source": "commitment_verifier"},
                ),
                timeout=self.INLINE_TIMEOUT_S,
            )
            succeeded = getattr(result, "succeeded", bool(result))
            summary = getattr(result, "summary", str(result))
            outcome = DispatchOutcome.COMPLETED if succeeded else DispatchOutcome.FAILED
            self._active_tasks[task_id].update({
                "status": outcome.value,
                "completed_at": time.time(),
                "summary": summary,
            })
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "TaskCommitmentVerifier: inline task %s → %s in %.0fms: %s",
                task_id, outcome.value, elapsed, summary[:80],
            )
            return TaskAcceptance(
                outcome=outcome,
                task_id=task_id,
                summary=summary,
                result_data=result,
                elapsed_ms=elapsed,
            )
        except asyncio.TimeoutError:
            # Timed out inline — treat as async-started
            self._active_tasks[task_id]["status"] = "running_async"
            elapsed = (time.monotonic() - t0) * 1000
            return TaskAcceptance(
                outcome=DispatchOutcome.STARTED,
                task_id=task_id,
                summary=f"This task is running in the background (took >={self.INLINE_TIMEOUT_S:.0f}s). I'll update you when it completes.",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            self._active_tasks[task_id].update({"status": "failed", "error": str(e)})
            elapsed = (time.monotonic() - t0) * 1000
            logger.warning("TaskCommitmentVerifier: inline task %s failed: %s", task_id, e)
            return TaskAcceptance(
                outcome=DispatchOutcome.FAILED,
                task_id=task_id,
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
    ) -> TaskAcceptance:
        """Launch task in the background and register a CommitmentEngine entry."""
        # Register with CommitmentEngine for cross-session tracking
        commitment_id = None
        try:
            from core.agency.commitment_engine import get_commitment_engine
            ce = get_commitment_engine()
            commitment = ce.commit(
                description=objective[:120],
                outcome="Task completed successfully with verified result.",
                deadline_hours=4.0,
            )
            commitment_id = commitment.id
        except Exception as ex:
            logger.debug("TaskCommitmentVerifier: CommitmentEngine registration failed: %s", ex)

        self._active_tasks[task_id] = {
            "task_id": task_id, "objective": objective[:120],
            "commitment_id": commitment_id,
            "status": "running_async", "started_at": time.time(),
        }

        # Fire-and-forget background execution with completion callback
        async def _bg_run():
            try:
                result = await task_engine.execute(
                    goal=objective,
                    context={"task_id": task_id, "source": "commitment_verifier_async"},
                )
                succeeded = getattr(result, "succeeded", bool(result))
                summary = getattr(result, "summary", str(result))
                self._active_tasks[task_id].update({
                    "status": "completed" if succeeded else "failed",
                    "completed_at": time.time(),
                    "summary": summary,
                })
                # Fulfil or mark the commitment
                if commitment_id:
                    try:
                        from core.agency.commitment_engine import get_commitment_engine
                        if succeeded:
                            get_commitment_engine().fulfill(commitment_id, note=summary[:200])
                        else:
                            get_commitment_engine().update_progress(commitment_id, 0.5,
                                                                     note=f"Partial: {summary[:100]}")
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                logger.info(
                    "TaskCommitmentVerifier: async task %s %s: %s",
                    task_id, "completed" if succeeded else "failed", summary[:80],
                )
            except Exception as e:
                self._active_tasks[task_id].update({"status": "failed", "error": str(e)})
                logger.warning("TaskCommitmentVerifier: async task %s failed: %s", task_id, e)

        asyncio.ensure_future(_bg_run())

        elapsed = (time.monotonic() - t0) * 1000
        return TaskAcceptance(
            outcome=DispatchOutcome.STARTED,
            task_id=task_id,
            commitment_id=commitment_id,
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


# ── Singleton ─────────────────────────────────────────────────────────────────

_verifier: Optional[TaskCommitmentVerifier] = None


def get_task_commitment_verifier(kernel: Any = None) -> TaskCommitmentVerifier:
    global _verifier
    if _verifier is None:
        _verifier = TaskCommitmentVerifier(kernel)
    return _verifier
