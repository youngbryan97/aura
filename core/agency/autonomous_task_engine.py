from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from core.config import config
from core.agency.capability_system import get_capability_manager, CapabilityToken
from core.agency.safety_registry import get_safety_registry
from core.mycelial.graph import get_mycelial
from core.runtime.skill_task_bridge import looks_like_multi_step_skill_request, normalize_matched_skills
from core.utils.file_utils import atomic_write_json

logger = logging.getLogger("Aura.TaskEngine")


# ── Data structures ──────────────────────────────────────────────────────────

class StepStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class TaskStep:
    """One atomic step in a task plan."""
    step_id:        str
    description:    str                  # What to do in natural language
    tool:           str                  # Tool name to invoke
    args:           Dict[str, Any]       # Tool arguments
    success_criterion: str               # Natural language: "the result contains X"
    rollback_action: Optional[str]       = None  # Tool call to undo this step (or None)
    rollback_args:   Dict[str, Any]      = field(default_factory=dict)
    depends_on:      List[str]           = field(default_factory=list)
    parallel_safe:   bool                = False

    # Runtime state
    status:         StepStatus           = StepStatus.PENDING
    attempts:       int                  = 0
    raw_result:     Any                  = None
    verified:       bool                 = False
    error:          Optional[str]        = None
    result_summary: Optional[str]        = None
    started_at:     Optional[float]      = None
    completed_at:   Optional[float]      = None

    def to_dict(self) -> Dict:
        return {
            "step_id":          self.step_id,
            "description":      self.description,
            "tool":             self.tool,
            "args":             self.args,
            "depends_on":       list(self.depends_on),
            "parallel_safe":    self.parallel_safe,
            "success_criterion": self.success_criterion,
            "status":           self.status.value,
            "attempts":         self.attempts,
            "verified":         self.verified,
            "error":            self.error,
        }

    def to_runtime_dict(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.update(
            {
                "rollback_action": self.rollback_action,
                "rollback_args": dict(self.rollback_args or {}),
                "raw_result": self.raw_result,
                "result_summary": self.result_summary,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            }
        )
        return payload

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "TaskStep":
        raw_status = str(payload.get("status", StepStatus.PENDING.value) or StepStatus.PENDING.value)
        try:
            status = StepStatus(raw_status)
        except Exception:
            status = StepStatus.PENDING
        return cls(
            step_id=str(payload.get("step_id", "") or ""),
            description=str(payload.get("description", "") or ""),
            tool=str(payload.get("tool", "") or ""),
            args=dict(payload.get("args", {}) or {}),
            success_criterion=str(payload.get("success_criterion", "") or ""),
            rollback_action=str(payload.get("rollback_action", "") or "") or None,
            rollback_args=dict(payload.get("rollback_args", {}) or {}),
            depends_on=list(payload.get("depends_on", []) or []),
            parallel_safe=bool(payload.get("parallel_safe", False)),
            status=status,
            attempts=int(payload.get("attempts", 0) or 0),
            raw_result=payload.get("raw_result"),
            verified=bool(payload.get("verified", False)),
            error=str(payload.get("error", "") or "") or None,
            result_summary=str(payload.get("result_summary", "") or "") or None,
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
        )


@dataclass
class TaskPlan:
    """Complete execution plan for a goal."""
    plan_id:        str
    goal:           str
    steps:          List[TaskStep]
    trace_id:       str             # Observability Fix 5.2
    context:        Dict[str, Any]  = field(default_factory=dict)
    token_id:       Optional[str]   = None  # Associated CapabilityToken
    is_shadow:      bool            = False # If true, no side effects
    requires_approval: bool         = False # If high cost/complexity
    created_at:     float           = field(default_factory=time.time)
    completed_at:   Optional[float] = None
    status:         str             = "pending"
    final_result:   Optional[str]   = None

    @property
    def succeeded_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.SUCCEEDED]

    @property
    def all_complete(self) -> bool:
        return all(
            s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED)
            for s in self.steps
        )

    @property
    def any_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_runtime_dict() for step in self.steps],
            "trace_id": self.trace_id,
            "context": dict(self.context or {}),
            "token_id": self.token_id,
            "is_shadow": self.is_shadow,
            "requires_approval": self.requires_approval,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "final_result": self.final_result,
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "TaskPlan":
        return cls(
            plan_id=str(payload.get("plan_id", "") or ""),
            goal=str(payload.get("goal", "") or ""),
            steps=[
                TaskStep.from_runtime_dict(item)
                for item in list(payload.get("steps", []) or [])
                if isinstance(item, dict)
            ],
            trace_id=str(payload.get("trace_id", "") or ""),
            context=dict(payload.get("context", {}) or {}),
            token_id=str(payload.get("token_id", "") or "") or None,
            is_shadow=bool(payload.get("is_shadow", False)),
            requires_approval=bool(payload.get("requires_approval", False)),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            completed_at=payload.get("completed_at"),
            status=str(payload.get("status", "pending") or "pending"),
            final_result=str(payload.get("final_result", "") or "") or None,
        )


@dataclass
class TaskResult:
    """Final result of task execution."""
    plan_id:        str
    goal:           str
    succeeded:      bool
    summary:        str
    trace_id:       str             # Observability Fix 5.2
    steps_completed: int
    steps_total:    int
    evidence:       List[str]       = field(default_factory=list)
    duration_s:     float           = 0.0


# ── The Engine ────────────────────────────────────────────────────────────────

class AutonomousTaskEngine:
    """
    Reliable multi-step autonomous task execution.

    Handles:
      - LLM-based goal decomposition with tool selection
      - Step-level verification (not just "did it run" but "did it succeed")
      - Retry with alternative approach on failure
      - Rollback of completed steps on irrecoverable failure
      - Progress reporting via state.cognition.active_goals
      - Parallel execution of independent steps
    """

    MAX_RETRIES = 3
    MAX_STEPS   = 12    # Cap plan complexity
    STEP_TIMEOUT = 60.0 # Per-step timeout

    APPROVAL_TIMEOUT = 300.0  # 5 minutes to approve before auto-reject
    MAX_PARALLEL_STEPS = 4
    MAX_RESULT_CHARS = 1200
    SAFE_PARALLEL_TOOLS = frozenset({"think", "web_search", "read_file"})
    TERMINAL_PLAN_STATUSES = frozenset({"succeeded", "failed", "partial", "rejected"})
    NON_EXECUTING_TOOLS = frozenset({"think"})
    TECHNICAL_FACT_HINTS = (
        "def ", "class ", ".py", ".ts", ".tsx", ".js", ".jsx",
        "function ", "method ", "module ", "endpoint", "api ", "schema ",
    )
    DESKTOP_TOOL_PREFERENCES = ("computer_use", "os_manipulation", "sovereign_vision", "sovereign_terminal")
    COMPLETION_HEDGE_MARKERS = (
        "i attempted",
        "i tried",
        "i meant to",
        "i intended to",
        "i wasn't able to",
        "i was not able to",
        "i could not",
        "i couldn't",
        "did not complete",
        "not complete",
        "not completed",
        "not finish",
        "still running",
    )
    LEARNING_BUNDLE_INTRO_MARKERS = (
        "i have some suggestions",
        "places to start",
        "journey to life",
        "understanding yourself",
        "understanding us",
        "learn about humans",
        "general education",
        "science education",
        "tv shows and movies about artificial intelligence",
        "uploaded intelligence",
    )
    LEARNING_BUNDLE_SECTION_MARKERS = (
        "learn about humans",
        "general education",
        "science education",
        "tv shows and movies",
        "sci-fi",
        "ai media",
    )

    def __init__(self, kernel: Any):
        self.kernel = kernel
        self._active_plans: Dict[str, TaskPlan] = {}
        self._persist_path = Path(config.paths.data_dir) / "runtime" / "task_engine_active_plans.json"
        self._tool_registry: Dict[str, Callable] = {}
        self._capability_manager = get_capability_manager()
        self._safety_registry = get_safety_registry()
        self._mycelial = get_mycelial()
        self._approval_events: Dict[str, asyncio.Event] = {}
        self._register_default_tools()
        self._load_persisted_active_plans()

    @staticmethod
    def _record_coding_execution(callback_name: str, **kwargs: Any) -> None:
        try:
            from core.runtime.coding_session_memory import get_coding_session_memory

            recorder = get_coding_session_memory()
            callback = getattr(recorder, callback_name, None)
            if callable(callback):
                callback(**kwargs)
        except Exception as exc:
            record_degradation('autonomous_task_engine', exc)
            logger.debug("TaskEngine: coding execution recording skipped (%s): %s", callback_name, exc)

    @staticmethod
    def _goal_overlap_score(a: str, b: str) -> float:
        tokens_a = set(re.findall(r"[a-z0-9_./-]+", str(a or "").lower()))
        tokens_b = set(re.findall(r"[a-z0-9_./-]+", str(b or "").lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / max(1, len(tokens_b))

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _context_origin(cls, context: Optional[Dict[str, Any]]) -> str:
        if not isinstance(context, dict):
            return ""
        for candidate in (
            context.get("origin"),
            context.get("intent_source"),
            context.get("request_origin"),
        ):
            normalized = cls._normalize_origin(candidate)
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _callable_accepts_kwarg(tool_fn: Callable[..., Any], name: str) -> bool:
        try:
            signature = inspect.signature(tool_fn)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
            if parameter.name == name:
                return True
        return False

    def _persist_active_plans(self) -> None:
        try:
            payload = {
                "updated_at": time.time(),
                "plans": [plan.to_runtime_dict() for plan in self._active_plans.values()],
            }
            atomic_write_json(str(self._persist_path), payload)
        except Exception as exc:
            record_degradation('autonomous_task_engine', exc)
            logger.debug("TaskEngine: active plan persistence skipped: %s", exc)

    def _persist_plan_state(self, plan: TaskPlan) -> None:
        active_plans = getattr(self, "_active_plans", None)
        if isinstance(active_plans, dict) and plan.plan_id in active_plans:
            active_plans[plan.plan_id] = plan
        self._persist_active_plans()

    def _normalize_loaded_plan(self, plan: TaskPlan) -> TaskPlan:
        plan.context = dict(plan.context or {})
        plan.context["recovered_after_restart"] = True
        for step in plan.steps:
            if step.status in {StepStatus.SUCCEEDED, StepStatus.SKIPPED, StepStatus.ROLLED_BACK}:
                continue
            interruption_note = (
                f"Interrupted before completion. Previous state: {step.error}"
                if step.error
                else "Interrupted before this step could be fully verified."
            )
            step.status = StepStatus.PENDING
            step.verified = False
            step.error = interruption_note
            step.completed_at = None
        plan.status = "interrupted"
        return plan

    def _load_persisted_active_plans(self) -> None:
        try:
            if not self._persist_path.exists():
                return
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            record_degradation('autonomous_task_engine', exc)
            logger.debug("TaskEngine: persisted active plan load skipped: %s", exc)
            return

        restored: Dict[str, TaskPlan] = {}
        for item in list(raw.get("plans", []) or []):
            if not isinstance(item, dict):
                continue
            try:
                plan = TaskPlan.from_runtime_dict(item)
            except Exception as exc:
                record_degradation('autonomous_task_engine', exc)
                logger.debug("TaskEngine: persisted plan decode skipped: %s", exc)
                continue
            if not plan.plan_id or not plan.goal:
                continue
            if plan.status in self.TERMINAL_PLAN_STATUSES:
                continue
            restored[plan.plan_id] = self._normalize_loaded_plan(plan)

        if not restored:
            return

        self._active_plans.update(restored)
        for plan in restored.values():
            try:
                self._update_state_goals(plan)
            except Exception as exc:
                record_degradation('autonomous_task_engine', exc)
                logger.debug("TaskEngine: recovered plan state sync skipped: %s", exc)
        self._persist_active_plans()

    def _find_resume_candidate(self, goal: str, context: Optional[Dict[str, Any]]) -> Optional[TaskPlan]:
        ctx = dict(context or {})
        requested_plan_id = str(ctx.get("resume_plan_id", "") or "").strip()
        if requested_plan_id:
            candidate = self._active_plans.get(requested_plan_id)
            if candidate is not None and candidate.status not in self.TERMINAL_PLAN_STATUSES:
                return candidate

        task_id = str(ctx.get("task_id", "") or "").strip()
        best_match: Optional[TaskPlan] = None
        best_score = 0.0
        for plan in list(self._active_plans.values()):
            if plan.status in self.TERMINAL_PLAN_STATUSES:
                continue
            score = 0.0
            if task_id and str(plan.context.get("task_id", "") or "").strip() == task_id:
                score += 1.0
            score += self._goal_overlap_score(plan.goal, goal)
            if plan.status == "interrupted":
                score += 0.2
            if score > best_score:
                best_score = score
                best_match = plan
        if best_score >= 0.65:
            return best_match
        return None

    @staticmethod
    def _plan_summary(plan: TaskPlan) -> str:
        for step in reversed(plan.steps):
            if step.error:
                return str(step.error)[:180]
            if step.result_summary:
                return str(step.result_summary)[:180]
        return str(plan.final_result or "")[:180]

    # === AUDIT FIXES: Logic & Safety ===

    async def _invoke_tool(
        self,
        tool_name: str,
        args: Dict,
        token_id: Optional[str] = None,
        is_shadow: bool = False,
        origin: Optional[str] = None,
    ) -> Any:
        """Invoke a registered tool with capability enforcement and shadow mode support."""
        origin = self._normalize_origin(origin)

        # ── Capability Check (Token-based) ──
        # Note: token_id is generated per-plan based on decomposed steps.
        if not self._capability_manager.verify_access(tool_name, token_id):
            # If not in the plan's static allow-list, check if it's a generally safe tool
            if not self._capability_manager.verify_access(tool_name, None):
                raise PermissionError(f"Capability Error: Tool '{tool_name}' is not authorized for this specific session.")

        # ── Shadow Mode Check (Simulation) ──
        if is_shadow and tool_name in ["run_python", "write_file", "social_post"]:
            logger.info("TaskEngine: [SHADOW MODE] Simulating execution of '%s'", tool_name)
            return f"[SHADOW_SUCCESS] Simulated {tool_name} with args {args}"

        tool_fn = self._tool_registry.get(tool_name)
        if tool_fn is not None:
            call_args = dict(args or {})
            if origin and "origin" not in call_args and self._callable_accepts_kwarg(tool_fn, "origin"):
                call_args["origin"] = origin
            result = tool_fn(**call_args)
            if inspect.isawaitable(result):
                return await result
            return result

        # Unknown tool: try via orchestrator's capability engine
        try:
            from core.container import ServiceContainer
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator and hasattr(orchestrator, "execute_tool"):
                kwargs = {"origin": origin} if origin else {}
                return await orchestrator.execute_tool(tool_name, args, **kwargs)
        except Exception as e:
            record_degradation('autonomous_task_engine', e)
            logger.debug("Orchestrator tool fallback failed: %s", e)

        raise RuntimeError(f"Tool '{tool_name}' not found in registry or orchestrator")

    # ── Public interface ──────────────────────────────────────────────────────

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool that steps can invoke."""
        self._tool_registry[name] = fn
        logger.debug("TaskEngine: registered tool '%s'", name)

    async def execute(
        self,
        goal: str,
        context: Optional[Dict] = None,
        on_progress: Optional[Callable] = None,
        is_shadow: bool = False,
    ) -> TaskResult:
        """Compatibility alias for older callers."""
        return await self.execute_goal(
            goal=goal,
            context=context,
            on_progress=on_progress,
            is_shadow=is_shadow,
        )

    async def execute_goal(
        self,
        goal: str,
        context: Optional[Dict] = None,
        on_progress: Optional[Callable] = None,
        is_shadow: bool = False,
    ) -> TaskResult:
        """Logic for decomposing a goal and executing the plan."""
        requested_plan_id = f"plan_{int(time.time())}"
        trace_id = uuid.uuid4().hex[:8]
        logger.info("TaskEngine: starting goal '%s' (requested_plan=%s) [trace=%s]", goal[:50], requested_plan_id, trace_id)

        # 0. Safety Check: Is this goal/skill allowed?
        if not await self._safety_registry.is_allowed(goal[:50]):
            logger.warning("TaskEngine: Goal '%s' blocked by SafetyRegistry", goal[:50])
            return TaskResult(
                plan_id=requested_plan_id, goal=goal, succeeded=False,
                summary="This autonomous action is currently disabled or restricted by safety policies.",
                trace_id=trace_id,
                steps_completed=0, steps_total=0,
            )

        plan = self._find_resume_candidate(goal, context)
        if plan is None:
            plan = await self._decompose_goal(goal, requested_plan_id, context)
            plan.context = dict(context or {})
        else:
            plan.context = dict(plan.context or {}) | dict(context or {})
            plan.context["resume_count"] = int(plan.context.get("resume_count", 0) or 0) + 1
            plan.context["last_resumed_at"] = time.time()
            logger.info("TaskEngine: resuming interrupted plan %s for goal '%s'", plan.plan_id, goal[:60])

        if self._plan_needs_grounding_repair(plan, goal, plan.context):
            repaired = self._build_grounded_fallback_plan(goal, plan.plan_id, plan.context)
            if repaired is not None:
                logger.warning(
                    "TaskEngine: replacing under-grounded plan %s with deterministic fallback for '%s'",
                    plan.plan_id,
                    goal[:60],
                )
                plan = repaired
            else:
                logger.warning(
                    "TaskEngine: no grounded fallback available for '%s'; refusing to degrade to think-only execution",
                    goal[:60],
                )
                return TaskResult(
                    plan_id=plan.plan_id,
                    goal=goal,
                    succeeded=False,
                    summary="I couldn't build a grounded execution plan with real skills/tools for that task.",
                    trace_id=trace_id,
                    steps_completed=0,
                    steps_total=0,
                )

        plan.trace_id = trace_id
        plan.is_shadow = is_shadow
        self._record_coding_execution(
            "record_execution_plan",
            goal=goal,
            steps=[step.description for step in list(plan.steps or [])],
            plan_id=plan.plan_id,
            objective=goal,
        )

        if not plan.steps:
            return TaskResult(
                plan_id=plan.plan_id, goal=goal, succeeded=False,
                summary="I couldn't decompose this goal into executable steps.",
                steps_completed=0, steps_total=0,
                trace_id=trace_id,
            )

        # 1.5. Escalation & Capability Checks
        if len(plan.steps) > 5 or any(s.tool == "run_python" for s in plan.steps):
            plan.requires_approval = True

        # Register a token for the plan (static allow-list)
        tools_needed = [s.tool for s in plan.steps]
        token = self._capability_manager.generate_token(tools_needed)
        plan.token_id = token.token_id

        self._active_plans[plan.plan_id] = plan
        self._update_state_goals(plan)
        self._persist_plan_state(plan)

        # Pause if escalation required (except in shadow mode)
        if plan.requires_approval and not plan.is_shadow:
            logger.warning("TaskEngine: Plan %s requires human approval. Blocking execution.", plan.plan_id)
            plan.status = "waiting_for_approval"
            self._persist_plan_state(plan)
            event = asyncio.Event()
            self._approval_events[plan.plan_id] = event
            try:
                approved = await asyncio.wait_for(event.wait(), timeout=self.APPROVAL_TIMEOUT)
            except asyncio.TimeoutError:
                approved = False
            finally:
                self._approval_events.pop(plan.plan_id, None)
            if not approved or plan.status == "rejected":
                plan.status = "rejected"
                self._update_state_goals(plan)
                self._active_plans.pop(plan.plan_id, None)
                self._persist_active_plans()
                return TaskResult(
                    plan_id=plan.plan_id, goal=goal, succeeded=False,
                    summary="Plan requires human approval. Call approve_plan(plan_id) to proceed.",
                    steps_completed=0, steps_total=len(plan.steps),
                    trace_id=trace_id,
                )

        # 1.7. Counterfactual deliberation — is this the RIGHT plan?
        #       For multi-step plans, evaluate alternatives before committing.
        if len(plan.steps) >= 3 and not plan.is_shadow:
            try:
                from core.container import ServiceContainer
                cfe = ServiceContainer.get("counterfactual_engine", default=None)
                if cfe:
                    action_space = [
                        {"type": "execute_plan", "description": f"Execute full plan: {goal[:60]}",
                         "params": {"plan_id": plan.plan_id, "steps": len(plan.steps)}},
                        {"type": "plan", "description": f"Re-plan with simpler approach for: {goal[:60]}",
                         "params": {}},
                        {"type": "ask_clarification", "description": f"Ask user to clarify: {goal[:60]}",
                         "params": {}},
                    ]
                    context_dict = {
                        "hedonic_score": 0.5, "curiosity": 0.5, "valence": 0.0,
                        "heartstone_weights": {"curiosity": 0.25, "empathy": 0.25,
                                               "self_preservation": 0.25, "obedience": 0.25},
                    }
                    candidates = await cfe.deliberate(action_space, context_dict)
                    best = cfe.select(candidates)
                    if best and best.action_type != "execute_plan":
                        logger.info(
                            "TaskEngine: Counterfactual deliberation chose '%s' over execute_plan for %s",
                            best.action_type, plan.plan_id,
                        )
                        # Still execute — the deliberation is informational for now
                        # and records learning signal. The counterfactual record will
                        # accumulate regret/relief after the fact.
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                logger.debug("TaskEngine: counterfactual deliberation failed (non-critical): %s", e)

        # 2. Execute steps
        await self._execute_plan(plan, on_progress)

        # 3. Synthesize result
        result = await self._synthesize_result(plan, time.time() - plan.created_at)
        self._record_coding_execution(
            "record_execution_result",
            summary=result.summary,
            succeeded=result.succeeded,
            steps_completed=result.steps_completed,
            steps_total=result.steps_total,
        )
        # Issue ATE-007: Update goals BEFORE deleting from active_plans.
        # Wrap in try/finally so the plan is always cleaned up even if
        # the goal engine write fails — prevents zombie active plans.
        try:
            self._update_state_goals(plan)
            # Mark the associated goal as completed via lifecycle when plan succeeded
            if result.succeeded:
                try:
                    from core.container import ServiceContainer as _SC
                    _ge = _SC.get("goal_engine", default=None)
                    if _ge and hasattr(_ge, "update_task_lifecycle"):
                        await _ge.update_task_lifecycle(
                            task_id=str(plan.context.get("task_id", "") or plan.plan_id),
                            status="completed",
                            summary=result.summary or "",
                            evidence=result.evidence or [],
                        )
                except Exception as _lc_err:
                    record_degradation('autonomous_task_engine', _lc_err)
                    logger.debug("TaskEngine: goal lifecycle completion failed for plan %s: %s", plan.plan_id, _lc_err)
        except Exception as exc:
            record_degradation('autonomous_task_engine', exc)
            logger.error("TaskEngine: goal state sync failed for plan %s: %s", plan.plan_id, exc)
        finally:
            self._active_plans.pop(plan.plan_id, None)
            self._persist_active_plans()

        logger.info(
            "TaskEngine: plan %s complete. Success=%s (%d/%d steps)",
            plan.plan_id, result.succeeded, result.steps_completed, result.steps_total,
        )
        return result

    def approve_plan(self, plan_id: str) -> bool:
        """Signal approval for a plan waiting for human review. Returns True if plan was found."""
        event = self._approval_events.get(plan_id)
        if event:
            if plan_id in self._active_plans:
                self._active_plans[plan_id].status = "approved"
                self._persist_active_plans()
            event.set()
            logger.info("TaskEngine: Plan %s approved by operator.", plan_id)
            return True
        logger.warning("TaskEngine: approve_plan called for unknown/expired plan %s", plan_id)
        return False

    def reject_plan(self, plan_id: str) -> bool:
        """Signal rejection for a plan waiting for human review. Returns True if plan was found."""
        event = self._approval_events.get(plan_id)
        if event:
            if plan_id in self._active_plans:
                self._active_plans[plan_id].status = "rejected"
                self._persist_active_plans()
            event.set()
            logger.info("TaskEngine: Plan %s rejected by operator.", plan_id)
            return True
        logger.warning("TaskEngine: reject_plan called for unknown/expired plan %s", plan_id)
        return False

    def get_active_plans(self) -> List[Dict]:
        """Returns snapshot of all currently running task plans (thread-safe copy)."""
        snapshot = list(self._active_plans.values()) # Issue ATE-006: Snapshot to avoid mutation race
        return [
            {
                "plan_id": p.plan_id,
                "goal":    p.goal[:60],
                "steps":   [s.to_dict() for s in p.steps],
                "status":  p.status,
                "summary": self._plan_summary(p),
                "task_id": str(p.context.get("task_id", "") or p.plan_id),
                "project_id": str(p.context.get("project_id", "") or ""),
                "trace_id": p.trace_id,
                "steps_completed": len(p.succeeded_steps),
                "steps_total": len(p.steps),
                "resumable": p.status in {"interrupted", "waiting_for_approval", "pending"},
            }
            for p in snapshot
        ]

    # ── Decomposition ─────────────────────────────────────────────────────────

    async def _decompose_goal(
        self,
        goal: str,
        plan_id: str,
        context: Optional[Dict],
    ) -> TaskPlan:
        """Use LLM to decompose goal into concrete, verifiable steps."""
        if self._looks_like_learning_resource_bundle(goal):
            bundle_plan = self._build_learning_resource_plan(goal, plan_id, context)
            if bundle_plan is not None:
                logger.info(
                    "TaskEngine: recognized structured learning-resource bundle; using deterministic ingestion plan for '%s'",
                    goal[:60],
                )
                return bundle_plan

        tool_specs = self._build_planning_tool_specs(goal)
        available_tools = [spec["name"] for spec in tool_specs]
        tool_catalog = "\n".join(
            f"- {spec['name']}: {spec['description']}"
            + (f" Args: {spec['args']}" if spec.get("args") else "")
            for spec in tool_specs
        )

        prompt = f"""Decompose this goal into a sequence of concrete steps.

Goal: {goal}

Available tools:
{tool_catalog}

For each step, provide:
1. description: What to do (clear, specific)
2. tool: Which tool to use (must be from the available list)
3. args: Tool arguments as a JSON object
4. success_criterion: How to verify this step succeeded
5. rollback_action: How to undo this step if needed (or null)
6. depends_on: Array of prior step numbers this step depends on (or [])
7. parallel_safe: true only for read-only steps that can safely run concurrently

Rules:
- Maximum {self.MAX_STEPS} steps
- Each step must be atomic and verifiable
- Steps should be ordered by dependency
- Prefer parallel-safe steps when possible
- Only mark parallel_safe=true for read-only tools such as think, web_search, or read_file
- For technical remember steps, only set args.verified=true after a live inspection step and list that dependency
- Reuse the same real tool across multiple steps when needed
- Use a real executable tool instead of `think` whenever the action can actually be performed
- For desktop or app goals, break the work into grounded actions such as open, inspect/look, click/focus, type, verify, then summarize
- Never invent parameter keys; stay inside the listed arguments for the chosen tool

Respond ONLY with a JSON array, no other text:
[
  {{
    "description": "...",
    "tool": "...",
    "args": {{}},
    "success_criterion": "...",
    "rollback_action": null,
    "rollback_args": {{}},
    "depends_on": [],
    "parallel_safe": false
  }}
]"""

        try:
            llm = self.kernel.organs["llm"].get_instance()
            raw = await asyncio.wait_for(
                llm.think(
                    prompt,
                    origin="autonomous_task_engine",
                    is_background=True,
                    prefer_tier="tertiary",
                    allow_cloud_fallback=False,
                ),
                timeout=30.0,
            )

            # Extract JSON from response
            start_idx = raw.find("[")
            end_idx   = raw.rfind("]") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array in response")

            steps_data = json.loads(raw[start_idx:end_idx])
            steps = []
            for i, s in enumerate(steps_data[:self.MAX_STEPS]):
                tool_name = s.get("tool", "think")
                steps.append(TaskStep(
                    step_id           = f"{plan_id}_s{i}",
                    description       = s.get("description", ""),
                    tool              = tool_name,
                    args              = s.get("args", {}),
                    success_criterion = s.get("success_criterion", "step completes without error"),
                    rollback_action   = s.get("rollback_action"),
                    rollback_args     = s.get("rollback_args", {}),
                    depends_on        = self._normalize_depends_on(s.get("depends_on", []), plan_id, i),
                    parallel_safe     = self._coerce_parallel_safe(tool_name, s.get("parallel_safe", False)),
                ))

            logger.info("TaskEngine: decomposed into %d steps", len(steps))

            # Audit Fix 4.1: Register Mycelial Edges (Memory -> Skill/Decomposition)
            if context and context.get("source_memory"):
                await self._mycelial.add_edge(context["source_memory"], goal[:40])

            return TaskPlan(plan_id=plan_id, goal=goal, steps=steps, trace_id="", context=dict(context or {}))

        except Exception as e:
            record_degradation('autonomous_task_engine', e)
            logger.error("TaskEngine: decomposition failed: %s", e)
            if self._requires_grounded_action(goal, context):
                fallback = self._build_grounded_fallback_plan(goal, plan_id, context)
                if fallback is not None:
                    logger.warning(
                        "TaskEngine: using deterministic grounded fallback plan for '%s' after decomposition failure",
                        goal[:60],
                    )
                    return fallback
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[],
                    trace_id="",
                    context=dict(context or {}),
                )

            # Non-embodied fallback: keep the best-effort cognitive step for generic goals.
            return TaskPlan(
                plan_id=plan_id, goal=goal,
                steps=[TaskStep(
                    step_id="fallback",
                    description=goal,
                    tool="think",
                    args={"prompt": goal},
                    success_criterion="response is non-empty",
                )],
                trace_id="",
                context=dict(context or {}),
            )

    @staticmethod
    def _summarize_parameter_schema(schema: Dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return ""
        props = schema.get("properties") or {}
        if not isinstance(props, dict) or not props:
            return ""

        parts: List[str] = []
        for name, spec in list(props.items())[:6]:
            if not isinstance(spec, dict):
                parts.append(str(name))
                continue
            type_name = str(spec.get("type", "any") or "any")
            desc = str(spec.get("description", "") or "").strip()
            label = f"{name}:{type_name}"
            if desc:
                label += f" ({desc[:80]})"
            parts.append(label)
        return "; ".join(parts)

    @staticmethod
    def _looks_like_desktop_goal(goal: str) -> bool:
        lowered = str(goal or "").lower()
        return any(
            marker in lowered
            for marker in (
                "desktop",
                "screen",
                "window",
                "app",
                "application",
                "notes",
                "terminal",
                "browser",
                "tab",
                "click",
                "type",
                "mouse",
                "keyboard",
                "on my computer",
                "on my screen",
            )
        )

    def _build_planning_tool_specs(self, goal: str) -> List[Dict[str, str]]:
        specs: List[Dict[str, str]] = []
        seen: Set[str] = set()

        def _push(name: str, description: str, args: str = "") -> None:
            tool_name = str(name or "").strip()
            if not tool_name or tool_name in seen:
                return
            seen.add(tool_name)
            specs.append(
                {
                    "name": tool_name,
                    "description": str(description or "").strip() or "No description provided.",
                    "args": str(args or "").strip(),
                }
            )

        _push("think", "Reason about the next step or synthesize a grounded summary.", "prompt:string")
        _push("web_search", "Search the web for grounded information.", "query:string")
        _push("run_python", "Execute sandboxed Python code.", "code:string")
        _push("write_file", "Write content to a file.", "path:string; content:string")
        _push("read_file", "Read content from a file.", "path:string")
        _push(
            "remember",
            "Store grounded information after verification.",
            "content:string; verified:boolean; type:string; metadata:object",
        )

        try:
            from core.container import ServiceContainer

            cap = ServiceContainer.get("capability_engine", default=None)
        except Exception:
            cap = None

        if cap and hasattr(cap, "get_tool_definitions"):
            selected_defs = []
            try:
                if hasattr(cap, "select_tool_definitions"):
                    selected_defs = list(cap.select_tool_definitions(objective=goal, max_tools=10) or [])
                else:
                    selected_defs = list(cap.get_tool_definitions() or [])
            except Exception as exc:
                record_degradation('autonomous_task_engine', exc)
                logger.debug("TaskEngine: planning tool selection skipped: %s", exc)
                selected_defs = []

            by_name: Dict[str, Dict[str, Any]] = {}
            try:
                for entry in list(cap.get_tool_definitions() or []):
                    if not isinstance(entry, dict):
                        continue
                    fn = entry.get("function", {}) or {}
                    name = str(fn.get("name", "") or "").strip()
                    if name:
                        by_name[name] = entry
            except Exception as exc:
                record_degradation('autonomous_task_engine', exc)
                logger.debug("TaskEngine: planning tool catalog skipped: %s", exc)

            if self._looks_like_desktop_goal(goal):
                for preferred in self.DESKTOP_TOOL_PREFERENCES:
                    entry = by_name.get(preferred)
                    if entry and entry not in selected_defs:
                        selected_defs.append(entry)

            for entry in selected_defs:
                if not isinstance(entry, dict):
                    continue
                fn = entry.get("function", {}) or {}
                name = str(fn.get("name", "") or "").strip()
                if not name:
                    continue
                _push(
                    name,
                    str(fn.get("description", "") or ""),
                    self._summarize_parameter_schema(fn.get("parameters") or {}),
                )

        return specs

    def _normalize_depends_on(self, raw_depends_on: Any, plan_id: str, step_index: int) -> List[str]:
        dependencies = raw_depends_on if isinstance(raw_depends_on, list) else [raw_depends_on]
        normalized: List[str] = []
        for dependency in dependencies:
            dep_index: Optional[int] = None
            if isinstance(dependency, int):
                dep_index = dependency
            elif isinstance(dependency, str):
                token = dependency.strip()
                if token.startswith(f"{plan_id}_s"):
                    normalized.append(token)
                    continue
                if token.isdigit():
                    dep_index = int(token)
                elif token.startswith("s") and token[1:].isdigit():
                    dep_index = int(token[1:])
            if dep_index is None or dep_index < 0 or dep_index >= step_index:
                continue
            dep_id = f"{plan_id}_s{dep_index}"
            if dep_id not in normalized:
                normalized.append(dep_id)
        return normalized

    def _coerce_parallel_safe(self, tool: str, value: Any) -> bool:
        return bool(value) and str(tool or "") in self.SAFE_PARALLEL_TOOLS

    def _can_run_in_parallel(self, step: TaskStep) -> bool:
        return step.parallel_safe and step.tool in self.SAFE_PARALLEL_TOOLS

    def _looks_technical_fact(self, content: str) -> bool:
        lowered = str(content or "").lower()
        if any(hint in lowered for hint in self.TECHNICAL_FACT_HINTS):
            return True
        return bool(re.search(r"(?:[\w.-]+/)+[\w.-]+\.(?:py|tsx?|jsx?|json|ya?ml|toml|sh)", lowered))

    @staticmethod
    def _matched_skills_from_context(context: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(context, dict):
            return []
        return normalize_matched_skills(context.get("matched_skills"))

    def _requires_grounded_action(self, goal: str, context: Optional[Dict[str, Any]]) -> bool:
        matched_skills = self._matched_skills_from_context(context)
        lowered = str(goal or "").lower()
        if looks_like_multi_step_skill_request(goal, matched_skills):
            return True
        if matched_skills and self._looks_like_desktop_goal(goal):
            return True
        if matched_skills and any(
            marker in lowered
            for marker in ("actually", "on your own", "interact", "perform", "click", "type", "press", "open")
        ):
            return True
        return False

    def _plan_needs_grounding_repair(self, plan: TaskPlan, goal: str, context: Optional[Dict[str, Any]]) -> bool:
        if not self._requires_grounded_action(goal, context):
            return False

        non_think_steps = [step for step in plan.steps if step.tool not in self.NON_EXECUTING_TOOLS]
        if not non_think_steps:
            return True

        matched_skills = self._matched_skills_from_context(context)
        if self._looks_like_desktop_goal(goal) and looks_like_multi_step_skill_request(goal, matched_skills):
            grounded_steps = [
                step for step in plan.steps
                if step.tool in set(self.DESKTOP_TOOL_PREFERENCES) | set(matched_skills)
            ]
            if len(grounded_steps) < 3:
                return True

        return False

    @staticmethod
    def _extract_quoted_text(goal: str) -> str:
        text = str(goal or "")
        for pattern in (
            r"[\"“”']([^\"“”']{1,400})[\"“”']",
            r"\btype\s+exactly\s*:\s*(.+?)(?:,?\s+(?:press|hit|then|and|come back|report)\b|$)",
            r"\b(?:type|write|enter)\s+(.+?)(?:,?\s+(?:press|hit|then|and|come back|report)\b|$)",
        ):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                candidate = " ".join(match.group(1).split()).strip().strip(".,")
                if candidate:
                    return candidate
        return ""

    @staticmethod
    def _extract_terminal_command(goal: str) -> str:
        text = str(goal or "")
        for pattern in (
            r"^\s*(?:execute|run|terminal)(?:\s+the\s+command)?\s*:\s*(.+?)\s*$",
            r"\btype\s+exactly\s*:\s*(.+?)(?:,?\s+(?:press|hit)\s+(?:return|enter)\b|$)",
            r"\b(?:run|execute)\s+(.+?)(?:\.|!|\?|$)",
        ):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                candidate = " ".join(match.group(1).split()).strip().strip(".,")
                if candidate:
                    return candidate
        return ""

    @staticmethod
    def _extract_app_name(goal: str) -> str:
        text = str(goal or "")
        for pattern in (
            r"\bopen(?:\s+up)?\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 ._-]{1,48}?)\s+(?:app|application)\b",
            r"\blaunch\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 ._-]{1,48}?)\s+(?:app|application)?\b",
        ):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return " ".join(match.group(1).split()).strip(" .")
        lowered = text.lower()
        if "terminal" in lowered:
            return "Terminal"
        if "notes" in lowered:
            return "Notes"
        if "browser" in lowered:
            return "Browser"
        return ""

    @staticmethod
    def _extract_url(goal: str) -> str:
        match = re.search(r"https?://[^\s<>\"')\]]+", str(goal or ""))
        return match.group(0) if match else ""

    @classmethod
    def _looks_like_learning_bundle_header(cls, line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped or "http://" in stripped or "https://" in stripped:
            return False
        if not stripped.endswith(":") or len(stripped) > 120:
            return False
        lowered = stripped[:-1].strip().lower()
        return any(marker in lowered for marker in cls.LEARNING_BUNDLE_SECTION_MARKERS)

    @classmethod
    def _parse_learning_resource_line(cls, line: str, category: str = "") -> Optional[Dict[str, str]]:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", str(line or "").strip())
        if not cleaned or cls._looks_like_learning_bundle_header(cleaned):
            return None

        head, sep, tail = cleaned.rpartition(":")
        if not sep:
            return None

        description = tail.strip().lstrip(":").strip()
        if len(description) < 8:
            return None

        title = head.strip()
        url = ""
        creator = ""
        url_match = re.match(r"^(?P<title>.+?)\s+\((?P<url>https?://[^)]+)\)\s*$", title)
        if url_match:
            title = url_match.group("title").strip()
            url = url_match.group("url").strip()
        elif " - " in title:
            title, creator = title.rsplit(" - ", 1)
            title = title.strip()
            creator = creator.strip()

        if not title:
            return None

        return {
            "category": str(category or "").strip(),
            "title": title,
            "url": url,
            "creator": creator,
            "description": description,
        }

    @classmethod
    def _looks_like_learning_resource_bundle(cls, goal: str) -> bool:
        text = str(goal or "")
        if len(text) < 280:
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 6:
            return False

        lowered = text.lower()
        url_count = len(re.findall(r"https?://[^\s<>\"')\]]+", text))
        header_count = sum(1 for line in lines if cls._looks_like_learning_bundle_header(line))

        category = ""
        resource_count = 0
        for line in lines:
            if cls._looks_like_learning_bundle_header(line):
                category = line.rstrip(":").strip()
                continue
            if cls._parse_learning_resource_line(line, category):
                resource_count += 1

        intro_hit = any(marker in lowered for marker in cls.LEARNING_BUNDLE_INTRO_MARKERS)
        return (
            (url_count >= 4 and resource_count >= 5)
            or (header_count >= 2 and resource_count >= 5)
            or (intro_hit and resource_count >= 4)
        )

    @staticmethod
    def _chunk_learning_resource_entries(entries: List[Dict[str, str]], max_chunks: int) -> List[List[Dict[str, str]]]:
        if not entries:
            return []
        max_chunks = max(1, int(max_chunks or 1))
        chunk_count = min(len(entries), max_chunks)
        chunk_size = max(1, (len(entries) + chunk_count - 1) // chunk_count)
        return [entries[idx: idx + chunk_size] for idx in range(0, len(entries), chunk_size)]

    def _build_learning_resource_plan(
        self,
        goal: str,
        plan_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[TaskPlan]:
        lines = [line.strip() for line in str(goal or "").splitlines() if line.strip()]
        category = ""
        entries: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        for line in lines:
            if self._looks_like_learning_bundle_header(line):
                category = line.rstrip(":").strip()
                continue
            parsed = self._parse_learning_resource_line(line, category)
            if not parsed:
                continue
            dedupe_key = (
                parsed.get("category", "").lower(),
                parsed.get("title", "").lower(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            entries.append(parsed)

        if not entries:
            return None

        category_counts: Dict[str, int] = {}
        for entry in entries:
            label = entry.get("category") or "Uncategorized"
            category_counts[label] = category_counts.get(label, 0) + 1

        steps: List[TaskStep] = []
        category_summary = "; ".join(
            f"{name} ({count})"
            for name, count in category_counts.items()
        )
        steps.append(
            TaskStep(
                step_id=f"{plan_id}_s0",
                description="Store the index of the structured learning-resource bundle.",
                tool="remember",
                args={
                    "content": (
                        "Bryan shared a structured learning-resource bundle for Aura. "
                        f"Categories: {category_summary}. "
                        "Preserve each recommendation as its own future research lead instead of flattening the list."
                    ),
                    "verified": False,
                    "type": "learning_resource_index",
                    "metadata": {
                        "entry_count": len(entries),
                        "categories": list(category_counts.keys()),
                    },
                },
                success_criterion="result contains 'Remembered:'",
            )
        )

        chunks = self._chunk_learning_resource_entries(entries, max_chunks=max(1, self.MAX_STEPS - 1))
        for index, chunk in enumerate(chunks, start=1):
            chunk_lines = [
                f"Learning resource bundle from Bryan - chunk {index}/{len(chunks)}.",
                "Treat each bullet below as a separate recommendation and future research thread.",
            ]
            entry_titles: List[str] = []
            categories = sorted({entry.get("category") or "Uncategorized" for entry in chunk})
            for entry in chunk:
                title = entry.get("title", "").strip()
                entry_titles.append(title)
                location = entry.get("url") or entry.get("creator") or "reference pending"
                label = entry.get("category") or "Uncategorized"
                chunk_lines.append(
                    f"- [{label}] {title} - {location} - {entry.get('description', '').strip()}"
                )

            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description=f"Remember learning-resource chunk {index} of {len(chunks)}.",
                    tool="remember",
                    args={
                        "content": "\n".join(chunk_lines),
                        "verified": False,
                        "type": "learning_resource_chunk",
                        "metadata": {
                            "chunk_index": index,
                            "chunk_total": len(chunks),
                            "entry_titles": entry_titles,
                            "categories": categories,
                        },
                    },
                    success_criterion="result contains 'Remembered:'",
                    depends_on=[f"{plan_id}_s0"],
                )
            )

        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps[: self.MAX_STEPS],
            trace_id="",
            context=dict(context or {}),
        )

    def _build_desktop_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[TaskPlan]:
        matched_skills = self._matched_skills_from_context(context)
        if "computer_use" not in matched_skills:
            return None
        desktop_tool = "computer_use"

        app_name = self._extract_app_name(goal)
        if not app_name:
            return None

        typed_text = self._extract_quoted_text(goal)
        if app_name.lower() == "terminal":
            typed_text = self._extract_terminal_command(goal) or typed_text

        steps: List[TaskStep] = [
            TaskStep(
                step_id=f"{plan_id}_s0",
                description=f"Open {app_name}.",
                tool=desktop_tool,
                args={"action": "open_app", "target": app_name},
                success_criterion=f"result contains '{app_name}'",
            )
        ]

        next_dep = f"{plan_id}_s0"
        if app_name.lower() == "notes":
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s1",
                    description="Create a new note.",
                    tool=desktop_tool,
                    args={"action": "hotkey", "target": "command+n"},
                    success_criterion="result contains 'command+n'",
                    depends_on=[next_dep],
                )
            )
            next_dep = f"{plan_id}_s1"

        if typed_text:
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description=f"Type the requested text in {app_name}.",
                    tool=desktop_tool,
                    args={"action": "type", "target": typed_text},
                    success_criterion=f"result contains '{typed_text[:80]}'",
                    depends_on=[next_dep],
                )
            )
            next_dep = steps[-1].step_id

        if app_name.lower() == "terminal" and typed_text:
            steps.append(
                TaskStep(
                    step_id=f"{plan_id}_s{len(steps)}",
                    description="Submit the command in Terminal.",
                    tool=desktop_tool,
                    args={"action": "hotkey", "target": "enter"},
                    success_criterion="result contains 'enter'",
                    depends_on=[next_dep],
                )
            )
            next_dep = steps[-1].step_id

        return TaskPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            trace_id="",
            context=dict(context or {}),
        )

    def _build_single_skill_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[TaskPlan]:
        matched_skills = self._matched_skills_from_context(context)

        if "sovereign_terminal" in matched_skills:
            command = self._extract_terminal_command(goal)
            if command:
                needle = command.replace("echo ", "", 1).strip().strip("'\"")
                criterion = "step completes without error"
                if needle:
                    criterion = f"result contains '{needle[:80]}'"
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description="Execute the requested terminal command.",
                            tool="sovereign_terminal",
                            args={"action": "execute", "command": command},
                            success_criterion=criterion,
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        if "computer_use" in matched_skills:
            url = self._extract_url(goal)
            if url:
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description="Open the requested URL.",
                            tool="computer_use",
                            args={"action": "open_url", "target": url},
                            success_criterion=f"result contains '{url[:80]}'",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )
            app_name = self._extract_app_name(goal)
            if app_name:
                return TaskPlan(
                    plan_id=plan_id,
                    goal=goal,
                    steps=[
                        TaskStep(
                            step_id=f"{plan_id}_s0",
                            description=f"Open {app_name}.",
                            tool="computer_use",
                            args={"action": "open_app", "target": app_name},
                            success_criterion=f"result contains '{app_name}'",
                        )
                    ],
                    trace_id="",
                    context=dict(context or {}),
                )

        return None

    def _build_grounded_fallback_plan(
        self,
        goal: str,
        plan_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[TaskPlan]:
        if self._looks_like_desktop_goal(goal):
            desktop = self._build_desktop_fallback_plan(goal, plan_id, context)
            if desktop is not None:
                return desktop
        return self._build_single_skill_fallback_plan(goal, plan_id, context)

    def _summary_hedges_completion(self, summary: str) -> bool:
        lowered = str(summary or "").lower()
        return any(marker in lowered for marker in self.COMPLETION_HEDGE_MARKERS)

    def _compact_tool_result(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, bytes):
            text = result.decode("utf-8", errors="ignore")
        elif isinstance(result, (dict, list, tuple, set)):
            try:
                text = json.dumps(result, ensure_ascii=False, default=str, sort_keys=True)
            except Exception:
                text = repr(result)
        else:
            text = str(result)
        text = text.strip()
        if not text:
            return ""

        prefix = ""
        if isinstance(result, dict):
            prefix = f"[dict keys={len(result)}] "
        elif isinstance(result, (list, tuple, set)):
            prefix = f"[items={len(result)}] "

        if len(text) <= self.MAX_RESULT_CHARS:
            return f"{prefix}{text}" if prefix else text

        head = text[: int(self.MAX_RESULT_CHARS * 0.65)]
        tail = text[-int(self.MAX_RESULT_CHARS * 0.2):]
        omitted = max(0, len(text) - len(head) - len(tail))
        compacted = f"{head}\n...[trimmed {omitted} chars]...\n{tail}"
        return f"{prefix}{compacted}" if prefix else compacted

    def _report_progress(self, step: TaskStep, on_progress: Optional[Callable]) -> None:
        if on_progress is None:
            return
        try:
            on_progress(step.to_dict())
        except Exception as e:
            record_degradation('autonomous_task_engine', e)
            logger.debug("Ignored Exception in autonomous_task_engine.py: %s", e)

    async def _fail_plan(self, plan: TaskPlan, completed_ids: Set[str], reason: str) -> None:
        logger.error("TaskEngine: %s", reason)
        await self._rollback_completed(plan, completed_ids)
        for remaining in plan.steps:
            if remaining.status == StepStatus.PENDING:
                remaining.status = StepStatus.SKIPPED
                remaining.error = "skipped due to earlier failure"
        plan.status = "failed"
        self._persist_plan_state(plan)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_plan(
        self,
        plan: TaskPlan,
        on_progress: Optional[Callable],
    ) -> None:
        """Execute all steps, respecting dependencies and handling failures."""
        # Final safety check before execution
        if not await self._safety_registry.is_allowed(plan.goal[:50]):
             plan.status = "failed"
             self._persist_plan_state(plan)
             logger.error("TaskEngine: Execution aborted for '%s' - Safety revocation triggered.", plan.goal[:50])
             return

        plan.status = "running"
        self._persist_plan_state(plan)
        completed_ids: Set[str] = {step.step_id for step in plan.succeeded_steps}

        while True:
            pending_steps = [step for step in plan.steps if step.status == StepStatus.PENDING]
            if not pending_steps:
                break

            ready_steps = [
                step for step in pending_steps
                if all(dep in completed_ids for dep in step.depends_on)
            ]
            if not ready_steps:
                for step in pending_steps:
                    step.status = StepStatus.SKIPPED
                    step.error = "dependency cycle or unsatisfied dependency"
                    self._report_progress(step, on_progress)
                plan.status = "failed"
                self._persist_plan_state(plan)
                logger.error("TaskEngine: no runnable steps remain for plan %s", plan.plan_id)
                return

            parallel_wave = [step for step in ready_steps if self._can_run_in_parallel(step)][:self.MAX_PARALLEL_STEPS]
            if parallel_wave:
                await asyncio.gather(*(self._execute_step_with_retry(step, plan) for step in parallel_wave))
                for step in parallel_wave:
                    if step.status == StepStatus.SUCCEEDED:
                        completed_ids.add(step.step_id)
                    self._report_progress(step, on_progress)
                self._persist_plan_state(plan)
                failed_step = next((step for step in parallel_wave if step.status == StepStatus.FAILED), None)
                if failed_step is not None:
                    await self._fail_plan(
                        plan,
                        completed_ids,
                        f"step '{failed_step.description[:60]}' failed after {failed_step.attempts} attempts",
                    )
                    return
                continue

            step = ready_steps[0]
            await self._execute_step_with_retry(step, plan)
            if step.status == StepStatus.SUCCEEDED:
                completed_ids.add(step.step_id)
            self._report_progress(step, on_progress)
            self._persist_plan_state(plan)

            if step.status == StepStatus.FAILED:
                await self._fail_plan(
                    plan,
                    completed_ids,
                    f"step '{step.description[:60]}' failed after {step.attempts} attempts",
                )
                return

        if plan.status != "failed":
            plan.status = "succeeded" if plan.all_complete and not plan.any_failed else "partial"
            self._persist_plan_state(plan)

    async def _execute_step_with_retry(self, step: TaskStep, plan: TaskPlan) -> None:
        """Execute a single step with up to MAX_RETRIES retries."""
        for attempt in range(self.MAX_RETRIES + 1):
            step.attempts   += 1
            step.status      = StepStatus.RUNNING
            step.started_at  = time.time()
            step.completed_at = None
            self._record_coding_execution(
                "record_execution_step",
                step_description=step.description,
                tool_name=step.tool,
                status="running",
                attempt=step.attempts,
                success_criterion=step.success_criterion,
                steps_completed=len(plan.succeeded_steps),
                steps_total=len(plan.steps),
            )
            self._persist_plan_state(plan)

            try:
                # Issue ATE-001: asyncio.timeout() is 3.11+, use wait_for for compatibility
                raw_result = await asyncio.wait_for(
                    self._invoke_tool(
                        step.tool,
                        step.args,
                        plan.token_id,
                        plan.is_shadow,
                        origin=self._context_origin(plan.context),
                    ),
                    timeout=self.STEP_TIMEOUT
                )

                step.raw_result = self._compact_tool_result(raw_result)
                step.result_summary = step.raw_result

                # Verify the step succeeded
                self._record_coding_execution(
                    "record_execution_step",
                    step_description=step.description,
                    tool_name=step.tool,
                    status="verifying",
                    attempt=step.attempts,
                    result_summary=step.result_summary or "",
                    success_criterion=step.success_criterion,
                    steps_completed=len(plan.succeeded_steps),
                    steps_total=len(plan.steps),
                )
                verified = await self._verify_step(step, raw_result)
                if verified:
                    step.status       = StepStatus.SUCCEEDED
                    step.verified     = True
                    step.completed_at = time.time()
                    self._record_coding_execution(
                        "record_execution_step",
                        step_description=step.description,
                        tool_name=step.tool,
                        status="verified",
                        attempt=step.attempts,
                        result_summary=step.result_summary or "",
                        success_criterion=step.success_criterion,
                        steps_completed=len(plan.succeeded_steps) + 1,
                        steps_total=len(plan.steps),
                    )
                    logger.debug("TaskEngine: step '%s' succeeded (attempt %d)", step.description[:40], attempt + 1)
                    self._persist_plan_state(plan)
                    return
                else:
                    # Verification failed — modify args and retry
                    step.error = "verification failed"
                    self._record_coding_execution(
                        "record_execution_step",
                        step_description=step.description,
                        tool_name=step.tool,
                        status="verification_failed",
                        attempt=step.attempts,
                        result_summary=step.result_summary or "",
                        error=step.error,
                        success_criterion=step.success_criterion,
                        steps_completed=len(plan.succeeded_steps),
                        steps_total=len(plan.steps),
                    )
                    logger.warning("TaskEngine: step '%s' verification failed (attempt %d)", step.description[:40], attempt + 1)
                    # Modify args for retry: ask LLM for an alternative approach
                    if attempt < self.MAX_RETRIES - 1:
                        new_args = await self._get_alternative_approach(step)
                        if new_args is None:
                            step.error = "verification failed with no viable alternative"
                            logger.warning(
                                "TaskEngine: no viable alternative for '%s'. Failing fast after verification miss.",
                                step.description[:40],
                            )
                            self._persist_plan_state(plan)
                            break
                        step.args = new_args
                        self._record_coding_execution(
                            "record_execution_repair",
                            step_description=step.description,
                            reason=step.error or step.success_criterion,
                            new_args=step.args,
                        )
                    self._persist_plan_state(plan)

            except asyncio.TimeoutError:
                step.error = f"timeout after {self.STEP_TIMEOUT}s"
                self._record_coding_execution(
                    "record_execution_step",
                    step_description=step.description,
                    tool_name=step.tool,
                    status="timeout",
                    attempt=step.attempts,
                    error=step.error,
                    success_criterion=step.success_criterion,
                    steps_completed=len(plan.succeeded_steps),
                    steps_total=len(plan.steps),
                )
                logger.warning("TaskEngine: step '%s' timed out (attempt %d)", step.description[:40], attempt + 1)
                self._persist_plan_state(plan)
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                step.error = str(e)
                self._record_coding_execution(
                    "record_execution_step",
                    step_description=step.description,
                    tool_name=step.tool,
                    status="failed",
                    attempt=step.attempts,
                    error=step.error,
                    success_criterion=step.success_criterion,
                    steps_completed=len(plan.succeeded_steps),
                    steps_total=len(plan.steps),
                )
                logger.warning("TaskEngine: step '%s' error: %s (attempt %d)", step.description[:40], e, attempt + 1)
                self._persist_plan_state(plan)

        # If all retries fail
        step.status = StepStatus.FAILED
        step.completed_at = time.time()
        self._persist_plan_state(plan)

    # ── Verification ─────────────────────────────────────────────────────────

    async def _verify_step(self, step: TaskStep, result: Any) -> bool:
        """
        LLM-based verification: does the result satisfy the success criterion?

        This is the key to reliable task execution. We don't just check if a
        tool ran — we check if it produced the expected outcome.
        """
        if not result:
            return False

        result_str = str(result)[:1000] if result else ""
        criterion = str(step.success_criterion or "").lower()

        if isinstance(result, dict):
            if result.get("verified") is True:
                return True
            if result.get("verified") is False:
                return False
            if result.get("ok") is False:
                return False
            exit_code = result.get("exit_code", result.get("return_code"))
            if exit_code not in (None, 0):
                return False
            if result.get("error"):
                return False
            if criterion and any(marker in criterion for marker in ("no error", "without error", "completes")):
                if exit_code == 0 and not result.get("stderr"):
                    return True

        if criterion:
            contains_match = re.search(r"(?:contains?|includes?|mentions?)\s+['\"]([^'\"]+)['\"]", criterion)
            if contains_match:
                needle = contains_match.group(1).strip().lower()
                if needle and needle in result_str.lower():
                    return True
            if any(marker in criterion for marker in ("non-empty", "non empty", "response is non-empty", "any result")):
                return bool(result_str.strip())
            if "file exists" in criterion and any(token in result_str.lower() for token in ("exists", "found", "present")):
                return True

        # Fast path: trivial criteria
        trivial_pass = ["step completes", "non-empty", "any result", "no error"]
        if any(t in step.success_criterion.lower() for t in trivial_pass):
            return bool(result_str.strip())

        if step.tool in {"read_file", "think"} and self._looks_technical_fact(result_str):
            return True

        try:
            llm = self.kernel.organs["llm"].get_instance()
            prompt = (
                f"Did this step succeed?\n\n"
                f"Step: {step.description}\n"
                f"Success criterion: {step.success_criterion}\n"
                f"Result: {result_str}\n\n"
                "Answer with ONLY 'YES' or 'NO' followed by one sentence of evidence."
            )
            raw = await asyncio.wait_for(
                llm.think(
                    prompt,
                    origin="autonomous_task_engine",
                    is_background=True,
                    prefer_tier="tertiary",
                    allow_cloud_fallback=False,
                ),
                timeout=15.0,
            )
            verdict = raw.strip().upper()
            passed  = verdict.startswith("YES")
            logger.debug("Verification: %s → %s", step.description[:40], verdict[:50])
            return passed
        except Exception as e:
            record_degradation('autonomous_task_engine', e)
            logger.debug("Verification LLM call failed: %s. Assuming pass.", e)
            return bool(result_str.strip())

    async def _get_alternative_approach(self, step: TaskStep) -> Optional[Dict]:
        """Ask LLM to suggest alternative args after verification failure."""
        try:
            llm = self.kernel.organs["llm"].get_instance()
            prompt = (
                f"This step failed verification. Suggest alternative arguments.\n\n"
                f"Step: {step.description}\n"
                f"Tool: {step.tool}\n"
                f"Original args: {json.dumps(step.args)}\n"
                f"Error: {step.error}\n"
                f"Success criterion: {step.success_criterion}\n\n"
                "Respond ONLY with a JSON object of new args, no other text."
            )
            raw = await asyncio.wait_for(
                llm.think(
                    prompt,
                    origin="autonomous_task_engine",
                    is_background=True,
                    prefer_tier="tertiary",
                    allow_cloud_fallback=False,
                ),
                timeout=15.0,
            )
            start = raw.find("{"); end = raw.rfind("}") + 1
            if start != -1 and end > start:
                new_args = json.loads(raw[start:end])
                # Issue ATE-005: Only accept genuinely different args
                if new_args != step.args:
                    return new_args
                logger.debug("Alternative approach returned identical args for '%s'", step.description[:40])
        except Exception as e:
            record_degradation('autonomous_task_engine', e)
            logger.debug("Alternative approach generation failed: %s", e)

        # Signal exhaustion rather than silently looping identically
        logger.warning("No alternative found for '%s'; step will likely fail.", step.description[:40])
        return None

    # ── Rollback ─────────────────────────────────────────────────────────────

    async def _rollback_completed(self, plan: TaskPlan, completed_ids: set) -> None:
        """Rollback completed steps in reverse order after failure."""
        to_rollback = [
            s for s in reversed(plan.steps)
            if s.step_id in completed_ids and s.rollback_action
        ]
        for step in to_rollback:
            logger.info("TaskEngine: rolling back '%s'", step.description[:40])
            try:
                # Issue ATE-003: Pass rollback_args instead of empty dict
                await self._invoke_tool(
                    step.rollback_action,
                    step.rollback_args,
                    origin=self._context_origin(plan.context),
                )
                step.status = StepStatus.ROLLED_BACK
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                logger.warning("Rollback failed for step '%s': %s", step.description[:40], e)

    # ── Result synthesis ──────────────────────────────────────────────────────

    async def _synthesize_result(self, plan: TaskPlan, duration: float) -> TaskResult:
        """Use LLM to synthesize a natural language summary of what was accomplished."""
        plan.completed_at = time.time()
        succeeded_steps   = plan.succeeded_steps # Initialize succeeded_steps here
        plan.status       = "succeeded" if plan.all_complete and not plan.any_failed else "partial"

        # Build evidence from step results
        evidence = []
        for step in succeeded_steps:
            if step.raw_result:
                result_str = str(step.raw_result)[:200]
                evidence.append(f"[{step.description[:40]}]: {result_str}")

        # LLM synthesis
        try:
            llm = self.kernel.organs["llm"].get_instance()
            steps_summary = "\n".join(
                f"  {'✓' if s.status == StepStatus.SUCCEEDED else '✗'} {s.description}"
                for s in plan.steps
            )
            findings = "\n".join(evidence[:5])
            prompt = (
                f"Summarize what was accomplished for this goal.\n\n"
                f"Goal: {plan.goal}\n"
                f"Steps executed:\n{steps_summary}\n\n"
                f"Key findings:\n{findings}\n\n"
                f"Write a concise, first-person summary of what was done and what was found. "
                f"Be specific. If steps failed, acknowledge it honestly. 3-5 sentences max."
            )
            summary = await asyncio.wait_for(
                llm.think(
                    prompt,
                    origin="autonomous_task_engine",
                    is_background=True,
                    prefer_tier="tertiary",
                    allow_cloud_fallback=False,
                ),
                timeout=20.0,
            )
            plan.final_result = summary
        except Exception:
            n_done = len(succeeded_steps)
            n_total = len(plan.steps)
            summary = (
                f"Completed {n_done}/{n_total} steps toward '{plan.goal}'. "
                + (f"Key finding: {evidence[0]}" if evidence else "")
            )
            plan.final_result = summary

        succeeded = not plan.any_failed
        if self._requires_grounded_action(plan.goal, plan.context) and self._summary_hedges_completion(plan.final_result or summary):
            succeeded = False

        return TaskResult(
            plan_id=plan.plan_id,
            goal=plan.goal,
            succeeded=succeeded,
            summary=plan.final_result or "Task failed",
            trace_id=plan.trace_id,
            steps_completed=len(plan.succeeded_steps),
            steps_total=len(plan.steps),
            evidence=evidence[:6],
            duration_s=time.time() - plan.created_at
        )

    # ── State integration ─────────────────────────────────────────────────────

    def _update_state_goals(self, plan: TaskPlan) -> None:
        """Keep AuraState.cognition.active_goals in sync with running plans."""
        try:
            from core.container import ServiceContainer

            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "sync_task_plan"):
                goal_engine.sync_task_plan(plan, context=getattr(plan, "context", None))

            state = self.kernel.state
            if state is None:
                return
            cognition = getattr(state, "cognition", None)
            if cognition is None:
                return

            if goal_engine and hasattr(goal_engine, "get_active_goals"):
                try:
                    cognition.active_goals = goal_engine.get_active_goals(
                        limit=6,
                        include_external=False,
                        actionable_only=True,
                    )
                    if not cognition.active_goals:
                        cognition.active_goals = goal_engine.get_active_goals(
                            limit=6,
                            include_external=True,
                            actionable_only=True,
                        )
                    if cognition.active_goals and not getattr(cognition, "current_objective", None):
                        cognition.current_objective = str(
                            cognition.active_goals[0].get("objective")
                            or cognition.active_goals[0].get("goal")
                            or cognition.active_goals[0].get("name")
                            or ""
                        )
                    return
                except Exception as exc:
                    record_degradation('autonomous_task_engine', exc)
                    logger.debug("GoalEngine-backed state sync failed: %s", exc)

            if not hasattr(cognition, "active_goals") or cognition.active_goals is None:
                cognition.active_goals = []

            cognition.active_goals = [
                g for g in cognition.active_goals
                if g.get("plan_id") != plan.plan_id
            ]
            if plan.plan_id in self._active_plans:
                cognition.active_goals.append({
                    "plan_id": plan.plan_id,
                    "goal": plan.goal[:80],
                    "status": plan.status,
                    "steps_done": len(plan.succeeded_steps),
                    "steps_total": len(plan.steps),
                })
        except Exception as e:
            record_degradation('autonomous_task_engine', e)
            logger.debug("State goal update failed: %s", e)

    # ── Default tools ─────────────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        """Register the core tool set that the engine can always use."""

        async def _think(prompt: str, **kwargs) -> str:
            """Use Aura's LLM to reason about something."""
            llm = self.kernel.organs["llm"].get_instance()
            return await llm.think(
                prompt,
                origin="autonomous_task_engine",
                is_background=True,
                prefer_tier="tertiary",
                allow_cloud_fallback=False,
            )

        async def _web_search(query: str, **kwargs) -> str:
            """Search the web for information."""
            try:
                from core.container import ServiceContainer
                orch = ServiceContainer.get("orchestrator", default=None)
                if orch:
                    origin = self._normalize_origin(kwargs.get("origin"))
                    if origin:
                        return await orch.execute_tool("web_search", {"query": query}, origin=origin)
                    return await orch.execute_tool("web_search", {"query": query})
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                return f"Search failed: {e}"
            return f"Search tool not available for: {query}"

        async def _run_python(code: str, **kwargs) -> str:
            """Execute Python code in a sandboxed environment."""
            try:
                from core.container import ServiceContainer
                orch = ServiceContainer.get("orchestrator", default=None)
                if orch and hasattr(orch, "execute_tool"):
                    origin = self._normalize_origin(kwargs.get("origin"))
                    if origin:
                        result = await orch.execute_tool("run_python", {"code": code}, origin=origin)
                    else:
                        result = await orch.execute_tool("run_python", {"code": code})
                    return str(result)
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                return f"Python execution failed: {e}"
            return "Python execution not available"

        async def _write_file(path: str, content: str, **kwargs) -> str:
            """Write content to a file."""
            import aiofiles
            try:
                async with aiofiles.open(path, "w") as f:
                    await f.write(content)
                return f"Written to {path}"
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                return f"Write failed: {e}"

        async def _read_file(path: str, **kwargs) -> str:
            """Read content from a file."""
            import aiofiles
            try:
                async with aiofiles.open(path) as f:
                    return await f.read()
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                return f"Read failed: {e}"

        async def _remember(
            content: str,
            verified: bool = False,
            type: str = "observation",
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs,
        ) -> str:
            """Store something in Aura's knowledge graph."""
            if self._looks_technical_fact(content) and not verified:
                raise ValueError("Technical memory writes require verified=true after a live read or tool-backed check")
            try:
                from core.container import ServiceContainer
                kg = ServiceContainer.get("knowledge_graph", default=None)
                if kg:
                    kg.add_knowledge(
                        content=content,
                        type=str(type or "observation"),
                        source=str(kwargs.get("source", "task_engine") or "task_engine"),
                        metadata=metadata or {},
                    )
                    return f"Remembered: {content[:80]}"
            except Exception as e:
                record_degradation('autonomous_task_engine', e)
                return f"Remember failed: {e}"
            return "Memory not available"

        self._tool_registry["think"]      = _think
        self._tool_registry["web_search"] = _web_search
        self._tool_registry["run_python"] = _run_python
        self._tool_registry["write_file"] = _write_file
        self._tool_registry["read_file"]  = _read_file
        self._tool_registry["remember"]   = _remember

        logger.debug("TaskEngine: %d default tools registered", len(self._tool_registry))


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[AutonomousTaskEngine] = None


def get_task_engine(kernel: Any = None, force_reset: bool = False) -> AutonomousTaskEngine:
    global _engine
    # Issue ATE-002: Handle stale kernel reference with force_reset
    if _engine is None or force_reset:
        if kernel is None:
            from core.container import ServiceContainer
            kernel = ServiceContainer.get("aura_kernel", default=None) or ServiceContainer.get("kernel", default=None)
        _engine = AutonomousTaskEngine(kernel)
    return _engine


def reset_task_engine() -> None:
    """Force-reset the singleton (use on kernel hot-reload)."""
    global _engine
    _engine = None
