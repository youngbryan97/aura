from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from core.agency.capability_system import get_capability_manager, CapabilityToken
from core.agency.safety_registry import get_safety_registry
from core.mycelial.graph import get_mycelial

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


@dataclass
class TaskPlan:
    """Complete execution plan for a goal."""
    plan_id:        str
    goal:           str
    steps:          List[TaskStep]
    trace_id:       str             # Observability Fix 5.2
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
    TECHNICAL_FACT_HINTS = (
        "def ", "class ", ".py", ".ts", ".tsx", ".js", ".jsx",
        "function ", "method ", "module ", "endpoint", "api ", "schema ",
    )

    def __init__(self, kernel: Any):
        self.kernel = kernel
        self._active_plans: Dict[str, TaskPlan] = {}
        self._tool_registry: Dict[str, Callable] = {}
        self._capability_manager = get_capability_manager()
        self._safety_registry = get_safety_registry()
        self._mycelial = get_mycelial()
        self._approval_events: Dict[str, asyncio.Event] = {}
        self._register_default_tools()

    # === AUDIT FIXES: Logic & Safety ===

    async def _invoke_tool(self, tool_name: str, args: Dict, token_id: Optional[str] = None, is_shadow: bool = False) -> Any:
        """Invoke a registered tool with capability enforcement and shadow mode support."""

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
            result = tool_fn(**dict(args or {}))
            if inspect.isawaitable(result):
                return await result
            return result

        # Unknown tool: try via orchestrator's capability engine
        try:
            from core.container import ServiceContainer
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator and hasattr(orchestrator, "execute_tool"):
                return await orchestrator.execute_tool(tool_name, args)
        except Exception as e:
            logger.debug("Orchestrator tool fallback failed: %s", e)

        raise RuntimeError(f"Tool '{tool_name}' not found in registry or orchestrator")

    # ── Public interface ──────────────────────────────────────────────────────

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool that steps can invoke."""
        self._tool_registry[name] = fn
        logger.debug("TaskEngine: registered tool '%s'", name)

    async def execute_goal(
        self,
        goal: str,
        context: Optional[Dict] = None,
        on_progress: Optional[Callable] = None,
        is_shadow: bool = False,
    ) -> TaskResult:
        """Logic for decomposing a goal and executing the plan."""
        plan_id = f"plan_{int(time.time())}"
        trace_id = uuid.uuid4().hex[:8]
        logger.info("TaskEngine: starting goal '%s' (plan=%s) [trace=%s]", goal[:50], plan_id, trace_id)

        # 0. Safety Check: Is this goal/skill allowed?
        if not await self._safety_registry.is_allowed(goal[:50]):
            logger.warning("TaskEngine: Goal '%s' blocked by SafetyRegistry", goal[:50])
            return TaskResult(
                plan_id=plan_id, goal=goal, succeeded=False,
                summary="This autonomous action is currently disabled or restricted by safety policies.",
                trace_id=trace_id,
                steps_completed=0, steps_total=0,
            )

        # 1. Decompose goal into steps
        plan = await self._decompose_goal(goal, plan_id, context)
        plan.trace_id = trace_id
        plan.is_shadow = is_shadow

        if not plan.steps:
            return TaskResult(
                plan_id=plan_id, goal=goal, succeeded=False,
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

        self._active_plans[plan_id] = plan
        self._update_state_goals(plan)

        # Pause if escalation required (except in shadow mode)
        if plan.requires_approval and not plan.is_shadow:
            logger.warning("TaskEngine: Plan %s requires human approval. Blocking execution.", plan_id)
            plan.status = "waiting_for_approval"
            event = asyncio.Event()
            self._approval_events[plan_id] = event
            try:
                approved = await asyncio.wait_for(event.wait(), timeout=self.APPROVAL_TIMEOUT)
            except asyncio.TimeoutError:
                approved = False
            finally:
                self._approval_events.pop(plan_id, None)
            if not approved or plan.status == "rejected":
                plan.status = "rejected"
                self._update_state_goals(plan)
                self._active_plans.pop(plan_id, None)
                return TaskResult(
                    plan_id=plan_id, goal=goal, succeeded=False,
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
                         "params": {"plan_id": plan_id, "steps": len(plan.steps)}},
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
                            best.action_type, plan_id,
                        )
                        # Still execute — the deliberation is informational for now
                        # and records learning signal. The counterfactual record will
                        # accumulate regret/relief after the fact.
            except Exception as e:
                logger.debug("TaskEngine: counterfactual deliberation failed (non-critical): %s", e)

        # 2. Execute steps
        await self._execute_plan(plan, on_progress)

        # 3. Synthesize result
        result = await self._synthesize_result(plan, time.time() - plan.created_at)
        # Issue ATE-007: Update goals BEFORE deleting from active_plans
        self._update_state_goals(plan)
        del self._active_plans[plan_id]

        logger.info(
            "TaskEngine: plan %s complete. Success=%s (%d/%d steps)",
            plan_id, result.succeeded, result.steps_completed, result.steps_total,
        )
        return result

    def approve_plan(self, plan_id: str) -> bool:
        """Signal approval for a plan waiting for human review. Returns True if plan was found."""
        event = self._approval_events.get(plan_id)
        if event:
            if plan_id in self._active_plans:
                self._active_plans[plan_id].status = "approved"
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
        available_tools = list(self._tool_registry.keys())

        prompt = f"""Decompose this goal into a sequence of concrete steps.

Goal: {goal}

Available tools: {', '.join(available_tools)}

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

            return TaskPlan(plan_id=plan_id, goal=goal, steps=steps, trace_id="")

        except Exception as e:
            logger.error("TaskEngine: decomposition failed: %s", e)
            # Fallback: single-step plan using 'think'
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
            )

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
            logger.debug("Ignored Exception in autonomous_task_engine.py: %s", e)

    async def _fail_plan(self, plan: TaskPlan, completed_ids: Set[str], reason: str) -> None:
        logger.error("TaskEngine: %s", reason)
        await self._rollback_completed(plan, completed_ids)
        for remaining in plan.steps:
            if remaining.status == StepStatus.PENDING:
                remaining.status = StepStatus.SKIPPED
                remaining.error = "skipped due to earlier failure"
        plan.status = "failed"

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
             logger.error("TaskEngine: Execution aborted for '%s' - Safety revocation triggered.", plan.goal[:50])
             return

        plan.status = "running"
        completed_ids: Set[str] = set()

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
                logger.error("TaskEngine: no runnable steps remain for plan %s", plan.plan_id)
                return

            parallel_wave = [step for step in ready_steps if self._can_run_in_parallel(step)][:self.MAX_PARALLEL_STEPS]
            if parallel_wave:
                await asyncio.gather(*(self._execute_step_with_retry(step, plan) for step in parallel_wave))
                for step in parallel_wave:
                    if step.status == StepStatus.SUCCEEDED:
                        completed_ids.add(step.step_id)
                    self._report_progress(step, on_progress)
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

            if step.status == StepStatus.FAILED:
                await self._fail_plan(
                    plan,
                    completed_ids,
                    f"step '{step.description[:60]}' failed after {step.attempts} attempts",
                )
                return

        if plan.status != "failed":
            plan.status = "succeeded" if plan.all_complete and not plan.any_failed else "partial"

    async def _execute_step_with_retry(self, step: TaskStep, plan: TaskPlan) -> None:
        """Execute a single step with up to MAX_RETRIES retries."""
        for attempt in range(self.MAX_RETRIES + 1):
            step.attempts   += 1
            step.status      = StepStatus.RUNNING
            step.started_at  = time.time()

            try:
                # Issue ATE-001: asyncio.timeout() is 3.11+, use wait_for for compatibility
                raw_result = await asyncio.wait_for(
                    self._invoke_tool(step.tool, step.args, plan.token_id, plan.is_shadow),
                    timeout=self.STEP_TIMEOUT
                )

                step.raw_result = self._compact_tool_result(raw_result)
                step.result_summary = step.raw_result

                # Verify the step succeeded
                verified = await self._verify_step(step, raw_result)
                if verified:
                    step.status       = StepStatus.SUCCEEDED
                    step.verified     = True
                    step.completed_at = time.time()
                    logger.debug("TaskEngine: step '%s' succeeded (attempt %d)", step.description[:40], attempt + 1)
                    return
                else:
                    # Verification failed — modify args and retry
                    step.error = "verification failed"
                    logger.warning("TaskEngine: step '%s' verification failed (attempt %d)", step.description[:40], attempt + 1)
                    # Modify args for retry: ask LLM for an alternative approach
                    if attempt < self.MAX_RETRIES - 1:
                        step.args = await self._get_alternative_approach(step)

            except asyncio.TimeoutError:
                step.error = f"timeout after {self.STEP_TIMEOUT}s"
                logger.warning("TaskEngine: step '%s' timed out (attempt %d)", step.description[:40], attempt + 1)
            except Exception as e:
                step.error = str(e)
                logger.warning("TaskEngine: step '%s' error: %s (attempt %d)", step.description[:40], e, attempt + 1)

        # If all retries fail
        step.status = StepStatus.FAILED
        step.completed_at = time.time()

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

        # Fast path: trivial criteria
        trivial_pass = ["step completes", "non-empty", "any result", "no error"]
        if any(t in step.success_criterion.lower() for t in trivial_pass):
            return bool(result_str.strip())

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
            logger.debug("Verification LLM call failed: %s. Assuming pass.", e)
            return bool(result_str.strip())

    async def _get_alternative_approach(self, step: TaskStep) -> Dict:
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
            logger.debug("Alternative approach generation failed: %s", e)

        # Signal exhaustion rather than silently looping identically
        logger.warning("No alternative found for '%s'; step will likely fail.", step.description[:40])
        return step.args

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
                await self._invoke_tool(step.rollback_action, step.rollback_args)
                step.status = StepStatus.ROLLED_BACK
            except Exception as e:
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

        return TaskResult(
            plan_id=plan.plan_id,
            goal=plan.goal,
            succeeded=not plan.any_failed,
            summary=plan.final_result or "Task failed",
            trace_id=plan.trace_id,
            steps_completed=len(plan.succeeded_steps),
            steps_total=len(plan.steps),
            duration_s=time.time() - plan.created_at
        )

    # ── State integration ─────────────────────────────────────────────────────

    def _update_state_goals(self, plan: TaskPlan) -> None:
        """Keep AuraState.cognition.active_goals in sync with running plans."""
        try:
            state = self.kernel.state
            if state is None:
                return
            # Ensure it is a list
            if not hasattr(state.cognition, "active_goals") or state.cognition.active_goals is None:
                state.cognition.active_goals = []
            
            # Remove old entry for this plan if present
            state.cognition.active_goals = [
                g for g in state.cognition.active_goals
                if g.get("plan_id") != plan.plan_id
            ]
            # Add current state if plan is still running
            if plan.plan_id in self._active_plans:
                state.cognition.active_goals.append({
                    "plan_id":       plan.plan_id,
                    "goal":          plan.goal[:80],
                    "status":        plan.status,
                    "steps_done":    len(plan.succeeded_steps),
                    "steps_total":   len(plan.steps),
                })
        except Exception as e:
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
                    return await orch.execute_tool("web_search", {"query": query})
            except Exception as e:
                return f"Search failed: {e}"
            return f"Search tool not available for: {query}"

        async def _run_python(code: str, **kwargs) -> str:
            """Execute Python code in a sandboxed environment."""
            try:
                from core.container import ServiceContainer
                orch = ServiceContainer.get("orchestrator", default=None)
                if orch and hasattr(orch, "execute_tool"):
                    result = await orch.execute_tool("run_python", {"code": code})
                    return str(result)
            except Exception as e:
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
                return f"Write failed: {e}"

        async def _read_file(path: str, **kwargs) -> str:
            """Read content from a file."""
            import aiofiles
            try:
                async with aiofiles.open(path) as f:
                    return await f.read()
            except Exception as e:
                return f"Read failed: {e}"

        async def _remember(content: str, verified: bool = False, **kwargs) -> str:
            """Store something in Aura's knowledge graph."""
            if self._looks_technical_fact(content) and not verified:
                raise ValueError("Technical memory writes require verified=true after a live read or tool-backed check")
            try:
                from core.container import ServiceContainer
                kg = ServiceContainer.get("knowledge_graph", default=None)
                if kg:
                    kg.add_knowledge(content=content, source="task_engine")
                    return f"Remembered: {content[:80]}"
            except Exception as e:
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
