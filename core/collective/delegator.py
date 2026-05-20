"""Agent Swarm / Collective Intelligence Delegator.

The delegator owns bounded fan-out for specialized thinking shards and
agentic workers.  Its contract is deliberately boring: no unbounded swarms,
no silent background failures, no callback failures poisoning successful work,
and no busy agents left behind after a timeout.
"""

# ruff: noqa: ASYNC109

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from core.base_module import AuraBaseModule
from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.task_tracker import get_task_tracker

DELEGATOR_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
)


class SwarmAgent:
    """A lightweight parallel executor with explicit lifecycle state."""

    def __init__(self, agent_id: str, specialty: str):
        self.id = agent_id
        self.specialty = specialty
        self.status = "IDLE"
        self.start_time: float | None = None
        self.result: Any = None
        self.done_event = asyncio.Event()
        self.completed_at: float | None = None
        self.task: asyncio.Task | None = None


class AgentDelegator(AuraBaseModule):
    """Bounded swarm coordinator for thinking shards and agentic workers."""

    DEFAULT_SHARD_TIMEOUT_S = 60.0
    DEFAULT_AGENTIC_TIMEOUT_S = 120.0
    MAX_AGENTIC_TIMEOUT_S = 900.0
    GPU_SEMAPHORE_TIMEOUT_S = 120.0
    COMPLETED_RETENTION_S = 60.0

    def __init__(self, orchestrator):
        super().__init__("AgentDelegator")
        self.orchestrator = orchestrator
        self.active_agents: dict[str, SwarmAgent] = {}
        self.max_parallel = 5

        self.agent_roles = {
            "critic": (
                "You are 'The Critic'. Analyze the provided proposal for flaws, "
                "edge cases, and security vulnerabilities. Be harsh but precise."
            ),
            "architect": (
                "You are 'The Architect'. Design the high-level structure to solve "
                "the problem. Focus on patterns, resilience, and scalability."
            ),
            "researcher": (
                "You are 'The Researcher'. Break down the problem and identify "
                "exactly what information is missing or needed to solve it."
            ),
            "optimizer": (
                "You are 'The Optimizer'. Look at the provided solution and find "
                "ways to make it faster, use less memory, or be more elegant."
            ),
        }
        self.running = False
        self._scavenger_task: asyncio.Task | None = None
        self._gpu_semaphore_obj: asyncio.Semaphore | None = None

    def _emit_delegator_fault(
        self,
        error: BaseException,
        *,
        action: str,
        severity: str = "degraded",
        stage: str = "",
        agent_id: str | None = None,
        receipt_required: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        metadata = dict(extra or {})
        if stage:
            metadata["stage"] = stage
        if agent_id:
            metadata["agent_id"] = agent_id
        record_degradation(
            "delegator",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=receipt_required,
            extra=metadata or None,
        )

    @staticmethod
    def _coerce_timeout(
        value: Any,
        *,
        default: float,
        minimum: float = 1.0,
        maximum: float | None = None,
    ) -> float:
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            timeout = default
        if timeout <= 0:
            timeout = default
        timeout = max(minimum, timeout)
        if maximum is not None:
            timeout = min(maximum, timeout)
        return timeout

    def _new_agent_id(self, prefix: str) -> str:
        for _ in range(8):
            agent_id = f"{prefix}{uuid.uuid4().hex[:8]}"
            if agent_id not in self.active_agents:
                return agent_id
        return f"{prefix}{uuid.uuid4().hex}"

    def _track_agent_task(self, agent: SwarmAgent, coro: Any, *, name: str) -> bool:
        try:
            try:
                task = get_task_tracker().create_task(coro, name=name)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                    raise
                task = get_task_tracker().create_task(coro)
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            if inspect.iscoroutine(coro):
                coro.close()
            self._emit_delegator_fault(
                exc,
                action="failed closed delegated agent because task tracking could not start",
                severity="critical",
                stage="task_spawn",
                agent_id=agent.id,
                receipt_required=True,
            )
            self._mark_agent_failed(
                agent,
                "[ERROR] Delegated task could not be scheduled.",
                cancel_task=False,
            )
            return False
        if isinstance(task, asyncio.Task):
            agent.task = task
        return True

    def _mark_agent_failed(
        self,
        agent: SwarmAgent,
        result: str,
        *,
        cancel_task: bool = False,
    ) -> None:
        agent.status = "FAILED"
        agent.result = result
        agent.completed_at = time.time()
        if not agent.done_event.is_set():
            agent.done_event.set()
        if cancel_task and agent.task and not agent.task.done():
            agent.task.cancel()

    def _mark_timed_out_agents(self, agent_ids: list[str], *, timeout_s: float, stage: str) -> int:
        timed_out = 0
        for agent_id in agent_ids:
            agent = self.active_agents.get(agent_id)
            if agent is None or agent.done_event.is_set():
                continue
            timed_out += 1
            self._mark_agent_failed(
                agent,
                f"[TIMEOUT] Agent did not complete within {timeout_s:.1f}s.",
                cancel_task=True,
            )
            self._emit_delegator_fault(
                TimeoutError(f"swarm agent {agent_id} timed out in {stage}"),
                action="cancelled timed-out swarm agent and released capacity",
                severity="warning",
                stage=stage,
                agent_id=agent_id,
            )
        return timed_out

    async def _wait_for_agents(self, agent_ids: list[str], *, timeout_s: float, stage: str) -> None:
        waits = [
            agent.done_event.wait()
            for agent_id in agent_ids
            if (agent := self.active_agents.get(agent_id)) is not None
        ]
        if not waits:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*waits), timeout=timeout_s)
        except TimeoutError:
            timed_out = self._mark_timed_out_agents(agent_ids, timeout_s=timeout_s, stage=stage)
            self.logger.warning("%s: %d agents timed out.", stage, timed_out)
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="marked incomplete swarm waits as failed after waiter failure",
                severity="degraded",
                stage=stage,
            )
            self._mark_timed_out_agents(agent_ids, timeout_s=timeout_s, stage=stage)

    async def _safe_publish_workspace(
        self,
        workspace: Any,
        *,
        priority: float,
        source: str,
        payload: dict[str, Any],
        reason: str,
        agent_id: str,
    ) -> None:
        if workspace is None:
            return
        try:
            await workspace.publish(
                priority=priority,
                source=source,
                payload=payload,
                reason=reason,
            )
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="continued swarm execution after workspace telemetry publish failed",
                severity="warning",
                stage="workspace_publish",
                agent_id=agent_id,
            )
            self.logger.debug("Workspace publish failed for swarm agent %s: %s", agent_id, exc)

    async def _notify_callback(
        self,
        callback: Callable | None,
        *,
        agent_id: str,
        result: Any,
    ) -> None:
        if callback is None:
            return
        try:
            callback_result = callback(agent_id=agent_id, result=result)
            if inspect.isawaitable(callback_result):
                await callback_result
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="preserved agent result after callback failed",
                severity="warning",
                stage="callback",
                agent_id=agent_id,
            )
            self.logger.warning("Swarm callback failed for %s: %s", agent_id, exc)

    def _pulse_mycelium(self, *, success: bool) -> None:
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if not mycelium:
                return
            hypha = mycelium.get_hypha("collective", "cognition")
            if hypha:
                hypha.pulse(success=success)
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="continued swarm execution after mycelial trace pulse failed",
                severity="warning",
                stage="mycelial_pulse",
            )

    async def start(self) -> None:
        """Start background tasks for the delegator."""
        if self.running:
            return
        self.running = True
        scavenger_coro = self._scavenger_loop()
        try:
            try:
                self._scavenger_task = get_task_tracker().create_task(
                    scavenger_coro,
                    name="AgentDelegator.scavenger",
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                    raise
                self._scavenger_task = get_task_tracker().create_task(scavenger_coro)
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            if inspect.iscoroutine(scavenger_coro):
                scavenger_coro.close()
            self.running = False
            self._emit_delegator_fault(
                exc,
                action="failed closed delegator startup because scavenger task could not start",
                severity="critical",
                stage="start",
                receipt_required=True,
            )
            raise RuntimeError("AgentDelegator failed to start scavenger task") from exc
        self.logger.info("AgentDelegator systems active (scavenger enabled, GPU semaphore=1)")

    async def stop(self) -> None:
        """Stop background tasks and leave no running scavenger behind."""
        self.running = False
        task = self._scavenger_task
        if task and hasattr(task, "cancel"):
            task.cancel()
        if isinstance(task, asyncio.Task):
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            except TimeoutError as exc:
                self._emit_delegator_fault(
                    exc,
                    action="delegator stop continued after scavenger did not cancel in time",
                    severity="warning",
                    stage="stop",
                )
            except DELEGATOR_RECOVERABLE_ERRORS as exc:
                self._emit_delegator_fault(
                    exc,
                    action="delegator stop continued after scavenger cancellation audit failed",
                    severity="warning",
                    stage="stop",
                )
        self.logger.info("AgentDelegator systems stopped")

    def is_alive(self) -> bool:
        """Return True if the delegator is active and its scavenger is healthy."""
        if not self.running:
            return False
        if self._scavenger_task and self._scavenger_task.done() and not self._scavenger_task.cancelled():
            try:
                if self._scavenger_task.exception():
                    return False
            except DELEGATOR_RECOVERABLE_ERRORS as exc:
                self.logger.debug("Failed to retrieve scavenger task exception: %s", exc)
                return False
        return True

    async def _scavenger_loop(self) -> None:
        """Periodically prune completed agents to prevent memory bloat."""
        while self.running:
            try:
                await asyncio.sleep(30)
                now = time.time()
                to_prune = [
                    agent_id
                    for agent_id, agent in list(self.active_agents.items())
                    if agent.status in {"COMPLETED", "FAILED"}
                    and agent.completed_at
                    and now - agent.completed_at > self.COMPLETED_RETENTION_S
                ]
                for agent_id in to_prune:
                    del self.active_agents[agent_id]
                    self.logger.debug("Scavenged swarm agent: %s", agent_id)
            except asyncio.CancelledError:
                break
            except DELEGATOR_RECOVERABLE_ERRORS as exc:
                await self._repair_scavenger_failure(exc)

    async def _repair_scavenger_failure(self, error: BaseException) -> None:
        try:
            import psutil

            if psutil.virtual_memory().percent < 90:
                from core.runtime.self_healing import get_healer

                self._emit_delegator_fault(
                    error,
                    action="scheduled deep repair for delegator scavenger loop failure",
                    severity="degraded",
                    stage="scavenger_loop",
                    receipt_required=True,
                    extra={"repair_requested": True},
                )
                get_healer().schedule_deep_repair(
                    "core/collective/delegator.py",
                    reason=f"scavenger_loop_exception: {error}",
                    metadata={"error_type": type(error).__name__},
                )
            else:
                self._emit_delegator_fault(
                    error,
                    action="suppressed deep repair under memory pressure and kept scavenger alive",
                    severity="warning",
                    stage="scavenger_loop",
                    receipt_required=True,
                )
        except DELEGATOR_RECOVERABLE_ERRORS as repair_exc:
            self._emit_delegator_fault(
                error,
                action="kept scavenger loop alive after repair scheduler was unavailable",
                severity="degraded",
                stage="scavenger_loop",
                extra={"repair_error": f"{type(repair_exc).__name__}: {repair_exc}"},
            )
        self.logger.error("Scavenger loop error: %s", error)

    def get_status(self) -> dict[str, Any]:
        return {
            "active_count": self._busy_count(),
            "agents": {
                agent_id: {"specialty": agent.specialty, "status": agent.status}
                for agent_id, agent in list(self.active_agents.items())
            },
            "capacity": f"{self._busy_count()}/{self.effective_max_parallel()}",
            "configured_max_parallel": self.max_parallel,
        }

    def _busy_count(self) -> int:
        return sum(agent.status == "BUSY" for agent in list(self.active_agents.values()))

    def effective_max_parallel(self) -> int:
        """Throttle swarm width from live integrity telemetry."""
        limit = max(1, int(self.max_parallel))
        try:
            monitor = ServiceContainer.get("integrity_monitor", default=None)
            report = getattr(monitor, "_last_report", None)
            if report is None and hasattr(monitor, "get_stats"):
                stats = monitor.get_stats()
                cpu = float(stats.get("cpu_percent", 0.0) or 0.0)
                memory = float(stats.get("memory_percent", 0.0) or 0.0)
                thermal = 0
            else:
                cpu = float(getattr(report, "cpu_percent", 0.0) or 0.0)
                memory = float(getattr(report, "memory_percent", 0.0) or 0.0)
                thermal = int(getattr(report, "thermal_level", 0) or 0)

            if thermal >= 2 or cpu >= 90.0 or memory >= 90.0:
                return min(limit, 1)
            if thermal == 1 or cpu >= 75.0 or memory >= 80.0:
                return min(limit, 2)
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="used conservative swarm parallelism after resource throttle probe failed",
                severity="warning",
                stage="resource_throttle",
            )
            self.logger.debug("Swarm resource throttle probe failed: %s", exc)
            return min(limit, 2)
        return limit

    async def delegate(
        self,
        specialty: str,
        task_prompt: str,
        callback: Callable | None = None,
        parent_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a thinking shard and return the agent id."""
        prompt = str(task_prompt or "").strip()
        if not prompt:
            self.logger.warning("Swarm delegation blocked: empty task prompt.")
            return ""

        if self._busy_count() >= self.effective_max_parallel():
            self.logger.warning("Swarm capacity reached. Blocking delegation.")
            return ""

        hierarchy = f"{parent_id}/" if parent_id else ""
        agent_id = self._new_agent_id("ag-")
        agent = SwarmAgent(agent_id, str(specialty or "generalist").strip() or "generalist")
        agent.status = "BUSY"
        agent.start_time = time.time()
        self.active_agents[agent_id] = agent

        self.logger.info("Spawning Swarm Agent: %s%s (%s)", hierarchy, agent_id, agent.specialty)

        if not self._track_agent_task(
            agent,
            self._run_agent(agent, prompt, callback, **kwargs),
            name=f"AgentDelegator.{agent_id}",
        ):
            return ""
        return agent_id

    async def get_swarm_results(self, agent_ids: list[str]) -> dict[str, Any]:
        """Gather completed results from a specific set of agents."""
        results: dict[str, Any] = {}
        for agent_id in agent_ids:
            agent = self.active_agents.get(agent_id)
            if agent and agent.status == "COMPLETED":
                results[agent.specialty] = agent.result
            elif agent and agent.status == "FAILED":
                results[agent.specialty] = f"ERROR: {agent.result}"
        return results

    async def delegate_debate(  # noqa: ASYNC109 - public API accepts timeout kwarg.
        self,
        topic: str,
        roles: list[str] | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> str:
        """Spawn multiple agents, wait for them, and synthesize consensus."""
        topic_text = str(topic or "").strip()
        if not topic_text:
            return "Swarm debate cancelled: empty topic."

        selected_roles = roles or ["architect", "critic"]
        timeout_s = self._coerce_timeout(
            timeout,
            default=self.DEFAULT_SHARD_TIMEOUT_S,
            minimum=0.05,
            maximum=300.0,
        )
        agent_timeout = min(timeout_s, self.DEFAULT_SHARD_TIMEOUT_S)
        self.logger.info("Forming swarm debate on: %s...", topic_text[:50])

        agent_ids: list[str] = []
        for role in selected_roles:
            agent_id = await self.delegate(
                role,
                (
                    "Analyze this topic from your perspective and return compact JSON with keys "
                    f"claim, evidence_refs, confidence, flaws: {topic_text}"
                ),
                agent_timeout=agent_timeout,
                **kwargs,
            )
            if agent_id:
                agent_ids.append(agent_id)

        if not agent_ids:
            return "Swarm capacity reached, debate cancelled."

        await self._wait_for_agents(agent_ids, timeout_s=timeout_s, stage="delegate_debate")

        results = []
        for agent_id in agent_ids:
            agent = self.active_agents.get(agent_id)
            if agent and agent.result and agent.status == "COMPLETED":
                results.append(f"[{agent.specialty.upper()}]:\n{agent.result}")

        if not results:
            return "Swarm failed to produce a consensus (timeout or execution failure)."

        return await self.synthesize_consensus(topic_text, results, **kwargs)

    async def synthesize_consensus(
        self,
        original_topic: str,
        agent_outputs: list[str],
        **kwargs: Any,
    ) -> str:
        """Synthesize swarm outputs into a single conclusion."""
        deterministic = self._deterministic_consensus(original_topic, agent_outputs)
        engine = getattr(self.orchestrator, "cognitive_engine", None)
        if not engine:
            return deterministic

        combined_outputs = "\n\n---\n\n".join(agent_outputs)
        prompt = f"""You are the Master Synthesizer. Review the original problem and the analyses from your specialized swarm agents.
Formulate a final, conclusive recommendation or plan that balances their insights.

ORIGINAL PROBLEM:
{original_topic}

SWARM ANALYSES:
{combined_outputs}

FINAL SYNTHESIS:"""

        try:
            from core.brain.cognitive_engine import ThinkingMode

            result = await asyncio.wait_for(
                engine.think(prompt, mode=ThinkingMode.DEEP, block_user=True, **kwargs),
                timeout=60.0,
            )
            content = result.content if hasattr(result, "content") else str(result)
            return str(content or "").strip() or deterministic
        except TimeoutError:
            self.logger.error("Synthesis failed: cognitive engine timed out (>60s).")
            return deterministic
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="returned deterministic consensus after LLM synthesis failed",
                severity="warning",
                stage="synthesize_consensus",
            )
            self.logger.error("Failed to synthesize consensus: %s", exc)
            return deterministic

    def _deterministic_consensus(self, original_topic: str, agent_outputs: list[str]) -> str:
        claims: list[str] = []
        flaws: list[str] = []
        confidences: list[float] = []
        for output in agent_outputs:
            parsed = self._parse_agent_output(output)
            claim = str(parsed.get("claim") or "").strip()
            if claim:
                claims.append(claim)
            flaws.extend(str(item).strip() for item in parsed.get("flaws", []) if str(item).strip())
            try:
                confidences.append(float(parsed.get("confidence")))
            except (TypeError, ValueError) as exc:
                self.logger.debug("Suppressed invalid swarm confidence value: %s", exc)

        if not claims:
            claims = [str(output).strip()[:240] for output in agent_outputs if str(output).strip()]
        confidence = sum(confidences) / len(confidences) if confidences else 0.5
        unique_flaws = list(dict.fromkeys(flaws))[:6]
        parts = [
            "Deterministic swarm consensus:",
            f"Topic: {original_topic[:240]}",
            "Claims:",
            *[f"- {claim[:300]}" for claim in claims[:6]],
            f"Confidence: {confidence:.2f}",
        ]
        if unique_flaws:
            parts.extend(["Flaws / cautions:", *[f"- {flaw[:240]}" for flaw in unique_flaws]])
        return "\n".join(parts)

    def _parse_agent_output(self, output: str) -> dict[str, Any]:
        text = str(output or "").strip()
        if not text:
            return {}
        if text.startswith("[") and "]:" in text:
            text = text.split("]:", 1)[1].strip()

        candidates = [text]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        )
        for candidate in candidates:
            try:
                parsed = json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                parsed.setdefault("flaws", [])
                return parsed
        return {"claim": text[:500], "evidence_refs": [], "confidence": 0.5, "flaws": []}

    async def delegate_agentic(  # noqa: ASYNC109 - public API accepts timeout kwarg.
        self,
        goal: str,
        timeout: float = DEFAULT_AGENTIC_TIMEOUT_S,
        callback: Callable | None = None,
    ) -> str:
        """Spawn an agent that can use tools through AutonomousTaskEngine."""
        goal_text = str(goal or "").strip()
        if not goal_text:
            self.logger.warning("Agentic delegation blocked: empty goal.")
            return ""
        if self._busy_count() >= self.effective_max_parallel():
            self.logger.warning("Swarm capacity reached. Blocking agentic delegation.")
            return ""

        timeout_s = self._coerce_timeout(
            timeout,
            default=self.DEFAULT_AGENTIC_TIMEOUT_S,
            minimum=0.05,
            maximum=self.MAX_AGENTIC_TIMEOUT_S,
        )
        agent_id = self._new_agent_id("ag-task-")
        agent = SwarmAgent(agent_id, "agentic_executor")
        agent.status = "BUSY"
        agent.start_time = time.time()
        self.active_agents[agent_id] = agent

        self.logger.info("Spawning agentic agent %s for goal: %s", agent_id, goal_text[:60])
        if not self._track_agent_task(
            agent,
            self._run_agentic_agent(agent, goal_text, timeout_s, callback),
            name=f"AgentDelegator.{agent_id}",
        ):
            return ""
        return agent_id

    async def delegate_parallel_goals(  # noqa: ASYNC109 - public API accepts timeout kwarg.
        self,
        goals: list[dict[str, str]],
        timeout: float = DEFAULT_AGENTIC_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Spawn multiple agentic agents and return combined results."""
        if not isinstance(goals, list):
            return {"ok": False, "error": "Goals must be a list."}

        timeout_s = self._coerce_timeout(
            timeout,
            default=self.DEFAULT_AGENTIC_TIMEOUT_S,
            minimum=0.05,
            maximum=self.MAX_AGENTIC_TIMEOUT_S,
        )
        agent_ids: list[str] = []
        skipped = 0
        for item in goals:
            if not isinstance(item, dict):
                skipped += 1
                continue
            goal_text = str(item.get("goal", "") or "").strip()
            if not goal_text:
                skipped += 1
                continue
            agent_id = await self.delegate_agentic(goal_text, timeout=timeout_s)
            if agent_id:
                agent_ids.append(agent_id)
            else:
                skipped += 1

        if not agent_ids:
            return {"ok": False, "error": "No agents could be spawned.", "skipped": skipped}

        await self._wait_for_agents(agent_ids, timeout_s=timeout_s, stage="delegate_parallel_goals")

        results = {}
        for agent_id in agent_ids:
            agent = self.active_agents.get(agent_id)
            if agent:
                results[agent_id] = {
                    "status": agent.status,
                    "result": agent.result,
                    "specialty": agent.specialty,
                }
        return {
            "ok": all(agent.get("status") == "COMPLETED" for agent in results.values()),
            "agents": results,
            "spawned": len(agent_ids),
            "skipped": skipped,
        }

    def _register_orchestrator_tools(self, engine: Any, orchestrator: Any) -> int:
        """Register capability-engine skills on an AutonomousTaskEngine."""
        if not orchestrator or not hasattr(orchestrator, "execute_tool"):
            return 0
        cap_engine = getattr(orchestrator, "capability_engine", None)
        skills = getattr(cap_engine, "skills", None)
        if not skills:
            self._emit_delegator_fault(
                RuntimeError("capability engine skills unavailable"),
                action="continued agentic execution with AutonomousTaskEngine default tools only",
                severity="warning",
                stage="agentic_tool_registration",
            )
            return 0

        registered = 0
        for tool_name in list(skills):
            name = str(tool_name)

            async def _tool_adapter(_name: str = name, **kwargs: Any) -> Any:
                origin = kwargs.pop("origin", None)
                tool_kwargs = {"origin": origin} if origin else {}
                return await orchestrator.execute_tool(_name, kwargs, **tool_kwargs)

            engine.register_tool(name, _tool_adapter)
            registered += 1
        return registered

    async def _run_agentic_agent(
        self,
        agent: SwarmAgent,
        goal: str,
        timeout_s: float,
        callback: Callable | None,
    ) -> None:
        """Execute a goal using AutonomousTaskEngine with full tool access."""
        try:
            from core.agency.autonomous_task_engine import AutonomousTaskEngine

            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel is None:
                raise RuntimeError("AuraKernel not available for agentic agent")

            engine = AutonomousTaskEngine(kernel)
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            registered_tools = self._register_orchestrator_tools(engine, orchestrator)

            self.logger.info(
                "Agentic Agent %s executing goal with %d registered tools",
                agent.id,
                registered_tools,
            )
            result = await asyncio.wait_for(
                engine.execute_goal(
                    goal=goal,
                    context={"origin": f"swarm_agent:{agent.id}", "drive": "delegation"},
                ),
                timeout=timeout_s,
            )

            if result and hasattr(result, "summary"):
                agent.result = result.summary
                agent.status = "COMPLETED" if result.succeeded else "FAILED"
            else:
                agent.result = str(result) if result else "No result returned"
                agent.status = "COMPLETED" if result else "FAILED"

            self.logger.info("Agentic Agent %s completed: %s", agent.id, agent.status)
            await self._notify_callback(callback, agent_id=agent.id, result=agent.result)
        except TimeoutError:
            self.logger.error("Agentic Agent %s timed out (>%.0fs)", agent.id, timeout_s)
            self._mark_agent_failed(
                agent,
                f"[TIMEOUT] Agent could not complete goal within {timeout_s:.1f}s",
                cancel_task=False,
            )
        except asyncio.CancelledError:
            if agent.status == "BUSY":
                self._mark_agent_failed(agent, "[CANCELLED] Agent task was cancelled.", cancel_task=False)
            raise
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="failed agentic worker with explicit error result",
                severity="degraded",
                stage="run_agentic_agent",
                agent_id=agent.id,
                receipt_required=True,
            )
            from core.utils.exceptions import capture_and_log

            capture_and_log(exc, {"module": "AgentDelegator", "method": "_run_agentic_agent"})
            self.logger.error("Agentic Agent %s failed: %s", agent.id, exc, exc_info=True)
            self._mark_agent_failed(agent, f"[ERROR] {exc}", cancel_task=False)
        finally:
            agent.completed_at = time.time()
            if not agent.done_event.is_set():
                agent.done_event.set()

    async def _run_agent(
        self,
        agent: SwarmAgent,
        prompt: str,
        callback: Callable | None,
        **kwargs: Any,
    ) -> None:
        workspace = ServiceContainer.get("global_workspace", default=None)

        try:
            await self._safe_publish_workspace(
                workspace,
                priority=0.5,
                source=f"Swarm::{agent.id}",
                payload={"status": "started", "role": agent.specialty},
                reason="Swarm sub-agent initialized",
                agent_id=agent.id,
            )

            local_brain = ServiceContainer.get("cognitive_engine", default=None)
            if not local_brain and self.orchestrator:
                local_brain = getattr(self.orchestrator, "cognitive_engine", None)
            if not local_brain:
                raise RuntimeError("No cognitive engine available for swarm delegation.")

            role_prompt = self.agent_roles.get(
                agent.specialty.lower(),
                f"You are an expert in {agent.specialty}.",
            )
            swarm_context = (
                f"[SWARM PROTOCOL: {role_prompt} "
                "Focus exclusively on your specialized perspective.]\n"
            )

            self.logger.debug("Swarm Agent %s waiting for GPU semaphore.", agent.id)
            try:
                await asyncio.wait_for(
                    self.gpu_semaphore.acquire(),
                    timeout=self.GPU_SEMAPHORE_TIMEOUT_S,
                )
            except TimeoutError as exc:
                self.logger.error(
                    "Could not acquire GPU semaphore within %.0fs for Swarm Agent %s",
                    self.GPU_SEMAPHORE_TIMEOUT_S,
                    agent.id,
                )
                raise RuntimeError(f"GPU semaphore deadlock for {agent.id}") from exc

            try:
                self.logger.debug("Swarm Agent %s acquired GPU. Beginning inference.", agent.id)
                agent_timeout = self._coerce_timeout(
                    kwargs.pop("agent_timeout", self.DEFAULT_SHARD_TIMEOUT_S),
                    default=self.DEFAULT_SHARD_TIMEOUT_S,
                    maximum=300.0,
                )
                result = await asyncio.wait_for(
                    local_brain.think(swarm_context + prompt, mode="fast", **kwargs),
                    timeout=agent_timeout,
                )
            finally:
                self.gpu_semaphore.release()

            content = result.content if hasattr(result, "content") else str(result)
            agent.result = str(content or "").strip()
            if not agent.result:
                raise RuntimeError("Swarm cognitive engine returned empty output")
            agent.status = "COMPLETED"
            self._pulse_mycelium(success=True)
            await self._notify_callback(callback, agent_id=agent.id, result=agent.result)

            await self._safe_publish_workspace(
                workspace,
                priority=0.8,
                source=f"Swarm::{agent.id}",
                payload={"status": "completed", "result_length": len(agent.result)},
                reason="Swarm internal monologue step completed",
                agent_id=agent.id,
            )
            self.logger.info("Swarm Agent %s completed task.", agent.id)
        except TimeoutError:
            self.logger.error("Swarm Agent %s failed: inference timeout.", agent.id)
            self._mark_agent_failed(
                agent,
                "[CRITICAL] Cognitive timeout. Agent failed to reach consensus.",
                cancel_task=False,
            )
        except asyncio.CancelledError:
            if agent.status == "BUSY":
                self._mark_agent_failed(agent, "[CANCELLED] Agent task was cancelled.", cancel_task=False)
            raise
        except DELEGATOR_RECOVERABLE_ERRORS as exc:
            self._emit_delegator_fault(
                exc,
                action="failed swarm shard with explicit error result",
                severity="degraded",
                stage="run_agent",
                agent_id=agent.id,
                receipt_required=True,
            )
            from core.utils.exceptions import capture_and_log

            capture_and_log(exc, {"module": "AgentDelegator", "method": "_run_agent"})
            self.logger.error("Swarm Agent %s failed: %s", agent.id, exc, exc_info=True)
            self._mark_agent_failed(agent, f"[ERROR] {exc}", cancel_task=False)
            self._pulse_mycelium(success=False)
        finally:
            agent.completed_at = time.time()
            if not agent.done_event.is_set():
                agent.done_event.set()

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        if self._gpu_semaphore_obj is None:
            self._gpu_semaphore_obj = asyncio.Semaphore(1)
        return self._gpu_semaphore_obj

    async def join_all(self, timeout: float = 30.0) -> bool:  # noqa: ASYNC109 - public API accepts timeout kwarg.
        """Wait for busy agents to complete; retained completed agents do not block."""
        busy_agents = [
            agent
            for agent in list(self.active_agents.values())
            if agent.status == "BUSY" and not agent.done_event.is_set()
        ]
        if not busy_agents:
            return True
        timeout_s = self._coerce_timeout(timeout, default=30.0, minimum=0.1)
        try:
            await asyncio.wait_for(
                asyncio.gather(*(agent.done_event.wait() for agent in busy_agents)),
                timeout=timeout_s,
            )
        except TimeoutError:
            return False
        return all(agent.status != "BUSY" for agent in list(self.active_agents.values()))
