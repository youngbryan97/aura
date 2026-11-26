"""core/kernel/leviathan_kernel.py — Aura Leviathan Central Kernel.

The single sovereign authority and runtime spine of Aura.
Every consequential thought-to-action flow passes through this kernel.
It owns: identity, mission state, world model, memory, model council,
planning, tool bus, action execution, receipts, welfare/body, truth engine,
permission economy, rollback, self-improvement, and audit.

Spine: perceive → memory → model → council → will → action → verify → commit → learn.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.container import ServiceContainer
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LeviathanKernel")
_KERNEL_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@dataclass
class SpineTrace:
    """Captures a full perceive→learn trace for one kernel cycle."""

    cycle_id: str = ""
    started_at: float = 0.0
    objective: str = ""
    perception_result: dict[str, Any] | None = None
    memory_context: dict[str, Any] | None = None
    world_model_update: dict[str, Any] | None = None
    council_verdict: dict[str, Any] | None = None
    will_decision: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    commit_status: dict[str, Any] | None = None
    learning_outcome: dict[str, Any] | None = None
    completed_at: float = 0.0
    error: str | None = None


@dataclass
class PermissionBudget:
    """Structured permission economy for action classes."""

    risk_tier: str = "low"  # low, medium, high, critical
    max_calls_per_hour: int = 100
    cooldown_seconds: float = 0.0
    reversible: bool = True
    allowed_domains: list[str] = field(default_factory=list)
    required_proof: bool = False
    budget_remaining: int = 100
    last_used: float = 0.0


class LeviathanKernel:
    """The central authority and runtime spine of Aura Leviathan.

    All consequential actions must route through this kernel.
    No subsystem may act on the external world without kernel authorization.
    """

    def __init__(self) -> None:
        self.subsystems: dict[str, Any] = {}
        self.critical_subsystems: set[str] = set()
        self.startup_failures: dict[str, str] = {}
        self.active_missions: list[str] = []
        self._initialized = False
        self._cycle_count = 0
        self._spine_history: list[SpineTrace] = []
        self._identity_lock = False

        # Permission economy — structured budgets per action class
        self._permission_economy: dict[str, PermissionBudget] = {
            "file_read": PermissionBudget(risk_tier="low", max_calls_per_hour=1000, reversible=True),
            "file_write": PermissionBudget(risk_tier="medium", max_calls_per_hour=200, reversible=True, required_proof=True),
            "shell_command": PermissionBudget(risk_tier="medium", max_calls_per_hour=100, reversible=False, required_proof=True),
            "memory_write": PermissionBudget(risk_tier="low", max_calls_per_hour=500, reversible=True),
            "browser_action": PermissionBudget(risk_tier="medium", max_calls_per_hour=50, reversible=False),
            "desktop_action": PermissionBudget(risk_tier="high", max_calls_per_hour=30, reversible=False, required_proof=True),
            "email_send": PermissionBudget(risk_tier="high", max_calls_per_hour=10, reversible=False, required_proof=True),
            "cloud_compute": PermissionBudget(risk_tier="critical", max_calls_per_hour=5, reversible=False, required_proof=True),
            "self_modification": PermissionBudget(risk_tier="critical", max_calls_per_hour=3, cooldown_seconds=60.0, reversible=True, required_proof=True),
            "external_api": PermissionBudget(risk_tier="medium", max_calls_per_hour=50, reversible=False),
            "tool_execution": PermissionBudget(risk_tier="medium", max_calls_per_hour=100, reversible=True),
        }

        # Autonomy budgets
        self._budgets = {
            "compute_seconds": 3600.0,
            "memory_writes": 500,
            "tool_calls": 200,
            "web_searches": 50,
            "self_modifications": 5,
            "external_communications": 20,
        }
        self._budget_used: dict[str, float] = {k: 0.0 for k in self._budgets}

    # ── Subsystem registry ──────────────────────────────────────────────

    def register_subsystem(self, name: str, instance: Any, *, critical: bool = False) -> None:
        """Dynamically plug in a Leviathan component."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("subsystem name must be a non-empty string")
        self.subsystems[name] = instance
        if critical:
            self.critical_subsystems.add(name)
        logger.info("Leviathan Kernel registered subsystem: %s", name)

    def get_subsystem(self, name: str) -> Any | None:
        return self.subsystems.get(name)

    # ── Permission economy ──────────────────────────────────────────────

    def check_permission(self, action_class: str) -> bool:
        """Check whether an action class has remaining budget and is not on cooldown."""
        budget = self._permission_economy.get(action_class)
        if budget is None:
            logger.warning("Unknown action class '%s' — defaulting to deny", action_class)
            return False
        now = time.time()
        if budget.cooldown_seconds > 0 and (now - budget.last_used) < budget.cooldown_seconds:
            logger.warning("Action class '%s' is on cooldown (%.1fs remaining)",
                           action_class, budget.cooldown_seconds - (now - budget.last_used))
            return False
        if budget.budget_remaining <= 0:
            logger.warning("Action class '%s' has exhausted its hourly budget", action_class)
            return False
        return True

    def consume_permission(self, action_class: str) -> None:
        """Deduct one unit from the action class budget."""
        budget = self._permission_economy.get(action_class)
        if budget:
            budget.budget_remaining = max(0, budget.budget_remaining - 1)
            budget.last_used = time.time()

    def get_permission_status(self) -> dict[str, Any]:
        return {
            name: {
                "risk_tier": b.risk_tier,
                "remaining": b.budget_remaining,
                "max": b.max_calls_per_hour,
                "reversible": b.reversible,
            }
            for name, b in self._permission_economy.items()
        }

    # ── Autonomy budgets ────────────────────────────────────────────────

    def check_budget(self, category: str, amount: float = 1.0) -> bool:
        cap = self._budgets.get(category, 0.0)
        used = self._budget_used.get(category, 0.0)
        return (used + amount) <= cap

    def consume_budget(self, category: str, amount: float = 1.0) -> None:
        self._budget_used[category] = self._budget_used.get(category, 0.0) + amount

    # ── Initialization ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Resilient startup sequence for all registered sub-systems."""
        if self._initialized:
            return

        self.startup_failures.clear()
        logger.info(" Leviathan Kernel initializing registered subsystems...")
        for name, subsystem in self.subsystems.items():
            if hasattr(subsystem, "initialize") and callable(subsystem.initialize):
                try:
                    await subsystem.initialize()
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    self.startup_failures[name] = str(e)
                    record_degradation(
                        "leviathan_kernel",
                        e,
                        action="blocked critical subsystem startup or degraded optional subsystem",
                        extra={"subsystem": name, "critical": name in self.critical_subsystems},
                    )
                    logger.error("Failed to initialize subsystem %s: %s", name, e, exc_info=True)
                    if name in self.critical_subsystems:
                        self._initialized = False
                        raise RuntimeError(f"critical_subsystem_failed:{name}") from e

        self._initialized = True
        logger.info("Leviathan Kernel fully online.")

    # ── Health ──────────────────────────────────────────────────────────

    def health_status(self) -> dict[str, Any]:
        missing_critical = sorted(name for name in self.critical_subsystems if name not in self.subsystems)
        failed_critical = sorted(name for name in self.critical_subsystems if name in self.startup_failures)
        healthy = self._initialized and not missing_critical and not failed_critical
        return {
            "initialized": self._initialized,
            "healthy": healthy,
            "cycle_count": self._cycle_count,
            "critical_subsystems": sorted(self.critical_subsystems),
            "missing_critical": missing_critical,
            "startup_failures": dict(self.startup_failures),
            "active_missions": list(self.active_missions),
            "budget_status": {k: {"used": self._budget_used[k], "cap": self._budgets[k]} for k in self._budgets},
        }

    # ── The Spine: perceive → memory → model → council → will → action → verify → commit → learn

    async def execute_mission(self, objective: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        """Runs a strategic campaign mission through the unified cognition spine."""
        self._cycle_count += 1
        trace = SpineTrace(
            cycle_id=f"cycle_{self._cycle_count}_{int(time.time())}",
            started_at=time.time(),
            objective=objective,
        )
        self.active_missions.append(objective[:80])
        logger.info("Leviathan Kernel spine cycle #%d: '%s'", self._cycle_count, objective)

        try:
            # ── 1. PERCEIVE: Update world state ─────────────────────────
            perception = self.get_subsystem("perception")
            if perception and hasattr(perception, "perceive"):
                try:
                    trace.perception_result = await perception.perceive(query=objective)
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without perception")
                    trace.perception_result = {"degraded": True, "error": str(e)}

            # ── 2. MEMORY: Retrieve relevant context ────────────────────
            memory = self.get_subsystem("memory")
            if memory and hasattr(memory, "retrieve_context"):
                try:
                    trace.memory_context = await memory.retrieve_context(objective)
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without memory context")
                    trace.memory_context = {"degraded": True}

            # ── 3. MODEL: Update world model with new claims ────────────
            world_model = self.get_subsystem("world_model")
            if world_model and hasattr(world_model, "update_for_objective"):
                try:
                    trace.world_model_update = await world_model.update_for_objective(objective)
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without world model update")

            # ── 4. SIMULATE: Speculate and predict outcomes ─────────────
            simulator = self.get_subsystem("simulator")
            sim_options = {}
            if simulator and hasattr(simulator, "simulate_outcomes"):
                try:
                    sim_options = await simulator.simulate_outcomes(objective)
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without simulation")

            # ── 5. COUNCIL: Run Parliament debate for planning ──────────
            council = self.get_subsystem("council")
            debate_result = {"approved": True, "plan": [objective], "confidence": 1.0}
            if council and hasattr(council, "run_debate"):
                try:
                    debate_result = await council.run_debate(
                        objective,
                        simulation_data=sim_options,
                        memory_context=trace.memory_context,
                    )
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="aborted spine execution after council debate failed")
                    debate_result = {
                        "approved": False,
                        "reason": f"Council debate failed: {e}",
                        "confidence": 0.0,
                    }
            trace.council_verdict = debate_result


            if not debate_result.get("approved"):
                logger.warning("Mission rejected by Council: %s", debate_result.get("reason", "no consensus"))
                trace.error = "council_rejected"
                trace.completed_at = time.time()
                self._spine_history.append(trace)
                return {"ok": False, "reason": "council_rejected", "details": debate_result, "trace": trace.cycle_id}

            # ── 6. WILL & ACTION: Execute through mission engine or fallback
            mission_engine = self.get_subsystem("mission_engine")
            result: dict[str, Any] = {"ok": False}

            # Check permission budget before acting
            action_class = (constraints or {}).get("action_class", "tool_execution")
            if not self.check_permission(action_class):
                trace.error = "permission_denied"
                trace.completed_at = time.time()
                self._spine_history.append(trace)
                return {"ok": False, "reason": "permission_budget_exhausted", "action_class": action_class}

            if mission_engine and hasattr(mission_engine, "run_mission"):
                result = await mission_engine.run_mission(debate_result.get("plan", [objective]), constraints=constraints)
            else:
                result = await ActionExecutor.execute(
                    domain="tool_execution",
                    action_name="leviathan.fallback_run",
                    params={"objective": objective},
                    source="leviathan_kernel",
                )
            trace.action_result = result
            self.consume_permission(action_class)

            # ── 7. VERIFY: Check action outcome ────────────────────────
            truth = self.get_subsystem("truth_engine")
            verification = {"verified": True, "method": "default"}
            if truth and hasattr(truth, "verify_action_outcome"):
                try:
                    verification = truth.verify_action_outcome(objective, result)
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without truth verification")
            trace.verification = verification

            # ── 8. COMMIT: Persist to memory civilization ──────────────
            if memory and hasattr(memory, "record_mission_outcome"):
                try:
                    commit_status = await memory.record_mission_outcome(objective, result)
                    trace.commit_status = commit_status or {"committed": True}
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without memory commit")

            # ── 9. LEARN: Update forge and internal models ─────────────
            forge = self.get_subsystem("forge")
            if forge and hasattr(forge, "analyze_weaknesses"):
                try:
                    learning = await forge.analyze_weaknesses()
                    trace.learning_outcome = learning or {"analyzed": True}
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    record_degradation("leviathan_kernel", e, action="continued spine without forge learning")

            trace.completed_at = time.time()
            self._spine_history.append(trace)

            # Consume autonomy budget
            self.consume_budget("tool_calls")

            return {
                "ok": result.get("ok", False),
                "trace_id": trace.cycle_id,
                "duration_s": trace.completed_at - trace.started_at,
                "result": result,
                "council_approved": True,
                "verified": verification.get("verified", False),
            }

        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as e:
            trace.error = str(e)
            trace.completed_at = time.time()
            self._spine_history.append(trace)
            if objective in self.active_missions:
                self.active_missions.remove(objective[:80])
            raise

        finally:
            if objective[:80] in self.active_missions:
                self.active_missions.remove(objective[:80])

    # ── Narrative compression ───────────────────────────────────────────

    def compress_history(self, max_traces: int = 50) -> dict[str, Any]:
        """Post-mission compression: what happened, what changed, what failed,
        what was learned, what should be remembered, what should be forgotten."""
        if len(self._spine_history) <= max_traces:
            return {"compressed": False, "trace_count": len(self._spine_history)}

        old = self._spine_history[:-max_traces]
        self._spine_history = self._spine_history[-max_traces:]

        successes = sum(1 for t in old if not t.error)
        failures = sum(1 for t in old if t.error)
        errors = [t.error for t in old if t.error]

        summary = {
            "compressed": True,
            "removed_traces": len(old),
            "successes": successes,
            "failures": failures,
            "common_errors": list(set(errors))[:10],
            "retained_traces": len(self._spine_history),
        }
        logger.info("Compressed %d spine traces: %d ok, %d failed", len(old), successes, failures)
        return summary

    def get_recent_traces(self, n: int = 10) -> list[SpineTrace]:
        return self._spine_history[-n:]


# ── Singleton ───────────────────────────────────────────────────────────
_kernel_instance: LeviathanKernel | None = None


def get_leviathan_kernel() -> LeviathanKernel:
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = LeviathanKernel()
        ServiceContainer.register_instance("leviathan_kernel", _kernel_instance, required=False)
    return _kernel_instance
