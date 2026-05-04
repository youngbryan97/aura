"""Environment-agnostic hierarchical goal management.

The goal manager is the strategic spine for embodied tasks. It keeps
survival/stability invariants live, interrupts stale plans when risk spikes,
and exposes a compact goal stack to slower reasoning layers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from .belief_state import EnvironmentBeliefState
from .environment_parser import EnvironmentState
from .reflex_layer import RiskProfile


class GoalStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EmbodiedGoal:
    name: str
    priority: float
    reason: str = ""
    status: GoalStatus = GoalStatus.ACTIVE
    invariant: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    constraints: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    abort_conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def refresh(self, *, priority: Optional[float] = None, reason: Optional[str] = None) -> None:
        if priority is not None:
            self.priority = max(0.0, min(1.0, priority))
        if reason is not None:
            self.reason = reason
        self.updated_at = time.time()
        self.status = GoalStatus.ACTIVE


class EnvironmentGoalManager:
    """Maintains an interruptible hierarchical goal stack."""

    def __init__(self) -> None:
        self.goals: Dict[str, EmbodiedGoal] = {}
        self.interrupt_log: List[Dict[str, Any]] = []
        self.push(
            "SURVIVE_AND_MAINTAIN_CONTROL",
            priority=1.0,
            reason="Embodied invariant: preserve agency and avoid catastrophic failure.",
            invariant=True,
            constraints=[
                "do not take irreversible actions under critical uncertainty",
                "resolve modal prompts before normal actions",
                "stabilize critical resources before progress-seeking",
            ],
        )

    def push(
        self,
        name: str,
        *,
        priority: float,
        reason: str = "",
        invariant: bool = False,
        constraints: Optional[Iterable[str]] = None,
        success_conditions: Optional[Iterable[str]] = None,
        abort_conditions: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbodiedGoal:
        key = self._key(name)
        if key in self.goals:
            goal = self.goals[key]
            goal.refresh(priority=priority, reason=reason or goal.reason)
            goal.constraints = list(constraints or goal.constraints)
            goal.success_conditions = list(success_conditions or goal.success_conditions)
            goal.abort_conditions = list(abort_conditions or goal.abort_conditions)
            goal.metadata.update(metadata or {})
            return goal
        goal = EmbodiedGoal(
            name=key,
            priority=max(0.0, min(1.0, priority)),
            reason=reason,
            invariant=invariant,
            constraints=list(constraints or []),
            success_conditions=list(success_conditions or []),
            abort_conditions=list(abort_conditions or []),
            metadata=dict(metadata or {}),
        )
        self.goals[key] = goal
        return goal

    def complete(self, name: str) -> None:
        goal = self.goals.get(self._key(name))
        if goal and not goal.invariant:
            goal.status = GoalStatus.COMPLETED
            goal.updated_at = time.time()

    def fail(self, name: str, reason: str = "") -> None:
        goal = self.goals.get(self._key(name))
        if goal and not goal.invariant:
            goal.status = GoalStatus.FAILED
            goal.reason = reason or goal.reason
            goal.updated_at = time.time()

    def update_from_state(
        self,
        state: EnvironmentState,
        risk: RiskProfile,
        belief: EnvironmentBeliefState,
    ) -> EmbodiedGoal:
        """Adapt the stack from live risk, prompts, uncertainty, and intentions."""
        # 1. Resolve Active Prompts
        if state.has_active_prompt():
            self.interrupt_with(
                "RESOLVE_ACTIVE_PROMPT",
                reason="Environment is awaiting modal input; normal control is blocked.",
                priority=0.96,
                constraints=["only emit prompt-safe actions until modal state clears"],
            )
        elif "RESOLVE_ACTIVE_PROMPT" in self.goals:
            self.complete("RESOLVE_ACTIVE_PROMPT")

        # 2. Risk Stabilization
        if risk.critical:
            self.interrupt_with(
                "STABILIZE_CRITICAL_RISK",
                reason="Reflex layer detected critical risk.",
                priority=0.99,
                constraints=["avoid exploration", "avoid irreversible actions", "prefer recovery or escape"],
            )
        elif "STABILIZE_CRITICAL_RISK" in self.goals:
            self.complete("STABILIZE_CRITICAL_RISK")

        if risk.danger_or_worse:
            self.push(
                "REDUCE_TACTICAL_RISK",
                priority=0.86,
                reason="Reflex layer detected danger.",
                constraints=["prefer reversible actions", "preserve escape routes", "reassess current plan"],
            )
        elif "REDUCE_TACTICAL_RISK" in self.goals:
            self.complete("REDUCE_TACTICAL_RISK")

        # 3. Epistemic Uncertainty
        uncertainty = belief.epistemic_uncertainty()
        if uncertainty >= 0.65:
            self.push(
                "REDUCE_UNCERTAINTY",
                priority=0.72,
                reason=f"Epistemic uncertainty is high ({uncertainty:.2f}).",
                constraints=["test safely", "gather evidence", "avoid irreversible unknown actions"],
            )
        elif "REDUCE_UNCERTAINTY" in self.goals:
            self.complete("REDUCE_UNCERTAINTY")

        # 4. Deferred Intentions
        for intention in belief.due_intentions(state)[:3]:
            self.push(
                f"REMEMBER_{intention.intention}",
                priority=max(0.55, intention.priority),
                reason=f"Deferred intention triggered by {intention.trigger}.",
                metadata={"trigger": intention.trigger, **intention.metadata},
            )

        # 5. Default Progress
        if not any(goal.status == GoalStatus.ACTIVE and not goal.invariant for goal in self.goals.values()):
            self.push(
                "MAKE_MEASURED_PROGRESS",
                priority=0.45,
                reason="No specific non-invariant goal is active; continue safe progress.",
                constraints=["respect survival invariant", "prefer information-gaining reversible actions"],
            )

        return self.current_goal()

    def interrupt_with(
        self,
        name: str,
        *,
        reason: str,
        priority: float,
        constraints: Optional[Iterable[str]] = None,
    ) -> EmbodiedGoal:
        for goal in self.goals.values():
            if not goal.invariant and goal.status == GoalStatus.ACTIVE:
                goal.status = GoalStatus.SUSPENDED
        goal = self.push(name, priority=priority, reason=reason, constraints=constraints)
        self.interrupt_log.append(
            {"at": time.time(), "goal": goal.name, "reason": reason, "priority": priority}
        )
        self.interrupt_log = self.interrupt_log[-200:]
        return goal

    def current_goal(self) -> EmbodiedGoal:
        active = [goal for goal in self.goals.values() if goal.status == GoalStatus.ACTIVE]
        if not active:
            return self.push("MAKE_MEASURED_PROGRESS", priority=0.45)
        operational = [goal for goal in active if not goal.invariant]
        if operational:
            return max(operational, key=lambda goal: (goal.priority, goal.updated_at))
        return max(active, key=lambda goal: (goal.priority, goal.updated_at))

    def active_goals(self) -> List[EmbodiedGoal]:
        return sorted(
            [goal for goal in self.goals.values() if goal.status == GoalStatus.ACTIVE],
            key=lambda goal: (goal.priority, goal.updated_at),
            reverse=True,
        )

    def to_prompt(self) -> str:
        lines = ["[GOAL STACK]"]
        for goal in self.active_goals()[:8]:
            invariant = " invariant" if goal.invariant else ""
            lines.append(f"- {goal.name} priority={goal.priority:.2f}{invariant}: {goal.reason}")
            if goal.constraints:
                lines.append("  constraints: " + "; ".join(goal.constraints[:4]))
        if self.interrupt_log:
            latest = self.interrupt_log[-1]
            lines.append(f"LATEST INTERRUPT: {latest['goal']} because {latest['reason']}")
        return "\n".join(lines)

    @staticmethod
    def _key(name: str) -> str:
        return str(name or "unnamed_goal").strip().upper().replace(" ", "_")
