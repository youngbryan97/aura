"""Reusable environment skill graph.

Skills here are options in the hierarchical-RL sense: multi-turn policies
with preconditions, constraints, termination/failure hints, and reliability.
They are generic control primitives, not domain-specific scripts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .environment_parser import EnvironmentState
from .goal_manager import EmbodiedGoal
from .reflex_layer import RiskProfile


Predicate = Callable[[EnvironmentState, RiskProfile, EmbodiedGoal], bool]


@dataclass
class SkillOption:
    name: str
    description: str
    preconditions: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    action_hints: List[str] = field(default_factory=list)
    priority: float = 0.5
    reliability: float = 0.5
    interruptible: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    predicate: Optional[Predicate] = None
    successes: int = 0
    failures: int = 0
    updated_at: float = field(default_factory=time.time)

    def matches(self, state: EnvironmentState, risk: RiskProfile, goal: EmbodiedGoal) -> bool:
        if self.predicate is not None:
            return bool(self.predicate(state, risk, goal))
        goal_text = f"{goal.name} {goal.reason}".lower()
        return any(tag.lower() in goal_text for tag in self.tags) if self.tags else True

    def score(self, state: EnvironmentState, risk: RiskProfile, goal: EmbodiedGoal) -> float:
        base = self.priority * 0.45 + self.reliability * 0.35 + goal.priority * 0.20
        if risk.critical and "emergency" in self.tags:
            base += 0.25
        if state.has_active_prompt() and "modal" in self.tags:
            base += 0.25
        if "uncertainty" in goal.name.lower() and "information" in self.tags:
            base += 0.15
        return max(0.0, min(1.0, base))

    def record_outcome(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1
        total = self.successes + self.failures
        self.reliability = self.successes / total if total else self.reliability
        self.updated_at = time.time()


class EnvironmentSkillGraph:
    """Selects the next option from the current goal/risk/state."""

    def __init__(self, macro_library: Any = None) -> None:
        self.options: Dict[str, SkillOption] = {}
        self._seed_generic_options()
        if macro_library is not None:
            self.load_from_macro_library(macro_library)

    def register(self, option: SkillOption) -> None:
        self.options[option.name] = option

    def select(
        self,
        state: EnvironmentState,
        risk: RiskProfile,
        goal: EmbodiedGoal,
    ) -> SkillOption:
        candidates = [
            option for option in self.options.values() if option.matches(state, risk, goal)
        ]
        if not candidates:
            candidates = list(self.options.values())
        return max(candidates, key=lambda option: option.score(state, risk, goal))

    def to_prompt(self, selected: Optional[SkillOption] = None) -> str:
        lines = ["[SKILL GRAPH]"]
        if selected:
            lines.append(f"SELECTED OPTION: {selected.name} - {selected.description}")
            if selected.constraints:
                lines.append("CONSTRAINTS: " + "; ".join(selected.constraints[:5]))
            if selected.action_hints:
                lines.append("ACTION HINTS: " + "; ".join(selected.action_hints[:8]))
        reliable = sorted(self.options.values(), key=lambda option: option.reliability, reverse=True)[:8]
        lines.append("AVAILABLE RELIABLE OPTIONS:")
        for option in reliable:
            lines.append(f"- {option.name} reliability={option.reliability:.2f}: {option.description}")
        return "\n".join(lines)

    def load_from_macro_library(self, macro_library: Any) -> None:
        """Expose existing persistent macro skills as selectable options.

        This bridges Aura's established ``core.agency.skill_library`` into
        embodied option selection instead of creating a second procedural
        memory store.
        """
        skills = getattr(macro_library, "skills", {}) or {}
        for name, learned in skills.items():
            reliability = float(getattr(learned, "reliability", 0.5))
            description = str(getattr(learned, "description", "")) or f"Existing learned macro {name}"
            self.register(
                SkillOption(
                    name=f"macro:{name}",
                    description=description,
                    action_hints=[f"consider existing macro skill {name}"],
                    priority=0.5,
                    reliability=reliability,
                    tags=["macro", "progress", "procedural_memory"],
                    metadata={"source": "core.agency.skill_library", "skill_name": name},
                )
            )

    def _seed_generic_options(self) -> None:
        self.register(
            SkillOption(
                name="resolve_active_prompt",
                description="Handle modal prompts or menus before continuing ordinary control.",
                preconditions=["active prompt or menu"],
                success_conditions=["normal control loop restored"],
                failure_conditions=["prompt persists or action has no effect"],
                constraints=["only use actions valid for the active prompt"],
                action_hints=["confirm safe prompt", "cancel unsafe prompt", "advance more/details prompt"],
                priority=0.9,
                reliability=0.7,
                tags=["modal", "interface", "prompt"],
                predicate=lambda state, risk, goal: state.has_active_prompt(),
            )
        )
        self.register(
            SkillOption(
                name="stabilize_critical_risk",
                description="Stop progress-seeking and reduce immediate failure risk.",
                preconditions=["critical or high danger risk"],
                success_conditions=["risk below danger", "resources stabilized", "control restored"],
                failure_conditions=["risk remains critical", "loop detected"],
                constraints=["avoid irreversible actions", "prefer escape/recovery", "preserve control"],
                action_hints=["retreat", "recover resource", "wait only if safe", "use known emergency tool"],
                priority=0.95,
                reliability=0.65,
                tags=["emergency", "survival", "risk"],
                predicate=lambda state, risk, goal: risk.danger_or_worse,
            )
        )
        self.register(
            SkillOption(
                name="gather_information_safely",
                description="Convert uncertainty into evidence using reversible probes.",
                preconditions=["uncertainty elevated"],
                success_conditions=["belief confidence improves", "new evidence recorded"],
                failure_conditions=["risk spikes", "unknown irreversible effect"],
                constraints=["prefer reversible tests", "avoid irreversible unknown use under pressure"],
                action_hints=["inspect", "query memory", "test in safe context", "defer if unsafe"],
                priority=0.7,
                reliability=0.6,
                tags=["information", "uncertainty"],
            )
        )
        self.register(
            SkillOption(
                name="break_stagnation",
                description="Interrupt repetitive patterns and try alternative strategies.",
                preconditions=["stagnation or loop detected in action outcomes"],
                success_conditions=["observation_id changes", "new state reached"],
                failure_conditions=["stalled again"],
                constraints=["avoid the recently failed actions", "prefer exploration or waiting"],
                action_hints=["change direction", "wait a turn", "search surroundings", "interact with a different object"],
                priority=0.85,
                reliability=0.5,
                tags=["stagnation", "loop", "reflex", "change"],
                predicate=lambda state, risk, goal: "stagnation" in risk.tags() or "loop" in risk.tags(),
            )
        )
        self.register(
            SkillOption(
                name="safe_progress",
                description="Advance the current task while respecting active invariants.",
                success_conditions=["new useful state reached", "goal progress increases"],
                failure_conditions=["stagnation", "risk spike", "loop detected"],
                constraints=["respect survival invariant", "interrupt on surprise"],
                action_hints=["move toward objective", "explore low-risk option", "continue current plan"],
                priority=0.45,
                reliability=0.55,
                tags=["progress", "exploration"],
            )
        )
