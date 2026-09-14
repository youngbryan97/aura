"""Plan from typed task requirements over the existing procedure registry.

Requirements carry grounded state keys, not prose similarities. Callers that
need a particular computation also restrict eligible procedure identities.
Planning never invokes a backend or treats a signature as an observed value.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from core.cognition.procedure import (
    Backend, Effect, Precondition, Procedure, ProcedureRegistry, _kind_accepts_value,
    _kinds_compose, compose,
)
from core.cognition.procedure_execution import (
    BackendExecutor, ProcedureExecution, execute_procedure,
)
from core.verify.invariants import invariant


@dataclass(frozen=True, slots=True)
class ProcedurePlan:
    """A dependency proposal; success requires subsequent observed execution."""

    requirements: tuple[Precondition, ...]
    procedure_ids: tuple[str, ...]
    found: bool
    expanded: int
    reason: str


@dataclass(frozen=True, slots=True)
class ProcedureGoalExecution:
    plan: ProcedurePlan
    execution: ProcedureExecution | None
    requirements_met: bool

    @property
    def completed(self) -> bool:
        return self.plan.found and self.requirements_met and (
            self.execution is None or self.execution.completed
        )


def _requirement_key(requirement: Precondition) -> str:
    # Typed JSON constants cover the shared signature's persistent value space.
    return json.dumps(
        [requirement.key, requirement.kind, requirement.negated, _constant_identity(requirement.equals)],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _constant_identity(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return (type(value).__name__, value)
    if type(value) in (list, tuple):
        return (type(value).__name__, [_constant_identity(item) for item in value])
    if type(value) is dict and all(type(key) is str for key in value):
        return ("dict", [(key, _constant_identity(value[key])) for key in sorted(value)])
    raise ValueError("procedure requirements need typed serializable constants")


def _canonical(requirements: Sequence[Precondition]) -> tuple[tuple[str, Precondition], ...]:
    return tuple(sorted({_requirement_key(item): item for item in requirements}.items()))


def _effect_meets(effect: Effect, requirement: Precondition) -> bool:
    if effect.key != requirement.key:
        return False
    if effect.value is not None:
        return _kind_accepts_value(effect.kind, effect.value) and requirement.satisfied_by(
            {effect.key: effect.value},
        )
    # A computed value supplies a type, never a guessed constant or absence.
    return (
        not requirement.negated and requirement.equals is None
        and _kinds_compose(effect.kind, requirement.kind)
    )


def _regress(
    requirements: Sequence[Precondition], procedure: Procedure,
) -> tuple[Precondition, ...] | None:
    effects = {effect.key: effect for effect in procedure.signature.effects}
    remaining = []
    contributed = False
    for requirement in requirements:
        effect = effects.get(requirement.key)
        if effect is None:
            remaining.append(requirement)
        elif not _effect_meets(effect, requirement):
            return None
        else:
            contributed = True
    if not contributed:
        return None
    return (*remaining, *procedure.signature.preconditions)


def plan_procedure(
    registry: ProcedureRegistry,
    state: Mapping[str, Any],
    requirements: Sequence[Precondition],
    *,
    eligible: Collection[str] | None = None,
    max_steps: int,
    max_expansions: int,
) -> ProcedurePlan:
    """Backward dependency search, with explicit caller-owned work bounds.

    Shorter plans are considered first, then the registry's measured values.
    Only declared effects discharge a requirement. A result whose value is
    computed cannot satisfy a required constant until execution observes it.
    No match is a statement about this finite search, not general inability.
    """

    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if type(max_expansions) is not int or max_expansions < 1:
        raise ValueError("max_expansions must be a positive integer")
    requested = tuple(requirements)
    if not requested or any(not isinstance(item, Precondition) or not item.key for item in requested):
        raise ValueError("a task needs nonempty typed requirements")
    initial = _canonical(requested)
    if all(item.satisfied_by(state) for item in requested):
        return ProcedurePlan(requested, (), True, 0, "already_satisfied")
    allowed = frozenset(eligible) if eligible is not None else None
    available = (
        registry.procedures(refresh=True) if allowed is None
        else [p for identity in allowed if (p := registry.get(identity)) is not None]
    )
    candidates = sorted(
        (p for p in available if not p.retired),
        key=lambda p: (-p.value.net, p.procedure_id),
    )
    by_effect: dict[str, set[str]] = {}
    for procedure in candidates:
        for effect in procedure.signature.effects:
            by_effect.setdefault(effect.key, set()).add(procedure.procedure_id)
    frontier = deque([(initial, ())])
    seen = {tuple(key for key, _ in initial)}
    expanded = 0
    depth_limited = False
    while frontier:
        needed, suffix = frontier.popleft()
        if len(suffix) >= max_steps:
            depth_limited = True
            continue
        relevant = set().union(*(by_effect.get(item.key, set()) for _, item in needed))
        for procedure in candidates:
            if procedure.procedure_id not in relevant:
                continue
            if expanded >= max_expansions:
                return ProcedurePlan(requested, (), False, expanded, "expansion_limit")
            expanded += 1
            regressed = _regress(tuple(item for _, item in needed), procedure)
            if regressed is None:
                continue
            canonical = _canonical(regressed)
            path = (procedure.procedure_id, *suffix)
            if all(item.satisfied_by(state) for _, item in canonical):
                return ProcedurePlan(requested, path, True, expanded, "dependencies_satisfied")
            key = tuple(key for key, _ in canonical)
            if key not in seen:
                seen.add(key)
                frontier.append((canonical, path))
    return ProcedurePlan(
        requested, (), False, expanded, "depth_limit" if depth_limited else "no_dependency_plan",
    )


def execute_procedure_plan(
    registry: ProcedureRegistry,
    plan: ProcedurePlan,
    state: Mapping[str, Any],
    *,
    backends: Mapping[Backend, BackendExecutor],
    context: Mapping[str, Any] | None = None,
) -> ProcedureGoalExecution:
    """Run through the shared executor and check the task's actual final state.

    A met structural requirement is not independent semantic correctness and
    does not update the registry's learned success rate.
    """

    if not plan.found:
        return ProcedureGoalExecution(plan, None, False)
    if not plan.procedure_ids:
        return ProcedureGoalExecution(
            plan, None, all(item.satisfied_by(state) for item in plan.requirements),
        )
    parts = [registry.get(identity) for identity in plan.procedure_ids]
    if any(part is None or part.retired for part in parts):
        raise ValueError("planned procedure is no longer available")
    procedure = parts[0] if len(parts) == 1 else compose(registry, parts, intern=True)
    execution = execute_procedure(
        registry, procedure.procedure_id, state, backends=backends, context=context,
    )
    return ProcedureGoalExecution(
        plan, execution,
        all(item.satisfied_by(execution.resulting_state) for item in plan.requirements),
    )


@invariant(
    "procedure.task_goal_constrains_reuse", scope="procedure",
    owner="core/cognition/procedure_planning.py", observational=False,
)
def _goal_bound_reuse_holds() -> tuple:
    """A matching input cannot select an effect opposed to the task goal."""

    from core.cognition.procedure import Signature

    registry = ProcedureRegistry()
    registry.register("opposite", Backend.TOOL,
        Signature(effects=(Effect("enabled", "boolean", False),)))
    intended = registry.register("requested", Backend.TOOL,
        Signature(effects=(Effect("enabled", "boolean", True),)))
    plan = plan_procedure(registry, {}, (Precondition("enabled", "boolean", True),),
        max_steps=1, max_expansions=2)
    assert plan.found and plan.procedure_ids == (intended.procedure_id,)
    return ()
