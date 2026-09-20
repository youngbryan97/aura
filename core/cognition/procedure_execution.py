"""Execute common procedures through the existing tool-plan executor.

Backend callbacks retain their own authority checks. Supplying a callback is
not permission to bypass its gateway. Signatures describe required outputs;
only values returned by a backend enter the execution state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from inspect import iscoroutinefunction
from types import MappingProxyType
from typing import Any
import time

from core.cognition.cognitive_event import EventGraph, get_event_graph
from core.cognition.procedure import (
    Backend,
    Procedure,
    ProcedureRegistry,
    _kind_accepts_value,
)
from core.cognition.tool_plan import Execution, Executor, Op, Plan, PlanFailed, Step
from core.cognition.procedure_trace import ObservedProcedureInputs, ProcedureTrace
from core.verify.invariants import invariant


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Explicit writes and backend evidence from one execution."""

    outputs: Mapping[str, Any]
    evidence: Any = None


BackendExecutor = Callable[[Procedure, Mapping[str, Any], Mapping[str, Any]], BackendResult]


@dataclass(frozen=True, slots=True)
class ProcedureStepResult:
    procedure_id: str
    backend: Backend
    output_keys: tuple[str, ...] = ()
    evidence: Any = None
    error: str = ""
    event_id: int = 0
    trace_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcedureExecution:
    """Execution evidence, without inferring task correctness or rollback."""

    procedure_id: str
    resulting_state: dict[str, Any]
    steps: tuple[ProcedureStepResult, ...]
    execution: Execution
    closed_types_checked: bool = False

    @property
    def completed(self) -> bool:
        return not self.execution.failed

    @property
    def trace_complete(self) -> bool:
        return bool(self.steps) and all(step.event_id and not step.trace_errors for step in self.steps)


def _events(
    registry: ProcedureRegistry,
    procedure_id: str,
    backends: Mapping[Backend, BackendExecutor],
) -> list[tuple[str, Procedure]]:
    """Resolve a finite composition before invoking any backend."""

    procedures = {item.procedure_id: item for item in registry.procedures()}
    events: list[tuple[str, Procedure]] = []
    active: set[str] = set()
    pending = [(False, procedure_id)]
    while pending:
        leaving, identity = pending.pop()
        procedure = procedures.get(identity)
        if procedure is None or procedure.retired:
            raise ValueError(f"procedure unavailable: {identity}")
        if leaving:
            active.remove(identity)
            events.append(("exit", procedure))
            continue
        if identity in active:
            raise ValueError(f"cyclic procedure composition: {identity}")
        composed = procedure.origin is not None and procedure.origin.learner == "compose"
        if composed:
            if not procedure.parts or procedure.program != procedure.parts:
                raise ValueError(f"composition program differs from its parts: {identity}")
            active.add(identity)
            events.append(("enter", procedure))
            pending.append((True, identity))
            pending.extend((False, part) for part in reversed(procedure.parts))
        else:
            handler = backends.get(procedure.backend)
            if not callable(handler):
                raise ValueError(f"procedure backend unavailable: {procedure.backend}")
            if iscoroutinefunction(handler) or iscoroutinefunction(getattr(handler, "__call__", None)):
                raise ValueError(f"procedure backend needs a synchronous adapter: {procedure.backend}")
            events.append(("leaf", procedure))
    return events


def _check_effects(procedure: Procedure, outputs: Mapping[str, Any]) -> None:
    for effect in procedure.signature.effects:
        if effect.key not in outputs:
            raise PlanFailed(f"{procedure.procedure_id}: missing output {effect.key!r}")
        actual = outputs[effect.key]
        if not _kind_accepts_value(effect.kind, actual):
            raise PlanFailed(f"{procedure.procedure_id}: wrong output type for {effect.key!r}")
        if effect.value is not None and actual != effect.value:
            raise PlanFailed(f"{procedure.procedure_id}: wrong output value for {effect.key!r}")


def execute_procedure(
    registry: ProcedureRegistry,
    procedure_id: str,
    state: Mapping[str, Any],
    *,
    backends: Mapping[Backend, BackendExecutor],
    context: Mapping[str, Any] | None = None,
    event_graph: EventGraph | None = None,
    parent_events: Sequence[int] = (),
    closed_types: bool = False,
) -> ProcedureExecution:
    """Run a registered composition with caller-supplied backend capabilities.

The call budget equals the finite program's leaf count. No elapsed timer
truncates it. Backend resource limits and cancellation remain in force.
Execution completion is not a correctness observation for ``record_use``.
External effects cannot be rolled back by discarding the local state.
"""

    if type(closed_types) is not bool:
        raise ValueError("closed type mode must be boolean")
    handlers = dict(backends)
    events = _events(registry, procedure_id, handlers)
    if closed_types:
        for _, procedure in events:
            procedure.signature.validate_structural_types()
    current = deepcopy(dict(state))
    request_context = MappingProxyType(dict(context or {}))
    records: list[ProcedureStepResult] = []
    tools: dict[str, Callable[..., Any]] = {}
    instructions: list[Step] = []
    trace = ProcedureTrace(event_graph if event_graph is not None else get_event_graph(), parent_events)

    def invoke(procedure: Procedure, state: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal current
        started = time.monotonic()
        observed_state = ObservedProcedureInputs(
            deepcopy(dict(state)), namespace="procedure:state:", owner=procedure.procedure_id,
        )
        observed_context = ObservedProcedureInputs(
            request_context, namespace="procedure:context:", owner=procedure.procedure_id,
        )
        try:
            if not procedure.signature.matches(observed_state):
                raise PlanFailed(f"{procedure.procedure_id}: preconditions do not match")
            result = handlers[procedure.backend](
                procedure, observed_state, observed_context,
            )
            if not isinstance(result, BackendResult) or not isinstance(result.outputs, Mapping):
                raise TypeError("procedure backend must return explicit BackendResult outputs")
            outputs = deepcopy(dict(result.outputs))
            declared = {effect.key for effect in procedure.signature.effects}
            if outputs.keys() - declared:
                raise PlanFailed(f"{procedure.procedure_id}: undeclared output writes")
            _check_effects(procedure, outputs)
            current = {**state, **outputs}
            trace_result = trace.record(
                procedure.procedure_id, procedure.backend.value, observed_state, observed_context,
                outputs=tuple(outputs), duration_s=time.monotonic() - started,
            )
            records.append(ProcedureStepResult(
                procedure.procedure_id, procedure.backend, tuple(outputs), result.evidence,
                event_id=trace_result.event_id, trace_errors=trace_result.errors,
            ))
            return current
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace_result = trace.record(
                procedure.procedure_id, procedure.backend.value, observed_state, observed_context,
                outputs=(), duration_s=time.monotonic() - started, error=error,
            )
            records.append(ProcedureStepResult(
                procedure.procedure_id, procedure.backend, error=error,
                event_id=trace_result.event_id, trace_errors=trace_result.errors,
            ))
            raise

    for event, procedure in events:
        if event == "leaf":
            identity = procedure.procedure_id
            tools[identity] = lambda state, p=procedure: invoke(p, state)
            instructions.append(Step(Op.CALL, into="state", tool=identity, args={"state": "$state"}))
        elif event == "enter":
            instructions.append(Step(
                Op.ASSERT, over="state", fn=procedure.signature.matches,
                message=f"{procedure.procedure_id}: composite preconditions do not match",
            ))
        else:
            def effects_hold(state: Mapping[str, Any], p: Procedure = procedure) -> bool:
                _check_effects(p, state)
                return True

            instructions.append(Step(Op.ASSERT, over="state", fn=effects_hold))
    instructions.append(Step(Op.RETURN, over="state"))
    plan = Plan(
        name=procedure_id, steps=tuple(instructions),
        max_calls=sum(event == "leaf" for event, _ in events),
    )
    execution = Executor().run(
        plan, tools=tools, permitted=frozenset(tools), bindings={"state": current},
    )
    return ProcedureExecution(procedure_id, current, tuple(records), execution, closed_types)


@invariant(
    "procedure.explicit_typed_outputs", scope="procedure",
    owner="core/cognition/procedure_execution.py", observational=False,
)
def _explicit_typed_outputs_hold() -> tuple:
    """A declared integer effect cannot replace a computed integer."""

    from core.cognition.procedure import Effect, Signature

    registry = ProcedureRegistry()
    procedure = registry.register(
        "typed output probe", Backend.TOOL, Signature(effects=(Effect("result", "integer"),)),
    )
    _check_effects(procedure, {"result": 0})
    for outputs in ({}, {"result": True}):
        try:
            _check_effects(procedure, outputs)
        except PlanFailed:
            continue
        raise AssertionError("procedure output check accepted a missing or mistyped value")
    return ()


@invariant(
    "procedure.closed_structural_declarations", scope="procedure",
    owner="core/cognition/procedure_execution.py", observational=False,
)
def _closed_structural_declarations_hold() -> tuple:
    """Nominal presence labels cannot grant closed structural evidence."""
    from core.cognition.procedure import Effect, Signature

    Signature(effects=(Effect("result", "integer", 1),)).validate_structural_types()
    for effect in (Effect("result", "unregistered"), Effect("result", "integer", True)):
        try:
            Signature(effects=(effect,)).validate_structural_types()
        except ValueError:
            continue
        raise AssertionError("closed structural evidence accepted an invalid declaration")
    return ()
