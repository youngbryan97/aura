"""Composition carries computed values across backend boundaries."""

from dataclasses import replace

import pytest

from core.cognition.procedure import (
    Backend, Effect, Origin, Precondition, ProcedureRegistry, Signature, compose,
)
from core.cognition.procedure_execution import BackendResult, execute_procedure


def _procedure(registry, name="produce", *, backend=Backend.TOOL, preconditions=(), effects=None):
    return registry.register(
        name, backend,
        Signature(preconditions=preconditions, effects=effects or (Effect("value", "integer"),)),
    )


def test_actual_computation_crosses_backends_and_nested_compositions():
    registry = ProcedureRegistry()
    fetch = _procedure(registry)
    transform = _procedure(
        registry, "transform", backend=Backend.MACRO,
        preconditions=(Precondition("value", "integer"),), effects=(Effect("answer", "integer"),),
    )
    chain = compose(registry, (compose(registry, (fetch,)), transform))
    context = {"request": object()}
    seen = []

    def macro(procedure, state, ctx):
        seen.append(ctx["request"])
        return BackendResult({"answer": state["value"] * 7}, {"computed": True})

    result = execute_procedure(registry, chain.procedure_id, {"untouched": "yes"}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult({"value": 6}), Backend.MACRO: macro,
    }, context=context)
    assert result.completed
    assert result.resulting_state == {"untouched": "yes", "value": 6, "answer": 42}
    assert result.execution.tool_calls == 2
    assert seen == [context["request"]]
    assert [step.procedure_id for step in result.steps] == [fetch.procedure_id, transform.procedure_id]
    assert registry.get(transform.procedure_id).value.uses == 0


@pytest.mark.parametrize("outputs, message", [
    ({}, "missing output"),
    ({"value": True}, "wrong output type"),
    ({"value": 5, "extra": 1}, "undeclared output"),
])
def test_invalid_writes_never_pass_as_declared_success(outputs, message):
    registry = ProcedureRegistry()
    procedure = _procedure(registry)
    result = execute_procedure(registry, procedure.procedure_id, {"value": 99}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult(outputs),
    })
    assert not result.completed
    assert message in result.execution.failed
    assert result.resulting_state == {"value": 99}
    assert result.steps[0].error


def test_declared_constant_is_checked_against_actual_output():
    registry = ProcedureRegistry()
    procedure = _procedure(registry, effects=(Effect("value", "integer", 12),))
    result = execute_procedure(registry, procedure.procedure_id, {}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult({"value": 13}),
    })
    assert "wrong output value" in result.execution.failed


def test_failure_retains_prior_computation_and_never_runs_following_step():
    registry = ProcedureRegistry()
    first = _procedure(registry)
    failed = _procedure(registry, "fail", backend=Backend.MACRO)
    chain = compose(registry, (first, failed, first))
    calls = []

    def fail(p, state, context):
        state["nested"].append(2)
        raise RuntimeError("backend broke")

    initial = {"nested": [1]}
    result = execute_procedure(registry, chain.procedure_id, initial, backends={
        Backend.TOOL: lambda p, s, c: calls.append(p.procedure_id) or BackendResult({"value": 8}),
        Backend.MACRO: fail,
    })
    assert not result.completed
    assert result.resulting_state == {"nested": [1], "value": 8}
    assert initial == {"nested": [1]}
    assert calls == [first.procedure_id]
    assert len(result.steps) == 2


def test_missing_backend_is_found_before_any_part_runs():
    registry = ProcedureRegistry()
    first = _procedure(registry)
    second = _procedure(registry, "missing", backend=Backend.MACRO)
    chain = compose(registry, (first, second))
    calls = []
    with pytest.raises(ValueError, match="backend unavailable"):
        execute_procedure(registry, chain.procedure_id, {}, backends={
            Backend.TOOL: lambda p, s, c: calls.append(p),
        })
    assert calls == []


def test_each_step_checks_values_not_only_the_composed_type():
    registry = ProcedureRegistry()
    first = _procedure(registry)
    second = _procedure(registry, "needs nine", preconditions=(Precondition("value", "integer", 9),))
    chain = compose(registry, (first, second))
    result = execute_procedure(registry, chain.procedure_id, {}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult({"value": 8}),
    })
    assert not result.completed
    assert result.resulting_state == {"value": 8}
    assert "preconditions" in result.steps[-1].error


def test_parts_metadata_on_a_non_composition_is_not_executed():
    registry = ProcedureRegistry()
    procedure = registry.register(
        "learned macro", Backend.MACRO, Signature(effects=(Effect("value", "integer"),)),
        parts=("historical source no longer registered",),
    )
    result = execute_procedure(registry, procedure.procedure_id, {}, backends={
        Backend.MACRO: lambda p, s, c: BackendResult({"value": 3}),
    })
    assert result.completed and result.resulting_state["value"] == 3


def test_cycle_is_found_without_running_a_backend(monkeypatch):
    registry = ProcedureRegistry()
    procedure = _procedure(registry)
    cyclic = replace(procedure, parts=(procedure.procedure_id,), program=(procedure.procedure_id,),
                     origin=Origin(learner="compose"))
    monkeypatch.setattr(registry, "procedures", lambda: [cyclic])
    with pytest.raises(ValueError, match="cyclic"):
        execute_procedure(registry, procedure.procedure_id, {}, backends={})


@pytest.mark.parametrize("retired", [False, True])
def test_missing_or_retired_part_refuses_before_execution(monkeypatch, retired):
    registry = ProcedureRegistry()
    procedure = _procedure(registry)
    monkeypatch.setattr(registry, "procedures", lambda: [replace(procedure, retired=True)] if retired else [])
    with pytest.raises(ValueError, match="unavailable"):
        execute_procedure(registry, procedure.procedure_id, {}, backends={})


def test_repeated_parts_execute_each_occurrence():
    registry = ProcedureRegistry()
    procedure = _procedure(registry, preconditions=(Precondition("value", "integer"),))
    chain = compose(registry, (procedure, procedure, procedure))
    result = execute_procedure(registry, chain.procedure_id, {"value": 0}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult({"value": s["value"] + 1}),
    })
    assert result.completed and result.resulting_state["value"] == 3
    assert result.execution.tool_calls == 3


def test_authority_denial_is_not_retried_or_replaced_with_signature_outputs():
    registry = ProcedureRegistry()
    procedure = _procedure(registry)

    def deny(p, s, c):
        assert "scoped_authority" not in c
        raise PermissionError("scoped authority required")

    result = execute_procedure(registry, procedure.procedure_id, {}, backends={Backend.TOOL: deny})
    assert not result.completed and result.resulting_state == {}
    assert result.execution.tool_calls == 1
    assert "PermissionError" in result.execution.failed


def test_false_is_a_boolean_value_between_procedures_not_a_missing_observation():
    registry = ProcedureRegistry()
    first = _procedure(registry, effects=(Effect("condition", "boolean"),))
    second = _procedure(
        registry, "use false", preconditions=(Precondition("condition", "boolean", False),),
    )
    chain = compose(registry, (first, second))
    result = execute_procedure(registry, chain.procedure_id, {}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult(
            {"condition": False} if p.procedure_id == first.procedure_id else {"value": 7},
        ),
    })
    assert result.completed and result.resulting_state["value"] == 7
    assert not Precondition("condition").satisfied_by({"condition": False})
    assert Precondition("condition", negated=True).satisfied_by({"condition": False})
    assert not Precondition("condition", "boolean").satisfied_by({})


def test_backend_cannot_mutate_the_callers_top_level_state():
    registry = ProcedureRegistry()
    procedure = _procedure(registry)

    def mutate(p, state, context):
        state["value"] = 9
        return BackendResult({"value": 9})

    result = execute_procedure(registry, procedure.procedure_id, {}, backends={Backend.TOOL: mutate})
    assert not result.completed and result.resulting_state == {}


def test_composition_must_agree_with_its_executable_parts(monkeypatch):
    registry = ProcedureRegistry()
    procedure = _procedure(registry)
    chain = compose(registry, (procedure,))
    broken = replace(chain, program=("a different program",))
    monkeypatch.setattr(registry, "procedures", lambda: [procedure, broken])
    with pytest.raises(ValueError, match="differs from its parts"):
        execute_procedure(registry, chain.procedure_id, {}, backends={})


def test_async_backend_is_not_called_from_the_synchronous_executor():
    registry = ProcedureRegistry()
    procedure = _procedure(registry)

    async def backend(p, s, c):
        return BackendResult({"value": 3})

    with pytest.raises(ValueError, match="synchronous adapter"):
        execute_procedure(registry, procedure.procedure_id, {}, backends={Backend.TOOL: backend})


def test_typed_output_invariant_is_executable():
    from core.cognition.procedure_execution import _explicit_typed_outputs_hold

    assert _explicit_typed_outputs_hold() == ()
