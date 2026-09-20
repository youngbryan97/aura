"""Structural execution evidence cannot inherit nominal presence semantics."""

import pytest

from core.cognition.procedure import Backend, Effect, Precondition, ProcedureRegistry, Signature, compose
from core.cognition.procedure_execution import BackendResult, execute_procedure


def test_unknown_nested_type_is_found_before_any_effect():
    registry = ProcedureRegistry()
    first = registry.register("first", Backend.TOOL, Signature(effects=(Effect("x", "integer"),)))
    second = registry.register("second", Backend.MACRO,
        Signature((Precondition("x", "integer"),), (Effect("result", "unregistered-proof-type"),)))
    combined = compose(registry, (first, second))
    calls = []
    with pytest.raises(ValueError, match="structural type"):
        execute_procedure(registry, combined.procedure_id, {}, closed_types=True, backends={
            Backend.TOOL: lambda p, s, c: calls.append(p) or BackendResult({"x": 1}),
            Backend.MACRO: lambda p, s, c: calls.append(p) or BackendResult({"result": "not proven"}),
        })
    assert calls == []


@pytest.mark.parametrize("kind,value", [("integer", True), ("boolean", 1), ("integer_sequence", [1, False])])
def test_declared_constant_must_inhabit_its_structural_type(kind, value):
    registry = ProcedureRegistry()
    procedure = registry.register("bad declaration", Backend.TOOL,
        Signature(effects=(Effect("answer", kind, value),)))
    calls = []
    with pytest.raises(ValueError, match="structural value"):
        execute_procedure(registry, procedure.procedure_id, {}, closed_types=True,
            backends={Backend.TOOL: lambda p, s, c: calls.append(p) or BackendResult({"answer": value})})
    assert calls == []


def test_nominal_legacy_labels_keep_their_original_execution_semantics():
    registry = ProcedureRegistry()
    procedure = registry.register("legacy", Backend.TOOL,
        Signature((Precondition("ready", "legacy-state"),), (Effect("answer", "legacy-answer"),)))
    result = execute_procedure(registry, procedure.procedure_id, {"ready": True},
        backends={Backend.TOOL: lambda p, s, c: BackendResult({"answer": object()})})
    assert result.completed
    assert not result.closed_types_checked


def test_closed_execution_checks_actual_values_without_claiming_task_correctness():
    registry = ProcedureRegistry()
    procedure = registry.register("compute", Backend.TOOL,
        Signature((Precondition("value", "integer"),), (Effect("answer", "number"),)))
    result = execute_procedure(registry, procedure.procedure_id, {"value": 3}, closed_types=True,
        backends={Backend.TOOL: lambda p, s, c: BackendResult({"answer": s["value"] + 2})})
    assert result.completed and result.closed_types_checked
    assert result.resulting_state["answer"] == 5
    assert registry.get(procedure.procedure_id).value.uses == 0


def test_closed_input_constraints_are_checked_before_the_backend():
    registry = ProcedureRegistry()
    procedure = registry.register("bad input declaration", Backend.TOOL,
        Signature((Precondition("value", "integer", True),), (Effect("answer", "integer"),)))
    calls = []
    with pytest.raises(ValueError, match="structural value"):
        execute_procedure(registry, procedure.procedure_id, {"value": 1}, closed_types=True,
            backends={Backend.TOOL: lambda p, s, c: calls.append(p) or BackendResult({"answer": 5})})
    assert calls == []


@pytest.mark.parametrize("flag", [1, "true", None])
def test_closed_type_mode_requires_an_explicit_boolean(flag):
    with pytest.raises(ValueError, match="closed type"):
        execute_procedure(ProcedureRegistry(), "absent", {}, closed_types=flag, backends={})


def test_plan_requirements_cannot_use_unknown_types_even_without_work():
    from core.cognition.procedure_planning import ProcedurePlan, execute_procedure_plan

    plan = ProcedurePlan((Precondition("answer", "unknown-goal-type"),), (), True, 0, "already_met")
    with pytest.raises(ValueError, match="structural type"):
        execute_procedure_plan(ProcedureRegistry(), plan, {"answer": True}, backends={}, closed_types=True)
