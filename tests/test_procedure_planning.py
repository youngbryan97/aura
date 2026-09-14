"""Task requirements, rather than input presence, constrain common procedures."""

from dataclasses import replace

import pytest

from core.cognition.procedure import (
    Backend,
    Effect,
    Origin,
    Precondition,
    ProceduralValue,
    ProcedureRegistry,
    Signature,
)
from core.cognition.procedure_execution import BackendResult
from core.cognition.procedure_planning import execute_procedure_plan, plan_procedure


def _register(registry, name, reads=(), writes=(), *, value=0, backend=Backend.TOOL):
    return registry.register(name, backend, Signature(tuple(reads), tuple(writes)),
        value=ProceduralValue(value_when_it_works=value), origin=Origin(learner="test_fixture"))


def _plan(registry, state, requirements, **kwargs):
    return plan_procedure(registry, state, requirements,
        max_steps=kwargs.pop("max_steps", 8), max_expansions=kwargs.pop("max_expansions", 100), **kwargs)


def test_novel_dependency_composition_is_selected_without_executing():
    registry = ProcedureRegistry()
    unrelated = _register(registry, "high value but unrelated", writes=(Effect("other", "integer"),), value=100)
    fetch = _register(registry, "fetch", reads=(Precondition("source", "string"),),
        writes=(Effect("numbers", "integer_sequence"),))
    total = _register(registry, "total", reads=(Precondition("numbers", "integer_sequence"),),
        writes=(Effect("total", "integer"),), backend=Backend.RLC)
    report = _register(registry, "report", reads=(Precondition("total", "integer"),),
        writes=(Effect("report", "string"),), backend=Backend.MACRO)
    plan = _plan(registry, {"source": "local.csv"}, (Precondition("report", "string"),))
    assert plan.found and plan.procedure_ids == (fetch.procedure_id, total.procedure_id, report.procedure_id)
    assert unrelated.procedure_id not in plan.procedure_ids
    assert plan.expanded == 3
    assert len(registry.procedures()) == 4


def test_constant_goal_rejects_a_more_valuable_opposite_effect():
    registry = ProcedureRegistry()
    _register(registry, "disable", writes=(Effect("enabled", "boolean", False),), value=100)
    enable = _register(registry, "enable", writes=(Effect("enabled", "boolean", True),))
    plan = _plan(registry, {"enabled": False}, (Precondition("enabled", "boolean", True),))
    assert plan.procedure_ids == (enable.procedure_id,)


def test_computed_output_is_not_a_prediction_of_its_exact_value():
    registry = ProcedureRegistry()
    procedure = _register(registry, "compute", writes=(Effect("answer", "integer"),))
    plan = _plan(registry, {}, (Precondition("answer", "integer", 42),))
    assert plan.found and plan.reason == "value_observation_required"
    assert plan.value_obligations == ((procedure.procedure_id, plan.requirements[0]),)
    for actual in (7, 42):
        run = execute_procedure_plan(registry, plan, {}, backends={
            Backend.TOOL: lambda p, s, c, actual=actual: BackendResult({"answer": actual}),
        })
        assert run.execution.completed
        assert run.completed is (actual == 42)
    assert registry.get(procedure.procedure_id).value.uses == 0


def test_computed_intermediate_is_observed_before_dependent_backend_executes():
    registry = ProcedureRegistry()
    first = _register(registry, "measure", writes=(Effect("x", "integer"),))
    second = _register(registry, "requires measured four", reads=(Precondition("x", "integer", 4),),
        writes=(Effect("answer", "integer"),), backend=Backend.RLC)
    plan = _plan(registry, {}, (Precondition("answer", "integer", 8),))
    assert plan.procedure_ids == (first.procedure_id, second.procedure_id)
    assert len(plan.value_obligations) == 2
    for actual in (3, 4):
        calls = []
        def compute(p, s, c, calls=calls):
            calls.append(s["x"])
            return BackendResult({"answer": 2 * s["x"]})
        run = execute_procedure_plan(registry, plan, {}, backends={
            Backend.TOOL: lambda p, s, c, actual=actual: BackendResult({"x": actual}), Backend.RLC: compute,
        })
        assert run.completed is (actual == 4)
        assert calls == ([4] if actual == 4 else [])


def test_impossible_constant_and_type_constraints_do_not_become_speculation():
    registry = ProcedureRegistry()
    _register(registry, "compute", writes=(Effect("answer", "integer"),))
    for goal in (
        (Precondition("answer", "integer", 1), Precondition("answer", "integer", 2)),
        (Precondition("answer", "any", "word"),),
        (Precondition("answer", "boolean", False),),
        (Precondition("answer", "integer", negated=True),),
    ):
        assert not _plan(registry, {}, goal).found


@pytest.mark.parametrize("produced,required,value", [
    ("integer", "number", 7), ("integer_sequence", "sequence", (2, 3)),
])
def test_structural_subtype_crosses_backend_boundary(produced, required, value):
    registry = ProcedureRegistry()
    first = _register(registry, "produce", writes=(Effect("x", produced),))
    second = _register(registry, "consume", reads=(Precondition("x", required),),
        writes=(Effect("answer", "string"),), backend=Backend.MACRO)
    plan = _plan(registry, {}, (Precondition("answer", "string", "observed"),))
    assert plan.procedure_ids == (first.procedure_id, second.procedure_id)
    result = execute_procedure_plan(registry, plan, {}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult({"x": value}),
        Backend.MACRO: lambda p, s, c: BackendResult({"answer": "observed"}),
    })
    assert result.completed


@pytest.mark.parametrize("produced,required", [
    ("number", "integer"), ("sequence", "integer_sequence"),
    ("boolean", "number"), ("string", "sequence"),
])
def test_subtyping_does_not_invent_a_downcast(produced, required):
    registry = ProcedureRegistry()
    _register(registry, "produce", writes=(Effect("x", produced),))
    _register(registry, "consume", reads=(Precondition("x", required),),
        writes=(Effect("answer", "string"),))
    assert not _plan(registry, {}, (Precondition("answer", "string"),)).found


def test_task_identity_prevents_type_matched_semantic_substitution():
    registry = ProcedureRegistry()
    reads, writes = (Precondition("x", "integer"),), (Effect("answer", "integer"),)
    intended = _register(registry, "intended", reads, writes)
    _register(registry, "different computation", reads, writes, value=100)
    plan = _plan(registry, {"x": 5}, (Precondition("answer", "integer"),),
        eligible=(intended.procedure_id,))
    assert plan.procedure_ids == (intended.procedure_id,)
    assert not _plan(registry, {"x": 5}, plan.requirements, eligible=()).found


def test_all_preconditions_and_preserved_goals_must_hold():
    registry = ProcedureRegistry()
    _register(registry, "unsafe shortcut", writes=(Effect("answer", "integer"), Effect("keep", "boolean", False)))
    _register(registry, "missing input", reads=(Precondition("missing", "integer"),),
        writes=(Effect("answer", "integer"),))
    goal = (Precondition("answer", "integer"), Precondition("keep", "boolean", True))
    assert not _plan(registry, {"keep": True}, goal).found


def test_contradictory_constant_dependency_is_not_inferred_from_type():
    registry = ProcedureRegistry()
    _register(registry, "wrong constant", writes=(Effect("x", "integer", 3),))
    _register(registry, "needs four", reads=(Precondition("x", "integer", 4),),
        writes=(Effect("answer", "integer"),))
    assert not _plan(registry, {}, (Precondition("answer", "integer"),)).found


def test_false_can_be_a_required_and_observed_boolean():
    registry = ProcedureRegistry()
    off = _register(registry, "off", writes=(Effect("flag", "boolean", False),))
    plan = _plan(registry, {}, (Precondition("flag", "boolean", False),))
    run = execute_procedure_plan(registry, plan, {}, backends={
        Backend.TOOL: lambda p, s, c: BackendResult({"flag": False}),
    })
    assert run.completed and run.execution.steps[0].procedure_id == off.procedure_id


def test_cycles_terminate_and_resource_exhaustion_is_distinct_from_no_plan():
    registry = ProcedureRegistry()
    _register(registry, "a from b", (Precondition("b", "integer"),), (Effect("a", "integer"),))
    _register(registry, "b from a", (Precondition("a", "integer"),), (Effect("b", "integer"),))
    goal = (Precondition("a", "integer"),)
    assert _plan(registry, {}, goal).reason == "no_dependency_plan"
    assert _plan(registry, {}, goal, max_steps=1).reason == "depth_limit"
    assert _plan(registry, {}, goal, max_expansions=1).reason == "expansion_limit"


def test_already_satisfied_goal_is_rechecked_against_execution_state():
    registry = ProcedureRegistry()
    plan = _plan(registry, {"answer": 3}, (Precondition("answer", "integer"),))
    assert plan.reason == "already_satisfied"
    assert execute_procedure_plan(registry, plan, {"answer": 4}, backends={}).completed
    assert not execute_procedure_plan(registry, plan, {}, backends={}).completed


def test_registry_refresh_supplies_recently_learned_procedures():
    registry = ProcedureRegistry()
    registry.keep_current_with(lambda: _register(registry, "new", writes=(Effect("answer", "integer"),)))
    assert _plan(registry, {}, (Precondition("answer", "integer"),)).found


def test_identity_bound_runtime_does_not_refresh_unrelated_learners():
    registry = ProcedureRegistry()
    p = _register(registry, "known", writes=(Effect("answer", "integer"),))
    registry.keep_current_with(lambda: pytest.fail("unrelated learner refresh"))
    assert _plan(registry, {}, (Precondition("answer", "integer"),), eligible=(p.procedure_id,)).found


def test_goal_execution_does_not_turn_completion_into_correctness_or_repeat_registration():
    registry = ProcedureRegistry()
    first = _register(registry, "source", writes=(Effect("x", "integer"),))
    last = _register(registry, "double", (Precondition("x", "integer"),), (Effect("answer", "integer"),))
    plan = _plan(registry, {}, (Precondition("answer", "integer"),))
    def backend(p, s, c):
        return BackendResult({"x": 3} if p.procedure_id == first.procedure_id else {"answer": s["x"] * 2})
    one = execute_procedure_plan(registry, plan, {}, backends={Backend.TOOL: backend})
    two = execute_procedure_plan(registry, plan, {}, backends={Backend.TOOL: backend})
    assert one.completed and two.completed
    assert two.execution.resulting_state["answer"] == 6
    assert one.execution.procedure_id == two.execution.procedure_id
    assert len(registry.procedures()) == 3
    assert registry.get(last.procedure_id).value.uses == 0


def test_failed_or_retired_plan_never_runs_or_claims_success(monkeypatch):
    registry = ProcedureRegistry()
    p = _register(registry, "run", writes=(Effect("answer", "integer"),))
    plan = _plan(registry, {}, (Precondition("answer", "integer"),))
    calls = []
    failed = replace(plan, found=False)
    assert not execute_procedure_plan(registry, failed, {}, backends={}).completed
    monkeypatch.setattr(registry, "get", lambda _: replace(p, retired=True))
    with pytest.raises(ValueError, match="no longer available"):
        execute_procedure_plan(registry, plan, {}, backends={Backend.TOOL: lambda *args: calls.append(1)})
    assert calls == []


def test_backend_failure_and_stale_state_are_not_successful_goals():
    registry = ProcedureRegistry()
    _register(registry, "run", reads=(Precondition("x", "integer"),), writes=(Effect("answer", "integer"),))
    plan = _plan(registry, {"x": 1}, (Precondition("answer", "integer"),))
    def backend(*args):
        raise RuntimeError("failed computation")
    run = execute_procedure_plan(registry, plan, {"x": 1, "answer": 99}, backends={Backend.TOOL: backend})
    assert run.requirements_met and not run.completed


@pytest.mark.parametrize("limits", [{"max_steps": -1}, {"max_steps": True}, {"max_expansions": 0}])
def test_invalid_bounds_are_not_interpreted_as_search(limits):
    with pytest.raises(ValueError):
        _plan(ProcedureRegistry(), {}, (Precondition("answer"),), **limits)


def test_empty_goal_is_not_evidence_of_task_completion():
    with pytest.raises(ValueError, match="nonempty"):
        _plan(ProcedureRegistry(), {}, ())


def test_task_goal_invariant_is_executable():
    from core.cognition.procedure_planning import _goal_bound_reuse_holds

    assert _goal_bound_reuse_holds() == ()


def test_constant_identity_preserves_list_tuple_distinctions_in_search():
    from core.cognition.procedure_planning import _canonical

    requirements = (Precondition("x", "sequence", [1]), Precondition("x", "sequence", (1,)))
    assert len(_canonical(requirements)) == 2
    registry = ProcedureRegistry()
    _register(registry, "list", writes=(Effect("x", "sequence", [1]),))
    assert not _plan(registry, {}, requirements).found


@pytest.mark.parametrize("seed", range(20))
def test_dependency_search_matches_exhaustive_constant_effect_plans(seed):
    import random
    from itertools import product

    rng = random.Random(seed)
    registry = ProcedureRegistry()
    keys = ("a", "b", "c")
    parts = []
    for index in range(5):
        reads = tuple(Precondition(key, "boolean", bool(rng.randrange(2)))
            for key in keys if rng.random() < .5)
        writes = tuple(Effect(key, "boolean", bool(rng.randrange(2)))
            for key in keys if rng.random() < .5)
        parts.append(_register(registry, str(index), reads, writes))
    state = {key: bool(rng.randrange(2)) for key in keys}
    goal = tuple(Precondition(key, "boolean", bool(rng.randrange(2))) for key in keys)
    expected = None
    for length in range(4):
        for path in product(parts, repeat=length):
            current = dict(state)
            for part in path:
                if not part.signature.matches(current):
                    break
                current.update({effect.key: effect.value for effect in part.signature.effects})
            else:
                if all(item.satisfied_by(current) for item in goal):
                    expected = length
                    break
        if expected is not None:
            break
    plan = _plan(registry, state, goal, max_steps=3, max_expansions=5000)
    assert plan.found == (expected is not None)
    if plan.found:
        assert len(plan.procedure_ids) == expected
        run = execute_procedure_plan(registry, plan, state, backends={
            Backend.TOOL: lambda p, s, c: BackendResult({e.key: e.value for e in p.signature.effects}),
        })
        assert run.completed
