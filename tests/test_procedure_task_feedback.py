"""Whole-task feedback across backends, without self-grading execution."""

import hashlib
from dataclasses import replace

import pytest

from core.cognition.outcome_ledger import OutcomeLedger
from core.cognition.procedure import (
    Backend,
    Effect,
    Origin,
    Precondition,
    ProcedureRegistry,
    Signature,
    compose,
)
from core.cognition.procedure_execution import BackendResult
from core.cognition.procedure_planning import ProcedurePlan
from core.reasoning.action_value import ActionValueModel
from core.reasoning.procedure_value import (
    execute_valued_procedure_plan,
    observe_procedure_outcome,
    plan_action_key,
    rank_procedure_plans,
)


def _registry(reverse=False):
    registry = ProcedureRegistry()
    parts = {}
    for name in reversed(("bad_read", "good_read", "sum")) if reverse else ("bad_read", "good_read", "sum"):
        signature = (
            Signature((Precondition("numbers", "integer_sequence"),), (Effect("total", "integer"),))
            if name == "sum" else
            Signature((Precondition("raw", "integer_sequence"),), (Effect("numbers", "integer_sequence"),))
        )
        parts[name] = registry.intern(
            name, hashlib.sha256(name.encode()).hexdigest(), name,
            Backend.RLC if name == "sum" else Backend.TOOL, signature,
            origin=Origin(learner="test_fixture"),
        )
    plans = tuple(ProcedurePlan(
        (Precondition("total", "integer"),),
        (parts[name].procedure_id, parts["sum"].procedure_id), True, 2, "dependencies_satisfied",
    ) for name in ("bad_read", "good_read"))
    return registry, parts, plans


def _backend(procedure, state, context):
    if procedure.name == "sum":
        return BackendResult({"total": sum(state["numbers"])})
    numbers = list(state["raw"])
    if procedure.name == "bad_read":
        numbers[0] += 1
    return BackendResult({"numbers": numbers})


BACKENDS = {Backend.TOOL: _backend, Backend.RLC: _backend}
SITUATION = {"input_kind": "integer_sequence", "goal": "sum"}


def test_measured_task_failure_changes_later_selection_across_restart(tmp_path):
    registry, parts, plans = _registry()
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    values = ActionValueModel(stats={})
    assert rank_procedure_plans(registry, plans, values, situation=SITUATION)[0].plan == plans[0]
    for plan, observed in zip(plans, (0.0, 1.0), strict=True):
        selected = rank_procedure_plans(registry, [plan], values, situation=SITUATION)[0]
        pending = execute_valued_procedure_plan(
            registry, selected, {"raw": [2, 3]}, backends=BACKENDS, ledger=ledger, situation=SITUATION,
        )
        # Both returned integers and completed. Only the external check differs.
        assert pending.result.completed
        assert len(ledger.pending()) == 1
        actual = pending.result.execution.resulting_state["total"]
        assert float(actual == 5) == observed
        before = ledger.measured_action_stats()
        assert selected.action_key not in before
        assert observe_procedure_outcome(
            ledger, pending, observed=observed, evaluator="independent_sum_fixture_v1",
            evidence_sha256=hashlib.sha256(str((actual, 5)).encode()).hexdigest(),
        ).is_evidence
        assert observe_procedure_outcome(
            ledger, pending, observed=observed, evaluator="independent_sum_fixture_v1",
            evidence_sha256="e" * 64,
        ) is None
    assert all(registry.get(part.procedure_id).value.uses == 0 for part in parts.values())

    restarted, _, replans = _registry(reverse=True)
    assert plans[0].procedure_ids != replans[0].procedure_ids
    assert plan_action_key(registry, plans[0]) == plan_action_key(restarted, replans[0])
    reopened = OutcomeLedger(db_path=ledger._db_path)
    learned = ActionValueModel()
    learned.refresh(reopened)
    ranked = rank_procedure_plans(restarted, replans, learned, situation=SITUATION)
    assert ranked[0].plan == replans[1]
    assert ranked[0].value.is_evidenced
    novel = execute_valued_procedure_plan(
        restarted, ranked[0], {"raw": [7, 8, 11]}, backends=BACKENDS,
        ledger=reopened, situation=SITUATION,
    )
    assert novel.result.completed and novel.result.execution.resulting_state["total"] == 26
    assert len(reopened.measured_action_stats()) == 2
    # The new ungraded task is not an extra successful observation.
    assert sum(row["n"] for row in reopened.measured_action_stats().values()) == 2


def test_contract_covers_composed_leaves_order_and_task():
    registry, parts, plans = _registry()
    other, other_parts, _ = _registry(reverse=True)
    first = compose(registry, [parts["good_read"], parts["sum"]], intern=True)
    second = compose(other, [other_parts["good_read"], other_parts["sum"]], intern=True)
    assert registry.execution_contract(first.procedure_id) == other.execution_contract(second.procedure_id)
    assert plan_action_key(registry, plans[0]) != plan_action_key(registry, plans[1])
    different_goal = replace(plans[0], requirements=(Precondition("total", "integer", 5),))
    assert plan_action_key(registry, plans[0]) != plan_action_key(registry, different_goal)
    with pytest.raises(ValueError, match="same task"):
        rank_procedure_plans(registry, [plans[0], different_goal], ActionValueModel(stats={}))


def test_unbound_legacy_procedure_is_not_given_a_durable_learning_identity():
    registry = ProcedureRegistry()
    legacy = registry.register("legacy", Backend.TOOL, Signature(effects=(Effect("x", "integer"),)))
    plan = ProcedurePlan((Precondition("x", "integer"),), (legacy.procedure_id,), True, 1, "dependencies_satisfied")
    assert registry.execution_contract(legacy.procedure_id) is None
    with pytest.raises(ValueError, match="stable executable contracts"):
        plan_action_key(registry, plan)


def test_separate_executions_never_collapse_and_context_cannot_drift(tmp_path):
    registry, _, plans = _registry()
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    selected = rank_procedure_plans(registry, plans, ActionValueModel(stats={}), situation=SITUATION)[0]
    with pytest.raises(ValueError, match="situation changed"):
        execute_valued_procedure_plan(registry, selected, {"raw": [2]}, backends=BACKENDS, ledger=ledger)
    assert not ledger.pending()
    pending = [execute_valued_procedure_plan(
        registry, selected, {"raw": [2]}, backends=BACKENDS, ledger=ledger, situation=SITUATION,
    ) for _ in range(2)]
    assert pending[0].receipt_id != pending[1].receipt_id
    assert ledger.measured_action_stats() == {}


@pytest.mark.parametrize("observed", [float("nan"), float("inf"), -1, 2, True])
def test_bad_assessment_preserves_pending_task(tmp_path, observed):
    registry, _, plans = _registry()
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    selected = rank_procedure_plans(registry, plans, ActionValueModel(stats={}))[0]
    pending = execute_valued_procedure_plan(registry, selected, {"raw": [2]}, backends=BACKENDS, ledger=ledger)
    with pytest.raises(ValueError, match="finite measured"):
        observe_procedure_outcome(ledger, pending, observed=observed, evaluator="fixture", evidence_sha256="e" * 64)
    assert len(ledger.pending()) == 1
