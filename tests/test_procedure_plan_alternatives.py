"""Alternative dependency paths remain available to whole-task learning."""

import hashlib
import itertools

import pytest

from core.cognition.outcome_ledger import OutcomeLedger
from core.cognition.procedure import Backend, Effect, Origin, Precondition, ProcedureRegistry, Signature
from core.cognition.procedure_planning import plan_procedure, plan_procedure_candidates
from core.reasoning.action_value import ActionValueModel
from core.reasoning.procedure_value import (
    execute_valued_procedure_plan,
    observe_procedure_outcome,
    search_valued_procedure_plans,
)
from tests.test_procedure_task_feedback import BACKENDS, SITUATION, _registry


def _search(registry, state, goals, **limits):
    return plan_procedure_candidates(registry, state, goals,
        max_steps=limits.pop("max_steps", 4), max_expansions=limits.pop("max_expansions", 100),
        max_plans=limits.pop("max_plans", 20), **limits)


@pytest.mark.parametrize("readers,writers", [(1, 2), (2, 1), (3, 2), (2, 3)])
def test_search_preserves_all_paths_through_identical_requirement_states(readers, writers):
    registry = ProcedureRegistry()
    before = [registry.register(f"reader-{i}", Backend.TOOL,
        Signature((Precondition("source", "integer"),), (Effect("middle", "integer"),)))
        for i in range(readers)]
    after = [registry.register(f"writer-{i}", Backend.RLC,
        Signature((Precondition("middle", "integer"),), (Effect("answer", "integer"),)),
        origin=Origin(learner="test_fixture"))
        for i in range(writers)]
    goals = (Precondition("answer", "integer", 42),)
    result = _search(registry, {"source": 1}, goals)
    expected = {(a.procedure_id, b.procedure_id) for a, b in itertools.product(before, after)}
    assert {plan.procedure_ids for plan in result.plans} == expected
    assert len(result.plans) == len(expected)
    assert result.complete and result.reason == "frontier_exhausted"
    assert all(plan.value_obligations == ((plan.procedure_ids[-1], goals[0]),) for plan in result.plans)
    legacy = plan_procedure(registry, {"source": 1}, goals, max_steps=4, max_expansions=100)
    assert legacy == result.plans[0]
    assert all(procedure.value.uses == 0 for procedure in (*before, *after))


def test_quota_and_expansion_limits_keep_candidates_but_do_not_claim_exhaustion():
    registry, _, _ = _registry()
    state, goals = {"raw": [1, 2]}, (Precondition("total", "integer"),)
    quota = _search(registry, state, goals, max_plans=1)
    assert len(quota.plans) == 1 and not quota.complete and quota.reason == "plan_limit"
    limited = _search(registry, state, goals, max_expansions=2)
    assert len(limited.plans) == 1 and not limited.complete and limited.reason == "expansion_limit"
    depth = _search(registry, state, goals, max_steps=1)
    assert not depth.plans and not depth.complete and depth.reason == "depth_limit"
    exhausted = _search(registry, state, goals, eligible=())
    assert not exhausted.plans and exhausted.complete and exhausted.reason == "no_dependency_plan"


def test_repeatable_dependencies_are_bounded_without_merging_distinct_paths():
    registry = ProcedureRegistry()
    registry.register("loop", Backend.TOOL,
        Signature((Precondition("answer", "integer"),), (Effect("answer", "integer"),)))
    result = _search(registry, {}, (Precondition("answer", "integer"),), max_steps=3)
    assert not result.plans and not result.complete and result.reason == "depth_limit"
    assert result.expanded == 3


def test_satisfied_goal_needs_no_ranked_action_or_execution():
    registry, _, _ = _registry()
    result = search_valued_procedure_plans(registry, {"total": 5},
        (Precondition("total", "integer", 5),), ActionValueModel(stats={}),
        max_steps=0, max_expansions=1, max_plans=1)
    assert result.search.complete and result.search.reason == "already_satisfied"
    assert len(result.search.plans) == 1 and not result.ranked


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_plan_quota_is_an_explicit_positive_integer(value):
    with pytest.raises(ValueError, match="max_plans"):
        _search(ProcedureRegistry(), {}, (Precondition("answer", "integer"),), max_plans=value)


def test_external_outcomes_rerank_actual_search_alternatives_across_restart(tmp_path):
    registry, _, _ = _registry()
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    goals = (Precondition("total", "integer"),)
    values = ActionValueModel(stats={})
    proposal = search_valued_procedure_plans(registry, {"raw": [2, 3]}, goals, values,
        max_steps=3, max_expansions=100, max_plans=10, situation=SITUATION)
    assert proposal.search.complete and len(proposal.ranked) == 2
    assert not ledger.pending() and not ledger.measured_action_stats()
    observed_values = []
    for selected in proposal.ranked:
        pending = execute_valued_procedure_plan(registry, selected, {"raw": [2, 3]},
            backends=BACKENDS, ledger=ledger, situation=SITUATION)
        actual = pending.result.execution.resulting_state["total"]
        assert pending.result.completed  # Type-correct execution is not correctness.
        observed = float(actual == 5)
        observed_values.append(observed)
        observe_procedure_outcome(ledger, pending, observed=observed,
            evaluator="independent_sum_fixture_v1",
            evidence_sha256=hashlib.sha256(str((actual, 5)).encode()).hexdigest())
    assert sorted(observed_values) == [0., 1.]
    restarted, _, _ = _registry(reverse=True)
    values.refresh(OutcomeLedger(db_path=ledger._db_path))
    proposal = search_valued_procedure_plans(restarted, {"raw": [7, 8, 11]}, goals, values,
        max_steps=3, max_expansions=100, max_plans=10, situation=SITUATION)
    assert proposal.ranked[0].value.value > proposal.ranked[1].value.value
    pending = execute_valued_procedure_plan(restarted, proposal.ranked[0], {"raw": [7, 8, 11]},
        backends=BACKENDS, ledger=ledger, situation=SITUATION)
    assert pending.result.execution.resulting_state["total"] == 26
    assert sum(row["n"] for row in ledger.measured_action_stats().values()) == 2
    assert len(ledger.pending()) == 1
