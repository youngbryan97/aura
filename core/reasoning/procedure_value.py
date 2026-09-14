"""Task-level procedure feedback through the existing outcome/value learner.

The caller supplies eligible plans and later supplies an independent observed
outcome. Execution and type checks cannot resolve their own correctness.
No task reward is attributed to individual leaves of a composition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.outcome_ledger import CreditSource, OutcomeLedger, OutcomeReceipt
from core.cognition.procedure import Backend, ProcedureRegistry
from core.cognition.procedure_execution import BackendExecutor
from core.cognition.procedure_planning import (
    ProcedureGoalExecution,
    ProcedurePlan,
    _canonical,
    execute_procedure_plan,
)
from core.reasoning.action_value import ActionValue, ActionValueModel


def plan_action_key(registry: ProcedureRegistry, plan: ProcedurePlan) -> str:
    """Bind the ordered computation and requested goal, never local IDs."""
    if not plan.found or not plan.procedure_ids:
        raise ValueError("feedback needs an executable nonempty plan")
    contracts = tuple(registry.execution_contract(identity) for identity in plan.procedure_ids)
    if None in contracts:
        raise ValueError("procedure feedback needs stable executable contracts")
    payload = ["procedure_task_v1", contracts, [key for key, _ in _canonical(plan.requirements)]]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return "procedure-task:" + digest


@dataclass(frozen=True, slots=True)
class ValuedProcedurePlan:
    plan: ProcedurePlan
    action_key: str
    value: ActionValue
    situation_key: str


def rank_procedure_plans(
    registry: ProcedureRegistry,
    plans: Sequence[ProcedurePlan],
    values: ActionValueModel,
    *,
    situation: Mapping[str, Any] | str | None = None,
) -> tuple[ValuedProcedurePlan, ...]:
    """Rank caller-eligible alternatives to the same goal using measured values.

    Reads are in-memory. The execution owner refreshes the shared value model
    off the event loop before planning. Equal values preserve shorter plans
    and then caller order; no name-based preference is invented.
    """
    candidates = []
    goal = None
    for plan in plans:
        key = plan_action_key(registry, plan)
        requested = tuple(key for key, _ in _canonical(plan.requirements))
        if goal is not None and requested != goal:
            raise ValueError("procedure alternatives must target the same task")
        goal = requested
        candidates.append(ValuedProcedurePlan(
            plan, key, values.value_for(key, state=situation), values.state_key(situation),
        ))
    return tuple(sorted(candidates, key=lambda item: (-item.value.value, len(item.plan.procedure_ids))))


@dataclass(frozen=True, slots=True)
class PendingProcedureOutcome:
    receipt_id: str
    action_key: str
    result: ProcedureGoalExecution


class ProcedureOutcomeExecutionError(RuntimeError):
    """An execution failed before producing its normal result; receipt retained."""

    def __init__(self, receipt_id: str) -> None:
        self.receipt_id = receipt_id
        super().__init__(f"procedure execution failed; outcome pending: {receipt_id}")


def execute_valued_procedure_plan(
    registry: ProcedureRegistry,
    selected: ValuedProcedurePlan,
    state: Mapping[str, Any],
    *,
    backends: Mapping[Backend, BackendExecutor],
    ledger: OutcomeLedger,
    situation: Mapping[str, Any] | str | None = None,
    context: Mapping[str, Any] | None = None,
) -> PendingProcedureOutcome:
    """Open one task receipt at actual execution, not during plan search."""
    if plan_action_key(registry, selected.plan) != selected.action_key:
        raise ValueError("selected procedure contract changed before execution")
    if ActionValueModel.state_key(situation) != selected.situation_key:
        raise ValueError("selected procedure situation changed before execution")
    receipt_id = ledger.open(
        selected.action_key, selected.value.value, category="procedure_task",
        sources=[CreditSource("plan", selected.action_key)],
        context={"state": selected.situation_key},
        # These are distinct executions even when their inputs are identical.
        collapse_window_s=0,
    )
    try:
        result = execute_procedure_plan(
            registry, selected.plan, state, backends=backends, context=context,
        )
    except Exception as exc:
        raise ProcedureOutcomeExecutionError(receipt_id) from exc
    return PendingProcedureOutcome(receipt_id, selected.action_key, result)


def observe_procedure_outcome(
    ledger: OutcomeLedger,
    pending: PendingProcedureOutcome,
    *,
    observed: float,
    evaluator: str,
    evidence_sha256: str,
) -> OutcomeReceipt | None:
    """Resolve an external task assessment; a named evaluator is provenance.

    This function does not certify that the evaluator is independent or that
    its result is true. Those remain the execution owner's responsibilities.
    Duplicate assessments use the ledger's existing one-resolution contract.
    """
    if type(observed) not in (int, float) or not math.isfinite(observed) or not 0 <= observed <= 1:
        raise ValueError("task outcome must be a finite measured value in [0, 1]")
    if not isinstance(evaluator, str) or not evaluator.strip() or len(evaluator) > 120:
        raise ValueError("task outcome needs a bounded evaluator identity")
    if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_sha256
    ):
        raise ValueError("task outcome needs an evidence digest")
    return ledger.resolve(
        pending.receipt_id, observed,
        note=f"procedure task assessed by {evaluator}; evidence={evidence_sha256}",
    )
