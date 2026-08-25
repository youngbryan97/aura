"""Adversarial contracts for RLC-selected host effects."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.brain.external_execute_coordinator import (
    EXTERNAL_EXECUTE_TRANSACTION_SCHEMA,
    ExternalExecuteCoordinator,
    ExternalExecutionInProgressError,
)
from core.brain.llm.latent_cortex.epistemic_runtime import (
    RUNTIME_OPERATION_SCHEMA,
)
from core.brain.llm.latent_cortex.epistemic_state import (
    ComputeBudgetState,
    EpistemicState,
    EpistemicTransaction,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    ProblemFrame,
)
from core.brain.llm.latent_cortex.external_execution import (
    build_external_execution_handoff,
    build_external_execution_offer,
    build_external_execution_readiness,
    validate_external_execution_handoff,
    validate_external_execution_offer,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    ACTION_TRANSITION_SCHEMA,
    CognitiveStateSignal,
    ValueOfComputationPolicy,
    build_evidence_snapshot,
    transition_reward,
)


def _offer(
    *,
    action_id: str = "action-rlc-1",
    request_digest: str = "sha256:" + "1" * 64,
) -> dict[str, Any]:
    return build_external_execution_offer(
        action_id=action_id,
        domain="external_action",
        action_name="write_note",
        request_digest=request_digest,
        will_receipt_id="will-rlc-1",
        objective="Create the already-admitted note.",
        expectation={
            "objective": "The note exists",
            "success_criteria": ["note is observable"],
        },
    )


def _execute_trace() -> list[dict[str, Any]]:
    state = CognitiveStateSignal(
        step_index=0,
        max_steps=8,
        neural_steps=2,
        min_neural_steps=1,
        active_branches=1,
        total_branches=1,
        residual=0.2,
        residual_delta=0.1,
        verifier_score=None,
        verifier_delta=None,
        disagreement=0.1,
        uncertainty=0.2,
        budget_remaining_fraction=0.8,
        has_memory=True,
        has_evidence=True,
        has_verifier=False,
        has_savepoint=True,
        can_execute=True,
        answer_verified=True,
        irreducible_uncertainty=False,
        previously_selected=(),
    )
    decision = ValueOfComputationPolicy(_action_policy()).choose(
        state,
        executors=_executors(),
    )
    assert decision["action"] == "execute"
    return [
        {
            "decision": decision,
            "transition": {
                "schema": ACTION_TRANSITION_SCHEMA,
                "bucket": _action_policy()["bucket"],
                "snapshot_sha256": _action_policy()["snapshot_sha256"],
                "decision_sha256": decision["decision_sha256"],
                "step_index": decision["step_index"],
                "action": decision["action"],
                "mode": decision["mode"],
                "outcome": "external_execute_requested",
                "checked": False,
                "metrics": transition_reward(
                    verified_delta=0.0,
                    information_gain=0.0,
                    diversity_gain=0.0,
                    unsupported_confidence=0.0,
                    cost=0.01,
                ),
            },
            "state_signal": state.to_dict(),
            "state_before": {
                "residual": 0.2,
                "disagreement": 0.1,
                "verifier_score": None,
                "budget_remaining_fraction": 0.8,
            },
            "state_after": {
                "residual": 0.2,
                "disagreement": 0.1,
                "verifier_score": None,
                "observed_verifier_score": None,
            },
            "affected_branches": 0,
            "verification": {
                "target_branch": None,
                "observation": {},
                "decision": "not_run",
                "restored": False,
            },
        }
    ]


def _campaign_forced_execute_trace() -> list[dict[str, Any]]:
    trace = _execute_trace()
    state = CognitiveStateSignal.from_dict(trace[0]["state_signal"])
    decision = ValueOfComputationPolicy(_action_policy()).choose_forced(
        state,
        executors=_executors(),
        action=OperationKind.EXECUTE,
    )
    trace[0]["decision"] = decision
    trace[0]["transition"].update(
        {
            "decision_sha256": decision["decision_sha256"],
            "mode": decision["mode"],
        }
    )
    return trace


def _non_execute_trace() -> list[dict[str, Any]]:
    state = CognitiveStateSignal(
        step_index=0,
        max_steps=8,
        neural_steps=2,
        min_neural_steps=1,
        active_branches=1,
        total_branches=1,
        residual=0.5,
        residual_delta=0.0,
        verifier_score=None,
        verifier_delta=None,
        disagreement=0.1,
        uncertainty=0.6,
        budget_remaining_fraction=0.8,
        has_memory=True,
        has_evidence=True,
        has_verifier=False,
        has_savepoint=True,
        can_execute=False,
        answer_verified=False,
        irreducible_uncertainty=False,
        previously_selected=(),
    )
    decision = ValueOfComputationPolicy(_action_policy()).choose(
        state,
        executors=tuple(
            item for item in _executors() if item is not OperationKind.EXECUTE
        ),
    )
    return [
        {
            "decision": decision,
            "transition": {
                "schema": ACTION_TRANSITION_SCHEMA,
                "bucket": _action_policy()["bucket"],
                "snapshot_sha256": _action_policy()["snapshot_sha256"],
                "decision_sha256": decision["decision_sha256"],
                "step_index": 0,
                "action": decision["action"],
                "mode": decision["mode"],
                "outcome": "completed",
                "checked": False,
                "metrics": transition_reward(
                    verified_delta=0.0,
                    information_gain=0.0,
                    diversity_gain=0.0,
                    unsupported_confidence=0.0,
                    cost=0.01,
                ),
            },
            "state_signal": state.to_dict(),
            "state_before": {
                "residual": 0.5,
                "disagreement": 0.1,
                "verifier_score": None,
                "budget_remaining_fraction": 0.8,
            },
            "state_after": {
                "residual": 0.5,
                "disagreement": 0.1,
                "verifier_score": None,
                "observed_verifier_score": None,
            },
            "affected_branches": 0,
            "verification": {
                "target_branch": None,
                "observation": {},
                "decision": "not_run",
                "restored": False,
            },
        }
    ]


def _policy_receipt(
    trace: list[dict[str, Any]],
    executors: list[OperationKind],
) -> dict[str, Any]:
    evidence = _action_policy()
    return {
        "schema": evidence["schema"],
        "bucket": evidence["bucket"],
        "snapshot_sha256": evidence["snapshot_sha256"],
        "active": True,
        "executors": [item.value for item in executors],
        "actions_selected": len(trace),
        "checked_transitions": sum(
            int(row["transition"]["checked"]) for row in trace
        ),
        "selected_actions": [row["decision"]["action"] for row in trace],
    }


def _runtime_operation_receipt(
    offer: dict[str, Any],
    trace: list[dict[str, Any]],
    policy_receipt: dict[str, Any],
    *,
    objective: str = "Create the already-admitted note.",
) -> dict[str, Any]:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def canonical_digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    controller = {
        "schema": "aura.latent_execution_controller.v1",
        "bucket": _action_policy()["bucket"],
        "arm": "base",
        "mode": "observe",
        "evidence": {},
    }
    decision_sha256 = canonical_digest(controller)
    config_sha256 = digest("config")
    budget_sha256 = digest("budget")
    operation_payload = {
        "objective_sha256": offer["objective_sha256"],
        "decision_sha256": decision_sha256,
        "config_sha256": config_sha256,
        "budget_sha256": budget_sha256,
        "action_policy_sha256": _action_policy()["snapshot_sha256"],
        "controller": controller,
        "external_execution_offer_sha256": offer["offer_sha256"],
    }
    input_payload_sha256 = canonical_digest(operation_payload)
    genesis = EpistemicState.genesis(
        episode_id="episode-external",
        problem=ProblemFrame.create(objective),
        budget=ComputeBudgetState(total=1.0),
    )
    attempt_sha256 = OperationRecord.compute_attempt_sha256(
        kind=OperationKind.BLIND_RESOLVE,
        operator_id="latent_execution_controller",
        operator_version="v2",
        input_payload_sha256=input_payload_sha256,
        input_claim_ids=(),
        input_hypothesis_ids=(),
        input_evidence_ids=(),
    )
    intent = OperationRecord.create(
        operation_id=f"rlc-op-{attempt_sha256[:20]}-a1",
        kind=OperationKind.BLIND_RESOLVE,
        outcome=OperationOutcome.UNKNOWN,
        input_state_sha256=genesis.state_sha256,
        cost=0.0,
        operator_id="latent_execution_controller",
        operator_version="v2",
        input_payload_sha256=input_payload_sha256,
        started_at=1.0,
        completed_at=1.0,
        failure_code="execution_pending",
    )
    admitted_state = EpistemicTransaction(genesis).add_operation(intent).commit()
    terminal = OperationRecord.create(
        operation_id=f"rlc-op-{attempt_sha256[:20]}-a2",
        kind=intent.kind,
        outcome=OperationOutcome.SUCCEEDED,
        input_state_sha256=admitted_state.state_sha256,
        cost=0.1,
        operator_id=intent.operator_id,
        operator_version=intent.operator_version,
        input_payload_sha256=intent.input_payload_sha256,
        started_at=1.0,
        completed_at=2.0,
        retry_of_operation_id=intent.operation_id,
        detail=(
            "worker outcome=succeeded; "
            "cost basis=token_layer_fraction_of_remaining_episode_budget"
        ),
    )
    action_operations = []
    for index, row in enumerate(trace):
        decision = row["decision"]
        action_operations.append(
            OperationRecord.create(
                operation_id=(
                    f"rlc-action-{decision['decision_sha256'][:20]}-{index:03d}"
                ),
                kind=OperationKind(decision["action"]),
                outcome=OperationOutcome.SUCCEEDED,
                input_state_sha256=admitted_state.state_sha256,
                cost=0.01,
                operator_id="value_of_computation",
                operator_version="v1",
                input_payload_sha256=decision["decision_sha256"],
                started_at=1.0,
                completed_at=2.0,
                detail=(
                    f"step={row['transition']['step_index']}; "
                    f"mode={row['transition']['mode']}; "
                    f"outcome={row['transition']['outcome']}; "
                    f"checked={row['transition']['checked']}"
                ),
            )
        )
    current_transaction = EpistemicTransaction(admitted_state).add_operation(
        terminal
    )
    for operation in action_operations:
        current_transaction.add_operation(operation)
    current_state = current_transaction.commit()
    journal_parent_sha256 = digest("journal-parent")
    journal_entry = {
        "schema": "aura.rlc.epistemic_journal.v1",
        "sequence": current_state.version,
        "previous_entry_sha256": journal_parent_sha256,
        "state_sha256": current_state.state_sha256,
        "state": current_state.to_dict(),
    }
    authority = {
        "schema": RUNTIME_OPERATION_SCHEMA,
        "episode_id": "episode-external",
        "objective_sha256": offer["objective_sha256"],
        "input_state_sha256": genesis.state_sha256,
        "admitted_state_sha256": admitted_state.state_sha256,
        "admitted_state_version": 1,
        "admitted_journal_head_sha256": journal_parent_sha256,
        "admitted_journal_entry_count": admitted_state.version + 1,
        "operation_id": intent.operation_id,
        "operation_kind": intent.kind.value,
        "operator_id": intent.operator_id,
        "operator_version": intent.operator_version,
        "attempt_sha256": intent.attempt_sha256,
        "input_payload_sha256": intent.input_payload_sha256,
        "decision_sha256": decision_sha256,
        "config_sha256": config_sha256,
        "budget_sha256": budget_sha256,
        "action_policy_sha256": _action_policy()["snapshot_sha256"],
        "controller_schema": controller["schema"],
        "controller_bucket": controller["bucket"],
        "controller_arm": controller["arm"],
        "controller_mode": controller["mode"],
        "controller_evidence": controller["evidence"],
        "input_claim_ids": [],
        "input_hypothesis_ids": [],
        "input_evidence_ids": [],
        "admission_reason": "admitted",
        "retry_of_operation_id": "",
        "external_execution_offer_sha256": offer["offer_sha256"],
    }
    return {
        "schema": RUNTIME_OPERATION_SCHEMA,
        "authority": authority,
        "intent": intent.to_dict(),
        "terminal": terminal.to_dict(),
        "action_operations": [row.to_dict() for row in action_operations],
        "completed": True,
        "admitted_state": admitted_state.to_dict(),
        "current_state_sha256": current_state.state_sha256,
        "current_state_version": current_state.version,
        "current_state": current_state.to_dict(),
        "journal": {
            "state_sha256": current_state.state_sha256,
            "state_version": current_state.version,
            "entry_count": current_state.version + 1,
            "previous_head_sha256": journal_parent_sha256,
            "head_sha256": canonical_digest(journal_entry),
            "size_bytes": 4096,
            "repaired_torn_tail_bytes": 0,
        },
        "compute": {
            "basis": "token_layer_fraction_of_remaining_episode_budget",
            "spent_layer_apps": 10 + len(action_operations),
            "max_layer_apps": 100,
            "fraction": (10 + len(action_operations)) / 100,
            "state_cost": 0.1 + (0.01 * len(action_operations)),
            "action_state_cost": 0.01 * len(action_operations),
            "action_operation_count": len(action_operations),
        },
    }


def _action_policy() -> dict[str, Any]:
    return build_evidence_snapshot(bucket="external-action-test", cells={})


def _executors() -> tuple[OperationKind, ...]:
    return (
        OperationKind.DECOMPOSE,
        OperationKind.EXECUTE,
        OperationKind.ANSWER,
        OperationKind.ABSTAIN,
    )


def _readiness_output(*, ready: bool = True) -> str:
    return json.dumps(
        {
            "action_ready": ready,
            "preconditions_met": ready,
            "risk_acceptable": ready,
            "expected_effect": "The note is created and observable.",
            "reason": (
                "All declared preconditions are met."
                if ready
                else "A required precondition is missing."
            ),
        },
        sort_keys=True,
    )


def _record_handoff(
    coordinator: ExternalExecuteCoordinator,
    offer: dict[str, Any],
    trace: list[dict[str, Any]],
    *,
    ready: bool = True,
) -> dict[str, Any]:
    model_output = _readiness_output(ready=ready)
    executors = list(_executors())
    policy_receipt = _policy_receipt(trace, executors)
    return coordinator.record_handoff(
        offer=offer,
        handoff=build_external_execution_handoff(offer, trace),
        cognitive_action_trace=trace,
        readiness=build_external_execution_readiness(offer, model_output),
        model_output=model_output,
        action_policy_evidence=_action_policy(),
        executors=[item.value for item in executors],
        action_policy_receipt=policy_receipt,
        runtime_operation=_runtime_operation_receipt(
            offer,
            trace,
            policy_receipt,
        ),
    )


def _downgrade_transaction_envelope_to_v2(
    coordinator: ExternalExecuteCoordinator,
) -> Path:
    import core.brain.external_execute_coordinator as coordinator_module

    path = next(coordinator.root.glob("*.json"))
    sealed = json.loads(path.read_text(encoding="utf-8"))
    sealed.pop("transaction_sha256")
    sealed.pop("action_intervention")
    sealed["schema"] = "aura.rlc.external_execute_transaction.v2"
    sealed["transaction_sha256"] = coordinator_module._canonical_sha256(sealed)
    path.write_text(
        json.dumps(sealed, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_offer_and_handoff_are_exact_digest_bound_contracts() -> None:
    offer = _offer()
    assert validate_external_execution_offer(offer) == offer
    trace = _execute_trace()
    handoff = build_external_execution_handoff(offer, trace)
    assert handoff["requested"] is True
    assert validate_external_execution_handoff(
        handoff,
        offer=offer,
        cognitive_action_trace=trace,
    ) == handoff

    tampered_offer = {**offer, "action_name": "delete_everything"}
    with pytest.raises(ValueError, match="digest"):
        validate_external_execution_offer(tampered_offer)
    with pytest.raises(ValueError, match="differs"):
        validate_external_execution_handoff(
            {**handoff, "requested": False},
            offer=offer,
            cognitive_action_trace=trace,
        )
    with pytest.raises(ValueError, match="at most once"):
        build_external_execution_handoff(offer, trace + trace)


def test_coordinator_is_duplicate_resistant_and_replays_terminal_result(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer()
    prepared = coordinator.prepare(offer)
    assert prepared["state"] == "PREPARED"
    assert coordinator.lookup(
        action_id=offer["action_id"],
        request_digest=offer["request_digest"],
    ) == prepared

    decided = _record_handoff(coordinator, offer, _execute_trace())
    assert decided["state"] == "DECIDED"
    assert decided["decision_source"] == "rlc"
    dispatching = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="test-task",
    )
    assert dispatching["state"] == "DISPATCHING"
    completed = coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatching["dispatch_owner"]["attempt_id"],
        result={
            "ok": True,
            "status": "success_verified",
            "transport_succeeded": True,
            "effect_verified": True,
            "created_path": "artifacts/output/hello.pdf",
            "credential": "must-not-persist",
        },
    )
    assert completed["state"] == "SUCCEEDED"
    assert (
        completed["result"]["replay_payload"]["created_path"]
        == "artifacts/output/hello.pdf"
    )
    assert completed["result"]["replay_payload"]["credential"] == "[REDACTED]"
    assert coordinator.prepare(offer) == completed

    conflicting = _offer(
        action_id=offer["action_id"],
        request_digest="sha256:" + "3" * 64,
    )
    with pytest.raises(ValueError, match="conflicts"):
        coordinator.prepare(conflicting)


def test_campaign_forced_execute_reaches_durable_host_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from core.brain.llm.latent_cortex import action_intervention as intervention_mod

    intervention = {
        "schema": "aura.rlc.action_intervention.v2",
        "authority_payload": {
            "action": "execute",
            "arm": "forced_action",
            "intervention_ordinal": 0,
        },
        "intervention_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention",
        lambda value, *, require_current_policy: (
            intervention if value == intervention else None
        ),
    )
    monkeypatch.setattr(
        intervention_mod,
        "validate_action_intervention_receipt",
        lambda value, *, intervention, cognitive_action_trace: value,
    )
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-campaign-forced")
    coordinator.prepare(offer)
    trace = _campaign_forced_execute_trace()
    executors = list(_executors())
    policy_receipt = {
        **_policy_receipt(trace, executors),
        "calibration_intervention": {"receipt_sha256": "b" * 64},
    }
    model_output = _readiness_output()
    decided = coordinator.record_handoff(
        offer=offer,
        handoff=build_external_execution_handoff(offer, trace),
        cognitive_action_trace=trace,
        readiness=build_external_execution_readiness(offer, model_output),
        model_output=model_output,
        action_policy_evidence=_action_policy(),
        executors=[item.value for item in executors],
        action_policy_receipt=policy_receipt,
        runtime_operation=_runtime_operation_receipt(
            offer,
            trace,
            policy_receipt,
        ),
        action_intervention=intervention,
    )
    assert decided["state"] == "DECIDED"
    assert decided["action_intervention"] == intervention
    assert decided["handoff"]["requested"] is True


def test_host_rejects_minimal_unvalidated_execute_trace(tmp_path: Path) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-minimal-trace")
    coordinator.prepare(offer)
    minimal = [
        {
            "decision": {
                "action": "execute",
                "decision_sha256": "2" * 64,
                "step_index": 2,
                "mode": "verified_execute",
            },
            "transition": {
                "action": "execute",
                "decision_sha256": "2" * 64,
                "outcome": "external_execute_requested",
                "checked": False,
            },
            "state_signal": {"can_execute": True},
        }
    ]
    model_output = _readiness_output()
    with pytest.raises(ValueError, match="trace fields differ"):
        coordinator.record_handoff(
            offer=offer,
            handoff=build_external_execution_handoff(offer, minimal),
            cognitive_action_trace=minimal,
            readiness=build_external_execution_readiness(
                offer,
                model_output,
            ),
                model_output=model_output,
                action_policy_evidence=_action_policy(),
                executors=[item.value for item in _executors()],
                action_policy_receipt={},
                runtime_operation={},
            )


def test_replay_payload_is_value_redacted_and_byte_bounded(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-redacted-replay")
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatching = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="test-task",
    )
    completed = coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatching["dispatch_owner"]["attempt_id"],
        result={
            "ok": True,
            "status": "success_verified",
            "transport_succeeded": True,
            "effect_verified": True,
            "stdout": "Authorization: Bearer live-secret",
            "body": "password=live-secret",
            "nested": (
                "Authorization: Bearer live-secret",
                {"items": ("password=live-secret",)},
            ),
            "unordered": {
                "password=live-secret",
                "ordinary-output",
            },
            "output": {
                f"row-{index}": "x" * 2048
                for index in range(128)
            },
        },
    )
    transaction_file = next(coordinator.root.glob("*.json"))
    persisted = transaction_file.read_text(encoding="utf-8")
    assert "live-secret" not in persisted
    assert transaction_file.stat().st_size < 512_000
    assert completed["result"]["replay_payload"]["_truncated"] is True
    assert coordinator.lookup(
        action_id=offer["action_id"],
        request_digest=offer["request_digest"],
    )["state"] == "SUCCEEDED"


def test_recovered_dispatch_is_unknown_and_never_blindly_retried(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(
        tmp_path / "transactions",
        owner_alive=lambda _owner: False,
    )
    offer = _offer()
    coordinator.prepare(offer)
    trace = _execute_trace()
    _record_handoff(coordinator, offer, trace)
    coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="test-task",
    )
    transaction_path = _downgrade_transaction_envelope_to_v2(coordinator)

    recovered = coordinator.prepare(offer)
    assert recovered["state"] == "UNKNOWN_EFFECT"
    assert recovered["result"]["transport_succeeded"] is None
    assert recovered["result"]["manual_reconciliation_required"] is True
    assert "unknown_effect" in recovered["result"]["error"]
    assert recovered["schema"] == EXTERNAL_EXECUTE_TRANSACTION_SCHEMA
    assert json.loads(transaction_path.read_text(encoding="utf-8"))["schema"] == (
        EXTERNAL_EXECUTE_TRANSACTION_SCHEMA
    )


def test_v2_terminal_transaction_is_atomically_migrated_and_replayed(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-v2-terminal-replay")
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatch = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="test-task",
    )
    completed = coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": True,
            "status": "success_verified",
            "transport_succeeded": True,
            "effect_verified": True,
            "retry_safe": False,
        },
    )
    assert completed["state"] == "SUCCEEDED"
    transaction_path = _downgrade_transaction_envelope_to_v2(coordinator)

    replayed = coordinator.lookup(
        action_id=offer["action_id"],
        request_digest=offer["request_digest"],
    )

    assert replayed is not None
    assert replayed["state"] == "SUCCEEDED"
    assert replayed["result"] == completed["result"]
    assert replayed["schema"] == EXTERNAL_EXECUTE_TRANSACTION_SCHEMA
    persisted = json.loads(transaction_path.read_text(encoding="utf-8"))
    assert persisted["schema"] == EXTERNAL_EXECUTE_TRANSACTION_SCHEMA
    assert persisted["action_intervention"] == {}


def test_live_dispatch_owner_blocks_duplicates_and_owner_token_is_required(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(
        tmp_path / "transactions",
        owner_alive=lambda _owner: True,
    )
    offer = _offer(action_id="action-live-owner")
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatching = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="test-task",
    )

    with pytest.raises(ExternalExecutionInProgressError, match="live dispatcher"):
        coordinator.prepare(offer)
    with pytest.raises(ValueError, match="owner differs"):
        coordinator.complete(
            offer=offer,
            dispatch_attempt_id="wrong-owner-token",
            result={
                "ok": True,
                "status": "success_verified",
                "transport_succeeded": True,
                "effect_verified": True,
            },
        )

    completed = coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatching["dispatch_owner"]["attempt_id"],
        result={
            "ok": True,
            "status": "success_verified",
            "transport_succeeded": True,
            "effect_verified": True,
        },
    )
    assert completed["state"] == "SUCCEEDED"


def test_host_bypass_accepts_unavailability_but_rejects_integrity_failures(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    unavailable = _offer(action_id="action-unavailable")
    coordinator.prepare(unavailable)
    bypassed = coordinator.record_bypass(
        offer=unavailable,
        reason="availability_failure:generation_gate_busy",
    )
    assert bypassed["state"] == "DECIDED"
    assert bypassed["decision_source"] == "host_fallback"

    malformed = _offer(action_id="action-malformed")
    coordinator.prepare(malformed)
    with pytest.raises(ValueError, match="not eligible"):
        coordinator.record_bypass(
            offer=malformed,
            reason="external_execution_handoff_invalid:ValueError",
        )
    forged_absence = _offer(action_id="action-forged-absence")
    coordinator.prepare(forged_absence)
    with pytest.raises(ValueError, match="not eligible"):
        coordinator.record_bypass(
            offer=forged_absence,
            reason="latent_cortex_absent_integrity_failure",
        )

    # An integrity failure never bypasses, however it is worded. This is the
    # case the eligibility rule exists for: the rehearsal RAN and refused, and
    # calling that unavailability would let a verdict masquerade as an absence.
    for index, reason in enumerate(
        (
            "episode_integrity_hash_mismatch",
            "episode_integrity_failure:trace_truncated",
            "not_an_availability_failure:client_unavailable:OSError",
        )
    ):
        integrity_failure = _offer(action_id=f"action-integrity-{index}")
        coordinator.prepare(integrity_failure)
        with pytest.raises(ValueError, match="not eligible"):
            coordinator.record_bypass(offer=integrity_failure, reason=reason)


def test_an_unlisted_exception_class_still_counts_as_unavailability(
    tmp_path: Path,
) -> None:
    """Eligibility is by class, because an ineligible bypass refuses the action.

    preaction_cortex composes its reasons as
    `availability_failure:{type(exc).__name__}`, so while the rule was an exact
    list, any exception nobody had enumerated made the bypass ineligible and
    took the whole action down with it. Live 2026-08-18 a user-requested
    browser task died that way after clearing every authority gate before it.
    """
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    for index, reason in enumerate(
        (
            "availability_failure:client_unavailable:RuntimeError",
            "availability_failure:generation_lease_unavailable:RuntimeError",
            "availability_failure:SomeClassInventedNextYear",
        )
    ):
        offer = _offer(action_id=f"action-unlisted-{index}")
        coordinator.prepare(offer)
        bypassed = coordinator.record_bypass(offer=offer, reason=reason)
        assert bypassed["state"] == "DECIDED"
        assert bypassed["decision_source"] == "host_fallback"


def test_complete_trace_rejects_missing_prior_action_lineage(tmp_path: Path) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-missing-trace-lineage")
    coordinator.prepare(offer)
    trace = _execute_trace()
    state_payload = {
        **trace[0]["state_signal"],
        "step_index": 1,
        "previously_selected": [OperationKind.DECOMPOSE.value],
    }
    state = CognitiveStateSignal.from_dict(state_payload)
    decision = ValueOfComputationPolicy(_action_policy()).choose(
        state,
        executors=_executors(),
    )
    trace[0]["state_signal"] = state.to_dict()
    trace[0]["decision"] = decision
    trace[0]["transition"] = {
        **trace[0]["transition"],
        "step_index": 1,
        "decision_sha256": decision["decision_sha256"],
        "action": decision["action"],
        "mode": decision["mode"],
    }
    model_output = _readiness_output()
    policy_receipt = _policy_receipt(trace, list(_executors()))
    with pytest.raises(ValueError, match="step lineage"):
        coordinator.record_handoff(
            offer=offer,
            handoff=build_external_execution_handoff(offer, trace),
            cognitive_action_trace=trace,
            readiness=build_external_execution_readiness(
                offer,
                model_output,
            ),
            model_output=model_output,
            action_policy_evidence=_action_policy(),
            executors=[item.value for item in _executors()],
            action_policy_receipt=policy_receipt,
            runtime_operation=_runtime_operation_receipt(
                offer,
                trace,
                policy_receipt,
            ),
        )


def test_failed_execute_operation_cannot_authorize_host_dispatch(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-failed-execute")
    coordinator.prepare(offer)
    trace = _execute_trace()
    model_output = _readiness_output()
    executors = list(_executors())
    policy_receipt = _policy_receipt(trace, executors)
    runtime_operation = _runtime_operation_receipt(
        offer,
        trace,
        policy_receipt,
    )
    failed_execute = {
        **runtime_operation["action_operations"][0],
        "outcome": OperationOutcome.FAILED.value,
        "failure_code": "action_execution_failed",
    }
    runtime_operation["action_operations"][0] = failed_execute

    with pytest.raises(ValueError, match="action lineage"):
        coordinator.record_handoff(
            offer=offer,
            handoff=build_external_execution_handoff(offer, trace),
            cognitive_action_trace=trace,
            readiness=build_external_execution_readiness(offer, model_output),
            model_output=model_output,
            action_policy_evidence=_action_policy(),
            executors=[item.value for item in executors],
            action_policy_receipt=policy_receipt,
            runtime_operation=runtime_operation,
        )

    prepared = coordinator.lookup(
        action_id=offer["action_id"],
        request_digest=offer["request_digest"],
    )
    assert prepared is not None
    assert prepared["state"] == "PREPARED"


def test_runtime_operation_requires_reconstructable_final_state_and_journal(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-forged-final-state")
    coordinator.prepare(offer)
    trace = _execute_trace()
    model_output = _readiness_output()
    executors = list(_executors())
    policy_receipt = _policy_receipt(trace, executors)
    runtime_operation = _runtime_operation_receipt(
        offer,
        trace,
        policy_receipt,
    )
    admitted = runtime_operation["admitted_state"]
    forged = {
        **runtime_operation,
        "current_state": admitted,
        "current_state_sha256": admitted["state_sha256"],
        "current_state_version": admitted["version"],
        "journal": {
            **runtime_operation["journal"],
            "state_sha256": admitted["state_sha256"],
            "state_version": admitted["version"],
            "entry_count": admitted["version"] + 1,
        },
    }

    with pytest.raises(ValueError, match="completion state"):
        coordinator.record_handoff(
            offer=offer,
            handoff=build_external_execution_handoff(offer, trace),
            cognitive_action_trace=trace,
            readiness=build_external_execution_readiness(offer, model_output),
            model_output=model_output,
            action_policy_evidence=_action_policy(),
            executors=[item.value for item in executors],
            action_policy_receipt=policy_receipt,
            runtime_operation=forged,
        )

    forged_journal = {
        **runtime_operation,
        "journal": {
            **runtime_operation["journal"],
            "head_sha256": "0" * 64,
            "size_bytes": 1,
        },
    }
    with pytest.raises(ValueError, match="completion state"):
        coordinator.record_handoff(
            offer=offer,
            handoff=build_external_execution_handoff(offer, trace),
            cognitive_action_trace=trace,
            readiness=build_external_execution_readiness(offer, model_output),
            model_output=model_output,
            action_policy_evidence=_action_policy(),
            executors=[item.value for item in executors],
            action_policy_receipt=policy_receipt,
            runtime_operation=forged_journal,
        )

    forged_parent_sha256 = hashlib.sha256(
        b"forged-journal-parent"
    ).hexdigest()
    forged_entry = {
        "schema": "aura.rlc.epistemic_journal.v1",
        "sequence": runtime_operation["current_state"]["version"],
        "previous_entry_sha256": forged_parent_sha256,
        "state_sha256": runtime_operation["current_state"]["state_sha256"],
        "state": runtime_operation["current_state"],
    }
    self_consistent_forgery = {
        **runtime_operation,
        "journal": {
            **runtime_operation["journal"],
            "previous_head_sha256": forged_parent_sha256,
            "head_sha256": hashlib.sha256(
                json.dumps(
                    forged_entry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "size_bytes": runtime_operation["journal"]["size_bytes"] + 1,
        },
    }
    with pytest.raises(ValueError, match="completion state"):
        coordinator.record_handoff(
            offer=offer,
            handoff=build_external_execution_handoff(offer, trace),
            cognitive_action_trace=trace,
            readiness=build_external_execution_readiness(offer, model_output),
            model_output=model_output,
            action_policy_evidence=_action_policy(),
            executors=[item.value for item in executors],
            action_policy_receipt=policy_receipt,
            runtime_operation=self_consistent_forgery,
        )


def test_abandoned_dispatch_cache_is_bounded_and_expires(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module

    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    for index in range(3000):
        coordinator._mark_abandoned_attempt(f"{index:032x}")
    assert len(coordinator._abandoned_attempt_ids) <= 2048

    now = coordinator_module.time.monotonic()
    monkeypatch.setattr(
        coordinator_module.time,
        "monotonic",
        lambda: now + 901.0,
    )
    coordinator._prune_abandoned_attempts()
    assert coordinator._abandoned_attempt_ids == {}


def test_invalid_abandonment_cannot_poison_a_live_dispatch(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(
        tmp_path / "transactions",
        owner_alive=lambda _owner: True,
    )
    first = _offer(action_id="action-live-owner")
    second = _offer(action_id="action-other-owner")
    for offer in (first, second):
        coordinator.prepare(offer)
        _record_handoff(coordinator, offer, _execute_trace())
    dispatch = coordinator.begin_dispatch(
        first,
        authorization_receipt_id=first["will_receipt_id"],
        task_id="live-owner",
    )
    coordinator.begin_dispatch(
        second,
        authorization_receipt_id=second["will_receipt_id"],
        task_id="other-owner",
    )
    attempt_id = dispatch["dispatch_owner"]["attempt_id"]

    with pytest.raises(ValueError, match="owner differs"):
        coordinator.abandon_dispatch(
            offer=second,
            dispatch_attempt_id=attempt_id,
            effect_may_have_occurred=True,
            reason="wrong transaction",
        )
    with pytest.raises(ValueError, match="attempt id is invalid"):
        coordinator.abandon_dispatch(
            offer=first,
            dispatch_attempt_id="x" * 2_000_000,
            effect_may_have_occurred=True,
            reason="oversized token",
        )

    assert coordinator._abandoned_attempt_ids == {}
    with pytest.raises(ExternalExecutionInProgressError):
        coordinator.prepare(first)


def test_nested_mapping_secrets_are_redacted_from_durable_replay(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-userdict-secret")
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatch = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="redaction-test",
    )
    secret = "Authorization: Bearer live-secret"
    completed = coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": True,
            "status": "success_verified",
            "transport_succeeded": True,
            "effect_verified": True,
            "manual_reconciliation_required": False,
            "retry_safe": False,
            "verification_evidence": UserDict(
                {"nested": UserDict({"header": secret})}
            ),
        },
    )

    assert secret not in json.dumps(completed, sort_keys=True)
    transaction_path = next(coordinator.root.glob("*.json"))
    assert secret not in transaction_path.read_text(encoding="utf-8")


def test_post_action_link_requires_a_matching_durable_receipt(
    tmp_path: Path,
) -> None:
    from core.runtime.post_action_receipt import (
        PostActionReceipt,
        PostActionReceiptStore,
    )

    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-fabricated-receipt")
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatch = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="receipt-test",
    )
    coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": True,
            "status": "success_verified",
            "transport_succeeded": True,
            "effect_verified": True,
            "manual_reconciliation_required": False,
            "retry_safe": False,
        },
    )
    receipt_store = PostActionReceiptStore(tmp_path / "receipts.jsonl")
    expected_id = (
        "post-external-"
        + hashlib.sha256(
            f"{offer['action_id']}\0{offer['request_digest']}".encode()
        ).hexdigest()[:32]
    )
    receipt = PostActionReceipt(
        receipt_id=expected_id,
        will_receipt_id=offer["will_receipt_id"],
        executor_name=offer["action_name"],
        actual_outcome="success",
        output_hash="sha256:" + "a" * 64,
        error_status="",
        welfare_transaction_id="welfare-test",
        action_id=offer["action_id"],
        domain=offer["domain"],
        source="test",
        request_digest=offer["request_digest"],
        status="success_verified",
        effect_verified=True,
        transport_succeeded=True,
        retry_safe=False,
    )

    with pytest.raises(ValueError, match="durable store evidence"):
        coordinator.link_post_action_receipt(
            offer=offer,
            persisted_receipt=receipt.to_dict(),
            receipt_store=receipt_store,
        )

    transaction = coordinator.lookup(
        action_id=offer["action_id"],
        request_digest=offer["request_digest"],
    )
    assert transaction is not None
    assert transaction["result"]["post_action_receipt_id"] == ""


def test_post_action_link_rejects_receipt_that_contradicts_staged_outcome(
    tmp_path: Path,
) -> None:
    from core.runtime.post_action_receipt import (
        PostActionReceipt,
        PostActionReceiptStore,
    )

    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-contradictory-receipt")
    coordinator.prepare(offer)
    _record_handoff(coordinator, offer, _execute_trace())
    dispatch = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="contradiction-test",
    )
    coordinator.complete(
        offer=offer,
        dispatch_attempt_id=dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": False,
            "status": "failed_recoverable",
            "error": "effect unavailable",
            "transport_succeeded": False,
            "effect_verified": False,
            "manual_reconciliation_required": False,
            "retry_safe": True,
        },
    )
    receipt_id = (
        "post-external-"
        + hashlib.sha256(
            f"{offer['action_id']}\0{offer['request_digest']}".encode()
        ).hexdigest()[:32]
    )
    base = {
        "receipt_id": receipt_id,
        "will_receipt_id": offer["will_receipt_id"],
        "executor_name": offer["action_name"],
        "welfare_transaction_id": "welfare-test",
        "action_id": offer["action_id"],
        "domain": offer["domain"],
        "source": "test",
        "request_digest": offer["request_digest"],
    }
    failed_receipt = PostActionReceipt(
        **base,
        actual_outcome="failure",
        output_hash="sha256:" + "a" * 64,
        error_status="effect unavailable",
        status="failed_recoverable",
        effect_verified=False,
        transport_succeeded=False,
        retry_safe=True,
    )
    coordinator.stage_post_action_receipt(
        offer=offer,
        receipt_contract=failed_receipt.to_dict(),
    )

    contradictory = PostActionReceipt(
        **base,
        actual_outcome="success",
        output_hash="sha256:" + "b" * 64,
        error_status="",
        status="success_verified",
        effect_verified=True,
        transport_succeeded=True,
        retry_safe=False,
        timestamp=failed_receipt.timestamp,
    )
    receipt_store = PostActionReceiptStore(tmp_path / "receipts.jsonl")
    receipt_store.record(contradictory)

    with pytest.raises(ValueError, match="recovery contract"):
        coordinator.link_post_action_receipt(
            offer=offer,
            persisted_receipt=contradictory.to_dict(),
            receipt_store=receipt_store,
        )


def test_transaction_file_tampering_is_rejected(tmp_path: Path) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer()
    prepared = coordinator.prepare(offer)
    transaction_files = [
        path
        for path in coordinator.root.glob("*.json")
        if path.name != ".transactions.lock"
    ]
    assert len(transaction_files) == 1
    payload = json.loads(transaction_files[0].read_text(encoding="utf-8"))
    payload["state"] = "SUCCEEDED"
    transaction_files[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        coordinator.lookup(
            action_id=prepared["action_id"],
            request_digest=prepared["request_digest"],
        )


def test_hash_valid_but_impossible_transaction_state_is_rejected(
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module

    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer()
    prepared = coordinator.prepare(offer)
    transaction_file = next(coordinator.root.glob("*.json"))
    payload = json.loads(transaction_file.read_text(encoding="utf-8"))
    payload.pop("transaction_sha256")
    payload["state"] = "SUCCEEDED"
    payload["transaction_sha256"] = coordinator_module._canonical_sha256(payload)
    transaction_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="owner|result"):
        coordinator.lookup(
            action_id=prepared["action_id"],
            request_digest=prepared["request_digest"],
        )


class _FakeWill:
    def __init__(self) -> None:
        self.calls = 0
        self.outcomes: list[tuple[str, Any]] = []

    def decide(self, **kwargs: Any) -> Any:
        from core.governance.will import WillDecision, WillOutcome

        self.calls += 1
        return WillDecision(
            receipt_id=f"will-external-{self.calls}",
            outcome=WillOutcome.PROCEED,
            domain=kwargs["domain"],
            reason="test_approved",
            source="test",
        )

    def record_outcome(self, _receipt_id: str, _outcome: Any) -> None:
        self.outcomes.append((_receipt_id, _outcome))


class _ExecutingLatentService:
    def __init__(
        self,
        *,
        execute: bool,
        tamper: bool = False,
        ready: bool | None = None,
    ) -> None:
        self.execute = execute
        self.tamper = tamper
        self.ready = execute if ready is None else ready

    async def deep_reason(self, objective: str, **kwargs: Any) -> dict[str, Any]:
        offer = kwargs["external_execution_offer"]
        trace = _execute_trace() if self.execute else _non_execute_trace()
        policy_receipt = _policy_receipt(trace, list(_executors()))
        handoff = build_external_execution_handoff(offer, trace)
        if self.tamper:
            handoff = {**handoff, "requested": not handoff["requested"]}
        return {
            "ok": True,
            "text": _readiness_output(ready=self.ready),
            "receipt": {
                "episode_id": "episode-external",
                "steps_taken": 3,
                "honest_flags": [],
                "cognitive_action_trace": trace,
                "external_execution_handoff": handoff,
                "host_action_policy_evidence": _action_policy(),
                "value_of_computation": policy_receipt,
                "epistemic_operation": _runtime_operation_receipt(
                    offer,
                    trace,
                    policy_receipt,
                    objective=objective,
                ),
            },
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("execute", [True, False])
async def test_action_executor_dispatches_only_after_rlc_handoff_and_replays_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    execute: bool,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    coordinator = ExternalExecuteCoordinator(tmp_path / "external")
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=execute),
    )
    will = _FakeWill()
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: PostActionReceiptStore(tmp_path / "post_action.jsonl"),
    )
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True, "observed": "done"}

    def verify(context: dict[str, Any]) -> dict[str, Any]:
        return {
            "effect_verified": context["result"].get("observed") == "done",
            "observation": {"observed": context["result"].get("observed")},
        }

    kwargs = {
        "domain": "external_action",
        "action_name": "write_note",
        "params": {"note": "hello"},
        "source": "rlc_test",
        "action_id": f"rlc-action-{execute}",
        "effect_handler": effect,
        "effect_verifier": verify,
    }
    first = await executor.ActionExecutor.execute(**kwargs)
    second = await executor.ActionExecutor.execute(**kwargs)

    if execute:
        assert first["ok"] is True
        assert first["effect_verified"] is True
        assert first["external_execution_transaction"]["state"] == "SUCCEEDED"
        assert second["external_execution_replayed"] is True
        assert second["external_execution_transaction"]["state"] == "SUCCEEDED"
        assert second["observed"] == "done"
        assert effects == ["fired"]
    else:
        assert first["status"] == "blocked_by_policy"
        assert first["external_execution_transaction"]["state"] == "ABSTAINED"
        assert first["receipt_persisted"] is True
        assert second["external_execution_replayed"] is True
        assert effects == []
    assert will.calls == 1
    assert len(will.outcomes) == 1


@pytest.mark.asyncio
async def test_malformed_worker_handoff_fails_closed_without_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor

    monkeypatch.setattr(
        coordinator_module,
        "_COORDINATOR",
        ExternalExecuteCoordinator(tmp_path / "external"),
    )
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True, tamper=True),
    )
    monkeypatch.setattr(executor, "get_will", lambda: _FakeWill())
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True}

    result = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="tampered_action",
        params={},
        source="rlc_test",
        action_id="rlc-action-tampered",
        effect_handler=effect,
        effect_verifier=lambda _context: {
            "effect_verified": True,
            "observation": {"done": True},
        },
    )
    assert result["ok"] is False
    assert result["status"] == "failed_recoverable"
    assert result["retry_safe"] is False
    assert effects == []


@pytest.mark.asyncio
async def test_execute_request_without_model_readiness_fails_before_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    will = _FakeWill()
    store = PostActionReceiptStore(tmp_path / "post_action.jsonl")
    monkeypatch.setattr(
        coordinator_module,
        "_COORDINATOR",
        ExternalExecuteCoordinator(tmp_path / "external"),
    )
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True, ready=False),
    )
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: store,
    )
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True}

    kwargs = {
        "domain": "external_action",
        "action_name": "missing_precondition",
        "params": {},
        "source": "rlc_test",
        "action_id": "rlc-action-not-ready",
        "effect_handler": effect,
        "effect_verifier": lambda _context: {
            "effect_verified": True,
            "observation": {"done": True},
        },
    }
    result = await executor.ActionExecutor.execute(
        **kwargs,
    )
    replay = await executor.ActionExecutor.execute(**kwargs)
    assert result["ok"] is False
    assert result["retry_safe"] is False
    assert "preparation_failed" in result["error"]
    assert effects == []
    assert will.calls == 1
    assert len(will.outcomes) == 1
    assert result["welfare_transaction_completed"] is True
    assert result["receipt_persisted"] is True
    assert store.get_receipt(result["post_action_receipt_id"]) is not None
    assert result["external_execution_transaction"]["state"] == "FAILED_PRE_DISPATCH"
    assert replay["external_execution_replayed"] is True
    assert replay["will_receipt_id"] == result["will_receipt_id"]
    assert will.calls == 1


@pytest.mark.asyncio
async def test_absent_cortex_still_uses_durable_exactly_once_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    coordinator = ExternalExecuteCoordinator(tmp_path / "external")
    will = _FakeWill()
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(preaction, "_latent_service", lambda: None)
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: PostActionReceiptStore(tmp_path / "post_action.jsonl"),
    )
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True, "observed": "done"}

    kwargs = {
        "domain": "external_action",
        "action_name": "cortex_absent_effect",
        "params": {},
        "source": "rlc_test",
        "action_id": "rlc-action-cortex-absent",
        "effect_handler": effect,
        "effect_verifier": lambda context: {
            "effect_verified": context["result"].get("observed") == "done",
            "observation": {"done": True},
        },
    }
    first = await executor.ActionExecutor.execute(**kwargs)
    second = await executor.ActionExecutor.execute(**kwargs)
    assert first["external_execution_transaction"]["state"] == "SUCCEEDED"
    assert second["external_execution_replayed"] is True
    assert effects == ["fired"]
    assert will.calls == 1


@pytest.mark.asyncio
async def test_external_transaction_lookup_failure_prevents_duplicate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.runtime.action_executor as executor

    class BrokenCoordinator:
        def lookup(self, **_kwargs: Any) -> None:
            raise OSError("transaction store unavailable")

    will = _FakeWill()
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", BrokenCoordinator())
    monkeypatch.setattr(executor, "get_will", lambda: will)
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True}

    result = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="lookup_failure",
        params={},
        source="rlc_test",
        action_id="rlc-action-lookup-failure",
        effect_handler=effect,
        effect_verifier=lambda _context: {"effect_verified": True},
    )
    assert result["ok"] is False
    assert result["error"] == "external_execution_preflight_failed:OSError"
    assert will.calls == 0
    assert effects == []


@pytest.mark.asyncio
async def test_cancellation_after_dispatch_intent_becomes_unknown_not_wedged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor

    coordinator = ExternalExecuteCoordinator(tmp_path / "external")
    will = _FakeWill()
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True),
    )
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    entered = asyncio.Event()

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        await asyncio.Event().wait()
        return {"ok": True}

    kwargs = {
        "domain": "external_action",
        "action_name": "cancelled_effect",
        "params": {},
        "source": "rlc_test",
        "action_id": "rlc-action-cancelled",
        "effect_handler": effect,
        "effect_verifier": lambda _context: {
            "effect_verified": True,
            "observation": {"done": True},
        },
    }
    task = asyncio.create_task(executor.ActionExecutor.execute(**kwargs))
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    transaction = json.loads(
        next(coordinator.root.glob("*.json")).read_text(encoding="utf-8")
    )
    assert transaction["state"] == "UNKNOWN_EFFECT"
    replay = await executor.ActionExecutor.execute(**kwargs)
    assert replay["external_execution_replayed"] is True
    assert replay["manual_reconciliation_required"] is True
    assert will.calls == 1


@pytest.mark.asyncio
async def test_cancellation_during_durable_completion_stops_lease_and_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor

    entered = threading.Event()
    release = threading.Event()

    class BlockingCompleteCoordinator(ExternalExecuteCoordinator):
        def complete(self, **kwargs: Any) -> dict[str, Any]:
            entered.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("test completion release timed out")
            return super().complete(**kwargs)

    coordinator = BlockingCompleteCoordinator(tmp_path / "external")
    will = _FakeWill()
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True),
    )
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True, "observed": "done"}

    kwargs = {
        "domain": "external_action",
        "action_name": "cancelled_completion",
        "params": {},
        "source": "rlc_test",
        "action_id": "rlc-action-cancelled-completion",
        "effect_handler": effect,
        "effect_verifier": lambda context: {
            "effect_verified": context["result"].get("observed") == "done",
            "observation": {"done": True},
        },
    }
    task = asyncio.create_task(executor.ActionExecutor.execute(**kwargs))
    assert await asyncio.to_thread(entered.wait, 2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    replay = await executor.ActionExecutor.execute(**kwargs)
    transaction = replay["external_execution_transaction"]
    assert transaction["state"] == "UNKNOWN_EFFECT"
    assert replay["external_execution_replayed"] is True
    assert replay["manual_reconciliation_required"] is True
    assert effects == ["fired"]
    assert will.calls == 1


def test_post_dispatch_failure_is_terminal_only_when_no_effect_is_proven(
    tmp_path: Path,
) -> None:
    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    uncertain = _offer(action_id="action-uncertain")
    coordinator.prepare(uncertain)
    trace = _execute_trace()
    _record_handoff(coordinator, uncertain, trace)
    uncertain_dispatch = coordinator.begin_dispatch(
        uncertain,
        authorization_receipt_id=uncertain["will_receipt_id"],
        task_id="test-task",
    )
    uncertain_result = coordinator.complete(
        offer=uncertain,
        dispatch_attempt_id=uncertain_dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": False,
            "status": "failed_recoverable",
            "transport_succeeded": False,
            "effect_verified": False,
            "retry_safe": False,
        },
    )
    assert uncertain_result["state"] == "UNKNOWN_EFFECT"

    proven = _offer(action_id="action-proven-no-effect")
    coordinator.prepare(proven)
    _record_handoff(coordinator, proven, trace)
    proven_dispatch = coordinator.begin_dispatch(
        proven,
        authorization_receipt_id=proven["will_receipt_id"],
        task_id="test-task",
    )
    failed = coordinator.complete(
        offer=proven,
        dispatch_attempt_id=proven_dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": False,
            "status": "failed_recoverable",
            "transport_succeeded": False,
            "effect_verified": False,
            "retry_safe": True,
        },
    )
    assert failed["state"] == "FAILED"

    verified = _offer(action_id="action-verified-receipt-failed")
    coordinator.prepare(verified)
    _record_handoff(coordinator, verified, trace)
    verified_dispatch = coordinator.begin_dispatch(
        verified,
        authorization_receipt_id=verified["will_receipt_id"],
        task_id="test-task",
    )
    completed = coordinator.complete(
        offer=verified,
        dispatch_attempt_id=verified_dispatch["dispatch_owner"]["attempt_id"],
        result={
            "ok": False,
            "status": "partial_success",
            "transport_succeeded": True,
            "effect_verified": True,
            "retry_safe": False,
            "error": "post_action_receipt_persistence_failed",
        },
    )
    assert completed["state"] == "SUCCEEDED"
    assert completed["result"]["ok"] is False
    assert completed["result"]["effect_verified"] is True


def test_expired_dispatch_lease_forces_reconciliation_while_process_is_alive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module

    coordinator = ExternalExecuteCoordinator(tmp_path / "transactions")
    offer = _offer(action_id="action-expired-lease")
    coordinator.prepare(offer)
    trace = _execute_trace()
    _record_handoff(coordinator, offer, trace)
    dispatch = coordinator.begin_dispatch(
        offer,
        authorization_receipt_id=offer["will_receipt_id"],
        task_id="still-live-task",
    )
    owner = dispatch["dispatch_owner"]
    monkeypatch.setattr(
        coordinator_module.time,
        "time",
        lambda: float(owner["lease_renewed_at"])
        + float(owner["lease_duration_s"])
        + 1.0,
    )

    recovered = coordinator.prepare(offer)

    assert recovered is not None
    assert recovered["state"] == "UNKNOWN_EFFECT"
    assert recovered["result"]["manual_reconciliation_required"] is True
    assert recovered["result"]["error"] == "unknown_effect_requires_reconciliation"


@pytest.mark.asyncio
async def test_verifier_exception_after_transport_is_not_retry_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    monkeypatch.setattr(executor, "get_will", lambda: _FakeWill())
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: PostActionReceiptStore(tmp_path / "post_action.jsonl"),
    )

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    async def failed_observer(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("observer crashed")

    monkeypatch.setattr(executor, "observe_action_effect", failed_observer)
    result = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="one_way_effect",
        params={},
        source="observer_test",
        effect_handler=effect,
        effect_verifier=lambda _context: {
            "effect_verified": True,
            "observation": {"done": True},
        },
    )
    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is False
    assert result["manual_reconciliation_required"] is True
    assert result["retry_safe"] is False
    assert result["status"] == "partial_success"


@pytest.mark.asyncio
async def test_external_closure_failure_is_reflected_in_durable_post_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    class FailingCompleteCoordinator(ExternalExecuteCoordinator):
        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            raise OSError("durable transaction store failed")

    coordinator = FailingCompleteCoordinator(tmp_path / "external")
    store = PostActionReceiptStore(tmp_path / "post_action.jsonl")
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True),
    )
    monkeypatch.setattr(executor, "get_will", lambda: _FakeWill())
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: store,
    )

    result = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="closure_failure",
        params={},
        source="rlc_test",
        action_id="rlc-action-closure-failure",
        effect_handler=lambda _context: {"ok": True, "observed": "done"},
        effect_verifier=lambda context: {
            "effect_verified": context["result"].get("observed") == "done",
            "observation": {"observed": context["result"].get("observed")},
        },
    )

    receipt = store.get_receipt(result["post_action_receipt_id"])
    assert receipt is not None
    assert result["status"] == "partial_success"
    assert receipt.status == result["status"]
    assert receipt.error_status == result["error"]
    assert receipt.output_hash == result["post_action_output_hash"]
    assert result["effect_verified"] is True
    assert result["transport_succeeded"] is True


@pytest.mark.asyncio
async def test_missing_post_receipt_self_heals_without_repeating_effect_or_will(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    class FailFirstStore(PostActionReceiptStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.failures_remaining = 1

        async def record_async(self, receipt: Any) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise OSError("receipt store temporarily unavailable")
            await super().record_async(receipt)

    coordinator = ExternalExecuteCoordinator(tmp_path / "external")
    store = FailFirstStore(tmp_path / "post_action.jsonl")
    will = _FakeWill()
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True),
    )
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: store,
    )
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True, "observed": "done"}

    kwargs = {
        "domain": "external_action",
        "action_name": "self_healed_receipt",
        "params": {},
        "source": "rlc_test",
        "action_id": "rlc-action-self-heal",
        "effect_handler": effect,
        "effect_verifier": lambda context: {
            "effect_verified": context["result"].get("observed") == "done",
            "observation": {"done": True},
        },
    }
    first = await executor.ActionExecutor.execute(**kwargs)
    assert first["receipt_persisted"] is False
    assert first["post_action_receipt_pending"] is True

    second = await executor.ActionExecutor.execute(**kwargs)
    assert second["external_execution_replayed"] is True
    assert second["receipt_persisted"] is True
    assert second["post_action_receipt_pending"] is False
    assert store.get_receipt(second["post_action_receipt_id"]) is not None
    assert effects == ["fired"]
    assert will.calls == 1


@pytest.mark.asyncio
async def test_completion_and_receipt_store_failures_preserve_self_healing_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.brain.external_execute_coordinator as coordinator_module
    import core.brain.preaction_cortex as preaction
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    class FailFirstCompleteCoordinator(ExternalExecuteCoordinator):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failures_remaining = 1

        def complete(self, **kwargs: Any) -> dict[str, Any]:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise OSError("transaction completion temporarily unavailable")
            return super().complete(**kwargs)

    class FailFirstReceiptStore(PostActionReceiptStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.failures_remaining = 1

        async def record_async(self, receipt: Any) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise OSError("receipt store temporarily unavailable")
            await super().record_async(receipt)

    coordinator = FailFirstCompleteCoordinator(tmp_path / "external")
    store = FailFirstReceiptStore(tmp_path / "post_action.jsonl")
    will = _FakeWill()
    monkeypatch.setattr(coordinator_module, "_COORDINATOR", coordinator)
    monkeypatch.setattr(
        preaction,
        "_latent_service",
        lambda: _ExecutingLatentService(execute=True),
    )
    monkeypatch.setattr(executor, "get_will", lambda: will)
    monkeypatch.setattr(
        executor.BodyStateService,
        "get",
        classmethod(lambda _cls: SimpleNamespace(snapshot=lambda: None)),
    )
    monkeypatch.setattr(
        executor.WelfareState,
        "get",
        classmethod(lambda _cls: SimpleNamespace(last_outputs=None)),
    )
    monkeypatch.setattr(
        executor,
        "get_post_action_receipt_store",
        lambda: store,
    )
    effects: list[str] = []

    async def effect(_context: dict[str, Any]) -> dict[str, Any]:
        effects.append("fired")
        return {"ok": True, "observed": "done"}

    kwargs = {
        "domain": "external_action",
        "action_name": "compound_receipt_recovery",
        "params": {},
        "source": "rlc_test",
        "action_id": "rlc-action-compound-recovery",
        "effect_handler": effect,
        "effect_verifier": lambda context: {
            "effect_verified": context["result"].get("observed") == "done",
            "observation": {"done": True},
        },
    }
    first = await executor.ActionExecutor.execute(**kwargs)
    assert first["receipt_persisted"] is False
    assert first["post_action_receipt_pending"] is True

    second = await executor.ActionExecutor.execute(**kwargs)
    assert second["external_execution_replayed"] is True
    assert second["receipt_persisted"] is True
    assert store.get_receipt(second["post_action_receipt_id"]) is not None
    assert effects == ["fired"]
    assert will.calls == 1
