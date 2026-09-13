"""Bounded, provenance-preserving neural communication between RLC branches.

The exchange channel is a mailbox, not another vote.  Each sender contributes
one fixed-width latent summary derived only from its private reasoning slots.
The communication slot and immutable organ/context slots are excluded.  The
first exchange is bound to independently sealed candidates; later exchanges
are explicitly cooperative and can never be counted as additional independent
support.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from core.brain.llm.latent_cortex.cognitive_operators import operator_for_role

BRANCH_EXCHANGE_SCHEMA = "aura.rlc.branch_exchange.v1"
BRANCH_EXCHANGE_TRACE_SCHEMA = "aura.rlc.branch_exchange_trace.v1"
MAX_EXCHANGE_SOURCE_SLOTS = 16
_SYNC_KINDS = {"interval", "schedule_bytecode", "controller_compare", "test"}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def private_exchange_slots(
    *,
    n_slots: int,
    comm_slot: int,
    context_slots: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    """Return the bounded private slot set allowed to enter a peer message."""

    if type(n_slots) is not int or not 2 <= n_slots <= 128:
        raise ValueError("exchange workspace cardinality is invalid")
    if type(comm_slot) is not int or not 0 <= comm_slot < n_slots:
        raise ValueError("exchange communication slot is invalid")
    protected = set()
    for index in context_slots:
        if type(index) is not int or not 0 <= index < n_slots:
            raise ValueError("exchange context slot is invalid")
        protected.add(index)
    protected.add(comm_slot)
    slots = tuple(index for index in range(n_slots) if index not in protected)
    if not slots:
        raise ValueError("exchange has no private reasoning slot")
    return slots[:MAX_EXCHANGE_SOURCE_SLOTS]


def eligible_exchange_steps(
    *,
    n_branches: int,
    max_steps: int,
    isolation_steps: int,
    exchange_interval: int,
) -> tuple[int, ...]:
    """Which recurrent steps of an episode can carry an interval exchange.

    Read off the ensemble's own three guards rather than off the spec.
    ``exchange`` returns without reading anything below two live branches. It
    returns again while isolation is unsealed, and isolation seals only once
    every branch has taken ``isolation_steps``. The interval trigger then fires
    on a step divisible by ``exchange_interval``.

    ``max_steps`` is a ceiling: convergence can halt an episode earlier, so a
    step named here is a step an exchange CAN reach, and an empty answer means
    no exchange is reachable even at full depth.
    """

    if any(
        type(value) is not int
        for value in (n_branches, max_steps, isolation_steps, exchange_interval)
    ):
        raise ValueError("exchange schedule arguments must be integers")
    if n_branches < 2 or exchange_interval < 1 or max_steps < 1 or isolation_steps < 1:
        return ()
    return tuple(
        step
        for step in range(1, max_steps + 1)
        if step % exchange_interval == 0 and step >= isolation_steps
    )


def exchange_steps_a_return_can_travel(
    *,
    n_branches: int,
    max_steps: int,
    isolation_steps: int,
    exchange_interval: int,
) -> tuple[int, ...]:
    """Eligible steps with at least one recurrent step left after them.

    The consensus is written into the communication slot, which sits ahead of
    the immutable context prefix and every private slot, so it is the one place
    in the workspace where influence moves backwards through position. An
    exchange on the FINAL step still reaches the answer -- the mailbox persists
    into the K/V the decode attends to -- and reaches no further thinking,
    because no window pass remains to carry it forward. A schedule that only
    ever lands there buys a second trajectory and integrates it nowhere.
    """

    return tuple(
        step
        for step in eligible_exchange_steps(
            n_branches=n_branches,
            max_steps=max_steps,
            isolation_steps=isolation_steps,
            exchange_interval=exchange_interval,
        )
        if step < max_steps
    )


def candidate_set_sha256(branch_isolation: Any) -> str:
    """Commit to the exact sealed candidates that existed before exposure."""

    if not isinstance(branch_isolation, dict):
        raise ValueError("branch isolation evidence is missing")
    candidates = branch_isolation.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("branch candidate evidence is missing")
    rows: list[dict[str, Any]] = []
    for expected_index, row in enumerate(candidates):
        if (
            not isinstance(row, dict)
            or row.get("index") != expected_index
            or not isinstance(row.get("role"), str)
            or not row["role"]
            or not is_sha256(row.get("candidate_sha256"))
            or type(row.get("candidate_step")) is not int
            or row["candidate_step"] < 1
        ):
            raise ValueError("branch candidate evidence is invalid")
        rows.append(
            {
                "index": expected_index,
                "role": row["role"],
                "candidate_sha256": row["candidate_sha256"],
                "candidate_step": row["candidate_step"],
            }
        )
    return canonical_sha256(rows)


def _finite_unit(value: Any, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (0.0 < float(value) <= 1.0 if positive else 0.0 <= float(value) <= 1.0)
    )


def _candidate_commitment_from_rows(rows: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "index": row["branch_index"],
                "role": row["role"],
                "candidate_sha256": row["candidate_sha256"],
                "candidate_step": row["candidate_step"],
            }
            for row in rows
        ]
    )


def validate_branch_exchange_receipt(
    value: Any,
    *,
    n_branches: int,
    n_slots: int,
    comm_slot: int,
    exchange_gamma: float,
    branch_isolation: Any,
    cognitive_slots: Any,
    expected_ordinal: int,
) -> dict[str, Any]:
    """Strictly reconstruct one hidden-state exchange receipt."""

    required = {
        "schema",
        "ordinal",
        "sync_kind",
        "sync_id",
        "generation",
        "n_branches",
        "n_slots",
        "comm_slot",
        "exchange_gamma",
        "source_policy",
        "message_representation",
        "message_slot_count",
        "hidden_dimension",
        "source_slot_limit",
        "context_slots_excluded",
        "comm_slot_excluded",
        "first_answer_text_exposed",
        "prior_peer_context_possible",
        "counts_as_independent_support",
        "candidate_set_sha256",
        "source_rows",
        "consensus_sha256",
        "recipient_rows",
        "tensor_accounting",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("branch exchange receipt schema is invalid")
    if value.get("schema") != BRANCH_EXCHANGE_SCHEMA:
        raise ValueError("branch exchange receipt version is invalid")
    if (
        type(n_branches) is not int
        or not 2 <= n_branches <= 8
        or value.get("n_branches") != n_branches
        or value.get("n_slots") != n_slots
        or value.get("comm_slot") != comm_slot
        or value.get("ordinal") != expected_ordinal
    ):
        raise ValueError("branch exchange topology differs")
    if (
        value.get("sync_kind") not in _SYNC_KINDS
        or not isinstance(value.get("sync_id"), str)
        or not 1 <= len(value["sync_id"]) <= 160
    ):
        raise ValueError("branch exchange synchronization point is invalid")
    if (
        not _finite_unit(exchange_gamma, positive=True)
        or not _finite_unit(value.get("exchange_gamma"), positive=True)
        or not math.isclose(
            float(value["exchange_gamma"]),
            float(exchange_gamma),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("branch exchange strength differs")

    context_indices: list[int] = []
    if not isinstance(cognitive_slots, list):
        raise ValueError("cognitive slot evidence must be a list")
    for row in cognitive_slots:
        if not isinstance(row, dict) or type(row.get("slot")) is not int:
            raise ValueError("cognitive slot evidence is invalid")
        context_indices.append(row["slot"])
    context_indices = sorted(set(context_indices))
    source_slots = list(
        private_exchange_slots(
            n_slots=n_slots,
            comm_slot=comm_slot,
            context_slots=context_indices,
        )
    )
    if (
        value.get("source_policy")
        != "bounded_private_reasoning_mean_excluding_mailbox_and_context_v1"
        or value.get("message_representation") != "latent_tensor_only"
        or value.get("message_slot_count") != 1
        or value.get("source_slot_limit") != MAX_EXCHANGE_SOURCE_SLOTS
        or value.get("context_slots_excluded") != context_indices
        or value.get("comm_slot_excluded") is not True
        or value.get("first_answer_text_exposed") is not False
    ):
        raise ValueError("branch exchange information policy is invalid")
    hidden_dimension = value.get("hidden_dimension")
    if type(hidden_dimension) is not int or not 1 <= hidden_dimension <= 1_000_000:
        raise ValueError("branch exchange hidden dimension is invalid")

    first = expected_ordinal == 0
    role_lesion = bool(
        isinstance(branch_isolation, dict)
        and branch_isolation.get("configured_role_lesion") is True
    )
    expected_generation = (
        "lesioned_candidates"
        if first and role_lesion
        else "independent_candidates"
        if first
        else "cooperative_refinement"
    )
    expected_independent = first and not role_lesion
    expected_prior_peer_context = not first
    if (
        value.get("generation") != expected_generation
        or value.get("prior_peer_context_possible")
        is not expected_prior_peer_context
        or value.get("counts_as_independent_support") is not expected_independent
    ):
        raise ValueError("branch exchange generation semantics are invalid")

    isolation_candidates = branch_isolation.get("candidates") if isinstance(
        branch_isolation, dict
    ) else None
    rows = value.get("source_rows")
    if (
        not isinstance(isolation_candidates, list)
        or len(isolation_candidates) != n_branches
        or not isinstance(rows, list)
        or len(rows) != n_branches
    ):
        raise ValueError("branch exchange source coverage is invalid")
    source_required = {
        "branch_index",
        "role",
        "operator",
        "step",
        "candidate_sha256",
        "candidate_step",
        "source_slots",
        "excluded_slots",
        "state_sha256",
        "private_state_sha256",
        "message_sha256",
        "support_weight",
        "consensus_weight",
    }
    excluded = sorted(set(context_indices + [comm_slot]))
    weight_sum = 0.0
    for index, row in enumerate(rows):
        candidate = isolation_candidates[index]
        if not isinstance(row, dict) or set(row) != source_required:
            raise ValueError("branch exchange source row schema is invalid")
        role = row.get("role")
        try:
            expected_operator = operator_for_role(role).value
        except (TypeError, ValueError) as exc:
            raise ValueError("branch exchange role is invalid") from exc
        if (
            row.get("branch_index") != index
            or not isinstance(candidate, dict)
            or candidate.get("index") != index
            or role != candidate.get("role")
            or row.get("operator") != expected_operator
            or type(row.get("step")) is not int
            or row["step"] < candidate.get("candidate_step", 1)
            or row.get("candidate_sha256") != candidate.get("candidate_sha256")
            or row.get("candidate_step") != candidate.get("candidate_step")
            or row.get("source_slots") != source_slots
            or row.get("excluded_slots") != excluded
            or not all(
                is_sha256(row.get(key))
                for key in (
                    "state_sha256",
                    "private_state_sha256",
                    "message_sha256",
                )
            )
            or not _finite_unit(row.get("support_weight"), positive=True)
            or not _finite_unit(row.get("consensus_weight"), positive=True)
        ):
            raise ValueError("branch exchange source provenance is invalid")
        weight_sum += float(row["consensus_weight"])
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("branch exchange consensus weights are not normalized")
    expected_candidates = candidate_set_sha256(branch_isolation)
    if (
        value.get("candidate_set_sha256") != expected_candidates
        or _candidate_commitment_from_rows(rows) != expected_candidates
        or not is_sha256(value.get("consensus_sha256"))
    ):
        raise ValueError("branch exchange candidate or consensus binding differs")

    recipients = value.get("recipient_rows")
    recipient_required = {
        "branch_index",
        "comm_pre_sha256",
        "comm_post_sha256",
        "non_comm_pre_sha256",
        "non_comm_post_sha256",
        "state_pre_sha256",
        "state_post_sha256",
        "causal",
    }
    if not isinstance(recipients, list) or len(recipients) != n_branches:
        raise ValueError("branch exchange recipient coverage is invalid")
    causal_writes = 0
    for index, row in enumerate(recipients):
        causal = row.get("causal")
        if (
            not isinstance(row, dict)
            or set(row) != recipient_required
            or row.get("branch_index") != index
            or not all(
                is_sha256(row.get(key))
                for key in recipient_required - {"branch_index", "causal"}
            )
            or row["non_comm_pre_sha256"] != row["non_comm_post_sha256"]
            or type(causal) is not bool
            or causal
            is not (
                row["comm_pre_sha256"] != row["comm_post_sha256"]
                and row["state_pre_sha256"] != row["state_post_sha256"]
            )
        ):
            raise ValueError("branch exchange causal write is invalid")
        causal_writes += int(causal)
    if causal_writes < 1:
        raise ValueError("branch exchange did not alter any recipient")

    accounting = value.get("tensor_accounting")
    expected_accounting = {
        "source_elements_read": n_branches
        * len(source_slots)
        * hidden_dimension,
        "message_elements_emitted": n_branches * hidden_dimension,
        "consensus_elements_written": n_branches * hidden_dimension,
        "tensor_scalar_ops": (
            n_branches * hidden_dimension * (len(source_slots) + 12)
            + 9 * n_branches
        ),
        "hidden_layer_apps": 0,
    }
    if accounting != expected_accounting:
        raise ValueError("branch exchange tensor accounting differs")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != canonical_sha256(payload):
        raise ValueError("branch exchange receipt digest differs")
    return dict(value)


def build_branch_exchange_trace(
    *,
    exchanges: list[dict[str, Any]],
    n_branches: int,
    n_slots: int,
    comm_slot: int,
    exchange_gamma: float,
    branch_isolation: Any,
    cognitive_slots: Any,
    exchange_interval: int,
    schedule_hash: str,
    bytecode_events: Any,
    cognitive_action_trace: Any,
) -> dict[str, Any]:
    validated = [
        validate_branch_exchange_receipt(
            row,
            n_branches=n_branches,
            n_slots=n_slots,
            comm_slot=comm_slot,
            exchange_gamma=exchange_gamma,
            branch_isolation=branch_isolation,
            cognitive_slots=cognitive_slots,
            expected_ordinal=index,
        )
        for index, row in enumerate(exchanges)
    ]
    sync_points = [f"{row['sync_kind']}:{row['sync_id']}" for row in validated]
    if len(sync_points) != len(set(sync_points)):
        raise ValueError("branch exchange synchronization points repeat")
    if type(exchange_interval) is not int or exchange_interval < 1:
        raise ValueError("branch exchange interval is invalid")
    if not isinstance(schedule_hash, str):
        raise ValueError("branch exchange schedule identity is invalid")
    if not isinstance(bytecode_events, list) or not isinstance(
        cognitive_action_trace, list
    ):
        raise ValueError("branch exchange declaration traces are invalid")
    bytecode_syncs = {
        f"schedule:{schedule_hash}:op:{event['op']}"
        for event in bytecode_events
        if isinstance(event, dict)
        and event.get("kind") == "exchange"
        and event.get("done") is True
        and type(event.get("op")) is int
        and event["op"] >= 0
    }
    controller_syncs = {
        f"controller-action:{index}"
        for index, action in enumerate(cognitive_action_trace)
        if isinstance(action, dict)
        and isinstance(action.get("transition"), dict)
        and action["transition"].get("action") == "compare"
        and action["transition"].get("outcome") == "branches_compared"
    }
    observed_bytecode: set[str] = set()
    observed_controller: set[str] = set()
    for row in validated:
        if row["sync_kind"] == "interval":
            steps = {source["step"] for source in row["source_rows"]}
            if (
                len(steps) != 1
                or row["sync_id"] != f"recurrent-step:{next(iter(steps))}"
                or next(iter(steps)) % exchange_interval != 0
            ):
                raise ValueError("interval exchange was not declared by the schedule")
        elif row["sync_kind"] == "schedule_bytecode":
            if row["sync_id"] not in bytecode_syncs:
                raise ValueError("bytecode exchange has no successful declaration")
            observed_bytecode.add(row["sync_id"])
        elif row["sync_kind"] == "controller_compare":
            if row["sync_id"] not in controller_syncs:
                raise ValueError("controller exchange has no successful declaration")
            observed_controller.add(row["sync_id"])
    if observed_bytecode != bytecode_syncs or observed_controller != controller_syncs:
        raise ValueError("declared branch exchange is missing from the trace")
    payload = {
        "schema": BRANCH_EXCHANGE_TRACE_SCHEMA,
        "exchange_count": len(validated),
        "first_generation_independent": bool(
            validated and validated[0]["counts_as_independent_support"] is True
        ),
        "later_generations_cooperative": len(validated) > 1,
        "independent_support_generations": sum(
            row["counts_as_independent_support"] is True for row in validated
        ),
        "sync_points": sync_points,
        "declared_sync_points_proven": True,
        "exchanges": validated,
    }
    return {**payload, "trace_sha256": canonical_sha256(payload)}


def validate_branch_exchange_trace(
    value: Any,
    *,
    exchange_count: int,
    n_branches: int,
    n_slots: int,
    comm_slot: int,
    exchange_gamma: float,
    branch_isolation: Any,
    cognitive_slots: Any,
    exchange_interval: int,
    schedule_hash: str,
    bytecode_events: Any,
    cognitive_action_trace: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("exchanges"), list):
        raise ValueError("branch exchange trace is missing")
    expected = build_branch_exchange_trace(
        exchanges=value["exchanges"],
        n_branches=n_branches,
        n_slots=n_slots,
        comm_slot=comm_slot,
        exchange_gamma=exchange_gamma,
        branch_isolation=branch_isolation,
        cognitive_slots=cognitive_slots,
        exchange_interval=exchange_interval,
        schedule_hash=schedule_hash,
        bytecode_events=bytecode_events,
        cognitive_action_trace=cognitive_action_trace,
    )
    if expected.get("exchange_count") != exchange_count or value != expected:
        raise ValueError("branch exchange trace differs from reconstructed evidence")
    return dict(value)


__all__ = [
    "BRANCH_EXCHANGE_SCHEMA",
    "BRANCH_EXCHANGE_TRACE_SCHEMA",
    "MAX_EXCHANGE_SOURCE_SLOTS",
    "build_branch_exchange_trace",
    "candidate_set_sha256",
    "canonical_sha256",
    "is_sha256",
    "private_exchange_slots",
    "validate_branch_exchange_receipt",
    "validate_branch_exchange_trace",
]
