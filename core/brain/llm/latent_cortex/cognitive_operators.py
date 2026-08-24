"""Executable latent cognitive operators for blind branch specialization.

An operator is a bounded transformation of one branch's private workspace
before recurrence. The nine programs below do not differ merely by embedded
instructions: each has a different state transition, target-slot policy, and
anchor/control relationship. Context-seeded evidence slots are immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.brain.llm.latent_cortex.recurrence import rms_match
from core.runtime.tensor_identity import tensor_identity_sha256

COGNITIVE_OPERATOR_SCHEMA = "aura.rlc.cognitive_operator.v1"


class CognitiveOperator(StrEnum):
    DIRECT_DERIVATION = "direct_derivation"
    CONSTRUCTIVE_SOLUTION = "constructive_solution"
    COUNTEREXAMPLE = "counterexample"
    INVERSE_REASONING = "inverse_reasoning"
    CAUSAL_SIMULATION = "causal_simulation"
    FORMALIZATION = "formalization"
    ANALOGY_MAPPING = "analogy_mapping"
    ASSUMPTION_REMOVAL = "assumption_removal"
    BOUNDARY_CASE = "boundary_case"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    operator: CognitiveOperator
    transform: str
    strength: float


OPERATOR_SPECS: dict[CognitiveOperator, OperatorSpec] = {
    CognitiveOperator.DIRECT_DERIVATION: OperatorSpec(
        CognitiveOperator.DIRECT_DERIVATION, "single_control_write", 0.10
    ),
    CognitiveOperator.CONSTRUCTIVE_SOLUTION: OperatorSpec(
        CognitiveOperator.CONSTRUCTIVE_SOLUTION, "progressive_scaffold", 0.12
    ),
    CognitiveOperator.COUNTEREXAMPLE: OperatorSpec(
        CognitiveOperator.COUNTEREXAMPLE, "hypothesis_sign_reversal", 0.11
    ),
    CognitiveOperator.INVERSE_REASONING: OperatorSpec(
        CognitiveOperator.INVERSE_REASONING, "reverse_slot_transport", 0.09
    ),
    CognitiveOperator.CAUSAL_SIMULATION: OperatorSpec(
        CognitiveOperator.CAUSAL_SIMULATION, "finite_difference_rollout", 0.08
    ),
    CognitiveOperator.FORMALIZATION: OperatorSpec(
        CognitiveOperator.FORMALIZATION, "control_axis_projection", 0.10
    ),
    CognitiveOperator.ANALOGY_MAPPING: OperatorSpec(
        CognitiveOperator.ANALOGY_MAPPING, "paired_relation_transport", 0.09
    ),
    CognitiveOperator.ASSUMPTION_REMOVAL: OperatorSpec(
        CognitiveOperator.ASSUMPTION_REMOVAL, "max_alignment_subtraction", 0.13
    ),
    CognitiveOperator.BOUNDARY_CASE: OperatorSpec(
        CognitiveOperator.BOUNDARY_CASE, "signed_boundary_extrapolation", 0.08
    ),
}

ROLE_OPERATOR: dict[str, CognitiveOperator] = {
    "direct_derivation": CognitiveOperator.DIRECT_DERIVATION,
    "simplification": CognitiveOperator.DIRECT_DERIVATION,
    "constructive_solution": CognitiveOperator.CONSTRUCTIVE_SOLUTION,
    "counterexample_search": CognitiveOperator.COUNTEREXAMPLE,
    "inverse_reasoning": CognitiveOperator.INVERSE_REASONING,
    "reverse_reasoning": CognitiveOperator.INVERSE_REASONING,
    "causal_simulation": CognitiveOperator.CAUSAL_SIMULATION,
    "causal_reconstruction": CognitiveOperator.CAUSAL_SIMULATION,
    "formalization": CognitiveOperator.FORMALIZATION,
    "constraint_checking": CognitiveOperator.FORMALIZATION,
    "analogy_mapping": CognitiveOperator.ANALOGY_MAPPING,
    "analogy": CognitiveOperator.ANALOGY_MAPPING,
    "assumption_removal": CognitiveOperator.ASSUMPTION_REMOVAL,
    "adversarial_criticism": CognitiveOperator.ASSUMPTION_REMOVAL,
    "critical_audit": CognitiveOperator.ASSUMPTION_REMOVAL,
    "boundary_case_analysis": CognitiveOperator.BOUNDARY_CASE,
}


def operator_for_role(role: str) -> CognitiveOperator:
    try:
        return ROLE_OPERATOR[role]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown executable cognitive role: {role!r}") from exc


def _tensor_sha256(array: Any) -> str:
    return tensor_identity_sha256(array)


def _receipt_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def execute_cognitive_operator(
    z,
    anchor,
    control,
    *,
    operator: CognitiveOperator,
    role: str,
    branch_index: int,
    action: str,
    action_step: int,
    protected_slots: tuple[int, ...] = (),
    comm_slot: int = 0,
    rms_clip_ratio: float = 3.0,
) -> tuple[Any, dict[str, Any]]:
    """Apply one bounded, role-specific state transition and receipt it."""

    import mlx.core as mx

    if operator_for_role(role) is not operator:
        raise ValueError("role and cognitive operator disagree")
    if type(branch_index) is not int or branch_index < 0:
        raise ValueError("branch_index must be non-negative")
    if type(action_step) is not int or action_step < 0:
        raise ValueError("action_step must be non-negative")
    if not isinstance(action, str) or not action or len(action) > 80:
        raise ValueError("action must be a bounded non-empty string")
    if len(z.shape) != 3 or z.shape != anchor.shape or int(z.shape[0]) != 1:
        raise ValueError("operator state and anchor must have equal (1, slots, dim) shape")
    slot_count = int(z.shape[1])
    if not 0 <= comm_slot < slot_count:
        raise ValueError("comm_slot is outside the workspace")
    protected = tuple(sorted(set(protected_slots)))
    if any(type(index) is not int or not 0 <= index < slot_count for index in protected):
        raise ValueError("protected slot index is invalid")
    mutable = [index for index in range(slot_count) if index not in protected]
    if not mutable:
        raise ValueError("operator has no mutable workspace slot")

    spec = OPERATOR_SPECS[operator]
    vector = mx.reshape(control, (1, 1, int(z.shape[-1])))
    rows = [z[:, index : index + 1, :] for index in range(slot_count)]
    original = list(rows)
    changed: list[int] = []

    def replace(index: int, target, *, scale: float = 1.0) -> None:
        strength = min(0.25, spec.strength * scale)
        prior = original[index]
        candidate = (1.0 - strength) * prior + strength * target
        rows[index] = rms_match(candidate, prior, rms_clip_ratio)
        if index not in changed:
            changed.append(index)

    if operator is CognitiveOperator.DIRECT_DERIVATION:
        target = comm_slot if comm_slot in mutable else mutable[0]
        replace(target, vector)
    elif operator is CognitiveOperator.CONSTRUCTIVE_SOLUTION:
        for order, index in enumerate(mutable, start=1):
            scaffold = 0.55 * anchor[:, index : index + 1, :] + 0.45 * vector
            replace(index, scaffold, scale=0.5 + 0.5 * order / len(mutable))
    elif operator is CognitiveOperator.COUNTEREXAMPLE:
        index = mutable[-1]
        delta = original[index] - anchor[:, index : index + 1, :]
        replace(index, anchor[:, index : index + 1, :] - delta + vector)
    elif operator is CognitiveOperator.INVERSE_REASONING:
        reversed_rows = list(reversed([original[index] for index in mutable]))
        for index, source in zip(mutable, reversed_rows, strict=True):
            replace(index, 0.8 * source + 0.2 * vector)
    elif operator is CognitiveOperator.CAUSAL_SIMULATION:
        for position, index in enumerate(mutable):
            previous = original[mutable[position - 1]] if position else anchor[:, index : index + 1, :]
            velocity = original[index] - previous
            replace(index, original[index] + velocity + 0.25 * vector)
    elif operator is CognitiveOperator.FORMALIZATION:
        unit = vector / mx.maximum(mx.linalg.norm(vector, axis=-1, keepdims=True), 1e-6)
        for index in mutable:
            projection = mx.sum(original[index] * unit, axis=-1, keepdims=True) * unit
            replace(index, projection + 0.25 * anchor[:, index : index + 1, :])
    elif operator is CognitiveOperator.ANALOGY_MAPPING:
        for position, index in enumerate(mutable):
            partner = mutable[(position + max(1, len(mutable) // 2)) % len(mutable)]
            relation = original[partner] - anchor[:, partner : partner + 1, :]
            replace(index, anchor[:, index : index + 1, :] + relation + 0.2 * vector)
    elif operator is CognitiveOperator.ASSUMPTION_REMOVAL:
        unit = vector / mx.maximum(mx.linalg.norm(vector, axis=-1, keepdims=True), 1e-6)
        alignments = [
            abs(float(mx.sum(original[index] * unit))) for index in mutable
        ]
        index = mutable[max(range(len(mutable)), key=alignments.__getitem__)]
        projection = mx.sum(original[index] * unit, axis=-1, keepdims=True) * unit
        replace(index, anchor[:, index : index + 1, :] - projection)
    elif operator is CognitiveOperator.BOUNDARY_CASE:
        endpoints = [mutable[0]]
        if mutable[-1] != mutable[0]:
            endpoints.append(mutable[-1])
        for sign, index in zip((-1.0, 1.0), endpoints, strict=False):
            delta = original[index] - anchor[:, index : index + 1, :]
            replace(index, anchor[:, index : index + 1, :] + sign * delta + sign * vector)
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported cognitive operator: {operator.value}")

    output = mx.concatenate(rows, axis=1)
    mx.eval(output)
    input_sha256 = _tensor_sha256(z)
    output_sha256 = _tensor_sha256(output)
    if input_sha256 == output_sha256:
        raise RuntimeError(f"cognitive operator {operator.value} was not causal")
    payload = {
        "schema": COGNITIVE_OPERATOR_SCHEMA,
        "operator": operator.value,
        "transform": spec.transform,
        "role": role,
        "branch_index": branch_index,
        "action": action,
        "action_step": action_step,
        "strength": spec.strength,
        "n_slots": slot_count,
        "hidden_dimension": int(z.shape[-1]),
        "changed_slots": sorted(changed),
        "protected_slots": list(protected),
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "anchor_sha256": _tensor_sha256(anchor),
        "control_sha256": _tensor_sha256(vector),
        "tensor_accounting": {
            "element_reads": (
                2 * slot_count + 3 * len(changed) + 1
            )
            * int(z.shape[-1]),
            "element_writes": (slot_count + len(changed)) * int(z.shape[-1]),
            "tensor_scalar_ops": (
                24 * len(changed) + 4 * slot_count + 8
            )
            * int(z.shape[-1]),
            "commitment_host_ops": (3 * slot_count + 1) * int(z.shape[-1]),
            "hidden_layer_apps": 0,
        },
        "causal": True,
    }
    return output, {**payload, "receipt_sha256": _receipt_sha256(payload)}


def validate_operator_receipt(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "operator",
        "transform",
        "role",
        "branch_index",
        "action",
        "action_step",
        "strength",
        "n_slots",
        "hidden_dimension",
        "changed_slots",
        "protected_slots",
        "input_sha256",
        "output_sha256",
        "anchor_sha256",
        "control_sha256",
        "tensor_accounting",
        "causal",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("cognitive operator receipt schema is invalid")
    if value.get("schema") != COGNITIVE_OPERATOR_SCHEMA:
        raise ValueError("cognitive operator receipt version is invalid")
    try:
        operator = CognitiveOperator(value.get("operator"))
    except (TypeError, ValueError) as exc:
        raise ValueError("cognitive operator identity is invalid") from exc
    spec = OPERATOR_SPECS[operator]
    if value.get("transform") != spec.transform:
        raise ValueError("cognitive operator transform differs from its program")
    role = value.get("role")
    if operator_for_role(role) is not operator:
        raise ValueError("cognitive operator role differs from its program")
    strength = value.get("strength")
    n_slots = value.get("n_slots")
    hidden_dimension = value.get("hidden_dimension")
    if (
        type(value.get("branch_index")) is not int
        or value["branch_index"] < 0
        or type(value.get("action_step")) is not int
        or value["action_step"] < 0
        or not isinstance(value.get("action"), str)
        or not value["action"]
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not math.isclose(float(strength), spec.strength)
        or type(n_slots) is not int
        or n_slots < 1
        or type(hidden_dimension) is not int
        or hidden_dimension < 1
        or value.get("causal") is not True
    ):
        raise ValueError("cognitive operator execution metadata is invalid")
    changed = value.get("changed_slots")
    protected = value.get("protected_slots")
    if (
        not isinstance(changed, list)
        or not changed
        or changed != sorted(set(changed))
        or any(type(index) is not int or index < 0 for index in changed)
        or any(index >= n_slots for index in changed)
        or not isinstance(protected, list)
        or protected != sorted(set(protected))
        or any(type(index) is not int or index < 0 for index in protected)
        or any(index >= n_slots for index in protected)
        or set(changed) & set(protected)
    ):
        raise ValueError("cognitive operator slot evidence is invalid")
    accounting = value.get("tensor_accounting")
    expected_accounting = {
        "element_reads": (2 * n_slots + 3 * len(changed) + 1)
        * hidden_dimension,
        "element_writes": (n_slots + len(changed)) * hidden_dimension,
        "tensor_scalar_ops": (
            24 * len(changed) + 4 * n_slots + 8
        )
        * hidden_dimension,
        "commitment_host_ops": (3 * n_slots + 1) * hidden_dimension,
        "hidden_layer_apps": 0,
    }
    if accounting != expected_accounting:
        raise ValueError("cognitive operator tensor accounting differs")
    for key in (
        "input_sha256",
        "output_sha256",
        "anchor_sha256",
        "control_sha256",
    ):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 64:
            raise ValueError("cognitive operator tensor commitment is invalid")
        try:
            int(item, 16)
        except ValueError as exc:
            raise ValueError("cognitive operator tensor commitment is invalid") from exc
    if value["input_sha256"] == value["output_sha256"]:
        raise ValueError("cognitive operator receipt is non-causal")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _receipt_sha256(payload):
        raise ValueError("cognitive operator receipt digest differs")
    return dict(value)


__all__ = [
    "COGNITIVE_OPERATOR_SCHEMA",
    "OPERATOR_SPECS",
    "ROLE_OPERATOR",
    "CognitiveOperator",
    "execute_cognitive_operator",
    "operator_for_role",
    "validate_operator_receipt",
]
