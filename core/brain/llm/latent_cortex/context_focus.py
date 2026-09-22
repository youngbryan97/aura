"""Source-selective latent effects for memory and evidence actions.

SEARCH_MEMORY and RETRIEVE_EVIDENCE are not synonyms for a generic recurrent
step. When their source material is already admitted into immutable cognitive
slots, this module reads only the matching source class and writes a bounded
summary into the branch communication slot. The evidence rows remain byte-for-
byte unchanged and the following cognitive operator consumes the focused state.

This is the in-episode focus effect. It does not claim that a new external
retrieval occurred; governed re-fetch is a separate service-level operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.epistemic_state import (
    OperationKind,
    canonical_sha256,
)
from core.brain.llm.latent_cortex.recurrence import rms_match
from core.brain.llm.latent_cortex.verified_best import tensor_sha256

CONTEXT_FOCUS_SCHEMA = "aura.rlc.context_focus.v1"
CONTEXT_FOCUS_ACTIONS = {
    OperationKind.SEARCH_MEMORY,
    OperationKind.RETRIEVE_EVIDENCE,
}
DEFAULT_CONTEXT_FOCUS_STRENGTH = 0.18

_EVIDENCE_SOURCES = {"reference", "world_model"}


def source_matches_action(source: Any, action: OperationKind | str) -> bool:
    """Return whether one typed context source belongs to an action class."""

    try:
        operation = action if isinstance(action, OperationKind) else OperationKind(action)
    # not a failure: a name that is not an operation kind belongs to no action class.
    except (TypeError, ValueError):
        return False
    label = str(source or "")
    if operation is OperationKind.SEARCH_MEMORY:
        return (
            label in {"memory", "one_shot_memory"}
            or label.startswith("memory.")
        )
    if operation is OperationKind.RETRIEVE_EVIDENCE:
        return (
            label in {*_EVIDENCE_SOURCES, "one_shot_memory"}
            or label.startswith("evidence")
            or label.startswith("tool_observation")
            or label.startswith("capability.")
        )
    return False


def context_sources_for_action(
    context_slots: Sequence[Mapping[str, Any]],
    action: OperationKind | str,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Normalize the exact slot/source inventory read by one focus action."""

    try:
        operation = action if isinstance(action, OperationKind) else OperationKind(action)
    except (TypeError, ValueError) as exc:
        raise ValueError("context focus action is unsupported") from exc
    if operation not in CONTEXT_FOCUS_ACTIONS:
        raise ValueError("context focus action is unsupported")
    if not isinstance(context_slots, Sequence) or isinstance(
        context_slots,
        (str, bytes),
    ):
        raise ValueError("context focus slot inventory is invalid")
    selected: list[tuple[int, str]] = []
    observed: set[int] = set()
    for raw in context_slots:
        if not isinstance(raw, Mapping):
            raise ValueError("context focus slot row is invalid")
        slot = raw.get("slot")
        source = raw.get("source")
        if (
            type(slot) is not int
            or slot < 0
            or slot in observed
            or not isinstance(source, str)
            or not source
            or len(source) > 160
        ):
            raise ValueError("context focus slot row is invalid")
        observed.add(slot)
        if source_matches_action(source, operation):
            selected.append((slot, source))
    selected.sort()
    return (
        tuple(slot for slot, _source in selected),
        tuple(source for _slot, source in selected),
    )


def _strength(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= 0.5
    ):
        raise ValueError("context focus strength must be finite in (0, 0.5]")
    return round(float(value), 8)


def apply_context_focus(
    state: Any,
    *,
    context_slots: Sequence[Mapping[str, Any]],
    action: OperationKind | str,
    branch_index: int,
    action_step: int,
    comm_slot: int = 0,
    strength: float = DEFAULT_CONTEXT_FOCUS_STRENGTH,
    rms_clip_ratio: float = 3.0,
) -> tuple[Any, dict[str, Any]]:
    """Focus one branch on matching immutable sources and receipt the write."""

    import mlx.core as mx

    try:
        operation = action if isinstance(action, OperationKind) else OperationKind(action)
    except (TypeError, ValueError) as exc:
        raise ValueError("context focus action is unsupported") from exc
    if operation not in CONTEXT_FOCUS_ACTIONS:
        raise ValueError("context focus action is unsupported")
    if type(branch_index) is not int or branch_index < 0:
        raise ValueError("context focus branch index is invalid")
    if type(action_step) is not int or action_step < 0:
        raise ValueError("context focus action step is invalid")
    if not hasattr(state, "shape") or len(state.shape) != 3 or int(state.shape[0]) != 1:
        raise ValueError("context focus state must have shape (1, slots, hidden)")
    slot_count = int(state.shape[1])
    hidden = int(state.shape[2])
    if not 0 <= comm_slot < slot_count:
        raise ValueError("context focus communication slot is invalid")
    normalized_strength = _strength(strength)
    source_slots, source_labels = context_sources_for_action(context_slots, operation)
    if not source_slots:
        available = sorted(
            str(row.get("source") or "")
            for row in context_slots
            if isinstance(row, Mapping)
        )
        raise ValueError(
            "context focus has no matching admitted source: "
            f"action={operation.value}; available={available}"
        )
    if any(slot >= slot_count or slot == comm_slot for slot in source_slots):
        raise ValueError("context focus source slot is invalid")

    source_tensor = mx.concatenate(
        [state[:, slot : slot + 1, :] for slot in source_slots],
        axis=1,
    )
    summary = mx.mean(source_tensor, axis=1, keepdims=True)
    prior = state[:, comm_slot : comm_slot + 1, :]
    focused = rms_match(
        (1.0 - normalized_strength) * prior + normalized_strength * summary,
        prior,
        rms_clip_ratio,
    )
    output = mx.concatenate(
        [
            focused if slot == comm_slot else state[:, slot : slot + 1, :]
            for slot in range(slot_count)
        ],
        axis=1,
    )
    preserved = mx.concatenate(
        [output[:, slot : slot + 1, :] for slot in source_slots],
        axis=1,
    )
    mx.eval(output, source_tensor, preserved)
    input_sha256 = tensor_sha256(state)
    output_sha256 = tensor_sha256(output)
    source_sha256 = tensor_sha256(source_tensor)
    preserved_sha256 = tensor_sha256(preserved)
    if input_sha256 == output_sha256:
        raise RuntimeError("context focus did not cause a state transition")
    if source_sha256 != preserved_sha256:
        raise RuntimeError("context focus mutated immutable source slots")

    accounting = {
        "element_reads": (len(source_slots) + 2) * hidden,
        "element_writes": (slot_count + 1) * hidden,
        "tensor_scalar_ops": (len(source_slots) + 7) * hidden,
        "commitment_host_ops": (3 * slot_count + 2 * len(source_slots)) * hidden,
        "hidden_layer_apps": 0,
    }
    payload = {
        "schema": CONTEXT_FOCUS_SCHEMA,
        "action": operation.value,
        "source_class": (
            "memory" if operation is OperationKind.SEARCH_MEMORY else "evidence"
        ),
        "branch_index": branch_index,
        "action_step": action_step,
        "n_slots": slot_count,
        "hidden_dimension": hidden,
        "target_slot": comm_slot,
        "source_slots": list(source_slots),
        "source_labels": list(source_labels),
        "strength": normalized_strength,
        "input_sha256": input_sha256,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "preserved_source_sha256": preserved_sha256,
        "tensor_accounting": accounting,
        "causal": True,
        "external_retrieval_effect": "none",
    }
    return output, {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_context_focus_receipt(
    value: Any,
    *,
    cognitive_slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate source selection, accounting, preservation, and commitment."""

    fields = {
        "schema",
        "action",
        "source_class",
        "branch_index",
        "action_step",
        "n_slots",
        "hidden_dimension",
        "target_slot",
        "source_slots",
        "source_labels",
        "strength",
        "input_sha256",
        "source_sha256",
        "output_sha256",
        "preserved_source_sha256",
        "tensor_accounting",
        "causal",
        "external_retrieval_effect",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("context focus receipt fields differ")
    try:
        action = OperationKind(value["action"])
    except (TypeError, ValueError) as exc:
        raise ValueError("context focus action is unsupported") from exc
    source_slots, source_labels = context_sources_for_action(cognitive_slots, action)
    n_slots = value["n_slots"]
    hidden = value["hidden_dimension"]
    target = value["target_slot"]
    if (
        value["schema"] != CONTEXT_FOCUS_SCHEMA
        or action not in CONTEXT_FOCUS_ACTIONS
        or value["source_class"]
        != ("memory" if action is OperationKind.SEARCH_MEMORY else "evidence")
        or type(value["branch_index"]) is not int
        or value["branch_index"] < 0
        or type(value["action_step"]) is not int
        or value["action_step"] < 0
        or type(n_slots) is not int
        or n_slots < 3
        or type(hidden) is not int
        or hidden < 1
        or type(target) is not int
        or not 0 <= target < n_slots
        or value["source_slots"] != list(source_slots)
        or value["source_labels"] != list(source_labels)
        or not source_slots
        or target in source_slots
        or any(slot >= n_slots for slot in source_slots)
        or value["strength"] != _strength(value["strength"])
        or value["causal"] is not True
        or value["external_retrieval_effect"] != "none"
    ):
        raise ValueError("context focus execution metadata differs")
    for name in (
        "input_sha256",
        "source_sha256",
        "output_sha256",
        "preserved_source_sha256",
    ):
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("context focus tensor commitment is invalid")
    if (
        value["input_sha256"] == value["output_sha256"]
        or value["source_sha256"] != value["preserved_source_sha256"]
    ):
        raise ValueError("context focus causality or preservation is invalid")
    expected_accounting = {
        "element_reads": (len(source_slots) + 2) * hidden,
        "element_writes": (n_slots + 1) * hidden,
        "tensor_scalar_ops": (len(source_slots) + 7) * hidden,
        "commitment_host_ops": (3 * n_slots + 2 * len(source_slots)) * hidden,
        "hidden_layer_apps": 0,
    }
    if value["tensor_accounting"] != expected_accounting:
        raise ValueError("context focus tensor accounting differs")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("context focus receipt commitment differs")
    return dict(value)


__all__ = [
    "CONTEXT_FOCUS_ACTIONS",
    "CONTEXT_FOCUS_SCHEMA",
    "DEFAULT_CONTEXT_FOCUS_STRENGTH",
    "apply_context_focus",
    "context_sources_for_action",
    "source_matches_action",
    "validate_context_focus_receipt",
]
