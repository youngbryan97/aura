"""Measured runtime-integrity proof for one latent-cortex episode.

The legacy ``params_unchanged`` and ``fast_weights_erased`` fields are useful
telemetry, but they are mutable assertions. This module reconstructs the
authority needed to keep a resident worker alive from measurements taken on
both sides of the episode and binds those measurements to the exact worker
that served it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from core.runtime.model_layers import require_model_layers

RUNTIME_INTEGRITY_SCHEMA = "aura.rlc.runtime_integrity.v1"
PARAMETER_CANARY_SCHEMA = "aura.rlc.parameter_canary.v1"
ADAPTED_LAYER_SCHEMA = "aura.rlc.adapted_layer_identity.v1"
STACK_MEASUREMENT_SCHEMA = "aura.rlc.serving_stack_measurement.v1"
FAST_WEIGHT_CLEANUP_SCHEMA = "aura.rlc.fast_weight_cleanup.v1"
POLICY = {
    "parameter_measurement": "fixed_stride_tensor_canary_sha256_v1",
    "adapted_layer_measurement": "exact_target_parameter_bytes_sha256_v1",
    "serving_stack": "pre_post_runtime_and_artifact_identity_v1",
    "fast_weight_erase": "exact_full_stack_probe_pre_post_v1",
    "cache_safety": "model_function_change_invalidation_and_empty_final_cache_v1",
    "worker_binding": "exact_worker_boot_model_and_stack_identity_v1",
}

_TARGET_ATTRS = {
    "o_proj": ("self_attn", "o_proj"),
    "down_proj": ("mlp", "down_proj"),
}
_TOP_LEVEL_FIELDS = {
    "schema",
    "policy_sha256",
    "episode_id",
    "input_tokens_sha256",
    "checkpoint",
    "parameters",
    "adapted_layers",
    "serving_stack",
    "fast_weight_erase",
    "cache",
    "worker",
    "verdict",
    "receipt_sha256",
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_fast_weight_cleanup_proof(
    *,
    episode_id: str,
    input_tokens_sha256: str,
    detached: bool,
    erase_proven: bool,
    lease_released: bool,
    conflicts: int,
    pre_probe_sha256: str,
    post_probe_sha256: str,
    layer_ids: Sequence[str],
) -> dict[str, Any]:
    payload = {
        "schema": FAST_WEIGHT_CLEANUP_SCHEMA,
        "episode_id": str(episode_id or ""),
        "input_tokens_sha256": str(input_tokens_sha256 or ""),
        "detached": detached,
        "erase_proven": erase_proven,
        "lease_released": lease_released,
        "conflicts": conflicts,
        "pre_probe_sha256": str(pre_probe_sha256 or ""),
        "post_probe_sha256": str(post_probe_sha256 or ""),
        "layer_ids": list(layer_ids),
    }
    proof = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_fast_weight_cleanup_proof(
        proof,
        expected_episode_id=episode_id,
        expected_input_tokens_sha256=input_tokens_sha256,
    )


def validate_fast_weight_cleanup_proof(
    value: Mapping[str, Any],
    *,
    expected_episode_id: str | None = None,
    expected_input_tokens_sha256: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "episode_id",
        "input_tokens_sha256",
        "detached",
        "erase_proven",
        "lease_released",
        "conflicts",
        "pre_probe_sha256",
        "post_probe_sha256",
        "layer_ids",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("fast-weight cleanup proof fields are invalid")
    payload = {
        key: value[key]
        for key in fields - {"receipt_sha256"}
    }
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("fast-weight cleanup proof commitment mismatch")
    if (
        value["schema"] != FAST_WEIGHT_CLEANUP_SCHEMA
        or not str(value["episode_id"] or "")
        or not _is_sha256(value["input_tokens_sha256"])
        or type(value["detached"]) is not bool
        or type(value["erase_proven"]) is not bool
        or type(value["lease_released"]) is not bool
        or type(value["conflicts"]) is not int
        or value["conflicts"] < 0
        or not isinstance(value["layer_ids"], list)
        or not value["layer_ids"]
        or len(set(value["layer_ids"])) != len(value["layer_ids"])
        or any(
            not isinstance(item, str) or not item
            for item in value["layer_ids"]
        )
    ):
        raise ValueError("fast-weight cleanup proof is invalid")
    if expected_episode_id is not None and value["episode_id"] != expected_episode_id:
        raise ValueError("fast-weight cleanup episode mismatch")
    if (
        expected_input_tokens_sha256 is not None
        and value["input_tokens_sha256"] != expected_input_tokens_sha256
    ):
        raise ValueError("fast-weight cleanup input mismatch")
    if value["erase_proven"] and (
        not value["detached"]
        or not value["lease_released"]
        or value["conflicts"] != 0
        or not _is_sha256(value["pre_probe_sha256"])
        or value["pre_probe_sha256"] != value["post_probe_sha256"]
    ):
        raise ValueError("fast-weight cleanup success does not reconstruct")
    return dict(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_bytes(value: Any) -> bytes:
    import numpy as np

    array = np.asarray(value)
    prefix = (
        f"{array.dtype}:{','.join(str(item) for item in array.shape)}:"
    ).encode("ascii")
    return prefix + array.tobytes()


def _parameter_tree_rows(module: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise ValueError("integrity target exposes no parameter tree")
    return sorted(tree_flatten(parameters()), key=lambda row: row[0])


def parameter_canary_fingerprint(
    model: Any,
    *,
    stride: int = 7,
    elements_per_tensor: int = 64,
) -> dict[str, Any]:
    """Hash a fixed, declared sample over the permanent parameter tree."""

    if type(stride) is not int or stride <= 0:
        raise ValueError("parameter canary stride must be positive")
    if type(elements_per_tensor) is not int or elements_per_tensor <= 0:
        raise ValueError("parameter canary element count must be positive")
    import mlx.core as mx

    rows = _parameter_tree_rows(model)
    selected = rows[::stride]
    digest = hashlib.sha256()
    sampled_elements = 0
    for name, tensor in selected:
        flat = mx.reshape(tensor, (-1,))
        size = int(flat.size)
        if size <= elements_per_tensor:
            sample = flat
        else:
            # Fixed leading/middle/trailing coverage prevents the canary from
            # observing only one tensor boundary while keeping 32B overhead
            # bounded.
            third = max(1, elements_per_tensor // 3)
            middle = max(0, (size - third) // 2)
            sample = mx.concatenate(
                (
                    flat[:third],
                    flat[middle : middle + third],
                    flat[-third:],
                )
            )[:elements_per_tensor]
        mx.eval(sample)
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(sample))
        sampled_elements += int(sample.size)
    payload = {
        "schema": PARAMETER_CANARY_SCHEMA,
        "method": POLICY["parameter_measurement"],
        "stride": stride,
        "elements_per_tensor": elements_per_tensor,
        "parameter_leaf_count": len(rows),
        "sampled_tensor_count": len(selected),
        "sampled_element_count": sampled_elements,
        "sha256": digest.hexdigest(),
    }
    return payload


def adapted_layer_fingerprint(
    model: Any,
    *,
    layer_indices: Sequence[int],
    target: str,
) -> dict[str, Any]:
    """Hash every permanent parameter byte in the layers fast weights target."""

    if target not in _TARGET_ATTRS:
        raise ValueError("adapted-layer target is unsupported")
    layers = require_model_layers(model).layers
    normalized = list(layer_indices)
    if (
        not normalized
        or any(type(index) is not int or not 0 <= index < len(layers) for index in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("adapted-layer index set is invalid")
    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    parent_attr, leaf_attr = _TARGET_ATTRS[target]
    for index in normalized:
        parent = getattr(layers[index], parent_attr)
        module = getattr(parent, leaf_attr)
        rows = _parameter_tree_rows(module)
        if not rows:
            raise ValueError("adapted-layer target has no parameters")
        for name, tensor in rows:
            digest.update(f"{index}:{target}:{name}".encode())
            digest.update(_tensor_bytes(tensor))
            tensor_count += 1
            element_count += int(tensor.size)
    return {
        "schema": ADAPTED_LAYER_SCHEMA,
        "method": POLICY["adapted_layer_measurement"],
        "target": target,
        "layer_ids": [f"layers.{index}.{target}" for index in normalized],
        "tensor_count": tensor_count,
        "element_count": element_count,
        "sha256": digest.hexdigest(),
    }


def serving_stack_measurement(
    model: Any,
    tokenizer: Any,
    model_path: str,
) -> dict[str, Any]:
    """Measure the function-defining tokenizer/adapter/quantization stack."""

    from core.brain.llm.latent_cortex.runtime_identity import (
        serving_stack_identity,
    )

    identity = serving_stack_identity(
        model,
        model_path,
        tokenizer=tokenizer,
    )
    payload = {
        "schema": STACK_MEASUREMENT_SCHEMA,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
    }
    return payload


def _comparison(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    identity_fields: Sequence[str],
) -> dict[str, Any]:
    if any(before.get(field) != after.get(field) for field in identity_fields):
        unchanged = False
    else:
        unchanged = before.get("sha256") == after.get("sha256")
    return {
        "before": dict(before),
        "after": dict(after),
        "unchanged": unchanged,
    }


def _fast_weight_erase(
    *,
    episode_id: str,
    input_tokens_sha256: str,
    fast_weights_applied: bool,
    fast_weight_learning: Mapping[str, Any] | None,
    fast_weight_cleanup: Mapping[str, Any] | None,
    fast_weights_attach_attempted: bool = False,
) -> dict[str, Any]:
    # An attach that mutated some layers and then raised leaves
    # fast_weights_applied False while the resident model is dirty. Keying
    # "cleanup required" off the completed attach made the proof optional in
    # exactly the state that needs it.
    required = fast_weights_applied is True or fast_weights_attach_attempted is True
    learning: dict[str, Any] = {}
    if isinstance(fast_weight_learning, Mapping):
        try:
            from core.brain.llm.latent_cortex.fast_weight_learning import (
                validate_fast_weight_learning_receipt,
            )

            learning = validate_fast_weight_learning_receipt(
                fast_weight_learning,
                expected_episode_id=episode_id,
                expected_input_tokens_sha256=input_tokens_sha256,
            )
        except (ImportError, TypeError, ValueError):
            learning = {}
    learning_cleanup = learning.get("cleanup")
    learning_cleanup = (
        dict(learning_cleanup)
        if isinstance(learning_cleanup, Mapping)
        else {}
    )
    admission = learning.get("admission")
    admitted = bool(
        isinstance(admission, Mapping)
        and admission.get("admitted") is True
    )
    cleanup_proof: dict[str, Any] = {}
    if isinstance(fast_weight_cleanup, Mapping):
        try:
            cleanup_proof = validate_fast_weight_cleanup_proof(
                fast_weight_cleanup,
                expected_episode_id=episode_id,
                expected_input_tokens_sha256=input_tokens_sha256,
            )
        except (TypeError, ValueError):
            cleanup_proof = {}
    before = str(cleanup_proof.get("pre_probe_sha256") or "")
    after = str(cleanup_proof.get("post_probe_sha256") or "")
    layer_ids = list(cleanup_proof.get("layer_ids") or [])
    detached = cleanup_proof.get("detached") is True
    erase_proven = cleanup_proof.get("erase_proven") is True
    lease_released = cleanup_proof.get("lease_released") is True
    conflicts = cleanup_proof.get("conflicts", 0)
    learning_agrees = True
    if learning_cleanup and cleanup_proof:
        learning_agrees = all(
            (
                learning_cleanup.get("detached") is detached,
                learning_cleanup.get("erase_proven") is erase_proven,
                learning_cleanup.get("lease_released") is lease_released,
                learning_cleanup.get("conflicts") == conflicts,
                learning_cleanup.get("pre_probe_sha256") == before,
                learning_cleanup.get("post_probe_sha256") == after,
                list(learning_cleanup.get("erased_layer_ids") or [])
                == layer_ids,
            )
        )
    exact = bool(
        (not required and not admitted and not cleanup_proof)
        or (
            required
            and _is_sha256(cleanup_proof.get("receipt_sha256"))
            and _is_sha256(before)
            and before == after
            and bool(layer_ids)
            and detached
            and erase_proven
            and lease_released
            and conflicts == 0
            and learning_agrees
        )
    )
    return {
        "required": required,
        "learning_receipt_sha256": str(
            learning.get("receipt_sha256") or ""
        ),
        "cleanup_receipt_sha256": str(
            cleanup_proof.get("receipt_sha256") or ""
        ),
        "admitted": admitted,
        "detached": detached,
        "erase_proven": erase_proven,
        "lease_released": lease_released,
        "conflicts": conflicts,
        "pre_probe_sha256": before,
        "post_probe_sha256": after,
        "layer_ids": layer_ids,
        "learning_agrees": learning_agrees,
        "exact": exact,
    }


def _cache_proof(
    *,
    fast_weights_applied: bool,
    probe_cache: Mapping[str, Any] | None,
    fast_weights_attach_attempted: bool = False,
) -> dict[str, Any]:
    cache = dict(probe_cache) if isinstance(probe_cache, Mapping) else {}
    enabled = bool(cache)
    invalidations = list(cache.get("invalidations") or []) if enabled else []
    entries = cache.get("entries", 0)
    touched = fast_weights_applied or fast_weights_attach_attempted
    safe = bool(
        type(entries) is int
        and entries >= 0
        and (
            not touched
            or not enabled
            or (
                entries == 0
                and "fast_weights_attached" in invalidations
                and "fast_weights_detached" in invalidations
            )
        )
    )
    return {
        "enabled": enabled,
        "probe_cache": cache,
        "invalidations": invalidations,
        "final_entry_count": entries,
        "safe": safe,
    }


def _engine_reasons(payload: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    checkpoint = payload["checkpoint"]
    if checkpoint["required"] and checkpoint["exact"] is not True:
        reasons.append("checkpoint_identity_unproven")
    if payload["parameters"]["unchanged"] is not True:
        reasons.append("parameter_canary_changed")
    if payload["adapted_layers"]["unchanged"] is not True:
        reasons.append("adapted_layer_identity_changed")
    if payload["serving_stack"]["unchanged"] is not True:
        reasons.append("serving_stack_changed")
    if checkpoint["required"]:
        for side in ("before", "after"):
            identity = payload["serving_stack"][side].get("identity")
            gaps = (
                identity.get("worker_stack_identity_gaps")
                if isinstance(identity, Mapping)
                else None
            )
            if not isinstance(gaps, list) or gaps:
                reasons.append(f"serving_stack_{side}_identity_incomplete")
    if payload["fast_weight_erase"]["exact"] is not True:
        reasons.append("fast_weight_erase_unproven")
    if payload["cache"]["safe"] is not True:
        reasons.append("cache_invalidation_unproven")
    return reasons


def build_engine_runtime_integrity(
    *,
    episode_id: str,
    input_tokens_sha256: str,
    checkpoint: Mapping[str, Any],
    parameters_before: Mapping[str, Any],
    parameters_after: Mapping[str, Any],
    adapted_layers_before: Mapping[str, Any],
    adapted_layers_after: Mapping[str, Any],
    serving_stack_before: Mapping[str, Any],
    serving_stack_after: Mapping[str, Any],
    fast_weights_applied: bool,
    fast_weight_learning: Mapping[str, Any] | None,
    fast_weight_cleanup: Mapping[str, Any] | None,
    probe_cache: Mapping[str, Any] | None,
    fast_weights_attach_attempted: bool = False,
) -> dict[str, Any]:
    checkpoint_payload = {
        "required": bool(checkpoint.get("required")),
        "fingerprint": str(checkpoint.get("fingerprint") or ""),
        "method": str(checkpoint.get("method") or ""),
        "file_count": int(checkpoint.get("files") or 0),
    }
    checkpoint_payload["exact"] = bool(
        not checkpoint_payload["required"]
        or (
            _is_sha256(checkpoint_payload["fingerprint"])
            and checkpoint_payload["method"] == "sha256"
            and checkpoint_payload["file_count"] > 0
        )
    )
    parameters = _comparison(
        parameters_before,
        parameters_after,
        identity_fields=(
            "schema",
            "method",
            "stride",
            "elements_per_tensor",
            "parameter_leaf_count",
            "sampled_tensor_count",
            "sampled_element_count",
        ),
    )
    adapted = _comparison(
        adapted_layers_before,
        adapted_layers_after,
        identity_fields=(
            "schema",
            "method",
            "target",
            "layer_ids",
            "tensor_count",
            "element_count",
        ),
    )
    stack = {
        "before": dict(serving_stack_before),
        "after": dict(serving_stack_after),
        "unchanged": (
            serving_stack_before == serving_stack_after
            and serving_stack_before.get("identity_sha256")
            == serving_stack_after.get("identity_sha256")
        ),
    }
    payload: dict[str, Any] = {
        "schema": RUNTIME_INTEGRITY_SCHEMA,
        "policy_sha256": canonical_sha256(POLICY),
        "episode_id": episode_id,
        "input_tokens_sha256": input_tokens_sha256,
        "checkpoint": checkpoint_payload,
        "parameters": parameters,
        "adapted_layers": adapted,
        "serving_stack": stack,
        "fast_weight_erase": _fast_weight_erase(
            episode_id=episode_id,
            input_tokens_sha256=input_tokens_sha256,
            fast_weights_applied=fast_weights_applied,
            fast_weight_learning=fast_weight_learning,
            fast_weight_cleanup=fast_weight_cleanup,
            fast_weights_attach_attempted=fast_weights_attach_attempted,
        ),
        "cache": _cache_proof(
            fast_weights_applied=fast_weights_applied,
            probe_cache=probe_cache,
            fast_weights_attach_attempted=fast_weights_attach_attempted,
        ),
        "worker": {
            "bound": False,
            "identity": {},
            "identity_sha256": "",
            "worker_boot_id": "",
            "worker_pid": 0,
            "worker_model_path": "",
            "stack_matches_engine": False,
        },
        "verdict": {},
    }
    reasons = _engine_reasons(payload)
    payload["verdict"] = {
        "engine_measurements_complete": not reasons,
        "worker_bound": False,
        "safe_to_continue": False,
        "reasons": reasons + ["worker_identity_unbound"],
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_runtime_integrity_receipt(receipt, require_worker=False)


def bind_worker_runtime_integrity(
    value: Mapping[str, Any],
    *,
    worker_identity: Mapping[str, Any],
) -> dict[str, Any]:
    engine = validate_runtime_integrity_receipt(value, require_worker=False)
    from core.brain.llm.latent_cortex.runtime_identity import (
        worker_identity_errors,
    )

    errors = worker_identity_errors(worker_identity)
    stack = engine["serving_stack"]["after"]["identity"]
    stack_matches = bool(
        not errors
        and worker_identity.get("worker_adapters")
        == stack.get("worker_adapters")
        and worker_identity.get("worker_adapter_stack_sha256")
        == stack.get("worker_adapter_stack_sha256")
        and worker_identity.get("worker_tokenizer")
        == stack.get("worker_tokenizer")
        and worker_identity.get("worker_quantization")
        == stack.get("worker_quantization")
        and worker_identity.get("worker_runtime_tokenizer")
        == stack.get("worker_runtime_tokenizer")
        and worker_identity.get("worker_stack_identity_gaps")
        == stack.get("worker_stack_identity_gaps")
    )
    payload = {
        key: value
        for key, value in engine.items()
        if key != "receipt_sha256"
    }
    bound = bool(not errors and stack_matches)
    payload["worker"] = (
        {
            "bound": True,
            "identity": dict(worker_identity),
            "identity_sha256": canonical_sha256(worker_identity),
            "worker_boot_id": str(
                worker_identity.get("worker_boot_id") or ""
            ),
            "worker_pid": int(worker_identity.get("worker_pid") or 0),
            "worker_model_path": str(
                worker_identity.get("worker_model_path") or ""
            ),
            "stack_matches_engine": True,
        }
        if bound
        else {
            "bound": False,
            "identity": {},
            "identity_sha256": "",
            "worker_boot_id": "",
            "worker_pid": 0,
            "worker_model_path": "",
            "stack_matches_engine": False,
        }
    )
    engine_reasons = _engine_reasons(payload)
    reasons = list(engine_reasons)
    reasons.extend(f"worker:{error}" for error in errors)
    if not stack_matches:
        reasons.append("worker_serving_stack_mismatch")
    if not bound:
        reasons.append("worker_identity_unbound")
    payload["verdict"] = {
        "engine_measurements_complete": not engine_reasons,
        "worker_bound": bound,
        "safe_to_continue": bool(
            not engine_reasons and bound
        ),
        "reasons": sorted(set(reasons)),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_runtime_integrity_receipt(receipt, require_worker=True)


def _validate_measurement_pair(
    pair: Any,
    *,
    schema: str,
    identity_fields: Sequence[str],
) -> None:
    if not isinstance(pair, Mapping) or set(pair) != {
        "before",
        "after",
        "unchanged",
    }:
        raise ValueError("runtime-integrity measurement pair is invalid")
    before = pair["before"]
    after = pair["after"]
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError("runtime-integrity measurement is not a mapping")
    expected_fields = {"sha256", *identity_fields}
    if set(before) != expected_fields or set(after) != expected_fields:
        raise ValueError("runtime-integrity measurement fields are invalid")
    if before.get("schema") != schema or after.get("schema") != schema:
        raise ValueError("runtime-integrity measurement schema mismatch")
    if not _is_sha256(before.get("sha256")) or not _is_sha256(
        after.get("sha256")
    ):
        raise ValueError("runtime-integrity measurement digest is invalid")
    for measurement in (before, after):
        if schema == PARAMETER_CANARY_SCHEMA:
            if (
                measurement.get("method") != POLICY["parameter_measurement"]
                or type(measurement.get("stride")) is not int
                or measurement["stride"] <= 0
                or type(measurement.get("elements_per_tensor")) is not int
                or measurement["elements_per_tensor"] <= 0
                or type(measurement.get("parameter_leaf_count")) is not int
                or measurement["parameter_leaf_count"] <= 0
                or type(measurement.get("sampled_tensor_count")) is not int
                or not (
                    0
                    < measurement["sampled_tensor_count"]
                    <= measurement["parameter_leaf_count"]
                )
                or type(measurement.get("sampled_element_count")) is not int
                or not (
                    0
                    < measurement["sampled_element_count"]
                    <= measurement["sampled_tensor_count"]
                    * measurement["elements_per_tensor"]
                )
            ):
                raise ValueError(
                    "runtime-integrity parameter canary is incomplete"
                )
        elif schema == ADAPTED_LAYER_SCHEMA:
            layer_ids = measurement.get("layer_ids")
            if (
                measurement.get("method")
                != POLICY["adapted_layer_measurement"]
                or measurement.get("target") not in _TARGET_ATTRS
                or not isinstance(layer_ids, list)
                or not layer_ids
                or len(set(layer_ids)) != len(layer_ids)
                or any(not isinstance(item, str) or not item for item in layer_ids)
                or type(measurement.get("tensor_count")) is not int
                or measurement["tensor_count"] <= 0
                or type(measurement.get("element_count")) is not int
                or measurement["element_count"] <= 0
            ):
                raise ValueError(
                    "runtime-integrity adapted-layer identity is incomplete"
                )
    expected = bool(
        all(before.get(field) == after.get(field) for field in identity_fields)
        and before["sha256"] == after["sha256"]
    )
    if type(pair["unchanged"]) is not bool or pair["unchanged"] is not expected:
        raise ValueError("runtime-integrity comparison does not reconstruct")


def validate_runtime_integrity_receipt(
    value: Mapping[str, Any],
    *,
    require_worker: bool,
    expected_episode_id: str | None = None,
    expected_input_tokens_sha256: str | None = None,
    expected_worker_identity: Mapping[str, Any] | None = None,
    expected_fast_weights_applied: bool | None = None,
    expected_fast_weights_attach_attempted: bool | None = None,
    expected_checkpoint_fingerprint: str | None = None,
    expected_checkpoint_method: str | None = None,
    expected_checkpoint_file_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("runtime-integrity fields do not match schema")
    payload = {
        key: value[key]
        for key in _TOP_LEVEL_FIELDS - {"receipt_sha256"}
    }
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("runtime-integrity commitment mismatch")
    if (
        value["schema"] != RUNTIME_INTEGRITY_SCHEMA
        or value["policy_sha256"] != canonical_sha256(POLICY)
        or not isinstance(value["episode_id"], str)
        or not value["episode_id"]
        or not _is_sha256(value["input_tokens_sha256"])
    ):
        raise ValueError("runtime-integrity episode identity is invalid")
    if expected_episode_id is not None and value["episode_id"] != expected_episode_id:
        raise ValueError("runtime-integrity episode mismatch")
    if (
        expected_input_tokens_sha256 is not None
        and value["input_tokens_sha256"] != expected_input_tokens_sha256
    ):
        raise ValueError("runtime-integrity input mismatch")
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "required",
        "fingerprint",
        "method",
        "file_count",
        "exact",
    }:
        raise ValueError("runtime-integrity checkpoint measurement is invalid")
    if (
        type(checkpoint["required"]) is not bool
        or type(checkpoint["file_count"]) is not int
        or checkpoint["file_count"] < 0
        or type(checkpoint["exact"]) is not bool
    ):
        raise ValueError("runtime-integrity checkpoint types are invalid")
    expected_checkpoint_exact = bool(
        not checkpoint["required"]
        or (
            _is_sha256(checkpoint["fingerprint"])
            and checkpoint["method"] == "sha256"
            and checkpoint["file_count"] > 0
        )
    )
    if checkpoint["exact"] is not expected_checkpoint_exact:
        raise ValueError("runtime-integrity checkpoint verdict does not reconstruct")
    if (
        expected_checkpoint_fingerprint is not None
        and checkpoint["fingerprint"] != expected_checkpoint_fingerprint
    ):
        raise ValueError("runtime-integrity checkpoint fingerprint mismatch")
    if (
        expected_checkpoint_method is not None
        and checkpoint["method"] != expected_checkpoint_method
    ):
        raise ValueError("runtime-integrity checkpoint method mismatch")
    if (
        expected_checkpoint_file_count is not None
        and checkpoint["file_count"] != expected_checkpoint_file_count
    ):
        raise ValueError("runtime-integrity checkpoint file count mismatch")
    _validate_measurement_pair(
        value["parameters"],
        schema=PARAMETER_CANARY_SCHEMA,
        identity_fields=(
            "schema",
            "method",
            "stride",
            "elements_per_tensor",
            "parameter_leaf_count",
            "sampled_tensor_count",
            "sampled_element_count",
        ),
    )
    _validate_measurement_pair(
        value["adapted_layers"],
        schema=ADAPTED_LAYER_SCHEMA,
        identity_fields=(
            "schema",
            "method",
            "target",
            "layer_ids",
            "tensor_count",
            "element_count",
        ),
    )
    stack = value["serving_stack"]
    if not isinstance(stack, Mapping) or set(stack) != {
        "before",
        "after",
        "unchanged",
    }:
        raise ValueError("runtime-integrity serving stack is invalid")
    for measurement in (stack["before"], stack["after"]):
        if (
            not isinstance(measurement, Mapping)
            or measurement.get("schema") != STACK_MEASUREMENT_SCHEMA
            or not isinstance(measurement.get("identity"), Mapping)
            or not _is_sha256(measurement.get("identity_sha256"))
            or measurement["identity_sha256"]
            != canonical_sha256(measurement["identity"])
        ):
            raise ValueError("runtime-integrity stack measurement is invalid")
        identity = measurement["identity"]
        if set(identity) != {
            "worker_adapters",
            "worker_adapter_stack_sha256",
            "worker_tokenizer",
            "worker_runtime_tokenizer",
            "worker_quantization",
            "worker_stack_identity_gaps",
        }:
            raise ValueError("runtime-integrity serving identity fields are invalid")
        if (
            not isinstance(identity["worker_adapters"], list)
            or identity["worker_adapter_stack_sha256"]
            != canonical_sha256(identity["worker_adapters"])
            or not isinstance(identity["worker_tokenizer"], Mapping)
            or not isinstance(identity["worker_runtime_tokenizer"], Mapping)
            or not isinstance(identity["worker_quantization"], Mapping)
            or not isinstance(identity["worker_stack_identity_gaps"], list)
            or any(
                not isinstance(item, str) or not item
                for item in identity["worker_stack_identity_gaps"]
            )
        ):
            raise ValueError("runtime-integrity serving identity is invalid")
        if checkpoint["required"]:
            from core.brain.llm.latent_cortex.runtime_identity import (
                serving_stack_identity_errors,
            )

            identity_errors = serving_stack_identity_errors(identity)
            if identity_errors:
                raise ValueError(
                    "runtime-integrity serving identity is incomplete:"
                    + ",".join(identity_errors)
                )
    expected_stack_unchanged = bool(
        stack["before"] == stack["after"]
        and stack["before"]["identity_sha256"]
        == stack["after"]["identity_sha256"]
    )
    if (
        type(stack["unchanged"]) is not bool
        or stack["unchanged"] is not expected_stack_unchanged
    ):
        raise ValueError("runtime-integrity stack comparison does not reconstruct")
    erase = value["fast_weight_erase"]
    if not isinstance(erase, Mapping) or set(erase) != {
        "required",
        "learning_receipt_sha256",
        "cleanup_receipt_sha256",
        "admitted",
        "detached",
        "erase_proven",
        "lease_released",
        "conflicts",
        "pre_probe_sha256",
        "post_probe_sha256",
        "layer_ids",
        "learning_agrees",
        "exact",
    }:
        raise ValueError("runtime-integrity erase proof is invalid")
    if (
        type(erase["required"]) is not bool
        or not isinstance(erase["learning_receipt_sha256"], str)
        or (
            erase["learning_receipt_sha256"]
            and not _is_sha256(erase["learning_receipt_sha256"])
        )
        or not isinstance(erase["cleanup_receipt_sha256"], str)
        or (
            erase["cleanup_receipt_sha256"]
            and not _is_sha256(erase["cleanup_receipt_sha256"])
        )
        or type(erase["admitted"]) is not bool
        or type(erase["detached"]) is not bool
        or type(erase["erase_proven"]) is not bool
        or type(erase["lease_released"]) is not bool
        or type(erase["conflicts"]) is not int
        or erase["conflicts"] < 0
        or type(erase["learning_agrees"]) is not bool
        or type(erase["exact"]) is not bool
        or not isinstance(erase["layer_ids"], list)
        or any(not isinstance(item, str) or not item for item in erase["layer_ids"])
    ):
        raise ValueError("runtime-integrity erase proof types are invalid")
    if erase["cleanup_receipt_sha256"]:
        cleanup_payload = {
            "schema": FAST_WEIGHT_CLEANUP_SCHEMA,
            "episode_id": value["episode_id"],
            "input_tokens_sha256": value["input_tokens_sha256"],
            "detached": erase["detached"],
            "erase_proven": erase["erase_proven"],
            "lease_released": erase["lease_released"],
            "conflicts": erase["conflicts"],
            "pre_probe_sha256": erase["pre_probe_sha256"],
            "post_probe_sha256": erase["post_probe_sha256"],
            "layer_ids": list(erase["layer_ids"]),
        }
        if erase["cleanup_receipt_sha256"] != canonical_sha256(
            cleanup_payload
        ):
            raise ValueError(
                "runtime-integrity cleanup commitment does not reconstruct"
            )
    elif (
        erase["detached"]
        or erase["erase_proven"]
        or erase["lease_released"]
        or erase["conflicts"]
        or erase["pre_probe_sha256"]
        or erase["post_probe_sha256"]
        or erase["layer_ids"]
    ):
        raise ValueError(
            "runtime-integrity cleanup measurements lack a commitment"
        )
    expected_erase = bool(
        (
            not erase["required"]
            and not erase["cleanup_receipt_sha256"]
            and erase["admitted"] is False
            and erase["detached"] is False
            and erase["erase_proven"] is False
            and erase["lease_released"] is False
            and erase["conflicts"] == 0
            and not erase["pre_probe_sha256"]
            and not erase["post_probe_sha256"]
            and not erase["layer_ids"]
            and erase["learning_agrees"] is True
        )
        or (
            erase["required"]
            and _is_sha256(erase["cleanup_receipt_sha256"])
            and _is_sha256(erase["pre_probe_sha256"])
            and erase["pre_probe_sha256"] == erase["post_probe_sha256"]
            and bool(erase["layer_ids"])
            and erase["detached"]
            and erase["erase_proven"]
            and erase["lease_released"]
            and erase["conflicts"] == 0
            and erase["learning_agrees"]
        )
    )
    if erase["exact"] is not expected_erase:
        raise ValueError("runtime-integrity erase verdict does not reconstruct")
    if (
        expected_fast_weights_applied is not None
        or expected_fast_weights_attach_attempted is not None
    ):
        expected_scope = bool(expected_fast_weights_applied) or bool(
            expected_fast_weights_attach_attempted
        )
        if erase["required"] is not expected_scope:
            raise ValueError("runtime-integrity fast-weight scope mismatch")
    cache = value["cache"]
    if not isinstance(cache, Mapping) or set(cache) != {
        "enabled",
        "probe_cache",
        "invalidations",
        "final_entry_count",
        "safe",
    }:
        raise ValueError("runtime-integrity cache proof is invalid")
    if (
        type(cache["enabled"]) is not bool
        or not isinstance(cache["probe_cache"], Mapping)
        or not isinstance(cache["invalidations"], list)
        or any(not isinstance(item, str) for item in cache["invalidations"])
        or type(cache["final_entry_count"]) is not int
        or cache["final_entry_count"] < 0
        or type(cache["safe"]) is not bool
    ):
        raise ValueError("runtime-integrity cache proof types are invalid")
    expected_cache_safe = bool(
        not erase["required"]
        or not cache["enabled"]
        or (
            cache["final_entry_count"] == 0
            and "fast_weights_attached" in cache["invalidations"]
            and "fast_weights_detached" in cache["invalidations"]
        )
    )
    if cache["safe"] is not expected_cache_safe:
        raise ValueError("runtime-integrity cache verdict does not reconstruct")
    worker = value["worker"]
    if not isinstance(worker, Mapping) or set(worker) != {
        "bound",
        "identity",
        "identity_sha256",
        "worker_boot_id",
        "worker_pid",
        "worker_model_path",
        "stack_matches_engine",
    }:
        raise ValueError("runtime-integrity worker proof is invalid")
    if (
        type(worker["bound"]) is not bool
        or not isinstance(worker["identity"], Mapping)
        or type(worker["worker_pid"]) is not int
        or worker["worker_pid"] < 0
        or type(worker["stack_matches_engine"]) is not bool
    ):
        raise ValueError("runtime-integrity worker proof types are invalid")
    if worker["bound"]:
        from core.brain.llm.latent_cortex.runtime_identity import (
            worker_identity_errors,
        )

        identity = dict(worker["identity"])
        if (
            worker_identity_errors(identity)
            or worker["identity_sha256"] != canonical_sha256(identity)
            or worker["worker_boot_id"] != identity.get("worker_boot_id")
            or worker["worker_pid"] != identity.get("worker_pid")
            or worker["worker_model_path"] != identity.get("worker_model_path")
            or worker["stack_matches_engine"] is not True
        ):
            raise ValueError("runtime-integrity worker binding is incomplete")
        measured_stack = stack["after"]["identity"]
        for field in (
            "worker_adapters",
            "worker_adapter_stack_sha256",
            "worker_tokenizer",
            "worker_runtime_tokenizer",
            "worker_quantization",
            "worker_stack_identity_gaps",
        ):
            if identity.get(field) != measured_stack.get(field):
                raise ValueError(
                    "runtime-integrity worker serving stack does not reconstruct"
                )
        if (
            expected_worker_identity is not None
            and any(
                expected_worker_identity.get(key) != item
                for key, item in identity.items()
            )
        ):
            raise ValueError("runtime-integrity worker differs from expected worker")
    if not worker["bound"] and (
        worker["identity"]
        or worker["identity_sha256"]
        or worker["worker_boot_id"]
        or worker["worker_pid"]
        or worker["worker_model_path"]
        or worker["stack_matches_engine"]
    ):
        raise ValueError("unbound runtime-integrity worker contains authority")
    verdict = value["verdict"]
    if not isinstance(verdict, Mapping) or set(verdict) != {
        "engine_measurements_complete",
        "worker_bound",
        "safe_to_continue",
        "reasons",
    }:
        raise ValueError("runtime-integrity verdict is invalid")
    if (
        type(verdict["engine_measurements_complete"]) is not bool
        or type(verdict["worker_bound"]) is not bool
        or type(verdict["safe_to_continue"]) is not bool
        or not isinstance(verdict["reasons"], list)
        or any(not isinstance(item, str) or not item for item in verdict["reasons"])
    ):
        raise ValueError("runtime-integrity verdict types are invalid")
    engine_reasons = _engine_reasons(value)
    expected_worker_bound = worker["bound"]
    expected_safe = bool(not engine_reasons and expected_worker_bound)
    if (
        verdict["engine_measurements_complete"] is not (not engine_reasons)
        or verdict["worker_bound"] is not expected_worker_bound
        or verdict["safe_to_continue"] is not expected_safe
    ):
        raise ValueError("runtime-integrity verdict does not reconstruct")
    expected_reasons = list(engine_reasons)
    if not expected_worker_bound:
        expected_reasons.append("worker_identity_unbound")
    if not set(expected_reasons).issubset(set(verdict["reasons"])):
        raise ValueError("runtime-integrity reasons omit a measured failure")
    if require_worker and not expected_worker_bound:
        raise ValueError("runtime-integrity worker binding is required")
    return dict(value)


def runtime_integrity_safe(
    value: Mapping[str, Any],
    *,
    require_worker: bool = True,
    expected_episode_id: str | None = None,
    expected_input_tokens_sha256: str | None = None,
    expected_worker_identity: Mapping[str, Any] | None = None,
    expected_fast_weights_applied: bool | None = None,
    expected_fast_weights_attach_attempted: bool | None = None,
    expected_checkpoint_fingerprint: str | None = None,
    expected_checkpoint_method: str | None = None,
    expected_checkpoint_file_count: int | None = None,
) -> bool:
    try:
        parsed = validate_runtime_integrity_receipt(
            value,
            require_worker=require_worker,
            expected_episode_id=expected_episode_id,
            expected_input_tokens_sha256=expected_input_tokens_sha256,
            expected_worker_identity=expected_worker_identity,
            expected_fast_weights_applied=expected_fast_weights_applied,
            expected_fast_weights_attach_attempted=(
                expected_fast_weights_attach_attempted
            ),
            expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
            expected_checkpoint_method=expected_checkpoint_method,
            expected_checkpoint_file_count=expected_checkpoint_file_count,
        )
    except (TypeError, ValueError):
        return False
    if require_worker:
        return parsed["verdict"]["safe_to_continue"] is True
    return parsed["verdict"]["engine_measurements_complete"] is True


def runtime_integrity_claim_verdict(
    value: Mapping[str, Any],
    claim: str,
    *,
    expected_episode_id: str | None = None,
    expected_input_tokens_sha256: str | None = None,
    expected_worker_identity: Mapping[str, Any] | None = None,
    expected_fast_weights_applied: bool | None = None,
    expected_fast_weights_attach_attempted: bool | None = None,
    expected_checkpoint_fingerprint: str | None = None,
    expected_checkpoint_method: str | None = None,
    expected_checkpoint_file_count: int | None = None,
) -> str:
    """Independently reconstruct one compatibility claim from measured proof."""

    try:
        proof = validate_runtime_integrity_receipt(
            value,
            require_worker=True,
            expected_episode_id=expected_episode_id,
            expected_input_tokens_sha256=expected_input_tokens_sha256,
            expected_worker_identity=expected_worker_identity,
            expected_fast_weights_applied=expected_fast_weights_applied,
            expected_fast_weights_attach_attempted=(
                expected_fast_weights_attach_attempted
            ),
            expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
            expected_checkpoint_method=expected_checkpoint_method,
            expected_checkpoint_file_count=expected_checkpoint_file_count,
        )
    except (TypeError, ValueError):
        return "unproven"
    if claim == "params_unchanged":
        if not (
            proof["parameters"]["unchanged"]
            and proof["adapted_layers"]["unchanged"]
            and proof["serving_stack"]["unchanged"]
        ):
            return "refuted"
        if (
            proof["checkpoint"]["exact"] is not True
            or proof["serving_stack"]["before"]["identity"][
                "worker_stack_identity_gaps"
            ]
            or proof["serving_stack"]["after"]["identity"][
                "worker_stack_identity_gaps"
            ]
        ):
            return "unproven"
        return "proven"
    if claim != "fast_weights_erased":
        return "unproven"
    erase = proof["fast_weight_erase"]
    cache = proof["cache"]
    if erase["exact"] and cache["safe"]:
        return "proven"
    if erase["required"] and (
        (
            erase["pre_probe_sha256"]
            and erase["post_probe_sha256"]
            and erase["pre_probe_sha256"] != erase["post_probe_sha256"]
        )
        or (cache["enabled"] and cache["safe"] is False)
    ):
        return "refuted"
    return "unproven"


__all__ = [
    "ADAPTED_LAYER_SCHEMA",
    "PARAMETER_CANARY_SCHEMA",
    "POLICY",
    "RUNTIME_INTEGRITY_SCHEMA",
    "STACK_MEASUREMENT_SCHEMA",
    "adapted_layer_fingerprint",
    "bind_worker_runtime_integrity",
    "build_engine_runtime_integrity",
    "canonical_sha256",
    "parameter_canary_fingerprint",
    "runtime_integrity_claim_verdict",
    "runtime_integrity_safe",
    "serving_stack_measurement",
    "validate_runtime_integrity_receipt",
]
