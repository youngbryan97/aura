"""Strict identity contract for adapters trained through recurrent GRPO.

The recurrence-native supervised trainer and recurrent GRPO optimize different
objects.  They therefore cannot share an adapter manifest merely because both
produce scoped LoRA tensors.  This module binds the GRPO behavior policy,
recurrent execution graph, immutable training evidence, and final tensor
topology without inheriting any supervised-training or causal-gain claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any, Never

from core.brain.llm.latent_cortex.adapter_identity import (
    TensorIdentity,
    normalize_tensor_metadata,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.grpo import GRPO_SCHEMA
from core.learning.recurrent_grpo import VerifiedTrajectoryGroupConfig
from core.learning.recurrent_grpo_artifact_schema import (
    PROTOCOL_SCHEMA,
    PROTOCOL_SCHEMA_V5,
    PROTOCOL_SCHEMA_V6,
    PROTOCOL_SCHEMA_V7,
    PROTOCOL_SCHEMA_V8,
    PROTOCOL_TRAINING_KEYS,
    PROTOCOL_TRAINING_KEYS_V5,
    PROTOCOL_TRAINING_KEYS_V8,
    STEP_RECEIPT_KEYS,
    RecurrentGRPOArtifactSchemaError,
    protocol_semantic_sha256,
    recurrent_training_adequacy_report,
    validate_step_reward_channels,
)
from core.learning.recurrent_grpo_artifact_schema import (
    TRAINING_RECEIPT_SCHEMA as SHARED_TRAINING_RECEIPT_SCHEMA,
)
from core.learning.verified_transition_trainer import (
    VERIFIED_TRANSITION_STEP_SCHEMA,
    validate_verified_transition_step_receipt,
)

MANIFEST_FILE = "recurrence_adapter_manifest.json"
MANIFEST_SCHEMA = "aura.recurrent_grpo_adapter_manifest.v2"
IDENTITY_RECEIPT_SCHEMA = "aura.recurrent_grpo_adapter_identity_receipt.v1"
VERIFIED_IDENTITY_RECEIPT_SCHEMA = "aura.recurrent_grpo_verified_adapter_identity_receipt.v1"
COMPLETION_SCHEMA = "aura.recurrent_grpo_training_completion.v1"
TRAINING_PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
TRAINING_RECEIPT_SCHEMA = SHARED_TRAINING_RECEIPT_SCHEMA
DATASET_SCHEMA = "aura.grpo_dataset.v1"
LOADER_CONFIG_SCHEMA = "aura.recurrent_grpo_scoped_lora_config.v1"
TRAINING_METHOD = "recurrent_grpo"
OBJECTIVE_NAME = "aura.recurrent_grpo_behavior_policy.v1"

BINDING_ROLES = (
    "adapter",
    "adapter_alias",
    "loader_config",
    "training_receipt",
    "training_protocol",
    "dataset_manifest",
    "execution_spec",
)
REQUIRED_SOURCE_ROLES = frozenset(
    {
        "trainer",
        "grpo",
        "curriculum",
        "tasks",
        "checkpoint",
        "artifact_schema",
        "adapter",
        "recurrent_grpo",
        "recurrent_objective",
        "execution_spec",
        "latent_engine",
        "recurrence",
        "verified_trainer",
        "transition_campaign",
        "transition_episode",
        "transition_reward",
        "scope_reachability",
        "transition_admission",
        "transition_update",
        "transition_training_evidence",
        "campaign_trust",
        "transition_provider",
        "transition_provider_factory",
        "transition_launch_bundle",
        "transition_launch_runner",
        "transition_launch_materializer",
        "transition_recurrent_evidence",
        "transition_recurrent_repository",
        "transition_policy_probe",
        "transition_measurement_chain",
        "transition_policy_state_replay",
        "transition_policy_state_replay_worker",
        "transition_policy_state_replay_resume",
        "durable_external_verifier_job",
        "recurrent_training_prompt",
        "atomic_writer",
        "file_read_gateway",
        "file_write_gateway",
        "transition_transaction",
        "transition_rejection_transaction",
        "transition_causal_campaign",
        "verified_training_task",
        "verified_token_trace",
    }
)
LEGACY_REQUIRED_SOURCE_ROLES = REQUIRED_SOURCE_ROLES - {"scope_reachability"}
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1 << 50
MAX_TENSORS = 1_000_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PROJECTION_RE = re.compile(r"model\.layers\.(?:0|[1-9][0-9]*)(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z")


class RecurrentGRPOAdapterIdentityError(ValueError):
    """Stable fail-closed identity error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrentGRPOAdapterIdentityError(code)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        _fail("identity_not_canonical_json")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safetensors_tensors(
    payload: bytes,
) -> tuple[tuple[str, str, list[int], bytes], ...]:
    import numpy as np

    if not isinstance(payload, bytes) or len(payload) < 10:
        _fail("adapter_safetensors_invalid")
    header_size = int.from_bytes(payload[:8], "little", signed=False)
    if header_size < 2 or header_size > MAX_JSON_BYTES or 8 + header_size > len(payload):
        _fail("adapter_safetensors_header_invalid")
    header = strict_json_loads(payload[8 : 8 + header_size], role="adapter_safetensors")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        _fail("adapter_safetensors_metadata_invalid")
    if not header:
        _fail("adapter_safetensors_empty")
    dtype_table = {
        "BOOL": ("mlx.core.bool_", 1, "?"),
        "U8": ("mlx.core.uint8", 1, "u1"),
        "I8": ("mlx.core.int8", 1, "i1"),
        "U16": ("mlx.core.uint16", 2, "<u2"),
        "I16": ("mlx.core.int16", 2, "<i2"),
        "U32": ("mlx.core.uint32", 4, "<u4"),
        "I32": ("mlx.core.int32", 4, "<i4"),
        "U64": ("mlx.core.uint64", 8, "<u8"),
        "I64": ("mlx.core.int64", 8, "<i8"),
        "F16": ("mlx.core.float16", 2, "<f2"),
        "F32": ("mlx.core.float32", 4, "<f4"),
        "F64": ("mlx.core.float64", 8, "<f8"),
        "BF16": ("mlx.core.bfloat16", 2, None),
    }
    data = memoryview(payload)[8 + header_size :]
    tensors: list[tuple[str, str, list[int], bytes, int, int]] = []
    for name, raw_record in header.items():
        if not isinstance(name, str) or not name:
            _fail("adapter_safetensors_tensor_name_invalid")
        record = _exact(
            raw_record,
            {"dtype", "shape", "data_offsets"},
            role="adapter_safetensors_tensor",
        )
        dtype_record = dtype_table.get(record.get("dtype"))
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            dtype_record is None
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
        ):
            _fail("adapter_safetensors_tensor_invalid")
        start, end = offsets
        elements = math.prod(shape)
        dtype_name, item_size, numpy_dtype = dtype_record
        if end < start or end - start != elements * item_size or end > len(data):
            _fail("adapter_safetensors_tensor_range_invalid")
        raw = bytes(data[start:end])
        if numpy_dtype is None:
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            tensor_payload = (words << 16).view(np.float32).tobytes(order="C")
        else:
            tensor_payload = np.frombuffer(raw, dtype=numpy_dtype).tobytes(order="C")
        tensors.append((name, dtype_name, shape, tensor_payload, start, end))
    ranges = sorted((start, end) for *_prefix, start, end in tensors)
    cursor = 0
    for start, end in ranges:
        if start != cursor:
            _fail("adapter_safetensors_tensor_ranges_noncanonical")
        cursor = end
    if cursor != len(data):
        _fail("adapter_safetensors_unbound_bytes")
    return tuple(
        (name, dtype_name, shape, tensor_payload)
        for name, dtype_name, shape, tensor_payload, _start, _end in sorted(tensors)
    )


def tensor_metadata_from_safetensors(payload: bytes) -> tuple[dict[str, Any], ...]:
    """Read exact tensor metadata from the same immutable bytes being hashed."""

    return tuple(
        {"key": name, "dtype": dtype_name, "shape": shape}
        for name, dtype_name, shape, _tensor_payload in _safetensors_tensors(payload)
    )


def recurrent_policy_sha256_from_safetensors(
    payload: bytes,
    *,
    execution_spec_sha256: str,
) -> str:
    """Reconstruct the trainer's recurrent-policy digest from frozen tensors."""

    spec_sha256 = _sha(execution_spec_sha256, role="frozen_policy_execution_spec")

    digest = hashlib.sha256()
    digest.update(b"aura.recurrent_policy.v1\0")
    digest.update(spec_sha256.encode("ascii"))
    for name, dtype_name, shape, tensor_payload in _safetensors_tensors(payload):
        for part in (
            name.encode("utf-8"),
            dtype_name.encode("ascii"),
            json.dumps(shape, separators=(",", ":")).encode("ascii"),
            tensor_payload,
        ):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def strict_json_loads(raw: bytes, *, role: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_JSON_BYTES:
        _fail(f"{role}_size_invalid")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        _fail(f"{role}_not_ascii")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{role}_duplicate_json_key")
            result[key] = value
        return result

    def finite_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            _fail(f"{role}_number_invalid")
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_float=finite_float,
            parse_constant=lambda _value: _fail(f"{role}_number_invalid"),
        )
    except RecurrentGRPOAdapterIdentityError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_schema_invalid")
    return value


def _exact(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"{role}_schema_invalid")
    return value


def _sha(value: Any, *, role: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{role}_sha256_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
    return value


def _relative_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{role}_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"{role}_path_invalid")
    return value


def artifact_binding(value: Any, *, role: str) -> dict[str, Any]:
    value = _exact(value, {"path", "sha256", "size_bytes"}, role=role)
    return {
        "path": _relative_path(value["path"], role=role),
        "sha256": _sha(value["sha256"], role=role),
        "size_bytes": _integer(
            value["size_bytes"],
            role=f"{role}_size",
            minimum=1,
            maximum=MAX_ARTIFACT_BYTES,
        ),
    }


def _verify_artifact(
    binding: Mapping[str, Any], artifacts: Mapping[str, bytes], *, role: str
) -> bytes:
    payload = artifacts.get(str(binding["path"]))
    if not isinstance(payload, bytes):
        _fail(f"{role}_bytes_missing")
    if len(payload) != binding["size_bytes"]:
        _fail(f"{role}_size_mismatch")
    if sha256_bytes(payload) != binding["sha256"]:
        _fail(f"{role}_sha256_mismatch")
    return payload


def declared_bindings(manifest: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return every content binding named by a GRPO manifest."""

    if manifest.get("schema") != MANIFEST_SCHEMA:
        _fail("manifest_schema_unsupported")
    bindings = [(role, artifact_binding(manifest.get(role), role=role)) for role in BINDING_ROLES]
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) not in {
        REQUIRED_SOURCE_ROLES,
        LEGACY_REQUIRED_SOURCE_ROLES,
    }:
        _fail("sources_schema_invalid")
    for role in sorted(sources):
        source = _exact(
            sources[role],
            {"origin_path", "snapshot_path", "sha256", "size_bytes"},
            role=f"source_{role}",
        )
        bindings.append(
            (
                f"source_{role}",
                artifact_binding(
                    {
                        "path": source["snapshot_path"],
                        "sha256": source["sha256"],
                        "size_bytes": source["size_bytes"],
                    },
                    role=f"source_{role}",
                ),
            )
        )
    paths = [binding["path"] for _role, binding in bindings]
    if len(paths) != len(set(paths)):
        _fail("artifact_path_duplicated")
    return bindings


def required_source_roles(protocol_schema: str) -> frozenset[str]:
    """Return the immutable source inventory required by one protocol version."""

    if protocol_schema == TRAINING_PROTOCOL_SCHEMA:
        return REQUIRED_SOURCE_ROLES
    if protocol_schema == PROTOCOL_SCHEMA_V8:
        return REQUIRED_SOURCE_ROLES
    if protocol_schema in {PROTOCOL_SCHEMA_V5, PROTOCOL_SCHEMA_V6, PROTOCOL_SCHEMA_V7}:
        return LEGACY_REQUIRED_SOURCE_ROLES
    _fail("training_protocol_schema_invalid")


def _normalize_lora(value: Any) -> dict[str, Any]:
    value = _exact(
        value,
        {"rank", "targets", "wrapped_projections", "projection_paths", "trainable_params"},
        role="lora",
    )
    rank = _integer(value["rank"], role="lora_rank", minimum=1, maximum=1 << 20)
    targets = value["targets"]
    projections = value["projection_paths"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(targets) != len(set(targets))
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        _fail("lora_targets_invalid")
    if (
        not isinstance(projections, list)
        or not projections
        or len(projections) != len(set(projections))
        or any(
            not isinstance(path, str) or _PROJECTION_RE.fullmatch(path) is None
            for path in projections
        )
    ):
        _fail("lora_projection_paths_invalid")
    if any(path.rsplit(".", 1)[-1] not in targets for path in projections):
        _fail("lora_projection_target_mismatch")
    wrapped = _integer(
        value["wrapped_projections"],
        role="lora_wrapped_projections",
        minimum=1,
        maximum=MAX_TENSORS // 2,
    )
    if wrapped != len(projections):
        _fail("lora_projection_count_mismatch")
    trainable_params = _integer(
        value["trainable_params"],
        role="lora_trainable_params",
        minimum=1,
        maximum=1 << 60,
    )
    return {
        "rank": rank,
        "targets": list(targets),
        "wrapped_projections": wrapped,
        "projection_paths": list(projections),
        "trainable_params": trainable_params,
    }


def _validate_tensors(
    expected: Any,
    actual: Iterable[TensorIdentity | Mapping[str, Any]],
    *,
    lora: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(expected, list) or not expected or len(expected) > MAX_TENSORS:
        _fail("tensor_inventory_invalid")
    try:
        expected_tensors = normalize_tensor_metadata(expected)
        actual_tensors = normalize_tensor_metadata(actual)
    except ValueError as exc:
        raise RecurrentGRPOAdapterIdentityError("tensor_inventory_invalid") from exc
    if expected_tensors != actual_tensors:
        _fail("tensor_metadata_mismatch")
    expected_keys = {
        f"{projection}.{suffix}"
        for projection in lora["projection_paths"]
        for suffix in ("lora_a", "lora_b")
    }
    if {tensor.key for tensor in actual_tensors} != expected_keys:
        _fail("tensor_topology_mismatch")
    by_key = {tensor.key: tensor for tensor in actual_tensors}
    trainable_params = 0
    for projection in lora["projection_paths"]:
        left = by_key[f"{projection}.lora_a"]
        right = by_key[f"{projection}.lora_b"]
        if (
            len(left.shape) != 2
            or len(right.shape) != 2
            or left.shape[1] != lora["rank"]
            or right.shape[0] != lora["rank"]
            or left.dtype != right.dtype
        ):
            _fail("tensor_lora_shape_invalid")
        trainable_params += math.prod(left.shape) + math.prod(right.shape)
    if trainable_params != lora["trainable_params"]:
        _fail("tensor_trainable_params_mismatch")
    return [tensor.to_dict() for tensor in actual_tensors]


def _validate_step_receipts(
    value: Any,
    *,
    steps: int,
    optimizer_updates: int,
    group_size: int,
    execution_spec_sha256: str,
    trajectory_credit_enabled: bool,
    trajectory_shaping_weight: float,
    advantage_clip: float,
) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != steps:
        _fail("step_receipt_count_mismatch")
    update_count = 0
    previous_policy_after: str | None = None
    final_policy_after: str | None = None
    for index, raw_step in enumerate(value, start=1):
        verified_transition = (
            isinstance(raw_step, Mapping)
            and raw_step.get("schema") == VERIFIED_TRANSITION_STEP_SCHEMA
        )
        if verified_transition:
            try:
                step = validate_verified_transition_step_receipt(
                    raw_step,
                    group_size=group_size,
                    execution_spec_sha256=execution_spec_sha256,
                )
            except ValueError as exc:
                _fail(str(exc))
            if trajectory_credit_enabled:
                _fail("verified_transition_trajectory_credit_forbidden")
        else:
            step = _exact(
                raw_step,
                set(STEP_RECEIPT_KEYS),
                role="step_receipt",
            )
        if (
            step.get("step") != index
            or not isinstance(step.get("task_id"), str)
            or not step["task_id"]
            or type(step.get("sample_seed")) is not int
            or step.get("execution_spec_sha256") != execution_spec_sha256
            or step.get("step_kind")
            not in {
                "optimizer_update",
                "degenerate_group",
                "verified_optimizer_update",
                "verified_rejected_group",
            }
        ):
            _fail("step_receipt_identity_invalid")
        samples = step.get("samples")
        if not isinstance(samples, list) or len(samples) != group_size:
            _fail("step_receipt_group_invalid")
        if not verified_transition:
            try:
                validate_step_reward_channels(
                    step,
                    group_size=group_size,
                    trajectory_credit_enabled=trajectory_credit_enabled,
                    shaping_weight=trajectory_shaping_weight,
                    advantage_clip=advantage_clip,
                )
            except RecurrentGRPOArtifactSchemaError as exc:
                _fail(exc.code)
        policy_at_sampling: str | None = None
        for sample in samples:
            if not isinstance(sample, Mapping):
                _fail("sample_receipt_invalid")
            activation = sample.get("cached_recurrence_adapter")
            try:
                from core.brain.llm.latent_cortex.runtime_integrity import (
                    runtime_integrity_safe,
                )

                measured_runtime_safe = runtime_integrity_safe(
                    sample.get("cached_runtime_integrity"),
                    require_worker=False,
                    expected_episode_id=str(sample.get("episode_id") or ""),
                    expected_input_tokens_sha256=str(sample.get("prompt_tokens_sha256") or ""),
                )
            # not a failure: without the integrity module nothing can be called safe, and
            # False is the refusing direction.
            except ImportError:
                measured_runtime_safe = False
            if not isinstance(activation, Mapping):
                _fail("sample_behavior_not_admitted")
            if (
                sample.get("schema") != "aura.recurrent_sampling_behavior.v4"
                or not isinstance(sample.get("episode_id"), str)
                or not sample["episode_id"]
                or sample.get("behavior_admitted") is not True
                or sample.get("execution_spec_sha256") != execution_spec_sha256
                or measured_runtime_safe is not True
                or sample.get("cached_nonparametric_memory_status") != "disabled_by_policy"
                or activation.get("schema") != "aura.recurrence_adapter_activation.v1"
                or activation.get("active") is not True
                or activation.get("scope") != "latent_slots_only"
                or type(activation.get("calls")) is not int
                or activation["calls"] <= 0
                or type(activation.get("adapted_positions")) is not int
                or activation["adapted_positions"] <= 0
                or type(activation.get("observed_positions")) is not int
                or activation["observed_positions"] < activation["adapted_positions"]
            ):
                _fail("sample_behavior_not_admitted")
            sample_policy = _sha(sample.get("policy_sha256"), role="sample_policy")
            if policy_at_sampling is None:
                policy_at_sampling = sample_policy
            elif sample_policy != policy_at_sampling:
                _fail("sample_policy_group_mismatch")
        if verified_transition and policy_at_sampling != step.get("policy_before_sha256"):
            _fail("verified_step_sampling_policy_mismatch")
        if previous_policy_after is not None and policy_at_sampling != previous_policy_after:
            _fail("step_policy_chain_mismatch")
        policy_after = _sha(step.get("policy_after_sha256"), role="policy_after")
        update = step.get("update")
        if step["step_kind"] == "optimizer_update":
            if (
                not isinstance(update, Mapping)
                or update.get("schema") != "aura.recurrent_grpo.v1"
                or update.get("has_gradient") is not True
            ):
                _fail("optimizer_update_receipt_invalid")
            update_count += 1
        elif step["step_kind"] == "verified_optimizer_update":
            if (
                not isinstance(update, Mapping)
                or update.get("schema") != "aura.verified_transition.update_receipt.v1"
                or update.get("optimizer_update_count") != 1
            ):
                _fail("verified_optimizer_update_receipt_invalid")
            update_count += 1
        elif update is not None:
            _fail("degenerate_step_has_update")
        previous_policy_after = policy_after
        final_policy_after = policy_after
    if update_count != optimizer_updates:
        _fail("optimizer_update_count_mismatch")
    return {
        "step_receipt_count": len(value),
        "optimizer_update_count": update_count,
        "final_policy_sha256": final_policy_after,
    }


def validate_recurrent_grpo_adapter_identity(
    manifest: Mapping[str, Any] | bytes,
    *,
    adapter_id: str,
    actual_base_checkpoint: Mapping[str, Any],
    actual_model_behavior_bundle: Mapping[str, Any],
    actual_personality_adapter: Mapping[str, Any],
    actual_runtime_environment: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    tensor_metadata: Iterable[TensorIdentity | Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one complete recurrent-GRPO training bundle model-free."""

    manifest_bytes = (
        manifest if isinstance(manifest, bytes) else canonical_json_bytes(dict(manifest)) + b"\n"
    )
    parsed = strict_json_loads(manifest_bytes, role="manifest")
    parsed = dict(
        _exact(
            parsed,
            {
                "schema",
                "adapter_id",
                "training_method",
                "base_checkpoint",
                "model_behavior_bundle",
                "personality_adapter",
                "training_runtime",
                *BINDING_ROLES,
                "protocol_sha256",
                "dataset_sha256",
                "execution_spec_sha256",
                "sources",
                "lora",
                "tensors",
            },
            role="manifest",
        )
    )
    if parsed["schema"] != MANIFEST_SCHEMA:
        _fail("manifest_schema_unsupported")
    if (
        not isinstance(parsed["adapter_id"], str)
        or _IDENTIFIER_RE.fullmatch(parsed["adapter_id"]) is None
        or parsed["adapter_id"] != adapter_id
    ):
        _fail("adapter_id_mismatch")
    if parsed["training_method"] != TRAINING_METHOD:
        _fail("training_method_mismatch")
    for declared, actual, role in (
        (parsed["base_checkpoint"], actual_base_checkpoint, "base_checkpoint"),
        (
            parsed["model_behavior_bundle"],
            actual_model_behavior_bundle,
            "model_behavior_bundle",
        ),
        (
            parsed["personality_adapter"],
            actual_personality_adapter,
            "personality_adapter",
        ),
        (
            parsed["training_runtime"],
            actual_runtime_environment,
            "training_runtime",
        ),
    ):
        if declared != dict(actual):
            _fail(f"{role}_mismatch")

    binding_items = declared_bindings(parsed)
    bindings = {role: binding for role, binding in binding_items}
    payloads = {
        role: _verify_artifact(binding, artifacts, role=role) for role, binding in binding_items
    }
    if payloads["adapter"] != payloads["adapter_alias"]:
        _fail("adapter_alias_mismatch")

    protocol = strict_json_loads(payloads["training_protocol"], role="training_protocol")
    receipt = strict_json_loads(payloads["training_receipt"], role="training_receipt")
    dataset = strict_json_loads(payloads["dataset_manifest"], role="dataset_manifest")
    loader = strict_json_loads(payloads["loader_config"], role="loader_config")
    spec_payload = strict_json_loads(payloads["execution_spec"], role="execution_spec")
    try:
        spec = RLCExecutionSpec.from_dict(spec_payload)
    except (TypeError, ValueError) as exc:
        raise RecurrentGRPOAdapterIdentityError("execution_spec_invalid") from exc

    protocol = dict(
        _exact(
            protocol,
            {
                "schema",
                "adapter_id",
                "model_path",
                "base_checkpoint",
                "model_behavior",
                "personality_adapter",
                "runtime",
                "dataset_sha256",
                "sources",
                "training",
            },
            role="training_protocol",
        )
    )
    protocol_schema = protocol.get("schema")
    if protocol_schema == TRAINING_PROTOCOL_SCHEMA:
        training_keys = set(PROTOCOL_TRAINING_KEYS)
    elif protocol_schema == PROTOCOL_SCHEMA_V8:
        training_keys = set(PROTOCOL_TRAINING_KEYS_V8)
    elif protocol_schema in {PROTOCOL_SCHEMA_V6, PROTOCOL_SCHEMA_V7}:
        training_keys = set(PROTOCOL_TRAINING_KEYS_V8) - {"max_invocation_steps"}
    elif protocol_schema == PROTOCOL_SCHEMA_V5:
        training_keys = set(PROTOCOL_TRAINING_KEYS_V5)
    else:
        _fail("training_protocol_schema_invalid")
    training = _exact(
        protocol["training"],
        training_keys,
        role="training_parameters",
    )
    trajectory_credit_enabled = training.get("trajectory_credit")
    trajectory_shaping_weight = training.get("trajectory_shaping_weight")
    min_signal_groups = training.get("min_signal_groups")
    provider_contract_sha256 = training.get("verified_transition_provider_contract_sha256")
    lora_initialization_seed = training.get("lora_initialization_seed")
    advantage_clip = training.get("advantage_clip", 4.0)
    max_invocation_steps = training.get("max_invocation_steps", 0)
    warm_start_contract_sha256 = training.get("warm_start_contract_sha256")
    trajectory_group_config = None
    trajectory_group_config_sha256 = None
    if protocol_schema in {
        TRAINING_PROTOCOL_SCHEMA,
        PROTOCOL_SCHEMA_V8,
        PROTOCOL_SCHEMA_V7,
    }:
        trajectory_group_config = training.get("verified_trajectory_config")
        trajectory_group_config_sha256 = training.get("verified_trajectory_config_sha256")
        if trajectory_group_config is None:
            if trajectory_group_config_sha256 is not None:
                _fail("verified_trajectory_config_binding_invalid")
        else:
            try:
                validated_trajectory_config = VerifiedTrajectoryGroupConfig.from_dict(
                    trajectory_group_config
                )
            except (TypeError, ValueError) as exc:
                raise RecurrentGRPOAdapterIdentityError(
                    "verified_trajectory_config_invalid"
                ) from exc
            if (
                validated_trajectory_config.to_dict() != trajectory_group_config
                or not isinstance(trajectory_group_config_sha256, str)
                or _SHA256_RE.fullmatch(trajectory_group_config_sha256) is None
                or protocol_semantic_sha256(trajectory_group_config)
                != trajectory_group_config_sha256
            ):
                _fail("verified_trajectory_config_binding_invalid")
    if (
        protocol.get("adapter_id") != adapter_id
        or protocol.get("base_checkpoint") != parsed["base_checkpoint"]
        or protocol.get("model_behavior") != parsed["model_behavior_bundle"]
        or protocol.get("personality_adapter") != parsed["personality_adapter"]
        or protocol.get("runtime") != parsed["training_runtime"]
        or training.get("execution_mode") != "recurrent"
        or training.get("temperature") != 1.0
        or training.get("rng_strategy") != "stateless_sha256_step_seeded_v1"
        or training.get("execution_spec") != spec.to_dict()
        or training.get("execution_spec_sha256") != spec.sha256
        or not isinstance(provider_contract_sha256, str)
        or _SHA256_RE.fullmatch(provider_contract_sha256) is None
        or type(lora_initialization_seed) is not int
        or not 0 <= lora_initialization_seed <= (1 << 32) - 1
        or type(trajectory_credit_enabled) is not bool
        or isinstance(advantage_clip, bool)
        or not isinstance(advantage_clip, (int, float))
        or not math.isfinite(float(advantage_clip))
        or not 0.0 < float(advantage_clip) <= 100.0
        or isinstance(trajectory_shaping_weight, bool)
        or not isinstance(trajectory_shaping_weight, (int, float))
        or not math.isfinite(float(trajectory_shaping_weight))
        or not 0.0 <= float(trajectory_shaping_weight) <= 0.49
        or type(min_signal_groups) is not int
        or min_signal_groups < 1
        or type(max_invocation_steps) is not int
        or max_invocation_steps < 0
        or (
            protocol_schema == TRAINING_PROTOCOL_SCHEMA
            and warm_start_contract_sha256 is not None
            and (
                not isinstance(warm_start_contract_sha256, str)
                or _SHA256_RE.fullmatch(warm_start_contract_sha256) is None
            )
        )
    ):
        _fail("training_protocol_cross_binding_mismatch")
    protocol_sha256 = sha256_bytes(payloads["training_protocol"])
    dataset_sha256 = sha256_bytes(payloads["dataset_manifest"])
    if (
        parsed.get("protocol_sha256") != protocol_sha256
        or parsed.get("dataset_sha256") != dataset_sha256
        or parsed.get("execution_spec_sha256") != spec.sha256
        or protocol.get("dataset_sha256") != dataset_sha256
    ):
        _fail("training_digest_cross_binding_mismatch")

    _exact(dataset, {"schema", "seed", "train", "holdout"}, role="dataset_manifest")
    train_tasks = dataset.get("train")
    holdout_tasks = dataset.get("holdout")
    domains = training.get("domains")
    depths = training.get("depths")
    if (
        dataset.get("schema") != DATASET_SCHEMA
        or dataset.get("seed") != training.get("seed")
        or not isinstance(train_tasks, list)
        or not train_tasks
        or not isinstance(holdout_tasks, list)
        or not holdout_tasks
        or not isinstance(domains, list)
        or not domains
        or not isinstance(depths, list)
        or not depths
        or len(train_tasks) != len(domains) * len(depths) * training.get("train_per_cell", 0)
        or len(holdout_tasks) != len(domains) * len(depths) * training.get("holdout_per_cell", 0)
    ):
        _fail("dataset_protocol_mismatch")
    train_ids = {task.get("task_id") for task in train_tasks if isinstance(task, Mapping)}
    holdout_ids = {task.get("task_id") for task in holdout_tasks if isinstance(task, Mapping)}
    if (
        len(train_ids) != len(train_tasks)
        or len(holdout_ids) != len(holdout_tasks)
        or None in train_ids
        or None in holdout_ids
        or train_ids & holdout_ids
    ):
        _fail("dataset_split_identity_invalid")

    receipt_keys = {
        "schema",
        "adapter_id",
        "protocol_sha256",
        "dataset_sha256",
        "model",
        "config",
        "execution_mode",
        "execution_spec",
        "execution_spec_sha256",
        "domains",
        "depths",
        "train_tasks",
        "holdout_tasks",
        "steps",
        "optimizer_updates",
        "invocation_count",
        "termination",
        "learning_signal",
        "curriculum",
        "calibration",
        "baseline",
        "history",
        "step_receipts",
        "final",
        "adapter_decode_delta",
        "adapter_standard_decode_delta",
        "adapter_recurrent_decode_delta",
        "checkpoint",
        "verdict",
        "elapsed_minutes",
    }
    if protocol_schema in {
        TRAINING_PROTOCOL_SCHEMA,
        PROTOCOL_SCHEMA_V8,
        PROTOCOL_SCHEMA_V7,
    }:
        receipt_keys.add("training_adequacy")
    _exact(receipt, receipt_keys, role="training_receipt")
    config = _exact(
        receipt.get("config"),
        {
            "schema",
            "group_size",
            "kl_coefficient",
            "advantage_clip",
            "max_degenerate_fraction",
        },
        role="receipt_config",
    )
    group_size = _integer(training.get("group_size"), role="group_size", minimum=2, maximum=4096)
    advantage_clip = config.get("advantage_clip")
    kl_coefficient = config.get("kl_coefficient")
    max_degenerate_fraction = config.get("max_degenerate_fraction")
    if (
        config.get("schema") != GRPO_SCHEMA
        or config.get("group_size") != group_size
        or isinstance(kl_coefficient, bool)
        or not isinstance(kl_coefficient, (int, float))
        or not math.isfinite(float(kl_coefficient))
        or float(kl_coefficient) != training.get("kl_coefficient")
        or isinstance(advantage_clip, bool)
        or not isinstance(advantage_clip, (int, float))
        or not math.isfinite(float(advantage_clip))
        or not 0.0 < float(advantage_clip) <= 100.0
        or isinstance(max_degenerate_fraction, bool)
        or not isinstance(max_degenerate_fraction, (int, float))
        or not math.isfinite(float(max_degenerate_fraction))
        or not 0.0 <= float(max_degenerate_fraction) <= 1.0
    ):
        _fail("receipt_config_cross_binding_mismatch")
    model = _exact(
        receipt.get("model"), {"path", "base_checkpoint", "behavior"}, role="receipt_model"
    )
    termination = _exact(
        receipt.get("termination"),
        {"reason", "completed_budget", "signal"},
        role="termination",
    )
    verdict = _exact(
        receipt.get("verdict"),
        {
            "had_signal",
            "point_estimate_improved",
            "causal_gain_proven",
            "causal_gain_blocker",
            "diagnosis",
        },
        role="verdict",
    )
    steps = _integer(receipt.get("steps"), role="training_steps", minimum=1, maximum=100_000_000)
    max_steps = _integer(
        training.get("max_steps"), role="training_max_steps", minimum=1, maximum=100_000_000
    )
    optimizer_updates = _integer(
        receipt.get("optimizer_updates"),
        role="optimizer_updates",
        minimum=1,
        maximum=steps,
    )
    if (
        receipt.get("schema") != TRAINING_RECEIPT_SCHEMA
        or receipt.get("adapter_id") != adapter_id
        or receipt.get("protocol_sha256") != protocol_sha256
        or receipt.get("dataset_sha256") != dataset_sha256
        or model.get("base_checkpoint") != parsed["base_checkpoint"]
        or model.get("behavior") != parsed["model_behavior_bundle"]
        or receipt.get("execution_mode") != "recurrent"
        or receipt.get("execution_spec") != spec.to_dict()
        or receipt.get("execution_spec_sha256") != spec.sha256
        or receipt.get("domains") != domains
        or receipt.get("depths") != depths
        or receipt.get("train_tasks") != len(train_tasks)
        or receipt.get("holdout_tasks") != len(holdout_tasks)
        or steps != max_steps
        or termination.get("reason") != "max_steps"
        or termination.get("completed_budget") is not True
        or termination.get("signal") is not None
        or verdict.get("causal_gain_proven") is not False
        or verdict.get("causal_gain_blocker")
        != "requires fresh powered base/adapter x standard/RLC factorial gate"
    ):
        _fail("training_receipt_cross_binding_mismatch")
    step_summary = _validate_step_receipts(
        receipt.get("step_receipts"),
        steps=steps,
        optimizer_updates=optimizer_updates,
        group_size=group_size,
        execution_spec_sha256=spec.sha256,
        trajectory_credit_enabled=trajectory_credit_enabled,
        trajectory_shaping_weight=float(trajectory_shaping_weight),
        advantage_clip=float(advantage_clip),
    )
    if protocol_schema in {
        TRAINING_PROTOCOL_SCHEMA,
        PROTOCOL_SCHEMA_V8,
        PROTOCOL_SCHEMA_V7,
    }:
        history = receipt.get("history")
        learning_signal = receipt.get("learning_signal")
        if not isinstance(history, list) or not isinstance(learning_signal, Mapping):
            _fail("training_adequacy_evidence_missing")
        expected_adequacy = recurrent_training_adequacy_report(
            step_receipts=receipt["step_receipts"],
            scheduled_task_ids=[task["task_id"] for task in train_tasks],
            max_steps=max_steps,
            eval_every=_integer(
                training.get("eval_every"),
                role="training_eval_every",
                minimum=1,
                maximum=max_steps,
            ),
            evaluation_steps=[
                int(report["step"])
                for report in history
                if isinstance(report, Mapping) and type(report.get("step")) is int
            ],
            learning_signal=learning_signal,
        )
        if (
            receipt.get("training_adequacy") != expected_adequacy
            or expected_adequacy["admitted"] is not True
        ):
            _fail("training_adequacy_not_admitted")

    source_roles = required_source_roles(str(protocol_schema))
    sources = _exact(parsed["sources"], set(source_roles), role="sources")
    protocol_sources = _exact(
        protocol["sources"], set(source_roles), role="protocol_sources"
    )
    normalized_sources: dict[str, dict[str, Any]] = {}
    for role in sorted(source_roles):
        source = _exact(
            sources[role],
            {"origin_path", "snapshot_path", "sha256", "size_bytes"},
            role=f"source_{role}",
        )
        protocol_source = _exact(
            protocol_sources[role],
            {"path", "sha256", "size_bytes"},
            role=f"protocol_source_{role}",
        )
        if (
            _relative_path(source["origin_path"], role=f"source_{role}_origin")
            != protocol_source["path"]
            or source["sha256"] != protocol_source["sha256"]
            or source["size_bytes"] != protocol_source["size_bytes"]
        ):
            _fail(f"source_{role}_protocol_mismatch")
        _verify_artifact(bindings[f"source_{role}"], artifacts, role=f"source_{role}")
        normalized_sources[role] = dict(source)

    lora = _normalize_lora(parsed["lora"])
    target_string = training.get("lora_targets")
    if (
        lora["rank"] != training.get("lora_rank")
        or not isinstance(target_string, str)
        or lora["targets"] != [part.strip() for part in target_string.split(",")]
    ):
        _fail("lora_protocol_mismatch")
    _exact(
        loader,
        {
            "schema",
            "fine_tune_type",
            "loader",
            "model",
            "num_layers",
            "wrapped_projection_count",
            "lora_parameters",
            "execution_spec_sha256",
            "training_method",
        },
        role="loader_config",
    )
    loader_lora = _exact(
        loader.get("lora_parameters"),
        {"rank", "scale", "dropout", "keys"},
        role="loader_lora",
    )
    unique_layers = {int(path.split(".")[2]) for path in lora["projection_paths"]}
    if (
        loader.get("schema") != LOADER_CONFIG_SCHEMA
        or loader.get("fine_tune_type") != "recurrent_grpo_scoped_lora"
        or loader.get("loader") != "aura_custom_loader_required"
        or loader.get("model") != protocol.get("model_path")
        or loader.get("num_layers") != len(unique_layers)
        or loader.get("wrapped_projection_count") != lora["wrapped_projections"]
        or loader.get("execution_spec_sha256") != spec.sha256
        or loader.get("training_method") != TRAINING_METHOD
        or loader_lora
        != {"rank": lora["rank"], "scale": 20.0, "dropout": 0.0, "keys": lora["targets"]}
    ):
        _fail("loader_config_cross_binding_mismatch")
    supplied_tensor_metadata = normalize_tensor_metadata(tensor_metadata)
    frozen_tensor_metadata = normalize_tensor_metadata(
        tensor_metadata_from_safetensors(payloads["adapter"])
    )
    if supplied_tensor_metadata != frozen_tensor_metadata:
        _fail("adapter_tensor_metadata_source_mismatch")
    tensors = _validate_tensors(parsed["tensors"], frozen_tensor_metadata, lora=lora)
    frozen_policy_sha256 = recurrent_policy_sha256_from_safetensors(
        payloads["adapter"],
        execution_spec_sha256=spec.sha256,
    )
    if step_summary["final_policy_sha256"] != frozen_policy_sha256:
        _fail("final_policy_adapter_mismatch")

    completion_raw = artifacts.get("training_completion.json")
    if not isinstance(completion_raw, bytes):
        _fail("training_completion_bytes_missing")
    completion = strict_json_loads(completion_raw, role="training_completion")
    _exact(
        completion,
        {
            "schema",
            "complete",
            "halt_reason",
            "step",
            "optimizer_updates",
            "adapter_sha256",
            "receipt_sha256",
            "protocol_sha256",
            "execution_spec_sha256",
            "manifest_sha256",
        },
        role="training_completion",
    )
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("complete") is not True
        or completion.get("halt_reason") != "max_steps"
        or completion.get("step") != steps
        or completion.get("optimizer_updates") != optimizer_updates
        or completion.get("adapter_sha256") != bindings["adapter"]["sha256"]
        or completion.get("receipt_sha256") != bindings["training_receipt"]["sha256"]
        or completion.get("protocol_sha256") != protocol_sha256
        or completion.get("execution_spec_sha256") != spec.sha256
        or completion.get("manifest_sha256") != manifest_sha256
    ):
        _fail("training_completion_mismatch")

    identity_material = {
        "schema": "aura.recurrent_grpo_adapter_identity.v1",
        "adapter_id": adapter_id,
        "training_method": TRAINING_METHOD,
        "base_checkpoint": parsed["base_checkpoint"],
        "model_behavior_bundle": parsed["model_behavior_bundle"],
        "personality_adapter": parsed["personality_adapter"],
        "training_runtime": parsed["training_runtime"],
        "bindings": {role: bindings[role] for role in BINDING_ROLES},
        "training_completion_sha256": sha256_bytes(completion_raw),
        "sources": normalized_sources,
        "lora": lora,
        "tensors": tensors,
        "step_summary": step_summary,
    }
    return {
        "schema": IDENTITY_RECEIPT_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "composite_identity_sha256": sha256_bytes(canonical_json_bytes(identity_material)),
        "adapter_id": adapter_id,
        "training_method": TRAINING_METHOD,
        "objective_name": OBJECTIVE_NAME,
        "base_checkpoint_fingerprint": parsed["base_checkpoint"]["fingerprint"],
        "model_behavior_bundle_sha256": parsed["model_behavior_bundle"]["bundle_sha256"],
        "personality_adapter_bundle_sha256": parsed["personality_adapter"]["bundle_sha256"],
        "training_runtime_identity_sha256": parsed["training_runtime"]["identity_sha256"],
        "adapter_sha256": bindings["adapter"]["sha256"],
        "training_receipt_sha256": bindings["training_receipt"]["sha256"],
        "training_protocol_sha256": protocol_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": spec.sha256,
        "training_completion_sha256": sha256_bytes(completion_raw),
        "rank": lora["rank"],
        "targets": lora["targets"],
        "wrapped_projection_count": lora["wrapped_projections"],
        "tensor_count": len(tensors),
        "tensor_metadata_sha256": sha256_bytes(canonical_json_bytes(tensors)),
        "steps": steps,
        "optimizer_updates": optimizer_updates,
        "final_policy_sha256": frozen_policy_sha256,
        "causal_gain_proven": False,
        "complete": True,
    }


def validate_recurrent_grpo_adapter_identity_with_verified_transitions(
    manifest: Mapping[str, Any] | bytes,
    *,
    adapter_id: str,
    actual_base_checkpoint: Mapping[str, Any],
    actual_model_behavior_bundle: Mapping[str, Any],
    actual_personality_adapter: Mapping[str, Any],
    actual_runtime_environment: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    tensor_metadata: Iterable[TensorIdentity | Mapping[str, Any]],
    transition_campaign_ledger: Any,
    transition_policy: Any,
    transition_groups: Iterable[Any],
) -> dict[str, Any]:
    """Validate adapter identity and replay every verified mutation source."""

    identity = validate_recurrent_grpo_adapter_identity(
        manifest,
        adapter_id=adapter_id,
        actual_base_checkpoint=actual_base_checkpoint,
        actual_model_behavior_bundle=actual_model_behavior_bundle,
        actual_personality_adapter=actual_personality_adapter,
        actual_runtime_environment=actual_runtime_environment,
        artifacts=artifacts,
        tensor_metadata=tensor_metadata,
    )
    from core.learning.recurrence_native_objective_v2 import (
        INTERVENTION_MEASUREMENT_TRUST_BOUNDARY,
    )
    from core.learning.verified_transition_training_evidence import (
        VERIFIED_EXTERNALLY_REPLAYED_INTERVENTION_TRAINING_EVIDENCE_SCHEMA,
        VERIFIED_INTERVENTION_TRAINING_EVIDENCE_SCHEMA,
        VERIFIED_TRAINING_EVIDENCE_SCHEMA,
        validate_verified_transition_training_evidence,
    )

    parsed_manifest = (
        strict_json_loads(manifest, role="verified_identity_manifest")
        if isinstance(manifest, bytes)
        else dict(manifest)
    )
    bindings = dict(declared_bindings(parsed_manifest))
    training_protocol = strict_json_loads(
        _verify_artifact(
            bindings["training_protocol"],
            artifacts,
            role="verified_identity_training_protocol",
        ),
        role="verified_identity_training_protocol",
    )
    training_parameters = training_protocol.get("training")
    if not isinstance(training_parameters, Mapping):
        _fail("verified_identity_training_protocol_invalid")
    try:
        execution_spec = RLCExecutionSpec.from_dict(training_parameters["execution_spec"])
        raw_trajectory_config = training_parameters.get("verified_trajectory_config")
        trajectory_group_config = (
            VerifiedTrajectoryGroupConfig.from_dict(raw_trajectory_config)
            if raw_trajectory_config is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrentGRPOAdapterIdentityError(
            "verified_identity_training_protocol_invalid"
        ) from exc
    advantage_clip = training_parameters.get("advantage_clip")
    groups = tuple(transition_groups)
    evidence = validate_verified_transition_training_evidence(
        transition_campaign_ledger,
        policy=transition_policy,
        groups=groups,
        execution_spec=execution_spec,
        trajectory_group_config=trajectory_group_config,
        advantage_clip=advantage_clip,
    )
    intervention_evidence = bool(
        trajectory_group_config is not None
        and trajectory_group_config.intervention_config is not None
    )
    expected_evidence_schema = (
        {
            VERIFIED_INTERVENTION_TRAINING_EVIDENCE_SCHEMA,
            VERIFIED_EXTERNALLY_REPLAYED_INTERVENTION_TRAINING_EVIDENCE_SCHEMA,
        }
        if intervention_evidence
        else {VERIFIED_TRAINING_EVIDENCE_SCHEMA}
    )
    if (
        evidence.get("schema") not in expected_evidence_schema
        or evidence.get("source_artifacts_replayed") is not True
        or evidence.get("legacy_scalar_reward_path_used") is not False
        or (
            intervention_evidence
            and (
                evidence.get("measurement_trust_boundary")
                != INTERVENTION_MEASUREMENT_TRUST_BOUNDARY
                or evidence.get("external_policy_state_replayed")
                is not (
                    evidence.get("schema")
                    == VERIFIED_EXTERNALLY_REPLAYED_INTERVENTION_TRAINING_EVIDENCE_SCHEMA
                )
            )
        )
        or identity.get("optimizer_updates") != evidence.get("optimizer_update_count")
        or identity.get("final_policy_sha256") != evidence.get("final_policy_sha256")
    ):
        _fail("verified_transition_identity_cross_binding_mismatch")

    training_receipt = strict_json_loads(
        _verify_artifact(
            bindings["training_receipt"],
            artifacts,
            role="verified_identity_training_receipt",
        ),
        role="verified_identity_training_receipt",
    )
    raw_steps = training_receipt.get("step_receipts")
    if not isinstance(raw_steps, list) or not raw_steps:
        _fail("verified_transition_step_chain_missing")
    campaign = transition_campaign_ledger.validate_closed(policy=transition_policy)
    close_payload = campaign.get("close_payload")
    if (
        not isinstance(close_payload, Mapping)
        or close_payload.get("group_count") != len(raw_steps)
        or close_payload.get("group_count") != identity.get("steps")
    ):
        _fail("verified_transition_campaign_step_count_mismatch")
    replay_by_sequence = {group.sequence: group for group in groups}
    if len(replay_by_sequence) != len(groups):
        _fail("verified_transition_replay_sequence_duplicated")
    updated_sequences: list[int] = []
    rejected_count = 0
    previous_policy: str | None = None
    chain_rows: list[dict[str, Any]] = []
    group_size = int(training_receipt["config"]["group_size"])
    execution_spec_sha256 = str(training_receipt["execution_spec_sha256"])
    for sequence, raw_step in enumerate(raw_steps):
        if (
            not isinstance(raw_step, Mapping)
            or raw_step.get("schema") != VERIFIED_TRANSITION_STEP_SCHEMA
        ):
            _fail("verified_transition_identity_legacy_step_forbidden")
        try:
            step = validate_verified_transition_step_receipt(
                raw_step,
                group_size=group_size,
                execution_spec_sha256=execution_spec_sha256,
            )
        except ValueError as exc:
            _fail(str(exc))
        start, terminal = transition_campaign_ledger.group_records(
            sequence=sequence,
            policy=transition_policy,
        )
        start_manifest = start.get("group_manifest")
        if (
            step.get("campaign_sequence") != sequence
            or step.get("step") != sequence + 1
            or not isinstance(start_manifest, Mapping)
            or step.get("task_id") != start_manifest.get("task_id")
            or step.get("group_manifest_sha256") != start_manifest.get("manifest_sha256")
            or step.get("terminal") != terminal
            or step.get("reward_receipt_sha256") != terminal.get("reward_receipt_sha256")
            or (previous_policy is not None and step.get("policy_before_sha256") != previous_policy)
        ):
            _fail("verified_transition_ordered_step_binding_mismatch")
        previous_policy = str(step["policy_after_sha256"])
        if step["step_kind"] == "verified_optimizer_update":
            replay = replay_by_sequence.get(sequence)
            if replay is None:
                _fail("verified_transition_updated_replay_missing")
            if (
                replay.reward_receipt.get("receipt_sha256") != step.get("reward_receipt_sha256")
                or replay.group_manifest.get("manifest_sha256") != step.get("group_manifest_sha256")
                or replay.group_admission_receipt.get("receipt_sha256")
                != step.get("group_admission_sha256")
                or replay.update_receipt.get("receipt_sha256") != step.get("update_receipt_sha256")
                or dict(replay.update_receipt) != step.get("update")
                or [sample.receipt() for sample in replay.samples] != step.get("samples")
            ):
                _fail("verified_transition_updated_source_binding_mismatch")
            updated_sequences.append(sequence)
        elif step["step_kind"] == "verified_rejected_group":
            if sequence in replay_by_sequence or terminal.get("status") != "rejected":
                _fail("verified_transition_rejected_source_binding_mismatch")
            rejected_count += 1
        else:
            _fail("verified_transition_identity_step_kind_invalid")
        chain_rows.append(
            {
                "sequence": sequence,
                "step_receipt_sha256": step["receipt_sha256"],
                "group_start_sha256": start["receipt_sha256"],
                "group_terminal_sha256": terminal["receipt_sha256"],
                "reward_receipt_sha256": step["reward_receipt_sha256"],
                "group_admission_sha256": step["group_admission_sha256"],
                "update_receipt_sha256": step["update_receipt_sha256"],
                "policy_before_sha256": step["policy_before_sha256"],
                "policy_after_sha256": step["policy_after_sha256"],
            }
        )
    if (
        updated_sequences != evidence.get("updated_sequences")
        or set(replay_by_sequence) != set(updated_sequences)
        or chain_rows[0]["policy_before_sha256"] != evidence.get("initial_policy_sha256")
        or previous_policy != identity.get("final_policy_sha256")
    ):
        _fail("verified_transition_ordered_campaign_mismatch")
    verified_step_chain_sha256 = sha256_bytes(canonical_json_bytes(chain_rows))
    base_identity_sha256 = sha256_bytes(canonical_json_bytes(identity))
    evidence_sha256 = _sha(evidence.get("receipt_sha256"), role="verified_evidence")
    material = {
        "schema": VERIFIED_IDENTITY_RECEIPT_SCHEMA,
        "adapter_id": identity["adapter_id"],
        "base_identity": identity,
        "base_identity_sha256": base_identity_sha256,
        "verified_transition_evidence": evidence,
        "verified_transition_evidence_sha256": evidence_sha256,
        "verified_step_chain": chain_rows,
        "verified_step_chain_sha256": verified_step_chain_sha256,
        "verified_group_count": len(chain_rows),
        "rejected_group_count": rejected_count,
        "adapter_sha256": identity["adapter_sha256"],
        "execution_spec_sha256": identity["execution_spec_sha256"],
        "optimizer_updates": evidence["optimizer_update_count"],
        "initial_policy_sha256": evidence["initial_policy_sha256"],
        "final_policy_sha256": evidence["final_policy_sha256"],
        "proof_grade_mutation": True,
        "legacy_scalar_reward_path_used": False,
        "causal_gain_proven": False,
        "complete": True,
    }
    return validate_verified_recurrent_grpo_adapter_identity_receipt(
        {
            **material,
            "composite_identity_sha256": sha256_bytes(canonical_json_bytes(material)),
        }
    )


def validate_verified_recurrent_grpo_adapter_identity_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct a proof-grade identity from its embedded validated receipts."""

    required = {
        "schema",
        "adapter_id",
        "base_identity",
        "base_identity_sha256",
        "verified_transition_evidence",
        "verified_transition_evidence_sha256",
        "verified_step_chain",
        "verified_step_chain_sha256",
        "verified_group_count",
        "rejected_group_count",
        "adapter_sha256",
        "execution_spec_sha256",
        "optimizer_updates",
        "initial_policy_sha256",
        "final_policy_sha256",
        "proof_grade_mutation",
        "legacy_scalar_reward_path_used",
        "causal_gain_proven",
        "complete",
        "composite_identity_sha256",
    }
    receipt = dict(_exact(value, required, role="verified_identity_receipt"))
    base_required = {
        "schema",
        "manifest_sha256",
        "composite_identity_sha256",
        "adapter_id",
        "training_method",
        "objective_name",
        "base_checkpoint_fingerprint",
        "model_behavior_bundle_sha256",
        "personality_adapter_bundle_sha256",
        "training_runtime_identity_sha256",
        "adapter_sha256",
        "training_receipt_sha256",
        "training_protocol_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "training_completion_sha256",
        "rank",
        "targets",
        "wrapped_projection_count",
        "tensor_count",
        "tensor_metadata_sha256",
        "steps",
        "optimizer_updates",
        "final_policy_sha256",
        "causal_gain_proven",
        "complete",
    }
    base = dict(_exact(receipt.get("base_identity"), base_required, role="verified_base_identity"))
    for field in (
        "manifest_sha256",
        "composite_identity_sha256",
        "base_checkpoint_fingerprint",
        "model_behavior_bundle_sha256",
        "training_runtime_identity_sha256",
        "adapter_sha256",
        "training_receipt_sha256",
        "training_protocol_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "training_completion_sha256",
        "tensor_metadata_sha256",
        "final_policy_sha256",
    ):
        _sha(base.get(field), role=f"verified_base_{field}")
    personality = base.get("personality_adapter_bundle_sha256")
    _sha(personality, role="verified_base_personality", allow_empty=True)
    if (
        base.get("schema") != IDENTITY_RECEIPT_SCHEMA
        or base.get("adapter_id") != receipt.get("adapter_id")
        or base.get("training_method") != TRAINING_METHOD
        or base.get("objective_name") != OBJECTIVE_NAME
        or base.get("causal_gain_proven") is not False
        or base.get("complete") is not True
        or type(base.get("optimizer_updates")) is not int
        or base["optimizer_updates"] < 1
        or type(base.get("steps")) is not int
        or base["steps"] < base["optimizer_updates"]
        or type(base.get("rank")) is not int
        or base["rank"] < 1
        or not isinstance(base.get("targets"), list)
        or not base["targets"]
        or type(base.get("wrapped_projection_count")) is not int
        or base["wrapped_projection_count"] < 1
        or type(base.get("tensor_count")) is not int
        or base["tensor_count"] < 1
    ):
        _fail("verified_base_identity_invalid")

    from core.learning.verified_transition_training_evidence import (
        validate_verified_transition_training_evidence_receipt,
    )

    evidence = validate_verified_transition_training_evidence_receipt(
        receipt.get("verified_transition_evidence")
    )
    base_sha256 = sha256_bytes(canonical_json_bytes(base))
    evidence_sha256 = _sha(evidence.get("receipt_sha256"), role="verified_evidence")
    step_chain_sha256 = _sha(receipt.get("verified_step_chain_sha256"), role="verified_step_chain")
    step_chain = receipt.get("verified_step_chain")
    if not isinstance(step_chain, list) or not step_chain:
        _fail("verified_step_chain_invalid")
    expected_step_keys = {
        "sequence",
        "step_receipt_sha256",
        "group_start_sha256",
        "group_terminal_sha256",
        "reward_receipt_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
    }
    updated_rows: list[Mapping[str, Any]] = []
    previous_after: str | None = None
    for sequence, raw_row in enumerate(step_chain):
        row = _exact(raw_row, expected_step_keys, role="verified_step_chain_row")
        for field in (
            "step_receipt_sha256",
            "group_start_sha256",
            "group_terminal_sha256",
            "reward_receipt_sha256",
            "policy_before_sha256",
            "policy_after_sha256",
        ):
            _sha(row.get(field), role=f"verified_step_chain_{field}")
        admission = row.get("group_admission_sha256")
        update = row.get("update_receipt_sha256")
        if (admission is None) != (update is None):
            _fail("verified_step_chain_update_pair_invalid")
        if admission is not None:
            _sha(admission, role="verified_step_chain_admission")
            _sha(update, role="verified_step_chain_update")
            updated_rows.append(row)
        if (
            row.get("sequence") != sequence
            or (previous_after is not None and row.get("policy_before_sha256") != previous_after)
            or (
                admission is None
                and row.get("policy_before_sha256") != row.get("policy_after_sha256")
            )
        ):
            _fail("verified_step_chain_continuity_invalid")
        previous_after = str(row["policy_after_sha256"])
    if step_chain_sha256 != sha256_bytes(canonical_json_bytes(step_chain)):
        _fail("verified_step_chain_digest_mismatch")
    updated_sequences = [int(row["sequence"]) for row in updated_rows]
    updated_rewards = [row["reward_receipt_sha256"] for row in updated_rows]
    updated_admissions = [row["group_admission_sha256"] for row in updated_rows]
    updated_receipts = [row["update_receipt_sha256"] for row in updated_rows]
    material = {key: receipt[key] for key in required - {"composite_identity_sha256"}}
    if (
        receipt.get("schema") != VERIFIED_IDENTITY_RECEIPT_SCHEMA
        or receipt.get("base_identity_sha256") != base_sha256
        or receipt.get("verified_transition_evidence_sha256") != evidence_sha256
        or type(receipt.get("verified_group_count")) is not int
        or receipt["verified_group_count"] != base["steps"]
        or receipt["verified_group_count"] != len(step_chain)
        or type(receipt.get("rejected_group_count")) is not int
        or receipt["rejected_group_count"] < 0
        or receipt["rejected_group_count"] != base["steps"] - evidence["optimizer_update_count"]
        or updated_sequences != evidence["updated_sequences"]
        or updated_rewards != evidence["reward_receipt_sha256s"]
        or updated_admissions != evidence["group_admission_sha256s"]
        or updated_receipts != evidence["update_receipt_sha256s"]
        or step_chain[0]["policy_before_sha256"] != evidence["initial_policy_sha256"]
        or previous_after != evidence["final_policy_sha256"]
        or receipt.get("adapter_sha256") != base["adapter_sha256"]
        or receipt.get("execution_spec_sha256") != base["execution_spec_sha256"]
        or receipt.get("optimizer_updates") != base["optimizer_updates"]
        or receipt.get("optimizer_updates") != evidence["optimizer_update_count"]
        or receipt.get("initial_policy_sha256") != evidence["initial_policy_sha256"]
        or receipt.get("final_policy_sha256") != base["final_policy_sha256"]
        or receipt.get("final_policy_sha256") != evidence["final_policy_sha256"]
        or receipt.get("proof_grade_mutation") is not True
        or receipt.get("legacy_scalar_reward_path_used") is not False
        or receipt.get("causal_gain_proven") is not False
        or receipt.get("complete") is not True
        or receipt.get("composite_identity_sha256") != sha256_bytes(canonical_json_bytes(material))
    ):
        _fail("verified_identity_receipt_reconstruction_mismatch")
    return receipt


__all__ = [
    "BINDING_ROLES",
    "COMPLETION_SCHEMA",
    "DATASET_SCHEMA",
    "IDENTITY_RECEIPT_SCHEMA",
    "LOADER_CONFIG_SCHEMA",
    "MANIFEST_FILE",
    "MANIFEST_SCHEMA",
    "OBJECTIVE_NAME",
    "LEGACY_REQUIRED_SOURCE_ROLES",
    "REQUIRED_SOURCE_ROLES",
    "TRAINING_METHOD",
    "TRAINING_PROTOCOL_SCHEMA",
    "TRAINING_RECEIPT_SCHEMA",
    "VERIFIED_IDENTITY_RECEIPT_SCHEMA",
    "RecurrentGRPOAdapterIdentityError",
    "artifact_binding",
    "canonical_json_bytes",
    "declared_bindings",
    "required_source_roles",
    "sha256_bytes",
    "recurrent_policy_sha256_from_safetensors",
    "tensor_metadata_from_safetensors",
    "strict_json_loads",
    "validate_recurrent_grpo_adapter_identity",
    "validate_recurrent_grpo_adapter_identity_with_verified_transitions",
    "validate_verified_recurrent_grpo_adapter_identity_receipt",
]
