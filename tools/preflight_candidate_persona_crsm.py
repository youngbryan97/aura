#!/usr/bin/env python3
"""Build a model-free readiness receipt for candidate persona/CRSM training.

The receipt binds the training corpus, candidate checkpoint descriptor, LoRA
topology, and all resume locations before a trainer is allowed to own Metal.
It reads JSON and safetensors headers only. It never imports MLX or loads model
weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREFLIGHT_SCHEMA = "aura.candidate_persona_crsm_preflight.v1"
ADAPTER_BINDING_SCHEMA = "aura.candidate_persona_adapter_binding.v1"
EXPECTED_DESCRIPTOR_SHA256 = (
    "79b8369af238a0d6ab197ecaea6002f9b0224adf4ae4a728bfc73e62b1e0703e"
)
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
LORA_RANK = 32
LORA_SCALE = 20.0
LORA_DROPOUT = 0.0
SAFETENSOR_DTYPES = {
    "BOOL": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


class CandidatePreflightError(ValueError):
    """One stable preflight-contract failure."""


def _fail(code: str) -> None:
    raise CandidatePreflightError(code)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CandidatePreflightError("canonical_json_invalid") from exc


def _document_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_bytes(raw: bytes, *, role: str) -> Any:
    if not raw or len(raw) > MAX_JSON_BYTES:
        _fail(f"{role}_size_invalid")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _fail(f"{role}_number_invalid")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except CandidatePreflightError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CandidatePreflightError(f"{role}_json_invalid") from exc


def _read_json(path: Path, *, role: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CandidatePreflightError(f"{role}_unreadable") from exc
    return _strict_json_bytes(raw, role=role)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        _fail("artifact_not_file")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": stat.st_size,
    }


def _safe_child(root: Path, relative: str, *, role: str) -> Path:
    candidate = PurePosixPath(relative)
    if (
        not relative
        or candidate.is_absolute()
        or str(candidate) != relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        _fail(f"{role}_path_invalid")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / Path(*candidate.parts)).resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        _fail(f"{role}_path_escape")
    return resolved


def _validate_descriptor(
    descriptor_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], Path]:
    descriptor = _read_json(descriptor_path, role="candidate_descriptor")
    required = {
        "schema",
        "canonical_path",
        "repository_id",
        "revision",
        "artifact_profile",
        "weight_identity",
        "behavior_identity",
        "descriptor_sha256",
    }
    if not isinstance(descriptor, dict) or set(descriptor) != required:
        _fail("candidate_descriptor_schema_invalid")
    material = dict(descriptor)
    claimed = material.pop("descriptor_sha256")
    if not _is_sha256(claimed) or claimed != _document_sha256(material):
        _fail("candidate_descriptor_digest_invalid")
    if claimed != expected_sha256:
        _fail("candidate_descriptor_not_admitted")
    model_root = Path(str(descriptor["canonical_path"])).expanduser().resolve(strict=True)
    if not model_root.is_dir():
        _fail("candidate_model_not_directory")

    behavior = descriptor.get("behavior_identity")
    if not isinstance(behavior, dict) or not isinstance(behavior.get("files"), list):
        _fail("candidate_behavior_identity_invalid")
    observed_files = []
    for index, record in enumerate(behavior["files"]):
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            _fail(f"candidate_behavior_file_{index}_schema_invalid")
        path = _safe_child(model_root, str(record["path"]), role="candidate_behavior")
        binding = _file_binding(path)
        if (
            binding["sha256"] != record["sha256"]
            or binding["size_bytes"] != record["size_bytes"]
        ):
            _fail(f"candidate_behavior_file_{index}_mismatch")
        observed_files.append(binding)
    if behavior.get("file_count") != len(observed_files):
        _fail("candidate_behavior_file_count_mismatch")
    return descriptor, model_root


def _integer(value: Any, *, role: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{role}_invalid")
    return value


def _projection(
    layer: int,
    relative: str,
    input_features: int,
    output_features: int,
    layer_type: str,
) -> dict[str, Any]:
    return {
        "path": f"language_model.model.layers.{layer}.{relative}",
        "relative_key": relative,
        "layer": layer,
        "layer_type": layer_type,
        "input_features": input_features,
        "output_features": output_features,
    }


def _module_specs(text_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    hidden = _integer(text_config.get("hidden_size"), role="hidden_size")
    intermediate = _integer(
        text_config.get("intermediate_size"), role="intermediate_size"
    )
    layers = _integer(text_config.get("num_hidden_layers"), role="num_hidden_layers")
    head_dim = _integer(text_config.get("head_dim"), role="head_dim")
    attention_heads = _integer(
        text_config.get("num_attention_heads"), role="num_attention_heads"
    )
    kv_heads = _integer(text_config.get("num_key_value_heads"), role="num_key_value_heads")
    key_heads = _integer(
        text_config.get("linear_num_key_heads"), role="linear_num_key_heads"
    )
    value_heads = _integer(
        text_config.get("linear_num_value_heads"), role="linear_num_value_heads"
    )
    key_head_dim = _integer(
        text_config.get("linear_key_head_dim"), role="linear_key_head_dim"
    )
    value_head_dim = _integer(
        text_config.get("linear_value_head_dim"), role="linear_value_head_dim"
    )
    layer_types = text_config.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != layers
        or any(kind not in {"linear_attention", "full_attention"} for kind in layer_types)
    ):
        _fail("layer_types_invalid")

    full_q = attention_heads * head_dim
    full_kv = kv_heads * head_dim
    linear_key = key_heads * key_head_dim
    linear_value = value_heads * value_head_dim
    specs: list[dict[str, Any]] = []
    for layer, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            specs.extend(
                [
                    _projection(
                        layer,
                        "linear_attn.in_proj_qkv",
                        hidden,
                        2 * linear_key + linear_value,
                        layer_type,
                    ),
                    _projection(
                        layer,
                        "linear_attn.in_proj_z",
                        hidden,
                        linear_value,
                        layer_type,
                    ),
                    _projection(
                        layer,
                        "linear_attn.in_proj_b",
                        hidden,
                        value_heads,
                        layer_type,
                    ),
                    _projection(
                        layer,
                        "linear_attn.in_proj_a",
                        hidden,
                        value_heads,
                        layer_type,
                    ),
                    _projection(
                        layer,
                        "linear_attn.out_proj",
                        linear_value,
                        hidden,
                        layer_type,
                    ),
                ]
            )
        else:
            specs.extend(
                [
                    _projection(layer, "self_attn.q_proj", hidden, full_q, layer_type),
                    _projection(layer, "self_attn.k_proj", hidden, full_kv, layer_type),
                    _projection(layer, "self_attn.v_proj", hidden, full_kv, layer_type),
                    _projection(layer, "self_attn.o_proj", full_q, hidden, layer_type),
                ]
            )
        specs.extend(
            [
                _projection(layer, "mlp.gate_proj", hidden, intermediate, layer_type),
                _projection(layer, "mlp.up_proj", hidden, intermediate, layer_type),
                _projection(layer, "mlp.down_proj", intermediate, hidden, layer_type),
            ]
        )
    return specs


def _target_topology(
    descriptor: Mapping[str, Any],
    model_root: Path,
    *,
    rank: int = LORA_RANK,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    config = _read_json(model_root / "config.json", role="candidate_config")
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        _fail("candidate_model_type_unsupported")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3_5_text":
        _fail("candidate_text_config_unsupported")
    profile = descriptor.get("artifact_profile")
    if not isinstance(profile, dict):
        _fail("candidate_profile_invalid")
    profile_checks = {
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "vocab_size": text_config.get("vocab_size"),
        "native_context_window": text_config.get("max_position_embeddings"),
        "layer_types": text_config.get("layer_types"),
    }
    for key, value in profile_checks.items():
        if profile.get(key) != value:
            _fail(f"candidate_profile_{key}_mismatch")

    specs = _module_specs(text_config)
    index = _read_json(
        model_root / "model.safetensors.index.json", role="candidate_weight_index"
    )
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        _fail("candidate_weight_index_invalid")
    required_weight_keys = {
        f"{spec['path']}.{suffix}"
        for spec in specs
        for suffix in ("weight", "scales", "biases")
    }
    missing = sorted(required_weight_keys - set(weight_map))
    rank_failures = [
        spec["path"]
        for spec in specs
        if rank > min(spec["input_features"], spec["output_features"])
    ]
    relative_keys = sorted({str(spec["relative_key"]) for spec in specs})
    layer_counts = Counter(str(spec["layer_type"]) for spec in specs)
    topology_sha256 = _document_sha256(specs)
    report = {
        "compatible": not missing and not rank_failures,
        "model_type": config["model_type"],
        "text_model_type": text_config["model_type"],
        "rank": rank,
        "scale": LORA_SCALE,
        "dropout": LORA_DROPOUT,
        "layers": text_config["num_hidden_layers"],
        "linear_attention_layers": text_config["layer_types"].count("linear_attention"),
        "full_attention_layers": text_config["layer_types"].count("full_attention"),
        "wrapped_projection_count": len(specs),
        "projection_count_by_layer_type": dict(sorted(layer_counts.items())),
        "relative_keys": relative_keys,
        "topology_sha256": topology_sha256,
        "missing_weight_keys": missing,
        "rank_incompatible_paths": rank_failures,
        "minimum_projection_dimension": min(
            min(spec["input_features"], spec["output_features"]) for spec in specs
        ),
        "mlx_lora_config": {
            "lora_parameters": {
                "rank": rank,
                "scale": LORA_SCALE,
                "dropout": LORA_DROPOUT,
                "keys": relative_keys,
            }
        },
    }
    return report, tuple(specs)


def _message_key(messages: Iterable[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "role": str(message["role"]).strip().casefold(),
            "content": " ".join(str(message["content"]).split()).casefold(),
        }
        for message in messages
    ]
    return _document_sha256(normalized)


def _inspect_message_dataset(path: Path) -> tuple[dict[str, Any], set[str]]:
    binding = _file_binding(path)
    invalid = Counter()
    identities: set[str] = set()
    duplicates = 0
    lines = 0
    with path.open("rb") as handle:
        for lines, raw in enumerate(handle, 1):
            try:
                value = _strict_json_bytes(raw, role=f"dataset_line_{lines}")
            except CandidatePreflightError as exc:
                invalid[str(exc)] += 1
                continue
            messages = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(messages, list) or not messages:
                invalid["messages_missing"] += 1
                continue
            normalized: list[Mapping[str, Any]] = []
            roles: list[str] = []
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                ):
                    invalid["message_schema_invalid"] += 1
                    normalized = []
                    break
                roles.append(message["role"])
                normalized.append(message)
            if not normalized:
                continue
            if "user" not in roles or roles[-1] != "assistant":
                invalid["conversation_shape_invalid"] += 1
                continue
            identity = _message_key(normalized)
            duplicates += int(identity in identities)
            identities.add(identity)
    report = {
        **binding,
        "lines": lines,
        "valid_records": len(identities) + duplicates,
        "unique_records": len(identities),
        "duplicate_records": duplicates,
        "invalid_records": sum(invalid.values()),
        "invalid_by_reason": dict(sorted(invalid.items())),
        "content_identity_sha256": _document_sha256(sorted(identities)),
    }
    return report, identities


def _dataset_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the location-independent identity material for one dataset."""

    return {
        key: report[key]
        for key in (
            "sha256",
            "size_bytes",
            "lines",
            "valid_records",
            "unique_records",
            "duplicate_records",
            "invalid_records",
            "content_identity_sha256",
        )
    }


def _inspect_crsm_source(path: Path) -> dict[str, Any]:
    binding = _file_binding(path)
    invalid = Counter()
    valid = 0
    lines = 0
    for lines, raw in enumerate(path.open("rb"), 1):
        try:
            value = _strict_json_bytes(raw, role=f"crsm_line_{lines}")
        except CandidatePreflightError as exc:
            invalid[str(exc)] += 1
            continue
        if not isinstance(value, dict):
            invalid["record_not_object"] += 1
            continue
        text = value.get("text")
        messages = value.get("messages")
        if not (
            (isinstance(text, str) and text.strip())
            or (isinstance(messages, list) and messages)
        ):
            invalid["content_missing"] += 1
            continue
        quality = value.get("_quality")
        if quality is not None and (
            isinstance(quality, bool)
            or not isinstance(quality, (int, float))
            or not math.isfinite(float(quality))
            or not 0.0 <= float(quality) <= 1.0
        ):
            invalid["quality_invalid"] += 1
            continue
        valid += 1
    return {
        **binding,
        "lines": lines,
        "valid_records": valid,
        "invalid_records": sum(invalid.values()),
        "invalid_by_reason": dict(sorted(invalid.items())),
    }


def _manifest_dataset_match(
    manifest_path: Path,
    *,
    train: Mapping[str, Any] | None,
    valid: Mapping[str, Any] | None,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"present": False, "current": False, "path": str(manifest_path)}
    manifest = _read_json(manifest_path, role="dataset_manifest")
    if not isinstance(manifest, dict):
        _fail("dataset_manifest_schema_invalid")
    output = manifest.get("output")
    output = output if isinstance(output, dict) else {}
    train_record = output.get("train") if isinstance(output.get("train"), dict) else {}
    valid_record = output.get("valid") if isinstance(output.get("valid"), dict) else {}
    output_current = train is not None and valid is not None
    if train is not None:
        output_current &= (
            train_record.get("sha256") == train.get("sha256")
            and train_record.get("lines") == train.get("lines")
        )
    if valid is not None:
        output_current &= (
            valid_record.get("sha256") == valid.get("sha256")
            and valid_record.get("lines") == valid.get("lines")
        )
    source_current = True
    if source is not None:
        source_current = (
            manifest.get("source_sha256") == source.get("sha256")
            and manifest.get("source_lines") == source.get("lines")
        )
    return {
        "present": True,
        "path": str(manifest_path.resolve()),
        "sha256": _sha256_file(manifest_path),
        "output_current": bool(output_current),
        "source_current": bool(source_current),
        "current": bool(output_current and source_current),
        "recorded_source_sha256": manifest.get("source_sha256"),
        "recorded_source_lines": manifest.get("source_lines"),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _run_paths(
    output_root: Path,
    *,
    descriptor_sha256: str,
    persona_identity_sha256: str,
    crsm_source_sha256: str,
    topology_sha256: str,
) -> dict[str, str]:
    material = {
        "model_descriptor_sha256": descriptor_sha256,
        "persona_dataset_identity_sha256": persona_identity_sha256,
        "crsm_source_sha256": crsm_source_sha256,
        "target_topology_sha256": topology_sha256,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "lora_dropout": LORA_DROPOUT,
    }
    run_id = _document_sha256(material)[:20]
    root = output_root.expanduser().resolve(strict=False)
    run_root = (root / descriptor_sha256[:16] / run_id).resolve(strict=False)
    if not _is_within(run_root, root):
        _fail("candidate_run_path_escape")
    persona = run_root / "persona"
    crsm = run_root / "crsm"
    return {
        "output_root": str(root),
        "run_id": run_id,
        "run_root": str(run_root),
        "persona_adapter_dir": str(persona / "adapter"),
        "persona_checkpoint_pattern": str(persona / "adapter" / "[0-9]*_adapters.safetensors"),
        "persona_resume_root": str(persona / "adapter"),
        "persona_final_adapter": str(persona / "adapter" / "adapters.safetensors"),
        "persona_binding": str(persona / "adapter" / "candidate_binding.json"),
        "persona_completion": str(persona / "completion.json"),
        "crsm_dataset_dir": str(crsm / "data"),
        "crsm_adapter_dir": str(crsm / "adapter"),
        "crsm_checkpoint_pattern": str(crsm / "adapter" / "[0-9]*_adapters.safetensors"),
        "crsm_resume_root": str(crsm / "adapter"),
        "crsm_final_adapter": str(crsm / "adapter" / "adapters.safetensors"),
        "crsm_binding": str(crsm / "adapter" / "candidate_binding.json"),
        "crsm_completion": str(crsm / "completion.json"),
        "fused_model_dir": str(run_root / "fused-model"),
    }


def _read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                _fail("adapter_tensor_header_missing")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 2 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                _fail("adapter_tensor_header_size_invalid")
            raw = handle.read(header_length)
    except OSError as exc:
        raise CandidatePreflightError("adapter_tensor_unreadable") from exc
    header = _strict_json_bytes(raw, role="adapter_tensor_header")
    if not isinstance(header, dict):
        _fail("adapter_tensor_header_invalid")
    data_size = size - 8 - header_length
    tensors: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    for key, value in header.items():
        if key == "__metadata__":
            if value is not None and (
                not isinstance(value, dict)
                or any(
                    not isinstance(meta_key, str) or not isinstance(meta_value, str)
                    for meta_key, meta_value in value.items()
                )
            ):
                _fail("adapter_tensor_metadata_invalid")
            continue
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, dict)
            or set(value) != {"dtype", "shape", "data_offsets"}
        ):
            _fail("adapter_tensor_entry_invalid")
        dtype = value["dtype"]
        shape = value["shape"]
        offsets = value["data_offsets"]
        if (
            dtype not in SAFETENSOR_DTYPES
            or not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
            or offsets[1] - offsets[0]
            != math.prod(shape) * SAFETENSOR_DTYPES[dtype]
        ):
            _fail("adapter_tensor_entry_invalid")
        intervals.append((offsets[0], offsets[1], key))
        tensors[key] = {"dtype": dtype, "shape": shape}
    intervals.sort()
    if (
        not intervals
        or intervals[0][0] != 0
        or intervals[-1][1] != data_size
        or any(
            left[1] != right[0]
            for left, right in zip(intervals, intervals[1:], strict=False)
        )
    ):
        _fail("adapter_tensor_layout_invalid")
    return tensors


def _expected_adapter_tensors(
    specs: Iterable[Mapping[str, Any]], rank: int
) -> dict[str, list[int]]:
    expected: dict[str, list[int]] = {}
    for spec in specs:
        path = str(spec["path"])
        expected[f"{path}.lora_a"] = [int(spec["input_features"]), rank]
        expected[f"{path}.lora_b"] = [rank, int(spec["output_features"])]
    return expected


def _adapter_report(
    adapter_dir: Path | None,
    *,
    model_root: Path,
    descriptor_sha256: str,
    dataset_identity_sha256: str,
    topology_sha256: str,
    run_id: str,
    run_root: Path,
    specs: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if adapter_dir is None:
        return {"present": False, "accepted": False, "reasons": ["adapter_not_supplied"]}
    resolved = adapter_dir.expanduser().resolve(strict=False)
    if not resolved.exists():
        return {
            "present": False,
            "accepted": False,
            "path": str(resolved),
            "reasons": ["adapter_missing"],
        }
    reasons: list[str] = []
    if not resolved.is_dir():
        reasons.append("adapter_not_directory")
    if not _is_within(resolved, run_root):
        reasons.append("adapter_outside_candidate_run")

    config_path = resolved / "adapter_config.json"
    config: dict[str, Any] = {}
    if not config_path.is_file():
        reasons.append("adapter_config_missing")
    else:
        loaded = _read_json(config_path, role="adapter_config")
        if not isinstance(loaded, dict):
            reasons.append("adapter_config_invalid")
        else:
            config = loaded
            configured_model = str(config.get("model") or "")
            try:
                configured = Path(configured_model).expanduser().resolve(strict=False)
            except (OSError, RuntimeError, ValueError):
                configured = Path("/__invalid_adapter_model__")
            if configured != model_root:
                reasons.append("adapter_model_mismatch")
            lora_parameters = config.get("lora_parameters")
            if not isinstance(lora_parameters, dict) or (
                lora_parameters.get("rank") != LORA_RANK
                or lora_parameters.get("scale") != LORA_SCALE
                or lora_parameters.get("dropout") != LORA_DROPOUT
            ):
                reasons.append("adapter_lora_contract_mismatch")

    binding_path = resolved / "candidate_binding.json"
    binding: dict[str, Any] = {}
    required_binding = {
        "schema",
        "model_descriptor_sha256",
        "model_path",
        "dataset_identity_sha256",
        "target_topology_sha256",
        "run_id",
    }
    if not binding_path.is_file():
        reasons.append("candidate_binding_missing")
    else:
        loaded = _read_json(binding_path, role="candidate_adapter_binding")
        if not isinstance(loaded, dict) or set(loaded) != required_binding:
            reasons.append("candidate_binding_schema_invalid")
        else:
            binding = loaded
            expected = {
                "schema": ADAPTER_BINDING_SCHEMA,
                "model_descriptor_sha256": descriptor_sha256,
                "model_path": str(model_root),
                "dataset_identity_sha256": dataset_identity_sha256,
                "target_topology_sha256": topology_sha256,
                "run_id": run_id,
            }
            if binding != expected:
                reasons.append("candidate_binding_mismatch")

    tensor_path = resolved / "adapters.safetensors"
    checkpoint_step: int | None = None
    if not tensor_path.is_file():
        checkpoints = []
        for candidate in resolved.glob("*_adapters.safetensors"):
            match = re.fullmatch(r"(\d+)_adapters\.safetensors", candidate.name)
            if match:
                checkpoints.append((int(match.group(1)), candidate))
        if checkpoints:
            checkpoint_step, tensor_path = max(checkpoints)
    tensor_count = 0
    missing_tensors: list[str] = []
    unexpected_tensors: list[str] = []
    shape_mismatches: list[str] = []
    tensor_sha256: str | None = None
    if not tensor_path.is_file():
        reasons.append("adapter_tensor_missing")
    else:
        tensors = _read_safetensors_header(tensor_path)
        tensor_count = len(tensors)
        expected_tensors = _expected_adapter_tensors(specs, LORA_RANK)
        missing_tensors = sorted(set(expected_tensors) - set(tensors))
        unexpected_tensors = sorted(set(tensors) - set(expected_tensors))
        shape_mismatches = sorted(
            key
            for key in set(tensors) & set(expected_tensors)
            if tensors[key]["shape"] != expected_tensors[key]
        )
        if missing_tensors:
            reasons.append("adapter_tensor_topology_missing")
        if unexpected_tensors:
            reasons.append("adapter_tensor_topology_unexpected")
        if shape_mismatches:
            reasons.append("adapter_tensor_shape_mismatch")
        if not reasons:
            tensor_sha256 = _sha256_file(tensor_path)

    return {
        "present": True,
        "accepted": not reasons,
        "path": str(resolved),
        "config_path": str(config_path),
        "configured_model": config.get("model"),
        "binding_path": str(binding_path),
        "binding": binding,
        "tensor_path": str(tensor_path),
        "checkpoint_step": checkpoint_step,
        "tensor_sha256": tensor_sha256,
        "tensor_count": tensor_count,
        "missing_tensor_count": len(missing_tensors),
        "unexpected_tensor_count": len(unexpected_tensors),
        "shape_mismatch_count": len(shape_mismatches),
        "missing_tensor_sample": missing_tensors[:12],
        "unexpected_tensor_sample": unexpected_tensors[:12],
        "shape_mismatch_sample": shape_mismatches[:12],
        "reasons": sorted(set(reasons)),
    }


def build_preflight(
    *,
    descriptor_path: Path,
    data_repo_root: Path,
    source_repo_root: Path,
    output_root: Path,
    expected_descriptor_sha256: str = EXPECTED_DESCRIPTOR_SHA256,
    resume_adapter: Path | None = None,
    legacy_adapter: Path | None = None,
) -> dict[str, Any]:
    """Return one deterministic, no-model-load training readiness receipt."""

    descriptor, model_root = _validate_descriptor(
        descriptor_path,
        expected_sha256=expected_descriptor_sha256,
    )
    descriptor_sha256 = str(descriptor["descriptor_sha256"])
    topology, specs = _target_topology(descriptor, model_root)

    data_root = data_repo_root.expanduser().resolve(strict=True)
    train, train_keys = _inspect_message_dataset(data_root / "training/data/train.jsonl")
    valid, valid_keys = _inspect_message_dataset(data_root / "training/data/valid.jsonl")
    overlap = len(train_keys & valid_keys)
    crsm_source = _inspect_crsm_source(
        data_root / "data/synthetic_training/lora_dataset.jsonl"
    )
    persona_identity_material = {
        "train": _dataset_identity(train),
        "valid": _dataset_identity(valid),
        "split_overlap_records": overlap,
    }
    persona_identity_sha256 = _document_sha256(persona_identity_material)
    integration_manifest = _manifest_dataset_match(
        data_root / "training/data/crsm_integration_manifest.json",
        train=train,
        valid=valid,
        source=crsm_source,
    )

    delta_train_path = data_root / "training/data/crsm_delta/train.jsonl"
    delta_valid_path = data_root / "training/data/crsm_delta/valid.jsonl"
    delta_train: dict[str, Any] | None = None
    delta_valid: dict[str, Any] | None = None
    if delta_train_path.is_file():
        delta_train, _ = _inspect_message_dataset(delta_train_path)
    if delta_valid_path.is_file():
        delta_valid, _ = _inspect_message_dataset(delta_valid_path)
    delta_manifest = _manifest_dataset_match(
        data_root / "training/data/crsm_delta_manifest.json",
        train=delta_train,
        valid=delta_valid,
        source=crsm_source,
    )

    paths = _run_paths(
        output_root,
        descriptor_sha256=descriptor_sha256,
        persona_identity_sha256=persona_identity_sha256,
        crsm_source_sha256=str(crsm_source["sha256"]),
        topology_sha256=str(topology["topology_sha256"]),
    )
    run_root = Path(paths["run_root"])
    candidate_resume_path = resume_adapter or Path(paths["persona_adapter_dir"])
    resume = _adapter_report(
        candidate_resume_path,
        model_root=model_root,
        descriptor_sha256=descriptor_sha256,
        dataset_identity_sha256=persona_identity_sha256,
        topology_sha256=str(topology["topology_sha256"]),
        run_id=paths["run_id"],
        run_root=run_root,
        specs=specs,
    )
    legacy_path = legacy_adapter or data_root / "training/adapters/aura-personality"
    legacy = _adapter_report(
        legacy_path,
        model_root=model_root,
        descriptor_sha256=descriptor_sha256,
        dataset_identity_sha256=persona_identity_sha256,
        topology_sha256=str(topology["topology_sha256"]),
        run_id=paths["run_id"],
        run_root=run_root,
        specs=specs,
    )

    source_root = source_repo_root.expanduser().resolve(strict=True)
    pipeline_sources = {
        relative: _file_binding(source_root / relative)
        for relative in (
            "training/build_dataset_v3.py",
            "training/finetune_lora.py",
            "training/train_and_fuse.py",
        )
    }

    persona_blockers: list[str] = []
    if not topology["compatible"]:
        persona_blockers.append("candidate_target_modules_incompatible")
    if train["invalid_records"] or valid["invalid_records"]:
        persona_blockers.append("persona_dataset_schema_invalid")
    if overlap:
        persona_blockers.append(f"persona_train_valid_overlap:{overlap}")
    if not integration_manifest.get("output_current"):
        persona_blockers.append("persona_dataset_manifest_output_stale")

    crsm_blockers: list[str] = []
    if crsm_source["invalid_records"]:
        crsm_blockers.append("crsm_source_schema_invalid")
    if not delta_manifest.get("current"):
        crsm_blockers.append("crsm_delta_dataset_requires_rebuild")
    if not resume["accepted"]:
        crsm_blockers.append("candidate_persona_adapter_required_for_crsm_resume")

    checks = {
        "descriptor_exact": descriptor_sha256 == expected_descriptor_sha256,
        "candidate_target_modules_compatible": bool(topology["compatible"]),
        "persona_dataset_schema_valid": not train["invalid_records"]
        and not valid["invalid_records"],
        "persona_split_disjoint": overlap == 0,
        "persona_manifest_output_current": bool(
            integration_manifest.get("output_current")
        ),
        "crsm_source_schema_valid": not crsm_source["invalid_records"],
        "crsm_delta_manifest_current": bool(delta_manifest.get("current")),
        "candidate_resume_path_isolated": _is_within(
            Path(paths["persona_adapter_dir"]), run_root
        ),
        "legacy_qwen25_adapter_refused": not legacy["accepted"],
        "no_model_load": True,
    }
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "source_revision": "CP912",
        "no_model_load": True,
        "candidate": {
            "descriptor": _file_binding(descriptor_path),
            "descriptor_sha256": descriptor_sha256,
            "canonical_path": str(model_root),
            "repository_id": descriptor["repository_id"],
            "revision": descriptor["revision"],
        },
        "pipeline_sources": pipeline_sources,
        "datasets": {
            "persona": {
                "identity_sha256": persona_identity_sha256,
                "train": train,
                "valid": valid,
                "split_overlap_records": overlap,
                "manifest": integration_manifest,
            },
            "crsm": {
                "source": crsm_source,
                "delta_train": delta_train,
                "delta_valid": delta_valid,
                "integration_manifest": integration_manifest,
                "delta_manifest": delta_manifest,
                "requires_rebuild": not bool(delta_manifest.get("current")),
            },
        },
        "target_modules": topology,
        "paths": paths,
        "resume": {
            "candidate_persona": resume,
            "legacy_adapter_audit": legacy,
            "candidate_binding_schema": ADAPTER_BINDING_SCHEMA,
            "candidate_binding_template": {
                "schema": ADAPTER_BINDING_SCHEMA,
                "model_descriptor_sha256": descriptor_sha256,
                "model_path": str(model_root),
                "dataset_identity_sha256": persona_identity_sha256,
                "target_topology_sha256": str(topology["topology_sha256"]),
                "run_id": paths["run_id"],
            },
        },
        "readiness": {
            "persona_fresh_training": not persona_blockers,
            "persona_blockers": persona_blockers,
            "crsm_delta_after_persona": not crsm_blockers,
            "crsm_blockers": crsm_blockers,
        },
        "checks": checks,
    }
    result["verdict"] = (
        "READY_FOR_PERSONA_AND_CRSM"
        if result["readiness"]["persona_fresh_training"]
        and result["readiness"]["crsm_delta_after_persona"]
        else "READY_FOR_PERSONA"
        if result["readiness"]["persona_fresh_training"]
        else "BLOCKED"
    )
    result["preflight_sha256"] = _document_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor",
        default=REPO_ROOT / "artifacts/closeout/cortex_upgrade/cp911/artifact_descriptor.json",
        type=Path,
    )
    parser.add_argument("--data-repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--source-repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument(
        "--output-root",
        default=REPO_ROOT / "training/candidate-runs",
        type=Path,
    )
    parser.add_argument("--resume-adapter", type=Path)
    parser.add_argument("--legacy-adapter", type=Path)
    parser.add_argument(
        "--expected-descriptor-sha256",
        default=EXPECTED_DESCRIPTOR_SHA256,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--require-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_preflight(
        descriptor_path=args.descriptor,
        data_repo_root=args.data_repo_root,
        source_repo_root=args.source_repo_root,
        output_root=args.output_root,
        expected_descriptor_sha256=args.expected_descriptor_sha256,
        resume_adapter=args.resume_adapter,
        legacy_adapter=args.legacy_adapter,
    )
    from core.runtime.atomic_writer import atomic_write_text

    atomic_write_text(
        args.out,
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_ready and result["verdict"] != "READY_FOR_PERSONA_AND_CRSM":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
