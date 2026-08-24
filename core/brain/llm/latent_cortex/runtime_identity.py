"""Identity receipts for resident Recursive Latent Cortex episodes."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.tensor_identity import tensor_identity_parts

WORKER_IDENTITY_SCHEMA = "aura.latent_cortex.worker_identity.v3"
WORKER_ACTIVATION_SCHEMA = (
    "aura.latent_cortex.worker_recurrent_adapter_activation.v1"
)
RUNTIME_IDENTITY_SCHEMA = "aura.latent_cortex.runtime_identity.v1"
MAX_AFFECTIVE_STEERING_ALPHA = 50.0


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_sha256(path: str | Path, *, max_bytes: int) -> str:
    return hashlib.sha256(read_stable_bytes(path, max_bytes=max_bytes)).hexdigest()


def _stable_model_artifact_bytes(path: str | Path, *, max_bytes: int) -> bytes:
    """Read a model artifact while preserving Hugging Face snapshot identity.

    Hugging Face snapshots intentionally contain symlinks into the immutable
    blob store. The general file gateway correctly rejects symlinks, but
    treating that standard checkpoint layout as absent evidence made every
    downloaded model's tokenizer and quantization identity incomplete. Resolve
    only the final artifact link, read the resolved regular file through the
    no-follow gateway, then prove the link itself did not change around the
    read. Arbitrary application/state reads retain the stricter no-link rule.
    """

    target = Path(path).expanduser()
    if not target.is_symlink():
        return read_stable_bytes(target, max_bytes=max_bytes)
    before = os.lstat(target)
    link_before = os.readlink(target)
    resolved = target.resolve(strict=True)
    payload = read_stable_bytes(resolved, max_bytes=max_bytes)
    after = os.lstat(target)
    if (
        os.readlink(target) != link_before
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise OSError(f"model artifact symlink changed during read: {target}")
    return payload


def _stable_model_artifact_sha256(path: str | Path, *, max_bytes: int) -> str:
    return hashlib.sha256(
        _stable_model_artifact_bytes(path, max_bytes=max_bytes)
    ).hexdigest()


def canonical_model_path(model_path: str | Path) -> str:
    return os.path.realpath(os.path.expanduser(str(model_path)))


def latent_request_payload_sha256(
    *,
    prompt: Any,
    messages: Any,
    domain: str,
    config: Any,
    budget: Any,
    runtime_controls: Any,
    cognitive_context: Any = None,
    operation_authority: Any = None,
    action_policy_evidence: Any = None,
    action_intervention: Any = None,
    external_execution_offer: Any = None,
    response_contract: Any = None,
    verifier_guidance: Any = None,
    facet_reliability: Any = None,
) -> str:
    payload = {
        "prompt": prompt,
        "messages": messages,
        "domain": str(domain or "general"),
        "config": config,
        "budget": budget,
        "runtime_controls": runtime_controls,
    }
    # Additive so pre-ingress request digests stay reproducible: episodes
    # without typed cognitive context hash exactly as they always did.
    if cognitive_context is not None:
        payload["cognitive_context"] = cognitive_context
    if operation_authority is not None:
        payload["operation_authority"] = operation_authority
    if action_policy_evidence is not None:
        payload["action_policy_evidence"] = action_policy_evidence
    if action_intervention is not None:
        payload["action_intervention"] = action_intervention
    if external_execution_offer is not None:
        payload["external_execution_offer"] = external_execution_offer
    if response_contract is not None:
        payload["response_contract"] = response_contract
    # CP126 9721b1be: verifier_guidance and facet_reliability are SEMANTIC
    # inputs to a latent episode — two episodes with different verifier
    # behavior used to share one expected request identity, so the
    # parent-side digest proof could not tell them apart. Additive, so
    # episodes that pass neither hash exactly as they always did.
    if verifier_guidance is not None:
        payload["verifier_guidance"] = verifier_guidance
    if facet_reliability is not None:
        payload["facet_reliability"] = facet_reliability
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_parameter_count(model: Any) -> int:
    """Count every resident parameter leaf once without copying tensor data."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError("resident model does not expose a parameter tree")

    def leaves(node: Any):
        if isinstance(node, Mapping):
            for child in node.values():
                yield from leaves(child)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                yield from leaves(child)
            return
        yield node

    total = 0
    for tensor in leaves(parameters()):
        size = getattr(tensor, "size", None)
        if callable(size):
            size = size()
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("resident model exposed an invalid parameter leaf size")
        total += size
    if total <= 0:
        raise ValueError("resident model exposed no countable parameters")
    return total


def logical_model_parameter_count(
    model_path: str | Path,
    *,
    stored_element_count: int,
) -> tuple[int, str]:
    """Derive logical weights from architecture config for packed checkpoints."""

    config_path = Path(canonical_model_path(model_path)) / "config.json"
    try:
        config = json.loads(
            read_stable_bytes(config_path, max_bytes=2 * 1024 * 1024).decode(
                "utf-8"
            )
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return stored_element_count, "stored_tensor_elements"
    if not isinstance(config, Mapping) or str(config.get("model_type") or "") != "qwen2":
        return stored_element_count, "stored_tensor_elements"

    names = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    values: dict[str, int] = {}
    for name in names:
        value = config.get(name)
        if type(value) is not int or value <= 0:
            return stored_element_count, "stored_tensor_elements"
        values[name] = value
    hidden = values["hidden_size"]
    attention_heads = values["num_attention_heads"]
    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden % attention_heads:
            return stored_element_count, "stored_tensor_elements"
        head_dim = hidden // attention_heads
    if type(head_dim) is not int or head_dim <= 0:
        return stored_element_count, "stored_tensor_elements"

    query_width = attention_heads * head_dim
    kv_width = values["num_key_value_heads"] * head_dim
    attention_weights = (
        hidden * query_width
        + 2 * hidden * kv_width
        + query_width * hidden
    )
    # Qwen2 q/k/v projections carry bias; o_proj and the gated MLP do not.
    attention_biases = query_width + 2 * kv_width
    mlp_weights = 3 * hidden * values["intermediate_size"]
    layer_norms = 2 * hidden
    per_layer = attention_weights + attention_biases + mlp_weights + layer_norms
    embeddings = values["vocab_size"] * hidden
    output_head = (
        0
        if config.get("tie_word_embeddings") is True
        else values["vocab_size"] * hidden
    )
    logical = (
        embeddings
        + values["num_hidden_layers"] * per_layer
        + hidden
        + output_head
    )
    if logical <= 0:
        return stored_element_count, "stored_tensor_elements"
    return logical, "architecture_config_logical"


def build_worker_identity(
    model: Any,
    *,
    model_path: str | Path,
    worker_boot_id: str,
    worker_source_path: str | Path,
    worker_action_capture_identity: Mapping[str, Any],
    tokenizer: Any = None,
    affective_steering_active: bool = False,
    affective_steering_alpha: float = 0.0,
    recurrent_adapter_activation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    boot_id = str(worker_boot_id or "").strip().lower()
    if len(boot_id) != 32 or any(character not in "0123456789abcdef" for character in boot_id):
        raise ValueError("worker_boot_id must be a 128-bit lowercase hex identifier")
    stored_element_count = model_parameter_count(model)
    logical_count, count_basis = logical_model_parameter_count(
        model_path,
        stored_element_count=stored_element_count,
    )
    from core.brain.llm.latent_cortex.worker_capture_identity import (
        validate_worker_capture_identity,
    )

    capture_identity = validate_worker_capture_identity(
        dict(worker_action_capture_identity)
    )
    if (
        capture_identity["worker_boot_id"] != boot_id
        or capture_identity["worker_pid"] != os.getpid()
    ):
        raise ValueError(
            "worker action-capture identity does not belong to this worker boot"
        )
    activation = (
        inactive_worker_recurrent_adapter_activation()
        if recurrent_adapter_activation is None
        else dict(recurrent_adapter_activation)
    )
    activation_errors = worker_recurrent_adapter_activation_errors(
        activation
    )
    if activation_errors:
        raise ValueError(
            "invalid worker recurrent-adapter activation: "
            + ",".join(activation_errors)
        )
    return {
        "schema": WORKER_IDENTITY_SCHEMA,
        "worker_boot_id": boot_id,
        "worker_pid": os.getpid(),
        "worker_model_path": canonical_model_path(model_path),
        "worker_model_parameter_count": logical_count,
        "worker_model_stored_parameter_element_count": stored_element_count,
        "worker_model_parameter_count_basis": count_basis,
        "worker_source_sha256": _stable_sha256(worker_source_path, max_bytes=8 * 1024 * 1024),
        "worker_affective_steering_active": bool(affective_steering_active),
        "worker_affective_steering_alpha": float(affective_steering_alpha),
        "worker_action_capture_identity": capture_identity,
        "worker_recurrent_adapter_activation": activation,
        # CP126 866530f6. Model path plus a parameter count cannot tell two
        # runtimes apart when the difference is WHICH adapter is attached,
        # which tokenizer resolved the text, or how the weights are
        # quantized. Two workers with identical path and count could be
        # serving materially different functions, so identity comparisons
        # and control/treatment claims built on this receipt were weaker
        # than they read.
        **serving_stack_identity(
            model,
            model_path,
            tokenizer=tokenizer,
        ),
    }


def inactive_worker_recurrent_adapter_activation(
    *,
    reason: str = "no_certified_activation",
) -> dict[str, Any]:
    """Canonical evidence that no trained recurrent adapter is serving."""

    return {
        "schema": WORKER_ACTIVATION_SCHEMA,
        "configured": False,
        "active": False,
        "reason": str(reason),
        "receipt_sha256": "",
        "activation_sha256": "",
        "adapter_composite_identity_sha256": "",
        "campaign_name": "",
        "claim_tier": "NONE",
        "verified_verdict": "none",
        "loaded_projection_count": 0,
    }


def worker_recurrent_adapter_activation_errors(value: Any) -> list[str]:
    """Validate the proof-bearing live recurrent-adapter state."""

    if not isinstance(value, Mapping):
        return ["worker_recurrent_adapter_activation_not_mapping"]
    expected = {
        "schema",
        "configured",
        "active",
        "reason",
        "receipt_sha256",
        "activation_sha256",
        "adapter_composite_identity_sha256",
        "campaign_name",
        "claim_tier",
        "verified_verdict",
        "loaded_projection_count",
    }
    if set(value) != expected:
        return ["invalid_worker_recurrent_adapter_activation_fields"]
    errors: list[str] = []
    if value.get("schema") != WORKER_ACTIVATION_SCHEMA:
        errors.append("invalid_worker_recurrent_adapter_activation_schema")
    configured = value.get("configured")
    active = value.get("active")
    if type(configured) is not bool or type(active) is not bool:
        errors.append("invalid_worker_recurrent_adapter_activation_state")
        return errors
    if not isinstance(value.get("reason"), str) or not value["reason"]:
        errors.append("invalid_worker_recurrent_adapter_activation_reason")
    count = value.get("loaded_projection_count")
    if type(count) is not int or count < 0:
        errors.append("invalid_worker_recurrent_adapter_projection_count")
    if active:
        if not configured:
            errors.append("active_worker_recurrent_adapter_not_configured")
        for key in (
            "receipt_sha256",
            "activation_sha256",
            "adapter_composite_identity_sha256",
        ):
            if not _sha256(value.get(key)):
                errors.append(f"invalid_{key}")
        if (
            not str(value.get("campaign_name") or "")
            or value.get("claim_tier") != "PROVEN"
            or value.get("verified_verdict") != "gain_proven"
            or type(count) is not int
            or count <= 0
        ):
            errors.append("worker_recurrent_adapter_positive_evidence_incomplete")
    elif (
        configured
        or value.get("receipt_sha256") != ""
        or value.get("activation_sha256") != ""
        or value.get("adapter_composite_identity_sha256") != ""
        or value.get("campaign_name") != ""
        or value.get("claim_tier") != "NONE"
        or value.get("verified_verdict") != "none"
        or count != 0
    ):
        errors.append("inactive_worker_recurrent_adapter_claims_evidence")
    return errors


def serving_stack_identity(
    model: Any,
    model_path: str | Path,
    *,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Identity of everything that changes what the model computes.

    Ordered adapter identity, tokenizer identity, and the quantization/dtype
    layout — each best-effort and each reporting its own absence rather than
    silently contributing nothing. A field that could not be determined is
    recorded as an empty value with a reason in
    ``worker_stack_identity_gaps``, so a consumer can see that identity is
    partial instead of assuming it is complete.
    """
    gaps: list[str] = []

    adapters = _attached_adapter_identity(model, gaps)
    tokenizer_artifacts = _tokenizer_identity(model_path, gaps)
    quantization = _quantization_identity(model_path, gaps)
    runtime_tokenizer = _runtime_tokenizer_identity(tokenizer, gaps)

    return {
        "worker_adapters": adapters,
        "worker_adapter_stack_sha256": _digest_of_json(adapters),
        "worker_tokenizer": tokenizer_artifacts,
        "worker_runtime_tokenizer": runtime_tokenizer,
        "worker_quantization": quantization,
        "worker_stack_identity_gaps": gaps,
    }


def _attached_adapter_identity(model: Any, gaps: list[str]) -> list[dict[str, Any]]:
    """Ordered identity of adapter-class modules resident on the model.

    Order matters: the same adapters applied in a different order can
    compose to a different function, so this is a list, not a set.
    """
    adapters: list[dict[str, Any]] = []
    try:
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            gaps.append("adapters:model_exposes_no_named_modules")
            return adapters
        for name, module in named_modules():
            type_name = type(module).__name__
            if not any(
                marker in type_name
                for marker in ("LoRA", "DoRA", "Adapter")
            ):
                continue
            parameter_sha256, parameter_scope = _module_parameter_identity(
                module
            )
            if not _sha256(parameter_sha256):
                gaps.append(f"adapters:{name}:parameter_identity_unavailable")
            adapters.append(
                {
                    "name": str(name),
                    "type": type_name,
                    "rank": _int_or_zero(getattr(module, "r", None)),
                    "scale": _float_or_zero(getattr(module, "scale", None)),
                    "parameter_sha256": parameter_sha256,
                    "parameter_scope": parameter_scope,
                }
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        gaps.append(f"adapters:{type(exc).__name__}")
    return adapters


def _module_parameter_identity(module: Any) -> tuple[str, str]:
    """Hash adapter-owned bytes without recursively rehashing the base model."""

    try:
        from mlx.utils import tree_flatten

        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return "", ""
        rows = sorted(tree_flatten(parameters()), key=lambda row: row[0])
        type_name = type(module).__name__
        if "LoRA" in type_name or "DoRA" in type_name:
            # mlx_lm wrappers recursively expose ``linear.*`` or
            # ``embedding.*`` base tensors through parameters(). Hashing that
            # tree for every adapter would reread much of the resident 32B on
            # every pre/post stack check. Those permanent bytes are measured
            # by the parameter canary and exact adapted-layer proof; adapter
            # identity owns the low-rank tensors and DoRA magnitude state.
            rows = [
                row
                for row in rows
                if not row[0].startswith(("linear.", "embedding."))
            ]
            scope = "adapter_owned_excluding_wrapped_base_v1"
        else:
            scope = "module_parameter_tree_v1"
        if not rows:
            return "", scope
        digest = hashlib.sha256()
        for name, tensor in rows:
            dtype, shape, payload = tensor_identity_parts(tensor)
            digest.update(str(name).encode("utf-8"))
            digest.update(dtype.encode("ascii"))
            digest.update(str(shape).encode("ascii"))
            digest.update(payload)
        return digest.hexdigest(), scope
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return "", ""


def _tokenizer_identity(model_path: str | Path, gaps: list[str]) -> dict[str, Any]:
    """Digest of the tokenizer artifacts that turn text into the token ids."""
    identity: dict[str, Any] = {}
    try:
        root = Path(str(model_path))
        found = False
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            candidate = root / filename
            if candidate.is_file():
                identity[filename] = _stable_model_artifact_sha256(
                    candidate,
                    max_bytes=32 * 1024 * 1024,
                )
                found = True
        if not found:
            gaps.append("tokenizer:no_tokenizer_artifacts_found")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        gaps.append(f"tokenizer:{type(exc).__name__}")
    return identity


def _runtime_tokenizer_identity(
    tokenizer: Any,
    gaps: list[str],
) -> dict[str, Any]:
    if tokenizer is None:
        gaps.append("runtime_tokenizer:unavailable")
        return {}
    identity: dict[str, Any] = {
        "type": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
    }
    for field in (
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ):
        value = getattr(tokenizer, field, None)
        if isinstance(value, bool) or (value is not None and not isinstance(value, int)):
            gaps.append(f"runtime_tokenizer:{field}_invalid")
            continue
        if field == "vocab_size" and (value is None or value <= 0):
            gaps.append("runtime_tokenizer:vocab_size_unavailable")
        identity[field] = value
    special_tokens = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_tokens, Mapping):
        identity["special_tokens_sha256"] = _digest_of_json(special_tokens)
    else:
        identity["special_tokens_sha256"] = _digest_of_json({})
    chat_template = getattr(tokenizer, "chat_template", None)
    identity["chat_template_sha256"] = hashlib.sha256(
        str(chat_template or "").encode("utf-8")
    ).hexdigest()
    return identity


def _quantization_identity(model_path: str | Path, gaps: list[str]) -> dict[str, Any]:
    """Quantization layout and dtype, which change the computed function."""
    identity: dict[str, Any] = {}
    try:
        config_path = Path(str(model_path)) / "config.json"
        if not config_path.is_file():
            gaps.append("quantization:no_config")
            return identity
        config_bytes = _stable_model_artifact_bytes(
            config_path,
            max_bytes=4 * 1024 * 1024,
        )
        config = json.loads(config_bytes.decode("utf-8"))
        if not isinstance(config, dict):
            gaps.append("quantization:config_not_an_object")
            return identity
        quantization = config.get("quantization")
        if isinstance(quantization, dict):
            identity["bits"] = _int_or_zero(quantization.get("bits"))
            identity["group_size"] = _int_or_zero(quantization.get("group_size"))
        else:
            identity["bits"] = 0
            identity["group_size"] = 0
        identity["dtype"] = str(
            config.get("torch_dtype") or config.get("dtype") or ""
        )
        identity["model_type"] = str(config.get("model_type") or "")
        identity["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    except (OSError, UnicodeDecodeError, ValueError, TypeError, RuntimeError) as exc:
        gaps.append(f"quantization:{type(exc).__name__}")
    return identity


def _digest_of_json(value: Any) -> str:
    try:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
    except (TypeError, ValueError):
        return ""


def _int_or_zero(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        if isinstance(value, bool):
            return 0.0
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def serving_stack_identity_errors(identity: Any) -> list[str]:
    """Validate the complete function-defining serving-stack identity."""

    if not isinstance(identity, Mapping):
        return ["worker_serving_stack_identity_not_mapping"]
    expected_fields = {
        "worker_adapters",
        "worker_adapter_stack_sha256",
        "worker_tokenizer",
        "worker_runtime_tokenizer",
        "worker_quantization",
        "worker_stack_identity_gaps",
    }
    errors: list[str] = []
    if set(identity) != expected_fields:
        errors.append("invalid_worker_serving_stack_fields")

    adapters = identity.get("worker_adapters")
    if (
        not isinstance(adapters, list)
        or identity.get("worker_adapter_stack_sha256")
        != _digest_of_json(adapters)
    ):
        errors.append("invalid_worker_adapter_identity")
    elif any(
        not isinstance(adapter, Mapping)
        or not str(adapter.get("name") or "")
        or not str(adapter.get("type") or "")
        or type(adapter.get("rank")) is not int
        or adapter["rank"] < 0
        or isinstance(adapter.get("scale"), bool)
        or not isinstance(adapter.get("scale"), (int, float))
        or not math.isfinite(float(adapter["scale"]))
        or not _sha256(adapter.get("parameter_sha256"))
        or adapter.get("parameter_scope")
        not in {
            "adapter_owned_excluding_wrapped_base_v1",
            "module_parameter_tree_v1",
        }
        for adapter in adapters
    ):
        errors.append("invalid_worker_adapter_identity")

    tokenizer = identity.get("worker_tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or not tokenizer
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not _sha256(digest)
            for name, digest in tokenizer.items()
        )
    ):
        errors.append("invalid_worker_tokenizer_identity")

    runtime_tokenizer = identity.get("worker_runtime_tokenizer")
    if (
        not isinstance(runtime_tokenizer, Mapping)
        or not str(runtime_tokenizer.get("type") or "")
        or type(runtime_tokenizer.get("vocab_size")) is not int
        or runtime_tokenizer["vocab_size"] <= 0
        or not _sha256(runtime_tokenizer.get("special_tokens_sha256"))
        or not _sha256(runtime_tokenizer.get("chat_template_sha256"))
        or any(
            value is not None
            and (
                type(value) is not int
                or value < 0
            )
            for value in (
                runtime_tokenizer.get("bos_token_id"),
                runtime_tokenizer.get("eos_token_id"),
                runtime_tokenizer.get("pad_token_id"),
                runtime_tokenizer.get("unk_token_id"),
            )
        )
    ):
        errors.append("invalid_worker_runtime_tokenizer_identity")

    quantization = identity.get("worker_quantization")
    if (
        not isinstance(quantization, Mapping)
        or type(quantization.get("bits")) is not int
        or quantization["bits"] < 0
        or type(quantization.get("group_size")) is not int
        or quantization["group_size"] < 0
        or not isinstance(quantization.get("dtype"), str)
        or not str(quantization.get("model_type") or "")
        or not _sha256(quantization.get("config_sha256"))
    ):
        errors.append("invalid_worker_quantization_identity")

    gaps = identity.get("worker_stack_identity_gaps")
    if (
        not isinstance(gaps, list)
        or any(not isinstance(item, str) or not item for item in gaps)
    ):
        errors.append("invalid_worker_stack_identity_gaps")
    elif gaps:
        errors.append("worker_serving_stack_identity_incomplete")
    return errors


def worker_identity_errors(
    receipt: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> list[str]:
    if not isinstance(receipt, Mapping):
        return ["worker_identity_receipt_not_mapping"]
    errors: list[str] = []
    schema = receipt.get("schema")
    if schema not in {
        "aura.latent_cortex.worker_identity.v1",
        "aura.latent_cortex.worker_identity.v2",
        WORKER_IDENTITY_SCHEMA,
    }:
        errors.append("invalid_worker_identity_schema")
    boot_id = receipt.get("worker_boot_id")
    if not (
        isinstance(boot_id, str)
        and len(boot_id) == 32
        and all(character in "0123456789abcdef" for character in boot_id)
    ):
        errors.append("invalid_worker_boot_id")
    if type(receipt.get("worker_pid")) is not int or receipt["worker_pid"] <= 0:
        errors.append("invalid_worker_pid")
    if not str(receipt.get("worker_model_path") or "").strip():
        errors.append("missing_worker_model_path")
    if (
        type(receipt.get("worker_model_parameter_count")) is not int
        or receipt["worker_model_parameter_count"] <= 0
    ):
        errors.append("invalid_worker_model_parameter_count")
    if (
        type(receipt.get("worker_model_stored_parameter_element_count")) is not int
        or receipt["worker_model_stored_parameter_element_count"] <= 0
    ):
        errors.append("invalid_worker_model_stored_parameter_element_count")
    count_basis = receipt.get("worker_model_parameter_count_basis")
    if count_basis not in {
        "architecture_config_logical",
        "stored_tensor_elements",
    }:
        errors.append("invalid_worker_model_parameter_count_basis")
    logical_count = receipt.get("worker_model_parameter_count")
    stored_count = receipt.get("worker_model_stored_parameter_element_count")
    if (
        type(logical_count) is int
        and type(stored_count) is int
        and (
            (count_basis == "architecture_config_logical" and logical_count < stored_count)
            or (count_basis == "stored_tensor_elements" and logical_count != stored_count)
        )
    ):
        errors.append("worker_model_parameter_count_basis_contradiction")
    if not _sha256(receipt.get("worker_source_sha256")):
        errors.append("invalid_worker_source_sha256")
    if type(receipt.get("worker_affective_steering_active")) is not bool:
        errors.append("invalid_worker_affective_steering_active")
    steering_alpha = receipt.get("worker_affective_steering_alpha")
    if (
        isinstance(steering_alpha, bool)
        or not isinstance(steering_alpha, (int, float))
        or not 0.0 <= float(steering_alpha) <= MAX_AFFECTIVE_STEERING_ALPHA
    ):
        errors.append("invalid_worker_affective_steering_alpha")
    if schema in {
        "aura.latent_cortex.worker_identity.v2",
        WORKER_IDENTITY_SCHEMA,
    }:
        try:
            from core.brain.llm.latent_cortex.worker_capture_identity import (
                validate_worker_capture_identity,
            )

            capture_identity = validate_worker_capture_identity(
                receipt.get("worker_action_capture_identity")
            )
            if (
                capture_identity.get("worker_boot_id") != boot_id
                or capture_identity.get("worker_pid")
                != receipt.get("worker_pid")
            ):
                errors.append("worker_action_capture_identity_mismatch")
        except (ImportError, TypeError, ValueError):
            errors.append("invalid_worker_action_capture_identity")
        stack = {
            key: receipt.get(key)
            for key in (
                "worker_adapters",
                "worker_adapter_stack_sha256",
                "worker_tokenizer",
                "worker_runtime_tokenizer",
                "worker_quantization",
                "worker_stack_identity_gaps",
            )
        }
        errors.extend(serving_stack_identity_errors(stack))
    if schema == WORKER_IDENTITY_SCHEMA:
        errors.extend(
            worker_recurrent_adapter_activation_errors(
                receipt.get("worker_recurrent_adapter_activation")
            )
        )
    if expected is not None:
        for key in (
            "worker_boot_id",
            "worker_pid",
            "worker_model_path",
            "worker_model_parameter_count",
            "worker_model_stored_parameter_element_count",
            "worker_model_parameter_count_basis",
            "worker_source_sha256",
            "worker_affective_steering_active",
            "worker_affective_steering_alpha",
            "worker_action_capture_identity",
            "worker_recurrent_adapter_activation",
            "worker_adapters",
            "worker_adapter_stack_sha256",
            "worker_tokenizer",
            "worker_runtime_tokenizer",
            "worker_quantization",
            "worker_stack_identity_gaps",
        ):
            if receipt.get(key) != expected.get(key):
                errors.append(f"{key}_mismatch")
    return errors


def _typed_issue_list(value: Any) -> list[str]:
    """Issue strings from an untrusted field, never a crash.

    A bare string is ONE issue, not a sequence of characters — iterating it
    was how a malformed provenance response turned into a wall of
    single-letter issues. Anything else non-iterable becomes a typed issue
    describing the shape problem itself.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        return [f"{key}:{val}" for key, val in value.items()]
    try:
        return [str(item) for item in value if str(item).strip()]
    except TypeError:
        return [f"provenance_issues_malformed:{type(value).__name__}"]


def _mapping_or_empty(value: Any) -> Mapping:
    """A mapping to read, or an empty one — never an AttributeError.

    A truthy non-mapping (a list, a string) previously reached .get and
    raised.
    """
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any) -> int:
    """Coerce an untrusted count, defaulting to 0 rather than raising."""
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def collect_latent_runtime_identity(
    project_root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind one episode to the exact source and, when applicable, Aura.app."""

    from core.runtime.launch_provenance import (
        collect_runtime_launch_provenance,
        collect_source_identity,
    )

    # CP126 79271a67. This collector reports on identity DEGRADATION, so it
    # is precisely the code that must not itself raise when the thing it is
    # inspecting is malformed. Three unguarded assumptions lived here: that
    # `issues` is an iterable collection (a bare string iterates into
    # characters; an int raises), that a truthy `expected` is a mapping
    # before calling .get on it, and that `source_change_count` converts
    # with a bare int(). Any of those turned "identity could not be
    # verified" into an exception on the caller.
    provenance = collect_runtime_launch_provenance(project_root, env=env)
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    required = provenance.get("required") is True
    source = provenance.get("actual") if required else collect_source_identity(project_root)
    source = dict(source) if isinstance(source, Mapping) else {}
    manifest = provenance.get("manifest")
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    issues = _typed_issue_list(provenance.get("issues"))

    app_executable_sha256 = ""
    launch_manifest_sha256 = ""
    if required:
        executable = str(provenance.get("app_executable") or "").strip()
        manifest_path = str(provenance.get("manifest_path") or "").strip()
        try:
            app_executable_sha256 = _stable_sha256(
                executable,
                max_bytes=256 * 1024 * 1024,
            )
            launch_manifest_sha256 = _stable_sha256(
                manifest_path,
                max_bytes=4 * 1024 * 1024,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            issues.append(f"app_identity_hash_failed:{type(exc).__name__}")

    commit_sha = str(source.get("commit_sha") or "").lower()
    workspace_sha256 = str(source.get("workspace_state_sha256") or "").lower()
    shell_assets_sha256 = str(
        source.get("shell_assets_sha256") or manifest.get("shell_assets_sha256") or ""
    ).lower()
    source_bound = bool(
        _git_oid(commit_sha)
        and _sha256(workspace_sha256)
        and _sha256(shell_assets_sha256)
        and provenance.get("source_verified") is True
    )
    installed_app_verified = bool(
        required
        and provenance.get("verified") is True
        and _sha256(app_executable_sha256)
        and _sha256(launch_manifest_sha256)
    )
    identity_bound = bool(source_bound and (not required or installed_app_verified))
    if not source_bound:
        issues.append("source_identity_unbound")
    if required and not installed_app_verified:
        issues.append("installed_app_identity_unbound")

    return {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "identity_bound": identity_bound,
        "launch_mode": str(provenance.get("launch_mode") or ""),
        "installed_app_required": required,
        "installed_app_verified": installed_app_verified,
        "source_verified": provenance.get("source_verified") is True,
        "source_root": str(source.get("source_root") or provenance.get("source_root") or ""),
        "source_commit": commit_sha,
        "source_branch": str(source.get("branch") or ""),
        "workspace_state_sha256": workspace_sha256,
        "source_dirty": source.get("source_dirty") is True,
        "source_change_count": _nonnegative_int(source.get("source_change_count")),
        "shell_assets_sha256": shell_assets_sha256,
        "bundle_identifier": str(
            manifest.get("bundle_identifier")
            or _mapping_or_empty(provenance.get("expected")).get("bundle_identifier")
            or ""
        ),
        "app_executable_sha256": app_executable_sha256,
        "launch_manifest_sha256": launch_manifest_sha256,
        "issues": sorted(set(issues)),
    }


__all__ = [
    "MAX_AFFECTIVE_STEERING_ALPHA",
    "RUNTIME_IDENTITY_SCHEMA",
    "WORKER_ACTIVATION_SCHEMA",
    "WORKER_IDENTITY_SCHEMA",
    "build_worker_identity",
    "canonical_model_path",
    "collect_latent_runtime_identity",
    "latent_request_payload_sha256",
    "inactive_worker_recurrent_adapter_activation",
    "logical_model_parameter_count",
    "model_parameter_count",
    "serving_stack_identity",
    "serving_stack_identity_errors",
    "worker_identity_errors",
    "worker_recurrent_adapter_activation_errors",
]
