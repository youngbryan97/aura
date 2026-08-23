"""Measured model-artifact identity for admission, QoS, and proof decisions.

CP126 semantic review flagged a whole class of defects rooted in one habit:
model footprint, minimum-headroom, deadline, cache-residency, and identity
decisions were derived from SPOOFABLE PATH SUBSTRINGS ("72b", "cortex",
"zenith"). A renamed heavy checkpoint inherited light-model budgets; an
unrelated path containing "32b" inherited a 20GB reservation.

Every sharded MLX artifact already carries machine-readable evidence:

- ``model.safetensors.index.json`` → ``metadata.total_parameters`` and
  ``metadata.total_size`` (exact weight bytes),
- ``config.json`` → architecture shape (a parameter count can be estimated
  when the index metadata is absent),
- the safetensors file listing itself (names + sizes).

This module turns that evidence into a cached :class:`ModelArtifactProfile`.
Classification prefers measured evidence and only falls back to declared
path naming when the artifact is absent (tests and pre-download paths use
fake names) — and the profile SAYS which evidence produced it, so receipts
can distinguish measured truth from naming convention.

The fingerprint is a cheap identity binding (config bytes + index metadata
+ weight-file listing), NOT a weight hash — hashing 20GB per admission
check is not viable. It changes whenever the artifact's declared shape,
quantization, shard layout, or file sizes change, which is the tamper
surface the admission and proof lanes actually consult.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_DESCRIPTOR_SCHEMA = "aura.model_artifact_descriptor.v1"
SERVING_PROFILE_SCHEMA = "aura.model_serving_profile.v1"
SERVING_QUALIFICATION_SCHEMA = "aura.model_serving_qualification.v2"

_REQUIRED_SERVING_LANES = frozenset(
    {
        "foreground_simple",
        "foreground_standard",
        "foreground_extended",
        "deep_reasoning",
        "tool_execution",
        "code",
        "document",
    }
)

# Parameter-count boundaries for the runtime's weight classes. The classes
# mirror the lanes the runtime actually provisions for (solver/cortex/
# brainstem/reflex); boundaries sit between real model families rather than
# on top of them.
_CLASS_BOUNDARIES: tuple[tuple[float, str], ...] = (
    (55e9, "72b"),
    (20e9, "32b"),
    (10e9, "14b"),
    (4e9, "7b"),
    (0.0, "small"),
)

_HEAVY_CLASSES = frozenset({"72b", "32b"})

_72B_PATH_TOKENS = ("72b", "solver")
_32B_PATH_TOKENS = ("32b", "27b", "cortex", "zenith")
_14B_PATH_TOKENS = ("14b", "24b", "40b")
_7B_PATH_TOKENS = ("7b", "brainstem")


@dataclass(frozen=True)
class ModelArtifactProfile:
    path: str
    exists: bool
    weight_bytes: int
    total_parameters: int
    quantization_bits: int
    size_class: str  # "72b" | "32b" | "14b" | "7b" | "small" | "unknown"
    evidence: str  # "index_metadata" | "config_estimate" | "file_sizes" | "path_tokens" | "absent"
    fingerprint: str
    model_type: str = ""
    architectures: tuple[str, ...] = ()
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    vocab_size: int = 0
    native_context_window: int = 0
    layer_types: tuple[str, ...] = ()
    linear_attention_layers: int = 0
    full_attention_layers: int = 0

    @property
    def weight_gb(self) -> float:
        return float(self.weight_bytes) / float(1024**3)

    @property
    def is_heavy(self) -> bool:
        return self.size_class in _HEAVY_CLASSES

    @property
    def measured(self) -> bool:
        """True when the class came from artifact evidence, not naming."""
        return self.evidence in {"index_metadata", "config_estimate", "file_sizes"}


_PROFILE_CACHE: dict[str, tuple[tuple[float, float], ModelArtifactProfile]] = {}
_PROFILE_CACHE_LOCK = threading.Lock()
_PROFILE_CACHE_MAX = 32
# Zero-syscall fast path keyed on the RAW path string: computing the
# mtime-validated key above costs realpath + three stats, and that ran on
# every call from an event-loop status read.
_PROFILE_FAST_CACHE: dict[str, tuple[float, ModelArtifactProfile]] = {}
_PROFILE_REVALIDATE_INTERVAL_S = 30.0


def _class_for_parameters(total_parameters: float) -> str:
    for boundary, name in _CLASS_BOUNDARIES:
        if total_parameters >= boundary and total_parameters > 0:
            return name
    return "unknown"


def _class_from_path_tokens(model_path: str) -> str:
    lowered = str(model_path or "").lower()

    def contains(token: str) -> bool:
        return re.search(
            rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
            lowered,
        ) is not None

    if any(contains(token) for token in _72B_PATH_TOKENS):
        return "72b"
    if any(contains(token) for token in _32B_PATH_TOKENS):
        return "32b"
    if any(contains(token) for token in _14B_PATH_TOKENS):
        return "14b"
    if any(contains(token) for token in _7B_PATH_TOKENS):
        return "7b"
    return "small"


def _estimate_parameters_from_config(config: dict) -> int:
    """Coarse transformer parameter estimate from architecture shape.

    Good to well within one class boundary for the dense decoder families
    this runtime serves (embedding + per-layer attention/MLP terms).
    """
    text = _text_model_config(config)
    try:
        hidden = float(text.get("hidden_size") or 0)
        layers = float(text.get("num_hidden_layers") or 0)
        inter = float(text.get("intermediate_size") or 0)
        vocab = float(text.get("vocab_size") or 0)
        heads = float(text.get("num_attention_heads") or 0)
        kv_heads = float(text.get("num_key_value_heads") or heads or 1)
    except (TypeError, ValueError):
        return 0
    if hidden <= 0 or layers <= 0:
        return 0
    if inter <= 0:
        inter = hidden * 4
    head_dim = hidden / heads if heads > 0 else hidden
    # Attention: Q + output are hidden×hidden; K/V shrink under GQA.
    attn = 2.0 * hidden * hidden + 2.0 * hidden * (kv_heads * head_dim)
    mlp = 3.0 * hidden * inter  # gate/up/down
    embed = 2.0 * vocab * hidden  # embed + lm_head (upper bound if tied)
    estimate = embed + layers * (attn + mlp)
    if not math.isfinite(estimate) or estimate <= 0:
        return 0
    return int(estimate)


def _quantization_bits(config: dict) -> int:
    quant = config.get("quantization")
    if isinstance(quant, dict):
        try:
            bits = int(quant.get("bits") or 0)
        except (TypeError, ValueError):
            return 0
        return bits if 0 < bits <= 32 else 0
    return 0


def _text_model_config(config: dict[str, object]) -> dict[str, object]:
    text = config.get("text_config")
    return dict(text) if isinstance(text, dict) else dict(config)


def _safe_config_int(config: dict[str, object], *keys: str) -> int:
    for key in keys:
        try:
            value = int(config.get(key) or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if value > 0:
            return value
    return 0


def _architecture_fields(config: dict[str, object]) -> dict[str, object]:
    text = _text_model_config(config)
    raw_architectures = config.get("architectures")
    architectures = tuple(
        str(value)
        for value in (raw_architectures if isinstance(raw_architectures, list) else [])
        if str(value).strip()
    )
    raw_layer_types = text.get("layer_types")
    layer_types = tuple(
        str(value)
        for value in (raw_layer_types if isinstance(raw_layer_types, list) else [])
    )
    layers = _safe_config_int(text, "num_hidden_layers", "n_layer", "num_layers")
    linear = sum(value == "linear_attention" for value in layer_types)
    full = sum(value == "full_attention" for value in layer_types)
    if not layer_types and layers:
        full = layers
    return {
        "model_type": str(text.get("model_type") or config.get("model_type") or ""),
        "architectures": architectures,
        "hidden_size": _safe_config_int(text, "hidden_size", "d_model", "n_embd"),
        "num_hidden_layers": layers,
        "num_attention_heads": _safe_config_int(text, "num_attention_heads", "n_head"),
        "num_key_value_heads": _safe_config_int(
            text, "num_key_value_heads", "num_kv_heads"
        ),
        "vocab_size": _safe_config_int(text, "vocab_size"),
        "native_context_window": _safe_config_int(
            text,
            "max_position_embeddings",
            "model_max_length",
            "seq_length",
            "max_sequence_length",
        ),
        "layer_types": layer_types,
        "linear_attention_layers": linear,
        "full_attention_layers": full,
    }


def _cache_key_stamp(config_path: Path, index_path: Path) -> tuple[float, float]:
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return (_mtime(config_path), _mtime(index_path))


def get_model_artifact_profile(model_path: str) -> ModelArtifactProfile:
    """Return the (cached) measured profile for a model artifact path.

    The mtime-validated cache below is the source of truth, but computing its
    key was itself the expensive part: ``exists()`` + ``resolve()`` (realpath
    walks every path component) + two ``stat()`` calls ran on EVERY call,
    including cache hits. That put four-plus filesystem syscalls on a hot
    status read — ``get_lane_status`` -> ``_model_is_deep_solver_lane`` ->
    ``model_size_class`` — which the background compute-budget policy calls
    from the event loop on every tick. Measured live: TICK STALL, background
    mean 19,823ms, with the event-loop stack sitting in ``posixpath.realpath``
    underneath exactly this function.

    So the hot path now does NO filesystem work at all: a monotonic fast cache
    keyed on the RAW path string answers repeat calls, and the mtime
    revalidation still runs, just at an interval instead of per call. A model
    directory's config does not change under a running process, so seconds of
    staleness is free; seconds of blocked event loop is not.
    """

    raw_key = str(model_path or "")
    now = time.monotonic()
    with _PROFILE_CACHE_LOCK:
        fresh = _PROFILE_FAST_CACHE.get(raw_key)
        if fresh is not None and (now - fresh[0]) < _PROFILE_REVALIDATE_INTERVAL_S:
            return fresh[1]

    resolved = raw_key
    try:
        root = Path(resolved).expanduser()
        real = root.resolve() if root.exists() else root
    except OSError:
        root = Path(resolved)
        real = root
    cache_id = str(real)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    stamp = _cache_key_stamp(config_path, index_path)
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(cache_id)
        if cached is not None and cached[0] == stamp:
            # Inline, NOT via a helper: `_PROFILE_CACHE_LOCK` is a plain
            # threading.Lock and therefore not reentrant, so any helper that
            # re-acquires it from inside this block self-deadlocks. That is not
            # hypothetical — it wedged the live runtime on the event-loop thread
            # during boot, with the faulthandler dump showing
            # get_model_artifact_profile -> _remember_fresh_profile blocked on
            # the lock its own caller already held.
            _store_fast_locked(raw_key, now, cached[1])
            return cached[1]

    profile = _build_profile(resolved, root, config_path, index_path)
    with _PROFILE_CACHE_LOCK:
        if len(_PROFILE_CACHE) >= _PROFILE_CACHE_MAX:
            _PROFILE_CACHE.pop(next(iter(_PROFILE_CACHE)), None)
        _PROFILE_CACHE[cache_id] = (stamp, profile)
        _store_fast_locked(raw_key, now, profile)
    return profile


def _store_fast_locked(
    raw_key: str, at: float, profile: ModelArtifactProfile
) -> None:
    """Record a profile under the zero-syscall fast key.

    CALLER MUST ALREADY HOLD ``_PROFILE_CACHE_LOCK``. It is never taken here:
    the lock is not reentrant.
    """

    _PROFILE_FAST_CACHE[raw_key] = (at, profile)
    while len(_PROFILE_FAST_CACHE) > _PROFILE_CACHE_MAX:
        _PROFILE_FAST_CACHE.pop(next(iter(_PROFILE_FAST_CACHE)), None)


def reset_model_artifact_profile_cache() -> None:
    """Drop both caches. For tests and for an artifact swapped under a run."""

    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.clear()
        _PROFILE_FAST_CACHE.clear()


def _build_profile(
    resolved: str,
    root: Path,
    config_path: Path,
    index_path: Path,
) -> ModelArtifactProfile:
    exists = False
    try:
        exists = root.exists()
    except OSError:
        exists = False
    if not exists or not root.is_dir():
        # Single-file or absent artifacts: declared naming is the only
        # evidence available. Say so.
        size_class = _class_from_path_tokens(resolved)
        weight_bytes = 0
        if exists:
            try:
                weight_bytes = int(root.stat().st_size)
            except OSError:
                weight_bytes = 0
        return ModelArtifactProfile(
            path=resolved,
            exists=exists,
            weight_bytes=weight_bytes,
            total_parameters=0,
            quantization_bits=0,
            size_class=size_class,
            evidence="path_tokens" if exists else "absent",
            fingerprint="",
        )

    config: dict = {}
    config_bytes = b""
    try:
        config_bytes = config_path.read_bytes()
        parsed = json.loads(config_bytes)
        if isinstance(parsed, dict):
            config = parsed
    except (OSError, ValueError):
        config = {}

    index_metadata: dict = {}
    try:
        index = json.loads(index_path.read_text())
        if isinstance(index, dict) and isinstance(index.get("metadata"), dict):
            index_metadata = index["metadata"]
    except (OSError, ValueError):
        index_metadata = {}

    weight_files: list[tuple[str, int]] = []
    summed_weight_bytes = 0
    try:
        for child in sorted(root.glob("*.safetensors")):
            try:
                size = int(child.stat().st_size)
            except OSError:
                continue
            weight_files.append((child.name, size))
            summed_weight_bytes += size
    except OSError:
        pass

    total_parameters = 0
    weight_bytes = 0
    evidence = "path_tokens"
    try:
        total_parameters = int(index_metadata.get("total_parameters") or 0)
        weight_bytes = int(index_metadata.get("total_size") or 0)
    except (TypeError, ValueError):
        total_parameters = 0
        weight_bytes = 0
    if total_parameters > 0:
        evidence = "index_metadata"
    else:
        total_parameters = _estimate_parameters_from_config(config)
        if total_parameters > 0:
            evidence = "config_estimate"
    if weight_bytes <= 0:
        weight_bytes = summed_weight_bytes
        if total_parameters <= 0 and weight_bytes > 0:
            evidence = "file_sizes"

    if total_parameters > 0:
        size_class = _class_for_parameters(float(total_parameters))
    elif weight_bytes > 0:
        # Weight bytes alone: infer through 4-bit density (~0.55 byte/param
        # after metadata) as a conservative bound, then classify.
        approx_params = float(weight_bytes) / 0.55
        size_class = _class_for_parameters(approx_params)
    else:
        size_class = _class_from_path_tokens(resolved)
        evidence = "path_tokens"

    digest = hashlib.sha256()
    digest.update(config_bytes)
    digest.update(
        json.dumps(index_metadata, sort_keys=True, default=str).encode("utf-8")
    )
    for name, size in weight_files:
        digest.update(f"{name}:{size}".encode())
    fingerprint = digest.hexdigest() if (config_bytes or weight_files) else ""

    architecture = _architecture_fields(config)
    profile = ModelArtifactProfile(
        path=resolved,
        exists=True,
        weight_bytes=weight_bytes,
        total_parameters=total_parameters,
        quantization_bits=_quantization_bits(config),
        size_class=size_class,
        evidence=evidence,
        fingerprint=fingerprint,
        model_type=str(architecture["model_type"]),
        architectures=tuple(architecture["architectures"]),
        hidden_size=int(architecture["hidden_size"]),
        num_hidden_layers=int(architecture["num_hidden_layers"]),
        num_attention_heads=int(architecture["num_attention_heads"]),
        num_key_value_heads=int(architecture["num_key_value_heads"]),
        vocab_size=int(architecture["vocab_size"]),
        native_context_window=int(architecture["native_context_window"]),
        layer_types=tuple(architecture["layer_types"]),
        linear_attention_layers=int(architecture["linear_attention_layers"]),
        full_attention_layers=int(architecture["full_attention_layers"]),
    )
    declared = _class_from_path_tokens(resolved)
    if profile.measured and declared != profile.size_class and declared != "small":
        # A measured artifact whose naming DISAGREES with its contents is
        # exactly the spoof/rename hazard this module exists for — surface
        # it instead of silently trusting either side.
        logger.warning(
            "Model artifact %s measures as %s-class (%.1fB params, %.1fGB) "
            "but is NAMED %s-class; measured evidence wins.",
            os.path.basename(resolved),
            profile.size_class,
            profile.total_parameters / 1e9,
            profile.weight_gb,
            declared,
        )
    return profile


def model_size_class(model_path: str) -> str:
    """Convenience: the measured (or declared-fallback) weight class."""
    return get_model_artifact_profile(model_path).size_class


def model_is_heavy(model_path: str) -> bool:
    return get_model_artifact_profile(model_path).is_heavy


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_normalize(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def build_model_artifact_descriptor(
    model_path: str | Path,
    *,
    repository_id: str = "",
    revision: str = "",
) -> dict[str, object]:
    """Build the promotion identity for one immutable local checkpoint.

    The hot-path profile fingerprint is intentionally cheap. Promotion needs
    the stronger object: every weight shard plus every tokenizer/config byte
    that changes behavior. This function is offline and may read the complete
    artifact.
    """

    root = Path(model_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model_artifact_not_directory")
    reset_model_artifact_profile_cache()
    profile = get_model_artifact_profile(str(root))
    if not profile.exists or not profile.measured:
        raise ValueError("model_artifact_unmeasured")

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
        model_behavior_bundle_identity,
    )

    material: dict[str, object] = {
        "schema": MODEL_ARTIFACT_DESCRIPTOR_SCHEMA,
        "canonical_path": str(root),
        "repository_id": str(repository_id).strip(),
        "revision": str(revision).strip(),
        "artifact_profile": _json_normalize(asdict(profile)),
        "weight_identity": full_weight_checkpoint_identity(root),
        "behavior_identity": model_behavior_bundle_identity(root),
    }
    material["descriptor_sha256"] = _canonical_digest(material)
    return material


def validate_model_artifact_descriptor(
    descriptor: dict[str, object],
    *,
    model_path: str | Path | None = None,
    verify_full_hash: bool = False,
) -> dict[str, object]:
    """Validate a descriptor and optionally re-hash the bound checkpoint."""

    if not isinstance(descriptor, dict):
        raise ValueError("descriptor_schema_invalid")
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
    if set(descriptor) != required or descriptor.get("schema") != MODEL_ARTIFACT_DESCRIPTOR_SCHEMA:
        raise ValueError("descriptor_schema_invalid")
    claimed = descriptor.get("descriptor_sha256")
    material = dict(descriptor)
    material.pop("descriptor_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_digest(material):
        raise ValueError("descriptor_digest_invalid")

    profile = descriptor.get("artifact_profile")
    weights = descriptor.get("weight_identity")
    behavior = descriptor.get("behavior_identity")
    if not isinstance(profile, dict) or not isinstance(weights, dict) or not isinstance(behavior, dict):
        raise ValueError("descriptor_schema_invalid")
    if not _is_sha256(weights.get("fingerprint")) or weights.get("method") != "sha256":
        raise ValueError("descriptor_weight_identity_invalid")
    if not _is_sha256(behavior.get("bundle_sha256")):
        raise ValueError("descriptor_behavior_identity_invalid")

    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve(strict=True)
        if str(resolved) != descriptor.get("canonical_path"):
            raise ValueError("descriptor_path_mismatch")
        if verify_full_hash:
            observed = build_model_artifact_descriptor(
                resolved,
                repository_id=str(descriptor.get("repository_id") or ""),
                revision=str(descriptor.get("revision") or ""),
            )
            if observed["descriptor_sha256"] != claimed:
                raise ValueError("descriptor_mismatch")
    return descriptor


def _validate_serving_qualification(
    value: dict[str, object],
    *,
    model_descriptor_sha256: str,
) -> str:
    required = {
        "schema",
        "verdict",
        "model_descriptor_sha256",
        "template_pass",
        "complete_answer_pass",
        "tool_contract_pass",
        "code_contract_pass",
        "context_pass",
        "latency_pass",
        "memory_pass",
        "served_context_tokens",
        "requested_context_tokens",
        "prefill_chunk_tokens",
        "evidence_sha256",
    }
    try:
        served_context = int(value.get("served_context_tokens") or 0)
        requested_context = int(value.get("requested_context_tokens") or 0)
        prefill_chunk = int(value.get("prefill_chunk_tokens") or 0)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("serving_qualification_incomplete") from None
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema") != SERVING_QUALIFICATION_SCHEMA
        or value.get("verdict") != "PASS"
        or value.get("model_descriptor_sha256") != model_descriptor_sha256
        or value.get("template_pass") is not True
        or value.get("complete_answer_pass") is not True
        or value.get("tool_contract_pass") is not True
        or value.get("code_contract_pass") is not True
        or value.get("context_pass") is not True
        or value.get("latency_pass") is not True
        or value.get("memory_pass") is not True
        or served_context <= 0
        or requested_context != served_context
        or prefill_chunk <= 0
        or not _is_sha256(value.get("evidence_sha256"))
    ):
        raise ValueError("serving_qualification_incomplete")
    return _canonical_digest(value)


def build_model_serving_profile(
    descriptor: dict[str, object],
    *,
    served_context_tokens: int,
    prefill_chunk_tokens: int,
    lane_limits: dict[str, dict[str, int]],
    qualification: dict[str, object],
) -> dict[str, object]:
    """Bind tested context/output budgets to one exact model artifact.

    A newer checkpoint does not receive larger limits by reputation. The
    limits become promotable only after complete-answer, latency, and memory
    evidence all pass for this exact artifact.
    """

    validate_model_artifact_descriptor(descriptor)
    descriptor_sha256 = str(descriptor["descriptor_sha256"])
    qualification_sha256 = _validate_serving_qualification(
        qualification,
        model_descriptor_sha256=descriptor_sha256,
    )
    artifact_profile = descriptor["artifact_profile"]
    assert isinstance(artifact_profile, dict)
    native_context = int(artifact_profile.get("native_context_window") or 0)
    served = int(served_context_tokens)
    chunk = int(prefill_chunk_tokens)
    if native_context <= 0 or served <= 0 or served > native_context:
        raise ValueError("serving_context_invalid")
    if chunk < 128 or chunk > min(served, 8192):
        raise ValueError("serving_prefill_chunk_invalid")
    if (
        int(qualification["served_context_tokens"]) != served
        or int(qualification["requested_context_tokens"]) != served
        or int(qualification["prefill_chunk_tokens"]) != chunk
    ):
        raise ValueError("serving_qualification_profile_mismatch")
    if not isinstance(lane_limits, dict) or set(lane_limits) != _REQUIRED_SERVING_LANES:
        raise ValueError("serving_lanes_incomplete")

    normalized_lanes: dict[str, dict[str, int]] = {}
    for lane in sorted(_REQUIRED_SERVING_LANES):
        limits = lane_limits.get(lane)
        if not isinstance(limits, dict) or set(limits) != {
            "max_input_tokens",
            "max_output_tokens",
        }:
            raise ValueError(f"serving_lane_invalid:{lane}")
        maximum_input = int(limits["max_input_tokens"])
        maximum_output = int(limits["max_output_tokens"])
        if maximum_input <= 0 or maximum_output <= 0:
            raise ValueError(f"serving_lane_invalid:{lane}")
        if maximum_input + maximum_output > served:
            raise ValueError(f"serving_context_overcommit:{lane}")
        normalized_lanes[lane] = {
            "max_input_tokens": maximum_input,
            "max_output_tokens": maximum_output,
        }

    material: dict[str, object] = {
        "schema": SERVING_PROFILE_SCHEMA,
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "native_context_tokens": native_context,
        "served_context_tokens": served,
        "prefill_chunk_tokens": chunk,
        "lanes": normalized_lanes,
        "qualification": _json_normalize(qualification),
        "qualification_sha256": qualification_sha256,
    }
    material["profile_sha256"] = _canonical_digest(material)
    return material


def validate_model_serving_profile(
    profile: dict[str, object],
    descriptor: dict[str, object],
) -> dict[str, object]:
    """Fail closed if a serving profile moved away from its measured model."""

    if not isinstance(profile, dict) or profile.get("schema") != SERVING_PROFILE_SCHEMA:
        raise ValueError("serving_profile_schema_invalid")
    if profile.get("model_descriptor_sha256") != descriptor.get("descriptor_sha256"):
        raise ValueError("serving_profile_model_identity_mismatch")
    validate_model_artifact_descriptor(descriptor)
    claimed = profile.get("profile_sha256")
    material = dict(profile)
    material.pop("profile_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_digest(material):
        raise ValueError("serving_profile_digest_invalid")
    qualification = profile.get("qualification")
    if not isinstance(qualification, dict):
        raise ValueError("serving_qualification_incomplete")
    if _validate_serving_qualification(
        qualification,
        model_descriptor_sha256=str(descriptor["descriptor_sha256"]),
    ) != profile.get(
        "qualification_sha256"
    ):
        raise ValueError("serving_qualification_digest_invalid")
    rebuilt = build_model_serving_profile(
        descriptor,
        served_context_tokens=int(profile.get("served_context_tokens") or 0),
        prefill_chunk_tokens=int(profile.get("prefill_chunk_tokens") or 0),
        lane_limits=dict(profile.get("lanes") or {}),
        qualification=qualification,
    )
    if rebuilt["profile_sha256"] != claimed:
        raise ValueError("serving_profile_invalid")
    return profile


__all__ = [
    "MODEL_ARTIFACT_DESCRIPTOR_SCHEMA",
    "SERVING_PROFILE_SCHEMA",
    "SERVING_QUALIFICATION_SCHEMA",
    "ModelArtifactProfile",
    "build_model_artifact_descriptor",
    "build_model_serving_profile",
    "get_model_artifact_profile",
    "model_is_heavy",
    "model_size_class",
    "reset_model_artifact_profile_cache",
    "validate_model_artifact_descriptor",
    "validate_model_serving_profile",
]
