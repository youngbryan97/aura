"""Compile verified internal corrections into persistent recurrent adapters.

The episodic RLC can already move a failed query into a verified answer basin
with a query-scoped minimum-norm write.  Supervised text imitation does not
teach that operation: it asks the adapter to reproduce answer tokens rather
than the internal state transition that caused the successful answer.

This module fits that transition directly.  Given input activations ``X`` and
teacher-minus-incumbent projection corrections ``Y``, it solves the dual ridge
problem

    delta_W = X.T @ inv(X @ X.T + lambda I) @ Y

and computes a rank-bounded factorization without materializing a full
model-width-by-model-width matrix.  The factors use the exact orientation and
scale of ``ScopedLoRALinear`` so they can become persistent recurrent tissue.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DISTILLATION_SCHEMA = "aura.verified_trajectory_distillation.v1"
EPISODIC_TRANSPLANT_SCHEMA = "aura.episodic_delta_transplant.v1"
TRAJECTORY_ARTIFACT_SCHEMA = "aura.verified_trajectory_artifact.v1"
TRAJECTORY_SAMPLE_COMPLEXITY_SCHEMA = (
    "aura.verified_trajectory_sample_complexity.v1"
)
TRAJECTORY_FACTORS_FILE = "factors.npz"
TRAJECTORY_MANIFEST_FILE = "manifest.json"
_MAX_TRAJECTORY_ARTIFACT_BYTES = 256 * 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DistilledTrajectoryFactors:
    """One named recurrent site's bounded low-rank correction."""

    site: str
    target_phase: str
    lora_a: np.ndarray
    lora_b: np.ndarray
    receipt: Mapping[str, Any]


def _validate_factor_receipt(factors: DistilledTrajectoryFactors) -> dict[str, Any]:
    receipt = dict(factors.receipt)
    receipt_sha256 = receipt.pop("receipt_sha256", None)
    if not _is_sha256(receipt_sha256) or _canonical_sha256(receipt) != receipt_sha256:
        raise ValueError(f"trajectory factor receipt is invalid at {factors.site}")
    if receipt.get("schema") not in {
        DISTILLATION_SCHEMA,
        EPISODIC_TRANSPLANT_SCHEMA,
    }:
        raise ValueError(f"trajectory factor schema is invalid at {factors.site}")
    if receipt.get("site") != factors.site:
        raise ValueError(f"trajectory factor site binding differs at {factors.site}")
    if receipt.get("target_phase") != factors.target_phase:
        raise ValueError(f"trajectory factor phase binding differs at {factors.site}")
    if receipt.get("lora_a_sha256") != _array_sha256(factors.lora_a):
        raise ValueError(f"trajectory A factor hash differs at {factors.site}")
    if receipt.get("lora_b_sha256") != _array_sha256(factors.lora_b):
        raise ValueError(f"trajectory B factor hash differs at {factors.site}")
    rank = factors.lora_a.shape[1]
    if factors.lora_b.shape[0] != rank:
        raise ValueError(f"trajectory factor rank differs at {factors.site}")
    if receipt["schema"] == EPISODIC_TRANSPLANT_SCHEMA:
        dimensions = (
            receipt.get("rank"),
            receipt.get("input_width"),
            receipt.get("output_width"),
        )
    else:
        dimensions = (
            receipt.get("effective_rank"),
            receipt.get("input_width"),
            receipt.get("output_width"),
        )
    if dimensions != (rank, factors.lora_a.shape[0], factors.lora_b.shape[1]):
        raise ValueError(f"trajectory factor dimensions differ at {factors.site}")
    return {**receipt, "receipt_sha256": receipt_sha256}


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode numeric arrays without ZIP timestamps or pickle payloads."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name or name.endswith(".npy"):
                raise ValueError("trajectory tensor key is invalid")
            array = np.ascontiguousarray(arrays[name])
            if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"trajectory tensor is not numeric: {name}")
            payload = io.BytesIO()
            np.save(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return output.getvalue()


def build_verified_trajectory_artifact(
    inventory: Mapping[str, DistilledTrajectoryFactors],
    *,
    checkpoint_fingerprint: str,
    source_evidence_sha256: str,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Build a deterministic, checkpoint-bound exact trajectory package."""

    if not _is_sha256(checkpoint_fingerprint):
        raise ValueError("trajectory artifact checkpoint fingerprint is invalid")
    if not _is_sha256(source_evidence_sha256):
        raise ValueError("trajectory artifact source evidence hash is invalid")
    sites = tuple(sorted(inventory))
    if not sites or len(sites) != len(set(sites)):
        raise ValueError("trajectory artifact inventory is empty or duplicated")

    arrays: dict[str, np.ndarray] = {}
    tensor_manifest: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for index, site in enumerate(sites):
        factors = inventory[site]
        if not isinstance(factors, DistilledTrajectoryFactors) or factors.site != site:
            raise ValueError(f"trajectory artifact inventory binding differs at {site}")
        if factors.target_phase not in {"recurrence", "decode"}:
            raise ValueError(f"trajectory artifact phase is invalid at {site}")
        lora_a = _finite_matrix(factors.lora_a, name=f"{site} lora A").astype(
            np.float32
        )
        lora_b = _finite_matrix(factors.lora_b, name=f"{site} lora B").astype(
            np.float32
        )
        if lora_a.shape[1] != lora_b.shape[0]:
            raise ValueError(f"trajectory artifact rank differs at {site}")
        a_key = f"site_{index:04d}_lora_a"
        b_key = f"site_{index:04d}_lora_b"
        arrays[a_key] = lora_a
        arrays[b_key] = lora_b
        tensor_manifest[site] = {
            "lora_a_key": a_key,
            "lora_a_shape": list(lora_a.shape),
            "lora_a_dtype": str(lora_a.dtype),
            "lora_a_sha256": _array_sha256(lora_a),
            "lora_b_key": b_key,
            "lora_b_shape": list(lora_b.shape),
            "lora_b_dtype": str(lora_b.dtype),
            "lora_b_sha256": _array_sha256(lora_b),
        }
        receipts[site] = _validate_factor_receipt(
            DistilledTrajectoryFactors(
                site=site,
                target_phase=factors.target_phase,
                lora_a=lora_a,
                lora_b=lora_b,
                receipt=factors.receipt,
            )
        )

    factors_payload = _deterministic_npz_bytes(arrays)
    body = {
        "schema": TRAJECTORY_ARTIFACT_SCHEMA,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "source_evidence_sha256": source_evidence_sha256,
        "sites": list(sites),
        "site_phases": {site: inventory[site].target_phase for site in sites},
        "operation_modes": {
            site: (
                "episodic_exact"
                if receipts[site]["schema"] == EPISODIC_TRANSPLANT_SCHEMA
                else "scoped_lora"
            )
            for site in sites
        },
        "factor_receipts": receipts,
        "tensor_manifest": tensor_manifest,
        "tensor_artifact": {
            "name": TRAJECTORY_FACTORS_FILE,
            "sha256": hashlib.sha256(factors_payload).hexdigest(),
            "size_bytes": len(factors_payload),
            "keys": sorted(arrays),
        },
    }
    manifest = {**body, "receipt_sha256": _canonical_sha256(body)}
    manifest_payload = _canonical_json_bytes(manifest) + b"\n"
    return factors_payload, manifest_payload, manifest


def publish_verified_trajectory_artifact(
    artifact_dir: Path | str,
    inventory: Mapping[str, DistilledTrajectoryFactors],
    *,
    checkpoint_fingerprint: str,
    source_evidence_sha256: str,
) -> dict[str, Any]:
    """Publish a complete trajectory package through the governed batch lane."""

    from core.brain.llm.latent_cortex.persistence import (
        get_latent_cortex_persistence,
    )

    target = Path(artifact_dir).expanduser()
    if target.is_symlink():
        raise ValueError("trajectory artifact destination cannot be a symlink")
    if target.exists() and any(target.iterdir()):
        raise ValueError("trajectory artifact destination is not empty")
    factors_payload, manifest_payload, manifest = build_verified_trajectory_artifact(
        inventory,
        checkpoint_fingerprint=checkpoint_fingerprint,
        source_evidence_sha256=source_evidence_sha256,
    )
    receipt = get_latent_cortex_persistence().publish_verified_trajectory_artifact(
        target,
        factors_payload=factors_payload,
        manifest_payload=manifest_payload,
    )
    expected_hashes = {
        str(target.resolve() / TRAJECTORY_FACTORS_FILE): hashlib.sha256(
            factors_payload
        ).hexdigest(),
        str(target.resolve() / TRAJECTORY_MANIFEST_FILE): hashlib.sha256(
            manifest_payload
        ).hexdigest(),
    }
    if set(receipt.paths) != set(expected_hashes) or dict(receipt.sha256) != expected_hashes:
        raise RuntimeError("trajectory artifact publication receipt differs")
    return {
        "artifact_dir": str(target.resolve()),
        "manifest": manifest,
        "publication": {
            "transaction_id": receipt.transaction_id,
            "paths": list(receipt.paths),
            "sha256": dict(receipt.sha256),
        },
    }


def _strict_manifest(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"trajectory manifest contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("trajectory artifact manifest is invalid JSON") from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value) + b"\n":
        raise ValueError("trajectory artifact manifest is not canonical")
    return value


def load_verified_trajectory_artifact(
    artifact_dir: Path | str,
    *,
    expected_checkpoint_fingerprint: str,
    expected_source_evidence_sha256: str | None = None,
) -> tuple[dict[str, DistilledTrajectoryFactors], dict[str, Any]]:
    """Load and verify a complete exact trajectory package without mutation."""

    from core.runtime.file_read_gateway import read_stable_directory_files

    if not _is_sha256(expected_checkpoint_fingerprint):
        raise ValueError("expected trajectory checkpoint fingerprint is invalid")
    if (
        expected_source_evidence_sha256 is not None
        and not _is_sha256(expected_source_evidence_sha256)
    ):
        raise ValueError("expected trajectory source evidence hash is invalid")
    payloads = read_stable_directory_files(
        artifact_dir,
        names={TRAJECTORY_FACTORS_FILE, TRAJECTORY_MANIFEST_FILE},
        max_bytes_per_file=_MAX_TRAJECTORY_ARTIFACT_BYTES,
    )
    manifest = _strict_manifest(payloads[TRAJECTORY_MANIFEST_FILE])
    receipt_sha256 = manifest.get("receipt_sha256")
    body = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
    if (
        manifest.get("schema") != TRAJECTORY_ARTIFACT_SCHEMA
        or not _is_sha256(receipt_sha256)
        or _canonical_sha256(body) != receipt_sha256
    ):
        raise ValueError("trajectory artifact manifest receipt is invalid")
    if manifest.get("checkpoint_fingerprint") != expected_checkpoint_fingerprint:
        raise ValueError("trajectory artifact checkpoint fingerprint differs")
    if not _is_sha256(manifest.get("source_evidence_sha256")):
        raise ValueError("trajectory artifact source evidence hash is invalid")
    if (
        expected_source_evidence_sha256 is not None
        and manifest.get("source_evidence_sha256")
        != expected_source_evidence_sha256
    ):
        raise ValueError("trajectory artifact source evidence hash differs")
    tensor_artifact = manifest.get("tensor_artifact")
    if not isinstance(tensor_artifact, dict):
        raise ValueError("trajectory tensor artifact binding is missing")
    factors_payload = payloads[TRAJECTORY_FACTORS_FILE]
    if (
        tensor_artifact.get("name") != TRAJECTORY_FACTORS_FILE
        or tensor_artifact.get("size_bytes") != len(factors_payload)
        or tensor_artifact.get("sha256")
        != hashlib.sha256(factors_payload).hexdigest()
    ):
        raise ValueError("trajectory tensor artifact binding differs")

    sites = manifest.get("sites")
    phases = manifest.get("site_phases")
    modes = manifest.get("operation_modes")
    tensor_manifest = manifest.get("tensor_manifest")
    receipts = manifest.get("factor_receipts")
    if (
        not isinstance(sites, list)
        or not sites
        or sites != sorted(sites)
        or len(sites) != len(set(sites))
        or not all(isinstance(site, str) and site for site in sites)
        or not all(isinstance(value, dict) for value in (phases, modes, tensor_manifest, receipts))
        or set(phases) != set(sites)
        or set(modes) != set(sites)
        or set(tensor_manifest) != set(sites)
        or set(receipts) != set(sites)
    ):
        raise ValueError("trajectory artifact site topology is invalid")

    expected_keys = tensor_artifact.get("keys")
    if (
        not isinstance(expected_keys, list)
        or not expected_keys
        or expected_keys != sorted(expected_keys)
        or len(expected_keys) != len(set(expected_keys))
        or not all(isinstance(key, str) and key for key in expected_keys)
    ):
        raise ValueError("trajectory tensor key manifest is invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(factors_payload)) as raw_archive:
            members = raw_archive.infolist()
            if (
                len(members) != len(expected_keys)
                or [member.filename for member in members]
                != [f"{key}.npy" for key in expected_keys]
                or any(
                    member.compress_type != zipfile.ZIP_STORED
                    or member.file_size <= 0
                    or member.file_size > _MAX_TRAJECTORY_ARTIFACT_BYTES
                    or member.compress_size != member.file_size
                    for member in members
                )
                or sum(member.file_size for member in members)
                > _MAX_TRAJECTORY_ARTIFACT_BYTES
            ):
                raise ValueError("trajectory tensor ZIP inventory is invalid")
        with np.load(io.BytesIO(factors_payload), allow_pickle=False) as archive:
            if archive.files != expected_keys:
                raise ValueError("trajectory tensor key inventory differs")
            arrays = {key: np.array(archive[key]) for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("trajectory tensor key"):
            raise
        raise ValueError("trajectory tensor artifact is unreadable") from exc

    inventory: dict[str, DistilledTrajectoryFactors] = {}
    for site in sites:
        metadata = tensor_manifest[site]
        receipt = receipts[site]
        if not isinstance(metadata, dict) or not isinstance(receipt, dict):
            raise ValueError(f"trajectory site metadata is invalid at {site}")
        a_key = metadata.get("lora_a_key")
        b_key = metadata.get("lora_b_key")
        if a_key not in arrays or b_key not in arrays:
            raise ValueError(f"trajectory tensor is missing at {site}")
        lora_a = arrays[a_key]
        lora_b = arrays[b_key]
        for label, array in (("lora_a", lora_a), ("lora_b", lora_b)):
            if (
                array.ndim != 2
                or not array.size
                or array.dtype != np.float32
                or not np.all(np.isfinite(array))
                or metadata.get(f"{label}_shape") != list(array.shape)
                or metadata.get(f"{label}_dtype") != str(array.dtype)
                or metadata.get(f"{label}_sha256") != _array_sha256(array)
            ):
                raise ValueError(f"trajectory tensor metadata differs at {site}:{label}")
        factors = DistilledTrajectoryFactors(
            site=site,
            target_phase=phases[site],
            lora_a=lora_a,
            lora_b=lora_b,
            receipt=receipt,
        )
        validated_receipt = _validate_factor_receipt(factors)
        expected_mode = (
            "episodic_exact"
            if validated_receipt["schema"] == EPISODIC_TRANSPLANT_SCHEMA
            else "scoped_lora"
        )
        if modes[site] != expected_mode:
            raise ValueError(f"trajectory operation mode differs at {site}")
        inventory[site] = factors
    return inventory, manifest


def compile_episodic_delta_factors(
    episodic_u: Any,
    episodic_v: Any,
    *,
    site: str,
    episodic_scale: float,
    adapter_scale: float,
    target_phase: str = "decode",
) -> DistilledTrajectoryFactors:
    """Compile one accepted temporary operator into an exact scoped LoRA.

    The episodic wrapper emits ``s * (x @ V.T) @ U.T`` while Aura's persistent
    scoped LoRA emits ``s * (x @ A) @ B``.  Preserving the scale and setting
    ``A = V.T`` and ``B = U.T`` keeps both algebra and MLX operation order
    identical, without fitting, normalization, or teacher text.
    """

    if not isinstance(site, str) or not site.strip() or site != site.strip():
        raise ValueError("episodic transplant site is invalid")
    if target_phase not in {"recurrence", "decode"}:
        raise ValueError("episodic transplant target phase is invalid")
    for value, label in (
        (episodic_scale, "episodic scale"),
        (adapter_scale, "adapter scale"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"episodic transplant {label} must be positive")
    if float(episodic_scale) != float(adapter_scale):
        raise ValueError("episodic transplant requires the original operator scale")

    u = _finite_matrix(episodic_u, name="episodic U")
    v = _finite_matrix(episodic_v, name="episodic V")
    if u.shape[1] != v.shape[0]:
        raise ValueError("episodic transplant rank dimensions differ")
    if np.linalg.norm(u) <= 1e-12 or np.linalg.norm(v) <= 1e-12:
        raise ValueError("episodic transplant operator is collapsed")

    lora_a = v.T.astype(np.float32)
    lora_b = u.T.astype(np.float32)
    body = {
        "schema": EPISODIC_TRANSPLANT_SCHEMA,
        "site": site,
        "target_phase": target_phase,
        "rank": int(u.shape[1]),
        "input_width": int(v.shape[1]),
        "output_width": int(u.shape[0]),
        "episodic_scale": float(episodic_scale),
        "adapter_scale": float(adapter_scale),
        "factor_scale": 1.0,
        "operator_relation": "A=V.T;B=U.T;adapter_scale=episodic_scale",
        "bitwise_operation_order_preserved": True,
        "episodic_u_sha256": _array_sha256(u),
        "episodic_v_sha256": _array_sha256(v),
        "lora_a_sha256": _array_sha256(lora_a),
        "lora_b_sha256": _array_sha256(lora_b),
    }
    return DistilledTrajectoryFactors(
        site=site,
        target_phase=target_phase,
        lora_a=lora_a,
        lora_b=lora_b,
        receipt={**body, "receipt_sha256": _canonical_sha256(body)},
    )


def compile_episodic_delta_inventory(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    target: str,
    target_phase: str = "decode",
) -> dict[str, DistilledTrajectoryFactors]:
    """Compile a complete ``snapshot_delta`` inventory without proxy pairing."""

    # o_proj lives under attention, which three layers in four of a hybrid
    # checkpoint do not have; the resolver below reports that rather than
    # assuming the mapping holds.
    parents = {"o_proj": "self_attn", "down_proj": "mlp"}
    if target not in parents:
        raise ValueError("episodic transplant projection target is unsupported")
    if (
        not isinstance(snapshots, Sequence)
        or isinstance(snapshots, (str, bytes, bytearray))
        or not snapshots
    ):
        raise ValueError("episodic transplant snapshot inventory is empty")
    result: dict[str, DistilledTrajectoryFactors] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise ValueError("episodic transplant snapshot row is invalid")
        try:
            layer = snapshot["layer"]
            scale = snapshot["scale"]
            u = snapshot["U"]
            v = snapshot["V"]
        except KeyError as exc:
            raise ValueError("episodic transplant snapshot row is incomplete") from exc
        if type(layer) is not int or layer < 0:
            raise ValueError("episodic transplant layer is invalid")
        site = f"model.layers.{layer}.{parents[target]}.{target}"
        if site in result:
            raise ValueError("episodic transplant layer inventory contains duplicates")
        result[site] = compile_episodic_delta_factors(
            u,
            v,
            site=site,
            episodic_scale=scale,
            adapter_scale=scale,
            target_phase=target_phase,
        )
    return result


def fit_verified_trajectory_factors(
    input_features: Any,
    output_corrections: Any,
    *,
    site: str,
    rank: int,
    regularization: float,
    gain: float,
    adapter_scale: float,
    normalize_corrections: bool = True,
    target_phase: str = "recurrence",
) -> DistilledTrajectoryFactors:
    """Fit a low-rank recurrent operator to verified activation corrections.

    ``input_features`` and ``output_corrections`` have one row per private,
    independently verified teaching pair.  By default each correction is
    normalized before fitting, matching the successful episodic trajectory
    transplant and preventing long teacher traces from receiving more
    authority merely because their residual norm is larger.
    """

    if not isinstance(site, str) or not site.strip() or site != site.strip():
        raise ValueError("trajectory distillation site is invalid")
    if target_phase not in {"recurrence", "decode"}:
        raise ValueError("trajectory distillation target phase is invalid")
    if type(rank) is not int or rank < 1:
        raise ValueError("trajectory distillation rank must be positive")
    for value, label, lower, upper in (
        (regularization, "regularization", 0.0, 1e6),
        (gain, "gain", 0.0, 16.0),
        (adapter_scale, "adapter scale", 0.0, 4096.0),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not lower < float(value) <= upper
        ):
            raise ValueError(f"trajectory distillation {label} is invalid")

    inputs = _finite_matrix(input_features, name="input features")
    corrections = _finite_matrix(output_corrections, name="output corrections")
    if inputs.shape[0] != corrections.shape[0]:
        raise ValueError("trajectory teaching pair counts differ")
    if inputs.shape[0] < 2:
        raise ValueError("trajectory distillation requires at least two teaching pairs")
    effective_rank = min(rank, inputs.shape[0], inputs.shape[1], corrections.shape[1])

    target = corrections.copy()
    correction_norms = np.linalg.norm(target, axis=1)
    if np.any(correction_norms <= 1e-10):
        raise ValueError("trajectory correction contains a collapsed row")
    if normalize_corrections:
        target /= correction_norms[:, None]

    # Solve in sample space. This keeps memory O(n*d), not O(d_in*d_out).
    gram = inputs @ inputs.T
    system = gram + float(regularization) * np.eye(inputs.shape[0])
    try:
        dual = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("trajectory ridge system is singular") from exc

    # W = P @ Q. Thin QR decompositions reduce its SVD to an n-by-n core.
    left = inputs.T
    right = dual
    q_left, r_left = np.linalg.qr(left, mode="reduced")
    q_right, r_right = np.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, singular_values, vt_core = np.linalg.svd(core, full_matrices=False)
    retained = singular_values[:effective_rank]
    if not retained.size or retained[0] <= 1e-12:
        raise ValueError("trajectory correction map collapsed")
    sqrt_s = np.sqrt(retained)
    left_factor = (q_left @ u_core[:, :effective_rank]) * sqrt_s[None, :]
    right_factor = sqrt_s[:, None] * (vt_core[:effective_rank] @ q_right.T)

    # ScopedLoRALinear emits scale * (x @ A) @ B.
    factor_scale = math.sqrt(float(gain) / float(adapter_scale))
    lora_a = (left_factor * factor_scale).astype(np.float32)
    lora_b = (right_factor * factor_scale).astype(np.float32)
    predicted = float(adapter_scale) * (inputs @ lora_a.astype(np.float64)) @ lora_b.astype(
        np.float64
    )
    residual = predicted - float(gain) * target
    target_energy = float(np.sum(np.square(float(gain) * target)))
    residual_energy = float(np.sum(np.square(residual)))
    relative_error = math.sqrt(residual_energy / max(target_energy, 1e-20))
    explained_energy = float(
        np.sum(np.square(retained)) / max(np.sum(np.square(singular_values)), 1e-20)
    )
    receipt_body = {
        "schema": DISTILLATION_SCHEMA,
        "site": site,
        "target_phase": target_phase,
        "teaching_pairs": int(inputs.shape[0]),
        "input_width": int(inputs.shape[1]),
        "output_width": int(corrections.shape[1]),
        "requested_rank": rank,
        "effective_rank": effective_rank,
        "regularization": float(regularization),
        "gain": float(gain),
        "adapter_scale": float(adapter_scale),
        "corrections_normalized": bool(normalize_corrections),
        "correction_norm_min": float(np.min(correction_norms)),
        "correction_norm_max": float(np.max(correction_norms)),
        "singular_values": [float(value) for value in retained],
        "retained_operator_energy": explained_energy,
        "training_relative_error": relative_error,
        "input_features_sha256": _array_sha256(inputs),
        "output_corrections_sha256": _array_sha256(corrections),
        "lora_a_sha256": _array_sha256(lora_a),
        "lora_b_sha256": _array_sha256(lora_b),
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": _canonical_sha256(receipt_body),
    }
    return DistilledTrajectoryFactors(
        site=site,
        target_phase=target_phase,
        lora_a=lora_a,
        lora_b=lora_b,
        receipt=receipt,
    )


def fit_verified_trajectory_inventory(
    teaching_pairs: Mapping[str, tuple[Any, Any]],
    *,
    rank: int,
    regularization: float,
    gain: float,
    adapter_scale: float,
    site_phases: Mapping[str, str] | None = None,
    normalize_corrections: bool = True,
) -> dict[str, DistilledTrajectoryFactors]:
    """Fit every named site and reject partial or inconsistent inventories."""

    if not isinstance(teaching_pairs, Mapping) or not teaching_pairs:
        raise ValueError("trajectory teaching inventory is empty")
    result: dict[str, DistilledTrajectoryFactors] = {}
    phases = dict(site_phases or {})
    if phases and set(phases) != set(teaching_pairs):
        raise ValueError("trajectory site phase inventory differs from teaching pairs")
    pair_counts: set[int] = set()
    for site in sorted(teaching_pairs):
        pair = teaching_pairs[site]
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("trajectory teaching inventory row is invalid")
        fitted = fit_verified_trajectory_factors(
            pair[0],
            pair[1],
            site=site,
            rank=rank,
            regularization=regularization,
            gain=gain,
            adapter_scale=adapter_scale,
            normalize_corrections=normalize_corrections,
            target_phase=phases.get(site, "recurrence"),
        )
        pair_counts.add(int(fitted.receipt["teaching_pairs"]))
        result[site] = fitted
    if len(pair_counts) != 1:
        raise ValueError("trajectory teaching inventories have unequal pair counts")
    return result


def evaluate_verified_trajectory_transfer(
    fitted: Mapping[str, DistilledTrajectoryFactors],
    validation_pairs: Mapping[str, tuple[Any, Any]],
    *,
    training_pairs: Mapping[str, tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Measure whether one fitted neural rule transfers beyond its fit rows."""

    if not fitted or set(fitted) != set(validation_pairs):
        raise ValueError("trajectory transfer site inventories differ")
    if training_pairs is not None and set(training_pairs) != set(fitted):
        raise ValueError("trajectory transfer training inventory differs")

    rows: dict[str, dict[str, Any]] = {}
    total_target_energy = 0.0
    total_residual_energy = 0.0
    total_dot = 0.0
    total_prediction_energy = 0.0
    input_coverage_energy = 0.0
    input_total_energy = 0.0
    correction_coverage_energy = 0.0
    correction_total_energy = 0.0
    for site in sorted(fitted):
        factors = fitted[site]
        receipt = _validate_factor_receipt(factors)
        pair = validation_pairs[site]
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"trajectory validation pair is invalid at {site}")
        inputs = _finite_matrix(pair[0], name=f"{site} validation inputs")
        corrections = _finite_matrix(pair[1], name=f"{site} validation corrections")
        if (
            inputs.shape[0] != corrections.shape[0]
            or inputs.shape[1] != factors.lora_a.shape[0]
            or corrections.shape[1] != factors.lora_b.shape[1]
        ):
            raise ValueError(f"trajectory validation geometry differs at {site}")
        correction_norms = np.linalg.norm(corrections, axis=1)
        if np.any(correction_norms <= 1e-10):
            raise ValueError(f"trajectory validation correction collapsed at {site}")
        target = corrections.copy()
        if bool(receipt.get("corrections_normalized")):
            target /= correction_norms[:, None]
        target *= float(receipt["gain"])
        predicted = (
            float(receipt["adapter_scale"])
            * (inputs @ factors.lora_a.astype(np.float64))
            @ factors.lora_b.astype(np.float64)
        )
        residual = predicted - target
        target_energy = float(np.sum(np.square(target)))
        residual_energy = float(np.sum(np.square(residual)))
        prediction_energy = float(np.sum(np.square(predicted)))
        dot = float(np.sum(predicted * target))
        relative_error = math.sqrt(residual_energy / max(target_energy, 1e-20))
        cosine = dot / math.sqrt(max(target_energy * prediction_energy, 1e-40))

        input_coverage = None
        correction_coverage = None
        if training_pairs is not None:
            training = training_pairs[site]
            if not isinstance(training, Sequence) or len(training) != 2:
                raise ValueError(f"trajectory training pair is invalid at {site}")
            train_inputs = _finite_matrix(
                training[0],
                name=f"{site} training inputs",
            )
            train_corrections = _finite_matrix(
                training[1],
                name=f"{site} training corrections",
            )
            if (
                train_inputs.shape[1] != inputs.shape[1]
                or train_corrections.shape[1] != corrections.shape[1]
            ):
                raise ValueError(f"trajectory training geometry differs at {site}")
            input_basis = np.linalg.svd(train_inputs, full_matrices=False)[2]
            input_projection = (inputs @ input_basis.T) @ input_basis
            input_projected_energy = float(np.sum(np.square(input_projection)))
            input_energy = float(np.sum(np.square(inputs)))
            input_coverage = input_projected_energy / max(input_energy, 1e-20)
            correction_basis = np.linalg.svd(
                train_corrections,
                full_matrices=False,
            )[2]
            correction_projection = (
                corrections @ correction_basis.T
            ) @ correction_basis
            correction_projected_energy = float(
                np.sum(np.square(correction_projection))
            )
            correction_energy = float(np.sum(np.square(corrections)))
            correction_coverage = correction_projected_energy / max(
                correction_energy,
                1e-20,
            )
            input_coverage_energy += input_projected_energy
            input_total_energy += input_energy
            correction_coverage_energy += correction_projected_energy
            correction_total_energy += correction_energy

        rows[site] = {
            "validation_pairs": int(inputs.shape[0]),
            "relative_error": relative_error,
            "cosine": cosine,
            "prediction_to_target_energy": prediction_energy
            / max(target_energy, 1e-20),
            "better_than_zero_operator": relative_error < 1.0,
            "input_subspace_coverage": input_coverage,
            "correction_subspace_coverage": correction_coverage,
        }
        total_target_energy += target_energy
        total_residual_energy += residual_energy
        total_prediction_energy += prediction_energy
        total_dot += dot

    aggregate_relative_error = math.sqrt(
        total_residual_energy / max(total_target_energy, 1e-20)
    )
    aggregate_cosine = total_dot / math.sqrt(
        max(total_target_energy * total_prediction_energy, 1e-40)
    )
    return {
        "schema": "aura.verified_trajectory_transfer_diagnostic.v1",
        "sites": rows,
        "aggregate": {
            "relative_error": aggregate_relative_error,
            "cosine": aggregate_cosine,
            "prediction_to_target_energy": total_prediction_energy
            / max(total_target_energy, 1e-20),
            "better_than_zero_operator": aggregate_relative_error < 1.0,
            "input_subspace_coverage": (
                input_coverage_energy / max(input_total_energy, 1e-20)
                if training_pairs is not None
                else None
            ),
            "correction_subspace_coverage": (
                correction_coverage_energy / max(correction_total_energy, 1e-20)
                if training_pairs is not None
                else None
            ),
        },
    }


def fit_verified_trajectory_sample_complexity(
    teaching_pairs: Mapping[str, tuple[Any, Any]],
    validation_cohorts: Mapping[str, Mapping[str, tuple[Any, Any]]],
    *,
    sample_rows: Sequence[int],
    rank: int,
    regularization: float,
    gain: float,
    adapter_scale: float,
    site_phases: Mapping[str, str] | None = None,
    normalize_corrections: bool = True,
) -> tuple[dict[str, Any], dict[str, DistilledTrajectoryFactors]]:
    """Fit nested prefixes and measure transfer on independent cohorts.

    The learner and every hyperparameter remain fixed while only the amount
    of admitted teaching evidence changes.  Validation cohorts are evaluated
    independently instead of being concatenated, so a favorable aggregate
    cannot hide a fresh seed where the operator is worse than zero.

    ``sample_rows`` must end at the complete teaching inventory.  Callers that
    capture multi-token examples should pass only complete-example boundaries;
    the model-facing canary records and enforces those boundaries.
    """

    if not isinstance(teaching_pairs, Mapping) or not teaching_pairs:
        raise ValueError("trajectory sample-complexity teaching inventory is empty")
    if not isinstance(validation_cohorts, Mapping) or len(validation_cohorts) < 2:
        raise ValueError(
            "trajectory sample complexity requires at least two validation cohorts"
        )
    cohort_names = tuple(sorted(validation_cohorts))
    if any(
        not isinstance(name, str) or not name.strip() or name != name.strip()
        for name in cohort_names
    ):
        raise ValueError("trajectory sample-complexity cohort identity is invalid")

    sites = tuple(sorted(teaching_pairs))
    row_counts: set[int] = set()
    validated_teaching: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for site in sites:
        pair = teaching_pairs[site]
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"trajectory teaching pair is invalid at {site}")
        inputs = _finite_matrix(pair[0], name=f"{site} teaching inputs")
        corrections = _finite_matrix(
            pair[1], name=f"{site} teaching corrections"
        )
        if inputs.shape[0] != corrections.shape[0]:
            raise ValueError(f"trajectory teaching pair counts differ at {site}")
        row_counts.add(int(inputs.shape[0]))
        validated_teaching[site] = (inputs, corrections)
    if len(row_counts) != 1:
        raise ValueError("trajectory teaching inventories have unequal pair counts")
    total_rows = row_counts.pop()

    if (
        isinstance(sample_rows, (str, bytes))
        or len(sample_rows) < 2
        or any(type(value) is not int or value < 2 for value in sample_rows)
        or tuple(sorted(set(sample_rows))) != tuple(sample_rows)
        or sample_rows[-1] != total_rows
    ):
        raise ValueError(
            "trajectory sample rows must be increasing complete prefixes ending at all rows"
        )

    validated_cohorts: dict[str, Mapping[str, tuple[Any, Any]]] = {}
    cohort_row_counts: dict[str, int] = {}
    for cohort_name in cohort_names:
        cohort = validation_cohorts[cohort_name]
        if not isinstance(cohort, Mapping) or set(cohort) != set(sites):
            raise ValueError(
                f"trajectory validation site inventory differs in {cohort_name}"
            )
        rows: set[int] = set()
        for site in sites:
            pair = cohort[site]
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(
                    f"trajectory validation pair is invalid at {cohort_name}:{site}"
                )
            inputs = _finite_matrix(
                pair[0], name=f"{cohort_name}:{site} validation inputs"
            )
            corrections = _finite_matrix(
                pair[1], name=f"{cohort_name}:{site} validation corrections"
            )
            if inputs.shape[0] != corrections.shape[0]:
                raise ValueError(
                    f"trajectory validation pair counts differ at {cohort_name}:{site}"
                )
            rows.add(int(inputs.shape[0]))
        if len(rows) != 1:
            raise ValueError(
                f"trajectory validation inventories have unequal rows in {cohort_name}"
            )
        cohort_row_counts[cohort_name] = rows.pop()
        validated_cohorts[cohort_name] = cohort

    stages: list[dict[str, Any]] = []
    final_inventory: dict[str, DistilledTrajectoryFactors] | None = None
    for row_count in sample_rows:
        subset = {
            site: (inputs[:row_count], corrections[:row_count])
            for site, (inputs, corrections) in validated_teaching.items()
        }
        fitted = fit_verified_trajectory_inventory(
            subset,
            rank=rank,
            regularization=regularization,
            gain=gain,
            adapter_scale=adapter_scale,
            site_phases=site_phases,
            normalize_corrections=normalize_corrections,
        )
        cohort_reports = {
            name: evaluate_verified_trajectory_transfer(
                fitted,
                validated_cohorts[name],
                training_pairs=subset,
            )
            for name in cohort_names
        }
        aggregates = [
            cohort_reports[name]["aggregate"] for name in cohort_names
        ]
        relative_errors = [float(row["relative_error"]) for row in aggregates]
        cosines = [float(row["cosine"]) for row in aggregates]
        site_rows = [
            site_row
            for name in cohort_names
            for site_row in cohort_reports[name]["sites"].values()
        ]
        stage = {
            "training_rows": row_count,
            "requested_rank": rank,
            "effective_ranks": {
                site: int(fitted[site].receipt["effective_rank"])
                for site in sites
            },
            "factor_receipt_sha256s": {
                site: str(fitted[site].receipt["receipt_sha256"])
                for site in sites
            },
            "cohorts": cohort_reports,
            "summary": {
                "mean_relative_error": float(np.mean(relative_errors)),
                "median_relative_error": float(np.median(relative_errors)),
                "worst_relative_error": max(relative_errors),
                "mean_cosine": float(np.mean(cosines)),
                "minimum_cosine": min(cosines),
                "all_cohorts_better_than_zero": all(
                    bool(row["better_than_zero_operator"])
                    for row in aggregates
                ),
                "all_cohorts_direction_positive": all(value > 0.0 for value in cosines),
                "site_pass_fraction": sum(
                    bool(row["better_than_zero_operator"]) and float(row["cosine"]) > 0.0
                    for row in site_rows
                )
                / len(site_rows),
                "all_sites_better_than_zero": all(
                    bool(row["better_than_zero_operator"]) for row in site_rows
                ),
                "all_sites_direction_positive": all(
                    float(row["cosine"]) > 0.0 for row in site_rows
                ),
            },
        }
        stages.append(stage)
        final_inventory = fitted

    if final_inventory is None:  # pragma: no cover - guarded by sample_rows
        raise RuntimeError("trajectory sample-complexity fit produced no stage")
    first = stages[0]["summary"]
    final = stages[-1]["summary"]
    log_rows = np.log2(np.asarray(sample_rows, dtype=np.float64))
    error_means = np.asarray(
        [stage["summary"]["mean_relative_error"] for stage in stages],
        dtype=np.float64,
    )
    cosine_means = np.asarray(
        [stage["summary"]["mean_cosine"] for stage in stages],
        dtype=np.float64,
    )
    gates = {
        "fresh_cohorts_all_better_than_zero": bool(
            final["all_cohorts_better_than_zero"]
        ),
        "fresh_cohorts_all_direction_positive": bool(
            final["all_cohorts_direction_positive"]
        ),
        "final_worst_case_beats_zero": float(final["worst_relative_error"]) < 1.0,
        "fresh_site_cells_all_better_than_zero": bool(
            final["all_sites_better_than_zero"]
        ),
        "fresh_site_cells_all_direction_positive": bool(
            final["all_sites_direction_positive"]
        ),
        "mean_error_improves_with_more_evidence": (
            float(final["mean_relative_error"])
            < float(first["mean_relative_error"])
            and float(np.polyfit(log_rows, error_means, 1)[0]) < 0.0
        ),
        "mean_direction_improves_with_more_evidence": (
            float(final["mean_cosine"]) > float(first["mean_cosine"])
            and float(np.polyfit(log_rows, cosine_means, 1)[0]) > 0.0
        ),
    }
    body = {
        "schema": TRAJECTORY_SAMPLE_COMPLEXITY_SCHEMA,
        "training_rows_total": total_rows,
        "sample_rows": list(sample_rows),
        "validation_cohort_rows": cohort_row_counts,
        "site_count": len(sites),
        "sites": list(sites),
        "configuration": {
            "rank": rank,
            "regularization": float(regularization),
            "gain": float(gain),
            "adapter_scale": float(adapter_scale),
            "corrections_normalized": bool(normalize_corrections),
            "site_phases": dict(sorted((site_phases or {}).items())),
            "training_subset_policy": "ordered_nested_complete_example_prefixes",
            "hyperparameters_fixed_across_stages": True,
        },
        "stages": stages,
        "trend": {
            "mean_relative_error_per_log2_row_slope": float(
                np.polyfit(log_rows, error_means, 1)[0]
            ),
            "mean_cosine_per_log2_row_slope": float(
                np.polyfit(log_rows, cosine_means, 1)[0]
            ),
        },
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_boundary": (
            "fresh_seed_internal_operator_scaling_only_not_behavioral_or_reasoning_gain"
        ),
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    return report, final_inventory


def install_verified_trajectory_inventory(
    model: Any,
    inventory: Mapping[str, DistilledTrajectoryFactors],
    *,
    expected_sites: Sequence[str],
) -> dict[str, Any]:
    """Atomically install fitted factors into their exact causal phase."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    expected = tuple(sorted(expected_sites))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected trajectory site inventory is invalid")
    if tuple(sorted(inventory)) != expected:
        raise ValueError("fitted trajectory site inventory differs from attachment")

    resolved: list[
        tuple[
            str,
            ScopedLoRALinear | ScopedCodaLoRALinear,
            DistilledTrajectoryFactors,
        ]
    ] = []
    for site in expected:
        parts = site.split(".")
        if len(parts) != 5 or parts[:2] != ["model", "layers"]:
            raise ValueError(f"trajectory site path is invalid: {site}")
        try:
            layer_index = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"trajectory site layer is invalid: {site}") from exc
        parent = getattr(model.model.layers[layer_index], parts[3], None)
        projection = getattr(parent, parts[4], None)
        factors = inventory[site]
        if not isinstance(factors, DistilledTrajectoryFactors):
            raise ValueError(f"trajectory factor inventory is invalid at {site}")
        factor_receipt = _validate_factor_receipt(factors)
        expected_type = (
            ScopedLoRALinear
            if factors.target_phase == "recurrence"
            else ScopedCodaLoRALinear
        )
        if not isinstance(projection, expected_type):
            raise ValueError(
                f"trajectory site phase differs from attachment: "
                f"{site} expected={factors.target_phase}"
            )
        if (
            tuple(factors.lora_a.shape) != tuple(projection.lora_a.shape)
            or tuple(factors.lora_b.shape) != tuple(projection.lora_b.shape)
        ):
            raise ValueError(f"trajectory factor shape differs at {site}")
        adapter_scale = factor_receipt.get("adapter_scale")
        projection_scale = getattr(projection, "scale", None)
        if (
            isinstance(adapter_scale, bool)
            or not isinstance(adapter_scale, (int, float))
            or not math.isfinite(float(adapter_scale))
            or isinstance(projection_scale, bool)
            or not isinstance(projection_scale, (int, float))
            or float(projection_scale) != float(adapter_scale)
        ):
            raise ValueError(f"trajectory adapter scale differs at {site}")
        resolved.append((site, projection, factors))

    snapshots = [
        (
            projection,
            projection.lora_a,
            projection.lora_b,
            bool(getattr(projection, "exact_episodic_operation", False)),
        )
        for _, projection, _ in resolved
    ]
    try:
        for _site, projection, factors in resolved:
            projection.lora_a = mx.array(factors.lora_a).astype(projection.lora_a.dtype)
            projection.lora_b = mx.array(factors.lora_b).astype(projection.lora_b.dtype)
            projection.exact_episodic_operation = (
                factors.receipt.get("schema") == EPISODIC_TRANSPLANT_SCHEMA
            )
        mx.eval(
            *(
                tensor
                for _, projection, _ in resolved
                for tensor in (projection.lora_a, projection.lora_b)
            )
        )
    except BaseException:
        for projection, lora_a, lora_b, exact_episodic_operation in snapshots:
            projection.lora_a = lora_a
            projection.lora_b = lora_b
            projection.exact_episodic_operation = exact_episodic_operation
        raise

    body = {
        "schema": "aura.verified_trajectory_installation.v1",
        "sites": list(expected),
        "site_phases": {
            site: inventory[site].target_phase for site in expected
        },
        "operation_modes": {
            site: (
                "episodic_exact"
                if inventory[site].receipt.get("schema")
                == EPISODIC_TRANSPLANT_SCHEMA
                else "scoped_lora"
            )
            for site in expected
        },
        "factor_receipt_sha256s": {
            site: str(inventory[site].receipt["receipt_sha256"]) for site in expected
        },
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


__all__ = [
    "DISTILLATION_SCHEMA",
    "EPISODIC_TRANSPLANT_SCHEMA",
    "TRAJECTORY_ARTIFACT_SCHEMA",
    "TRAJECTORY_SAMPLE_COMPLEXITY_SCHEMA",
    "DistilledTrajectoryFactors",
    "build_verified_trajectory_artifact",
    "compile_episodic_delta_factors",
    "compile_episodic_delta_inventory",
    "evaluate_verified_trajectory_transfer",
    "fit_verified_trajectory_factors",
    "fit_verified_trajectory_inventory",
    "fit_verified_trajectory_sample_complexity",
    "install_verified_trajectory_inventory",
    "load_verified_trajectory_artifact",
    "publish_verified_trajectory_artifact",
]
