"""CAA readiness verification for the exact active cortex basis.

Whether steering vectors were genuinely extracted from contrastive activation
differences in the exact resident cortex, or merely derived at runtime, determines
the readiness level and therefore how much the steering alpha is damped. Same-width
or same-config artifacts are different neural bases and are never interchangeable.

This module verifies it from ground truth: it reads each steering-vector file's
provenance (``source`` / ``extracted`` / ``derived_at``) directly off disk, ties it to
the active fused model (``training/fused-model/active.json``), classifies the readiness
level, estimates the resulting steering capacity, and warns loudly when steering is
running below design capacity. "Are the Zenith vectors registered?" becomes a queryable,
surfaced fact instead of a guess.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CAA.Readiness")

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Estimated steering capacity (alpha fraction of design) per readiness level.
_CAPACITY = {"bootstrap": 0.3, "mixed": 0.6, "validated": 0.85, "production": 1.0}


def _parse_vector_stem(stem: str) -> tuple[str, int]:
    import re

    match = re.match(r"^(?P<dimension>.+)_layer(?P<layer>\d+)$", stem)
    if match:
        return match.group("dimension"), int(match.group("layer"))
    return stem, -1


def _runtime_expected_keys() -> list[str]:
    try:
        from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

        return [str(spec["key"]) for spec in AFFECTIVE_DIMENSIONS if spec.get("key")]
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        record_degradation("caa_readiness_report", exc)
        return ["valence_positive", "arousal", "curiosity", "frustration", "energy"]


def _target_layers_for_active_model(active: dict[str, Any]) -> list[int]:
    if not active.get("identity_valid"):
        return []
    try:
        profile = active.get("artifact_profile")
        if not isinstance(profile, dict):
            raise ValueError("active_model_profile_missing")
        n_layers = int(profile.get("num_hidden_layers") or 0)
    except (ValueError, TypeError) as exc:
        record_degradation("caa_readiness_report", exc)
        return []
    if n_layers <= 0:
        return []
    lo = int(n_layers * 0.40)
    hi = int(n_layers * 0.65)
    span = hi - lo
    if span <= 2:
        return [lo]
    if span <= 5:
        return [lo, lo + span // 2]
    return [lo, lo + span // 3, lo + 2 * span // 3]


#: Parsed provenance per vector file, keyed by path and invalidated by
#: (mtime, size). The integrity block calls scan_vector_files() on every
#: collection and this used to np.load() every ``.npz`` each time — a
#: pure-Python decompression burst holding the GIL underneath a runtime whose
#: event loop is the scarce resource (it sat in the 2026-07-29 stall dumps).
#: A vector file's provenance is fixed at write time, so a file whose stat is
#: unchanged cannot have new provenance to read.
_VECTOR_PROVENANCE_CACHE: dict[str, tuple[tuple[float, int], dict[str, Any]]] = {}


def _vector_provenance(npz: Path, dimension: Any, layer: Any) -> dict[str, Any] | None:
    """Provenance for one vector file, re-read only when the file changes."""
    key = str(npz)
    try:
        stat = npz.stat()
        stamp = (stat.st_mtime, stat.st_size)
    except OSError as exc:
        record_degradation("caa_readiness_report", exc)
        return None
    cached = _VECTOR_PROVENANCE_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        d = np.load(npz, allow_pickle=True)
        source = str(d["source"]) if "source" in d else "unknown"
        is_extracted = bool(d["extracted"]) if "extracted" in d else False
        derived_at = float(d["derived_at"]) if "derived_at" in d else 0.0
        vector_dim = (
            int(np.asarray(d["v"] if "v" in d else d[d.files[0]]).reshape(-1).shape[0])
            if d.files
            else 0
        )
        recorded_model_path = (
            str(d["model_path"]) if "model_path" in d else str(d["model"]) if "model" in d else None
        )
        model_config_sha256 = str(d["model_config_sha256"]) if "model_config_sha256" in d else None
        model_descriptor_sha256 = (
            str(d["model_descriptor_sha256"]) if "model_descriptor_sha256" in d else None
        )
    except (OSError, ValueError, KeyError) as exc:
        record_degradation("caa_readiness_report", exc)
        return None
    entry = {
        "path": key,
        "dimension": dimension,
        "layer": layer,
        "source": source,
        "extracted": is_extracted,
        "derived_at": derived_at,
        "vector_dim": vector_dim,
        "model_path": recorded_model_path,
        "model_config_sha256": model_config_sha256,
        "model_descriptor_sha256": model_descriptor_sha256,
    }
    _VECTOR_PROVENANCE_CACHE[key] = (stamp, entry)
    return entry


def scan_vector_files(vectors_dir: Path) -> dict[str, Any]:
    """Read provenance off every steering-vector ``.npz`` on disk."""
    extracted = 0
    runtime_derived = 0
    fallback = 0
    other = 0
    newest_derived_at = 0.0
    files = 0
    by_source: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    try:
        seen: set[str] = set()
        for npz in sorted(vectors_dir.glob("*.npz")):
            files += 1
            dimension, layer = _parse_vector_stem(npz.stem)
            entry = _vector_provenance(npz, dimension, layer)
            if entry is None:
                continue
            seen.add(entry["path"])
            details.append(dict(entry))
            source = entry["source"]
            derived_at = entry["derived_at"]
            by_source[source] = by_source.get(source, 0) + 1
            newest_derived_at = max(newest_derived_at, derived_at)
            if entry["extracted"]:
                extracted += 1
            elif source == "runtime_derived_caa":
                runtime_derived += 1
            elif source == "fallback_random":
                fallback += 1
            else:
                other += 1
        for stale_key in _VECTOR_PROVENANCE_CACHE.keys() - seen:
            _VECTOR_PROVENANCE_CACHE.pop(stale_key, None)
    except OSError as exc:
        record_degradation("caa_readiness_report", exc)
    return {
        "files": files,
        "extracted": extracted,
        "runtime_derived": runtime_derived,
        "fallback": fallback,
        "other": other,
        "by_source": by_source,
        "newest_derived_at": newest_derived_at,
        "details": details,
    }


def _active_model(fused_model_dir: Path) -> dict[str, Any]:
    unavailable = {
        "path": None,
        "fused_at": 0.0,
        "descriptor_sha256": None,
        "artifact_profile": None,
        "identity_valid": False,
        "identity_error": "active_model_pointer_unavailable",
        "steering_authority_status": "unmanaged",
        "steering_authority_kind": "",
    }
    try:
        aj = fused_model_dir / "active.json"
        if aj.exists():
            data = json.loads(aj.read_text(encoding="utf-8"))
            model_path = str(data.get("active_model_path") or "")
            descriptor = data.get("artifact_descriptor")
            migration = data.get("migration_contract")
            components = migration.get("components") if isinstance(migration, dict) else None
            steering = components.get("steering") if isinstance(components, dict) else None
            steering_status = (
                str(steering.get("status") or "") if isinstance(steering, dict) else "unmanaged"
            )
            steering_kind = (
                str(steering.get("authority_kind") or "") if isinstance(steering, dict) else ""
            )
            if not model_path or not isinstance(descriptor, dict):
                return {
                    **unavailable,
                    "path": model_path or None,
                    "fused_at": float(data.get("fused_at", 0.0) or 0.0),
                    "identity_error": "active_model_descriptor_missing",
                }
            from core.brain.llm.model_artifact_profile import (
                validate_model_artifact_descriptor,
            )

            validated = validate_model_artifact_descriptor(
                descriptor,
                model_path=model_path,
                verify_full_hash=False,
            )
            return {
                "path": model_path,
                "fused_at": float(data.get("fused_at", 0.0) or 0.0),
                "descriptor_sha256": validated["descriptor_sha256"],
                "artifact_profile": validated["artifact_profile"],
                "identity_valid": True,
                "identity_error": "",
                "steering_authority_status": steering_status,
                "steering_authority_kind": steering_kind,
            }
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        record_degradation("caa_readiness_report", exc)
        return {**unavailable, "identity_error": f"{type(exc).__name__}: {exc}"}
    return unavailable


def _matches_active_model(item: dict[str, Any], active: dict[str, Any]) -> bool:
    """Return true only for an exact descriptor match to the active cortex."""

    active_digest = str(active.get("descriptor_sha256") or "")
    item_digest = str(item.get("model_descriptor_sha256") or "")
    return bool(active.get("identity_valid") and active_digest and item_digest == active_digest)


def verify_readiness(
    *,
    vectors_dir: Path | None = None,
    fused_model_dir: Path | None = None,
) -> dict[str, Any]:
    """Classify CAA readiness from on-disk provenance + the active model."""
    vectors_dir = vectors_dir or (_REPO_ROOT / "training" / "vectors")
    fused_model_dir = fused_model_dir or (_REPO_ROOT / "training" / "fused-model")
    scan = scan_vector_files(vectors_dir)
    active = _active_model(fused_model_dir)
    total = scan["files"] or 0
    extracted_ratio = (scan["extracted"] / total) if total else 0.0
    expected_keys = _runtime_expected_keys()
    expected_layers = _target_layers_for_active_model(active)
    expected_total = len(expected_keys) * len(expected_layers)
    details = list(scan.get("details") or [])
    expected_files = [
        item
        for item in details
        if item.get("dimension") in expected_keys and item.get("layer") in expected_layers
    ]
    expected_extracted = [
        item
        for item in expected_files
        if item.get("extracted") and str(item.get("source", "")).startswith("extracted")
    ]
    expected_extracted_active = [
        item for item in expected_extracted if _matches_active_model(item, active)
    ]
    stale_expected = [
        item for item in expected_extracted if not _matches_active_model(item, active)
    ]
    expected_ratio = (len(expected_extracted_active) / expected_total) if expected_total else 0.0
    missing_expected = []
    present_pairs = {(item.get("dimension"), item.get("layer")) for item in expected_files}
    for key in expected_keys:
        for layer in expected_layers:
            if (key, layer) not in present_pairs:
                missing_expected.append({"dimension": key, "layer": layer})
    ignored_files = [
        item
        for item in details
        if item.get("dimension") not in expected_keys or item.get("layer") not in expected_layers
    ]

    steering_authority_status = str(active.get("steering_authority_status") or "unmanaged")
    if steering_authority_status in {"deferred", "retired"}:
        level = steering_authority_status
        detail = (
            f"steering tissue is intentionally {steering_authority_status} for the exact "
            "active cortex; no residual hooks are authorized"
        )
        readiness_ratio = 0.0
    elif total == 0:
        level, detail = "bootstrap", "no steering vectors present"
        readiness_ratio = 0.0
    elif not active.get("identity_valid"):
        level = "bootstrap"
        detail = (
            "exact active-model identity unavailable; no steering vector can be "
            "credited to the resident neural basis"
        )
        readiness_ratio = 0.0
    elif expected_total and len(expected_extracted_active) == expected_total:
        level, detail = (
            "production",
            "all runtime target vectors extracted from and bound to the active model",
        )
        readiness_ratio = 1.0
    elif expected_total and expected_files:
        readiness_ratio = expected_ratio
        if expected_ratio < 0.5:
            level = "bootstrap"
            detail = (
                f"{len(expected_extracted_active)}/{expected_total} runtime target vectors extracted "
                "and bound to the active model; runtime steering will still rely on derived/nearest vectors"
            )
        else:
            level = "mixed"
            detail = (
                f"{len(expected_extracted_active)}/{expected_total} runtime target vectors extracted "
                "and bound to the active model; missing exact production coverage"
            )
    else:
        level = "bootstrap"
        detail = "no exact runtime-target vectors are bound to the active model"
        readiness_ratio = 0.0

    capacity = 0.0 if level in {"deferred", "retired"} else _CAPACITY.get(level, 0.3)
    below_design_capacity = capacity < 1.0 and level not in {"deferred", "retired"}
    return {
        "level": level,
        "detail": detail,
        "extracted_ratio": round(readiness_ratio, 3),
        "all_files_extracted_ratio": round(extracted_ratio, 3),
        "steering_capacity_pct": round(capacity * 100, 1),
        "below_design_capacity": below_design_capacity,
        "serving_authorized": steering_authority_status == "qualified",
        "steering_authority_status": steering_authority_status,
        "steering_authority_kind": str(active.get("steering_authority_kind") or ""),
        "active_model": active["path"],
        "active_model_identity_valid": bool(active.get("identity_valid")),
        "active_model_identity_error": str(active.get("identity_error") or ""),
        "runtime_contract": {
            "expected_keys": expected_keys,
            "expected_layers": expected_layers,
            "expected_total": expected_total,
            "expected_extracted": len(expected_extracted_active),
            "expected_extracted_unbound": len(stale_expected),
            "missing_expected": missing_expected,
            "ignored_file_count": len(ignored_files),
            "active_model_descriptor_sha256": active.get("descriptor_sha256"),
        },
        "vectors": scan,
    }


def audit(**kwargs: Any) -> dict[str, Any]:
    """Verify readiness and warn loudly when steering runs below design capacity."""
    report = verify_readiness(**kwargs)
    if report["below_design_capacity"]:
        logger.warning(
            "🎚️ [CAA] steering BELOW design capacity (%.0f%%): readiness=%s — %s. "
            "Extract CAA vectors from the fused model to reach production steering.",
            report["steering_capacity_pct"],
            report["level"],
            report["detail"],
        )
        try:
            from core.observability.metrics import get_metrics

            get_metrics().increment_counter("caa_below_design_capacity_total")
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass
    return report


def governance_signal() -> dict[str, Any]:
    return verify_readiness()
