"""Exact model identity for LoRA training, resume, and fusion."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_MODEL_BASIS_SCHEMA = "aura.training_model_basis.v1"


class TrainingModelBasisError(RuntimeError):
    """The requested training base cannot be identified exactly."""


@dataclass(frozen=True, slots=True)
class TrainingModelBasis:
    path: Path
    descriptor: dict[str, object]
    descriptor_sha256: str
    source: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema": TRAINING_MODEL_BASIS_SCHEMA,
            "model_path": str(self.path),
            "descriptor_sha256": self.descriptor_sha256,
            "artifact_descriptor": self.descriptor,
        }


def _local_model_path(value: str | os.PathLike[str]) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise TrainingModelBasisError("training_model_path_missing")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        raise TrainingModelBasisError("training_model_must_be_local")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise TrainingModelBasisError("training_model_path_unavailable") from exc
    if not resolved.is_dir():
        raise TrainingModelBasisError("training_model_not_directory")
    return resolved


def _read_descriptor(path: str | os.PathLike[str]) -> dict[str, object]:
    descriptor_path = Path(path).expanduser()
    if descriptor_path.is_symlink():
        raise TrainingModelBasisError("training_model_descriptor_invalid")
    try:
        descriptor_path = descriptor_path.resolve(strict=True)
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingModelBasisError("training_model_descriptor_unreadable") from exc
    if not descriptor_path.is_file() or not isinstance(payload, dict):
        raise TrainingModelBasisError("training_model_descriptor_invalid")
    return payload


def _validated_basis(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    source: str,
    verify_full_hash: bool,
) -> TrainingModelBasis:
    from core.brain.llm.model_artifact_profile import validate_model_artifact_descriptor

    try:
        validated = validate_model_artifact_descriptor(
            dict(descriptor),
            model_path=path,
            verify_full_hash=verify_full_hash,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingModelBasisError("training_model_descriptor_mismatch") from exc
    digest = str(validated.get("descriptor_sha256") or "")
    if len(digest) != 64:
        raise TrainingModelBasisError("training_model_descriptor_invalid")
    return TrainingModelBasis(
        path=path,
        descriptor=dict(validated),
        descriptor_sha256=digest,
        source=source,
    )


def resolve_training_model_basis(
    model_path: str | os.PathLike[str] | None = None,
    *,
    descriptor_path: str | os.PathLike[str] | None = None,
    verify_full_hash: bool = True,
) -> TrainingModelBasis:
    """Resolve one local model and prove the exact bytes training will use."""

    explicit = str(model_path or os.environ.get("AURA_LORA_BASE_MODEL", "")).strip()
    if explicit:
        locator = explicit
        source = "explicit"
    else:
        from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

        locator = get_runtime_model_path(ACTIVE_MODEL)
        source = "active_cortex"
    path = _local_model_path(locator)

    configured_descriptor = str(
        descriptor_path or os.environ.get("AURA_LORA_BASE_MODEL_DESCRIPTOR", "")
    ).strip()
    descriptor: dict[str, object] | None = None
    if configured_descriptor:
        descriptor = _read_descriptor(configured_descriptor)
        source += ":descriptor_file"
    else:
        from core.brain.llm.model_registry import get_active_model_artifact_descriptor

        descriptor = get_active_model_artifact_descriptor(path)
        if descriptor is not None:
            source += ":active_descriptor"

    if descriptor is None:
        from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor

        try:
            descriptor = build_model_artifact_descriptor(path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise TrainingModelBasisError("training_model_descriptor_build_failed") from exc
        source += ":measured_descriptor"
        verify_full_hash = False

    return _validated_basis(
        path,
        descriptor,
        source=source,
        verify_full_hash=verify_full_hash,
    )


def load_recorded_training_model_basis(
    config_path: str | os.PathLike[str],
    *,
    model_override: str | os.PathLike[str] | None = None,
    verify_full_hash: bool = True,
) -> TrainingModelBasis:
    """Load the model identity committed before a training run began."""

    path = Path(config_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingModelBasisError("training_config_unreadable") from exc
    if not isinstance(payload, dict):
        raise TrainingModelBasisError("training_config_invalid")
    record = payload.get("training_basis")
    if not isinstance(record, dict) or set(record) != {
        "schema",
        "model_path",
        "descriptor_sha256",
        "artifact_descriptor",
    }:
        raise TrainingModelBasisError("training_config_model_basis_missing")
    if record.get("schema") != TRAINING_MODEL_BASIS_SCHEMA:
        raise TrainingModelBasisError("training_config_model_basis_invalid")

    recorded_path = _local_model_path(str(record.get("model_path") or ""))
    configured_model = str(payload.get("model") or "").strip()
    if configured_model and _local_model_path(configured_model) != recorded_path:
        raise TrainingModelBasisError("training_config_model_path_mismatch")
    if model_override and _local_model_path(model_override) != recorded_path:
        raise TrainingModelBasisError("training_resume_model_basis_change")
    descriptor = record.get("artifact_descriptor")
    if not isinstance(descriptor, dict):
        raise TrainingModelBasisError("training_config_model_basis_invalid")
    observed = _validated_basis(
        recorded_path,
        descriptor,
        source="training_config",
        verify_full_hash=verify_full_hash,
    )
    if observed.descriptor_sha256 != str(record.get("descriptor_sha256") or ""):
        raise TrainingModelBasisError("training_config_model_descriptor_mismatch")
    return observed


def assert_adapter_matches_basis(
    adapter_dir: str | os.PathLike[str],
    expected: TrainingModelBasis,
) -> None:
    observed = load_recorded_training_model_basis(
        Path(adapter_dir) / "training_config.json",
        model_override=expected.path,
        verify_full_hash=False,
    )
    if observed.descriptor_sha256 != expected.descriptor_sha256:
        raise TrainingModelBasisError("adapter_model_basis_mismatch")


__all__ = [
    "TRAINING_MODEL_BASIS_SCHEMA",
    "TrainingModelBasis",
    "TrainingModelBasisError",
    "assert_adapter_matches_basis",
    "load_recorded_training_model_basis",
    "resolve_training_model_basis",
]
