"""Checking a model artifact and an adapter artifact before the worker is asked to load them: the files present, the manifest coherent, the identity the registry expects.

Lifted whole out of `mlx_client`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.brain.llm.how_big_is_the_checkpoint import _weight_files

if TYPE_CHECKING:
    from .mlx_client import (
        AdapterVerdict,
        ArtifactVerdict,
    )


def _validate_model_artifact(resolved: Path, incumbent: str = "") -> ArtifactVerdict:
    """Prove a directory is a servable model BEFORE the live worker is recycled.

    CP126 a996d77f: this was ``is_dir()``. An empty, partial, half-copied, or
    wrong-architecture directory passed, the healthy worker was torn down, and
    the failure surfaced at the next load — by which time the lane that was
    serving fine had been destroyed to make room for something that could not
    load at all.

    ``incumbent`` is the currently-served path. When both sides declare their
    architectures, a mismatch is refused: promoting a Llama checkpoint onto a
    lane whose callers, adapters and admission classes were built for Qwen is
    a different model wearing the lane's name.
    """
    from .mlx_client import (
        _REQUIRED_ARTIFACT_CONFIG,
        _TOKENIZER_CANDIDATES,
        ArtifactVerdict,
        logger,
    )

    if not resolved.is_dir():
        return ArtifactVerdict(False, f"artifact_missing:{resolved}")

    config_path = resolved / _REQUIRED_ARTIFACT_CONFIG
    if not config_path.is_file():
        return ArtifactVerdict(False, f"artifact_missing_config:{_REQUIRED_ARTIFACT_CONFIG}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ArtifactVerdict(False, f"artifact_config_unreadable:{type(exc).__name__}")
    if not isinstance(config, dict):
        return ArtifactVerdict(False, "artifact_config_not_an_object")

    if not any((resolved / name).is_file() for name in _TOKENIZER_CANDIDATES):
        return ArtifactVerdict(False, "artifact_missing_tokenizer")

    weights = list(_weight_files(resolved))
    if not weights:
        return ArtifactVerdict(False, "artifact_missing_weights")

    architectures = tuple(
        str(entry)
        for entry in (config.get("architectures") or [])
        if isinstance(entry, str)
    )
    profile = None
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(str(resolved))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Model artifact profile unavailable: %s", exc)
        profile = None

    verdict = ArtifactVerdict(
        True,
        architectures=architectures,
        size_class=getattr(profile, "size_class", "unknown") or "unknown",
        fingerprint=getattr(profile, "fingerprint", "") or "",
        weight_files=len(weights),
    )

    incumbent_path = Path(str(incumbent or "")).expanduser()
    incumbent_config = incumbent_path / _REQUIRED_ARTIFACT_CONFIG
    if architectures and incumbent_config.is_file():
        try:
            current = json.loads(incumbent_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            current = {}
        current_arch = tuple(
            str(entry)
            for entry in (current.get("architectures") or [])
            if isinstance(entry, str)
        )
        if current_arch and not set(current_arch) & set(architectures):
            return ArtifactVerdict(
                False,
                f"artifact_architecture_mismatch:{'/'.join(current_arch)}"
                f"->{'/'.join(architectures)}",
                architectures=architectures,
                size_class=verdict.size_class,
                fingerprint=verdict.fingerprint,
                weight_files=verdict.weight_files,
            )
    return verdict


def _validate_adapter_artifact(
    path: Path, *, expected_base_fingerprint: str = ""
) -> AdapterVerdict:
    """Prove a directory is an attachable adapter before live weights change.

    CP126 d665aa64: admission was ``is_dir()``. Any directory — a
    half-finished training output, an empty scratch folder, a symlink to
    somewhere else entirely — was handed to the resident worker as an adapter,
    and the failure surfaced inside the process holding twenty gigabytes of
    live weights.

    When the adapter names the base checkpoint it was trained against AND the
    caller supplies the resident one, a mismatch is refused here: LoRA deltas
    are only meaningful against the weights they were fitted to, and attaching
    them to different ones does not fail loudly — it quietly degrades every
    answer the model gives afterwards.

    When the caller cannot supply the resident fingerprint, the verdict says
    ``declared_unverified`` rather than passing quietly. An unchecked
    compatibility claim recorded as a checked one is the failure this whole
    remediation keeps finding, and writing it into a new validator to make the
    validator look thorough would be the same mistake with a fresh coat.
    """
    from .mlx_client import (
        _ADAPTER_CONFIG_NAME,
        _ADAPTER_MAX_BYTES,
        _ADAPTER_WEIGHT_NAMES,
        AdapterVerdict,
    )

    if not path.is_dir():
        return AdapterVerdict(False, f"adapter_missing:{path}")

    weight_path: Path | None = None
    for name in _ADAPTER_WEIGHT_NAMES:
        candidate = path / name
        if candidate.is_file():
            weight_path = candidate
            break
    if weight_path is None:
        return AdapterVerdict(False, "adapter_missing_weights")
    try:
        weight_bytes = int(weight_path.stat().st_size)
    except OSError as exc:
        return AdapterVerdict(False, f"adapter_weights_unreadable:{type(exc).__name__}")
    if weight_bytes <= 0:
        return AdapterVerdict(False, "adapter_weights_empty")
    if weight_bytes > _ADAPTER_MAX_BYTES:
        return AdapterVerdict(False, f"adapter_weights_oversized:{weight_bytes}")

    config: dict[str, Any] = {}
    config_path = path / _ADAPTER_CONFIG_NAME
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return AdapterVerdict(False, f"adapter_config_unreadable:{type(exc).__name__}")
        if not isinstance(loaded, dict):
            return AdapterVerdict(False, "adapter_config_not_an_object")
        config = loaded

    base_fingerprint = str(config.get("base_checkpoint_fingerprint") or "")
    expected = str(expected_base_fingerprint or "")
    if not base_fingerprint:
        compatibility = "not_declared"
    elif not expected:
        compatibility = "declared_unverified"
    elif base_fingerprint != expected:
        return AdapterVerdict(
            False,
            f"adapter_base_mismatch:{base_fingerprint[:12]}!={expected[:12]}",
            weight_file=weight_path.name,
            weight_bytes=weight_bytes,
            base_checkpoint_fingerprint=base_fingerprint,
            fine_tune_type=str(config.get("fine_tune_type") or ""),
            base_compatibility="mismatch",
        )
    else:
        compatibility = "verified"
    return AdapterVerdict(
        True,
        weight_file=weight_path.name,
        weight_bytes=weight_bytes,
        base_checkpoint_fingerprint=base_fingerprint,
        fine_tune_type=str(config.get("fine_tune_type") or ""),
        base_compatibility=compatibility,
    )
