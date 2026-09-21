"""Whether the deep specialist may be admitted, and what it is signed over.

Lifted whole out of `model_registry`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def deep_solver_is_distinctly_configured() -> bool:
    """Whether a second local model was deliberately assigned to deep work."""
    from .model_registry import (
        ACTIVE_MODEL,
        _canonical_model_locator,
        _configured_deep_model_name,
        _configured_deep_model_path,
        get_runtime_model_path,
    )


    configured_name = _configured_deep_model_name()
    configured_path = _configured_deep_model_path()
    if not configured_name and not configured_path:
        return False

    active_path = _canonical_model_locator(get_runtime_model_path(ACTIVE_MODEL))
    candidate_path = _canonical_model_locator(
        configured_path
        or get_runtime_model_path(configured_name)
    )
    if candidate_path and active_path and candidate_path == active_path:
        return False
    if configured_name and configured_name.lower() == ACTIVE_MODEL.lower() and not configured_path:
        return False
    return bool(candidate_path or configured_name)


def deep_solver_artifact_is_ready() -> bool:
    """Whether the configured specialist is a measured local model artifact."""
    from .model_registry import (
        deep_solver_is_distinctly_configured,
        get_deep_model_path,
    )


    if not deep_solver_is_distinctly_configured():
        return False
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(get_deep_model_path())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return bool(
        profile.exists
        and profile.measured
        and profile.weight_bytes > 0
        and profile.total_parameters > 0
    )


def get_deep_specialist_certificate_path() -> Path:
    from .model_registry import (
        _FLAG_DEEP_SPECIALIST_CERTIFICATE,
        get_fused_model_root,
    )

    configured = str(_FLAG_DEEP_SPECIALIST_CERTIFICATE.value() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return get_fused_model_root() / "specialists" / "deep" / "admission.json"


def get_deep_specialist_trust_root_path() -> Path:
    from .model_registry import (
        _FLAG_DEEP_SPECIALIST_TRUST_ROOT,
        get_fused_model_root,
    )

    configured = str(_FLAG_DEEP_SPECIALIST_TRUST_ROOT.value() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return get_fused_model_root() / "specialists" / "deep" / "admission.pub.pem"


def _path_stat_signature(path: Path) -> tuple[object, ...]:
    try:
        stat = path.stat()
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(path), 0, 0, 0, 0)


def _deep_specialist_identity_signature(
    *,
    certificate_path: Path,
    trust_root_path: Path,
    specialist_path: Path,
) -> tuple[object, ...]:
    """Cheap invalidation key; a cache miss performs the full model hash."""
    from .model_registry import (
        _path_stat_signature,
        model_paths,
    )


    signature: list[object] = [
        _path_stat_signature(certificate_path),
        _path_stat_signature(trust_root_path),
    ]
    try:
        children = sorted(
            child
            for child in specialist_path.iterdir()
            if child.is_file() or child.is_symlink()
        )
    except OSError:
        children = []
    signature.extend(_path_stat_signature(child) for child in children)
    try:
        from core.learning.specialist_cortex_admission import REQUIRED_SOURCE_CLOSURE

        signature.extend(
            _path_stat_signature(model_paths.BASE_DIR / relative)
            for relative in sorted(REQUIRED_SOURCE_CLOSURE)
        )
    except ImportError:
        signature.append(("specialist_admission_import", 0))
    return tuple(signature)


def _current_specialist_source_commit() -> str:
    for name in ("AURA_RUNTIME_SOURCE_COMMIT", "AURA_LAUNCH_EXPECTED_COMMIT"):
        expected = str(os.getenv(name, "") or "").strip().lower()
        if expected:
            return expected
    from core.runtime.release_certificate import current_commit

    return str(current_commit() or "").strip().lower()


def get_deep_solver_admission_status(
    requested_domain: str | None=None,
    *,
    force_refresh: bool=False,
) -> Any:
    """Return evidence qualification for the optional specialist.

    This is the one model-free admission authority used by startup, routing,
    health, and the API.  Host availability remains a live check in the
    inference gate; this status supplies the measured minima it must enforce.
    """
    from core.learning.specialist_cortex_admission import (
        SpecialistAdmissionError,
        denied_status,
        verify_specialist_qualification_certificate,
    )

    from .model_registry import (
        _DEEP_SPECIALIST_STATUS_TTL_S,
        _current_specialist_source_commit,
        _deep_specialist_identity_signature,
        _deep_specialist_status_cache,
        deep_solver_artifact_is_ready,
        deep_solver_is_distinctly_configured,
        get_active_cortex_spec,
        get_deep_model_path,
        get_deep_specialist_certificate_path,
        get_deep_specialist_trust_root_path,
        logger,
        model_paths,
    )

    if not deep_solver_is_distinctly_configured():
        return denied_status("specialist_not_configured")
    if not deep_solver_artifact_is_ready():
        return denied_status("specialist_artifact_unmeasured")
    resident = get_active_cortex_spec(force_refresh=force_refresh)
    if (
        resident is None
        or not resident.exact_identity
        or not resident.promotion_qualified
        or not resident.descriptor_sha256
        or not resident.pointer_sha256
    ):
        return denied_status("resident_cortex_unqualified")

    certificate_path = get_deep_specialist_certificate_path()
    trust_root_path = get_deep_specialist_trust_root_path()
    specialist_path = Path(get_deep_model_path()).expanduser().resolve(strict=False)
    cache_key = str(requested_domain or "").strip().lower()
    signature = _deep_specialist_identity_signature(
        certificate_path=certificate_path,
        trust_root_path=trust_root_path,
        specialist_path=specialist_path,
    )
    now = time.monotonic()
    cached = _deep_specialist_status_cache.get(cache_key)
    cached_status = cached[2] if cached is not None else None
    cached_expiry = getattr(cached_status, "expires_at", None)
    cached_time_valid = not bool(getattr(cached_status, "admitted", False)) or (
        isinstance(cached_expiry, (int, float)) and time.time() < float(cached_expiry)
    )
    if (
        not force_refresh
        and cached is not None
        and (now - cached[0]) < _DEEP_SPECIALIST_STATUS_TTL_S
        and cached[1] == signature
        and cached_time_valid
    ):
        return cached_status
    try:
        status = verify_specialist_qualification_certificate(
            certificate_path,
            trusted_public_key_path=trust_root_path,
            source_root=model_paths.BASE_DIR,
            current_source_commit=_current_specialist_source_commit(),
            resident_descriptor_sha256=resident.descriptor_sha256,
            resident_pointer_sha256=resident.pointer_sha256,
            specialist_model_path=specialist_path,
            requested_domain=requested_domain,
            verify_full_model_hash=True,
        )
    except SpecialistAdmissionError as exc:
        status = denied_status(exc.code)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Optional specialist qualification failed closed: %s", exc)
        status = denied_status("specialist_qualification_error")
    _deep_specialist_status_cache[cache_key] = (now, signature, status)
    return status


