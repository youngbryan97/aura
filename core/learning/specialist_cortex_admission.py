"""Evidence-bound admission for an optional local reasoning specialist.

The resident cortex plus Aura's reasoning architecture is the canonical deep
reasoning lane.  A second model is therefore an optional specialist, not an
authority conferred by parameter count or a directory name.  This module
admits that specialist only when one externally attested certificate binds:

* the exact resident and specialist model artifacts;
* a powered exact paired comparison with no regressed domain;
* equal compute and equal information accounting;
* independent raw-artifact verification;
* a measured serving and host lifecycle envelope; and
* the load-bearing source files that interpret the certificate.

It performs no model evaluation.  It composes and validates evidence produced
by Aura's existing evaluation machinery.  Missing evidence is unmeasured and
denied; a feature flag can disable admission but cannot create it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.architecture_quality.attestation import (
    payload_sha256,
    verify_attested_payload,
)
from core.brain.llm.latent_cortex.exact_paired_grade import (
    EXACT_PAIRED_COMPARISON_SCHEMA,
)
from core.brain.llm.latent_cortex.experiments import PROVEN
from core.brain.llm.latent_cortex.resource_accounting import (
    validate_comparison_accounting_certificate,
)
from core.brain.llm.model_artifact_profile import (
    validate_model_artifact_descriptor,
    validate_model_serving_profile,
)

CERTIFICATE_SCHEMA: Final = "aura.specialist_cortex_qualification.v1"
SOURCE_CLOSURE_SCHEMA: Final = "aura.specialist_cortex.source_closure.v1"
COMPARATIVE_SCHEMA: Final = "aura.specialist_cortex.comparative_evidence.v1"
INDEPENDENT_VERIFICATION_SCHEMA: Final = (
    "aura.specialist_cortex.independent_verification.v1"
)
HOST_ENVELOPE_SCHEMA: Final = "aura.specialist_cortex.host_envelope.v1"

MAX_CERTIFICATE_BYTES: Final = 16 * 1024 * 1024
MAX_SOURCE_FILES: Final = 128
MAX_SOURCE_FILE_BYTES: Final = 16 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES: Final = 64 * 1024 * 1024
MAX_FUTURE_SKEW_S: Final = 300.0

# These files decide what the specialist means, whether its evidence is valid,
# and where requests are routed.  A certificate omitting any of them is not a
# certificate for the running admission path.
REQUIRED_SOURCE_CLOSURE: Final = frozenset(
    {
        "core/architecture_quality/attestation.py",
        "core/brain/inference_gate.py",
        "core/brain/lane_admission.py",
        "core/brain/llm_health_router.py",
        "core/brain/llm/llm_router.py",
        "core/brain/llm/model_artifact_profile.py",
        "core/brain/llm/model_registry.py",
        "core/brain/llm/latent_cortex/exact_paired_grade.py",
        "core/brain/llm/latent_cortex/resource_accounting.py",
        "core/learning/specialist_cortex_admission.py",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40,64}")
_DOMAIN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,79}")


class SpecialistAdmissionError(ValueError):
    """Stable fail-closed error at the specialist admission boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SpecialistAdmissionStatus:
    admitted: bool
    reason: str
    certificate_sha256: str = ""
    resident_descriptor_sha256: str = ""
    specialist_descriptor_sha256: str = ""
    admitted_domains: tuple[str, ...] = ()
    evidence_age_s: float | None = None
    expires_at: float | None = None
    minimum_total_gb: float = 0.0
    minimum_available_gb: float = 0.0
    topology: str = ""

    def admits_domain(self, domain: str | None) -> bool:
        if not self.admitted:
            return False
        normalized = _normalize_domain(domain) if domain else ""
        if not normalized:
            return "general" in self.admitted_domains
        return normalized in self.admitted_domains or "general" in self.admitted_domains


def _fail(code: str) -> None:
    raise SpecialistAdmissionError(code)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _normalize_domain(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if not _DOMAIN_RE.fullmatch(normalized):
        _fail("specialist_domain_invalid")
    return normalized


def _rational(value: object, *, role: str) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        _fail(f"{role}_invalid")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator <= 0
    ):
        _fail(f"{role}_invalid")
    return numerator, denominator


def _rational_less(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] * right[1] < right[0] * left[1]


def _stable_file_binding(root: Path, relative: str) -> dict[str, Any]:
    if relative.startswith("/") or ".." in Path(relative).parts:
        _fail("specialist_source_path_invalid")
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if path.is_symlink() or not resolved.is_file():
            _fail("specialist_source_file_invalid")
        before = resolved.stat()
        if before.st_size > MAX_SOURCE_FILE_BYTES:
            _fail("specialist_source_file_too_large")
        payload = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise SpecialistAdmissionError("specialist_source_file_unreadable") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail("specialist_source_changed_while_hashing")
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def build_source_closure(
    source_root: str | Path,
    *,
    commit: str,
    relative_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the exact load-bearing source closure for offline certification."""

    normalized_commit = str(commit or "").strip().lower()
    if _GIT_OID_RE.fullmatch(normalized_commit) is None:
        _fail("specialist_source_commit_invalid")
    root = Path(source_root).expanduser().resolve(strict=True)
    requested = {str(path).strip() for path in relative_paths if str(path).strip()}
    paths = sorted(REQUIRED_SOURCE_CLOSURE | requested)
    if len(paths) > MAX_SOURCE_FILES:
        _fail("specialist_source_closure_too_large")
    files = [_stable_file_binding(root, path) for path in paths]
    if sum(int(row["size_bytes"]) for row in files) > MAX_SOURCE_TOTAL_BYTES:
        _fail("specialist_source_closure_too_large")
    body = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "commit": normalized_commit,
        "files": files,
    }
    return {**body, "closure_sha256": _sha256(body)}


def _validate_source_closure(
    value: object,
    *,
    source_root: Path,
    current_source_commit: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "commit",
        "files",
        "closure_sha256",
    }:
        _fail("specialist_source_closure_schema_invalid")
    if value.get("schema") != SOURCE_CLOSURE_SCHEMA:
        _fail("specialist_source_closure_schema_invalid")
    commit = str(value.get("commit") or "").lower()
    current = str(current_source_commit or "").strip().lower()
    if _GIT_OID_RE.fullmatch(commit) is None or commit != current:
        _fail("specialist_source_commit_stale")
    files = value.get("files")
    if not isinstance(files, list) or not files or len(files) > MAX_SOURCE_FILES:
        _fail("specialist_source_closure_invalid")
    observed_paths: list[str] = []
    normalized: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            _fail("specialist_source_closure_invalid")
        relative = str(row.get("path") or "")
        if relative in observed_paths or not _is_sha256(row.get("sha256")):
            _fail("specialist_source_closure_invalid")
        observed = _stable_file_binding(source_root, relative)
        if dict(row) != observed:
            _fail("specialist_source_closure_stale")
        observed_paths.append(relative)
        normalized.append(observed)
    if observed_paths != sorted(observed_paths):
        _fail("specialist_source_closure_noncanonical")
    if not REQUIRED_SOURCE_CLOSURE.issubset(observed_paths):
        _fail("specialist_source_closure_incomplete")
    if sum(int(row["size_bytes"]) for row in normalized) > MAX_SOURCE_TOTAL_BYTES:
        _fail("specialist_source_closure_too_large")
    body = {"schema": SOURCE_CLOSURE_SCHEMA, "commit": commit, "files": normalized}
    if value.get("closure_sha256") != _sha256(body):
        _fail("specialist_source_closure_digest_invalid")
    return {**body, "closure_sha256": value["closure_sha256"]}


def _validate_independent_verification(
    value: object,
    *,
    specialist_descriptor_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema",
        "verdict",
        "claim",
        "subject_identity",
        "verifier_identity",
        "verifier_execution",
        "verifier_code_sha256",
        "raw_artifact_package_sha256",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("specialist_independent_verification_schema_invalid")
    body = {key: value[key] for key in required - {"receipt_sha256"}}
    if (
        value.get("schema") != INDEPENDENT_VERIFICATION_SCHEMA
        or value.get("verdict") != "PASS"
        or value.get("claim") != "specialist_over_resident"
        or value.get("subject_identity") != specialist_descriptor_sha256
        or not isinstance(value.get("verifier_identity"), str)
        or not value.get("verifier_identity")
        or value.get("subject_identity") == value.get("verifier_identity")
        or value.get("verifier_execution")
        not in {"separate_process", "separate_trust_domain"}
        or not _is_sha256(value.get("verifier_code_sha256"))
        or not _is_sha256(value.get("raw_artifact_package_sha256"))
        or value.get("receipt_sha256") != _sha256(body)
    ):
        _fail("specialist_independent_verification_invalid")
    return dict(value)


def _validate_arm_bindings(
    value: object,
    *,
    accounting: Mapping[str, Any],
    resident_descriptor_sha256: str,
    specialist_descriptor_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"treatment", "control"}:
        _fail("specialist_arm_binding_schema_invalid")
    expected = {
        "treatment": {
            "model_descriptor_sha256": specialist_descriptor_sha256,
            "resource_receipt_sha256": accounting["treatment_resource_sha256"],
            "information_receipt_sha256": accounting["treatment_information_sha256"],
        },
        "control": {
            "model_descriptor_sha256": resident_descriptor_sha256,
            "resource_receipt_sha256": accounting["control_resource_sha256"],
            "information_receipt_sha256": accounting["control_information_sha256"],
        },
    }
    normalized: dict[str, Any] = {}
    for arm, expected_binding in expected.items():
        observed = value.get(arm)
        if not isinstance(observed, Mapping) or dict(observed) != expected_binding:
            _fail("specialist_arm_binding_invalid")
        normalized[arm] = dict(observed)
    return normalized


def _validate_grade(
    value: object,
    *,
    resident_descriptor_sha256: str,
    specialist_descriptor_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != {
        "experiment",
        "statement",
        "tier",
        "evidence",
    }:
        _fail("specialist_comparative_grade_schema_invalid")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("schema") != EXACT_PAIRED_COMPARISON_SCHEMA:
        _fail("specialist_comparative_grade_schema_invalid")
    if (
        value.get("tier") != PROVEN
        or evidence.get("treatment") != specialist_descriptor_sha256
        or evidence.get("control") != resident_descriptor_sha256
        or evidence.get("require_compute") is not True
        or evidence.get("all_families_noninferior") is not True
        or evidence.get("regressed_families") != []
        or evidence.get("underpowered_families") != []
        or evidence.get("invalid_compute_families") != []
    ):
        _fail("specialist_comparative_grade_not_admissible")
    positive = evidence.get("positive_families")
    families = evidence.get("families")
    if (
        not isinstance(positive, list)
        or not positive
        or positive != sorted(set(positive))
        or not isinstance(families, Mapping)
    ):
        _fail("specialist_comparative_domains_invalid")
    domains = tuple(_normalize_domain(str(domain)) for domain in positive)
    alpha = _rational(evidence.get("alpha"), role="specialist_comparative_alpha")
    minimum = _rational(
        evidence.get("minimum_effect"), role="specialist_comparative_minimum"
    )
    for domain in domains:
        row = families.get(domain)
        if not isinstance(row, Mapping):
            _fail("specialist_comparative_domain_missing")
        adjusted = _rational(
            row.get("holm_adjusted_p"), role="specialist_comparative_pvalue"
        )
        bounds = row.get("effect_bounds")
        if not isinstance(bounds, Mapping) or bounds.get("certified") is not True:
            _fail("specialist_comparative_bounds_invalid")
        lower = _rational(bounds.get("lower"), role="specialist_comparative_lower")
        if not _rational_less(adjusted, alpha) or not _rational_less(minimum, lower):
            _fail("specialist_comparative_domain_not_positive")
        if (
            row.get("missing_compute") is not False
            or row.get("nonpositive_compute") is not False
            or row.get("compute_mismatch_task_ids") != []
        ):
            _fail("specialist_comparative_compute_invalid")
    return dict(value), domains


def _validate_comparative(
    value: object,
    *,
    resident_descriptor_sha256: str,
    specialist_descriptor_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    required = {
        "schema",
        "resident_descriptor_sha256",
        "specialist_descriptor_sha256",
        "task_manifest_sha256",
        "evaluator_sha256",
        "grade",
        "grade_sha256",
        "accounting",
        "arm_bindings",
        "independent_verification",
        "admitted_domains",
        "comparative_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("specialist_comparative_schema_invalid")
    if (
        value.get("schema") != COMPARATIVE_SCHEMA
        or value.get("resident_descriptor_sha256") != resident_descriptor_sha256
        or value.get("specialist_descriptor_sha256") != specialist_descriptor_sha256
        or not _is_sha256(value.get("task_manifest_sha256"))
        or not _is_sha256(value.get("evaluator_sha256"))
    ):
        _fail("specialist_comparative_binding_invalid")
    grade, domains = _validate_grade(
        value.get("grade"),
        resident_descriptor_sha256=resident_descriptor_sha256,
        specialist_descriptor_sha256=specialist_descriptor_sha256,
    )
    if value.get("grade_sha256") != _sha256(grade):
        _fail("specialist_comparative_grade_digest_invalid")
    try:
        accounting = validate_comparison_accounting_certificate(value.get("accounting"))
    except (TypeError, ValueError) as exc:
        raise SpecialistAdmissionError("specialist_accounting_invalid") from exc
    if (
        accounting.get("admitted") is not True
        or accounting.get("require_compute_parity") is not True
        or accounting.get("information_matched") is not True
        or accounting.get("reasons") != []
    ):
        _fail("specialist_accounting_not_admissible")
    arm_bindings = _validate_arm_bindings(
        value.get("arm_bindings"),
        accounting=accounting,
        resident_descriptor_sha256=resident_descriptor_sha256,
        specialist_descriptor_sha256=specialist_descriptor_sha256,
    )
    independent = _validate_independent_verification(
        value.get("independent_verification"),
        specialist_descriptor_sha256=specialist_descriptor_sha256,
    )
    admitted = value.get("admitted_domains")
    if not isinstance(admitted, list) or tuple(admitted) != domains:
        _fail("specialist_comparative_domains_mismatch")
    body = {key: value[key] for key in required - {"comparative_sha256"}}
    if value.get("comparative_sha256") != _sha256(body):
        _fail("specialist_comparative_digest_invalid")
    return {
        **body,
        "comparative_sha256": value["comparative_sha256"],
        "grade": grade,
        "accounting": accounting,
        "arm_bindings": arm_bindings,
        "independent_verification": independent,
    }, domains


def _validate_host_envelope(
    value: object,
    *,
    resident_descriptor_sha256: str,
    specialist_descriptor_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema",
        "resident_descriptor_sha256",
        "specialist_descriptor_sha256",
        "topology",
        "minimum_total_gb",
        "minimum_available_gb",
        "maximum_peak_gb",
        "load_pass",
        "cancel_pass",
        "unload_pass",
        "resident_restore_pass",
        "evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("specialist_host_envelope_schema_invalid")
    body = {key: value[key] for key in required - {"evidence_sha256"}}
    numbers: dict[str, float] = {}
    for name in ("minimum_total_gb", "minimum_available_gb", "maximum_peak_gb"):
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            _fail("specialist_host_envelope_invalid")
        numbers[name] = float(raw)
    if (
        value.get("schema") != HOST_ENVELOPE_SCHEMA
        or value.get("resident_descriptor_sha256") != resident_descriptor_sha256
        or value.get("specialist_descriptor_sha256") != specialist_descriptor_sha256
        or value.get("topology") not in {"exclusive_swap", "co_resident"}
        or any(not math.isfinite(number) or number <= 0 for number in numbers.values())
        or any(
            value.get(name) is not True
            for name in ("load_pass", "cancel_pass", "unload_pass", "resident_restore_pass")
        )
        or value.get("evidence_sha256") != _sha256(body)
    ):
        _fail("specialist_host_envelope_invalid")
    return {**body, "evidence_sha256": value["evidence_sha256"]}


def verify_specialist_qualification_certificate(
    certificate_path: str | Path,
    *,
    trusted_public_key_path: str | Path,
    source_root: str | Path,
    current_source_commit: str,
    resident_descriptor_sha256: str,
    resident_pointer_sha256: str,
    specialist_model_path: str | Path,
    requested_domain: str | None = None,
    now: float | None = None,
    verify_full_model_hash: bool = True,
) -> SpecialistAdmissionStatus:
    """Validate one externally attested specialist certificate.

    The caller owns host admission because current memory availability is a
    live runtime measurement.  This function returns the measured host minima
    from the certificate so that every caller applies the same evidence.
    """

    path = Path(certificate_path).expanduser()
    trust_path = Path(trusted_public_key_path).expanduser()
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CERTIFICATE_BYTES:
            _fail("specialist_certificate_missing")
        if trust_path.is_symlink() or not trust_path.is_file():
            _fail("specialist_trust_root_missing")
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except SpecialistAdmissionError:
        raise
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SpecialistAdmissionError("specialist_certificate_unreadable") from exc
    if not isinstance(raw, dict) or raw.get("schema") != CERTIFICATE_SCHEMA:
        _fail("specialist_certificate_schema_invalid")
    try:
        verify_attested_payload(
            raw,
            digest_field="certificate_sha256",
            trusted_public_key_pem=trust_path.read_bytes(),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SpecialistAdmissionError("specialist_certificate_attestation_invalid") from exc

    required = {
        "schema",
        "issued_at",
        "expires_at",
        "source",
        "resident",
        "specialist_descriptor",
        "serving_profile",
        "comparative",
        "host_envelope",
        "certificate_sha256",
        "signature",
    }
    if set(raw) != required:
        _fail("specialist_certificate_schema_invalid")
    clock = float(time.time() if now is None else now)
    issued = raw.get("issued_at")
    expires = raw.get("expires_at")
    if (
        isinstance(issued, bool)
        or not isinstance(issued, (int, float))
        or isinstance(expires, bool)
        or not isinstance(expires, (int, float))
        or not math.isfinite(float(issued))
        or not math.isfinite(float(expires))
        or float(issued) > clock + MAX_FUTURE_SKEW_S
        or float(expires) <= float(issued)
        or clock >= float(expires)
    ):
        _fail("specialist_certificate_expired_or_invalid")

    resident = raw.get("resident")
    if not isinstance(resident, Mapping) or set(resident) != {
        "descriptor_sha256",
        "pointer_sha256",
    }:
        _fail("specialist_resident_binding_invalid")
    if (
        resident.get("descriptor_sha256") != resident_descriptor_sha256
        or resident.get("pointer_sha256") != resident_pointer_sha256
    ):
        _fail("specialist_resident_binding_stale")
    descriptor = raw.get("specialist_descriptor")
    if not isinstance(descriptor, dict):
        _fail("specialist_descriptor_invalid")
    try:
        validate_model_artifact_descriptor(
            descriptor,
            model_path=specialist_model_path,
            verify_full_hash=verify_full_model_hash,
        )
        profile = raw.get("serving_profile")
        if not isinstance(profile, dict):
            _fail("specialist_serving_profile_invalid")
        validate_model_serving_profile(profile, descriptor)
    except SpecialistAdmissionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise SpecialistAdmissionError("specialist_model_or_serving_evidence_invalid") from exc
    specialist_sha = str(descriptor.get("descriptor_sha256") or "")
    if specialist_sha == resident_descriptor_sha256:
        _fail("specialist_not_distinct")

    root = Path(source_root).expanduser().resolve(strict=True)
    _validate_source_closure(
        raw.get("source"),
        source_root=root,
        current_source_commit=current_source_commit,
    )
    _, domains = _validate_comparative(
        raw.get("comparative"),
        resident_descriptor_sha256=resident_descriptor_sha256,
        specialist_descriptor_sha256=specialist_sha,
    )
    host = _validate_host_envelope(
        raw.get("host_envelope"),
        resident_descriptor_sha256=resident_descriptor_sha256,
        specialist_descriptor_sha256=specialist_sha,
    )
    normalized_domain = _normalize_domain(requested_domain) if requested_domain else ""
    domain_admitted = (
        normalized_domain in domains or "general" in domains
        if normalized_domain
        else "general" in domains
    )
    if not domain_admitted:
        return SpecialistAdmissionStatus(
            admitted=False,
            reason="domain_not_qualified",
            certificate_sha256=str(raw["certificate_sha256"]),
            resident_descriptor_sha256=resident_descriptor_sha256,
            specialist_descriptor_sha256=specialist_sha,
            admitted_domains=domains,
            evidence_age_s=max(0.0, clock - float(issued)),
            expires_at=float(expires),
            minimum_total_gb=float(host["minimum_total_gb"]),
            minimum_available_gb=float(host["minimum_available_gb"]),
            topology=str(host["topology"]),
        )
    return SpecialistAdmissionStatus(
        admitted=True,
        reason="qualified",
        certificate_sha256=str(raw["certificate_sha256"]),
        resident_descriptor_sha256=resident_descriptor_sha256,
        specialist_descriptor_sha256=specialist_sha,
        admitted_domains=domains,
        evidence_age_s=max(0.0, clock - float(issued)),
        expires_at=float(expires),
        minimum_total_gb=float(host["minimum_total_gb"]),
        minimum_available_gb=float(host["minimum_available_gb"]),
        topology=str(host["topology"]),
    )


def denied_status(reason: str) -> SpecialistAdmissionStatus:
    return SpecialistAdmissionStatus(admitted=False, reason=str(reason or "unmeasured"))


def certificate_payload_sha256(value: Mapping[str, Any]) -> str:
    """Public helper for offline producers before external attestation."""

    return payload_sha256(dict(value))


__all__ = [
    "CERTIFICATE_SCHEMA",
    "COMPARATIVE_SCHEMA",
    "HOST_ENVELOPE_SCHEMA",
    "INDEPENDENT_VERIFICATION_SCHEMA",
    "REQUIRED_SOURCE_CLOSURE",
    "SOURCE_CLOSURE_SCHEMA",
    "SpecialistAdmissionError",
    "SpecialistAdmissionStatus",
    "build_source_closure",
    "certificate_payload_sha256",
    "denied_status",
    "verify_specialist_qualification_certificate",
]
