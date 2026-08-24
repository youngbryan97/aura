"""Evidence-bound activation for the qualified semantic neural machine."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from core.learning.recovery_package_identity import (
    DescriptorIdentity,
    descriptor_from_manifest,
    evidence_namespace_errors,
)
from core.learning.recovery_package_identity import (
    package_id as recovery_package_id,
)

SEMANTIC_NEURAL_SERVING_SCHEMA: Final = "aura.semantic_neural_serving.v2"
SEMANTIC_NEURAL_RUNTIME_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_neural_runtime_verification.v3"
)
SEMANTIC_NEURAL_SERVING_MODE: Final = "qualified_exact_semantic_v1"
PACKAGE_ID: Final = "cp568-resident-semantic-neural-active-r1"
RECOVERY_PACKAGE_CAMPAIGN: Final = "rlc-27b-recovery"
PROMOTION_MODE: Final = "active"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_ACTIVATION_PATH: Final = (
    REPO_ROOT / "artifacts/closeout/latent_cortex/cp568_semantic_neural_active_r1/activation.json"
)
ACTIVE_ACTIVATION_PATH: Final = (
    REPO_ROOT / "training/fused-model/semantic-neural-active.json"
)
RESIDENT_RESULT_PATH: Final = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp566_resident_mixed_multidomain_replication/result.json"
)
RESIDENT_VERIFICATION_PATH: Final = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp566_resident_mixed_multidomain_replication/verification.json"
)
RESIDENT_ADJUDICATION_PATH: Final = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp566_resident_mixed_multidomain_replication/adjudication.json"
)
MEASURED_SOURCE_FILES: Final = (
    "core/brain/llm/latent_cortex/semantic_neural_decode_context.py",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/manifest.json",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/weights.safetensors",
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/brain/llm/latent_cortex/semantic_surface_adapter.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/semantic_neural_controls.py",
    "core/learning/semantic_neural_machine.py",
)
ACTIVATION_SOURCE_FILES: Final = (
    *MEASURED_SOURCE_FILES,
    "core/learning/semantic_neural_runtime_machine.py",
    "core/brain/llm/latent_cortex/persistence.py",
    "core/brain/llm/qualified_recurrent_ingress.py",
)
# Whole-file hashes remain mandatory for the measured mechanism and its direct
# loaders. Integration modules change often, so bind their load-bearing AST
# instead of disabling a proven machine when unrelated code in a large module
# moves. A contract selector names either a function/method or a specific call.
INTEGRATION_SOURCE_CONTRACTS: Final = {
    "core/brain/foreground_latent_runtime.py": ("symbol:run_foreground_latent_episode",),
    "core/brain/latent_cortex_service.py": (
        "symbol:LatentCortexService.qualified_recurrent_reason",
    ),
}
INTEGRATION_SOURCE_FILES: Final = tuple(INTEGRATION_SOURCE_CONTRACTS)
ALLOWED_FAMILIES: Final = (
    "frontier_calibration",
    "frontier_coding",
    "frontier_misleading_premise",
    "frontier_scientific_inference",
)
ALLOWED_SURFACE_PROFILES: Final = (
    "lab_report",
    "narrative",
    "compact",
)
EVIDENCE_DOMAINS: Final = (
    "coding",
    "calibration",
    "misleading_premise",
    "scientific_inference",
)
ACTIVATION_CLAIM_BOUNDARY: Final = (
    "runtime qualification of a replicated lesion-dependent resident-32B effective "
    "reasoning gain over ordinary decode on the frozen four-domain semantic cohort; "
    "bounded executable families only, not open-domain general reasoning, static "
    "fusion, frontier performance, consciousness evidence, or unrestricted promotion"
)
LEGACY_ADJUDICATION_CLAIM: Final = (
    "replicated lesion-dependent resident-32B effective reasoning gain over "
    "ordinary decode on the frozen four-domain semantic cohort"
)
MODEL_BOUND_ADJUDICATION_CLAIM: Final = (
    "replicated lesion-dependent resident-model effective reasoning gain over "
    "ordinary decode on the frozen four-domain semantic cohort"
)
_FALSE_VALUES: Final = frozenset({"0", "false", "no", "off", "disabled"})
_OPTIONAL_AST_FIELDS: Final = frozenset({"type_params"})


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _symbol_node(tree: ast.Module, qualified_name: str) -> ast.AST:
    body: list[ast.stmt] = tree.body
    current: ast.AST | None = None
    for part in qualified_name.split("."):
        current = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if current is None:
            raise RuntimeError(f"semantic integration symbol is missing: {qualified_name}")
        body = list(getattr(current, "body", ()))
    assert current is not None
    return current


def _canonical_ast(value: Any) -> Any:
    """Serialize Python syntax without binding seals to an interpreter AST schema."""

    if isinstance(value, ast.AST):
        fields = {}
        for name, child in ast.iter_fields(value):
            # Python 3.12 and 3.14 expose different AST field inventories.
            # Empty optional syntax carries no program semantics, so omit it;
            # non-empty generic parameters remain part of the contract.
            if name in _OPTIONAL_AST_FIELDS and not child:
                continue
            fields[name] = _canonical_ast(child)
        return {"node": type(value).__name__, "fields": fields}
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    if isinstance(value, tuple):
        return {"tuple": [_canonical_ast(item) for item in value]}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if value is Ellipsis:
        return {"ellipsis": True}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(f"unsupported semantic integration AST value: {type(value).__name__}")


def _integration_contract_sha(path: Path, selector: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kind, separator, target = selector.partition(":")
    if not separator or not target:
        raise RuntimeError(f"semantic integration selector is invalid: {selector}")
    if kind == "symbol":
        payload: Any = _canonical_ast(_symbol_node(tree, target))
    elif kind == "call":
        calls = sorted(
            _sha(_canonical_ast(node))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == target
        )
        if not calls:
            raise RuntimeError(f"semantic integration call is missing: {target}")
        payload = calls
    else:
        raise RuntimeError(f"semantic integration selector kind is invalid: {kind}")
    return _sha(payload)


def _integration_contract_hashes(repo_root: Path) -> dict[str, str]:
    return {
        f"{relative}::{selector}": _integration_contract_sha(
            repo_root / relative,
            selector,
        )
        for relative, selectors in INTEGRATION_SOURCE_CONTRACTS.items()
        for selector in selectors
    }


def _read_bounded_json(path: Path, *, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve(strict=True)
    metadata = resolved.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_size <= 1
        or metadata.st_size > maximum_bytes
    ):
        raise RuntimeError(f"semantic serving evidence file is invalid: {path.name}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"semantic serving evidence JSON is invalid: {path.name}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"semantic serving evidence is not an object: {path.name}")
    return payload, raw


def _identity_for_model(model_path: Path) -> dict[str, str]:
    resolved = model_path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "config_sha256": _file_sha(resolved / "config.json"),
        "weights_index_sha256": _file_sha(resolved / "model.safetensors.index.json"),
    }


def _identity_for_manifest(manifest_path: Path) -> dict[str, Any]:
    payload, raw = _read_bounded_json(manifest_path, maximum_bytes=64 * 1024)
    active_path = (
        Path(str(payload.get("active_model_path") or "")).expanduser().resolve(strict=True)
    )
    return {
        "path": str(manifest_path.expanduser().resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "active_model_path": str(active_path),
        "schema_version": payload.get("schema_version"),
        "base_model": str(payload.get("base_model") or ""),
        "tag": str(payload.get("tag") or ""),
        "fused_at": payload.get("fused_at"),
    }


def _recovery_descriptor(value: Any) -> DescriptorIdentity:
    if not isinstance(value, dict):
        raise RuntimeError("semantic recovery descriptor is missing")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    try:
        descriptor = DescriptorIdentity(**body)
    except (TypeError, ValueError):
        raise RuntimeError("semantic recovery descriptor is invalid") from None
    if value.get("fingerprint") != descriptor.fingerprint():
        raise RuntimeError("semantic recovery descriptor fingerprint is invalid")
    return descriptor


def _activation_claim_boundary(claim: str) -> str:
    if claim == LEGACY_ADJUDICATION_CLAIM:
        return ACTIVATION_CLAIM_BOUNDARY
    if claim != MODEL_BOUND_ADJUDICATION_CLAIM:
        raise RuntimeError("semantic adjudication claim is not recognized")
    return (
        "runtime qualification of a replicated lesion-dependent resident-model "
        "effective reasoning gain over ordinary decode on the frozen four-domain "
        "semantic cohort; bounded executable families only, not open-domain general "
        "reasoning, static fusion, frontier performance, consciousness evidence, or "
        "unrestricted promotion"
    )


def active_semantic_neural_activation_path() -> Path:
    """Resolve the operational activation without rewriting historical evidence."""

    return ACTIVE_ACTIVATION_PATH if ACTIVE_ACTIVATION_PATH.exists() else DEFAULT_ACTIVATION_PATH


def _manifest_identity_matches_activation(
    *,
    expected: dict[str, Any],
    current: dict[str, Any],
    selected_model: Path,
) -> bool:
    """Accept an exact manifest or a verified identity-only normalization."""

    if current == expected:
        return True
    try:
        from core.brain.llm.model_registry import read_active_cortex_spec

        spec = read_active_cortex_spec(str(expected["path"]))
        return bool(
            spec is not None
            and spec.identity_transition_verified
            and spec.predecessor_pointer_sha256 == expected.get("sha256")
            and spec.manifest_path == Path(str(expected["path"])).resolve(strict=True)
            and spec.pointer_sha256 == current.get("sha256")
            and spec.model_path == selected_model
            and current.get("active_model_path") == expected.get("active_model_path")
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _relative_evidence_path(repo_root: Path, path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        raise RuntimeError("semantic serving evidence is outside the repository") from None
    return relative.as_posix()


def _resolve_evidence_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RuntimeError("semantic serving evidence path is not relative")
    candidate = (repo_root / value).resolve(strict=True)
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        raise RuntimeError("semantic serving evidence escaped the repository") from None
    return candidate


def _verify_resident_evidence(
    *,
    repo_root: Path,
    result_path: Path,
    verification_path: Path,
    adjudication_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes, bytes]:
    result, result_raw = _read_bounded_json(result_path, maximum_bytes=4 * 1024 * 1024)
    verification, verification_raw = _read_bounded_json(
        verification_path,
        maximum_bytes=512 * 1024,
    )
    adjudication, adjudication_raw = _read_bounded_json(
        adjudication_path,
        maximum_bytes=512 * 1024,
    )
    result_body = {key: value for key, value in result.items() if key != "receipt_sha256"}
    verification_body = {
        key: value for key, value in verification.items() if key != "verification_receipt_sha256"
    }
    adjudication_body = {
        key: value for key, value in adjudication.items() if key != "adjudication_receipt_sha256"
    }
    exact_by_arm = verification.get("independent_exact_by_arm")
    task_count = verification.get("task_count")
    claim = str(adjudication.get("claim") or "")
    adjudication_checks = adjudication.get("checks")
    if not isinstance(adjudication_checks, dict):
        adjudication_checks = {}
    model_identity = result.get("model_identity")
    manifest_identity = result.get("resident_manifest_identity")
    identities_match = (
        isinstance(model_identity, dict)
        and model_identity == verification.get("model_identity")
        and isinstance(manifest_identity, dict)
        and manifest_identity == verification.get("resident_manifest_identity")
    )
    if claim == MODEL_BOUND_ADJUDICATION_CLAIM:
        identities_match = bool(
            identities_match
            and adjudication.get("model_identity") == model_identity
            and adjudication.get("resident_manifest_identity") == manifest_identity
            and adjudication_checks.get("model_identity_match") is True
            and adjudication_checks.get("resident_manifest_identity_match") is True
        )
    elif claim == LEGACY_ADJUDICATION_CLAIM:
        legacy_activation, _legacy_raw = _read_bounded_json(
            DEFAULT_ACTIVATION_PATH,
            maximum_bytes=512 * 1024,
        )
        identities_match = bool(
            identities_match and model_identity == legacy_activation.get("model_identity")
        )
    else:
        identities_match = False
    if (
        result.get("receipt_sha256") != _sha(result_body)
        or verification.get("verification_receipt_sha256") != _sha(verification_body)
        or adjudication.get("adjudication_receipt_sha256") != _sha(adjudication_body)
        or adjudication.get("passed") is not True
        or adjudication.get("verdict") != "BOUNDED_WOW_SIGNAL"
        or not identities_match
        or "not open-domain general reasoning" not in str(adjudication.get("limitations") or "")
        or adjudication.get("input_receipts", {}).get("result") != result.get("receipt_sha256")
        or adjudication.get("input_receipts", {}).get("verification")
        != verification.get("verification_receipt_sha256")
        or verification.get("verified") is not True
        or verification.get("artifact_sha256") != hashlib.sha256(result_raw).hexdigest()
        or verification.get("artifact_receipt_sha256") != result.get("receipt_sha256")
        or verification.get("gain_count", 0) <= 0
        or verification.get("regression_count") != 0
        or result.get("surface_profile") != "mixed_multidomain_v1"
        or verification.get("surface_profile") != "mixed_multidomain_v1"
        or result.get("domains") != list(EVIDENCE_DOMAINS)
        or verification.get("domains") != list(EVIDENCE_DOMAINS)
        or result.get("task_count") != task_count
        or task_count != 60
        or not isinstance(exact_by_arm, dict)
        or exact_by_arm.get("treatment") != task_count
        or any(
            exact_by_arm.get(arm, task_count) >= task_count
            for arm in ("matched_wire_base", "coefficient_lesion", "matched_wrong_state")
        )
        or verification.get("coefficient_lesion_contract_verified") is not True
        or "resident model bound by" not in str(verification.get("claim_boundary") or "")
        or "not open-domain" not in str(verification.get("claim_boundary") or "")
    ):
        raise RuntimeError("resident semantic serving evidence is not admissible")
    measured_hashes = result.get("source_sha256s")
    if not isinstance(measured_hashes, dict) or any(
        measured_hashes.get(relative) != _file_sha(repo_root / relative)
        for relative in MEASURED_SOURCE_FILES
    ):
        raise RuntimeError("resident measured semantic source has drifted")
    return (
        result,
        verification,
        adjudication,
        result_raw,
        verification_raw,
        adjudication_raw,
    )


def build_semantic_neural_activation(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path = RESIDENT_RESULT_PATH,
    verification_path: Path = RESIDENT_VERIFICATION_PATH,
    adjudication_path: Path = RESIDENT_ADJUDICATION_PATH,
    resident_manifest_path: Path,
    model_path: Path,
    runtime_verification_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize a source/model/evidence-bound qualified activation."""

    root = repo_root.expanduser().resolve(strict=True)
    (
        result,
        verification,
        adjudication,
        result_raw,
        verification_raw,
        adjudication_raw,
    ) = _verify_resident_evidence(
        repo_root=root,
        result_path=result_path,
        verification_path=verification_path,
        adjudication_path=adjudication_path,
    )
    model_identity = _identity_for_model(model_path)
    manifest_identity = _identity_for_manifest(resident_manifest_path)
    if (
        model_identity != verification.get("model_identity")
        or manifest_identity != verification.get("resident_manifest_identity")
        or manifest_identity["active_model_path"] != model_identity["path"]
    ):
        raise RuntimeError("semantic serving resident identity differs from evidence")
    manifest_payload, _manifest_raw = _read_bounded_json(
        resident_manifest_path,
        maximum_bytes=512 * 1024,
    )
    adjudication_claim = str(adjudication.get("claim") or "")
    if adjudication_claim == LEGACY_ADJUDICATION_CLAIM:
        serving_package_id = PACKAGE_ID
        descriptor_identity = None
    else:
        descriptor = descriptor_from_manifest(manifest_payload)
        serving_package_id = recovery_package_id(
            descriptor,
            campaign=RECOVERY_PACKAGE_CAMPAIGN,
        )
        descriptor_identity = descriptor.as_dict()
    body: dict[str, Any] = {
        "schema": SEMANTIC_NEURAL_SERVING_SCHEMA,
        "package_id": serving_package_id,
        "mode": SEMANTIC_NEURAL_SERVING_MODE,
        "promotion_mode": PROMOTION_MODE,
        "active_by_default": True,
        "allowed_families": list(ALLOWED_FAMILIES),
        "allowed_surface_profiles": list(ALLOWED_SURFACE_PROFILES),
        "source_sha256s": {
            relative: _file_sha(root / relative) for relative in ACTIVATION_SOURCE_FILES
        },
        "integration_contract_sha256s": _integration_contract_hashes(root),
        "model_identity": model_identity,
        "resident_manifest_identity": manifest_identity,
        "evidence": {
            "result_path": _relative_evidence_path(root, result_path),
            "result_sha256": hashlib.sha256(result_raw).hexdigest(),
            "result_receipt_sha256": result["receipt_sha256"],
            "verification_path": _relative_evidence_path(root, verification_path),
            "verification_sha256": hashlib.sha256(verification_raw).hexdigest(),
            "verification_receipt_sha256": verification["verification_receipt_sha256"],
            "adjudication_path": _relative_evidence_path(root, adjudication_path),
            "adjudication_sha256": hashlib.sha256(adjudication_raw).hexdigest(),
            "adjudication_receipt_sha256": adjudication["adjudication_receipt_sha256"],
            "adjudication_verdict": adjudication["verdict"],
            "adjudication_claim": adjudication["claim"],
            "adjudication_limitations": adjudication["limitations"],
            "gain_count": verification["gain_count"],
            "regression_count": verification["regression_count"],
            "paired_one_sided_exact_p": verification["paired_one_sided_exact_p"],
            "domains": verification["domains"],
            "task_count": verification["task_count"],
            "independent_exact_by_arm": verification["independent_exact_by_arm"],
            "coefficient_lesion_contract_verified": verification[
                "coefficient_lesion_contract_verified"
            ],
        },
        "claim_boundary": _activation_claim_boundary(adjudication_claim),
    }
    if descriptor_identity is not None:
        body["descriptor_identity"] = descriptor_identity
    if runtime_verification_path is not None:
        runtime_verification, runtime_raw = _read_bounded_json(
            runtime_verification_path,
            maximum_bytes=4 * 1024 * 1024,
        )
        candidate_activation_sha256 = _sha(body)
        runtime_body = {
            key: value
            for key, value in runtime_verification.items()
            if key != "verification_receipt_sha256"
        }
        runtime_receipt = runtime_verification.get("activation_receipt")
        if (
            runtime_verification.get("schema")
            != SEMANTIC_NEURAL_RUNTIME_VERIFICATION_SCHEMA
            or
            runtime_verification.get("verified") is not True
            or runtime_verification.get("task_count") != 120
            or runtime_verification.get("exact_count") != 120
            or runtime_verification.get("lesion_disruption_count") != 120
            or runtime_verification.get("measured_backend_receipt_equivalence_count") != 120
            or runtime_verification.get("foreground_integration_count") != 120
            or runtime_verification.get("service_integration_count") != 120
            or runtime_verification.get("unsupported_language_refused") is not True
            or runtime_verification.get("verification_receipt_sha256") != _sha(runtime_body)
            or not isinstance(runtime_receipt, dict)
            or runtime_receipt.get("activation_sha256") != candidate_activation_sha256
            or runtime_receipt.get("package_id") != serving_package_id
            or runtime_receipt.get("promotion_mode") != PROMOTION_MODE
        ):
            raise RuntimeError("semantic runtime qualification is not admissible")
        body["runtime_qualification"] = {
            "path": _relative_evidence_path(root, runtime_verification_path),
            "sha256": hashlib.sha256(runtime_raw).hexdigest(),
            "verification_receipt_sha256": runtime_verification["verification_receipt_sha256"],
            "candidate_activation_sha256": candidate_activation_sha256,
            "task_count": runtime_verification["task_count"],
            "exact_count": runtime_verification["exact_count"],
            "lesion_disruption_count": runtime_verification["lesion_disruption_count"],
            "measured_backend_receipt_equivalence_count": runtime_verification[
                "measured_backend_receipt_equivalence_count"
            ],
            "foreground_integration_count": runtime_verification[
                "foreground_integration_count"
            ],
            "service_integration_count": runtime_verification[
                "service_integration_count"
            ],
            "unsupported_language_refused": runtime_verification["unsupported_language_refused"],
            "max_latency_ms": runtime_verification["max_latency_ms"],
        }
    return {**body, "activation_sha256": _sha(body)}


def semantic_neural_activation_errors(
    activation: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    model_path: Path | None = None,
    verify_live_identity: bool = True,
    require_runtime_qualification: bool = True,
) -> list[str]:
    """Recompute every mutable dependency of a serving activation."""

    errors: list[str] = []
    recovery_descriptor: DescriptorIdentity | None = None
    body = {key: value for key, value in activation.items() if key != "activation_sha256"}
    if activation.get("schema") != SEMANTIC_NEURAL_SERVING_SCHEMA:
        errors.append("schema")
    package = activation.get("package_id")
    claim = str((activation.get("evidence") or {}).get("adjudication_claim") or "")
    if package == PACKAGE_ID:
        if claim != LEGACY_ADJUDICATION_CLAIM or "descriptor_identity" in activation:
            errors.append("package_id")
    else:
        try:
            recovery_descriptor = _recovery_descriptor(
                activation.get("descriptor_identity")
            )
            if package != recovery_package_id(
                recovery_descriptor,
                campaign=RECOVERY_PACKAGE_CAMPAIGN,
            ):
                errors.append("package_id")
        except RuntimeError:
            errors.append("package_id")
    if activation.get("mode") != SEMANTIC_NEURAL_SERVING_MODE:
        errors.append("mode")
    if activation.get("promotion_mode") != PROMOTION_MODE:
        errors.append("promotion_mode")
    if activation.get("active_by_default") is not True:
        errors.append("active_by_default")
    if activation.get("allowed_families") != list(ALLOWED_FAMILIES):
        errors.append("allowed_families")
    if activation.get("allowed_surface_profiles") != list(ALLOWED_SURFACE_PROFILES):
        errors.append("allowed_surface_profiles")
    try:
        expected_claim_boundary = _activation_claim_boundary(claim)
    except RuntimeError:
        expected_claim_boundary = None
    if activation.get("claim_boundary") != expected_claim_boundary:
        errors.append("claim_boundary")
    if activation.get("activation_sha256") != _sha(body):
        errors.append("activation_sha256")
    root = repo_root.expanduser().resolve(strict=True)
    source_hashes = activation.get("source_sha256s")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(ACTIVATION_SOURCE_FILES):
        errors.append("source_inventory")
    else:
        # Name the files. "source_drift" alone is a dead end: it says a proven
        # surface has been disabled without saying what disabled it, so the
        # only way to find out is to re-derive twenty hashes by hand.
        #
        # LIVE 2026-08-17: this surface had been dark since 2026-08-15, the day
        # it was sealed, and nobody knew. Four of the twenty bound files had
        # moved — semantic_neural_shadow.py (Aug 15), foreground_latent_runtime
        # .py (Aug 16), latent_cortex_service.py (Aug 17) and
        # response_generation_unitary.py (Aug 17). A BOUNDED_WOW_SIGNAL
        # established at p=5.7e-14 with lesion controls was switched off by
        # ordinary development, silently, for two days.
        drifted = sorted(
            relative
            for relative in ACTIVATION_SOURCE_FILES
            if source_hashes.get(relative) != _file_sha(root / relative)
        )
        if drifted:
            errors.append(f"source_drift:{','.join(drifted)}")
    contract_hashes = activation.get("integration_contract_sha256s")
    expected_contracts = _integration_contract_hashes(root)
    if not isinstance(contract_hashes, dict) or set(contract_hashes) != set(expected_contracts):
        errors.append("integration_contract_inventory")
    else:
        drifted_contracts = sorted(
            key
            for key, expected in expected_contracts.items()
            if contract_hashes.get(key) != expected
        )
        if drifted_contracts:
            errors.append("integration_contract_drift:" + ",".join(drifted_contracts))
    runtime_qualification = activation.get("runtime_qualification")
    if not isinstance(runtime_qualification, dict) and require_runtime_qualification:
        errors.append("runtime_qualification")
    elif isinstance(runtime_qualification, dict):
        try:
            runtime_path = _resolve_evidence_path(root, runtime_qualification["path"])
            runtime_verification, runtime_raw = _read_bounded_json(
                runtime_path,
                maximum_bytes=4 * 1024 * 1024,
            )
            runtime_body = {
                key: value
                for key, value in runtime_verification.items()
                if key != "verification_receipt_sha256"
            }
            candidate_body = {
                key: value for key, value in body.items() if key != "runtime_qualification"
            }
            runtime_receipt = runtime_verification.get("activation_receipt")
            if (
                runtime_verification.get("schema")
                != SEMANTIC_NEURAL_RUNTIME_VERIFICATION_SCHEMA
                or
                hashlib.sha256(runtime_raw).hexdigest() != runtime_qualification.get("sha256")
                or runtime_verification.get("verification_receipt_sha256")
                != runtime_qualification.get("verification_receipt_sha256")
                or runtime_verification.get("verification_receipt_sha256") != _sha(runtime_body)
                or not isinstance(runtime_receipt, dict)
                or runtime_receipt.get("activation_sha256") != _sha(candidate_body)
                or runtime_receipt.get("package_id") != package
                or runtime_receipt.get("promotion_mode") != PROMOTION_MODE
                or runtime_qualification.get("candidate_activation_sha256") != _sha(candidate_body)
                or runtime_verification.get("verified") is not True
                or runtime_verification.get("task_count") != 120
                or runtime_verification.get("exact_count") != 120
                or runtime_verification.get("lesion_disruption_count") != 120
                or runtime_verification.get(
                    "measured_backend_receipt_equivalence_count"
                )
                != 120
                or runtime_qualification.get(
                    "measured_backend_receipt_equivalence_count"
                )
                != 120
                or runtime_verification.get("foreground_integration_count") != 120
                or runtime_qualification.get("foreground_integration_count") != 120
                or runtime_verification.get("service_integration_count") != 120
                or runtime_qualification.get("service_integration_count") != 120
                or runtime_verification.get("unsupported_language_refused") is not True
            ):
                errors.append("runtime_qualification_drift")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append("runtime_qualification_invalid")
    evidence = activation.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence")
    else:
        try:
            result_path = _resolve_evidence_path(root, evidence["result_path"])
            verification_path = _resolve_evidence_path(
                root,
                evidence["verification_path"],
            )
            adjudication_path = _resolve_evidence_path(
                root,
                evidence["adjudication_path"],
            )
            relative_evidence_paths = [
                _relative_evidence_path(root, path)
                for path in (result_path, verification_path, adjudication_path)
            ]
            if package != PACKAGE_ID and evidence_namespace_errors(relative_evidence_paths):
                raise RuntimeError("semantic recovery evidence namespace is invalid")
            (
                result,
                verification,
                adjudication,
                result_raw,
                verification_raw,
                adjudication_raw,
            ) = _verify_resident_evidence(
                repo_root=root,
                result_path=result_path,
                verification_path=verification_path,
                adjudication_path=adjudication_path,
            )
            if (
                hashlib.sha256(result_raw).hexdigest() != evidence.get("result_sha256")
                or result.get("receipt_sha256") != evidence.get("result_receipt_sha256")
                or hashlib.sha256(verification_raw).hexdigest()
                != evidence.get("verification_sha256")
                or verification.get("verification_receipt_sha256")
                != evidence.get("verification_receipt_sha256")
                or hashlib.sha256(adjudication_raw).hexdigest()
                != evidence.get("adjudication_sha256")
                or adjudication.get("adjudication_receipt_sha256")
                != evidence.get("adjudication_receipt_sha256")
                or adjudication.get("verdict") != evidence.get("adjudication_verdict")
                or adjudication.get("claim") != evidence.get("adjudication_claim")
                or adjudication.get("limitations") != evidence.get("adjudication_limitations")
                or activation.get("model_identity") != verification.get("model_identity")
                or activation.get("resident_manifest_identity")
                != verification.get("resident_manifest_identity")
            ):
                errors.append("evidence_drift")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append("evidence_invalid")
    if verify_live_identity:
        try:
            manifest_identity = activation["resident_manifest_identity"]
            current_manifest = _identity_for_manifest(Path(manifest_identity["path"]))
            current_manifest_payload, _current_manifest_raw = _read_bounded_json(
                Path(manifest_identity["path"]),
                maximum_bytes=512 * 1024,
            )
            expected_model = activation["model_identity"]
            selected_model = (
                Path(model_path).expanduser().resolve(strict=True)
                if model_path is not None
                else Path(str(expected_model["path"])).resolve(strict=True)
            )
            if not _manifest_identity_matches_activation(
                expected=manifest_identity,
                current=current_manifest,
                selected_model=selected_model,
            ):
                errors.append("resident_manifest_drift")
            if _identity_for_model(selected_model) != expected_model:
                errors.append("model_identity_drift")
            if (
                recovery_descriptor is not None
                and descriptor_from_manifest(current_manifest_payload).fingerprint()
                != recovery_descriptor.fingerprint()
            ):
                errors.append("descriptor_identity_drift")
            if current_manifest["active_model_path"] != str(selected_model):
                errors.append("active_model_mismatch")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append("live_identity_invalid")
    return sorted(set(errors))


def semantic_neural_serving_status(model_path: str | Path) -> dict[str, Any]:
    """Return active only while code, evidence, manifest, and model still agree."""

    if str(os.getenv("AURA_SEMANTIC_NEURAL_SERVING", "1")).strip().lower() in _FALSE_VALUES:
        return {
            "active": False,
            "reason": "semantic_neural_serving_disabled",
        }
    try:
        activation_path = active_semantic_neural_activation_path()
        activation, _raw = _read_bounded_json(
            activation_path,
            maximum_bytes=512 * 1024,
        )
        root = REPO_ROOT.resolve(strict=True)
        evidence = activation["evidence"]
        resident_identity = activation["resident_manifest_identity"]
        selected_model = Path(model_path).expanduser().resolve(strict=True)
        resident_manifest = Path(resident_identity["path"])
        resident_pointer, _resident_raw = _read_bounded_json(
            resident_manifest,
            maximum_bytes=64 * 1024,
        )
        resident_dependencies = [resident_manifest]
        if isinstance(resident_pointer.get("identity_transition"), dict):
            resident_dependencies.append(
                resident_manifest.with_name("active.json.identity-backup")
            )
        runtime_qualification = activation.get("runtime_qualification")
        runtime_dependencies = (
            (_resolve_evidence_path(root, runtime_qualification["path"]),)
            if isinstance(runtime_qualification, dict)
            else ()
        )
        dependencies = (
            activation_path,
            *tuple(root / relative for relative in ACTIVATION_SOURCE_FILES),
            *tuple(root / relative for relative in INTEGRATION_SOURCE_FILES),
            _resolve_evidence_path(root, evidence["result_path"]),
            _resolve_evidence_path(root, evidence["verification_path"]),
            _resolve_evidence_path(root, evidence["adjudication_path"]),
            *runtime_dependencies,
            *resident_dependencies,
            selected_model / "config.json",
            selected_model / "model.safetensors.index.json",
        )
        signature_items: list[tuple[str, int, int, int, int]] = []
        for path in dependencies:
            resolved = path.expanduser().resolve(strict=True)
            metadata = resolved.stat()
            signature_items.append(
                (
                    str(resolved),
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )
        return deepcopy(
            _cached_semantic_neural_serving_status(
                str(selected_model),
                str(activation_path.expanduser().resolve(strict=True)),
                tuple(signature_items),
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "active": False,
            "reason": f"semantic_neural_activation_unavailable:{type(exc).__name__}",
        }


def semantic_neural_default_serving_status(
    *, authority_key_path: Path | None = None
) -> dict[str, Any]:
    """Resolve serving or a signed model-migration disposition.

    A model-bound activation cannot follow a cortex replacement merely because
    the runtime pointer moved.  During a deliberate migration, preserve the old
    proof as historical evidence and expose the new model's signed recurrence
    quarantine as a first-class state.  This keeps incompatible tissue off the
    live model without misreporting the intentional quarantine as an outage.
    """

    try:
        activation_path = active_semantic_neural_activation_path()
        activation, _raw = _read_bounded_json(
            activation_path,
            maximum_bytes=512 * 1024,
        )
        model_identity = activation.get("model_identity")
        model_path = model_identity.get("path") if isinstance(model_identity, dict) else None
        if not isinstance(model_path, str) or not model_path.strip():
            raise RuntimeError("semantic neural model identity is unavailable")

        from core.brain.llm.model_registry import get_active_cortex_spec

        active = get_active_cortex_spec(
            force_refresh=True,
            authority_key_path=authority_key_path,
        )
        if active is None or not active.exact_identity or not active.promotion_qualified:
            raise RuntimeError("active cortex exact identity is unavailable")
        proven_model = Path(model_path).expanduser().resolve(strict=True)
        if active.model_path == proven_model:
            return semantic_neural_serving_status(proven_model)

        # A quarantine is authoritative only while the historical activation
        # itself is still intact. Otherwise migration could hide source or
        # evidence drift in the proof being preserved.
        historical_errors = semantic_neural_activation_errors(
            activation,
            model_path=proven_model,
            verify_live_identity=False,
        )
        if historical_errors:
            return {
                "active": False,
                "lifecycle": "invalid",
                "reason": "semantic_neural_historical_activation_invalid:"
                + ",".join(historical_errors),
            }

        migration = active.migration_contract()
        components = migration.get("components") if isinstance(migration, dict) else None
        authority = (
            components.get("recurrence_native") if isinstance(components, dict) else None
        )
        if not isinstance(authority, dict):
            raise RuntimeError("recurrence migration authority is unavailable")

        from core.learning.cortex_migration_authority import (
            validate_component_authority,
        )

        validated = validate_component_authority(
            authority,
            component="recurrence_native",
            descriptor_sha256=active.descriptor_sha256,
            authority_key_path=authority_key_path,
        )
        if validated.get("status") != "deferred":
            return {
                "active": False,
                "lifecycle": "invalid",
                "reason": "semantic_neural_qualified_activation_not_materialized",
                "model_descriptor_sha256": active.descriptor_sha256,
            }
        return {
            "active": False,
            "lifecycle": "deferred",
            "reason": "semantic_neural_model_basis_migration_deferred",
            "serving_authorized": False,
            "historical_activation_preserved": True,
            "model_descriptor_sha256": active.descriptor_sha256,
            "authority_sha256": validated.get("authority_sha256"),
        }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "active": False,
            "lifecycle": "invalid",
            "reason": f"semantic_neural_activation_unavailable:{type(exc).__name__}",
        }


@lru_cache(maxsize=8)
def _cached_semantic_neural_serving_status(
    model_path: str,
    activation_path: str,
    _dependency_signature: tuple[tuple[str, int, int, int, int], ...],
) -> dict[str, Any]:
    activation, _raw = _read_bounded_json(
        Path(activation_path),
        maximum_bytes=512 * 1024,
    )
    errors = semantic_neural_activation_errors(
        activation,
        model_path=Path(model_path),
        require_runtime_qualification=(
            str(os.getenv("AURA_SEMANTIC_NEURAL_QUALIFICATION_CANDIDATE", "0")).strip().lower()
            not in {"1", "true", "yes", "on"}
        ),
    )
    if errors:
        return {
            "active": False,
            "reason": "semantic_neural_activation_invalid:" + ",".join(errors),
        }
    public_receipt = {
        key: activation[key]
        for key in (
            "schema",
            "package_id",
            "mode",
            "promotion_mode",
            "allowed_families",
            "allowed_surface_profiles",
            "model_identity",
            "resident_manifest_identity",
            "claim_boundary",
            "activation_sha256",
        )
    }
    return {
        "active": True,
        "reason": "semantic_neural_serving_active",
        "receipt": public_receipt,
    }


__all__ = [
    "ACTIVATION_CLAIM_BOUNDARY",
    "ACTIVE_ACTIVATION_PATH",
    "ACTIVATION_SOURCE_FILES",
    "ALLOWED_FAMILIES",
    "ALLOWED_SURFACE_PROFILES",
    "DEFAULT_ACTIVATION_PATH",
    "EVIDENCE_DOMAINS",
    "INTEGRATION_SOURCE_CONTRACTS",
    "INTEGRATION_SOURCE_FILES",
    "PACKAGE_ID",
    "PROMOTION_MODE",
    "RESIDENT_RESULT_PATH",
    "RESIDENT_ADJUDICATION_PATH",
    "RESIDENT_VERIFICATION_PATH",
    "SEMANTIC_NEURAL_SERVING_MODE",
    "SEMANTIC_NEURAL_SERVING_SCHEMA",
    "active_semantic_neural_activation_path",
    "build_semantic_neural_activation",
    "semantic_neural_activation_errors",
    "semantic_neural_default_serving_status",
    "semantic_neural_serving_status",
]
