#!/usr/bin/env python3
"""Independently verify one completed composed RLC reconciliation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex import frontier_tasks as ft  # noqa: E402
from core.brain.llm.latent_cortex.resource_accounting import (  # noqa: E402
    validate_control_resource_dominance_certificate,
    validate_information_receipt,
    validate_resource_receipt,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools.rlc_complete_system_closed_book import (  # noqa: E402
    _complete_system_evidence,
)
from tools.rlc_reconciliation_evidence import full_stack_evidence  # noqa: E402

SCHEMA = "aura.rlc.reconciliation_independent_verification.v1"
CONTROLLER_SCHEMA = "aura.rlc_reconciliation_controller.v1"
FINGERPRINT_SCHEMA = "aura.rlc.reconciliation_evidence_manifest.v3"
EXPECTED_REQUESTED_ARMS = ("complete_system_recurrent_composed",)
COMPONENT_REQUIRED_ARMS = (
    "vanilla",
    "complete_system_closed_book",
    "complete_system_recurrent_composed",
    "complete_system_recurrent_initial_control",
    "complete_system_recurrent_depth_lesion",
    "complete_system_adaptation_ablation",
    "complete_system_executable_ablation",
)
PILOT_REQUIRED_ARMS = (
    "vanilla",
    "vanilla_equal_compute",
    *COMPONENT_REQUIRED_ARMS[1:],
)
CERTIFICATE_REQUIRED_ARMS = (
    *PILOT_REQUIRED_ARMS,
    "vanilla_resource_dominating",
)
# Compatibility names retained for callers constructing component fixtures.
EXPECTED_REQUIRED_ARMS = COMPONENT_REQUIRED_ARMS
RUNTIME_ARMS = frozenset(COMPONENT_REQUIRED_ARMS[1:])
RECURRENT_ARMS = frozenset(
    {
        "complete_system_recurrent_composed",
        "complete_system_recurrent_initial_control",
        "complete_system_recurrent_depth_lesion",
    }
)
TREATMENT_ARM = "complete_system_recurrent_composed"
MAX_JSON_BYTES = 512 * 1024 * 1024


def _required_arms_for_stage(stage: Any) -> tuple[str, ...]:
    if stage == "component":
        return COMPONENT_REQUIRED_ARMS
    if stage == "pilot":
        return PILOT_REQUIRED_ARMS
    if stage == "certificate":
        return CERTIFICATE_REQUIRED_ARMS
    _fail("campaign_stage_invalid")


class ReconciliationVerificationError(RuntimeError):
    """The completed campaign did not independently verify."""


def _fail(code: str) -> Never:
    raise ReconciliationVerificationError(str(code or "reconciliation_verification_failed"))


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ReconciliationVerificationError(
            "reconciliation_verification_noncanonical_value"
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_bytes(path: Path, *, role: str, maximum: int = MAX_JSON_BYTES) -> bytes:
    try:
        return read_stable_bytes(path.resolve(strict=True), max_bytes=maximum)
    except OSError as exc:
        raise ReconciliationVerificationError(f"{role}_unreadable") from exc


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_bytes(path, role=role))
    except json.JSONDecodeError as exc:
        raise ReconciliationVerificationError(f"{role}_invalid_json") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    return value


def _verify_regular_file(path: Path, *, size: int, digest: str, role: str) -> None:
    """Verify large bound files without materializing model shards in RAM."""

    try:
        metadata = path.stat()
        if not path.is_file() or metadata.st_size != size:
            _fail(f"{role}_file_mismatch")
        observed = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                observed.update(chunk)
    except OSError as exc:
        raise ReconciliationVerificationError(f"{role}_file_unreadable") from exc
    if observed.hexdigest() != digest:
        _fail(f"{role}_file_mismatch")


def _verify_self_hash(value: Mapping[str, Any], field: str, *, role: str) -> str:
    body = dict(value)
    observed = body.pop(field, None)
    if not _is_sha(observed) or observed != _sha(body):
        _fail(f"{role}_commitment_invalid")
    return str(observed)


def _verify_file_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_root: Path | None,
    role: str,
    commitment_field: str,
) -> str:
    root_value = manifest.get("root")
    records = manifest.get("files")
    declared = manifest.get(commitment_field)
    if (
        not isinstance(root_value, str)
        or not isinstance(records, list)
        or not records
        or not _is_sha(declared)
    ):
        _fail(f"{role}_manifest_invalid")
    root = Path(root_value).expanduser().resolve(strict=True)
    if expected_root is not None and root != expected_root.resolve(strict=True):
        _fail(f"{role}_root_mismatch")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            _fail(f"{role}_record_invalid")
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_sha(digest)
            or type(size) is not int
            or size < 0
        ):
            _fail(f"{role}_record_invalid")
        seen.add(relative)
        lexical = root / relative
        if lexical.is_symlink():
            _fail(f"{role}_record_path_invalid")
        path = lexical.resolve(strict=True)
        if root not in path.parents:
            _fail(f"{role}_record_path_invalid")
        _verify_regular_file(path, size=size, digest=str(digest), role=role)
        normalized.append({"path": relative, "sha256": digest, "size": size})
    body = dict(manifest)
    body.pop(commitment_field, None)
    body["root"] = str(root)
    body["files"] = normalized
    if _sha(body) != declared:
        _fail(f"{role}_manifest_commitment_invalid")
    observed_paths = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    if observed_paths != sorted(seen):
        _fail(f"{role}_file_set_drift")
    return str(declared)


def _verify_source_manifest(config: Mapping[str, Any], source_root: Path) -> str:
    path = Path(str(config.get("source_manifest_path") or "")).resolve(strict=True)
    manifest = _read_json(path, role="source_manifest")
    body = dict(manifest)
    declared = body.pop("manifest_sha256", None)
    records = body.get("files")
    if (
        manifest.get("schema") != "aura.rlc_reconciliation_source_manifest.v1"
        or manifest.get("source_commit") != config.get("source_commit")
        or not isinstance(records, list)
        or not records
        or not _is_sha(declared)
        or _sha(body) != declared
    ):
        _fail("source_manifest_invalid")
    root = source_root.resolve(strict=True)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            _fail("source_manifest_record_invalid")
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_sha(digest)
            or type(size) is not int
            or size < 0
        ):
            _fail("source_manifest_record_invalid")
        seen.add(relative)
        lexical = root / relative
        if lexical.is_symlink():
            _fail("source_manifest_path_invalid")
        candidate = lexical.resolve(strict=True)
        if root not in candidate.parents:
            _fail("source_manifest_path_invalid")
        _verify_regular_file(
            candidate,
            size=size,
            digest=str(digest),
            role="source_manifest",
        )
    return str(declared)


def _verify_implementation_files(fingerprint: Mapping[str, Any], source_root: Path) -> str:
    files = fingerprint.get("implementation_files")
    declared = fingerprint.get("implementation_sha256")
    if not isinstance(files, Mapping) or not files or not _is_sha(declared):
        _fail("implementation_manifest_invalid")
    normalized: dict[str, str] = {}
    root = source_root.resolve(strict=True)
    for relative, digest in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_sha(digest)
        ):
            _fail("implementation_record_invalid")
        lexical = root / relative
        if lexical.is_symlink():
            _fail("implementation_path_invalid")
        path = lexical.resolve(strict=True)
        if root not in path.parents:
            _fail("implementation_path_invalid")
        observed = hashlib.sha256(_read_bytes(path, role="implementation_file")).hexdigest()
        if observed != digest:
            _fail("implementation_file_mismatch")
        normalized[relative] = str(digest)
    if _sha(normalized) != declared:
        _fail("implementation_manifest_commitment_invalid")
    return str(declared)


def _verify_controller(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path, role="controller_config")
    if config.get("schema") != CONTROLLER_SCHEMA:
        _fail("controller_schema_invalid")
    _verify_self_hash(config, "config_sha256", role="controller")
    if config.get("arms") != EXPECTED_REQUESTED_ARMS[0]:
        _fail("controller_requested_arm_invalid")
    source_root = Path(str(config.get("source_root") or "")).expanduser().resolve(strict=True)
    source_identity = config.get("source_git_identity")
    if not isinstance(source_identity, Mapping):
        _fail("controller_source_identity_invalid")
    _verify_self_hash(source_identity, "identity_sha256", role="source_identity")
    if (
        Path(str(source_identity.get("source_root") or "")).resolve(strict=True) != source_root
        or source_identity.get("source_commit") != config.get("source_commit")
        or source_identity.get("workspace_status_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        _fail("controller_source_identity_mismatch")
    try:
        observed_commit = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        observed_branch = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(source_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        observed_status = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReconciliationVerificationError("source_git_probe_failed") from exc
    if (
        observed_commit != config.get("source_commit")
        or observed_branch != "HEAD"
        or observed_status
    ):
        _fail("source_git_identity_drift")
    program = Path(str(config.get("controller_program") or "")).resolve(strict=True)
    if source_root not in program.parents:
        _fail("controller_program_outside_source")
    if hashlib.sha256(_read_bytes(program, role="controller_program")).hexdigest() != config.get(
        "controller_program_sha256"
    ):
        _fail("controller_program_mismatch")
    model_root = Path(str(config.get("model") or "")).expanduser().resolve(strict=True)
    model_manifest_sha = _verify_file_manifest(
        config.get("model_manifest") or {},
        expected_root=model_root,
        role="model",
        commitment_field="manifest_sha256",
    )
    package = config.get("integrated_recurrent_package")
    if not isinstance(package, Mapping):
        _fail("controller_package_invalid")
    package_root = Path(str(package.get("root") or "")).expanduser().resolve(strict=True)
    package_files_sha = _verify_file_manifest(
        package,
        expected_root=package_root,
        role="package",
        commitment_field="input_sha256",
    )
    return {
        "config": config,
        "source_root": source_root,
        "source_manifest_sha256": _verify_source_manifest(config, source_root),
        "model_manifest_sha256": model_manifest_sha,
        "package_files_sha256": package_files_sha,
    }


def _regenerate_tasks(
    commitment: Mapping[str, Any], fingerprint: Mapping[str, Any]
) -> tuple[tuple[Any, ...], str]:
    required_commitment_fields = {
        "seed": int,
        "per_domain": int,
        "difficulty": int,
        "registry_version": str,
        "domains": list,
        "task_count": int,
        "commitment_sha256": str,
    }
    for field, expected_type in required_commitment_fields.items():
        value = commitment.get(field)
        if expected_type is int:
            if type(value) is not int:
                _fail("task_commitment_shape_invalid")
        elif not isinstance(value, expected_type):
            _fail("task_commitment_shape_invalid")
    domains = tuple(commitment["domains"])
    if (
        not domains
        or len(set(domains)) != len(domains)
        or any(not isinstance(domain, str) or domain not in ft.FRONTIER_DOMAINS for domain in domains)
    ):
        _fail("task_commitment_domains_invalid")
    seeds = [commitment["seed"] + offset for offset in range(commitment["per_domain"])]
    tasks = tuple(
        ft.generate_task_battery(
            seeds,
            domains=domains,
            difficulty=commitment["difficulty"],
            registry_version=commitment["registry_version"],
        )
    )
    rebuilt = ft.build_task_commitment(ft.build_task_manifest(tasks))
    expected_ids = [task.task_id for task in tasks]
    if (
        len(tasks) != commitment["task_count"]
        or rebuilt.commitment_sha256 != commitment["commitment_sha256"]
        or fingerprint.get("task_commitment_sha256") != rebuilt.commitment_sha256
        or fingerprint.get("expected_task_ids") != expected_ids
        or fingerprint.get("domains") != list(domains)
        or fingerprint.get("difficulty") != commitment["difficulty"]
        or fingerprint.get("task_registry_version") != commitment["registry_version"]
    ):
        _fail("task_commitment_reconstruction_mismatch")
    return tasks, rebuilt.commitment_sha256


def _task_decode_max_tokens(task: Any, requested: int) -> int:
    domain = str(getattr(task, "domain", "") or "").strip().lower()
    prompt = str(getattr(getattr(task, "public", None), "prompt", "") or "").lower()
    floor = 512
    if domain == "coding" or any(
        phrase in prompt
        for phrase in ("return the corrected function", "complete program", "implementation")
    ):
        floor = 768
    elif domain == "long_horizon_planning" or any(
        phrase in prompt for phrase in ("complete plan", "schedule", "prerequisite", "deadline")
    ):
        floor = 640
    return max(requested, floor)


def _decode_fingerprint(
    *,
    config: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    arm: str,
    max_tokens: int,
    implementation_sha256: str,
) -> str:
    fast_weight_site = fingerprint.get("fast_weight_site")
    if not isinstance(fast_weight_site, Mapping):
        _fail("fast_weight_site_invalid")
    package = config.get("integrated_recurrent_package") or {}
    package_sha = str(package.get("manifest_sha256") or "") if arm in RECURRENT_ARMS else ""
    recurrent_budget = int(config["integrated_recurrent_max_tokens"]) if arm in RECURRENT_ARMS else 2048
    body = {
        "adapter": str(config.get("adapter") or ""),
        "arm": arm,
        "contract": "rlc_reconciliation_decode.v4",
        "completion_budget_policy": "semantic_completion_floor.v1",
        "campaign_stage": str(config["campaign_stage"]),
        "difficulty": int(config["difficulty"]),
        "episode_wall_s": float(config["episode_wall_s"]),
        "fast_weight_layer_placement": str(fast_weight_site.get("layer_placement") or ""),
        "fast_weight_rank": int(fast_weight_site.get("rank", 2)),
        "fast_weight_key_source": str(
            fast_weight_site.get("key_source", "live_query")
        ),
        "fast_weight_target": str(fast_weight_site.get("target") or ""),
        "output_memory_diagnostic": bool(fingerprint.get("output_memory_diagnostic")),
        "implementation_sha256": implementation_sha256,
        "integrated_recurrent_package_sha256": package_sha,
        "integrated_recurrent_max_tokens": recurrent_budget,
        "max_tokens": int(max_tokens),
        "model": str(config["model"]),
        "n_slots": int(config["n_slots"]),
        "per_domain": int(config["per_domain"]),
        "seed": int(config["seed"]),
        "task_registry_version": str(config["task_registry_version"]),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _verify_decode_contract(
    *,
    config: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    tasks: Sequence[Any],
    implementation_sha256: str,
    required_arms: Sequence[str],
) -> Mapping[str, str]:
    if (
        config.get("seed") != fingerprint.get("seed", config.get("seed"))
        or config.get("per_domain") != fingerprint.get("per_domain", config.get("per_domain"))
        or config.get("difficulty") != fingerprint.get("difficulty")
        or config.get("domains") != fingerprint.get("domains")
        or config.get("task_registry_version") != fingerprint.get("task_registry_version")
        or fingerprint.get("completion_budget_policy") != "semantic_completion_floor.v1"
    ):
        _fail("controller_task_contract_mismatch")
    package = config.get("integrated_recurrent_package")
    recorded_package = fingerprint.get("integrated_recurrent_package")
    if not isinstance(package, Mapping) or not isinstance(recorded_package, Mapping):
        _fail("recurrent_package_contract_invalid")
    expected_package = {
        "package_id": package.get("package_id"),
        "manifest_sha256": package.get("manifest_sha256"),
        "controller_sha256": package.get("controller_sha256"),
        "activation_sha256": package.get("activation_sha256"),
    }
    if (
        recorded_package != expected_package
        or fingerprint.get("integrated_recurrent_max_tokens")
        != config.get("integrated_recurrent_max_tokens")
    ):
        _fail("recurrent_package_contract_mismatch")
    arm_tokens = fingerprint.get("arm_max_tokens")
    task_tokens = fingerprint.get("task_max_tokens")
    fingerprints = fingerprint.get("decode_fingerprint")
    if (
        not isinstance(arm_tokens, Mapping)
        or not isinstance(task_tokens, Mapping)
        or not isinstance(fingerprints, Mapping)
        or set(arm_tokens) != set(required_arms)
        or set(task_tokens) != set(required_arms)
        or set(fingerprints) != set(required_arms)
    ):
        _fail("decode_fingerprint_matrix_invalid")
    expected_task_ids = {task.task_id for task in tasks}
    for arm in required_arms:
        if type(arm_tokens[arm]) is not int or arm_tokens[arm] != config["max_tokens"]:
            _fail("arm_token_contract_mismatch")
        expected_task_tokens = {
            task.task_id: _task_decode_max_tokens(task, int(arm_tokens[arm])) for task in tasks
        }
        if task_tokens[arm] != expected_task_tokens or set(task_tokens[arm]) != expected_task_ids:
            _fail("task_token_contract_mismatch")
        expected_fingerprint = _decode_fingerprint(
            config=config,
            fingerprint=fingerprint,
            arm=arm,
            max_tokens=int(arm_tokens[arm]),
            implementation_sha256=implementation_sha256,
        )
        if fingerprints[arm] != expected_fingerprint:
            _fail("decode_fingerprint_reconstruction_mismatch")
    return fingerprints


def _read_journal(path: Path) -> list[dict[str, Any]]:
    payload = _read_bytes(path, role="journal")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReconciliationVerificationError(
                f"journal_line_{line_number}_invalid"
            ) from exc
        if not isinstance(value, dict) or value.get("event") != "CELL":
            _fail("journal_record_invalid")
        records.append(value)
    if not records:
        _fail("journal_empty")
    return records


def _reconstruct_runtime_evidence(campaign_dir: Path, cell: Mapping[str, Any]) -> str:
    relative = cell.get("runtime_receipt_path")
    declared = cell.get("runtime_receipt_sha256")
    if not isinstance(relative, str) or not relative or not _is_sha(declared):
        _fail("runtime_receipt_binding_invalid")
    receipt_root = (campaign_dir / "runtime_receipts").resolve(strict=True)
    lexical = campaign_dir / relative
    if lexical.is_symlink():
        _fail("runtime_receipt_path_invalid")
    receipt_path = lexical.resolve(strict=True)
    if receipt_root not in receipt_path.parents:
        _fail("runtime_receipt_path_invalid")
    receipt = _read_json(receipt_path, role="runtime_receipt")
    if _sha(receipt) != declared:
        _fail("runtime_receipt_digest_mismatch")
    adaptive = cell.get("arm_profile") != "complete_closed_book_adaptation_ablation"
    engine = full_stack_evidence(receipt, adaptive_neural_expected=adaptive)
    complete = _complete_system_evidence(receipt, engine_evidence=engine)
    if cell.get("full_stack_evidence") != engine or cell.get("complete_system_evidence") != complete:
        _fail("runtime_receipt_summary_mismatch")
    if not engine.get("valid") or not complete.get("valid"):
        _fail("runtime_mechanism_invalid")
    text_sha = hashlib.sha256(str(cell.get("text") or "").encode("utf-8")).hexdigest()
    if complete.get("final_text_sha256") != text_sha:
        _fail("runtime_final_text_mismatch")
    return str(declared)


def _paired(
    scored: Mapping[str, Mapping[str, bool]],
    *,
    control: str,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    treatment = scored[TREATMENT_ARM]
    baseline = scored[control]
    lifts = [task_id for task_id in task_ids if treatment[task_id] and not baseline[task_id]]
    regressions = [
        task_id for task_id in task_ids if baseline[task_id] and not treatment[task_id]
    ]
    discordant = len(lifts) + len(regressions)
    p_value = (
        sum(math.comb(discordant, index) for index in range(len(lifts), discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "treatment_arm": TREATMENT_ARM,
        "control_arm": control,
        "paired_cells": len(task_ids),
        "missing_task_ids": [],
        "lifts": len(lifts),
        "lift_task_ids": lifts,
        "regressions": len(regressions),
        "regression_task_ids": regressions,
        "discordant_pairs": discordant,
        "one_sided_exact_sign_p": round(p_value, 12),
    }


def _independent_adjudication(
    scored: Mapping[str, Mapping[str, bool]],
    task_ids: Sequence[str],
    *,
    resource_target_control_proven: bool = False,
) -> dict[str, Any]:
    comparisons = {
        "learned_parameters": _paired(
            scored, control="complete_system_recurrent_initial_control", task_ids=task_ids
        ),
        "marginal_composition": _paired(
            scored, control="complete_system_closed_book", task_ids=task_ids
        ),
        "recurrent_depth": _paired(
            scored, control="complete_system_recurrent_depth_lesion", task_ids=task_ids
        ),
        "vanilla_floor": _paired(scored, control="vanilla", task_ids=task_ids),
    }
    learned = comparisons["learned_parameters"]
    marginal = comparisons["marginal_composition"]
    depth = comparisons["recurrent_depth"]
    floor = comparisons["vanilla_floor"]
    bounded = bool(
        learned["lifts"] > 0
        and learned["regressions"] == 0
        and marginal["lifts"] > 0
        and marginal["regressions"] == 0
        and floor["regressions"] == 0
    )
    depth_positive = bool(depth["lifts"] > 0 and depth["regressions"] == 0)
    powered = bool(
        bounded
        and depth_positive
        and resource_target_control_proven
        and all(
            row["one_sided_exact_sign_p"] <= 0.05
            for row in (learned, marginal, depth)
        )
    )
    if not bounded:
        decision = "no_bounded_learned_tissue_gain"
    elif not depth_positive:
        decision = "learned_tissue_gain_without_depth_causality"
    elif powered:
        decision = "powered_causal_result_requires_independent_replication"
    else:
        decision = "bounded_causal_canary_positive_replication_required"
    return {
        "schema": "aura.rlc.composed_recurrent_adjudication.v1",
        "authority": "independent_component_verification_only",
        "measured": True,
        "comparisons": comparisons,
        "bounded_learned_tissue_positive": bounded,
        "recurrent_depth_positive": depth_positive,
        "resource_target_control_proven": resource_target_control_proven,
        "powered_causal_result": powered,
        "independent_replication_required": True,
        "wow_signal_authorized": False,
        "fusion_authorized": False,
        "decision": decision,
    }


def _verify_resource_target_controls(
    unique: Mapping[tuple[str, str], Mapping[str, Any]],
    task_ids: Sequence[str],
) -> bool:
    for task_id in task_ids:
        treatment = unique[(TREATMENT_ARM, task_id)]
        control = unique[("vanilla_resource_dominating", task_id)]
        try:
            treatment_resource = validate_resource_receipt(
                treatment.get("resource_accounting")
            )
            treatment_information = validate_information_receipt(
                treatment.get("information_accounting")
            )
            control_resource = validate_resource_receipt(control.get("resource_accounting"))
            control_information = validate_information_receipt(
                control.get("information_accounting")
            )
            certificate = validate_control_resource_dominance_certificate(
                control.get("resource_dominance_certificate")
            )
        except (TypeError, ValueError) as exc:
            raise ReconciliationVerificationError(
                "resource_dominance_evidence_invalid"
            ) from exc
        bindings = {
            "treatment_resource_sha256": treatment_resource["receipt_sha256"],
            "control_resource_sha256": control_resource["receipt_sha256"],
            "treatment_information_sha256": treatment_information["receipt_sha256"],
            "control_information_sha256": control_information["receipt_sha256"],
        }
        if certificate.get("admitted") is not True or any(
            certificate.get(name) != digest for name, digest in bindings.items()
        ):
            _fail("resource_dominance_evidence_invalid")
    return True


def verify(*, config_path: Path, campaign_dir: Path) -> dict[str, Any]:
    custody = _verify_controller(config_path)
    config = custody["config"]
    required_arms = _required_arms_for_stage(config.get("campaign_stage"))
    expected_campaign_dir = Path(str(config.get("out_dir") or "")).resolve(strict=True)
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    if campaign_dir != expected_campaign_dir:
        _fail("campaign_directory_mismatch")
    fingerprint = _read_json(campaign_dir / "decode_fingerprint.json", role="fingerprint")
    if (
        fingerprint.get("schema") != FINGERPRINT_SCHEMA
        or tuple(fingerprint.get("requested_arms") or ()) != EXPECTED_REQUESTED_ARMS
        or tuple(fingerprint.get("required_arms") or ()) != required_arms
        or fingerprint.get("campaign_stage") != config.get("campaign_stage")
        or fingerprint.get("resource_dominating_target_arm") != TREATMENT_ARM
    ):
        _fail("fingerprint_contract_invalid")
    implementation_sha = _verify_implementation_files(fingerprint, custody["source_root"])
    commitment = _read_json(campaign_dir / "task_commitment.json", role="task_commitment")
    tasks, commitment_sha = _regenerate_tasks(commitment, fingerprint)
    task_by_id = {task.task_id: task for task in tasks}
    task_ids = [task.task_id for task in tasks]
    if (
        commitment.get("schema") != "aura.rlc_reconciliation_sweep.v1"
        or commitment.get("seed") != config.get("seed")
        or commitment.get("per_domain") != config.get("per_domain")
        or commitment.get("difficulty") != config.get("difficulty")
        or commitment.get("domains") != config.get("domains")
        or commitment.get("registry_version") != config.get("task_registry_version")
    ):
        _fail("controller_task_commitment_mismatch")
    fingerprints = _verify_decode_contract(
        config=config,
        fingerprint=fingerprint,
        tasks=tasks,
        implementation_sha256=implementation_sha,
        required_arms=required_arms,
    )
    records = _read_journal(campaign_dir / "journal.jsonl")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    runtime_receipt_shas: list[str] = []
    for cell in records:
        arm = str(cell.get("arm") or "")
        task_id = str(cell.get("task_id") or "")
        key = (arm, task_id)
        if arm not in required_arms or task_id not in task_by_id:
            _fail("journal_cell_outside_matrix")
        if key in unique:
            _fail("journal_duplicate_cell")
        if cell.get("decode_fingerprint") != fingerprints[arm]:
            _fail("journal_decode_fingerprint_mismatch")
        if cell.get("domain") != task_by_id[task_id].domain or cell.get("error"):
            _fail("journal_cell_invalid")
        if arm in RUNTIME_ARMS:
            runtime_receipt_shas.append(_reconstruct_runtime_evidence(campaign_dir, cell))
        unique[key] = cell
    expected_keys = {(arm, task_id) for task_id in task_ids for arm in required_arms}
    if set(unique) != expected_keys:
        _fail("campaign_incomplete")
    scored: dict[str, dict[str, bool]] = {arm: {} for arm in required_arms}
    score_reasons: dict[str, dict[str, str]] = {arm: {} for arm in required_arms}
    for (arm, task_id), cell in unique.items():
        result = ft.score_task(task_by_id[task_id], str(cell.get("text") or ""))
        scored[arm][task_id] = bool(result.correct)
        score_reasons[arm][task_id] = str(result.reason or "correct")
    resource_target_control_proven = (
        _verify_resource_target_controls(unique, task_ids)
        if config.get("campaign_stage") == "certificate"
        else False
    )
    adjudication = _independent_adjudication(
        scored,
        task_ids,
        resource_target_control_proven=resource_target_control_proven,
    )
    frozen_verdict = _read_json(campaign_dir / "verdict.json", role="frozen_verdict")
    if (
        frozen_verdict.get("primary_claim_target") != "composed_recurrent_tissue"
        or frozen_verdict.get("decision") != adjudication["decision"]
        or frozen_verdict.get("composed_recurrent_adjudication", {}).get("comparisons")
        != adjudication["comparisons"]
        or frozen_verdict.get("composed_recurrent_adjudication", {}).get(
            "resource_target_control_proven"
        )
        is not adjudication["resource_target_control_proven"]
        or frozen_verdict.get("composed_recurrent_adjudication", {}).get(
            "powered_causal_result"
        )
        is not adjudication["powered_causal_result"]
        or frozen_verdict.get("claims", {}).get("reasoning_gain_proven") is not False
        or frozen_verdict.get("claims", {}).get("fusion_authorized") is not False
        or frozen_verdict.get("claims", {}).get("frontier_level_proven") is not False
    ):
        _fail("frozen_adjudication_mismatch")
    material = {
        "schema": SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": f"resident_32b_composed_{config['campaign_stage']}_only",
        "verified": True,
        "source_commit": config["source_commit"],
        "source_manifest_sha256": custody["source_manifest_sha256"],
        "controller_config_sha256": config["config_sha256"],
        "model_manifest_sha256": custody["model_manifest_sha256"],
        "package_files_sha256": custody["package_files_sha256"],
        "implementation_sha256": implementation_sha,
        "task_commitment_sha256": commitment_sha,
        "task_count": len(tasks),
        "required_arms": list(required_arms),
        "cell_count": len(unique),
        "journal_sha256": hashlib.sha256(
            _read_bytes(campaign_dir / "journal.jsonl", role="journal")
        ).hexdigest(),
        "runtime_receipts_sha256": _sha(sorted(runtime_receipt_shas)),
        "scores": {
            arm: sum(scored[arm].values()) for arm in required_arms
        },
        "score_reasons": score_reasons,
        "adjudication": adjudication,
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "fusion_authorized": False,
        "wow_signal_authorized": False,
        "required_next_gate": (
            "fresh_powered_preregistered_replication"
            if adjudication["bounded_learned_tissue_positive"]
            and adjudication["recurrent_depth_positive"]
            else "mechanism_repair_before_replication"
        ),
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return {**material, "verification_sha256": _sha(material)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = verify(
            config_path=Path(args.config),
            campaign_dir=Path(args.campaign_dir),
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ReconciliationVerificationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
