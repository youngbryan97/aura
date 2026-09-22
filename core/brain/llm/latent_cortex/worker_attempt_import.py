"""Verify and transactionally import one supervisor-certified worker attempt."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignJournalError,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    VerifiedCampaignTrustPolicy,
    prepare_role_signature_request,
)
from core.brain.llm.latent_cortex.worker_origin import (
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    ZERO_SHA256,
    WorkerOriginError,
    compute_allowed_cell_digest,
    validate_worker_authorization_payload,
    verify_worker_authorization,
    verify_worker_lifecycle_event_origin,
    verify_worker_result_origin,
)
from core.runtime.detached_subprocess_broker import BrokeredProcessResult

PAIRED_CAMPAIGN_CELL_TYPE = "paired_campaign_cell"
VERIFIED_STAGE_SCHEMA = "aura.latent_cortex.verified_worker_attempt_stage.v1"
IMPORT_INTENT_SCHEMA = "aura.latent_cortex.worker_stage_import_intent.v1"
IMPORT_RECEIPT_SCHEMA = "aura.latent_cortex.worker_stage_import_receipt.v1"
LIFECYCLE_ARTIFACT_SCHEMA = "aura.detached_step.worker_origin_lifecycle_artifact.v1"

_MAX_LIFECYCLE_BYTES = 4 * 1024 * 1024
_MAX_STAGE_BYTES = 1024 * 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class WorkerAttemptImportError(ValueError):
    """Stable fail-closed error for staged worker evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise WorkerAttemptImportError(code)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(raw: bytes, *, role: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{role}_not_utf8")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_json_key")
            result[key] = value
        return result

    def parse_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            _fail(f"{role}_non_finite_number")
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_float=parse_float,
            parse_constant=lambda _value: _fail(f"{role}_non_finite_number"),
        )
    except WorkerAttemptImportError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    return value


def _read_private_bytes(path: Path, *, maximum: int, role: str) -> bytes:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
        )
    except OSError:
        _fail(f"{role}_unavailable")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o077
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            _fail(f"{role}_storage_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"{role}_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(f"{role}_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_private_json(path: Path, *, maximum: int, role: str) -> dict[str, Any]:
    raw = _read_private_bytes(path, maximum=maximum, role=role)
    value = _strict_json(raw, role=role)
    if raw != canonical_json_bytes(value) + b"\n":
        _fail(f"{role}_noncanonical")
    return value


def _atomic_create_or_verify(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_stat = path.parent.stat()
    if (
        path.parent.is_symlink()
        or parent_stat.st_uid != os.geteuid()
        or parent_stat.st_mode & 0o077
    ):
        _fail("import_artifact_directory_invalid")
    if path.exists() or path.is_symlink():
        if (
            _read_private_bytes(
                path,
                maximum=max(1, len(payload)),
                role="import_artifact",
            )
            != payload
        ):
            _fail("import_artifact_conflict")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("import_artifact_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if (
            _read_private_bytes(
                path,
                maximum=max(1, len(payload)),
                role="import_artifact",
            )
            != payload
        ):
            _fail("import_artifact_conflict")
    finally:
        temporary.unlink(missing_ok=True)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


@dataclass(frozen=True)
class VerifiedWorkerAttemptStage:
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]


def _expected_arm_cells(
    plan: CampaignPlan,
    arm: str,
) -> tuple[dict[str, str], ...]:
    cells = tuple(
        {
            "cell_id": cell_id,
            "cell_type": PAIRED_CAMPAIGN_CELL_TYPE,
        }
        for cell_id in plan.cell_ids
        if plan.cell_definition(cell_id).get("arm") == arm
    )
    if not cells:
        _fail("worker_stage_arm_empty")
    return cells


def verify_terminal_worker_stage(
    *,
    stage_path: Path,
    lifecycle_path: Path,
    plan: CampaignPlan,
    policy: VerifiedCampaignTrustPolicy,
    broker_result: BrokeredProcessResult,
    arm: str,
    worker_attempt_slot: int,
    expected_protocol_sha256: str,
    expected_detached_plan_sha256: str,
    expected_broker_policy_sha256: str,
    expected_model_identity_sha256: str,
    expected_adapter_identity_sha256: str,
) -> VerifiedWorkerAttemptStage:
    if (
        not isinstance(worker_attempt_slot, int)
        or isinstance(worker_attempt_slot, bool)
        or worker_attempt_slot <= 0
    ):
        _fail("worker_stage_attempt_slot_invalid")
    for value, role in (
        (expected_protocol_sha256, "protocol"),
        (expected_detached_plan_sha256, "detached_plan"),
        (expected_broker_policy_sha256, "broker_policy"),
        (expected_model_identity_sha256, "model_identity"),
        (expected_adapter_identity_sha256, "adapter_identity"),
    ):
        if not _is_sha256(value):
            _fail(f"worker_stage_{role}_invalid")
    if (
        broker_result.returncode != 0
        or broker_result.status != "passed"
        or broker_result.timed_out
        or not broker_result.containment_verified
        or broker_result.policy_sha256 != expected_broker_policy_sha256
        or broker_result.worker_origin_lifecycle is None
    ):
        _fail("worker_stage_broker_not_terminal")

    expected_cells = _expected_arm_cells(plan, arm)
    stage_raw = _read_private_bytes(
        stage_path,
        maximum=_MAX_STAGE_BYTES,
        role="worker_stage_journal",
    )
    try:
        with CampaignJournal(stage_path, plan) as stage:
            snapshot = stage.resume()
            records = stage.committed_records()
    except CampaignJournalError as exc:
        raise WorkerAttemptImportError("worker_stage_journal_invalid") from exc
    if (
        _read_private_bytes(
            stage_path,
            maximum=_MAX_STAGE_BYTES,
            role="worker_stage_journal",
        )
        != stage_raw
    ):
        _fail("worker_stage_journal_changed_during_verification")
    expected_cell_ids = tuple(cell["cell_id"] for cell in expected_cells)
    if (
        snapshot.incomplete_cell_ids
        or snapshot.committed_cell_ids != expected_cell_ids
        or tuple(record["cell_id"] for record in records) != expected_cell_ids
    ):
        _fail("worker_stage_not_complete")

    lifecycle = _read_private_json(
        lifecycle_path,
        maximum=_MAX_LIFECYCLE_BYTES,
        role="worker_stage_lifecycle",
    )
    lifecycle_keys = {
        "schema",
        "broker_policy_sha256",
        "authorization_payload",
        "authorization_request",
        "authorization_attestation",
        "event_origin",
        "completion_error",
        "artifact_sha256",
    }
    lifecycle_body = {key: value for key, value in lifecycle.items() if key != "artifact_sha256"}
    if (
        set(lifecycle) != lifecycle_keys
        or lifecycle.get("schema") != LIFECYCLE_ARTIFACT_SCHEMA
        or lifecycle.get("broker_policy_sha256") != expected_broker_policy_sha256
        or lifecycle.get("artifact_sha256") != _sha256(lifecycle_body)
        or lifecycle.get("completion_error") is not None
    ):
        _fail("worker_stage_lifecycle_invalid")
    authorization = lifecycle.get("authorization_payload")
    request = lifecycle.get("authorization_request")
    attestation = lifecycle.get("authorization_attestation")
    event_origin = lifecycle.get("event_origin")
    if not all(
        isinstance(value, dict) for value in (authorization, request, attestation, event_origin)
    ):
        _fail("worker_stage_lifecycle_evidence_invalid")
    try:
        authorization = validate_worker_authorization_payload(authorization)
        allowed_cell_digest = compute_allowed_cell_digest(expected_cells)
    except WorkerOriginError as exc:
        raise WorkerAttemptImportError("worker_stage_authorization_invalid") from exc
    expected_authorization = {
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "detached_plan_sha256": expected_detached_plan_sha256,
        "broker_policy_sha256": expected_broker_policy_sha256,
        "supervisor_attempt": authorization.get("supervisor_attempt"),
        "arm": arm,
        "worker_attempt_slot": worker_attempt_slot,
        "allowed_cell_digest": allowed_cell_digest,
        "model_identity_sha256": expected_model_identity_sha256,
        "adapter_identity_sha256": expected_adapter_identity_sha256,
        "worker_key_custody": WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    }
    if any(authorization.get(key) != value for key, value in expected_authorization.items()):
        _fail("worker_stage_authorization_binding_invalid")
    signed_request = request.get("signed_payload")
    signed_at_unix = (
        signed_request.get("signed_at_unix") if isinstance(signed_request, dict) else None
    )
    if isinstance(signed_at_unix, bool) or not isinstance(signed_at_unix, int):
        _fail("worker_stage_authorization_time_invalid")
    try:
        expected_request = prepare_role_signature_request(
            policy,
            role=CAMPAIGN_RUNNER,
            payload=authorization,
            signed_at_unix=signed_at_unix,
        )
        if request != expected_request:
            _fail("worker_stage_authorization_request_invalid")
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=authorization,
        )
    except WorkerAttemptImportError:
        raise
    except (ValueError, WorkerOriginError) as exc:
        raise WorkerAttemptImportError("worker_stage_authorization_attestation_invalid") from exc

    previous_origin_sha256 = ZERO_SHA256
    result_origin_sha256: list[str] = []
    for sequence, (record, expected_cell) in enumerate(
        zip(records, expected_cells, strict=True),
        start=1,
    ):
        try:
            verify_worker_result_origin(
                policy,
                authorization_attestation=attestation,
                expected_authorization_payload=authorization,
                result=record["result"],
                expected_cell_id=expected_cell["cell_id"],
                expected_cell_type=expected_cell["cell_type"],
                expected_attempt_id=record["attempt_id"],
                expected_sequence=sequence,
                expected_previous_origin_sha256=previous_origin_sha256,
            )
        except WorkerOriginError as exc:
            raise WorkerAttemptImportError("worker_stage_result_origin_invalid") from exc
        origin_sha256 = record["result"]["worker_origin"]["origin_sha256"]
        result_origin_sha256.append(origin_sha256)
        previous_origin_sha256 = origin_sha256

    signed_lifecycle = event_origin.get("signed_payload")
    if not isinstance(signed_lifecycle, dict):
        _fail("worker_stage_lifecycle_payload_invalid")
    occurred_at_unix = signed_lifecycle.get("occurred_at_unix")
    try:
        verify_worker_lifecycle_event_origin(
            policy=policy,
            authorization_payload=authorization,
            authorization_attestation=attestation,
            event_origin=event_origin,
            expected_event_type="terminal",
            expected_prior_state="running",
            expected_result_count=len(expected_cells),
            expected_previous_origin_sha256=previous_origin_sha256,
            expected_completed_cell_ids=list(expected_cell_ids),
            expected_occurred_at_unix=occurred_at_unix,
            expected_return_code=0,
            expected_reason=None,
        )
    except WorkerOriginError as exc:
        raise WorkerAttemptImportError("worker_stage_terminal_lifecycle_invalid") from exc
    expected_lifecycle_path = lifecycle_path.resolve(strict=True)
    expected_summary = {
        "artifact_path": str(expected_lifecycle_path),
        "artifact_sha256": lifecycle["artifact_sha256"],
        "event_type": "terminal",
        "event_sha256": event_origin["event_sha256"],
        "result_count": len(expected_cells),
        "session_id": authorization["session_id"],
    }
    observed_summary = dict(broker_result.worker_origin_lifecycle)
    observed_path = observed_summary.get("artifact_path")
    try:
        observed_resolved = (
            Path(observed_path).resolve(strict=True)
            if isinstance(observed_path, str)
            else None
        )
    # not a failure: a path that does not resolve on disk is not an observed one.
    except OSError:
        observed_resolved = None
    if observed_resolved != expected_lifecycle_path:
        _fail("worker_stage_broker_lifecycle_path_invalid")
    observed_summary["artifact_path"] = str(expected_lifecycle_path)
    if observed_summary != expected_summary:
        _fail("worker_stage_broker_lifecycle_summary_invalid")

    manifest_body = {
        "schema": VERIFIED_STAGE_SCHEMA,
        "campaign_name": plan.campaign_name,
        "campaign_plan_sha256": plan.plan_sha256,
        "arm": arm,
        "worker_attempt_slot": worker_attempt_slot,
        "stage_path": str(stage_path.resolve(strict=True)),
        "stage_sha256": _sha256_bytes(stage_raw),
        "stage_journal_head_sha256": snapshot.journal_head_sha256,
        "cell_ids": list(expected_cell_ids),
        "result_origin_sha256": result_origin_sha256,
        "result_chain_head_sha256": previous_origin_sha256,
        "authorization_payload_sha256": _sha256(authorization),
        "authorization_attestation_sha256": _sha256(attestation),
        "detached_plan_sha256": expected_detached_plan_sha256,
        "broker_policy_sha256": expected_broker_policy_sha256,
        "broker_request_id": broker_result.request_id,
        "broker_receipt_sha256": broker_result.receipt_sha256,
        "broker_response_hmac_sha256": broker_result.response_hmac_sha256,
        "lifecycle_path": str(expected_lifecycle_path),
        "lifecycle_artifact_sha256": lifecycle["artifact_sha256"],
        "lifecycle_event_sha256": event_origin["event_sha256"],
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": _sha256(manifest_body),
    }
    return VerifiedWorkerAttemptStage(
        manifest=manifest,
        records=tuple(dict(record) for record in records),
    )


def import_verified_worker_stage(
    *,
    canonical_journal_path: Path,
    intent_path: Path,
    receipt_path: Path,
    plan: CampaignPlan,
    verified_stage: VerifiedWorkerAttemptStage,
) -> dict[str, Any]:
    manifest = verified_stage.manifest
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema") != VERIFIED_STAGE_SCHEMA
        or manifest.get("campaign_plan_sha256") != plan.plan_sha256
        or manifest.get("manifest_sha256") != _sha256(manifest_body)
    ):
        _fail("verified_worker_stage_manifest_invalid")
    with CampaignJournal(canonical_journal_path, plan) as canonical:
        existing_intent: dict[str, Any] | None = None
        if intent_path.exists() or intent_path.is_symlink():
            existing_intent = _read_private_json(
                intent_path,
                maximum=_MAX_LIFECYCLE_BYTES,
                role="worker_stage_import_intent",
            )
        if existing_intent is None:
            intent_body = {
                "schema": IMPORT_INTENT_SCHEMA,
                "campaign_plan_sha256": plan.plan_sha256,
                "arm": manifest["arm"],
                "worker_attempt_slot": manifest["worker_attempt_slot"],
                "stage_manifest_sha256": manifest["manifest_sha256"],
                "prior_canonical_head_sha256": canonical.resume().journal_head_sha256,
                "cell_ids": manifest["cell_ids"],
            }
            intent = {**intent_body, "intent_sha256": _sha256(intent_body)}
            _atomic_create_or_verify(intent_path, intent)
        else:
            intent = existing_intent
            intent_body = {key: value for key, value in intent.items() if key != "intent_sha256"}
            if (
                intent.get("schema") != IMPORT_INTENT_SCHEMA
                or intent.get("campaign_plan_sha256") != plan.plan_sha256
                or intent.get("arm") != manifest["arm"]
                or intent.get("worker_attempt_slot") != manifest["worker_attempt_slot"]
                or intent.get("stage_manifest_sha256") != manifest["manifest_sha256"]
                or intent.get("cell_ids") != manifest["cell_ids"]
                or intent.get("intent_sha256") != _sha256(intent_body)
            ):
                _fail("worker_stage_import_intent_invalid")

        imported: list[dict[str, Any]] = []
        for record in verified_stage.records:
            result_origin_sha256 = record["result"]["worker_origin"]["origin_sha256"]
            imported_receipt = canonical.import_staged_arm_result(
                record["cell_id"],
                expected_attempt_id=record["attempt_id"],
                result=record["result"],
            )
            imported.append(
                {
                    "cell_id": record["cell_id"],
                    "attempt_id": imported_receipt["attempt_id"],
                    "arm_result_event_sha256": imported_receipt["arm_result_event_sha256"],
                    "result_origin_sha256": result_origin_sha256,
                    "stage_verification_sha256": _sha256(record["verification"]),
                    "stage_commit_sha256": _sha256(record["commit"]),
                }
            )
        final_head = canonical.resume().journal_head_sha256

    receipt_body = {
        "schema": IMPORT_RECEIPT_SCHEMA,
        "campaign_plan_sha256": plan.plan_sha256,
        "arm": manifest["arm"],
        "worker_attempt_slot": manifest["worker_attempt_slot"],
        "stage_manifest_sha256": manifest["manifest_sha256"],
        "import_intent_sha256": intent["intent_sha256"],
        "imported": imported,
        "final_canonical_head_sha256": final_head,
    }
    receipt = {**receipt_body, "receipt_sha256": _sha256(receipt_body)}
    _atomic_create_or_verify(receipt_path, receipt)
    return receipt


__all__ = [
    "IMPORT_INTENT_SCHEMA",
    "IMPORT_RECEIPT_SCHEMA",
    "PAIRED_CAMPAIGN_CELL_TYPE",
    "VERIFIED_STAGE_SCHEMA",
    "VerifiedWorkerAttemptStage",
    "WorkerAttemptImportError",
    "import_verified_worker_stage",
    "verify_terminal_worker_stage",
]
