"""Fail-closed state capture for paired RLC cognitive-action campaigns.

The public protocol contains identities and commitments only. Private resident
state is published through :class:`PrivateActionSnapshotStore`, restored once
per pair arm, and erased after both arms are sealed. Model weights are never
snapshotted: the protocol binds their externally computed identity instead.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Never

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.brain.llm.latent_cortex.action_intervention import (
    CONTROL_ARM,
    TREATMENT_ARM,
)
from core.brain.llm.latent_cortex.action_state_key_custody import (
    SNAPSHOT_KEY_CUSTODY_IDENTITY_SCHEMA,
    SnapshotKeyCustodian,
    SnapshotKeyCustodyError,
)
from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    VerifiedCampaignTrustPolicy,
    build_role_attestation,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.runtime_identity import (
    latent_request_payload_sha256,
)
from core.brain.llm.latent_cortex.worker_capture_identity import (
    validate_worker_capture_identity,
    validate_worker_capture_origin_binding,
)

ACTION_STATE_CAPTURE_REQUEST_PAYLOAD_SCHEMA: Final = (
    "aura.rlc.action_state_capture.request_payload.v2"
)
ACTION_STATE_CAPTURE_REQUEST_SCHEMA: Final = "aura.rlc.action_state_capture.request.v2"
ACTION_STATE_CAPTURE_RECEIPT_SCHEMA: Final = "aura.rlc.action_state_capture.receipt.v1"
ACTION_STATE_CAPTURE_OPPORTUNITY_SCHEMA: Final = (
    "aura.rlc.action_state_capture.first_opportunity.v1"
)
ACTION_STATE_CAPTURE_WORKER_ORIGIN_SCHEMA: Final = "aura.rlc.action_state_capture.worker_origin.v1"
PRIVATE_ACTION_SNAPSHOT_ENVELOPE_SCHEMA: Final = "aura.rlc.action_state_capture.private_snapshot.v2"
PRIVATE_ACTION_SNAPSHOT_CHUNK_AAD_SCHEMA: Final = (
    "aura.rlc.action_state_capture.private_chunk_aad.v1"
)
PRIVATE_ACTION_SNAPSHOT_BINDING_SCHEMA: Final = (
    "aura.rlc.action_state_capture.private_snapshot_binding.v1"
)
PRIVATE_ACTION_SNAPSHOT_KEY_CONTEXT_SCHEMA: Final = (
    "aura.rlc.action_state_capture.private_key_context.v1"
)
PRIVATE_ACTION_SNAPSHOT_HANDLE_SCHEMA: Final = "aura.rlc.action_state_capture.private_handle.v1"
PRIVATE_ACTION_SNAPSHOT_PUBLICATION_SCHEMA: Final = (
    "aura.rlc.action_state_capture.private_publication.v1"
)
PRIVATE_ACTION_SNAPSHOT_LEDGER_SCHEMA: Final = "aura.rlc.action_state_capture.private_use_ledger.v1"
PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA_V1: Final = (
    "aura.rlc.action_state_capture.private_operation.v1"
)
PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA: Final = (
    "aura.rlc.action_state_capture.private_operation.v2"
)
PRIVATE_ACTION_SNAPSHOT_RESTORE_SCHEMA: Final = "aura.rlc.action_state_capture.private_restore.v1"
PRIVATE_ACTION_SNAPSHOT_SEAL_SCHEMA: Final = "aura.rlc.action_state_capture.private_seal.v1"
PRIVATE_ACTION_SNAPSHOT_ERASURE_SCHEMA: Final = "aura.rlc.action_state_capture.private_erasure.v1"

PAIR_ARMS: Final = (TREATMENT_ARM, CONTROL_ARM)

STATE_COMPONENT_NAMES: Final = (
    "branch_state_sha256",
    "durable_state_sha256",
    "evidence_state_sha256",
    "kv_cache_sha256",
    "latent_slots_sha256",
    "memory_state_sha256",
    "public_action_state_sha256",
    "rng_state_sha256",
)
_STATE_VALUE_NAMES = tuple(name.removesuffix("_sha256") for name in STATE_COMPONENT_NAMES)
_RUNNER_SUPPLIED_COMPONENTS: Final = {
    "durable_state_sha256",
    "rng_state_sha256",
}
_COMPONENT_OBSERVATION_OWNERS: Final = {
    name: (
        "runner_supplied_worker_commitment_verified_and_snapshotted"
        if name in _RUNNER_SUPPLIED_COMPONENTS
        else "resident_worker_measured_before_first_action_opportunity"
    )
    for name in STATE_COMPONENT_NAMES
}
_CHUNK_BYTES = 1024 * 1024
# Publication encodes and encrypts bounded chunks directly into its hidden
# transaction. Restore retains at most one reconstructed component because the
# state-application callback still consumes concrete Python values.
_MAX_COMPONENT_BYTES = 128 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NAMESPACE_NAMES: Final = (
    "bundles",
    "snapshots",
    "handles",
    "keys",
    "ledgers",
    "operations",
    "tombstones",
    "transactions",
)

_LATENT_REQUIRED_FIELDS = {
    "prompt",
    "messages",
    "domain",
    "config",
    "budget",
    "runtime_controls",
}
_LATENT_OPTIONAL_FIELDS = {
    "cognitive_context",
    "operation_authority",
    "action_policy_evidence",
    "action_intervention",
    "external_execution_offer",
    "response_contract",
    "verifier_guidance",
    "facet_reliability",
}
_PRIVATE_ANSWER_KEYS = {
    "answer_key",
    "expected_answer",
    "gold_answer",
    "private_answer",
    "reference_answer",
    "sealed_answer",
    "solution_key",
}
_REQUEST_PAYLOAD_FIELDS = {
    "schema",
    "capture_id",
    "capture_not_after_unix",
    "campaign_name",
    "campaign_design_sha256",
    "campaign_protocol_sha256",
    "policy_sha256",
    "policy_revision",
    "pair_id",
    "task_id",
    "task_payload_sha256",
    "action",
    "model_identity_sha256",
    "model_weights_identity_sha256",
    "execution_identity_sha256",
    "calibration_bucket",
    "bucket_classifier_sha256",
    "bucket_evidence_sha256",
    "latent_reason_request_sha256",
    "runner_durable_state_commitment_sha256",
    "runner_rng_root_commitment_sha256",
    "worker_origin_binding",
    "worker_origin_binding_sha256",
    "expected_action_opportunity_ordinal",
}
_REQUEST_FIELDS = {
    "schema",
    "request_payload",
    "policy_document",
    "runner_attestation",
    "request_sha256",
}
_OPPORTUNITY_FIELDS = {
    "schema",
    "ordinal",
    "action",
    "episode_step",
    "schedule_step",
    "branch_id",
    "layer_index",
    "kv_position",
    "pre_action_state_sha256",
    "pre_action_kv_sha256",
    "opportunity_sha256",
}
_WORKER_ORIGIN_FIELDS = {
    "schema",
    "algorithm",
    "public_key_b64",
    "key_id",
    "signed_payload_sha256",
    "signature_b64",
    "origin_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "request_sha256",
    "capture_id",
    "captured_at_unix",
    "campaign_name",
    "campaign_design_sha256",
    "campaign_protocol_sha256",
    "policy_sha256",
    "policy_revision",
    "pair_id",
    "task_id",
    "task_payload_sha256",
    "action",
    "model_identity_sha256",
    "model_weights_identity_sha256",
    "execution_identity_sha256",
    "runtime_identity_sha256",
    "calibration_bucket",
    "bucket_classifier_sha256",
    "bucket_evidence_sha256",
    "latent_reason_request_sha256",
    "runner_durable_state_commitment_sha256",
    "runner_rng_root_commitment_sha256",
    "worker_origin_binding_sha256",
    "private_snapshot_envelope_sha256",
    "state_components",
    "state_sha256",
    "component_observation_owners",
    "first_action_opportunity",
    "capture_boundary",
    "action_executed",
    "action_trace_count",
    "decode_started",
    "decoded_token_count",
    "output_present",
    "output_sha256",
    "output_byte_count",
    "worker_origin",
    "receipt_sha256",
}
_PUBLIC_BINDING_FIELDS = (
    "campaign_name",
    "campaign_design_sha256",
    "campaign_protocol_sha256",
    "policy_sha256",
    "policy_revision",
    "pair_id",
    "task_id",
    "task_payload_sha256",
    "action",
    "model_identity_sha256",
    "model_weights_identity_sha256",
    "execution_identity_sha256",
    "calibration_bucket",
    "bucket_classifier_sha256",
    "bucket_evidence_sha256",
    "latent_reason_request_sha256",
    "runner_durable_state_commitment_sha256",
    "runner_rng_root_commitment_sha256",
    "worker_origin_binding_sha256",
)


class ActionStateCaptureError(ValueError):
    """Stable fail-closed protocol or private snapshot error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UnknownActionStateApplicationError(ActionStateCaptureError):
    """Resident mutation may have started; this process must not serve again."""

    def __init__(self, operation: Mapping[str, Any]) -> None:
        super().__init__("private_snapshot_unknown_application_process_replacement_required")
        evidence = {
            "state": "UNKNOWN_APPLICATION",
            "operation_id": str(operation.get("operation_id") or ""),
            "arm": str(operation.get("arm") or ""),
            "worker_boot_id": str(operation.get("worker_boot_id") or ""),
            "worker_pid": operation.get("worker_pid"),
            "request_sha256": str(operation.get("request_sha256") or ""),
            "snapshot_sha256": str(operation.get("snapshot_sha256") or ""),
            "process_replacement_required": True,
            "same_process_retry_allowed": False,
        }
        self._quarantine_evidence_bytes = canonical_json_bytes(evidence)

    @property
    def quarantine_evidence(self) -> dict[str, Any]:
        """Return a copy so callers cannot rewrite the fatal evidence in place."""

        return json.loads(self._quarantine_evidence_bytes)


def _fail(code: str) -> Never:
    if not isinstance(code, str) or not code or code != code.strip():
        raise ActionStateCaptureError("action_state_capture_error_code_invalid")
    raise ActionStateCaptureError(code)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_json_bytes(value))


def _keyed_digest(secret: bytes, value: Any) -> str:
    return hmac.new(secret, canonical_json_bytes(value), hashlib.sha256).hexdigest()


def _chunk_aad(
    *,
    request_sha256: str,
    component_name: str,
    ordinal: int,
    plaintext_byte_count: int,
    plaintext_sha256: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema": PRIVATE_ACTION_SNAPSHOT_CHUNK_AAD_SCHEMA,
            "request_sha256": request_sha256,
            "component_name": component_name,
            "ordinal": ordinal,
            "plaintext_byte_count": plaintext_byte_count,
            "plaintext_sha256": plaintext_sha256,
        }
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: Any, *, role: str) -> str:
    if not _is_sha256(value):
        _fail(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str, maximum: int = 240) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        _fail(f"{role}_invalid")
    return value


def _hex_identifier(value: Any, *, role: str, length: int = 32) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _positive_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _nonnegative_int(value: Any, *, role: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{role}_invalid")
    return value


def _normalize(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_invalid")


def _normalized_mapping(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{role}_invalid")
    normalized = _normalize(dict(value), role=role)
    if not isinstance(normalized, dict):
        _fail(f"{role}_invalid")
    return normalized


def _contains_private_answer_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _PRIVATE_ANSWER_KEYS or _contains_private_answer_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_private_answer_key(item) for item in value)
    return False


def normalized_latent_reason_request_sha256(value: Mapping[str, Any]) -> str:
    """Hash exactly the normalized semantic request used by ``latent_reason``.

    The request preimage is never included in a public capture artifact.
    Explicit private-answer fields are rejected before hashing.
    """

    request = _normalized_mapping(value, role="action_state_capture_latent_request")
    fields = set(request)
    if (
        not _LATENT_REQUIRED_FIELDS <= fields
        or not fields <= _LATENT_REQUIRED_FIELDS | _LATENT_OPTIONAL_FIELDS
        or _contains_private_answer_key(request)
    ):
        _fail("action_state_capture_latent_request_invalid")
    prompt = request["prompt"]
    messages = request["messages"]
    if not (isinstance(prompt, str) and prompt.strip() or isinstance(messages, list) and messages):
        _fail("action_state_capture_latent_request_invalid")
    if (
        not isinstance(request["domain"], str)
        or not request["domain"].strip()
        or any(
            request[name] is not None and not isinstance(request[name], dict)
            for name in ("config", "budget", "runtime_controls")
        )
    ):
        _fail("action_state_capture_latent_request_invalid")
    try:
        return latent_request_payload_sha256(
            prompt=prompt,
            messages=messages,
            domain=request["domain"],
            config=request["config"],
            budget=request["budget"],
            runtime_controls=request["runtime_controls"],
            cognitive_context=request.get("cognitive_context"),
            operation_authority=request.get("operation_authority"),
            action_policy_evidence=request.get("action_policy_evidence"),
            action_intervention=request.get("action_intervention"),
            external_execution_offer=request.get("external_execution_offer"),
            response_contract=request.get("response_contract"),
            verifier_guidance=request.get("verifier_guidance"),
            facet_reliability=request.get("facet_reliability"),
        )
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("action_state_capture_latent_request_invalid")


def _public_key_raw(value: Any, *, role: str) -> bytes:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    if isinstance(value, Ed25519PublicKey):
        return value.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    if isinstance(value, bytes) and len(value) == 32:
        return value
    if isinstance(value, str) and value == value.strip():
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            _fail(f"{role}_invalid")
        if len(raw) == 32 and base64.b64encode(raw).decode("ascii") == value:
            return raw
    _fail(f"{role}_invalid")


def _private_key_public_raw(value: Any, *, role: str) -> bytes:
    try:
        return _public_key_raw(value.public_key(), role=role)
    except AttributeError:
        _fail(f"{role}_invalid")


def action_state_capture_request_payload(
    *,
    policy: VerifiedCampaignTrustPolicy,
    capture_id: str,
    capture_not_after_unix: int,
    campaign_design_sha256: str,
    pair_id: str,
    task_id: str,
    task_payload_sha256: str,
    action: OperationKind | str,
    model_identity: Mapping[str, Any],
    model_weights_identity_sha256: str,
    execution_identity: Mapping[str, Any],
    calibration_bucket: str,
    bucket_classifier_sha256: str,
    bucket_evidence_sha256: str,
    latent_reason_request: Mapping[str, Any],
    runner_durable_state_commitment_sha256: str,
    runner_rng_root_commitment_sha256: str,
    worker_origin_binding: Mapping[str, Any],
    expected_supervisor_public_key: Any,
) -> dict[str, Any]:
    """Build the strict, hash-only material signed by the campaign runner."""

    if not isinstance(policy, VerifiedCampaignTrustPolicy):
        _fail("action_state_capture_policy_invalid")
    try:
        operation = OperationKind(str(action))
    except ValueError:
        _fail("action_state_capture_action_invalid")
    deadline = _positive_int(capture_not_after_unix, role="action_state_capture_deadline")
    if deadline >= policy.document["expires_at_unix"]:
        _fail("action_state_capture_deadline_outside_policy")
    try:
        origin_binding = validate_worker_capture_origin_binding(
            worker_origin_binding,
            expected_supervisor_public_key=expected_supervisor_public_key,
        )
    except (TypeError, ValueError) as exc:
        if str(exc) == "worker_capture_launch_challenge_not_current":
            raise ActionStateCaptureError(
                "action_state_capture_worker_origin_binding_not_current"
            ) from exc
        raise ActionStateCaptureError(
            "action_state_capture_worker_origin_binding_invalid"
        ) from exc
    return {
        "schema": ACTION_STATE_CAPTURE_REQUEST_PAYLOAD_SCHEMA,
        "capture_id": _hex_identifier(capture_id, role="action_state_capture_id"),
        "capture_not_after_unix": deadline,
        "campaign_name": policy.document["campaign_name"],
        "campaign_design_sha256": _sha256(
            campaign_design_sha256,
            role="action_state_capture_design",
        ),
        "campaign_protocol_sha256": policy.document["protocol_sha256"],
        "policy_sha256": policy.policy_sha256,
        "policy_revision": policy.document["policy_revision"],
        "pair_id": _identifier(pair_id, role="action_state_capture_pair"),
        "task_id": _identifier(task_id, role="action_state_capture_task"),
        "task_payload_sha256": _sha256(
            task_payload_sha256, role="action_state_capture_task_payload"
        ),
        "action": operation.value,
        "model_identity_sha256": _digest(
            _normalized_mapping(model_identity, role="action_state_capture_model_identity")
        ),
        "model_weights_identity_sha256": _sha256(
            model_weights_identity_sha256,
            role="action_state_capture_model_weights_identity",
        ),
        "execution_identity_sha256": _digest(
            _normalized_mapping(execution_identity, role="action_state_capture_execution_identity")
        ),
        "calibration_bucket": _identifier(
            calibration_bucket,
            role="action_state_capture_bucket",
            maximum=512,
        ),
        "bucket_classifier_sha256": _sha256(
            bucket_classifier_sha256,
            role="action_state_capture_bucket_classifier",
        ),
        "bucket_evidence_sha256": _sha256(
            bucket_evidence_sha256,
            role="action_state_capture_bucket_evidence",
        ),
        "latent_reason_request_sha256": normalized_latent_reason_request_sha256(
            latent_reason_request
        ),
        "runner_durable_state_commitment_sha256": _sha256(
            runner_durable_state_commitment_sha256,
            role="action_state_capture_durable_commitment",
        ),
        "runner_rng_root_commitment_sha256": _sha256(
            runner_rng_root_commitment_sha256,
            role="action_state_capture_rng_commitment",
        ),
        "worker_origin_binding": origin_binding,
        "worker_origin_binding_sha256": origin_binding["binding_sha256"],
        "expected_action_opportunity_ordinal": 1,
    }


def _validate_request_payload(
    value: Any,
    *,
    policy: VerifiedCampaignTrustPolicy,
    expected_supervisor_public_key: Any,
    now_unix: int | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_PAYLOAD_FIELDS:
        _fail("action_state_capture_request_payload_fields")
    payload = _normalized_mapping(value, role="action_state_capture_request_payload")
    if (
        payload.get("schema") != ACTION_STATE_CAPTURE_REQUEST_PAYLOAD_SCHEMA
        or payload.get("campaign_name") != policy.document["campaign_name"]
        or payload.get("campaign_protocol_sha256") != policy.document["protocol_sha256"]
        or payload.get("policy_sha256") != policy.policy_sha256
        or type(payload.get("policy_revision")) is not int
        or payload.get("policy_revision") != policy.document["policy_revision"]
        or type(payload.get("expected_action_opportunity_ordinal")) is not int
        or payload.get("expected_action_opportunity_ordinal") != 1
    ):
        _fail("action_state_capture_request_identity_mismatch")
    _hex_identifier(payload.get("capture_id"), role="action_state_capture_id")
    deadline = _positive_int(
        payload.get("capture_not_after_unix"),
        role="action_state_capture_deadline",
    )
    if deadline >= policy.document["expires_at_unix"]:
        _fail("action_state_capture_deadline_outside_policy")
    for name in (
        "campaign_design_sha256",
        "campaign_protocol_sha256",
        "policy_sha256",
        "task_payload_sha256",
        "model_identity_sha256",
        "model_weights_identity_sha256",
        "execution_identity_sha256",
        "bucket_classifier_sha256",
        "bucket_evidence_sha256",
        "latent_reason_request_sha256",
        "runner_durable_state_commitment_sha256",
        "runner_rng_root_commitment_sha256",
    ):
        _sha256(payload.get(name), role=f"action_state_capture_{name}")
    try:
        origin_binding = validate_worker_capture_origin_binding(
            payload.get("worker_origin_binding"),
            expected_supervisor_public_key=expected_supervisor_public_key,
            now_unix=now_unix,
        )
    except (TypeError, ValueError) as exc:
        if str(exc) == "worker_capture_launch_challenge_not_current":
            raise ActionStateCaptureError(
                "action_state_capture_worker_origin_binding_not_current"
            ) from exc
        raise ActionStateCaptureError(
            "action_state_capture_worker_origin_binding_invalid"
        ) from exc
    if (
        payload.get("worker_origin_binding_sha256")
        != origin_binding["binding_sha256"]
    ):
        _fail("action_state_capture_worker_origin_binding_mismatch")
    for name in ("pair_id", "task_id", "calibration_bucket"):
        _identifier(
            payload.get(name),
            role=f"action_state_capture_{name}",
            maximum=512,
        )
    try:
        OperationKind(payload.get("action"))
    except ValueError:
        _fail("action_state_capture_action_invalid")
    return payload


def build_action_state_capture_request(
    *,
    policy: VerifiedCampaignTrustPolicy,
    runner_private_key: Any,
    signed_at_unix: int,
    expected_supervisor_public_key: Any,
    **payload_arguments: Any,
) -> dict[str, Any]:
    """Create a runner-attested state-capture request."""

    signed_at = _positive_int(signed_at_unix, role="action_state_capture_signed_at")
    payload = action_state_capture_request_payload(
        policy=policy,
        expected_supervisor_public_key=expected_supervisor_public_key,
        **payload_arguments,
    )
    try:
        validate_worker_capture_origin_binding(
            payload["worker_origin_binding"],
            expected_supervisor_public_key=expected_supervisor_public_key,
            now_unix=signed_at,
        )
    except (TypeError, ValueError) as exc:
        raise ActionStateCaptureError(
            "action_state_capture_worker_origin_binding_not_current"
        ) from exc
    if signed_at > payload["capture_not_after_unix"]:
        _fail("action_state_capture_signature_after_deadline")
    attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=payload,
        signed_at_unix=signed_at,
        private_key=runner_private_key,
    )
    body = {
        "schema": ACTION_STATE_CAPTURE_REQUEST_SCHEMA,
        "request_payload": payload,
        "policy_document": dict(policy.document),
        "runner_attestation": attestation,
    }
    return {**body, "request_sha256": _digest(body)}


@dataclass(frozen=True, slots=True)
class VerifiedActionStateCaptureRequest:
    """Immutable result of current admission or historical replay."""

    _request_bytes: bytes = field(repr=False)
    policy: VerifiedCampaignTrustPolicy = field(repr=False)
    signed_at_unix: int
    current_policy_admission: bool

    @property
    def request(self) -> dict[str, Any]:
        return json.loads(self._request_bytes)

    @property
    def payload(self) -> dict[str, Any]:
        return self.request["request_payload"]

    @property
    def request_sha256(self) -> str:
        return self.request["request_sha256"]


def _request_signed_at(value: Mapping[str, Any]) -> int:
    attestation = value.get("runner_attestation")
    signed_payload = attestation.get("signed_payload") if isinstance(attestation, Mapping) else None
    signed_at = (
        signed_payload.get("signed_at_unix") if isinstance(signed_payload, Mapping) else None
    )
    return _positive_int(signed_at, role="action_state_capture_signed_at")


def _verify_request(
    value: Any,
    *,
    trusted_root_public_key_pem: bytes,
    expected_supervisor_public_key: Any,
    current_policy_document: Mapping[str, Any] | None,
    now_unix: int | None,
) -> VerifiedActionStateCaptureRequest:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        _fail("action_state_capture_request_fields")
    request = _normalized_mapping(value, role="action_state_capture_request")
    if request.get("schema") != ACTION_STATE_CAPTURE_REQUEST_SCHEMA:
        _fail("action_state_capture_request_schema")
    body = {name: request[name] for name in _REQUEST_FIELDS - {"request_sha256"}}
    if request.get("request_sha256") != _digest(body):
        _fail("action_state_capture_request_hash_mismatch")
    signed_at = _request_signed_at(request)
    current = current_policy_document is not None
    if current:
        observed_at = _positive_int(now_unix, role="action_state_capture_admission_time")
        embedded = _normalized_mapping(
            request.get("policy_document"),
            role="action_state_capture_embedded_policy",
        )
        supplied = _normalized_mapping(
            current_policy_document,
            role="action_state_capture_current_policy",
        )
        if canonical_json_bytes(embedded) != canonical_json_bytes(supplied):
            _fail("action_state_capture_superseded_policy")
        policy_document = supplied
        validation_time = observed_at
    else:
        policy_document = request.get("policy_document")
        validation_time = signed_at
    raw_payload = request.get("request_payload")
    if not isinstance(raw_payload, Mapping):
        _fail("action_state_capture_request_payload_fields")
    try:
        policy = validate_campaign_trust_policy(
            policy_document,
            trusted_root_public_key_pem=trusted_root_public_key_pem,
            expected_campaign_name=raw_payload.get("campaign_name"),
            expected_policy_sha256=raw_payload.get("policy_sha256"),
            expected_protocol_sha256=raw_payload.get("campaign_protocol_sha256"),
            minimum_policy_revision=raw_payload.get("policy_revision"),
            now_unix=validation_time,
        )
        payload = _validate_request_payload(
            raw_payload,
            policy=policy,
            expected_supervisor_public_key=expected_supervisor_public_key,
            now_unix=validation_time if current else None,
        )
        verify_role_attestation(
            policy,
            request.get("runner_attestation"),
            role=CAMPAIGN_RUNNER,
            expected_payload=payload,
            not_after_unix=validation_time if current else None,
        )
    except ActionStateCaptureError:
        raise
    except (TypeError, ValueError, RuntimeError, ImportError) as exc:
        raise ActionStateCaptureError("action_state_capture_trust_verification_failed") from exc
    if signed_at > payload["capture_not_after_unix"]:
        _fail("action_state_capture_signature_after_deadline")
    if current and validation_time > payload["capture_not_after_unix"]:
        _fail("action_state_capture_request_expired")
    return VerifiedActionStateCaptureRequest(
        _request_bytes=canonical_json_bytes(request),
        policy=policy,
        signed_at_unix=signed_at,
        current_policy_admission=current,
    )


def admit_action_state_capture_request(
    value: Any,
    *,
    trusted_root_public_key_pem: bytes,
    expected_supervisor_public_key: Any,
    current_policy_document: Mapping[str, Any],
    now_unix: int,
) -> VerifiedActionStateCaptureRequest:
    """Admit only the exact externally supplied current policy at current time."""

    return _verify_request(
        value,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        expected_supervisor_public_key=expected_supervisor_public_key,
        current_policy_document=current_policy_document,
        now_unix=now_unix,
    )


def replay_action_state_capture_request(
    value: Any,
    *,
    trusted_root_public_key_pem: bytes,
    expected_supervisor_public_key: Any,
) -> VerifiedActionStateCaptureRequest:
    """Replay an embedded root-signed policy at its runner signature time."""

    return _verify_request(
        value,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        expected_supervisor_public_key=expected_supervisor_public_key,
        current_policy_document=None,
        now_unix=None,
    )


@dataclass(frozen=True, slots=True)
class PrivateSnapshotPublication:
    """Opaque private snapshot reference used by the receipt builder."""

    handle: str = field(repr=False)
    snapshot_sha256: str
    request_sha256: str
    _component_items: tuple[tuple[str, str], ...]

    @property
    def state_components(self) -> dict[str, str]:
        return dict(self._component_items)


@dataclass(frozen=True, slots=True)
class PrivateSnapshotRestore:
    """One committed pair-arm restore; state is intentionally hidden in repr."""

    state: dict[str, Any] = field(repr=False)
    receipt: dict[str, Any]


def _snapshot_binding(
    admission: VerifiedActionStateCaptureRequest,
) -> dict[str, Any]:
    if not isinstance(admission, VerifiedActionStateCaptureRequest):
        _fail("private_snapshot_admission_required")
    payload = admission.payload
    return {
        "schema": PRIVATE_ACTION_SNAPSHOT_BINDING_SCHEMA,
        "request_sha256": admission.request_sha256,
        **{name: payload[name] for name in _PUBLIC_BINDING_FIELDS},
    }


def _state_value_stream(value: Any) -> tuple[str, Iterator[bytes]]:
    from core.brain.llm.latent_cortex.action_continuation import (
        PortableStateComponent,
    )

    if isinstance(value, PortableStateComponent):
        return "portable_state_v1", value.iter_encoded_chunks(chunk_bytes=_CHUNK_BYTES)
    if isinstance(value, (bytes, bytearray, memoryview)):
        view = memoryview(value)

        def iter_bytes() -> Iterator[bytes]:
            for offset in range(0, len(view), _CHUNK_BYTES):
                yield view[offset : offset + _CHUNK_BYTES].tobytes()

        return "bytes", iter_bytes()

    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    def iter_json() -> Iterator[bytes]:
        try:
            for text in encoder.iterencode(value):
                for offset in range(0, len(text), _CHUNK_BYTES):
                    yield text[offset : offset + _CHUNK_BYTES].encode("ascii")
        except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError):
            _fail("private_snapshot_state_value_invalid")

    return "canonical_json", iter_json()


def _fixed_chunks(parts: Iterator[bytes]) -> Iterator[bytes]:
    pending = bytearray()
    observed = False
    for part in parts:
        if not isinstance(part, bytes):
            _fail("private_snapshot_state_stream_invalid")
        observed = observed or bool(part)
        view = memoryview(part)
        offset = 0
        while offset < len(view):
            amount = min(_CHUNK_BYTES - len(pending), len(view) - offset)
            pending.extend(view[offset : offset + amount])
            offset += amount
            if len(pending) == _CHUNK_BYTES:
                yield bytes(pending)
                pending.clear()
    if pending or not observed:
        yield bytes(pending)


def _strict_json_loads(raw: bytes, *, role: str) -> dict[str, Any]:
    def object_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_constant=lambda _raw: _fail(f"{role}_non_finite"),
        )
    except ActionStateCaptureError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, OverflowError):
        _fail(f"{role}_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


def _validate_bound_runtime_identity(value: Any) -> dict[str, Any]:
    identity = _normalized_mapping(
        value,
        role="action_state_capture_runtime_identity",
    )
    source_commit = identity.get("source_commit")
    issues = identity.get("issues")
    if (
        identity.get("schema") != "aura.latent_cortex.runtime_identity.v1"
        or identity.get("identity_bound") is not True
        or identity.get("source_verified") is not True
        or identity.get("source_dirty") is not False
        or not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source_commit)
        or not _is_sha256(identity.get("workspace_state_sha256"))
        or not _is_sha256(identity.get("shell_assets_sha256"))
        or not isinstance(issues, list)
        or issues
        or (
            identity.get("installed_app_required") is True
            and (
                identity.get("installed_app_verified") is not True
                or not _is_sha256(identity.get("app_executable_sha256"))
                or not _is_sha256(identity.get("launch_manifest_sha256"))
            )
        )
    ):
        _fail("action_state_capture_runtime_identity_unbound")
    return identity


def _crash_boundary(_name: str) -> None:
    """No-op fault-injection seam used by crash-recovery contract tests."""


class PrivateActionSnapshotStore:
    """Content-addressed, pair-local private resident-state lifecycle."""

    def __init__(self, root: str | Path, *, key_custodian: SnapshotKeyCustodian) -> None:
        if not isinstance(key_custodian, SnapshotKeyCustodian):
            _fail("private_snapshot_key_custodian_required")
        custody_identity = _normalized_mapping(
            key_custodian.identity,
            role="private_snapshot_key_custody_identity",
        )
        identity_body = {
            name: item for name, item in custody_identity.items() if name != "identity_sha256"
        }
        if (
            custody_identity.get("schema") != SNAPSHOT_KEY_CUSTODY_IDENTITY_SCHEMA
            or custody_identity.get("custody_class")
            not in {"macos_keychain", "external_service", "hardware_keystore"}
            or custody_identity.get("identity_sha256") != _digest(identity_body)
        ):
            _fail("private_snapshot_key_custody_identity_invalid")
        self._key_custodian = key_custodian
        self._key_custody_identity = custody_identity
        candidate = Path(root).expanduser().absolute()
        self.root = candidate
        self._root_fd = self._open_root(candidate)
        self._namespace_fds: dict[str, int] = {}
        self._lock_fd = -1
        for name in _NAMESPACE_NAMES:
            self._ensure_directory(candidate / name)
            self._namespace_fds[name] = self._open_directory(candidate / name)
        self._lock_path = candidate / ".action-state-capture.lock"
        self._lock_fd = self._open_lock_file()

    def close(self) -> None:
        """Release descriptors without changing the durable store."""

        lock_fd = getattr(self, "_lock_fd", -1)
        self._lock_fd = -1
        if lock_fd >= 0:
            os.close(lock_fd)
        namespace_fds = getattr(self, "_namespace_fds", {})
        self._namespace_fds = {}
        for descriptor in namespace_fds.values():
            os.close(descriptor)
        root_fd = getattr(self, "_root_fd", -1)
        self._root_fd = -1
        if root_fd >= 0:
            os.close(root_fd)

    def __del__(self) -> None:
        try:
            self.close()
        # not a failure: a capture already closed is the state __del__ is reaching.
        except OSError:
            pass

    @staticmethod
    def _validate_private_directory_stat(observed: os.stat_result) -> None:
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            _fail("private_snapshot_directory_unsafe")

    @classmethod
    def _open_root(cls, path: Path) -> int:
        try:
            observed = path.lstat()
        except FileNotFoundError:
            path.mkdir(parents=True, mode=0o700, exist_ok=False)
            os.chmod(path, 0o700)
            observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode):
            _fail("private_snapshot_directory_unsafe")
        cls._validate_private_directory_stat(observed)
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | _DIRECTORY | _CLOEXEC | _NOFOLLOW,
            )
        except OSError as exc:
            raise ActionStateCaptureError("private_snapshot_directory_unavailable") from exc
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
            os.close(descriptor)
            _fail("private_snapshot_root_changed_during_open")
        try:
            cls._validate_private_directory_stat(current)
        except ActionStateCaptureError:
            os.close(descriptor)
            raise
        return descriptor

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            _fail("private_snapshot_path_escape")
        if relative == Path("."):
            return ()
        if any(part in {"", ".", ".."} for part in relative.parts):
            _fail("private_snapshot_path_escape")
        return relative.parts

    def _open_directory(self, path: Path, *, create: bool = False) -> int:
        parts = self._relative_parts(path)
        if self._root_fd < 0:
            _fail("private_snapshot_store_closed")
        offset = 0
        if parts and parts[0] in self._namespace_fds:
            descriptor = os.dup(self._namespace_fds[parts[0]])
            offset = 1
        else:
            descriptor = os.dup(self._root_fd)
        try:
            self._validate_private_directory_stat(os.fstat(descriptor))
            for part in parts[offset:]:
                if create:
                    created = False
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                        created = True
                    # not a failure: the directory already exists, which is what mkdir was for.
                    except FileExistsError:
                        pass
                    if created:
                        os.fsync(descriptor)
                child = os.open(
                    part,
                    os.O_RDONLY | _DIRECTORY | _CLOEXEC | _NOFOLLOW,
                    dir_fd=descriptor,
                )
                self._validate_private_directory_stat(os.fstat(child))
                os.close(descriptor)
                descriptor = child
            return descriptor
        except ActionStateCaptureError:
            os.close(descriptor)
            raise
        except OSError as exc:
            os.close(descriptor)
            raise ActionStateCaptureError("private_snapshot_directory_unavailable") from exc

    @contextmanager
    def _directory(self, path: Path, *, create: bool = False) -> Iterator[int]:
        descriptor = self._open_directory(path, create=create)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    def _open_lock_file(self) -> int:
        name = self._lock_path.name
        flags = os.O_RDWR | _CLOEXEC | _NOFOLLOW
        try:
            try:
                descriptor = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self._root_fd)
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                os.fsync(self._root_fd)
            except FileExistsError:
                descriptor = os.open(name, flags, dir_fd=self._root_fd)
            self._validate_private_file_stat(os.fstat(descriptor), allow_empty=True)
            return descriptor
        except ActionStateCaptureError:
            raise
        except OSError as exc:
            raise ActionStateCaptureError("private_snapshot_lock_failed") from exc

    def _assert_descriptor_bindings(self) -> None:
        try:
            root_path_stat = self.root.lstat()
            root_fd_stat = os.fstat(self._root_fd)
            if stat.S_ISLNK(root_path_stat.st_mode) or (
                root_path_stat.st_dev,
                root_path_stat.st_ino,
            ) != (root_fd_stat.st_dev, root_fd_stat.st_ino):
                _fail("private_snapshot_root_binding_changed")
            self._validate_private_directory_stat(root_fd_stat)
            for name, descriptor in self._namespace_fds.items():
                path_stat = os.stat(name, dir_fd=self._root_fd, follow_symlinks=False)
                descriptor_stat = os.fstat(descriptor)
                if (
                    stat.S_ISLNK(path_stat.st_mode)
                    or (path_stat.st_dev, path_stat.st_ino)
                    != (descriptor_stat.st_dev, descriptor_stat.st_ino)
                ):
                    _fail("private_snapshot_namespace_binding_changed")
                self._validate_private_directory_stat(descriptor_stat)
            lock_path_stat = os.stat(
                self._lock_path.name,
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            lock_fd_stat = os.fstat(self._lock_fd)
            if (lock_path_stat.st_dev, lock_path_stat.st_ino) != (
                lock_fd_stat.st_dev,
                lock_fd_stat.st_ino,
            ):
                _fail("private_snapshot_lock_binding_changed")
            self._validate_private_file_stat(lock_fd_stat, allow_empty=True)
        except ActionStateCaptureError:
            raise
        except OSError as exc:
            raise ActionStateCaptureError("private_snapshot_descriptor_binding_unavailable") from exc

    def _ensure_directory(self, path: Path) -> None:
        descriptor = self._open_directory(path, create=True)
        os.close(descriptor)

    @staticmethod
    def _validate_private_file_stat(
        observed: os.stat_result,
        *,
        allow_empty: bool = False,
    ) -> None:
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or (not allow_empty and observed.st_size <= 0)
        ):
            _fail("private_snapshot_file_unsafe")

    def _stat_path(self, path: Path) -> os.stat_result:
        parts = self._relative_parts(path)
        if not parts:
            return os.fstat(self._root_fd)
        with self._directory(path.parent) as parent_fd:
            try:
                return os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise ActionStateCaptureError("private_snapshot_file_unavailable") from exc

    def _path_exists(self, path: Path) -> bool:
        parts = self._relative_parts(path)
        if not parts:
            return self._root_fd >= 0
        try:
            with self._directory(path.parent) as parent_fd:
                os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            return True
        # not a failure: a file that is already gone is the state this is reaching.
        except FileNotFoundError:
            return False
        except ActionStateCaptureError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return False
            raise

    def _assert_private_file(self, path: Path, *, allow_empty: bool = False) -> os.stat_result:
        observed = self._stat_path(path)
        self._validate_private_file_stat(observed, allow_empty=allow_empty)
        return observed

    def _unlink(self, path: Path, *, missing_ok: bool = False, sync: bool = False) -> None:
        parts = self._relative_parts(path)
        if not parts:
            _fail("private_snapshot_path_escape")
        with self._directory(path.parent) as parent_fd:
            try:
                os.unlink(parts[-1], dir_fd=parent_fd)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            if sync:
                os.fsync(parent_fd)

    def _rmdir(self, path: Path, *, sync: bool = False) -> None:
        parts = self._relative_parts(path)
        if not parts:
            _fail("private_snapshot_path_escape")
        with self._directory(path.parent) as parent_fd:
            os.rmdir(parts[-1], dir_fd=parent_fd)
            if sync:
                os.fsync(parent_fd)

    def _list_directory(self, path: Path) -> tuple[str, ...]:
        with self._directory(path) as descriptor:
            return tuple(sorted(os.listdir(descriptor)))

    def _remove_tree(self, path: Path) -> None:
        observed = self._stat_path(path)
        if stat.S_ISREG(observed.st_mode):
            self._validate_private_file_stat(observed, allow_empty=True)
            self._unlink(path)
            return
        if not stat.S_ISDIR(observed.st_mode):
            _fail("private_snapshot_temporary_entry_unsafe")
        self._validate_private_directory_stat(observed)
        for name in self._list_directory(path):
            self._remove_tree(path / name)
        self._rmdir(path, sync=True)

    def _atomic_publish_directory(self, source: Path, target: Path) -> None:
        self._ensure_directory(source.parent)
        self._ensure_directory(target.parent)
        with self._directory(source.parent) as source_parent_fd, self._directory(
            target.parent
        ) as target_parent_fd:
            try:
                source_stat = os.stat(
                    source.name,
                    dir_fd=source_parent_fd,
                    follow_symlinks=False,
                )
                self._validate_private_directory_stat(source_stat)
                try:
                    os.stat(
                        target.name,
                        dir_fd=target_parent_fd,
                        follow_symlinks=False,
                    )
                # not a failure: a file that is already gone is the state this is reaching.
                except FileNotFoundError:
                    pass
                else:
                    _fail("private_snapshot_bundle_collision")
                os.rename(
                    source.name,
                    target.name,
                    src_dir_fd=source_parent_fd,
                    dst_dir_fd=target_parent_fd,
                )
                os.fsync(target_parent_fd)
                os.fsync(source_parent_fd)
            except ActionStateCaptureError:
                raise
            except OSError as exc:
                raise ActionStateCaptureError("private_snapshot_bundle_commit_failed") from exc

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if self._lock_fd < 0:
            _fail("private_snapshot_store_closed")
        acquired = False
        try:
            self._validate_private_directory_stat(os.fstat(self._root_fd))
            self._validate_private_file_stat(os.fstat(self._lock_fd), allow_empty=True)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            acquired = True
            self._assert_descriptor_bindings()
            self._cleanup_temporary_files()
            yield
            self._assert_descriptor_bindings()
        except ActionStateCaptureError:
            raise
        except OSError as exc:
            raise ActionStateCaptureError("private_snapshot_lock_failed") from exc
        finally:
            if acquired and self._lock_fd >= 0:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _cleanup_temporary_files(self) -> None:
        for directory in (self.root, *(self.root / name for name in _NAMESPACE_NAMES)):
            for name in self._list_directory(directory):
                if not name.startswith(".tmp-"):
                    continue
                path = directory / name
                self._remove_tree(path)
        for handle_hash in self._list_directory(self.root / "bundles"):
            _hex_identifier(
                handle_hash,
                role="private_snapshot_bundle_handle",
                length=64,
            )
            self._cleanup_nested_temporaries(self.root / "bundles" / handle_hash)

    def _cleanup_nested_temporaries(self, directory: Path) -> None:
        for name in self._list_directory(directory):
            path = directory / name
            if name.startswith(".tmp-"):
                self._remove_tree(path)
                continue
            observed = self._stat_path(path)
            if stat.S_ISDIR(observed.st_mode):
                self._validate_private_directory_stat(observed)
                self._cleanup_nested_temporaries(path)

    def _fsync_directory(self, path: Path) -> None:
        with self._directory(path) as descriptor:
            os.fsync(descriptor)

    def _atomic_publish(
        self,
        path: Path,
        payload: bytes,
        *,
        replace: bool,
    ) -> None:
        self._ensure_directory(path.parent)
        temporary_name = f".tmp-{path.name}-{secrets.token_hex(16)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW
        descriptor = -1
        with self._directory(path.parent) as parent_fd:
            try:
                descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
                os.fchmod(descriptor, 0o600)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        _fail("private_snapshot_write_failed")
                    view = view[written:]
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1
                target_exists = True
                try:
                    target_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                # not a failure: a file that is already gone is the state this is reaching.
                except FileNotFoundError:
                    target_exists = False
                if target_exists:
                    self._validate_private_file_stat(target_stat, allow_empty=True)
                    if not replace:
                        if (
                            self._read_owned_at(parent_fd, path.name, maximum=max(1, len(payload)))
                            != payload
                        ):
                            _fail("private_snapshot_content_address_collision")
                        os.unlink(temporary_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                        return
                os.replace(
                    temporary_name,
                    path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
                observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                self._validate_private_file_stat(observed, allow_empty=True)
                if observed.st_size != len(payload):
                    _fail("private_snapshot_publish_size_mismatch")
            except ActionStateCaptureError:
                raise
            except OSError as exc:
                raise ActionStateCaptureError("private_snapshot_publish_failed") from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                # not a failure: the temporary is already gone, which is what unlinking it was for.
                except FileNotFoundError:
                    pass

    def _read_owned(self, path: Path, *, maximum: int) -> bytes:
        with self._directory(path.parent) as parent_fd:
            return self._read_owned_at(parent_fd, path.name, maximum=maximum)

    def _read_owned_at(self, parent_fd: int, name: str, *, maximum: int) -> bytes:
        try:
            descriptor = os.open(name, os.O_RDONLY | _CLOEXEC | _NOFOLLOW, dir_fd=parent_fd)
        except OSError as exc:
            raise ActionStateCaptureError("private_snapshot_file_unavailable") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 0
                or before.st_size > maximum
            ):
                _fail("private_snapshot_file_unsafe")
            parts: list[bytes] = []
            remaining = before.st_size
            while remaining:
                part = os.read(descriptor, min(_CHUNK_BYTES, remaining))
                if not part:
                    _fail("private_snapshot_short_read")
                parts.append(part)
                remaining -= len(part)
            if os.read(descriptor, 1):
                _fail("private_snapshot_changed_during_read")
            after = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            )
            if identity != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            ) or identity != (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
                current.st_nlink,
            ):
                _fail("private_snapshot_changed_during_read")
            return b"".join(parts)
        except OSError as exc:
            raise ActionStateCaptureError("private_snapshot_read_failed") from exc
        finally:
            os.close(descriptor)

    def _assert_directory_chain(self, directory: Path) -> None:
        descriptor = self._open_directory(directory)
        os.close(descriptor)

    def _read_json(self, path: Path, *, role: str) -> dict[str, Any]:
        raw = self._read_owned(path, maximum=_MAX_JSON_BYTES)
        value = _strict_json_loads(raw, role=role)
        if raw != canonical_json_bytes(value) + b"\n":
            _fail(f"{role}_noncanonical")
        return value

    @staticmethod
    def _handle_hash(handle: str) -> str:
        if not isinstance(handle, str) or not handle.startswith("asc1_") or len(handle) != 69:
            _fail("private_snapshot_handle_invalid")
        _hex_identifier(handle[5:], role="private_snapshot_handle_token", length=64)
        return _digest_bytes(handle.encode("ascii"))

    @staticmethod
    def _handle_secret(handle: str) -> bytes:
        PrivateActionSnapshotStore._handle_hash(handle)
        return bytes.fromhex(handle[5:])

    def _paths_for_handle_hash(self, handle_hash: str) -> dict[str, Path]:
        bundle = self.root / "bundles" / handle_hash
        return {
            "bundle": bundle,
            "publication": bundle / "publication.json",
            "handle": bundle / "handle.json",
            "key": bundle / "wrapped-key.json",
            "ledger": bundle / "ledger.json",
            "operation": bundle / "operation.json",
            "tombstone": bundle / "tombstone.json",
            "snapshot": bundle / "snapshot",
        }

    @staticmethod
    def _hash_document(
        value: Mapping[str, Any], *, hash_field: str, fields: set[str], role: str
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != fields:
            _fail(f"{role}_fields")
        normalized = _normalized_mapping(value, role=role)
        body = {name: normalized[name] for name in fields - {hash_field}}
        if normalized.get(hash_field) != _digest(body):
            _fail(f"{role}_hash_mismatch")
        return normalized

    def _load_handle(
        self,
        handle_hash: str,
        *,
        request_sha256: str,
        handle_secret: bytes,
    ) -> dict[str, Any]:
        paths = self._paths_for_handle_hash(handle_hash)
        if self._path_exists(paths["tombstone"]):
            self._read_erasure(paths["tombstone"], handle_secret=handle_secret)
            _fail("private_snapshot_erased")
        value = self._read_json(paths["handle"], role="private_snapshot_handle")
        fields = {
            "schema",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "pair_id",
            "task_id",
            "dek_sha256",
            "state",
            "created_at_unix",
            "seal_receipt_sha256",
            "handle_authentication_sha256",
            "record_sha256",
        }
        value = self._hash_document(
            value,
            hash_field="record_sha256",
            fields=fields,
            role="private_snapshot_handle",
        )
        if (
            value["schema"] != PRIVATE_ACTION_SNAPSHOT_HANDLE_SCHEMA
            or value["handle_sha256"] != handle_hash
            or value["request_sha256"] != request_sha256
            or not _is_sha256(value.get("dek_sha256"))
            or value["state"] not in {"active", "sealed"}
        ):
            _fail("private_snapshot_handle_binding_mismatch")
        self._verify_handle_authentication(
            value,
            hash_field="record_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_handle",
        )
        _positive_int(value["created_at_unix"], role="private_snapshot_created_at")
        if (value["state"] == "active" and value["seal_receipt_sha256"] is not None) or (
            value["state"] == "sealed" and not _is_sha256(value["seal_receipt_sha256"])
        ):
            _fail("private_snapshot_handle_state_invalid")
        return value

    def _load_ledger(
        self,
        handle_hash: str,
        *,
        request_sha256: str,
        snapshot_sha256: str,
        handle_secret: bytes,
    ) -> dict[str, Any]:
        path = self._paths_for_handle_hash(handle_hash)["ledger"]
        value = self._read_json(path, role="private_snapshot_ledger")
        fields = {
            "schema",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "pair_id",
            "uses",
            "sealed",
            "sealed_at_unix",
            "sequence",
            "handle_authentication_sha256",
            "ledger_sha256",
        }
        value = self._hash_document(
            value,
            hash_field="ledger_sha256",
            fields=fields,
            role="private_snapshot_ledger",
        )
        uses = value.get("uses")
        if (
            value["schema"] != PRIVATE_ACTION_SNAPSHOT_LEDGER_SCHEMA
            or value["handle_sha256"] != handle_hash
            or value["snapshot_sha256"] != snapshot_sha256
            or value["request_sha256"] != request_sha256
            or not isinstance(uses, dict)
            or set(uses) != set(PAIR_ARMS)
            or type(value.get("sealed")) is not bool
            or type(value.get("sequence")) is not int
            or value["sequence"] < 0
        ):
            _fail("private_snapshot_ledger_binding_mismatch")
        self._verify_handle_authentication(
            value,
            hash_field="ledger_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_ledger",
        )
        sealed_at = value.get("sealed_at_unix")
        if (value["sealed"] is False and sealed_at is not None) or (
            value["sealed"] is True and (type(sealed_at) is not int or sealed_at <= 0)
        ):
            _fail("private_snapshot_ledger_seal_invalid")
        for arm, use in uses.items():
            if use is None:
                continue
            if (
                not isinstance(use, dict)
                or set(use)
                != {
                    "arm",
                    "operation_id",
                    "restored_at_unix",
                    "post_apply_state_sha256",
                    "restore_receipt_sha256",
                }
                or use["arm"] != arm
            ):
                _fail("private_snapshot_ledger_use_invalid")
            _hex_identifier(
                use["operation_id"],
                role="private_snapshot_operation_id",
            )
            _positive_int(
                use["restored_at_unix"],
                role="private_snapshot_restored_at",
            )
            _sha256(
                use["post_apply_state_sha256"],
                role="private_snapshot_post_apply_state",
            )
            _sha256(
                use["restore_receipt_sha256"],
                role="private_snapshot_restore_receipt",
            )
        return value

    @staticmethod
    def _key_context_sha256(*, handle_hash: str, request_sha256: str) -> str:
        return _digest(
            {
                "schema": PRIVATE_ACTION_SNAPSHOT_KEY_CONTEXT_SCHEMA,
                "handle_sha256": handle_hash,
                "request_sha256": request_sha256,
            }
        )

    @staticmethod
    def _zeroize(value: bytearray) -> None:
        for index in range(len(value)):
            value[index] = 0
        value.clear()

    def _load_dek(
        self,
        handle_hash: str,
        *,
        request_sha256: str,
        expected_sha256: str,
    ) -> bytearray:
        path = self._paths_for_handle_hash(handle_hash)["key"]
        envelope = self._read_json(path, role="private_snapshot_wrapped_key")
        try:
            key = self._key_custodian.unwrap_data_key(
                envelope,
                context_sha256=self._key_context_sha256(
                    handle_hash=handle_hash,
                    request_sha256=request_sha256,
                ),
            )
        except SnapshotKeyCustodyError as exc:
            raise ActionStateCaptureError("private_snapshot_key_custody_failed") from exc
        if len(key) != 32 or _digest_bytes(bytes(key)) != expected_sha256:
            self._zeroize(key)
            _fail("private_snapshot_dek_invalid")
        return key

    def _load_envelope(
        self,
        handle_hash: str,
        snapshot_sha256: str,
        *,
        binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self._paths_for_handle_hash(handle_hash)["snapshot"] / "envelope.json"
        value = self._read_json(path, role="private_snapshot_envelope")
        fields = {
            "schema",
            "binding",
            "created_at_unix",
            "chunk_size_bytes",
            "component_count",
            "total_bytes",
            "components",
            "envelope_sha256",
        }
        value = self._hash_document(
            value,
            hash_field="envelope_sha256",
            fields=fields,
            role="private_snapshot_envelope",
        )
        if (
            value["schema"] != PRIVATE_ACTION_SNAPSHOT_ENVELOPE_SCHEMA
            or value["envelope_sha256"] != snapshot_sha256
            or value["binding"] != dict(binding)
            or value["chunk_size_bytes"] != _CHUNK_BYTES
            or value["component_count"] != len(STATE_COMPONENT_NAMES)
            or not isinstance(value["components"], list)
            or len(value["components"]) != len(STATE_COMPONENT_NAMES)
            or type(value.get("total_bytes")) is not int
            or not 0 <= value["total_bytes"] <= _MAX_SNAPSHOT_BYTES
        ):
            _fail("private_snapshot_envelope_binding_mismatch")
        return value

    @staticmethod
    def _authenticated_document(
        body: Mapping[str, Any],
        *,
        handle_secret: bytes,
        authentication_field: str,
        hash_field: str,
    ) -> dict[str, Any]:
        normalized = dict(body)
        authenticated = {
            **normalized,
            authentication_field: _keyed_digest(handle_secret, normalized),
        }
        return {**authenticated, hash_field: _digest(authenticated)}

    @staticmethod
    def _verify_handle_authentication(
        value: Mapping[str, Any],
        *,
        hash_field: str,
        handle_secret: bytes,
        role: str,
    ) -> None:
        authentication_field = "handle_authentication_sha256"
        body = {
            name: item
            for name, item in value.items()
            if name not in {hash_field, authentication_field}
        }
        expected = _keyed_digest(handle_secret, body)
        observed = value.get(authentication_field)
        if not isinstance(observed, str) or not hmac.compare_digest(observed, expected):
            _fail(f"{role}_authentication_mismatch")

    @classmethod
    def _ledger_document(cls, body: Mapping[str, Any], *, handle_secret: bytes) -> dict[str, Any]:
        return cls._authenticated_document(
            body,
            handle_secret=handle_secret,
            authentication_field="handle_authentication_sha256",
            hash_field="ledger_sha256",
        )

    @classmethod
    def _handle_document(cls, body: Mapping[str, Any], *, handle_secret: bytes) -> dict[str, Any]:
        return cls._authenticated_document(
            body,
            handle_secret=handle_secret,
            authentication_field="handle_authentication_sha256",
            hash_field="record_sha256",
        )

    @classmethod
    def _publication_document(
        cls,
        body: Mapping[str, Any],
        *,
        handle_secret: bytes,
    ) -> dict[str, Any]:
        return cls._authenticated_document(
            body,
            handle_secret=handle_secret,
            authentication_field="handle_authentication_sha256",
            hash_field="publication_sha256",
        )

    def _read_publication(self, path: Path) -> dict[str, Any]:
        value = self._read_json(path, role="private_snapshot_publication")
        fields = {
            "schema",
            "handle",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "state_components",
            "handle_authentication_sha256",
            "publication_sha256",
        }
        value = self._hash_document(
            value,
            hash_field="publication_sha256",
            fields=fields,
            role="private_snapshot_publication",
        )
        handle = value.get("handle")
        handle_hash = self._handle_hash(handle)
        handle_secret = self._handle_secret(handle)
        components = value.get("state_components")
        if (
            value.get("schema") != PRIVATE_ACTION_SNAPSHOT_PUBLICATION_SCHEMA
            or value.get("handle_sha256") != handle_hash
            or not _is_sha256(value.get("snapshot_sha256"))
            or not _is_sha256(value.get("request_sha256"))
            or not isinstance(components, dict)
            or set(components) != set(STATE_COMPONENT_NAMES)
            or any(not _is_sha256(item) for item in components.values())
        ):
            _fail("private_snapshot_publication_invalid")
        self._verify_handle_authentication(
            value,
            hash_field="publication_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_publication",
        )
        return value

    def _recover_published_capture(
        self,
        *,
        admission: VerifiedActionStateCaptureRequest,
        component_hashes: Mapping[str, str],
    ) -> PrivateSnapshotPublication | None:
        bundles_root = self.root / "bundles"
        for handle_hash in self._list_directory(bundles_root):
            _hex_identifier(
                handle_hash,
                role="private_snapshot_bundle_handle",
                length=64,
            )
            paths = self._paths_for_handle_hash(handle_hash)
            if not self._path_exists(paths["publication"]):
                continue
            publication = self._read_publication(paths["publication"])
            if publication["handle_sha256"] != handle_hash:
                _fail("private_snapshot_publication_directory_mismatch")
            if publication["request_sha256"] != admission.request_sha256:
                continue
            if publication["state_components"] != dict(component_hashes):
                _fail("private_snapshot_publication_retry_state_mismatch")
            handle = publication["handle"]
            handle_secret = self._handle_secret(handle)
            handle_record = self._load_handle(
                handle_hash,
                request_sha256=admission.request_sha256,
                handle_secret=handle_secret,
            )
            data_key = self._load_dek(
                handle_hash,
                request_sha256=admission.request_sha256,
                expected_sha256=handle_record["dek_sha256"],
            )
            self._zeroize(data_key)
            self._load_ledger(
                handle_hash,
                request_sha256=admission.request_sha256,
                snapshot_sha256=handle_record["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            self._load_envelope(
                handle_hash,
                handle_record["snapshot_sha256"],
                binding=_snapshot_binding(admission),
            )
            return PrivateSnapshotPublication(
                handle=handle,
                snapshot_sha256=handle_record["snapshot_sha256"],
                request_sha256=admission.request_sha256,
                _component_items=tuple(sorted(component_hashes.items())),
            )
        return None

    def publication_for_request(
        self,
        admission: VerifiedActionStateCaptureRequest,
    ) -> PrivateSnapshotPublication:
        """Recover the unique authenticated private publication for a request.

        This is deliberately a worker-private lookup.  The bearer handle is
        reconstructed only inside the custody boundary, which lets a replaced
        resident worker continue a paired experiment without putting the
        handle in a public IPC frame or campaign artifact.
        """

        binding = _snapshot_binding(admission)
        matches: list[PrivateSnapshotPublication] = []
        with self._locked():
            for handle_hash in self._list_directory(self.root / "bundles"):
                _hex_identifier(
                    handle_hash,
                    role="private_snapshot_bundle_handle",
                    length=64,
                )
                paths = self._paths_for_handle_hash(handle_hash)
                if not self._path_exists(paths["publication"]):
                    continue
                publication = self._read_publication(paths["publication"])
                if publication["handle_sha256"] != handle_hash:
                    _fail("private_snapshot_publication_directory_mismatch")
                if publication["request_sha256"] != admission.request_sha256:
                    continue
                handle = publication["handle"]
                handle_secret = self._handle_secret(handle)
                recovered = self._recover_operation(
                    handle_hash,
                    request_sha256=binding["request_sha256"],
                    handle_secret=handle_secret,
                )
                if recovered is not None:
                    publication = self._read_publication(paths["publication"])
                handle_record = self._load_handle(
                    handle_hash,
                    request_sha256=admission.request_sha256,
                    handle_secret=handle_secret,
                )
                self._load_ledger(
                    handle_hash,
                    request_sha256=admission.request_sha256,
                    snapshot_sha256=handle_record["snapshot_sha256"],
                    handle_secret=handle_secret,
                )
                envelope = self._load_envelope(
                    handle_hash,
                    handle_record["snapshot_sha256"],
                    binding=binding,
                )
                components = {
                    f"{item['name']}_sha256": item["value_sha256"]
                    for item in envelope["components"]
                }
                if components != publication["state_components"]:
                    _fail("private_snapshot_publication_envelope_mismatch")
                matches.append(
                    PrivateSnapshotPublication(
                        handle=handle,
                        snapshot_sha256=handle_record["snapshot_sha256"],
                        request_sha256=admission.request_sha256,
                        _component_items=tuple(sorted(components.items())),
                    )
                )
        if not matches:
            _fail("private_snapshot_publication_not_found")
        if len(matches) != 1:
            _fail("private_snapshot_publication_ambiguous")
        return matches[0]

    def _load_target_ledger(
        self,
        value: Mapping[str, Any],
        *,
        handle_hash: str,
        request_sha256: str,
        snapshot_sha256: str,
        handle_secret: bytes,
    ) -> dict[str, Any]:
        fields = {
            "schema",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "pair_id",
            "uses",
            "sealed",
            "sealed_at_unix",
            "sequence",
            "handle_authentication_sha256",
            "ledger_sha256",
        }
        normalized = self._hash_document(
            value,
            hash_field="ledger_sha256",
            fields=fields,
            role="private_snapshot_target_ledger",
        )
        self._verify_handle_authentication(
            normalized,
            hash_field="ledger_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_target_ledger",
        )
        if (
            normalized["schema"] != PRIVATE_ACTION_SNAPSHOT_LEDGER_SCHEMA
            or normalized["handle_sha256"] != handle_hash
            or normalized["snapshot_sha256"] != snapshot_sha256
            or normalized["request_sha256"] != request_sha256
            or normalized["sealed"] is not True
            or type(normalized["sealed_at_unix"]) is not int
            or normalized["sealed_at_unix"] <= 0
        ):
            _fail("private_snapshot_target_ledger_invalid")
        return normalized

    def _load_target_handle(
        self,
        value: Mapping[str, Any],
        *,
        handle_hash: str,
        request_sha256: str,
        snapshot_sha256: str,
        handle_secret: bytes,
    ) -> dict[str, Any]:
        fields = {
            "schema",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "pair_id",
            "task_id",
            "dek_sha256",
            "state",
            "created_at_unix",
            "seal_receipt_sha256",
            "handle_authentication_sha256",
            "record_sha256",
        }
        normalized = self._hash_document(
            value,
            hash_field="record_sha256",
            fields=fields,
            role="private_snapshot_target_handle",
        )
        self._verify_handle_authentication(
            normalized,
            hash_field="record_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_target_handle",
        )
        if (
            normalized["schema"] != PRIVATE_ACTION_SNAPSHOT_HANDLE_SCHEMA
            or normalized["handle_sha256"] != handle_hash
            or normalized["snapshot_sha256"] != snapshot_sha256
            or normalized["request_sha256"] != request_sha256
            or not _is_sha256(normalized.get("dek_sha256"))
            or normalized["state"] != "sealed"
            or not _is_sha256(normalized["seal_receipt_sha256"])
        ):
            _fail("private_snapshot_target_handle_invalid")
        return normalized

    def publish(
        self,
        admission: VerifiedActionStateCaptureRequest,
        private_state: Mapping[str, Any],
        *,
        created_at_unix: int,
    ) -> PrivateSnapshotPublication:
        """Atomically publish a typed private snapshot and return an opaque handle."""

        binding = _snapshot_binding(admission)
        created_at = _positive_int(created_at_unix, role="private_snapshot_created_at")
        if (
            created_at < admission.signed_at_unix
            or created_at > admission.payload["capture_not_after_unix"]
        ):
            _fail("private_snapshot_capture_time_invalid")
        if not isinstance(private_state, Mapping) or set(private_state) != set(_STATE_VALUE_NAMES):
            _fail("private_snapshot_state_components_invalid")
        handle = f"asc1_{secrets.token_hex(32)}"
        handle_hash = self._handle_hash(handle)
        handle_secret = self._handle_secret(handle)
        encryption_key = bytearray(secrets.token_bytes(32))
        paths = self._paths_for_handle_hash(handle_hash)
        try:
            dek_sha256 = _digest_bytes(bytes(encryption_key))
            cipher = AESGCM(bytes(encryption_key))
            wrapped_key = dict(
                self._key_custodian.wrap_data_key(
                    bytes(encryption_key),
                    context_sha256=self._key_context_sha256(
                        handle_hash=handle_hash,
                        request_sha256=admission.request_sha256,
                    ),
                )
            )
            with self._locked():
                stage_dir = (
                    self.root
                    / "transactions"
                    / f".tmp-publish-{handle_hash}-{secrets.token_hex(16)}"
                )
                stage_snapshot = stage_dir / "snapshot"
                stage_paths = {
                    "publication": stage_dir / "publication.json",
                    "handle": stage_dir / "handle.json",
                    "key": stage_dir / "wrapped-key.json",
                    "ledger": stage_dir / "ledger.json",
                }
                try:
                    self._ensure_directory(stage_dir)
                    self._atomic_publish(
                        stage_paths["key"],
                        canonical_json_bytes(wrapped_key) + b"\n",
                        replace=False,
                    )
                    self._ensure_directory(stage_snapshot)
                    chunks_root = stage_snapshot / "chunks"
                    self._ensure_directory(chunks_root)
                    total_bytes = 0
                    component_documents: list[dict[str, Any]] = []
                    for name in _STATE_VALUE_NAMES:
                        value_type, parts = _state_value_stream(private_state[name])
                        component_bytes = 0
                        component_digest = hashlib.sha256()
                        chunks: list[dict[str, Any]] = []
                        component_dir = chunks_root / name
                        self._ensure_directory(component_dir)
                        for ordinal, chunk in enumerate(_fixed_chunks(parts)):
                            component_bytes += len(chunk)
                            total_bytes += len(chunk)
                            if component_bytes > _MAX_COMPONENT_BYTES:
                                _fail("private_snapshot_component_too_large")
                            if total_bytes > _MAX_SNAPSHOT_BYTES:
                                _fail("private_snapshot_too_large")
                            component_digest.update(chunk)
                            plaintext_sha256 = _digest_bytes(chunk)
                            nonce = secrets.token_bytes(12)
                            aad = _chunk_aad(
                                request_sha256=admission.request_sha256,
                                component_name=name,
                                ordinal=ordinal,
                                plaintext_byte_count=len(chunk),
                                plaintext_sha256=plaintext_sha256,
                            )
                            encrypted = cipher.encrypt(nonce, chunk, aad)
                            file_sha256 = _digest_bytes(encrypted)
                            chunk_info = {
                                "ordinal": ordinal,
                                "plaintext_byte_count": len(chunk),
                                "plaintext_sha256": plaintext_sha256,
                                "byte_count": len(encrypted),
                                "file_sha256": file_sha256,
                                "nonce_b64": base64.b64encode(nonce).decode("ascii"),
                            }
                            chunk_path = component_dir / f"{ordinal:08d}-{file_sha256}.bin"
                            self._atomic_publish(chunk_path, encrypted, replace=False)
                            chunks.append(chunk_info)
                        component_documents.append(
                            {
                                "name": name,
                                "value_type": value_type,
                                "byte_count": component_bytes,
                                "value_sha256": component_digest.hexdigest(),
                                "chunks": chunks,
                            }
                        )

                    envelope_body = {
                        "schema": PRIVATE_ACTION_SNAPSHOT_ENVELOPE_SCHEMA,
                        "binding": binding,
                        "created_at_unix": created_at,
                        "chunk_size_bytes": _CHUNK_BYTES,
                        "component_count": len(component_documents),
                        "total_bytes": total_bytes,
                        "components": component_documents,
                    }
                    envelope = {
                        **envelope_body,
                        "envelope_sha256": _digest(envelope_body),
                    }
                    component_hashes = {
                        f"{item['name']}_sha256": item["value_sha256"]
                        for item in component_documents
                    }
                    if (
                        component_hashes["durable_state_sha256"]
                        != admission.payload["runner_durable_state_commitment_sha256"]
                        or component_hashes["rng_state_sha256"]
                        != admission.payload["runner_rng_root_commitment_sha256"]
                    ):
                        _fail("private_snapshot_runner_state_commitment_mismatch")
                    recovered = self._recover_published_capture(
                        admission=admission,
                        component_hashes=component_hashes,
                    )
                    if recovered is not None:
                        self._remove_tree(stage_dir)
                        return recovered

                    snapshot_sha256 = envelope["envelope_sha256"]
                    handle_body = {
                        "schema": PRIVATE_ACTION_SNAPSHOT_HANDLE_SCHEMA,
                        "handle_sha256": handle_hash,
                        "snapshot_sha256": snapshot_sha256,
                        "request_sha256": admission.request_sha256,
                        "pair_id": admission.payload["pair_id"],
                        "task_id": admission.payload["task_id"],
                        "dek_sha256": dek_sha256,
                        "state": "active",
                        "created_at_unix": created_at,
                        "seal_receipt_sha256": None,
                    }
                    ledger_body = {
                        "schema": PRIVATE_ACTION_SNAPSHOT_LEDGER_SCHEMA,
                        "handle_sha256": handle_hash,
                        "snapshot_sha256": snapshot_sha256,
                        "request_sha256": admission.request_sha256,
                        "pair_id": admission.payload["pair_id"],
                        "uses": {arm: None for arm in PAIR_ARMS},
                        "sealed": False,
                        "sealed_at_unix": None,
                        "sequence": 0,
                    }
                    publication_body = {
                        "schema": PRIVATE_ACTION_SNAPSHOT_PUBLICATION_SCHEMA,
                        "handle": handle,
                        "handle_sha256": handle_hash,
                        "snapshot_sha256": snapshot_sha256,
                        "request_sha256": admission.request_sha256,
                        "state_components": component_hashes,
                    }
                    self._atomic_publish(
                        stage_snapshot / "envelope.json",
                        canonical_json_bytes(envelope) + b"\n",
                        replace=False,
                    )
                    self._atomic_publish(
                        stage_paths["ledger"],
                        canonical_json_bytes(
                            self._ledger_document(ledger_body, handle_secret=handle_secret)
                        )
                        + b"\n",
                        replace=False,
                    )
                    self._atomic_publish(
                        stage_paths["handle"],
                        canonical_json_bytes(
                            self._handle_document(handle_body, handle_secret=handle_secret)
                        )
                        + b"\n",
                        replace=False,
                    )
                    self._atomic_publish(
                        stage_paths["publication"],
                        canonical_json_bytes(
                            self._publication_document(
                                publication_body,
                                handle_secret=handle_secret,
                            )
                        )
                        + b"\n",
                        replace=False,
                    )
                    self._fsync_directory(stage_dir)
                    _crash_boundary("publish_before_bundle_commit")
                    self._atomic_publish_directory(stage_dir, paths["bundle"])
                    _crash_boundary("publish_after_bundle_commit")
                except ActionStateCaptureError as exc:
                    if self._path_exists(stage_dir):
                        self._remove_tree(stage_dir)
                    if isinstance(exc.__cause__, OSError):
                        raise ActionStateCaptureError(
                            "private_snapshot_publication_io_failed"
                        ) from exc.__cause__
                    raise
                except OSError as exc:
                    if self._path_exists(stage_dir):
                        self._remove_tree(stage_dir)
                    raise ActionStateCaptureError(
                        "private_snapshot_publication_io_failed"
                    ) from exc
                except Exception:  # noqa: BLE001 - rollback every staged publication failure
                    if self._path_exists(stage_dir):
                        self._remove_tree(stage_dir)
                    raise

                return PrivateSnapshotPublication(
                    handle=handle,
                    snapshot_sha256=snapshot_sha256,
                    request_sha256=admission.request_sha256,
                    _component_items=tuple(sorted(component_hashes.items())),
                )
        except SnapshotKeyCustodyError as exc:
            raise ActionStateCaptureError("private_snapshot_key_custody_failed") from exc
        finally:
            self._zeroize(encryption_key)

    def _read_private_state(
        self,
        handle_hash: str,
        snapshot_sha256: str,
        envelope: Mapping[str, Any],
        *,
        encryption_key: bytes,
    ) -> dict[str, Any]:
        request_sha256 = envelope["binding"]["request_sha256"]
        cipher = AESGCM(encryption_key)
        state: dict[str, Any] = {}
        total = 0
        for component in envelope["components"]:
            if (
                not isinstance(component, dict)
                or set(component)
                != {
                    "name",
                    "value_type",
                    "byte_count",
                    "value_sha256",
                    "chunks",
                }
                or component["name"] not in _STATE_VALUE_NAMES
                or component["value_type"]
                not in {"bytes", "canonical_json", "portable_state_v1"}
                or type(component["byte_count"]) is not int
                or not 0 <= component["byte_count"] <= _MAX_COMPONENT_BYTES
                or not _is_sha256(component["value_sha256"])
                or not isinstance(component["chunks"], list)
                or not component["chunks"]
            ):
                _fail("private_snapshot_component_manifest_invalid")
            payload = bytearray()
            component_digest = hashlib.sha256()
            expected_ordinal = 0
            for chunk in component["chunks"]:
                if (
                    not isinstance(chunk, dict)
                    or set(chunk)
                    != {
                        "ordinal",
                        "plaintext_byte_count",
                        "plaintext_sha256",
                        "byte_count",
                        "file_sha256",
                        "nonce_b64",
                    }
                    or chunk["ordinal"] != expected_ordinal
                    or type(chunk["plaintext_byte_count"]) is not int
                    or not 0 <= chunk["plaintext_byte_count"] <= _CHUNK_BYTES
                    or not _is_sha256(chunk["plaintext_sha256"])
                    or type(chunk["byte_count"]) is not int
                    or chunk["byte_count"] != chunk["plaintext_byte_count"] + 16
                    or not _is_sha256(chunk["file_sha256"])
                ):
                    _fail("private_snapshot_chunk_manifest_invalid")
                chunk_path = (
                    self._paths_for_handle_hash(handle_hash)["snapshot"]
                    / "chunks"
                    / component["name"]
                    / f"{expected_ordinal:08d}-{chunk['file_sha256']}.bin"
                )
                raw = self._read_owned(chunk_path, maximum=_CHUNK_BYTES + 16)
                if len(raw) != chunk["byte_count"] or _digest_bytes(raw) != chunk["file_sha256"]:
                    _fail("private_snapshot_chunk_hash_mismatch")
                try:
                    nonce = base64.b64decode(chunk["nonce_b64"], validate=True)
                except (binascii.Error, TypeError, ValueError):
                    _fail("private_snapshot_chunk_nonce_invalid")
                if len(nonce) != 12:
                    _fail("private_snapshot_chunk_nonce_invalid")
                aad = _chunk_aad(
                    request_sha256=request_sha256,
                    component_name=component["name"],
                    ordinal=expected_ordinal,
                    plaintext_byte_count=chunk["plaintext_byte_count"],
                    plaintext_sha256=chunk["plaintext_sha256"],
                )
                try:
                    plaintext = cipher.decrypt(nonce, raw, aad)
                except (InvalidTag, ValueError) as exc:
                    raise ActionStateCaptureError(
                        "private_snapshot_chunk_authentication_failed"
                    ) from exc
                if (
                    len(plaintext) != chunk["plaintext_byte_count"]
                    or _digest_bytes(plaintext) != chunk["plaintext_sha256"]
                ):
                    _fail("private_snapshot_chunk_plaintext_mismatch")
                payload.extend(plaintext)
                component_digest.update(plaintext)
                if len(payload) > _MAX_COMPONENT_BYTES:
                    self._zeroize(payload)
                    _fail("private_snapshot_component_too_large")
                expected_ordinal += 1
            if (
                len(payload) != component["byte_count"]
                or component_digest.hexdigest() != component["value_sha256"]
            ):
                self._zeroize(payload)
                _fail("private_snapshot_component_hash_mismatch")
            total += len(payload)
            if component["value_type"] == "bytes":
                state[component["name"]] = bytes(payload)
            elif component["value_type"] == "portable_state_v1":
                from core.brain.llm.latent_cortex.action_continuation import (
                    PortableStateComponent,
                )

                try:
                    portable = PortableStateComponent.from_bytes(payload)
                    canonical_digest = portable.sha256()
                except (TypeError, ValueError) as exc:
                    self._zeroize(payload)
                    raise ActionStateCaptureError(
                        "private_snapshot_portable_state_invalid"
                    ) from exc
                if canonical_digest != component["value_sha256"]:
                    self._zeroize(payload)
                    _fail("private_snapshot_structured_value_noncanonical")
                state[component["name"]] = portable
            else:
                try:
                    value = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._zeroize(payload)
                    _fail("private_snapshot_structured_value_invalid")
                canonical_digest = hashlib.sha256()
                canonical_bytes = 0
                _value_type, canonical_parts = _state_value_stream(value)
                for canonical_chunk in _fixed_chunks(canonical_parts):
                    canonical_digest.update(canonical_chunk)
                    canonical_bytes += len(canonical_chunk)
                if (
                    canonical_bytes != len(payload)
                    or canonical_digest.hexdigest() != component["value_sha256"]
                ):
                    self._zeroize(payload)
                    _fail("private_snapshot_structured_value_noncanonical")
                state[component["name"]] = value
            self._zeroize(payload)
        if set(state) != set(_STATE_VALUE_NAMES) or total != envelope["total_bytes"]:
            _fail("private_snapshot_state_reconstruction_mismatch")
        return state

    @staticmethod
    def _restore_receipt(
        *,
        handle_hash: str,
        snapshot_sha256: str,
        request_sha256: str,
        pair_id: str,
        arm: str,
        operation_id: str,
        restored_at_unix: int,
        envelope: Mapping[str, Any],
        post_apply_state_sha256: str,
    ) -> dict[str, Any]:
        component_hashes = {
            f"{item['name']}_sha256": item["value_sha256"] for item in envelope["components"]
        }
        body = {
            "schema": PRIVATE_ACTION_SNAPSHOT_RESTORE_SCHEMA,
            "handle_sha256": handle_hash,
            "snapshot_sha256": snapshot_sha256,
            "request_sha256": request_sha256,
            "pair_id": pair_id,
            "arm": arm,
            "operation_id": operation_id,
            "restored_at_unix": restored_at_unix,
            "state_components": component_hashes,
            "state_sha256": _digest(component_hashes),
            "post_apply_state_sha256": post_apply_state_sha256,
            "all_bytes_verified_before_return": True,
            "state_applied_before_return": True,
        }
        return {**body, "restore_receipt_sha256": _digest(body)}

    @classmethod
    def _operation_document(
        cls, body: Mapping[str, Any], *, handle_secret: bytes
    ) -> dict[str, Any]:
        return cls._authenticated_document(
            body,
            handle_secret=handle_secret,
            authentication_field="handle_authentication_sha256",
            hash_field="operation_sha256",
        )

    def _read_operation(self, path: Path, *, handle_secret: bytes) -> dict[str, Any]:
        value = self._read_json(path, role="private_snapshot_operation")
        common = {
            "schema",
            "kind",
            "stage",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "handle_authentication_sha256",
            "operation_sha256",
        }
        expected_by_kind_v1 = {
            "restore": common
            | {
                "arm",
                "operation_id",
                "ledger_before_sha256",
                "state_sha256",
            },
            "seal": common | {"target_ledger", "target_handle", "seal_receipt"},
            "erase": common | {"dek_sha256", "erase_files", "erasure_receipt"},
        }
        expected_by_kind_v2 = {
            "restore": common
            | {
                "arm",
                "operation_id",
                "ledger_before_sha256",
                "state_sha256",
                "application_started_at_unix",
                "worker_boot_id",
                "worker_pid",
            },
            "seal": common | {"target_ledger", "target_handle", "seal_receipt"},
            "erase": common
            | {
                "dek_sha256",
                "erase_files",
                "erasure_receipt",
            },
        }
        kind = value.get("kind")
        schema = value.get("schema")
        expected_by_kind = (
            expected_by_kind_v2
            if schema == PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA
            else expected_by_kind_v1
        )
        if (
            schema
            not in {
                PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA_V1,
                PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA,
            }
            or kind not in expected_by_kind
            or set(value) != expected_by_kind[kind]
        ):
            _fail("private_snapshot_operation_fields")
        body = {name: item for name, item in value.items() if name != "operation_sha256"}
        if value["operation_sha256"] != _digest(body):
            _fail("private_snapshot_operation_hash_mismatch")
        if (
            value.get("kind") not in {"restore", "seal", "erase"}
            or (
                value.get("stage") != "prepared"
                and not (
                    schema == PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA
                    and kind == "restore"
                    and value.get("stage") == "application_started"
                )
            )
        ):
            _fail("private_snapshot_operation_invalid")
        self._verify_handle_authentication(
            value,
            hash_field="operation_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_operation",
        )
        for name in ("handle_sha256", "snapshot_sha256", "request_sha256"):
            _sha256(value.get(name), role=f"private_snapshot_operation_{name}")
        if kind == "restore":
            if value.get("arm") not in PAIR_ARMS:
                _fail("private_snapshot_operation_arm_invalid")
            _hex_identifier(
                value.get("operation_id"),
                role="private_snapshot_operation_id",
            )
            _sha256(
                value.get("ledger_before_sha256"),
                role="private_snapshot_operation_ledger",
            )
            _sha256(
                value.get("state_sha256"),
                role="private_snapshot_operation_state",
            )
            if schema == PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA:
                application_started_at = value.get("application_started_at_unix")
                if value["stage"] == "prepared":
                    if application_started_at is not None:
                        _fail("private_snapshot_operation_start_time_invalid")
                else:
                    _positive_int(
                        application_started_at,
                        role="private_snapshot_operation_start_time",
                    )
                _hex_identifier(
                    value.get("worker_boot_id"),
                    role="private_snapshot_operation_worker_boot_id",
                )
                _positive_int(
                    value.get("worker_pid"),
                    role="private_snapshot_operation_worker_pid",
                )
        elif kind == "erase":
            _sha256(
                value.get("dek_sha256"),
                role="private_snapshot_operation_dek",
            )
        return value

    def _recover_operation(
        self,
        handle_hash: str,
        *,
        request_sha256: str,
        handle_secret: bytes,
    ) -> dict[str, Any] | None:
        paths = self._paths_for_handle_hash(handle_hash)
        operation_path = paths["operation"]
        if not self._path_exists(operation_path):
            return None
        operation = self._read_operation(operation_path, handle_secret=handle_secret)
        if (
            operation.get("handle_sha256") != handle_hash
            or operation.get("request_sha256") != request_sha256
        ):
            _fail("private_snapshot_operation_binding_mismatch")
        kind = operation["kind"]
        if kind == "restore":
            handle = self._load_handle(
                handle_hash,
                request_sha256=request_sha256,
                handle_secret=handle_secret,
            )
            ledger = self._load_ledger(
                handle_hash,
                request_sha256=request_sha256,
                snapshot_sha256=handle["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            arm = operation.get("arm")
            use = ledger["uses"].get(arm)
            if use is None:
                if (
                    operation["schema"] == PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA_V1
                    or operation["stage"] == "application_started"
                ):
                    raise UnknownActionStateApplicationError(operation)
                self._unlink(operation_path, sync=True)
                return {"recovered": "rolled_back_precommit_restore"}
            if (
                isinstance(use, dict)
                and use.get("operation_id") == operation.get("operation_id")
                and use.get("post_apply_state_sha256") == operation.get("state_sha256")
            ):
                self._unlink(operation_path, sync=True)
                return {"recovered": "finalized_committed_restore"}
            _fail("private_snapshot_restore_recovery_conflict")
        if kind == "seal":
            target_ledger = operation.get("target_ledger")
            target_handle = operation.get("target_handle")
            if not isinstance(target_ledger, dict) or not isinstance(target_handle, dict):
                _fail("private_snapshot_seal_recovery_invalid")
            self._load_target_ledger(
                target_ledger,
                handle_hash=handle_hash,
                request_sha256=request_sha256,
                snapshot_sha256=operation["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            self._load_target_handle(
                target_handle,
                handle_hash=handle_hash,
                request_sha256=request_sha256,
                snapshot_sha256=operation["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            self._atomic_publish(
                paths["ledger"],
                canonical_json_bytes(target_ledger) + b"\n",
                replace=True,
            )
            self._atomic_publish(
                paths["handle"],
                canonical_json_bytes(target_handle) + b"\n",
                replace=True,
            )
            self._unlink(operation_path, sync=True)
            return {"recovered": "completed_seal"}
        self._complete_erasure(operation, paths=paths)
        return {"recovered": "completed_erasure"}

    def recover(
        self,
        handle: str,
        admission: VerifiedActionStateCaptureRequest,
    ) -> dict[str, Any]:
        """Deterministically resolve a prepared operation after process death."""

        binding = _snapshot_binding(admission)
        handle_hash = self._handle_hash(handle)
        handle_secret = self._handle_secret(handle)
        with self._locked():
            recovered = self._recover_operation(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            return recovered or {"recovered": "none"}

    def restore_and_apply(
        self,
        handle: str,
        admission: VerifiedActionStateCaptureRequest,
        *,
        arm: str,
        restored_at_unix: int,
        apply_state: Callable[[dict[str, Any]], str],
        application_worker_identity: Mapping[str, Any] | None = None,
    ) -> PrivateSnapshotRestore:
        """Install one arm's state, then durably consume that arm exactly once.

        ``apply_state`` runs after a durable ``application_started`` marker and
        before the one-use ledger commits. It must return the resident aggregate
        state hash measured after installation. Once application starts, any
        exception, mismatch, or process death leaves an ``UNKNOWN_APPLICATION``
        quarantine that requires process replacement; it is never retried as a
        pre-apply operation. A committed arm is never blindly replayed.
        """

        binding = _snapshot_binding(admission)
        if arm not in PAIR_ARMS:
            _fail("private_snapshot_arm_invalid")
        if not callable(apply_state):
            _fail("private_snapshot_apply_callback_required")
        restored_at = _positive_int(restored_at_unix, role="private_snapshot_restored_at")
        if restored_at > admission.payload["capture_not_after_unix"]:
            _fail("private_snapshot_restore_after_deadline")
        handle_hash = self._handle_hash(handle)
        handle_secret = self._handle_secret(handle)
        paths = self._paths_for_handle_hash(handle_hash)
        with self._locked():
            self._recover_operation(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            handle_record = self._load_handle(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            if restored_at < handle_record["created_at_unix"]:
                _fail("private_snapshot_restore_time_invalid")
            if handle_record["state"] != "active":
                _fail("private_snapshot_sealed")
            ledger = self._load_ledger(
                handle_hash,
                request_sha256=binding["request_sha256"],
                snapshot_sha256=handle_record["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            if ledger["sealed"] or ledger["uses"][arm] is not None:
                _fail("private_snapshot_arm_already_used")
            envelope = self._load_envelope(
                handle_hash,
                handle_record["snapshot_sha256"],
                binding=binding,
            )
            operation_id = secrets.token_hex(16)
            worker_identity = (
                validate_worker_capture_identity(application_worker_identity)
                if application_worker_identity is not None
                else admission.payload["worker_origin_binding"]["worker_identity"]
            )
            expected_state_sha256 = _digest(
                {f"{item['name']}_sha256": item["value_sha256"] for item in envelope["components"]}
            )
            operation = self._operation_document(
                {
                    "schema": PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA,
                    "kind": "restore",
                    "stage": "prepared",
                    "handle_sha256": handle_hash,
                    "snapshot_sha256": handle_record["snapshot_sha256"],
                    "request_sha256": binding["request_sha256"],
                    "arm": arm,
                    "operation_id": operation_id,
                    "ledger_before_sha256": ledger["ledger_sha256"],
                    "state_sha256": expected_state_sha256,
                    "application_started_at_unix": None,
                    "worker_boot_id": worker_identity["worker_boot_id"],
                    "worker_pid": worker_identity["worker_pid"],
                },
                handle_secret=handle_secret,
            )
            self._atomic_publish(
                paths["operation"],
                canonical_json_bytes(operation) + b"\n",
                replace=False,
            )
            _crash_boundary("restore_after_prepare")
            encryption_key = self._load_dek(
                handle_hash,
                request_sha256=binding["request_sha256"],
                expected_sha256=handle_record["dek_sha256"],
            )
            try:
                state = self._read_private_state(
                    handle_hash,
                    handle_record["snapshot_sha256"],
                    envelope,
                    encryption_key=bytes(encryption_key),
                )
            finally:
                self._zeroize(encryption_key)
            started_body = {
                name: item
                for name, item in operation.items()
                if name not in {"handle_authentication_sha256", "operation_sha256"}
            }
            started_body["stage"] = "application_started"
            started_body["application_started_at_unix"] = restored_at
            operation = self._operation_document(
                started_body,
                handle_secret=handle_secret,
            )
            self._atomic_publish(
                paths["operation"],
                canonical_json_bytes(operation) + b"\n",
                replace=True,
            )
            try:
                _crash_boundary("restore_after_application_started")
                post_apply_state_sha256 = apply_state(state)
                if (
                    not _is_sha256(post_apply_state_sha256)
                    or post_apply_state_sha256 != expected_state_sha256
                ):
                    _fail("private_snapshot_post_apply_state_mismatch")
                _crash_boundary("restore_after_state_apply")
                receipt = self._restore_receipt(
                    handle_hash=handle_hash,
                    snapshot_sha256=handle_record["snapshot_sha256"],
                    request_sha256=binding["request_sha256"],
                    pair_id=binding["pair_id"],
                    arm=arm,
                    operation_id=operation_id,
                    restored_at_unix=restored_at,
                    envelope=envelope,
                    post_apply_state_sha256=post_apply_state_sha256,
                )
                next_ledger_body = {
                    name: item
                    for name, item in ledger.items()
                    if name not in {"ledger_sha256", "handle_authentication_sha256"}
                }
                next_ledger_body["uses"] = dict(ledger["uses"])
                next_ledger_body["uses"][arm] = {
                    "arm": arm,
                    "operation_id": operation_id,
                    "restored_at_unix": restored_at,
                    "post_apply_state_sha256": post_apply_state_sha256,
                    "restore_receipt_sha256": receipt["restore_receipt_sha256"],
                }
                next_ledger_body["sequence"] = ledger["sequence"] + 1
                next_ledger = self._ledger_document(
                    next_ledger_body,
                    handle_secret=handle_secret,
                )
                self._atomic_publish(
                    paths["ledger"],
                    canonical_json_bytes(next_ledger) + b"\n",
                    replace=True,
                )
            except UnknownActionStateApplicationError:
                raise
            # An installer is an injected resident boundary. Any ordinary failure
            # after the durable start marker must quarantine the whole process.
            except Exception as exc:  # noqa: BLE001
                raise UnknownActionStateApplicationError(operation) from exc
            _crash_boundary("restore_after_ledger_commit")
            self._unlink(paths["operation"], sync=True)
            return PrivateSnapshotRestore(state=state, receipt=receipt)

    def seal(
        self,
        handle: str,
        admission: VerifiedActionStateCaptureRequest,
        *,
        sealed_at_unix: int,
    ) -> dict[str, Any]:
        """Seal a pair only after both one-use arm restores committed."""

        binding = _snapshot_binding(admission)
        sealed_at = _positive_int(sealed_at_unix, role="private_snapshot_sealed_at")
        handle_hash = self._handle_hash(handle)
        handle_secret = self._handle_secret(handle)
        paths = self._paths_for_handle_hash(handle_hash)
        with self._locked():
            self._recover_operation(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            handle_record = self._load_handle(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            ledger = self._load_ledger(
                handle_hash,
                request_sha256=binding["request_sha256"],
                snapshot_sha256=handle_record["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            if handle_record["state"] == "sealed":
                _fail("private_snapshot_already_sealed")
            if ledger["sealed"] or any(ledger["uses"][arm] is None for arm in PAIR_ARMS):
                _fail("private_snapshot_pair_incomplete")
            latest_restore = max(ledger["uses"][arm]["restored_at_unix"] for arm in PAIR_ARMS)
            if sealed_at < latest_restore:
                _fail("private_snapshot_seal_time_invalid")
            seal_body = {
                "schema": PRIVATE_ACTION_SNAPSHOT_SEAL_SCHEMA,
                "handle_sha256": handle_hash,
                "snapshot_sha256": handle_record["snapshot_sha256"],
                "request_sha256": binding["request_sha256"],
                "pair_id": binding["pair_id"],
                "sealed_at_unix": sealed_at,
                "arm_restore_receipts": {
                    arm: ledger["uses"][arm]["restore_receipt_sha256"] for arm in PAIR_ARMS
                },
                "both_arms_used_exactly_once": True,
            }
            seal_receipt = {
                **seal_body,
                "seal_receipt_sha256": _digest(seal_body),
            }
            next_ledger_body = {
                name: item
                for name, item in ledger.items()
                if name not in {"ledger_sha256", "handle_authentication_sha256"}
            }
            next_ledger_body.update(
                {
                    "sealed": True,
                    "sealed_at_unix": sealed_at,
                    "sequence": ledger["sequence"] + 1,
                }
            )
            next_handle_body = {
                name: item
                for name, item in handle_record.items()
                if name not in {"record_sha256", "handle_authentication_sha256"}
            }
            next_handle_body.update(
                {
                    "state": "sealed",
                    "seal_receipt_sha256": seal_receipt["seal_receipt_sha256"],
                }
            )
            operation = self._operation_document(
                {
                    "schema": PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA,
                    "kind": "seal",
                    "stage": "prepared",
                    "handle_sha256": handle_hash,
                    "snapshot_sha256": handle_record["snapshot_sha256"],
                    "request_sha256": binding["request_sha256"],
                    "target_ledger": self._ledger_document(
                        next_ledger_body, handle_secret=handle_secret
                    ),
                    "target_handle": self._handle_document(
                        next_handle_body, handle_secret=handle_secret
                    ),
                    "seal_receipt": seal_receipt,
                },
                handle_secret=handle_secret,
            )
            self._atomic_publish(
                paths["operation"],
                canonical_json_bytes(operation) + b"\n",
                replace=False,
            )
            _crash_boundary("seal_after_prepare")
            self._recover_operation(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            return seal_receipt

    def _erasure_manifest(
        self,
        handle_hash: str,
        snapshot_sha256: str,
        envelope: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        snapshot_dir = self._paths_for_handle_hash(handle_hash)["snapshot"]
        for component in envelope["components"]:
            for chunk in component["chunks"]:
                relative = (
                    Path("chunks")
                    / component["name"]
                    / f"{chunk['ordinal']:08d}-{chunk['file_sha256']}.bin"
                )
                files.append(
                    {
                        "relative_path": relative.as_posix(),
                        "file_sha256": chunk["file_sha256"],
                        "byte_count": chunk["byte_count"],
                    }
                )
        envelope_path = snapshot_dir / "envelope.json"
        envelope_raw = self._read_owned(envelope_path, maximum=_MAX_JSON_BYTES)
        files.append(
            {
                "relative_path": "envelope.json",
                "file_sha256": _digest_bytes(envelope_raw),
                "byte_count": len(envelope_raw),
            }
        )
        return files

    def _read_erasure(self, path: Path, *, handle_secret: bytes) -> dict[str, Any]:
        value = self._read_json(path, role="private_snapshot_erasure")
        fields = {
            "schema",
            "handle_sha256",
            "snapshot_sha256",
            "request_sha256",
            "pair_id",
            "seal_receipt_sha256",
            "erased_at_unix",
            "erased_file_count",
            "all_snapshot_files_absent",
            "cryptographic_key_destroyed",
            "ciphertext_namespace_deleted",
            "handle_authentication_sha256",
            "erasure_receipt_sha256",
        }
        if set(value) != fields:
            _fail("private_snapshot_erasure_fields")
        body = {name: item for name, item in value.items() if name != "erasure_receipt_sha256"}
        if (
            value.get("schema") != PRIVATE_ACTION_SNAPSHOT_ERASURE_SCHEMA
            or value["erasure_receipt_sha256"] != _digest(body)
            or value.get("all_snapshot_files_absent") is not True
            or value.get("cryptographic_key_destroyed") is not True
            or value.get("ciphertext_namespace_deleted") is not True
            or type(value.get("erased_file_count")) is not int
            or value["erased_file_count"] <= 0
        ):
            _fail("private_snapshot_erasure_invalid")
        self._verify_handle_authentication(
            value,
            hash_field="erasure_receipt_sha256",
            handle_secret=handle_secret,
            role="private_snapshot_erasure",
        )
        return value

    def _complete_erasure(
        self,
        operation: Mapping[str, Any],
        *,
        paths: Mapping[str, Path],
    ) -> None:
        files = operation.get("erase_files")
        erasure_receipt = operation.get("erasure_receipt")
        snapshot_sha256 = operation.get("snapshot_sha256")
        dek_sha256 = operation.get("dek_sha256")
        if (
            not isinstance(files, list)
            or not isinstance(erasure_receipt, dict)
            or not _is_sha256(snapshot_sha256)
            or not _is_sha256(dek_sha256)
        ):
            _fail("private_snapshot_erasure_operation_invalid")
        key_path = paths["key"]
        if self._path_exists(key_path):
            key = self._load_dek(
                operation["handle_sha256"],
                request_sha256=operation["request_sha256"],
                expected_sha256=dek_sha256,
            )
            if len(key) != 32:
                _fail("private_snapshot_erasure_dek_mismatch")
            self._zeroize(key)
            self._unlink(key_path, sync=True)
        snapshot_dir = paths["snapshot"]
        for item in files:
            if (
                not isinstance(item, dict)
                or set(item) != {"relative_path", "file_sha256", "byte_count"}
                or not _is_sha256(item["file_sha256"])
                or type(item["byte_count"]) is not int
                or item["byte_count"] < 0
            ):
                _fail("private_snapshot_erasure_manifest_invalid")
            relative = Path(item["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                _fail("private_snapshot_erasure_path_invalid")
            target = snapshot_dir / relative
            if self._path_exists(target):
                raw = self._read_owned(target, maximum=max(_MAX_JSON_BYTES, _CHUNK_BYTES))
                if len(raw) != item["byte_count"] or _digest_bytes(raw) != item["file_sha256"]:
                    _fail("private_snapshot_erasure_source_mismatch")
                self._unlink(target)
        chunks_root = snapshot_dir / "chunks"
        if self._path_exists(chunks_root):
            for component_name in self._list_directory(chunks_root):
                component_dir = chunks_root / component_name
                if not stat.S_ISDIR(self._stat_path(component_dir).st_mode):
                    _fail("private_snapshot_erasure_directory_invalid")
                if self._list_directory(component_dir):
                    _fail("private_snapshot_erasure_directory_not_empty")
                self._rmdir(component_dir)
            self._rmdir(chunks_root)
        if self._path_exists(snapshot_dir):
            if self._list_directory(snapshot_dir):
                _fail("private_snapshot_erasure_directory_not_empty")
            self._rmdir(snapshot_dir, sync=True)
        for metadata in (paths["ledger"], paths["handle"], paths["publication"]):
            if self._path_exists(metadata):
                self._assert_private_file(metadata)
                self._unlink(metadata)
        self._atomic_publish(
            paths["tombstone"],
            canonical_json_bytes(erasure_receipt) + b"\n",
            replace=False,
        )
        if self._path_exists(snapshot_dir):
            _fail("private_snapshot_erasure_incomplete")
        if self._path_exists(paths["operation"]):
            self._assert_private_file(paths["operation"])
            self._unlink(paths["operation"])
        self._fsync_directory(paths["operation"].parent)

    def erase(
        self,
        handle: str,
        admission: VerifiedActionStateCaptureRequest,
        *,
        erased_at_unix: int,
    ) -> dict[str, Any]:
        """Erase a sealed snapshot and publish a durable post-erasure receipt."""

        binding = _snapshot_binding(admission)
        erased_at = _positive_int(erased_at_unix, role="private_snapshot_erased_at")
        handle_hash = self._handle_hash(handle)
        handle_secret = self._handle_secret(handle)
        paths = self._paths_for_handle_hash(handle_hash)
        with self._locked():
            self._recover_operation(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            if self._path_exists(paths["tombstone"]):
                return self._read_erasure(paths["tombstone"], handle_secret=handle_secret)
            handle_record = self._load_handle(
                handle_hash,
                request_sha256=binding["request_sha256"],
                handle_secret=handle_secret,
            )
            ledger = self._load_ledger(
                handle_hash,
                request_sha256=binding["request_sha256"],
                snapshot_sha256=handle_record["snapshot_sha256"],
                handle_secret=handle_secret,
            )
            if handle_record["state"] != "sealed" or not ledger["sealed"]:
                _fail("private_snapshot_erase_requires_seal")
            if erased_at < ledger["sealed_at_unix"]:
                _fail("private_snapshot_erasure_time_invalid")
            envelope = self._load_envelope(
                handle_hash,
                handle_record["snapshot_sha256"],
                binding=binding,
            )
            erase_files = self._erasure_manifest(
                handle_hash,
                handle_record["snapshot_sha256"],
                envelope,
            )
            erasure_body = {
                "schema": PRIVATE_ACTION_SNAPSHOT_ERASURE_SCHEMA,
                "handle_sha256": handle_hash,
                "snapshot_sha256": handle_record["snapshot_sha256"],
                "request_sha256": binding["request_sha256"],
                "pair_id": binding["pair_id"],
                "seal_receipt_sha256": handle_record["seal_receipt_sha256"],
                "erased_at_unix": erased_at,
                "erased_file_count": len(erase_files),
                "all_snapshot_files_absent": True,
                "cryptographic_key_destroyed": True,
                "ciphertext_namespace_deleted": True,
            }
            erasure_receipt = self._authenticated_document(
                erasure_body,
                handle_secret=handle_secret,
                authentication_field="handle_authentication_sha256",
                hash_field="erasure_receipt_sha256",
            )
            operation = self._operation_document(
                {
                    "schema": PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA,
                    "kind": "erase",
                    "stage": "prepared",
                    "handle_sha256": handle_hash,
                    "snapshot_sha256": handle_record["snapshot_sha256"],
                    "request_sha256": binding["request_sha256"],
                    "dek_sha256": handle_record["dek_sha256"],
                    "erase_files": erase_files,
                    "erasure_receipt": erasure_receipt,
                },
                handle_secret=handle_secret,
            )
            self._atomic_publish(
                paths["operation"],
                canonical_json_bytes(operation) + b"\n",
                replace=False,
            )
            _crash_boundary("erase_after_prepare")
            self._complete_erasure(operation, paths=paths)
            return erasure_receipt


def _capture_opportunity(
    *,
    action: str,
    episode_step: int,
    schedule_step: int,
    branch_id: str,
    layer_index: int,
    kv_position: int,
    state_sha256: str,
    kv_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": ACTION_STATE_CAPTURE_OPPORTUNITY_SCHEMA,
        "ordinal": 1,
        "action": action,
        "episode_step": _nonnegative_int(episode_step, role="action_state_capture_episode_step"),
        "schedule_step": _nonnegative_int(schedule_step, role="action_state_capture_schedule_step"),
        "branch_id": _identifier(branch_id, role="action_state_capture_branch"),
        "layer_index": _nonnegative_int(layer_index, role="action_state_capture_layer"),
        "kv_position": _nonnegative_int(kv_position, role="action_state_capture_kv_position"),
        "pre_action_state_sha256": state_sha256,
        "pre_action_kv_sha256": kv_sha256,
    }
    return {**body, "opportunity_sha256": _digest(body)}


def _worker_origin(
    body: Mapping[str, Any],
    *,
    worker_private_key: Any,
    expected_key_id: str,
) -> dict[str, Any]:
    raw = _private_key_public_raw(
        worker_private_key, role="action_state_capture_worker_private_key"
    )
    key_id = _digest_bytes(raw)
    if key_id != expected_key_id:
        _fail("action_state_capture_worker_origin_key_mismatch")
    signed = canonical_json_bytes(body)
    signed_sha256 = _digest_bytes(signed)
    try:
        signature = worker_private_key.sign(signed)
    except (AttributeError, TypeError, ValueError):
        _fail("action_state_capture_worker_private_key_invalid")
    origin_body = {
        "schema": ACTION_STATE_CAPTURE_WORKER_ORIGIN_SCHEMA,
        "algorithm": "Ed25519",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": key_id,
        "signed_payload_sha256": signed_sha256,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    return {**origin_body, "origin_sha256": _digest(origin_body)}


def build_action_state_capture_receipt(
    *,
    admission: VerifiedActionStateCaptureRequest,
    publication: PrivateSnapshotPublication,
    worker_private_key: Any,
    captured_at_unix: int,
    latent_reason_request: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    episode_step: int,
    schedule_step: int,
    branch_id: str,
    layer_index: int,
    kv_position: int,
) -> dict[str, Any]:
    """Sign a public no-action/no-decode receipt backed by a private snapshot."""

    if (
        not isinstance(admission, VerifiedActionStateCaptureRequest)
        or not isinstance(publication, PrivateSnapshotPublication)
        or publication.request_sha256 != admission.request_sha256
    ):
        _fail("action_state_capture_publication_binding_mismatch")
    payload = admission.payload
    captured_at = _positive_int(captured_at_unix, role="action_state_capture_captured_at")
    if not (admission.signed_at_unix <= captured_at <= payload["capture_not_after_unix"]):
        _fail("action_state_capture_time_invalid")
    state_components = publication.state_components
    if set(state_components) != set(STATE_COMPONENT_NAMES) or any(
        not _is_sha256(value) for value in state_components.values()
    ):
        _fail("action_state_capture_state_components_invalid")
    model_hash = _digest(
        _normalized_mapping(model_identity, role="action_state_capture_model_identity")
    )
    execution_hash = _digest(
        _normalized_mapping(execution_identity, role="action_state_capture_execution_identity")
    )
    request_hash = normalized_latent_reason_request_sha256(latent_reason_request)
    if (
        model_hash != payload["model_identity_sha256"]
        or execution_hash != payload["execution_identity_sha256"]
        or request_hash != payload["latent_reason_request_sha256"]
    ):
        _fail("action_state_capture_runtime_request_drift")
    runtime_hash = _digest(
        _validate_bound_runtime_identity(runtime_identity)
    )
    state_sha256 = _digest(state_components)
    opportunity = _capture_opportunity(
        action=payload["action"],
        episode_step=episode_step,
        schedule_step=schedule_step,
        branch_id=branch_id,
        layer_index=layer_index,
        kv_position=kv_position,
        state_sha256=state_sha256,
        kv_sha256=state_components["kv_cache_sha256"],
    )
    body = {
        "schema": ACTION_STATE_CAPTURE_RECEIPT_SCHEMA,
        "request_sha256": admission.request_sha256,
        "capture_id": payload["capture_id"],
        "captured_at_unix": captured_at,
        **{name: payload[name] for name in _PUBLIC_BINDING_FIELDS},
        "runtime_identity_sha256": runtime_hash,
        "private_snapshot_envelope_sha256": publication.snapshot_sha256,
        "state_components": state_components,
        "state_sha256": state_sha256,
        "component_observation_owners": dict(_COMPONENT_OBSERVATION_OWNERS),
        "first_action_opportunity": opportunity,
        "capture_boundary": "immediately_before_first_action_opportunity_v1",
        "action_executed": False,
        "action_trace_count": 0,
        "decode_started": False,
        "decoded_token_count": 0,
        "output_present": False,
        "output_sha256": None,
        "output_byte_count": 0,
    }
    origin = _worker_origin(
        body,
        worker_private_key=worker_private_key,
        expected_key_id=payload["worker_origin_binding"]["worker_identity"]["key_id"],
    )
    complete = {**body, "worker_origin": origin}
    return {**complete, "receipt_sha256": _digest(complete)}


def _verify_worker_origin(
    body: Mapping[str, Any],
    value: Any,
    *,
    expected_key_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _WORKER_ORIGIN_FIELDS:
        _fail("action_state_capture_worker_origin_fields")
    origin = _normalized_mapping(value, role="action_state_capture_worker_origin")
    origin_body = {name: origin[name] for name in _WORKER_ORIGIN_FIELDS - {"origin_sha256"}}
    if (
        origin["schema"] != ACTION_STATE_CAPTURE_WORKER_ORIGIN_SCHEMA
        or origin["algorithm"] != "Ed25519"
        or origin["origin_sha256"] != _digest(origin_body)
        or origin["key_id"] != expected_key_id
    ):
        _fail("action_state_capture_worker_origin_invalid")
    raw = _public_key_raw(
        origin["public_key_b64"],
        role="action_state_capture_worker_origin_key",
    )
    if _digest_bytes(raw) != origin["key_id"]:
        _fail("action_state_capture_worker_origin_key_mismatch")
    signed = canonical_json_bytes(body)
    if origin["signed_payload_sha256"] != _digest_bytes(signed):
        _fail("action_state_capture_worker_origin_payload_mismatch")
    try:
        signature = base64.b64decode(origin["signature_b64"], validate=True)
    except (ValueError, binascii.Error):
        _fail("action_state_capture_worker_origin_signature_invalid")
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    try:
        Ed25519PublicKey.from_public_bytes(raw).verify(signature, signed)
    except (InvalidSignature, ValueError):
        _fail("action_state_capture_worker_origin_signature_invalid")
    return origin


def _validate_action_state_capture_receipt(
    value: Any,
    *,
    request: Mapping[str, Any],
    publication: PrivateSnapshotPublication | None,
    trusted_root_public_key_pem: bytes,
    expected_supervisor_public_key: Any,
    latent_reason_request: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    expected_campaign_design_sha256: str,
) -> dict[str, Any]:
    """Historically replay one worker-origin capture receipt."""

    admission = replay_action_state_capture_request(
        request,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        expected_supervisor_public_key=expected_supervisor_public_key,
    )
    if publication is not None and (
        not isinstance(publication, PrivateSnapshotPublication)
        or publication.request_sha256 != admission.request_sha256
    ):
        _fail("action_state_capture_publication_binding_mismatch")
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        _fail("action_state_capture_receipt_fields")
    receipt = _normalized_mapping(value, role="action_state_capture_receipt")
    if receipt.get("schema") != ACTION_STATE_CAPTURE_RECEIPT_SCHEMA:
        _fail("action_state_capture_receipt_schema")
    complete = {name: receipt[name] for name in _RECEIPT_FIELDS - {"receipt_sha256"}}
    if receipt["receipt_sha256"] != _digest(complete):
        _fail("action_state_capture_receipt_hash_mismatch")
    body = {name: receipt[name] for name in _RECEIPT_FIELDS - {"worker_origin", "receipt_sha256"}}
    payload = admission.payload
    if (
        not _is_sha256(expected_campaign_design_sha256)
        or payload["campaign_design_sha256"]
        != expected_campaign_design_sha256
        or receipt["request_sha256"] != admission.request_sha256
        or receipt["capture_id"] != payload["capture_id"]
        or any(receipt[name] != payload[name] for name in _PUBLIC_BINDING_FIELDS)
        or type(receipt.get("captured_at_unix")) is not int
        or not (
            admission.signed_at_unix
            <= receipt["captured_at_unix"]
            <= payload["capture_not_after_unix"]
        )
    ):
        _fail("action_state_capture_receipt_request_mismatch")
    if (
        _digest(_normalized_mapping(model_identity, role="action_state_capture_model_identity"))
        != payload["model_identity_sha256"]
        or _digest(
            _normalized_mapping(
                execution_identity,
                role="action_state_capture_execution_identity",
            )
        )
        != payload["execution_identity_sha256"]
        or _digest(_validate_bound_runtime_identity(runtime_identity))
        != receipt["runtime_identity_sha256"]
        or normalized_latent_reason_request_sha256(latent_reason_request)
        != payload["latent_reason_request_sha256"]
    ):
        _fail("action_state_capture_receipt_runtime_request_mismatch")
    components = receipt.get("state_components")
    owners = receipt.get("component_observation_owners")
    if (
        not isinstance(components, dict)
        or set(components) != set(STATE_COMPONENT_NAMES)
        or any(not _is_sha256(item) for item in components.values())
        or receipt.get("state_sha256") != _digest(components)
        or not isinstance(owners, dict)
        or owners != _COMPONENT_OBSERVATION_OWNERS
        or not _is_sha256(receipt.get("private_snapshot_envelope_sha256"))
        or (
            publication is not None
            and (
                receipt["private_snapshot_envelope_sha256"] != publication.snapshot_sha256
                or components != publication.state_components
            )
        )
    ):
        _fail("action_state_capture_receipt_state_invalid")
    opportunity = receipt.get("first_action_opportunity")
    if not isinstance(opportunity, dict) or set(opportunity) != _OPPORTUNITY_FIELDS:
        _fail("action_state_capture_opportunity_fields")
    opportunity_body = {
        name: opportunity[name] for name in _OPPORTUNITY_FIELDS - {"opportunity_sha256"}
    }
    if (
        opportunity["schema"] != ACTION_STATE_CAPTURE_OPPORTUNITY_SCHEMA
        or type(opportunity.get("ordinal")) is not int
        or opportunity["ordinal"] != 1
        or opportunity["action"] != payload["action"]
        or opportunity["pre_action_state_sha256"] != receipt["state_sha256"]
        or opportunity["pre_action_kv_sha256"] != components["kv_cache_sha256"]
        or opportunity["opportunity_sha256"] != _digest(opportunity_body)
    ):
        _fail("action_state_capture_opportunity_invalid")
    for name in ("episode_step", "schedule_step", "layer_index", "kv_position"):
        _nonnegative_int(opportunity.get(name), role=f"action_state_capture_{name}")
    _identifier(
        opportunity.get("branch_id"),
        role="action_state_capture_branch",
    )
    if (
        receipt.get("capture_boundary") != "immediately_before_first_action_opportunity_v1"
        or receipt.get("action_executed") is not False
        or type(receipt.get("action_trace_count")) is not int
        or receipt["action_trace_count"] != 0
        or receipt.get("decode_started") is not False
        or type(receipt.get("decoded_token_count")) is not int
        or receipt["decoded_token_count"] != 0
        or receipt.get("output_present") is not False
        or receipt.get("output_sha256") is not None
        or type(receipt.get("output_byte_count")) is not int
        or receipt["output_byte_count"] != 0
    ):
        _fail("action_state_capture_action_or_output_leakage")
    _verify_worker_origin(
        body,
        receipt["worker_origin"],
        expected_key_id=payload["worker_origin_binding"]["worker_identity"]["key_id"],
    )
    return receipt


def validate_action_state_capture_receipt_public(
    value: Any,
    *,
    request: Mapping[str, Any],
    trusted_root_public_key_pem: bytes,
    expected_supervisor_public_key: Any,
    latent_reason_request: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    expected_campaign_design_sha256: str,
) -> dict[str, Any]:
    """Replay a public capture receipt without private snapshot custody.

    This verifies campaign authority, request binding, worker origin,
    component commitments, the first-opportunity boundary declaration, and
    the no-action/no-output assertions. It deliberately cannot prove that a
    private store still possesses the committed bytes; that stronger local
    check is provided by :func:`validate_action_state_capture_receipt`.
    """

    return _validate_action_state_capture_receipt(
        value,
        request=request,
        publication=None,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        expected_supervisor_public_key=expected_supervisor_public_key,
        latent_reason_request=latent_reason_request,
        model_identity=model_identity,
        execution_identity=execution_identity,
        runtime_identity=runtime_identity,
        expected_campaign_design_sha256=expected_campaign_design_sha256,
    )


def validate_action_state_capture_receipt(
    value: Any,
    *,
    request: Mapping[str, Any],
    publication: PrivateSnapshotPublication,
    trusted_root_public_key_pem: bytes,
    expected_supervisor_public_key: Any,
    latent_reason_request: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    expected_campaign_design_sha256: str,
) -> dict[str, Any]:
    """Replay a capture receipt and bind it to local private custody."""

    if not isinstance(publication, PrivateSnapshotPublication):
        _fail("action_state_capture_publication_binding_mismatch")
    return _validate_action_state_capture_receipt(
        value,
        request=request,
        publication=publication,
        trusted_root_public_key_pem=trusted_root_public_key_pem,
        expected_supervisor_public_key=expected_supervisor_public_key,
        latent_reason_request=latent_reason_request,
        model_identity=model_identity,
        execution_identity=execution_identity,
        runtime_identity=runtime_identity,
        expected_campaign_design_sha256=expected_campaign_design_sha256,
    )


__all__ = [
    "ACTION_STATE_CAPTURE_OPPORTUNITY_SCHEMA",
    "ACTION_STATE_CAPTURE_RECEIPT_SCHEMA",
    "ACTION_STATE_CAPTURE_REQUEST_PAYLOAD_SCHEMA",
    "ACTION_STATE_CAPTURE_REQUEST_SCHEMA",
    "ACTION_STATE_CAPTURE_WORKER_ORIGIN_SCHEMA",
    "CONTROL_ARM",
    "PAIR_ARMS",
    "PRIVATE_ACTION_SNAPSHOT_CHUNK_AAD_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_BINDING_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_ENVELOPE_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_ERASURE_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_HANDLE_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_KEY_CONTEXT_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_LEDGER_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_OPERATION_SCHEMA_V1",
    "PRIVATE_ACTION_SNAPSHOT_PUBLICATION_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_RESTORE_SCHEMA",
    "PRIVATE_ACTION_SNAPSHOT_SEAL_SCHEMA",
    "STATE_COMPONENT_NAMES",
    "TREATMENT_ARM",
    "ActionStateCaptureError",
    "UnknownActionStateApplicationError",
    "PrivateActionSnapshotStore",
    "PrivateSnapshotPublication",
    "PrivateSnapshotRestore",
    "VerifiedActionStateCaptureRequest",
    "action_state_capture_request_payload",
    "admit_action_state_capture_request",
    "build_action_state_capture_receipt",
    "build_action_state_capture_request",
    "normalized_latent_reason_request_sha256",
    "replay_action_state_capture_request",
    "validate_action_state_capture_receipt",
    "validate_action_state_capture_receipt_public",
]
