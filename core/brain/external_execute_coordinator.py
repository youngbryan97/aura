"""Durable host-side coordinator for RLC-selected external effects.

The MLX worker can request an already-admitted effect but cannot execute it.
This store makes that handoff crash-legible and duplicate-resistant:

    PREPARED -> DECIDED -> DISPATCHING -> terminal

Any process restart that observes DISPATCHING converts it to UNKNOWN_EFFECT.
Aura must reconcile that state instead of blindly repeating a potentially
completed external action.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.external_execution import (
    validate_external_execution_handoff,
    validate_external_execution_offer,
    validate_external_execution_readiness,
)
from core.config import DATA_DIR
from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.diagnostics_bundle import redact_value
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lease import Identity, _holder_is_live

EXTERNAL_EXECUTE_TRANSACTION_SCHEMA = "aura.rlc.external_execute_transaction.v3"
_EXTERNAL_EXECUTE_TRANSACTION_SCHEMA_V2 = "aura.rlc.external_execute_transaction.v2"
_MAX_TRANSACTION_BYTES = 8 * 1024 * 1024
_MAX_REPLAY_BYTES = 128_000
_DISPATCH_LEASE_DURATION_S = 120.0
_ABANDONED_ATTEMPT_TTL_S = 900.0
_MAX_ABANDONED_ATTEMPTS = 2048
_TERMINAL = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "FAILED_PRE_DISPATCH",
        "ABSTAINED",
        "UNKNOWN_EFFECT",
    }
)
_BYPASS_REASONS = frozenset(
    {
        "disabled:AURA_PREACTION_RLC=0",
        "latent_cortex_absent",
        "availability_failure:disabled:AURA_LATENT_CORTEX=0",
        "availability_failure:generation_gate_busy",
        "availability_failure:no_resident_model",
        "availability_failure:client_error:OSError",
        "availability_failure:client_error:TimeoutError",
        "availability_failure:client_unavailable:ImportError",
        "availability_failure:client_unavailable:OSError",
        "availability_failure:client_unavailable:TimeoutError",
        "availability_failure:client_unavailable:DependencyUnavailable",
        "availability_failure:client_unavailable:ModelUnavailable",
        "availability_failure:generation_lease_unavailable:ImportError",
        "availability_failure:generation_lease_unavailable:OSError",
        "availability_failure:generation_lease_unavailable:TimeoutError",
        "availability_failure:generation_lease_unavailable:DependencyUnavailable",
    }
)


#: An availability failure is not a decision, and that distinction is what
#: eligibility is really about.
#:
#: The set below enumerates exact strings, and `preaction_cortex` composes its
#: reasons — `availability_failure:{type(exc).__name__}` — so any exception
#: class nobody had listed produced a reason that was not eligible, and an
#: INELIGIBLE BYPASS REFUSES THE WHOLE ACTION. Live 2026-08-18 a user-requested
#: browser task died here after clearing every authority gate before it.
#:
#: Eligibility is therefore by class. "The rehearsal could not run" is exactly
#: what a bypass is for, however the unavailability spelled itself. An
#: `episode_integrity_*` reason is the opposite — the rehearsal ran and refused
#: — and must never bypass, because that is the case the allowlist exists to
#: stop from masquerading as a decision.
_BYPASS_REASON_PREFIXES = ("availability_failure:",)


def _bypass_reason_is_eligible(reason: str) -> bool:
    """Whether this reason describes an unavailable rehearsal rather than a verdict."""
    if reason in _BYPASS_REASONS:
        return True
    return any(reason.startswith(prefix) for prefix in _BYPASS_REASON_PREFIXES)


_TRANSACTION_FIELDS = frozenset(
    {
        "schema",
        "action_id",
        "request_digest",
        "offer_sha256",
        "offer",
        "state",
        "decision_source",
        "handoff",
        "readiness",
        "cognitive_action_trace",
        "action_policy_evidence",
        "action_policy_receipt",
        "action_intervention",
        "executors",
        "runtime_operation",
        "dispatch_owner",
        "dispatch_authorization_receipt_id",
        "result",
        "terminal_reason",
        "created_at_unix",
        "updated_at_unix",
    }
)
_TRANSACTION_FIELDS_V2 = _TRANSACTION_FIELDS - {"action_intervention"}
_HANDOFF_FIELDS = frozenset(
    {
        "schema",
        "offer_sha256",
        "requested",
        "decision_sha256",
        "step_index",
        "mode",
        "outcome",
        "trace_sha256",
        "handoff_sha256",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "ok",
        "status",
        "error",
        "transport_succeeded",
        "effect_verified",
        "manual_reconciliation_required",
        "retry_safe",
        "post_action_receipt_id",
        "replay_payload",
        "replay_payload_sha256",
    }
)
_SENSITIVE_RESULT_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "session_id",
    "token",
)
_REPLAY_TEXT_FIELDS = frozenset(
    {
        "action_expectation",
        "created_path",
        "error",
        "observed",
        "output",
        "path",
        "record_id",
        "status",
        "stderr",
        "stdout",
        "text",
        "url",
        "verification_evidence",
    }
)


class ExternalExecutionInProgressError(RuntimeError):
    """A duplicate request observed a live dispatch owner."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transaction_path(root: Path, action_id: str) -> Path:
    return root / f"{hashlib.sha256(action_id.encode('utf-8')).hexdigest()}.json"


def _expected_post_action_receipt_id(action_id: str, request_digest: str) -> str:
    identity = f"{action_id}\0{request_digest}".encode()
    return f"post-external-{hashlib.sha256(identity).hexdigest()[:32]}"


def _validate_post_action_receipt_contract(
    receipt_contract: Mapping[str, Any],
    *,
    offer: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    from core.runtime.post_action_receipt import PostActionReceipt

    try:
        receipt = PostActionReceipt(**dict(receipt_contract))
    except (TypeError, ValueError) as exc:
        raise ValueError("post-action receipt contract is invalid") from exc
    contract = receipt.to_dict()
    expected_id = _expected_post_action_receipt_id(
        offer["action_id"],
        offer["request_digest"],
    )
    if (
        receipt.receipt_id != expected_id
        or receipt.action_id != offer["action_id"]
        or receipt.request_digest != offer["request_digest"]
        or receipt.will_receipt_id != offer["will_receipt_id"]
        or receipt.executor_name != offer["action_name"]
        or receipt.domain != offer["domain"]
        or not receipt.output_hash.startswith("sha256:")
        or isinstance(receipt.timestamp, bool)
        or not isinstance(receipt.timestamp, (int, float))
        or not math.isfinite(float(receipt.timestamp))
        or float(receipt.timestamp) < 0.0
    ):
        raise ValueError("post-action receipt contract identity differs")
    return contract, _canonical_sha256(contract)


def _validate_post_action_receipt_outcome(
    receipt: Mapping[str, Any],
    *,
    transaction: Mapping[str, Any],
) -> None:
    result = transaction.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("terminal transaction result is unavailable")
    verified_unknown_effect = (
        transaction.get("state") == "UNKNOWN_EFFECT"
        and result.get("effect_verified") is False
        and receipt.get("effect_verified") is True
        and receipt.get("transport_succeeded") is True
    )
    if verified_unknown_effect:
        return
    if (
        receipt.get("status") != result.get("status")
        or str(receipt.get("error_status") or "")[:500] != result.get("error")
        or receipt.get("effect_verified") is not result.get("effect_verified")
        or receipt.get("transport_succeeded") is not result.get("transport_succeeded")
        or receipt.get("retry_safe") is not result.get("retry_safe")
        or receipt.get("manual_reconciliation_required")
        is not result.get("manual_reconciliation_required")
    ):
        raise ValueError("post-action receipt outcome contradicts terminal transaction")


def _bounded_result(result: Mapping[str, Any]) -> dict[str, Any]:
    replay_payload = _bounded_replay_mapping(result)
    return {
        "ok": result.get("ok") is True,
        "status": str(result.get("status") or "")[:80],
        "error": str(result.get("error") or "")[:500],
        "transport_succeeded": result.get("transport_succeeded") is True,
        "effect_verified": result.get("effect_verified") is True,
        "manual_reconciliation_required": bool(result.get("manual_reconciliation_required")),
        "retry_safe": result.get("retry_safe") is True,
        "post_action_receipt_id": str(result.get("post_action_receipt_id") or "")[:192],
        "replay_payload": replay_payload,
        "replay_payload_sha256": _canonical_sha256(replay_payload),
    }


def _bounded_replay_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    state = {"items": 0, "truncated": False}

    def bound(
        item: Any,
        *,
        key: str = "",
        depth: int = 0,
        preserve_text: bool = True,
    ) -> Any:
        state["items"] += 1
        if depth > 8 or state["items"] > 512:
            state["truncated"] = True
            return "<truncated>"
        if any(marker in key.casefold() for marker in _SENSITIVE_RESULT_MARKERS):
            return "[REDACTED]"
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else str(item)
        if isinstance(item, str):
            if not preserve_text:
                encoded = item.encode("utf-8")
                return {
                    "type": "redacted_text_digest",
                    "characters": len(item),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            if len(item) <= 2048:
                return item
            state["truncated"] = True
            return {
                "prefix": item[:2048],
                "characters": len(item),
                "sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(),
            }
        if isinstance(item, (bytes, bytearray, memoryview)):
            payload = bytes(item)
            return {
                "type": "bytes",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for index, (raw_key, child) in enumerate(item.items()):
                if index >= 128:
                    state["truncated"] = True
                    break
                child_key = str(raw_key)[:240]
                if child_key == "external_execution_transaction":
                    continue
                result[child_key] = bound(
                    child,
                    key=child_key,
                    depth=depth + 1,
                    preserve_text=(depth > 0 or child_key in _REPLAY_TEXT_FIELDS),
                )
            return result
        if isinstance(item, (list, tuple, set, frozenset)):
            values = list(item)
            if len(values) > 128:
                state["truncated"] = True
            return [bound(child, depth=depth + 1) for child in values[:128]]
        return f"<{type(item).__module__}.{type(item).__qualname__}>"

    scrubbed = redact_value(dict(value))
    bounded = bound(scrubbed)
    result = bounded if isinstance(bounded, dict) else {"value": bounded}
    if state["truncated"]:
        result = {**result, "_truncated": True}
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_REPLAY_BYTES:
        summary = {
            key: result[key]
            for key in (
                "ok",
                "status",
                "error",
                "transport_succeeded",
                "effect_verified",
                "manual_reconciliation_required",
                "retry_safe",
                "receipt_persisted",
                "post_action_receipt_pending",
                "post_action_receipt_attempt_id",
                "_post_action_recovery_contract",
            )
            if key in result
        }
        result = {
            **summary,
            "_truncated": True,
            "_payload_bytes": len(encoded),
            "_payload_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return result


def _validate_transaction(value: Mapping[str, Any]) -> dict[str, Any]:
    if frozenset(value) != _TRANSACTION_FIELDS:
        raise ValueError("external execution transaction fields differ")
    offer = validate_external_execution_offer(value.get("offer"))
    if (
        value.get("action_id") != offer["action_id"]
        or value.get("request_digest") != offer["request_digest"]
        or value.get("offer_sha256") != offer["offer_sha256"]
    ):
        raise ValueError("external execution transaction offer identity differs")
    state = str(value.get("state") or "")
    if state not in {"PREPARED", "DECIDED", "DISPATCHING", *_TERMINAL}:
        raise ValueError("external execution transaction state is invalid")
    decision_source = str(value.get("decision_source") or "")
    if decision_source not in {"", "rlc", "host_fallback"}:
        raise ValueError("external execution decision source is invalid")
    handoff = value.get("handoff")
    readiness = value.get("readiness")
    cognitive_action_trace = value.get("cognitive_action_trace")
    action_policy_evidence = value.get("action_policy_evidence")
    action_policy_receipt = value.get("action_policy_receipt")
    action_intervention = value.get("action_intervention")
    executors = value.get("executors")
    runtime_operation = value.get("runtime_operation")
    dispatch_owner = value.get("dispatch_owner")
    dispatch_authorization_receipt_id = value.get("dispatch_authorization_receipt_id")
    result = value.get("result")
    if (
        not isinstance(handoff, Mapping)
        or not isinstance(readiness, Mapping)
        or not isinstance(cognitive_action_trace, list)
        or not isinstance(action_policy_evidence, Mapping)
        or not isinstance(action_policy_receipt, Mapping)
        or not isinstance(action_intervention, Mapping)
        or not isinstance(executors, list)
        or not isinstance(runtime_operation, Mapping)
        or not isinstance(dispatch_owner, Mapping)
        or not isinstance(result, Mapping)
    ):
        raise ValueError("external execution transaction payload is invalid")
    if not isinstance(dispatch_authorization_receipt_id, str):
        raise ValueError("external execution dispatch authorization is invalid")
    if dispatch_authorization_receipt_id:
        _bounded_authorization = dispatch_authorization_receipt_id.strip()
        if (
            _bounded_authorization != dispatch_authorization_receipt_id
            or len(_bounded_authorization) > 192
            or any(ord(character) < 32 for character in _bounded_authorization)
        ):
            raise ValueError("external execution dispatch authorization is invalid")
    if handoff:
        if frozenset(handoff) != _HANDOFF_FIELDS:
            raise ValueError("external execution stored handoff fields differ")
        handoff_payload = {key: handoff[key] for key in handoff if key != "handoff_sha256"}
        if handoff.get("offer_sha256") != offer["offer_sha256"] or handoff.get(
            "handoff_sha256"
        ) != _canonical_sha256(handoff_payload):
            raise ValueError("external execution stored handoff integrity failed")
    if readiness:
        validate_external_execution_readiness(readiness, offer=offer)
    if result and frozenset(result) != _RESULT_FIELDS:
        raise ValueError("external execution stored result fields differ")
    if state == "PREPARED" and (
        decision_source
        or handoff
        or readiness
        or cognitive_action_trace
        or action_policy_evidence
        or action_policy_receipt
        or action_intervention
        or executors
        or runtime_operation
        or dispatch_owner
        or dispatch_authorization_receipt_id
        or result
    ):
        raise ValueError("prepared external execution carries premature outcome")
    if state == "DECIDED" and decision_source == "":
        raise ValueError("decided external execution lacks a decision source")
    if decision_source == "rlc":
        if (
            not handoff
            or handoff.get("requested") is not (state != "ABSTAINED")
            or not cognitive_action_trace
            or not action_policy_evidence
            or not action_policy_receipt
            or not executors
            or not runtime_operation
        ):
            raise ValueError("RLC external execution handoff differs from state")
        if not readiness:
            raise ValueError("RLC external execution lacks readiness evidence")
        from core.brain.llm.latent_cortex.epistemic_runtime import (
            validate_completed_runtime_operation_receipt,
        )
        from core.brain.llm.latent_cortex.epistemic_state import OperationKind
        from core.brain.llm.latent_cortex.value_of_computation import (
            validate_action_trace,
            validate_evidence_snapshot,
        )

        normalized_evidence = validate_evidence_snapshot(action_policy_evidence)
        normalized_intervention = None
        if action_intervention:
            from core.brain.llm.latent_cortex.action_intervention import (
                validate_action_intervention,
            )

            normalized_intervention = validate_action_intervention(
                action_intervention,
                require_current_policy=False,
            )
        try:
            normalized_executors = tuple(OperationKind(item) for item in executors)
        except (TypeError, ValueError) as exc:
            raise ValueError("external execution executor inventory is invalid") from exc
        if (
            not normalized_executors
            or len(set(normalized_executors)) != len(normalized_executors)
            or OperationKind.EXECUTE not in normalized_executors
        ):
            raise ValueError("external execution executor inventory is invalid")
        normalized_trace = validate_action_trace(
            cognitive_action_trace,
            evidence_snapshot=normalized_evidence,
            executors=normalized_executors,
            action_intervention=normalized_intervention,
        )
        policy_fields = {
            "schema",
            "bucket",
            "snapshot_sha256",
            "active",
            "executors",
            "actions_selected",
            "checked_transitions",
            "selected_actions",
        }
        if normalized_intervention is not None:
            policy_fields.add("calibration_intervention")
        checked_transitions = sum(
            int(row["transition"]["checked"]) for row in normalized_trace["rows"]
        )
        if (
            set(action_policy_receipt) != policy_fields
            or action_policy_receipt.get("schema") != normalized_evidence["schema"]
            or action_policy_receipt.get("bucket") != normalized_evidence["bucket"]
            or action_policy_receipt.get("snapshot_sha256")
            != normalized_evidence["snapshot_sha256"]
            or action_policy_receipt.get("active") is not True
            or action_policy_receipt.get("executors") != list(executors)
            or action_policy_receipt.get("actions_selected") != len(cognitive_action_trace)
            or action_policy_receipt.get("selected_actions") != normalized_trace["selected_actions"]
            or action_policy_receipt.get("checked_transitions") != checked_transitions
        ):
            raise ValueError("external execution policy summary differs")
        if normalized_intervention is not None:
            from core.brain.llm.latent_cortex.action_intervention import (
                validate_action_intervention_receipt,
            )

            validate_action_intervention_receipt(
                action_policy_receipt["calibration_intervention"],
                intervention=normalized_intervention,
                cognitive_action_trace=cognitive_action_trace,
            )
        validate_external_execution_handoff(
            handoff,
            offer=offer,
            cognitive_action_trace=cognitive_action_trace,
        )
        validate_completed_runtime_operation_receipt(
            runtime_operation,
            external_execution_offer=offer,
            action_policy_evidence=normalized_evidence,
            action_policy_receipt=action_policy_receipt,
            cognitive_action_trace=cognitive_action_trace,
            action_intervention=normalized_intervention,
        )
    if decision_source == "host_fallback" and (
        handoff
        or cognitive_action_trace
        or action_policy_evidence
        or action_policy_receipt
        or action_intervention
        or executors
        or runtime_operation
    ):
        raise ValueError("host fallback cannot claim RLC execution evidence")
    if state == "ABSTAINED" and (
        decision_source != "rlc" or dispatch_owner or dispatch_authorization_receipt_id
    ):
        raise ValueError("abstained external execution state is inconsistent")
    if state in {"DISPATCHING", "SUCCEEDED", "FAILED", "UNKNOWN_EFFECT"}:
        if frozenset(dispatch_owner) != {
            "attempt_id",
            "identity",
            "task_id",
            "lease_renewed_at",
            "lease_duration_s",
        } or not isinstance(dispatch_owner.get("identity"), Mapping):
            raise ValueError("dispatching external execution owner is invalid")
        attempt_id = dispatch_owner.get("attempt_id")
        identity = dispatch_owner.get("identity")
        if (
            not isinstance(attempt_id, str)
            or len(attempt_id) != 32
            or any(character not in "0123456789abcdef" for character in attempt_id)
            or frozenset(identity) != {"holder", "pid", "boot_id", "host", "started_at"}
            or not isinstance(identity.get("holder"), str)
            or not identity["holder"]
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or not isinstance(identity.get("boot_id"), str)
            or not identity["boot_id"]
            or not isinstance(identity.get("host"), str)
            or not identity["host"]
            or isinstance(identity.get("started_at"), bool)
            or not isinstance(identity.get("started_at"), (int, float))
            or not math.isfinite(float(identity["started_at"]))
            or float(identity["started_at"]) <= 0.0
            or not isinstance(dispatch_owner.get("task_id"), str)
            or not dispatch_owner["task_id"]
            or len(dispatch_owner["task_id"]) > 192
            or any(ord(character) < 32 for character in dispatch_owner["task_id"])
            or isinstance(dispatch_owner.get("lease_renewed_at"), bool)
            or not isinstance(
                dispatch_owner.get("lease_renewed_at"),
                (int, float),
            )
            or not math.isfinite(float(dispatch_owner["lease_renewed_at"]))
            or float(dispatch_owner["lease_renewed_at"]) <= 0.0
            or isinstance(dispatch_owner.get("lease_duration_s"), bool)
            or not isinstance(
                dispatch_owner.get("lease_duration_s"),
                (int, float),
            )
            or not math.isfinite(float(dispatch_owner["lease_duration_s"]))
            or not 10.0 <= float(dispatch_owner["lease_duration_s"]) <= 900.0
        ):
            raise ValueError("dispatching external execution identity is invalid")
        if not dispatch_authorization_receipt_id:
            raise ValueError("external execution lacks dispatch authorization")
    elif state not in {"SUCCEEDED", "FAILED", "UNKNOWN_EFFECT"} and dispatch_owner:
        raise ValueError("external execution carries a premature dispatch owner")
    if state in {"PREPARED", "DECIDED", "ABSTAINED"} and (dispatch_authorization_receipt_id):
        raise ValueError("external execution carries premature dispatch authorization")
    if state in _TERMINAL:
        if not result:
            raise ValueError("terminal external execution lacks a result")
    elif result:
        raise ValueError("nonterminal external execution carries a result")
    if result:
        transport = result.get("transport_succeeded")
        if transport not in {True, False, None}:
            raise ValueError("external execution transport result is invalid")
        for name in (
            "ok",
            "effect_verified",
            "manual_reconciliation_required",
            "retry_safe",
        ):
            if type(result.get(name)) is not bool:
                raise ValueError(f"external execution {name} result is invalid")
        replay_payload = result.get("replay_payload")
        if not isinstance(replay_payload, Mapping) or result.get(
            "replay_payload_sha256"
        ) != _canonical_sha256(replay_payload):
            raise ValueError("external execution replay payload integrity failed")
        if state == "SUCCEEDED" and result.get("effect_verified") is not True:
            raise ValueError("successful external execution lacks verified effect")
        if state == "FAILED" and (
            result.get("transport_succeeded") is not False
            or result.get("effect_verified") is not False
            or result.get("retry_safe") is not True
            or result.get("manual_reconciliation_required") is not False
        ):
            raise ValueError("failed external execution is not proven retry-safe")
        if state == "FAILED_PRE_DISPATCH" and (
            result.get("transport_succeeded") is not False
            or result.get("effect_verified") is not False
            or result.get("manual_reconciliation_required") is not False
        ):
            raise ValueError("pre-dispatch external execution failure is inconsistent")
        if state == "UNKNOWN_EFFECT" and (
            result.get("effect_verified") is not False
            or result.get("manual_reconciliation_required") is not True
            or result.get("retry_safe") is not False
        ):
            raise ValueError("unknown external execution lacks reconciliation guard")
        if state == "ABSTAINED" and (
            result.get("transport_succeeded") is not False
            or result.get("effect_verified") is not False
            or result.get("manual_reconciliation_required") is not False
            or result.get("retry_safe") is not False
        ):
            raise ValueError("abstained external execution result is inconsistent")
    created = value.get("created_at_unix")
    updated = value.get("updated_at_unix")
    if (
        isinstance(created, bool)
        or isinstance(updated, bool)
        or not isinstance(created, (int, float))
        or not isinstance(updated, (int, float))
        or not math.isfinite(float(created))
        or not math.isfinite(float(updated))
        or float(updated) < float(created)
    ):
        raise ValueError("external execution transaction timestamps are invalid")
    return dict(value)


def _migrate_v2_transaction(value: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade the pre-intervention envelope without rewriting its evidence."""

    if (
        frozenset(value) != _TRANSACTION_FIELDS_V2
        or value.get("schema") != _EXTERNAL_EXECUTE_TRANSACTION_SCHEMA_V2
    ):
        raise ValueError("external execution v2 transaction fields differ")
    migrated = {
        **dict(value),
        "schema": EXTERNAL_EXECUTE_TRANSACTION_SCHEMA,
        "action_intervention": {},
    }
    return _validate_transaction(migrated)


class ExternalExecuteCoordinator:
    """One durable transaction per host action identity."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        owner_alive: Any = None,
    ) -> None:
        self.root = Path(root or Path(DATA_DIR) / "latent_cortex" / "external_execution")
        self._owner_alive = owner_alive or self._default_owner_alive
        self._abandoned_attempt_ids: dict[str, float] = {}

    def _prune_abandoned_attempts(self) -> None:
        now = time.monotonic()
        expired = [
            attempt_id
            for attempt_id, expires_at in self._abandoned_attempt_ids.items()
            if expires_at <= now
        ]
        for attempt_id in expired:
            self._abandoned_attempt_ids.pop(attempt_id, None)
        overflow = len(self._abandoned_attempt_ids) - _MAX_ABANDONED_ATTEMPTS
        if overflow > 0:
            for attempt_id in tuple(self._abandoned_attempt_ids)[:overflow]:
                self._abandoned_attempt_ids.pop(attempt_id, None)

    def _mark_abandoned_attempt(self, attempt_id: str) -> None:
        if (
            not isinstance(attempt_id, str)
            or len(attempt_id) != 32
            or any(character not in "0123456789abcdef" for character in attempt_id)
        ):
            raise ValueError("external execution dispatch attempt id is invalid")
        self._prune_abandoned_attempts()
        self._abandoned_attempt_ids[attempt_id] = time.monotonic() + _ABANDONED_ATTEMPT_TTL_S
        self._prune_abandoned_attempts()

    def _dispatch_owner_active(self, owner: Mapping[str, Any]) -> bool:
        self._prune_abandoned_attempts()
        attempt_id = str(owner.get("attempt_id") or "")
        if attempt_id in self._abandoned_attempt_ids:
            return False
        renewed_at = owner.get("lease_renewed_at")
        duration = owner.get("lease_duration_s")
        if (
            isinstance(renewed_at, bool)
            or isinstance(duration, bool)
            or not isinstance(renewed_at, (int, float))
            or not isinstance(duration, (int, float))
            or time.time() >= float(renewed_at) + float(duration)
        ):
            return False
        return bool(self._owner_alive(owner))

    @staticmethod
    def _default_owner_alive(owner: Mapping[str, Any]) -> bool:
        try:
            identity = owner.get("identity") or {}
            return _holder_is_live(
                Identity(
                    holder=str(identity.get("holder") or ""),
                    pid=int(identity.get("pid") or 0),
                    boot_id=str(identity.get("boot_id") or ""),
                    host=str(identity.get("host") or ""),
                    started_at=float(identity.get("started_at") or 0.0),
                )
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return True

    def _load(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        value = json.loads(
            read_stable_bytes(path, max_bytes=_MAX_TRANSACTION_BYTES).decode("utf-8")
        )
        if not isinstance(value, dict):
            raise ValueError("external execution transaction is not an object")
        digest = value.pop("transaction_sha256", None)
        if digest != _canonical_sha256(value):
            raise ValueError("external execution transaction integrity failed")
        if value.get("schema") == _EXTERNAL_EXECUTE_TRANSACTION_SCHEMA_V2:
            return self._write(path, _migrate_v2_transaction(value))
        if value.get("schema") != EXTERNAL_EXECUTE_TRANSACTION_SCHEMA:
            raise ValueError("external execution transaction integrity failed")
        return {
            **_validate_transaction(value),
            "transaction_sha256": digest,
        }

    def _write(self, path: Path, transaction: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(transaction)
        payload.pop("transaction_sha256", None)
        payload["updated_at_unix"] = time.time()
        _validate_transaction(payload)
        sealed = {**payload, "transaction_sha256": _canonical_sha256(payload)}
        rendered = json.dumps(sealed, sort_keys=True, separators=(",", ":"))
        if len(rendered.encode("utf-8")) > _MAX_TRANSACTION_BYTES:
            raise ValueError("external execution transaction exceeds durable byte limit")
        with local_internal_governed_scope(
            "rlc_external_execute_coordinator",
            domain="state_mutation",
        ):
            get_file_write_gateway().write_text(
                path,
                rendered,
                source="rlc_external_execute_coordinator",
            )
        return sealed

    def prepare(self, offer: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            existing = self._load(path)
            if existing is not None:
                if (
                    existing.get("offer_sha256") != normalized["offer_sha256"]
                    or existing.get("request_digest") != normalized["request_digest"]
                ):
                    raise ValueError("external execution action identity conflicts")
                if existing.get("state") == "DISPATCHING":
                    if self._dispatch_owner_active(existing.get("dispatch_owner") or {}):
                        raise ExternalExecutionInProgressError(
                            "external execution is owned by a live dispatcher"
                        )
                    existing = self._write(
                        path,
                        {
                            **existing,
                            "state": "UNKNOWN_EFFECT",
                            "terminal_reason": "recovered_in_flight_dispatch",
                            "result": {
                                "ok": False,
                                "status": "failed_recoverable",
                                "error": "unknown_effect_requires_reconciliation",
                                "transport_succeeded": None,
                                "effect_verified": False,
                                "manual_reconciliation_required": True,
                                "retry_safe": False,
                                "post_action_receipt_id": "",
                                "replay_payload": {},
                                "replay_payload_sha256": _canonical_sha256({}),
                            },
                        },
                    )
                return existing
            return self._write(
                path,
                {
                    "schema": EXTERNAL_EXECUTE_TRANSACTION_SCHEMA,
                    "action_id": normalized["action_id"],
                    "request_digest": normalized["request_digest"],
                    "offer_sha256": normalized["offer_sha256"],
                    "offer": normalized,
                    "state": "PREPARED",
                    "decision_source": "",
                    "handoff": {},
                    "readiness": {},
                    "cognitive_action_trace": [],
                    "action_policy_evidence": {},
                    "action_policy_receipt": {},
                    "action_intervention": {},
                    "executors": [],
                    "runtime_operation": {},
                    "dispatch_owner": {},
                    "dispatch_authorization_receipt_id": "",
                    "result": {},
                    "terminal_reason": "",
                    "created_at_unix": time.time(),
                },
            )

    def lookup(
        self,
        *,
        action_id: str,
        request_digest: str,
    ) -> dict[str, Any] | None:
        """Return a prior transaction only when its request identity matches."""

        path = _transaction_path(self.root, str(action_id))
        if not path.exists():
            return None
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if transaction is None:
                return None
            if transaction.get("action_id") != str(action_id) or transaction.get(
                "request_digest"
            ) != str(request_digest):
                raise ValueError("external execution action identity conflicts")
            offer = validate_external_execution_offer(transaction.get("offer"))
            if offer["offer_sha256"] != transaction.get("offer_sha256"):
                raise ValueError("external execution stored offer differs")
            return transaction

    def record_handoff(
        self,
        *,
        offer: Mapping[str, Any],
        handoff: Mapping[str, Any],
        cognitive_action_trace: list[Mapping[str, Any]],
        readiness: Mapping[str, Any],
        model_output: str,
        action_policy_evidence: Mapping[str, Any],
        executors: list[str],
        action_policy_receipt: Mapping[str, Any],
        runtime_operation: Mapping[str, Any],
        action_intervention: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from core.brain.llm.latent_cortex.epistemic_runtime import (
            validate_completed_runtime_operation_receipt,
        )
        from core.brain.llm.latent_cortex.epistemic_state import OperationKind
        from core.brain.llm.latent_cortex.value_of_computation import (
            validate_action_trace,
            validate_evidence_snapshot,
        )

        normalized_offer = validate_external_execution_offer(offer)
        normalized_evidence = validate_evidence_snapshot(action_policy_evidence)
        normalized_intervention = None
        if action_intervention is not None:
            from core.brain.llm.latent_cortex.action_intervention import (
                validate_action_intervention,
            )

            normalized_intervention = validate_action_intervention(
                action_intervention,
                require_current_policy=False,
            )
        try:
            normalized_executors = tuple(OperationKind(item) for item in executors)
        except (TypeError, ValueError) as exc:
            raise ValueError("external execution executor inventory is invalid") from exc
        if (
            not normalized_executors
            or len(set(normalized_executors)) != len(normalized_executors)
            or OperationKind.EXECUTE not in normalized_executors
        ):
            raise ValueError("external execution executor inventory is invalid")
        normalized_trace = validate_action_trace(
            cognitive_action_trace,
            evidence_snapshot=normalized_evidence,
            executors=normalized_executors,
            action_intervention=normalized_intervention,
        )
        policy_fields = {
            "schema",
            "bucket",
            "snapshot_sha256",
            "active",
            "executors",
            "actions_selected",
            "checked_transitions",
            "selected_actions",
        }
        if normalized_intervention is not None:
            policy_fields.add("calibration_intervention")
        normalized_policy_receipt = dict(action_policy_receipt)
        checked_transitions = sum(
            int(row["transition"]["checked"]) for row in normalized_trace["rows"]
        )
        if (
            set(normalized_policy_receipt) != policy_fields
            or normalized_policy_receipt.get("schema") != normalized_evidence["schema"]
            or normalized_policy_receipt.get("bucket") != normalized_evidence["bucket"]
            or normalized_policy_receipt.get("snapshot_sha256")
            != normalized_evidence["snapshot_sha256"]
            or normalized_policy_receipt.get("active") is not True
            or normalized_policy_receipt.get("executors") != list(executors)
            or normalized_policy_receipt.get("actions_selected") != len(cognitive_action_trace)
            or normalized_policy_receipt.get("checked_transitions") != checked_transitions
            or normalized_policy_receipt.get("selected_actions")
            != normalized_trace["selected_actions"]
        ):
            raise ValueError("external execution policy summary differs")
        if normalized_intervention is not None:
            from core.brain.llm.latent_cortex.action_intervention import (
                validate_action_intervention_receipt,
            )

            validate_action_intervention_receipt(
                normalized_policy_receipt["calibration_intervention"],
                intervention=normalized_intervention,
                cognitive_action_trace=cognitive_action_trace,
            )
        normalized_runtime_operation = validate_completed_runtime_operation_receipt(
            runtime_operation,
            external_execution_offer=normalized_offer,
            action_policy_evidence=normalized_evidence,
            action_policy_receipt=normalized_policy_receipt,
            cognitive_action_trace=cognitive_action_trace,
            action_intervention=normalized_intervention,
        )
        normalized_handoff = validate_external_execution_handoff(
            handoff,
            offer=normalized_offer,
            cognitive_action_trace=cognitive_action_trace,
        )
        normalized_readiness = validate_external_execution_readiness(
            readiness,
            offer=normalized_offer,
            model_output=model_output,
        )
        if normalized_handoff["requested"] and not (
            normalized_readiness["action_ready"]
            and normalized_readiness["preconditions_met"]
            and normalized_readiness["risk_acceptable"]
        ):
            raise ValueError("external execution handoff lacks action readiness")
        path = _transaction_path(self.root, normalized_offer["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if (
                transaction is None
                or transaction.get("offer_sha256") != normalized_offer["offer_sha256"]
            ):
                raise ValueError("external execution transaction was not prepared")
            state = str(transaction.get("state") or "")
            if state in _TERMINAL:
                return transaction
            if state not in {"PREPARED", "DECIDED"}:
                raise ValueError("external execution handoff arrived after dispatch")
            return self._write(
                path,
                {
                    **transaction,
                    "state": ("DECIDED" if normalized_handoff["requested"] else "ABSTAINED"),
                    "decision_source": "rlc",
                    "handoff": normalized_handoff,
                    "readiness": normalized_readiness,
                    "cognitive_action_trace": normalized_trace["rows"],
                    "action_policy_evidence": normalized_evidence,
                    "action_policy_receipt": normalized_policy_receipt,
                    "action_intervention": dict(normalized_intervention or {}),
                    "executors": list(executors),
                    "runtime_operation": normalized_runtime_operation,
                    "result": (
                        {}
                        if normalized_handoff["requested"]
                        else _bounded_result(
                            {
                                "ok": False,
                                "status": "blocked_by_policy",
                                "error": ("latent_cortex_declined_external_execution"),
                                "transport_succeeded": False,
                                "effect_verified": False,
                                "manual_reconciliation_required": False,
                                "retry_safe": False,
                            }
                        )
                    ),
                    "terminal_reason": (
                        "" if normalized_handoff["requested"] else "rlc_execute_not_selected"
                    ),
                },
            )

    def fail_preparation(
        self,
        *,
        offer: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Close an admitted action that failed before any dispatch intent."""

        normalized = validate_external_execution_offer(offer)
        bounded = _bounded_result(
            {
                **dict(result),
                "ok": False,
                "transport_succeeded": False,
                "effect_verified": False,
                "manual_reconciliation_required": False,
                "retry_safe": False,
            }
        )
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if transaction is None or transaction.get("offer_sha256") != normalized["offer_sha256"]:
                raise ValueError("external execution transaction was not prepared")
            if transaction.get("state") in _TERMINAL:
                return transaction
            if transaction.get("state") != "PREPARED":
                raise ValueError("external execution preparation failure arrived after decision")
            return self._write(
                path,
                {
                    **transaction,
                    "state": "FAILED_PRE_DISPATCH",
                    "result": bounded,
                    "terminal_reason": "pre_dispatch_preparation_failed",
                },
            )

    def record_bypass(
        self,
        *,
        offer: Mapping[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Record an explicit host fallback when cognition cannot answer.

        The bypass preserves the pre-existing action lane during transient
        cortex unavailability, but it never masquerades as an RLC decision.
        """

        normalized = validate_external_execution_offer(offer)
        bounded_reason = str(reason or "unknown")[:240]
        if not _bypass_reason_is_eligible(bounded_reason):
            raise ValueError(
                f"external execution bypass reason is not eligible: {bounded_reason}"
            )
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if transaction is None or transaction.get("offer_sha256") != normalized["offer_sha256"]:
                raise ValueError("external execution transaction was not prepared")
            if transaction.get("state") in _TERMINAL:
                return transaction
            if transaction.get("state") != "PREPARED":
                raise ValueError("external execution bypass arrived after a decision")
            return self._write(
                path,
                {
                    **transaction,
                    "state": "DECIDED",
                    "decision_source": "host_fallback",
                    "handoff": {},
                    "readiness": {},
                    "terminal_reason": f"rlc_unavailable:{bounded_reason}",
                },
            )

    def begin_dispatch(
        self,
        offer: Mapping[str, Any],
        *,
        authorization_receipt_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        bounded_authorization = str(authorization_receipt_id or "").strip()
        if (
            not bounded_authorization
            or len(bounded_authorization) > 192
            or any(ord(character) < 32 for character in bounded_authorization)
        ):
            raise ValueError("external execution dispatch authorization is invalid")
        if bounded_authorization != normalized["will_receipt_id"]:
            raise ValueError("external execution dispatch authorization differs from offer")
        bounded_task_id = str(task_id or "").strip()
        if (
            not bounded_task_id
            or len(bounded_task_id) > 192
            or any(ord(character) < 32 for character in bounded_task_id)
        ):
            raise ValueError("external execution dispatch task identity is invalid")
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if transaction is None:
                raise ValueError("external execution transaction was not prepared")
            if transaction.get("offer_sha256") != normalized["offer_sha256"]:
                raise ValueError("external execution offer changed before dispatch")
            if transaction.get("state") != "DECIDED":
                raise ValueError(
                    f"external execution cannot dispatch from {transaction.get('state')}"
                )
            owner = {
                "attempt_id": uuid.uuid4().hex,
                "identity": Identity.current(f"rlc-external-{os.getpid()}").to_dict(),
                "task_id": bounded_task_id,
                "lease_renewed_at": time.time(),
                "lease_duration_s": _DISPATCH_LEASE_DURATION_S,
            }
            return self._write(
                path,
                {
                    **transaction,
                    "state": "DISPATCHING",
                    "dispatch_owner": owner,
                    "dispatch_authorization_receipt_id": bounded_authorization,
                },
            )

    def renew_dispatch(
        self,
        *,
        offer: Mapping[str, Any],
        dispatch_attempt_id: str,
    ) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if (
                transaction is None
                or transaction.get("offer_sha256") != normalized["offer_sha256"]
                or transaction.get("state") != "DISPATCHING"
            ):
                raise ValueError("external execution dispatch lease is unavailable")
            owner = dict(transaction.get("dispatch_owner") or {})
            if owner.get("attempt_id") != str(dispatch_attempt_id):
                raise ValueError("external execution dispatch owner differs")
            owner["lease_renewed_at"] = time.time()
            return self._write(
                path,
                {**transaction, "dispatch_owner": owner},
            )

    def abandon_dispatch(
        self,
        *,
        offer: Mapping[str, Any],
        dispatch_attempt_id: str,
        effect_may_have_occurred: bool,
        reason: str,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        if (
            not isinstance(dispatch_attempt_id, str)
            or len(dispatch_attempt_id) != 32
            or any(character not in "0123456789abcdef" for character in dispatch_attempt_id)
        ):
            raise ValueError("external execution dispatch attempt id is invalid")
        attempt_id = dispatch_attempt_id
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if (
                transaction is None
                or transaction.get("offer_sha256") != normalized["offer_sha256"]
                or transaction.get("state") != "DISPATCHING"
            ):
                raise ValueError("external execution dispatch cannot be abandoned")
            owner = transaction.get("dispatch_owner") or {}
            if owner.get("attempt_id") != attempt_id:
                raise ValueError("external execution dispatch owner differs")
            self._mark_abandoned_attempt(attempt_id)
            bounded_reason = str(reason or "dispatch_abandoned")[:240]
            bounded_result = _bounded_result(
                {
                    **dict(result or {}),
                    "ok": False,
                    "status": "failed_recoverable",
                    "error": bounded_reason,
                    "transport_succeeded": effect_may_have_occurred,
                    "effect_verified": False,
                    "manual_reconciliation_required": effect_may_have_occurred,
                    "retry_safe": not effect_may_have_occurred,
                }
            )
            written = self._write(
                path,
                {
                    **transaction,
                    "state": ("UNKNOWN_EFFECT" if effect_may_have_occurred else "FAILED"),
                    "result": bounded_result,
                    "terminal_reason": (
                        "dispatch_abandoned_effect_unknown"
                        if effect_may_have_occurred
                        else "dispatch_abandoned_before_effect"
                    ),
                },
            )
            self._abandoned_attempt_ids.pop(attempt_id, None)
            return written

    def complete(
        self,
        *,
        offer: Mapping[str, Any],
        result: Mapping[str, Any],
        dispatch_attempt_id: str,
    ) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        path = _transaction_path(self.root, normalized["action_id"])
        bounded = _bounded_result(result)
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if transaction is None or transaction.get("offer_sha256") != normalized["offer_sha256"]:
                raise ValueError("external execution transaction identity differs")
            if transaction.get("state") != "DISPATCHING":
                raise ValueError("external execution completion has no dispatch intent")
            owner = transaction.get("dispatch_owner") or {}
            if owner.get("attempt_id") != str(dispatch_attempt_id):
                raise ValueError("external execution dispatch owner differs")
            if bounded["effect_verified"]:
                state = "SUCCEEDED"
                reason = "effect_verified"
            elif (bounded["transport_succeeded"] or not bounded["retry_safe"]) and not bounded[
                "effect_verified"
            ]:
                state = "UNKNOWN_EFFECT"
                reason = "dispatched_effect_unverified"
                bounded["manual_reconciliation_required"] = True
            else:
                state = "FAILED"
                reason = "effect_failed_before_verification"
            return self._write(
                path,
                {
                    **transaction,
                    "state": state,
                    "result": bounded,
                    "terminal_reason": reason,
                },
            )

    def link_post_action_receipt(
        self,
        *,
        offer: Mapping[str, Any],
        persisted_receipt: Mapping[str, Any],
        receipt_store: Any = None,
    ) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        contract, receipt_sha256 = _validate_post_action_receipt_contract(
            persisted_receipt,
            offer=normalized,
        )
        bounded_receipt_id = contract["receipt_id"]
        if receipt_store is None:
            from core.runtime.post_action_receipt import (
                get_post_action_receipt_store,
            )

            receipt_store = get_post_action_receipt_store()
        stored_receipt = receipt_store.get_receipt(bounded_receipt_id)
        if stored_receipt is None or stored_receipt.to_dict() != contract:
            raise ValueError("post-action receipt lacks matching durable store evidence")
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if (
                transaction is None
                or transaction.get("offer_sha256") != normalized["offer_sha256"]
                or transaction.get("state") not in _TERMINAL
            ):
                raise ValueError("external execution terminal transaction is unavailable")
            result = dict(transaction.get("result") or {})
            existing = str(result.get("post_action_receipt_id") or "")
            if existing and existing != bounded_receipt_id:
                raise ValueError("external execution post-action receipt conflicts")
            replay_payload = dict(result.get("replay_payload") or {})
            recovery_contract = replay_payload.get("_post_action_recovery_contract")
            if not isinstance(recovery_contract, Mapping) or dict(recovery_contract) != contract:
                raise ValueError("persisted post-action receipt differs from recovery contract")
            _validate_post_action_receipt_outcome(
                contract,
                transaction=transaction,
            )
            reconciled_unknown = (
                transaction.get("state") == "UNKNOWN_EFFECT"
                and contract.get("effect_verified") is True
                and contract.get("transport_succeeded") is True
            )
            result.update(
                {
                    "ok": (
                        contract.get("actual_outcome") == "success"
                        and contract.get("status") == "success_verified"
                    ),
                    "status": contract["status"],
                    "error": str(contract.get("error_status") or "")[:500],
                    "transport_succeeded": contract["transport_succeeded"],
                    "effect_verified": contract["effect_verified"],
                    "manual_reconciliation_required": contract["manual_reconciliation_required"],
                    "retry_safe": contract["retry_safe"],
                }
            )
            replay_payload.update(
                {
                    "ok": result["ok"],
                    "status": result["status"],
                    "error": result["error"],
                    "transport_succeeded": result["transport_succeeded"],
                    "effect_verified": result["effect_verified"],
                    "manual_reconciliation_required": result["manual_reconciliation_required"],
                    "retry_safe": result["retry_safe"],
                    "post_action_receipt_id": bounded_receipt_id,
                    "post_action_receipt_sha256": receipt_sha256,
                    "post_action_output_hash": contract["output_hash"],
                    "receipt_persisted": True,
                    "post_action_receipt_pending": False,
                }
            )
            replay_payload.pop("_post_action_recovery_contract", None)
            return self._write(
                path,
                {
                    **transaction,
                    "state": ("SUCCEEDED" if reconciled_unknown else transaction["state"]),
                    "terminal_reason": (
                        "effect_verified_by_durable_post_action_receipt"
                        if reconciled_unknown
                        else transaction["terminal_reason"]
                    ),
                    "result": {
                        **result,
                        "post_action_receipt_id": bounded_receipt_id,
                        "replay_payload": replay_payload,
                        "replay_payload_sha256": _canonical_sha256(replay_payload),
                    },
                },
            )

    def stage_post_action_receipt(
        self,
        *,
        offer: Mapping[str, Any],
        receipt_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = validate_external_execution_offer(offer)
        contract, _receipt_sha256 = _validate_post_action_receipt_contract(
            receipt_contract,
            offer=normalized,
        )
        receipt_id = contract["receipt_id"]
        path = _transaction_path(self.root, normalized["action_id"])
        with interprocess_file_lock(path.with_suffix(".lock")):
            transaction = self._load(path)
            if (
                transaction is None
                or transaction.get("offer_sha256") != normalized["offer_sha256"]
                or transaction.get("state") not in _TERMINAL
            ):
                raise ValueError("external execution terminal transaction is unavailable")
            _validate_post_action_receipt_outcome(
                contract,
                transaction=transaction,
            )
            result = dict(transaction.get("result") or {})
            if result.get("post_action_receipt_id"):
                raise ValueError("linked post-action receipt contract is immutable")
            replay_payload = dict(result.get("replay_payload") or {})
            replay_payload.update(
                {
                    "receipt_persisted": False,
                    "post_action_receipt_pending": True,
                    "post_action_receipt_attempt_id": receipt_id,
                    "_post_action_recovery_contract": contract,
                }
            )
            return self._write(
                path,
                {
                    **transaction,
                    "result": {
                        **result,
                        "replay_payload": replay_payload,
                        "replay_payload_sha256": _canonical_sha256(replay_payload),
                    },
                },
            )


_COORDINATOR: ExternalExecuteCoordinator | None = None


def get_external_execute_coordinator() -> ExternalExecuteCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = ExternalExecuteCoordinator()
    return _COORDINATOR


__all__ = [
    "EXTERNAL_EXECUTE_TRANSACTION_SCHEMA",
    "ExternalExecutionInProgressError",
    "ExternalExecuteCoordinator",
    "get_external_execute_coordinator",
]
