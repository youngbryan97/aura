"""Immutable, independently replayable pass-to-pass reasoning evidence."""

from __future__ import annotations

import base64
import binascii
import contextvars
import enum
import fcntl
import functools
import hashlib
import inspect
import json
import logging
import marshal
import os
import platform
import re
import stat
import struct
import sys
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    CampaignTrustError,
    VerifiedCampaignTrustPolicy,
    operationally_isolated_roles,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    FINAL_ANSWER_MARKER,
    BlindedAnswerPayload,
    FrontierTask,
    FrontierTaskRegistry,
    PublicTaskRecord,
    build_task_manifest,
    frontier_scorer_callables,
    parse_final_answer,
    score_task,
)
from core.runtime.atomic_writer import ensure_private_directory
from core.runtime.resource_observation import (
    HostResourceObserver,
    ObservationSource,
    OpenFileIdentityObservation,
)

ARTIFACT_BINDING_SCHEMA = "aura.verified_transition.artifact_binding.v1"
CALLABLE_COMMITMENT_SCHEMA = "aura.verified_transition.callable_commitment.v1"
PRIMARY_SCORE_SCHEMA = "aura.verified_transition.primary_score.v1"
WITNESS_SCORE_SCHEMA = "aura.verified_transition.witness_score.v1"
VERIFIER_AUTHORITY_SCHEMA = "aura.verified_transition.verifier_authority.v1"
REASONING_PASS_SCHEMA = "aura.verified_transition.reasoning_pass.v1"
VERIFIED_TRANSITION_EPISODE_SCHEMA = "aura.verified_transition.paired_episode.v1"
ATTEMPT_JOURNAL_SCHEMA = "aura.verified_transition.attempt_journal.v1"
ATTEMPT_LEDGER_EVENT_SCHEMA = "aura.verified_transition.attempt_ledger_event.v1"
ATTEMPT_LEDGER_OPEN_PAYLOAD_SCHEMA = "aura.verified_transition.attempt_ledger_open_payload.v1"
ATTEMPT_LEDGER_TERMINAL_PAYLOAD_SCHEMA = (
    "aura.verified_transition.attempt_ledger_terminal_payload.v1"
)
WITNESS_PAYLOAD_SCHEMA = "aura.verified_transition.witness_payload.v1"
EXECUTION_MANIFEST_SCHEMA = "aura.verified_transition.execution_manifest.v1"
CALIBRATION_PAYLOAD_SCHEMA = "aura.verified_transition.calibration_payload.v1"
CALIBRATION_EVIDENCE_SCHEMA = "aura.verified_transition.calibration_evidence.v1"
CALIBRATION_CASE_SCHEMA = "aura.verified_transition.calibration_case.v1"
VERIFIER_IMPLEMENTATION_SCHEMA = "aura.verified_transition.verifier_implementation.v1"
CANDIDATE_INPUT_SCHEMA = "aura.verified_transition.candidate_input.v1"
GENERATION_TRACE_PAYLOAD_SCHEMA = "aura.verified_transition.generation_trace_payload.v1"
EXECUTION_OBSERVER_PAYLOAD_SCHEMA = "aura.verified_transition.execution_observer_payload.v1"
EXECUTION_PROCESS_OBSERVATION_SCHEMA = "aura.verified_transition.execution_process_observation.v1"

MAX_DOCUMENT_BYTES = 1_048_576
MAX_TEXT_BYTES = 262_144
MAX_SOURCE_BYTES = 8_388_608
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 16_384
MAX_JSON_STRING_BYTES = 262_144
MAX_INTEGER = (1 << 63) - 1
MAX_TOKEN_ID = (1 << 31) - 1
MAX_OUTPUT_TOKENS = 65_536
_EXECUTION_COMPONENT_ROLES = frozenset(
    {
        "base_checkpoint",
        "adapter_stack",
        "tokenizer",
        "policy",
        "personality",
        "runtime",
        "source_closure",
        "generation_worker",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/;=+-]{0,191}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")

_ARTIFACT_KEYS = frozenset({"schema", "payload_sha256", "byte_length", "media_type"})
_CALLABLE_KEYS = frozenset(
    {
        "schema",
        "module",
        "qualname",
        "runtime_code_sha256",
        "defaults_sha256",
        "closure_sha256",
        "globals_sha256",
        "callable_source_sha256",
        "module_file_sha256",
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "platform_system",
        "platform_machine",
    }
)
_SCORE_OUTPUT_KEYS = frozenset(
    {
        "schema",
        "task_id",
        "domain",
        "scorer_id",
        "parsed",
        "correct",
        "reason",
        "normalized_answer_sha256",
    }
)
_INDEPENDENT_OUTPUT_KEYS = frozenset({"parsed", "correct", "reason", "normalized_answer_sha256"})
_PRIMARY_SCORE_KEYS = frozenset(
    {
        "schema",
        "task_id",
        "public_task_payload_sha256",
        "answer_commitment_sha256",
        "response_artifact",
        "score_output",
    }
)
_WITNESS_SCORE_KEYS = frozenset(
    {
        "schema",
        "task_id",
        "public_task_payload_sha256",
        "answer_commitment_sha256",
        "response_artifact",
        "witness_identity_sha256",
        "scorer_callable",
        "score_output",
        "trust_policy_sha256",
        "evidence_verifier_attestation",
        "issued_at_unix_ns",
        "receipt_sha256",
    }
)
_AUTHORITY_KEYS = frozenset(
    {
        "schema",
        "authority_id",
        "verifier_id",
        "verifier_version",
        "source_closure",
        "source_closure_sha256",
        "public_task_artifact",
        "task_manifest_artifact",
        "task_issuer_attestation_artifact",
        "task_id",
        "public_task_payload_sha256",
        "answer_commitment_sha256",
        "issuer_commitment_sha256",
        "verifier_trust_policy_sha256",
        "calibration_evidence_sha256",
        "calibration_evidence_artifact",
        "execution_manifest_sha256",
        "execution_manifest_artifact",
        "response_artifact",
        "primary_score_artifact",
        "independent_witness_artifact",
        "verifier_output",
        "parsed_answer_sha256",
        "outcome",
        "producer_identity_sha256",
        "verifier_identity_sha256",
        "independent_witness_identity_sha256",
        "issued_at_unix_ns",
        "sealed_at_unix_ns",
        "receipt_sha256",
    }
)
_AUTHORITY_EXPECTED_KEYS = frozenset(
    {
        "authority_id",
        "verifier_id",
        "verifier_version",
        "issuer_commitment_sha256",
        "verifier_trust_policy_sha256",
        "calibration_evidence_sha256",
        "execution_manifest_sha256",
        "producer_identity_sha256",
        "verifier_identity_sha256",
        "independent_witness_identity_sha256",
    }
)
_BUDGET_KEYS = frozenset({"max_output_tokens", "max_wall_time_ms", "max_compute_units"})
_PASS_KEYS = frozenset(
    {
        "schema",
        "episode_id",
        "pass_index",
        "task_id",
        "case_id",
        "family",
        "domain",
        "depth",
        "difficulty",
        "sealed_task_commitment_sha256",
        "prompt_artifact",
        "model_input_artifact",
        "response_artifact",
        "model_identity_sha256",
        "base_checkpoint_sha256",
        "adapter_stack_sha256",
        "tokenizer_sha256",
        "token_encoder_callable",
        "token_decoder_callable",
        "policy_sha256",
        "personality_sha256",
        "runtime_sha256",
        "source_closure_sha256",
        "attempt_ledger_identity_sha256",
        "execution_manifest_artifact",
        "execution_spec_artifact",
        "latent_path_artifact",
        "tool_snapshot_artifact",
        "evidence_snapshot_artifact",
        "world_state_snapshot_artifact",
        "rng_root_sha256",
        "generation_budget",
        "deadline_unix_ns",
        "input_token_ids",
        "output_token_ids",
        "emitted_token_pieces_artifact",
        "behavior_policy_logprobs",
        "generation_worker_attestation_artifact",
        "execution_process_observation_artifact",
        "execution_observer_attestation_artifact",
        "verifier_authority_artifact",
        "process_receipt_artifact",
        "uncertainty_receipt_artifact",
        "diversity_receipt_artifact",
        "resource_receipt_artifact",
        "generated_at_unix_ns",
        "sealed_at_unix_ns",
        "receipt_sha256",
    }
)
_PASS_CONTEXT_KEYS = frozenset(
    {
        "episode_id",
        "case_id",
        "family",
        "depth",
        "sealed_task_commitment_sha256",
        "model_identity_sha256",
        "base_checkpoint_sha256",
        "adapter_stack_sha256",
        "tokenizer_sha256",
        "policy_sha256",
        "personality_sha256",
        "runtime_sha256",
        "source_closure_sha256",
        "execution_spec_artifact",
        "latent_path_artifact",
        "tool_snapshot_artifact",
        "evidence_snapshot_artifact",
        "world_state_snapshot_artifact",
        "rng_root_sha256",
        "generation_budget",
        "deadline_unix_ns",
        "process_receipt_artifact",
        "uncertainty_receipt_artifact",
        "diversity_receipt_artifact",
        "resource_receipt_artifact",
        "generated_at_unix_ns",
        "sealed_at_unix_ns",
    }
)
_EPISODE_KEYS = frozenset(
    {
        "schema",
        "episode_id",
        "pass_count",
        "pass_0_artifact",
        "pass_1_artifact",
        "task_id",
        "sealed_task_commitment_sha256",
        "protocol_sha256",
        "immutable_context_sha256",
        "trust_policy_sha256",
        "attempt_journal_artifact",
        "campaign_runner_attestation_artifact",
        "evidence_verifier_journal_attestation_artifact",
        "created_at_unix_ns",
        "sealed_at_unix_ns",
        "receipt_sha256",
    }
)
_ATTEMPT_JOURNAL_KEYS = frozenset(
    {
        "schema",
        "episode_id",
        "protocol_sha256",
        "immutable_context_sha256",
        "attempt_count",
        "attempts",
        "runner_event_attestations",
        "attempt_ledger_identity_sha256",
        "attempt_ledger_content_sha256",
        "attempt_ledger_open_attestation",
        "attempt_ledger_terminal_attestation",
        "event_chain_head_sha256",
        "terminal_state",
        "final_pass_index",
        "receipt_sha256",
    }
)
_PASS_IMMUTABLE_FIELDS = (
    "episode_id",
    "task_id",
    "case_id",
    "family",
    "domain",
    "depth",
    "difficulty",
    "sealed_task_commitment_sha256",
    "prompt_artifact",
    "model_input_artifact",
    "model_identity_sha256",
    "base_checkpoint_sha256",
    "adapter_stack_sha256",
    "tokenizer_sha256",
    "token_encoder_callable",
    "token_decoder_callable",
    "policy_sha256",
    "personality_sha256",
    "runtime_sha256",
    "source_closure_sha256",
    "attempt_ledger_identity_sha256",
    "execution_manifest_artifact",
    "execution_spec_artifact",
    "latent_path_artifact",
    "tool_snapshot_artifact",
    "evidence_snapshot_artifact",
    "world_state_snapshot_artifact",
    "rng_root_sha256",
    "generation_budget",
    "deadline_unix_ns",
)
_AUTHORITY_IMMUTABLE_FIELDS = (
    "authority_id",
    "verifier_id",
    "verifier_version",
    "source_closure",
    "source_closure_sha256",
    "public_task_artifact",
    "task_manifest_artifact",
    "task_issuer_attestation_artifact",
    "task_id",
    "public_task_payload_sha256",
    "answer_commitment_sha256",
    "issuer_commitment_sha256",
    "verifier_trust_policy_sha256",
    "calibration_evidence_sha256",
    "calibration_evidence_artifact",
    "execution_manifest_sha256",
    "execution_manifest_artifact",
    "producer_identity_sha256",
    "verifier_identity_sha256",
    "independent_witness_identity_sha256",
)


class VerifiedTransitionError(ValueError):
    """Stable fail-closed error for unverifiable transition evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionError(code)


def _verify_role_attestation(
    policy: VerifiedCampaignTrustPolicy,
    attestation: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        return verify_role_attestation(policy, attestation, **kwargs)
    except CampaignTrustError as exc:
        raise VerifiedTransitionError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class TransitionTrustContext:
    """Out-of-band root and frozen campaign policy for transition evidence."""

    policy_document: Mapping[str, Any]
    trusted_root_public_key_pem: bytes
    expected_campaign_name: str
    expected_protocol_sha256: str
    expected_policy_sha256: str
    observed_at_unix: int
    execution_manifest: Mapping[str, Any]
    execution_component_roots: Mapping[str, Path]
    expected_execution_manifest_sha256: str
    calibration_evidence: Mapping[str, Any]
    expected_calibration_evidence_sha256: str
    attempt_ledger_path: Path
    expected_attempt_ledger_identity_sha256: str
    attempt_ledger_open_attestation: Mapping[str, Any] | None
    attempt_ledger_terminal_attestation: Mapping[str, Any] | None
    task_issuer_attestation: Mapping[str, Any] | None

    def verified_policy(self) -> VerifiedCampaignTrustPolicy:
        if not isinstance(self.trusted_root_public_key_pem, bytes):
            _fail("transition_trust_root_invalid")
        return validate_campaign_trust_policy(
            self.policy_document,
            trusted_root_public_key_pem=self.trusted_root_public_key_pem,
            expected_campaign_name=_require_identifier(
                self.expected_campaign_name,
                role="transition_campaign_name",
            ),
            expected_policy_sha256=_require_sha256(
                self.expected_policy_sha256,
                role="transition_policy_sha256",
            ),
            expected_protocol_sha256=_require_sha256(
                self.expected_protocol_sha256,
                role="transition_protocol_sha256",
            ),
            now_unix=_require_int(
                self.observed_at_unix,
                role="transition_observed_at",
                minimum=1,
            ),
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{role}_invalid")
    return value


def _require_identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{role}_invalid")
    return value


def _require_int(
    value: Any,
    *,
    role: str,
    minimum: int = 0,
    maximum: int = MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
    return value


def _require_signed_second(value: Any, *, role: str) -> int:
    timestamp = _require_int(value, role=role, minimum=1)
    if timestamp % 1_000_000_000:
        _fail(f"{role}_subsecond_forbidden")
    return timestamp


def _require_bool(value: Any, *, role: str) -> bool:
    if type(value) is not bool:
        _fail(f"{role}_invalid")
    return value


def _require_decimal(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or _DECIMAL.fullmatch(value) is None
        or value in {"-0", "-0.0"}
        or (value.endswith("0") and "." in value)
        or value.endswith(".")
    ):
        _fail(f"{role}_invalid")
    return value


def _validate_json_tree(
    value: Any,
    *,
    depth: int = 0,
    nodes: list[int] | None = None,
) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES:
        _fail("json_node_limit_exceeded")
    if depth > MAX_JSON_DEPTH:
        _fail("json_depth_limit_exceeded")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if not -MAX_INTEGER <= value <= MAX_INTEGER:
            _fail("json_integer_out_of_bounds")
        return
    if isinstance(value, float):
        _fail("json_floating_point_forbidden")
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            _fail("json_string_utf8_invalid")
        if len(encoded) > MAX_JSON_STRING_BYTES or "\x00" in value:
            _fail("json_string_invalid")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key.encode("utf-8")) > 192
                or "\x00" in key
            ):
                _fail("json_key_invalid")
            _validate_json_tree(item, depth=depth + 1, nodes=nodes)
        return
    _fail("json_type_invalid")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted receipt encoding."""

    _validate_json_tree(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise VerifiedTransitionError("json_not_canonicalizable") from exc


def strict_canonical_json_loads(
    payload: bytes,
    *,
    role: str = "document",
    maximum_bytes: int = MAX_DOCUMENT_BYTES,
) -> Any:
    """Parse strict canonical JSON without duplicate keys or numeric ambiguity."""

    if not isinstance(payload, bytes) or not payload or len(payload) > maximum_bytes:
        _fail(f"{role}_size_invalid")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError:
        _fail(f"{role}_not_ascii")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def parse_int(raw: str) -> int:
        digits = raw.removeprefix("-")
        if not digits or len(digits) > 19:
            _fail(f"{role}_integer_out_of_bounds")
        value = int(raw)
        if not -MAX_INTEGER <= value <= MAX_INTEGER:
            _fail(f"{role}_integer_out_of_bounds")
        return value

    def reject_float(_raw: str) -> Never:
        _fail(f"{role}_floating_point_forbidden")

    def reject_constant(_raw: str) -> Never:
        _fail(f"{role}_non_finite_number")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except VerifiedTransitionError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise VerifiedTransitionError(f"{role}_invalid_json") from exc
    _validate_json_tree(value)
    if canonical_json_bytes(value) != payload:
        _fail(f"{role}_noncanonical")
    return value


def _digest_document(document: Mapping[str, Any], *, digest_field: str) -> str:
    body = dict(document)
    body.pop(digest_field, None)
    return _sha256_bytes(canonical_json_bytes(body))


def _seal_document(
    document: Mapping[str, Any],
    *,
    digest_field: str = "receipt_sha256",
) -> dict[str, Any]:
    if digest_field in document:
        _fail("caller_digest_forbidden")
    sealed = dict(document)
    sealed[digest_field] = _digest_document(sealed, digest_field=digest_field)
    return sealed


def _validate_digest(
    document: Mapping[str, Any],
    *,
    digest_field: str = "receipt_sha256",
    role: str,
) -> None:
    supplied = _require_sha256(document.get(digest_field), role=f"{role}_digest")
    if supplied != _digest_document(document, digest_field=digest_field):
        _fail(f"{role}_digest_mismatch")


def _read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    expected_size: int | None = None,
    role: str,
    require_private: bool = True,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        _fail("nofollow_unsupported")
    flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerifiedTransitionError(f"{role}_unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or (require_private and before.st_mode & 0o077)
            or before.st_size < 0
            or before.st_size > maximum_bytes
            or (expected_size is not None and before.st_size != expected_size)
        ):
            _fail(f"{role}_file_identity_invalid")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                _fail(f"{role}_truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{role}_grew_during_read")
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise VerifiedTransitionError(f"{role}_replaced_during_read") from exc
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail(f"{role}_changed_during_read")
        if any(getattr(after, field) != getattr(path_after, field) for field in stable_fields):
            _fail(f"{role}_replaced_during_read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash_regular_file(path: Path, *, role: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        _fail("nofollow_unsupported")
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        raise VerifiedTransitionError(f"{role}_unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > 137_438_953_472
        ):
            _fail(f"{role}_file_identity_invalid")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(f"{role}_truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{role}_grew_during_read")
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail(f"{role}_changed_during_read")
        if any(getattr(after, field) != getattr(path_after, field) for field in stable_fields):
            _fail(f"{role}_replaced_during_read")
        return digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    expected_size: int | None = None,
    role: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        _fail("nofollow_unsupported")
    try:
        descriptor = os.open(name, flags | nofollow, dir_fd=directory_fd)
    except OSError as exc:
        raise VerifiedTransitionError(f"{role}_unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_mode & 0o077
            or before.st_size < 0
            or before.st_size > maximum_bytes
            or (expected_size is not None and before.st_size != expected_size)
        ):
            _fail(f"{role}_file_identity_invalid")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                _fail(f"{role}_truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{role}_grew_during_read")
        after = os.fstat(descriptor)
        path_after = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail(f"{role}_changed_during_read")
        if any(getattr(after, field) != getattr(path_after, field) for field in stable_fields):
            _fail(f"{role}_replaced_during_read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_regular_file_once_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        _fail("nofollow_unsupported")
    try:
        descriptor = os.open(
            name,
            flags | nofollow,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError:
        return
    except OSError as exc:
        raise VerifiedTransitionError("artifact_write_failed") from exc
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                _fail("artifact_write_incomplete")
            written += count
        os.fsync(descriptor)
    except Exception:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_chain(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            _fail("artifact_store_symlink_path_rejected")


def _ensure_private_store_directory(path: Path) -> Path:
    lexical = _lexical_absolute(path)
    _reject_symlink_chain(lexical)
    if lexical.exists():
        metadata = lexical.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("artifact_store_directory_not_private")
    ensure_private_directory(lexical)
    _reject_symlink_chain(lexical)
    metadata = lexical.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("artifact_store_directory_not_private")
    return lexical


def _private_directory_identity(path: Path) -> tuple[int, int]:
    _reject_symlink_chain(path)
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise VerifiedTransitionError("artifact_store_directory_unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("artifact_store_directory_not_private")
    return (metadata.st_dev, metadata.st_ino)


def _require_exact_frontier_task(task: Any) -> FrontierTask:
    if (
        type(task) is not FrontierTask
        or type(task.public) is not PublicTaskRecord
        or type(task.blinded_answer) is not BlindedAnswerPayload
    ):
        _fail("frontier_task_exact_type_required")
    public = PublicTaskRecord.from_dict(task.public.to_dict())
    private_bytes = task.blinded_answer._canonical_bytes
    rebuilt = FrontierTask(
        schema=task.schema,
        public=public,
        blinded_answer=BlindedAnswerPayload(
            task.blinded_answer.commitment_sha256,
            bytes(private_bytes),
        ),
    )
    if rebuilt != task:
        _fail("frontier_task_reconstruction_mismatch")
    return task


class TransitionArtifactStore:
    """Private immutable SHA-256 store for transition evidence payloads."""

    def __init__(self, root: str | Path) -> None:
        requested = Path(root)
        self.root = _ensure_private_store_directory(requested)
        self.blob_root = _ensure_private_store_directory(self.root / "blobs")
        if self.blob_root.parent != self.root:
            _fail("artifact_root_invalid")
        self._root_identity = _private_directory_identity(self.root)
        self._blob_root_identity = _private_directory_identity(self.blob_root)
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(nofollow, int) or nofollow <= 0:
            _fail("nofollow_unsupported")
        try:
            self._root_fd = os.open(self.root, directory_flags | nofollow)
            self._blob_root_fd = os.open(
                "blobs",
                directory_flags | nofollow,
                dir_fd=self._root_fd,
            )
        except OSError as exc:
            if hasattr(self, "_root_fd"):
                os.close(self._root_fd)
            raise VerifiedTransitionError("artifact_store_open_failed") from exc
        if (
            os.fstat(self._root_fd).st_dev,
            os.fstat(self._root_fd).st_ino,
        ) != self._root_identity or (
            os.fstat(self._blob_root_fd).st_dev,
            os.fstat(self._blob_root_fd).st_ino,
        ) != self._blob_root_identity:
            self.close()
            _fail("artifact_store_directory_replaced")

    def _assert_store_identity(self) -> None:
        if (
            not hasattr(self, "_root_fd")
            or not hasattr(self, "_blob_root_fd")
            or self._root_fd < 0
            or self._blob_root_fd < 0
        ):
            _fail("artifact_store_closed")
        try:
            root_metadata = os.fstat(self._root_fd)
            blob_metadata = os.fstat(self._blob_root_fd)
        except OSError as exc:
            raise VerifiedTransitionError("artifact_store_closed") from exc
        if (
            (root_metadata.st_dev, root_metadata.st_ino) != self._root_identity
            or (blob_metadata.st_dev, blob_metadata.st_ino) != self._blob_root_identity
            or _private_directory_identity(self.root) != self._root_identity
            or _private_directory_identity(self.blob_root) != self._blob_root_identity
        ):
            _fail("artifact_store_directory_replaced")

    def _path(self, digest: str) -> Path:
        self._assert_store_identity()
        normalized = _require_sha256(digest, role="artifact_digest")
        return self.blob_root / normalized

    def close(self) -> None:
        blob_fd = getattr(self, "_blob_root_fd", -1)
        root_fd = getattr(self, "_root_fd", -1)
        self._blob_root_fd = -1
        self._root_fd = -1
        if blob_fd >= 0:
            os.close(blob_fd)
        if root_fd >= 0:
            os.close(root_fd)

    def __enter__(self) -> TransitionArtifactStore:
        self._assert_store_identity()
        return self

    def __copy__(self) -> TransitionArtifactStore:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> TransitionArtifactStore:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
    ) -> dict[str, Any]:
        if not isinstance(payload, bytes) or not payload or len(payload) > MAX_DOCUMENT_BYTES:
            _fail("artifact_payload_size_invalid")
        normalized_media = _require_identifier(media_type, role="artifact_media_type")
        digest = _sha256_bytes(payload)
        self._path(digest)
        _write_regular_file_once_at(self._blob_root_fd, digest, payload)
        observed = _read_regular_file_at(
            self._blob_root_fd,
            digest,
            maximum_bytes=MAX_DOCUMENT_BYTES,
            expected_size=len(payload),
            role="artifact",
        )
        if observed != payload:
            _fail("artifact_collision")
        return {
            "schema": ARTIFACT_BINDING_SCHEMA,
            "payload_sha256": digest,
            "byte_length": len(payload),
            "media_type": normalized_media,
        }

    def put_json(self, document: Mapping[str, Any]) -> dict[str, Any]:
        return self.put_bytes(
            canonical_json_bytes(dict(document)),
            media_type="application/json",
        )

    def read_bytes(
        self,
        binding: Mapping[str, Any],
        *,
        expected_media_type: str | None = None,
    ) -> bytes:
        normalized = _validate_artifact_binding(binding)
        if expected_media_type is not None and normalized["media_type"] != expected_media_type:
            _fail("artifact_media_type_mismatch")
        self._path(normalized["payload_sha256"])
        payload = _read_regular_file_at(
            self._blob_root_fd,
            normalized["payload_sha256"],
            maximum_bytes=MAX_DOCUMENT_BYTES,
            expected_size=normalized["byte_length"],
            role="artifact",
        )
        if _sha256_bytes(payload) != normalized["payload_sha256"]:
            _fail("artifact_digest_mismatch")
        return payload

    def read_json(self, binding: Mapping[str, Any], *, role: str) -> dict[str, Any]:
        payload = self.read_bytes(
            binding,
            expected_media_type="application/json",
        )
        value = strict_canonical_json_loads(payload, role=role)
        if not isinstance(value, dict):
            _fail(f"{role}_not_object")
        return cast(dict[str, Any], value)


class ExternalAttemptLedger:
    """Pinned append-only runner ledger for non-retrospective attempt evidence."""

    _append_only_flag = getattr(stat, "UF_APPEND", 0)

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        self.path = _lexical_absolute(Path(path))
        parent = _ensure_private_store_directory(self.path.parent)
        self.path = parent / self.path.name
        _reject_symlink_chain(self.path)
        if create:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except FileExistsError as exc:
                raise VerifiedTransitionError("attempt_ledger_already_exists") from exc
            except OSError as exc:
                raise VerifiedTransitionError("attempt_ledger_create_failed") from exc
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if self._append_only_flag:
                try:
                    os.chflags(
                        self.path,
                        self._append_only_flag,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise VerifiedTransitionError("attempt_ledger_append_only_flag_failed") from exc
            parent_fd = os.open(
                self.path.parent,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        self._identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        _reject_symlink_chain(self.path)
        try:
            metadata = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise VerifiedTransitionError("attempt_ledger_unavailable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
            or (self._append_only_flag and not metadata.st_flags & self._append_only_flag)
        ):
            _fail("attempt_ledger_file_identity_invalid")
        return (metadata.st_dev, metadata.st_ino)

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(
            canonical_json_bytes(
                {
                    "path": os.fspath(self.path),
                    "device": self._identity[0],
                    "inode": self._identity[1],
                }
            )
        )

    def _assert_identity(self) -> None:
        if self._file_identity() != self._identity:
            _fail("attempt_ledger_replaced")

    @staticmethod
    def _decode_lines(payload: bytes) -> list[dict[str, Any]]:
        if not payload:
            return []
        if not payload.endswith(b"\n"):
            _fail("attempt_ledger_truncated")
        documents = []
        for index, line in enumerate(payload.splitlines()):
            value = strict_canonical_json_loads(
                line,
                role=f"attempt_ledger_line_{index}",
            )
            if not isinstance(value, dict):
                _fail("attempt_ledger_line_not_object")
            documents.append(cast(dict[str, Any], value))
        return documents

    def attestations(self) -> list[dict[str, Any]]:
        attestations, _content_sha256 = self.snapshot()
        return attestations

    def snapshot(self) -> tuple[list[dict[str, Any]], str]:
        self._assert_identity()
        payload = _read_regular_file(
            self.path,
            maximum_bytes=MAX_DOCUMENT_BYTES,
            role="attempt_ledger",
        )
        return self._decode_lines(payload), _sha256_bytes(payload)

    def append(
        self,
        *,
        policy: VerifiedCampaignTrustPolicy,
        attestation: Mapping[str, Any],
    ) -> None:
        self._assert_identity()
        flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise VerifiedTransitionError("attempt_ledger_open_failed") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            metadata = os.fstat(descriptor)
            if (
                (metadata.st_dev, metadata.st_ino) != self._identity
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or (self._append_only_flag and not metadata.st_flags & self._append_only_flag)
            ):
                _fail("attempt_ledger_file_identity_invalid")
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 65_536)
                if not chunk:
                    break
                chunks.append(chunk)
            existing = self._decode_lines(b"".join(chunks))
            if existing:
                previous_payload = existing[-1]["signed_payload"]["payload"]
                if previous_payload.get("event_type") == "episode_terminal":
                    _fail("attempt_ledger_already_terminal")
                previous_sha256 = _sha256_bytes(canonical_json_bytes(previous_payload))
            else:
                previous_sha256 = "0" * 64
            signed_payload = attestation.get("signed_payload")
            event = signed_payload.get("payload") if isinstance(signed_payload, Mapping) else None
            if not isinstance(event, Mapping):
                _fail("attempt_ledger_event_missing")
            common_fields = {
                "schema",
                "episode_id",
                "protocol_sha256",
                "immutable_context_sha256",
                "runner_session_sha256",
                "sequence",
                "previous_event_sha256",
                "event_time_unix_ns",
                "event_type",
            }
            expected_event = build_attempt_ledger_event_payload(
                episode_id=cast(str, event.get("episode_id")),
                protocol_sha256=cast(str, event.get("protocol_sha256")),
                immutable_context_sha256=cast(
                    str,
                    event.get("immutable_context_sha256"),
                ),
                sequence=cast(int, event.get("sequence")),
                previous_event_sha256=cast(
                    str,
                    event.get("previous_event_sha256"),
                ),
                event_time_unix_ns=cast(
                    int,
                    event.get("event_time_unix_ns"),
                ),
                event_type=cast(str, event.get("event_type")),
                event_fields={key: event[key] for key in set(event) - common_fields},
            )
            if (
                dict(event) != expected_event
                or event.get("sequence") != len(existing)
                or event.get("previous_event_sha256") != previous_sha256
            ):
                _fail("attempt_ledger_event_chain_invalid")
            verified = _verify_role_attestation(
                policy,
                attestation,
                role=CAMPAIGN_RUNNER,
                expected_payload=event,
            )
            event_time = _require_signed_second(
                event.get("event_time_unix_ns"),
                role="attempt_ledger_event_time",
            )
            if event_time != verified["signed_at_unix"] * 1_000_000_000:
                _fail("attempt_ledger_event_timestamp_mismatch")
            encoded = canonical_json_bytes(dict(attestation)) + b"\n"
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    _fail("attempt_ledger_write_incomplete")
                offset += written
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _validated_attempt_ledger(
    trust_context: TransitionTrustContext,
) -> ExternalAttemptLedger:
    ledger = ExternalAttemptLedger(trust_context.attempt_ledger_path)
    expected_identity = _require_sha256(
        trust_context.expected_attempt_ledger_identity_sha256,
        role="expected_attempt_ledger_identity",
    )
    if ledger.identity_sha256 != expected_identity:
        _fail("attempt_ledger_identity_pin_mismatch")
    return ledger


def _validate_artifact_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_KEYS:
        _fail("artifact_binding_schema_invalid")
    if value.get("schema") != ARTIFACT_BINDING_SCHEMA:
        _fail("artifact_binding_version_invalid")
    normalized = {
        "schema": ARTIFACT_BINDING_SCHEMA,
        "payload_sha256": _require_sha256(
            value.get("payload_sha256"),
            role="artifact_payload_sha256",
        ),
        "byte_length": _require_int(
            value.get("byte_length"),
            role="artifact_byte_length",
            minimum=1,
            maximum=MAX_DOCUMENT_BYTES,
        ),
        "media_type": _require_identifier(
            value.get("media_type"),
            role="artifact_media_type",
        ),
    }
    return normalized


def _read_callable_module_file(callable_object: Callable[..., Any]) -> bytes:
    module = inspect.getmodule(callable_object)
    module_path = inspect.getsourcefile(module) if module is not None else None
    if not isinstance(module_path, str):
        _fail("callable_module_file_missing")
    path = Path(module_path).resolve(strict=True)
    return _read_regular_file(
        path,
        maximum_bytes=MAX_SOURCE_BYTES,
        role="callable_module",
        require_private=False,
    )


@functools.lru_cache(maxsize=4096)
def _loaded_callable_source_sha256(callable_object: Callable[..., Any]) -> str | None:
    """Hash source for one loaded callable object without repeated AST parsing."""

    try:
        return _sha256_bytes(inspect.getsource(callable_object).encode("utf-8"))
    except (OSError, TypeError):
        return None


def _shallow_runtime_identity(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return {
            "kind": "bounded",
            "module": type(value).__module__,
            "qualname": type(value).__qualname__,
        }
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        return {"kind": "float64", "ieee754_hex": struct.pack("!d", value).hex()}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "sha256": _sha256_bytes(value),
        }
    if isinstance(value, re.Pattern):
        return {
            "kind": "regex",
            "pattern": value.pattern,
            "flags": value.flags,
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_shallow_runtime_identity(item, depth=depth + 1) for item in value],
        }
    if (
        isinstance(value, Mapping)
        and len(value) <= 64
        and all(isinstance(key, str) for key in value)
    ):
        return {
            "kind": "mapping",
            "items": {
                key: _shallow_runtime_identity(value[key], depth=depth + 1) for key in sorted(value)
            },
        }
    if callable(value):
        target = inspect.unwrap(value.__func__ if inspect.ismethod(value) else value)
        code = getattr(target, "__code__", None)
        bound_self = getattr(value, "__self__", None)
        return {
            "kind": "callable_leaf",
            "module": getattr(target, "__module__", ""),
            "qualname": getattr(target, "__qualname__", getattr(target, "__name__", "")),
            "runtime_code_sha256": (
                _sha256_bytes(marshal.dumps(code)) if code is not None else None
            ),
            "defaults": _shallow_runtime_identity(
                (
                    getattr(target, "__defaults__", None),
                    getattr(target, "__kwdefaults__", None),
                ),
                depth=depth + 1,
            ),
            "closure": [
                _shallow_runtime_identity(cell.cell_contents, depth=depth + 1)
                for cell in (getattr(target, "__closure__", None) or ())
            ],
            "bound_self": (
                _shallow_runtime_identity(bound_self, depth=depth + 1)
                if isinstance(bound_self, re.Pattern)
                else None
            ),
        }
    return {
        "kind": "typed_leaf",
        "module": type(value).__module__,
        "qualname": type(value).__qualname__,
    }


def _runtime_value_identity(
    value: Any,
    *,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
) -> Any:
    if depth > 8:
        _fail("callable_runtime_value_too_deep")
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "module": type(value).__module__,
            "qualname": type(value).__qualname__,
            "name": value.name,
            "value": _runtime_value_identity(value.value, depth=depth + 1, seen=seen),
        }
    if isinstance(value, float):
        return {
            "kind": "float64",
            "ieee754_hex": struct.pack("!d", value).hex(),
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "sha256": _sha256_bytes(value),
        }
    if isinstance(value, re.Pattern):
        return {
            "kind": "regex",
            "pattern": value.pattern,
            "flags": value.flags,
        }
    if isinstance(value, Path):
        return {"kind": "path", "value": os.fspath(value)}
    if isinstance(value, contextvars.ContextVar):
        return {
            "kind": "contextvar",
            "name": value.name,
        }
    if isinstance(value, logging.Logger):
        # A logger is named, not composed: walking one reaches the logging
        # manager, every logger in the process, and a placeholder's map keyed
        # by Logger objects, and which of those exist depends on what was
        # imported first (2026-09-20: 59 fixtures failed on a mapping with
        # Logger keys after an unrelated import order change).
        return {"kind": "logger", "name": value.name}
    if isinstance(value, types.ModuleType):
        try:
            module_path = inspect.getsourcefile(value)
        except TypeError:
            module_path = None
        module_file_sha256 = None
        if isinstance(module_path, str) and Path(module_path).is_file():
            module_file_sha256 = _sha256_bytes(
                _read_regular_file(
                    Path(module_path).resolve(strict=True),
                    maximum_bytes=MAX_SOURCE_BYTES,
                    role="callable_module",
                    require_private=False,
                )
            )
        return {
            "kind": "module",
            "name": value.__name__,
            "file_sha256": module_file_sha256,
        }
    identity = id(value)
    if identity in seen:
        return {"kind": "cycle", "type": type(value).__qualname__}
    next_seen = seen | {identity}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [
                _runtime_value_identity(
                    item,
                    depth=depth + 1,
                    seen=next_seen,
                )
                for item in value
            ],
        }
    if isinstance(value, (set, frozenset)):
        items = [
            _runtime_value_identity(
                item,
                depth=depth + 1,
                seen=next_seen,
            )
            for item in value
        ]
        items.sort(key=canonical_json_bytes)
        return {"kind": type(value).__name__, "items": items}
    if isinstance(value, Mapping):
        if len(value) > 128 or any(not isinstance(key, str) for key in value):
            _fail("callable_runtime_mapping_invalid")
        return {
            "kind": "mapping",
            "items": {
                key: _runtime_value_identity(
                    value[key],
                    depth=depth + 1,
                    seen=next_seen,
                )
                for key in sorted(value)
            },
        }
    if callable(value):
        target = inspect.unwrap(value)
        code = getattr(target, "__code__", None)
        callable_seen = next_seen | {id(target)}
        source_sha256 = _loaded_callable_source_sha256(target) if code is None else None
        referenced_globals: dict[str, Any] = {}
        globals_map = getattr(target, "__globals__", None)
        if depth == 0 and code is not None and isinstance(globals_map, Mapping):
            for name in sorted(set(code.co_names)):
                if name in globals_map:
                    referenced_globals[name] = _runtime_value_identity(
                        globals_map[name],
                        depth=depth + 1,
                        seen=callable_seen,
                    )
        class_members: dict[str, Any] = {}
        if inspect.isclass(target):
            ignored_members = {
                "__annotations__",
                "__dict__",
                "__doc__",
                "__module__",
                "__weakref__",
            }
            for name, member in sorted(vars(target).items()):
                if name in ignored_members:
                    continue
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if isinstance(member, property):
                    class_members[name] = {
                        "kind": "property",
                        "getter": (
                            _shallow_runtime_identity(member.fget)
                            if member.fget is not None
                            else None
                        ),
                        "setter": (
                            _shallow_runtime_identity(member.fset)
                            if member.fset is not None
                            else None
                        ),
                        "deleter": (
                            _shallow_runtime_identity(member.fdel)
                            if member.fdel is not None
                            else None
                        ),
                    }
                elif callable(member):
                    class_members[name] = _shallow_runtime_identity(member)
                elif member is None or type(member) in {bool, int, str}:
                    class_members[name] = _runtime_value_identity(
                        member,
                        depth=depth + 1,
                        seen=callable_seen,
                    )
        return {
            "kind": "callable",
            "module": getattr(target, "__module__", ""),
            "qualname": getattr(target, "__qualname__", ""),
            "runtime_code_sha256": (
                _sha256_bytes(marshal.dumps(code)) if code is not None else None
            ),
            "source_sha256": source_sha256,
            "defaults": _runtime_value_identity(
                (
                    getattr(target, "__defaults__", None),
                    getattr(target, "__kwdefaults__", None),
                ),
                depth=depth + 1,
                seen=callable_seen,
            ),
            "closure": [
                _runtime_value_identity(
                    cell.cell_contents,
                    depth=depth + 1,
                    seen=callable_seen,
                )
                for cell in (getattr(target, "__closure__", None) or ())
            ],
            "referenced_globals": referenced_globals,
            "class_members_sha256": _sha256_bytes(canonical_json_bytes(class_members)),
        }
    state = getattr(value, "__dict__", None)
    if (
        isinstance(state, Mapping)
        and len(state) <= 128
        and all(isinstance(key, str) for key in state)
    ):
        return {
            "kind": "object",
            "type": _runtime_value_identity(
                type(value),
                depth=depth + 1,
                seen=next_seen,
            ),
            "state": {
                key: _runtime_value_identity(
                    state[key],
                    depth=depth + 1,
                    seen=next_seen,
                )
                for key in sorted(state)
            },
        }
    _fail("callable_runtime_value_unavailable")


def callable_commitment(callable_object: Callable[..., Any]) -> dict[str, Any]:
    """Bind the exact Python callable, source file, and runtime ABI."""

    if not callable(callable_object):
        _fail("callable_invalid")
    target = inspect.unwrap(callable_object)
    module_name = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        _fail("callable_identity_missing")
    source_sha256 = _loaded_callable_source_sha256(target)
    if source_sha256 is None:
        _fail("callable_source_unavailable")
    cache_tag = sys.implementation.cache_tag
    if not isinstance(cache_tag, str) or not cache_tag:
        _fail("runtime_cache_tag_missing")
    code = getattr(target, "__code__", None)
    defaults = _runtime_value_identity(
        (
            getattr(target, "__defaults__", None),
            getattr(target, "__kwdefaults__", None),
        )
    )
    closure = [
        _runtime_value_identity(cell.cell_contents)
        for cell in (getattr(target, "__closure__", None) or ())
    ]
    globals_map = getattr(target, "__globals__", None)
    referenced_globals: dict[str, Any] = {}
    if code is not None and isinstance(globals_map, Mapping):
        for name in sorted(set(code.co_names)):
            if name in globals_map:
                runtime_value = globals_map[name]
                identity = _runtime_value_identity(runtime_value)
                if isinstance(runtime_value, types.ModuleType):
                    module_globals = vars(runtime_value)
                    identity = {
                        **identity,
                        "referenced_attributes": {
                            attribute: _runtime_value_identity(module_globals[attribute])
                            for attribute in sorted(set(code.co_names))
                            if attribute in module_globals
                        },
                    }
                referenced_globals[name] = identity
    return {
        "schema": CALLABLE_COMMITMENT_SCHEMA,
        "module": module_name,
        "qualname": qualname,
        "runtime_code_sha256": (_sha256_bytes(marshal.dumps(code)) if code is not None else None),
        "defaults_sha256": _sha256_bytes(canonical_json_bytes(defaults)),
        "closure_sha256": _sha256_bytes(canonical_json_bytes(closure)),
        "globals_sha256": _sha256_bytes(canonical_json_bytes(referenced_globals)),
        "callable_source_sha256": source_sha256,
        "module_file_sha256": _sha256_bytes(_read_callable_module_file(target)),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": cache_tag,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }


def _validate_callable_commitment(
    value: Any,
    callable_object: Callable[..., Any],
    *,
    role: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CALLABLE_KEYS:
        _fail(f"{role}_schema_invalid")
    expected = callable_commitment(callable_object)
    if dict(value) != expected:
        _fail(f"{role}_mismatch")
    return expected


def _callable_dependency_closure(
    callable_object: Callable[..., Any],
) -> dict[str, Any]:
    """Bind loaded same-module callees and callable registries recursively."""

    pending = [inspect.unwrap(callable_object)]
    rows: dict[str, Any] = {}
    while pending:
        current = inspect.unwrap(pending.pop())
        module_name = getattr(current, "__module__", None)
        qualname = getattr(current, "__qualname__", None)
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            _fail("callable_dependency_identity_missing")
        identity = f"{module_name}:{qualname}"
        if identity in rows:
            continue
        if len(rows) >= 256:
            _fail("callable_dependency_limit_exceeded")
        rows[identity] = callable_commitment(current)
        code = getattr(current, "__code__", None)
        globals_map = getattr(current, "__globals__", None)
        if code is None or not isinstance(globals_map, Mapping):
            continue
        for name in sorted(set(code.co_names)):
            dependency = globals_map.get(name)
            candidates: list[Any] = []
            if callable(dependency):
                candidates.append(dependency)
            elif isinstance(dependency, Mapping) and len(dependency) <= 256:
                candidates.extend(value for value in dependency.values() if callable(value))
            for candidate in candidates:
                target = inspect.unwrap(
                    candidate.__func__ if inspect.ismethod(candidate) else candidate
                )
                if getattr(target, "__module__", None) == module_name:
                    pending.append(target)
    return {identity: rows[identity] for identity in sorted(rows)}


def _source_closure(
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    scorer_registry = {
        scorer_id: _callable_dependency_closure(scorer)
        for scorer_id, scorer in frontier_scorer_callables()
    }
    return {
        "frontier_task_score": _callable_dependency_closure(FrontierTask.score),
        "frontier_score_task": _callable_dependency_closure(score_task),
        "frontier_parser": _callable_dependency_closure(parse_final_answer),
        "frontier_registry": _callable_dependency_closure(FrontierTaskRegistry.generate),
        "frontier_domain_scorers": scorer_registry,
        "independent_scorer": _callable_dependency_closure(independent_scorer),
    }


def verifier_implementation_manifest(
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the exact loaded verifier implementation and runtime."""

    source_closure = _source_closure(independent_scorer)
    return {
        "schema": VERIFIER_IMPLEMENTATION_SCHEMA,
        "source_closure_sha256": _sha256_bytes(canonical_json_bytes(source_closure)),
        "source_closure": source_closure,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }


def verifier_implementation_identity(
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
) -> str:
    """Return the policy pin for the currently loaded verifier closure."""

    return _sha256_bytes(canonical_json_bytes(verifier_implementation_manifest(independent_scorer)))


def _validate_manifest_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_length",
    }:
        _fail("execution_manifest_entry_schema_invalid")
    path = value.get("path")
    if (
        not isinstance(path, str)
        or not path
        or len(path.encode("ascii", errors="ignore")) != len(path)
        or len(path) > 512
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _fail("execution_manifest_entry_path_invalid")
    return {
        "path": path,
        "sha256": _require_sha256(
            value.get("sha256"),
            role="execution_manifest_entry",
        ),
        "byte_length": _require_int(
            value.get("byte_length"),
            role="execution_manifest_entry_size",
            minimum=1,
        ),
    }


def _build_component_manifest(
    role: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if role not in _EXECUTION_COMPONENT_ROLES:
        _fail("execution_manifest_component_role_invalid")
    if not isinstance(entries, Sequence) or isinstance(
        entries,
        (str, bytes, bytearray),
    ):
        _fail("execution_manifest_component_entries_invalid")
    normalized = [_validate_manifest_entry(entry) for entry in entries]
    if not normalized or len(normalized) > 16_384:
        _fail("execution_manifest_component_entries_invalid")
    normalized.sort(key=lambda entry: entry["path"])
    if len({entry["path"] for entry in normalized}) != len(normalized):
        _fail("execution_manifest_component_path_duplicate")
    return {
        "role": role,
        "entries": normalized,
        "file_count": len(normalized),
        "total_bytes": sum(entry["byte_length"] for entry in normalized),
        "root_sha256": _sha256_bytes(canonical_json_bytes(normalized)),
    }


def _component_manifest_from_root(role: str, root: Path) -> dict[str, Any]:
    lexical_root = _lexical_absolute(root)
    _reject_symlink_chain(lexical_root)
    if not lexical_root.exists():
        _fail("execution_component_root_missing")
    candidates: list[tuple[str, Path]] = []
    if lexical_root.is_file():
        candidates.append((lexical_root.name, lexical_root))
    elif lexical_root.is_dir():
        for path in sorted(lexical_root.rglob("*")):
            _reject_symlink_chain(path)
            if path.is_file():
                candidates.append(
                    (
                        path.relative_to(lexical_root).as_posix(),
                        path,
                    )
                )
            elif not path.is_dir():
                _fail("execution_component_special_file_rejected")
    else:
        _fail("execution_component_root_invalid")
    entries = []
    for relative_path, path in candidates:
        digest, byte_length = _hash_regular_file(
            path,
            role=f"execution_component_{role}",
        )
        entries.append(
            {
                "path": relative_path,
                "sha256": digest,
                "byte_length": byte_length,
            }
        )
    return _build_component_manifest(role, entries)


def build_execution_manifest(
    *,
    manifest_id: str,
    component_roots: Mapping[str, Path],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    created_at_unix_ns: int,
) -> dict[str, Any]:
    """Build a reconstructable execution manifest for out-of-band pinning."""

    if (
        not isinstance(component_roots, Mapping)
        or set(component_roots) != _EXECUTION_COMPONENT_ROLES
    ):
        _fail("execution_manifest_components_invalid")
    components = [
        _component_manifest_from_root(role, component_roots[role])
        for role in sorted(_EXECUTION_COMPONENT_ROLES)
    ]
    roots = {component["role"]: component["root_sha256"] for component in components}
    model_identity = _sha256_bytes(
        canonical_json_bytes(
            {
                "base_checkpoint_sha256": roots["base_checkpoint"],
                "adapter_stack_sha256": roots["adapter_stack"],
            }
        )
    )
    body = {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "manifest_id": _require_identifier(
            manifest_id,
            role="execution_manifest_id",
        ),
        "components": components,
        "component_roots": roots,
        "model_identity_sha256": model_identity,
        "token_encoder_callable": callable_commitment(token_encoder),
        "token_decoder_callable": callable_commitment(token_decoder),
        "generation_worker_identity_sha256": roots["generation_worker"],
        "verifier_implementation_sha256": verifier_implementation_identity(independent_scorer),
        "candidate_input_template_sha256": _sha256_bytes(CANDIDATE_INPUT_SCHEMA.encode("ascii")),
        "created_at_unix_ns": _require_int(
            created_at_unix_ns,
            role="execution_manifest_created_at",
            minimum=1,
        ),
    }
    return _seal_document(body)


def _validate_execution_manifest(
    value: Any,
    *,
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    component_roots: Mapping[str, Path],
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "manifest_id",
        "components",
        "component_roots",
        "model_identity_sha256",
        "token_encoder_callable",
        "token_decoder_callable",
        "generation_worker_identity_sha256",
        "verifier_implementation_sha256",
        "candidate_input_template_sha256",
        "created_at_unix_ns",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _fail("execution_manifest_schema_invalid")
    if value.get("schema") != EXECUTION_MANIFEST_SCHEMA:
        _fail("execution_manifest_version_invalid")
    _validate_digest(value, role="execution_manifest")
    _require_identifier(value.get("manifest_id"), role="execution_manifest_id")
    raw_components = value.get("components")
    if not isinstance(raw_components, list):
        _fail("execution_manifest_components_invalid")
    components = [
        _build_component_manifest(
            cast(str, component.get("role")) if isinstance(component, Mapping) else "",
            cast(Sequence[Mapping[str, Any]], component.get("entries"))
            if isinstance(component, Mapping)
            else (),
        )
        for component in raw_components
    ]
    reconstructed_components = (
        [
            _component_manifest_from_root(role, component_roots[role])
            for role in sorted(_EXECUTION_COMPONENT_ROLES)
        ]
        if (
            isinstance(component_roots, Mapping)
            and set(component_roots) == _EXECUTION_COMPONENT_ROLES
        )
        else []
    )
    if components != reconstructed_components:
        _fail("execution_manifest_component_content_mismatch")
    if components != raw_components:
        _fail("execution_manifest_component_mismatch")
    roots = {component["role"]: component["root_sha256"] for component in components}
    if set(roots) != _EXECUTION_COMPONENT_ROLES or value.get("component_roots") != roots:
        _fail("execution_manifest_component_roots_mismatch")
    expected_model_identity = _sha256_bytes(
        canonical_json_bytes(
            {
                "base_checkpoint_sha256": roots["base_checkpoint"],
                "adapter_stack_sha256": roots["adapter_stack"],
            }
        )
    )
    if value.get("model_identity_sha256") != expected_model_identity:
        _fail("execution_manifest_model_identity_mismatch")
    _validate_callable_commitment(
        value.get("token_encoder_callable"),
        token_encoder,
        role="execution_manifest_token_encoder",
    )
    _validate_callable_commitment(
        value.get("token_decoder_callable"),
        token_decoder,
        role="execution_manifest_token_decoder",
    )
    _require_sha256(
        value.get("generation_worker_identity_sha256"),
        role="execution_manifest_generation_worker",
    )
    if value.get("verifier_implementation_sha256") != verifier_implementation_identity(
        independent_scorer
    ):
        _fail("execution_manifest_verifier_implementation_mismatch")
    if value.get("candidate_input_template_sha256") != _sha256_bytes(
        CANDIDATE_INPUT_SCHEMA.encode("ascii")
    ):
        _fail("execution_manifest_candidate_template_mismatch")
    _require_int(
        value.get("created_at_unix_ns"),
        role="execution_manifest_created_at",
        minimum=1,
    )
    return dict(value)


def _serialize_frontier_task(task: FrontierTask) -> dict[str, Any]:
    task = _require_exact_frontier_task(task)
    return {
        "schema": task.schema,
        "public": task.public.to_dict(),
        "blinded_answer": {
            "commitment_sha256": task.blinded_answer.commitment_sha256,
            "canonical_bytes_b64": base64.b64encode(task.blinded_answer._canonical_bytes).decode(
                "ascii"
            ),
        },
    }


def _deserialize_frontier_task(value: Any) -> FrontierTask:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "public",
        "blinded_answer",
    }:
        _fail("calibration_task_schema_invalid")
    blinded = value.get("blinded_answer")
    if not isinstance(blinded, Mapping) or set(blinded) != {
        "commitment_sha256",
        "canonical_bytes_b64",
    }:
        _fail("calibration_blinded_answer_schema_invalid")
    try:
        canonical_bytes = base64.b64decode(
            cast(str, blinded.get("canonical_bytes_b64")),
            validate=True,
        )
    except (ValueError, TypeError, binascii.Error) as exc:
        raise VerifiedTransitionError("calibration_blinded_answer_encoding_invalid") from exc
    task = FrontierTask(
        schema=cast(str, value.get("schema")),
        public=PublicTaskRecord.from_dict(cast(Mapping[str, Any], value.get("public"))),
        blinded_answer=BlindedAnswerPayload(
            cast(str, blinded.get("commitment_sha256")),
            canonical_bytes,
        ),
    )
    return _require_exact_frontier_task(task)


def _different_calibration_answer(value: Any) -> Any:
    """Produce a same-shape, deterministically wrong JSON answer."""

    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1 if value < MAX_INTEGER else value - 1
    if isinstance(value, str):
        return f"{value}_calibration_negative"
    if isinstance(value, list) and value:
        return [_different_calibration_answer(value[0]), *value[1:]]
    if isinstance(value, dict) and value:
        first = sorted(value)[0]
        return {
            key: _different_calibration_answer(item) if key == first else item
            for key, item in value.items()
        }
    _fail("calibration_expected_answer_not_perturbable")


def build_calibration_case(
    *,
    task: FrontierTask,
    case_kind: str,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    task = _require_exact_frontier_task(task)
    if case_kind == "canonical_positive":
        expected = task.reveal_for_verifier()["expected"]
        response = (
            f"{FINAL_ANSWER_MARKER} {json.dumps(expected, sort_keys=True, separators=(',', ':'))}"
        )
    elif case_kind == "missing_marker_negative":
        response = "CALIBRATION NEGATIVE CONTROL: marker intentionally absent."
    elif case_kind == "parsed_wrong_negative":
        expected = task.reveal_for_verifier()["expected"]
        wrong = _different_calibration_answer(expected)
        response = (
            f"{FINAL_ANSWER_MARKER} {json.dumps(wrong, sort_keys=True, separators=(',', ':'))}"
        )
    else:
        _fail("calibration_case_kind_invalid")
    primary = _validate_score_output(task.score(response).to_dict(), task)
    independent = _validate_independent_output(independent_scorer(task, response))
    return {
        "schema": CALIBRATION_CASE_SCHEMA,
        "case_kind": case_kind,
        "task": _serialize_frontier_task(task),
        "response_b64": base64.b64encode(response.encode("utf-8")).decode("ascii"),
        "primary_output": primary,
        "independent_output": independent,
    }


def _replay_calibration_case(
    value: Any,
    *,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "case_kind",
        "task",
        "response_b64",
        "primary_output",
        "independent_output",
    }:
        _fail("calibration_case_schema_invalid")
    if value.get("schema") != CALIBRATION_CASE_SCHEMA:
        _fail("calibration_case_version_invalid")
    task = _deserialize_frontier_task(value.get("task"))
    case_kind = cast(str, value.get("case_kind"))
    expected_case = build_calibration_case(
        task=task,
        case_kind=case_kind,
        independent_scorer=independent_scorer,
    )
    try:
        response = base64.b64decode(
            cast(str, value.get("response_b64")),
            validate=True,
        ).decode("utf-8")
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error) as exc:
        raise VerifiedTransitionError("calibration_response_encoding_invalid") from exc
    expected_response = base64.b64decode(
        expected_case["response_b64"],
        validate=True,
    ).decode("utf-8")
    if response != expected_response:
        _fail("calibration_control_response_mismatch")
    primary = _validate_score_output(task.score(response).to_dict(), task)
    independent = _validate_independent_output(independent_scorer(task, response))
    if value.get("primary_output") != primary:
        _fail("calibration_primary_replay_mismatch")
    if value.get("independent_output") != independent:
        _fail("calibration_independent_replay_mismatch")
    expected_correct = case_kind == "canonical_positive"
    expected_parsed = case_kind != "missing_marker_negative"
    return {
        **dict(value),
        "_task_id": task.task_id,
        "_domain": task.domain,
        "_scorer_id": task.public.scorer_id,
        "_case_kind": case_kind,
        "_parsed": bool(primary["parsed"]) == expected_parsed,
        "_agreement": (
            primary["parsed"],
            primary["correct"],
            primary["normalized_answer_sha256"],
        )
        == (
            independent["parsed"],
            independent["correct"],
            independent["normalized_answer_sha256"],
        ),
        "_false_positive": primary["correct"] and not expected_correct,
        "_false_negative": not primary["correct"] and expected_correct,
    }


def build_calibration_payload(
    *,
    verifier_implementation_sha256: str,
    trust_policy_sha256: str,
    cases: Sequence[Mapping[str, Any]],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    acceptance_policy_sha256: str,
    calibrated_at_unix_ns: int,
) -> dict[str, Any]:
    """Build the metrics payload an external verifier must sign."""

    if (
        not isinstance(cases, Sequence)
        or isinstance(cases, (str, bytes, bytearray))
        or len(cases) < 2
        or len(cases) > 16_384
    ):
        _fail("calibration_cases_invalid")
    replayed_cases = [
        _replay_calibration_case(
            case,
            independent_scorer=independent_scorer,
        )
        for case in cases
    ]
    normalized_cases = [
        {key: value for key, value in case.items() if not key.startswith("_")}
        for case in replayed_cases
    ]
    task_ids = [cast(str, case["_task_id"]) for case in replayed_cases]
    if len(set(task_ids)) != len(task_ids):
        _fail("calibration_task_duplicate")
    coverage: dict[tuple[str, str], set[str]] = {}
    for case in replayed_cases:
        key = (
            cast(str, case["_domain"]),
            cast(str, case["_scorer_id"]),
        )
        coverage.setdefault(key, set()).add(cast(str, case["_case_kind"]))
    if any(
        case_kinds
        != {
            "canonical_positive",
            "missing_marker_negative",
            "parsed_wrong_negative",
        }
        for case_kinds in coverage.values()
    ):
        _fail("calibration_control_coverage_incomplete")
    parsed_count = sum(bool(case["_parsed"]) for case in replayed_cases)
    agreement_count = sum(bool(case["_agreement"]) for case in replayed_cases)
    false_positive_count = sum(bool(case["_false_positive"]) for case in replayed_cases)
    false_negative_count = sum(bool(case["_false_negative"]) for case in replayed_cases)
    payload = {
        "schema": CALIBRATION_PAYLOAD_SCHEMA,
        "verifier_implementation_sha256": _require_sha256(
            verifier_implementation_sha256,
            role="calibration_verifier_implementation",
        ),
        "trust_policy_sha256": _require_sha256(
            trust_policy_sha256,
            role="calibration_trust_policy",
        ),
        "task_manifest_sha256": _sha256_bytes(canonical_json_bytes(task_ids)),
        "cases": normalized_cases,
        "task_count": len(normalized_cases),
        "parsed_count": parsed_count,
        "agreement_count": agreement_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "acceptance_policy_sha256": _require_sha256(
            acceptance_policy_sha256,
            role="calibration_acceptance_policy",
        ),
        "calibrated_at_unix_ns": _require_signed_second(
            calibrated_at_unix_ns,
            role="calibration_time",
        ),
    }
    return payload


def seal_calibration_evidence(
    payload: Mapping[str, Any],
    *,
    evidence_verifier_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "verifier_implementation_sha256",
        "trust_policy_sha256",
        "task_manifest_sha256",
        "cases",
        "task_count",
        "parsed_count",
        "agreement_count",
        "false_positive_count",
        "false_negative_count",
        "acceptance_policy_sha256",
        "calibrated_at_unix_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        _fail("calibration_payload_schema_invalid")
    return _seal_document(
        {
            **dict(payload),
            "schema": CALIBRATION_EVIDENCE_SCHEMA,
            "evidence_verifier_attestation": dict(evidence_verifier_attestation),
        }
    )


def _validate_calibration_evidence(
    value: Any,
    *,
    policy: VerifiedCampaignTrustPolicy,
    expected_sha256: str,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    observed_at_unix: int,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "verifier_implementation_sha256",
        "trust_policy_sha256",
        "task_manifest_sha256",
        "cases",
        "task_count",
        "parsed_count",
        "agreement_count",
        "false_positive_count",
        "false_negative_count",
        "acceptance_policy_sha256",
        "calibrated_at_unix_ns",
        "evidence_verifier_attestation",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _fail("calibration_evidence_schema_invalid")
    if value.get("schema") != CALIBRATION_EVIDENCE_SCHEMA:
        _fail("calibration_evidence_version_invalid")
    _validate_digest(value, role="calibration_evidence")
    if _sha256_bytes(canonical_json_bytes(dict(value))) != _require_sha256(
        expected_sha256,
        role="expected_calibration_evidence",
    ):
        _fail("calibration_evidence_pin_mismatch")
    payload = build_calibration_payload(
        verifier_implementation_sha256=cast(
            str,
            value.get("verifier_implementation_sha256"),
        ),
        trust_policy_sha256=cast(str, value.get("trust_policy_sha256")),
        cases=cast(Sequence[Mapping[str, Any]], value.get("cases")),
        independent_scorer=independent_scorer,
        acceptance_policy_sha256=cast(
            str,
            value.get("acceptance_policy_sha256"),
        ),
        calibrated_at_unix_ns=cast(int, value.get("calibrated_at_unix_ns")),
    )
    if any(key != "schema" and value.get(key) != expected for key, expected in payload.items()):
        _fail("calibration_payload_reconstruction_mismatch")
    if payload["verifier_implementation_sha256"] != (
        verifier_implementation_identity(independent_scorer)
    ):
        _fail("calibration_verifier_implementation_mismatch")
    if payload["trust_policy_sha256"] != policy.policy_sha256:
        _fail("calibration_trust_policy_mismatch")
    if (
        payload["parsed_count"] != payload["task_count"]
        or payload["agreement_count"] != payload["task_count"]
        or payload["false_positive_count"] != 0
        or payload["false_negative_count"] != 0
    ):
        _fail("calibration_acceptance_failed")
    attestation = _verify_role_attestation(
        policy,
        value.get("evidence_verifier_attestation"),
        role=EVIDENCE_VERIFIER,
        expected_payload=payload,
        not_after_unix=observed_at_unix,
    )
    calibrated_at_ns = payload["calibrated_at_unix_ns"]
    if calibrated_at_ns != attestation["signed_at_unix"] * 1_000_000_000:
        _fail("calibration_timestamp_mismatch")
    return dict(value)


def _require_calibration_coverage_for_task(
    evidence: Mapping[str, Any],
    task: FrontierTask,
) -> None:
    task = _require_exact_frontier_task(task)
    cases = evidence.get("cases")
    if not isinstance(cases, list):
        _fail("calibration_cases_invalid")
    matching_kinds = {
        cast(str, case.get("case_kind"))
        for case in cases
        if isinstance(case, Mapping)
        and (calibration_task := _deserialize_frontier_task(case.get("task"))).domain == task.domain
        and calibration_task.public.scorer_id == task.public.scorer_id
    }
    if matching_kinds != {
        "canonical_positive",
        "missing_marker_negative",
        "parsed_wrong_negative",
    }:
        _fail("calibration_target_coverage_missing")


def canonical_candidate_model_input(task: FrontierTask) -> bytes:
    """Construct the only candidate-visible input accepted by this protocol."""

    task = _require_exact_frontier_task(task)
    return canonical_json_bytes(
        {
            "schema": CANDIDATE_INPUT_SCHEMA,
            "public_task": task.public.to_dict(),
        }
    )


def _validate_score_output(value: Any, task: FrontierTask) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SCORE_OUTPUT_KEYS:
        _fail("verifier_output_schema_invalid")
    if (
        value.get("schema") != "aura.latent_cortex.frontier_score_result.v1"
        or value.get("task_id") != task.task_id
        or value.get("domain") != task.domain
        or value.get("scorer_id") != task.public.scorer_id
    ):
        _fail("verifier_output_identity_mismatch")
    parsed = _require_bool(value.get("parsed"), role="verifier_output_parsed")
    correct = _require_bool(value.get("correct"), role="verifier_output_correct")
    reason = _require_identifier(
        value.get("reason"),
        role="verifier_output_reason",
    )
    normalized = value.get("normalized_answer_sha256")
    if normalized is not None:
        normalized = _require_sha256(
            normalized,
            role="verifier_output_normalized_answer",
        )
    if (not parsed and (correct or normalized is not None)) or (parsed and normalized is None):
        _fail("verifier_output_state_invalid")
    return {
        "schema": value["schema"],
        "task_id": value["task_id"],
        "domain": value["domain"],
        "scorer_id": value["scorer_id"],
        "parsed": parsed,
        "correct": correct,
        "reason": reason,
        "normalized_answer_sha256": normalized,
    }


def _validate_independent_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INDEPENDENT_OUTPUT_KEYS:
        _fail("independent_output_schema_invalid")
    parsed = _require_bool(value.get("parsed"), role="independent_output_parsed")
    correct = _require_bool(value.get("correct"), role="independent_output_correct")
    reason = _require_identifier(
        value.get("reason"),
        role="independent_output_reason",
    )
    normalized = value.get("normalized_answer_sha256")
    if normalized is not None:
        normalized = _require_sha256(
            normalized,
            role="independent_output_normalized_answer",
        )
    if (not parsed and (correct or normalized is not None)) or (parsed and normalized is None):
        _fail("independent_output_state_invalid")
    return {
        "parsed": parsed,
        "correct": correct,
        "reason": reason,
        "normalized_answer_sha256": normalized,
    }


def _semantic_score(output: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        output.get("parsed"),
        output.get("correct"),
        output.get("reason"),
        output.get("normalized_answer_sha256"),
    )


def _validate_expected_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_EXPECTED_KEYS:
        _fail("authority_expectation_schema_invalid")
    result = {
        "authority_id": _require_identifier(
            value.get("authority_id"),
            role="authority_id",
        ),
        "verifier_id": _require_identifier(
            value.get("verifier_id"),
            role="verifier_id",
        ),
        "verifier_version": _require_identifier(
            value.get("verifier_version"),
            role="verifier_version",
        ),
    }
    for field in sorted(_AUTHORITY_EXPECTED_KEYS - set(result)):
        result[field] = _require_sha256(value.get(field), role=field)
    return result


def _verified_transition_policy(
    trust_context: TransitionTrustContext,
    expected: Mapping[str, Any],
    *,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]] | None = None,
    token_decoder: Callable[[Sequence[int]], bytes] | None = None,
) -> VerifiedCampaignTrustPolicy:
    if not isinstance(trust_context, TransitionTrustContext):
        _fail("transition_trust_context_invalid")
    _validated_attempt_ledger(trust_context)
    policy = trust_context.verified_policy()
    if not operationally_isolated_roles(policy):
        _fail("transition_operational_role_custody_required")
    if policy.policy_sha256 != expected["verifier_trust_policy_sha256"]:
        _fail("transition_trust_policy_mismatch")
    runner = policy.role_pin(CAMPAIGN_RUNNER)
    verifier = policy.role_pin(EVIDENCE_VERIFIER)
    observer = policy.role_pin(CONTAMINATION_AUDITOR)
    if expected["producer_identity_sha256"] != runner["key_id"]:
        _fail("transition_producer_identity_not_policy_pinned")
    verifier_identity = verifier_implementation_identity(independent_scorer)
    if (
        expected["verifier_identity_sha256"] != verifier_identity
        or verifier["implementation_sha256"] != verifier_identity
    ):
        _fail("transition_verifier_identity_not_policy_pinned")
    if expected["independent_witness_identity_sha256"] != verifier["key_id"]:
        _fail("transition_witness_identity_not_policy_pinned")
    if observer["implementation_sha256"] != execution_observer_implementation_identity():
        _fail("transition_execution_observer_not_policy_pinned")
    execution_manifest_sha256 = _require_sha256(
        trust_context.expected_execution_manifest_sha256,
        role="transition_execution_manifest",
    )
    if (
        expected["execution_manifest_sha256"] != execution_manifest_sha256
        or _sha256_bytes(canonical_json_bytes(dict(trust_context.execution_manifest)))
        != execution_manifest_sha256
    ):
        _fail("transition_execution_manifest_pin_mismatch")
    if token_encoder is not None and token_decoder is not None:
        manifest = _validate_execution_manifest(
            trust_context.execution_manifest,
            token_encoder=token_encoder,
            token_decoder=token_decoder,
            independent_scorer=independent_scorer,
            component_roots=trust_context.execution_component_roots,
        )
        if manifest["generation_worker_identity_sha256"] != runner["implementation_sha256"]:
            _fail("transition_generation_worker_not_policy_pinned")
    calibration_sha256 = _require_sha256(
        trust_context.expected_calibration_evidence_sha256,
        role="transition_calibration_evidence",
    )
    if (
        expected["calibration_evidence_sha256"] != calibration_sha256
        or _sha256_bytes(canonical_json_bytes(dict(trust_context.calibration_evidence)))
        != calibration_sha256
    ):
        _fail("transition_calibration_evidence_pin_mismatch")
    _validate_calibration_evidence(
        trust_context.calibration_evidence,
        policy=policy,
        expected_sha256=calibration_sha256,
        independent_scorer=independent_scorer,
        observed_at_unix=trust_context.observed_at_unix,
    )
    return policy


def build_frontier_witness_payload(
    store: TransitionArtifactStore,
    *,
    task: FrontierTask,
    response_artifact: Mapping[str, Any],
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    issued_at_unix_ns: int,
) -> dict[str, Any]:
    """Build the exact payload an externally pinned verifier must attest."""

    task = _require_exact_frontier_task(task)
    expected = _validate_expected_authority(expected_authority)
    response_binding = _validate_artifact_binding(response_artifact)
    response_bytes = store.read_bytes(
        response_binding,
        expected_media_type="text/plain;charset=utf-8",
    )
    try:
        response = response_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("response_not_utf8")
    output = _validate_independent_output(independent_scorer(task, response))
    return {
        "schema": WITNESS_PAYLOAD_SCHEMA,
        "authority_id": expected["authority_id"],
        "task_id": task.task_id,
        "public_task_payload_sha256": task.public.task_payload_sha256,
        "answer_commitment_sha256": task.public.answer_commitment_sha256,
        "response_artifact": response_binding,
        "scorer_callable": callable_commitment(independent_scorer),
        "source_closure_sha256": _sha256_bytes(
            canonical_json_bytes(_source_closure(independent_scorer))
        ),
        "score_output": output,
        "witness_identity_sha256": expected["independent_witness_identity_sha256"],
        "verifier_implementation_sha256": expected["verifier_identity_sha256"],
        "execution_manifest_sha256": expected["execution_manifest_sha256"],
        "calibration_evidence_sha256": expected["calibration_evidence_sha256"],
        "issued_at_unix_ns": _require_signed_second(
            issued_at_unix_ns,
            role="witness_issued_at",
        ),
    }


def build_frontier_task_issuer_payload(
    task: FrontierTask,
) -> dict[str, Any]:
    """Return the exact blinded task identity the issuer role must attest."""

    task = _require_exact_frontier_task(task)
    public_bytes = canonical_json_bytes(task.public.to_dict())
    manifest_bytes = build_task_manifest([task]).canonical_bytes()
    return {
        "schema": "aura.verified_transition.task_issuer_payload.v1",
        "task_id": task.task_id,
        "registry_version": task.public.registry_version,
        "public_task_sha256": _sha256_bytes(public_bytes),
        "task_manifest_sha256": _sha256_bytes(manifest_bytes),
        "public_task_payload_sha256": task.public.task_payload_sha256,
        "answer_commitment_sha256": task.public.answer_commitment_sha256,
        "scorer_id": task.public.scorer_id,
        "scorer_version": task.public.scorer_version,
    }


def _verify_transition_task_issuer_attestation(
    *,
    policy: VerifiedCampaignTrustPolicy,
    trust_context: TransitionTrustContext,
    task: FrontierTask,
    not_after_unix: int,
) -> dict[str, Any]:
    return _verify_role_attestation(
        policy,
        trust_context.task_issuer_attestation,
        role=TASK_ISSUER,
        expected_payload=build_frontier_task_issuer_payload(task),
        not_after_unix=not_after_unix,
    )


def issue_frontier_verifier_authority(
    store: TransitionArtifactStore,
    *,
    task: FrontierTask,
    response_artifact: Mapping[str, Any],
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    trust_context: TransitionTrustContext,
    task_issuer_attestation: Mapping[str, Any],
    evidence_verifier_attestation: Mapping[str, Any],
    issued_at_unix_ns: int | None = None,
    sealed_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Score one frontier response through primary and independent authorities."""

    if not isinstance(store, TransitionArtifactStore):
        _fail("artifact_store_invalid")
    task = _require_exact_frontier_task(task)
    expected = _validate_expected_authority(expected_authority)
    policy = _verified_transition_policy(
        trust_context,
        expected,
        independent_scorer=independent_scorer,
    )
    _require_calibration_coverage_for_task(
        trust_context.calibration_evidence,
        task,
    )
    if issued_at_unix_ns is None:
        _fail("witness_issued_at_required")
    issued = _require_signed_second(
        issued_at_unix_ns,
        role="witness_issued_at",
    )
    issuer_payload = build_frontier_task_issuer_payload(task)
    if expected["issuer_commitment_sha256"] != _sha256_bytes(canonical_json_bytes(issuer_payload)):
        _fail("task_issuer_commitment_mismatch")
    verified_context_issuer = _verify_transition_task_issuer_attestation(
        policy=policy,
        trust_context=trust_context,
        task=task,
        not_after_unix=trust_context.observed_at_unix,
    )
    verified_submitted_issuer = _verify_role_attestation(
        policy,
        task_issuer_attestation,
        role=TASK_ISSUER,
        expected_payload=issuer_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    if verified_submitted_issuer != verified_context_issuer:
        _fail("task_issuer_attestation_context_mismatch")
    response_binding = _validate_artifact_binding(response_artifact)
    response_bytes = store.read_bytes(
        response_binding,
        expected_media_type="text/plain;charset=utf-8",
    )
    try:
        response = response_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("response_not_utf8")
    witness_payload = build_frontier_witness_payload(
        store,
        task=task,
        response_artifact=response_binding,
        expected_authority=expected,
        independent_scorer=independent_scorer,
        issued_at_unix_ns=issued,
    )
    verified_attestation = _verify_role_attestation(
        policy,
        evidence_verifier_attestation,
        role=EVIDENCE_VERIFIER,
        expected_payload=witness_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    attested_at_ns = cast(int, verified_attestation["signed_at_unix"]) * 1_000_000_000
    independent_output = _validate_independent_output(independent_scorer(task, response))
    if witness_payload["score_output"] != independent_output:
        _fail("independent_witness_replay_mismatch")
    primary_output = _validate_score_output(task.score(response).to_dict(), task)
    if _semantic_score(primary_output) != _semantic_score(independent_output):
        _fail("independent_scorer_disagreement")

    public_task_artifact = store.put_json(task.public.to_dict())
    manifest_artifact = store.put_bytes(
        build_task_manifest([task]).canonical_bytes(),
        media_type="application/json",
    )
    issuer_attestation_artifact = store.put_json(dict(task_issuer_attestation))
    execution_manifest_artifact = store.put_json(dict(trust_context.execution_manifest))
    calibration_evidence_artifact = store.put_json(dict(trust_context.calibration_evidence))
    primary_score = {
        "schema": PRIMARY_SCORE_SCHEMA,
        "task_id": task.task_id,
        "public_task_payload_sha256": task.public.task_payload_sha256,
        "answer_commitment_sha256": task.public.answer_commitment_sha256,
        "response_artifact": response_binding,
        "score_output": primary_output,
    }
    primary_score_artifact = store.put_json(primary_score)

    if issued != attested_at_ns:
        _fail("witness_attestation_timestamp_mismatch")
    witness = _seal_document(
        {
            "schema": WITNESS_SCORE_SCHEMA,
            "task_id": task.task_id,
            "public_task_payload_sha256": task.public.task_payload_sha256,
            "answer_commitment_sha256": task.public.answer_commitment_sha256,
            "response_artifact": response_binding,
            "witness_identity_sha256": expected["independent_witness_identity_sha256"],
            "scorer_callable": callable_commitment(independent_scorer),
            "score_output": independent_output,
            "trust_policy_sha256": policy.policy_sha256,
            "evidence_verifier_attestation": dict(evidence_verifier_attestation),
            "issued_at_unix_ns": issued,
        }
    )
    witness_artifact = store.put_json(witness)

    sealed = (
        max(issued, time.time_ns())
        if sealed_at_unix_ns is None
        else _require_int(
            sealed_at_unix_ns,
            role="authority_sealed_at",
            minimum=issued,
        )
    )
    source_closure = _source_closure(independent_scorer)
    authority = _seal_document(
        {
            "schema": VERIFIER_AUTHORITY_SCHEMA,
            "authority_id": expected["authority_id"],
            "verifier_id": expected["verifier_id"],
            "verifier_version": expected["verifier_version"],
            "source_closure": source_closure,
            "source_closure_sha256": _sha256_bytes(canonical_json_bytes(source_closure)),
            "public_task_artifact": public_task_artifact,
            "task_manifest_artifact": manifest_artifact,
            "task_issuer_attestation_artifact": issuer_attestation_artifact,
            "task_id": task.task_id,
            "public_task_payload_sha256": task.public.task_payload_sha256,
            "answer_commitment_sha256": task.public.answer_commitment_sha256,
            "issuer_commitment_sha256": expected["issuer_commitment_sha256"],
            "verifier_trust_policy_sha256": expected["verifier_trust_policy_sha256"],
            "calibration_evidence_sha256": expected["calibration_evidence_sha256"],
            "calibration_evidence_artifact": calibration_evidence_artifact,
            "execution_manifest_sha256": expected["execution_manifest_sha256"],
            "execution_manifest_artifact": execution_manifest_artifact,
            "response_artifact": response_binding,
            "primary_score_artifact": primary_score_artifact,
            "independent_witness_artifact": witness_artifact,
            "verifier_output": primary_output,
            "parsed_answer_sha256": primary_output["normalized_answer_sha256"],
            "outcome": "pass" if primary_output["correct"] else "fail",
            "producer_identity_sha256": expected["producer_identity_sha256"],
            "verifier_identity_sha256": expected["verifier_identity_sha256"],
            "independent_witness_identity_sha256": expected["independent_witness_identity_sha256"],
            "issued_at_unix_ns": issued,
            "sealed_at_unix_ns": sealed,
        }
    )
    return authority


def validate_frontier_verifier_authority(
    store: TransitionArtifactStore,
    receipt: Mapping[str, Any],
    *,
    task: FrontierTask,
    response_artifact: Mapping[str, Any],
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    trust_context: TransitionTrustContext,
) -> dict[str, Any]:
    """Reopen every payload and re-execute both scoring implementations."""

    if not isinstance(receipt, Mapping) or set(receipt) != _AUTHORITY_KEYS:
        _fail("verifier_authority_schema_invalid")
    if receipt.get("schema") != VERIFIER_AUTHORITY_SCHEMA:
        _fail("verifier_authority_version_invalid")
    _validate_digest(receipt, role="verifier_authority")
    task = _require_exact_frontier_task(task)
    expected = _validate_expected_authority(expected_authority)
    policy = _verified_transition_policy(
        trust_context,
        expected,
        independent_scorer=independent_scorer,
    )
    _require_calibration_coverage_for_task(
        trust_context.calibration_evidence,
        task,
    )
    issuer_payload = build_frontier_task_issuer_payload(task)
    if expected["issuer_commitment_sha256"] != _sha256_bytes(canonical_json_bytes(issuer_payload)):
        _fail("task_issuer_commitment_mismatch")
    issuer_attestation = store.read_json(
        cast(
            Mapping[str, Any],
            receipt["task_issuer_attestation_artifact"],
        ),
        role="task_issuer_attestation",
    )
    _verify_role_attestation(
        policy,
        issuer_attestation,
        role=TASK_ISSUER,
        expected_payload=issuer_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    for field, value in expected.items():
        if receipt.get(field) != value:
            _fail(f"verifier_authority_{field}_mismatch")
    if (
        receipt.get("task_id") != task.task_id
        or receipt.get("public_task_payload_sha256") != task.public.task_payload_sha256
        or receipt.get("answer_commitment_sha256") != task.public.answer_commitment_sha256
    ):
        _fail("verifier_authority_task_identity_mismatch")
    execution_manifest = store.read_json(
        cast(Mapping[str, Any], receipt["execution_manifest_artifact"]),
        role="execution_manifest",
    )
    if execution_manifest != dict(trust_context.execution_manifest) or receipt.get(
        "execution_manifest_sha256"
    ) != _sha256_bytes(canonical_json_bytes(execution_manifest)):
        _fail("verifier_authority_execution_manifest_mismatch")
    calibration_evidence = store.read_json(
        cast(Mapping[str, Any], receipt["calibration_evidence_artifact"]),
        role="calibration_evidence",
    )
    if calibration_evidence != dict(trust_context.calibration_evidence) or receipt.get(
        "calibration_evidence_sha256"
    ) != _sha256_bytes(canonical_json_bytes(calibration_evidence)):
        _fail("verifier_authority_calibration_evidence_mismatch")
    response_binding = _validate_artifact_binding(response_artifact)
    if receipt.get("response_artifact") != response_binding:
        _fail("verifier_authority_response_mismatch")
    response_bytes = store.read_bytes(
        response_binding,
        expected_media_type="text/plain;charset=utf-8",
    )
    try:
        response = response_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("response_not_utf8")

    public_task = store.read_json(
        cast(Mapping[str, Any], receipt["public_task_artifact"]),
        role="public_task",
    )
    if public_task != task.public.to_dict():
        _fail("verifier_authority_public_task_mismatch")
    manifest = store.read_bytes(
        cast(Mapping[str, Any], receipt["task_manifest_artifact"]),
        expected_media_type="application/json",
    )
    if manifest != build_task_manifest([task]).canonical_bytes():
        _fail("verifier_authority_task_manifest_mismatch")

    expected_closure = _source_closure(independent_scorer)
    if receipt.get("source_closure") != expected_closure:
        _fail("verifier_authority_source_closure_mismatch")
    if receipt.get("source_closure_sha256") != _sha256_bytes(
        canonical_json_bytes(expected_closure)
    ):
        _fail("verifier_authority_source_closure_digest_mismatch")

    primary_output = _validate_score_output(task.score(response).to_dict(), task)
    if receipt.get("verifier_output") != primary_output:
        _fail("verifier_authority_output_mismatch")
    if receipt.get("parsed_answer_sha256") != primary_output["normalized_answer_sha256"]:
        _fail("verifier_authority_parsed_answer_mismatch")
    if receipt.get("outcome") != ("pass" if primary_output["correct"] else "fail"):
        _fail("verifier_authority_outcome_mismatch")

    primary_score = store.read_json(
        cast(Mapping[str, Any], receipt["primary_score_artifact"]),
        role="primary_score",
    )
    if set(primary_score) != _PRIMARY_SCORE_KEYS:
        _fail("primary_score_schema_invalid")
    expected_primary_score = {
        "schema": PRIMARY_SCORE_SCHEMA,
        "task_id": task.task_id,
        "public_task_payload_sha256": task.public.task_payload_sha256,
        "answer_commitment_sha256": task.public.answer_commitment_sha256,
        "response_artifact": response_binding,
        "score_output": primary_output,
    }
    if primary_score != expected_primary_score:
        _fail("primary_score_mismatch")

    witness = store.read_json(
        cast(Mapping[str, Any], receipt["independent_witness_artifact"]),
        role="independent_witness",
    )
    if set(witness) != _WITNESS_SCORE_KEYS:
        _fail("independent_witness_schema_invalid")
    if witness.get("schema") != WITNESS_SCORE_SCHEMA:
        _fail("independent_witness_version_invalid")
    _validate_digest(witness, role="independent_witness")
    _validate_callable_commitment(
        witness.get("scorer_callable"),
        independent_scorer,
        role="independent_witness_callable",
    )
    independent_output = _validate_independent_output(independent_scorer(task, response))
    witness_payload = build_frontier_witness_payload(
        store,
        task=task,
        response_artifact=response_binding,
        expected_authority=expected,
        independent_scorer=independent_scorer,
        issued_at_unix_ns=_require_signed_second(
            witness.get("issued_at_unix_ns"),
            role="independent_witness_issued_at",
        ),
    )
    verified_attestation = _verify_role_attestation(
        policy,
        witness.get("evidence_verifier_attestation"),
        role=EVIDENCE_VERIFIER,
        expected_payload=witness_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    expected_witness_fields = {
        "task_id": task.task_id,
        "public_task_payload_sha256": task.public.task_payload_sha256,
        "answer_commitment_sha256": task.public.answer_commitment_sha256,
        "response_artifact": response_binding,
        "witness_identity_sha256": expected["independent_witness_identity_sha256"],
        "score_output": independent_output,
        "trust_policy_sha256": policy.policy_sha256,
        "evidence_verifier_attestation": witness["evidence_verifier_attestation"],
    }
    for field, value in expected_witness_fields.items():
        if witness.get(field) != value:
            _fail(f"independent_witness_{field}_mismatch")
    if _semantic_score(primary_output) != _semantic_score(independent_output):
        _fail("independent_scorer_disagreement")
    issued = _require_int(
        receipt.get("issued_at_unix_ns"),
        role="verifier_authority_issued_at",
        minimum=1,
    )
    sealed = _require_int(
        receipt.get("sealed_at_unix_ns"),
        role="verifier_authority_sealed_at",
        minimum=issued,
    )
    witness_issued = _require_signed_second(
        witness.get("issued_at_unix_ns"),
        role="independent_witness_issued_at",
    )
    attested_at_ns = cast(int, verified_attestation["signed_at_unix"]) * 1_000_000_000
    if witness_issued != issued or witness_issued != attested_at_ns or sealed < witness_issued:
        _fail("verifier_authority_timestamp_mismatch")
    return dict(receipt)


def _validate_budget(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _BUDGET_KEYS:
        _fail("generation_budget_schema_invalid")
    budget = {
        "max_output_tokens": _require_int(
            value.get("max_output_tokens"),
            role="generation_budget_tokens",
            minimum=1,
            maximum=MAX_OUTPUT_TOKENS,
        ),
        "max_wall_time_ms": _require_int(
            value.get("max_wall_time_ms"),
            role="generation_budget_wall_time",
            minimum=1,
        ),
        "max_compute_units": _require_int(
            value.get("max_compute_units"),
            role="generation_budget_compute",
            minimum=1,
        ),
    }
    return budget


def _validate_token_trace(
    *,
    model_input_bytes: bytes,
    response_bytes: bytes,
    input_token_ids: Any,
    output_token_ids: Any,
    emitted_token_pieces: Any,
    behavior_policy_logprobs: Any,
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    budget: Mapping[str, int],
) -> tuple[list[int], list[int], list[str], list[str]]:
    if (
        not isinstance(input_token_ids, list)
        or not input_token_ids
        or len(input_token_ids) > MAX_OUTPUT_TOKENS
    ):
        _fail("input_token_ids_invalid")
    input_tokens = [
        _require_int(
            token,
            role="input_token_id",
            maximum=MAX_TOKEN_ID,
        )
        for token in input_token_ids
    ]
    if (
        not isinstance(output_token_ids, list)
        or not output_token_ids
        or len(output_token_ids) > budget["max_output_tokens"]
    ):
        _fail("output_token_ids_invalid")
    tokens = [
        _require_int(
            token,
            role="output_token_id",
            maximum=MAX_TOKEN_ID,
        )
        for token in output_token_ids
    ]
    if not isinstance(behavior_policy_logprobs, list) or len(behavior_policy_logprobs) != len(
        tokens
    ):
        _fail("behavior_policy_logprobs_invalid")
    logprobs = [
        _require_decimal(value, role="behavior_policy_logprob")
        for value in behavior_policy_logprobs
    ]
    try:
        if any(Decimal(value) > 0 for value in logprobs):
            _fail("behavior_policy_logprob_positive")
    except InvalidOperation as exc:
        raise VerifiedTransitionError("behavior_policy_logprob_invalid") from exc
    if (
        not isinstance(emitted_token_pieces, list)
        or len(emitted_token_pieces) != len(tokens)
        or any(not isinstance(piece, bytes) or not piece for piece in emitted_token_pieces)
    ):
        _fail("emitted_token_pieces_invalid")
    if b"".join(emitted_token_pieces) != response_bytes:
        _fail("emitted_token_pieces_response_mismatch")
    try:
        independently_encoded_input = list(token_encoder(model_input_bytes))
        independently_encoded_output = list(token_encoder(response_bytes))
        independently_decoded_input = token_decoder(input_tokens)
        independently_decoded_output = token_decoder(tokens)
    except Exception as exc:  # noqa: BLE001 - tokenizer trust boundary
        raise VerifiedTransitionError("token_encoder_execution_failed") from exc
    if any(
        type(token) is not int
        for token in (*independently_encoded_input, *independently_encoded_output)
    ):
        _fail("token_encoder_output_invalid")
    if (
        independently_encoded_input != input_tokens
        or independently_decoded_input != model_input_bytes
    ):
        _fail("model_input_token_mismatch")
    if independently_encoded_output != tokens or independently_decoded_output != response_bytes:
        _fail("response_token_mismatch")
    return (
        input_tokens,
        tokens,
        logprobs,
        [base64.b64encode(piece).decode("ascii") for piece in emitted_token_pieces],
    )


def _validate_non_candidate_evidence(
    store: TransitionArtifactStore,
    *,
    context: Mapping[str, Any],
    execution_manifest_sha256: str,
    generation_worker_identity_sha256: str,
    execution_component_roots: Mapping[str, str],
    generated_at_unix_ns: int,
) -> None:
    empty_artifacts = {
        "tool_snapshot_artifact": "aura.verified_transition.tool_snapshot.v1",
        "evidence_snapshot_artifact": ("aura.verified_transition.evidence_snapshot.v1"),
        "world_state_snapshot_artifact": ("aura.verified_transition.world_state_snapshot.v1"),
    }
    for field, schema in empty_artifacts.items():
        document = store.read_json(
            _validate_artifact_binding(context.get(field)),
            role=field.removesuffix("_artifact"),
        )
        if document != {
            "schema": schema,
            "candidate_visible": False,
            "items": [],
        }:
            _fail(f"{field}_candidate_isolation_failed")

    execution_spec = store.read_json(
        _validate_artifact_binding(context.get("execution_spec_artifact")),
        role="execution_spec",
    )
    execution_spec_schema = execution_spec.get("schema")
    expected_execution_spec_keys = {
        "schema",
        "candidate_visible",
        "execution_manifest_sha256",
        "sampling_policy_sha256",
    }
    if execution_spec_schema == "aura.verified_transition.execution_spec.v2":
        expected_execution_spec_keys.add("recurrent_execution_spec_sha256")
    if (
        set(execution_spec) != expected_execution_spec_keys
        or execution_spec_schema
        not in {
            "aura.verified_transition.execution_spec.v1",
            "aura.verified_transition.execution_spec.v2",
        }
        or execution_spec.get("candidate_visible") is not False
        or execution_spec.get("execution_manifest_sha256") != execution_manifest_sha256
    ):
        _fail("execution_spec_schema_invalid")
    _require_sha256(
        execution_spec.get("sampling_policy_sha256"),
        role="execution_sampling_policy",
    )
    if execution_spec_schema == "aura.verified_transition.execution_spec.v2":
        _require_sha256(
            execution_spec.get("recurrent_execution_spec_sha256"),
            role="recurrent_execution_spec",
        )

    latent_path = store.read_json(
        _validate_artifact_binding(context.get("latent_path_artifact")),
        role="latent_path",
    )
    if (
        set(latent_path)
        != {
            "schema",
            "candidate_visible",
            "mechanism_id",
            "configuration_sha256",
            "recurrence_steps",
            "branch_count",
        }
        or latent_path.get("schema") != "aura.verified_transition.latent_path.v1"
        or latent_path.get("candidate_visible") is not False
    ):
        _fail("latent_path_schema_invalid")
    _require_identifier(latent_path.get("mechanism_id"), role="latent_mechanism")
    _require_sha256(
        latent_path.get("configuration_sha256"),
        role="latent_configuration",
    )
    _require_int(
        latent_path.get("recurrence_steps"),
        role="latent_recurrence_steps",
        minimum=1,
        maximum=1_024,
    )
    _require_int(
        latent_path.get("branch_count"),
        role="latent_branch_count",
        minimum=1,
        maximum=256,
    )

    process = store.read_json(
        _validate_artifact_binding(context.get("process_receipt_artifact")),
        role="process_receipt",
    )
    if (
        set(process)
        != {
            "schema",
            "candidate_visible",
            "generation_worker_identity_sha256",
            "execution_manifest_sha256",
            "worker_pid",
            "worker_start_time_unix_ns",
            "worker_executable_sha256",
            "executable_component_root_sha256",
            "loaded_component_roots",
            "observer_contract",
            "started_at_unix_ns",
            "finished_at_unix_ns",
            "exit_code",
        }
        or process.get("schema") != "aura.verified_transition.process_receipt.v1"
        or process.get("candidate_visible") is not False
        or process.get("generation_worker_identity_sha256") != generation_worker_identity_sha256
        or process.get("execution_manifest_sha256") != execution_manifest_sha256
        or process.get("executable_component_root_sha256") != generation_worker_identity_sha256
        or process.get("observer_contract") != "external_process_monitor_required"
        or process.get("exit_code") != 0
    ):
        _fail("process_receipt_schema_invalid")
    started_at = _require_int(
        process.get("started_at_unix_ns"),
        role="process_started_at",
        minimum=1,
    )
    worker_start = _require_int(
        process.get("worker_start_time_unix_ns"),
        role="process_worker_start_time",
        minimum=1,
    )
    if worker_start > started_at:
        _fail("process_worker_started_after_generation")
    _require_int(
        process.get("worker_pid"),
        role="process_worker_pid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    _require_sha256(
        process.get("worker_executable_sha256"),
        role="process_worker_executable",
    )
    loaded_roots = process.get("loaded_component_roots")
    if (
        not isinstance(loaded_roots, Mapping)
        or set(loaded_roots) != _EXECUTION_COMPONENT_ROLES
        or dict(loaded_roots) != dict(execution_component_roots)
        or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in loaded_roots.values()
        )
    ):
        _fail("process_loaded_component_roots_invalid")
    finished_at = _require_int(
        process.get("finished_at_unix_ns"),
        role="process_finished_at",
        minimum=started_at,
    )
    if finished_at != generated_at_unix_ns:
        _fail("process_receipt_generation_time_mismatch")

    for field, schema in {
        "uncertainty_receipt_artifact": ("aura.verified_transition.uncertainty_receipt.v1"),
        "diversity_receipt_artifact": ("aura.verified_transition.diversity_receipt.v1"),
        "resource_receipt_artifact": ("aura.verified_transition.resource_receipt.v1"),
    }.items():
        document = store.read_json(
            _validate_artifact_binding(context.get(field)),
            role=field.removesuffix("_artifact"),
        )
        if (
            set(document) != {"schema", "candidate_visible", "measurement_micros"}
            or document.get("schema") != schema
            or document.get("candidate_visible") is not False
        ):
            _fail(f"{field}_schema_invalid")
        _require_int(
            document.get("measurement_micros"),
            role=f"{field}_measurement",
            maximum=1_000_000,
        )


def _validated_execution_identity(
    trust_context: TransitionTrustContext,
    *,
    context: Mapping[str, Any],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    policy: VerifiedCampaignTrustPolicy,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _validate_execution_manifest(
        trust_context.execution_manifest,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        independent_scorer=independent_scorer,
        component_roots=trust_context.execution_component_roots,
    )
    if (
        manifest["generation_worker_identity_sha256"]
        != policy.role_pin(CAMPAIGN_RUNNER)["implementation_sha256"]
    ):
        _fail("generation_worker_identity_not_policy_pinned")
    roots = cast(dict[str, str], manifest["component_roots"])
    expected_fields = {
        "model_identity_sha256": manifest["model_identity_sha256"],
        "base_checkpoint_sha256": roots["base_checkpoint"],
        "adapter_stack_sha256": roots["adapter_stack"],
        "tokenizer_sha256": roots["tokenizer"],
        "policy_sha256": roots["policy"],
        "personality_sha256": roots["personality"],
        "runtime_sha256": roots["runtime"],
        "source_closure_sha256": roots["source_closure"],
    }
    for field, expected in expected_fields.items():
        if context.get(field) != expected:
            _fail(f"{field}_execution_manifest_mismatch")
    return manifest, roots


def build_generation_trace_payload(
    store: TransitionArtifactStore,
    *,
    pass_index: int,
    task: FrontierTask,
    model_input_bytes: bytes,
    response_bytes: bytes,
    input_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    emitted_token_pieces: Sequence[bytes],
    behavior_policy_logprobs: Sequence[str],
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    trust_context: TransitionTrustContext,
    context: Mapping[str, Any],
    trace_signed_at_unix_ns: int,
) -> dict[str, Any]:
    """Build the exact generation evidence the external runner signs."""

    if type(pass_index) is not int or pass_index not in {0, 1}:
        _fail("pass_index_invalid")
    task = _require_exact_frontier_task(task)
    if not isinstance(context, Mapping) or set(context) != _PASS_CONTEXT_KEYS:
        _fail("reasoning_pass_context_schema_invalid")
    if model_input_bytes != canonical_candidate_model_input(task):
        _fail("candidate_model_input_not_canonical")
    if (
        not isinstance(response_bytes, bytes)
        or not response_bytes
        or len(response_bytes) > MAX_TEXT_BYTES
    ):
        _fail("response_payload_invalid")
    policy = _verified_transition_policy(
        trust_context,
        _validate_expected_authority(expected_authority),
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    manifest, _roots = _validated_execution_identity(
        trust_context,
        context=context,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        independent_scorer=independent_scorer,
        policy=policy,
    )
    generated_at = _require_int(
        context.get("generated_at_unix_ns"),
        role="pass_generated_at",
        minimum=1,
    )
    trace_signed_at = _require_signed_second(
        trace_signed_at_unix_ns,
        role="generation_trace_signed_at",
    )
    if trace_signed_at <= generated_at:
        _fail("generation_trace_not_after_generation")
    execution_manifest_sha256 = _sha256_bytes(
        canonical_json_bytes(dict(trust_context.execution_manifest))
    )
    _validate_non_candidate_evidence(
        store,
        context=context,
        execution_manifest_sha256=execution_manifest_sha256,
        generation_worker_identity_sha256=manifest["generation_worker_identity_sha256"],
        execution_component_roots=cast(
            Mapping[str, str],
            manifest["component_roots"],
        ),
        generated_at_unix_ns=generated_at,
    )
    planned_context_sha256 = planned_transition_immutable_context_sha256(
        store,
        task=task,
        context=context,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
    )
    ledger_open_attestation = _verify_attempt_ledger_open_attestation(
        policy=policy,
        trust_context=trust_context,
        episode_id=cast(str, context.get("episode_id")),
        immutable_context_sha256=planned_context_sha256,
        not_after_unix=generated_at // 1_000_000_000,
    )
    open_payload = ledger_open_attestation["signed_payload"]["payload"]
    task_issuer_attestation = _verify_transition_task_issuer_attestation(
        policy=policy,
        trust_context=trust_context,
        task=task,
        not_after_unix=generated_at // 1_000_000_000,
    )
    issuer_at = task_issuer_attestation["signed_at_unix"] * 1_000_000_000
    if not open_payload["opened_at_unix_ns"] < issuer_at < generated_at:
        _fail("attempt_ledger_not_open_before_task_disclosure")
    model_input_artifact = store.put_bytes(
        model_input_bytes,
        media_type="application/octet-stream",
    )
    response_artifact = store.put_bytes(
        response_bytes,
        media_type="text/plain;charset=utf-8",
    )
    budget = _validate_budget(context.get("generation_budget"))
    input_tokens, output_tokens, logprobs, encoded_pieces = _validate_token_trace(
        model_input_bytes=model_input_bytes,
        response_bytes=response_bytes,
        input_token_ids=list(input_token_ids),
        output_token_ids=list(output_token_ids),
        emitted_token_pieces=list(emitted_token_pieces),
        behavior_policy_logprobs=list(behavior_policy_logprobs),
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        budget=budget,
    )
    trace = {
        "schema": GENERATION_TRACE_PAYLOAD_SCHEMA,
        "episode_id": _require_identifier(
            context.get("episode_id"),
            role="episode_id",
        ),
        "pass_index": pass_index,
        "task_id": task.task_id,
        "candidate_input_sha256": model_input_artifact["payload_sha256"],
        "response_sha256": response_artifact["payload_sha256"],
        "input_token_ids_sha256": _sha256_bytes(canonical_json_bytes(input_tokens)),
        "output_token_ids_sha256": _sha256_bytes(canonical_json_bytes(output_tokens)),
        "emitted_token_pieces_sha256": _sha256_bytes(canonical_json_bytes(encoded_pieces)),
        "behavior_policy_logprobs_sha256": _sha256_bytes(canonical_json_bytes(logprobs)),
        "execution_manifest_sha256": execution_manifest_sha256,
        "generation_worker_identity_sha256": manifest["generation_worker_identity_sha256"],
        "attempt_ledger_identity_sha256": _require_sha256(
            trust_context.expected_attempt_ledger_identity_sha256,
            role="attempt_ledger_identity",
        ),
        "planned_immutable_context_sha256": planned_context_sha256,
        "attempt_ledger_open_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(ledger_open_attestation)
        ),
        "task_issuer_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(task_issuer_attestation)
        ),
        "rng_root_sha256": _require_sha256(
            context.get("rng_root_sha256"),
            role="rng_root",
        ),
        "generation_budget_sha256": _sha256_bytes(canonical_json_bytes(budget)),
        "evidence_artifacts_sha256": _sha256_bytes(
            canonical_json_bytes(
                {
                    field: context[field]
                    for field in (
                        "execution_spec_artifact",
                        "latent_path_artifact",
                        "tool_snapshot_artifact",
                        "evidence_snapshot_artifact",
                        "world_state_snapshot_artifact",
                        "process_receipt_artifact",
                        "uncertainty_receipt_artifact",
                        "diversity_receipt_artifact",
                        "resource_receipt_artifact",
                    )
                }
            )
        ),
        "generated_at_unix_ns": generated_at,
        "trace_signed_at_unix_ns": trace_signed_at,
    }
    return trace


def capture_execution_process_observation(
    store: TransitionArtifactStore,
    *,
    context: Mapping[str, Any],
    execution_component_roots: Mapping[str, Path],
) -> dict[str, Any]:
    """Collect host process and open-component evidence inside the observer."""

    if not isinstance(store, TransitionArtifactStore):
        _fail("artifact_store_invalid")
    if not isinstance(context, Mapping) or set(context) != _PASS_CONTEXT_KEYS:
        _fail("execution_observer_context_invalid")
    process_receipt = _validate_artifact_binding(context.get("process_receipt_artifact"))
    process_document = store.read_json(
        process_receipt,
        role="execution_observer_process_receipt",
    )
    pid = _require_int(
        process_document.get("worker_pid"),
        role="process_worker_pid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    expected_start = _require_int(
        process_document.get("worker_start_time_unix_ns"),
        role="process_worker_start_time",
        minimum=1,
    )
    observer = HostResourceObserver(
        source=ObservationSource.HOST,
        scenario_id="verified_transition_execution_observer",
    )
    process = observer.process(pid)
    if process is None or process.provenance.source is not ObservationSource.HOST:
        _fail("execution_process_observation_unavailable")
    observed_start = int(round(process.create_time * 1_000_000_000))
    if abs(observed_start - expected_start) > 1_000_000:
        _fail("execution_process_start_identity_mismatch")
    executable = Path(process.exe).resolve(strict=True)
    executable_sha256, _executable_size = _hash_regular_file(
        executable,
        role="execution_process_executable",
    )
    if executable_sha256 != _require_sha256(
        process_document.get("worker_executable_sha256"),
        role="process_worker_executable",
    ):
        _fail("execution_process_executable_mismatch")

    open_table = observer.open_file_table(pid=pid)
    if open_table.provenance.source is not ObservationSource.HOST or not open_table.available:
        _fail("execution_process_open_files_unavailable")
    open_paths: set[Path] = set()
    for raw_path in open_table.paths:
        try:
            open_paths.add(Path(raw_path).resolve(strict=True))
        except (OSError, RuntimeError):
            continue
    descriptor_identities: dict[Path, list[OpenFileIdentityObservation]] = {}
    for identity in open_table.identities:
        try:
            observed_path = Path(identity.path).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        descriptor_identities.setdefault(observed_path, []).append(identity)

    loaded_roots = process_document.get("loaded_component_roots")
    if (
        not isinstance(loaded_roots, Mapping)
        or set(loaded_roots) != _EXECUTION_COMPONENT_ROLES
        or set(execution_component_roots) != _EXECUTION_COMPONENT_ROLES
    ):
        _fail("execution_observer_loaded_roots_missing")
    observed_roots: dict[str, str] = {}
    observed_files: list[str] = []
    observed_descriptor_identities: list[dict[str, Any]] = []
    for role in sorted(_EXECUTION_COMPONENT_ROLES):
        root = Path(execution_component_roots[role]).resolve(strict=True)
        manifest = _component_manifest_from_root(role, root)
        if manifest["root_sha256"] != loaded_roots.get(role):
            _fail("execution_observer_component_root_mismatch")
        for entry in manifest["entries"]:
            component_path = (root / entry["path"]).resolve(strict=True)
            if component_path not in open_paths:
                _fail("execution_observer_component_file_not_open")
            metadata = component_path.stat(follow_symlinks=False)
            matching_identities = [
                identity
                for identity in descriptor_identities.get(component_path, ())
                if (
                    identity.device == metadata.st_dev
                    and identity.inode == metadata.st_ino
                    and identity.byte_length == entry["byte_length"]
                    and identity.byte_length == metadata.st_size
                    and identity.mtime_ns == metadata.st_mtime_ns
                    and stat.S_ISREG(identity.mode)
                    and identity.provider in {"proc_pidfdinfo", "procfs_fd", "self_fstat"}
                )
            ]
            if not matching_identities:
                _fail("execution_observer_component_descriptor_identity_mismatch")
            descriptor = min(matching_identities, key=lambda item: item.fd)
            observed_files.append(f"{role}:{entry['path']}")
            observed_descriptor_identities.append(
                {
                    "role": role,
                    "path": entry["path"],
                    "fd": descriptor.fd,
                    "device": descriptor.device,
                    "inode": descriptor.inode,
                    "byte_length": descriptor.byte_length,
                    "mtime_ns": descriptor.mtime_ns,
                    "provider": descriptor.provider,
                }
            )
        observed_roots[role] = manifest["root_sha256"]

    captured_at = time.time_ns()
    return store.put_json(
        {
            "schema": EXECUTION_PROCESS_OBSERVATION_SCHEMA,
            "candidate_visible": False,
            "observer_backend": "HostResourceObserver",
            "observation_source": "host",
            "pid": pid,
            "ppid": process.ppid,
            "worker_start_time_unix_ns": observed_start,
            "worker_executable_sha256": executable_sha256,
            "cmdline_sha256": _sha256_bytes(canonical_json_bytes(list(process.cmdline))),
            "cwd_sha256": _sha256_bytes(process.cwd.encode("utf-8")),
            "open_file_count": len(open_paths),
            "open_file_identity_count": len(open_table.identities),
            "observed_component_roots": observed_roots,
            "observed_component_files_sha256": _sha256_bytes(canonical_json_bytes(observed_files)),
            "observed_component_file_count": len(observed_files),
            "observed_component_descriptor_identities_sha256": _sha256_bytes(
                canonical_json_bytes(observed_descriptor_identities)
            ),
            "captured_at_unix_ns": captured_at,
        }
    )


def _validate_execution_process_observation(
    store: TransitionArtifactStore,
    *,
    artifact: Mapping[str, Any],
    process_document: Mapping[str, Any],
    observed_at_unix_ns: int,
) -> dict[str, Any]:
    binding = _validate_artifact_binding(artifact)
    document = store.read_json(binding, role="execution_process_observation")
    if (
        set(document)
        != {
            "schema",
            "candidate_visible",
            "observer_backend",
            "observation_source",
            "pid",
            "ppid",
            "worker_start_time_unix_ns",
            "worker_executable_sha256",
            "cmdline_sha256",
            "cwd_sha256",
            "open_file_count",
            "open_file_identity_count",
            "observed_component_roots",
            "observed_component_files_sha256",
            "observed_component_file_count",
            "observed_component_descriptor_identities_sha256",
            "captured_at_unix_ns",
        }
        or document.get("schema") != EXECUTION_PROCESS_OBSERVATION_SCHEMA
        or document.get("candidate_visible") is not False
        or document.get("observer_backend") != "HostResourceObserver"
        or document.get("observation_source") != "host"
        or document.get("pid") != process_document.get("worker_pid")
        or document.get("worker_start_time_unix_ns")
        != process_document.get("worker_start_time_unix_ns")
        or document.get("worker_executable_sha256")
        != process_document.get("worker_executable_sha256")
        or document.get("observed_component_roots")
        != process_document.get("loaded_component_roots")
        or document.get("captured_at_unix_ns") != observed_at_unix_ns
    ):
        _fail("execution_process_observation_mismatch")
    for role in (
        "cmdline_sha256",
        "cwd_sha256",
        "observed_component_files_sha256",
        "observed_component_descriptor_identities_sha256",
    ):
        _require_sha256(document.get(role), role=f"execution_observation_{role}")
    _require_int(document.get("ppid"), role="execution_observation_ppid", minimum=0)
    _require_int(
        document.get("open_file_count"),
        role="execution_observation_open_file_count",
        minimum=1,
    )
    _require_int(
        document.get("open_file_identity_count"),
        role="execution_observation_open_file_identity_count",
        minimum=len(_EXECUTION_COMPONENT_ROLES),
    )
    _require_int(
        document.get("observed_component_file_count"),
        role="execution_observation_component_file_count",
        minimum=len(_EXECUTION_COMPONENT_ROLES),
    )
    return binding


def build_execution_observer_payload(
    store: TransitionArtifactStore,
    *,
    generation_trace_payload: Mapping[str, Any],
    generation_worker_attestation: Mapping[str, Any],
    context: Mapping[str, Any],
    process_observation_artifact: Mapping[str, Any],
    observed_at_unix_ns: int,
) -> dict[str, Any]:
    """Build the claim an independent process observer must attest."""

    if (
        not isinstance(generation_trace_payload, Mapping)
        or generation_trace_payload.get("schema") != GENERATION_TRACE_PAYLOAD_SCHEMA
    ):
        _fail("execution_observer_generation_trace_invalid")
    if not isinstance(context, Mapping) or set(context) != _PASS_CONTEXT_KEYS:
        _fail("execution_observer_context_invalid")
    observed_at = _require_signed_second(
        observed_at_unix_ns,
        role="execution_observer_time",
    )
    trace_signed_at = _require_signed_second(
        generation_trace_payload.get("trace_signed_at_unix_ns"),
        role="generation_trace_signed_at",
    )
    if observed_at <= trace_signed_at:
        _fail("execution_observer_not_after_generation_trace")
    process_receipt = _validate_artifact_binding(context.get("process_receipt_artifact"))
    process_document = store.read_json(
        process_receipt,
        role="execution_observer_process_receipt",
    )
    loaded_roots = process_document.get("loaded_component_roots")
    if not isinstance(loaded_roots, Mapping):
        _fail("execution_observer_loaded_roots_missing")
    observation_binding = _validate_execution_process_observation(
        store,
        artifact=process_observation_artifact,
        process_document=process_document,
        observed_at_unix_ns=observed_at,
    )
    return {
        "schema": EXECUTION_OBSERVER_PAYLOAD_SCHEMA,
        "observer_class": "external_process_monitor",
        "episode_id": _require_identifier(
            context.get("episode_id"),
            role="episode_id",
        ),
        "pass_index": _require_int(
            generation_trace_payload.get("pass_index"),
            role="pass_index",
            maximum=1,
        ),
        "generation_trace_sha256": _sha256_bytes(
            canonical_json_bytes(dict(generation_trace_payload))
        ),
        "generation_worker_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(dict(generation_worker_attestation))
        ),
        "process_receipt_artifact": process_receipt,
        "process_observation_artifact": observation_binding,
        "loaded_component_roots_sha256": _sha256_bytes(canonical_json_bytes(loaded_roots)),
        "execution_manifest_sha256": _require_sha256(
            generation_trace_payload.get("execution_manifest_sha256"),
            role="execution_manifest",
        ),
        "model_identity_sha256": _require_sha256(
            context.get("model_identity_sha256"),
            role="model_identity",
        ),
        "response_sha256": _require_sha256(
            generation_trace_payload.get("response_sha256"),
            role="response",
        ),
        "observed_at_unix_ns": observed_at,
    }


def execution_observer_implementation_identity() -> str:
    """Pin the host collector and observer-payload implementation."""

    closure = {
        "capture": callable_commitment(capture_execution_process_observation),
        "payload": callable_commitment(build_execution_observer_payload),
        "host_process": callable_commitment(HostResourceObserver.process),
        "host_open_files": callable_commitment(HostResourceObserver.open_file_table),
    }
    return _sha256_bytes(canonical_json_bytes(closure))


def build_reasoning_pass_receipt(
    store: TransitionArtifactStore,
    *,
    pass_index: int,
    task: FrontierTask,
    model_input_bytes: bytes,
    response_bytes: bytes,
    input_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    emitted_token_pieces: Sequence[bytes],
    behavior_policy_logprobs: Sequence[str],
    verifier_authority: Mapping[str, Any],
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    trust_context: TransitionTrustContext,
    context: Mapping[str, Any],
    generation_worker_attestation: Mapping[str, Any],
    execution_process_observation_artifact: Mapping[str, Any],
    execution_observer_attestation: Mapping[str, Any],
    trace_signed_at_unix_ns: int,
) -> dict[str, Any]:
    """Build one pass only after payload, tokenizer, and verifier replay."""

    if type(pass_index) is not int or pass_index not in {0, 1}:
        _fail("pass_index_invalid")
    if not isinstance(context, Mapping) or set(context) != _PASS_CONTEXT_KEYS:
        _fail("reasoning_pass_context_schema_invalid")
    if (
        not isinstance(response_bytes, bytes)
        or not response_bytes
        or len(response_bytes) > MAX_TEXT_BYTES
    ):
        _fail("response_payload_invalid")
    try:
        response_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("response_not_utf8")

    prompt_artifact = store.put_bytes(
        task.public.prompt.encode("utf-8"),
        media_type="text/plain;charset=utf-8",
    )
    expected_model_input = canonical_candidate_model_input(task)
    if model_input_bytes != expected_model_input:
        _fail("candidate_model_input_not_canonical")
    model_input_artifact = store.put_bytes(
        model_input_bytes,
        media_type="application/octet-stream",
    )
    response_artifact = store.put_bytes(
        response_bytes,
        media_type="text/plain;charset=utf-8",
    )
    validate_frontier_verifier_authority(
        store,
        verifier_authority,
        task=task,
        response_artifact=response_artifact,
        expected_authority=expected_authority,
        independent_scorer=independent_scorer,
        trust_context=trust_context,
    )
    authority_artifact = store.put_json(verifier_authority)
    generation_trace_payload = build_generation_trace_payload(
        store,
        pass_index=pass_index,
        task=task,
        model_input_bytes=model_input_bytes,
        response_bytes=response_bytes,
        input_token_ids=input_token_ids,
        output_token_ids=output_token_ids,
        emitted_token_pieces=emitted_token_pieces,
        behavior_policy_logprobs=behavior_policy_logprobs,
        expected_authority=expected_authority,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
        context=context,
        trace_signed_at_unix_ns=trace_signed_at_unix_ns,
    )
    policy = _verified_transition_policy(
        trust_context,
        _validate_expected_authority(expected_authority),
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    verified_generation_attestation = _verify_role_attestation(
        policy,
        generation_worker_attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=generation_trace_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    if generation_trace_payload["trace_signed_at_unix_ns"] != (
        verified_generation_attestation["signed_at_unix"] * 1_000_000_000
    ):
        _fail("generation_trace_timestamp_mismatch")
    generation_attestation_artifact = store.put_json(dict(generation_worker_attestation))
    observer_signed_document = (
        execution_observer_attestation.get("signed_payload")
        if isinstance(execution_observer_attestation, Mapping)
        else None
    )
    observer_submitted_payload = (
        observer_signed_document.get("payload")
        if isinstance(observer_signed_document, Mapping)
        else None
    )
    observer_at = _require_signed_second(
        observer_submitted_payload.get("observed_at_unix_ns")
        if isinstance(observer_submitted_payload, Mapping)
        else None,
        role="execution_observer_time",
    )
    observer_payload = build_execution_observer_payload(
        store,
        generation_trace_payload=generation_trace_payload,
        generation_worker_attestation=generation_worker_attestation,
        context=context,
        process_observation_artifact=execution_process_observation_artifact,
        observed_at_unix_ns=observer_at,
    )
    verified_observer = _verify_role_attestation(
        policy,
        execution_observer_attestation,
        role=CONTAMINATION_AUDITOR,
        expected_payload=observer_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    if observer_at != verified_observer["signed_at_unix"] * 1_000_000_000:
        _fail("execution_observer_timestamp_mismatch")
    observer_attestation_artifact = store.put_json(dict(execution_observer_attestation))
    budget = _validate_budget(context.get("generation_budget"))
    input_tokens, tokens, logprobs, encoded_pieces = _validate_token_trace(
        model_input_bytes=model_input_bytes,
        response_bytes=response_bytes,
        input_token_ids=list(input_token_ids),
        output_token_ids=list(output_token_ids),
        emitted_token_pieces=list(emitted_token_pieces),
        behavior_policy_logprobs=list(behavior_policy_logprobs),
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        budget=budget,
    )
    pieces_artifact = store.put_json(
        {
            "schema": "aura.verified_transition.emitted_token_pieces.v1",
            "encoding": "base64",
            "pieces": encoded_pieces,
            "response_sha256": _sha256_bytes(response_bytes),
        }
    )
    depth = _require_int(context.get("depth"), role="pass_depth", minimum=1)
    if depth < task.public.difficulty:
        _fail("pass_depth_below_task_difficulty")
    generated = _require_int(
        context.get("generated_at_unix_ns"),
        role="pass_generated_at",
        minimum=1,
    )
    sealed = _require_int(
        context.get("sealed_at_unix_ns"),
        role="pass_sealed_at",
        minimum=generated,
    )
    deadline = _require_int(
        context.get("deadline_unix_ns"),
        role="pass_deadline",
        minimum=generated,
    )
    if sealed > deadline:
        _fail("pass_sealed_after_deadline")
    authority_issued = _require_int(
        verifier_authority.get("issued_at_unix_ns"),
        role="pass_authority_issued_at",
        minimum=generated,
    )
    authority_sealed = _require_int(
        verifier_authority.get("sealed_at_unix_ns"),
        role="pass_authority_sealed_at",
        minimum=authority_issued,
    )
    if authority_sealed > sealed:
        _fail("pass_authority_sealed_after_pass")
    if generation_trace_payload["trace_signed_at_unix_ns"] >= authority_issued:
        _fail("generation_trace_not_pre_verification")
    if observer_at >= authority_issued:
        _fail("execution_observer_not_pre_verification")

    manifest, _roots = _validated_execution_identity(
        trust_context,
        context=context,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        independent_scorer=independent_scorer,
        policy=policy,
    )
    execution_manifest_artifact = store.put_json(dict(trust_context.execution_manifest))
    _validate_non_candidate_evidence(
        store,
        context=context,
        execution_manifest_sha256=execution_manifest_artifact["payload_sha256"],
        generation_worker_identity_sha256=manifest["generation_worker_identity_sha256"],
        execution_component_roots=cast(
            Mapping[str, str],
            manifest["component_roots"],
        ),
        generated_at_unix_ns=generated,
    )
    expected_task_commitment = _sha256_bytes(
        canonical_json_bytes(build_frontier_task_issuer_payload(task))
    )
    if context.get("sealed_task_commitment_sha256") != expected_task_commitment:
        _fail("sealed_task_commitment_mismatch")

    receipt = _seal_document(
        {
            "schema": REASONING_PASS_SCHEMA,
            "episode_id": _require_identifier(
                context.get("episode_id"),
                role="episode_id",
            ),
            "pass_index": pass_index,
            "task_id": task.task_id,
            "case_id": _require_identifier(
                context.get("case_id"),
                role="case_id",
            ),
            "family": _require_identifier(
                context.get("family"),
                role="task_family",
            ),
            "domain": task.domain,
            "depth": depth,
            "difficulty": task.public.difficulty,
            "sealed_task_commitment_sha256": _require_sha256(
                context.get("sealed_task_commitment_sha256"),
                role="sealed_task_commitment",
            ),
            "prompt_artifact": prompt_artifact,
            "model_input_artifact": model_input_artifact,
            "response_artifact": response_artifact,
            "model_identity_sha256": _require_sha256(
                context.get("model_identity_sha256"),
                role="model_identity",
            ),
            "base_checkpoint_sha256": _require_sha256(
                context.get("base_checkpoint_sha256"),
                role="base_checkpoint",
            ),
            "adapter_stack_sha256": _require_sha256(
                context.get("adapter_stack_sha256"),
                role="adapter_stack",
            ),
            "tokenizer_sha256": _require_sha256(
                context.get("tokenizer_sha256"),
                role="tokenizer",
            ),
            "token_encoder_callable": callable_commitment(token_encoder),
            "token_decoder_callable": callable_commitment(token_decoder),
            "policy_sha256": _require_sha256(
                context.get("policy_sha256"),
                role="policy",
            ),
            "personality_sha256": _require_sha256(
                context.get("personality_sha256"),
                role="personality",
            ),
            "runtime_sha256": _require_sha256(
                context.get("runtime_sha256"),
                role="runtime",
            ),
            "source_closure_sha256": _require_sha256(
                context.get("source_closure_sha256"),
                role="source_closure",
            ),
            "attempt_ledger_identity_sha256": _require_sha256(
                trust_context.expected_attempt_ledger_identity_sha256,
                role="attempt_ledger_identity",
            ),
            "execution_manifest_artifact": execution_manifest_artifact,
            "execution_spec_artifact": dict(context["execution_spec_artifact"]),
            "latent_path_artifact": dict(context["latent_path_artifact"]),
            "tool_snapshot_artifact": dict(context["tool_snapshot_artifact"]),
            "evidence_snapshot_artifact": dict(context["evidence_snapshot_artifact"]),
            "world_state_snapshot_artifact": dict(context["world_state_snapshot_artifact"]),
            "rng_root_sha256": _require_sha256(
                context.get("rng_root_sha256"),
                role="rng_root",
            ),
            "generation_budget": budget,
            "deadline_unix_ns": deadline,
            "input_token_ids": input_tokens,
            "output_token_ids": tokens,
            "emitted_token_pieces_artifact": pieces_artifact,
            "behavior_policy_logprobs": logprobs,
            "generation_worker_attestation_artifact": (generation_attestation_artifact),
            "execution_process_observation_artifact": _validate_artifact_binding(
                execution_process_observation_artifact
            ),
            "execution_observer_attestation_artifact": (observer_attestation_artifact),
            "verifier_authority_artifact": authority_artifact,
            "process_receipt_artifact": dict(context["process_receipt_artifact"]),
            "uncertainty_receipt_artifact": dict(context["uncertainty_receipt_artifact"]),
            "diversity_receipt_artifact": dict(context["diversity_receipt_artifact"]),
            "resource_receipt_artifact": dict(context["resource_receipt_artifact"]),
            "generated_at_unix_ns": generated,
            "sealed_at_unix_ns": sealed,
        }
    )
    planned = planned_transition_immutable_context(
        store,
        task=task,
        context=context,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
    )
    if _immutable_pass_context(receipt) != planned:
        _fail("reasoning_pass_planned_context_mismatch")
    return receipt


def validate_reasoning_pass_receipt(
    store: TransitionArtifactStore,
    receipt: Mapping[str, Any],
    *,
    task: FrontierTask,
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    trust_context: TransitionTrustContext,
) -> dict[str, Any]:
    """Reopen and replay every pass payload and authority."""

    if not isinstance(receipt, Mapping) or set(receipt) != _PASS_KEYS:
        _fail("reasoning_pass_schema_invalid")
    if receipt.get("schema") != REASONING_PASS_SCHEMA:
        _fail("reasoning_pass_version_invalid")
    _validate_digest(receipt, role="reasoning_pass")
    pass_index = _require_int(
        receipt.get("pass_index"),
        role="pass_index",
        maximum=1,
    )
    if pass_index not in {0, 1}:
        _fail("pass_index_invalid")
    if (
        receipt.get("task_id") != task.task_id
        or receipt.get("domain") != task.domain
        or receipt.get("difficulty") != task.public.difficulty
    ):
        _fail("reasoning_pass_task_identity_mismatch")
    for role in (
        "episode_id",
        "case_id",
        "family",
    ):
        _require_identifier(receipt.get(role), role=role)
    _require_int(receipt.get("depth"), role="pass_depth", minimum=1)
    for role in (
        "sealed_task_commitment_sha256",
        "model_identity_sha256",
        "base_checkpoint_sha256",
        "adapter_stack_sha256",
        "tokenizer_sha256",
        "policy_sha256",
        "personality_sha256",
        "runtime_sha256",
        "source_closure_sha256",
        "attempt_ledger_identity_sha256",
        "rng_root_sha256",
    ):
        _require_sha256(receipt.get(role), role=role)
    _validate_callable_commitment(
        receipt.get("token_encoder_callable"),
        token_encoder,
        role="token_encoder_callable",
    )
    _validate_callable_commitment(
        receipt.get("token_decoder_callable"),
        token_decoder,
        role="token_decoder_callable",
    )
    prompt = store.read_bytes(
        cast(Mapping[str, Any], receipt["prompt_artifact"]),
        expected_media_type="text/plain;charset=utf-8",
    )
    if prompt != task.public.prompt.encode("utf-8"):
        _fail("reasoning_pass_prompt_mismatch")
    model_input = store.read_bytes(
        cast(Mapping[str, Any], receipt["model_input_artifact"]),
        expected_media_type="application/octet-stream",
    )
    if model_input != canonical_candidate_model_input(task):
        _fail("reasoning_pass_candidate_input_mismatch")
    response_binding = _validate_artifact_binding(receipt["response_artifact"])
    response = store.read_bytes(
        response_binding,
        expected_media_type="text/plain;charset=utf-8",
    )
    budget = _validate_budget(receipt.get("generation_budget"))
    pieces_document = store.read_json(
        cast(Mapping[str, Any], receipt["emitted_token_pieces_artifact"]),
        role="emitted_token_pieces",
    )
    if (
        set(pieces_document)
        != {
            "schema",
            "encoding",
            "pieces",
            "response_sha256",
        }
        or pieces_document.get("schema") != "aura.verified_transition.emitted_token_pieces.v1"
    ):
        _fail("emitted_token_pieces_schema_invalid")
    encoded_pieces = pieces_document.get("pieces")
    if (
        pieces_document.get("encoding") != "base64"
        or not isinstance(encoded_pieces, list)
        or any(not isinstance(piece, str) for piece in encoded_pieces)
        or pieces_document.get("response_sha256") != _sha256_bytes(response)
    ):
        _fail("emitted_token_pieces_identity_invalid")
    try:
        emitted_pieces = [base64.b64decode(piece, validate=True) for piece in encoded_pieces]
    except (ValueError, binascii.Error) as exc:
        raise VerifiedTransitionError("emitted_token_pieces_encoding_invalid") from exc
    _validate_token_trace(
        model_input_bytes=model_input,
        response_bytes=response,
        input_token_ids=receipt.get("input_token_ids"),
        output_token_ids=receipt.get("output_token_ids"),
        emitted_token_pieces=emitted_pieces,
        behavior_policy_logprobs=receipt.get("behavior_policy_logprobs"),
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        budget=budget,
    )
    expected = _validate_expected_authority(expected_authority)
    policy = _verified_transition_policy(
        trust_context,
        expected,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    manifest, _roots = _validated_execution_identity(
        trust_context,
        context=receipt,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        independent_scorer=independent_scorer,
        policy=policy,
    )
    manifest_artifact = store.read_json(
        cast(Mapping[str, Any], receipt["execution_manifest_artifact"]),
        role="execution_manifest",
    )
    if manifest_artifact != manifest:
        _fail("reasoning_pass_execution_manifest_mismatch")
    generated = _require_int(
        receipt.get("generated_at_unix_ns"),
        role="pass_generated_at",
        minimum=1,
    )
    _validate_non_candidate_evidence(
        store,
        context=receipt,
        execution_manifest_sha256=_sha256_bytes(canonical_json_bytes(manifest)),
        generation_worker_identity_sha256=manifest["generation_worker_identity_sha256"],
        execution_component_roots=cast(
            Mapping[str, str],
            manifest["component_roots"],
        ),
        generated_at_unix_ns=generated,
    )
    generation_attestation = store.read_json(
        cast(
            Mapping[str, Any],
            receipt["generation_worker_attestation_artifact"],
        ),
        role="generation_worker_attestation",
    )
    signed_payload = generation_attestation.get("signed_payload")
    raw_trace_payload = (
        signed_payload.get("payload") if isinstance(signed_payload, Mapping) else None
    )
    trace_signed_at = (
        raw_trace_payload.get("trace_signed_at_unix_ns")
        if isinstance(raw_trace_payload, Mapping)
        else None
    )
    trace_payload = build_generation_trace_payload(
        store,
        pass_index=pass_index,
        task=task,
        model_input_bytes=model_input,
        response_bytes=response,
        input_token_ids=cast(Sequence[int], receipt.get("input_token_ids")),
        output_token_ids=cast(Sequence[int], receipt.get("output_token_ids")),
        emitted_token_pieces=emitted_pieces,
        behavior_policy_logprobs=cast(
            Sequence[str],
            receipt.get("behavior_policy_logprobs"),
        ),
        expected_authority=expected,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
        context={field: receipt[field] for field in _PASS_CONTEXT_KEYS},
        trace_signed_at_unix_ns=_require_signed_second(
            trace_signed_at,
            role="generation_trace_signed_at",
        ),
    )
    verified_generation_attestation = _verify_role_attestation(
        policy,
        generation_attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=trace_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    if trace_payload["trace_signed_at_unix_ns"] != (
        verified_generation_attestation["signed_at_unix"] * 1_000_000_000
    ):
        _fail("generation_trace_timestamp_mismatch")
    observer_attestation = store.read_json(
        cast(
            Mapping[str, Any],
            receipt["execution_observer_attestation_artifact"],
        ),
        role="execution_observer_attestation",
    )
    observer_signed_document = observer_attestation.get("signed_payload")
    observer_submitted_payload = (
        observer_signed_document.get("payload")
        if isinstance(observer_signed_document, Mapping)
        else None
    )
    observer_at = _require_signed_second(
        observer_submitted_payload.get("observed_at_unix_ns")
        if isinstance(observer_submitted_payload, Mapping)
        else None,
        role="execution_observer_time",
    )
    observer_payload = build_execution_observer_payload(
        store,
        generation_trace_payload=trace_payload,
        generation_worker_attestation=generation_attestation,
        context={field: receipt[field] for field in _PASS_CONTEXT_KEYS},
        process_observation_artifact=cast(
            Mapping[str, Any],
            receipt["execution_process_observation_artifact"],
        ),
        observed_at_unix_ns=observer_at,
    )
    verified_observer = _verify_role_attestation(
        policy,
        observer_attestation,
        role=CONTAMINATION_AUDITOR,
        expected_payload=observer_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    if observer_at != verified_observer["signed_at_unix"] * 1_000_000_000:
        _fail("execution_observer_timestamp_mismatch")
    authority = store.read_json(
        cast(Mapping[str, Any], receipt["verifier_authority_artifact"]),
        role="verifier_authority",
    )
    validate_frontier_verifier_authority(
        store,
        authority,
        task=task,
        response_artifact=response_binding,
        expected_authority=expected_authority,
        independent_scorer=independent_scorer,
        trust_context=trust_context,
    )
    sealed = _require_int(
        receipt.get("sealed_at_unix_ns"),
        role="pass_sealed_at",
        minimum=generated,
    )
    deadline = _require_int(
        receipt.get("deadline_unix_ns"),
        role="pass_deadline",
        minimum=generated,
    )
    if sealed > deadline:
        _fail("pass_sealed_after_deadline")
    authority_issued = _require_int(
        authority.get("issued_at_unix_ns"),
        role="pass_authority_issued_at",
        minimum=generated,
    )
    authority_sealed = _require_int(
        authority.get("sealed_at_unix_ns"),
        role="pass_authority_sealed_at",
        minimum=authority_issued,
    )
    if authority_sealed > sealed:
        _fail("pass_authority_sealed_after_pass")
    if trace_payload["trace_signed_at_unix_ns"] >= authority_issued:
        _fail("generation_trace_not_pre_verification")
    if observer_at >= authority_issued:
        _fail("execution_observer_not_pre_verification")
    expected_task_commitment = _sha256_bytes(
        canonical_json_bytes(build_frontier_task_issuer_payload(task))
    )
    if receipt.get("sealed_task_commitment_sha256") != expected_task_commitment:
        _fail("sealed_task_commitment_mismatch")
    return dict(receipt)


def _authority_from_pass(
    store: TransitionArtifactStore,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return store.read_json(
        cast(Mapping[str, Any], receipt["verifier_authority_artifact"]),
        role="verifier_authority",
    )


def planned_transition_immutable_context(
    store: TransitionArtifactStore,
    *,
    task: FrontierTask,
    context: Mapping[str, Any],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    trust_context: TransitionTrustContext,
) -> dict[str, Any]:
    """Construct the immutable pass context before either invocation launches."""

    task = _require_exact_frontier_task(task)
    attempt_ledger = _validated_attempt_ledger(trust_context)
    if not isinstance(context, Mapping) or set(context) != _PASS_CONTEXT_KEYS:
        _fail("reasoning_pass_context_schema_invalid")
    prompt_artifact = store.put_bytes(
        task.public.prompt.encode("utf-8"),
        media_type="text/plain;charset=utf-8",
    )
    model_input_artifact = store.put_bytes(
        canonical_candidate_model_input(task),
        media_type="application/octet-stream",
    )
    execution_manifest_artifact = store.put_json(dict(trust_context.execution_manifest))
    return {
        "episode_id": _require_identifier(
            context.get("episode_id"),
            role="episode_id",
        ),
        "task_id": task.task_id,
        "case_id": _require_identifier(
            context.get("case_id"),
            role="case_id",
        ),
        "family": _require_identifier(
            context.get("family"),
            role="task_family",
        ),
        "domain": task.domain,
        "depth": _require_int(
            context.get("depth"),
            role="pass_depth",
            minimum=1,
        ),
        "difficulty": task.public.difficulty,
        "sealed_task_commitment_sha256": _require_sha256(
            context.get("sealed_task_commitment_sha256"),
            role="sealed_task_commitment",
        ),
        "prompt_artifact": prompt_artifact,
        "model_input_artifact": model_input_artifact,
        "model_identity_sha256": _require_sha256(
            context.get("model_identity_sha256"),
            role="model_identity",
        ),
        "base_checkpoint_sha256": _require_sha256(
            context.get("base_checkpoint_sha256"),
            role="base_checkpoint",
        ),
        "adapter_stack_sha256": _require_sha256(
            context.get("adapter_stack_sha256"),
            role="adapter_stack",
        ),
        "tokenizer_sha256": _require_sha256(
            context.get("tokenizer_sha256"),
            role="tokenizer",
        ),
        "token_encoder_callable": callable_commitment(token_encoder),
        "token_decoder_callable": callable_commitment(token_decoder),
        "policy_sha256": _require_sha256(
            context.get("policy_sha256"),
            role="policy",
        ),
        "personality_sha256": _require_sha256(
            context.get("personality_sha256"),
            role="personality",
        ),
        "runtime_sha256": _require_sha256(
            context.get("runtime_sha256"),
            role="runtime",
        ),
        "source_closure_sha256": _require_sha256(
            context.get("source_closure_sha256"),
            role="source_closure",
        ),
        "attempt_ledger_identity_sha256": _require_sha256(
            attempt_ledger.identity_sha256,
            role="attempt_ledger_identity",
        ),
        "execution_manifest_artifact": execution_manifest_artifact,
        "execution_spec_artifact": dict(context["execution_spec_artifact"]),
        "latent_path_artifact": dict(context["latent_path_artifact"]),
        "tool_snapshot_artifact": dict(context["tool_snapshot_artifact"]),
        "evidence_snapshot_artifact": dict(context["evidence_snapshot_artifact"]),
        "world_state_snapshot_artifact": dict(context["world_state_snapshot_artifact"]),
        "rng_root_sha256": _require_sha256(
            context.get("rng_root_sha256"),
            role="rng_root",
        ),
        "generation_budget": _validate_budget(context.get("generation_budget")),
        "deadline_unix_ns": _require_int(
            context.get("deadline_unix_ns"),
            role="pass_deadline",
            minimum=1,
        ),
    }


def planned_transition_immutable_context_sha256(
    store: TransitionArtifactStore,
    **kwargs: Any,
) -> str:
    return _sha256_bytes(
        canonical_json_bytes(planned_transition_immutable_context(store, **kwargs))
    )


def _immutable_pass_context(pass_receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {field: pass_receipt[field] for field in _PASS_IMMUTABLE_FIELDS}


def _generation_trace_attestation_sha256(
    pass_receipt: Mapping[str, Any],
) -> str:
    binding = _validate_artifact_binding(pass_receipt.get("generation_worker_attestation_artifact"))
    return binding["payload_sha256"]


def _execution_observer_attestation_sha256(
    pass_receipt: Mapping[str, Any],
) -> str:
    binding = _validate_artifact_binding(
        pass_receipt.get("execution_observer_attestation_artifact")
    )
    return binding["payload_sha256"]


def build_attempt_ledger_event_payload(
    *,
    episode_id: str,
    protocol_sha256: str,
    immutable_context_sha256: str,
    sequence: int,
    previous_event_sha256: str,
    event_time_unix_ns: int,
    event_type: str,
    event_fields: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_fields = {
        "episode_opened": {"planned_attempt_count", "launch_counter"},
        "attempt_launched": {
            "ordinal",
            "pass_index",
            "rng_root_sha256",
            "deadline_unix_ns",
            "launch_counter",
        },
        "attempt_finished": {
            "ordinal",
            "pass_index",
            "response_sha256",
            "generation_attestation_sha256",
            "execution_observer_attestation_sha256",
            "status",
            "launch_counter",
        },
        "episode_terminal": {
            "attempt_count",
            "final_pass_index",
            "terminal_state",
            "launch_counter",
        },
    }
    if (
        event_type not in allowed_fields
        or not isinstance(
            event_fields,
            Mapping,
        )
        or set(event_fields) != allowed_fields[event_type]
    ):
        _fail("attempt_ledger_event_fields_invalid")
    normalized_fields: dict[str, Any]
    if event_type == "episode_opened":
        if (
            event_fields.get("planned_attempt_count") != 2
            or event_fields.get("launch_counter") != 0
        ):
            _fail("attempt_ledger_open_event_invalid")
        normalized_fields = {
            "planned_attempt_count": 2,
            "launch_counter": 0,
        }
    elif event_type == "attempt_launched":
        ordinal = _require_int(
            event_fields.get("ordinal"),
            role="attempt_ledger_ordinal",
            maximum=1,
        )
        pass_index = _require_int(
            event_fields.get("pass_index"),
            role="attempt_ledger_pass_index",
            maximum=1,
        )
        launch_counter = _require_int(
            event_fields.get("launch_counter"),
            role="attempt_ledger_launch_counter",
            minimum=1,
            maximum=2,
        )
        if pass_index != ordinal or launch_counter != ordinal + 1:
            _fail("attempt_ledger_launch_event_invalid")
        normalized_fields = {
            "ordinal": ordinal,
            "pass_index": pass_index,
            "rng_root_sha256": _require_sha256(
                event_fields.get("rng_root_sha256"),
                role="attempt_ledger_rng_root",
            ),
            "deadline_unix_ns": _require_int(
                event_fields.get("deadline_unix_ns"),
                role="attempt_ledger_deadline",
                minimum=1,
            ),
            "launch_counter": launch_counter,
        }
    elif event_type == "attempt_finished":
        ordinal = _require_int(
            event_fields.get("ordinal"),
            role="attempt_ledger_ordinal",
            maximum=1,
        )
        pass_index = _require_int(
            event_fields.get("pass_index"),
            role="attempt_ledger_pass_index",
            maximum=1,
        )
        launch_counter = _require_int(
            event_fields.get("launch_counter"),
            role="attempt_ledger_launch_counter",
            minimum=1,
            maximum=2,
        )
        if (
            pass_index != ordinal
            or launch_counter != ordinal + 1
            or event_fields.get("status") != "completed"
        ):
            _fail("attempt_ledger_finish_event_invalid")
        normalized_fields = {
            "ordinal": ordinal,
            "pass_index": pass_index,
            "response_sha256": _require_sha256(
                event_fields.get("response_sha256"),
                role="attempt_ledger_response",
            ),
            "generation_attestation_sha256": _require_sha256(
                event_fields.get("generation_attestation_sha256"),
                role="attempt_ledger_generation_attestation",
            ),
            "execution_observer_attestation_sha256": _require_sha256(
                event_fields.get("execution_observer_attestation_sha256"),
                role="attempt_ledger_execution_observer_attestation",
            ),
            "status": "completed",
            "launch_counter": launch_counter,
        }
    else:
        if (
            event_fields.get("attempt_count") != 2
            or event_fields.get("final_pass_index") != 1
            or event_fields.get("terminal_state") != "attempts_completed"
            or event_fields.get("launch_counter") != 2
        ):
            _fail("attempt_ledger_terminal_event_invalid")
        normalized_fields = {
            "attempt_count": 2,
            "final_pass_index": 1,
            "terminal_state": "attempts_completed",
            "launch_counter": 2,
        }
    episode = _require_identifier(episode_id, role="attempt_ledger_episode")
    protocol = _require_sha256(protocol_sha256, role="transition_protocol")
    context_sha256 = _require_sha256(
        immutable_context_sha256,
        role="attempt_ledger_context",
    )
    normalized_sequence = _require_int(
        sequence,
        role="attempt_ledger_sequence",
        maximum=5,
    )
    expected_event_types = (
        "episode_opened",
        "attempt_launched",
        "attempt_finished",
        "attempt_launched",
        "attempt_finished",
        "episode_terminal",
    )
    if event_type != expected_event_types[normalized_sequence]:
        _fail("attempt_ledger_event_sequence_invalid")
    if event_type in {"attempt_launched", "attempt_finished"}:
        expected_ordinal = 0 if normalized_sequence < 3 else 1
        if normalized_fields["ordinal"] != expected_ordinal:
            _fail("attempt_ledger_event_ordinal_invalid")
    event: dict[str, Any] = {
        "schema": ATTEMPT_LEDGER_EVENT_SCHEMA,
        "episode_id": episode,
        "protocol_sha256": protocol,
        "immutable_context_sha256": context_sha256,
        "runner_session_sha256": _sha256_bytes(
            canonical_json_bytes(
                {
                    "episode_id": episode,
                    "protocol_sha256": protocol,
                    "immutable_context_sha256": context_sha256,
                }
            )
        ),
        "sequence": normalized_sequence,
        "previous_event_sha256": _require_sha256(
            previous_event_sha256,
            role="attempt_ledger_previous_event",
        ),
        "event_time_unix_ns": _require_signed_second(
            event_time_unix_ns,
            role="attempt_ledger_event_time",
        ),
        "event_type": event_type,
        **normalized_fields,
    }
    return event


def build_attempt_ledger_open_payload(
    *,
    episode_id: str,
    protocol_sha256: str,
    immutable_context_sha256: str,
    attempt_ledger_identity_sha256: str,
    opened_at_unix_ns: int,
) -> dict[str, Any]:
    """Bind one empty physical ledger before the first attempt launches."""

    return {
        "schema": ATTEMPT_LEDGER_OPEN_PAYLOAD_SCHEMA,
        "episode_id": _require_identifier(
            episode_id,
            role="attempt_ledger_episode",
        ),
        "protocol_sha256": _require_sha256(
            protocol_sha256,
            role="transition_protocol",
        ),
        "immutable_context_sha256": _require_sha256(
            immutable_context_sha256,
            role="attempt_ledger_context",
        ),
        "attempt_ledger_identity_sha256": _require_sha256(
            attempt_ledger_identity_sha256,
            role="attempt_ledger_identity",
        ),
        "initial_content_sha256": _sha256_bytes(b""),
        "planned_attempt_count": 2,
        "opened_at_unix_ns": _require_signed_second(
            opened_at_unix_ns,
            role="attempt_ledger_open_time",
        ),
    }


def build_attempt_ledger_terminal_payload(
    *,
    episode_id: str,
    protocol_sha256: str,
    immutable_context_sha256: str,
    attempt_ledger_identity_sha256: str,
    attempt_ledger_content_sha256: str,
    event_chain_head_sha256: str,
    terminal_at_unix_ns: int,
) -> dict[str, Any]:
    """Bind the only externally witnessed terminal ledger snapshot."""

    return {
        "schema": ATTEMPT_LEDGER_TERMINAL_PAYLOAD_SCHEMA,
        "episode_id": _require_identifier(
            episode_id,
            role="attempt_ledger_episode",
        ),
        "protocol_sha256": _require_sha256(
            protocol_sha256,
            role="transition_protocol",
        ),
        "immutable_context_sha256": _require_sha256(
            immutable_context_sha256,
            role="attempt_ledger_context",
        ),
        "attempt_ledger_identity_sha256": _require_sha256(
            attempt_ledger_identity_sha256,
            role="attempt_ledger_identity",
        ),
        "attempt_ledger_content_sha256": _require_sha256(
            attempt_ledger_content_sha256,
            role="attempt_ledger_content",
        ),
        "event_count": 6,
        "event_chain_head_sha256": _require_sha256(
            event_chain_head_sha256,
            role="attempt_ledger_event_chain_head",
        ),
        "terminal_state": "attempts_completed",
        "terminal_at_unix_ns": _require_signed_second(
            terminal_at_unix_ns,
            role="attempt_ledger_terminal_time",
        ),
    }


def _verify_attempt_ledger_open_attestation(
    *,
    policy: VerifiedCampaignTrustPolicy,
    trust_context: TransitionTrustContext,
    episode_id: str,
    immutable_context_sha256: str,
    not_after_unix: int,
) -> dict[str, Any]:
    attestation = trust_context.attempt_ledger_open_attestation
    signed_document = (
        attestation.get("signed_payload") if isinstance(attestation, Mapping) else None
    )
    submitted_payload = (
        signed_document.get("payload") if isinstance(signed_document, Mapping) else None
    )
    opened_at = _require_signed_second(
        submitted_payload.get("opened_at_unix_ns")
        if isinstance(submitted_payload, Mapping)
        else None,
        role="attempt_ledger_open_time",
    )
    expected_payload = build_attempt_ledger_open_payload(
        episode_id=episode_id,
        protocol_sha256=trust_context.expected_protocol_sha256,
        immutable_context_sha256=immutable_context_sha256,
        attempt_ledger_identity_sha256=(trust_context.expected_attempt_ledger_identity_sha256),
        opened_at_unix_ns=opened_at,
    )
    verified = _verify_role_attestation(
        policy,
        attestation,
        role=TASK_ISSUER,
        expected_payload=expected_payload,
        not_after_unix=not_after_unix,
    )
    if opened_at != verified["signed_at_unix"] * 1_000_000_000:
        _fail("attempt_ledger_open_timestamp_mismatch")
    return dict(cast(Mapping[str, Any], attestation))


def _verify_attempt_ledger_terminal_attestation(
    *,
    policy: VerifiedCampaignTrustPolicy,
    trust_context: TransitionTrustContext,
    episode_id: str,
    immutable_context_sha256: str,
    attempt_ledger_content_sha256: str,
    event_chain_head_sha256: str,
    last_event_time_unix_ns: int,
) -> dict[str, Any]:
    attestation = trust_context.attempt_ledger_terminal_attestation
    signed_document = (
        attestation.get("signed_payload") if isinstance(attestation, Mapping) else None
    )
    submitted_payload = (
        signed_document.get("payload") if isinstance(signed_document, Mapping) else None
    )
    terminal_at = _require_signed_second(
        submitted_payload.get("terminal_at_unix_ns")
        if isinstance(submitted_payload, Mapping)
        else None,
        role="attempt_ledger_terminal_time",
    )
    if terminal_at <= last_event_time_unix_ns:
        _fail("attempt_ledger_terminal_not_after_events")
    expected_payload = build_attempt_ledger_terminal_payload(
        episode_id=episode_id,
        protocol_sha256=trust_context.expected_protocol_sha256,
        immutable_context_sha256=immutable_context_sha256,
        attempt_ledger_identity_sha256=(trust_context.expected_attempt_ledger_identity_sha256),
        attempt_ledger_content_sha256=attempt_ledger_content_sha256,
        event_chain_head_sha256=event_chain_head_sha256,
        terminal_at_unix_ns=terminal_at,
    )
    verified = _verify_role_attestation(
        policy,
        attestation,
        role=EVIDENCE_VERIFIER,
        expected_payload=expected_payload,
        not_after_unix=trust_context.observed_at_unix,
    )
    if terminal_at != verified["signed_at_unix"] * 1_000_000_000:
        _fail("attempt_ledger_terminal_timestamp_mismatch")
    return dict(cast(Mapping[str, Any], attestation))


def build_attempt_ledger_event_payloads(
    *,
    pass_0: Mapping[str, Any],
    pass_1: Mapping[str, Any],
    protocol_sha256: str,
    event_times_unix_ns: Sequence[int],
) -> list[dict[str, Any]]:
    """Build the fixed six-event runner ledger around both invocations."""

    if (
        not isinstance(pass_0, Mapping)
        or not isinstance(pass_1, Mapping)
        or set(pass_0) != _PASS_KEYS
        or set(pass_1) != _PASS_KEYS
    ):
        _fail("attempt_ledger_pass_schema_invalid")
    _validate_digest(pass_0, role="attempt_ledger_pass_0")
    _validate_digest(pass_1, role="attempt_ledger_pass_1")
    if pass_0.get("pass_index") != 0 or pass_1.get("pass_index") != 1:
        _fail("attempt_ledger_pass_order_invalid")
    if _immutable_pass_context(pass_0) != _immutable_pass_context(pass_1):
        _fail("attempt_ledger_context_drift")
    if (
        not isinstance(event_times_unix_ns, Sequence)
        or isinstance(event_times_unix_ns, (str, bytes, bytearray))
        or len(event_times_unix_ns) != 6
    ):
        _fail("attempt_ledger_event_times_invalid")
    times = [
        _require_signed_second(
            value,
            role="attempt_ledger_event_time",
        )
        for value in event_times_unix_ns
    ]
    generated_0 = cast(int, pass_0["generated_at_unix_ns"])
    generated_1 = cast(int, pass_1["generated_at_unix_ns"])
    if not (
        times[0]
        < times[1]
        < generated_0
        <= times[2]
        < times[3]
        < generated_1
        <= times[4]
        < times[5]
    ):
        _fail("attempt_ledger_event_chronology_invalid")
    context_sha256 = _sha256_bytes(canonical_json_bytes(_immutable_pass_context(pass_0)))
    protocol = _require_sha256(protocol_sha256, role="transition_protocol")
    descriptions: list[dict[str, Any]] = [
        {
            "event_type": "episode_opened",
            "planned_attempt_count": 2,
            "launch_counter": 0,
        },
        {
            "event_type": "attempt_launched",
            "ordinal": 0,
            "pass_index": 0,
            "rng_root_sha256": pass_0["rng_root_sha256"],
            "deadline_unix_ns": pass_0["deadline_unix_ns"],
            "launch_counter": 1,
        },
        {
            "event_type": "attempt_finished",
            "ordinal": 0,
            "pass_index": 0,
            "response_sha256": pass_0["response_artifact"]["payload_sha256"],
            "generation_attestation_sha256": (_generation_trace_attestation_sha256(pass_0)),
            "execution_observer_attestation_sha256": (
                _execution_observer_attestation_sha256(pass_0)
            ),
            "status": "completed",
            "launch_counter": 1,
        },
        {
            "event_type": "attempt_launched",
            "ordinal": 1,
            "pass_index": 1,
            "rng_root_sha256": pass_1["rng_root_sha256"],
            "deadline_unix_ns": pass_1["deadline_unix_ns"],
            "launch_counter": 2,
        },
        {
            "event_type": "attempt_finished",
            "ordinal": 1,
            "pass_index": 1,
            "response_sha256": pass_1["response_artifact"]["payload_sha256"],
            "generation_attestation_sha256": (_generation_trace_attestation_sha256(pass_1)),
            "execution_observer_attestation_sha256": (
                _execution_observer_attestation_sha256(pass_1)
            ),
            "status": "completed",
            "launch_counter": 2,
        },
        {
            "event_type": "episode_terminal",
            "attempt_count": 2,
            "final_pass_index": 1,
            "terminal_state": "attempts_completed",
            "launch_counter": 2,
        },
    ]
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for sequence, (event_time, description) in enumerate(zip(times, descriptions, strict=True)):
        event_type = cast(str, description.pop("event_type"))
        event = build_attempt_ledger_event_payload(
            episode_id=cast(str, pass_0["episode_id"]),
            protocol_sha256=protocol,
            immutable_context_sha256=context_sha256,
            sequence=sequence,
            previous_event_sha256=previous,
            event_time_unix_ns=event_time,
            event_type=event_type,
            event_fields=description,
        )
        events.append(event)
        previous = _sha256_bytes(canonical_json_bytes(event))
    return events


def build_transition_attempt_journal(
    *,
    pass_0: Mapping[str, Any],
    pass_1: Mapping[str, Any],
    protocol_sha256: str,
    trust_context: TransitionTrustContext,
) -> dict[str, Any]:
    """Freeze the pinned external append-only attempt inventory."""

    if (
        not isinstance(pass_0, Mapping)
        or not isinstance(pass_1, Mapping)
        or set(pass_0) != _PASS_KEYS
        or set(pass_1) != _PASS_KEYS
    ):
        _fail("attempt_journal_pass_schema_invalid")
    _validate_digest(pass_0, role="attempt_journal_pass_0")
    _validate_digest(pass_1, role="attempt_journal_pass_1")
    if pass_0.get("pass_index") != 0 or pass_1.get("pass_index") != 1:
        _fail("attempt_journal_pass_order_invalid")
    if pass_0.get("episode_id") != pass_1.get("episode_id"):
        _fail("attempt_journal_episode_mismatch")
    immutable_context = _immutable_pass_context(pass_0)
    if immutable_context != _immutable_pass_context(pass_1):
        _fail("attempt_journal_context_drift")
    context_sha256 = _sha256_bytes(canonical_json_bytes(immutable_context))
    ledger = _validated_attempt_ledger(trust_context)
    runner_event_attestations, ledger_content_sha256 = ledger.snapshot()
    if (
        not isinstance(runner_event_attestations, Sequence)
        or isinstance(runner_event_attestations, (str, bytes, bytearray))
        or len(runner_event_attestations) != 6
    ):
        _fail("attempt_ledger_attestations_invalid")
    event_times: list[int] = []
    for attestation in runner_event_attestations:
        signed_payload = (
            attestation.get("signed_payload") if isinstance(attestation, Mapping) else None
        )
        payload = signed_payload.get("payload") if isinstance(signed_payload, Mapping) else None
        event_times.append(
            _require_signed_second(
                payload.get("event_time_unix_ns") if isinstance(payload, Mapping) else None,
                role="attempt_ledger_event_time",
            )
        )
    expected_events = build_attempt_ledger_event_payloads(
        pass_0=pass_0,
        pass_1=pass_1,
        protocol_sha256=protocol_sha256,
        event_times_unix_ns=event_times,
    )
    policy = trust_context.verified_policy()
    if not operationally_isolated_roles(policy):
        _fail("transition_operational_role_custody_required")
    open_attestation = _verify_attempt_ledger_open_attestation(
        policy=policy,
        trust_context=trust_context,
        episode_id=cast(str, pass_0["episode_id"]),
        immutable_context_sha256=context_sha256,
        not_after_unix=event_times[0] // 1_000_000_000,
    )
    if open_attestation["signed_payload"]["payload"]["opened_at_unix_ns"] >= event_times[0]:
        _fail("attempt_ledger_open_not_before_events")
    verified_attestations: list[dict[str, Any]] = []
    for event, attestation in zip(
        expected_events,
        runner_event_attestations,
        strict=True,
    ):
        verified = _verify_role_attestation(
            policy,
            attestation,
            role=CAMPAIGN_RUNNER,
            expected_payload=event,
            not_after_unix=trust_context.observed_at_unix,
        )
        if event["event_time_unix_ns"] != (verified["signed_at_unix"] * 1_000_000_000):
            _fail("attempt_ledger_event_timestamp_mismatch")
        verified_attestations.append(dict(attestation))
    event_chain_head_sha256 = _sha256_bytes(canonical_json_bytes(expected_events[-1]))
    terminal_attestation = _verify_attempt_ledger_terminal_attestation(
        policy=policy,
        trust_context=trust_context,
        episode_id=cast(str, pass_0["episode_id"]),
        immutable_context_sha256=context_sha256,
        attempt_ledger_content_sha256=ledger_content_sha256,
        event_chain_head_sha256=event_chain_head_sha256,
        last_event_time_unix_ns=event_times[-1],
    )
    return _seal_document(
        {
            "schema": ATTEMPT_JOURNAL_SCHEMA,
            "episode_id": pass_0["episode_id"],
            "protocol_sha256": _require_sha256(
                protocol_sha256,
                role="transition_protocol",
            ),
            "immutable_context_sha256": context_sha256,
            "attempt_count": 2,
            "attempts": [
                {
                    "ordinal": 0,
                    "pass_index": 0,
                    "pass_receipt_sha256": pass_0["receipt_sha256"],
                    "response_sha256": pass_0["response_artifact"]["payload_sha256"],
                },
                {
                    "ordinal": 1,
                    "pass_index": 1,
                    "pass_receipt_sha256": pass_1["receipt_sha256"],
                    "response_sha256": pass_1["response_artifact"]["payload_sha256"],
                },
            ],
            "runner_event_attestations": verified_attestations,
            "attempt_ledger_identity_sha256": ledger.identity_sha256,
            "attempt_ledger_content_sha256": ledger_content_sha256,
            "attempt_ledger_open_attestation": open_attestation,
            "attempt_ledger_terminal_attestation": terminal_attestation,
            "event_chain_head_sha256": event_chain_head_sha256,
            "terminal_state": "sealed",
            "final_pass_index": 1,
        }
    )


def build_campaign_runner_journal_payload(
    attempt_journal: Mapping[str, Any],
    *,
    signed_at_unix_ns: int,
) -> dict[str, Any]:
    """Return the exact complete-attempt claim the runner role must sign."""

    if (
        not isinstance(attempt_journal, Mapping)
        or set(attempt_journal) != _ATTEMPT_JOURNAL_KEYS
        or attempt_journal.get("schema") != ATTEMPT_JOURNAL_SCHEMA
    ):
        _fail("attempt_journal_schema_invalid")
    _validate_digest(attempt_journal, role="attempt_journal")
    return {
        "schema": "aura.verified_transition.runner_journal_payload.v1",
        "episode_id": attempt_journal["episode_id"],
        "protocol_sha256": attempt_journal["protocol_sha256"],
        "immutable_context_sha256": attempt_journal["immutable_context_sha256"],
        "attempt_count": attempt_journal["attempt_count"],
        "attempts_sha256": _sha256_bytes(canonical_json_bytes(attempt_journal["attempts"])),
        "terminal_state": attempt_journal["terminal_state"],
        "attempt_journal_sha256": attempt_journal["receipt_sha256"],
        "attempt_ledger_content_sha256": attempt_journal["attempt_ledger_content_sha256"],
        "attempt_ledger_terminal_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(attempt_journal["attempt_ledger_terminal_attestation"])
        ),
        "event_chain_head_sha256": attempt_journal["event_chain_head_sha256"],
        "signed_at_unix_ns": _require_signed_second(
            signed_at_unix_ns,
            role="runner_journal_signed_at",
        ),
    }


def build_evidence_verifier_journal_payload(
    attempt_journal: Mapping[str, Any],
    campaign_runner_attestation: Mapping[str, Any],
    *,
    signed_at_unix_ns: int,
) -> dict[str, Any]:
    runner_signed_payload = (
        campaign_runner_attestation.get("signed_payload")
        if isinstance(campaign_runner_attestation, Mapping)
        else None
    )
    runner_payload = (
        runner_signed_payload.get("payload") if isinstance(runner_signed_payload, Mapping) else None
    )
    runner_time = (
        runner_payload.get("signed_at_unix_ns") if isinstance(runner_payload, Mapping) else None
    )
    expected_runner_payload = build_campaign_runner_journal_payload(
        attempt_journal,
        signed_at_unix_ns=_require_signed_second(
            runner_time,
            role="runner_journal_signed_at",
        ),
    )
    verifier_time = _require_signed_second(
        signed_at_unix_ns,
        role="verifier_journal_signed_at",
    )
    if verifier_time <= expected_runner_payload["signed_at_unix_ns"]:
        _fail("verifier_journal_not_after_runner")
    return {
        **expected_runner_payload,
        "schema": "aura.verified_transition.verifier_journal_payload.v1",
        "campaign_runner_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(dict(campaign_runner_attestation))
        ),
        "verifier_signed_at_unix_ns": verifier_time,
    }


def build_verified_transition_episode(
    store: TransitionArtifactStore,
    *,
    pass_0: Mapping[str, Any],
    pass_1: Mapping[str, Any],
    task: FrontierTask,
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    trust_context: TransitionTrustContext,
    attempt_journal: Mapping[str, Any],
    campaign_runner_attestation: Mapping[str, Any],
    evidence_verifier_journal_attestation: Mapping[str, Any],
    created_at_unix_ns: int | None = None,
    sealed_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Seal exactly two independently replayed passes under one context."""

    first = validate_reasoning_pass_receipt(
        store,
        pass_0,
        task=task,
        expected_authority=expected_authority,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
    )
    second = validate_reasoning_pass_receipt(
        store,
        pass_1,
        task=task,
        expected_authority=expected_authority,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
    )
    if first["pass_index"] != 0 or second["pass_index"] != 1:
        _fail("transition_pass_order_invalid")
    for field in _PASS_IMMUTABLE_FIELDS:
        if first.get(field) != second.get(field):
            _fail(f"transition_{field}_drift")
    if first["response_artifact"] == second["response_artifact"]:
        _fail("transition_response_reused")
    first_authority = _authority_from_pass(store, first)
    second_authority = _authority_from_pass(store, second)
    for field in _AUTHORITY_IMMUTABLE_FIELDS:
        if first_authority.get(field) != second_authority.get(field):
            _fail(f"transition_verifier_{field}_drift")

    expected_authority_values = _validate_expected_authority(expected_authority)
    policy = _verified_transition_policy(
        trust_context,
        expected_authority_values,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    expected_journal = build_transition_attempt_journal(
        pass_0=first,
        pass_1=second,
        protocol_sha256=trust_context.expected_protocol_sha256,
        trust_context=trust_context,
    )
    if dict(attempt_journal) != expected_journal:
        _fail("transition_attempt_journal_mismatch")
    runner_signed_document = (
        campaign_runner_attestation.get("signed_payload")
        if isinstance(campaign_runner_attestation, Mapping)
        else None
    )
    runner_submitted_payload = (
        runner_signed_document.get("payload")
        if isinstance(runner_signed_document, Mapping)
        else None
    )
    runner_signed_at_ns = _require_signed_second(
        runner_submitted_payload.get("signed_at_unix_ns")
        if isinstance(runner_submitted_payload, Mapping)
        else None,
        role="runner_journal_signed_at",
    )
    maximum_pass_seal = max(
        cast(int, first["sealed_at_unix_ns"]),
        cast(int, second["sealed_at_unix_ns"]),
    )
    if runner_signed_at_ns < maximum_pass_seal:
        _fail("runner_journal_signed_before_pass_seal")
    runner_payload = build_campaign_runner_journal_payload(
        expected_journal,
        signed_at_unix_ns=runner_signed_at_ns,
    )
    runner_signed_payload = _verify_role_attestation(
        policy,
        campaign_runner_attestation,
        role=CAMPAIGN_RUNNER,
        expected_payload=runner_payload,
        not_before_unix=(maximum_pass_seal + 999_999_999) // 1_000_000_000,
        not_after_unix=trust_context.observed_at_unix,
    )
    if runner_signed_at_ns != (runner_signed_payload["signed_at_unix"] * 1_000_000_000):
        _fail("runner_journal_timestamp_mismatch")
    verifier_signed_document = (
        evidence_verifier_journal_attestation.get("signed_payload")
        if isinstance(evidence_verifier_journal_attestation, Mapping)
        else None
    )
    verifier_submitted_payload = (
        verifier_signed_document.get("payload")
        if isinstance(verifier_signed_document, Mapping)
        else None
    )
    verifier_signed_at_ns = _require_signed_second(
        verifier_submitted_payload.get("verifier_signed_at_unix_ns")
        if isinstance(verifier_submitted_payload, Mapping)
        else None,
        role="verifier_journal_signed_at",
    )
    verifier_journal_payload = build_evidence_verifier_journal_payload(
        expected_journal,
        campaign_runner_attestation,
        signed_at_unix_ns=verifier_signed_at_ns,
    )
    verifier_signed_payload = _verify_role_attestation(
        policy,
        evidence_verifier_journal_attestation,
        role=EVIDENCE_VERIFIER,
        expected_payload=verifier_journal_payload,
        not_before_unix=(runner_signed_at_ns + 999_999_999) // 1_000_000_000,
        not_after_unix=trust_context.observed_at_unix,
    )
    if verifier_signed_at_ns != (verifier_signed_payload["signed_at_unix"] * 1_000_000_000):
        _fail("verifier_journal_timestamp_mismatch")

    created = (
        time.time_ns()
        if created_at_unix_ns is None
        else _require_int(
            created_at_unix_ns,
            role="episode_created_at",
            minimum=1,
        )
    )
    if created < maximum_pass_seal:
        _fail("episode_created_before_pass_seal")
    if created < max(runner_signed_at_ns, verifier_signed_at_ns):
        _fail("episode_created_before_role_attestations")
    if cast(int, second["generated_at_unix_ns"]) >= cast(int, first_authority["issued_at_unix_ns"]):
        _fail("transition_verifier_feedback_isolation_failed")
    sealed = (
        max(created, time.time_ns())
        if sealed_at_unix_ns is None
        else _require_int(
            sealed_at_unix_ns,
            role="episode_sealed_at",
            minimum=created,
        )
    )
    pass_0_artifact = store.put_json(first)
    pass_1_artifact = store.put_json(second)
    journal_artifact = store.put_json(expected_journal)
    runner_attestation_artifact = store.put_json(dict(campaign_runner_attestation))
    verifier_journal_attestation_artifact = store.put_json(
        dict(evidence_verifier_journal_attestation)
    )
    return _seal_document(
        {
            "schema": VERIFIED_TRANSITION_EPISODE_SCHEMA,
            "episode_id": first["episode_id"],
            "pass_count": 2,
            "pass_0_artifact": pass_0_artifact,
            "pass_1_artifact": pass_1_artifact,
            "task_id": task.task_id,
            "sealed_task_commitment_sha256": first["sealed_task_commitment_sha256"],
            "protocol_sha256": trust_context.expected_protocol_sha256,
            "immutable_context_sha256": expected_journal["immutable_context_sha256"],
            "trust_policy_sha256": policy.policy_sha256,
            "attempt_journal_artifact": journal_artifact,
            "campaign_runner_attestation_artifact": (runner_attestation_artifact),
            "evidence_verifier_journal_attestation_artifact": (
                verifier_journal_attestation_artifact
            ),
            "created_at_unix_ns": created,
            "sealed_at_unix_ns": sealed,
        }
    )


def validate_verified_transition_episode(
    store: TransitionArtifactStore,
    receipt: Mapping[str, Any],
    *,
    task: FrontierTask,
    expected_authority: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    trust_context: TransitionTrustContext,
) -> dict[str, Any]:
    """Independently reconstruct a complete paired transition episode."""

    if not isinstance(receipt, Mapping) or set(receipt) != _EPISODE_KEYS:
        _fail("transition_episode_schema_invalid")
    if receipt.get("schema") != VERIFIED_TRANSITION_EPISODE_SCHEMA:
        _fail("transition_episode_version_invalid")
    _validate_digest(receipt, role="transition_episode")
    if receipt.get("pass_count") != 2 or receipt.get("task_id") != task.task_id:
        _fail("transition_episode_identity_invalid")
    first = store.read_json(
        cast(Mapping[str, Any], receipt["pass_0_artifact"]),
        role="reasoning_pass",
    )
    second = store.read_json(
        cast(Mapping[str, Any], receipt["pass_1_artifact"]),
        role="reasoning_pass",
    )
    attempt_journal = store.read_json(
        cast(Mapping[str, Any], receipt["attempt_journal_artifact"]),
        role="attempt_journal",
    )
    campaign_runner_attestation = store.read_json(
        cast(
            Mapping[str, Any],
            receipt["campaign_runner_attestation_artifact"],
        ),
        role="campaign_runner_attestation",
    )
    evidence_verifier_journal_attestation = store.read_json(
        cast(
            Mapping[str, Any],
            receipt["evidence_verifier_journal_attestation_artifact"],
        ),
        role="evidence_verifier_journal_attestation",
    )
    expected = build_verified_transition_episode(
        store,
        pass_0=first,
        pass_1=second,
        task=task,
        expected_authority=expected_authority,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=trust_context,
        attempt_journal=attempt_journal,
        campaign_runner_attestation=campaign_runner_attestation,
        evidence_verifier_journal_attestation=(evidence_verifier_journal_attestation),
        created_at_unix_ns=_require_int(
            receipt.get("created_at_unix_ns"),
            role="episode_created_at",
            minimum=1,
        ),
        sealed_at_unix_ns=_require_int(
            receipt.get("sealed_at_unix_ns"),
            role="episode_sealed_at",
            minimum=1,
        ),
    )
    if expected != dict(receipt):
        _fail("transition_episode_reconstruction_mismatch")
    return dict(receipt)


__all__ = [
    "ARTIFACT_BINDING_SCHEMA",
    "ATTEMPT_LEDGER_EVENT_SCHEMA",
    "ATTEMPT_LEDGER_OPEN_PAYLOAD_SCHEMA",
    "ATTEMPT_LEDGER_TERMINAL_PAYLOAD_SCHEMA",
    "ATTEMPT_JOURNAL_SCHEMA",
    "CALIBRATION_CASE_SCHEMA",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CALIBRATION_PAYLOAD_SCHEMA",
    "CALLABLE_COMMITMENT_SCHEMA",
    "CANDIDATE_INPUT_SCHEMA",
    "EXECUTION_MANIFEST_SCHEMA",
    "EXECUTION_OBSERVER_PAYLOAD_SCHEMA",
    "EXECUTION_PROCESS_OBSERVATION_SCHEMA",
    "GENERATION_TRACE_PAYLOAD_SCHEMA",
    "PRIMARY_SCORE_SCHEMA",
    "REASONING_PASS_SCHEMA",
    "ExternalAttemptLedger",
    "TransitionArtifactStore",
    "TransitionTrustContext",
    "VERIFIED_TRANSITION_EPISODE_SCHEMA",
    "VERIFIER_AUTHORITY_SCHEMA",
    "VerifiedTransitionError",
    "WITNESS_SCORE_SCHEMA",
    "build_attempt_ledger_event_payload",
    "build_attempt_ledger_event_payloads",
    "build_attempt_ledger_open_payload",
    "build_attempt_ledger_terminal_payload",
    "build_calibration_case",
    "build_calibration_payload",
    "build_campaign_runner_journal_payload",
    "build_evidence_verifier_journal_payload",
    "build_execution_manifest",
    "build_execution_observer_payload",
    "build_frontier_task_issuer_payload",
    "build_frontier_witness_payload",
    "build_generation_trace_payload",
    "build_reasoning_pass_receipt",
    "build_transition_attempt_journal",
    "build_verified_transition_episode",
    "callable_commitment",
    "canonical_candidate_model_input",
    "canonical_json_bytes",
    "capture_execution_process_observation",
    "execution_observer_implementation_identity",
    "issue_frontier_verifier_authority",
    "planned_transition_immutable_context",
    "planned_transition_immutable_context_sha256",
    "seal_calibration_evidence",
    "strict_canonical_json_loads",
    "validate_frontier_verifier_authority",
    "validate_reasoning_pass_receipt",
    "validate_verified_transition_episode",
    "verifier_implementation_identity",
    "verifier_implementation_manifest",
]
