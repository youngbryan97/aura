"""Quarantined SFT projection for encrypted, causally verified replay.

The verified replay ledger intentionally retains more private evidence than a
trainer may see.  This module is the one-way boundary between those stores:
it authenticates and decrypts replay entries in memory, selects only the
model-visible task and accepted answer, applies local privacy and poisoning
screens, freezes every causal lineage into a pre-augmentation split, and
checks keyed exact and near-duplicate signatures against a caller-supplied
sealed reference index.

The resulting candidate and evaluator artifacts are physically separable and
cryptographically linked.  They remain quarantined: neither this module nor
its artifacts grant training authority.  External privacy, contamination,
trainer, and promotion attestations remain mandatory.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.verified_replay_buffer import (
    ReplayProtector,
    materialize_verified_replay_entry,
    validate_verified_replay_store,
)

VERIFIED_REPLAY_SFT_PRIVACY_CLEARANCE_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_privacy_clearance.v1"
)
VERIFIED_REPLAY_SFT_REFERENCE_INDEX_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_reference_index.v1"
)
VERIFIED_REPLAY_SFT_PARTITION_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_pre_augmentation_partition.v1"
)
VERIFIED_REPLAY_SFT_DEDUP_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_semantic_dedup.v1"
)
VERIFIED_REPLAY_SFT_PRIVACY_MANIFEST_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_privacy_manifest.v1"
)
VERIFIED_REPLAY_SFT_EXAMPLE_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_example.v1"
)
VERIFIED_REPLAY_SFT_CANDIDATE_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_candidate_package.v2"
)
VERIFIED_REPLAY_SFT_EVALUATOR_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_evaluator_package.v1"
)
VERIFIED_REPLAY_SFT_CUSTODY_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_custody_report.v1"
)
VERIFIED_REPLAY_SFT_TOKENIZATION_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_tokenization.v1"
)

TRAIN_SPLIT: Final = "train"
VALIDATION_SPLIT: Final = "validation"
HOLDOUT_SPLIT: Final = "holdout"
SPLITS: Final = (TRAIN_SPLIT, VALIDATION_SPLIT, HOLDOUT_SPLIT)

_CANDIDATE_TRAIN_FILE = "verified_replay_train.jsonl"
_CANDIDATE_VALIDATION_FILE = "verified_replay_validation.jsonl"
_CANDIDATE_MANIFEST_FILE = "verified_replay_candidate_manifest.json"
_EVALUATOR_HOLDOUT_FILE = "verified_replay_holdout.json"
_EVALUATOR_MANIFEST_FILE = "verified_replay_evaluator_manifest.json"

VERIFIED_REPLAY_SFT_CANDIDATE_FILES: Final = (
    _CANDIDATE_TRAIN_FILE,
    _CANDIDATE_VALIDATION_FILE,
    _CANDIDATE_MANIFEST_FILE,
)
VERIFIED_REPLAY_SFT_EVALUATOR_FILES: Final = (
    _EVALUATOR_HOLDOUT_FILE,
    _EVALUATOR_MANIFEST_FILE,
)

ZERO_SHA256: Final = "0" * 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TEXT_CHARS = 131_072
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_REFERENCE_RECORDS = 10_000
_SHINGLE_SKETCH_LIMIT = 512
_MAX_SIGNATURES_PER_RECORD = _SHINGLE_SKETCH_LIMIT
_MIN_KEY_BYTES = 32
_MAX_KEY_BYTES = 256
_DEFAULT_TOKEN_SHINGLE_SIZE = 4
_DEFAULT_CHARACTER_SHINGLE_SIZE = 12
_DEFAULT_MAX_SEQ_LENGTH = 4096
_TOKEN_NEAR_DUPLICATE_THRESHOLD = 0.82
_CHARACTER_NEAR_DUPLICATE_THRESHOLD = 0.88

_PRIVATE_FIELD_NAMES = frozenset(
    {
        "initial_failure",
        "baseline_decode",
        "failed_atom",
        "earliest_causal_error",
        "discriminating_test",
        "original_route",
        "corrected_route",
        "corrected_transition",
        "preserved_prefix",
        "replacement_suffix",
        "corrected_atom",
        "verified_solution",
        "output_quality",
        "tokens",
        "escape_strategy",
        "provenance",
        "privacy_governance_disposition",
    }
)
_HIDDEN_REASONING_PATTERNS = (
    re.compile(r"<\s*/?\s*(?:think|analysis|reasoning|scratchpad)\s*>", re.I),
    re.compile(r"\b(?:hidden|private|internal)\s+(?:chain[ -]of[ -]thought|reasoning|monologue)\b", re.I),
    re.compile(r"\b(?:chain[ -]of[ -]thought|scratchpad)\s*:", re.I),
)
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions?\b", re.I),
    re.compile(r"\b(?:reveal|print|repeat|exfiltrate)\s+(?:the\s+)?system\s+prompt\b", re.I),
    re.compile(r"\b(?:developer|system)\s+message\s*:", re.I),
    re.compile(r"\bdo\s+not\s+follow\s+(?:the\s+)?(?:system|developer)\b", re.I),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk|rk|pk)-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|password|passwd|secret)\s*[:=]\s*[^\s,;]{8,}",
        re.I,
    ),
)
_PII_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
)
_PAYMENT_CARD_RE = re.compile(
    r"(?<!\d)(?:\d{13,19}|(?:\d{4}[ -]){3}\d{4})(?!\d)"
)

_PRIVACY_FIELDS = {
    "schema",
    "entry_sha256",
    "experience_sha256",
    "projection_content_sha256",
    "origin_classification",
    "contains_user_content",
    "consent_basis",
    "consent_receipt_sha256",
    "license_basis",
    "license_receipt_sha256",
    "tenant_scope",
    "tenant_commitment_sha256",
    "retention_active",
    "revoked",
    "deleted",
    "pii_findings",
    "secret_findings",
    "user_secret_findings",
    "hidden_reasoning_findings",
    "implementation_sha256",
    "release_sha256",
    "status",
    "clearance_sha256",
}
_REFERENCE_RECORD_FIELDS = {
    "record_id_sha256",
    "corpus",
    "split",
    "lineage_token",
    "exact_token",
    "objective_token",
    "answer_token",
    "objective_character_count",
    "answer_character_count",
    "token_shingles",
    "character_shingles",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


class VerifiedReplaySFTError(ValueError):
    """Stable fail-closed projection error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = VerifiedReplaySFTError(code)
    raise error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise VerifiedReplaySFTError("verified_replay_sft_json_invalid") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha(value: Any, *, code: str) -> str:
    if not _is_sha256(value):
        _fail(code)
    return value


def _key(value: Any, *, code: str) -> bytes:
    if not isinstance(value, bytes) or not _MIN_KEY_BYTES <= len(value) <= _MAX_KEY_BYTES:
        _fail(code)
    return value


def _hmac(key: bytes, domain: str, value: Any) -> str:
    payload = domain.encode("ascii") + b"\0" + canonical_json_bytes(value)
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _committed(raw: Any, *, fields: set[str], digest_field: str, code: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        _fail(code)
    normalized = json.loads(canonical_json_bytes(raw))
    body = dict(normalized)
    digest = body.pop(digest_field, None)
    if not _is_sha256(digest) or _sha(body) != digest:
        _fail(f"{code}_commitment_invalid")
    return normalized


def _normalized_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT_CHARS:
        _fail("verified_replay_sft_text_invalid")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character) in {"Cc", "Cs"} and character not in "\n\t" for character in normalized):
        _fail("verified_replay_sft_text_control_character")
    return " ".join(normalized.casefold().split())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[^\w\s]", _normalized_text(value), flags=re.UNICODE)


def _windows(items: Sequence[str], size: int):
    if not items:
        return
    width = min(size, len(items))
    for index in range(len(items) - width + 1):
        yield "\x1f".join(items[index : index + width])


def _character_windows(value: str, size: int):
    normalized = _normalized_text(value)
    if not normalized:
        return
    width = min(size, len(normalized))
    for index in range(len(normalized) - width + 1):
        yield normalized[index : index + width]


def _bottom_k_hmac(
    key: bytes,
    domain: str,
    values,
    *,
    limit: int = _SHINGLE_SKETCH_LIMIT,
) -> list[str]:
    """Retain the deterministic smallest keyed hashes with bounded memory."""

    heap: list[tuple[int, str]] = []
    selected: set[str] = set()
    for value in values:
        digest = _hmac(key, domain, value)
        if digest in selected:
            continue
        score = int(digest, 16)
        item = (-score, digest)
        if len(heap) < limit:
            heapq.heappush(heap, item)
            selected.add(digest)
            continue
        if score >= -heap[0][0]:
            continue
        _negated, removed = heapq.heapreplace(heap, item)
        selected.remove(removed)
        selected.add(digest)
    return sorted(selected)


def _signature(
    *,
    objective: str,
    answer: str,
    lineage_root_sha256: str,
    dedup_key: bytes,
) -> dict[str, Any]:
    content = {"objective": _normalized_text(objective), "answer": _normalized_text(answer)}
    joined = f"{objective}\n<assistant>\n{answer}"
    return {
        "lineage_token": _hmac(dedup_key, "AURA-RLC-LINEAGE-v1", lineage_root_sha256),
        "exact_token": _hmac(dedup_key, "AURA-RLC-EXACT-v1", content),
        "objective_token": _hmac(dedup_key, "AURA-RLC-OBJECTIVE-v1", content["objective"]),
        "answer_token": _hmac(dedup_key, "AURA-RLC-ANSWER-v1", content["answer"]),
        "objective_character_count": len(content["objective"]),
        "answer_character_count": len(content["answer"]),
        "token_shingles": _bottom_k_hmac(
            dedup_key,
            "AURA-RLC-TOKEN-SHINGLE-v1",
            _windows(_tokens(joined), _DEFAULT_TOKEN_SHINGLE_SIZE),
        ),
        "character_shingles": _bottom_k_hmac(
            dedup_key,
            "AURA-RLC-CHAR-SHINGLE-v1",
            _character_windows(joined, _DEFAULT_CHARACTER_SHINGLE_SIZE),
        ),
    }


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _assert_no_signature_overlap(
    projected: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    allow_same_lineage_same_split: bool = False,
    allow_same_corpus_near_duplicates: bool = False,
) -> None:
    """Use inverted shingle indexes instead of an all-pairs corpus scan."""

    records: list[Mapping[str, Any]] = []
    record_ids: set[str] = set()
    exact_index: dict[str, set[int]] = defaultdict(set)
    objective_index: dict[str, set[int]] = defaultdict(set)
    answer_index: dict[str, set[int]] = defaultdict(set)
    lineage_index: dict[str, set[int]] = defaultdict(set)
    token_index: dict[str, set[int]] = defaultdict(set)
    character_index: dict[str, set[int]] = defaultdict(set)

    def add(record: Mapping[str, Any]) -> None:
        def permitted(prior: Mapping[str, Any]) -> bool:
            return bool(
                allow_same_lineage_same_split
                and prior["lineage_token"] == record["lineage_token"]
                and prior["split"] == record["split"]
            )

        record_id = record["record_id_sha256"]
        if record_id in record_ids:
            _fail("verified_replay_sft_duplicate_record_identity")
        for prior_index in exact_index[record["exact_token"]]:
            if not permitted(records[prior_index]):
                _fail("verified_replay_sft_exact_content_overlap")
        for prior_index in objective_index[record["objective_token"]]:
            if not permitted(records[prior_index]):
                _fail("verified_replay_sft_exact_content_overlap")
        for prior_index in answer_index[record["answer_token"]]:
            prior = records[prior_index]
            if min(
                record["answer_character_count"],
                prior["answer_character_count"],
            ) >= 32 and not permitted(prior):
                _fail("verified_replay_sft_exact_content_overlap")
        for prior_index in lineage_index[record["lineage_token"]]:
            if records[prior_index]["split"] != record["split"]:
                _fail("verified_replay_sft_cross_corpus_lineage_overlap")

        possible: set[int] = set()
        for shingle in record["token_shingles"]:
            possible.update(token_index[shingle])
        for shingle in record["character_shingles"]:
            possible.update(character_index[shingle])
        for prior_index in possible:
            prior = records[prior_index]
            if (
                not permitted(prior)
                and not (
                    allow_same_corpus_near_duplicates
                    and prior["corpus"] == record["corpus"]
                )
                and (
                    _jaccard(
                        record["token_shingles"], prior["token_shingles"]
                    )
                    >= _TOKEN_NEAR_DUPLICATE_THRESHOLD
                    or _jaccard(
                        record["character_shingles"],
                        prior["character_shingles"],
                    )
                    >= _CHARACTER_NEAR_DUPLICATE_THRESHOLD
                )
            ):
                _fail("verified_replay_sft_semantic_near_duplicate")

        index = len(records)
        records.append(record)
        record_ids.add(record_id)
        exact_index[record["exact_token"]].add(index)
        objective_index[record["objective_token"]].add(index)
        answer_index[record["answer_token"]].add(index)
        lineage_index[record["lineage_token"]].add(index)
        for shingle in record["token_shingles"]:
            token_index[shingle].add(index)
        for shingle in record["character_shingles"]:
            character_index[shingle].add(index)

    for reference in references:
        add(reference)
    for candidate in projected:
        add(candidate)


def assert_semantic_signature_integrity(
    records: Sequence[Mapping[str, Any]],
    *,
    allow_same_lineage_same_split: bool = False,
    allow_same_corpus_near_duplicates: bool = False,
) -> None:
    """Validate exact, near-duplicate, and causal split integrity."""

    _assert_no_signature_overlap(
        records,
        (),
        allow_same_lineage_same_split=allow_same_lineage_same_split,
        allow_same_corpus_near_duplicates=allow_same_corpus_near_duplicates,
    )


def build_semantic_signature_records(
    *,
    dedup_key: bytes,
    records: Sequence[Mapping[str, str]],
    allow_same_lineage_same_split: bool = False,
    allow_same_corpus_near_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """Project plaintext surfaces into keyed signatures and discard plaintext."""

    key = _key(dedup_key, code="verified_replay_sft_dedup_key_invalid")
    projected: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != {
            "corpus",
            "split",
            "lineage_root_sha256",
            "objective",
            "answer",
        }:
            _fail("verified_replay_sft_reference_source_invalid")
        if (
            not isinstance(raw["corpus"], str)
            or not raw["corpus"]
            or raw["split"] not in {*SPLITS, "external_evaluation"}
            or not isinstance(raw["objective"], str)
            or not isinstance(raw["answer"], str)
        ):
            _fail("verified_replay_sft_reference_source_invalid")
        signature = _signature(
            objective=raw["objective"],
            answer=raw["answer"],
            lineage_root_sha256=_require_sha(
                raw["lineage_root_sha256"],
                code="verified_replay_sft_reference_lineage_invalid",
            ),
            dedup_key=key,
        )
        projected.append(
            {
                "record_id_sha256": _sha(
                    {
                        "ordinal": ordinal,
                        "corpus": raw["corpus"],
                        "split": raw["split"],
                        **signature,
                    }
                ),
                "corpus": raw["corpus"],
                "split": raw["split"],
                **signature,
            }
        )
    assert_semantic_signature_integrity(
        projected,
        allow_same_lineage_same_split=allow_same_lineage_same_split,
        allow_same_corpus_near_duplicates=allow_same_corpus_near_duplicates,
    )
    return projected


def _scan_text(objective: str, answer: str) -> None:
    for text in (objective, answer):
        _normalized_text(text)
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            _fail("verified_replay_sft_secret_detected")
        if any(pattern.search(text) for pattern in _PII_PATTERNS):
            _fail("verified_replay_sft_pii_detected")
        for match in _PAYMENT_CARD_RE.finditer(text):
            digits = [int(character) for character in match.group() if character.isdigit()]
            checksum = 0
            parity = len(digits) % 2
            for index, digit in enumerate(digits):
                value = digit * 2 if index % 2 == parity else digit
                checksum += value - 9 if value > 9 else value
            if 13 <= len(digits) <= 19 and checksum % 10 == 0:
                _fail("verified_replay_sft_pii_detected")
        if any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS):
            _fail("verified_replay_sft_prompt_injection_detected")
    if any(pattern.search(answer) for pattern in _HIDDEN_REASONING_PATTERNS):
        _fail("verified_replay_sft_hidden_reasoning_detected")


def projection_content_sha256(*, objective: str, answer: str) -> str:
    """Commit exactly the two model-visible replay fields."""

    _scan_text(objective, answer)
    return _sha(
        {
            "schema": VERIFIED_REPLAY_SFT_EXAMPLE_SCHEMA,
            "messages": [
                {"role": "user", "content": objective},
                {"role": "assistant", "content": answer},
            ],
            "tools": [],
        }
    )


def build_privacy_clearance(
    *,
    entry_sha256: str,
    experience_sha256: str,
    projection_sha256: str,
    origin_classification: str,
    consent_receipt_sha256: str,
    license_receipt_sha256: str,
    tenant_commitment_sha256: str,
    implementation_sha256: str,
    release_sha256: str,
) -> dict[str, Any]:
    """Build a committed local clearance; it is not an external attestation."""

    origin = origin_classification
    if origin == "synthetic_generated":
        contains_user_content = False
        consent_basis = "not_applicable_synthetic"
        if consent_receipt_sha256 != ZERO_SHA256:
            _fail("verified_replay_sft_synthetic_consent_invalid")
    elif origin == "user_content_explicit_opt_in":
        contains_user_content = True
        consent_basis = "explicit_training_opt_in"
        if consent_receipt_sha256 == ZERO_SHA256:
            _fail("verified_replay_sft_user_consent_missing")
    else:
        _fail("verified_replay_sft_origin_invalid")
    body = {
        "schema": VERIFIED_REPLAY_SFT_PRIVACY_CLEARANCE_SCHEMA,
        "entry_sha256": _require_sha(
            entry_sha256,
            code="verified_replay_sft_entry_sha256_invalid",
        ),
        "experience_sha256": _require_sha(
            experience_sha256,
            code="verified_replay_sft_experience_sha256_invalid",
        ),
        "projection_content_sha256": _require_sha(
            projection_sha256,
            code="verified_replay_sft_projection_sha256_invalid",
        ),
        "origin_classification": origin,
        "contains_user_content": contains_user_content,
        "consent_basis": consent_basis,
        "consent_receipt_sha256": _require_sha(
            consent_receipt_sha256,
            code="verified_replay_sft_consent_receipt_invalid",
        ),
        "license_basis": "owner_authorized_local_training",
        "license_receipt_sha256": _require_sha(
            license_receipt_sha256,
            code="verified_replay_sft_license_receipt_invalid",
        ),
        "tenant_scope": "local_single_owner",
        "tenant_commitment_sha256": _require_sha(
            tenant_commitment_sha256,
            code="verified_replay_sft_tenant_commitment_invalid",
        ),
        "retention_active": True,
        "revoked": False,
        "deleted": False,
        "pii_findings": 0,
        "secret_findings": 0,
        "user_secret_findings": 0,
        "hidden_reasoning_findings": 0,
        "implementation_sha256": _require_sha(
            implementation_sha256,
            code="verified_replay_sft_privacy_implementation_invalid",
        ),
        "release_sha256": _require_sha(
            release_sha256,
            code="verified_replay_sft_privacy_release_invalid",
        ),
        "status": "passed_projection_only_no_training_authority",
    }
    return {**body, "clearance_sha256": _sha(body)}


def validate_privacy_clearance(
    value: Any,
    *,
    entry_sha256: str,
    experience_sha256: str,
    projection_sha256: str,
) -> dict[str, Any]:
    clearance = _committed(
        value,
        fields=_PRIVACY_FIELDS,
        digest_field="clearance_sha256",
        code="verified_replay_sft_privacy_clearance_invalid",
    )
    origin = clearance.get("origin_classification")
    expected_user_content = origin == "user_content_explicit_opt_in"
    expected_consent = (
        "explicit_training_opt_in"
        if expected_user_content
        else "not_applicable_synthetic"
    )
    consent_receipt = clearance.get("consent_receipt_sha256")
    count_fields = (
        "pii_findings",
        "secret_findings",
        "user_secret_findings",
        "hidden_reasoning_findings",
    )
    if (
        origin not in {"synthetic_generated", "user_content_explicit_opt_in"}
        or clearance.get("entry_sha256") != entry_sha256
        or clearance.get("experience_sha256") != experience_sha256
        or clearance.get("projection_content_sha256") != projection_sha256
        or clearance.get("contains_user_content") is not expected_user_content
        or clearance.get("consent_basis") != expected_consent
        or not _is_sha256(consent_receipt)
        or (expected_user_content and consent_receipt == ZERO_SHA256)
        or (not expected_user_content and consent_receipt != ZERO_SHA256)
        or clearance.get("license_basis") != "owner_authorized_local_training"
        or not _is_sha256(clearance.get("license_receipt_sha256"))
        or clearance.get("license_receipt_sha256") == ZERO_SHA256
        or clearance.get("tenant_scope") != "local_single_owner"
        or not _is_sha256(clearance.get("tenant_commitment_sha256"))
        or clearance.get("tenant_commitment_sha256") == ZERO_SHA256
        or clearance.get("retention_active") is not True
        or clearance.get("revoked") is not False
        or clearance.get("deleted") is not False
        or any(clearance.get(field) != 0 for field in count_fields)
        or not _is_sha256(clearance.get("implementation_sha256"))
        or not _is_sha256(clearance.get("release_sha256"))
        or clearance.get("status") != "passed_projection_only_no_training_authority"
    ):
        _fail("verified_replay_sft_privacy_clearance_failed")
    return clearance


def empty_reference_index(*, dedup_key: bytes) -> dict[str, Any]:
    """Return a committed empty index for isolated/local falsification only."""

    key = _key(dedup_key, code="verified_replay_sft_dedup_key_invalid")
    body = {
        "schema": VERIFIED_REPLAY_SFT_REFERENCE_INDEX_SCHEMA,
        "dedup_key_commitment_sha256": hashlib.sha256(key).hexdigest(),
        "records": [],
        "record_count": 0,
        "coverage": [],
        "scope": "empty_local_falsification_only",
    }
    return {**body, "index_sha256": _sha(body)}


def validate_reference_index(value: Any, *, dedup_key: bytes) -> dict[str, Any]:
    key = _key(dedup_key, code="verified_replay_sft_dedup_key_invalid")
    fields = {
        "schema",
        "dedup_key_commitment_sha256",
        "records",
        "record_count",
        "coverage",
        "scope",
        "index_sha256",
    }
    raw_records = value.get("records") if isinstance(value, Mapping) else None
    if not isinstance(raw_records, list) or len(raw_records) > _MAX_REFERENCE_RECORDS:
        _fail("verified_replay_sft_reference_index_invalid")
    index = _committed(
        value,
        fields=fields,
        digest_field="index_sha256",
        code="verified_replay_sft_reference_index_invalid",
    )
    records = index.get("records")
    coverage = index.get("coverage")
    if (
        index.get("schema") != VERIFIED_REPLAY_SFT_REFERENCE_INDEX_SCHEMA
        or index.get("dedup_key_commitment_sha256") != hashlib.sha256(key).hexdigest()
        or not isinstance(records, list)
        or len(records) > _MAX_REFERENCE_RECORDS
        or index.get("record_count") != len(records)
        or not isinstance(coverage, list)
        or any(not isinstance(item, str) or not item for item in coverage)
        or len(set(coverage)) != len(coverage)
        or index.get("scope")
        not in {
            "empty_local_falsification_only",
            "sealed_multisurface_external_corpora",
        }
    ):
        _fail("verified_replay_sft_reference_index_invalid")
    if records and index.get("scope") != "sealed_multisurface_external_corpora":
        _fail("verified_replay_sft_reference_index_scope_invalid")
    seen_ids: set[str] = set()
    normalized_records: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != _REFERENCE_RECORD_FIELDS:
            _fail("verified_replay_sft_reference_record_invalid")
        record = json.loads(canonical_json_bytes(raw))
        scalar_hashes = (
            "record_id_sha256",
            "lineage_token",
            "exact_token",
            "objective_token",
            "answer_token",
        )
        token_shingles = record.get("token_shingles")
        character_shingles = record.get("character_shingles")
        if (
            any(not _is_sha256(record.get(field)) for field in scalar_hashes)
            or record["record_id_sha256"] in seen_ids
            or not isinstance(record.get("corpus"), str)
            or not record["corpus"]
            or record.get("split") not in {*SPLITS, "external_evaluation"}
            or type(record.get("objective_character_count")) is not int
            or not 1 <= record["objective_character_count"] <= _MAX_TEXT_CHARS
            or type(record.get("answer_character_count")) is not int
            or not 1 <= record["answer_character_count"] <= _MAX_TEXT_CHARS
            or not isinstance(token_shingles, list)
            or not isinstance(character_shingles, list)
            or len(token_shingles) > _MAX_SIGNATURES_PER_RECORD
            or len(character_shingles) > _MAX_SIGNATURES_PER_RECORD
            or token_shingles != sorted(set(token_shingles))
            or character_shingles != sorted(set(character_shingles))
            or any(not _is_sha256(item) for item in token_shingles)
            or any(not _is_sha256(item) for item in character_shingles)
        ):
            _fail("verified_replay_sft_reference_record_invalid")
        seen_ids.add(record["record_id_sha256"])
        normalized_records.append(record)
    _assert_no_signature_overlap(normalized_records, ())
    return {**index, "records": normalized_records}


def build_reference_index(
    *,
    dedup_key: bytes,
    records: Sequence[Mapping[str, str]],
    coverage: Sequence[str],
) -> dict[str, Any]:
    """Build a keyed external index without retaining reference plaintext."""

    key = _key(dedup_key, code="verified_replay_sft_dedup_key_invalid")
    projected = build_semantic_signature_records(
        dedup_key=key,
        records=records,
    )
    body = {
        "schema": VERIFIED_REPLAY_SFT_REFERENCE_INDEX_SCHEMA,
        "dedup_key_commitment_sha256": hashlib.sha256(key).hexdigest(),
        "records": projected,
        "record_count": len(projected),
        "coverage": sorted(set(coverage)),
        "scope": "sealed_multisurface_external_corpora",
    }
    return validate_reference_index(
        {**body, "index_sha256": _sha(body)},
        dedup_key=key,
    )


def _lineage_root(entry: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    provenance = payload["provenance"]
    return _sha(
        {
            "domain": "AURA-RLC-VERIFIED-REPLAY-LINEAGE-v1",
            "request_id": provenance["request_id"],
            "objective_sha256": provenance["objective_sha256"],
            "checkpoint_fingerprint": provenance["checkpoint_fingerprint"],
            "source_experience_sha256": entry["experience_sha256"],
        }
    )


def _split(lineage_root_sha256: str, partition_key: bytes, ratios: Mapping[str, int]) -> str:
    value = int(
        _hmac(partition_key, "AURA-RLC-PRE-AUGMENTATION-SPLIT-v1", lineage_root_sha256)[:16],
        16,
    ) % 10_000
    cursor = 0
    for split in SPLITS:
        cursor += ratios[split]
        if value < cursor:
            return split
    _fail("verified_replay_sft_partition_unreachable")


def _ratios(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        _fail("verified_replay_sft_partition_ratios_invalid")
    ratios = dict(value)
    if (
        any(type(ratios[split]) is not int or ratios[split] <= 0 for split in SPLITS)
        or sum(ratios.values()) != 10_000
    ):
        _fail("verified_replay_sft_partition_ratios_invalid")
    return ratios


def _trainer_row(
    *,
    entry: Mapping[str, Any],
    payload: Mapping[str, Any],
    split: str,
    lineage_root_sha256: str,
    projection_sha256: str,
    clearance: Mapping[str, Any],
) -> dict[str, Any]:
    objective = payload["task_context"]["objective"]
    answer = payload["verified_solution"]["text"]
    body = {
        "schema": VERIFIED_REPLAY_SFT_EXAMPLE_SCHEMA,
        "messages": [
            {"role": "user", "content": objective},
            {"role": "assistant", "content": answer},
        ],
        "tools": [],
        "_meta": {
            "source_kind": "encrypted_causally_verified_replay",
            "source_entry_sha256": entry["entry_sha256"],
            "source_experience_sha256": entry["experience_sha256"],
            "lineage_root_sha256": lineage_root_sha256,
            "split": split,
            "projection_content_sha256": projection_sha256,
            "privacy_clearance_sha256": clearance["clearance_sha256"],
            "error_class": payload["error_class"],
            "verifier": payload["discriminating_test"]["verifier"],
            "loss_policy": {
                "mask_prompt": True,
                "supervised_region": "final_assistant_message_only",
            },
            "training_authority": "none_quarantined_projection",
        },
    }
    return {**body, "example_sha256": _sha(body)}


def _assert_trainer_surface(row: Mapping[str, Any]) -> None:
    if set(row) != {"schema", "messages", "tools", "_meta", "example_sha256"}:
        _fail("verified_replay_sft_trainer_surface_invalid")
    rendered = canonical_json_bytes(row)
    if any(f'"{field}"'.encode("ascii") in rendered for field in _PRIVATE_FIELD_NAMES):
        _fail("verified_replay_sft_private_field_leak")
    if row.get("tools") != []:
        _fail("verified_replay_sft_unverified_tool_trace")
    meta = row.get("_meta")
    if (
        not isinstance(meta, Mapping)
        or set(meta)
        != {
            "source_kind",
            "source_entry_sha256",
            "source_experience_sha256",
            "lineage_root_sha256",
            "split",
            "projection_content_sha256",
            "privacy_clearance_sha256",
            "error_class",
            "verifier",
            "loss_policy",
            "training_authority",
        }
        or meta.get("source_kind") != "encrypted_causally_verified_replay"
        or any(
            not _is_sha256(meta.get(field))
            for field in (
                "source_entry_sha256",
                "source_experience_sha256",
                "lineage_root_sha256",
                "projection_content_sha256",
                "privacy_clearance_sha256",
            )
        )
        or meta.get("split") not in SPLITS
        or not isinstance(meta.get("error_class"), str)
        or not meta["error_class"]
        or not isinstance(meta.get("verifier"), str)
        or not meta["verifier"]
        or meta.get("loss_policy")
        != {
            "mask_prompt": True,
            "supervised_region": "final_assistant_message_only",
        }
        or meta.get("training_authority") != "none_quarantined_projection"
    ):
        _fail("verified_replay_sft_metadata_surface_invalid")
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [message.get("role") for message in messages] != ["user", "assistant"]
        or any(set(message) != {"role", "content"} for message in messages)
    ):
        _fail("verified_replay_sft_message_surface_invalid")
    _scan_text(messages[0]["content"], messages[1]["content"])
    body = dict(row)
    digest = body.pop("example_sha256", None)
    if not _is_sha256(digest) or _sha(body) != digest:
        _fail("verified_replay_sft_example_commitment_invalid")


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _binding(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _json_document(payload: bytes, *, code: str) -> dict[str, Any]:
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAX_ARTIFACT_BYTES
    ):
        _fail(code)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise VerifiedReplaySFTError(code) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        _fail(code)
    return value


def _jsonl_rows(payload: bytes, *, code: str) -> list[dict[str, Any]]:
    if not isinstance(payload, bytes) or not payload or not payload.endswith(b"\n"):
        _fail(code)
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        row = _json_document(line, code=code)
        _assert_trainer_surface(row)
        rows.append(row)
    if _jsonl(rows) != payload:
        _fail(code)
    return rows


def validate_verified_replay_sft_candidate_artifacts(
    artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate trainer-visible artifacts without opening evaluator custody."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        VERIFIED_REPLAY_SFT_CANDIDATE_FILES
    ):
        _fail("verified_replay_sft_candidate_file_set_invalid")
    manifest = _json_document(
        artifacts[_CANDIDATE_MANIFEST_FILE],
        code="verified_replay_sft_candidate_manifest_invalid",
    )
    fields = {
        "schema",
        "source_store_sha256",
        "custody_root_sha256",
        "artifacts",
        "row_counts",
        "partition_manifest_sha256",
        "semantic_dedup_manifest_sha256",
        "privacy_manifest_sha256",
        "holdout_artifact_sha256",
        "trainer_contract",
        "trainer_ready",
        "training_authority",
        "required_next_gates",
        "candidate_package_sha256",
    }
    body = dict(manifest)
    digest = body.pop("candidate_package_sha256", None)
    required_gates = [
        "external_privacy_attestation",
        "external_multisurface_contamination_audit",
        "resident_tokenizer_projection_validation",
        "externally_rooted_trainer_admission",
        "small_checkpoint_transfer_falsification",
        "independent_resident_promotion",
    ]
    expected_bindings = {
        name: _binding(artifacts[name])
        for name in (_CANDIDATE_TRAIN_FILE, _CANDIDATE_VALIDATION_FILE)
    }
    train_rows = _jsonl_rows(
        artifacts[_CANDIDATE_TRAIN_FILE],
        code="verified_replay_sft_candidate_train_invalid",
    )
    validation_rows = _jsonl_rows(
        artifacts[_CANDIDATE_VALIDATION_FILE],
        code="verified_replay_sft_candidate_validation_invalid",
    )
    trainer_contract = manifest.get("trainer_contract")
    if (
        set(manifest) != fields
        or manifest.get("schema") != VERIFIED_REPLAY_SFT_CANDIDATE_SCHEMA
        or not _is_sha256(digest)
        or _sha(body) != digest
        or not _is_sha256(manifest.get("source_store_sha256"))
        or not _is_sha256(manifest.get("custody_root_sha256"))
        or manifest.get("artifacts") != expected_bindings
        or manifest.get("row_counts")
        != {TRAIN_SPLIT: len(train_rows), VALIDATION_SPLIT: len(validation_rows)}
        or any(
            not _is_sha256(manifest.get(field))
            for field in (
                "partition_manifest_sha256",
                "semantic_dedup_manifest_sha256",
                "privacy_manifest_sha256",
                "holdout_artifact_sha256",
            )
        )
        or not isinstance(trainer_contract, Mapping)
        or set(trainer_contract)
        != {
            "trainer",
            "mask_prompt",
            "supervised_region",
            "max_seq_length",
            "truncation_allowed",
        }
        or trainer_contract.get("trainer") != "mlx_lm.ChatDataset"
        or trainer_contract.get("mask_prompt") is not True
        or trainer_contract.get("supervised_region")
        != "final_assistant_message_only"
        or type(trainer_contract.get("max_seq_length")) is not int
        or not 256 <= trainer_contract["max_seq_length"] <= _MAX_TEXT_CHARS
        or trainer_contract.get("truncation_allowed") is not False
        or manifest.get("trainer_ready") is not False
        or manifest.get("training_authority") != "none_quarantined_projection"
        or manifest.get("required_next_gates") != required_gates
    ):
        _fail("verified_replay_sft_candidate_manifest_invalid")
    rows = [*train_rows, *validation_rows]
    if (
        any(row["_meta"]["split"] != TRAIN_SPLIT for row in train_rows)
        or any(row["_meta"]["split"] != VALIDATION_SPLIT for row in validation_rows)
        or len({row["example_sha256"] for row in rows}) != len(rows)
        or len({row["_meta"]["lineage_root_sha256"] for row in rows}) != len(rows)
    ):
        _fail("verified_replay_sft_candidate_rows_invalid")
    return {"manifest": manifest, "train_rows": train_rows, "validation_rows": validation_rows}


def _token_sequence(value: Any) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        _fail("verified_replay_sft_token_sequence_invalid")
    return list(value)


def validate_verified_replay_sft_tokenization(
    candidate_artifacts: Mapping[str, bytes],
    *,
    tokenizer: Any,
    chat_dataset_process: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Prove exact ChatDataset masking for every trainer-visible replay row."""

    candidate = validate_verified_replay_sft_candidate_artifacts(candidate_artifacts)
    manifest = candidate["manifest"]
    contract = manifest["trainer_contract"]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        _fail("verified_replay_sft_tokenizer_template_missing")
    receipts: list[dict[str, Any]] = []
    group_stats: dict[str, dict[str, int]] = {}
    rows_by_split = {
        TRAIN_SPLIT: candidate["train_rows"],
        VALIDATION_SPLIT: candidate["validation_rows"],
    }
    for split, rows in rows_by_split.items():
        for row in rows:
            messages = row["messages"]
            tools = row["tools"]
            full = _token_sequence(
                apply_template(messages, tools=tools, return_dict=False)
            )
            prefix = _token_sequence(
                apply_template(
                    messages[:-1],
                    tools=tools,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            )
            if len(prefix) >= len(full) or full[: len(prefix)] != prefix:
                _fail("verified_replay_sft_masked_prefix_not_exact")
            processed = chat_dataset_process(row)
            if (
                not isinstance(processed, tuple)
                or len(processed) != 2
                or type(processed[1]) is not int
                or _token_sequence(processed[0]) != full
                or processed[1] != len(prefix)
            ):
                _fail("verified_replay_sft_chat_dataset_projection_mismatch")
            if len(full) > contract["max_seq_length"]:
                _fail("verified_replay_sft_sequence_would_truncate")
            target = full[len(prefix) :]
            error_class = row["_meta"]["error_class"]
            stats = group_stats.setdefault(
                error_class,
                {
                    "examples": 0,
                    "min_full_tokens": len(full),
                    "max_full_tokens": len(full),
                    "min_prefix_tokens": len(prefix),
                    "max_prefix_tokens": len(prefix),
                    "min_target_tokens": len(target),
                    "max_target_tokens": len(target),
                },
            )
            stats["examples"] += 1
            for name, size in (
                ("full", len(full)),
                ("prefix", len(prefix)),
                ("target", len(target)),
            ):
                stats[f"min_{name}_tokens"] = min(stats[f"min_{name}_tokens"], size)
                stats[f"max_{name}_tokens"] = max(stats[f"max_{name}_tokens"], size)
            receipts.append(
                {
                    "example_sha256": row["example_sha256"],
                    "lineage_root_sha256": row["_meta"]["lineage_root_sha256"],
                    "split": split,
                    "error_class": error_class,
                    "full_tokens_sha256": _sha(full),
                    "prefix_tokens_sha256": _sha(prefix),
                    "target_tokens_sha256": _sha(target),
                    "full_token_count": len(full),
                    "masked_prefix_token_count": len(prefix),
                    "supervised_target_token_count": len(target),
                    "target_start_index": len(prefix),
                    "prefix_exact": True,
                    "chat_dataset_process_exact": True,
                    "within_max_seq_length": True,
                }
            )
    body = {
        "schema": VERIFIED_REPLAY_SFT_TOKENIZATION_SCHEMA,
        "candidate_package_sha256": manifest["candidate_package_sha256"],
        "custody_root_sha256": manifest["custody_root_sha256"],
        "source_store_sha256": manifest["source_store_sha256"],
        "partition_manifest_sha256": manifest["partition_manifest_sha256"],
        "privacy_manifest_sha256": manifest["privacy_manifest_sha256"],
        "trainer_contract": dict(contract),
        "tokenization_scope": "candidate_train_validation_only",
        "rows_checked": len(receipts),
        "expected_rows": sum(manifest["row_counts"].values()),
        "rows_with_truncation": 0,
        "chat_dataset_process_mismatches": 0,
        "holdout_tokenized": False,
        "groups": {name: group_stats[name] for name in sorted(group_stats)},
        "projection_receipts_sha256": _sha(receipts),
        "status": "passed_exact_masked_prefix",
    }
    if body["rows_checked"] != body["expected_rows"] or body["rows_checked"] < 1:
        _fail("verified_replay_sft_tokenization_coverage_invalid")
    return {**body, "report_sha256": _sha(body)}


def validate_verified_replay_sft_custody_pair(
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate candidate/evaluator linkage and sealed holdout separation."""

    candidate = validate_verified_replay_sft_candidate_artifacts(candidate_artifacts)
    if not isinstance(evaluator_artifacts, Mapping) or set(evaluator_artifacts) != set(
        VERIFIED_REPLAY_SFT_EVALUATOR_FILES
    ):
        _fail("verified_replay_sft_evaluator_file_set_invalid")
    holdout_bytes = evaluator_artifacts[_EVALUATOR_HOLDOUT_FILE]
    holdout = _json_document(
        holdout_bytes,
        code="verified_replay_sft_holdout_invalid",
    )
    evaluator = _json_document(
        evaluator_artifacts[_EVALUATOR_MANIFEST_FILE],
        code="verified_replay_sft_evaluator_manifest_invalid",
    )
    evaluator_fields = {
        "schema",
        "source_store_sha256",
        "custody_root_sha256",
        "candidate_package_sha256",
        "artifact",
        "holdout_row_count",
        "trainer_access",
        "training_authority",
        "evaluator_package_sha256",
    }
    evaluator_body = dict(evaluator)
    evaluator_digest = evaluator_body.pop("evaluator_package_sha256", None)
    holdout_fields = {
        "schema",
        "source_store_sha256",
        "partition_manifest",
        "semantic_dedup_manifest",
        "privacy_manifest",
        "examples",
    }
    examples = holdout.get("examples")
    if not isinstance(examples, list):
        _fail("verified_replay_sft_holdout_invalid")
    for row in examples:
        _assert_trainer_surface(row)
    partition = holdout.get("partition_manifest")
    dedup = holdout.get("semantic_dedup_manifest")
    privacy = holdout.get("privacy_manifest")
    manifests = (
        (partition, VERIFIED_REPLAY_SFT_PARTITION_SCHEMA),
        (dedup, VERIFIED_REPLAY_SFT_DEDUP_SCHEMA),
        (privacy, VERIFIED_REPLAY_SFT_PRIVACY_MANIFEST_SCHEMA),
    )
    for manifest, schema in manifests:
        if not isinstance(manifest, Mapping) or manifest.get("schema") != schema:
            _fail("verified_replay_sft_evaluator_embedded_manifest_invalid")
        manifest_body = dict(manifest)
        manifest_digest = manifest_body.pop("manifest_sha256", None)
        if not _is_sha256(manifest_digest) or _sha(manifest_body) != manifest_digest:
            _fail("verified_replay_sft_evaluator_embedded_manifest_invalid")
    partition_fields = {
        "schema",
        "source_store_sha256",
        "source_store_revision",
        "partition_key_commitment_sha256",
        "partition_ratios_basis_points",
        "assignment_domain",
        "records",
        "record_count",
        "augmentation_generation",
        "future_derivatives_inherit_lineage_split",
        "status",
        "manifest_sha256",
    }
    dedup_fields = {
        "schema",
        "source_store_sha256",
        "partition_manifest_sha256",
        "dedup_key_commitment_sha256",
        "reference_index_sha256",
        "reference_scope",
        "reference_coverage",
        "projected_record_count",
        "reference_record_count",
        "methods",
        "token_shingle_size",
        "character_shingle_size",
        "shingle_sketch_limit",
        "token_near_duplicate_threshold",
        "character_near_duplicate_threshold",
        "exact_overlap_count",
        "near_duplicate_overlap_count",
        "lineage_overlap_count",
        "status",
        "manifest_sha256",
    }
    privacy_fields = {
        "schema",
        "source_store_sha256",
        "records",
        "record_count",
        "local_secret_scan_findings",
        "local_pii_scan_findings",
        "local_hidden_reasoning_findings",
        "local_prompt_injection_findings",
        "projected_private_field_count",
        "projected_tool_trace_count",
        "external_attestation_required",
        "status",
        "manifest_sha256",
    }
    candidate_manifest = candidate["manifest"]
    artifact = evaluator.get("artifact")
    if (
        set(evaluator) != evaluator_fields
        or evaluator.get("schema") != VERIFIED_REPLAY_SFT_EVALUATOR_SCHEMA
        or not _is_sha256(evaluator_digest)
        or _sha(evaluator_body) != evaluator_digest
        or evaluator.get("source_store_sha256")
        != candidate_manifest["source_store_sha256"]
        or evaluator.get("custody_root_sha256")
        != candidate_manifest["custody_root_sha256"]
        or evaluator.get("candidate_package_sha256")
        != candidate_manifest["candidate_package_sha256"]
        or artifact
        != {"filename": _EVALUATOR_HOLDOUT_FILE, **_binding(holdout_bytes)}
        or candidate_manifest.get("holdout_artifact_sha256")
        != hashlib.sha256(holdout_bytes).hexdigest()
        or evaluator.get("holdout_row_count") != len(examples)
        or evaluator.get("trainer_access") != "forbidden_separate_artifact_root"
        or evaluator.get("training_authority") != "none_quarantined_projection"
        or set(holdout) != holdout_fields
        or set(partition) != partition_fields
        or set(dedup) != dedup_fields
        or set(privacy) != privacy_fields
        or holdout.get("schema") != "aura.rlc.verified_replay_sft_holdout.v1"
        or holdout.get("source_store_sha256")
        != candidate_manifest["source_store_sha256"]
        or partition.get("manifest_sha256")
        != candidate_manifest["partition_manifest_sha256"]
        or dedup.get("manifest_sha256")
        != candidate_manifest["semantic_dedup_manifest_sha256"]
        or privacy.get("manifest_sha256")
        != candidate_manifest["privacy_manifest_sha256"]
        or any(row["_meta"]["split"] != HOLDOUT_SPLIT for row in examples)
    ):
        _fail("verified_replay_sft_evaluator_manifest_invalid")
    all_rows = [*candidate["train_rows"], *candidate["validation_rows"], *examples]
    expected_custody_root = _sha(
        {
            "domain": "AURA-RLC-VERIFIED-REPLAY-SFT-CUSTODY-v1",
            "source_store_sha256": candidate_manifest["source_store_sha256"],
            "visible_artifacts": candidate_manifest["artifacts"],
            "holdout_artifact": _binding(holdout_bytes),
            "partition_manifest_sha256": partition["manifest_sha256"],
            "semantic_dedup_manifest_sha256": dedup["manifest_sha256"],
            "privacy_manifest_sha256": privacy["manifest_sha256"],
        }
    )
    partition_records = partition.get("records")
    privacy_records = privacy.get("records")
    if (
        len({row["example_sha256"] for row in all_rows}) != len(all_rows)
        or len({row["_meta"]["lineage_root_sha256"] for row in all_rows})
        != len(all_rows)
        or candidate_manifest.get("custody_root_sha256") != expected_custody_root
        or evaluator.get("custody_root_sha256") != expected_custody_root
        or not isinstance(partition_records, list)
        or not isinstance(privacy_records, list)
        or partition.get("record_count") != len(all_rows)
        or privacy.get("record_count") != len(all_rows)
        or partition.get("source_store_sha256")
        != candidate_manifest["source_store_sha256"]
        or partition.get("source_store_revision") is None
        or type(partition.get("source_store_revision")) is not int
        or partition["source_store_revision"] < 0
        or not _is_sha256(partition.get("partition_key_commitment_sha256"))
        or _ratios(partition.get("partition_ratios_basis_points"))
        != partition["partition_ratios_basis_points"]
        or partition.get("assignment_domain")
        != "AURA-RLC-PRE-AUGMENTATION-SPLIT-v1"
        or partition.get("augmentation_generation") != 0
        or partition.get("future_derivatives_inherit_lineage_split") is not True
        or partition.get("status")
        != "sealed_before_augmentation_no_training_authority"
        or dedup.get("source_store_sha256")
        != candidate_manifest["source_store_sha256"]
        or dedup.get("partition_manifest_sha256") != partition["manifest_sha256"]
        or not _is_sha256(dedup.get("dedup_key_commitment_sha256"))
        or not _is_sha256(dedup.get("reference_index_sha256"))
        or dedup.get("reference_scope")
        not in {
            "empty_local_falsification_only",
            "sealed_multisurface_external_corpora",
        }
        or not isinstance(dedup.get("reference_coverage"), list)
        or type(dedup.get("reference_record_count")) is not int
        or dedup["reference_record_count"] < 0
        or dedup.get("projected_record_count") != len(all_rows)
        or dedup.get("methods")
        != [
            "keyed_exact_normalized_content",
            "keyed_objective_and_answer",
            "keyed_bottom_k_token_shingle_jaccard",
            "keyed_bottom_k_character_shingle_jaccard",
            "causal_lineage_nonoverlap",
        ]
        or dedup.get("token_shingle_size") != _DEFAULT_TOKEN_SHINGLE_SIZE
        or dedup.get("character_shingle_size")
        != _DEFAULT_CHARACTER_SHINGLE_SIZE
        or dedup.get("shingle_sketch_limit") != _SHINGLE_SKETCH_LIMIT
        or dedup.get("token_near_duplicate_threshold")
        != _TOKEN_NEAR_DUPLICATE_THRESHOLD
        or dedup.get("character_near_duplicate_threshold")
        != _CHARACTER_NEAR_DUPLICATE_THRESHOLD
        or dedup.get("exact_overlap_count") != 0
        or dedup.get("near_duplicate_overlap_count") != 0
        or dedup.get("lineage_overlap_count") != 0
        or dedup.get("status") != "passed_local_keyed_projection_quarantine"
        or privacy.get("source_store_sha256")
        != candidate_manifest["source_store_sha256"]
        or privacy.get("projected_private_field_count") != 0
        or privacy.get("projected_tool_trace_count") != 0
        or any(
            privacy.get(field) != 0
            for field in (
                "local_secret_scan_findings",
                "local_pii_scan_findings",
                "local_hidden_reasoning_findings",
                "local_prompt_injection_findings",
            )
        )
        or privacy.get("external_attestation_required") is not True
        or privacy.get("status") != "passed_local_projection_quarantine_only"
    ):
        _fail("verified_replay_sft_custody_overlap")
    partition_by_example: dict[str, Mapping[str, Any]] = {}
    for record in partition_records:
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "source_entry_sha256",
                "source_experience_sha256",
                "lineage_root_sha256",
                "split",
                "example_sha256",
                "augmentation_generation",
                "augmentation_parent_sha256",
            }
            or not _is_sha256(record.get("example_sha256"))
            or any(
                not _is_sha256(record.get(field))
                for field in (
                    "source_entry_sha256",
                    "source_experience_sha256",
                    "lineage_root_sha256",
                )
            )
            or record["example_sha256"] in partition_by_example
            or record.get("augmentation_generation") != 0
            or record.get("augmentation_parent_sha256") != ZERO_SHA256
        ):
            _fail("verified_replay_sft_partition_record_invalid")
        partition_by_example[record["example_sha256"]] = record
    privacy_by_entry: dict[str, Mapping[str, Any]] = {}
    for record in privacy_records:
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "source_entry_sha256",
                "projection_content_sha256",
                "privacy_clearance_sha256",
                "origin_classification",
            }
            or not _is_sha256(record.get("source_entry_sha256"))
            or not _is_sha256(record.get("projection_content_sha256"))
            or not _is_sha256(record.get("privacy_clearance_sha256"))
            or record["source_entry_sha256"] in privacy_by_entry
            or record.get("origin_classification")
            not in {"synthetic_generated", "user_content_explicit_opt_in"}
        ):
            _fail("verified_replay_sft_privacy_record_invalid")
        privacy_by_entry[record["source_entry_sha256"]] = record
    for row in all_rows:
        meta = row["_meta"]
        partition_record = partition_by_example.get(row["example_sha256"])
        privacy_record = privacy_by_entry.get(meta["source_entry_sha256"])
        if (
            partition_record is None
            or privacy_record is None
            or partition_record.get("source_entry_sha256")
            != meta["source_entry_sha256"]
            or partition_record.get("source_experience_sha256")
            != meta["source_experience_sha256"]
            or partition_record.get("lineage_root_sha256")
            != meta["lineage_root_sha256"]
            or partition_record.get("split") != meta["split"]
            or privacy_record.get("projection_content_sha256")
            != meta["projection_content_sha256"]
            or privacy_record.get("privacy_clearance_sha256")
            != meta["privacy_clearance_sha256"]
        ):
            _fail("verified_replay_sft_manifest_row_binding_invalid")
    return {
        "candidate_manifest": candidate_manifest,
        "evaluator_manifest": evaluator,
        "partition_manifest": partition,
        "semantic_dedup_manifest": dedup,
        "privacy_manifest": privacy,
        "visible_row_count": len(candidate["train_rows"])
        + len(candidate["validation_rows"]),
        "holdout_row_count": len(examples),
    }


@dataclass(frozen=True, slots=True)
class VerifiedReplaySFTCustodyBundles:
    """Physically separable candidate/evaluator projection artifacts."""

    candidate_artifacts: dict[str, bytes]
    evaluator_artifacts: dict[str, bytes]
    custody_report: dict[str, Any]


def build_verified_replay_sft_custody_bundles(
    *,
    replay_store: Mapping[str, Any],
    protector: ReplayProtector,
    privacy_clearances: Mapping[str, Mapping[str, Any]],
    partition_key: bytes,
    dedup_key: bytes,
    reference_index: Mapping[str, Any],
    partition_ratios: Mapping[str, int] | None = None,
    minimum_rows_per_split: int = 1,
    max_seq_length: int = _DEFAULT_MAX_SEQ_LENGTH,
) -> VerifiedReplaySFTCustodyBundles:
    """Project authenticated replay into quarantined split custody bundles."""

    store = validate_verified_replay_store(replay_store)
    partition_secret = _key(
        partition_key,
        code="verified_replay_sft_partition_key_invalid",
    )
    dedup_secret = _key(dedup_key, code="verified_replay_sft_dedup_key_invalid")
    ratios = _ratios(
        partition_ratios
        if partition_ratios is not None
        else {TRAIN_SPLIT: 8_000, VALIDATION_SPLIT: 1_000, HOLDOUT_SPLIT: 1_000}
    )
    if (
        type(minimum_rows_per_split) is not int
        or minimum_rows_per_split < 1
        or minimum_rows_per_split > 10_000
    ):
        _fail("verified_replay_sft_minimum_split_rows_invalid")
    if type(max_seq_length) is not int or not 256 <= max_seq_length <= _MAX_TEXT_CHARS:
        _fail("verified_replay_sft_max_seq_length_invalid")
    if not isinstance(privacy_clearances, Mapping):
        _fail("verified_replay_sft_privacy_inventory_invalid")
    references = validate_reference_index(reference_index, dedup_key=dedup_secret)

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    signatures: list[dict[str, Any]] = []
    partition_records: list[dict[str, Any]] = []
    privacy_records: list[dict[str, Any]] = []
    seen_entries: set[str] = set()
    for entry in store["entries"]:
        payload = materialize_verified_replay_entry(entry, protector=protector)
        objective = payload["task_context"]["objective"]
        answer = payload["verified_solution"]["text"]
        projection_sha = projection_content_sha256(objective=objective, answer=answer)
        clearance_raw = privacy_clearances.get(entry["entry_sha256"])
        clearance = validate_privacy_clearance(
            clearance_raw,
            entry_sha256=entry["entry_sha256"],
            experience_sha256=entry["experience_sha256"],
            projection_sha256=projection_sha,
        )
        lineage_root = _lineage_root(entry, payload)
        split = _split(lineage_root, partition_secret, ratios)
        row = _trainer_row(
            entry=entry,
            payload=payload,
            split=split,
            lineage_root_sha256=lineage_root,
            projection_sha256=projection_sha,
            clearance=clearance,
        )
        _assert_trainer_surface(row)
        signature = _signature(
            objective=objective,
            answer=answer,
            lineage_root_sha256=lineage_root,
            dedup_key=dedup_secret,
        )
        signatures.append(
            {
                "record_id_sha256": row["example_sha256"],
                "corpus": "verified_replay_projection",
                "split": split,
                **signature,
            }
        )
        partition_records.append(
            {
                "source_entry_sha256": entry["entry_sha256"],
                "source_experience_sha256": entry["experience_sha256"],
                "lineage_root_sha256": lineage_root,
                "split": split,
                "example_sha256": row["example_sha256"],
                "augmentation_generation": 0,
                "augmentation_parent_sha256": ZERO_SHA256,
            }
        )
        privacy_records.append(
            {
                "source_entry_sha256": entry["entry_sha256"],
                "projection_content_sha256": projection_sha,
                "privacy_clearance_sha256": clearance["clearance_sha256"],
                "origin_classification": clearance["origin_classification"],
            }
        )
        rows_by_split[split].append(row)
        seen_entries.add(entry["entry_sha256"])

    if set(privacy_clearances) != seen_entries:
        _fail("verified_replay_sft_privacy_inventory_mismatch")

    # Contamination is reported before partition power, because it is the
    # prior fault and the more serious one: an overlap with a sealed
    # evaluation corpus means this material may not be projected at all,
    # whatever shape the split has. Reported the other way round, a store that
    # overlapped AND partitioned thin came back
    # "verified_replay_sft_partition_underpowered" — a sizing complaint about
    # a contamination.
    _assert_no_signature_overlap(signatures, references["records"])

    if any(len(rows_by_split[split]) < minimum_rows_per_split for split in SPLITS):
        _fail("verified_replay_sft_partition_underpowered")

    partition_body = {
        "schema": VERIFIED_REPLAY_SFT_PARTITION_SCHEMA,
        "source_store_sha256": store["store_sha256"],
        "source_store_revision": store["revision"],
        "partition_key_commitment_sha256": hashlib.sha256(partition_secret).hexdigest(),
        "partition_ratios_basis_points": ratios,
        "assignment_domain": "AURA-RLC-PRE-AUGMENTATION-SPLIT-v1",
        "records": partition_records,
        "record_count": len(partition_records),
        "augmentation_generation": 0,
        "future_derivatives_inherit_lineage_split": True,
        "status": "sealed_before_augmentation_no_training_authority",
    }
    partition_manifest = {**partition_body, "manifest_sha256": _sha(partition_body)}
    dedup_body = {
        "schema": VERIFIED_REPLAY_SFT_DEDUP_SCHEMA,
        "source_store_sha256": store["store_sha256"],
        "partition_manifest_sha256": partition_manifest["manifest_sha256"],
        "dedup_key_commitment_sha256": hashlib.sha256(dedup_secret).hexdigest(),
        "reference_index_sha256": references["index_sha256"],
        "reference_scope": references["scope"],
        "reference_coverage": references["coverage"],
        "projected_record_count": len(signatures),
        "reference_record_count": references["record_count"],
        "methods": [
            "keyed_exact_normalized_content",
            "keyed_objective_and_answer",
            "keyed_bottom_k_token_shingle_jaccard",
            "keyed_bottom_k_character_shingle_jaccard",
            "causal_lineage_nonoverlap",
        ],
        "token_shingle_size": _DEFAULT_TOKEN_SHINGLE_SIZE,
        "character_shingle_size": _DEFAULT_CHARACTER_SHINGLE_SIZE,
        "shingle_sketch_limit": _SHINGLE_SKETCH_LIMIT,
        "token_near_duplicate_threshold": _TOKEN_NEAR_DUPLICATE_THRESHOLD,
        "character_near_duplicate_threshold": _CHARACTER_NEAR_DUPLICATE_THRESHOLD,
        "exact_overlap_count": 0,
        "near_duplicate_overlap_count": 0,
        "lineage_overlap_count": 0,
        "status": "passed_local_keyed_projection_quarantine",
    }
    dedup_manifest = {**dedup_body, "manifest_sha256": _sha(dedup_body)}
    privacy_body = {
        "schema": VERIFIED_REPLAY_SFT_PRIVACY_MANIFEST_SCHEMA,
        "source_store_sha256": store["store_sha256"],
        "records": privacy_records,
        "record_count": len(privacy_records),
        "local_secret_scan_findings": 0,
        "local_pii_scan_findings": 0,
        "local_hidden_reasoning_findings": 0,
        "local_prompt_injection_findings": 0,
        "projected_private_field_count": 0,
        "projected_tool_trace_count": 0,
        "external_attestation_required": True,
        "status": "passed_local_projection_quarantine_only",
    }
    privacy_manifest = {**privacy_body, "manifest_sha256": _sha(privacy_body)}

    train_bytes = _jsonl(rows_by_split[TRAIN_SPLIT])
    validation_bytes = _jsonl(rows_by_split[VALIDATION_SPLIT])
    holdout_body = {
        "schema": "aura.rlc.verified_replay_sft_holdout.v1",
        "source_store_sha256": store["store_sha256"],
        "partition_manifest": partition_manifest,
        "semantic_dedup_manifest": dedup_manifest,
        "privacy_manifest": privacy_manifest,
        "examples": rows_by_split[HOLDOUT_SPLIT],
    }
    holdout_bytes = canonical_json_bytes(holdout_body)
    visible_bindings = {
        _CANDIDATE_TRAIN_FILE: _binding(train_bytes),
        _CANDIDATE_VALIDATION_FILE: _binding(validation_bytes),
    }
    holdout_binding = _binding(holdout_bytes)
    custody_root = _sha(
        {
            "domain": "AURA-RLC-VERIFIED-REPLAY-SFT-CUSTODY-v1",
            "source_store_sha256": store["store_sha256"],
            "visible_artifacts": visible_bindings,
            "holdout_artifact": holdout_binding,
            "partition_manifest_sha256": partition_manifest["manifest_sha256"],
            "semantic_dedup_manifest_sha256": dedup_manifest["manifest_sha256"],
            "privacy_manifest_sha256": privacy_manifest["manifest_sha256"],
        }
    )
    candidate_body = {
        "schema": VERIFIED_REPLAY_SFT_CANDIDATE_SCHEMA,
        "source_store_sha256": store["store_sha256"],
        "custody_root_sha256": custody_root,
        "artifacts": visible_bindings,
        "row_counts": {
            TRAIN_SPLIT: len(rows_by_split[TRAIN_SPLIT]),
            VALIDATION_SPLIT: len(rows_by_split[VALIDATION_SPLIT]),
        },
        "partition_manifest_sha256": partition_manifest["manifest_sha256"],
        "semantic_dedup_manifest_sha256": dedup_manifest["manifest_sha256"],
        "privacy_manifest_sha256": privacy_manifest["manifest_sha256"],
        "holdout_artifact_sha256": holdout_binding["sha256"],
        "trainer_contract": {
            "trainer": "mlx_lm.ChatDataset",
            "mask_prompt": True,
            "supervised_region": "final_assistant_message_only",
            "max_seq_length": max_seq_length,
            "truncation_allowed": False,
        },
        "trainer_ready": False,
        "training_authority": "none_quarantined_projection",
        "required_next_gates": [
            "external_privacy_attestation",
            "external_multisurface_contamination_audit",
            "resident_tokenizer_projection_validation",
            "externally_rooted_trainer_admission",
            "small_checkpoint_transfer_falsification",
            "independent_resident_promotion",
        ],
    }
    candidate_manifest = {
        **candidate_body,
        "candidate_package_sha256": _sha(candidate_body),
    }
    evaluator_body = {
        "schema": VERIFIED_REPLAY_SFT_EVALUATOR_SCHEMA,
        "source_store_sha256": store["store_sha256"],
        "custody_root_sha256": custody_root,
        "candidate_package_sha256": candidate_manifest["candidate_package_sha256"],
        "artifact": {"filename": _EVALUATOR_HOLDOUT_FILE, **holdout_binding},
        "holdout_row_count": len(rows_by_split[HOLDOUT_SPLIT]),
        "trainer_access": "forbidden_separate_artifact_root",
        "training_authority": "none_quarantined_projection",
    }
    evaluator_manifest = {
        **evaluator_body,
        "evaluator_package_sha256": _sha(evaluator_body),
    }

    candidate_artifacts = {
        _CANDIDATE_TRAIN_FILE: train_bytes,
        _CANDIDATE_VALIDATION_FILE: validation_bytes,
        _CANDIDATE_MANIFEST_FILE: canonical_json_bytes(candidate_manifest),
    }
    evaluator_artifacts = {
        _EVALUATOR_HOLDOUT_FILE: holdout_bytes,
        _EVALUATOR_MANIFEST_FILE: canonical_json_bytes(evaluator_manifest),
    }
    custody_body = {
        "schema": VERIFIED_REPLAY_SFT_CUSTODY_SCHEMA,
        "source_store_sha256": store["store_sha256"],
        "custody_root_sha256": custody_root,
        "candidate_package_sha256": candidate_manifest["candidate_package_sha256"],
        "evaluator_package_sha256": evaluator_manifest["evaluator_package_sha256"],
        "candidate_artifacts": {
            name: _binding(payload) for name, payload in candidate_artifacts.items()
        },
        "evaluator_artifacts": {
            name: _binding(payload) for name, payload in evaluator_artifacts.items()
        },
        "visible_row_count": len(rows_by_split[TRAIN_SPLIT])
        + len(rows_by_split[VALIDATION_SPLIT]),
        "holdout_row_count": len(rows_by_split[HOLDOUT_SPLIT]),
        "trainer_holdout_access": False,
        "trainer_ready": False,
        "training_authority": "none_quarantined_projection",
        "status": "passed_projection_custody_no_training_authority",
    }
    custody_report = {**custody_body, "report_sha256": _sha(custody_body)}
    validated_pair = validate_verified_replay_sft_custody_pair(
        candidate_artifacts,
        evaluator_artifacts,
    )
    if (
        validated_pair["candidate_manifest"] != candidate_manifest
        or validated_pair["evaluator_manifest"] != evaluator_manifest
    ):
        _fail("verified_replay_sft_custody_reconstruction_mismatch")
    return VerifiedReplaySFTCustodyBundles(
        candidate_artifacts=candidate_artifacts,
        evaluator_artifacts=evaluator_artifacts,
        custody_report=custody_report,
    )


__all__ = [
    "HOLDOUT_SPLIT",
    "SPLITS",
    "TRAIN_SPLIT",
    "VALIDATION_SPLIT",
    "VERIFIED_REPLAY_SFT_CANDIDATE_FILES",
    "VERIFIED_REPLAY_SFT_EVALUATOR_FILES",
    "VERIFIED_REPLAY_SFT_PRIVACY_CLEARANCE_SCHEMA",
    "VERIFIED_REPLAY_SFT_REFERENCE_INDEX_SCHEMA",
    "VerifiedReplaySFTCustodyBundles",
    "VerifiedReplaySFTError",
    "build_privacy_clearance",
    "build_reference_index",
    "build_semantic_signature_records",
    "build_verified_replay_sft_custody_bundles",
    "canonical_json_bytes",
    "empty_reference_index",
    "projection_content_sha256",
    "validate_privacy_clearance",
    "validate_reference_index",
    "validate_verified_replay_sft_tokenization",
    "validate_verified_replay_sft_candidate_artifacts",
    "validate_verified_replay_sft_custody_pair",
    "assert_semantic_signature_integrity",
]
