"""Durable token alignment and resident features for semantic programs.

This module owns acquisition evidence, not language interpretation. It binds
the synthetic program-first corpus to one local tokenizer and one resident
model basis, then stores the measured token states in compact immutable
records. The completion manifest is written only after every record survives
an independent reload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from core.brain.llm.hidden_sequence_contract import (
    HIDDEN_SEQUENCE_REPRESENTATIONS,
    LEXICAL_CONTEXTUAL_V1,
    hidden_sequence_channels,
    hidden_sequence_schema,
)
from core.governance_context import local_internal_governed_scope
from core.learning.semantic_program_corpus import (
    SemanticProgramExample,
    project_example_to_ir,
)
from core.learning.semantic_program_ir import semantic_program_ir_from_dict
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import FileWriteGateway, get_file_write_gateway

FEATURE_RECORD_SCHEMA: Final = "aura.semantic_program_feature_record.v1"
FEATURE_MANIFEST_SCHEMA: Final = "aura.semantic_program_feature_manifest.v1"
FEATURE_STATUS_SCHEMA: Final = "aura.semantic_program_feature_status.v1"
FEATURE_CONFIG_SCHEMA: Final = "aura.semantic_program_feature_config.v2"
GOLD_PROJECTION_SCHEMA: Final = "aura.semantic_program_gold_projection.v1"

_RECORD_MAGIC: Final = b"AURASPF1"
_RECORD_SUFFIX: Final = ".spf"
_MANIFEST_NAME: Final = "manifest.json"
_STATUS_NAME: Final = "status.json"
_MAX_EXAMPLES: Final = 4096
_MAX_RECORD_BYTES: Final = 128 * 1024 * 1024
_MAX_RECORD_BODY_BYTES: Final = 70 * 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
_MAX_TOKEN_COUNT: Final = 512
_MAX_HIDDEN_SIZE: Final = 32768
_NORMALIZATION_TOLERANCE: Final = 1e-4
_TOKENIZER_FILES: Final = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
)
_EVIDENCE_ABSENCE: Final = {
    "expected_outputs_available": False,
    "verifier_traces_available": False,
    "generated_compiler_text_available": False,
    "raw_text_persisted": False,
}
_LANE_RECEIPT_SCHEMA: Final = "aura.mlx_model_lane_ownership.v1"


class SemanticFeatureMaterializationError(RuntimeError):
    """The acquisition bundle could not establish its declared evidence."""


class HiddenSequenceClient(Protocol):
    """Resident-only feature surface used by the acquisition sidecar."""

    async def encode_hidden_sequence(
        self,
        text: str,
        *,
        timeout_s: float = 8.0,
        representation: str = "final_hidden_v1",
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class SemanticFeatureConfig:
    """Frozen corpus and acquisition bounds for one materialization."""

    seed: int = 271828
    examples_per_operation_pair: int = 1
    max_examples: int = 576
    representation: str = LEXICAL_CONTEXTUAL_V1
    hidden_timeout_s: float = 120.0
    idle_wait_s: float = 300.0
    idle_poll_s: float = 1.0
    schema: str = FEATURE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != FEATURE_CONFIG_SCHEMA
            or type(self.seed) is not int
            or type(self.examples_per_operation_pair) is not int
            or self.examples_per_operation_pair < 1
            or type(self.max_examples) is not int
            or not 1 <= self.max_examples <= _MAX_EXAMPLES
            or self.representation not in HIDDEN_SEQUENCE_REPRESENTATIONS
            or not 1.0 <= float(self.hidden_timeout_s) <= 3600.0
            or not 0.0 <= float(self.idle_wait_s) <= 86400.0
            or not 0.05 <= float(self.idle_poll_s) <= 60.0
        ):
            raise ValueError("semantic feature materialization config is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "examples_per_operation_pair": self.examples_per_operation_pair,
            "max_examples": self.max_examples,
            "representation": self.representation,
            "hidden_timeout_s": float(self.hidden_timeout_s),
            "idle_wait_s": float(self.idle_wait_s),
            "idle_poll_s": float(self.idle_poll_s),
        }


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Bounded acquisition outcome; incomplete work remains resumable."""

    complete: bool
    completed_examples: int
    total_examples: int
    output_directory: Path
    manifest_sha256: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class LoadedSemanticFeatureExample:
    """One strictly reloaded feature record."""

    metadata: dict[str, Any]
    token_ids: np.ndarray
    hidden_states: np.ndarray
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedSemanticFeatureBundle:
    """A complete, identity-consistent acquisition bundle."""

    manifest: dict[str, Any]
    examples: tuple[LoadedSemanticFeatureExample, ...]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SemanticFeatureMaterializationError(
            "semantic feature evidence is not canonical JSON"
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _example_public_identity(example: SemanticProgramExample) -> dict[str, Any]:
    return {
        "schema": example.schema,
        "example_id": example.example_id,
        "construction_id": example.construction_id,
        "topology_id": example.topology_id,
        "split": example.split,
        "source_text_sha256": hashlib.sha256(
            example.source_text.encode("utf-8")
        ).hexdigest(),
        "inputs": list(example.inputs),
        "program": {
            "n_inputs": example.program.n_inputs,
            "instructions": [
                {"op": instruction.op, "args": list(instruction.args)}
                for instruction in example.program.instructions
            ],
            "report_value": example.report_value,
        },
        "contrast_id": example.contrast_id,
    }


def select_bounded_semantic_examples(
    corpus: Sequence[SemanticProgramExample],
    *,
    max_examples: int,
) -> tuple[SemanticProgramExample, ...]:
    """Select deterministically while retaining every factorial corpus cell."""

    if not corpus or not 1 <= max_examples <= _MAX_EXAMPLES:
        raise ValueError("semantic feature selection bound is invalid")
    grouped: dict[
        tuple[str, str, tuple[str, ...]],
        list[SemanticProgramExample],
    ] = defaultdict(list)
    for example in corpus:
        grouped[
            (
                example.construction_id,
                example.topology_id,
                tuple(item.instruction.op for item in example.instructions),
            )
        ].append(example)
    if max_examples < len(grouped):
        raise ValueError("selection bound cannot represent every factorial corpus cell")
    for rows in grouped.values():
        rows.sort(key=lambda item: item.example_id)
    selected: list[SemanticProgramExample] = []
    ordinal = 0
    cell_ids = sorted(grouped)
    while len(selected) < min(max_examples, len(corpus)):
        advanced = False
        for cell_id in cell_ids:
            rows = grouped[cell_id]
            if ordinal < len(rows) and len(selected) < max_examples:
                selected.append(rows[ordinal])
                advanced = True
        if not advanced:
            break
        ordinal += 1
    if len({item.example_id for item in selected}) != len(selected):
        raise SemanticFeatureMaterializationError("semantic corpus ids are not unique")
    return tuple(selected)


def tokenizer_checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Hash the supplied local tokenizer files without loading model weights."""

    root = checkpoint.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SemanticFeatureMaterializationError(
            "tokenizer checkpoint must be a real local directory"
        )
    files: dict[str, dict[str, Any]] = {}
    for name in _TOKENIZER_FILES:
        path = root / name
        if not path.exists():
            continue
        payload = read_stable_bytes(path, max_bytes=512 * 1024 * 1024)
        files[name] = {"bytes": len(payload), "sha256": _sha_bytes(payload)}
    if not any(name in files for name in ("tokenizer.json", "vocab.json")):
        raise SemanticFeatureMaterializationError(
            "checkpoint does not contain a local tokenizer vocabulary"
        )
    body = {
        "schema": "aura.semantic_program_tokenizer_identity.v1",
        "checkpoint_path": str(root),
        "files": files,
        "offsets_required": True,
        "generated_text": False,
    }
    return {**body, "identity_sha256": _sha(body)}


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Run one local offset-aware tokenization with no decode or generation."""

    if type(text) is not str or not text:
        raise ValueError("semantic tokenizer input must be non-empty text")
    if not callable(tokenizer):
        raise TypeError("semantic tokenizer does not expose offset tokenization")
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except Exception as exc:  # noqa: BLE001 - tokenizer is an evidence boundary
        raise SemanticFeatureMaterializationError(
            "local tokenizer could not return offsets"
        ) from exc
    if not isinstance(encoded, Mapping):
        raise SemanticFeatureMaterializationError("local tokenizer result is not a mapping")
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(raw_ids, (list, tuple)) or not isinstance(
        raw_offsets, (list, tuple)
    ):
        raise SemanticFeatureMaterializationError(
            "local tokenizer omitted token ids or offsets"
        )
    token_ids = list(raw_ids)
    offsets: list[tuple[int, int]] = []
    for value in raw_offsets:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise SemanticFeatureMaterializationError("tokenizer offset is malformed")
        start, end = value
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > len(text)
        ):
            raise SemanticFeatureMaterializationError("tokenizer offset is out of range")
        offsets.append((start, end))
    if (
        not 1 <= len(token_ids) <= _MAX_TOKEN_COUNT
        or len(token_ids) != len(offsets)
        or any(type(token_id) is not int or not 0 <= token_id < 2**31 for token_id in token_ids)
    ):
        raise SemanticFeatureMaterializationError(
            "local tokenizer returned invalid bounded token ids"
        )
    return token_ids, offsets


def offset_tokenizer_for_worker(tokenizer: Any) -> Any:
    """Return the offset surface paired with a worker tokenization wrapper."""

    if callable(tokenizer):
        return tokenizer
    underlying = getattr(tokenizer, "_tokenizer", None)
    if callable(underlying):
        return underlying
    raise TypeError("resident tokenizer has no offset-aware tokenization surface")


def _validate_model_basis(model_basis: Any, checkpoint: Path) -> dict[str, Any]:
    if not isinstance(model_basis, Mapping) or not model_basis:
        raise SemanticFeatureMaterializationError("worker model basis is unavailable")
    basis = json.loads(_canonical_bytes(dict(model_basis)))
    worker_path = basis.get("worker_model_path")
    if not isinstance(worker_path, str) or not worker_path:
        raise SemanticFeatureMaterializationError("worker model basis omits its checkpoint")
    if os.path.realpath(worker_path) != os.path.realpath(checkpoint):
        raise SemanticFeatureMaterializationError(
            "resident worker model differs from the supplied tokenizer checkpoint"
        )
    return basis


def validate_exclusive_lane_receipt(
    receipt: Mapping[str, Any],
    *,
    checkpoint: Path,
    model_basis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the standalone campaign's committed model-lane ownership."""

    if not isinstance(receipt, Mapping):
        raise SemanticFeatureMaterializationError("exclusive model-lane receipt is missing")
    accepted = json.loads(_canonical_bytes(dict(receipt)))
    expected_fields = {
        "schema",
        "exclusive",
        "owner_id",
        "fencing_token",
        "terminal_receipt_id",
        "model_path",
        "campaign_pid",
        "worker_pid",
        "worker_boot_id",
        "receipt_sha256",
    }
    if set(accepted) != expected_fields:
        raise SemanticFeatureMaterializationError("exclusive model-lane receipt fields differ")
    receipt_sha256 = accepted.pop("receipt_sha256")
    if not _is_sha256(receipt_sha256) or _sha(accepted) != receipt_sha256:
        raise SemanticFeatureMaterializationError("exclusive model-lane receipt hash differs")
    accepted["receipt_sha256"] = receipt_sha256
    if (
        accepted.get("schema") != _LANE_RECEIPT_SCHEMA
        or accepted.get("exclusive") is not True
        or not isinstance(accepted.get("owner_id"), str)
        or not accepted["owner_id"]
        or type(accepted.get("fencing_token")) is not int
        or accepted["fencing_token"] <= 0
        or not isinstance(accepted.get("terminal_receipt_id"), str)
        or not accepted["terminal_receipt_id"]
        or type(accepted.get("campaign_pid")) is not int
        or accepted["campaign_pid"] <= 0
        or type(accepted.get("worker_pid")) is not int
        or accepted["worker_pid"] <= 0
        or not isinstance(accepted.get("worker_boot_id"), str)
        or not accepted["worker_boot_id"]
        or os.path.realpath(str(accepted.get("model_path") or ""))
        != os.path.realpath(checkpoint)
    ):
        raise SemanticFeatureMaterializationError("exclusive model-lane receipt is invalid")
    if model_basis is not None and (
        accepted["worker_pid"] != model_basis.get("worker_pid")
        or accepted["worker_boot_id"] != model_basis.get("worker_boot_id")
        or os.path.realpath(str(model_basis.get("worker_model_path") or ""))
        != os.path.realpath(checkpoint)
    ):
        raise SemanticFeatureMaterializationError(
            "exclusive model-lane receipt differs from worker identity"
        )
    return accepted


def _validate_hidden_observation(
    observation: Any,
    *,
    text: str,
    local_token_ids: Sequence[int],
    checkpoint: Path,
    representation: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not isinstance(observation, Mapping):
        raise SemanticFeatureMaterializationError("resident feature observation is malformed")
    worker_token_ids = observation.get("token_ids")
    states = observation.get("hidden_states")
    receipt = observation.get("receipt")
    if list(worker_token_ids or []) != list(local_token_ids):
        raise SemanticFeatureMaterializationError(
            "local tokenizer ids differ from resident worker token ids"
        )
    if not isinstance(states, np.ndarray) or states.dtype != np.dtype("float32"):
        raise SemanticFeatureMaterializationError(
            "resident hidden states must be a float32 array"
        )
    if (
        states.ndim != 2
        or states.shape[0] != len(local_token_ids)
        or not 1 <= states.shape[1] <= _MAX_HIDDEN_SIZE
        or not states.flags.c_contiguous
        or not np.all(np.isfinite(states))
    ):
        raise SemanticFeatureMaterializationError(
            "resident hidden-state shape or values are invalid"
        )
    norms = np.linalg.norm(states, axis=1)
    if np.any(np.abs(norms - 1.0) > _NORMALIZATION_TOLERANCE):
        raise SemanticFeatureMaterializationError(
            "resident hidden states are not unit normalized"
        )
    if not isinstance(receipt, Mapping):
        raise SemanticFeatureMaterializationError("resident feature receipt is missing")
    accepted_receipt = json.loads(_canonical_bytes(dict(receipt)))
    required = {
        "schema": hidden_sequence_schema(representation),
        "action": "encode_hidden_sequence",
        "input_char_count": len(text),
        "token_count": len(local_token_ids),
        "hidden_size": states.shape[1],
        "hidden_state_bytes": states.nbytes,
        "hidden_state_sha256": _sha_bytes(states.astype("<f4", copy=False).tobytes()),
        "transport": "packed_float32_le",
        "representation": representation,
        "channels": list(hidden_sequence_channels(representation)),
        "forward_passes": 1,
        "causal_full_sequence": True,
        "sampling": False,
        "generated_tokens": 0,
        "generated_text": False,
    }
    if any(accepted_receipt.get(key) != value for key, value in required.items()):
        raise SemanticFeatureMaterializationError(
            "resident feature receipt differs from measured payload"
        )
    request_id = accepted_receipt.get("request_id")
    limits = accepted_receipt.get("limits")
    if not isinstance(request_id, str) or not request_id or not isinstance(limits, dict):
        raise SemanticFeatureMaterializationError("resident feature receipt is incomplete")
    if (
        limits.get("max_input_chars", 0) < len(text)
        or limits.get("max_tokens", 0) < len(local_token_ids)
        or limits.get("max_hidden_size", 0) < states.shape[1]
    ):
        raise SemanticFeatureMaterializationError("resident feature receipt limits are invalid")
    model_basis = _validate_model_basis(accepted_receipt.get("model_basis"), checkpoint)
    return states, accepted_receipt, model_basis


def _gold_projection_receipt(
    *,
    corpus_sha256: str,
    example_id: str,
) -> dict[str, Any]:
    body = {
        "schema": GOLD_PROJECTION_SCHEMA,
        "corpus_sha256": corpus_sha256,
        "example_id": example_id,
        "authority": "synthetic_corpus_character_annotations_only",
        "serving_authority": False,
        **_EVIDENCE_ABSENCE,
    }
    return {**body, "receipt_sha256": _sha(body)}


def _encode_record(
    metadata_without_hash: dict[str, Any],
    token_ids: Sequence[int],
    hidden_states: np.ndarray,
) -> bytes:
    tokens = np.asarray(token_ids, dtype="<i4")
    states = np.asarray(hidden_states, dtype="<f4", order="C")
    metadata = {
        **metadata_without_hash,
        "token_dtype": "int32_le",
        "hidden_dtype": "float32_le",
        "token_count": int(tokens.shape[0]),
        "hidden_size": int(states.shape[1]),
        "token_ids_sha256": _sha_bytes(tokens.tobytes(order="C")),
        "hidden_states_sha256": _sha_bytes(states.tobytes(order="C")),
    }
    metadata = {**metadata, "logical_payload_sha256": _sha(metadata)}
    header = _canonical_bytes(metadata)
    body = struct.pack("<I", len(header)) + header + tokens.tobytes() + states.tobytes()
    if len(body) > _MAX_RECORD_BODY_BYTES:
        raise SemanticFeatureMaterializationError("semantic feature record exceeds bound")
    return _RECORD_MAGIC + zlib.compress(body, level=9)


def _decompress_record(payload: bytes) -> bytes:
    if not payload.startswith(_RECORD_MAGIC) or len(payload) > _MAX_RECORD_BYTES:
        raise SemanticFeatureMaterializationError("semantic feature record envelope is invalid")
    decoder = zlib.decompressobj()
    body = decoder.decompress(payload[len(_RECORD_MAGIC) :], _MAX_RECORD_BODY_BYTES + 1)
    if len(body) > _MAX_RECORD_BODY_BYTES or decoder.unconsumed_tail:
        raise SemanticFeatureMaterializationError("semantic feature record expands past bound")
    tail = decoder.flush()
    if len(body) + len(tail) > _MAX_RECORD_BODY_BYTES:
        raise SemanticFeatureMaterializationError("semantic feature record expands past bound")
    body += tail
    if not decoder.eof or decoder.unused_data:
        raise SemanticFeatureMaterializationError("semantic feature record compression is invalid")
    return body


def load_semantic_feature_record(
    path: Path,
    *,
    expected_payload_sha256: str | None = None,
) -> LoadedSemanticFeatureExample:
    """Load and validate one immutable compressed feature record."""

    payload = read_stable_bytes(path, max_bytes=_MAX_RECORD_BYTES)
    payload_sha256 = _sha_bytes(payload)
    if expected_payload_sha256 is not None and payload_sha256 != expected_payload_sha256:
        raise SemanticFeatureMaterializationError("semantic feature payload hash differs")
    body = _decompress_record(payload)
    if len(body) < 4:
        raise SemanticFeatureMaterializationError("semantic feature record body is truncated")
    (header_size,) = struct.unpack("<I", body[:4])
    if not 1 <= header_size <= 4 * 1024 * 1024 or 4 + header_size > len(body):
        raise SemanticFeatureMaterializationError("semantic feature record header is invalid")
    try:
        metadata = json.loads(body[4 : 4 + header_size].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticFeatureMaterializationError(
            "semantic feature record metadata is invalid"
        ) from exc
    expected_fields = {
        "schema",
        "example_id",
        "split",
        "construction_id",
        "topology_id",
        "contrast_id",
        "inputs",
        "source_text_sha256",
        "corpus_sha256",
        "config_sha256",
        "tokenizer_identity_sha256",
        "model_basis_sha256",
        "worker_receipt",
        "lane_ownership_receipt",
        "lane_ownership_sha256",
        "gold_projection_receipt",
        "gold_ir",
        "evidence_absence",
        "token_dtype",
        "hidden_dtype",
        "token_count",
        "hidden_size",
        "token_ids_sha256",
        "hidden_states_sha256",
        "logical_payload_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise SemanticFeatureMaterializationError("semantic feature metadata fields differ")
    logical_hash = metadata.pop("logical_payload_sha256")
    if not _is_sha256(logical_hash) or _sha(metadata) != logical_hash:
        raise SemanticFeatureMaterializationError("semantic feature logical hash differs")
    metadata["logical_payload_sha256"] = logical_hash
    token_count = metadata.get("token_count")
    hidden_size = metadata.get("hidden_size")
    if (
        metadata.get("schema") != FEATURE_RECORD_SCHEMA
        or metadata.get("token_dtype") != "int32_le"
        or metadata.get("hidden_dtype") != "float32_le"
        or type(token_count) is not int
        or not 1 <= token_count <= _MAX_TOKEN_COUNT
        or type(hidden_size) is not int
        or not 1 <= hidden_size <= _MAX_HIDDEN_SIZE
        or metadata.get("evidence_absence") != _EVIDENCE_ABSENCE
    ):
        raise SemanticFeatureMaterializationError("semantic feature metadata contract differs")
    array_offset = 4 + header_size
    token_bytes = token_count * 4
    hidden_bytes = token_count * hidden_size * 4
    if len(body) != array_offset + token_bytes + hidden_bytes:
        raise SemanticFeatureMaterializationError("semantic feature array lengths differ")
    token_payload = body[array_offset : array_offset + token_bytes]
    hidden_payload = body[array_offset + token_bytes :]
    if (
        _sha_bytes(token_payload) != metadata.get("token_ids_sha256")
        or _sha_bytes(hidden_payload) != metadata.get("hidden_states_sha256")
    ):
        raise SemanticFeatureMaterializationError("semantic feature array hash differs")
    token_ids = np.frombuffer(token_payload, dtype="<i4").copy()
    hidden_states = np.frombuffer(hidden_payload, dtype="<f4").reshape(
        token_count, hidden_size
    ).copy()
    if np.any(token_ids < 0) or not np.all(np.isfinite(hidden_states)):
        raise SemanticFeatureMaterializationError("semantic feature arrays are invalid")
    norms = np.linalg.norm(hidden_states, axis=1)
    if np.any(np.abs(norms - 1.0) > _NORMALIZATION_TOLERANCE):
        raise SemanticFeatureMaterializationError(
            "semantic feature rows are not unit normalized"
        )
    ir = semantic_program_ir_from_dict(metadata["gold_ir"])
    if list(ir.source_token_ids) != token_ids.tolist():
        raise SemanticFeatureMaterializationError("gold IR token ids differ from record")
    projection = metadata.get("gold_projection_receipt")
    worker = metadata.get("worker_receipt")
    if not isinstance(projection, dict) or not isinstance(worker, dict):
        raise SemanticFeatureMaterializationError("semantic feature receipts are missing")
    projection_hash = projection.get("receipt_sha256")
    projection_body = dict(projection)
    projection_body.pop("receipt_sha256", None)
    if not _is_sha256(projection_hash) or _sha(projection_body) != projection_hash:
        raise SemanticFeatureMaterializationError("gold projection receipt hash differs")
    if ir.transducer_receipt_sha256 != projection_hash:
        raise SemanticFeatureMaterializationError("gold IR projection binding differs")
    basis = worker.get("model_basis")
    if not isinstance(basis, dict) or _sha(basis) != metadata.get("model_basis_sha256"):
        raise SemanticFeatureMaterializationError("worker model basis hash differs")
    if ir.model_basis_receipt_sha256 != metadata.get("model_basis_sha256"):
        raise SemanticFeatureMaterializationError("gold IR model basis binding differs")
    lane_receipt = validate_exclusive_lane_receipt(
        metadata["lane_ownership_receipt"],
        checkpoint=Path(str(basis.get("worker_model_path") or "")),
        model_basis=basis,
    )
    if lane_receipt["receipt_sha256"] != metadata["lane_ownership_sha256"]:
        raise SemanticFeatureMaterializationError("feature record lane binding differs")
    return LoadedSemanticFeatureExample(
        metadata=metadata,
        token_ids=token_ids,
        hidden_states=hidden_states,
        payload_sha256=payload_sha256,
    )


def _validate_construction_disjointness(examples: Sequence[LoadedSemanticFeatureExample]) -> None:
    construction_splits: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        construction_splits[str(example.metadata["construction_id"])].add(
            str(example.metadata["split"])
        )
    if any(len(splits) != 1 for splits in construction_splits.values()):
        raise SemanticFeatureMaterializationError(
            "semantic feature constructions cross evaluation splits"
        )
    split_constructions: dict[str, set[str]] = defaultdict(set)
    for construction, splits in construction_splits.items():
        split_constructions[next(iter(splits))].add(construction)
    split_names = sorted(split_constructions)
    if any(
        split_constructions[left] & split_constructions[right]
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    ):
        raise SemanticFeatureMaterializationError(
            "semantic feature split construction sets overlap"
        )


def load_semantic_feature_bundle(
    output_directory: Path,
    *,
    expected_examples: Sequence[SemanticProgramExample] | None = None,
) -> LoadedSemanticFeatureBundle:
    """CPU-only validation of one complete materialization bundle."""

    root = output_directory.expanduser().resolve(strict=True)
    manifest_payload = read_stable_bytes(
        root / _MANIFEST_NAME,
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    try:
        manifest = json.loads(manifest_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticFeatureMaterializationError("feature manifest is invalid") from exc
    expected_manifest_fields = {
        "schema",
        "complete",
        "config",
        "config_sha256",
        "corpus_sha256",
        "tokenizer_identity",
        "exact_model_path",
        "model_bases",
        "lane_ownership_receipts",
        "example_count",
        "split_counts",
        "split_constructions",
        "records",
        "evidence_absence",
        "manifest_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise SemanticFeatureMaterializationError("feature manifest fields differ")
    manifest_hash = manifest.pop("manifest_sha256")
    if not _is_sha256(manifest_hash) or _sha(manifest) != manifest_hash:
        raise SemanticFeatureMaterializationError("feature manifest hash differs")
    manifest["manifest_sha256"] = manifest_hash
    if manifest.get("schema") != FEATURE_MANIFEST_SCHEMA or manifest.get("complete") is not True:
        raise SemanticFeatureMaterializationError("feature materialization is incomplete")
    records = manifest.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= _MAX_EXAMPLES:
        raise SemanticFeatureMaterializationError("feature manifest records are invalid")
    allowed_names = {_MANIFEST_NAME, _STATUS_NAME, *(record.get("file", "") for record in records)}
    observed_names = {path.name for path in root.iterdir() if path.name != ".aura_file_write_batch.lock"}
    if observed_names != allowed_names:
        raise SemanticFeatureMaterializationError("feature bundle inventory differs")
    loaded: list[LoadedSemanticFeatureExample] = []
    record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "example_id",
            "file",
            "payload_sha256",
        }:
            raise SemanticFeatureMaterializationError("feature manifest record is malformed")
        example_id = record["example_id"]
        filename = record["file"]
        if (
            not isinstance(example_id, str)
            or example_id in record_ids
            or filename != f"{example_id}{_RECORD_SUFFIX}"
            or not _is_sha256(record["payload_sha256"])
        ):
            raise SemanticFeatureMaterializationError("feature manifest record identity differs")
        item = load_semantic_feature_record(
            root / filename,
            expected_payload_sha256=record["payload_sha256"],
        )
        if item.metadata["example_id"] != example_id:
            raise SemanticFeatureMaterializationError("feature record example id differs")
        record_ids.add(example_id)
        loaded.append(item)
    common_fields = (
        "config_sha256",
        "corpus_sha256",
        "tokenizer_identity_sha256",
    )
    for item in loaded:
        if any(
            item.metadata[field]
            != (
                manifest["tokenizer_identity"]["identity_sha256"]
                if field == "tokenizer_identity_sha256"
                else manifest[field]
            )
            for field in common_fields
        ):
            raise SemanticFeatureMaterializationError("feature record identity differs from manifest")
    model_bases = manifest.get("model_bases")
    lane_receipts = manifest.get("lane_ownership_receipts")
    if not isinstance(model_bases, list) or not model_bases:
        raise SemanticFeatureMaterializationError("feature manifest model bases are missing")
    accepted_basis_hashes: set[str] = set()
    for entry in model_bases:
        if not isinstance(entry, dict) or set(entry) != {"sha256", "receipt"}:
            raise SemanticFeatureMaterializationError("feature manifest model basis is malformed")
        if not _is_sha256(entry["sha256"]) or _sha(entry["receipt"]) != entry["sha256"]:
            raise SemanticFeatureMaterializationError("feature manifest model basis hash differs")
        _validate_model_basis(entry["receipt"], Path(manifest["exact_model_path"]))
        accepted_basis_hashes.add(entry["sha256"])
    if not isinstance(lane_receipts, list) or not lane_receipts:
        raise SemanticFeatureMaterializationError("feature manifest lane receipts are missing")
    accepted_lane_hashes = {
        validate_exclusive_lane_receipt(
            receipt,
            checkpoint=Path(manifest["exact_model_path"]),
        )["receipt_sha256"]
        for receipt in lane_receipts
    }
    if any(
        item.metadata["model_basis_sha256"] not in accepted_basis_hashes
        or item.metadata["lane_ownership_sha256"] not in accepted_lane_hashes
        for item in loaded
    ):
        raise SemanticFeatureMaterializationError("feature record ownership is absent from manifest")
    if manifest["config_sha256"] != _sha(manifest["config"]):
        raise SemanticFeatureMaterializationError("feature manifest config hash differs")
    split_counts = Counter(str(item.metadata["split"]) for item in loaded)
    if dict(sorted(split_counts.items())) != manifest.get("split_counts"):
        raise SemanticFeatureMaterializationError("feature manifest split counts differ")
    _validate_construction_disjointness(loaded)
    observed_split_constructions: dict[str, list[str]] = defaultdict(list)
    for item in loaded:
        split = str(item.metadata["split"])
        construction = str(item.metadata["construction_id"])
        if construction not in observed_split_constructions[split]:
            observed_split_constructions[split].append(construction)
    normalized_constructions = {
        split: sorted(values) for split, values in observed_split_constructions.items()
    }
    if normalized_constructions != manifest.get("split_constructions"):
        raise SemanticFeatureMaterializationError(
            "feature manifest split construction inventory differs"
        )
    if manifest.get("example_count") != len(loaded) or manifest.get(
        "evidence_absence"
    ) != _EVIDENCE_ABSENCE:
        raise SemanticFeatureMaterializationError("feature manifest count or authority differs")
    if expected_examples is not None:
        expected = {item.example_id: _example_public_identity(item) for item in expected_examples}
        if set(expected) != record_ids:
            raise SemanticFeatureMaterializationError("feature bundle corpus membership differs")
        for item in loaded:
            identity = expected[item.metadata["example_id"]]
            for field in (
                "construction_id",
                "topology_id",
                "split",
                "source_text_sha256",
                "inputs",
                "contrast_id",
            ):
                if item.metadata[field] != identity[field]:
                    raise SemanticFeatureMaterializationError(
                        "feature record differs from rebuilt synthetic corpus"
                    )
    return LoadedSemanticFeatureBundle(manifest=manifest, examples=tuple(loaded))


def _write_bytes(
    gateway: FileWriteGateway,
    path: Path,
    payload: bytes,
    *,
    source: str,
    if_absent: bool = False,
) -> bool:
    with local_internal_governed_scope(source, domain="file_write"):
        if if_absent:
            return gateway.write_bytes_if_absent(
                path,
                payload,
                mode=0o600,
                source=source,
            )
        gateway.write_bytes(path, payload, source=source)
    return True


def _resolve_materialization_paths(
    checkpoint: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    return (
        checkpoint.expanduser().resolve(strict=True),
        output_directory.expanduser().resolve(strict=False),
    )


def _ensure_output_directory(gateway: FileWriteGateway, root: Path) -> None:
    with local_internal_governed_scope(
        "semantic_program_features.directory",
        domain="file_write",
    ):
        gateway.ensure_directory(root, source="semantic_program_features.directory")


def _partial_inventory(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.name != ".aura_file_write_batch.lock"
    }


def _status_payload(
    *,
    complete: bool,
    reason: str,
    completed: int,
    total: int,
    config_sha256: str,
    corpus_sha256: str,
) -> bytes:
    body = {
        "schema": FEATURE_STATUS_SCHEMA,
        "complete": complete,
        "reason": reason,
        "completed_examples": completed,
        "total_examples": total,
        "config_sha256": config_sha256,
        "corpus_sha256": corpus_sha256,
        "updated_at_unix": time.time(),
    }
    return _canonical_bytes({**body, "status_sha256": _sha(body)})


async def materialize_semantic_program_features(
    *,
    client: HiddenSequenceClient,
    tokenizer: Any,
    checkpoint: Path,
    output_directory: Path,
    corpus: Sequence[SemanticProgramExample],
    config: SemanticFeatureConfig,
    lane_ownership_receipt: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any] | None = None,
    gateway: FileWriteGateway | None = None,
) -> MaterializationResult:
    """Acquire sequential resident features and publish a resumable bundle."""

    selected = select_bounded_semantic_examples(corpus, max_examples=config.max_examples)
    selected_identities = [_example_public_identity(example) for example in selected]
    corpus_sha256 = _sha(
        {
            "schema": "aura.semantic_program_selected_corpus.v1",
            "examples": selected_identities,
        }
    )
    config_body = {**config.to_dict(), "selected_example_ids": [item.example_id for item in selected]}
    config_sha256 = _sha(config_body)
    resolved_checkpoint, root = await asyncio.to_thread(
        _resolve_materialization_paths,
        checkpoint,
        output_directory,
    )
    measured_tokenizer_identity = tokenizer_identity
    if measured_tokenizer_identity is None:
        measured_tokenizer_identity = await asyncio.to_thread(
            tokenizer_checkpoint_identity,
            resolved_checkpoint,
        )
    accepted_tokenizer_identity = dict(measured_tokenizer_identity)
    if accepted_tokenizer_identity.get("identity_sha256") != _sha(
        {key: value for key, value in accepted_tokenizer_identity.items() if key != "identity_sha256"}
    ):
        raise SemanticFeatureMaterializationError("tokenizer identity hash differs")
    current_lane_receipt = validate_exclusive_lane_receipt(
        lane_ownership_receipt,
        checkpoint=resolved_checkpoint,
    )
    writer = gateway or get_file_write_gateway()
    await asyncio.to_thread(_ensure_output_directory, writer, root)
    if (root / _MANIFEST_NAME).exists():
        bundle = await asyncio.to_thread(
            load_semantic_feature_bundle,
            root,
            expected_examples=selected,
        )
        if bundle.manifest["config_sha256"] != config_sha256:
            raise SemanticFeatureMaterializationError("existing bundle config differs")
        return MaterializationResult(
            complete=True,
            completed_examples=len(bundle.examples),
            total_examples=len(selected),
            output_directory=root,
            manifest_sha256=bundle.manifest["manifest_sha256"],
            reason="already_complete",
        )

    allowed_partial = {_STATUS_NAME, *(f"{item.example_id}{_RECORD_SUFFIX}" for item in selected)}
    observed_partial = await asyncio.to_thread(_partial_inventory, root)
    if not observed_partial.issubset(allowed_partial):
        raise SemanticFeatureMaterializationError("partial feature directory contains foreign files")

    records: dict[str, LoadedSemanticFeatureExample] = {}
    for example in selected:
        path = root / f"{example.example_id}{_RECORD_SUFFIX}"
        if not path.exists():
            continue
        loaded = await asyncio.to_thread(load_semantic_feature_record, path)
        if (
            loaded.metadata["example_id"] != example.example_id
            or loaded.metadata["config_sha256"] != config_sha256
            or loaded.metadata["corpus_sha256"] != corpus_sha256
            or loaded.metadata["tokenizer_identity_sha256"]
            != accepted_tokenizer_identity["identity_sha256"]
        ):
            raise SemanticFeatureMaterializationError("partial feature record identity differs")
        records[example.example_id] = loaded

    idle_deadline = time.monotonic() + float(config.idle_wait_s)
    for example in selected:
        if example.example_id in records:
            continue
        local_token_ids, offsets = await asyncio.to_thread(
            tokenize_with_offsets,
            tokenizer,
            example.source_text,
        )
        observation: dict[str, Any] | None = None
        while observation is None:
            observation = await client.encode_hidden_sequence(
                example.source_text,
                timeout_s=float(config.hidden_timeout_s),
                representation=config.representation,
            )
            if observation is not None:
                break
            if time.monotonic() >= idle_deadline:
                status = _status_payload(
                    complete=False,
                    reason="resident_lane_busy",
                    completed=len(records),
                    total=len(selected),
                    config_sha256=config_sha256,
                    corpus_sha256=corpus_sha256,
                )
                await asyncio.to_thread(
                    _write_bytes,
                    writer,
                    root / _STATUS_NAME,
                    status,
                    source="semantic_program_features.partial_status",
                )
                return MaterializationResult(
                    complete=False,
                    completed_examples=len(records),
                    total_examples=len(selected),
                    output_directory=root,
                    manifest_sha256=None,
                    reason="resident_lane_busy",
                )
            await asyncio.sleep(float(config.idle_poll_s))
        states, worker_receipt, model_basis = _validate_hidden_observation(
            observation,
            text=example.source_text,
            local_token_ids=local_token_ids,
            checkpoint=resolved_checkpoint,
            representation=config.representation,
        )
        current_lane_receipt = validate_exclusive_lane_receipt(
            current_lane_receipt,
            checkpoint=resolved_checkpoint,
            model_basis=model_basis,
        )
        model_basis_sha256 = _sha(model_basis)
        projection_receipt = _gold_projection_receipt(
            corpus_sha256=corpus_sha256,
            example_id=example.example_id,
        )
        gold_ir = project_example_to_ir(
            example,
            source_token_ids=local_token_ids,
            offset_mapping=offsets,
            model_basis_receipt_sha256=model_basis_sha256,
            transducer_receipt_sha256=projection_receipt["receipt_sha256"],
        )
        metadata = {
            "schema": FEATURE_RECORD_SCHEMA,
            "example_id": example.example_id,
            "split": example.split,
            "construction_id": example.construction_id,
            "topology_id": example.topology_id,
            "contrast_id": example.contrast_id,
            "inputs": list(example.inputs),
            "source_text_sha256": hashlib.sha256(
                example.source_text.encode("utf-8")
            ).hexdigest(),
            "corpus_sha256": corpus_sha256,
            "config_sha256": config_sha256,
            "tokenizer_identity_sha256": accepted_tokenizer_identity["identity_sha256"],
            "model_basis_sha256": model_basis_sha256,
            "worker_receipt": worker_receipt,
            "lane_ownership_receipt": current_lane_receipt,
            "lane_ownership_sha256": current_lane_receipt["receipt_sha256"],
            "gold_projection_receipt": projection_receipt,
            "gold_ir": gold_ir.to_dict(),
            "evidence_absence": dict(_EVIDENCE_ABSENCE),
        }
        payload = _encode_record(metadata, local_token_ids, states)
        path = root / f"{example.example_id}{_RECORD_SUFFIX}"
        created = await asyncio.to_thread(
            _write_bytes,
            writer,
            path,
            payload,
            source="semantic_program_features.example",
            if_absent=True,
        )
        loaded = await asyncio.to_thread(load_semantic_feature_record, path)
        if not created and loaded.payload_sha256 != _sha_bytes(payload):
            raise SemanticFeatureMaterializationError(
                "concurrent feature record differs from this acquisition"
            )
        records[example.example_id] = loaded
        progress_status = _status_payload(
                complete=False,
                reason="acquiring",
                completed=len(records),
                total=len(selected),
                config_sha256=config_sha256,
                corpus_sha256=corpus_sha256,
            )
        await asyncio.to_thread(
            _write_bytes,
            writer,
            root / _STATUS_NAME,
            progress_status,
            source="semantic_program_features.progress",
        )

    ordered_records = [records[example.example_id] for example in selected]
    split_counts = Counter(example.split for example in selected)
    split_constructions: dict[str, set[str]] = defaultdict(set)
    for example in selected:
        split_constructions[example.split].add(example.construction_id)
    model_bases_by_hash: dict[str, dict[str, Any]] = {}
    lane_receipts_by_hash: dict[str, dict[str, Any]] = {}
    for record in ordered_records:
        basis = record.metadata["worker_receipt"]["model_basis"]
        model_bases_by_hash[record.metadata["model_basis_sha256"]] = basis
        lane_receipt = record.metadata["lane_ownership_receipt"]
        lane_receipts_by_hash[record.metadata["lane_ownership_sha256"]] = lane_receipt
    manifest_body = {
        "schema": FEATURE_MANIFEST_SCHEMA,
        "complete": True,
        "config": config_body,
        "config_sha256": config_sha256,
        "corpus_sha256": corpus_sha256,
        "tokenizer_identity": accepted_tokenizer_identity,
        "exact_model_path": str(resolved_checkpoint),
        "model_bases": [
            {"sha256": value, "receipt": model_bases_by_hash[value]}
            for value in sorted(model_bases_by_hash)
        ],
        "lane_ownership_receipts": [
            lane_receipts_by_hash[value] for value in sorted(lane_receipts_by_hash)
        ],
        "example_count": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "split_constructions": {
            split: sorted(values) for split, values in sorted(split_constructions.items())
        },
        "records": [
            {
                "example_id": example.example_id,
                "file": f"{example.example_id}{_RECORD_SUFFIX}",
                "payload_sha256": record.payload_sha256,
            }
            for example, record in zip(selected, ordered_records, strict=True)
        ],
        "evidence_absence": dict(_EVIDENCE_ABSENCE),
    }
    manifest = {**manifest_body, "manifest_sha256": _sha(manifest_body)}
    await asyncio.to_thread(
        _write_bytes,
        writer,
        root / _MANIFEST_NAME,
        _canonical_bytes(manifest),
        source="semantic_program_features.manifest",
        if_absent=True,
    )
    bundle = await asyncio.to_thread(
        load_semantic_feature_bundle,
        root,
        expected_examples=selected,
    )
    complete_status = _status_payload(
        complete=True,
        reason="complete",
        completed=len(bundle.examples),
        total=len(selected),
        config_sha256=config_sha256,
        corpus_sha256=corpus_sha256,
    )
    await asyncio.to_thread(
        _write_bytes,
        writer,
        root / _STATUS_NAME,
        complete_status,
        source="semantic_program_features.complete_status",
    )
    return MaterializationResult(
        complete=True,
        completed_examples=len(bundle.examples),
        total_examples=len(selected),
        output_directory=root,
        manifest_sha256=bundle.manifest["manifest_sha256"],
        reason="complete",
    )


__all__ = [
    "FEATURE_CONFIG_SCHEMA",
    "FEATURE_MANIFEST_SCHEMA",
    "FEATURE_RECORD_SCHEMA",
    "LoadedSemanticFeatureBundle",
    "LoadedSemanticFeatureExample",
    "MaterializationResult",
    "SemanticFeatureConfig",
    "SemanticFeatureMaterializationError",
    "load_semantic_feature_bundle",
    "load_semantic_feature_record",
    "materialize_semantic_program_features",
    "select_bounded_semantic_examples",
    "offset_tokenizer_for_worker",
    "tokenize_with_offsets",
    "tokenizer_checkpoint_identity",
]
