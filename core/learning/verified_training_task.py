"""Proof-grade envelopes for Aura's training-only verifier curricula.

This module is intentionally independent of the frontier evaluation task
registry.  It freezes the public task before answer disclosure, commits a
salted verifier-only answer, verifies a detached task-issuer attestation only
after both recurrent outputs are sealed, and replays grading through a frozen
local scorer registry.  No API here accepts or generates a signing private
key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Never, Protocol, cast

from core.learning.answer_channel_curriculum import (
    ANSWER_CHANNEL_FAMILIES,
    ANSWER_CHANNEL_VERSION,
    MAX_ANSWER_CHANNEL_DEPTH,
    AnswerChannelTask,
)
from core.learning.recurrence_curriculum import (
    CURRICULUM_VERSION,
    MAX_TRAINING_DEPTH,
    RECURRENCE_TRAINING_FAMILIES,
    RecurrenceTrainingTask,
)
from core.learning.verifiable_tasks import (
    KNOWLEDGE_FREE,
    NEEDS_RETRIEVAL,
    VerifiableTask,
)

PUBLIC_TRAINING_TASK_SCHEMA = "aura.verified_training.public_task.v1"
SEALED_TRAINING_ANSWER_SCHEMA = "aura.verified_training.sealed_answer.v1"
ANSWER_AUTHORITY_PAYLOAD_SCHEMA = (
    "aura.verified_training.answer_authority_payload.v1"
)
ANSWER_AUTHORITY_SCHEMA = "aura.verified_training.answer_authority.v1"
SCORER_REGISTRY_SCHEMA = "aura.verified_training.scorer_registry.v1"
SCORE_RECEIPT_SCHEMA = "aura.verified_training.score_receipt.v1"
SCORER_REGISTRY_VERSION = "2026.07.27.1"
TASK_ISSUER = "task_issuer"
CAMPAIGN_ROLE_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.campaign_role_attestation.v1"
)
CAMPAIGN_ROLE_PAYLOAD_SCHEMA = "aura.latent_cortex.campaign_role_payload.v2"

PASS_NAMES = ("pass_0", "pass_1")
MIN_ANSWER_NONCE_BYTES = 32
MAX_ANSWER_NONCE_BYTES = 256
MAX_TASK_ID_BYTES = 512
MAX_PROMPT_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 4096
MAX_JSON_STRING_BYTES = 64 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_DEPTH = 1_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answers",
        "expected",
        "ground_truth",
        "label",
        "labels",
        "nonce",
        "secret",
        "secrets",
        "solution",
        "solutions",
    }
)


class VerifiedTrainingTaskError(ValueError):
    """Stable fail-closed task, authority, or scorer protocol error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTrainingTaskError(code)


class VerifiedTrainingTrustPolicy(Protocol):
    """Public-only trust policy surface required by answer verification."""

    policy_sha256: str
    document: dict[str, Any]

    def role_pin(self, role: str) -> dict[str, str]: ...


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one deterministic JSON value without importing cortex packages."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("value_not_canonical_json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{role}_invalid")
    return value


def _require_identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{role}_invalid")
    return value


def _require_text(value: Any, *, role: str, maximum_bytes: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        _fail(f"{role}_invalid")
    return value


def _validate_json_graph(
    value: Any,
    *,
    role: str,
    depth: int = 0,
    nodes: list[int] | None = None,
) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES:
        _fail(f"{role}_too_complex")
    if depth > MAX_JSON_DEPTH:
        _fail(f"{role}_too_deep")
    if value is None or type(value) in {bool, int}:
        if type(value) is int and not -(1 << 63) <= value <= (1 << 63) - 1:
            _fail(f"{role}_integer_out_of_bounds")
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{role}_non_finite_number")
        return
    if isinstance(value, str):
        if "\x00" in value or len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
            _fail(f"{role}_string_invalid")
        return
    if type(value) is list:
        for item in value:
            _validate_json_graph(
                item,
                role=role,
                depth=depth + 1,
                nodes=nodes,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or "\x00" in key
                or len(key.encode("utf-8")) > 256
            ):
                _fail(f"{role}_key_invalid")
            _validate_json_graph(
                item,
                role=role,
                depth=depth + 1,
                nodes=nodes,
            )
        return
    _fail(f"{role}_non_json_type")


def _clone_json(value: Any, *, role: str) -> Any:
    _validate_json_graph(value, role=role)
    try:
        encoded = canonical_json_bytes(value)
        if len(encoded) > MAX_METADATA_BYTES and role.endswith("parameters"):
            _fail(f"{role}_too_large")
        return json.loads(encoded)
    except VerifiedTrainingTaskError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_invalid")


def _strict_json_bytes(raw: bytes, *, role: str) -> Any:
    if not isinstance(raw, bytes) or not raw:
        _fail(f"{role}_bytes_invalid")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def parse_int(text: str) -> int:
        digits = text.removeprefix("-")
        if not digits or len(digits) > 19:
            _fail(f"{role}_integer_out_of_bounds")
        value = int(text)
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            _fail(f"{role}_integer_out_of_bounds")
        return value

    def parse_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            _fail(f"{role}_non_finite_number")
        return value

    def reject_constant(_text: str) -> None:
        _fail(f"{role}_non_finite_number")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
    except VerifiedTrainingTaskError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        _fail(f"{role}_json_invalid")
    _validate_json_graph(decoded, role=role)
    if canonical_json_bytes(decoded) != raw:
        _fail(f"{role}_not_canonical")
    return decoded


def _forbid_public_secrets(value: Any, *, path: str = "public_parameters") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                _fail(f"public_task_secret_field:{path}.{key}")
            _forbid_public_secrets(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _forbid_public_secrets(item, path=f"{path}[{index}]")


def _extract_relaxed_final(response: str) -> str:
    plain = re.sub(r"[*`#]", "", response)
    for pattern in (
        r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)",
        r"(?:the )?answer(?: is)?\s*[:=]\s*(.+?)(?:\n|$)",
        r"\\boxed\{([^}]*)\}",
    ):
        found = re.findall(pattern, plain, flags=re.IGNORECASE | re.DOTALL)
        if found:
            return found[-1].strip()
    lines = [line.strip() for line in plain.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _as_decimal(value: Any) -> Decimal | None:
    if type(value) in {int, float}:
        if type(value) is float and not math.isfinite(value):
            return None
        text = str(value)
    else:
        text = str(value or "").strip()
        text = text.replace(",", "").replace("$", "").rstrip(".")
        text = re.sub(r"^[^\d\-+.]*", "", text)
        text = re.sub(r"[^\d\-+./eE]*$", "", text)
    if not text:
        return None
    try:
        if "/" in text:
            parts = text.split("/")
            if len(parts) != 2:
                return None
            denominator = Decimal(parts[1])
            return Decimal(parts[0]) / denominator if denominator else None
        return Decimal(text)
    # not a failure: a value that is not a number is not one this can read.
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def _decimal_json(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.normalize(), "f")


def _score_numeric(
    response: str,
    expected: Any,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    produced = _as_decimal(_extract_relaxed_final(response))
    target = _as_decimal(expected)
    tolerance = _as_decimal(parameters.get("tolerance", 1e-6))
    if produced is None or target is None or tolerance is None or tolerance < 0:
        return {"correct": False, "parsed": _decimal_json(produced), "reason": "unparseable"}
    delta = abs(produced - target)
    threshold = max(tolerance, tolerance * max(abs(produced), abs(target)))
    return {
        "correct": delta <= threshold,
        "parsed": _decimal_json(produced),
        "expected": _decimal_json(target),
    }


def _score_ordered(
    response: str,
    expected: Any,
    _parameters: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _extract_relaxed_final(response)
    produced = [part.strip().lower() for part in re.split(r"[,\s]+", raw) if part.strip()]
    if type(expected) is not list:
        _fail("scorer_expected_ordered_invalid")
    target = [str(item).strip().lower() for item in expected]
    return {"correct": produced == target, "parsed": produced, "expected": target}


def _score_exact_set(
    response: str,
    expected: Any,
    _parameters: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _extract_relaxed_final(response)
    produced = {part.strip().lower() for part in re.split(r"[,\s]+", raw) if part.strip()}
    if type(expected) is not list:
        _fail("scorer_expected_set_invalid")
    target = {str(item).strip().lower() for item in expected}
    return {
        "correct": produced == target,
        "parsed": sorted(produced),
        "expected": sorted(target),
    }


def _score_boolean(
    response: str,
    expected: Any,
    _parameters: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _extract_relaxed_final(response).strip().lower()
    if raw in {"true", "yes", "1", "t"}:
        produced: bool | None = True
    elif raw in {"false", "no", "0", "f"}:
        produced = False
    else:
        produced = None
    if produced is None:
        return {"correct": False, "parsed": None, "reason": "unparseable"}
    if type(expected) is not bool:
        _fail("scorer_expected_boolean_invalid")
    return {"correct": produced is expected, "parsed": produced}


def _extract_json_value(response: str) -> Any:
    raw = _extract_relaxed_final(response)
    for opening, closing in (("{", "}"), ("[", "]")):
        start = raw.find(opening)
        if start < 0:
            continue
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(raw)):
            character = raw[index]
            if escaped:
                escaped = False
                continue
            if character == "\\" and quoted:
                escaped = True
                continue
            if character == '"':
                quoted = not quoted
            if quoted:
                continue
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth == 0:
                    raw = raw[start : index + 1]
                    break
        break
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_text: str) -> Never:
        raise ValueError("non-finite number")

    try:
        produced = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
        _validate_json_graph(produced, role="json_scorer_output")
        return produced
    # not a failure: text that does not parse is not the shape this was reading for.
    except (TypeError, ValueError, RecursionError, VerifiedTrainingTaskError):
        return None


def _score_json(
    response: str,
    expected: Any,
    _parameters: Mapping[str, Any],
) -> dict[str, Any]:
    produced = _extract_json_value(response)
    if produced is None:
        return {"correct": False, "parsed": None, "reason": "unparseable"}
    return {"correct": produced == expected, "parsed": produced}


def _parse_terminal_json_object(response: str) -> dict[str, Any] | None:
    if (
        not isinstance(response, str)
        or not response.strip()
        or "\x00" in response
        or len(response.encode("utf-8")) > MAX_RESPONSE_BYTES
        or response.count("FINAL_ANSWER:") != 1
    ):
        return None
    lines = response.rstrip().splitlines()
    if not lines or not lines[-1].startswith("FINAL_ANSWER:"):
        return None
    encoded = lines[-1].removeprefix("FINAL_ANSWER:").strip()
    if not encoded:
        return None
    try:
        raw = encoded.encode("utf-8")

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        def parse_int(text: str) -> int:
            digits = text.removeprefix("-")
            if not digits or len(digits) > 19:
                raise ValueError("integer out of bounds")
            value = int(text)
            if not -(1 << 63) <= value <= (1 << 63) - 1:
                raise ValueError("integer out of bounds")
            return value

        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=lambda _text: (_ for _ in ()).throw(ValueError("float")),
            parse_constant=lambda _text: (_ for _ in ()).throw(ValueError("constant")),
        )
        _validate_json_graph(parsed, role="terminal_answer")
    # not a failure: text that does not parse is not the shape this was reading for.
    except (TypeError, ValueError, RecursionError, UnicodeError):
        return None
    return parsed if type(parsed) is dict else None


def _score_exact_terminal_json(
    response: str,
    expected: Any,
    _parameters: Mapping[str, Any],
) -> dict[str, Any]:
    if type(expected) is not dict:
        _fail("scorer_expected_json_object_invalid")
    produced = _parse_terminal_json_object(response)
    if produced is None:
        return {
            "correct": False,
            "parsed": None,
            "expected": expected,
            "reason": "unparseable",
        }
    return {"correct": produced == expected, "parsed": produced, "expected": expected}


_SCORERS: dict[str, Callable[[str, Any, Mapping[str, Any]], dict[str, Any]]] = {
    "verifiable.numeric.v1": _score_numeric,
    "verifiable.ordered.v1": _score_ordered,
    "verifiable.exact_set.v1": _score_exact_set,
    "verifiable.boolean.v1": _score_boolean,
    "verifiable.json.v1": _score_json,
    "recurrence.exact_terminal_json.v1": _score_exact_terminal_json,
    "answer_channel.exact_terminal_json.v1": _score_exact_terminal_json,
}
SCORER_REGISTRY: Mapping[
    str,
    Callable[[str, Any, Mapping[str, Any]], dict[str, Any]],
] = MappingProxyType(_SCORERS)


def _callable_source_sha256(value: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(inspect.unwrap(value)).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise VerifiedTrainingTaskError("scorer_source_unavailable") from exc
    return _sha256_bytes(source)


def verified_training_task_source_sha256() -> str:
    path = Path(__file__).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        _fail("verified_training_task_source_unavailable")
    return _sha256_bytes(path.read_bytes())


def scorer_registry_identity() -> dict[str, Any]:
    """Return the current source-bound immutable scorer registry identity."""

    body = {
        "schema": SCORER_REGISTRY_SCHEMA,
        "version": SCORER_REGISTRY_VERSION,
        "module": "core.learning.verified_training_task",
        "module_source_sha256": verified_training_task_source_sha256(),
        "scorers": [
            {
                "scorer_id": scorer_id,
                "implementation_sha256": _callable_source_sha256(scorer),
            }
            for scorer_id, scorer in sorted(SCORER_REGISTRY.items())
        ],
    }
    return {**body, "registry_sha256": _digest(body)}


def validate_scorer_registry_identity(raw: Any) -> dict[str, Any]:
    identity = _clone_json(raw, role="scorer_registry_identity")
    expected = scorer_registry_identity()
    if identity != expected:
        _fail("scorer_registry_identity_mismatch")
    return cast(dict[str, Any], identity)


@dataclass(frozen=True, slots=True, repr=False)
class PublicVerifiedTrainingTask:
    """Immutable canonical candidate-visible training task commitment."""

    task_commitment_sha256: str
    _canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.task_commitment_sha256, role="public_task_commitment")
        if not isinstance(self._canonical_bytes, bytes):
            _fail("public_task_bytes_invalid")
        document = _strict_json_bytes(self._canonical_bytes, role="public_task")
        if document.get("task_commitment_sha256") != self.task_commitment_sha256:
            _fail("public_task_commitment_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _strict_json_bytes(self._canonical_bytes, role="public_task"))

    def canonical_bytes(self) -> bytes:
        return bytes(self._canonical_bytes)


@dataclass(frozen=True, slots=True, repr=False)
class SealedTrainingAnswer:
    """Verifier-only answer material; repr and public task reveal no secret."""

    answer_commitment_sha256: str
    task_core_sha256: str
    _canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.answer_commitment_sha256, role="sealed_answer_commitment")
        _require_sha256(self.task_core_sha256, role="sealed_answer_task_core")
        if not isinstance(self._canonical_bytes, bytes):
            _fail("sealed_answer_bytes_invalid")
        material = _strict_json_bytes(self._canonical_bytes, role="sealed_answer")
        if _digest(material) != self.answer_commitment_sha256:
            _fail("sealed_answer_commitment_mismatch")
        if material.get("task_core_sha256") != self.task_core_sha256:
            _fail("sealed_answer_task_core_mismatch")


def _task_core(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: document[key]
        for key in (
            "schema",
            "task_type",
            "task_id",
            "prompt",
            "domain",
            "depth",
            "public_parameters",
            "scorer_id",
            "scorer_registry_sha256",
        )
    }


def _scorer_for_task(task: Any) -> tuple[str, dict[str, Any], Any]:
    if type(task) is VerifiableTask:
        scorer_id = f"verifiable.{task.grader}.v1"
        if scorer_id not in SCORER_REGISTRY:
            _fail("verifiable_task_scorer_unsupported")
        metadata = _clone_json(task.metadata, role="verifiable_public_parameters")
        public = {"knowledge": task.knowledge, "grader_metadata": metadata}
        return scorer_id, public, _clone_json(task.expected, role="verifiable_expected")
    if type(task) is RecurrenceTrainingTask:
        public = {
            "knowledge": "parametric",
            "family": task.family,
            "seed": task.seed,
            "curriculum_version": CURRICULUM_VERSION,
        }
        return (
            "recurrence.exact_terminal_json.v1",
            public,
            _clone_json(task.expected, role="recurrence_expected"),
        )
    if type(task) is AnswerChannelTask:
        public = {
            "knowledge": "parametric",
            "family": task.family,
            "seed": task.seed,
            "curriculum_version": ANSWER_CHANNEL_VERSION,
            "claim_boundary": "format_parseability_only_not_reasoning_gain",
        }
        return (
            "answer_channel.exact_terminal_json.v1",
            public,
            _clone_json(task.expected, role="answer_channel_expected"),
        )
    _fail("verified_training_task_type_unsupported")


def _validate_scorer_contract(
    scorer_id: str,
    expected: Any,
    parameters: Mapping[str, Any],
) -> None:
    if type(parameters) is not dict:
        _fail("scorer_parameters_invalid")
    if scorer_id == "verifiable.numeric.v1":
        if set(parameters) - {"tolerance"}:
            _fail("numeric_scorer_parameters_invalid")
        tolerance = parameters.get("tolerance", 1e-6)
        if (
            isinstance(tolerance, bool)
            or _as_decimal(tolerance) is None
            or cast(Decimal, _as_decimal(tolerance)) < 0
            or _as_decimal(expected) is None
        ):
            _fail("numeric_scorer_contract_invalid")
        return
    if parameters:
        _fail("unused_scorer_parameters_forbidden")
    if scorer_id in {"verifiable.ordered.v1", "verifiable.exact_set.v1"}:
        if type(expected) is not list or any(
            type(item) not in {str, int, float, bool} for item in expected
        ):
            _fail("sequence_scorer_expected_invalid")
        return
    if scorer_id == "verifiable.boolean.v1":
        if type(expected) is not bool:
            _fail("boolean_scorer_expected_invalid")
        return
    if scorer_id == "verifiable.json.v1":
        _validate_json_graph(expected, role="json_scorer_expected")
        return
    if scorer_id in {
        "recurrence.exact_terminal_json.v1",
        "answer_channel.exact_terminal_json.v1",
    }:
        if type(expected) is not dict:
            _fail("terminal_json_scorer_expected_invalid")
        _validate_json_graph(expected, role="terminal_json_scorer_expected")
        return
    _fail("scorer_contract_unknown")


def _task_type_and_coordinates(task: Any) -> tuple[str, str, str, int]:
    if type(task) is VerifiableTask:
        task_type = "verifiable"
        task_id, domain, depth = task.task_id, task.domain, task.depth
        if task.knowledge not in {KNOWLEDGE_FREE, NEEDS_RETRIEVAL}:
            _fail("verifiable_task_knowledge_invalid")
    elif type(task) is RecurrenceTrainingTask:
        task_type = "recurrence_training"
        task_id, domain, depth = task.task_id, task.family, task.depth
        if task.family not in RECURRENCE_TRAINING_FAMILIES:
            _fail("recurrence_task_family_invalid")
        if not 1 <= depth <= MAX_TRAINING_DEPTH:
            _fail("recurrence_task_depth_invalid")
    elif type(task) is AnswerChannelTask:
        task_type = "answer_channel"
        task_id, domain, depth = task.task_id, task.family, task.depth
        if task.family not in ANSWER_CHANNEL_FAMILIES:
            _fail("answer_channel_task_family_invalid")
        if not 1 <= depth <= MAX_ANSWER_CHANNEL_DEPTH:
            _fail("answer_channel_task_depth_invalid")
    else:
        _fail("verified_training_task_type_unsupported")
    _require_text(task_id, role="public_task_id", maximum_bytes=MAX_TASK_ID_BYTES)
    _require_identifier(domain, role="public_task_domain")
    if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
        _fail("public_task_depth_invalid")
    prompt = _require_text(task.prompt, role="public_task_prompt", maximum_bytes=MAX_PROMPT_BYTES)
    return task_type, task_id, prompt, depth


def build_verified_training_task(
    task: VerifiableTask | RecurrenceTrainingTask | AnswerChannelTask,
    *,
    answer_nonce: bytes,
) -> tuple[PublicVerifiedTrainingTask, SealedTrainingAnswer]:
    """Adapt one exact training task into public and verifier-only halves."""

    if (
        not isinstance(answer_nonce, bytes)
        or not MIN_ANSWER_NONCE_BYTES <= len(answer_nonce) <= MAX_ANSWER_NONCE_BYTES
    ):
        _fail("answer_nonce_invalid")
    task_type, task_id, prompt, depth = _task_type_and_coordinates(task)
    scorer_id, public_parameters, expected = _scorer_for_task(task)
    public_parameters = _clone_json(public_parameters, role="public_parameters")
    _forbid_public_secrets(public_parameters)
    registry = scorer_registry_identity()
    core = {
        "schema": PUBLIC_TRAINING_TASK_SCHEMA,
        "task_type": task_type,
        "task_id": task_id,
        "prompt": prompt,
        "domain": _require_identifier(task.domain, role="public_task_domain"),
        "depth": depth,
        "public_parameters": public_parameters,
        "scorer_id": scorer_id,
        "scorer_registry_sha256": registry["registry_sha256"],
    }
    task_core_sha256 = _digest(core)
    scorer_parameters = (
        public_parameters.get("grader_metadata", {})
        if task_type == "verifiable"
        else {}
    )
    _validate_scorer_contract(scorer_id, expected, scorer_parameters)
    answer_material = {
        "schema": SEALED_TRAINING_ANSWER_SCHEMA,
        "task_core_sha256": task_core_sha256,
        "task_id": task_id,
        "scorer_id": scorer_id,
        "scorer_registry_sha256": registry["registry_sha256"],
        "scorer_parameters": scorer_parameters,
        "expected": expected,
        "answer_nonce_b64": base64.b64encode(answer_nonce).decode("ascii"),
    }
    answer_commitment = _digest(answer_material)
    unsigned = {
        **core,
        "task_core_sha256": task_core_sha256,
        "answer_commitment_sha256": answer_commitment,
    }
    document = {**unsigned, "task_commitment_sha256": _digest(unsigned)}
    public = PublicVerifiedTrainingTask(
        task_commitment_sha256=document["task_commitment_sha256"],
        _canonical_bytes=canonical_json_bytes(document),
    )
    sealed = SealedTrainingAnswer(
        answer_commitment_sha256=answer_commitment,
        task_core_sha256=task_core_sha256,
        _canonical_bytes=canonical_json_bytes(answer_material),
    )
    validate_public_training_task(public)
    return public, sealed


def _load_document(raw: Any, *, role: str) -> dict[str, Any]:
    if isinstance(raw, PublicVerifiedTrainingTask):
        decoded = _strict_json_bytes(raw.canonical_bytes(), role=role)
    elif isinstance(raw, bytes):
        decoded = _strict_json_bytes(raw, role=role)
    else:
        decoded = _clone_json(raw, role=role)
    if type(decoded) is not dict:
        _fail(f"{role}_not_object")
    return cast(dict[str, Any], decoded)


def _verify_task_issuer_attestation(
    policy: VerifiedTrainingTrustPolicy,
    raw: Any,
    *,
    expected_payload: Mapping[str, Any],
) -> None:
    """Verify the existing campaign envelope using only public key material."""

    attestation = _clone_json(raw, role="answer_authority_attestation")
    if type(attestation) is not dict or set(attestation) != {
        "schema",
        "signed_payload",
        "signed_payload_sha256",
        "signature_b64",
    }:
        _fail("answer_authority_attestation_schema_invalid")
    if attestation.get("schema") != CAMPAIGN_ROLE_ATTESTATION_SCHEMA:
        _fail("answer_authority_attestation_version_invalid")
    signed_payload = attestation.get("signed_payload")
    if type(signed_payload) is not dict or set(signed_payload) != {
        "schema",
        "policy_sha256",
        "campaign_name",
        "protocol_sha256",
        "role",
        "signer_id",
        "operation",
        "purpose",
        "idempotency_key",
        "signed_at_unix",
        "payload",
    }:
        _fail("answer_authority_signed_payload_schema_invalid")
    try:
        pin = policy.role_pin(TASK_ISSUER)
        policy_document = policy.document
        policy_sha256 = _require_sha256(
            policy.policy_sha256,
            role="answer_authority_policy",
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise VerifiedTrainingTaskError(
            "answer_authority_trust_policy_invalid"
        ) from exc
    if (
        signed_payload.get("schema") != CAMPAIGN_ROLE_PAYLOAD_SCHEMA
        or signed_payload.get("policy_sha256") != policy_sha256
        or signed_payload.get("campaign_name") != policy_document.get("campaign_name")
        or signed_payload.get("protocol_sha256") != policy_document.get("protocol_sha256")
        or signed_payload.get("role") != TASK_ISSUER
        or signed_payload.get("signer_id") != pin.get("signer_id")
        or not isinstance(signed_payload.get("operation"), str)
        or not isinstance(signed_payload.get("purpose"), str)
        or not isinstance(signed_payload.get("idempotency_key"), str)
        or signed_payload.get("payload")
        != _clone_json(expected_payload, role="answer_authority_expected_payload")
    ):
        _fail("answer_authority_attestation_identity_mismatch")
    signed_at = signed_payload.get("signed_at_unix")
    not_before = policy_document.get("not_before_unix")
    expires_at = policy_document.get("expires_at_unix")
    if (
        type(signed_at) is not int
        or type(not_before) is not int
        or type(expires_at) is not int
        or not not_before <= signed_at < expires_at
    ):
        _fail("answer_authority_attestation_time_invalid")
    signed_bytes = canonical_json_bytes(signed_payload)
    if attestation.get("signed_payload_sha256") != _sha256_bytes(signed_bytes):
        _fail("answer_authority_attestation_digest_mismatch")
    public_key_b64 = pin.get("public_key_b64")
    try:
        public_key_raw = base64.b64decode(public_key_b64, validate=True)
        signature = base64.b64decode(attestation.get("signature_b64"), validate=True)
    except (TypeError, ValueError, binascii.Error):
        _fail("answer_authority_attestation_signature_invalid")
    if (
        len(public_key_raw) != 32
        or base64.b64encode(public_key_raw).decode("ascii") != public_key_b64
        or pin.get("key_id") != _sha256_bytes(public_key_raw)
    ):
        _fail("answer_authority_task_issuer_key_invalid")
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            signature,
            signed_bytes,
        )
    except (InvalidSignature, ValueError):
        _fail("answer_authority_attestation_signature_invalid")


def validate_public_training_task(raw: Any) -> dict[str, Any]:
    """Validate one exact current-registry public task commitment."""

    document = _load_document(raw, role="public_task")
    required = {
        "schema",
        "task_type",
        "task_id",
        "prompt",
        "domain",
        "depth",
        "public_parameters",
        "scorer_id",
        "scorer_registry_sha256",
        "task_core_sha256",
        "answer_commitment_sha256",
        "task_commitment_sha256",
    }
    if set(document) != required or document.get("schema") != PUBLIC_TRAINING_TASK_SCHEMA:
        _fail("public_task_schema_invalid")
    task_type = document.get("task_type")
    if task_type not in {"verifiable", "recurrence_training", "answer_channel"}:
        _fail("public_task_type_invalid")
    _require_text(document.get("task_id"), role="public_task_id", maximum_bytes=MAX_TASK_ID_BYTES)
    _require_text(document.get("prompt"), role="public_task_prompt", maximum_bytes=MAX_PROMPT_BYTES)
    domain = _require_identifier(document.get("domain"), role="public_task_domain")
    depth = document.get("depth")
    if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
        _fail("public_task_depth_invalid")
    parameters = document.get("public_parameters")
    if type(parameters) is not dict:
        _fail("public_task_parameters_invalid")
    _forbid_public_secrets(parameters)
    if task_type == "verifiable":
        if set(parameters) != {"knowledge", "grader_metadata"}:
            _fail("verifiable_public_parameters_invalid")
        if parameters["knowledge"] not in {KNOWLEDGE_FREE, NEEDS_RETRIEVAL}:
            _fail("verifiable_public_knowledge_invalid")
        if type(parameters["grader_metadata"]) is not dict:
            _fail("verifiable_public_metadata_invalid")
    elif task_type == "recurrence_training":
        if set(parameters) != {"knowledge", "family", "seed", "curriculum_version"}:
            _fail("recurrence_public_parameters_invalid")
        if (
            parameters["knowledge"] != "parametric"
            or parameters["family"] != domain
            or domain not in RECURRENCE_TRAINING_FAMILIES
            or parameters["curriculum_version"] != CURRICULUM_VERSION
            or type(parameters["seed"]) is not int
            or parameters["seed"] < 0
            or depth > MAX_TRAINING_DEPTH
        ):
            _fail("recurrence_public_parameters_invalid")
        expected_task_id = (
            f"recurrence-{domain}-d{depth}-s{parameters['seed']}"
        )
        if document["task_id"] != expected_task_id:
            _fail("recurrence_public_task_id_mismatch")
    else:
        expected_keys = {
            "knowledge",
            "family",
            "seed",
            "curriculum_version",
            "claim_boundary",
        }
        if set(parameters) != expected_keys:
            _fail("answer_channel_public_parameters_invalid")
        if (
            parameters["knowledge"] != "parametric"
            or parameters["family"] != domain
            or domain not in ANSWER_CHANNEL_FAMILIES
            or parameters["curriculum_version"] != ANSWER_CHANNEL_VERSION
            or parameters["claim_boundary"]
            != "format_parseability_only_not_reasoning_gain"
            or type(parameters["seed"]) is not int
            or parameters["seed"] < 0
            or depth > MAX_ANSWER_CHANNEL_DEPTH
        ):
            _fail("answer_channel_public_parameters_invalid")
        expected_task_id = (
            f"answer-channel-{domain}-d{depth}-s{parameters['seed']}"
        )
        if document["task_id"] != expected_task_id:
            _fail("answer_channel_public_task_id_mismatch")
    scorer_id = _require_identifier(document.get("scorer_id"), role="public_task_scorer")
    if scorer_id not in SCORER_REGISTRY:
        _fail("public_task_scorer_unknown")
    expected_prefix = {
        "verifiable": "verifiable.",
        "recurrence_training": "recurrence.",
        "answer_channel": "answer_channel.",
    }[cast(str, task_type)]
    if not scorer_id.startswith(expected_prefix):
        _fail("public_task_scorer_type_mismatch")
    registry = scorer_registry_identity()
    if document.get("scorer_registry_sha256") != registry["registry_sha256"]:
        _fail("public_task_scorer_registry_mismatch")
    core = _task_core(document)
    if document.get("task_core_sha256") != _digest(core):
        _fail("public_task_core_digest_mismatch")
    _require_sha256(document.get("answer_commitment_sha256"), role="public_task_answer_commitment")
    unsigned = dict(document)
    observed = unsigned.pop("task_commitment_sha256", None)
    if observed != _digest(unsigned):
        _fail("public_task_commitment_digest_mismatch")
    return document


def seal_training_output(response: str) -> str:
    if (
        not isinstance(response, str)
        or "\x00" in response
        or len(response.encode("utf-8")) > MAX_RESPONSE_BYTES
    ):
        _fail("training_output_invalid")
    return _sha256_bytes(response.encode("utf-8"))


def _validate_output_seals(raw: Any) -> dict[str, str]:
    seals = _clone_json(raw, role="sealed_outputs")
    if type(seals) is not dict or set(seals) != set(PASS_NAMES):
        _fail("sealed_outputs_schema_invalid")
    for name in PASS_NAMES:
        _require_sha256(seals.get(name), role=f"sealed_output_{name}")
    return cast(dict[str, str], seals)


def prepare_answer_authority_payload(
    public_task: PublicVerifiedTrainingTask | Mapping[str, Any] | bytes,
    sealed_answer: SealedTrainingAnswer,
    *,
    sealed_outputs: Mapping[str, str],
) -> dict[str, Any]:
    """Reveal answer material only after two exact output seals are supplied."""

    public = validate_public_training_task(public_task)
    if not isinstance(sealed_answer, SealedTrainingAnswer):
        _fail("sealed_answer_type_invalid")
    material = cast(
        dict[str, Any],
        _strict_json_bytes(sealed_answer._canonical_bytes, role="sealed_answer"),
    )
    if (
        _digest(material) != public["answer_commitment_sha256"]
        or material.get("task_core_sha256") != public["task_core_sha256"]
        or material.get("task_id") != public["task_id"]
        or material.get("scorer_id") != public["scorer_id"]
        or material.get("scorer_registry_sha256")
        != public["scorer_registry_sha256"]
    ):
        _fail("sealed_answer_public_task_mismatch")
    parameters = (
        public["public_parameters"].get("grader_metadata", {})
        if public["task_type"] == "verifiable"
        else {}
    )
    if material.get("scorer_parameters") != parameters:
        _fail("sealed_answer_scorer_parameters_mismatch")
    nonce = material.get("answer_nonce_b64")
    try:
        nonce_raw = base64.b64decode(nonce, validate=True)
    except (TypeError, ValueError, binascii.Error):
        _fail("sealed_answer_nonce_invalid")
    if (
        not MIN_ANSWER_NONCE_BYTES <= len(nonce_raw) <= MAX_ANSWER_NONCE_BYTES
        or base64.b64encode(nonce_raw).decode("ascii") != nonce
    ):
        _fail("sealed_answer_nonce_invalid")
    output_seals = _validate_output_seals(sealed_outputs)
    return {
        "schema": ANSWER_AUTHORITY_PAYLOAD_SCHEMA,
        "public_task_sha256": public["task_commitment_sha256"],
        "task_core_sha256": public["task_core_sha256"],
        "task_id": public["task_id"],
        "answer_commitment_sha256": public["answer_commitment_sha256"],
        "scorer_id": public["scorer_id"],
        "scorer_registry_sha256": public["scorer_registry_sha256"],
        "scorer_parameters": material["scorer_parameters"],
        "expected": material["expected"],
        "answer_nonce_b64": material["answer_nonce_b64"],
        "sealed_outputs": output_seals,
    }


def _validate_answer_authority_payload(
    public: Mapping[str, Any],
    raw: Any,
) -> dict[str, Any]:
    payload = _clone_json(raw, role="answer_authority_payload")
    required = {
        "schema",
        "public_task_sha256",
        "task_core_sha256",
        "task_id",
        "answer_commitment_sha256",
        "scorer_id",
        "scorer_registry_sha256",
        "scorer_parameters",
        "expected",
        "answer_nonce_b64",
        "sealed_outputs",
    }
    if type(payload) is not dict or set(payload) != required:
        _fail("answer_authority_payload_schema_invalid")
    if payload.get("schema") != ANSWER_AUTHORITY_PAYLOAD_SCHEMA:
        _fail("answer_authority_payload_version_invalid")
    for field_name, public_name in (
        ("public_task_sha256", "task_commitment_sha256"),
        ("task_core_sha256", "task_core_sha256"),
        ("task_id", "task_id"),
        ("answer_commitment_sha256", "answer_commitment_sha256"),
        ("scorer_id", "scorer_id"),
        ("scorer_registry_sha256", "scorer_registry_sha256"),
    ):
        if payload.get(field_name) != public.get(public_name):
            _fail(f"answer_authority_{field_name}_mismatch")
    expected_parameters = (
        public["public_parameters"].get("grader_metadata", {})
        if public["task_type"] == "verifiable"
        else {}
    )
    if payload.get("scorer_parameters") != expected_parameters:
        _fail("answer_authority_scorer_parameters_mismatch")
    _validate_output_seals(payload.get("sealed_outputs"))
    nonce = payload.get("answer_nonce_b64")
    try:
        nonce_raw = base64.b64decode(nonce, validate=True)
    except (TypeError, ValueError, binascii.Error):
        _fail("answer_authority_nonce_invalid")
    if (
        not MIN_ANSWER_NONCE_BYTES <= len(nonce_raw) <= MAX_ANSWER_NONCE_BYTES
        or base64.b64encode(nonce_raw).decode("ascii") != nonce
    ):
        _fail("answer_authority_nonce_invalid")
    _validate_scorer_contract(
        cast(str, payload["scorer_id"]),
        payload["expected"],
        cast(dict[str, Any], payload["scorer_parameters"]),
    )
    material = {
        "schema": SEALED_TRAINING_ANSWER_SCHEMA,
        "task_core_sha256": payload["task_core_sha256"],
        "task_id": payload["task_id"],
        "scorer_id": payload["scorer_id"],
        "scorer_registry_sha256": payload["scorer_registry_sha256"],
        "scorer_parameters": payload["scorer_parameters"],
        "expected": payload["expected"],
        "answer_nonce_b64": payload["answer_nonce_b64"],
    }
    if _digest(material) != public["answer_commitment_sha256"]:
        _fail("answer_authority_answer_commitment_mismatch")
    return cast(dict[str, Any], payload)


def assemble_answer_authority(
    public_task: PublicVerifiedTrainingTask | Mapping[str, Any] | bytes,
    *,
    payload: Mapping[str, Any],
    task_issuer_attestation: Mapping[str, Any],
    policy: VerifiedTrainingTrustPolicy,
) -> dict[str, Any]:
    """Verify the detached task-issuer signature and seal the authority."""

    public = validate_public_training_task(public_task)
    normalized_payload = _validate_answer_authority_payload(public, payload)
    _verify_task_issuer_attestation(
        policy,
        task_issuer_attestation,
        expected_payload=normalized_payload,
    )
    body = {
        "schema": ANSWER_AUTHORITY_SCHEMA,
        "policy_sha256": policy.policy_sha256,
        "payload": normalized_payload,
        "task_issuer_attestation": _clone_json(
            task_issuer_attestation,
            role="answer_authority_attestation",
        ),
    }
    return {**body, "authority_sha256": _digest(body)}


def validate_answer_authority(
    public_task: PublicVerifiedTrainingTask | Mapping[str, Any] | bytes,
    authority: Mapping[str, Any],
    *,
    policy: VerifiedTrainingTrustPolicy,
) -> dict[str, Any]:
    public = validate_public_training_task(public_task)
    document = _clone_json(authority, role="answer_authority")
    if type(document) is not dict or set(document) != {
        "schema",
        "policy_sha256",
        "payload",
        "task_issuer_attestation",
        "authority_sha256",
    }:
        _fail("answer_authority_schema_invalid")
    if (
        document.get("schema") != ANSWER_AUTHORITY_SCHEMA
        or document.get("policy_sha256") != policy.policy_sha256
    ):
        _fail("answer_authority_identity_mismatch")
    payload = _validate_answer_authority_payload(public, document.get("payload"))
    _verify_task_issuer_attestation(
        policy,
        document.get("task_issuer_attestation"),
        expected_payload=payload,
    )
    unsigned = dict(document)
    observed = unsigned.pop("authority_sha256", None)
    if observed != _digest(unsigned):
        _fail("answer_authority_digest_mismatch")
    return cast(dict[str, Any], document)


def score_verified_training_outputs(
    public_task: PublicVerifiedTrainingTask | Mapping[str, Any] | bytes,
    authority: Mapping[str, Any],
    *,
    outputs: Mapping[str, str],
    policy: VerifiedTrainingTrustPolicy,
) -> dict[str, Any]:
    """Replay both sealed outputs against the independently revealed answer."""

    public = validate_public_training_task(public_task)
    verified_authority = validate_answer_authority(
        public,
        authority,
        policy=policy,
    )
    if not isinstance(outputs, Mapping) or set(outputs) != set(PASS_NAMES):
        _fail("training_outputs_schema_invalid")
    scorer_id = public["scorer_id"]
    scorer = SCORER_REGISTRY.get(scorer_id)
    if scorer is None:
        _fail("training_output_scorer_missing")
    payload = cast(dict[str, Any], verified_authority["payload"])
    results: dict[str, Any] = {}
    for pass_name in PASS_NAMES:
        response = outputs.get(pass_name)
        if not isinstance(response, str):
            _fail(f"training_output_{pass_name}_invalid")
        response_sha256 = seal_training_output(response)
        if response_sha256 != payload["sealed_outputs"][pass_name]:
            _fail(f"training_output_{pass_name}_seal_mismatch")
        verdict = _clone_json(
            scorer(
                response,
                payload["expected"],
                payload["scorer_parameters"],
            ),
            role=f"training_output_{pass_name}_verdict",
        )
        if type(verdict) is not dict or type(verdict.get("correct")) is not bool:
            _fail("training_output_verdict_invalid")
        results[pass_name] = {
            "response_sha256": response_sha256,
            "verdict": verdict,
        }
    body = {
        "schema": SCORE_RECEIPT_SCHEMA,
        "public_task_sha256": public["task_commitment_sha256"],
        "answer_authority_sha256": verified_authority["authority_sha256"],
        "scorer_id": scorer_id,
        "scorer_registry_sha256": public["scorer_registry_sha256"],
        "results": results,
    }
    return {**body, "receipt_sha256": _digest(body)}


__all__ = [
    "ANSWER_AUTHORITY_PAYLOAD_SCHEMA",
    "ANSWER_AUTHORITY_SCHEMA",
    "PASS_NAMES",
    "PUBLIC_TRAINING_TASK_SCHEMA",
    "SCORER_REGISTRY",
    "SCORER_REGISTRY_SCHEMA",
    "SCORE_RECEIPT_SCHEMA",
    "SEALED_TRAINING_ANSWER_SCHEMA",
    "PublicVerifiedTrainingTask",
    "SealedTrainingAnswer",
    "VerifiedTrainingTaskError",
    "VerifiedTrainingTrustPolicy",
    "assemble_answer_authority",
    "build_verified_training_task",
    "prepare_answer_authority_payload",
    "score_verified_training_outputs",
    "scorer_registry_identity",
    "seal_training_output",
    "validate_answer_authority",
    "validate_public_training_task",
    "validate_scorer_registry_identity",
    "verified_training_task_source_sha256",
]
