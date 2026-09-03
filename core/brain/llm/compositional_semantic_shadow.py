"""Source-bound shadow execution for the frozen compositional semantic tissue."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_feature_materialization import (
    offset_tokenizer_for_worker,
    tokenize_with_offsets,
    tokenizer_checkpoint_identity,
)
from core.learning.semantic_program_ir import semantic_value_to_json
from core.learning.semantic_program_runtime import (
    SemanticProgramDecodeRejectedError,
    execute_compositional_semantic_observation,
)
from core.learning.semantic_public_inputs import semantic_public_character_inputs
from core.runtime.file_read_gateway import read_stable_bytes

COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA: Final = (
    "aura.compositional_semantic_shadow.v1"
)
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
ARTIFACT_DIRECTORY: Final = (
    REPO_ROOT / "artifacts/rlc/semantic_program_27b_compositional_v13"
)
TRANSDUCER_PATH: Final = ARTIFACT_DIRECTORY / "transducer.json"
SOURCE_REPORT_PATH: Final = ARTIFACT_DIRECTORY / "source_campaign.json"
SOURCE_VERIFICATION_PATH: Final = ARTIFACT_DIRECTORY / "verification.json"
ENDOGENOUS_VERIFICATION_PATH: Final = (
    ARTIFACT_DIRECTORY / "endogenous_runtime_verification.json"
)
_FALSE_VALUES: Final = frozenset({"0", "false", "no", "off", "disabled"})
_TOKENIZER_IDENTITY_FILES: Final = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
)


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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    value = json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError("compositional semantic artifact is not an object")
    return value, payload


def _file_sha(path: Path) -> str:
    return hashlib.sha256(
        read_stable_bytes(path.resolve(strict=True), max_bytes=512 * 1024 * 1024)
    ).hexdigest()


def _logical_hash_matches(value: dict[str, Any], field: str) -> bool:
    expected = value.get(field)
    return isinstance(expected, str) and expected == _sha(
        {key: item for key, item in value.items() if key != field}
    )


def _dependency_signature(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int, int], ...]:
    result = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
        result.append((str(resolved), stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(result)


def _shadow_dependencies() -> tuple[Path, ...]:
    try:
        verification, _raw = _read_json(
            ENDOGENOUS_VERIFICATION_PATH,
            max_bytes=4 * 1024 * 1024,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return (
            TRANSDUCER_PATH,
            SOURCE_REPORT_PATH,
            SOURCE_VERIFICATION_PATH,
            ENDOGENOUS_VERIFICATION_PATH,
        )
    source_paths = verification.get("source_sha256s")
    extra = (
        tuple(REPO_ROOT / relative for relative in source_paths)
        if isinstance(source_paths, dict)
        else ()
    )
    return (
        TRANSDUCER_PATH,
        SOURCE_REPORT_PATH,
        SOURCE_VERIFICATION_PATH,
        ENDOGENOUS_VERIFICATION_PATH,
        *extra,
    )


def _model_identity_dependencies(model_path: Path) -> tuple[Path, ...]:
    return tuple(
        model_path / name
        for name in _TOKENIZER_IDENTITY_FILES
        if (model_path / name).is_file()
    )


@lru_cache(maxsize=2)
def _cached_shadow_status(
    model_path: str,
    _signature: tuple[tuple[str, int, int, int], ...],
) -> dict[str, Any]:
    from core.learning.semantic_program_endogenous_verification import (
        ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES,
    )

    transducer, transducer_raw = _read_json(TRANSDUCER_PATH, max_bytes=32 * 1024 * 1024)
    source_report, source_report_raw = _read_json(
        SOURCE_REPORT_PATH,
        max_bytes=32 * 1024 * 1024,
    )
    source_verification, _source_verification_raw = _read_json(
        SOURCE_VERIFICATION_PATH,
        max_bytes=8 * 1024 * 1024,
    )
    endogenous, endogenous_raw = _read_json(
        ENDOGENOUS_VERIFICATION_PATH,
        max_bytes=4 * 1024 * 1024,
    )
    source_body = {key: value for key, value in source_report.items() if key != "report_sha256"}
    if (
        source_report.get("report_sha256") != _sha(source_body)
        or not _logical_hash_matches(source_verification, "verification_sha256")
        or not _logical_hash_matches(endogenous, "verification_sha256")
        or source_verification.get("verified") is not True
        or endogenous.get("verified") is not True
        or source_verification.get("serving_authority") is not False
        or endogenous.get("serving_authority") is not False
        or endogenous.get("source_verification_sha256")
        != source_verification.get("verification_sha256")
        or source_verification.get("transducer_receipt_sha256")
        != transducer.get("training_receipt", {}).get("receipt_sha256")
        or source_verification.get("stored_file_sha256s", {}).get("model")
        != hashlib.sha256(transducer_raw).hexdigest()
        or source_verification.get("stored_file_sha256s", {}).get("source_report")
        != hashlib.sha256(source_report_raw).hexdigest()
        or set(endogenous.get("source_sha256s", {}))
        != set(ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES)
        or any(
            endogenous["source_sha256s"].get(relative)
            != _file_sha(REPO_ROOT / relative)
            for relative in ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES
        )
    ):
        raise RuntimeError("compositional semantic frozen evidence differs")

    selected_model = Path(model_path).expanduser().resolve(strict=True)
    expected_model = Path(str(source_verification.get("model_path") or "")).resolve(
        strict=True
    )
    if selected_model != expected_model:
        raise RuntimeError("compositional semantic resident model differs")
    tokenizer_identity = tokenizer_checkpoint_identity(selected_model)
    if (
        tokenizer_identity.get("identity_sha256")
        != source_verification.get("tokenizer_identity_sha256")
    ):
        raise RuntimeError("compositional semantic tokenizer differs")
    compatibility = source_report.get("representation_compatibility")
    if (
        not isinstance(compatibility, dict)
        or compatibility.get("hidden_states_changed") is not False
        or compatibility.get("serving_authority") is not False
        or compatibility.get("representation_basis_sha256")
        != endogenous.get("representation_basis_sha256")
    ):
        raise RuntimeError("compositional semantic representation evidence differs")
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA,
        "available": True,
        "mode": "shadow",
        "serving_authority": False,
        "model_path": str(selected_model),
        "tokenizer_identity_sha256": tokenizer_identity["identity_sha256"],
        "representation_basis_sha256": endogenous["representation_basis_sha256"],
        "transducer_receipt_sha256": source_verification[
            "transducer_receipt_sha256"
        ],
        "source_verification_sha256": source_verification["verification_sha256"],
        "endogenous_verification_sha256": endogenous["verification_sha256"],
        "endogenous_verification_file_sha256": hashlib.sha256(endogenous_raw).hexdigest(),
        "qualified_cohorts": sorted(endogenous["cohorts"]),
        "claim_boundary": endogenous["claim_boundary"],
    }
    return {**body, "receipt_sha256": _sha(body)}


def compositional_semantic_shadow_status(model_path: str | Path) -> dict[str, Any]:
    """Return availability only while evidence, code, tokenizer, and model agree."""

    if str(os.getenv("AURA_COMPOSITIONAL_SEMANTIC_SHADOW", "1")).strip().lower() in (
        _FALSE_VALUES
    ):
        return {"available": False, "reason": "compositional_semantic_shadow_disabled"}
    try:
        selected_model = Path(model_path).expanduser().resolve(strict=True)
        resolved = str(selected_model)
        return deepcopy(
            _cached_shadow_status(
                resolved,
                _dependency_signature(
                    (*_shadow_dependencies(), *_model_identity_dependencies(selected_model))
                ),
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"compositional_semantic_shadow_unavailable:{type(exc).__name__}",
        }


@lru_cache(maxsize=2)
def _load_transducer(path: str, expected_receipt: str) -> CompositionalSemanticProgramTransducer:
    payload, _raw = _read_json(Path(path), max_bytes=32 * 1024 * 1024)
    model = compositional_semantic_program_transducer_from_dict(payload)
    if model.receipt_sha256 != expected_receipt:
        raise RuntimeError("compositional semantic transducer receipt differs")
    return model


@lru_cache(maxsize=2)
def _load_offset_tokenizer(model_path: str) -> Any:
    from mlx_lm.utils import load_tokenizer

    return offset_tokenizer_for_worker(load_tokenizer(Path(model_path)))


def _render_result(value: Any) -> str:
    normalized = semantic_value_to_json(value)
    if type(normalized) is int:
        return str(normalized)
    return json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)


async def execute_compositional_semantic_shadow(
    *,
    client: Any,
    prompt: str,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Run one answer-blind resident observation without claiming its answer."""

    if type(prompt) is not str or not prompt.strip():
        raise ValueError("compositional semantic shadow prompt is invalid")
    character_inputs = semantic_public_character_inputs(prompt)
    if not character_inputs.literals:
        return {
            "eligible": False,
            "attempted": False,
            "ok": False,
            "reason": "compositional_semantic_no_public_inputs",
        }
    model_path = str(getattr(client, "model_path", "") or "")
    status = compositional_semantic_shadow_status(model_path)
    if status.get("available") is not True:
        return {
            "eligible": True,
            "attempted": False,
            "ok": False,
            "reason": str(status.get("reason") or "compositional_semantic_shadow_unavailable"),
        }
    tokenizer = _load_offset_tokenizer(model_path)
    token_ids, offsets = await asyncio.to_thread(tokenize_with_offsets, tokenizer, prompt)
    observation = await client.encode_hidden_sequence(
        prompt,
        timeout_s=max(1.0, float(timeout_s)),
        representation="lexical_mid_final_v1",
    )
    if observation is None:
        return {
            "eligible": True,
            "attempted": False,
            "ok": False,
            "reason": "compositional_semantic_resident_lane_busy",
        }
    if observation.get("token_ids") != token_ids:
        raise RuntimeError("compositional semantic local and worker tokens differ")
    receipt = observation.get("receipt")
    if not isinstance(receipt, dict) or not isinstance(receipt.get("model_basis"), dict):
        raise RuntimeError("compositional semantic worker basis is unavailable")
    model = _load_transducer(
        str(TRANSDUCER_PATH.resolve(strict=True)),
        str(status["transducer_receipt_sha256"]),
    )
    try:
        outcome = await asyncio.to_thread(
            execute_compositional_semantic_observation,
            model=model,
            source_text=prompt,
            source_token_ids=token_ids,
            offset_mapping=offsets,
            hidden_states=observation.get("hidden_states"),
            worker_model_basis=receipt["model_basis"],
            expected_representation_basis_sha256=str(
                status["representation_basis_sha256"]
            ),
        )
    except SemanticProgramDecodeRejectedError as exc:
        return {
            "eligible": True,
            "attempted": True,
            "ok": False,
            "reason": str(exc),
            "activation_receipt": status,
        }
    text = _render_result(outcome.execution.result)
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA,
        "eligible": True,
        "attempted": True,
        "ok": True,
        "mode": "shadow",
        "serving_authority": False,
        "activation_receipt_sha256": status["receipt_sha256"],
        "runtime_receipt_sha256": outcome.receipt["receipt_sha256"],
        "worker_hidden_state_sha256": receipt["hidden_state_sha256"],
        "result_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "result": semantic_value_to_json(outcome.execution.result),
        "text": text,
    }
    return {**body, "receipt_sha256": _sha(body)}


__all__ = [
    "COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA",
    "ENDOGENOUS_VERIFICATION_PATH",
    "execute_compositional_semantic_shadow",
    "compositional_semantic_shadow_status",
]
