"""Source-bound shadow execution for the frozen compositional semantic tissue."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from core.cognition.procedure import get_procedure_registry
from core.learning.compositional_semantic_qualification import (
    COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS,
    compositional_semantic_activation_errors,
)
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
    SemanticProgramObservationError,
    execute_compositional_semantic_observation,
)
from core.learning.semantic_public_inputs import semantic_public_character_inputs
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.flags import FlagKind, declare
from core.runtime.lockdep import checked_lock

COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA: Final = (
    "aura.compositional_semantic_shadow.v1"
)
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
LEGACY_ARTIFACT_DIRECTORY: Final = (
    REPO_ROOT / "artifacts/rlc/semantic_program_27b_compositional_v14"
)
ARTIFACT_DIRECTORY: Final = (
    REPO_ROOT / "artifacts/rlc/semantic_program_27b_frozen_path_v1"
)
DEFAULT_ACTIVATION_PATH: Final = ARTIFACT_DIRECTORY / "activation.json"
ACTIVE_ACTIVATION_PATH: Final = (
    REPO_ROOT / "training/fused-model/compositional-semantic-active.json"
)
TRANSDUCER_PATH: Final = ARTIFACT_DIRECTORY / "transducer.json"
SOURCE_REPORT_PATH: Final = LEGACY_ARTIFACT_DIRECTORY / "source_campaign.json"
SOURCE_VERIFICATION_PATH: Final = LEGACY_ARTIFACT_DIRECTORY / "verification.json"
ENDOGENOUS_VERIFICATION_PATH: Final = (
    LEGACY_ARTIFACT_DIRECTORY / "endogenous_runtime_verification.json"
)
_TOKENIZER_IDENTITY_FILES: Final = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
)
_FLAG_SHADOW_AVAILABLE = declare(
    "AURA_COMPOSITIONAL_SEMANTIC_SHADOW",
    kind=FlagKind.BOOL,
    default=True,
    description="Permit source-bound compositional semantic shadow observations.",
    owner="core.brain.llm.compositional_semantic_shadow",
)
_FLAG_LIVE_SHADOW = declare(
    "AURA_COMPOSITIONAL_SEMANTIC_LIVE_SHADOW",
    kind=FlagKind.BOOL,
    default=False,
    description="Observe eligible foreground turns with the resident semantic tissue.",
    owner="core.brain.llm.compositional_semantic_shadow",
)
_OBSERVATIONS: deque[dict[str, Any]] = deque(maxlen=64)
_OBSERVATIONS_LOCK = checked_lock("core.brain.llm.compositional_semantic_shadow.singleton")


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


def _dependency_signature(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int, int], ...]:
    result = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
        result.append((str(resolved), stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(result)


def active_compositional_semantic_activation_path() -> Path:
    """Resolve an explicit candidate, then the operational or default package."""

    candidate_value = str(os.getenv("AURA_COMPOSITIONAL_SEMANTIC_ACTIVATION", "")).strip()
    if candidate_value:
        requested = Path(candidate_value).expanduser()
        if requested.is_symlink():
            raise RuntimeError("compositional semantic activation cannot be a symlink")
        candidate = requested.resolve(strict=True)
        root = REPO_ROOT.resolve(strict=True)
        if not candidate.is_file() or not candidate.is_relative_to(root):
            raise RuntimeError("compositional semantic activation is outside the repository")
        return candidate
    return ACTIVE_ACTIVATION_PATH if ACTIVE_ACTIVATION_PATH.exists() else DEFAULT_ACTIVATION_PATH


def _shadow_dependencies(activation_path: Path) -> tuple[Path, ...]:
    try:
        activation, _raw = _read_json(
            activation_path,
            max_bytes=4 * 1024 * 1024,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return (activation_path,)
    paths = [activation_path]
    transducer = activation.get("transducer")
    if isinstance(transducer, dict) and isinstance(transducer.get("path"), str):
        paths.append(REPO_ROOT / transducer["path"])
    evidence = activation.get("evidence")
    if isinstance(evidence, dict):
        paths.extend(
            REPO_ROOT / record["path"]
            for record in evidence.values()
            if isinstance(record, dict) and isinstance(record.get("path"), str)
        )
    paths.extend(REPO_ROOT / relative for relative in COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS)
    return tuple(dict.fromkeys(paths))


def _model_identity_dependencies(model_path: Path) -> tuple[Path, ...]:
    return tuple(
        model_path / name
        for name in _TOKENIZER_IDENTITY_FILES
        if (model_path / name).is_file()
    )


@lru_cache(maxsize=2)
def _cached_shadow_status(
    model_path: str,
    activation_path: str,
    _signature: tuple[tuple[str, int, int, int], ...],
) -> dict[str, Any]:
    activation, _activation_raw = _read_json(Path(activation_path), max_bytes=4 * 1024 * 1024)
    selected_model = Path(model_path).expanduser().resolve(strict=True)
    errors = compositional_semantic_activation_errors(
        activation,
        repo_root=REPO_ROOT,
        selected_model_path=selected_model,
    )
    if errors:
        raise RuntimeError(",".join(errors))
    tokenizer_identity = tokenizer_checkpoint_identity(selected_model)
    if (
        tokenizer_identity.get("identity_sha256")
        != activation.get("model", {}).get("tokenizer_identity_sha256")
    ):
        raise RuntimeError("compositional semantic tokenizer differs")
    transducer = activation["transducer"]
    model = activation["model"]
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA,
        "available": True,
        "mode": "shadow",
        "serving_authority": False,
        "package_id": activation["package_id"],
        "activation_sha256": activation["activation_sha256"],
        "activation_path": str(Path(activation_path).resolve(strict=True)),
        "model_path": str(selected_model),
        "tokenizer_identity_sha256": tokenizer_identity["identity_sha256"],
        "representation_basis_sha256": model["representation_basis_sha256"],
        "transducer_path": str((REPO_ROOT / transducer["path"]).resolve(strict=True)),
        "transducer_receipt_sha256": transducer["receipt_sha256"],
        "measured": activation["measured"],
        "composition_policy": activation["composition_policy"],
        "claim_boundary": activation["claim_boundary"],
    }
    return {**body, "receipt_sha256": _sha(body)}


def compositional_semantic_shadow_status(model_path: str | Path) -> dict[str, Any]:
    """Return availability only while evidence, code, tokenizer, and model agree."""

    if not bool(_FLAG_SHADOW_AVAILABLE.value()):
        return {"available": False, "reason": "compositional_semantic_shadow_disabled"}
    try:
        selected_model = Path(model_path).expanduser().resolve(strict=True)
        resolved = str(selected_model)
        activation_path = active_compositional_semantic_activation_path().resolve(strict=True)
        return deepcopy(
            _cached_shadow_status(
                resolved,
                str(activation_path),
                _dependency_signature(
                    (
                        *_shadow_dependencies(activation_path),
                        *_model_identity_dependencies(selected_model),
                    )
                ),
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "reason": (
                "compositional_semantic_shadow_unavailable:"
                f"{type(exc).__name__}:{exc}"
            ),
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


def _resolve_real_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=True)


def compositional_semantic_live_shadow_enabled() -> bool:
    """Whether foreground cognition should spend the resident observation pass."""

    return bool(_FLAG_LIVE_SHADOW.value())


def _record_observation(result: dict[str, Any]) -> None:
    record = {
        "observed_at_ns": time.time_ns(),
        **deepcopy(result),
    }
    with _OBSERVATIONS_LOCK:
        _OBSERVATIONS.append(record)


def compositional_semantic_shadow_observations() -> list[dict[str, Any]]:
    """Return the bounded diagnostic ledger; it is never an answer input."""

    with _OBSERVATIONS_LOCK:
        return deepcopy(list(_OBSERVATIONS))


async def observe_resident_compositional_semantics(
    prompt: str,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Observe through the already-resident cortex without creating a client."""

    if not semantic_public_character_inputs(prompt).literals:
        result = {
            "eligible": False,
            "attempted": False,
            "ok": False,
            "reason": "compositional_semantic_no_public_inputs",
        }
        _record_observation(result)
        return result

    from core.brain.llm.mlx_client import clients_snapshot
    from core.brain.llm.model_registry import get_runtime_model_path

    selected_model = _resolve_real_path(get_runtime_model_path())
    status = compositional_semantic_shadow_status(selected_model)
    if status.get("available") is not True:
        result = {
            "eligible": True,
            "attempted": False,
            "ok": False,
            "reason": str(status.get("reason") or "compositional_semantic_shadow_unavailable"),
        }
        _record_observation(result)
        return result
    matching = []
    for _registry_key, client in clients_snapshot():
        try:
            candidate = _resolve_real_path(str(getattr(client, "model_path", "")))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if candidate == selected_model and not bool(getattr(client, "_closed", False)):
            matching.append(client)
    if len(matching) != 1:
        result = {
            "eligible": True,
            "attempted": False,
            "ok": False,
            "reason": (
                "compositional_semantic_resident_client_missing"
                if not matching
                else "compositional_semantic_resident_client_ambiguous"
            ),
        }
        _record_observation(result)
        return result
    result = await execute_compositional_semantic_shadow(
        client=matching[0],
        prompt=prompt,
        timeout_s=timeout_s,
    )
    _record_observation(result)
    return result


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
    model = _load_transducer(
        str(status["transducer_path"]),
        str(status["transducer_receipt_sha256"]),
    )
    if model.inference_step_limit(len(character_inputs.literals)) is None:
        return {
            "eligible": False,
            "attempted": False,
            "ok": False,
            "reason": "compositional_semantic_public_input_count_unsupported",
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
            procedure_registry=get_procedure_registry(),
        )
    except SemanticProgramDecodeRejectedError as exc:
        return {
            "eligible": True,
            "attempted": True,
            "ok": False,
            "reason": str(exc),
            "activation_receipt": status,
        }
    except SemanticProgramObservationError as exc:
        from core.brain.llm.latent_cortex.runtime_identity import (
            worker_representation_basis,
        )

        observed_basis = worker_representation_basis(receipt["model_basis"])
        observed_basis_sha256 = _sha(observed_basis)
        return {
            "eligible": True,
            "attempted": True,
            "ok": False,
            "reason": str(exc),
            "observed_representation_basis": observed_basis,
            "observed_representation_basis_sha256": observed_basis_sha256,
            "expected_representation_basis_sha256": status[
                "representation_basis_sha256"
            ],
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
        "procedure_id": getattr(getattr(outcome, "procedure", None), "procedure_id", None),
        "worker_hidden_state_sha256": receipt["hidden_state_sha256"],
        "result_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "result": semantic_value_to_json(outcome.execution.result),
        "text": text,
    }
    return {**body, "receipt_sha256": _sha(body)}


__all__ = [
    "ACTIVE_ACTIVATION_PATH",
    "COMPOSITIONAL_SEMANTIC_SHADOW_SCHEMA",
    "DEFAULT_ACTIVATION_PATH",
    "ENDOGENOUS_VERIFICATION_PATH",
    "active_compositional_semantic_activation_path",
    "compositional_semantic_live_shadow_enabled",
    "compositional_semantic_shadow_observations",
    "execute_compositional_semantic_shadow",
    "compositional_semantic_shadow_status",
    "observe_resident_compositional_semantics",
]
