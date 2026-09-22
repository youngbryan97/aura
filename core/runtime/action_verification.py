"""Observed-effect verification for canonical consequential actions."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from core.governance.will import ActionDomain
from core.runtime.skill_contract import ActionExpectation

logger = logging.getLogger(__name__)

EffectVerifier = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]

_READ_ONLY_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MAX_MANIFEST_ENTRIES = 10_000
_MAX_HASHED_BYTES = 128 * 1024 * 1024


def default_action_expectation(
    domain: ActionDomain,
    action_name: str,
) -> ActionExpectation:
    return ActionExpectation(
        objective=f"complete {action_name} through the {domain.value} effect lane",
        acceptance_criteria=["effect_verified"],
        required_evidence=["verification_evidence.observation"],
        repair_hint="observe the requested effect or run a domain-specific verifier",
        rollback_hint="use the action receipt rollback target when compensation is supported",
        allow_partial=True,
    )


async def capture_pre_action_state(
    domain: ActionDomain,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    if domain != ActionDomain.FILE_WRITE:
        return {}
    operation = str(params.get("op") or "write").strip().lower()
    if operation not in {"move", "copy", "delete"}:
        return {}
    path = params.get("path")
    if path is None:
        return {}
    source = await asyncio.to_thread(_expanded_path, path)
    if operation == "delete":
        return await asyncio.to_thread(
            _path_presence_snapshot,
            source,
        )
    return await asyncio.to_thread(_path_snapshot, source)


async def observe_action_effect(
    domain: ActionDomain,
    params: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    pre_state: Mapping[str, Any] | None = None,
    verifier: EffectVerifier | None = None,
    verifier_timeout_s: float = 5.0,
) -> dict[str, Any]:
    automatic = await _automatic_observation(
        domain,
        params,
        result,
        pre_state=dict(pre_state or {}),
    )
    evidence: dict[str, Any] = {
        "observation": automatic,
        "automatic_effect_verified": bool(automatic.get("effect_verified", False)),
    }
    effect_verified = bool(automatic.get("effect_verified", False))

    if verifier is not None:
        verifier_context = {
            "domain": domain.value,
            "params": dict(params),
            "result": dict(result),
            "pre_state": dict(pre_state or {}),
            "automatic_observation": dict(automatic),
        }
        verifier_result = await _run_verifier(
            verifier,
            verifier_context,
            timeout_s=verifier_timeout_s,
        )
        evidence["custom_verifier"] = verifier_result
        effect_verified = bool(verifier_result.get("effect_verified", False))

    evidence["effect_verified"] = effect_verified
    return {
        "effect_verified": effect_verified,
        "verification_evidence": evidence,
        "criteria_results": {"effect_verified": effect_verified},
    }


async def _automatic_observation(
    domain: ActionDomain,
    params: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    pre_state: Mapping[str, Any],
) -> dict[str, Any]:
    action_ok = bool(result.get("ok", False))
    if not action_ok:
        return {"effect_verified": False, "reason": "execution_failed"}

    if domain == ActionDomain.FILE_WRITE:
        return await _observe_file_effect(params, result=result, pre_state=pre_state)
    if domain == ActionDomain.TOOL_EXECUTION and "argv" in params:
        exit_code = result.get("exit_code")
        return {
            "effect_verified": exit_code == 0,
            "kind": "process_exit",
            "exit_code": exit_code,
        }
    if domain in {
        ActionDomain.NETWORK_CALL,
        ActionDomain.CLOUD_CALL,
        ActionDomain.CLOUD_FALLBACK,
    }:
        method = str(params.get("method") or "GET").strip().upper()
        status_code = int(result.get("status_code") or 0)
        read_response_observed = method in _READ_ONLY_HTTP_METHODS and 100 <= status_code < 400
        return {
            "effect_verified": read_response_observed,
            "kind": "http_response",
            "method": method,
            "status_code": status_code,
            "transport_verified": 100 <= status_code < 600,
            "reason": (
                "read_response_observed"
                if read_response_observed
                else "mutating_network_effect_requires_readback"
            ),
        }
    if domain == ActionDomain.MEMORY_WRITE:
        verified = bool(
            result.get("record_id")
            and result.get("receipt_id")
            and int(result.get("bytes_written") or 0) > 0
        )
        return {
            "effect_verified": verified,
            "kind": "memory_receipt",
            "record_id": result.get("record_id"),
            "receipt_id": result.get("receipt_id"),
            "bytes_written": int(result.get("bytes_written") or 0),
        }
    if domain == ActionDomain.STATE_MUTATION:
        verified = bool(
            result.get("receipt_id")
            and result.get("key")
            and result.get("readback_verified") is True
        )
        return {
            "effect_verified": verified,
            "kind": "state_readback",
            "key": result.get("key"),
            "receipt_id": result.get("receipt_id"),
            "readback_verified": result.get("readback_verified") is True,
        }
    if domain == ActionDomain.SELF_MODIFICATION:
        verified = bool(result.get("applied") and result.get("canary_passed"))
        return {
            "effect_verified": verified,
            "kind": "self_modification_canary",
            "applied": bool(result.get("applied", False)),
            "canary_passed": bool(result.get("canary_passed", False)),
        }
    if domain == ActionDomain.TOOL_EXECUTION:
        verdict = result.get("expectation_verdict")
        if not isinstance(verdict, Mapping):
            raw_evidence = result.get("verification_evidence")
            verdict = (
                raw_evidence.get("expectation_verdict")
                if isinstance(raw_evidence, Mapping)
                else None
            )
        expectation_receipt_id = result.get("expectation_receipt_id")
        verified = bool(
            result.get("effect_verified") is True
            and isinstance(verdict, Mapping)
            and verdict.get("passed") is True
            and expectation_receipt_id
        )
        return {
            "effect_verified": verified,
            "kind": "capability_expectation_receipt",
            "expectation_receipt_id": expectation_receipt_id,
            "expectation_verdict_passed": bool(
                isinstance(verdict, Mapping) and verdict.get("passed") is True
            ),
            "downstream_effect_claimed": result.get("effect_verified") is True,
            "reason": (
                "capability_effect_and_expectation_receipt_verified"
                if verified
                else "capability_effect_claim_lacks_passed_durable_expectation_receipt"
            ),
        }
    return {
        "effect_verified": False,
        "kind": domain.value,
        "reason": "transport_completed_without_observed_effect",
    }


async def _observe_file_effect(
    params: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    pre_state: Mapping[str, Any],
) -> dict[str, Any]:
    operation = str(params.get("op") or "write").strip().lower()
    source = await asyncio.to_thread(_expanded_path, params.get("path") or "")
    destination_raw = params.get("destination")

    if operation == "delete":
        absent = not await asyncio.to_thread(os.path.lexists, source)
        existed_before = pre_state.get("exists") is True
        return {
            "effect_verified": absent,
            "kind": "file_delete_readback",
            "path": str(source),
            "absent": absent,
            "existed_before": existed_before,
            "changed": bool(existed_before and absent),
        }

    if operation in {"copy", "move"}:
        if destination_raw is None:
            return {
                "effect_verified": False,
                "kind": f"file_{operation}_readback",
                "reason": "destination_missing",
            }
        actual_destination = result.get("destination") or destination_raw
        destination, requested_destination = await asyncio.gather(
            asyncio.to_thread(_expanded_path, actual_destination),
            asyncio.to_thread(_expanded_path, destination_raw),
        )
        destination_state = await asyncio.to_thread(_path_snapshot, destination)
        expected_state = dict(pre_state)
        equivalent = _snapshots_equivalent(expected_state, destination_state)
        source_absent = not await asyncio.to_thread(os.path.lexists, source)
        verified = equivalent and (operation != "move" or source_absent)
        return {
            "effect_verified": verified,
            "kind": f"file_{operation}_readback",
            "source": str(source),
            "destination": str(destination),
            "content_equivalent": equivalent,
            "source_absent": source_absent,
            "requested_destination": str(requested_destination),
            "destination_state": destination_state,
        }

    state = await asyncio.to_thread(_path_snapshot, source)
    verified = bool(state.get("exists", False))
    expected_sha256 = ""
    if "text" in params:
        expected_sha256 = hashlib.sha256(
            str(params["text"]).encode(str(params.get("encoding") or "utf-8"))
        ).hexdigest()
        verified = verified and state.get("sha256") == expected_sha256
    elif "payload" in params:
        expected_sha256 = hashlib.sha256(bytes(params["payload"])).hexdigest()
        verified = verified and state.get("sha256") == expected_sha256
    elif "obj" in params:
        payload_matches = await asyncio.to_thread(
            _json_envelope_payload_matches,
            source,
            params["obj"],
        )
        state["payload_matches"] = payload_matches
        verified = verified and payload_matches
    return {
        "effect_verified": verified,
        "kind": "file_write_readback",
        "path": str(source),
        "expected_sha256": expected_sha256,
        "state": state,
    }


def _expanded_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


async def _run_verifier(
    verifier: EffectVerifier,
    context: Mapping[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    timeout = min(120.0, max(0.1, float(timeout_s)))

    async def invoke() -> Mapping[str, Any]:
        if inspect.iscoroutinefunction(verifier):
            value = await verifier(dict(context))
        else:
            value = await asyncio.to_thread(verifier, dict(context))
            if inspect.isawaitable(value):
                value = await value
        if not isinstance(value, Mapping):
            raise TypeError("effect verifier must return a mapping")
        return value

    started = time.monotonic()
    try:
        checked = await asyncio.wait_for(invoke(), timeout=timeout)
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        return {
            "effect_verified": False,
            "error_type": type(exc).__qualname__,
            "error": str(exc)[:320],
            "verifier": _callable_name(verifier),
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
    result = dict(checked)
    claimed = result.get("effect_verified") is True
    supporting_keys = {
        "artifact_hash",
        "comparison",
        "evidence",
        "observation",
        "observed_state",
        "readback",
        "receipt_id",
    }
    if claimed and not any(_has_supporting_value(result.get(key)) for key in supporting_keys):
        result["effect_verified"] = False
        result["error"] = "verifier_evidence_missing"
    result["verifier"] = _callable_name(verifier)
    result["duration_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    return result


def _path_snapshot(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"exists": False, "path": str(path)}
    try:
        if path.is_symlink():
            target = os.readlink(path)
            return {
                "exists": True,
                "kind": "symlink",
                "path": str(path),
                "target": target,
                "fingerprint": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            }
        if path.is_file():
            size = path.stat().st_size
            if size > _MAX_HASHED_BYTES:
                return {
                    "exists": True,
                    "kind": "file",
                    "path": str(path),
                    "size": size,
                    "truncated": True,
                    "fingerprint": "",
                }
            digest = _hash_file(path)
            return {
                "exists": True,
                "kind": "file",
                "path": str(path),
                "size": size,
                "sha256": digest,
                "fingerprint": f"file:{size}:{digest}",
            }
        if path.is_dir():
            manifest: list[tuple[str, str, int, str]] = []
            hashed_bytes = 0
            truncated = False
            for entry in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
                if len(manifest) >= _MAX_MANIFEST_ENTRIES:
                    truncated = True
                    break
                relative = entry.relative_to(path).as_posix()
                if entry.is_symlink():
                    manifest.append((relative, "symlink", 0, os.readlink(entry)))
                elif entry.is_dir():
                    manifest.append((relative, "dir", 0, ""))
                elif entry.is_file():
                    size = entry.stat().st_size
                    if hashed_bytes + size > _MAX_HASHED_BYTES:
                        truncated = True
                        break
                    manifest.append((relative, "file", size, _hash_file(entry)))
                    hashed_bytes += size
            fingerprint = ""
            if not truncated:
                fingerprint = hashlib.sha256(
                    json.dumps(manifest, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            return {
                "exists": True,
                "kind": "directory",
                "path": str(path),
                "entries": len(manifest),
                "hashed_bytes": hashed_bytes,
                "truncated": truncated,
                "fingerprint": fingerprint,
            }
        return {"exists": True, "kind": "other", "path": str(path)}
    except OSError as exc:
        return {
            "exists": True,
            "path": str(path),
            "error_type": type(exc).__qualname__,
            "error": str(exc)[:320],
        }


def _path_presence_snapshot(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"exists": False, "path": str(path)}
    try:
        if path.is_symlink():
            kind = "symlink"
        elif path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            kind = "other"
        return {"exists": True, "kind": kind, "path": str(path)}
    except OSError as exc:
        return {
            "exists": True,
            "path": str(path),
            "error_type": type(exc).__qualname__,
            "error": str(exc)[:320],
        }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshots_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("exists") is True
        and right.get("exists") is True
        and left.get("kind") == right.get("kind")
        and left.get("fingerprint")
        and left.get("fingerprint") == right.get("fingerprint")
        and not left.get("truncated", False)
        and not right.get("truncated", False)
    )


def _json_envelope_payload_matches(path: Path, expected: Any) -> bool:
    try:
        from core.runtime.atomic_writer import read_json_envelope

        envelope = read_json_envelope(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug(
            "verification could not read the envelope at %s (%s: %s); "
            "reporting no match",
            path,
            type(exc).__name__,
            exc,
        )
        return False
    return bool(envelope.get("payload") == expected)


def _callable_name(value: Any) -> str:
    module = str(getattr(value, "__module__", "") or "")
    name = str(
        getattr(value, "__qualname__", "")
        or getattr(value, "__name__", "")
        or type(value).__qualname__
    )
    return f"{module}.{name}".strip(".")[:240]


def _has_supporting_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, bytearray, list, tuple, set, frozenset, Mapping)):
        return bool(value)
    return True


__all__ = [
    "EffectVerifier",
    "capture_pre_action_state",
    "default_action_expectation",
    "observe_action_effect",
]
