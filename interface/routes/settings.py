"""Authenticated API for Aura's transactional runtime settings control plane."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_orchestrator
from core.runtime.settings_control_plane import (
    RuntimeSettingsStore,
    SettingsConflictError,
    SettingsControlPlaneError,
    SettingsIdempotencyError,
    SettingsIntegrityError,
    SettingsVersionError,
)
from core.runtime.settings_schema import (
    SCHEMA,
    SETTINGS_SCHEMA_NAME,
    SETTINGS_SCHEMA_VERSION,
    SettingDef,
)
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Settings")

router = APIRouter(prefix="/settings", tags=["settings"])
SETTINGS_PATCH_BODY = Body(...)
SETTINGS_RESET_BODY = Body(...)
SETTINGS_ROLLBACK_BODY = Body(...)
SETTINGS_APPLICATION_ACK_BODY = Body(...)
SETTINGS_CONFIRM_BODY = Body(...)
SETTINGS_CONFIRM_CANCEL_BODY = Body(...)

_SETTINGS_DIR = Path.home() / ".aura" / "data" / "settings"
_SETTINGS_PATH = Path(
    os.environ.get("AURA_SETTINGS_PATH", str(_SETTINGS_DIR / "runtime.json"))
).expanduser()
_SETTINGS_AUDIT_PATH = _SETTINGS_PATH.with_name(
    f"{_SETTINGS_PATH.stem}.audit.jsonl"
)
_SETTINGS_APPLICATION_AUDIT_PATH = _SETTINGS_PATH.with_name(
    f"{_SETTINGS_PATH.stem}.application.jsonl"
)

_SETTINGS_API_ERRORS = (
    KeyError,
    OSError,
    SettingsControlPlaneError,
    TypeError,
    ValueError,
)


class SettingsStore(RuntimeSettingsStore):
    """Compatibility name backed by the canonical core control plane."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        audit_path: str | Path | None = None,
        application_audit_path: str | Path | None = None,
    ) -> None:
        resolved_path = Path(path or _SETTINGS_PATH).expanduser()
        super().__init__(
            resolved_path,
            audit_path=audit_path or resolved_path.with_name(
                f"{resolved_path.stem}.audit.jsonl"
            ),
            application_audit_path=(
                application_audit_path
                or resolved_path.with_name(
                    f"{resolved_path.stem}.application.jsonl"
                )
            ),
        )


_STORE: SettingsStore | None = None

# Emergency containment reconfigures the resident orchestrator immediately.
# Autonomy is an intrinsic runtime invariant, not an operator-selected mode.
# Other settings are read by their owner at the next action/tick boundary.
# Voice and camera also have resident bridges because open hardware handles
# must change immediately rather than waiting for another action.
_RUNTIME_MODE_KEYS = frozenset({"safety.safe_mode"})
_VOICE_RUNTIME_KEYS = frozenset(
    {
        "voice.input_enabled",
        "voice.output_enabled",
        "voice.auto_listen",
    }
)
_CAMERA_RUNTIME_KEYS = frozenset({"permissions.camera"})


def _runtime_should_restrict(store: SettingsStore) -> bool:
    return bool(store.get("safety.safe_mode"))


def _apply_runtime_mode_from_settings(
    key: str,
    _previous: Any,
    _new: Any,
) -> dict[str, str]:
    if key not in _RUNTIME_MODE_KEYS:
        return {
            "owner": "safe_mode",
            "status": "unchanged",
            "detail": "setting is outside the runtime posture bridge",
        }
    try:
        from core.container import ServiceContainer
        from core.runtime.safe_mode import set_safe_mode

        peek = getattr(ServiceContainer, "peek", None)
        orchestrator = None
        if callable(peek):
            orchestrator = peek("orchestrator", default=None)
        if orchestrator is None:
            orchestrator = resolve_orchestrator()
        if orchestrator is None:
            return {
                "owner": "safe_mode",
                "status": "deferred",
                "detail": "persisted; orchestrator will apply the posture during boot",
            }
        restrict = _runtime_should_restrict(get_settings())
        set_safe_mode(orchestrator, restrict)
        logger.info(
            "Runtime posture %s applied from settings revision (%s).",
            "restricted" if restrict else "full",
            key,
        )
        return {
            "owner": "safe_mode",
            "status": "applied",
            "detail": "resident orchestrator posture updated",
        }
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        record_degradation("settings.runtime_mode_apply", exc)
        return {
            "owner": "safe_mode",
            "status": "failed",
            "detail": f"{type(exc).__name__}:{str(exc)[:180]}",
        }


def _apply_voice_setting(
    key: str,
    previous: Any,
    new: Any,
) -> dict[str, str]:
    """Bridge persisted voice policy to the one resident hardware owner."""

    if key not in _VOICE_RUNTIME_KEYS:
        return {
            "owner": "voice_runtime",
            "status": "unchanged",
            "detail": "setting is outside the resident voice bridge",
        }
    try:
        revocation: dict[str, Any] | None = None
        if key == "voice.input_enabled" and not bool(new):
            from core.voice.microphone_authority import get_microphone_authority

            # Invalidate every browser/native lease before asking individual
            # consumers to clean up their device handles. This closes ingress
            # immediately even if a remote browser needs one monitor tick to
            # receive the close frame and stop its MediaStream track.
            revocation = get_microphone_authority().revoke_all(
                reason="runtime_setting_disabled"
            )

        from core.senses.voice_engine import get_voice_engine

        voice = get_voice_engine()
        apply_setting = getattr(voice, "apply_runtime_setting", None)
        if not callable(apply_setting):
            return {
                "owner": "voice_runtime",
                "status": "failed",
                "detail": "resident voice engine lacks the runtime-settings contract",
            }
        result = apply_setting(key, previous, new)
        if not isinstance(result, dict):
            return {
                "owner": "voice_runtime",
                "status": "failed",
                "detail": "resident voice engine returned an invalid application receipt",
            }
        if revocation and int(revocation.get("revoked", 0)):
            result = dict(result)
            result["detail"] = (
                f"{result.get('detail', 'voice setting applied')}; revoked "
                f"{revocation['revoked']} active microphone lease(s)"
            )
        return result
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        record_degradation(
            "settings.voice_apply",
            exc,
            severity="warning",
            action="kept the durable voice preference and reported owner failure",
            enforce_failure_policy=False,
        )
        return {
            "owner": "voice_runtime",
            "status": "failed",
            "detail": f"{type(exc).__name__}:{str(exc)[:180]}",
        }


def _apply_camera_setting(
    key: str,
    _previous: Any,
    new: Any,
) -> dict[str, str]:
    """Bridge the durable camera gate to every resident capture transport."""
    if key not in _CAMERA_RUNTIME_KEYS:
        return {
            "owner": "camera_runtime",
            "status": "unchanged",
            "detail": "setting is outside the resident camera bridge",
        }
    try:
        from interface.routes.privacy import apply_camera_runtime_state

        state = apply_camera_runtime_state(
            bool(new),
            reason="applied from the transactional runtime settings control plane",
        )
        return {
            "owner": "camera_runtime",
            "status": "applied",
            "detail": (
                f"camera {state['mode']} via {state['transport']}; "
                f"native_capture={state['native_capture_enabled']}"
            ),
        }
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        record_degradation(
            "settings.camera_apply",
            exc,
            severity="warning",
            action="kept the durable camera preference and reported owner failure",
            enforce_failure_policy=False,
        )
        return {
            "owner": "camera_runtime",
            "status": "failed",
            "detail": f"{type(exc).__name__}:{str(exc)[:180]}",
        }


def get_settings() -> SettingsStore:
    global _STORE
    if _STORE is None:
        _STORE = SettingsStore()
        _STORE.subscribe(
            _apply_runtime_mode_from_settings,
            owner="safe_mode",
            keys=_RUNTIME_MODE_KEYS,
        )
        _STORE.subscribe(
            _apply_voice_setting,
            owner="voice_runtime",
            keys=_VOICE_RUNTIME_KEYS,
        )
        _STORE.subscribe(
            _apply_camera_setting,
            owner="camera_runtime",
            keys=_CAMERA_RUNTIME_KEYS,
        )
    return _STORE


def _schema_payload() -> list[dict[str, Any]]:
    return [
        {
            "key": setting.key,
            "label": setting.label,
            "section": setting.section,
            "default": setting.default,
            "explanation": setting.explanation,
            "type": setting.type_,
            "choices": list(setting.choices) if setting.choices else None,
            "min": setting.min_,
            "max": setting.max_,
            "owner": setting.owner,
            "apply_mode": setting.apply_mode,
            "mutable": setting.mutable,
        }
        for setting in SCHEMA
    ]


def _conflict_response(exc: SettingsConflictError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "settings_revision_conflict",
            "expected_revision": exc.expected_revision,
            "current_revision": exc.current_revision,
            "retryable": True,
        },
    )


def _settings_error_response(exc: Exception) -> JSONResponse:
    status_code = 409 if isinstance(
        exc,
        (
            SettingsIdempotencyError,
            SettingsIntegrityError,
            SettingsVersionError,
        ),
    ) else 422
    error = (
        "settings_request_id_reused"
        if isinstance(exc, SettingsIdempotencyError)
        else "settings_integrity_failed"
        if isinstance(exc, SettingsIntegrityError)
        else "settings_version_incompatible"
        if isinstance(exc, SettingsVersionError)
        else "settings_validation_failed"
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": str(exc)[:500],
            "retryable": False,
        },
    )


def _required_revision(payload: Any) -> int:
    if not isinstance(payload, dict):
        raise TypeError("settings request body must be an object")
    revision = payload.get("expected_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("expected_revision must be a non-negative integer")
    return revision


@router.get("")
async def get_all(_: None = Depends(_require_internal)) -> JSONResponse:
    store = get_settings()
    state = await asyncio.to_thread(store.describe)
    payload: dict[str, Any] = {**state}
    # Route-owned keys win over any same-named keys inside describe() —
    # spreading state LAST once silently replaced the `schema` alias (the
    # controls-panel schema list) with the store's own schema string.
    payload.update(
        {
            "control_plane": {
                "schema": SETTINGS_SCHEMA_NAME,
                "schema_version": SETTINGS_SCHEMA_VERSION,
                "cas_required": True,
                "atomic_patch": True,
                "rollback_supported": True,
            },
            "settings_schema": _schema_payload(),
            # `schema` remains as a compatibility alias for older internal clients.
            "schema": _schema_payload(),
        }
    )
    return JSONResponse(payload)


@router.patch("")
async def patch_settings(
    payload: dict[str, Any] = SETTINGS_PATCH_BODY,
    _: None = Depends(_require_internal),
) -> JSONResponse:
    store = get_settings()
    try:
        expected_revision = _required_revision(payload)
        changes = payload.get("changes")
        request_id = payload.get("request_id")
        result = await asyncio.to_thread(
            store.patch,
            changes,
            expected_revision=expected_revision,
            actor="authenticated_internal_settings_api",
            request_id=str(request_id) if request_id is not None else None,
        )
        return JSONResponse(result.public())
    except SettingsConflictError as exc:
        return _conflict_response(exc)
    except _SETTINGS_API_ERRORS as exc:
        record_degradation(
            "settings.patch",
            exc,
            severity="warning",
            action="rejected settings transaction without partial commit",
            enforce_failure_policy=False,
        )
        return _settings_error_response(exc)


@router.post("/reset")
async def reset_section(
    payload: dict[str, Any] = SETTINGS_RESET_BODY,
    _: None = Depends(_require_internal),
) -> JSONResponse:
    store = get_settings()
    try:
        expected_revision = _required_revision(payload)
        section = str(payload.get("section") or "").strip()
        if not section:
            raise ValueError("section is required")
        result = await asyncio.to_thread(
            store.reset_section,
            section,
            expected_revision=expected_revision,
            actor="authenticated_internal_settings_api",
            request_id=(
                str(payload["request_id"])
                if payload.get("request_id") is not None
                else None
            ),
        )
        return JSONResponse(result.public())
    except SettingsConflictError as exc:
        return _conflict_response(exc)
    except _SETTINGS_API_ERRORS as exc:
        record_degradation(
            "settings.reset",
            exc,
            severity="warning",
            action="rejected settings reset without partial commit",
            enforce_failure_policy=False,
        )
        return _settings_error_response(exc)


@router.post("/rollback")
async def rollback_settings(
    payload: dict[str, Any] = SETTINGS_ROLLBACK_BODY,
    _: None = Depends(_require_internal),
) -> JSONResponse:
    store = get_settings()
    try:
        expected_revision = _required_revision(payload)
        target_revision = payload.get("target_revision")
        if isinstance(target_revision, bool) or not isinstance(target_revision, int):
            raise ValueError("target_revision must be a non-negative integer")
        result = await asyncio.to_thread(
            store.rollback,
            target_revision,
            expected_revision=expected_revision,
            actor="authenticated_internal_settings_api",
            request_id=(
                str(payload["request_id"])
                if payload.get("request_id") is not None
                else None
            ),
        )
        return JSONResponse(result.public())
    except SettingsConflictError as exc:
        return _conflict_response(exc)
    except _SETTINGS_API_ERRORS as exc:
        record_degradation(
            "settings.rollback",
            exc,
            severity="warning",
            action="rejected settings rollback without changing current revision",
            enforce_failure_policy=False,
        )
        return _settings_error_response(exc)


@router.get("/integrity")
async def settings_integrity(_: None = Depends(_require_internal)) -> JSONResponse:
    try:
        report = await asyncio.to_thread(get_settings().verify_integrity)
        return JSONResponse(report)
    except _SETTINGS_API_ERRORS as exc:
        return _settings_error_response(exc)


@router.post("/application-ack")
async def acknowledge_settings_application(
    payload: dict[str, Any] = SETTINGS_APPLICATION_ACK_BODY,
    _: None = Depends(_require_internal),
) -> JSONResponse:
    try:
        if not isinstance(payload, dict):
            raise TypeError("settings acknowledgement body must be an object")
        result = await asyncio.to_thread(
            get_settings().acknowledge_application,
            str(payload.get("settings_receipt_hash") or ""),
            payload.get("acknowledgements"),
            actor="authenticated_desktop_settings_shell",
        )
        return JSONResponse(result)
    except _SETTINGS_API_ERRORS as exc:
        record_degradation(
            "settings.application_ack",
            exc,
            severity="warning",
            action="rejected invalid owner acknowledgement without changing settings",
            enforce_failure_policy=False,
        )
        return _settings_error_response(exc)


@router.post("/auth/fresh")
async def acknowledge_fresh_auth(
    payload: dict[str, Any] = SETTINGS_CONFIRM_BODY,
    _: None = Depends(_require_internal),
) -> JSONResponse:
    """Authorize one expiring challenge bound to one canonical action."""

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "action_confirmation_invalid",
                "detail": "action confirmation body must be an object",
            },
        )
    challenge_id = str(payload.get("challenge_id") or "")
    try:
        from core.executive.action_confirmation import (
            get_action_confirmation_registry,
        )

        confirmations = get_action_confirmation_registry()
        authorization = confirmations.authorize(challenge_id)
    except KeyError as exc:
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error": "action_confirmation_challenge_not_found",
                "detail": str(exc),
            },
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "error": "action_confirmation_unavailable",
                "detail": str(exc),
            },
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "action_confirmation_invalid",
                "detail": str(exc),
            },
        )
    except ImportError as exc:
        record_degradation("settings.fresh_auth", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        from core.ethics.conscience import get_conscience

        conscience = get_conscience()
        conscience.acknowledge_user_authorization()
        conscience_window_seconds = conscience.fresh_user_authorization_window_s()
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        confirmations.revoke_authorization(challenge_id)
        record_degradation("settings.fresh_auth", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(
        {
            **authorization,
            "conscience_window_seconds": conscience_window_seconds,
            "confirmation_does_not_bypass_governance": True,
        }
    )


@router.post("/auth/revoke")
async def revoke_action_confirmation(
    payload: dict[str, Any] = SETTINGS_CONFIRM_CANCEL_BODY,
    _: None = Depends(_require_internal),
) -> JSONResponse:
    """Cancel one unconsumed confirmation challenge."""

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "action_confirmation_invalid",
                "detail": "action confirmation body must be an object",
            },
        )
    try:
        from core.executive.action_confirmation import (
            get_action_confirmation_registry,
        )

        cancelled = get_action_confirmation_registry().cancel(
            str(payload.get("challenge_id") or "")
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "action_confirmation_invalid",
                "detail": str(exc),
            },
        )
    except ImportError as exc:
        record_degradation("settings.revoke_action_confirmation", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "cancelled": cancelled})


__all__ = [
    "SCHEMA",
    "SettingDef",
    "SettingsStore",
    "get_settings",
    "router",
]
