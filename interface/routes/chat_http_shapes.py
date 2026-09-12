"""What a chat turn looks like once it is an HTTP response.

The last step of the route, and nothing here decides anything about the answer:
a JSON encoder that will not refuse an object it has not met before, the body a
caller gets when the runtime is shutting down or the gate never opened, and the
record that this turn was served.
"""
from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import record_shutdown_admission_event
from interface.routes.chat_common import (
    _CHAT_RECOVERABLE_ERRORS,
    logger,
)


def _export_json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _early_chat_json_response(
    payload: dict[str, Any],
    *,
    status_code: int,
) -> JSONResponse:
    """Return an early payload; the paired boundary settles actual delivery."""

    return JSONResponse(payload, status_code=status_code)


def _pre_gate_unavailable_response(gate: str) -> JSONResponse:
    gate_label = {
        "defensive_runtime": "security",
        "conscience": "conscience",
    }.get(str(gate), "required")
    response = (
        f"I could not safely process that turn because my {gate_label} preflight "
        "is unavailable. I did not send the request into cognition or act on it. "
        "Please retry after the runtime recovers."
    )
    return _early_chat_json_response(
        {
            "response": response,
            "message": response,
            "error": "chat_preflight_unavailable",
            "status": "chat_preflight_unavailable",
            "gate": str(gate),
            "retryable": True,
            "processed": False,
            "response_confidence": "fail_closed",
        },
        status_code=503,
    )


def _mark_http_turn_served(outcome: Any, response: Any) -> None:
    """Record what the person actually received, from the response itself."""

    if outcome is None:
        return
    try:
        payload = getattr(response, "body", None)
        served = ""
        if payload is not None:
            data = json.loads(payload)
            if isinstance(data, dict):
                served = str(data.get("response") or "")
                live_contract = data.get("live_turn_contract")
                if isinstance(live_contract, dict):
                    outcome.record_receipt(
                        "served_response_authority",
                        {
                            # Use the turn ledger's redaction-safe evidence
                            # vocabulary. Raw HTTP contract names containing
                            # "auth" are intentionally redacted as possible
                            # credentials by record_receipt().
                            "evidence_kind": live_contract.get(
                                "response_authority_kind"
                            ),
                            "authority_verified": live_contract.get(
                                "response_authority_proven"
                            )
                            is True,
                            "evidence_reason": live_contract.get(
                                "response_authority_reason"
                            ),
                            "delivery_verified": live_contract.get(
                                "answer_delivery_proven"
                            )
                            is True,
                        },
                    )
        if served.strip():
            outcome.mark_served(served)
        else:
            from core.runtime.turn_outcome import UserVisibleState

            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
    except (_CHAT_RECOVERABLE_ERRORS, json.JSONDecodeError) as exc:
        record_degradation("chat.turn_outcome", exc, severity="info")


def _runtime_shutdown_response(
    checkpoint: str,
    *,
    slot_acquired: bool,
    error: BaseException | None = None,
) -> JSONResponse:
    """The 503 a turn returns when the runtime is going down under it.

    Module scope with `slot_acquired` passed in, rather than nested and
    closing over `foreground_slot_acquired`. That was the only thing it took
    from the turn, and one boolean is a cheaper contract than a closure.
    """
    outcome = "reaped" if slot_acquired else "suppressed"
    record_shutdown_admission_event(
        "chat.foreground_turn",
        resource_kind="foreground_turn",
        outcome=outcome,
        detail=f"checkpoint={checkpoint}",
    )
    logger.info(
        "Foreground chat stopped by runtime shutdown (checkpoint=%s error_type=%s).",
        checkpoint,
        type(error).__name__ if error is not None else "none",
    )
    return JSONResponse(
        {
            "response": (
                "The runtime is shutting down, so I stopped this turn cleanly "
                "before starting more cognitive work."
            ),
            "status": "runtime_shutdown",
            "checkpoint": checkpoint,
            "response_confidence": "not_generated",
        },
        status_code=503,
        headers={"Retry-After": "1"},
    )


def _launcher_desktop_runtime_active() -> bool:
    return any(
        str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("AURA_LAUNCHED_FROM_APP", "AURA_EXTERNAL_GUI_OWNER", "AURA_GUI_PROXY")
    )


def _request_from_local_desktop_client(request: Request) -> bool:
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    if not host:
        return True
    return host in {"127.0.0.1", "::1", "localhost", "test", "local"}


def _normalize_response_body(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()
