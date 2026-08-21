"""The chat delivery fence: one reply reaches the caller exactly once.

A chat turn is expensive, and a caller that retries must not pay for it
twice or receive a second, different answer. This module holds the delivery
receipt, the idempotency claim, the replay path and the heartbeat that keeps
a waiting caller from timing out mid-turn, plus the conversation-epoch clock
that decides when a session id starts a new conversation.
"""

from __future__ import annotations

from core.runtime.chat_delivery_journal import (
    AdmissionKind,
    ChatDeliveryFenceLost,
    ChatDeliveryJournalCorruption,
    ChatDeliveryJournalError,
    ChatDeliveryJournalUnavailable,
    DeliveryAdmission,
    DeliveryIdentity,
    DeliveryRecord,
    DeliveryState,
    canonical_request_hash,
    get_chat_delivery_journal,
)
from typing import TYPE_CHECKING, Any
from starlette.background import BackgroundTask
from collections.abc import Callable, Sequence
from core.runtime.flags import FlagKind, declare
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_DELIVERY_IDEMPOTENCY_KEY,  # noqa: F401
    _CHAT_PENDING_DELIVERY_CLAIM,  # noqa: F401
    _CHAT_SESSION_ID_MAX_CHARS,  # noqa: F401
    _INTERNAL_SURFACE_CONTEXT,  # noqa: F401
    _UNSET,  # noqa: F401
)
from core.conversation.session_scope import (
    conversation_turn_var as _CHAT_DELIVERY_TURN_ID,  # noqa: N812
)
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes import chat_preflight as _chat_preflight
import asyncio
from core.runtime.chat_delivery_progress import (
    bind_chat_delivery_progress,
    report_chat_delivery_progress,
)
from core.runtime.receipts import digest_output_content, digest_principal_binding
from core.utils.task_tracker import get_task_tracker
import hashlib
import json
from core.runtime.service_access import optional_service
from interface.auth import (
    CHEAT_CODE_COOKIE_NAME,
    CHEAT_CODE_COOKIE_TTL_SECS,
    _activate_cheat_code_for_request,
    _check_rate_limit,
    _encode_owner_session_cookie,
    _require_internal,
    _restore_owner_session_from_request,
    paired_device_session_id,
    relational_principal_id_for_request,
    request_access_profile,
    validate_runtime_security_request,
)
import re
from core.runtime.errors import describe_error, record_degradation
from core.runtime.principal_context import (
    relational_principal_scope,
)
import time
import uuid
from functools import lru_cache, wraps

from interface.routes.chat_common import (
    MAX_CHAT_MESSAGE_BYTES,
)


_CHAT_DELIVERY_WAIT_TIMEOUT_FLAG = declare(
    "AURA_CHAT_DELIVERY_WAIT_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=180.0,
    description="Maximum wait for another owner of an admitted chat turn",
    owner="interface.routes.chat",
)

_PAIRED_CHAT_RESPONSE_KEYS = frozenset(
    {
        "error",
        "message",
        "response",
        "response_confidence",
        "status",
    }
)


def _paired_chat_response_payload(value: Any) -> dict[str, Any]:
    """Project every paired chat response onto its negotiated wire contract."""

    payload = value if isinstance(value, dict) else {}
    projected = {key: payload.get(key) for key in _PAIRED_CHAT_RESPONSE_KEYS if key in payload}
    if "conversation_lane" in payload:
        projected["conversation_lane"] = _chat_preflight._paired_conversation_lane_payload(
            payload.get("conversation_lane")
        )
    if not str(projected.get("response") or "").strip():
        message = str(projected.get("message") or "").strip()
        projected["response"] = message or "The paired conversation request could not be completed."
    return projected


def _authenticated_chat_principal(request: Request | None) -> str:
    if request is None:
        return ""
    try:
        return " ".join(str(relational_principal_id_for_request(request) or "").strip().split())[
            :160
        ]
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.relational_principal", exc)
        return ""


def _chat_turn_session_key(request: Request | None, body: Any) -> str:
    try:
        paired = paired_device_session_id(request) if request is not None else None
    except _CHAT_RECOVERABLE_ERRORS:
        paired = None
    supplied = str(getattr(body, "session_id", "") or "").strip()
    if paired:
        return paired
    if supplied:
        return supplied
    client = getattr(request, "client", None) if request is not None else None
    return str(getattr(client, "host", "default") or "default")


def _chat_delivery_wait_timeout_s() -> float:
    try:
        configured = float(_CHAT_DELIVERY_WAIT_TIMEOUT_FLAG.value())
    except (TypeError, ValueError):
        configured = 180.0
    return max(1.0, min(configured, 600.0))


def _chat_delivery_principal(
    request: Request | None,
    exact_principal: str,
    session_key: str,
) -> str:
    normalized = " ".join(str(exact_principal or "").strip().split())
    if normalized:
        return normalized
    profile = request_access_profile(request)
    surface = str(profile.get("surface") or "internal").strip().casefold()
    if surface == "owner":
        return "authenticated-local-owner"
    if surface == "paired_device":
        return f"authenticated-{session_key}"
    client = getattr(request, "client", None) if request is not None else None
    host = str(getattr(client, "host", "internal") or "internal").strip().casefold()
    return f"authenticated-{surface}:{host}"


def _chat_delivery_request_contract(
    request: Request | None,
    body: Any,
    *,
    exact_principal: str,
) -> tuple[DeliveryIdentity, str, str]:
    session_key = _chat_turn_session_key(request, body)
    raw_key = (
        str(request.headers.get("X-Idempotency-Key") or "").strip() if request is not None else ""
    )
    idempotency_key = raw_key or f"server-{uuid.uuid4().hex}"
    principal = _chat_delivery_principal(request, exact_principal, session_key)
    identity = DeliveryIdentity.create(
        principal=principal,
        session_id=session_key,
        idempotency_key=idempotency_key,
    )
    profile = request_access_profile(request)
    headers = getattr(request, "headers", {}) if request is not None else {}
    request_hash = canonical_request_hash(
        {
            "schema": "aura.chat.delivery.request.v1",
            "method": str(getattr(request, "method", "POST") or "POST").upper(),
            "path": str(getattr(getattr(request, "url", None), "path", "/api/chat") or "/api/chat"),
            "message": str(getattr(body, "message", "") or ""),
            "session_id": session_key,
            "surface": str(profile.get("surface") or "internal"),
            "conversation_only": bool(profile.get("conversation_only", False)),
            "behavior_headers": {
                "benchmark": str(headers.get("X-Aura-Benchmark") or "").casefold() == "true",
                "require_cognitive_engine": str(
                    headers.get("X-Aura-Require-CognitiveEngine") or ""
                ).casefold()
                == "true",
                "allow_legacy_orchestrator": str(
                    headers.get("X-Aura-Allow-Legacy-Orchestrator") or ""
                ).casefold()
                == "true",
                "desktop_request": str(headers.get("X-Aura-Desktop-Request") or "").casefold(),
                "surface": str(headers.get("X-Aura-Surface") or "").casefold(),
            },
        }
    )
    approval_resume_token = str(headers.get("X-Aura-Approval-Resume") or "").strip().casefold()
    if approval_resume_token and not re.fullmatch(
        r"[0-9a-f]{32}",
        approval_resume_token,
    ):
        raise ValueError("invalid approval-resume token")
    return identity, request_hash, approval_resume_token


def _chat_delivery_state_for_response(
    payload: dict[str, Any],
    status_code: int,
) -> DeliveryState:
    status = str(payload.get("status") or "").strip().casefold()
    confidence = str(payload.get("response_confidence") or "").strip().casefold()
    if status in {"approval_required", "require_fresh_user_auth"}:
        return DeliveryState.AWAITING_APPROVAL
    if "cancel" in status or status == "delivery_ambiguous":
        return DeliveryState.AMBIGUOUS
    failure_markers = (
        "blocked",
        "denied",
        "error",
        "failed",
        "guard",
        "refused",
        "rejected",
        "timeout",
        "unavailable",
    )
    if (
        int(status_code) >= 400
        or confidence == "failed"
        or any(marker in status for marker in failure_markers)
    ):
        return DeliveryState.FAILED
    return DeliveryState.COMPLETED


def _note_chat_surface_delivery_response(
    response: JSONResponse,
    *,
    request: Request | None,
    body: Any,
) -> None:
    """Close the one-message/one-reply surface state with delivered text."""

    try:
        decoded = json.loads(bytes(response.body))
        if not isinstance(decoded, dict):
            raise TypeError("chat delivery body must be a JSON object")
        delivered = str(decoded.get("response") or decoded.get("message") or "").strip()
        if not delivered:
            return
        turn_id = str(response.headers.get("X-Aura-Turn-ID") or "").strip()
        if not turn_id:
            return
        conversation_id = _resolved_conversation_session(request, body)
        from core.conversation.surface_delivery import note_route_delivered

        note_route_delivered(
            delivered,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        record_degradation(
            "chat.surface_delivery",
            exc,
            severity="warning",
            action="delivered the response while route-delivery settlement failed",
            enforce_failure_policy=False,
        )


def _contains_private_affordance_control_syntax(value: Any) -> bool:
    """Recognize private affordance controls without parsing or executing them."""

    text = str(value or "")
    lowered = text.casefold()
    if "affordance:" not in lowered:
        return False
    return any(marker in text for marker in ("⟦", "[[", "<<", "["))


def _chat_delivery_payload(
    payload: dict[str, Any],
    admission: DeliveryAdmission,
    *,
    state: DeliveryState,
    replayed: bool = False,
) -> dict[str, Any]:
    result = dict(payload)
    result.update(
        {
            "turn_id": admission.record.turn_id,
            "idempotency_key": admission.record.identity.idempotency_key,
            "delivery_state": state.value,
            "delivery_generation": admission.record.generation,
            "delivery_replayed": bool(replayed),
        }
    )
    return result


def _chat_delivery_json_response(
    payload: dict[str, Any],
    *,
    status_code: int,
    turn_id: str = "",
    idempotency_key: str = "",
    replayed: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers["Cache-Control"] = "no-store"
    if turn_id:
        response_headers["X-Aura-Turn-ID"] = turn_id
    if idempotency_key:
        response_headers["X-Aura-Idempotency-Key"] = idempotency_key
    if replayed:
        response_headers["X-Aura-Delivery-Replayed"] = "true"
    return JSONResponse(
        payload,
        status_code=int(status_code),
        headers=response_headers,
    )


def _chat_delivery_replay_response(record: DeliveryRecord) -> JSONResponse:
    if not record.terminal or record.response is None or record.http_status is None:
        raise ChatDeliveryJournalCorruption(
            "chat delivery replay requested without a terminal receipt"
        )
    payload = dict(record.response)
    payload["delivery_replayed"] = True
    return _chat_delivery_json_response(
        payload,
        status_code=record.http_status,
        turn_id=record.turn_id,
        idempotency_key=record.identity.idempotency_key,
        replayed=True,
    )


async def _chat_delivery_heartbeat(
    journal: Any,
    admission: DeliveryAdmission,
    fence_lost: asyncio.Event,
) -> None:
    interval_s = max(0.02, min(5.0, float(journal.stale_after_s) / 3.0))
    try:
        while not fence_lost.is_set():
            await asyncio.sleep(interval_s)
            if not await journal.renew(admission):
                fence_lost.set()
                return
    except asyncio.CancelledError:
        raise
    except ChatDeliveryJournalError as exc:
        fence_lost.set()
        logger.error("Chat delivery lease renewal failed closed: %s", exc)


async def _stop_chat_delivery_heartbeat(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _finalize_chat_delivery(
    journal: Any,
    admission: DeliveryAdmission,
    *,
    state: DeliveryState,
    status_code: int,
    payload: dict[str, Any],
) -> DeliveryRecord:
    operation = get_task_tracker().create_task(
        journal.finalize(
            admission,
            state=state,
            http_status=status_code,
            response=payload,
        ),
        name=f"ChatDeliveryFinalize:{admission.record.turn_id}",
    )
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        try:
            await operation
        except ChatDeliveryJournalError as exc:
            logger.error(
                "Chat delivery terminal receipt failed during cancellation: %s",
                exc,
            )
        raise


async def _chat_delivery_fence_response(
    journal: Any,
    admission: DeliveryAdmission,
) -> JSONResponse:
    current = await journal.get(admission.record.identity)
    if current is not None and current.terminal:
        return _chat_delivery_replay_response(current)
    record = current or admission.record
    return _chat_delivery_json_response(
        {
            "response": (
                "This chat execution was superseded by the current fenced owner. "
                "Use the delivery status contract instead of replaying it."
            ),
            "status": "delivery_pending",
            "delivery_state": "running",
            "turn_id": record.turn_id,
            "idempotency_key": record.identity.idempotency_key,
            "delivery_generation": record.generation,
            "delivery_replayed": False,
        },
        status_code=202,
        turn_id=record.turn_id,
        idempotency_key=record.identity.idempotency_key,
        headers={"Retry-After": "1"},
    )


def _observe_authenticated_chat_turn(
    request: Request | None,
    body: Any,
) -> str:
    """Apply exact-agent consent and observe the original HTTP chat turn."""
    principal = _authenticated_chat_principal(request)
    message = str(getattr(body, "message", "") or "")
    if not principal or not message:
        return principal
    if len(message.encode("utf-8", errors="replace")) > MAX_CHAT_MESSAGE_BYTES:
        return principal

    session_key = _chat_turn_session_key(request, body)
    idempotency_key = (
        str(request.headers.get("X-Idempotency-Key") or "").strip()[:240]
        if request is not None
        else ""
    )
    observed_at = time.time()
    event_nonce = (
        f"idempotency:{idempotency_key}" if idempotency_key else f"request:{uuid.uuid4().hex}"
    )
    evidence_digest = hashlib.sha256(
        f"http-chat-v1\n{principal}\n{session_key}\n{event_nonce}\n{message}".encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()

    try:
        authority = optional_service("relational_memory")
        if authority is not None:
            from core.runtime.memory_consent import apply_relational_memory_command

            control = apply_relational_memory_command(
                authority,
                principal,
                message,
                receipt_id=f"chat-command-{evidence_digest}",
            )
            if control is not None and request is not None:
                request.state.relational_memory_control = control
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.relational_memory_control",
            exc,
            action="continued the authenticated turn after relational memory control failed",
        )

    try:
        estimator = optional_service("other_agent_model")
        if estimator is None or not hasattr(estimator, "observe_message"):
            return principal
        estimator.observe_message(
            principal,
            message,
            now=observed_at,
            persist=False,
            evidence_digest=evidence_digest,
        )
        snapshot = (
            estimator.cognitive_snapshot(principal, observed_at)
            if hasattr(estimator, "cognitive_snapshot")
            else None
        )
        observer = optional_service("recursive_tom")
        if observer is not None and hasattr(observer, "observe_agent"):
            observer.observe_agent(
                principal,
                kind="conversation_turn",
                strength=0.8,
                evidence_digest=evidence_digest,
                observed_at=observed_at,
            )
        if (
            observer is not None
            and isinstance(snapshot, dict)
            and hasattr(observer, "register_interaction")
        ):
            snapshot["evidence_digest"] = evidence_digest
            snapshot["at"] = observed_at
            observer.register_interaction(principal, snapshot)
        if hasattr(estimator, "save_if_due"):
            get_task_tracker().track(
                asyncio.to_thread(estimator.save_if_due),
                name="chat.other_agent_model_persist",
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.other_agent_observation",
            exc,
            action="continued the authenticated turn without social-state observation",
        )
    return principal


async def _record_http_chat_delivery(
    response_text: str,
    *,
    principal: str,
    session_key: str,
    status_code: int,
    status: str,
    turn_id: str,
    terminal_at: float,
) -> None:
    """Emit and consume a principal-bound receipt after the HTTP body is sent."""
    if not response_text or not principal:
        return

    def _record() -> None:
        from core.runtime.receipts import OutputReceipt, get_receipt_store

        principal_digest = digest_principal_binding(principal)
        receipt = get_receipt_store().emit(
            OutputReceipt(
                receipt_id=f"output-chat-http-{turn_id}",
                cause="chat_http_response",
                created_at=terminal_at,
                origin="api",
                target="primary",
                digest=digest_output_content(response_text),
                metadata={
                    "accepted_sinks": ["http_response_body"],
                    "delivery_stage": "transport_accepted",
                    "recipient_principal_digest": principal_digest,
                    "session_digest": hashlib.sha256(
                        session_key.encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "status": status[:120],
                    "status_code": int(status_code),
                    "turn_id": turn_id,
                },
            )
        )
        estimator = optional_service("other_agent_model")
        if estimator is not None and hasattr(estimator, "record_response"):
            estimator.record_response(
                principal,
                response_text,
                receipt.receipt_id,
            )

    try:
        from core.runtime.executors import run_durable_receipt_io

        await run_durable_receipt_io(
            _record,
            timeout_s=10.0,
            label="chat_http_delivery_receipt",
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.http_delivery_receipt",
            exc,
            action="reported HTTP delivery without opening an unreceipted feedback window",
        )


def _attach_http_chat_delivery_receipt(
    response: JSONResponse,
    *,
    request: Request | None,
    body: Any,
    payload: dict[str, Any],
    record: DeliveryRecord | None = None,
) -> None:
    response_text = str(payload.get("response") or "").strip()
    principal = _authenticated_chat_principal(request)
    if not response_text or not principal:
        return
    session_key = _chat_turn_session_key(request, body)
    status = str(payload.get("status") or "")
    existing_background = response.background
    # The delivery record enriches the receipt when the paired-boundary
    # produced one; a transport receipt without it is still honest (fresh
    # terminal time, no turn binding) — the social/feedback path exercises
    # this standalone.
    turn_id = str(getattr(record, "turn_id", "") or "")
    terminal_at = float(
        getattr(record, "terminal_at", 0.0) or getattr(record, "updated_at", 0.0) or time.time()
    )

    async def _after_send() -> None:
        try:
            if existing_background is not None:
                await existing_background()
        finally:
            await _record_http_chat_delivery(
                response_text,
                principal=principal,
                session_key=session_key,
                status_code=response.status_code,
                status=status,
                turn_id=turn_id,
                terminal_at=terminal_at,
            )

    response.background = BackgroundTask(_after_send)


def _invalidate_answer_proof_after_delivery_mutation(
    payload: dict[str, Any],
    *,
    original_text: Any,
    reason: str,
) -> None:
    """Prevent a terminal byte rewrite from inheriting route-level proof."""

    contract = payload.get("live_turn_contract")
    if not isinstance(contract, dict):
        return
    contract = dict(contract)
    missing = [str(item) for item in contract.get("full_mind_missing_proofs") or ()]
    marker = "delivery_bytes_changed_after_proof"
    if marker not in missing:
        missing.append(marker)
    contract.update(
        {
            "answer_delivery_proven": False,
            "certification_complete": False,
            "authentic_cognitive_reply": False,
            "authored_generation_source_proven": False,
            "authored_answer_completion_proven": False,
            "final_requested_output_contract_evaluated": False,
            "final_requested_output_contract_satisfied": False,
            "final_requested_output_contract_proven": False,
            "model_native_output": False,
            "final_text_authorship": "delivery_boundary_rewrite",
            "post_generation_repair_applied": True,
            "deterministic_repair_applied": True,
            "authorship_replacement_applied": True,
            "unreceipted_runtime_replacement": True,
            "delivery_payload_mutated_after_proof": True,
            "delivery_payload_mutation_reason": str(reason or "terminal_mutation"),
            "pre_mutation_response_sha256": hashlib.sha256(
                str(original_text or "").encode("utf-8")
            ).hexdigest(),
            "delivered_response_sha256": hashlib.sha256(
                str(payload.get("response") or "").encode("utf-8")
            ).hexdigest(),
            "full_mind_missing_proofs": missing,
        }
    )
    payload["live_turn_contract"] = contract


def _paired_chat_response_boundary(handler: Callable[..., Any]) -> Callable[..., Any]:
    """Fence every chat turn before side effects and durably seal its outcome."""

    @wraps(handler)
    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        body = kwargs.get("body")
        if body is None and args:
            body = args[0]
        request = kwargs.get("request")
        if request is None and len(args) > 1:
            request = args[1]
        if isinstance(body, Request):
            request = body
            body = None
        exact_principal = _authenticated_chat_principal(request)
        conversation_only = bool(request_access_profile(request).get("conversation_only", True))
        strict_output_status = bool(
            not conversation_only
            and request is not None
            and str(request.headers.get("X-Aura-Benchmark") or "").casefold() == "true"
        )

        identity: DeliveryIdentity | None = None
        try:
            identity, request_hash, approval_resume_token = _chat_delivery_request_contract(
                request,
                body,
                exact_principal=exact_principal,
            )
            journal = await asyncio.to_thread(get_chat_delivery_journal)
            admission = await journal.reserve(
                identity,
                request_hash,
                wait_timeout_s=_chat_delivery_wait_timeout_s(),
                approval_resume_token=approval_resume_token,
            )
        except ValueError as exc:
            return _chat_delivery_json_response(
                {
                    "response": "The chat delivery identity or request contract was invalid.",
                    "status": "invalid_chat_delivery_contract",
                    "detail": str(exc),
                    "delivery_state": "failed",
                },
                status_code=400,
                idempotency_key=(identity.idempotency_key if identity else ""),
            )
        except (ChatDeliveryJournalCorruption, ChatDeliveryJournalUnavailable) as exc:
            logger.error("Chat delivery journal admission failed closed: %s", exc)
            return _chat_delivery_json_response(
                {
                    "response": (
                        "The durable chat delivery authority is unavailable. I did not "
                        "start this turn or any of its side effects."
                    ),
                    "status": "chat_delivery_journal_unavailable",
                    "delivery_state": "failed",
                    "response_confidence": "failed",
                },
                status_code=503,
                idempotency_key=(identity.idempotency_key if identity else ""),
                headers={"Retry-After": "1"},
            )

        if admission.kind is AdmissionKind.MISMATCH:
            return _chat_delivery_json_response(
                {
                    "response": (
                        "That idempotency key is already bound to a different chat "
                        "request. The original turn was not changed."
                    ),
                    "status": "idempotency_payload_mismatch",
                    "delivery_state": "mismatch",
                    "turn_id": admission.record.turn_id,
                    "idempotency_key": admission.record.identity.idempotency_key,
                    "response_confidence": "failed",
                },
                status_code=409,
                turn_id=admission.record.turn_id,
                idempotency_key=admission.record.identity.idempotency_key,
            )

        if admission.kind is AdmissionKind.REPLAY:
            try:
                response = _chat_delivery_replay_response(admission.record)
                replay_payload = dict(admission.record.response or {})
                replay_payload["delivery_replayed"] = True
                _attach_http_chat_delivery_receipt(
                    response,
                    request=request,
                    body=body,
                    payload=replay_payload,
                    record=admission.record,
                )
                return response
            except ChatDeliveryJournalCorruption as exc:
                logger.error("Chat delivery replay failed closed: %s", exc)
                return _chat_delivery_json_response(
                    {
                        "response": "The stored chat delivery receipt failed validation.",
                        "status": "chat_delivery_journal_corrupt",
                        "delivery_state": "failed",
                        "response_confidence": "failed",
                    },
                    status_code=503,
                    turn_id=admission.record.turn_id,
                    idempotency_key=admission.record.identity.idempotency_key,
                )

        if admission.kind is AdmissionKind.PENDING:
            return _chat_delivery_json_response(
                admission.record.public_status(include_result=False),
                status_code=202,
                turn_id=admission.record.turn_id,
                idempotency_key=admission.record.identity.idempotency_key,
                headers={"Retry-After": "1"},
            )

        turn_token = _CHAT_DELIVERY_TURN_ID.set(admission.record.turn_id)
        key_token = _CHAT_DELIVERY_IDEMPOTENCY_KEY.set(admission.record.identity.idempotency_key)
        pending_claim_token = _CHAT_PENDING_DELIVERY_CLAIM.set(("", ()))
        fence_lost = asyncio.Event()
        heartbeat_task = get_task_tracker().create_task(
            _chat_delivery_heartbeat(journal, admission, fence_lost),
            name=f"ChatDeliveryHeartbeat:{admission.record.turn_id}",
        )
        try:
            try:
                observed_principal = _observe_authenticated_chat_turn(request, body)
                with (
                    relational_principal_scope(observed_principal or exact_principal),
                    bind_chat_delivery_progress(journal, admission),
                ):
                    await report_chat_delivery_progress(
                        phase="understanding",
                        message="Understanding the request and gathering its relevant context.",
                        details={"surface": request_access_profile(request).get("surface", "")},
                    )
                    response = await handler(*args, **kwargs)
                    await report_chat_delivery_progress(
                        phase="finalizing",
                        message="Checking the result and its evidence before replying.",
                    )
            except asyncio.CancelledError:
                cancelled_payload = _chat_delivery_payload(
                    {
                        "response": (
                            "The transport ended before this chat execution produced "
                            "an authoritative terminal response. Automatic replay is fenced."
                        ),
                        "status": "delivery_ambiguous",
                        "response_confidence": "failed",
                    },
                    admission,
                    state=DeliveryState.AMBIGUOUS,
                )
                try:
                    await _finalize_chat_delivery(
                        journal,
                        admission,
                        state=DeliveryState.AMBIGUOUS,
                        status_code=409,
                        payload=cancelled_payload,
                    )
                except (ChatDeliveryFenceLost, ChatDeliveryJournalError) as exc:
                    logger.error(
                        "Chat cancellation could not seal its authoritative state: %s",
                        exc,
                    )
                raise
            except HTTPException as exc:
                if conversation_only:
                    response = JSONResponse(
                        {
                            "response": "The paired conversation request was rejected.",
                            "status": "paired_request_rejected",
                            "response_confidence": "failed",
                        },
                        status_code=exc.status_code,
                        headers=exc.headers,
                    )
                else:
                    response = JSONResponse(
                        {
                            "detail": exc.detail,
                            "status": "request_rejected",
                            "response_confidence": "failed",
                        },
                        status_code=exc.status_code,
                        headers=exc.headers,
                    )
            except Exception as exc:  # noqa: BLE001 - terminal route boundary
                record_degradation("chat.delivery_boundary", exc)
                logger.error("Chat delivery boundary caught an uncaught failure", exc_info=True)
                response = JSONResponse(
                    {
                        "response": (
                            "The chat execution failed before an authoritative answer formed."
                        ),
                        "status": "chat_delivery_execution_failed",
                        "error_type": type(exc).__name__,
                        "response_confidence": "failed",
                    },
                    status_code=500,
                )

            if not isinstance(response, JSONResponse):
                # A handler that returned the reply TEXT has produced an answer;
                # the only thing wrong is its envelope. Erasing it and shipping a
                # 500 destroys real work for a type error — measured live: Aura
                # opened Notes and wrote the requested note (it is on disk), the
                # salvage path returned the reply as a bare str, and the person
                # saw "The chat route returned an unsupported response format."
                #
                # Deliver the text and record the envelope defect, rather than
                # letting a transport detail decide whether the user gets their
                # answer. The 500 remains for genuinely unusable returns.
                coerced = response if isinstance(response, str) else None
                if coerced is not None and coerced.strip():
                    logger.error(
                        "Chat handler returned a bare %s instead of a JSONResponse "
                        "(%d chars); delivering the text and recording the envelope "
                        "defect rather than discarding a real answer.",
                        type(response).__name__,
                        len(coerced),
                    )
                    record_degradation(
                        "chat.response_envelope",
                        TypeError(
                            f"chat handler returned {type(response).__name__}, not JSONResponse"
                        ),
                        action="delivered the handler's reply text and continued",
                        severity="warning",
                    )
                    response = JSONResponse(
                        {
                            "response": coerced,
                            "status": "chat_response_envelope_coerced",
                            "response_confidence": "degraded",
                        },
                        status_code=200,
                    )
                else:
                    response = JSONResponse(
                        {
                            "response": "The chat route returned an unsupported response format.",
                            "status": "chat_response_format_rejected",
                            "response_confidence": "failed",
                        },
                        # Real chat failures are delivered in-band so the UI
                        # can render the authoritative status without a retry
                        # storm. Proof/benchmark callers retain strict HTTP
                        # failure semantics.
                        status_code=500 if strict_output_status else 200,
                    )

            payload: dict[str, Any]
            try:
                decoded = json.loads(bytes(response.body))
                if not isinstance(decoded, dict):
                    raise TypeError("chat response body must be a JSON object")
                payload = decoded
                if conversation_only:
                    payload = _paired_chat_response_payload(payload)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                record_degradation("chat.paired_response_projection", exc)
                payload = {
                    "response": "The chat response could not be projected safely.",
                    "status": "chat_response_projection_failed",
                    "response_confidence": "failed",
                }
                response.status_code = 500 if strict_output_status else 200

            try:
                from core.cognition.expressive_affordances import (
                    sanitize_affordance_control_syntax,
                )

                affordance_sanitization = sanitize_affordance_control_syntax(
                    str(payload.get("response") or "")
                )
                if affordance_sanitization.changed:
                    pre_sanitization_text = str(payload.get("response") or "")
                    record_degradation(
                        "chat.affordance_visibility_boundary",
                        ValueError("private affordance control syntax reached final delivery"),
                        severity="warning",
                        action=(
                            "stripped private affordance syntax before journaling and delivery"
                        ),
                        extra={
                            "removed_controls": affordance_sanitization.removed_controls,
                            "malformed_controls": affordance_sanitization.malformed_controls,
                        },
                    )
                    payload["response"] = (
                        affordance_sanitization.text
                        or "I wasn't able to realize that action in this turn."
                    )
                    if str(payload.get("response_confidence") or "").casefold() not in {
                        "failed",
                        "failed_closed",
                    }:
                        payload["response_confidence"] = "degraded"
                    if str(payload.get("status") or "").casefold() in {"", "ok"}:
                        payload["status"] = "chat_affordance_control_sanitized"
                    _invalidate_answer_proof_after_delivery_mutation(
                        payload,
                        original_text=pre_sanitization_text,
                        reason="private_affordance_control_removed",
                    )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                unsafe_control_visible = _contains_private_affordance_control_syntax(
                    payload.get("response")
                )
                record_degradation(
                    "chat.affordance_visibility_boundary",
                    exc,
                    severity="warning",
                    action=(
                        "replaced the response because private control visibility could "
                        "not be verified"
                        if unsafe_control_visible
                        else "kept control-free prose after the affordance sanitizer failed"
                    ),
                )
                if unsafe_control_visible:
                    pre_sanitization_text = str(payload.get("response") or "")
                    payload["response"] = (
                        "I couldn't verify that the action control stayed private, so I "
                        "did not deliver that draft."
                    )
                    payload["status"] = "chat_affordance_visibility_unavailable"
                    payload["response_confidence"] = "failed"
                    _invalidate_answer_proof_after_delivery_mutation(
                        payload,
                        original_text=pre_sanitization_text,
                        reason="affordance_visibility_unavailable",
                    )
                    response.status_code = 500 if strict_output_status else 200

            terminal_state = _chat_delivery_state_for_response(
                payload,
                response.status_code,
            )
            payload = _chat_delivery_payload(
                payload,
                admission,
                state=terminal_state,
            )

            if fence_lost.is_set():
                return await _chat_delivery_fence_response(journal, admission)

            try:
                terminal_record = await _finalize_chat_delivery(
                    journal,
                    admission,
                    state=terminal_state,
                    status_code=response.status_code,
                    payload=payload,
                )
            except ChatDeliveryFenceLost:
                return await _chat_delivery_fence_response(journal, admission)
            except ChatDeliveryJournalError as exc:
                logger.error("Chat terminal receipt failed closed: %s", exc)
                return _chat_delivery_json_response(
                    {
                        "response": (
                            "The turn ran, but its durable terminal receipt could not be "
                            "sealed. The result is withheld to prevent unsafe replay."
                        ),
                        "status": "chat_delivery_terminal_unsealed",
                        "delivery_state": "ambiguous",
                        "turn_id": admission.record.turn_id,
                        "idempotency_key": admission.record.identity.idempotency_key,
                        "response_confidence": "failed",
                    },
                    status_code=503,
                    turn_id=admission.record.turn_id,
                    idempotency_key=admission.record.identity.idempotency_key,
                )

            response.body = response.render(payload)
            response.headers["content-length"] = str(len(response.body))
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Aura-Turn-ID"] = admission.record.turn_id
            response.headers["X-Aura-Idempotency-Key"] = admission.record.identity.idempotency_key
            _attach_http_chat_delivery_receipt(
                response,
                request=request,
                body=body,
                payload=payload,
                record=terminal_record,
            )
            pending_owner, pending_ids = _CHAT_PENDING_DELIVERY_CLAIM.get()
            if terminal_state is DeliveryState.COMPLETED and pending_owner and pending_ids:
                try:
                    from core.conversation.chat_preflight import acknowledge_delivery

                    acknowledged = acknowledge_delivery(
                        pending_ids,
                        delivery_owner=pending_owner,
                    )
                    if acknowledged == len(pending_ids):
                        _CHAT_PENDING_DELIVERY_CLAIM.set(("", ()))
                    else:
                        logger.warning(
                            "Pending-chat terminal acknowledgement removed %d/%d rows",
                            acknowledged,
                            len(pending_ids),
                        )
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat.pending_delivery_ack", exc)
            return response
        finally:
            pending_owner, pending_ids = _CHAT_PENDING_DELIVERY_CLAIM.get()
            if pending_owner and pending_ids:
                try:
                    from core.conversation.chat_preflight import release_delivery_claims

                    release_delivery_claims(pending_ids, delivery_owner=pending_owner)
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat.pending_delivery_release", exc)
            await _stop_chat_delivery_heartbeat(heartbeat_task)
            _CHAT_PENDING_DELIVERY_CLAIM.reset(pending_claim_token)
            _CHAT_DELIVERY_IDEMPOTENCY_KEY.reset(key_token)
            _CHAT_DELIVERY_TURN_ID.reset(turn_token)

    @wraps(handler)
    async def _surface_settled(*args: Any, **kwargs: Any) -> JSONResponse:
        response = await _wrapped(*args, **kwargs)
        body = kwargs.get("body")
        if body is None and args:
            body = args[0]
        request = kwargs.get("request")
        if request is None and len(args) > 1:
            request = args[1]
        if isinstance(body, Request):
            request = body
            body = None
        _note_chat_surface_delivery_response(response, request=request, body=body)
        return response

    return _surface_settled


def _resolved_conversation_session(request: Request | None, body: Any) -> str:
    """Resolve the same conversation identity before, during, or after a turn."""

    supplied_session = str(getattr(body, "session_id", "") or "").strip()
    try:
        paired_session = paired_device_session_id(request) if request is not None else None
    except _CHAT_RECOVERABLE_ERRORS:
        paired_session = None
    conversation_session = str(paired_session or supplied_session or "").strip()
    if conversation_session:
        return conversation_session[:_CHAT_SESSION_ID_MAX_CHARS]
    request_session = _chat_turn_session_key(request, body)
    return _chat_preflight._conversation_session_id(request_session)[:_CHAT_SESSION_ID_MAX_CHARS]
