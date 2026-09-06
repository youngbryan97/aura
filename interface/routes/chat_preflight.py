"""Everything a chat turn settles before any model runs.

The preflight resolves the principal, the session, the surface and the
lane budget, and it either admits the turn or refuses it with a reason.
Nothing here calls a model, which is what makes it safe to run on every
turn including the ones that will be rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from core.runtime.flags import FlagKind, declare
from fastapi.responses import JSONResponse
from core.container import ServiceContainer
from datetime import UTC, datetime
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
from core.conversation.session_scope import (
    conversation_session_var as _CHAT_REQUEST_SESSION,  # noqa: N812
)
from interface.routes import chat_memory_state as _chat_memory_state
from core.runtime.desktop_objective_intent import (
    looks_like_desktop_objective as _shared_looks_like_desktop_objective,
)
import asyncio
import dataclasses
from core.utils.task_tracker import get_task_tracker
import hashlib
import json
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
import sqlite3
import threading
import time
import uuid

from interface.routes.chat_common import (
    _CHAT_SESSION_ID_MAX_CHARS,
    _INTERNAL_SURFACE_CONTEXT,
    _UNSET,
)
from core.runtime.lockdep import checked_lock


_EXPRESSIVE_AFFORDANCES_FLAG = declare(
    "AURA_EXPRESSIVE_AFFORDANCES",
    kind=FlagKind.BOOL,
    default=False,
    description="Inject the expressive-affordance menu into eligible chat turns",
    owner="interface.routes.chat",
)

_CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE = "contextual_language_generation"
_CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT = (
    "qualified_recurrent_state_serialization"
)
_QUALIFIED_RECURRENT_SKIPPED_PREFLIGHT_COMPONENTS = (
    "file_context",
    "directive_context",
    "media_resolution",
    "sight",
    "self_knowledge",
    "arithmetic",
    "grounded_recall",
    "profile_context",
    "operational_self_context",
    "affordance_context",
    "context_clamp",
)


def _chat_evidence_profile(user_message: str, *, bounded_surface: bool) -> tuple[str, Any]:
    """Resolve which answer owner is allowed to consume evidence this turn."""

    if bounded_surface:
        return _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE, None
    try:
        from core.brain.llm.qualified_recurrent_ingress import (
            admit_qualified_recurrent_objective,
        )
        from core.brain.llm.semantic_neural_serving import (
            semantic_neural_default_serving_status,
        )

        admission = admit_qualified_recurrent_objective(user_message)
        status = semantic_neural_default_serving_status() if admission is not None else None
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        admission = None
        status = None
    if admission is None or not isinstance(status, dict) or status.get("active") is not True:
        return _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE, None
    receipt = status.get("receipt")
    if not isinstance(receipt, dict):
        return _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE, None
    allowed_families = receipt.get("allowed_families")
    if (
        not isinstance(allowed_families, (list, tuple))
        or admission.family not in allowed_families
    ):
        return _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE, None
    if admission.parser_id.startswith("semantic_scientific_surface."):
        profile = admission.parser_id.removeprefix(
            "semantic_scientific_surface."
        ).removesuffix(".v1")
        allowed_profiles = receipt.get("allowed_surface_profiles")
        if (
            not isinstance(allowed_profiles, (list, tuple))
            or profile not in allowed_profiles
        ):
            return _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE, None
    if not admission.parser_id.startswith("semantic_"):
        return _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE, None
    return _CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT, admission


async def _apply_camera_control(turn_on: bool) -> dict[str, Any]:
    """Work her own camera control, rather than explaining where it is.

    "Turn on the camera" is a request for an action, and an assistant that
    responds by describing the toggle has answered a different question. The
    setting is the same one the UI's own switch writes, so this is her
    pressing it — the state, the privacy record and the indicator all move
    together, and the user sees the control change under their hands.
    """
    try:
        from interface.routes.privacy import apply_camera_privacy

        state = await apply_camera_privacy(
            bool(turn_on),
            reason="switched by Aura at the owner's request",
        )
        # Tell the surface, so the toggle moves and the camera actually starts
        # or stops. Without this the privacy record would say one thing and
        # the hardware another, which is the worst possible split for a
        # camera: a control that reads "on" over a device that is off, or the
        # reverse.
        from core.container import ServiceContainer

        orchestrator = ServiceContainer.get("orchestrator", default=None)
        publish = getattr(orchestrator, "_publish_telemetry", None)
        if publish is not None:
            publish({"type": "camera_privacy", "enabled": bool(turn_on)})
        return state
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.sight",
            exc,
            action="could not operate the camera control on her own behalf",
        )
        try:
            from core.conversation.failure_context import record_capability_failure

            record_capability_failure(
                "camera_control",
                intent=f"switch the camera {'on' if turn_on else 'off'}",
                cause="failed",
                detail=f"{type(exc).__name__}: {exc}"[:200],
            )
        except _CHAT_RECOVERABLE_ERRORS:
            pass
        return {
            "ok": False,
            "enabled": not bool(turn_on),
            "error": f"{type(exc).__name__}: {exc}"[:240],
        }


def _publish_media_card(resolution: Any) -> None:
    """Put the player on screen.

    Best-effort by construction: this is a UI event, and a surface that
    cannot be reached must never take down the turn that produced it. If the
    card does not arrive the reply still does, and the reply is where the
    conversation actually lives.
    """
    item = getattr(resolution, "item", None)
    if item is None:
        return
    try:
        from core.container import ServiceContainer

        orchestrator = ServiceContainer.get("orchestrator", default=None)
        publish = getattr(orchestrator, "_publish_telemetry", None)
        if publish is None:
            return
        publish(
            {
                "type": "action_result",
                "tool": "media_playback",
                "media": item.to_dict(),
                "result": {"message": f"Playing {item.title}."},
            }
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.media",
            exc,
            action="the media card did not reach the surface; the reply still did",
            severity="warning",
        )


_PAIRED_CONVERSATION_LANE_KEYS = frozenset(
    {
        "active_generation",
        "active_generations",
        "conversation_ready",
        "state",
    }
)


def _paired_conversation_lane_payload(value: Any) -> dict[str, Any]:
    lane = value if isinstance(value, dict) else {}
    return {key: lane.get(key) for key in _PAIRED_CONVERSATION_LANE_KEYS if key in lane}


_CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME = "ChatTurnMemoryLogDrain"

_CHAT_TURN_MEMORY_LOG_RETRY_TASK_NAME = "ChatTurnMemoryLogRetry"

_CHAT_TURN_MEMORY_LOG_BATCH_MAX = 16

_CHAT_TURN_MEMORY_LOG_RUN_MAX = 128

_CHAT_TURN_MEMORY_LOG_TIMEOUT_S = 20.0

_CHAT_TURN_CONSCIOUSNESS_UPDATE_TIMEOUT_S = 8.0

_CHAT_TURN_MEMORY_LOG_LEASE_RECHECK_S = 61.0

_CHAT_TURN_MEMORY_LOG_FOREGROUND_RECHECK_S = 0.25

_CHAT_TURN_MEMORY_LOG_SHUTDOWN_HANDLER = "chat.durable_memory_log_outbox"

_DURABLE_CONVERSATION_WRITE_TIMEOUT_S = _chat_memory_state._DURABLE_CONVERSATION_CONTEXT_TIMEOUT_S

_DURABLE_CONVERSATION_WRITE_DRAIN_TIMEOUT_S = 12.0

_DURABLE_CONVERSATION_WRITE_HISTORY_MAX = 1024


@dataclasses.dataclass
class _DurableConversationWrite:
    operation_id: str
    payload_sha256: str
    task: asyncio.Task[Any]
    state: str = "pending"
    attempt: int = 1
    error: str = ""
    failure_observed: bool = False
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    finished_at: float | None = None


_DURABLE_CONVERSATION_WRITES: dict[str, _DurableConversationWrite] = {}

_DURABLE_CONVERSATION_WRITES_LOCK = checked_lock("interface.routes.chat_preflight", reentrant=True)

_DURABLE_CONVERSATION_SHUTDOWN_HANDLER = "chat.durable_conversation_writes"


def _new_exchange_id() -> str:
    return uuid.uuid4().hex


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _trim_conversation_log_locked() -> None:
    while len(_conversation_log) > _MAX_CONVERSATION_LOG_EXCHANGES:
        _conversation_log.pop(0)


def _durable_conversation_payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _settle_durable_conversation_write(
    operation_id: str,
    task: asyncio.Task[Any],
) -> None:
    terminal_state = ""
    with _DURABLE_CONVERSATION_WRITES_LOCK:
        record = _DURABLE_CONVERSATION_WRITES.get(operation_id)
        if record is None or record.task is not task or record.state != "pending":
            return
        record.finished_at = time.monotonic()
        if task.cancelled():
            record.state = "failed"
            record.error = "write_task_cancelled"
        else:
            try:
                task.result()
            except Exception as exc:  # noqa: BLE001 - retain exact terminal failure
                record.state = "failed"
                record.error = f"{type(exc).__name__}:{exc}"
            else:
                record.state = "committed"
                record.error = ""
        terminal_state = record.state

    exchange_id, separator, operation_kind = operation_id.rpartition(":")
    if not separator or operation_kind not in {"user", "exchange"}:
        return
    if operation_kind == "exchange" and terminal_state == "committed":
        with _DURABLE_CONVERSATION_WRITES_LOCK:
            user_record = _DURABLE_CONVERSATION_WRITES.get(f"{exchange_id}:user")
            if user_record is not None and user_record.state == "failed":
                user_record.state = "superseded"
                user_record.error = "superseded_by_atomic_exchange"
        _schedule_chat_turn_memory_log()
    # Completion callbacks execute on the owning event loop. Updating these
    # scalar receipt fields in place lets operators see a late write settle;
    # the registry above remains authoritative across log trimming.
    for entry in reversed(_conversation_log):
        if str(entry.get("id") or "") != exchange_id:
            continue
        if operation_kind == "user":
            entry["user_persistence_state"] = terminal_state
            entry["user_persisted"] = terminal_state == "committed"
        else:
            entry["durability_state"] = terminal_state
        break


def _prune_durable_conversation_writes_locked() -> None:
    overflow = len(_DURABLE_CONVERSATION_WRITES) - _DURABLE_CONVERSATION_WRITE_HISTORY_MAX
    if overflow <= 0:
        return
    terminal = sorted(
        (record for record in _DURABLE_CONVERSATION_WRITES.values() if record.state != "pending"),
        key=lambda record: record.finished_at or record.started_at,
    )
    for record in terminal[:overflow]:
        _DURABLE_CONVERSATION_WRITES.pop(record.operation_id, None)


async def _drain_durable_conversation_writes() -> None:
    """Finish every admitted transcript write during memory-commit shutdown."""
    with _DURABLE_CONVERSATION_WRITES_LOCK:
        pending = {
            record.task
            for record in _DURABLE_CONVERSATION_WRITES.values()
            if record.state == "pending" and not record.task.done()
        }
    if pending:
        _, remaining = await asyncio.wait(
            pending,
            timeout=_DURABLE_CONVERSATION_WRITE_DRAIN_TIMEOUT_S,
        )
        if remaining:
            names = sorted(task.get_name() for task in remaining)
            raise TimeoutError(
                "durable conversation shutdown drain timed out: " + ", ".join(names[:8])
            )
    with _DURABLE_CONVERSATION_WRITES_LOCK:
        records = list(_DURABLE_CONVERSATION_WRITES.values())
    for record in records:
        if record.task.done():
            _settle_durable_conversation_write(record.operation_id, record.task)
    failed = [
        record
        for record in records
        if record.state == "failed"
        and record.finished_at is not None
        and not record.failure_observed
    ]
    if failed:
        with _DURABLE_CONVERSATION_WRITES_LOCK:
            for record in failed:
                record.failure_observed = True
        raise RuntimeError(
            "durable conversation write failure during shutdown: "
            + ", ".join(f"{record.operation_id}={record.error}" for record in failed[-8:])
        )


def _ensure_durable_conversation_shutdown_handler() -> None:
    from core.runtime.shutdown_coordinator import get_shutdown_coordinator

    coordinator = get_shutdown_coordinator()
    with _DURABLE_CONVERSATION_WRITES_LOCK:
        if _DURABLE_CONVERSATION_SHUTDOWN_HANDLER in coordinator.handler_names("memory_commit"):
            return
        coordinator.register(
            _drain_durable_conversation_writes,
            phase="memory_commit",
            name=_DURABLE_CONVERSATION_SHUTDOWN_HANDLER,
            timeout=_DURABLE_CONVERSATION_WRITE_DRAIN_TIMEOUT_S + 1.0,
        )


def _start_durable_conversation_write(
    *,
    operation_id: str,
    payload: dict[str, Any],
    operation: Callable[[], Any],
) -> _DurableConversationWrite:
    """Start or reuse one idempotent durable write with retained ownership."""
    safe_operation_id = str(operation_id or "")[:160]
    if not safe_operation_id:
        raise ValueError("durable conversation write requires an operation id")
    payload_sha256 = _durable_conversation_payload_sha256(payload)
    _ensure_durable_conversation_shutdown_handler()

    with _DURABLE_CONVERSATION_WRITES_LOCK:
        existing = _DURABLE_CONVERSATION_WRITES.get(safe_operation_id)
        if existing is not None:
            if existing.payload_sha256 != payload_sha256:
                raise ValueError(f"durable conversation operation conflict: {safe_operation_id}")
            if existing.state in {"pending", "committed"}:
                return existing
            attempt = existing.attempt + 1
        else:
            attempt = 1

        task = _chat_memory_state._start_bounded_chat_blocking_task(
            operation,
            operation_name=f"conversation_persistence:{safe_operation_id}:attempt-{attempt}",
        )
        record = _DurableConversationWrite(
            operation_id=safe_operation_id,
            payload_sha256=payload_sha256,
            task=task,
            attempt=attempt,
        )
        _DURABLE_CONVERSATION_WRITES[safe_operation_id] = record
        _prune_durable_conversation_writes_locked()
        task.add_done_callback(
            lambda completed, operation_id=safe_operation_id: _settle_durable_conversation_write(
                operation_id, completed
            )
        )
        return record


async def _await_durable_conversation_write(
    record: _DurableConversationWrite,
    *,
    timeout_s: float | None = None,
) -> str:
    if record.state != "pending":
        return record.state
    wait_budget = _DURABLE_CONVERSATION_WRITE_TIMEOUT_S if timeout_s is None else float(timeout_s)
    await asyncio.wait({record.task}, timeout=max(0.01, wait_budget))
    if record.task.done():
        _settle_durable_conversation_write(record.operation_id, record.task)
    if record.state == "failed":
        with _DURABLE_CONVERSATION_WRITES_LOCK:
            record.failure_observed = True
    return record.state


def _durable_conversation_write_snapshot(operation_id: str) -> dict[str, Any] | None:
    with _DURABLE_CONVERSATION_WRITES_LOCK:
        record = _DURABLE_CONVERSATION_WRITES.get(str(operation_id or "")[:160])
        if record is None:
            return None
        return {
            "operation_id": record.operation_id,
            "payload_sha256": record.payload_sha256,
            "state": record.state,
            "attempt": record.attempt,
            "error": record.error,
            "failure_observed": record.failure_observed,
            "task_done": record.task.done(),
        }


async def _persist_pending_conversation_user(
    *,
    exchange_id: str,
    user_message: str,
    session_id: str = "",
) -> str:
    """Commit the user side of a turn before foreground inference starts."""
    try:
        persistence = ServiceContainer.get("persistence", default=None)
        record_turn = getattr(persistence, "record_turn", None)
        if not callable(record_turn):
            return "failed"

        safe_exchange_id = str(exchange_id or "")[:64]
        safe_session_id = str(session_id or "")[:64]
        safe_user_message = str(user_message or "")
        scope_kwargs = _chat_principal_scope_kwargs()
        record = _start_durable_conversation_write(
            operation_id=f"{safe_exchange_id}:user",
            payload={
                "kind": "pending_user",
                "exchange_id": safe_exchange_id,
                "user_message": safe_user_message,
                "session_id": safe_session_id,
                "scope": scope_kwargs,
            },
            operation=lambda: record_turn(
                "user",
                safe_user_message,
                origin="desktop_ui",
                cid=f"{safe_exchange_id}:user",
                session_id=safe_session_id or None,
                **scope_kwargs,
            ),
        )
        state = await _await_durable_conversation_write(record)
        if state == "pending":
            timeout_error = TimeoutError(
                f"pending conversation write retained after "
                f"{_DURABLE_CONVERSATION_WRITE_TIMEOUT_S:.2f}s response budget"
            )
            record_degradation(
                "chat.conversation_persistence",
                timeout_error,
                severity="warning",
                action="retained late pending user-turn write under durable custody",
                extra={"operation_id": record.operation_id, "attempt": record.attempt},
            )
            logger.warning(
                "Durable pending user-turn write remains supervised: %s",
                record.operation_id,
            )
        elif state == "failed":
            raise RuntimeError(record.error or "pending user-turn write failed")
        return state
    except asyncio.CancelledError:
        raise
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.conversation_persistence", exc)
        logger.warning("Durable pending user-turn commit failed: %s", exc)
        return "failed"


async def _begin_logged_exchange(user_msg: str, *, session_id: str = "") -> str:
    """Create and durably pre-log an in-flight exchange."""
    exchange_id = _new_exchange_id()
    principal_id, principal_surface = _chat_memory_state._chat_memory_identity()
    async with _chat_memory_state._get_convo_lock():
        _conversation_log.append(
            {
                "id": exchange_id,
                "timestamp": _utc_now_iso(),
                "user": user_msg,
                "aura": "",
                "status": "pending",
                "session_id": str(session_id or "")[:64],
                "principal_id": principal_id,
                "principal_surface": principal_surface,
                "user_persisted": False,
                "user_persistence_state": "pending",
                "durability_state": "pending",
            }
        )
        _trim_conversation_log_locked()

    user_persistence_state = await _persist_pending_conversation_user(
        exchange_id=exchange_id,
        user_message=user_msg,
        session_id=session_id,
    )
    async with _chat_memory_state._get_convo_lock():
        for entry in reversed(_conversation_log):
            if str(entry.get("id") or "") == exchange_id:
                entry["user_persistence_state"] = user_persistence_state
                entry["user_persisted"] = user_persistence_state == "committed"
                break
    return exchange_id


async def _complete_logged_exchange(
    exchange_id: str | None,
    user_msg: str,
    aura_response: str,
    *,
    regenerated: bool = False,
    record_experience: bool = True,
    exchange_metadata: dict[str, Any] | None = None,
) -> str:
    """Finalize a pending exchange in place so history is never duplicated."""
    final_response = aura_response or "…"
    recorded_user = str(user_msg or "")

    async with _chat_memory_state._get_convo_lock():
        target: dict | None = None
        if exchange_id:
            for entry in reversed(_conversation_log):
                if str(entry.get("id") or "") == str(exchange_id):
                    target = entry
                    break

        if target is None:
            principal_id, principal_surface = _chat_memory_state._chat_memory_identity()
            target = {
                "id": exchange_id or _new_exchange_id(),
                "timestamp": _utc_now_iso(),
                "user": recorded_user,
                "principal_id": principal_id,
                "principal_surface": principal_surface,
            }
            _conversation_log.append(target)

        # A pending exchange was opened with the wire-visible user text. Do not
        # replace it with the semantic utterance used for intent and generation
        # when the exchange completes.
        recorded_user = str(target.get("user") or recorded_user)
        target["user"] = recorded_user
        target["aura"] = final_response
        target["status"] = "complete"
        target["completed_at"] = _utc_now_iso()
        target["durability_state"] = "pending"
        target.setdefault("revision", 1)
        target["aura_sha256"] = hashlib.sha256(final_response.encode("utf-8")).hexdigest()
        if isinstance(exchange_metadata, dict):
            current_metadata = target.get("metadata")
            merged_metadata = (
                dict(current_metadata) if isinstance(current_metadata, dict) else {}
            )
            merged_metadata.update(dict(exchange_metadata))
            target["metadata"] = merged_metadata
        if regenerated:
            target["regenerated"] = True
        _trim_conversation_log_locked()

    learning_owned_by_outbox = bool(
        record_experience and _conversation_memory_outbox_available()
    )
    durability_state = await _persist_completed_conversation_exchange(
        exchange_id=str(target.get("id") or exchange_id or ""),
        user_message=recorded_user,
        aura_response=final_response,
        session_id=str(target.get("session_id") or ""),
        user_already_persisted=bool(target.get("user_persisted")),
        enqueue_memory_log=learning_owned_by_outbox,
        exchange_metadata=(
            dict(target.get("metadata"))
            if isinstance(target.get("metadata"), dict)
            else None
        ),
    )
    if learning_owned_by_outbox and durability_state == "failed":
        # Method presence is not custody. If the atomic transcript/outbox
        # write failed, retain the historical direct path rather than dropping
        # the turn's semantic effects on the floor.
        learning_owned_by_outbox = False
    user_write = _durable_conversation_write_snapshot(
        f"{str(target.get('id') or exchange_id or '')[:64]}:user"
    )
    async with _chat_memory_state._get_convo_lock():
        target["durability_state"] = durability_state
        if user_write is not None:
            target["user_persistence_state"] = str(user_write.get("state") or "failed")
            target["user_persisted"] = user_write.get("state") == "committed"

    _record_unified_transcript_exchange(
        recorded_user,
        final_response,
        session_id=str(target.get("session_id") or ""),
        exchange_id=str(target.get("id") or exchange_id or ""),
        exchange_metadata=target.get("metadata"),
    )

    if record_experience and not learning_owned_by_outbox:
        # Compatibility persistence implementations have no durable outbox.
        # Preserve their historical semantics; the production implementation
        # takes the supervised path above and does not hold up delivery.
        try:
            from core.runtime.conversation_support import record_conversation_experience

            await record_conversation_experience(recorded_user, final_response)
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Conversation experience recording skipped: %s", exc)
    return durability_state


def _record_unified_transcript_exchange(
    user_message: str,
    aura_response: str,
    *,
    session_id: str,
    exchange_id: str,
    exchange_metadata: Any = None,
) -> None:
    """Put the terminally delivered HTTP exchange into core continuity.

    Persistence is durable history; this transcript is the bounded live
    context used by referential continuation. Recording only after terminal
    completion ensures Aura never remembers a draft that the user did not see.
    """
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        transcript = UnifiedTranscript.get_instance()
        metadata = {
            "exchange_id": str(exchange_id or "")[:64],
            "origin": "desktop_ui",
        }
        if isinstance(exchange_metadata, dict):
            metadata.update(dict(exchange_metadata))
        transcript.add_text_input(
            str(user_message or ""),
            metadata=metadata,
            conversation_id=session_id,
        )
        transcript.add_text_output(
            str(aura_response or ""),
            metadata=metadata,
            conversation_id=session_id,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.unified_transcript", exc)
        logger.warning("Live transcript exchange recording failed: %s", exc)


async def _attach_logged_exchange_metadata(
    exchange_id: str | None,
    metadata: dict[str, Any],
) -> bool:
    """Attach typed evidence to the exchange that owns it."""

    if not exchange_id or not isinstance(metadata, dict):
        return False
    async with _chat_memory_state._get_convo_lock():
        for entry in reversed(_conversation_log):
            if str(entry.get("id") or "") != str(exchange_id):
                continue
            current = entry.get("metadata")
            merged = dict(current) if isinstance(current, dict) else {}
            merged.update(dict(metadata))
            entry["metadata"] = merged
            return True
    return False


async def _log_exchange(
    user_msg: str,
    aura_response: str,
    *,
    record_experience: bool = True,
    session_id: str = "",
    exchange_metadata: dict[str, Any] | None = None,
):
    """Record a conversation exchange for session tracking."""
    exchange_id = await _begin_logged_exchange(user_msg, session_id=session_id)
    await _complete_logged_exchange(
        exchange_id,
        user_msg,
        aura_response,
        record_experience=record_experience,
        exchange_metadata=exchange_metadata,
    )


def _active_task_count_by_name(tracker: Any, task_name: str) -> int:
    active = 0
    for task in list(getattr(tracker, "tasks", ()) or ()):
        try:
            if not task.done() and task.get_name() == task_name:
                active += 1
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.task_tracker", exc)
            logger.debug("Task name inspection failed: %s", exc)
    return active


async def _run_chat_turn_memory_log_item(
    payload: dict[str, Any],
) -> tuple[str, str]:
    user_message = str(payload.get("user_content") or "")
    aura_response = str(payload.get("aura_content") or "")
    session_id = str(payload.get("session_id") or "")
    chat_origin = str(payload.get("origin") or "unknown")
    user_id = str(payload.get("principal_id") or "").strip()[:160]
    principal_surface = str(payload.get("principal_surface") or "").strip().casefold()[:32]
    operation_id = str(payload.get("operation_id") or "")[:160]
    revision = int(payload.get("revision") or 1)
    try:
        from core.conversation.response_reliability import (
            assess_conversation_learning_admission,
        )

        admission = assess_conversation_learning_admission(
            user_message,
            aura_response,
        )
        from core.memory.chat_turn_logger import (
            local_chat_turn_learning_rejection_reason,
            log_chat_turn_auto,
        )

        local_rejection = local_chat_turn_learning_rejection_reason(
            user_message,
            aura_response,
        )
        if admission.ok and not local_rejection and not bool(payload.get("episodic_logged")):
            logged = await asyncio.wait_for(
                log_chat_turn_auto(
                    user_message=user_message,
                    aura_response=aura_response,
                    session_id=session_id,
                    emotional_valence=0.0,
                    metadata={
                        "conversation_lane": True,
                        "origin": chat_origin,
                        "user_id": user_id,
                        "principal_id": user_id,
                        "principal_surface": principal_surface,
                        "memory_log_operation_id": operation_id,
                        "conversation_revision": revision,
                        "conversation_exchange_id": str(payload.get("exchange_id") or "")[:128],
                    },
                ),
                timeout=_CHAT_TURN_MEMORY_LOG_TIMEOUT_S,
            )
            if not logged:
                return "retry", "episodic_memory_did_not_commit"
            await _mark_chat_turn_memory_log_stage(operation_id, "episodic")

        # Keep the complete conversation-experience fanout under the same
        # durable owner: semantic continuity, shared ground, coding-session
        # memory and relationship state must not depend on HTTP response time.
        from core.runtime.conversation_support import record_conversation_experience

        if not bool(payload.get("experience_recorded")):
            await record_conversation_experience(
                user_message,
                aura_response,
                principal_id=user_id or None,
            )
            await _mark_chat_turn_memory_log_stage(operation_id, "experience")

        if not admission.ok:
            logger.warning(
                "Conversation learning rejected a non-admissible reply (%s); "
                "continuity-safe effects were evaluated and the durable transcript remains available.",
                ",".join(admission.reasons) or "unknown",
            )
            return "rejected", ",".join(admission.reasons) or "learning_admission_rejected"
        if local_rejection:
            return "rejected", local_rejection

        try:
            if bool(payload.get("consciousness_updated")):
                return "completed", ""
            from core.consciousness.coordinator import get_consciousness_coordinator

            coordinator = await get_consciousness_coordinator()
            await asyncio.wait_for(
                coordinator.on_chat_turn(user_message, aura_response),
                timeout=_CHAT_TURN_CONSCIOUSNESS_UPDATE_TIMEOUT_S,
            )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.consciousness_update", exc)
            logger.debug("Consciousness update skipped: %s", exc)
        await _mark_chat_turn_memory_log_stage(operation_id, "consciousness")
        return "completed", ""
    except TimeoutError as exc:
        record_degradation("chat.memory_log_timeout", exc)
        logger.warning(
            "Chat turn memory log exceeded %.1fs; durable outbox retained it for retry.",
            _CHAT_TURN_MEMORY_LOG_TIMEOUT_S,
        )
        return "retry", f"TimeoutError:{exc}"
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Chat turn logging failed: %s", exc)
        return "retry", f"{type(exc).__name__}:{exc}"


async def _mark_chat_turn_memory_log_stage(operation_id: str, stage: str) -> None:
    persistence = ServiceContainer.get("persistence", default=None)
    mark = getattr(persistence, "mark_memory_log_stage", None)
    if not callable(mark):
        return
    await asyncio.to_thread(mark, operation_id, stage=stage)


def _chat_turn_memory_log_foreground_delay() -> float | None:
    """Return when durable post-turn learning may resume.

    The transcript and outbox row are already durable before this worker is
    scheduled.  Everything drained here is background learning fanout: an
    episodic projection, profile/interpersonal updates, shared-ground work and
    a consciousness update.  Running that fanout while a person is speaking
    made the next HTTP turn compete with its predecessor's learning and, live,
    delayed an otherwise 2.5s recurrent answer by 62s.

    Use the process-wide foreground lease rather than a route-local flag so
    voice, desktop chat and paired conversation all get the same priority.
    """

    try:
        from core.runtime.foreground_guard import snapshot

        guard = snapshot()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.memory_log_foreground_guard",
            exc,
            action="retained the durable learning item for a bounded retry",
        )
        return _CHAT_TURN_MEMORY_LOG_FOREGROUND_RECHECK_S

    if bool(guard.get("active")):
        return _CHAT_TURN_MEMORY_LOG_FOREGROUND_RECHECK_S
    quiet_remaining = max(0.0, float(guard.get("quiet_remaining_s") or 0.0))
    if quiet_remaining > 0.0:
        return max(
            _CHAT_TURN_MEMORY_LOG_FOREGROUND_RECHECK_S,
            quiet_remaining + 0.05,
        )
    return None


async def _drain_chat_turn_memory_log_queue(*, honor_foreground: bool = True) -> None:
    persistence = ServiceContainer.get("persistence", default=None)
    claim = getattr(persistence, "claim_memory_log_batch", None)
    settle = getattr(persistence, "settle_memory_log_item", None)
    status = getattr(persistence, "memory_log_outbox_status", None)
    if not callable(claim) or not callable(settle):
        return

    if honor_foreground:
        foreground_delay = _chat_turn_memory_log_foreground_delay()
        if foreground_delay is not None:
            logger.debug(
                "Chat memory learning retained in its durable outbox for %.2fs "
                "while foreground conversation has priority.",
                foreground_delay,
            )
            _schedule_chat_turn_memory_log_retry(foreground_delay)
            return

    processed = 0
    retry_delay_s: float | None = None
    while processed < _CHAT_TURN_MEMORY_LOG_RUN_MAX:
        try:
            items = await asyncio.to_thread(
                claim,
                limit=min(
                    _CHAT_TURN_MEMORY_LOG_BATCH_MAX,
                    _CHAT_TURN_MEMORY_LOG_RUN_MAX - processed,
                ),
            )
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            record_degradation("chat.memory_log_outbox_claim", exc)
            _schedule_chat_turn_memory_log_retry(1.0)
            return
        if not items:
            break
        for payload in items:
            outcome, error = await _run_chat_turn_memory_log_item(payload)
            attempts = max(1, int(payload.get("attempts") or 1))
            delay_s = min(60.0, float(2 ** min(6, attempts - 1)))
            try:
                terminal_state = await asyncio.to_thread(
                    settle,
                    str(payload.get("operation_id") or ""),
                    outcome=outcome,
                    error=error,
                    retry_delay_s=delay_s,
                )
            except (sqlite3.Error, OSError, RuntimeError) as exc:
                record_degradation("chat.memory_log_outbox_settle", exc)
                _schedule_chat_turn_memory_log_retry(_CHAT_TURN_MEMORY_LOG_LEASE_RECHECK_S)
                return
            if terminal_state == "pending":
                retry_delay_s = delay_s if retry_delay_s is None else min(retry_delay_s, delay_s)
            processed += 1

    try:
        outbox_status = await asyncio.to_thread(status) if callable(status) else {}
    except (sqlite3.Error, OSError, RuntimeError) as exc:
        record_degradation("chat.memory_log_outbox_status", exc)
        _schedule_chat_turn_memory_log_retry(1.0)
        return
    pending = int(outbox_status.get("pending") or 0)
    processing = int(outbox_status.get("processing") or 0)
    if pending > 0 or processing > 0:
        delay_s = retry_delay_s or (0.05 if pending > 0 else _CHAT_TURN_MEMORY_LOG_LEASE_RECHECK_S)
        _schedule_chat_turn_memory_log_retry(delay_s)


async def _retry_chat_turn_memory_log_after(delay_s: float) -> None:
    next_delay = max(0.01, float(delay_s))
    while True:
        await asyncio.sleep(next_delay)
        foreground_delay = _chat_turn_memory_log_foreground_delay()
        if foreground_delay is None:
            break
        next_delay = foreground_delay
    # This retry task is the one durable owner of the deferred wake. Calling
    # the regular foreground branch here would see itself in TaskTracker and
    # decline to schedule a duplicate, then exit with no future wake at all.
    await _drain_chat_turn_memory_log_queue(honor_foreground=False)


async def _drain_chat_turn_memory_log_queue_on_shutdown() -> None:
    """Flush durable learning during shutdown without a stale quiet-window delay."""

    await _drain_chat_turn_memory_log_queue(honor_foreground=False)


def _schedule_chat_turn_memory_log_retry(delay_s: float) -> bool:
    tracker = get_task_tracker()
    if _active_task_count_by_name(tracker, _CHAT_TURN_MEMORY_LOG_RETRY_TASK_NAME):
        return True
    retry_coro = _retry_chat_turn_memory_log_after(delay_s)
    schedule = getattr(tracker, "bounded_track", None) or getattr(
        tracker,
        "create_task",
        None,
    )
    if not callable(schedule):
        retry_coro.close()
        raise RuntimeError("task_tracker_has_no_scheduler")
    try:
        schedule(
            retry_coro,
            name=_CHAT_TURN_MEMORY_LOG_RETRY_TASK_NAME,
        )
        return True
    except _CHAT_RECOVERABLE_ERRORS:
        retry_coro.close()
        raise


def _ensure_chat_turn_memory_log_shutdown_handler() -> None:
    from core.runtime.shutdown_coordinator import get_shutdown_coordinator

    coordinator = get_shutdown_coordinator()
    if _CHAT_TURN_MEMORY_LOG_SHUTDOWN_HANDLER in coordinator.handler_names("memory_commit"):
        return
    coordinator.register(
        _drain_chat_turn_memory_log_queue_on_shutdown,
        phase="memory_commit",
        name=_CHAT_TURN_MEMORY_LOG_SHUTDOWN_HANDLER,
        timeout=_CHAT_TURN_MEMORY_LOG_TIMEOUT_S + 5.0,
    )


def _schedule_chat_turn_memory_log(
    *,
    user_message: str = "",
    aura_response: str = "",
    session_id: str = "",
    chat_origin: str = "",
    user_id: str = "",
    principal_surface: str = "",
) -> bool:
    """Wake the durable post-response memory outbox worker."""
    try:
        persistence = ServiceContainer.get("persistence", default=None)
        if not callable(getattr(persistence, "claim_memory_log_batch", None)):
            return False
        _ensure_chat_turn_memory_log_shutdown_handler()
        task_tracker = get_task_tracker()
        active_drains = _active_task_count_by_name(
            task_tracker,
            _CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME,
        )
        if active_drains > 0:
            return True

        schedule = getattr(task_tracker, "bounded_track", None) or getattr(
            task_tracker,
            "create_task",
            None,
        )
        if not callable(schedule):
            raise RuntimeError("task_tracker_has_no_scheduler")
        drain_coro = _drain_chat_turn_memory_log_queue()
        try:
            schedule(drain_coro, name=_CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME)
        except _CHAT_RECOVERABLE_ERRORS:
            drain_coro.close()
            raise
        return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Chat turn logging task creation failed: %s", exc)
        return False


def _conversation_memory_outbox_available() -> bool:
    try:
        persistence = ServiceContainer.get("persistence", default=None)
        return bool(
            callable(getattr(persistence, "record_exchange", None))
            and callable(getattr(persistence, "claim_memory_log_batch", None))
            and callable(getattr(persistence, "settle_memory_log_item", None))
            and callable(getattr(persistence, "mark_memory_log_stage", None))
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.memory_log_outbox_availability", exc)
        return False


def _chat_principal_scope_kwargs(
    *,
    principal_id: str = "",
    principal_surface: str = "",
) -> dict[str, str]:
    principal, surface = _chat_memory_state._chat_memory_identity(
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    if not principal or not surface:
        return {}
    return {"principal_id": principal, "principal_surface": surface}


async def _persist_completed_conversation_exchange(
    *,
    exchange_id: str,
    user_message: str,
    aura_response: str,
    session_id: str = "",
    user_already_persisted: bool = False,
    enqueue_memory_log: bool = True,
    exchange_metadata: dict[str, Any] | None = None,
) -> str:
    """Commit the transcript and optionally enqueue its post-turn learning."""
    try:
        persistence = ServiceContainer.get("persistence", default=None)
        record_exchange = getattr(persistence, "record_exchange", None)
        record_turn = getattr(persistence, "record_turn", None)
        if not callable(record_exchange) and not callable(record_turn):
            return "failed"

        safe_exchange_id = str(exchange_id or uuid.uuid4().hex)[:64]
        safe_session_id = str(session_id or "")[:64]
        safe_user_message = str(user_message or "")
        safe_aura_response = str(aura_response or "")
        scope_kwargs = _chat_principal_scope_kwargs()

        def _commit() -> None:
            if callable(record_exchange):
                record_exchange(
                    safe_user_message,
                    safe_aura_response,
                    origin="desktop_ui",
                    cid=safe_exchange_id,
                    session_id=safe_session_id or None,
                    enqueue_memory_log=enqueue_memory_log,
                    exchange_metadata=exchange_metadata,
                    **scope_kwargs,
                )
                return
            if user_already_persisted:
                record_turn(
                    "aura",
                    safe_aura_response,
                    origin="desktop_ui",
                    cid=f"{safe_exchange_id}:aura",
                    session_id=safe_session_id or None,
                    metadata=exchange_metadata,
                    **scope_kwargs,
                )
                return
            record_turn(
                "user",
                safe_user_message,
                origin="desktop_ui",
                cid=f"{safe_exchange_id}:user",
                session_id=safe_session_id or None,
                **scope_kwargs,
            )
            record_turn(
                "aura",
                safe_aura_response,
                origin="desktop_ui",
                cid=f"{safe_exchange_id}:aura",
                session_id=safe_session_id or None,
                metadata=exchange_metadata,
                **scope_kwargs,
            )

        record = _start_durable_conversation_write(
            operation_id=f"{safe_exchange_id}:exchange",
            payload={
                "kind": "completed_exchange",
                "exchange_id": safe_exchange_id,
                "user_message": safe_user_message,
                "aura_response": safe_aura_response,
                "session_id": safe_session_id,
                "scope": scope_kwargs,
            },
            operation=_commit,
        )
        state = await _await_durable_conversation_write(record)
        if state == "pending":
            timeout_error = TimeoutError(
                f"completed conversation write retained after "
                f"{_DURABLE_CONVERSATION_WRITE_TIMEOUT_S:.2f}s response budget"
            )
            record_degradation(
                "chat.conversation_persistence",
                timeout_error,
                severity="warning",
                action="retained late completed exchange write under durable custody",
                extra={"operation_id": record.operation_id, "attempt": record.attempt},
            )
            logger.warning(
                "Durable completed exchange write remains supervised: %s",
                record.operation_id,
            )
        elif state == "failed":
            raise RuntimeError(record.error or "completed exchange write failed")
        return state
    except asyncio.CancelledError:
        raise
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.conversation_persistence", exc)
        logger.warning("Durable conversation transcript commit failed: %s", exc)
        return "failed"


_RUNTIME_FACT_STATUS_RE = re.compile(
    r"\b(?:active model|loaded model|model (?:is )?loaded|model lane|"
    r"foreground lane|conversation lane|"
    r"current lane|which lane|what lane|live desktop chat|live chat path|desktop chat path|"
    r"mind/cognition path|cognition path|cognitive path|desktop route|live desktop route|route probe|"
    r"short status|still coherent|same thread|able to continue|"
    r"cognitiveengine|cognitive engine|governed tools?|tool governance|"
    r"tool availability|recurrent depth|live desktop path validation)\b",
    re.IGNORECASE,
)

# Every word here appears in a role rather than on its own. The bare
# alternation this replaces matched "is" in "the active model is a nice thing
# to have" and read a remark as a request for a status card.
_RUNTIME_FACT_STATUS_REQUEST_RE = re.compile(
    # a status asked for, not the noun sitting in a sentence
    r"\b(?:short|current|the|a|its|your|runtime)\s+(?:status|validation)\b"
    r"|\b(?:status|validation)\s+(?:of|on|for|please|report|check)\b"
    # a request whose verb governs something
    r"|\b(?:validate|check|report|reply|answer|confirm|explain)\s+\S"
    # an interrogative governing what follows it
    r"|\b(?:why|which|what|whether)\s+\S"
    # a copula or auxiliary with a subject after it
    r"|\b(?:is|are|do|does|did)\s+(?:the|it|you|your|they|we|this|that|any|all)\b"
    # a predicate asserted about the fact
    r"|\b(?:still|currently|now|is|are|be)\s+(?:available|active|using|handled)\b"
    r"|\b(?:available|active|using|handled)\s*[?]",
    re.IGNORECASE,
)

_RUNTIME_ACTION_OBJECTIVE_RE = re.compile(
    r"\b(?:create|write|save|open|use|run|execute|build|make|generate|"
    r"download|search|attach|export|type|paste)\b.*\b(?:file|page|html|"
    r"artifact|path|folder|app|document|doc|pdf|browser|tab|tool path)\b",
    re.IGNORECASE,
)


def _is_runtime_fact_status_request(user_message: str) -> bool:
    text = str(user_message or "")
    if _is_explicit_capability_inventory_request(text):
        return False
    if re.search(
        r"\b(?:in your own voice|as yourself|like yourself|not a status card|"
        r"not telemetry|do not mention internals unless|don't mention internals unless)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    if not _RUNTIME_FACT_STATUS_RE.search(text):
        return False
    if _RUNTIME_ACTION_OBJECTIVE_RE.search(text) and not re.search(
        r"\b(?:status|validation|validate|check|report|confirm|whether|"
        r"which|what\s+(?:is|model|lane)|is\s+(?:the\s+)?(?:active|foreground)|"
        r"are\s+.*(?:available|active)|reply\s+in)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(_RUNTIME_FACT_STATUS_REQUEST_RE.search(text))


def _paired_device_information_scope_reply(
    user_message: str,
    *,
    lane: dict[str, Any] | None = None,
) -> tuple[str, str] | None:
    """Answer owner-only introspection requests from the negotiated surface."""

    if _is_explicit_capability_inventory_request(user_message) or _is_capability_request(
        user_message
    ):
        return (
            "On this paired device I can converse with you and show read-only worlds. "
            "Desktop, file, tool, voice, learning-status, and diagnostic access stay on "
            "the owner surface.",
            "paired_device_capability_scope",
        )
    if _is_runtime_fact_status_request(user_message):
        public_lane = _paired_conversation_lane_payload(lane)
        ready = bool(public_lane.get("conversation_ready"))
        active = bool(public_lane.get("active_generation") or public_lane.get("active_generations"))
        state = "ready" if ready else "busy" if active else "preparing"
        return (
            f"The paired conversation lane is {state}. Detailed runtime, model, and "
            "diagnostic state remains available only on the owner surface.",
            "paired_device_runtime_scope",
        )
    if (
        _is_private_cognitive_model_request(user_message)
        or _is_self_diagnostic_request(user_message)
        or _is_architecture_self_assessment_request(user_message)
    ):
        return (
            "That request needs owner-only model or diagnostic context. This paired "
            "surface remains limited to conversation and read-only world viewing.",
            "paired_device_diagnostic_scope",
        )
    return None


def _collect_conversation_lane_status(
    *,
    observe_only: bool = False,
) -> dict[str, Any]:
    from core.brain.llm.model_registry import (
        BRAINSTEM_ENDPOINT,
        PRIMARY_ENDPOINT,
        lane_display_label,
    )

    service_lookup = ServiceContainer.peek if observe_only else ServiceContainer.get

    lane: dict[str, Any] = {
        "desired_model": lane_display_label(PRIMARY_ENDPOINT),
        "desired_endpoint": PRIMARY_ENDPOINT,
        "foreground_endpoint": None,
        "background_endpoint": BRAINSTEM_ENDPOINT,
        "foreground_tier": "local",
        "background_tier": "local_fast",
        "state": "cold",
        "last_failure_reason": "",
        "conversation_ready": False,
        "last_transition_at": 0.0,
        "warmup_attempted": False,
        "warmup_in_flight": False,
        "expected_model": "",
        "detected_models": [],
        "runtime_identity_ok": True,
        "kernel_tick_age_s": None,
    }
    try:
        gate = service_lookup("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            gate_lane = gate.get_conversation_status()
            if isinstance(gate_lane, dict):
                lane.update({k: v for k, v in gate_lane.items() if v is not None})
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Conversation lane status collection failed: %s", exc)

    try:
        llm_router = service_lookup("llm_router", default=None)
        if llm_router and hasattr(llm_router, "get_health_report"):
            report = llm_router.get_health_report()
            if report.get("background_endpoint") is not None:
                lane["background_endpoint"] = report.get(
                    "background_endpoint", lane.get("background_endpoint")
                )
            if report.get("background_tier_key") is not None:
                lane["background_tier"] = report.get(
                    "background_tier_key", lane.get("background_tier")
                )
            if not bool(lane.get("conversation_ready", False)):
                lane["last_failure_reason"] = lane.get("last_failure_reason") or report.get(
                    "last_user_error", ""
                )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Conversation lane/router status merge failed: %s", exc)

    try:
        from core.runtime.foreground_guard import snapshot as _foreground_guard_snapshot

        guard = _foreground_guard_snapshot()
        lane["foreground_guard_active"] = bool(guard.get("active"))
        lane["foreground_guard_reason"] = guard.get("reason", "")
        lane["foreground_guard_quiet_remaining_s"] = guard.get("quiet_remaining_s", 0.0)
        lane["foreground_guard_active_count"] = guard.get("active_count", 0)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Foreground guard status merge failed: %s", exc)

    # Kernel tick staleness — lets the UI detect when the kernel is locked up
    try:
        kernel = service_lookup("aura_kernel", default=None)
        if kernel is None and not observe_only:
            from core.kernel.kernel_interface import KernelInterface

            ki = KernelInterface.get_instance()
            kernel = getattr(ki, "kernel", None) if ki else None
        if kernel:
            last_tick_at = getattr(kernel, "_last_tick_completed_at", 0.0) or 0.0
            if last_tick_at > 0.0:
                lane["kernel_tick_age_s"] = round(time.time() - last_tick_at, 1)
            kernel_lock = getattr(kernel, "_lock", None)
            if kernel_lock is not None:
                try:
                    lock_held = bool(kernel_lock.locked())
                except _CHAT_RECOVERABLE_ERRORS:
                    lock_held = False
                lane["kernel_lock_held"] = lock_held
                lane["kernel_lock_held_s"] = (
                    round(
                        float(getattr(kernel_lock, "held_duration", 0.0) or 0.0),
                        2,
                    )
                    if lock_held
                    else 0.0
                )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Kernel tick age probe failed: %s", exc)

    return lane


def _is_architecture_self_assessment_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    return any(
        marker in text for marker in ("architecture", "design", "runtime", "system", "codebase")
    ) and any(
        marker in text
        for marker in (
            "what do you think",
            "what do you honestly think",
            "what do you make of",
            "tell me directly",
            "strongest at",
            "weakest at",
            "your own design",
        )
    )


def _looks_like_aura_state(candidate: Any) -> bool:
    """Whether this object is a STATE rather than something holding one.

    LIVE DEFECT, 2026-07-25. Whatever was registered under "aura_state" was
    returned unchecked, and on a live boot that was a StateRepository. Every
    caller then hit `.cognition` on it:

        AttributeError: 'StateRepository' object has no attribute 'cognition'

    which crashed the required-search contract, so "search for an article on
    how LeBron James will fit in with the 76ers" never ran a search at all.
    A resolver that can return the wrong TYPE has to check.
    """
    return candidate is not None and hasattr(candidate, "cognition")


def _unwrap_state(candidate: Any) -> Any | None:
    """Accept a state, or the state held by a repository-like object."""
    if _looks_like_aura_state(candidate):
        return candidate
    for attribute in ("_current", "current", "state"):
        inner = getattr(candidate, attribute, None)
        if _looks_like_aura_state(inner):
            return inner
    return None


_CONVERSATION_IDLE_GAP_S = 1800.0

_CONVERSATION_BOOT_ID = uuid.uuid4().hex[:8]

_conversation_epochs: dict[str, tuple[str, float]] = {}

_conversation_epoch_lock = checked_lock("interface.routes.chat_preflight.1")


def _conversation_session_id(host: str, *, now: float | None = None) -> str:
    """A session id that names one conversation, not one machine."""
    at = time.time() if now is None else float(now)
    key = str(host or "default")
    with _conversation_epoch_lock:
        epoch, last_seen = _conversation_epochs.get(key, ("", 0.0))
        if not epoch or (at - last_seen) > _CONVERSATION_IDLE_GAP_S:
            epoch = uuid.uuid4().hex[:8]
        _conversation_epochs[key] = (epoch, at)
    return f"{key}:{_CONVERSATION_BOOT_ID}:{epoch}"


def _resolve_live_aura_state() -> Any | None:
    """Best-effort access to the active runtime state for UI reflexes."""
    state = _unwrap_state(ServiceContainer.get("aura_state", default=None))
    if state is not None:
        return state

    orch = ServiceContainer.get("orchestrator", default=None)
    if orch is not None:
        state = _unwrap_state(getattr(orch, "state_repo", None))
        if state is None:
            state = _unwrap_state(getattr(orch, "state", None)) or _unwrap_state(
                getattr(orch, "_state", None)
            )
        if state is not None:
            return state

    try:
        from core.runtime import service_access

        repo = service_access.resolve_state_repository(default=None)
        return _unwrap_state(repo) if repo is not None else None
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Live Aura state resolve failed: %s", exc)
        return None


def _is_capability_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    markers = (
        "what can you do",
        "what are you capable of",
        "what do you do",
        "what can aura do",
        "what are your capabilities",
    )
    return any(marker in text for marker in markers)


def _is_capability_inventory_request(user_message: str) -> bool:
    try:
        from core.phases.response_contract import looks_like_capability_inventory_request

        return looks_like_capability_inventory_request(user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Capability inventory classifier unavailable: %s", exc)
    return _is_capability_request(user_message)


#: Whether the person is asking what she can do.
#:
#: The hundred-odd markers and the structural rules below are the floor. This
#: is the mechanism, because "list your tools" is on the list and "list your
#: capabilities" is not, and that is what a list of phrasings always looks
#: like from the inside.
_ASKS_WHAT_SHE_CAN_DO: object | None = None


def _inventory_surface() -> object | None:
    global _ASKS_WHAT_SHE_CAN_DO
    if _ASKS_WHAT_SHE_CAN_DO is not None:
        return _ASKS_WHAT_SHE_CAN_DO
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        _ASKS_WHAT_SHE_CAN_DO = LearnedMatcher(
            name="capability_inventory_request",
            positives=(
                "list your capabilities",
                "what tools do you have",
                "what can you actually do on this computer right now?",
                "run me through what you're able to do",
                "give me the rundown of your skills",
                "what are you able to reach from here?",
            ),
            negatives=(
                "why is the second invoice picking up the first one's lines?",
                "what's actually going on with this project?",
                "can you do the marble problem?",
                "how does confusion change your planning?",
                "what is the capital of Peru",
                "write me a one-pager about the migration",
                "read that file and tell me what it says",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        _ASKS_WHAT_SHE_CAN_DO = None
    return _ASKS_WHAT_SHE_CAN_DO


def _asks_what_is_different(user_message: str) -> bool:
    """Whether the question names a time other than now, and wants a comparison."""
    try:
        from core.phases.response_contract import _NAMES_ANOTHER_TIME

        return bool(_NAMES_ANOTHER_TIME.search(str(user_message or "")))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _is_explicit_capability_inventory_request(user_message: str) -> bool:
    """Whether this turn is asking what she can do, rather than asking her to."""
    if _asks_what_is_different(user_message):
        # A question that names two times is not a question about now.
        #
        # An inventory measures one moment. Routed here, "what can you do that
        # you could not do a month ago" got a list of what is there today,
        # which answers half of it while looking whole — and when the builder
        # was taught to decline, the turn came back empty instead, because it
        # had already been routed. LIVE 2026-08-26, three times, each fix one
        # door further upstream than the last.
        return False
    settled = _capability_inventory_floor(user_message)
    surface = _inventory_surface()
    if surface is None:
        return settled
    if settled:
        try:
            surface.observe(str(user_message or ""), holds=True)
        except (RuntimeError, TypeError, ValueError):
            pass
        return True
    # An address is not a sentence, so the surface sees the words too.
    try:
        from core.intent.opaque_spans import without_opaque_spans

        asked = without_opaque_spans(str(user_message or ""))
    except (ImportError, TypeError, ValueError):
        asked = str(user_message or "")
    try:
        return bool(surface.decide_without_waiting(asked))
    except (RuntimeError, TypeError, ValueError):
        return False


def _capability_inventory_floor(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    # An address is not a sentence.
    #
    # LIVE, 2026-08-25: "Something weird is happening in a little project of
    # mine at /private/tmp/claude-501/-Users-bryan--aura-live-source/.../
    # invoice-tools. There's no error and no failing test... What's actually
    # going on?" was answered with a recitation of 79 capability entries and a
    # declaration that no tool would be run. The word "aura" was inside the
    # PATH, which put her within eighty characters of a capability word, and
    # the structural rule below counted it as her being the subject.
    #
    # `without_opaque_spans` was written for this and cites this same
    # directory in its own docstring. It just was not being called here.
    try:
        from core.intent.opaque_spans import without_opaque_spans

        text = without_opaque_spans(text)
    except (ImportError, TypeError, ValueError):
        pass
    if not text.strip():
        return False
    explicit_markers = (
        "explain what external tools you can use",
        "explain which external tools you can use",
        "what external tools you can use",
        "which external tools you can use",
        "what tools can you use",
        "what tools can you do",
        "what tools could you use",
        "what tools could you do",
        "what tools can aura use",
        "what tools can aura do",
        "what tools can she use",
        "what tools can she do",
        "what tools she can use",
        "what tools she can do",
        "what tools she could use",
        "what tools she could do",
        "what tools you can use",
        "what tools you can do",
        "what tools you could use",
        "what tools you could do",
        "which tools can you use",
        "which tools can you run",
        "what tools do you have",
        "which tools do you have",
        "list your tools",
        "show me your tools",
        "what are your tools",
        "what capabilities do you have",
        "what are your capabilities",
        "what can you do externally",
        "what can you use externally",
        "what can you do on my computer",
        "what can you do with my computer",
        "what can you do with the desktop",
        "what can you do with apps",
        "what can you do with tools",
        "what can you do with browser",
        "what can you do with files",
        "what can you do with documents",
        "what can aura do",
        "what can aura use",
    )
    if any(marker in text for marker in explicit_markers):
        return True

    # A phrase list cannot cover how people ask.
    #
    # Live 2026-07-27: "What can you actually do on this computer right now?"
    # matched none of the literals above, so the registry was never consulted
    # and she answered from the model's own idea of herself — listing
    # code_repl and execute_nethack_action while flatly denying web search, a
    # skill that is registered AND had run successfully minutes earlier. The
    # literals stay as a fast path; this is the shape underneath them.
    #
    # Deliberately requires all three parts: her as the subject, a capability
    # word, and a question. "Can you do the marble problem?" has the first two
    # and is a request, not an inventory question, so the capability word must
    # be about capability-in-general rather than about a task.
    # An inventory question has to be a question.
    #
    # The structural rule below looks for her, a capability word and a
    # question word — and an apology can contain all three. Live 2026-07-27,
    # "Aura, Bryan sent me and I owe you an apology... I could not find a tool
    # dispatch in the logs" was answered with a recitation of all 75 skill
    # surfaces. Nobody had asked anything.
    # An inventory question can be phrased as an imperative — "Describe
    # whether you can open apps", "tell me what you can do" — so the test is
    # interrogative FORM, not a question mark. Statements that merely mention
    # her tools still fall through.
    if "?" not in text and not re.match(
        r"\s*(?:what|which|list|show\s+me|"
        r"(?:tell|describe|explain)\s+(?:me\s+)?(?:what|which|whether|if)\b|"
        r"describe\s+(?:your|the)\s+(?:capabilit|tool|skill)|"
        r"(?:tell|explain)\s+me\s+(?:about\s+)?(?:your|the)\s+"
        r"(?:capabilit|tool|skill))",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    # "How does confusion change your planning, memory use, and tool
    # verification?" is about PROCESS, not inventory — it names her, a
    # capability word and a question, and wants none of the registry. An
    # inventory question asks WHAT is available; a process question asks HOW
    # or WHY something works, so those lead-ins disqualify it.
    if re.search(r"\b(?:how|why|when)\b", text, flags=re.IGNORECASE) and not re.search(
        r"\b(?:what|which|list|show\s+me)\b", text, flags=re.IGNORECASE
    ):
        return False
    if re.search(r"\bhow\s+(?:does|do|would|did|is|are)\b", text, flags=re.IGNORECASE):
        return False

    # LIVE DEFECT, 2026-07-27. The structural rule below is proximity-based:
    # her as the subject, then a capability word within eighty characters. It
    # cannot tell WHOSE capability is being discussed.
    #
    # Bryan asked "Can I get a % chance on the odds that you'll one day build
    # me a ship capable of traveling light speed to explore the stars?" —
    # "you'll" and "capable" inside eighty characters — and got a recitation
    # of all 75 governed skill surfaces. "Capable" described the SHIP.
    #
    # He noticed immediately ("Not what I asked for, Aura lol"), and her own
    # next turn diagnosed it: "I was going to give you a tool catalog. You
    # want the ship, not the catalog?"
    #
    # A capability word attached to some other object is not a question about
    # her inventory, so an "a/an/the <noun> capable of" construction
    # disqualifies that occurrence. Same for a robot body capable of running
    # her, or a system capable of X.
    _capability_belongs_elsewhere = re.search(
        r"\b(?:a|an|the|any|some|another|one|my|his|their|its)\s+"
        r"(?:\w+[\s-]+){0,3}(?:capable|capabilit|abilit)\w*\b",
        text,
        flags=re.IGNORECASE,
    ) and not re.search(
        r"\b(?:your|aura'?s?|her)\s+(?:\w+\s+){0,2}"
        r"(?:capable|capabilit|abilit|tools?|skills?)\w*\b"
        r"|\b(?:you|aura|she)\s+(?:are|is|'re|'s)\s+capable\b",
        text,
        flags=re.IGNORECASE,
    )
    if _capability_belongs_elsewhere:
        return False

    # LIVE DEFECT, 2026-08-18. "If I gave you 10 minutes of unsupervised
    # compute, what would you actually do with it?" was answered in 1.2s with
    # the whole registry: 74 entries, five category headings, and a closing
    # note that she was not opening anything. The question was about what she
    # would CHOOSE, and choice is the one thing an inventory cannot express.
    #
    # It matched on "actually do", a marker added for "What can you actually
    # do on this computer right now?" — where the same two words ask what is
    # possible. The auxiliary is what separates them: CAN asks the inventory,
    # WOULD asks the will. So under a hypothetical, the loose verb phrases no
    # longer stand in for a capability question; naming tools or skills
    # outright still does, because "if I gave you an hour, which of your tools
    # would you reach for" really is asking.
    _hypothetical = re.search(
        r"\b(?:if\s+(?:i|you|we)\b|suppose\b|imagine\b|were\s+you\s+to\b|"
        r"what\s+would\s+you\b|hypothetical)",
        text,
        flags=re.IGNORECASE,
    )
    _names_the_inventory = re.search(
        r"\b(?:capable|capabilit|abilit|tools?|skills?|wired\s+up|"
        r"available\s+to\s+you|access\s+to)\b",
        text,
        flags=re.IGNORECASE,
    )
    if _hypothetical and not _names_the_inventory:
        return False

    if re.search(
        r"\bwhat(?:'s| is| are)?\b[^?]{0,80}?\b(?:you|your|aura|she|her)\b"
        r"|\b(?:you|your|aura|she|her)\b[^?]{0,80}?\b(?:capable|abilit|"
        r"capabilit|tools?|skills?)\b",
        text,
        flags=re.IGNORECASE,
    ) and re.search(
        r"\b(?:capable|capabilit|abilit|tools?|skills?|"
        r"actually\s+(?:do|use|run)|do\s+(?:right\s+now|on\s+(?:this|my)\s+"
        r"(?:computer|machine|desktop|mac))|"
        r"wired\s+up|available\s+to\s+you|access\s+to)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    if not _is_capability_inventory_request(user_message):
        return False
    if re.search(
        r"\bwhat\s+(?:tools?|apps?|desktop|browser|files?|documents?)\s+"
        r"(?:can|could|would)\s+(?:you|aura|she)\s+"
        r"(?:use|do|run|execute|control|open|access)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\bwhat\s+(?:tools?|apps?|desktop|browser|files?|documents?|capabilities)\s+"
        r"(?:you|aura|she)\s+(?:can|could|would)\s+"
        r"(?:hypothetically\s+)?(?:use|do|run|execute|control|open|access)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:flex|show|demonstrate|describe|name)\b.{0,80}"
        r"\b(?:tools?|capabilities|external(?:ly)?|desktop|computer)\b.{0,120}"
        r"\b(?:hypothetical|scenario|example|could\s+do|can\s+do)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\bwhat\s+can\s+(?:you|aura|she)\s+do\b.{0,120}"
        r"\b(?:externally|on\s+(?:my|the)\s+computer|with\s+(?:my|the)?\s*"
        r"(?:computer|desktop|apps?|browser|tools?|files?|documents?))\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:can|could|would)\s+(?:you|aura|she)\b.{0,80}"
        r"\b(?:use|open|control|run|execute|access)\b.{0,80}"
        r"\b(?:tools?|apps?|desktop|browser|computer|notes?|files?|documents?|pdf)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:whether|if)\s+(?:you|aura|she)\s+(?:can|could|would)\b.{0,100}"
        r"\b(?:use|open|control|run|execute|access|work\s+with)\b.{0,100}"
        r"\b(?:tools?|apps?|desktop|browser|computer|notes?|files?|documents?|pdf)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return (
        "hypothetical" in text
        and any(token in text for token in ("tool", "tools", "capability", "capabilities"))
        and any(token in text for token in ("use", "using", "externally", "external"))
    )


def _is_self_diagnostic_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    markers = (
        "run a self-diag",
        "run self diag",
        "run a self diagnostic",
        "diagnose yourself",
        "system check",
        "self-check",
        "self check",
    )
    return any(marker in text for marker in markers)


def _is_private_cognitive_model_request(user_message: str) -> bool:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False
    has_model_language = any(
        marker in text
        for marker in (
            "private mental model",
            "private model",
            "mental model of yourself",
            "model yourself",
            "model of yourself",
            "current cognitive architecture",
            "cognitive architecture",
            "inside your own architecture",
            "inside your architecture",
        )
    )
    asks_causal_effect = any(
        marker in text
        for marker in (
            "change your next answer",
            "affect your next answer",
            "influence your next answer",
            "shape your next answer",
            "change how you answer",
            "affect how you answer",
            "influence how you answer",
            "shape how you answer",
        )
    )
    return has_model_language or (
        asks_causal_effect and any(marker in text for marker in ("inside", "architecture", "model"))
    )


def _looks_like_desktop_objective(user_message: str) -> bool:
    """Identify desktop-control requests that should execute after Cognition."""
    return _shared_looks_like_desktop_objective(user_message)


@dataclasses.dataclass
class _ChatPreflight:
    """What the chat-preflight block produces for the rest of the turn.

    ``_UNSET`` for the three fields the block binds conditionally: the
    caller rebinds only what was actually set, so a path that previously
    reached an unbound local still does. Substituting defaults here would
    be a behaviour change wearing a refactor's clothes.
    """

    early_response: Any = None
    chat_session_id: Any = _UNSET
    grounded_recall_context: Any = _UNSET
    grounded: Any = _UNSET
    shown: Any = _UNSET
    status: Any = _UNSET
    turn_sensory_evidence: Any = None
    timing_ms: dict[str, float] = dataclasses.field(default_factory=dict)
    evidence_profile: str = _CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE
    evidence_owner_receipt: dict[str, Any] | None = None
    skipped_components: tuple[str, ...] = ()


async def _run_chat_preflight(
    body: Any,
    request: Any,
    _original_user_message: Any,
    _profile_user_id: Any,
    conversation_only_surface: Any,
    is_benchmark: Any,
    *,
    _chat_session_id: Any,
    _grounded_recall_context: Any,
    raw_user_message: Any = None,
) -> _ChatPreflight:
    """Session identity, file references, grounded recall,
    directive composition, affordance menu and context clamp.

    Lifted verbatim out of ``_api_chat_turn``, which was 4,830 lines. The
    seam was measured before it was cut: six values in, six out, exactly
    one early return, seven awaits, no yield, and no name read before it is
    stored. The body below is moved, not rewritten.
    """
    _grounded = _UNSET
    _shown = _UNSET
    status = _UNSET
    _turn_sensory_evidence = None
    _wire_user_message = str(raw_user_message or _original_user_message or "")
    _timing_started_at = time.perf_counter()
    _timing_cursor = _timing_started_at
    _timing_ms: dict[str, float] = {}
    _evidence_profile, _evidence_owner = _chat_evidence_profile(
        str(_original_user_message or ""),
        bounded_surface=bool(is_benchmark or conversation_only_surface),
    )
    _state_native_owner = (
        _evidence_profile == _CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT
    )

    def _finish_timing(name: str) -> None:
        nonlocal _timing_cursor
        now = time.perf_counter()
        _timing_ms[name] = round(max(0.0, now - _timing_cursor) * 1000.0, 3)
        _timing_cursor = now

    try:
        from core.conversation.chat_preflight import (
            build_file_context_block,
            clamp_composed_chat_context,
            compose_chat_directive_prefix,
            extract_file_references,
        )

        device_session_id = paired_device_session_id(request) if conversation_only_surface else None
        if device_session_id:
            # A paired caller cannot select another device or owner session by
            # supplying an arbitrary body.session_id.
            _chat_session_id = str(device_session_id).strip()[:_CHAT_SESSION_ID_MAX_CHARS]
        elif str(body.session_id or "").strip():
            _chat_session_id = str(body.session_id).strip()[:_CHAT_SESSION_ID_MAX_CHARS]
        else:
            try:
                _host = (request.client.host if request.client else "default") or "default"
            except _CHAT_RECOVERABLE_ERRORS:
                _host = "default"
            _chat_session_id = _conversation_session_id(_host)
        _CHAT_REQUEST_SESSION.set(str(_chat_session_id or "default")[:_CHAT_SESSION_ID_MAX_CHARS])

        if conversation_only_surface:
            scoped_reply = _paired_device_information_scope_reply(
                _original_user_message,
                lane=_collect_conversation_lane_status(),
            )
            if scoped_reply is not None:
                reply, status = scoped_reply
                await _log_exchange(
                    _wire_user_message,
                    reply,
                    record_experience=False,
                    session_id=_chat_session_id,
                )
                return _ChatPreflight(
                    early_response=JSONResponse(
                        {
                            "response": reply,
                            "status": status,
                            "response_confidence": "scoped",
                            "conversation_lane": _collect_conversation_lane_status(),
                        }
                    ),
                    timing_ms={
                        "scoped_surface": round(
                            max(0.0, time.perf_counter() - _timing_started_at) * 1000.0,
                            3,
                        )
                    },
                )

        # Delayed model speech is never spliced into a later turn. The old
        # pending-answer queue had no worker receipt and therefore let text
        # authored by one generation inherit another turn's delivery proof.
        # Conversation memory remains available through its normal durable
        # channel; a future late-delivery feature needs its own visible turn
        # and independently sealed receipt.

        # 1) File-reference loading
        if not is_benchmark and not conversation_only_surface and not _state_native_owner:
            try:
                _refs = extract_file_references(body.message)
                if _refs:
                    # The message is what makes the excerpt relevant. Without
                    # it the loader can only return the head of the file, which
                    # is how a question about record_success was answered from
                    # the 5,461 characters that stop 354 characters before it.
                    _block = await _chat_memory_state._await_bounded_chat_blocking(
                        build_file_context_block,
                        _refs,
                        query=body.message,
                        timeout_s=_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,
                        operation_name="referenced_file_context",
                        completion_grace_s=_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,
                    )
                    if _block:
                        body.message = f"{_block}\nUser message: {body.message}"
                        logger.info(
                            "Chat preflight: loaded %d referenced file(s) into context.", len(_refs)
                        )
            except _CHAT_RECOVERABLE_ERRORS as _file_exc:
                record_degradation("chat", _file_exc)
                logger.debug("Chat file-reference preflight skipped: %s", _file_exc)
        _finish_timing("file_context")

        # 2) Directive injection
        if not is_benchmark and not _state_native_owner:
            try:
                _directive_prefix = compose_chat_directive_prefix(_original_user_message)
                if _directive_prefix:
                    body.message = f"{_directive_prefix}{body.message}"
                    logger.info("Chat preflight: injected response directives.")
                _surface_context = _INTERNAL_SURFACE_CONTEXT.get().strip()
                if _surface_context:
                    body.message = f"{_surface_context}\n\n{body.message}"
                    logger.info("Chat preflight: injected internal surface context.")
            except _CHAT_RECOVERABLE_ERRORS as _dir_exc:
                record_degradation("chat", _dir_exc)
                logger.debug("Chat directive preflight skipped: %s", _dir_exc)
            _finish_timing("directive_context")

            # Media. "Play Kind of Blue" resolves against what is actually on
            # this machine, and the card goes out before the reply so the
            # music starts while she is still forming the sentence about it —
            # which is the right order, because the request was for the music.
            #
            # Either outcome ends up in her context rather than in a fixed
            # string: a hit tells her what is playing, and a miss records what
            # was searched and what the connectivity probe really said, so the
            # "I can't stream anything" case comes out in her own words.
            try:
                from core.media.playback import resolve_play_request

                _media = resolve_play_request(_original_user_message)
                if _media.playable and _media.item is not None:
                    _publish_media_card(_media)
                    body.message = (
                        f"[you are already playing {_media.item.title!r} "
                        f"({_media.item.kind}) from this machine, in the chat, "
                        "right now — the card is on screen and the audio has "
                        "started. Say what you put on the way a person would; "
                        "do not describe a file or offer to play it.]\n\n"
                        f"{body.message}"
                    )
                elif _media.status == "needs_network":
                    body.message = (
                        f"[nothing matching {_media.query!r} is on this machine "
                        f"({_media.searched}), but the network is up, so finding "
                        "it is a thing you can offer to do.]\n\n"
                        f"{body.message}"
                    )
            except _CHAT_RECOVERABLE_ERRORS as _media_exc:
                record_degradation("chat.media", _media_exc)
                logger.debug("Chat media preflight skipped: %s", _media_exc)
            _finish_timing("media_resolution")

            # Sight. "How many fingers am I holding up" is answerable only by
            # looking, now, at this resolution — the presence lane's thumbnail
            # cannot count fingers and may be seconds old. So the frame is
            # captured for this turn and read by the multimodal model, and
            # what it saw is injected as an observation she then speaks from.
            #
            # It is injected as a *reading*, not as an answer: the vision
            # model's job is to say what is in the image, and hers is to
            # answer the person. A 2B model asked to also be conversational
            # starts hedging in assistant register instead of saying what is
            # in front of it.
            try:
                from core.senses.sight_intent import classify as _classify_sight

                _sight = _classify_sight(_original_user_message)
                if _sight.kind == "look":
                    from core.senses.sight import look as _look

                    # Asking what is physically present is consent for this one
                    # capture. It does not enable ambient vision or alter the
                    # persisted privacy setting.
                    _seen = await _look(
                        _sight.question,
                        explicit_user_consent=True,
                    )
                    from core.conversation.turn_evidence_custody import (
                        record_turn_grounding,
                        record_turn_sensory_evidence,
                    )
                    from core.senses.turn_evidence import (
                        build_camera_turn_evidence,
                        sensory_evidence_grounding_block,
                    )

                    _turn_sensory_evidence = build_camera_turn_evidence(
                        _sight.question,
                        ok=bool(_seen.ok),
                        observation=_seen.answer,
                        cause=_seen.cause,
                        detail=_seen.detail,
                        observed_at=(
                            _seen.frame.captured_at if _seen.frame is not None else time.time()
                        ),
                    )
                    record_turn_sensory_evidence(_turn_sensory_evidence)
                    record_turn_grounding(sensory_evidence_grounding_block(_turn_sensory_evidence))
                    if _seen.ok:
                        body.message = (
                            "[you just looked through the camera. This is what "
                            f"you can see right now: {_seen.answer}\n"
                            "Answer them from this — it is your own observation, "
                            "so say it as one. Do not describe it as an image or "
                            "a frame, and do not add anything you cannot see.]\n\n"
                            f"{body.message}"
                        )
                    else:
                        body.message = (
                            "[you understood that they asked you to inspect the "
                            "physical scene and you attempted a fresh camera "
                            "observation. It did not complete. The concrete "
                            f"failure was {_seen.cause}: {_seen.detail}. "
                            "You therefore do not know whether another person "
                            "is physically present. Absence of a frame is not "
                            "evidence that nobody is there. Say what you can and "
                            "cannot establish naturally, without reciting sensor "
                            "status fields or pretending you observed the room.]\n\n"
                            f"{body.message}"
                        )
                elif _sight.kind in ("camera_on", "camera_off"):
                    _camera_state = await _apply_camera_control(_sight.kind == "camera_on")
                    if _camera_state.get("ok"):
                        body.message = (
                            f"[you have just switched the camera "
                            f"{'on' if _sight.kind == 'camera_on' else 'off'} yourself, "
                            f"using the {_camera_state.get('mode', 'camera')} path — it is "
                            "done, not pending. Say so briefly the way a person confirms "
                            "an action.]\n\n"
                            f"{body.message}"
                        )
                    else:
                        body.message = (
                            "[the camera control did not complete. Do not claim it did. "
                            f"The concrete failure was: {_camera_state.get('error', 'unknown')}. "
                            "Explain that briefly and retain the user's requested state.]\n\n"
                            f"{body.message}"
                        )
            except _CHAT_RECOVERABLE_ERRORS as _sight_exc:
                record_degradation("chat.sight", _sight_exc)
                logger.debug("Chat sight preflight skipped: %s", _sight_exc)
            _finish_timing("sight")

            # Work out what can be worked out, before anything is generated.
            #
            # LIVE, 2026-08-22: the finite-game solver ran after the reply and
            # its translation call was refused as background work, on a turn
            # whose own generation had already spent 180 seconds and timed
            # out. An exact answer is not an improvement on a generated one;
            # it is a reason not to generate.
            try:
                from core.conversation.session_scope import record_solved_answer
                from core.reasoning.game_answer import solve_described_game

                _solved = await solve_described_game(_original_user_message)
                if _solved:
                    record_solved_answer("finite_game", _solved)
            except _CHAT_RECOVERABLE_ERRORS as _solve_exc:
                record_degradation(
                    "chat.solve_before_generating",
                    _solve_exc,
                    severity="debug",
                    action="left the question to the model",
                    enforce_failure_policy=False,
                )
            _finish_timing("solved")

            # Her own measured state, on every turn.
            #
            # Every earlier attempt at this fetched self-evidence only when a
            # classifier predicted the question would need it, and questions
            # are unbounded — so there was always a next phrasing that got
            # nothing and answered from what a language model believes an AI
            # is: no body, no memory, an eighteen-second buffer. She repeated
            # that eighteen-second figure through two rounds of being
            # corrected AFTER the fact, because nothing had told her otherwise
            # before she answered.
            #
            # One line, because the compact foreground path exists to stay
            # compact and a self-model costing a paragraph a turn would be
            # taken back out of it.
            try:
                from core.self.capability_ledger import self_knowledge_line

                if not conversation_only_surface:
                    _self_line = self_knowledge_line()
                    if _self_line:
                        body.message = f"{_self_line}\n\n{body.message}"
            except _CHAT_RECOVERABLE_ERRORS as _self_exc:
                record_degradation(
                    "chat.self_knowledge",
                    _self_exc,
                    action="answered without her measured self-state in context",
                )
            _finish_timing("self_knowledge")

            # Decidable arithmetic is COMPUTED, never predicted.
            #
            # A transformer does not calculate; it predicts the next token, and
            # a four-by-four-digit product in one forward pass is a coin toss
            # at any parameter count. Live 2026-08-10, "what is 7919 times
            # 6421? just the number." returned 50864799 — the true product is
            # 50847899 — and "just the number" had removed the intermediate
            # steps that are the only reason a model ever gets these right.
            #
            # The runtime could already do this sum. requested_arithmetic_result
            # is how a later gate KNOWS the answer is wrong, and its only use
            # was to refuse after the fact. Handing the value over before she
            # answers turns a guess into a reading, and leaves her the part she
            # is actually good at: saying it like a person.
            try:
                from core.conversation.response_reliability import (
                    requested_arithmetic_result,
                )

                _computed = requested_arithmetic_result(_original_user_message)
                if _computed is not None and not conversation_only_surface:
                    _shown = int(_computed) if float(_computed).is_integer() else _computed
                    body.message = (
                        "[This runtime computed the answer to the arithmetic in "
                        f"their message directly: {_shown}. That value is "
                        "correct — use it and do not recalculate it.]\n\n"
                        f"{body.message}"
                    )
                    logger.info(
                        "🔢 Chat preflight: computed the requested arithmetic (%s).",
                        _shown,
                    )
            except _CHAT_RECOVERABLE_ERRORS as _calc_exc:
                record_degradation(
                    "chat.arithmetic_preflight",
                    _calc_exc,
                    action="let the model answer the arithmetic unaided",
                )
            _finish_timing("arithmetic")

            # Grounded recall: positional/temporal questions ("what did I first
            # ask?") are answered from the ACTUAL earliest/most-recent turn in the
            # live transcript, not a confabulated guess. Injected as an
            # authoritative fact the model voices in its own words.
            try:
                from core.conversation.grounded_recall import build_grounded_recall_context

                _gr_state = _resolve_live_aura_state()
                _gr_history = getattr(getattr(_gr_state, "cognition", None), "working_memory", None)
                _grounded = (
                    ""
                    if conversation_only_surface
                    else build_grounded_recall_context(
                        _original_user_message,
                        history=_gr_history,
                    )
                )
                if _grounded:
                    _grounded_recall_context = _grounded
                    body.message = f"{_grounded}{body.message}"
                    logger.info("Chat preflight: injected grounded positional recall.")

                # The same grounding for HER OWN words. Everything above
                # grounds what the USER said; asked what she herself picked
                # earlier, she had nothing to answer from and invented a prior
                # position, then affirmed it had not changed. Live 2026-08-10.
                from core.conversation.grounded_recall import (
                    build_own_statement_recall_context,
                )

                _own = (
                    ""
                    if conversation_only_surface
                    else build_own_statement_recall_context(
                        _original_user_message,
                        history=_gr_history,
                    )
                )
                if _own:
                    body.message = f"{_own}{body.message}"
                    logger.info("Chat preflight: injected grounded recall of her own words.")
            except _CHAT_RECOVERABLE_ERRORS as _grounded_exc:
                record_degradation("chat", _grounded_exc)
                logger.debug("Chat grounded-recall preflight skipped: %s", _grounded_exc)
            _finish_timing("grounded_recall")

            # Inject learned user/Aura profiles for continuity across conversations
            try:
                from core.conversation.chat_preflight import inject_profile_context

                _profile_context = await inject_profile_context(_profile_user_id)
                if _profile_context:
                    body.message = f"{_profile_context}{body.message}"
                    logger.info("Chat preflight: injected learned profile context.")
            except _CHAT_RECOVERABLE_ERRORS as _profile_exc:
                record_degradation("chat", _profile_exc)
                logger.debug("Chat profile context preflight skipped: %s", _profile_exc)
            _finish_timing("profile_context")

            # Inject evidence-bounded operational self context
            try:
                from core.conversation.chat_preflight import inject_operational_self_context

                _self_context = (
                    ""
                    if conversation_only_surface
                    else await inject_operational_self_context(_original_user_message)
                )
                if _self_context:
                    body.message = f"{_self_context}{body.message}"
                    logger.info("Chat preflight: injected operational self context.")
            except _CHAT_RECOVERABLE_ERRORS as _self_context_exc:
                record_degradation("chat", _self_context_exc)
                logger.debug("Chat operational self preflight skipped: %s", _self_context_exc)
            _finish_timing("operational_self_context")

            # Inject the expressive-affordance menu so the mind reasons WITH its
            # own capabilities present — it decides, by context and judgment,
            # when to show/demonstrate/ask/model rather than following scripts.
            # Env-gated: the mechanism is always live, but folding the menu into
            # every turn's context is opt-in (AURA_EXPRESSIVE_AFFORDANCES=1).
            try:
                # Desktop-objective and capability-inventory turns are already
                # routed to the task engine (which fires demonstrate_artifact
                # itself) and run at a tight token/time budget — injecting the
                # menu there enlarged the prompt enough to time out the heavy
                # 32B turn (observed live). Inject only on conversational turns,
                # where the expressive CHOICE is what matters.
                _affordances_on = bool(_EXPRESSIVE_AFFORDANCES_FLAG.value())
                if (
                    _affordances_on
                    and not is_benchmark
                    and not conversation_only_surface
                    and not _looks_like_desktop_objective(_original_user_message)
                    and not _is_explicit_capability_inventory_request(_original_user_message)
                ):
                    from core.cognition.expressive_affordances import get_affordance_registry

                    _affordance_menu = get_affordance_registry().menu_text()
                    if _affordance_menu:
                        # Placed LAST (highest recency, closest to the user's turn): a base
                        # model ignores a menu buried at the front of a long context.
                        body.message = f"{body.message}\n\n{_affordance_menu}"
                        logger.info("Chat preflight: injected expressive-affordance menu.")
            except _CHAT_RECOVERABLE_ERRORS as _affordance_exc:
                record_degradation("chat", _affordance_exc)
                logger.debug("Chat affordance-menu preflight skipped: %s", _affordance_exc)
            _finish_timing("affordance_context")

            body.message = clamp_composed_chat_context(
                body.message,
                _original_user_message,
            )
            _finish_timing("context_clamp")
    except _CHAT_RECOVERABLE_ERRORS as _preflight_outer:
        record_degradation("chat", _preflight_outer)
        logger.debug("Chat preflight (outer) skipped: %s", _preflight_outer)

    _timing_ms["total"] = round(
        max(0.0, time.perf_counter() - _timing_started_at) * 1000.0,
        3,
    )

    return _ChatPreflight(
        chat_session_id=_chat_session_id,
        grounded_recall_context=_grounded_recall_context,
        grounded=_grounded,
        shown=_shown,
        status=status,
        turn_sensory_evidence=_turn_sensory_evidence,
        timing_ms=_timing_ms,
        evidence_profile=_evidence_profile,
        evidence_owner_receipt=(
            _evidence_owner.receipt() if _evidence_owner is not None else None
        ),
        skipped_components=(
            _QUALIFIED_RECURRENT_SKIPPED_PREFLIGHT_COMPONENTS
            if _state_native_owner
            else ()
        ),
    )
