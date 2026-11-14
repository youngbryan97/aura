"""interface/routes/voice_duplex.py — the /ws/voice transport.

A dedicated socket rather than a mode on ``/ws``. Three reasons, all
practical: voice pushes continuous binary in both directions and would
starve the chat socket's event queue; a voice session owns models and tasks
whose lifetime must be exactly the socket's; and a failure in the voice lane
must never be able to drop the text conversation.

Authentication mirrors ``/ws`` exactly — same local-origin trust, same paired
-device token flow — with one addition: a paired device also needs the
explicit ``voice`` scope, because a microphone is a materially larger grant
than a chat box.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import threading
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.brain.llm.her_pinned_weights import a_voice_session_closed, a_voice_session_opened
from core.brain.llm_health_router_gate_owners import a_live_turn, a_voice_session
from core.conversation.delivered_history import VISIBLE_CONVERSATION_EXCHANGES
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
from core.voice.duplex.config import DuplexConfig
from core.voice.duplex.live_context import HerLiveContext
from core.voice.duplex.mind_bridge import MindBridge
from core.voice.duplex.model_runtime import (
    get_voice_model_runtime,
    voice_model_runtime_status,
)
from core.voice.duplex.session import DuplexVoiceSession
from core.voice.microphone_authority import (
    MicrophoneDenial,
    MicrophoneLease,
    get_audio_ingress_broker,
    get_microphone_authority,
)

logger = logging.getLogger("Aura.Routes.VoiceDuplex")

router = APIRouter()

_VOICE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

# One live session per socket. Tracked so the runtime can report and close
# them, and so a reload cannot strand model handles.
_SESSIONS: dict[str, DuplexVoiceSession] = {}
_SESSION_MICROPHONE_LEASES: dict[str, MicrophoneLease] = {}
_SESSION_RESERVATIONS: dict[str, str] = {}
_SESSION_RESERVATION_LOCK = threading.Lock()
_CHAT_AUTH_HEADERS = frozenset(
    {b"authorization", b"cookie", b"x-api-token", b"x-aura-device-token"}
)


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError) as exc:
        record_degradation("voice_duplex.config", exc, severity="warning")
        value = default
    return max(minimum, min(value, maximum))


MAX_VOICE_SESSIONS = _bounded_env_int(
    "AURA_VOICE_MAX_SESSIONS",
    2,
    minimum=1,
    maximum=8,
)
MAX_VOICE_SESSIONS_PER_PRINCIPAL = _bounded_env_int(
    "AURA_VOICE_MAX_SESSIONS_PER_PRINCIPAL",
    1,
    minimum=1,
    maximum=4,
)
MAX_VOICE_AUDIO_MESSAGE_BYTES = 128 * 1024
MAX_VOICE_COMMAND_MESSAGE_BYTES = 64 * 1024


def _reserve_voice_session(session_id: str, principal: str) -> bool:
    with _SESSION_RESERVATION_LOCK:
        if len(_SESSION_RESERVATIONS) >= MAX_VOICE_SESSIONS:
            return False
        principal_count = sum(
            existing == principal for existing in _SESSION_RESERVATIONS.values()
        )
        if principal_count >= MAX_VOICE_SESSIONS_PER_PRINCIPAL:
            return False
        _SESSION_RESERVATIONS[session_id] = principal
        return True


def _release_voice_session(session_id: str) -> None:
    with _SESSION_RESERVATION_LOCK:
        _SESSION_RESERVATIONS.pop(session_id, None)


def _governed_chat_source_headers(
    source_headers: tuple[tuple[bytes, bytes], ...],
    *,
    owner_token: str | None,
    device_token: str | None,
    local_owner: bool = False,
) -> tuple[tuple[bytes, bytes], ...]:
    """Carry the authenticated voice principal into the governed chat route."""
    principal_count = sum(bool(item) for item in (owner_token, device_token, local_owner))
    if principal_count > 1:
        raise ValueError("voice principal must be exactly one authenticated identity")
    if principal_count == 0:
        return source_headers

    filtered = tuple(
        (name, value)
        for name, value in source_headers
        if name.lower() not in _CHAT_AUTH_HEADERS
    )
    if owner_token:
        return (*filtered, (b"x-api-token", owner_token.encode("utf-8")))
    if device_token:
        return (*filtered, (b"x-aura-device-token", device_token.encode("utf-8")))
    return filtered


def conversation_key_for(device_id: str | None, client_host: str) -> str:
    """The thread a spoken turn belongs to.

    One conversation, however it is being conducted. The socket's own id is a
    fresh uuid per connection — right for bookkeeping, wrong for identity.
    Passing it downstream as the conversation key made a spoken turn and a
    typed turn two different threads to everything that keys on it, and a
    voice socket that merely *reconnected* (the client retries with backoff,
    so this is routine rather than exotic) started a third. Switching from
    talking to typing mid-thought is the most ordinary thing a person does
    with something that sits on their desk, and it must not be the moment she
    loses the thread.

    So the key is derived from the principal, exactly as the desktop's own
    chat turns derive theirs: same person, same conversation, whichever way
    they said it.
    """
    if device_id:
        return f"paired:{device_id}"
    return str(client_host or "")


def active_sessions() -> dict[str, dict[str, Any]]:
    """Status of every live voice session, for health surfaces."""
    return {
        sid: {
            **session.status(),
            "microphone_lease": (
                _SESSION_MICROPHONE_LEASES[sid].to_dict()
                if sid in _SESSION_MICROPHONE_LEASES
                else None
            ),
        }
        for sid, session in _SESSIONS.items()
    }


def model_runtime_status() -> dict[str, Any]:
    """Voice model residency without opening a device or loading weights."""
    return voice_model_runtime_status()


def _voice_output_permitted() -> bool:
    """Respect the user's runtime toggles.

    Voice is the most intrusive output channel Aura has — it makes noise in
    a room. If the user turned it off, the lane refuses to open rather than
    opening silently and surprising them later.
    """
    try:
        from core.runtime.runtime_settings import get_runtime_setting

        return bool(get_runtime_setting("voice.output_enabled", True))
    except _VOICE_ERRORS as exc:
        record_degradation(
            "voice_duplex.route",
            exc,
            action="refused voice output because runtime settings were unreadable",
            severity="warning",
        )
        return False


def _voice_input_permitted() -> bool:
    try:
        from core.runtime.permission_gates import microphone_allowed

        return microphone_allowed()
    except _VOICE_ERRORS as exc:
        record_degradation(
            "voice_duplex.route",
            exc,
            action="refused microphone input because runtime settings were unreadable",
            severity="warning",
        )
        return False


@router.websocket("/ws/voice")
async def voice_duplex_endpoint(ws: WebSocket) -> None:
    # Import here rather than at module scope: interface.server imports this
    # module, so a top-level import would be circular.
    from interface.auth import (
        _extract_device_token,
        device_for_request,
        request_has_allowed_local_browser_origin,
    )
    from interface.server import _live_device_scopes, _verify_ws_device_token, config

    await ws.accept()

    host = ws.client.host if ws.client else "unknown"
    is_local = host in ("127.0.0.1", "::1", "localhost")
    local_origin_ok = request_has_allowed_local_browser_origin(ws)
    expected = str(getattr(config, "api_token", "") or "")

    local_owner_authenticated = is_local and local_origin_ok
    authenticated = local_owner_authenticated
    device_session = None
    explicit_device_token: str | None = None
    explicit_owner_token: str | None = None

    if not authenticated:
        device_session = device_for_request(ws)
        authenticated = device_session is not None
        if device_session is not None:
            explicit_device_token = _extract_device_token(ws)
            authenticated = bool(explicit_device_token)

    reserved_session_id = ""
    microphone_lease: MicrophoneLease | None = None
    try:
        if not authenticated:
            # Same 5-second credential exchange the chat socket uses.
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                data = json.loads(raw)
            # not a failure: text that is not the JSON this expects is not a record it can
            # read back.
            except TimeoutError:
                await ws.close(code=4001, reason="Auth Timeout")
                return
            # not a failure: text that is not the JSON this expects is not a record it can
            # read back.
            except json.JSONDecodeError:
                await ws.close(code=4001, reason="Invalid Auth Payload")
                return
            if not isinstance(data, dict):
                await ws.close(code=4001, reason="Invalid Auth Payload")
                return

            token = str(data.get("token", "") or "")
            if data.get("type") == "auth" and token.startswith("adt1."):
                device_session = _verify_ws_device_token(token)
                if device_session is not None:
                    explicit_device_token = token
                    authenticated = True
            elif data.get("type") == "auth" and expected:
                authenticated = hmac.compare_digest(token, expected)
                if authenticated:
                    explicit_owner_token = token

            if not authenticated:
                await ws.send_text(json.dumps({"type": "voice.error", "message": "Unauthorized"}))
                await ws.close(code=4001, reason="Unauthorized")
                return

        # A microphone is a bigger grant than a chat box, so a paired device
        # needs the voice scope explicitly. Deny by default.
        if device_session is not None:
            if "voice" not in _live_device_scopes(device_session.device_id):
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "voice.error",
                            "status": "paired_device_voice_scope_denied",
                            "message": (
                                "Voice is not enabled for this paired device. Ask the "
                                "owner to grant the voice scope."
                            ),
                        }
                    )
                )
                await ws.close(code=4003, reason="Voice scope required")
                return

        if not _voice_input_permitted() or not _voice_output_permitted():
            await ws.send_text(
                json.dumps(
                    {
                        "type": "voice.error",
                        "status": "voice_disabled_in_settings",
                        "message": "Voice is turned off in Runtime Settings.",
                    }
                )
            )
            await ws.close(code=4004, reason="Voice disabled")
            return

        session_id = uuid.uuid4().hex[:12]
        principal = (
            f"paired:{device_session.device_id}"
            if device_session is not None
            else f"owner:{host}"
        )
        if not _reserve_voice_session(session_id, principal):
            await ws.send_text(
                json.dumps(
                    {
                        "type": "voice.error",
                        "status": "voice_session_capacity_reached",
                        "message": "The bounded voice-session capacity is already in use.",
                    }
                )
            )
            await ws.close(code=4429, reason="Voice session capacity reached")
            return
        reserved_session_id = session_id

        requested_mode = "focused"
        query_params = getattr(ws, "query_params", None)
        if query_params is not None:
            try:
                candidate_mode = str(
                    query_params.get("mode", "focused") or "focused"
                )
            except (AttributeError, TypeError, ValueError):
                candidate_mode = "focused"
            if candidate_mode.strip().lower() == "ambient":
                requested_mode = "ambient"

        microphone_authority = get_microphone_authority()
        lease_result = await asyncio.to_thread(
            microphone_authority.acquire,
            f"voice_duplex:{session_id}",
            principal=principal,
            source="browser_duplex",
            mode=requested_mode,
            session_id=session_id,
            preemptible=requested_mode != "focused",
        )
        if isinstance(lease_result, MicrophoneDenial):
            await ws.send_text(
                json.dumps(
                    {
                        "type": "voice.error",
                        "status": f"microphone_{lease_result.reason}",
                        "message": lease_result.detail
                        or "The microphone is unavailable.",
                        "remedy": lease_result.remedy,
                    }
                )
            )
            await ws.close(code=4004, reason="Microphone unavailable")
            return
        microphone_lease = lease_result
        _SESSION_MICROPHONE_LEASES[session_id] = microphone_lease
        ingress_broker = get_audio_ingress_broker()

        send_lock = asyncio.Lock()
        transport_stop = asyncio.Event()
        authorization_revoked = asyncio.Event()
        source_headers = _governed_chat_source_headers(
            tuple(ws.scope.get("headers") or ()),
            owner_token=explicit_owner_token,
            device_token=explicit_device_token,
            local_owner=local_owner_authenticated,
        )

        def principal_authorized() -> bool:
            nonlocal device_session
            if explicit_owner_token is not None:
                current_expected = str(getattr(config, "api_token", "") or "")
                if not current_expected or not hmac.compare_digest(
                    explicit_owner_token,
                    current_expected,
                ):
                    authorization_revoked.set()
                    return False
                return True
            if device_session is None:
                return True
            refreshed = (
                _verify_ws_device_token(explicit_device_token)
                if explicit_device_token
                else device_for_request(ws)
            )
            if (
                refreshed is None
                or refreshed.device_id != device_session.device_id
                or "voice" not in _live_device_scopes(refreshed.device_id)
            ):
                authorization_revoked.set()
                return False
            device_session = refreshed
            return True

        async def send_json(payload: dict[str, Any]) -> None:
            # Serialise sends: the session emits from several tasks at once
            # (reflex loop, turn loop, filler loop) and Starlette's socket
            # is not safe against interleaved concurrent writes.
            async with send_lock:
                if not principal_authorized():
                    raise RuntimeError("voice_principal_authorization_revoked")
                await ws.send_text(json.dumps(payload, default=str))
            if payload.get("type") == "voice.state":
                # A live turn runs from their first word to her last played (llm_health_router_gate_owners.py).
                a_live_turn(conversation_key, payload.get("state") in ("user_speaking", "thinking", "speaking"))
                if payload.get("state") == "user_speaking":
                    live.a_person_began_to_speak()  # her live context reads the turn's evidence while they talk

        async def send_binary(payload: bytes) -> None:
            async with send_lock:
                if not principal_authorized():
                    raise RuntimeError("voice_principal_authorization_revoked")
                await ws.send_bytes(payload)

        # One conversation, however it is being conducted. See
        # `conversation_key_for` for why this is not the socket's own id.
        conversation_key = conversation_key_for(
            device_session.device_id if device_session is not None else None,
            host,
        )

        async def her_live_record(words: str, reply: str) -> None:
            from interface.routes import chat_preflight

            await chat_preflight._log_exchange(words, reply, session_id=conversation_key)

        live = HerLiveContext(conversation_key, record=her_live_record)

        async def responder(
            transcript: str,
            *,
            effective_message: str,
            session_id: str,
            timeout_s: float,
            reply_stream: Any = None,
        ) -> str | None:
            """Her live context answers a conversational turn; her governed route answers the rest, and any turn the
            live context could not (core/voice/duplex/live_context.py)."""
            if live.answers(transcript):
                reply = await live.answer(transcript, reply_stream=reply_stream, timeout_s=timeout_s)
                if reply is not None:
                    return reply
            else:
                logger.info("🗣️ her governed route takes this turn (live context ready=%s)", live.ready)
            live.governed_turn(transcript)
            reply = await governed_responder(transcript, effective_message=effective_message, session_id=session_id,
                                             timeout_s=timeout_s, reply_stream=reply_stream)
            live.heard_elsewhere(transcript, reply or "")
            return reply

        async def governed_responder(
            transcript: str,
            *,
            effective_message: str,
            session_id: str,
            timeout_s: float,
            reply_stream: Any = None,
        ) -> str | None:
            from interface.routes import chat as chat_routes

            surface_context = effective_message
            if transcript and surface_context.endswith(transcript):
                surface_context = surface_context[: -len(transcript)].strip()
            return await chat_routes.run_governed_voice_chat_turn(
                transcript,
                surface_context=surface_context,
                session_id=conversation_key,
                timeout_s=timeout_s,
                source_headers=source_headers,
                client_host=host,
                reply_stream=reply_stream,
            )

        session_config = DuplexConfig.load()
        model_runtime = get_voice_model_runtime(session_config)
        session = DuplexVoiceSession(
            session_id=session_id,
            send_json=send_json,
            send_binary=send_binary,
            config=session_config,
            mind=MindBridge(session_id=session_id, responder=responder, on_cut_off=live.cut_off),
            asr=model_runtime.new_asr(),
            tts=model_runtime.tts,
        )
        _SESSIONS[session_id] = session

        async def authorization_monitor() -> None:
            while not transport_stop.is_set():
                principal_ok = principal_authorized()
                input_ok = _voice_input_permitted()
                output_ok = _voice_output_permitted()
                lease_ok, lease_reason = microphone_authority.validate(
                    microphone_lease
                )
                if not principal_ok or not input_ok or not output_ok or not lease_ok:
                    paired_revocation = device_session is not None
                    authorization_revoked.set()
                    if not principal_ok:
                        status = (
                            "paired_device_session_revoked"
                            if paired_revocation
                            else "owner_session_revoked"
                        )
                        message = "This voice session's authorization was revoked."
                        close_reason = "Voice authorization revoked"
                    elif not input_ok or not output_ok:
                        status = "voice_disabled_in_settings"
                        message = "Voice was turned off in Runtime Settings."
                        close_reason = "Voice disabled"
                    else:
                        status = "microphone_lease_revoked"
                        message = f"Microphone ownership ended: {lease_reason}."
                        close_reason = "Microphone lease revoked"
                    async with send_lock:
                        with contextlib.suppress(*_VOICE_ERRORS):
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "voice.error",
                                        "status": status,
                                        "message": message,
                                    }
                                )
                            )
                            await ws.close(
                                code=4003 if not principal_ok else 4004,
                                reason=close_reason,
                            )
                    return
                microphone_authority.heartbeat(microphone_lease)
                try:
                    await asyncio.wait_for(transport_stop.wait(), timeout=0.25)
                except TimeoutError:
                    continue

        authorization_task = get_task_tracker().create_task(
            authorization_monitor(),
            name=f"VoiceAuthorization:{session_id}",
        )

        async def her_live_context_opens() -> None:
            from interface.routes import chat_memory_state

            # The fetch goes in unawaited, so the context counts as opening from the start (a turn ending now waits).
            await live.open(chat_memory_state._recent_completed_conversation_exchanges(
                current_user_message="", session_id=conversation_key, limit=VISIBLE_CONVERSATION_EXCHANGES,
                allow_cross_session=True))

        pinning: asyncio.Task | None = None
        opening: asyncio.Task | None = None
        held_for_voice = requested_mode == "focused"  # her voice mode, opened by the person; not ambient listening
        try:
            if held_for_voice:
                a_voice_session(session_id, True)
            await session.start()
            logger.info("Voice session %s opened from %s", session_id, host)
            # Her weights pinned while voice mode is open (core/brain/llm/her_pinned_weights.py), and her live context's
            # first read. Both wait on her worker, which answers between jobs, so they run beside the receive loop: LIVE
            # 11 November 2026, awaited before it, they left the socket unread while the browser streamed audio, and the
            # client's keepalive ping went unanswered for 20 s and closed the session before its first turn.
            # The context's read goes to her worker first: it is short, and a first turn waits on it.
            opening = get_task_tracker().create_task(her_live_context_opens(), name=f"VoiceLiveContext:{session_id}")
            pinning = get_task_tracker().create_task(asyncio.to_thread(a_voice_session_opened),
                                                     name=f"VoicePinnedWeights:{session_id}")

            connected = True
            while connected and not authorization_revoked.is_set():
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    connected = False
                    continue

                if not principal_authorized():
                    connected = False
                    continue

                if "bytes" in message and message["bytes"]:
                    if len(message["bytes"]) > MAX_VOICE_AUDIO_MESSAGE_BYTES:
                        await ws.close(code=1009, reason="Voice audio message too large")
                        connected = False
                        continue
                    if not ingress_broker.admit(
                        microphone_lease,
                        len(message["bytes"]),
                    ):
                        await ws.close(code=4004, reason="Microphone lease inactive")
                        connected = False
                        continue
                    await session.feed_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    if (
                        len(message["text"].encode("utf-8", errors="replace"))
                        > MAX_VOICE_COMMAND_MESSAGE_BYTES
                    ):
                        await ws.close(code=1009, reason="Voice command message too large")
                        connected = False
                        continue
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        await session.handle_command(payload)

        finally:
            transport_stop.set()
            authorization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await authorization_task
            _SESSIONS.pop(session_id, None)
            _SESSION_MICROPHONE_LEASES.pop(session_id, None)
            with contextlib.suppress(*_VOICE_ERRORS, asyncio.CancelledError):
                await session.close()
            microphone_authority.release(
                microphone_lease,
                reason="duplex_session_closed",
            )
            microphone_lease = None
            _release_voice_session(session_id)
            reserved_session_id = ""
            if opening is not None and not opening.done():
                opening.cancel()
            if pinning is not None:
                # The release must follow the pin it undoes; the pin is bounded by its own wait on her worker.
                with contextlib.suppress(*_VOICE_ERRORS, asyncio.CancelledError):
                    await pinning
                await asyncio.to_thread(a_voice_session_closed)
            a_live_turn(conversation_key, False)  # a socket that died mid-turn sent no last state
            a_voice_session(session_id, False)
            logger.info("Voice session %s closed", session_id)

    except WebSocketDisconnect as exc:
        logger.debug("Voice socket disconnected: %s", exc)
    except _VOICE_ERRORS as exc:
        record_degradation(
            "voice_duplex.route",
            exc,
            action="closed the voice socket after a transport error",
        )
        with contextlib.suppress(RuntimeError):
            await ws.close(code=1011, reason="Voice lane error")
    finally:
        if microphone_lease is not None:
            if microphone_lease.session_id:
                _SESSION_MICROPHONE_LEASES.pop(
                    microphone_lease.session_id,
                    None,
                )
            get_microphone_authority().release(
                microphone_lease,
                reason="duplex_transport_closed",
            )
        if reserved_session_id:
            _release_voice_session(reserved_session_id)
