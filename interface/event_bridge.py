"""interface/event_bridge.py
────────────────────────────
Extracted from server.py — EventBus → WebSocket bridge,
telemetry broadcasting, and mycelial UI callback.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested

logger = logging.getLogger("Aura.Server.EventBridge")

_EVENT_BRIDGE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    TypeError,
    ValueError,
)


#: Message types that land in the chat window as Aura speaking.
_SPOKEN_WS_TYPES = frozenset({"aura_message", "chat_response"})


def _suppress_internal_leak(ws_msg: dict[str, Any]) -> bool:
    """Drop an unsolicited message that is internal machinery, not speech.

    The chat route runs the reliability gate against the user's question.
    Everything that arrives unprompted — initiative, action results,
    autonomous speech — reaches the same window through this bridge, and had
    no gate at all. Live 2026-07-26 the chat window rendered, verbatim:

        ROUTER_ERROR: unknown (at all_failed)

    which is a diagnostic label meant for string consumers inside the runtime.
    This is the one seam every such publisher passes through, so the check
    belongs here rather than in each of them.
    """
    try:
        if str(ws_msg.get("type", "")) not in _SPOKEN_WS_TYPES:
            return False
        body = str(ws_msg.get("message") or ws_msg.get("content") or "")
        if not body.strip():
            return False
        from core.conversation.surface_delivery import route_answer_supersedes

        metadata = ws_msg.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        conversation_id = str(
            ws_msg.get("conversation_id") or metadata.get("conversation_id") or ""
        ).strip()
        turn_id = str(
            ws_msg.get("conversation_turn_id")
            or metadata.get("conversation_turn_id")
            or ""
        ).strip()

        # "chat_response" is the route answering the person; "aura_message" is
        # her speaking on her own account. Only the latter has to wait.
        if route_answer_supersedes(
            body,
            conversation_id=conversation_id,
            turn_id=turn_id,
            unprompted=str(ws_msg.get("type", "")) != "chat_response",
            # A commentary the person asked for is not competing to answer
            # their question. It is what is happening while the answer is
            # being worked out, and the turn is open for the whole of it.
            answering=not bool(metadata.get("narration")),
        ):
            logger.warning(
                "EventBridge: withheld a second answer to a turn the route "
                "already answered: %r",
                body[:160],
            )
            return True

        from core.conversation.response_reliability import internal_leak_reasons

        reasons = internal_leak_reasons(body)
        if not reasons:
            return False
        record_degradation(
            "event_bridge",
            RuntimeError("internal_text_suppressed:" + ",".join(reasons)),
            severity="warning",
            action="withheld an unsolicited message that was internal machinery, not speech",
        )
        logger.warning(
            "EventBridge: withheld %s carrying internal text (%s): %r",
            ws_msg.get("type", "unknown"),
            ",".join(reasons),
            body[:160],
        )
        return True
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        # This is user-visible autonomous egress. Treating an unavailable
        # integrity verdict as clean can publish the exact internal control
        # text this seam exists to contain. Quarantine this one message; the
        # next independently generated message gets a fresh bounded check.
        record_degradation(
            "event_bridge.integrity_gate",
            exc,
            severity="warning",
            action="quarantined one autonomous spoken message until integrity checking recovered",
            enforce_failure_policy=False,
        )
        logger.warning("EventBridge integrity gate unavailable; message quarantined: %s", exc)
        return str(ws_msg.get("type", "")) in _SPOKEN_WS_TYPES


def _complete_spoken_tail(ws_msg: dict[str, Any]) -> None:
    """Finish a sentence that a token budget cut off, in place.

    The chat route repairs a mid-clause cutoff before serving. The kernel's
    own publish path reaches the same window without passing through it, so a
    reply could arrive here whole in substance and broken in its last three
    characters. Live 2026-07-27 a four-part deploy-risk analysis — correct,
    ordered, genuinely useful — ended on the word "And".

    Same argument as the leak check above: this is the seam every publisher
    passes through, so the repair belongs here rather than in each of them.
    """
    try:
        if str(ws_msg.get("type", "")) not in _SPOKEN_WS_TYPES:
            return
        key = "message" if ws_msg.get("message") is not None else "content"
        body = str(ws_msg.get(key) or "")
        if not body.strip():
            return
        from core.conversation.response_reliability import complete_truncated_tail

        repaired = complete_truncated_tail(body)
        if repaired and repaired != body:
            logger.info(
                "EventBridge: completed a reply cut off mid-clause (%d -> %d chars).",
                len(body),
                len(repaired),
            )
            ws_msg[key] = repaired
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        # Fail open: a broken repair must not silence Aura.
        logger.debug("EventBridge tail completion skipped: %s", exc)


async def mycelial_ui_callback(message: str):
    """Direct, unblockable UI delivery via Mycelial Network.
    Bypasses EventBus/Queue infrastructure for emergency status.
    """
    if not message:
        return

    from interface.websocket_manager import ws_manager

    if ws_manager.count() == 0:
        return

    payload = {
        "type": "aura_message",
        "message": message,
        "timestamp": time.time(),
        "origin": "mycelial_failsafe",
    }
    logger.info("🍄 [MYCELIUM] ⚡ Direct Broadcast: %s", message[:50])
    await ws_manager.broadcast(payload)


async def broadcast_telemetry(data: dict):
    """Direct telemetry broadcasting.
    Bypasses the EventBus for sub-100ms gauge updates.
    """
    if not isinstance(data, dict):
        return

    from core.utils.telemetry_enrichment import enrich_telemetry
    from interface.websocket_manager import ws_manager

    if ws_manager.count() == 0:
        return

    enrich_telemetry(data)
    await ws_manager.broadcast(data)


def _shape_user_facing_ws_message(
    ws_msg: dict[str, Any],
    *,
    is_gui_proxy: bool = False,
) -> dict[str, Any]:
    """Shape complete spoken replies without mutating internal event payloads."""

    if (
        is_gui_proxy
        or not isinstance(ws_msg, dict)
        or str(ws_msg.get("type") or "") not in _SPOKEN_WS_TYPES
    ):
        return ws_msg
    shaped = dict(ws_msg)
    try:
        from core.brain.personality_engine import get_personality_engine

        personality = get_personality_engine()
        for key in ("message", "content", "text"):
            value = shaped.get(key)
            if isinstance(value, str) and value:
                shaped[key] = personality.filter_response(value, user_facing=True)
    except _EVENT_BRIDGE_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "event_bridge",
            exc,
            severity="warning",
            action="broadcast the unshaped spoken reply after personality filtering failed",
        )
        logger.debug("Personality filtering failed: %s", exc)
        return ws_msg
    return shaped


async def run_event_bridge(is_gui_proxy: bool = False) -> None:
    """Bridge EventBus events to the WebSocket broadcast bus.

    This is the core pipeline that forwards orchestrator/cognitive events
    to the frontend HUD in real-time.
    """
    from interface.websocket_manager import broadcast_bus, ws_manager

    bus = None
    q = None
    try:
        from core.event_bus import get_event_bus
        from core.schemas import (
            ActionResultPayload,
            AuraMessagePayload,
            ChatStreamChunkPayload,
            ChatThoughtChunkPayload,
            CognitiveThoughtPayload,
            WebsocketMessage,
        )

        bus = get_event_bus()
        q = await bus.subscribe("*")
        logger.info(
            "📡 EventBus → WebSocket bridge (Pydantic Zenith) ACTIVE (Bus ID: %s)",
            bus._bus_id,
        )

        # Prime the canonical observer's non-blocking CPU sample.
        from core.runtime import resource_psutil as psutil
        psutil.cpu_percent(interval=None)

        if bus._use_redis and not bus._redis:
            logger.warning(
                "EventBus Redis connection missing – HUD may be limited to local events."
            )

        while not is_shutdown_requested():
            _priority, _seq, event = await q.get()
            try:
                topic = event.get("topic")
                data = event.get("data")

                # With no UI consumers attached, the bridge should stay dormant instead of
                # serializing every internal event into a websocket shape that nobody will read.
                if ws_manager.count() == 0 and broadcast_bus.subscriber_count() <= 1:
                    continue

                if isinstance(data, dict):
                    data = dict(data)
                else:
                    data = {"content": str(data)}

                ws_msg = _map_event_to_ws_message(
                    topic, data,
                    CognitiveThoughtPayload=CognitiveThoughtPayload,
                    WebsocketMessage=WebsocketMessage,
                    ChatStreamChunkPayload=ChatStreamChunkPayload,
                    ChatThoughtChunkPayload=ChatThoughtChunkPayload,
                    AuraMessagePayload=AuraMessagePayload,
                    ActionResultPayload=ActionResultPayload,
                )

                if ws_msg is not None:
                    _complete_spoken_tail(ws_msg)
                if ws_msg is not None and _suppress_internal_leak(ws_msg):
                    ws_msg = None
                if ws_msg is not None:
                    ws_msg = _shape_user_facing_ws_message(
                        ws_msg,
                        is_gui_proxy=is_gui_proxy,
                    )
                    p_val = 10
                    msg_type = ws_msg.get("type", "")
                    if msg_type in ("aura_message", "chat_response", "chat_stream_chunk"):
                        p_val = 0
                    elif msg_type in ("thought", "neural_event", "log", "telemetry"):
                        p_val = 20

                    try:
                        await asyncio.wait_for(
                            broadcast_bus.publish(ws_msg, priority=p_val), timeout=2.0
                        )
                    except TimeoutError:
                        logger.warning(
                            "EventBridge: dropped %s event (broadcast bus timeout)",
                            ws_msg.get("type", "unknown"),
                        )
            except asyncio.CancelledError:
                raise
            except _EVENT_BRIDGE_RECOVERABLE_ERRORS as e:
                record_degradation('event_bridge', e)
                logger.warning(
                    "EventBridge: dropped malformed %s event: %s",
                    event.get("topic", "unknown") if isinstance(event, dict) else "unknown",
                    e,
                )
            finally:
                try:
                    q.task_done()
                except ValueError as e:
                    record_degradation('event_bridge', e)
                    logger.warning("EventBridge queue task accounting failed: %s", e)

    except asyncio.CancelledError:
        logger.info("EventBus bridge cancelled")
        raise
    except _EVENT_BRIDGE_RECOVERABLE_ERRORS as e:
        record_degradation('event_bridge', e)
        logger.error("EventBus bridge failure: %s", e, exc_info=True)
    finally:
        if bus is not None and q is not None:
            try:
                await bus.unsubscribe("*", q)
            except _EVENT_BRIDGE_RECOVERABLE_ERRORS as e:
                record_degradation('event_bridge', e)
                logger.warning("EventBridge unsubscribe failed during shutdown: %s", e)


def _map_event_to_ws_message(
    topic: str,
    data: dict[str, Any],
    **schema_classes,
) -> dict[str, Any] | None:
    """Convert an EventBus event into a WebSocket-deliverable message dict."""
    cognitive_thought_payload_cls = schema_classes["CognitiveThoughtPayload"]
    websocket_message_cls = schema_classes["WebsocketMessage"]
    chat_stream_chunk_payload_cls = schema_classes["ChatStreamChunkPayload"]
    chat_thought_chunk_payload_cls = schema_classes["ChatThoughtChunkPayload"]
    aura_message_payload_cls = schema_classes["AuraMessagePayload"]
    action_result_payload_cls = schema_classes["ActionResultPayload"]

    def _model_dict(instance):
        model_dump = getattr(instance, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        legacy_dict = getattr(instance, "dict", None)
        if callable(legacy_dict):
            return legacy_dict()
        raise TypeError(f"{type(instance).__name__} does not expose a model dump method")

    if topic in ("thoughts", "neural_event", "cognition"):
        extra = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "content",
                "message",
                "urgency",
                "phase",
                "type",
            }
        }
        return _model_dict(cognitive_thought_payload_cls(
            type="thought",
            content=data.get("content", data.get("message", "...")),
            urgency=data.get("urgency", "NORMAL"),
            cognitive_phase=data.get("phase"),
            **extra,
        ))

    if topic == "telemetry":
        msg_type = data.get("type", "telemetry")
        if msg_type == "telemetry":
            from core.utils.telemetry_enrichment import enrich_telemetry
            enrich_telemetry(data)
            return data
        elif msg_type == "chat_stream_chunk":
            return _model_dict(chat_stream_chunk_payload_cls(**data))
        elif msg_type == "chat_thought_chunk":
            return _model_dict(chat_thought_chunk_payload_cls(**data))
        elif msg_type in ("aura_message", "chat_response"):
            safe_data = data.copy() if isinstance(data, dict) else {"message": str(data)}
            if "content" in safe_data and "message" not in safe_data:
                safe_data["message"] = safe_data.pop("content")
            return _model_dict(aura_message_payload_cls(**safe_data))
        elif msg_type == "action_result":
            return _model_dict(action_result_payload_cls(**data))
        else:
            safe_data = data.copy() if isinstance(data, dict) else {"content": str(data)}
            safe_data.pop("type", None)
            return _model_dict(websocket_message_cls(type=msg_type, **safe_data))

    # Topic-level schema mapping
    if topic == "chat_stream_chunk":
        return _model_dict(chat_stream_chunk_payload_cls(**data))
    elif topic in ("aura_message", "chat_response"):
        safe_data = data.copy() if isinstance(data, dict) else {"message": str(data)}
        if "content" in safe_data and "message" not in safe_data:
            safe_data["message"] = safe_data.pop("content")
        return _model_dict(aura_message_payload_cls(**safe_data))
    else:
        safe_data = data.copy() if isinstance(data, dict) else {"content": str(data)}
        safe_data.pop("type", None)
        return _model_dict(websocket_message_cls(type=topic, **safe_data))
