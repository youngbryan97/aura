"""interface/event_bridge.py
────────────────────────────
Extracted from server.py — EventBus → WebSocket bridge,
telemetry broadcasting, and mycelial UI callback.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.Server.EventBridge")


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


async def run_event_bridge(is_gui_proxy: bool = False) -> None:
    """Bridge EventBus events to the WebSocket broadcast bus.

    This is the core pipeline that forwards orchestrator/cognitive events
    to the frontend HUD in real-time.
    """
    from interface.websocket_manager import broadcast_bus, ws_manager

    try:
        from core.event_bus import get_event_bus
        from core.schemas import (
            TelemetryPayload,
            CognitiveThoughtPayload,
            WebsocketMessage,
            ChatStreamChunkPayload,
            ChatThoughtChunkPayload,
            AuraMessagePayload,
            ActionResultPayload,
        )

        bus = get_event_bus()
        q = await bus.subscribe("*")
        logger.info(
            "📡 EventBus → WebSocket bridge (Pydantic Zenith) ACTIVE (Bus ID: %s)",
            bus._bus_id,
        )

        # Initialize psutil for accurate non-blocking percent calculation
        import psutil
        psutil.cpu_percent(interval=None)

        if bus._use_redis and not bus._redis:
            logger.warning(
                "EventBus Redis connection missing – HUD may be limited to local events."
            )

        def _filter_msg(text):
            if is_gui_proxy:
                return text
            if not text or not isinstance(text, str):
                return text
            try:
                from core.brain.personality_engine import get_personality_engine
                return get_personality_engine().filter_response(text)
            except Exception as e:
                logger.debug("Personality filtering failed: %s", e)
                return text

        while True:
            _priority, _seq, event = await q.get()
            topic = event.get("topic")
            data = event.get("data")

            # With no UI consumers attached, the bridge should stay dormant instead of
            # serializing every internal event into a websocket shape that nobody will read.
            if ws_manager.count() == 0 and broadcast_bus.subscriber_count() <= 1:
                q.task_done()
                continue

            # Apply personality filtering to any broadcasted text
            if isinstance(data, dict):
                for key in ["content", "message", "text", "thought", "chunk"]:
                    if key in data:
                        data[key] = _filter_msg(data[key])
            else:
                data = {"content": _filter_msg(str(data))}

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
                except asyncio.TimeoutError:
                    logger.warning(
                        "EventBridge: dropped %s event (broadcast bus timeout)",
                        ws_msg.get("type", "unknown"),
                    )
            q.task_done()

    except Exception as e:
        logger.error("EventBus bridge failure: %s", e, exc_info=True)


def _map_event_to_ws_message(
    topic: str,
    data: Dict[str, Any],
    **schema_classes,
) -> Optional[Dict[str, Any]]:
    """Convert an EventBus event into a WebSocket-deliverable message dict."""
    CognitiveThoughtPayload = schema_classes["CognitiveThoughtPayload"]
    WebsocketMessage = schema_classes["WebsocketMessage"]
    ChatStreamChunkPayload = schema_classes["ChatStreamChunkPayload"]
    ChatThoughtChunkPayload = schema_classes["ChatThoughtChunkPayload"]
    AuraMessagePayload = schema_classes["AuraMessagePayload"]
    ActionResultPayload = schema_classes["ActionResultPayload"]

    def _model_dict(instance):
        return getattr(instance, "model_dump", getattr(instance, "dict"))()

    if topic in ("thoughts", "neural_event", "cognition"):
        return CognitiveThoughtPayload(
            type="thought",
            content=data.get("content", data.get("message", "...")),
            urgency=data.get("urgency", "NORMAL"),
            cognitive_phase=data.get("phase"),
        ).dict()

    if topic == "telemetry":
        msg_type = data.get("type", "telemetry")
        if msg_type == "telemetry":
            from core.utils.telemetry_enrichment import enrich_telemetry
            enrich_telemetry(data)
            return data
        elif msg_type == "chat_stream_chunk":
            return _model_dict(ChatStreamChunkPayload(**data))
        elif msg_type == "chat_thought_chunk":
            return _model_dict(ChatThoughtChunkPayload(**data))
        elif msg_type in ("aura_message", "chat_response"):
            safe_data = data.copy() if isinstance(data, dict) else {"message": str(data)}
            if "content" in safe_data and "message" not in safe_data:
                safe_data["message"] = safe_data.pop("content")
            return _model_dict(AuraMessagePayload(**safe_data))
        elif msg_type == "action_result":
            return _model_dict(ActionResultPayload(**data))
        else:
            safe_data = data.copy() if isinstance(data, dict) else {"content": str(data)}
            safe_data.pop("type", None)
            return _model_dict(WebsocketMessage(type=msg_type, **safe_data))

    # Topic-level schema mapping
    if topic == "chat_stream_chunk":
        return _model_dict(ChatStreamChunkPayload(**data))
    elif topic in ("aura_message", "chat_response"):
        safe_data = data.copy() if isinstance(data, dict) else {"message": str(data)}
        if "content" in safe_data and "message" not in safe_data:
            safe_data["message"] = safe_data.pop("content")
        return _model_dict(AuraMessagePayload(**safe_data))
    else:
        safe_data = data.copy() if isinstance(data, dict) else {"content": str(data)}
        safe_data.pop("type", None)
        return _model_dict(WebsocketMessage(type=topic, **safe_data))
