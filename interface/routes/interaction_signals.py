from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.senses.interaction_signals import decode_data_url_image
from interface.auth import _require_internal
from interface.routes.privacy import get_browser_camera_privacy

logger = logging.getLogger("Aura.Server.InteractionSignals")

router = APIRouter()


class TypingSignalPayload(BaseModel):
    timestamp: float
    active: bool = False
    session_ms: float = 0.0
    key_count: int = 0
    correction_count: int = 0
    max_pause_ms: float = 0.0
    pause_before_submit_ms: float = 0.0
    message_chars: int = 0
    submitted: bool = False


class VoiceSignalPayload(BaseModel):
    timestamp: float
    duration_ms: float = 0.0
    speech_ratio: float = 0.0
    rms_avg: float = 0.0
    rms_std: float = 0.0
    peak_avg: float = 0.0
    zcr_avg: float = 0.0
    clipping_ratio: float = 0.0


class VisionSignalPayload(BaseModel):
    timestamp: float
    frame_data_url: str = Field(min_length=16)
    width: Optional[int] = None
    height: Optional[int] = None


def _get_engine():
    engine = ServiceContainer.get("interaction_signals", default=None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Interaction signal engine unavailable")
    return engine


def _microphone_signal_allowed() -> bool:
    voice_engine = ServiceContainer.get("voice_engine", default=None)
    if voice_engine is None:
        return True
    return bool(getattr(voice_engine, "microphone_enabled", True))


def _camera_signal_allowed() -> bool:
    camera_privacy = get_browser_camera_privacy()
    return bool(camera_privacy.get("enabled", False))


@router.post("/signals/typing")
async def api_signal_typing(payload: TypingSignalPayload, _: None = Depends(_require_internal)):
    engine = _get_engine()
    await engine.publish_typing(payload.model_dump())
    return JSONResponse({"ok": True})


@router.post("/signals/voice")
async def api_signal_voice(payload: VoiceSignalPayload, _: None = Depends(_require_internal)):
    if not _microphone_signal_allowed():
        raise HTTPException(status_code=409, detail="Microphone privacy is disabled")
    engine = _get_engine()
    await engine.publish_voice(payload.model_dump())
    return JSONResponse({"ok": True})


@router.post("/signals/vision")
async def api_signal_vision(payload: VisionSignalPayload, _: None = Depends(_require_internal)):
    if not _camera_signal_allowed():
        raise HTTPException(status_code=409, detail="Camera privacy is disabled")
    engine = _get_engine()
    try:
        frame_bytes = decode_data_url_image(payload.frame_data_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc

    if len(frame_bytes) > 512 * 1024:
        raise HTTPException(status_code=413, detail="Vision frame too large")

    metadata: Dict[str, Any] = {
        "timestamp": payload.timestamp,
        "width": payload.width,
        "height": payload.height,
    }
    await engine.publish_vision_frame(frame_bytes, metadata=metadata)
    return JSONResponse({"ok": True})


@router.get("/signals/status")
async def api_signal_status(_: None = Depends(_require_internal)):
    engine = _get_engine()
    return JSONResponse({"ok": True, "signals": engine.get_status()})
