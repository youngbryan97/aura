from __future__ import annotations

import binascii
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.senses.interaction_signals import decode_data_url_image
from interface.auth import _require_internal
from interface.routes.privacy import get_browser_camera_privacy

logger = logging.getLogger("Aura.Server.InteractionSignals")

router = APIRouter()
_VISION_PAYLOAD_ERRORS = (binascii.Error, TypeError, ValueError)


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
    engine = get_runtime_service("interaction_signals", default=None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Interaction signal engine unavailable")
    return engine


def _microphone_signal_allowed() -> bool:
    voice_engine = get_runtime_service("voice_engine", default=None)
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
    except _VISION_PAYLOAD_ERRORS as exc:
        record_degradation('interaction_signals', exc)
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


class CameraCapturePayload(BaseModel):
    request_id: str
    frame_data_url: str = ""
    width: int = 0
    height: int = 0
    #: When the window could not capture: the browser's error name ("NotAllowedError") and its message.
    error_name: str = ""
    error: str = ""


@router.post("/signals/camera_capture")
async def api_camera_capture(
    payload: CameraCapturePayload, _: None = Depends(_require_internal)
):
    """A frame she asked for, on its way back to the turn that is waiting.

    Distinct from `/signals/vision`, which is the presence lane: that one
    samples a thumbnail every few seconds to know whether somebody is there,
    and it is deliberately too small and too stale to answer a question about
    right now. This is the reply to a specific request, correlated by id, and
    the turn holding that id is blocked on it.
    """
    from core.senses.sight import CaptureRefusal, decode_frame, get_capture_broker

    broker = get_capture_broker()
    if payload.error_name:
        # Why the window has no frame. It carries no image, so it needs no camera permission; the pending request id
        # is what makes it this turn's answer, and one that is not pending is dropped.
        refusal = CaptureRefusal(name=str(payload.error_name)[:64], message=str(payload.error or "")[:300])
        return JSONResponse({"ok": True, "delivered": await broker.deliver(payload.request_id, refusal)})
    one_shot_authorized = await broker.one_shot_authorized(payload.request_id)
    if not _camera_signal_allowed() and not one_shot_authorized:
        raise HTTPException(status_code=409, detail="Camera privacy is disabled")

    frame = decode_frame(payload.frame_data_url)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or oversized frame")
    frame.width = int(payload.width or 0)
    frame.height = int(payload.height or 0)

    delivered = await broker.deliver(payload.request_id, frame)
    # A late frame is not an error worth raising — the turn that wanted it has
    # already given up and said so — but the client is told it was too late so
    # it can stop retrying.
    return JSONResponse({"ok": True, "delivered": delivered})


@router.get("/signals/status")
async def api_signal_status(_: None = Depends(_require_internal)):
    engine = _get_engine()
    return JSONResponse({"ok": True, "signals": engine.get_status()})
