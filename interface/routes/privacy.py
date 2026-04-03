"""interface/routes/privacy.py
──────────────────────────────
Extracted from server.py — Privacy toggles, voice endpoints,
and source download.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.config import config
from core.container import ServiceContainer

from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Privacy")

router = APIRouter()

# ── Voice Engine Accessor ─────────────────────────────────────
# The voice engine factory is set by the main server lifespan.
# This module provides a getter/setter so system.py can also access it.

_voice_engine_fn: Optional[Callable] = None


def set_voice_engine_fn(fn: Optional[Callable]) -> None:
    global _voice_engine_fn
    _voice_engine_fn = fn


def get_voice_engine_fn() -> Optional[Callable]:
    return _voice_engine_fn


# ── Models ────────────────────────────────────────────────────

class PrivacyPayload(BaseModel):
    enabled: bool


# ── Routes ────────────────────────────────────────────────────

@router.post("/privacy/camera")
async def api_privacy_camera(payload: PrivacyPayload, _: None = Depends(_require_internal)):
    """Toggle the visual cortex camera processing."""
    enabled = payload.enabled
    smc = ServiceContainer.get("sensory_motor_cortex", default=None)
    vision_buffer = ServiceContainer.get("continuous_vision", default=None)

    if not smc and not vision_buffer:
        return JSONResponse({"error": "Camera systems unavailable"}, status_code=503)

    if enabled:
        from core.runtime.boot_safety import main_process_camera_policy

        camera_allowed, reason = main_process_camera_policy(True)
        if not camera_allowed:
            if smc is not None:
                smc.camera_enabled = False
            if vision_buffer is not None:
                vision_buffer.camera_enabled = False
            logger.warning("\U0001f512 Privacy: Camera enable denied: %s", reason)
            return JSONResponse(
                {"ok": False, "enabled": False, "reason": reason},
                status_code=409,
            )

    if smc is not None:
        smc.camera_enabled = enabled
    if vision_buffer is not None:
        vision_buffer.camera_enabled = enabled
    logger.info("\U0001f512 Privacy: Camera %s", 'enabled' if enabled else 'disabled')
    return {"ok": True, "enabled": enabled}


@router.post("/privacy/microphone")
async def api_privacy_microphone(payload: PrivacyPayload, _: None = Depends(_require_internal)):
    """Toggle the voice engine microphone processing."""
    enabled = payload.enabled
    voice = _voice_engine_fn() if _voice_engine_fn else None
    if voice:
        voice.microphone_enabled = enabled
        logger.info("\U0001f512 Privacy: Microphone %s", 'enabled' if enabled else 'disabled')
        return {"ok": True, "enabled": enabled}
    return JSONResponse({"error": "VoiceEngine unavailable"}, status_code=503)


@router.post("/voice/chunk")
async def api_voice_chunk(request: Request):
    """Receive raw PCM audio chunk from browser AudioWorklet.
    M-01 FIX: Size limit enforced before reading body."""
    content_length = int(request.headers.get("content-length", 0))
    MAX_VOICE_CHUNK = 512 * 1024  # 512KB max
    if content_length > MAX_VOICE_CHUNK:
        raise HTTPException(status_code=413, detail="Voice chunk too large")
    chunk = await request.body()
    if len(chunk) > MAX_VOICE_CHUNK:
        raise HTTPException(status_code=413, detail="Voice chunk too large")
    voice = _voice_engine_fn() if _voice_engine_fn else None
    if voice and hasattr(voice, "feed_chunk"):
        await voice.feed_chunk(chunk)
    return JSONResponse({"ok": True})


@router.get("/source")
async def api_source_download(
    _: None = Depends(_require_internal),
):
    """Bundle and return the current source code as a download."""
    PROJECT_ROOT = config.paths.project_root
    try:
        from utils.bundler import write_bundle
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            out = Path(tmp.name)
        write_bundle(PROJECT_ROOT, out, lite=True)
        return FileResponse(
            str(out),
            media_type="text/plain",
            filename=f"aura_source_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        )
    except Exception as exc:
        logger.error("Source download failed: %s", exc)
        raise HTTPException(status_code=500, detail="Source bundle generation failed")


@router.get("/stream/voice")
async def voice_sse_stream(request: Request):
    """Server-Sent Events stream for voice pipeline output."""
    async def gen():
        sse_q: asyncio.Queue = asyncio.Queue(maxsize=50)
        voice = _voice_engine_fn() if _voice_engine_fn else None
        if voice and hasattr(voice, "subscribe"):
            await voice.subscribe(sse_q)
        try:
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(sse_q.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"type\":\"ping\"}\n\n"
        finally:
            if voice and hasattr(voice, "unsubscribe"):
                await voice.unsubscribe(sse_q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
