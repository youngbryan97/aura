from __future__ import annotations
"""interface/routes/multimodal.py — SSE for the multimodal coordinator."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from core.multimodal.coordinator import get_coordinator
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Multimodal")

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


@router.get("/stream")
async def stream(turn_id: str = Query(...), _: None = Depends(_require_internal)) -> StreamingResponse:
    coordinator = get_coordinator()

    async def gen():
        try:
            async for ev in coordinator.subscribe(turn_id):
                payload = {"kind": ev.kind, "offset_ms": ev.offset_ms, "seq": ev.seq, "payload": ev.payload}
                yield f"data: {json.dumps(payload, default=str)}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(gen(), media_type="text/event-stream")


__all__ = ["router"]
