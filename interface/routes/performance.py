"""interface/routes/performance.py

Endpoint that the frontend posts frame samples + ack samples to. The
PerformanceGuard owns the throttle decision; this route is a thin
collector. Returns the current motion-throttle state so the UI can
flip the ``aura-throttle-motion`` class on its own ``<body>`` without
waiting for a websocket round-trip.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from interface.auth import _require_internal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/performance", tags=["performance"])
_PERFORMANCE_ROUTE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _mark_runtime_service_progress(source: str) -> None:
    try:
        from core.resilience.stall_watchdog import mark_runtime_service_progress

        mark_runtime_service_progress(source)
    except _PERFORMANCE_ROUTE_ERRORS as exc:
        logger.debug(
            "%s unavailable (%s: %s); the reading is missing from the record",
            "mark_runtime_service_progress",
            type(exc).__name__,
            exc,
        )


@router.post("/frame")
async def frame(
    payload: Dict[str, Any] = Body(...),
    _: None = Depends(_require_internal),
) -> JSONResponse:
    _mark_runtime_service_progress("api.performance.frame")
    try:
        from core.runtime.performance_guard import get_guard
        guard = get_guard()
        duration_ms = float(payload.get("duration_ms", 0.0))
        source = str(payload.get("source", "ui"))
        guard.record_frame(duration_ms, source=source)
        report = guard.report()
        return JSONResponse({"ok": True, "throttled": report.get("motion_throttled", False)})
    except _PERFORMANCE_ROUTE_ERRORS as exc:
        record_degradation('performance', exc)
        return JSONResponse({"ok": False, "error": str(exc)})


@router.post("/ack")
async def ack(
    payload: Dict[str, Any] = Body(...),
    _: None = Depends(_require_internal),
) -> JSONResponse:
    _mark_runtime_service_progress("api.performance.ack")
    try:
        from core.runtime.performance_guard import get_guard
        guard = get_guard()
        guard.record_ack(str(payload.get("request_id", "")), float(payload.get("latency_ms", 0.0)))
        return JSONResponse({"ok": True})
    except _PERFORMANCE_ROUTE_ERRORS as exc:
        record_degradation('performance', exc)
        return JSONResponse({"ok": False, "error": str(exc)})


@router.get("/report")
async def report(_: None = Depends(_require_internal)) -> JSONResponse:
    _mark_runtime_service_progress("api.performance.report")
    try:
        from core.runtime.performance_guard import get_guard
        return JSONResponse(get_guard().report())
    except _PERFORMANCE_ROUTE_ERRORS as exc:
        record_degradation('performance', exc)
        return JSONResponse({"error": str(exc)})


__all__ = ["router"]
