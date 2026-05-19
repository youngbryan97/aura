from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.container import ServiceContainer
from core.runtime.health_contract import evaluate_health

router = APIRouter()


@router.get("/")
@router.get("/report")
async def get_full_health_report() -> dict[str, Any]:
    """Provide a comprehensive rollup of all registered services and their deep health."""
    return ServiceContainer.get_health_report()


@router.get("/runtime")
@router.get("/contract")
async def get_runtime_health_contract() -> JSONResponse:
    """Canonical runtime contract: what must be alive for Aura to be healthy."""
    verdict = evaluate_health()
    return JSONResponse(verdict.to_report(), status_code=verdict.status_code)


@router.get("/v2")
async def get_health_v2() -> dict[str, Any]:
    """Extended system health endpoints via the [ZENITH] Tricorder."""
    tricorder = ServiceContainer.get("tricorder", default=None)
    if not tricorder:
        return {"status": "error", "message": "Tricorder organ not found."}

    # Trigger a real-time scan
    from core.state.aura_state import get_current_state

    state = get_current_state()
    report = await tricorder.scan(state)

    # Add legacy metadata for compatibility
    report["legacy_status"] = "ok" if tricorder.healthy else "degraded"
    return report
