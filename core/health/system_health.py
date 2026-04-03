from fastapi import APIRouter
import psutil
from typing import Dict, Any
from core.container import ServiceContainer

router = APIRouter()

@router.get("/")
@router.get("/report")
async def get_full_health_report() -> Dict[str, Any]:
    """Provide a comprehensive rollup of all registered services and their deep health."""
    return ServiceContainer.get_health_report()

@router.get("/v2")
async def get_health_v2() -> Dict[str, Any]:
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
