from core.runtime.errors import record_degradation
import json
import logging
import os
import time
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse

from core.runtime.service_registry import get_runtime_service

router = APIRouter()
logger = logging.getLogger("Aura.UI.Memory")
_MEMORY_UI_RECOVERABLE_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_memory_ui_degradation(reason: str, exc: BaseException, degradations: list[str]) -> None:
    degradations.append(reason)
    record_degradation('memory_ui', exc)
    logger.warning("%s: %s", reason, exc)

# 1. API endpoint to return vault stats
@router.get("/api/memory")
async def get_vault_stats():
    degradations: list[str] = []

    # Use DI to get BlackHoleVault via MemoryFacade
    facade = get_runtime_service("memory_facade", default=None)
    if facade and hasattr(facade, "setup"):
        try:
            facade.setup()
        except _MEMORY_UI_RECOVERABLE_ERRORS as e:
            _record_memory_ui_degradation("Memory facade setup failed", e, degradations)

    vault = getattr(facade, "vector", None) or getattr(facade, "vault", None)
    if not facade or not vault:
        return {"status": "offline", "error": "BlackHoleVault not found"}

    try:
        memories = list(getattr(vault, "memories", []) or [])
    except _MEMORY_UI_RECOVERABLE_ERRORS as e:
        _record_memory_ui_degradation("Memory vault read failed", e, degradations)
        memories = []
    
    # 1. Mass/Density (Bekenstein)
    try:
        total_bytes = len(json.dumps(memories, default=str).encode()) if memories else 0
    except (OverflowError, TypeError, ValueError) as e:
        _record_memory_ui_degradation("Failed to calculate Vault mass", e, degradations)
        total_bytes = 0
    total_bits = total_bytes * 8
    # Constant radius of 10cm for now
    radius = 10.0
    # Bekenstein limit: I = 2pi * E * R / (hbar * c * ln2)
    # Mapping bytes to a fictional energy/radius ratio
    try:
        density = (total_bits / 1000000.0) / radius if radius > 0 else 0
    except ZeroDivisionError:
        density = 0
    
    # 2. Hawking Decay Analysis
    active_mems = [m for m in memories if m.get("access_count", 0) > 0]
    evaporating = len(memories) - len(active_mems)
    
    # 3. Horcrux Health
    horcrux_status = "STABLE"
    shard_count = 0
    if hasattr(vault, "horcrux"):
        try:
            shards = await vault.horcrux.check_shards()
            shard_count = sum(1 for s in shards.values() if s)
            if shard_count < 3:
                horcrux_status = "CRITICAL"
            elif shard_count < 5:
                horcrux_status = "UNSTABLE"
        except _MEMORY_UI_RECOVERABLE_ERRORS as e:
            _record_memory_ui_degradation("Horcrux shard check failed", e, degradations)
            horcrux_status = "UNKNOWN"

    # 4. Recent Spaghettified Strings (with Episodic Fallback)
    recent = []
    source_memories = memories
    
    # Fallback to Episodic if Vault is empty (Common during first launch/sync)
    is_fallback = False
    if not source_memories:
        try:
            recent_episodes = await facade.episodic.recall_recent_async(limit=10)
            if recent_episodes:
                is_fallback = True
                source_memories = [
                    {
                        "text": f"{e.context} -> {e.action}: {e.outcome}",
                        "created": e.timestamp * 1000,
                        "access_count": 1 if e.importance > 0.5 else 0
                    } for e in recent_episodes
                ]
        except _MEMORY_UI_RECOVERABLE_ERRORS as e:
            _record_memory_ui_degradation("Episodic fallback failed", e, degradations)

    for m in sorted(source_memories, key=lambda x: x.get("created", 0), reverse=True)[:10]:
        text = m.get("text", "")
        # Visual spaghettification: truncate and add "..."
        display_text = (text[:120] + "...") if len(text) > 120 else text
        
        try:
            created_ts = float(m.get("created", 0))
            age_sec = int(time.time() - created_ts / 1000.0)
        # not a failure: a value that is not a number is not one this can read.
        except (ValueError, TypeError):
            age_sec = 0
            
        recent.append({
            "text": display_text,
            "gravity": m.get("access_count", 0),
            "created": m.get("created", 0),
            "age_seconds": age_sec
        })
    
    return {
        "status": "degraded" if degradations else "online",
        "degraded": bool(degradations),
        "degradation_reasons": degradations,
        "horcrux": horcrux_status,
        "shards_active": shard_count,
        "total_nodes": len(memories) if memories else len(source_memories),
        "is_fallback": is_fallback,
        "total_mass_kb": round(total_bytes / 1024, 2),
        "bit_density": round(density, 4),
        "radius_cm": radius,
        "evaporation_count": evaporating,
        "recent_events": recent,
        # No "entropy_fidelity" here. It was a hardcoded 0.998 carrying the
        # comment "Fictional metric for UI", and the memory panel rendered it
        # as though the vault had measured it. Everything else in this payload
        # is derived from real vault state; an invented number sitting beside
        # them borrows their credibility. If entropy fidelity is worth showing,
        # it has to be computed.
    }

# 2a. Root route — served when btn-mem-map loads /memory
@router.get("")
@router.get("/")
async def serve_memory_root():
    from core.config import config
    # The source Vite entry is not a valid packaged desktop fallback: it points
    # at /src/*.jsx and can render as a black window outside a Vite dev server.
    # Keep /memory on the dependency-free panel unless the operator explicitly
    # opts into the development UI.
    static_path = config.paths.project_root / "interface" / "static" / "memory_panel.html"
    if static_path.exists():
        return FileResponse(str(static_path))
    if str(os.getenv("AURA_MEMORY_DEV_UI", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        dist_path = config.paths.project_root / "interface" / "static" / "memory" / "dist" / "index.html"
        if dist_path.exists():
            return FileResponse(str(dist_path))
        react_path = config.paths.project_root / "interface" / "static" / "memory" / "index.html"
        if react_path.exists():
            return FileResponse(str(react_path))
    return HTMLResponse(
        "<html><body style='background:#05030a;color:#8a2be2;font-family:monospace;padding:2rem'>"
        "<h2>Memory Nexus — Offline</h2><p>Static assets not found.</p></body></html>",
        status_code=200
    )

# 2b. /dash — explicit React app entry
@router.get("/dash")
async def serve_memory_ui():
    from core.config import config
    dist_path = config.paths.project_root / "interface" / "static" / "memory" / "dist" / "index.html"
    if dist_path.exists():
        return FileResponse(str(dist_path))
    ui_path = config.paths.project_root / "interface" / "static" / "memory" / "index.html"
    if not ui_path.exists():
        return HTMLResponse("<html><body><h1>Singularity Error</h1><p>Event horizon static assets not found.</p></body></html>", status_code=404)
    return FileResponse(str(ui_path))
