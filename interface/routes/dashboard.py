"""interface/routes/dashboard.py
─────────────────────────────────
Live evidence dashboard. The single endpoint a skeptic uses to inspect
Aura's inner life *without trusting her words*. Every value is read
directly from the substrate, not from generated text.

Endpoints:

  GET /dashboard/snapshot     — full live state (system tab)
  GET /dashboard/receipts     — recent action receipts (drive→outcome)
  GET /dashboard/projects     — self-originated projects ledger view
  GET /dashboard/tokens       — active capability tokens (count + scopes)
  GET /dashboard/relationships — relationship dossiers (anonymized fields)
  GET /dashboard/integration   — phi / GWT / HOT / qualia
  GET /dashboard/viability    — viability state machine + transitions
  GET /dashboard/world        — world-bridge channel permissions
  GET /dashboard/conscience   — recent conscience violations (read-only)
  GET /trace/{receipt_id}     — full causal trace for one action

The dashboard is read-only. Every response is built from authoritative
service registries; it never paraphrases nor synthesizes.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import JSONResponse

from core.container import ServiceContainer
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Dashboard")

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
trace_router = APIRouter(prefix="/trace", tags=["trace"])


# ─── helpers ────────────────────────────────────────────────────────────────


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:
        record_degradation('dashboard', exc)
        logger.debug("dashboard helper failed: %s", exc)
        return default


def _percentile_summary(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0}
    s = sorted(samples)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(0.95 * n))],
        "max": s[-1],
    }


# ─── snapshot ──────────────────────────────────────────────────────────────


@router.get("/snapshot")
async def snapshot(_: None = Depends(_require_internal)) -> JSONResponse:
    payload: Dict[str, Any] = {"when": time.time()}

    # Self snapshot
    payload["self"] = _safe(lambda: __import__("core.identity.self_object", fromlist=["get_self"]).get_self().snapshot().as_dict()) or {}

    # Viability
    try:
        from core.organism.viability import get_viability
        payload["viability"] = get_viability().report()
    except Exception:
        payload["viability"] = {}

    # Recent receipts
    try:
        from core.agency.agency_orchestrator import get_receipt_log
        payload["recent_receipts"] = get_receipt_log().recent(limit=20)
    except Exception:
        payload["recent_receipts"] = []

    # Active projects
    try:
        from core.agency.projects import get_ledger
        payload["projects"] = [p.to_dict() for p in get_ledger().active()]
    except Exception:
        payload["projects"] = []

    # Capability tokens
    try:
        from core.agency.capability_token import get_token_store
        store = get_token_store()
        active = []
        for t in store._tokens.values():  # type: ignore[attr-defined]
            if not t.is_consumed() and not t.revoked and not t.is_expired():
                active.append({
                    "token": t.token[:12] + "…",
                    "origin": t.origin,
                    "domain": t.domain,
                    "scope": t.scope,
                    "ttl_remaining_s": max(0.0, t.issued_at + t.ttl_seconds - time.time()),
                })
        payload["capability_tokens"] = active
    except Exception:
        payload["capability_tokens"] = []

    # Phi / GWT / HOT / qualia
    payload["integration"] = _safe(_collect_integration) or {}

    # System / runtime
    try:
        import psutil
        payload["system"] = {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_pct": psutil.virtual_memory().percent,
            "disk_pct": _safe(lambda: psutil.disk_usage("/").percent, default=0.0),
            "uptime_s": time.time() - psutil.boot_time(),
        }
    except Exception:
        payload["system"] = {}

    # Stability + lane
    try:
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian is not None and hasattr(guardian, "last_report"):
            r = guardian.last_report
            payload["stability"] = {
                "overall_healthy": getattr(r, "overall_healthy", None),
                "checks": [
                    {"name": getattr(c, "name", "?"), "healthy": getattr(c, "healthy", None), "message": getattr(c, "message", "")[:240]}
                    for c in (getattr(r, "checks", []) or [])
                ],
            }
    except Exception:
        payload["stability"] = {}

    # Conversation lane
    try:
        from interface.routes.chat import _collect_conversation_lane_status
        payload["conversation_lane"] = _collect_conversation_lane_status()
    except Exception:
        payload["conversation_lane"] = {}

    return JSONResponse(payload)


def _collect_integration() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        phi = ServiceContainer.get("phi_core", default=None)
        if phi is not None:
            for attr in ("phi_s", "is_complex", "structure_entropy", "history_len"):
                v = getattr(phi, attr, None)
                if v is None:
                    continue
                out[attr] = v
    except Exception:
        pass  # no-op: intentional
    try:
        hp = ServiceContainer.get("hierarchical_phi", default=None)
        if hp is not None and hasattr(hp, "last_result"):
            r = hp.last_result
            if r is not None:
                out["hierarchical_max_complex"] = getattr(r, "max_complex_name", None)
                out["hierarchical_max_phi"] = getattr(r, "max_phi", None)
    except Exception:
        pass  # no-op: intentional
    try:
        gw = ServiceContainer.get("global_workspace", default=None)
        if gw is not None and hasattr(gw, "last_winner"):
            w = gw.last_winner
            out["gw_winner_source"] = getattr(w, "source", None)
            out["gw_winner_priority"] = getattr(w, "priority", None)
    except Exception:
        pass  # no-op: intentional
    return out


# ─── slim endpoints (each is its own snapshot for the dashboard tabs) ─────


@router.get("/receipts")
async def receipts(limit: int = 100, _: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from core.agency.agency_orchestrator import get_receipt_log
        return JSONResponse({"receipts": get_receipt_log().recent(limit=int(limit))})
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc), "receipts": []})


@router.get("/projects")
async def projects(_: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from core.agency.projects import get_ledger
        return JSONResponse({"projects": [p.to_dict() for p in get_ledger().all()]})
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc), "projects": []})


@router.get("/tokens")
async def tokens(_: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from core.agency.capability_token import get_token_store
        store = get_token_store()
        all_tokens = []
        for t in store._tokens.values():  # type: ignore[attr-defined]
            all_tokens.append({
                "token": t.token[:12] + "…",
                "origin": t.origin,
                "domain": t.domain,
                "scope": t.scope,
                "issued_at": t.issued_at,
                "ttl_seconds": t.ttl_seconds,
                "revoked": t.revoked,
                "revoked_reason": t.revoked_reason,
                "consumed": t.is_consumed(),
                "expired": t.is_expired(),
            })
        return JSONResponse({"tokens": all_tokens})
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc), "tokens": []})


@router.get("/relationships")
async def relationships(_: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from core.social.relationship_model import get_store
        all_d = get_store().list_all()
        out = []
        for d in all_d:
            out.append({
                "relationship_id": d.relationship_id,
                "name": d.name,
                "trust_score": d.trust_score(),
                "fulfilled_rate": d.fulfilled_rate(),
                "open_commitments": [c.commitment_id for c in d.open_commitments()],
                "open_threads": d.open_threads,
                "boundaries_active": [b.description for b in d.boundaries if b.active],
                "topics_top": sorted(d.topics, key=lambda t: -t.weight)[:5],
            })
        return JSONResponse({"relationships": out})
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc), "relationships": []})


@router.get("/integration")
async def integration(_: None = Depends(_require_internal)) -> JSONResponse:
    return JSONResponse(_collect_integration())


@router.get("/viability")
async def viability(_: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from core.organism.viability import get_viability
        return JSONResponse(get_viability().report())
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc)})


@router.get("/world")
async def world(_: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from core.embodiment.world_bridge import get_permissions
        store = get_permissions()
        return JSONResponse({"channels": {c: {"granted": p.is_active(), "notes": p.notes} for c, p in store.all_channels().items()}})
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc), "channels": {}})


@router.get("/conscience")
async def conscience_recent(limit: int = 50, _: None = Depends(_require_internal)) -> JSONResponse:
    try:
        from pathlib import Path as _P
        path = _P.home() / ".aura" / "data" / "conscience" / "violations.jsonl"
        if not path.exists():
            return JSONResponse({"violations": []})
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return JSONResponse({"violations": rows[-int(limit):]})
    except Exception as exc:
        record_degradation('dashboard', exc)
        return JSONResponse({"error": str(exc), "violations": []})


# ─── single-action causal trace ────────────────────────────────────────────


@trace_router.get("/{receipt_id}")
async def trace(receipt_id: str = Path(...), _: None = Depends(_require_internal)) -> JSONResponse:
    """Return the full causal chain for one action receipt — drive → state
    → proposal → score → simulation → will → authority → token → execution
    → outcome → lesson.
    """
    try:
        from core.agency.agency_orchestrator import get_receipt_log
        recent = get_receipt_log().recent(limit=512)
        for r in reversed(recent):
            if r.get("proposal_id") == receipt_id:
                return JSONResponse({"trace": r})
        raise HTTPException(status_code=404, detail="receipt_not_found")
    except HTTPException:
        raise
    except Exception as exc:
        record_degradation('dashboard', exc)
        raise HTTPException(status_code=500, detail=str(exc))


__all__ = ["router", "trace_router"]
