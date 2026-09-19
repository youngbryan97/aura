"""interface/routes/memory.py
──────────────────────────────
Extracted from server.py — Memory retrieval endpoints:
episodic, semantic, recent, and goal memory.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.container import ServiceContainer
from core.governance.will import ActionDomain, get_will
from core.governance_context import governed_scope
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_orchestrator
from interface.auth import _check_rate_limit, _require_internal

logger = logging.getLogger("Aura.Server.Memory")

router = APIRouter()
_MEMORY_ROUTE_RECOVERABLE_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


# ── Pagination Helpers ────────────────────────────────────────

_MEMORY_PAGE_LIMIT = 200
_MEMORY_WINDOW_LIMIT = 1000


def _normalize_memory_pagination(limit: int, offset: int) -> tuple[int, int, int]:
    safe_limit = max(1, min(int(limit or 20), _MEMORY_PAGE_LIMIT))
    requested_offset = max(0, int(offset or 0))
    max_offset = max(0, _MEMORY_WINDOW_LIMIT - safe_limit)
    safe_offset = min(requested_offset, max_offset)
    return safe_limit, safe_offset, safe_offset + safe_limit


def _memory_page_payload(
    items: List[Dict[str, Any]],
    *,
    limit: int,
    offset: int,
    window_limit: int,
    degradation_reasons: List[str] | None = None,
) -> Dict[str, Any]:
    page_items = list(items[offset : offset + limit])
    page_end = offset + len(page_items)
    has_more = len(items) > page_end or (len(items) == window_limit and len(page_items) == limit)
    payload = {
        "items": page_items,
        "limit": limit,
        "offset": offset,
        "count": len(page_items),
        "has_more": has_more,
    }
    if degradation_reasons is not None:
        payload["degraded"] = bool(degradation_reasons)
        payload["degradation_reasons"] = list(degradation_reasons)
    return payload


def _record_memory_route_degradation(stage: str, exc: BaseException, reasons: List[str]) -> None:
    reasons.append(stage)
    record_degradation('memory', exc)
    logger.debug("%s: %s", stage, exc)


async def _build_episodic_memory_response(limit: int, offset: int) -> JSONResponse:
    safe_limit, safe_offset, window_limit = _normalize_memory_pagination(limit, offset)
    degradation_reasons: List[str] = []
    try:
        ep_mem = ServiceContainer.get("episodic_memory", default=None)
        if ep_mem and hasattr(ep_mem, "recall_recent"):
            episodes = ep_mem.recall_recent(limit=window_limit)
            items = [e.to_dict() for e in episodes]
            return JSONResponse(
                _memory_page_payload(
                    items,
                    limit=safe_limit,
                    offset=safe_offset,
                    window_limit=window_limit,
                    degradation_reasons=degradation_reasons,
                )
            )
        mem = ServiceContainer.get("memory_manager", default=None)
        if mem and hasattr(mem, "recall"):
            try:
                recalled = await mem.recall("recent", limit=window_limit)
                items = []
                for i in recalled:
                    if hasattr(i, "to_dict"):
                        items.append(i.to_dict())
                    else:
                        items.append({"context": str(i), "timestamp": time.time()})
                return JSONResponse(
                    _memory_page_payload(
                        items,
                        limit=safe_limit,
                        offset=safe_offset,
                        window_limit=window_limit,
                        degradation_reasons=degradation_reasons,
                    )
                )
            except _MEMORY_ROUTE_RECOVERABLE_ERRORS as e:
                _record_memory_route_degradation("Memory manager recall failed", e, degradation_reasons)
    except _MEMORY_ROUTE_RECOVERABLE_ERRORS as exc:
        _record_memory_route_degradation("Episodic memory recall failed", exc, degradation_reasons)
    return JSONResponse(
        _memory_page_payload(
            [],
            limit=safe_limit,
            offset=safe_offset,
            window_limit=window_limit,
            degradation_reasons=degradation_reasons,
        )
    )


def _semantic_items_from_bulk_result(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    ids = list(payload.get("ids") or [])
    items: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        items.append(
            {
                "id": str(ids[idx]) if idx < len(ids) else "",
                "content": str(doc or ""),
                "metadata": dict(metadatas[idx] or {}) if idx < len(metadatas) else {},
            }
        )
    return items


async def _build_semantic_memory_response(limit: int, offset: int) -> JSONResponse:
    safe_limit, safe_offset, window_limit = _normalize_memory_pagination(limit, offset)
    degradation_reasons: List[str] = []
    try:
        sem = ServiceContainer.get("semantic_memory", default=None)
        if sem:
            if hasattr(sem, "get"):
                raw = await asyncio.to_thread(sem.get, None, window_limit)
                if isinstance(raw, dict):
                    items = _semantic_items_from_bulk_result(raw)
                    if items:
                        items.reverse()
                        return JSONResponse(
                            _memory_page_payload(
                                items,
                                limit=safe_limit,
                                offset=safe_offset,
                                window_limit=window_limit,
                                degradation_reasons=degradation_reasons,
                            )
                        )
            if hasattr(sem, "memories"):
                raw_memories = list(getattr(sem, "memories", []) or [])[-window_limit:]
                items = [
                    {
                        "id": str(entry.get("created", "")),
                        "content": str(entry.get("text", "") or ""),
                        "metadata": dict(entry.get("metadata", {}) or {}),
                        "timestamp": entry.get("created"),
                    }
                    for entry in raw_memories
                    if isinstance(entry, dict)
                ]
                items.reverse()
                return JSONResponse(
                    _memory_page_payload(
                        items,
                        limit=safe_limit,
                        offset=safe_offset,
                        window_limit=window_limit,
                        degradation_reasons=degradation_reasons,
                    )
                )
            if hasattr(sem, "search"):
                raw = await asyncio.to_thread(sem.search, "", window_limit)
                if isinstance(raw, list):
                    items = []
                    for entry in raw:
                        if isinstance(entry, dict):
                            items.append(
                                {
                                    "id": str(entry.get("id", "")),
                                    "content": str(entry.get("content") or entry.get("text") or ""),
                                    "metadata": dict(entry.get("metadata", {}) or {}),
                                }
                            )
                    if items:
                        return JSONResponse(
                            _memory_page_payload(
                                items,
                                limit=safe_limit,
                                offset=safe_offset,
                                window_limit=window_limit,
                                degradation_reasons=degradation_reasons,
                            )
                        )

        kg = ServiceContainer.get("knowledge_graph", default=None)
        if kg:
            if hasattr(kg, "search_knowledge"):
                results = kg.search_knowledge("*", limit=window_limit)
                items = []
                for r in results:
                    if isinstance(r, dict):
                        items.append(r)
                    elif hasattr(r, "items"):
                        items.append(dict(r))
                    else:
                        items.append({"key": str(r), "value": ""})
                return JSONResponse(
                    _memory_page_payload(
                        items,
                        limit=safe_limit,
                        offset=safe_offset,
                        window_limit=window_limit,
                        degradation_reasons=degradation_reasons,
                    )
                )
            if hasattr(kg, "get_stats"):
                stats = kg.get_stats()
                items = [{"key": k, "value": str(v)} for k, v in stats.items()]
                return JSONResponse(
                    _memory_page_payload(
                        items,
                        limit=safe_limit,
                        offset=safe_offset,
                        window_limit=window_limit,
                        degradation_reasons=degradation_reasons,
                    )
                )
    except _MEMORY_ROUTE_RECOVERABLE_ERRORS as exc:
        _record_memory_route_degradation("Semantic memory failed", exc, degradation_reasons)
    return JSONResponse(
        _memory_page_payload(
            [],
            limit=safe_limit,
            offset=safe_offset,
            window_limit=window_limit,
            degradation_reasons=degradation_reasons,
        )
    )


# ── Routes ────────────────────────────────────────────────────

@router.get("/memory/recent")
async def api_memory_recent(
    limit: int = 20,
    offset: int = 0,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Legacy endpoint — redirects to episodic memory."""
    return await _build_episodic_memory_response(limit=limit, offset=offset)


@router.get("/memory/episodic")
async def api_memory_episodic(
    limit: int = 20,
    offset: int = 0,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Retrieve recent episodic memories as structured dicts."""
    return await _build_episodic_memory_response(limit=limit, offset=offset)


@router.get("/memory/semantic")
async def api_memory_semantic(
    limit: int = 20,
    offset: int = 0,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Retrieve semantic knowledge entries."""
    return await _build_semantic_memory_response(limit=limit, offset=offset)


@router.get("/memory/goals")
async def api_memory_goals(limit: int = 20, _: None = Depends(_require_internal)):
    """Retrieve the unified goal lifecycle snapshot."""
    goals: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    degradation_reasons: List[str] = []
    try:
        goal_engine = ServiceContainer.get("goal_engine", default=None)
        if goal_engine and hasattr(goal_engine, "build_snapshot"):
            snapshot = await asyncio.to_thread(
                goal_engine.build_snapshot,
                max(30, int(limit or 20) * 3),
                include_external=True,
            )
            return JSONResponse(
                {
                    "items": list(snapshot.get("items") or []),
                    "summary": dict(snapshot.get("summary") or {}),
                }
            )

        # 1. Knowledge Graph learning goals
        kg = ServiceContainer.get("knowledge_graph", default=None)
        if kg and hasattr(kg, "get_active_learning_goals"):
            learning_goals = kg.get_active_learning_goals()
            for g in learning_goals:
                if isinstance(g, dict):
                    goals.append({"description": g.get("goal", str(g)), "status": "active", "source": "learning"})
                elif hasattr(g, "items"):
                    d = dict(g)
                    goals.append({"description": d.get("goal", str(g)), "status": "active", "source": "learning"})
                else:
                    goals.append({"description": str(g), "status": "active", "source": "learning"})

        # 2. Strategic Planner projects
        planner = ServiceContainer.get("strategic_planner", default=None)
        if planner and hasattr(planner, "store"):
            try:
                projects = planner.store.get_active_projects()
                for p in projects[:limit]:
                    goals.append({
                        "id": getattr(p, "id", ""),
                        "description": getattr(p, "goal", str(p)),
                        "status": getattr(p, "status", "active"),
                        "source": "strategic",
                    })
            except _MEMORY_ROUTE_RECOVERABLE_ERRORS as _exc:
                _record_memory_route_degradation("Strategic planner goals failed", _exc, degradation_reasons)

        # 3. Orchestrator goal queue
        orch = resolve_orchestrator()
        if orch and hasattr(orch, "goals"):
            goals_list = list(orch.goals)
            for i in range(min(len(goals_list), limit)):
                g = goals_list[i]
                goals.append({
                    "description": getattr(g, "objective", str(g)),
                    "status": "queued",
                    "source": "orchestrator",
                })

        # 4. BeliefGraph goal edges
        bg = ServiceContainer.get("belief_graph", default=None)
        if bg and hasattr(bg, "graph"):
            for u, v, data in bg.graph.edges(data=True):
                if data.get("is_goal"):
                    goals.append({
                        "description": f"{u} → {data.get('relation', '?')} → {v}",
                        "confidence": data.get("confidence", 0.0),
                        "centrality": data.get("centrality", 0.0),
                        "status": "active",
                        "source": "belief_graph",
                    })
    except _MEMORY_ROUTE_RECOVERABLE_ERRORS as exc:
        _record_memory_route_degradation("Goals retrieval failed", exc, degradation_reasons)
    summary = {
        "active_count": sum(1 for item in goals if item.get("status") not in {"completed", "failed"}),
        "completed_count": sum(1 for item in goals if item.get("status") == "completed"),
        "failed_count": sum(1 for item in goals if item.get("status") == "failed"),
    }
    payload: Dict[str, Any] = {"items": goals[:limit], "summary": summary}
    if degradation_reasons:
        payload["degraded"] = True
        payload["degradation_reasons"] = degradation_reasons
    return JSONResponse(payload)


class MemoryEditRequest(BaseModel):
    id: str
    text: str


class MemoryControlRequest(BaseModel):
    id: str


def _get_semantic_memory() -> Any:
    sem = ServiceContainer.get("semantic_memory", default=None)
    if sem is None:
        raise LookupError("semantic_memory not initialized")
    return sem


async def _governed_memory_control(
    *,
    operation: str,
    record_id: str,
    summary: str,
    method_name: str,
    method_args: tuple[Any, ...] = (),
) -> Dict[str, Any]:
    record_id = str(record_id or "").strip()
    if not record_id:
        return {"ok": False, "error": "memory id is required"}
    try:
        sem = _get_semantic_memory()
        method = getattr(sem, method_name, None)
        if not callable(method):
            return {"ok": False, "error": f"semantic_memory missing {method_name}"}
        decision = get_will().decide(
            content=f"semantic_memory.{operation}:{record_id}:{summary[:180]}",
            source="interface.memory",
            domain=ActionDomain.MEMORY_WRITE,
            priority=0.65,
        )
        if not decision.is_approved():
            return {
                "ok": False,
                "status": "refused",
                "error": decision.reason,
                "will_receipt_id": decision.receipt_id,
            }
        async with governed_scope(decision):
            ok = await asyncio.to_thread(method, record_id, *method_args)
        return {"ok": bool(ok), "will_receipt_id": decision.receipt_id}
    except _MEMORY_ROUTE_RECOVERABLE_ERRORS as exc:
        record_degradation("memory", exc)
        logger.warning("Governed memory control failed for %s: %s", operation, exc)
        return {"ok": False, "error": str(exc)}


@router.post("/memory/edit")
async def api_memory_edit(req: MemoryEditRequest, _: None = Depends(_require_internal)):
    return await _governed_memory_control(
        operation="edit",
        record_id=req.id,
        summary=req.text,
        method_name="edit_memory",
        method_args=(req.text,),
    )


@router.post("/memory/delete")
async def api_memory_delete(req: MemoryControlRequest, _: None = Depends(_require_internal)):
    return await _governed_memory_control(
        operation="delete",
        record_id=req.id,
        summary="delete memory",
        method_name="delete_memory",
    )


@router.post("/memory/freeze")
async def api_memory_freeze(req: MemoryControlRequest, frozen: bool = True, _: None = Depends(_require_internal)):
    return await _governed_memory_control(
        operation="freeze",
        record_id=req.id,
        summary=f"set frozen={bool(frozen)}",
        method_name="freeze_memory",
        method_args=(bool(frozen),),
    )


@router.post("/memory/contest")
async def api_memory_contest(req: MemoryControlRequest, contested: bool = True, _: None = Depends(_require_internal)):
    return await _governed_memory_control(
        operation="contest",
        record_id=req.id,
        summary=f"set contested={bool(contested)}",
        method_name="contest_memory",
        method_args=(bool(contested),),
    )


@router.post("/memory/mark_false")
async def api_memory_mark_false(req: MemoryControlRequest, is_false: bool = True, _: None = Depends(_require_internal)):
    return await _governed_memory_control(
        operation="mark_false",
        record_id=req.id,
        summary=f"set false={bool(is_false)}",
        method_name="mark_false",
        method_args=(bool(is_false),),
    )


@router.get("/memory/provenance")
async def api_memory_provenance(id: str, _: None = Depends(_require_internal)):
    sem = ServiceContainer.get("semantic_memory", default=None)
    if sem and hasattr(sem, "get_provenance"):
        return sem.get_provenance(id)
    return {"error": "semantic_memory not initialized or missing get_provenance"}

@router.get("/memory/export")
async def api_memory_export(_: None = Depends(_require_internal)):
    sem = ServiceContainer.get("semantic_memory", default=None)
    if sem and hasattr(sem, "list_memory_records"):
        return {"memories": await asyncio.to_thread(sem.list_memory_records)}
    return {"error": "semantic_memory not initialized"}
