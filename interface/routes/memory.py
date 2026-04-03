"""interface/routes/memory.py
──────────────────────────────
Extracted from server.py — Memory retrieval endpoints:
episodic, semantic, recent, and goal memory.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from core.container import ServiceContainer

from interface.auth import _check_rate_limit, _require_internal

logger = logging.getLogger("Aura.Server.Memory")

router = APIRouter()


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
) -> Dict[str, Any]:
    page_items = list(items[offset : offset + limit])
    page_end = offset + len(page_items)
    has_more = len(items) > page_end or (len(items) == window_limit and len(page_items) == limit)
    return {
        "items": page_items,
        "limit": limit,
        "offset": offset,
        "count": len(page_items),
        "has_more": has_more,
    }


async def _build_episodic_memory_response(limit: int, offset: int) -> JSONResponse:
    safe_limit, safe_offset, window_limit = _normalize_memory_pagination(limit, offset)
    try:
        ep_mem = ServiceContainer.get("episodic_memory", default=None)
        if ep_mem and hasattr(ep_mem, "recall_recent"):
            episodes = ep_mem.recall_recent(limit=window_limit)
            items = [e.to_dict() for e in episodes]
            return JSONResponse(
                _memory_page_payload(items, limit=safe_limit, offset=safe_offset, window_limit=window_limit)
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
                    _memory_page_payload(items, limit=safe_limit, offset=safe_offset, window_limit=window_limit)
                )
            except Exception as e:
                logger.debug("Memory recall failed: %s", e)
    except Exception as exc:
        logger.debug("Episodic memory recall failed: %s", exc)
    return JSONResponse(
        _memory_page_payload([], limit=safe_limit, offset=safe_offset, window_limit=window_limit)
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
async def api_memory_semantic(limit: int = 20, _: None = Depends(_require_internal)):
    """Retrieve semantic knowledge entries."""
    try:
        kg = ServiceContainer.get("knowledge_graph", default=None)
        if kg:
            if hasattr(kg, "search_knowledge"):
                results = kg.search_knowledge("*", limit=limit)
                items = []
                for r in results:
                    if isinstance(r, dict):
                        items.append(r)
                    elif hasattr(r, "items"):
                        items.append(dict(r))
                    else:
                        items.append({"key": str(r), "value": ""})
                return JSONResponse({"items": items})
            elif hasattr(kg, "get_stats"):
                stats = kg.get_stats()
                return JSONResponse({"items": [{"key": k, "value": str(v)} for k, v in stats.items()]})
    except Exception as exc:
        logger.debug("Semantic memory failed: %s", exc)
    return JSONResponse({"items": []})


@router.get("/memory/goals")
async def api_memory_goals(limit: int = 20, _: None = Depends(_require_internal)):
    """Retrieve active goals from knowledge graph and strategic planner."""
    goals: List[Dict[str, Any]] = []
    try:
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
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        # 3. Orchestrator goal queue
        orch = ServiceContainer.get("orchestrator", default=None)
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
            import networkx as _nx
            for u, v, data in bg.graph.edges(data=True):
                if data.get("is_goal"):
                    goals.append({
                        "description": f"{u} → {data.get('relation', '?')} → {v}",
                        "confidence": data.get("confidence", 0.0),
                        "centrality": data.get("centrality", 0.0),
                        "status": "active",
                        "source": "belief_graph",
                    })
    except Exception as exc:
        logger.debug("Goals retrieval failed: %s", exc)
    return JSONResponse({"items": goals[:limit]})
