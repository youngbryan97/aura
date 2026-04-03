"""interface/routes/rpc.py
─────────────────────────
Extracted from server.py — Cosmic consciousness RPC endpoints
for collective intelligence peer communication.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from core.container import ServiceContainer

from interface.auth import _check_rate_limit, _require_internal

logger = logging.getLogger("Aura.Server.RPC")

router = APIRouter()


@router.post("/query_beliefs")
async def rpc_query_beliefs(
    params: Dict[str, Any],
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Remote peer querying beliefs about an entity."""
    sync = ServiceContainer.get("belief_sync", default=None)
    if sync:
        return await sync.handle_rpc_request("query_beliefs", params)
    return JSONResponse({"error": "BeliefSync not active"}, status_code=503)


@router.post("/receive_beliefs")
async def rpc_receive_beliefs(
    payload: Dict[str, Any],
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Remote peer pushing beliefs to this node."""
    sync = ServiceContainer.get("belief_sync", default=None)
    if sync:
        await sync.handle_incoming_beliefs(payload)
        return {"status": "accepted"}
    return JSONResponse({"error": "BeliefSync not active"}, status_code=503)


@router.post("/receive_principles")
async def rpc_receive_principles(
    payload: Dict[str, Any],
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Remote peer pushing principles to this node."""
    sync = ServiceContainer.get("belief_sync", default=None)
    if sync:
        await sync.handle_incoming_principles(payload)
        return {"status": "accepted"}
    return JSONResponse({"error": "BeliefSync not active"}, status_code=503)


@router.post("/receive_resonance")
async def rpc_receive_resonance(
    payload: Dict[str, Any],
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Remote peer pushing affective resonance to this node."""
    sync = ServiceContainer.get("belief_sync", default=None)
    if sync:
        return await sync.handle_rpc_request("receive_resonance", payload)
    return JSONResponse({"error": "BeliefSync not active"}, status_code=503)


@router.post("/attention_spike")
async def rpc_attention_spike(
    payload: Dict[str, Any],
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Remote peer pushing a collective attention spike."""
    sync = ServiceContainer.get("belief_sync", default=None)
    if sync:
        return await sync.handle_rpc_request("attention_spike", payload)
    return JSONResponse({"error": "BeliefSync not active"}, status_code=503)
