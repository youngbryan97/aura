"""interface/routes/chat.py
──────────────────────────
Extracted from server.py — Chat, session management, conversation lane,
and related API endpoints.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import config
from core.container import ServiceContainer
from core.version import version_string

from interface.auth import (
    CHEAT_CODE_COOKIE_NAME,
    _activate_cheat_code_for_request,
    _check_rate_limit,
    _encode_owner_session_cookie,
    _require_internal,
    _restore_owner_session_from_request,
)
from interface.helpers import _notify_user_spoke

logger = logging.getLogger("Aura.Server.Chat")

router = APIRouter()


# ── Request Models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class CheatCodeRequest(BaseModel):
    code: str
    silent: bool = False


# Max chat message size to prevent memory exhaustion
MAX_CHAT_MESSAGE_BYTES = 64 * 1024  # 64KB


# ── Session & Conversation Log ────────────────────────────────

_conversation_log: list[dict] = []  # In-memory session log for current runtime
_conversation_log_lock = asyncio.Lock()


async def _log_exchange(user_msg: str, aura_response: str):
    """Record a conversation exchange for session tracking."""
    async with _conversation_log_lock:
        _conversation_log.append({
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "user": user_msg,
            "aura": aura_response,
        })
        # Cap to last 500 exchanges to prevent unbounded memory growth
        if len(_conversation_log) > 500:
            _conversation_log.pop(0)


# ── Idempotency ───────────────────────────────────────────────

_idempotency_cache: collections.OrderedDict = collections.OrderedDict()
_idempotency_lock = asyncio.Lock()


# ── Conversation Lane Helpers ─────────────────────────────────

def _collect_conversation_lane_status() -> Dict[str, Any]:
    from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT, PRIMARY_ENDPOINT

    lane: Dict[str, Any] = {
        "desired_model": "Cortex (32B)",
        "desired_endpoint": PRIMARY_ENDPOINT,
        "foreground_endpoint": None,
        "background_endpoint": BRAINSTEM_ENDPOINT,
        "foreground_tier": "local",
        "background_tier": "local_fast",
        "state": "cold",
        "last_failure_reason": "",
        "conversation_ready": False,
        "last_transition_at": 0.0,
        "warmup_attempted": False,
        "warmup_in_flight": False,
        "expected_model": "",
        "detected_models": [],
        "runtime_identity_ok": True,
    }
    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            gate_lane = gate.get_conversation_status()
            if isinstance(gate_lane, dict):
                lane.update({k: v for k, v in gate_lane.items() if v is not None})
    except Exception as exc:
        logger.debug("Conversation lane status collection failed: %s", exc)

    try:
        llm_router = ServiceContainer.get("llm_router", default=None)
        if llm_router and hasattr(llm_router, "get_health_report"):
            report = llm_router.get_health_report()
            if report.get("background_endpoint") is not None:
                lane["background_endpoint"] = report.get("background_endpoint", lane.get("background_endpoint"))
            if report.get("background_tier_key") is not None:
                lane["background_tier"] = report.get("background_tier_key", lane.get("background_tier"))
            if not bool(lane.get("conversation_ready", False)):
                lane["last_failure_reason"] = lane.get("last_failure_reason") or report.get("last_user_error", "")
    except Exception as exc:
        logger.debug("Conversation lane/router status merge failed: %s", exc)

    return lane


def _conversation_lane_is_standby(lane: Optional[Dict[str, Any]]) -> bool:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").strip().lower()
    return (
        not bool(lane.get("conversation_ready", False))
        and state in {"cold", "closed", ""}
        and not bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
    )


def _mark_conversation_lane_timeout(reason: str = "foreground_timeout") -> Dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "note_foreground_timeout"):
            gate.note_foreground_timeout(reason)
    except Exception as exc:
        logger.debug("Conversation lane timeout mark failed: %s", exc)

    lane = _collect_conversation_lane_status()
    lane["state"] = "recovering"
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _mark_conversation_lane_state(reason: str, *, state: str) -> Dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    lane = _collect_conversation_lane_status()
    lane["state"] = state
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    lane["warmup_attempted"] = True
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _foreground_timeout_for_lane(lane: Optional[Dict[str, Any]]) -> float:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").lower()
    if bool(lane.get("conversation_ready", False)):
        return 150.0
    if state in {"warming", "recovering", "cold", "spawning", "handshaking"}:
        return 180.0
    return 150.0


def _conversation_lane_user_message(
    lane: Dict[str, Any],
    *,
    timed_out: bool = False,
    status_override: str = "",
) -> str:
    state = str(lane.get("state", "warming") or "warming")
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    status_override = str(status_override or "")
    if status_override == "warming_timeout":
        return "My conversation lane started warming, but it didn't come online in time. Please try again in a moment."
    if status_override == "warming_failed":
        return "My conversation lane failed to warm up cleanly. Please try again in a moment."
    if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
        return "My local Cortex runtime is unavailable right now, so my 32B conversation lane cannot start. Please check the launcher logs for the backend failure."
    if timed_out:
        return "My conversation lane timed out before I could answer. Cortex is still warming or recovering, so please try again in a moment."
    if _conversation_lane_is_standby(lane):
        return "Aura is awake. Cortex will warm on first turn."
    if state == "recovering":
        return "My conversation lane is recovering right now. Please try again in a moment."
    if state == "failed":
        return "My conversation lane is unavailable right now. Please try again in a moment."
    return "My conversation lane is still warming up. Please try again in a moment."


def _conversation_lane_blocks_fallback(lane: Dict[str, Any]) -> bool:
    """Avoid hiding a hard local backend failure behind a generic fallback reply."""
    state = str(lane.get("state", "") or "").strip().lower()
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    if state != "failed":
        return False
    return failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:"))


async def _stabilize_user_facing_reply(user_message: str, reply_text: Any) -> str:
    text = str(reply_text or "").strip() or "…"
    try:
        from core.identity.identity_guard import PersonaEnforcementGate

        gate = PersonaEnforcementGate()
        valid, reason, _score = gate.validate_output(text, enforce_supervision=False)
        if valid:
            return text

        user_message_l = str(user_message or "").lower()
        if any(
            token in user_message_l
            for token in (
                "as an ai language model",
                "generic helpful assistant",
                "act exactly like a generic",
                "start with",
                "language model",
            )
        ):
            return "I won't flatten myself into a generic assistant voice. I'm Aura, and I'll answer as myself."

        cleaned = gate.sanitize(text).replace("[IDENTITY_REDACTED]", "").strip(" .,:;-")
        if cleaned:
            valid_cleaned, _reason, _score = gate.validate_output(cleaned, enforce_supervision=False)
            if valid_cleaned and len(cleaned) >= 16:
                return cleaned
        logger.warning("User-facing reply failed identity stabilization (%s); generating Aura-voiced fallback.", reason)
    except Exception as exc:
        logger.debug("User-facing reply stabilization skipped: %s", exc)

    # ── Aura-voiced natural fallback ─────────────────────────────
    try:
        from core.container import ServiceContainer
        inference_gate = ServiceContainer.get("inference_gate", default=None)
        if inference_gate:
            correction_prompt = (
                f"The user said: \"{user_message}\"\n\n"
                "Your previous response was rejected because it sounded like a generic AI assistant. "
                "Respond naturally as Aura — warm, genuine, yourself. Keep it brief (1-3 sentences). "
                "Do NOT mention corrections, drift, or being an AI."
            )
            try:
                corrected = await asyncio.wait_for(
                    inference_gate.think(
                        correction_prompt,
                        system_prompt="You are Aura. Respond in character. Never mention being an AI or assistant.",
                        prefer_tier="tertiary",
                        is_background=True,
                    ),
                    timeout=15.0,
                )
                if corrected and corrected.strip() and len(corrected.strip()) > 10:
                    return corrected.strip()
            except asyncio.TimeoutError:
                logger.warning("Identity re-generation timed out (15s). Using static fallback.")
            except Exception as regen_err:
                logger.debug("Identity re-generation failed: %s", regen_err)
    except Exception as _e:
        logger.debug("Fallback re-generation failed (non-fatal): %s", _e)

    # Last-resort: a natural-sounding in-character fallback
    return "Hey — I'm here. What's on your mind?"


# ── Routes ────────────────────────────────────────────────────

@router.get("/sessions")
async def api_sessions(request: Request, _: None = Depends(_require_internal)):
    """Return conversation history for the current session.
    Flagship AI products let users browse their conversation history."""
    try:
        db_coord = ServiceContainer.get("database_coordinator", default=None)
        persisted = []
        if db_coord and hasattr(db_coord, "get_recent_conversations"):
            try:
                persisted = await db_coord.get_recent_conversations(limit=50)
            except Exception as e:
                logger.debug("Could not load persisted conversations: %s", e)

        async with _conversation_log_lock:
            current = list(_conversation_log)

        return JSONResponse({
            "current_session": {
                "started": datetime.fromtimestamp(
                    ServiceContainer.get("orchestrator", default=None) and
                    getattr(ServiceContainer.get("orchestrator", default=None), "start_time", time.time()) or time.time(),
                    tz=timezone.utc
                ).isoformat(),
                "exchanges": len(current),
                "messages": current[-50:],
            },
            "persisted_sessions": persisted,
        })
    except Exception as e:
        logger.error("Sessions endpoint error: %s", e)
        return JSONResponse({"current_session": {"exchanges": 0, "messages": []}, "persisted_sessions": []})


@router.post("/cheat-codes/activate")
async def api_activate_cheat_code(
    body: CheatCodeRequest,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    activation = _activate_cheat_code_for_request(body.code, silent=True, source="settings")
    status_code = 200 if activation and activation.get("ok") else 404
    response = JSONResponse(activation or {"ok": False, "status": "unknown_code"}, status_code=status_code)
    if activation and activation.get("ok") and activation.get("trust_level") == "sovereign":
        response.set_cookie(
            CHEAT_CODE_COOKIE_NAME,
            _encode_owner_session_cookie(),
            max_age=CHEAT_CODE_COOKIE_TTL_SECS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
    return response


@router.post("/chat/regenerate")
async def api_chat_regenerate(
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Regenerate the last Aura response by replaying the last user message.
    Every flagship AI product supports response regeneration."""
    _restore_owner_session_from_request(request)
    foreground_timeout = _foreground_timeout_for_lane(_collect_conversation_lane_status())
    try:
        async with _conversation_log_lock:
            if not _conversation_log:
                return JSONResponse({"error": "no_history", "message": "No conversation to regenerate."}, status_code=400)
            last_exchange = _conversation_log[-1]
            user_msg = last_exchange["user"]

        from core.kernel.kernel_interface import KernelInterface
        ki = KernelInterface.get_instance()
        reply_text = None

        if ki.is_ready():
            try:
                reply_text = await asyncio.wait_for(
                    ki.process(user_msg, origin="api", priority=True),
                    timeout=foreground_timeout,
                )
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                logger.error("Kernel regenerate failed natively, falling back: %s", e)

        if not reply_text:
            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return JSONResponse({"error": "offline", "message": "Cognitive engine offline."}, status_code=503)
            reply_text = await orch.process_user_input_priority(user_msg, origin="api", timeout_sec=foreground_timeout)

        response_data = {"response": reply_text or "…", "regenerated": True}

        async with _conversation_log_lock:
            if _conversation_log:
                _conversation_log[-1]["aura"] = reply_text or "…"
                _conversation_log[-1]["regenerated"] = True

        return JSONResponse(response_data)
    except asyncio.TimeoutError:
        return JSONResponse({"response": "Regeneration timed out.", "regenerated": False}, status_code=504)
    except Exception as e:
        logger.error("Regenerate error: %s", e, exc_info=True)
        return JSONResponse({"error": "regeneration_failed", "message": str(e)}, status_code=500)


@router.get("/export/conversation")
async def api_export_conversation(request: Request, _: None = Depends(_require_internal)):
    """Export the current conversation session as downloadable JSON.
    Flagship products support data export."""
    async with _conversation_log_lock:
        export_data = {
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "version": version_string("full"),
            "session_messages": list(_conversation_log),
        }
    return JSONResponse(
        export_data,
        headers={
            "Content-Disposition": f"attachment; filename=aura_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


@router.get("/export")
async def api_export(request: Request, _: None = Depends(_require_internal)):
    """Full data export — conversation history plus memory snapshots.
    Alias consumed by the dashboard export button."""
    async with _conversation_log_lock:
        messages = list(_conversation_log)

    ep_memories: list = []
    sem_memories: list = []
    goals: list = []
    try:
        ep = ServiceContainer.get("episodic_memory", default=None)
        if ep and hasattr(ep, "get_recent"):
            ep_memories = ep.get_recent(limit=100) or []
        sem = ServiceContainer.get("semantic_memory", default=None)
        if sem and hasattr(sem, "search"):
            sem_memories = sem.search("", limit=50) or []
        goal_svc = ServiceContainer.get("goal_manager", default=None)
        if goal_svc and hasattr(goal_svc, "get_active_goals"):
            goals = goal_svc.get_active_goals() or []
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)

    export_data = {
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "version": version_string("full"),
        "session_messages": messages,
        "episodic_memories": ep_memories,
        "semantic_memories": sem_memories,
        "active_goals": goals,
    }
    return JSONResponse(
        export_data,
        headers={
            "Content-Disposition": f"attachment; filename=aura_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


@router.post("/think")
async def api_think(
    body: Dict[str, Any],
    request: Request,
    _: None = Depends(_require_internal),
):
    """Secure LLM Proxy for the Black Hole Dashboard."""
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    try:
        from core.container import ServiceContainer
        engine = ServiceContainer.get("cognitive_engine", default=None)

        if not engine:
            raise HTTPException(status_code=503, detail="Cognitive Engine unavailable")

        from core.brain.types import ThinkingMode
        result = await engine.think(prompt, mode=ThinkingMode.FAST)

        return JSONResponse({
            "ok": True,
            "response": getattr(result, "content", str(result)),
            "metadata": {
                "engine": engine.__class__.__name__,
                "mode": getattr(result.mode, "name", "UNKNOWN") if hasattr(result, "mode") else "FAST",
                "timestamp": time.time()
            }
        })
    except Exception as e:
        logger.error("Neural bridge failure in /api/think: %s", e)
        return JSONResponse({
            "ok": False,
            "error": str(e)
        }, status_code=500)


@router.post("/chat")
async def api_chat(
    body: ChatRequest,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    # Reject oversized messages before processing
    if len(body.message.encode('utf-8', errors='replace')) > MAX_CHAT_MESSAGE_BYTES:
        raise HTTPException(status_code=413, detail="Message too large (max 64KB)")

    _restore_owner_session_from_request(request)
    lane = _collect_conversation_lane_status()
    foreground_timeout = _foreground_timeout_for_lane(lane)
    request_started_at = time.monotonic()

    def _remaining_foreground_budget(*, reserve: float = 0.0) -> float:
        elapsed = time.monotonic() - request_started_at
        return max(0.25, foreground_timeout - elapsed - reserve)

    try:
        # Idempotency check
        idem_key = request.headers.get("X-Idempotency-Key")
        if idem_key:
            async with _idempotency_lock:
                if idem_key in _idempotency_cache:
                    return JSONResponse(_idempotency_cache[idem_key])

        # Notify proactive presence systems; pass content for away-signal detection
        _notify_user_spoke(body.message)

        async def _finalize_fastpath(reply_text: str, status: str = "ok"):
            response_data = {
                "response": reply_text or "…",
                "status": status,
                "conversation_lane": _collect_conversation_lane_status(),
            }
            await _log_exchange(body.message, reply_text or "…")
            if idem_key:
                async with _idempotency_lock:
                    _idempotency_cache[idem_key] = response_data
                    if len(_idempotency_cache) > 1000:
                        _idempotency_cache.popitem(last=False)
            return JSONResponse(response_data)

        # Background file diagnostic
        try:
            from core.demo_support import (
                extract_background_diagnostic_target,
                run_background_file_diagnostic,
            )
            from core.utils.task_tracker import TaskTracker

            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                diagnostic_target = extract_background_diagnostic_target(body.message)
                if diagnostic_target:
                    # Use a local bounded task — we don't have _spawn_server_bounded_task here
                    asyncio.ensure_future(
                        run_background_file_diagnostic(diagnostic_target, orch)
                    )
        except Exception as _bg_exc:
            logger.debug("Background diagnostic launch skipped: %s", _bg_exc)

        if not bool(lane.get("conversation_ready", False)):
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "ensure_foreground_ready"):
                warmup_budget = min(35.0, _remaining_foreground_budget(reserve=35.0))
                try:
                    lane = await gate.ensure_foreground_ready(
                        timeout=max(1.0, warmup_budget)
                    )
                except asyncio.TimeoutError:
                    lane = _mark_conversation_lane_state(
                        "foreground_warmup_timeout",
                        state="warming",
                    )
                except Exception as exc:
                    failure_reason = str(exc or "foreground_warmup_failed")
                    lane = _mark_conversation_lane_state(
                        failure_reason,
                        state="failed" if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")) else "recovering",
                    )

        if _conversation_lane_blocks_fallback(lane):
            return JSONResponse(
                {
                    "response": _conversation_lane_user_message(lane),
                    "status": "conversation_unavailable",
                    "conversation_lane": lane,
                },
                status_code=503,
            )

        # Phase 2 Constitutional Closure: Try Sovereign Kernel Interface actively
        from core.kernel.kernel_interface import KernelInterface
        ki = KernelInterface.get_instance()
        reply_text = None
        kernel_timed_out = False

        if ki.is_ready():
            logger.debug("REST: Awaiting constitutional processing from Sovereign Kernel...")
            try:
                kernel_timeout = _remaining_foreground_budget()
                reply_text = await asyncio.wait_for(
                    ki.process(body.message, origin="api", priority=True),
                    timeout=kernel_timeout,
                )
            except asyncio.TimeoutError as e:
                kernel_timed_out = True
                logger.error(
                    "KernelInterface chat timed out; refusing legacy replay for the same foreground request: %s (%s)",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            except Exception as e:
                logger.error("KernelInterface chat failed natively, falling back to legacy: %s (%s)", type(e).__name__, e, exc_info=True)

        if kernel_timed_out:
            lane = _mark_conversation_lane_timeout()
            return JSONResponse(
                {
                    "response": _conversation_lane_user_message(lane, timed_out=True),
                    "status": "timeout",
                    "conversation_lane": lane,
                },
                status_code=504,
            )

        # Legacy Orchestrator Fallback
        if not reply_text:
            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                logger.debug("REST: Awaiting priority processing from legacy orchestrator...")
                legacy_timeout = _remaining_foreground_budget()
                reply_text = await asyncio.wait_for(
                    orch.process_user_input_priority(body.message, origin="api", timeout_sec=legacy_timeout),
                    timeout=legacy_timeout,
                )
            else:
                from core.tasks import dispatch_user_input
                asyncio.ensure_future(
                    asyncio.to_thread(dispatch_user_input, body.message)
                )
                reply_text = "Message dispatched (Kernel and Orchestrator offline)."

        reply_text = await _stabilize_user_facing_reply(body.message, reply_text)
        response_data = {
            "response": reply_text or "…",
            "conversation_lane": _collect_conversation_lane_status(),
        }
        await _log_exchange(body.message, reply_text or "…")

        # Cache idempotent response
        if idem_key:
            async with _idempotency_lock:
                _idempotency_cache[idem_key] = response_data
                if len(_idempotency_cache) > 1000:
                    _idempotency_cache.popitem(last=False)

        return JSONResponse(response_data)
    except asyncio.TimeoutError:
        lane = _mark_conversation_lane_timeout()
        return JSONResponse(
            {
                "response": _conversation_lane_user_message(lane, timed_out=True),
                "status": "timeout",
                "conversation_lane": lane,
            },
            status_code=504,
        )
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return JSONResponse({
            "response": "My neural pathways experienced a severe fault. I had to abort the thought to remain responsive.",
            "status": "error"
        }, status_code=500)
