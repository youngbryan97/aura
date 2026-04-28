"""interface/server.py
────────────────────
Aura Luna — FastAPI entry-point.

Decomposed: Routes live in interface/routes/*, auth in interface/auth.py,
WebSocket infrastructure in interface/websocket_manager.py, event bridge
in interface/event_bridge.py. This file retains only:
  - Imports and app creation
  - Lifespan context manager
  - Middleware stack
  - WebSocket endpoint and broadcaster
  - SPA catch-all
  - Entry-point
"""
# ruff: noqa: E402
# This module bootstraps logging, middleware, and route registration in phases;
# several imports intentionally stay next to the phase they wire.
from __future__ import annotations
from core.runtime.errors import record_degradation


# ── stdlib ────────────────────────────────────────────────────
import asyncio
import contextvars
import hmac
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import psutil

# ── Third-party ───────────────────────────────────────────────
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

try:
    from fastapi.responses import ORJSONResponse
except Exception:
    ORJSONResponse = JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

try:
    import sounddevice as sd
except ImportError:
    sd = None  # Audio features degrade gracefully

# ── Internal — logging first (no other internal imports before this) ──
from core.config import config
from core.container import ServiceContainer
from core.event_bus import get_event_bus

bus = get_event_bus()
from core.logging_config import setup_logging

logger = setup_logging("Aura.Server")

from core.health.boot_status import build_boot_health_snapshot
from core.runtime_tools import get_runtime_state
from core.utils.task_tracker import TaskTracker
from core.version import VERSION, version_string

PROJECT_ROOT = config.paths.project_root
_server_task_tracker = TaskTracker(name="AuraServer", max_concurrent=128)


def _spawn_server_task(coro, *, name: str) -> asyncio.Task:
    return _server_task_tracker.create_task(coro, name=name)


def _spawn_server_bounded_task(coro, *, name: str) -> asyncio.Task:
    return _server_task_tracker.bounded_track(coro, name=name)

logger.info("🚀 KERNEL LIFESPAN: Starting... EventBus ID: %s", bus._bus_id)

# Diagnostic: Identify process role
_is_proxy = os.environ.get("AURA_GUI_PROXY") == "1"
logger.info("📡 [PROCESS_BOOT] PID: %s | Role: %s", os.getpid(), "GUI_PROXY" if _is_proxy else "KERNEL")

# Lazy-loaded heavy subsystems (via lifespan)
_LocalBrain       = None
_LatentCore       = None
_PredictiveSelf   = None
_FastMouth        = None
_LocalVision      = None
_voice_engine_fn  = None


# ── WebSocket broadcast infrastructure (extracted to interface/websocket_manager.py) ──
from interface.websocket_manager import (
    MessageBroadcastBus as MessageBroadcastBus,
)
from interface.websocket_manager import (
    WebSocketManager as WebSocketManager,
)
from interface.websocket_manager import (
    broadcast_bus,
    log_queue,
    ws_manager,
)

# Wire task spawner into ws_manager now that _spawn_server_task is defined
ws_manager.set_task_spawner(lambda coro, name: _spawn_server_task(coro, name=name))


main_loop: asyncio.AbstractEventLoop | None = None
_event_bridge_task: asyncio.Task | None = None


class _QueueHandler(logging.Handler):
    """Sends structured log records to the async broadcast queue.
    Implements a circular buffer for log_queue to prevent OOM/silencing.
    """

    _recursion_guard: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "_qh_recursion_guard", default=False
    )
    _overflow_logged: bool = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._recursion_guard.get():
            return
        token = self._recursion_guard.set(True)
        try:
            msg = self.format(record)
            if "Error receiving data from connection" in msg or "Stream broken" in msg:
                return

            log_entry = {
                "type": "log",
                "message": msg,
                "level": record.levelname.lower(),
                "timestamp": record.created,
                "module": record.name
            }

            log_queue.append(log_entry)

            if not self._overflow_logged and len(log_queue) >= log_queue.maxlen:
                logger.warning("Log buffer reached capacity: Circular buffer active (dropping oldest).")
                self._overflow_logged = True
            elif len(log_queue) < log_queue.maxlen:
                self._overflow_logged = False

            if main_loop is not None and not main_loop.is_closed() and main_loop.is_running():
                publish_coro = broadcast_bus.publish(log_entry)
                try:
                    asyncio.run_coroutine_threadsafe(publish_coro, main_loop)
                except Exception:
                    try:
                        publish_coro.close()
                    except Exception:
                        pass  # no-op: intentional
                    raise

        except Exception:
            print(f"CRITICAL LOG FALLBACK: {record.levelname} - {record.getMessage()}", file=sys.stderr)
        finally:
            self._recursion_guard.reset(token)


# Attach queue handler to root logger
_qh = _QueueHandler()
_qh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_qh)


# ── Event bridge functions (extracted to interface/event_bridge.py) ──
from interface.auth import _restore_owner_session_from_request, validate_runtime_security_request
from interface.event_bridge import mycelial_ui_callback, run_event_bridge

# ── Shared helpers ──
from interface.helpers import _notify_user_spoke

# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all subsystems on boot; shut them down cleanly on exit."""
    global main_loop
    global _LocalBrain, _LatentCore
    global _PredictiveSelf, _FastMouth, _LocalVision, _voice_engine_fn

    main_loop = asyncio.get_running_loop()
    logger.info("Aura Server %s starting… (Lifespan Enter)", version_string("short"))

    # Initialize EventBus loop for threadsafe publication from background tasks
    from core.event_bus import get_event_bus
    get_event_bus().set_loop(main_loop)

    # 0. Global Registration
    is_gui_proxy = os.environ.get("AURA_GUI_PROXY") == "1"
    from core.service_registration import register_all_services
    register_all_services(is_proxy=is_gui_proxy)

    if is_gui_proxy:
        bus = ServiceContainer.get("actor_bus", default=None)
        if bus:
            logger.info("📡 Igniting deferred ActorBus transports...")
            bus.start_transports()

    # 0.1 Mycelial Network
    from core.mycelium import MycelialNetwork

    mycelial = ServiceContainer.get("mycelial_network", default=None)
    if not mycelial:
        mycelial = MycelialNetwork()
        ServiceContainer.register_instance("mycelial_network", mycelial)
        _spawn_server_bounded_task(
            asyncio.to_thread(mycelial.map_infrastructure, base_dir=str(config.paths.project_root)),
            name="server.mycelium.map_infrastructure",
        )

    ServiceContainer.register_instance("mycelium", mycelial)

    mycelial.set_ui_callback(mycelial_ui_callback)
    if is_gui_proxy:
        logger.info("📡 GUI Proxy: Mycelial Network synchronized.")

    # Ensure data directories exist
    config.paths.create_directories()
    logger.info("📡 Lifespan: Directories verified.")

    # ── Boot heavy subsystems (each gracefully degraded) ──
    from core.utils.safe_import import async_safe_import, is_missing

    if not is_gui_proxy:
        try:
            mod = await async_safe_import("core.local_chat_brain", optional=True)
            if not is_missing(mod):
                _LocalBrain = mod.LocalChatBrain
            else:
                logger.warning("LocalBrain (legacy) unavailable — Fallback mode active")
        except Exception as _exc:
            record_degradation('server', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            mod = await async_safe_import("core.latent.latent_core", optional=True)
            if not is_missing(mod):
                _LatentCore = mod.LatentCore
        except Exception as _exc:
            record_degradation('server', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            mod = await async_safe_import("core.predictive.predictive_self_model", optional=True)
            if not is_missing(mod):
                _PredictiveSelf = mod.PredictiveSelfModel
        except Exception as _exc:
            record_degradation('server', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.tts_stream", optional=True)
            if not is_missing(mod):
                _FastMouth = mod.FastMouth
        except Exception as _exc:
            record_degradation('server', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.screen_vision", optional=True)
            if not is_missing(mod):
                _LocalVision = mod.LocalVision
        except Exception as _exc:
            record_degradation('server', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.voice_engine", optional=True)
            if not is_missing(mod):
                _voice_engine_fn = mod.get_voice_engine
                try:
                    _ve_check = _voice_engine_fn()
                    if _ve_check is None:
                        logger.warning("⚠️ Voice engine factory returned None — voice features unavailable.")
                        _voice_engine_fn = None
                    else:
                        logger.info("✓ Voice engine health check passed.")
                except Exception as ve_err:
                    record_degradation('server', ve_err)
                    logger.warning("⚠️ Voice engine health check failed: %s — disabling voice.", ve_err)
                    _voice_engine_fn = None
        except Exception as _exc:
            record_degradation('server', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
    else:
        logger.info("📡 GUI Proxy Mode: Skipping heavy subsystem initialization (Brain, TTS, Vision).")

    # Share voice engine factory with privacy route module
    from interface.routes.privacy import set_voice_engine_fn
    set_voice_engine_fn(_voice_engine_fn)

    # ── Trigger cognitive substrate ──
    if not is_gui_proxy:
        logger.info("📡 Kernel Mode: Orchestrator startup deferred to aura_main (to prevent double-boot).")
    else:
        logger.info("📡 GUI Proxy Mode: Cognitive Orchestrator boot SKIPPED.")

    # ── Start WS broadcaster ──
    _spawn_server_task(_ws_broadcaster(), name="ws_broadcaster")

    # ── Bridge EventBus to WS broadcaster (Live HUD) ──
    is_gui_proxy = os.environ.get("AURA_GUI_PROXY") == "1"
    global _event_bridge_task
    if _event_bridge_task is None or _event_bridge_task.done():
        _event_bridge_task = _spawn_server_task(
            run_event_bridge(is_gui_proxy=is_gui_proxy), name="event_bus_bridge"
        )
    else:
        logger.debug("EventBridge task already running; skipping redundant spawn.")

    logger.info("Aura Server online — %s", version_string("full"))
    yield  # ← app is live here

    # ── Shutdown ──
    logger.info("Aura Server shutting down…")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Aura Luna Agent",
    description="Secure interface for the Aura Luna autonomous engine.",
    version=VERSION,
    lifespan=lifespan,
)

# 0.1 Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

# 0.2 Correlation ID Middleware & Context

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    correlation_id.set(req_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = req_id
    return response

# SEC-02: Defense-in-depth token verification middleware
@app.middleware("http")
async def verify_token_middleware(request: Request, call_next):
    try:
        validate_runtime_security_request(request)
    except HTTPException as exc:
        return Response(status_code=exc.status_code, content=str(exc.detail))
    return await call_next(request)

# ── Storage & Resource Management ─────────────────────────────

DATA_DIR = Path(config.paths.data_dir)
UPLOAD_DIR = DATA_DIR / "uploads"
GEN_IMAGES_DIR = DATA_DIR / "generated_images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GEN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def _cache_policy_for_path(path: str) -> dict[str, str] | None:
    normalized = str(path or "")
    live_shell_paths = {
        "/",
        "/static/aura.css",
        "/static/aura.js",
        "/static/manifest.json",
        "/static/service-worker.js",
    }
    if normalized in live_shell_paths or normalized.endswith("/index.html"):
        return dict(NO_CACHE_HEADERS)
    if normalized.startswith(("/static", "/data")):
        return {"Cache-Control": "public, max-age=31536000, immutable"}
    return None


# Mount static files for uploads and generated media
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR, html=False), name="uploads")
app.mount("/data/generated_images", StaticFiles(directory=GEN_IMAGES_DIR, html=False), name="generated_images")


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    policy = _cache_policy_for_path(request.url.path)
    if policy and hasattr(response, "headers"):
        for key, value in policy.items():
            cast(Response, response).headers[key] = value
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not config.security.internal_only_mode else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Api-Token", "X-Idempotency-Key", "Authorization"],
)

STATIC_DIR = config.paths.project_root / "interface" / "static"
SHELL_DIST_DIR = STATIC_DIR / "shell" / "dist"
LEGACY_UI_INDEX = STATIC_DIR / "index.html"


def _react_shell_enabled() -> bool:
    """Keep the original Aura HUD as the canonical shell unless explicitly opted in."""
    return os.environ.get("AURA_ENABLE_REACT_SHELL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Magnum Opus: Request ID Middleware ─────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject a unique request ID for distributed tracing and error correlation."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Magnum Opus: Global Exception Handler ─────────────────────

from datetime import UTC, datetime


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Phenomenal error envelope for every unhandled exception.

    The user never sees a Python traceback. core/resilience/phenomenal_error_map
    classifies the exception, pushes a substrate signal (cognitive fog,
    sensory deprivation, etc.), and emits the four-button recovery envelope
    that the frontend's error_banner.js renders automatically.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception [req=%s] %s: %s",
        request_id, type(exc).__name__, exc,
        exc_info=True,
    )
    try:
        from core.resilience.phenomenal_error_map import PhenomenalRaise, build_envelope
        if isinstance(exc, PhenomenalRaise):
            envelope = exc.envelope
        else:
            envelope = build_envelope(exc, correlation_id=request_id)
        return JSONResponse(
            status_code=200,  # always 200 so the chat never appears broken
            content={
                "status": "phenomenal",
                "envelope": envelope.to_dict(),
                "user_message": envelope.user_message,
                "request_id": request_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    except Exception as inner:
        record_degradation('server', inner)
        # Fall back to a structured 500 only when the envelope builder
        # itself crashes — should never happen in practice, but we never
        # want this handler to compound the problem.
        logger.error("phenomenal envelope build failed: %s", inner)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred. Aura's cognitive systems are recovering.",
                "request_id": request_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )


# ── Route Registration ────────────────────────────────────────
# Extracted route modules
from core.health.system_health import router as system_health_router
from core.session.checkpointing import CheckpointService
from interface import memory_ui
from interface.routes import chat as chat_routes
from interface.routes import dashboard as dashboard_routes
from interface.routes import inner_state as inner_state_routes
from interface.routes import interaction_signals as interaction_signal_routes
from interface.routes import memory as memory_routes
from interface.routes import multimodal as multimodal_routes
from interface.routes import performance as performance_routes
from interface.routes import privacy as privacy_routes
from interface.routes import rpc as rpc_routes
from interface.routes import settings as settings_routes
from interface.routes import subsystems as subsystem_routes
from interface.routes import system as system_routes

checkpoint_service = CheckpointService()

app.include_router(system_health_router, prefix="/api/health", tags=["health"])
app.include_router(memory_ui.router, prefix="/memory", tags=["memory"])
app.include_router(chat_routes.router, prefix="/api", tags=["chat"])
app.include_router(system_routes.router, prefix="/api", tags=["system"])
app.include_router(subsystem_routes.router, prefix="/api", tags=["subsystems"])
app.include_router(memory_routes.router, prefix="/api", tags=["memory-api"])
app.include_router(interaction_signal_routes.router, prefix="/api", tags=["interaction-signals"])
app.include_router(privacy_routes.router, prefix="/api", tags=["privacy"])
app.include_router(rpc_routes.router, prefix="/rpc", tags=["rpc"])
app.include_router(inner_state_routes.router, tags=["proof-surface"])
app.include_router(dashboard_routes.router, prefix="/api", tags=["dashboard"])
app.include_router(dashboard_routes.trace_router, prefix="/api", tags=["trace"])
app.include_router(settings_routes.router, prefix="/api", tags=["settings"])
app.include_router(multimodal_routes.router, prefix="/api", tags=["multimodal"])
app.include_router(performance_routes.router, prefix="/api", tags=["performance"])

_system_collect_liquid_state_payload = system_routes._collect_liquid_state_payload


def _collect_conversation_lane_status() -> dict[str, Any]:
    return chat_routes._collect_conversation_lane_status()


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    return chat_routes._conversation_lane_is_standby(lane)


def _collect_liquid_state_payload(
    ls_data: dict[str, Any],
    *,
    runtime_state: dict[str, Any],
    homeostasis_data: dict[str, Any],
) -> dict[str, Any]:
    return _system_collect_liquid_state_payload(
        ls_data,
        runtime_state=runtime_state,
        homeostasis_data=homeostasis_data,
    )


def _sync_legacy_system_exports() -> None:
    system_routes._restore_owner_session_from_request = _restore_owner_session_from_request
    system_routes._collect_conversation_lane_status = _collect_conversation_lane_status
    system_routes._conversation_lane_is_standby = _conversation_lane_is_standby
    system_routes._collect_liquid_state_payload = _collect_liquid_state_payload
    system_routes._collect_legacy_shell_status = _collect_legacy_shell_status
    system_routes.build_boot_health_snapshot = build_boot_health_snapshot
    system_routes.get_runtime_state = get_runtime_state
    system_routes.psutil = psutil


def _collect_stability_details() -> dict[str, Any]:
    _sync_legacy_system_exports()
    return system_routes._collect_stability_details()


def _collect_runtime_capabilities(conversation_lane: dict[str, Any] | None = None) -> dict[str, Any]:
    _sync_legacy_system_exports()
    return system_routes._collect_runtime_capabilities(conversation_lane)


def _collect_legacy_shell_status() -> dict[str, Any]:
    react_shell_enabled = _react_shell_enabled()
    return {
        "shell": "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell",
        "legacy_fallback_available": LEGACY_UI_INDEX.exists(),
        "experimental_shell_available": (SHELL_DIST_DIR / "index.html").exists(),
        "experimental_shell_enabled": react_shell_enabled,
        "canonical_shell": "legacy_shell" if LEGACY_UI_INDEX.exists() and not react_shell_enabled else "react_shell",
    }


# ── Compatibility re-exports ──────────────────────────────────────
# These functions were refactored into interface/routes/ but existing tests
# and internal callers still import them from interface.server.

ChatRequest = chat_routes.ChatRequest
api_chat = chat_routes.api_chat
_foreground_timeout_for_lane = chat_routes._foreground_timeout_for_lane
_conversation_lane_user_message = chat_routes._conversation_lane_user_message
_log_exchange = chat_routes._log_exchange
api_action_log = subsystem_routes.api_action_log


async def api_health(request: Request):
    _sync_legacy_system_exports()
    return await system_routes.api_health(request)


async def api_ui_bootstrap(request: Request = None):
    _sync_legacy_system_exports()
    return await system_routes.api_ui_bootstrap(request)


async def api_memory_episodic(limit: int = 20, offset: int = 0):
    return await memory_routes.api_memory_episodic(limit=limit, offset=offset)


# ── WebSocket broadcaster ─────────────────────────────────────

async def _ws_broadcaster() -> None:
    """Forward messages from broadcast_bus to all WebSocket clients."""
    q = await broadcast_bus.subscribe()
    try:
        while True:
            try:
                ptr, ts, msg = await asyncio.wait_for(q.get(), timeout=10.0)

                if ws_manager.count() == 0:
                    q.task_done()
                    continue

                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except json.JSONDecodeError:
                        msg = {"type": "message", "content": msg}
                elif not isinstance(msg, dict):
                    msg = {"type": "message", "content": str(msg)}

                try:
                    await asyncio.wait_for(ws_manager.broadcast(msg), timeout=15.0)
                except TimeoutError:
                    logger.warning("WS Broadcaster timeout - serious delivery lag detected")

                q.task_done()
            except TimeoutError:
                continue  # Pulsing
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('server', e)
                logger.error("WebSocket broadcaster error: %s", e)
                await asyncio.sleep(1.0)
    finally:
        await broadcast_bus.unsubscribe(q)


# ── Routes — UI ───────────────────────────────────────────────

from interface.auth import _require_internal


@app.get("/", include_in_schema=False)
async def serve_ui(request: Request):
    """Main entry point for the Sovereign HUD."""
    _require_internal(request)
    ui = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
    if not ui.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(str(ui), headers=NO_CACHE_HEADERS)


@app.get("/telemetry", include_in_schema=False)
async def serve_telemetry(request: Request):
    _require_internal(request)
    p = STATIC_DIR / "telemetry.html"
    return FileResponse(str(p), headers=NO_CACHE_HEADERS) if p.exists() else ORJSONResponse({"error": "not found"}, status_code=404)

# ── Routes — Checkpoints (Phase 5A) ───────────────────────────

@app.post("/api/checkpoints/save", tags=["checkpoints"])
async def save_checkpoint(request: Request):
    """Manually trigger a conversation checkpoint save."""
    _require_internal(request)
    data = await request.json()
    
    label = data.get("label", "manual")
    # In a full integration, these states would be pulled from the active KernelInterface
    messages = data.get("messages", [])
    
    filepath = checkpoint_service.save(
        messages=messages,
        label=label
    )
    if filepath:
        return {"ok": True, "filepath": filepath}
    return JSONResponse(status_code=500, content={"ok": False, "error": "Save failed"})

@app.post("/api/checkpoints/restore", tags=["checkpoints"])
async def restore_checkpoint(request: Request):
    """Restore conversation from a checkpoint."""
    _require_internal(request)
    data = await request.json()
    
    label = data.get("label")
    if label:
        cp = checkpoint_service.restore_by_label(label)
    else:
        cp = checkpoint_service.restore_latest()
        
    if cp:
        # Here we would inject the state back into the KernelInterface
        return {"ok": True, "turn_count": cp.turn_count, "messages": len(cp.messages)}
    return JSONResponse(status_code=404, content={"ok": False, "error": "Checkpoint not found"})

# ── Routes — WebSocket ────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)

    expected = os.environ.get("AURA_API_TOKEN", "")
    host = ws.client.host if ws.client else "unknown"
    is_local = host in ("127.0.0.1", "::1", "localhost")

    authenticated = not bool(expected) or is_local
    auth_timeout = 5.0

    try:
        if not authenticated:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=auth_timeout)
                data = json.loads(raw)
                if data.get("type") == "auth" and hmac.compare_digest(data.get("token", ""), expected):
                    authenticated = True
                    await ws.send_text(json.dumps({"type": "auth_success"}))
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "Unauthorized"}))
                    await ws.close(code=4001, reason="Unauthorized")
                    return
            except TimeoutError:
                await ws.close(code=4001, reason="Auth Timeout")
                return
            except json.JSONDecodeError:
                await ws.close(code=4001, reason="Invalid Auth Payload")
                return
        elif is_local and expected:
            await ws.send_text(json.dumps({"type": "auth_success", "note": "local_trust"}))

        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                msg_type = data.get("type")
                if msg_type == "user_message":
                    content = data.get("content", "")
                    if content:
                        logger.debug("WS: Received user_message: %s", content[:100])
                        _notify_user_spoke(content)

                        async def _handle_ws_message(ws_ref, user_content: str):
                            """Process user message and send response back via WebSocket."""
                            try:
                                from core.kernel.kernel_interface import KernelInterface
                                ki = KernelInterface.get_instance()
                                if ki.is_ready():
                                    # [STABILITY v53] Match HTTP timeout (180s) — was 120s,
                                    # causing WS to timeout while HTTP still succeeding.
                                    reply = await asyncio.wait_for(
                                        ki.process(user_content, origin="ws", priority=True),
                                        timeout=180.0,
                                    )
                                else:
                                    from core.event_bus import get_event_bus
                                    bus = get_event_bus()
                                    await bus.publish("user_input", {"message": user_content})
                                    return

                                if reply:
                                    await ws_ref.send_text(json.dumps({
                                        "type": "aura_message",
                                        "content": reply,
                                    }))
                                else:
                                    # [STABILITY v53] Never leave user without a response
                                    await ws_ref.send_text(json.dumps({
                                        "type": "aura_message",
                                        "content": "I lost my thread for a second. Say that again?",
                                    }))
                            except TimeoutError:
                                logger.error("WS: KernelInterface.process() timed out after 180s")
                                # [STABILITY v53] Try fast brainstem fallback before giving up
                                try:
                                    from core.container import ServiceContainer
                                    gate = ServiceContainer.get("inference_gate", default=None)
                                    if gate and hasattr(gate, "generate"):
                                        fallback = await asyncio.wait_for(
                                            gate.generate(
                                                user_content,
                                                context={
                                                    "origin": "ws",
                                                    "foreground_request": True,
                                                    "prefer_tier": "tertiary",
                                                    "allow_cloud_fallback": True,
                                                },
                                                timeout=15.0,
                                            ),
                                            timeout=15.0,
                                        )
                                        if fallback and str(fallback).strip():
                                            await ws_ref.send_text(json.dumps({
                                                "type": "aura_message",
                                                "content": str(fallback).strip(),
                                            }))
                                            return
                                except Exception:
                                    pass  # no-op: intentional
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": "I was thinking but my cortex took too long. Try again — I should be warmer now.",
                                }))
                            except Exception as e:
                                record_degradation('server', e)
                                logger.error("WS: Message handling failed: %s (%s)", type(e).__name__, e, exc_info=True)
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": "I hit a bump in my thinking. Try me again?",
                                }))

                        _spawn_server_bounded_task(
                            _handle_ws_message(ws, content),
                            name="server.ws.handle_message",
                        )
                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

            elif "bytes" in msg:
                if _voice_engine_fn:
                    ve = _voice_engine_fn()
                    if ve:
                        _spawn_server_bounded_task(
                            ve.feed_chunk(msg["bytes"]),
                            name="server.ws.feed_chunk",
                        )

    except WebSocketDisconnect as _exc:
        logger.debug("Suppressed WebSocketDisconnect: %s", _exc)
    except Exception as exc:
        record_degradation('server', exc)
        logger.debug("WS error: %s", exc)
    finally:
        await ws_manager.disconnect(ws)


# ── SPA Catch-all — v6.0 Traverse Hardened ────────────────────

@app.get("/{path:path}", include_in_schema=False)
async def spa_catchall(path: str, request: Request):
    """Secure catch-all to support SPA routing and static resolution with traversal protection."""
    _require_internal(request)

    if ".." in path or path.startswith("/") or "./" in path:
         fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
         return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)

    if path == "memory" or path.startswith("memory/"):
        dist_dir = STATIC_DIR / "memory" / "dist"
        if path == "memory":
            return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)
        sub_path = path[len("memory/"):]
        if not sub_path:
            return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)
        requested_path = (dist_dir / sub_path).resolve()
        if requested_path.is_file():
            return FileResponse(str(requested_path), headers=NO_CACHE_HEADERS)
        raw_path = (STATIC_DIR / "memory" / sub_path).resolve()
        if raw_path.is_file():
             return FileResponse(str(raw_path), headers=NO_CACHE_HEADERS)
        return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)

    if path == "shell" or path.startswith("shell/"):
        if LEGACY_UI_INDEX.exists() and not _react_shell_enabled():
            return FileResponse(str(LEGACY_UI_INDEX), headers=NO_CACHE_HEADERS)
        dist_dir = SHELL_DIST_DIR
        if path == "shell":
            return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)
        sub_path = path[len("shell/"):]
        requested_shell_path = (dist_dir / sub_path).resolve()
        if requested_shell_path.is_file():
            return FileResponse(str(requested_shell_path), headers=NO_CACHE_HEADERS)
        return FileResponse(str(dist_dir / "index.html"), headers=NO_CACHE_HEADERS)

    requested_path = (STATIC_DIR / path).resolve()

    if not str(requested_path).startswith(str(STATIC_DIR)) or not requested_path.exists():
         fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
         return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)

    if requested_path.is_file():
        return FileResponse(str(requested_path), headers=NO_CACHE_HEADERS)

    fallback = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
    return FileResponse(str(fallback), headers=NO_CACHE_HEADERS)


# ── Entry-point ───────────────────────────────────────────────

def main() -> None:
    from core.logging_config import setup_logging as _sl
    _sl(log_dir=config.paths.log_dir)

    host = "127.0.0.1" if config.security.internal_only_mode else "0.0.0.0"
    logger.info("Binding to %s:8000", host)

    uvicorn.run(
        "interface.server:app",
        host=host,
        port=8000,
        reload=False,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=10,
    )


if __name__ == "__main__":
    main()
