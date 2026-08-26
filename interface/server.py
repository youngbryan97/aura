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

# ── stdlib ────────────────────────────────────────────────────
import asyncio
import contextvars
import hmac
import json
import logging
import mimetypes
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlsplit

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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.launch_provenance import (
    RUNTIME_SHELL_ASSETS,
    RUNTIME_SHELL_PUBLIC_ASSETS,
    runtime_shell_request_path,
)
from core.runtime.shutdown_coordinator import is_shutdown_requested

try:
    from fastapi.responses import ORJSONResponse
except ImportError:
    ORJSONResponse = JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import sounddevice as sd
except ImportError:
    sd = None  # Audio features degrade gracefully

# ── Internal — logging first (no other internal imports before this) ──
from core.config import config
from core.container import ServiceContainer
from core.event_bus import get_event_bus

bus = get_event_bus()
from core.observability.logging_config import setup_logging

logger = setup_logging("Aura.Server")

from core.health.boot_status import build_boot_health_snapshot
from core.runtime.version import VERSION, version_string
from core.tools.runtime_tools import get_runtime_state
from core.utils.task_tracker import TaskTracker

PROJECT_ROOT = config.paths.project_root
_server_task_tracker = TaskTracker(name="AuraServer", max_concurrent=128)
_SERVER_BOUNDARY_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_HTTP_REQUESTS_TOTAL = Counter(
    "aura_http_requests_total",
    "HTTP requests served by the Aura interface.",
    ("method", "path", "status"),
)
_HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "aura_http_request_latency_seconds",
    "HTTP request latency for the Aura interface.",
    ("method", "path"),
)
_HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "aura_http_requests_in_progress",
    "HTTP requests currently being processed by the Aura interface.",
)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


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


async def _prewarm_chat_dependencies_after_cortex_ready(
    *,
    readiness_timeout_s: float = 180.0,
    poll_interval_s: float = 0.5,
    dependency_attempts: int = 3,
    dependency_retry_delay_s: float = 0.5,
) -> None:
    """Materialize the complete foreground read path before advertising chat."""

    set_ready = None

    deadline = asyncio.get_running_loop().time() + max(1.0, readiness_timeout_s)
    while not is_shutdown_requested():
        gate = ServiceContainer.get("inference_gate", default=None)
        candidate_set_ready = getattr(gate, "set_chat_dependencies_ready", None)
        if set_ready is None and callable(candidate_set_ready):
            set_ready = candidate_set_ready
            set_ready(False, blocker="chat_dependencies_warming")
        get_status = getattr(gate, "get_cortex_readiness_status", None)
        if not callable(get_status):
            get_status = getattr(gate, "get_conversation_status", None)
        if callable(get_status):
            try:
                lane = get_status()
            except _SERVER_BOUNDARY_ERRORS as exc:
                logger.debug("Chat dependency warmup readiness probe deferred: %s", exc)
            else:
                if isinstance(lane, dict) and lane.get("conversation_ready") is True:
                    break
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "Chat dependency warmup skipped because Cortex did not become "
                "conversation-ready within %.1fs.",
                readiness_timeout_s,
            )
            return
        await asyncio.sleep(max(0.05, poll_interval_s))
    if is_shutdown_requested():
        return

    from core.cognition.evidence_relevance import prewarm_evidence_relevance
    from core.consciousness.unified_self import get_unified_self
    from core.memory.embedding_runtime import prewarm_shared_embedding_runtime
    from core.memory.profile_manager import ProfileManager
    from core.self.self_condition import build_self_condition_projection
    from interface.chat_dependencies import materialize_foreground_chat_dependencies

    async def _complete_named_stage(
        stage: str,
        awaitables: tuple[Any, ...],
        names: tuple[str, ...],
    ) -> tuple[Any, ...]:
        results = await asyncio.gather(*awaitables, return_exceptions=True)
        failures = [
            (name, result)
            for name, result in zip(names, results, strict=True)
            if isinstance(result, _SERVER_BOUNDARY_ERRORS)
        ]
        if failures:
            name, failure = failures[0]
            raise RuntimeError(
                f"chat_dependency_stage_failed:{stage}:{name}:"
                f"{type(failure).__name__}:{failure}"
            ) from failure
        unexpected = [result for result in results if isinstance(result, BaseException)]
        if unexpected:
            raise unexpected[0]
        return tuple(results)

    async def _materialize_once() -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
        # Await every member even when one fails.  Abandoning a to_thread task
        # while retrying starts a second dependency transaction beside the
        # first and recreates the import/lifecycle race this owner prevents.
        # Handed to gather as coroutines. asyncio.gather schedules each one
        # itself, so wrapping them in create_task first added a raw task
        # creation for something that is awaited in the same expression — the
        # shape the ownership rule exists to catch is a task nobody awaits.
        _, _, snapshot, foreground_services = await _complete_named_stage(
            "readers",
            (
                ProfileManager.get_instance(),
                get_unified_self(),
                asyncio.to_thread(prewarm_shared_embedding_runtime),
                asyncio.to_thread(materialize_foreground_chat_dependencies),
            ),
            ("profile", "unified_self", "embedding", "foreground_services"),
        )
        projection, evidence_routing = await _complete_named_stage(
            "semantic_projection",
            (
                asyncio.to_thread(build_self_condition_projection),
                asyncio.to_thread(prewarm_evidence_relevance),
            ),
            ("self_condition", "evidence_relevance"),
        )
        if not getattr(projection, "evidence_id", ""):
            raise RuntimeError("self-condition warmup produced no evidence identity")
        return snapshot, foreground_services, projection, evidence_routing

    started = time.perf_counter()
    attempts = max(1, int(dependency_attempts))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            snapshot, foreground_services, projection, evidence_routing = (
                await _materialize_once()
            )
            break
        except _SERVER_BOUNDARY_ERRORS as exc:
            last_error = exc
            if attempt >= attempts:
                if callable(set_ready):
                    set_ready(False, blocker="chat_dependencies_failed")
                record_degradation(
                    "server.chat_dependency_warmup",
                    exc,
                    severity="degraded",
                    action=(
                        "kept conversation readiness blocked after bounded "
                        "dependency recovery was exhausted"
                    ),
                )
                logger.error(
                    "Chat dependency warmup failed after %d attempt(s): %s",
                    attempt,
                    exc,
                )
                return
            logger.warning(
                "Chat dependency warmup attempt %d/%d failed; retrying the "
                "completed transaction: %s",
                attempt,
                attempts,
                exc,
            )
            await asyncio.sleep(max(0.0, float(dependency_retry_delay_s)))
    else:  # pragma: no cover - loop exits through success or terminal return
        raise RuntimeError("chat dependency warmup exhausted without a verdict") from last_error

    if callable(set_ready):
        set_ready(True)
    logger.info(
        "Foreground chat dependencies prewarmed after Cortex readiness in %.2fs "
        "(embedding_dimensions=%s, leases=%s, skills=%s, expression_ms=%s, "
        "condition_evidence=%s, evidence_routing_ms=%s).",
        time.perf_counter() - started,
        snapshot.get("vector_dimensions"),
        snapshot.get("lease_count"),
        foreground_services.get("skill_count"),
        (foreground_services.get("expression_path") or {}).get("elapsed_ms"),
        str(getattr(projection, "evidence_id", ""))[:16],
        evidence_routing.get("elapsed_ms"),
    )


class _QueueHandler(logging.Handler):
    """Sends structured log records to the async broadcast queue.
    Implements a circular buffer for log_queue to prevent OOM/silencing.
    """

    _recursion_guard: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "_qh_recursion_guard", default=False
    )
    _dropped_count: int = 0
    _dropped_warn_count: int = 0
    _last_reported_warn_drops: int = 0
    _last_overflow_warning_at: float = 0.0

    @staticmethod
    def _proof_logging_active() -> bool:
        return any(os.environ.get(name) for name in ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS", "AURA_TESTING"))

    @classmethod
    def _should_buffer_record(cls, record: logging.LogRecord) -> bool:
        if cls._proof_logging_active() and record.levelno < logging.WARNING:
            return False
        return True

    @staticmethod
    def _entry_is_warning_or_worse(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        level = str(entry.get("level") or "").strip().lower()
        return level in {"warning", "error", "critical", "fatal"}

    # Tasks scheduled from the loop thread are held here so the loop keeps a
    # strong reference; asyncio only weakly references its tasks, and a
    # garbage-collected task drops the log line it was publishing.
    _inline_publishes: set[asyncio.Task] = set()

    @classmethod
    def _publish_without_blocking_the_loop(cls, publish_coro: Any, loop: Any) -> None:
        """Schedule a UI log publish, never blocking the event loop to do it.

        ``run_coroutine_threadsafe`` wakes its target loop by writing a byte to
        the loop's self-pipe. That is correct from a foreign thread, but when
        the caller IS the loop thread the write is both unnecessary and
        dangerous: the self-pipe socket buffer is finite, so a loop that has
        fallen behind cannot drain it, and the next ``csock.send`` blocks —
        on the loop thread, inside a logging call, deepening the very stall
        that caused the backlog.

        Measured live: an 8.4s event-loop stall whose captured loop stack was
        logging.warning -> emit -> run_coroutine_threadsafe ->
        call_soon_threadsafe -> _write_to_self -> csock.send. The runtime then
        latched DEGRADED for its whole 48-minute life on that one reading.
        """

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None and running is loop:
            task = _server_task_tracker.create_task(
                publish_coro,
                name="AuraServer.inline_log_publish",
            )
            cls._inline_publishes.add(task)
            task.add_done_callback(cls._inline_publishes.discard)
            return

        asyncio.run_coroutine_threadsafe(publish_coro, loop)


    def emit(self, record: logging.LogRecord) -> None:
        if self._recursion_guard.get():
            return
        if not self._should_buffer_record(record):
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
            full_message = getattr(record, "neural_full_message", None)
            if isinstance(full_message, str) and full_message.strip():
                log_entry["full_message"] = full_message
            query_chars = getattr(record, "query_chars", None)
            if isinstance(query_chars, int):
                log_entry["query_chars"] = query_chars

            queue_was_full = len(log_queue) >= log_queue.maxlen
            dropped_entry = log_queue[0] if queue_was_full and log_queue else None
            log_queue.append(log_entry)

            if queue_was_full:
                self._dropped_count += 1
                if self._entry_is_warning_or_worse(dropped_entry):
                    self._dropped_warn_count += 1
                    if record.created - self._last_overflow_warning_at >= 60.0:
                        rotated = self._dropped_warn_count - self._last_reported_warn_drops
                        logger.warning(
                            "UI log buffer at capacity: rotated out %d warning+ records "
                            "since last report (session totals: %d warning+, %d all levels); "
                            "newest records preserved.",
                            rotated,
                            self._dropped_warn_count,
                            self._dropped_count,
                        )
                        self._last_reported_warn_drops = self._dropped_warn_count
                        self._last_overflow_warning_at = record.created

            if main_loop is not None and not main_loop.is_closed() and main_loop.is_running():
                publish_coro = broadcast_bus.publish(log_entry)
                try:
                    self._publish_without_blocking_the_loop(publish_coro, main_loop)
                except _SERVER_BOUNDARY_ERRORS:
                    try:
                        publish_coro.close()
                    except _SERVER_BOUNDARY_ERRORS as close_exc:
                        print(f"CRITICAL LOG CLOSE FALLBACK: {close_exc}", file=sys.stderr)
                    raise

        except _SERVER_BOUNDARY_ERRORS:
            print(f"CRITICAL LOG FALLBACK: {record.levelname} - {record.getMessage()}", file=sys.stderr)
        finally:
            self._recursion_guard.reset(token)


# Attach queue handler to root logger
_qh = _QueueHandler()
_qh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_qh)


# ── Event bridge functions (extracted to interface/event_bridge.py) ──
from interface.auth import (
    _restore_owner_session_from_request,
    allowed_local_ui_origins,
    device_for_request,
    request_has_allowed_local_browser_origin,
    validate_runtime_security_request,
)
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
    messages_transport = None
    logger.info("Aura Server %s starting… (Lifespan Enter)", version_string("short"))

    # Initialize EventBus loop for threadsafe publication from background tasks
    from core.event_bus import get_event_bus
    get_event_bus().set_loop(main_loop)

    # 0. Global Registration
    #
    # LIVE DEFECT, 2026-08-09. A SIGTERM that arrives DURING boot does not stop
    # the API server's lifespan: the runtime logged "Shutdown requested" and
    # then started registering the entire service graph anyway. Every
    # registration was suppressed by the shutdown latch, the first
    # register-then-get pair raised, and the launch ended in a traceback and
    # "Application startup failed. Exiting." — for what was simply a quit.
    #
    # There is nothing to boot into. Yield so the ASGI app starts and the
    # teardown that is already in flight can finish cleanly.
    from core.runtime.shutdown_coordinator import is_shutdown_requested

    if is_shutdown_requested():
        logger.warning(
            "Lifespan entered while runtime shutdown is already requested; "
            "skipping subsystem boot."
        )
        yield
        return

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
    mapping_scheduled = mycelial.setup()
    logger.info(
        "📡 Mycelial infrastructure map: %s.",
        "scheduled"
        if mapping_scheduled
        else mycelial.get_infrastructure_report()["mapping_state"],
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
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional legacy local brain import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.latent.latent_core", optional=True)
            if not is_missing(mod):
                _LatentCore = mod.LatentCore
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional latent core import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.predictive.predictive_self_model", optional=True)
            if not is_missing(mod):
                _PredictiveSelf = mod.PredictiveSelfModel
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional predictive self model import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.tts_stream", optional=True)
            if not is_missing(mod):
                _FastMouth = mod.FastMouth
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional TTS stream import failed; continuing degraded: %s", _exc)

        try:
            mod = await async_safe_import("core.senses.screen_vision", optional=True)
            if not is_missing(mod):
                _LocalVision = mod.LocalVision
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional local vision import failed; continuing degraded: %s", _exc)

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
                except _SERVER_BOUNDARY_ERRORS as ve_err:
                    record_degradation('server', ve_err)
                    logger.warning("⚠️ Voice engine health check failed: %s — disabling voice.", ve_err)
                    _voice_engine_fn = None
        except _SERVER_BOUNDARY_ERRORS as _exc:
            record_degradation('server', _exc)
            logger.warning("Optional voice engine import failed; continuing degraded: %s", _exc)
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

    from core.resilience.degradation_repair import get_degradation_repair_router
    from interface.routes import system as system_routes

    get_degradation_repair_router().bind_owner_loop(
        asyncio.get_running_loop(),
        replace=True,
    )
    system_routes.start_health_read_model()
    if not is_gui_proxy:
        _spawn_server_task(
            _prewarm_chat_dependencies_after_cortex_ready(),
            name="chat_dependency_prewarm",
        )
    from interface.routes.chat import start_chat_turn_memory_log_worker

    if not start_chat_turn_memory_log_worker():
        logger.warning(
            "Durable chat memory outbox worker could not start; pending work remains on disk."
        )

    # Private Messages is another presentation surface over the canonical chat
    # lane. Its Keychain and SQLite work stays off the event loop, and missing
    # TCC permissions degrade only this transport rather than runtime boot.
    if not is_gui_proxy:
        try:
            from core.communication.messages_journal import MessagesDeliveryJournal
            from core.communication.messages_transport import MessagesTransport
            from interface.routes.chat import run_governed_surface_chat_turn

            messages_journal = await asyncio.to_thread(MessagesDeliveryJournal)
            messages_transport = MessagesTransport(
                chat_turn=run_governed_surface_chat_turn,
                journal=messages_journal,
            )
            ServiceContainer.register_instance(
                "messages_transport",
                messages_transport,
                required=False,
                owner="interface.server",
                registered_by="interface.server.lifespan",
                required_for="private_messages_surface",
                failure_policy="degrade",
            )
            await messages_transport.start(task_factory=_spawn_server_task)
            logger.info("Private Messages transport initialized on the canonical chat lane.")
        except _SERVER_BOUNDARY_ERRORS as exc:
            record_degradation(
                "server.messages_transport",
                exc,
                severity="warning",
                action="kept Aura online while private Messages remains explicitly unavailable",
                enforce_failure_policy=False,
            )
            logger.warning(
                "Private Messages transport unavailable: %s",
                type(exc).__name__,
            )
    logger.info("Aura Server online — %s", version_string("full"))
    try:
        yield  # ← app is live here
    finally:
        # ── Shutdown ──
        logger.info("Aura Server shutting down…")
        system_routes.stop_health_read_model()
        if messages_transport is not None:
            try:
                await messages_transport.stop()
            except _SERVER_BOUNDARY_ERRORS as exc:
                record_degradation(
                    "server.messages_transport",
                    exc,
                    severity="warning",
                    action="continued bounded server shutdown after Messages transport stop failed",
                    enforce_failure_policy=False,
                )
        await _server_task_tracker.shutdown(timeout=2.0)
        _event_bridge_task = None
        main_loop = None


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Aura Luna Agent",
    description="Secure interface for the Aura Luna autonomous engine.",
    version=VERSION,
    lifespan=lifespan,
)

# 0.1 Prometheus instrumentation. Kept native to avoid Starlette-version
# coupling in third-party middleware.
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    _HTTP_REQUESTS_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        status = int(response.status_code)
        return response
    finally:
        path = _route_template(request)
        method = request.method
        elapsed = max(0.0, time.perf_counter() - start)
        _HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed)
        _HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()
        _HTTP_REQUESTS_IN_PROGRESS.dec()


@app.get("/metrics", include_in_schema=False)
@app.get("/metrics/prometheus", include_in_schema=False)
async def prometheus_metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
_RUNTIME_SHELL_RECOVERY_SCRIPT = b"""(function () {
    'use strict';
    var next = new URL(window.location.href);
    next.searchParams.delete('_aura_runtime');
    var retire = Promise.resolve();
    if ('serviceWorker' in navigator && typeof navigator.serviceWorker.getRegistrations === 'function') {
        retire = navigator.serviceWorker.getRegistrations().then(function (registrations) {
            return Promise.all((registrations || []).map(function (registration) {
                var workers = [registration.active, registration.waiting, registration.installing];
                var ownsAuraShell = workers.some(function (worker) {
                    try {
                        var script = new URL(String(worker && worker.scriptURL || ''));
                        return script.origin === window.location.origin
                            && script.pathname === '/static/service-worker.js';
                    } catch (_err) {
                        return false;
                    }
                });
                if (!ownsAuraShell) return false;
                workers.forEach(function (worker) {
                    try { worker && worker.postMessage({ type: 'AURA_RETIRE_RUNTIME_SHELL' }); }
                    catch (_err) {}
                });
                try { return registration.unregister(); }
                catch (_err) { return false; }
            }));
        }).catch(function () {});
    }
    retire.then(function () {
        if (typeof caches === 'undefined' || typeof caches.keys !== 'function') return;
        return caches.keys().then(function (keys) {
            return Promise.all(keys.filter(function (key) {
                return String(key).indexOf('aura-runtime-shell-') === 0;
            }).map(function (key) { return caches.delete(key); }));
        });
    }).catch(function () {}).then(function () {
        window.location.replace(next.toString());
    });
}());
"""
_RUNTIME_REVISION_SHELL_PATHS = frozenset(
    runtime_shell_request_path(relative) for relative in RUNTIME_SHELL_ASSETS
)
_RUNTIME_REVISION_ADDRESSED_PATHS = frozenset(
    runtime_shell_request_path(relative) for relative in RUNTIME_SHELL_PUBLIC_ASSETS
)
_RUNTIME_REVISION_NO_STORE_PATHS = frozenset(
    {"/", *(_RUNTIME_REVISION_SHELL_PATHS - _RUNTIME_REVISION_ADDRESSED_PATHS)}
)
def _cache_policy_for_path(
    path: str,
    *,
    revision_addressed: bool = False,
) -> dict[str, str] | None:
    normalized = str(path or "")
    if (
        normalized in _RUNTIME_REVISION_NO_STORE_PATHS
        or normalized.endswith("/index.html")
    ):
        if revision_addressed and normalized in _RUNTIME_REVISION_NO_STORE_PATHS:
            return {"Cache-Control": "private, max-age=31536000, immutable"}
        return dict(NO_CACHE_HEADERS)
    if normalized in _RUNTIME_REVISION_ADDRESSED_PATHS:
        if revision_addressed:
            return {"Cache-Control": "public, max-age=31536000, immutable"}
        return dict(NO_CACHE_HEADERS)
    if normalized.startswith("/static/"):
        return dict(NO_CACHE_HEADERS)
    if normalized.startswith("/data"):
        return dict(NO_CACHE_HEADERS)
    return None


@app.middleware("http")
async def serve_immutable_runtime_shell(request: Request, call_next):
    """Serve revision-addressed shell requests from verified frozen bytes only."""

    from core.runtime.runtime_shell_snapshot import (
        runtime_shell_request_path,
        runtime_shell_snapshot_asset,
    )

    path = str(request.url.path or "")
    if not runtime_shell_request_path(path):
        return await call_next(request)
    if path == "/static/service-worker.js":
        # The one asset that must never be frozen. A registered worker's script
        # URL carries the revision it was registered under, and the browser
        # revalidates exactly that URL to discover a newer worker. Answering it
        # from that revision's snapshot means the update check can only ever see
        # the bytes it already has, so the registration — and every asset it
        # serves from its own cache — becomes permanent.
        #
        # Measured live 2026-08-03: a desktop window was controlled by a worker
        # from a revision hours old, across four runtime restarts and three
        # revision changes, and could not be rescued by reloading. Freezing the
        # update channel leaves no channel.
        #
        # The worker still binds itself to the revision in its own URL, so the
        # immutability of everything it serves is unchanged; only its own code
        # is allowed to move forward.
        return await call_next(request)
    directly_addressed = "_aura_runtime" in request.query_params
    revision = str(request.query_params.get("_aura_runtime") or "").strip().lower()
    if not revision:
        referer = str(request.headers.get("referer") or "")
        try:
            referer_url = urlsplit(referer)
        except ValueError:
            referer_url = None
        if (
            referer_url is not None
            and referer_url.scheme.lower() == request.url.scheme.lower()
            and referer_url.netloc.lower() == request.url.netloc.lower()
        ):
            revision = str(
                (parse_qs(referer_url.query).get("_aura_runtime") or [""])[0]
            ).strip().lower()
    if not revision:
        return await call_next(request)
    try:
        validate_runtime_security_request(request)
    except HTTPException as exc:
        return Response(
            status_code=exc.status_code,
            content=str(exc.detail),
            headers=NO_CACHE_HEADERS,
        )
    if len(revision) != 64 or any(character not in "0123456789abcdef" for character in revision):
        return Response(
            status_code=400,
            content="invalid runtime revision",
            headers=NO_CACHE_HEADERS,
        )
    content = runtime_shell_snapshot_asset(revision, path)
    if content is None:
        if request.method == "GET" and path in {"/", "/static/index.html"}:
            # A native window can outlive the in-process snapshot that supplied
            # its revision (runtime restart, or enough source revisions to evict
            # the old entry). Returning the 409 body as the document strands the
            # whole WKWebView on raw diagnostic text, where no shell JavaScript
            # exists to retire the stale worker. Recover only document
            # navigations through the unaddressed bootstrap. Revisioned
            # subresources remain fail-closed below so one page can never mix
            # assets from different attested shells.
            recovery_query = urlencode(
                [
                    (key, value)
                    for key, value in request.query_params.multi_items()
                    if key != "_aura_runtime"
                ],
                doseq=True,
            )
            recovery_url = request.url.replace(query=recovery_query)
            if path == "/static/index.html":
                recovery_url = recovery_url.replace(path="/")
            return RedirectResponse(
                url=str(recovery_url),
                status_code=307,
                headers={
                    **NO_CACHE_HEADERS,
                    "X-Aura-Runtime-Recovery": "revision_snapshot_unavailable",
                },
            )
        if request.method == "GET" and path == "/static/aura.js":
            # The application entrypoint is the only code able to retire a
            # stale service worker. Returning 409 here makes an older worker
            # use its cached aura.js, which permanently prevents that repair
            # code from loading. Serve a non-application recovery entrypoint:
            # it discards the obsolete shell and navigates the whole document
            # back through the unaddressed bootstrap, so revision bytes are
            # never mixed inside one running page.
            return Response(
                status_code=200,
                content=_RUNTIME_SHELL_RECOVERY_SCRIPT,
                media_type="text/javascript",
                headers={
                    **NO_CACHE_HEADERS,
                    "X-Aura-Runtime-Recovery": "retire_unknown_shell_revision",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        return Response(
            status_code=409,
            content="runtime revision is not available from verified immutable storage",
            headers=NO_CACHE_HEADERS,
        )
    media_path = "/static/index.html" if path == "/" else path
    media_type = mimetypes.guess_type(media_path)[0] or "application/octet-stream"
    headers = {
        "Cache-Control": (
            "private, max-age=31536000, immutable"
            if directly_addressed
            else "no-store, no-cache, must-revalidate, max-age=0"
        ),
        "X-Aura-Runtime-Revision": revision,
        "X-Content-Type-Options": "nosniff",
    }
    if path == "/static/service-worker.js":
        headers["Service-Worker-Allowed"] = "/"
    return Response(content=content, media_type=media_type, headers=headers)


# Mount static files for uploads and generated media
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR, html=False), name="uploads")
app.mount("/data/generated_images", StaticFiles(directory=GEN_IMAGES_DIR, html=False), name="generated_images")


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    query_params = getattr(request, "query_params", {})
    try:
        revision = str(query_params.get("_aura_runtime") or "")
    except (AttributeError, TypeError, ValueError):
        revision = ""
    response_revision = str(
        getattr(response, "headers", {}).get("X-Aura-Runtime-Revision", "")
    )
    revision_addressed = bool(
        len(revision) == 64
        and all(character in "0123456789abcdef" for character in revision)
        and int(getattr(response, "status_code", 0) or 0) == 200
        and hmac.compare_digest(response_revision, revision)
    )
    policy = _cache_policy_for_path(
        request.url.path,
        revision_addressed=revision_addressed,
    )
    if policy and hasattr(response, "headers"):
        for key, value in policy.items():
            cast(Response, response).headers[key] = value
        if request.url.path == "/static/service-worker.js" and int(
            getattr(response, "status_code", 0) or 0
        ) == 200:
            cast(Response, response).headers["Service-Worker-Allowed"] = "/"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_local_ui_origins(),
    allow_methods=["GET", "HEAD", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-Api-Token",
        "X-Idempotency-Key",
        "Authorization",
        "X-Aura-Surface",
        "X-Aura-Desktop-Request",
        "X-Aura-Require-CognitiveEngine",
    ],
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


# ── Reliability: HTTP trace root spans ─────────────────────────
# Every /api request gets a (sampled) root span; downstream spans —
# inference, Will refusals — nest under it via contextvars, so a slow or
# failing turn reads as one connected trace at /api/diagnostics/reliability/traces.

@app.middleware("http")
async def trace_root_middleware(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    try:
        from core.observability.tracing import get_tracer
        tracer = get_tracer()
    except (ImportError, AttributeError, RuntimeError):
        return await call_next(request)
    if not tracer.enabled:
        return await call_next(request)
    with tracer.span(
        "http.request",
        attributes={
            "http.method": request.method,
            "http.path": request.url.path,
            "request.id": getattr(request.state, "request_id", ""),
        },
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.status_code", response.status_code)
        if response.status_code >= 500:
            span.set_status("ERROR", f"HTTP {response.status_code}")
        return response


# ── Magnum Opus: Global Exception Handler ─────────────────────

from datetime import UTC, datetime


def _phenomenal_error_status(envelope) -> int:
    """Map graceful error envelopes to truthful HTTP status codes."""
    state = str(getattr(envelope, "phenomenal_state", "") or "")
    if state == "permission_denied":
        return 403
    if state == "disk_pressure":
        return 507
    if state in {"cognitive_fog", "metabolic_strain", "model_unavailable", "network_offline"}:
        return 503
    return 500


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
        http_status = _phenomenal_error_status(envelope)
        return JSONResponse(
            status_code=http_status,
            content={
                "ok": False,
                "status": "phenomenal",
                "http_status": http_status,
                "envelope": envelope.to_dict(),
                "user_message": envelope.user_message,
                "request_id": request_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    except _SERVER_BOUNDARY_ERRORS as inner:
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
from interface.routes import allostasis as allostasis_routes
from interface.routes import ambient as ambient_routes
from interface.routes import chat as chat_routes
from interface.routes import dashboard as dashboard_routes
from interface.routes import devices as devices_routes
from interface.routes import engineering as engineering_routes
from interface.routes import inner_state as inner_state_routes
from interface.routes import interaction_signals as interaction_signal_routes
from interface.routes import media as media_routes
from interface.routes import memory as memory_routes
from interface.routes import mission_control as mission_control_routes
from interface.routes import multimodal as multimodal_routes
from interface.routes import performance as performance_routes
from interface.routes import privacy as privacy_routes
from interface.routes import reality_reach as reality_reach_routes
from interface.routes import rpc as rpc_routes
from interface.routes import settings as settings_routes
from interface.routes import subsystems as subsystem_routes
from interface.routes import system as system_routes
from interface.routes import worlds as worlds_routes

checkpoint_service = CheckpointService()

app.include_router(system_health_router, prefix="/api/health", tags=["health"])
app.include_router(memory_ui.router, prefix="/memory", tags=["memory"])
app.include_router(chat_routes.router, prefix="/api", tags=["chat"])
app.include_router(devices_routes.router, prefix="/api", tags=["devices"])
app.include_router(worlds_routes.router, prefix="/api", tags=["worlds"])
app.include_router(engineering_routes.router, prefix="/api", tags=["engineering"])
app.include_router(system_routes.router, prefix="/api", tags=["system"])
app.include_router(subsystem_routes.router, prefix="/api", tags=["subsystems"])
app.include_router(memory_routes.router, prefix="/api", tags=["memory-api"])
app.include_router(interaction_signal_routes.router, prefix="/api", tags=["interaction-signals"])
app.include_router(privacy_routes.router, prefix="/api", tags=["privacy"])
# The bubble's surface. Routes are mounted with their own absolute
# paths (/api/ambient/...) so the companion endpoints stay findable as
# a group rather than being scattered under the shared /api prefix.
app.include_router(ambient_routes.router, tags=["ambient"])
app.include_router(reality_reach_routes.router, prefix="/api", tags=["reality-reach"])
app.include_router(rpc_routes.router, prefix="/rpc", tags=["rpc"])
app.include_router(inner_state_routes.router, tags=["proof-surface"])
app.include_router(allostasis_routes.router, prefix="/api", tags=["allostasis"])
app.include_router(dashboard_routes.router, prefix="/api", tags=["dashboard"])
app.include_router(dashboard_routes.trace_router, prefix="/api", tags=["trace"])
app.include_router(settings_routes.router, prefix="/api", tags=["settings"])
app.include_router(multimodal_routes.router, prefix="/api", tags=["multimodal"])
app.include_router(media_routes.router, prefix="/api", tags=["media"])
app.include_router(performance_routes.router, prefix="/api", tags=["performance"])
app.include_router(mission_control_routes.router, prefix="/api", tags=["mission_control"])

# Full-duplex voice (/ws/voice). Imported late and defensively: the voice
# lane pulls in ONNX, Silero and the ASR stack, and a runtime that cannot
# load them must still serve text chat rather than failing to boot.
try:
    from interface.routes import voice_duplex as voice_duplex_routes

    app.include_router(voice_duplex_routes.router, tags=["voice"])
except (ImportError, OSError, RuntimeError, AttributeError) as _voice_exc:
    record_degradation(
        "server.voice_duplex",
        _voice_exc,
        action="text chat stayed up; the full-duplex voice lane is unavailable",
    )
    logger.error("Full-duplex voice lane unavailable: %s", _voice_exc)

# ── Reliability diagnostics ────────────────────────────────────────
# Live at /api/diagnostics/reliability — exposes fault taxonomy, SLO burn
# rates, FMEA coverage, contract violations, and tracing statistics.
try:
    from core.resilience.diagnostics_dashboard import create_diagnostics_router
    app.include_router(create_diagnostics_router(), prefix="/api/diagnostics", tags=["reliability-diagnostics"])
except (ImportError, RuntimeError) as _reliability_exc:
    import logging as _logging
    _logging.getLogger("Aura.Server").debug("Reliability diagnostics unavailable: %s", _reliability_exc)

_system_collect_liquid_state_payload = system_routes._collect_liquid_state_payload


def _collect_conversation_lane_status() -> dict[str, Any]:
    from interface.routes import chat_preflight

    return chat_preflight._collect_conversation_lane_status()


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
_desktop_required_cognitive_budget = chat_routes._desktop_required_cognitive_budget
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
        while not is_shutdown_requested():
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
            except _SERVER_BOUNDARY_ERRORS as e:
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
    host = request.client.host if request.client else "unknown"
    if host not in ("127.0.0.1", "::1", "localhost") and device_for_request(request) is None:
        # Unpaired LAN visitor: the shell's assets would 401 anyway, so
        # route them straight into the pairing ceremony.
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/pair", status_code=307)
    ui = LEGACY_UI_INDEX if LEGACY_UI_INDEX.exists() else (SHELL_DIST_DIR / "index.html")
    if not ui.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(str(ui), headers=NO_CACHE_HEADERS)


@app.get("/worlds", include_in_schema=False)
async def serve_worlds(request: Request):
    """WebGL viewer for Aura's persistent physics worlds. Read-only for
    paired devices; stepping is owner-only at the API layer."""
    _require_internal(request)
    host = request.client.host if request.client else "unknown"
    if host not in ("127.0.0.1", "::1", "localhost") and device_for_request(request) is None:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/pair", status_code=307)
    p = STATIC_DIR / "worlds.html"
    if not p.exists():
        return ORJSONResponse({"error": "worlds viewer not found"}, status_code=404)
    return FileResponse(str(p), headers=NO_CACHE_HEADERS)


@app.get("/pair", include_in_schema=False)
async def serve_pair(request: Request):
    """Device pairing page: shows codes on the desktop, accepts them on
    the phone. Self-contained HTML — no /static dependencies, because an
    unpaired device cannot fetch those yet."""
    _require_internal(request)
    p = STATIC_DIR / "pair.html"
    if not p.exists():
        return ORJSONResponse({"error": "pairing UI not found"}, status_code=404)
    return FileResponse(str(p), headers=NO_CACHE_HEADERS)


@app.get("/telemetry", include_in_schema=False)
async def serve_telemetry(request: Request):
    _require_internal(request)
    p = STATIC_DIR / "telemetry.html"
    return FileResponse(str(p), headers=NO_CACHE_HEADERS) if p.exists() else ORJSONResponse({"error": "not found"}, status_code=404)


@app.get("/mind", include_in_schema=False)
async def serve_mind(request: Request):
    """Real-time mind visualizer (#39) — renders the live /api/inner-state surface."""
    _require_internal(request)
    p = STATIC_DIR / "mind.html"
    return FileResponse(str(p), headers=NO_CACHE_HEADERS) if p.exists() else ORJSONResponse({"error": "not found"}, status_code=404)


@app.get("/activity", include_in_schema=False)
async def serve_activity(request: Request):
    """Activity / receipts view (#35) — plain-language record of Aura's self-directed actions."""
    _require_internal(request)
    p = STATIC_DIR / "activity.html"
    return FileResponse(str(p), headers=NO_CACHE_HEADERS) if p.exists() else ORJSONResponse({"error": "not found"}, status_code=404)


@app.get("/controls", include_in_schema=False)
async def serve_controls(request: Request):
    """Controls panel (#35) — plain-language safe-mode, autonomy, and sensor switches."""
    _require_internal(request)
    p = STATIC_DIR / "controls.html"
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

def _live_device_scopes(device_id: str) -> tuple[str, ...]:
    """Current scopes from the live registry — grants and revocations
    take effect per-frame, not per-connection."""
    try:
        from core.security.device_pairing import get_device_registry

        device = get_device_registry().devices.get(str(device_id))
        if device is None or device.revoked:
            return ()
        return tuple(device.scopes)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("server.ws_device_scopes", exc)
        return ()


def _verify_ws_device_token(token: str):
    """Resolve a paired device from an explicit WS auth message token."""
    try:
        from core.security.device_pairing import get_device_registry

        return get_device_registry().verify_token(token)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        record_degradation("server.ws_device_auth", exc)
        return None


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Accept the transport so credentials can be exchanged, but do not
    # register it with the global broadcast manager until authentication has
    # succeeded. Pre-auth registration leaked queued runtime events during the
    # five-second auth window.
    await ws.accept()

    expected = str(config.api_token or "")
    host = ws.client.host if ws.client else "unknown"
    is_local = host in ("127.0.0.1", "::1", "localhost")
    local_browser_origin_allowed = request_has_allowed_local_browser_origin(ws)

    # No configured token means no way to authenticate a remote peer:
    # only same-host UI connections may proceed. WS handshakes bypass the
    # HTTP middleware, so this must fail closed here.
    authenticated = is_local and local_browser_origin_allowed
    auth_timeout = 5.0

    # Paired LAN devices authenticate via their session cookie on the
    # handshake itself (browsers attach it automatically), scoped by
    # interface/auth.py to the conversation surface — /ws is in scope.
    device_session = None
    explicit_device_token: str | None = None
    connection_message_tasks: set[asyncio.Task[Any]] = set()
    if not authenticated:
        device_session = device_for_request(ws)
        if device_session is not None:
            authenticated = True

    try:
        if device_session is not None:
            await ws.send_text(json.dumps({
                "type": "auth_success",
                "note": "paired_device",
            }))
        if not authenticated:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=auth_timeout)
                data = json.loads(raw)
                supplied_ws_token = str(data.get("token", "") or "")
                if data.get("type") == "auth" and supplied_ws_token.startswith("adt1."):
                    device_session = _verify_ws_device_token(supplied_ws_token)
                    if device_session is not None:
                        explicit_device_token = supplied_ws_token
                        authenticated = True
                        await ws.send_text(json.dumps({
                            "type": "auth_success",
                            "note": "paired_device",
                        }))
                    else:
                        await ws.send_text(json.dumps({"type": "error", "message": "Unauthorized"}))
                        await ws.close(code=4001, reason="Unauthorized")
                        return
                elif data.get("type") == "auth" and expected and hmac.compare_digest(supplied_ws_token, expected):
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
        elif is_local and expected and local_browser_origin_allowed:
            await ws.send_text(json.dumps({"type": "auth_success", "note": "local_trust"}))

        await ws_manager.connect(
            ws,
            accepted=True,
            scope="conversation" if device_session is not None else "owner",
        )

        while not is_shutdown_requested():
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=20.0)
            except TimeoutError:
                if device_session is None:
                    continue
                refreshed = (
                    _verify_ws_device_token(explicit_device_token)
                    if explicit_device_token
                    else device_for_request(ws)
                )
                if (
                    refreshed is not None
                    and refreshed.device_id == device_session.device_id
                ):
                    device_session = refreshed
                    continue
                await ws.send_text(json.dumps({
                    "type": "error",
                    "status": "paired_device_session_revoked",
                    "message": "This paired-device session is no longer authorized.",
                }))
                await ws.close(code=4003, reason="Paired device session revoked")
                return

            if device_session is not None:
                refreshed = (
                    _verify_ws_device_token(explicit_device_token)
                    if explicit_device_token
                    else device_for_request(ws)
                )
                if (
                    refreshed is None
                    or refreshed.device_id != device_session.device_id
                ):
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "status": "paired_device_session_revoked",
                        "message": "This paired-device session is no longer authorized.",
                    }))
                    await ws.close(code=4003, reason="Paired device session revoked")
                    return
                device_session = refreshed

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
                        if any(not task.done() for task in connection_message_tasks):
                            await ws.send_text(json.dumps({
                                "type": "aura_message",
                                "content": (
                                    "The previous WebSocket turn is still running. "
                                    "Wait for its terminal reply before sending another."
                                ),
                                "status": "conversation_turn_in_progress",
                            }))
                            continue
                        logger.debug("WS: Received user_message: %s", content[:100])
                        _notify_user_spoke(content)

                        async def _handle_ws_message(
                            ws_ref,
                            user_content: str,
                            paired_session,
                        ):
                            """Process user message and send response back via WebSocket."""
                            try:
                                from interface.routes import chat_preflight

                                lane_snapshot = (
                                    chat_preflight._collect_conversation_lane_status()
                                )
                                reply = await chat_routes._run_cognitive_engine_chat_turn(
                                    user_content,
                                    visible_user_message=user_content,
                                    session_id=(
                                        f"paired-device:{paired_session.device_id}"
                                        if paired_session is not None
                                        else ""
                                    ),
                                    origin=(
                                        "user"
                                        if paired_session is not None
                                        else "desktop-ui"
                                    ),
                                    timeout_s=300.0,
                                    lane=lane_snapshot,
                                    source=(
                                        "paired_device_websocket"
                                        if paired_session is not None
                                        else "desktop_websocket"
                                    ),
                                    require_engine=True,
                                    conversation_only_surface=paired_session is not None,
                                )
                                
                                if not reply:
                                    logger.warning(
                                        "WS: required desktop CognitiveEngine produced no clean reply; refusing legacy fallback."
                                    )
                                    await ws_ref.send_text(json.dumps({
                                        "type": "aura_message",
                                        "content": (
                                            "The desktop WebSocket chat path requires CognitiveEngine, and it did not "
                                            "return a clean reply. I refused the legacy fallback so this surface cannot "
                                            "display an incoherent answer."
                                        ),
                                        "status": "desktop_cognitive_engine_unavailable",
                                    }))
                                    return

                                _authored_reply = str(reply or "")
                                reply = await chat_routes._stabilize_user_facing_reply(
                                    user_content,
                                    reply,
                                )
                                reply = chat_routes._strip_user_visible_context_leaks(reply)
                                if not str(reply or "").strip() and _authored_reply.strip():
                                    # Never hand back an ellipsis for an answer
                                    # she actually wrote.
                                    #
                                    # LIVE 2026-08-17: "what's on my screen
                                    # right now?" was served as a bare "…" with
                                    # a 172-character cortex reply in hand and
                                    # the screen reading delivered. Whatever
                                    # emptied it — stabilisation or leak
                                    # stripping — the person got the shape of
                                    # an answer instead of the answer.
                                    salvaged = chat_routes._strip_user_visible_context_leaks(
                                        _authored_reply
                                    )
                                    if salvaged.strip():
                                        logger.warning(
                                            "Reply post-processing emptied a %d-char authored "
                                            "answer; served the authored text instead.",
                                            len(_authored_reply),
                                        )
                                        reply = salvaged
                                reply = str(reply or "").strip() or "…"
                                reply_status = (
                                    chat_routes._desktop_required_bounded_reply_status(
                                        user_content,
                                        reply,
                                        lane_snapshot,
                                    )
                                    or "cognitive_engine"
                                )
                                desktop_result = None
                                if paired_session is None:
                                    desktop_result = await chat_routes._execute_desktop_objective_from_chat(
                                        user_content,
                                        cognitive_reply=reply,
                                    )
                                elif chat_routes._looks_like_desktop_objective(user_content):
                                    await ws_ref.send_text(json.dumps({
                                        "type": "aura_message",
                                        "content": (
                                            "This paired device is scoped to conversation and read-only world viewing. "
                                            "Desktop, file, tool, and control actions require the owner surface."
                                        ),
                                        "status": "paired_device_action_scope_denied",
                                    }))
                                    return
                                if isinstance(desktop_result, dict):
                                    await ws_ref.send_text(json.dumps({
                                        "type": "aura_message",
                                        "content": chat_routes._strip_user_visible_context_leaks(
                                            desktop_result.get("response") or reply
                                        ) or "…",
                                        "status": desktop_result.get("status"),
                                        "data": {
                                            "desktop_result": desktop_result.get("result"),
                                        },
                                        "conversation_lane": {
                                            "source": "desktop_websocket",
                                            "governed_action_result": bool(desktop_result.get("ok")),
                                            "governed_action_status": desktop_result.get("status"),
                                        },
                                    }, default=str))
                                    return
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": reply,
                                    "status": reply_status,
                                    "conversation_lane": {
                                        "source": "desktop_websocket",
                                        "governed_action_result": False,
                                    },
                                }))
                            except TimeoutError:
                                logger.error("WS: live CognitiveEngine processing timed out")
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": "The live reasoning lane exceeded its timeout. I logged the timeout and preserved this turn instead of fabricating a recovered answer.",
                                }))
                            except _SERVER_BOUNDARY_ERRORS as e:
                                record_degradation('server', e)
                                logger.error("WS: Message handling failed: %s (%s)", type(e).__name__, e, exc_info=True)
                                await ws_ref.send_text(json.dumps({
                                    "type": "aura_message",
                                    "content": "The live message handler failed before a coherent answer formed. I logged the failure with the current turn context.",
                                }))

                        message_task = _spawn_server_bounded_task(
                            _handle_ws_message(ws, content, device_session),
                            name="server.ws.handle_message",
                        )
                        connection_message_tasks.add(message_task)
                        message_task.add_done_callback(connection_message_tasks.discard)
                elif msg_type == "ping":
                    await ws.send_text(
                        json.dumps(ws_manager.heartbeat_payload(ws, "pong"))
                    )

            elif "bytes" in msg:
                if device_session is not None and "voice" not in _live_device_scopes(
                    device_session.device_id
                ):
                    # Deny-by-default stands: only an explicit owner grant
                    # (POST /api/devices/grant-scope {scope: "voice"})
                    # opens the microphone lane for a paired device.
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "status": "paired_device_voice_scope_denied",
                        "message": (
                            "Voice streaming is not enabled for this paired device. "
                            "Ask the owner to grant the voice scope, or use the "
                            "owner surface for microphone input."
                        ),
                    }))
                    continue
                await ws.send_text(json.dumps({
                    "type": "error",
                    "status": "legacy_voice_transport_retired",
                    "message": "Use the authenticated duplex voice endpoint at /ws/voice.",
                }))

    except WebSocketDisconnect as _exc:
        logger.debug("Suppressed WebSocketDisconnect: %s", _exc)
    except _SERVER_BOUNDARY_ERRORS as exc:
        record_degradation('server', exc)
        logger.debug("WS error: %s", exc)
    finally:
        pending_message_tasks = [
            task for task in connection_message_tasks if not task.done()
        ]
        for task in pending_message_tasks:
            task.cancel()
        if pending_message_tasks:
            await asyncio.gather(*pending_message_tasks, return_exceptions=True)
        await ws_manager.disconnect(ws)


# ── SPA Catch-all — v6.0 Traverse Hardened ────────────────────

@app.get("/{path:path}", include_in_schema=False)
async def spa_catchall(path: str, request: Request):
    """Secure catch-all to support SPA routing and static resolution with traversal protection."""
    _require_internal(request)

    if path.startswith("api/"):
        # An unmatched API path must be a machine-readable 404, never the
        # SPA shell. Serving index.html at 200 here made a mistyped health
        # endpoint look 'up' to automation while returning HTML — a soak
        # driver polled one for 15 minutes on 2026-07-10 believing the
        # runtime was never ready.
        return JSONResponse(
            {"error": "not_found", "path": f"/{path}", "detail": "unknown API route"},
            status_code=404,
        )

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
    from core.observability.logging_config import setup_logging as _sl
    _sl(log_dir=config.paths.log_dir)

    host = "127.0.0.1" if config.security.internal_only_mode else "0.0.0.0"
    logger.info("Binding to %s:8000", host)

    tls_kwargs: dict = {}
    try:
        from core.security.tls_local import ensure_local_certificate, tls_enabled

        if tls_enabled():
            certificate = ensure_local_certificate()
            if certificate is not None:
                cert_path, key_path = certificate
                tls_kwargs = {
                    "ssl_certfile": str(cert_path),
                    "ssl_keyfile": str(key_path),
                }
                logger.info(
                    "Serving HTTPS with the local certificate — phones get a "
                    "secure context (accept the cert once) and the voice lane "
                    "becomes possible for scope-granted devices."
                )
            else:
                logger.error("AURA_ENABLE_TLS=1 but no certificate; staying on HTTP")
    except (ImportError, AttributeError, RuntimeError, OSError) as _tls_exc:
        record_degradation("server.tls", _tls_exc)

    uvicorn.run(
        "interface.server:app",
        host=host,
        port=8000,
        reload=False,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=10,
        **tls_kwargs,
    )


if __name__ == "__main__":
    main()
