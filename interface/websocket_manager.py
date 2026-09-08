"""interface/websocket_manager.py
─────────────────────────────────
Extracted from server.py — WebSocket connection management,
broadcast infrastructure, and UI event normalization.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import WebSocket

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.task_ownership import create_tracked_task

try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    class ConnectionClosed(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Compatibility fallback for environments without websockets."""

from fastapi import WebSocketDisconnect

from core.health.conversation_lane import (
    conversation_lane_is_busy,
)

logger = logging.getLogger("Aura.Server.WebSocket")

_QUEUE_REPAIR_ERRORS = (
    asyncio.QueueEmpty,
    asyncio.QueueFull,
    RuntimeError,
    TypeError,
    ValueError,
)
_WEBSOCKET_DELIVERY_ERRORS = (
    WebSocketDisconnect,
    ConnectionClosed,
    RuntimeError,
    OSError,
    TypeError,
    ValueError,
)
_WEBSOCKET_HEARTBEAT_ERRORS = (asyncio.TimeoutError,) + _WEBSOCKET_DELIVERY_ERRORS

type BroadcastItem = tuple[int, float, Any]
type BroadcastQueue = asyncio.PriorityQueue[BroadcastItem]
type ClientItem = tuple[int, float, str]
type ClientQueue = asyncio.PriorityQueue[ClientItem]
type TaskSpawner = Callable[..., asyncio.Task[Any]]


def _spawn_websocket_task(
    awaitable: Awaitable[Any],
    *,
    name: str | None = None,
) -> asyncio.Task[Any]:
    return cast(
        asyncio.Task[Any],
        create_tracked_task(
            awaitable,
            name=name,
            owner="websocket_manager",
        ),
    )


def _env_positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(minimum, value)


def runtime_heartbeat_payload(kind: str = "heartbeat") -> dict[str, Any]:
    """Project the same versioned readiness evidence used by HTTP health."""
    try:
        from interface.routes.system import (
            _runtime_revision_response_projection,
            read_runtime_health_snapshot,
        )

        snapshot = _runtime_revision_response_projection(
            read_runtime_health_snapshot(), include_diagnostics=False
        )
        return heartbeat_from_health_snapshot(snapshot, kind)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("websocket_manager", exc)
        return {
            "type": kind,
            "timestamp": time.time(),
            "transport_connected": True,
            "transport_only": False,
            "status": "unhealthy",
            "healthy": False,
            "runtime_probe_healthy": False,
            "runtime_status": "unknown",
            "required_probes": {"all_passed": False},
            "conversation_ready": False,
            "conversation_lane": {"conversation_ready": False, "state": "unknown"},
            "blockers": ["runtime_health_probe_error"],
        }


def heartbeat_from_health_snapshot(
    snapshot: dict[str, Any], kind: str = "heartbeat"
) -> dict[str, Any]:
    """Compact one health generation without probing or upgrading its verdict."""
    from core.runtime.health_contract import (
        required_probe_blockers,
        required_probe_groups_pass,
    )

    required = dict(snapshot.get("required_probes") or {})
    runtime_probe_healthy = (
        snapshot.get("runtime_probe_healthy") is True
        and required_probe_groups_pass(required)
    )
    lane = dict(snapshot.get("conversation_lane") or {})
    conversation_ready = snapshot.get("conversation_ready") is True and lane.get("conversation_ready") is True
    conversation_busy = conversation_lane_is_busy(lane)
    blockers = list(snapshot.get("blockers") or []) + required_probe_blockers(required)
    if not conversation_ready and not conversation_busy:
        blockers.extend(_conversation_lane_blockers(lane))
    healthy = snapshot.get("healthy") is True and runtime_probe_healthy and conversation_ready and not blockers
    working = runtime_probe_healthy and conversation_busy and not blockers
    return {
        "type": kind,
        "timestamp": time.time(),
        "transport_connected": True,
        "transport_only": False,
        "status": "healthy" if healthy else "working" if working else "unhealthy",
        "healthy": healthy,
        "runtime_probe_healthy": runtime_probe_healthy,
        "runtime_status": str(snapshot.get("status", "unknown")),
        "required_probes": required,
        "conversation_ready": conversation_ready,
        "conversation_busy": conversation_busy,
        "conversation_lane": lane,
        "blockers": list(dict.fromkeys(blockers)),
        "health_read_model": dict(snapshot.get("health_read_model") or {}),
        "runtime_revision": dict(snapshot.get("runtime_revision") or {}),
        "proof_readiness_healthy": snapshot.get("proof_readiness_healthy") is True,
        "certification_ready": healthy and snapshot.get("certification_ready") is True,
        "integrity": dict(snapshot.get("integrity") or {}),
        "integrity_blockers": list(snapshot.get("integrity_blockers") or []),
        "boot_phase": (snapshot.get("boot") or {}).get("boot_phase"),
    }


def conversation_heartbeat_payload(kind: str = "heartbeat") -> dict[str, Any]:
    """Return only conversation transport/readiness state for paired clients."""

    full = runtime_heartbeat_payload(kind)
    lane = full.get("conversation_lane")
    lane = lane if isinstance(lane, dict) else {}
    public_lane = {
        key: lane.get(key)
        for key in (
            "state",
            "conversation_ready",
            "active_generation",
            "active_generations",
        )
        if key in lane
    }
    return {
        "type": kind,
        "timestamp": full.get("timestamp", time.time()),
        "transport_connected": True,
        "status": full.get("status", "unknown"),
        "healthy": bool(full.get("conversation_ready", False)),
        "conversation_ready": bool(full.get("conversation_ready", False)),
        "conversation_busy": bool(full.get("conversation_busy", False)),
        "conversation_lane": public_lane,
    }


def _conversation_lane_readiness() -> tuple[dict[str, Any], bool]:
    """Return live conversation readiness for transport heartbeat payloads."""
    try:
        from interface.routes.chat_preflight import _collect_conversation_lane_status

        lane = _collect_conversation_lane_status()
        if not isinstance(lane, dict):
            raise TypeError(f"conversation lane collector returned {type(lane).__name__}")
        ready = bool(lane.get("conversation_ready", False))
        return lane, ready
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "websocket_manager",
            exc,
            severity="critical",
            action="heartbeat failed closed: conversation lane readiness unavailable",
            enforce_failure_policy=False,
        )
        return (
            {
                "conversation_ready": False,
                "state": "unknown",
                "last_failure_reason": str(exc)[:240],
            },
            False,
        )


def _conversation_lane_blockers(lane: dict[str, Any]) -> list[str]:
    state = str(lane.get("state", "unknown") or "unknown").strip().lower()
    blockers = ["conversation_ready"]
    if state:
        blockers.append(f"conversation_lane:{state}")
    reason = str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "").strip()
    if reason:
        blockers.append(f"conversation_reason:{reason[:80]}")
    return blockers


def _runtime_report_blockers(report: dict[str, Any]) -> list[str]:
    blockers = list(report.get("probe_blockers", []) or [])
    failures = report.get("failures", {}) if isinstance(report.get("failures"), dict) else {}
    for tier in ("critical", "important"):
        for failure in failures.get(tier, []) or []:
            if not isinstance(failure, dict):
                continue
            key = str(failure.get("container_key") or "").strip()
            blockers.append(f"{tier}:{key}" if key else f"runtime_{tier}_failure")
    if not blockers:
        status = str(report.get("status", "unknown") or "unknown")
        blockers.append(f"runtime_status:{status}")
    return list(dict.fromkeys(blockers))


# ── Broadcast Bus ────────────────────────────────────────────

class MessageBroadcastBus:
    """A simple pub/sub distributor for server events.
    Ensures that multiple consumers (WebSockets, SSE) get all messages.
    """

    def __init__(self, maxsize: int = 2000) -> None:
        self._subs: list[BroadcastQueue] = []
        self._lock = asyncio.Lock()
        self._maxsize = maxsize

    async def subscribe(self) -> BroadcastQueue:
        q: BroadcastQueue = asyncio.PriorityQueue(maxsize=self._maxsize)
        async with self._lock:
            self._subs.append(q)
        return q

    async def unsubscribe(self, q: BroadcastQueue) -> None:
        async with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def subscriber_count(self) -> int:
        return len(self._subs)

    @staticmethod
    def _worse_than(lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> bool:
        return (lhs[0], -float(lhs[1])) > (rhs[0], -float(rhs[1]))

    @staticmethod
    def _drain_queue_snapshot(q: BroadcastQueue) -> list[BroadcastItem]:
        """Drain the current queue contents without relying on an unbounded loop."""
        drained: list[BroadcastItem] = []
        for _ in range(q.qsize()):
            drained.append(q.get_nowait())
        return drained

    async def _replace_lowest_priority_item(
        self,
        q: BroadcastQueue,
        item: BroadcastItem,
    ) -> None:
        drained = self._drain_queue_snapshot(q)
        if not drained:
            return

        for _ in drained:
            try:
                q.task_done()
            except ValueError:
                break

        worst_idx = 0
        worst_item = drained[0]
        for idx, existing in enumerate(drained[1:], start=1):
            if self._worse_than(existing, worst_item):
                worst_item = existing
                worst_idx = idx

        if item[0] < worst_item[0]:
            drained[worst_idx] = item

        for existing in drained:
            q.put_nowait(existing)

    async def publish(self, message: Any, priority: int = 10) -> None:
        """Push message to all subscriber queues.
        Priority: 0=Critical, 10=Standard, 20=Logs
        """
        item = (priority, time.monotonic(), message)
        async with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    try:
                        await self._replace_lowest_priority_item(q, item)
                    except _QUEUE_REPAIR_ERRORS as _exc:
                        record_degradation('websocket_manager', _exc)
                        logger.warning("Broadcast queue replacement failed: %s", _exc)


def _normalize_ui_event(message: Any) -> dict[str, Any]:
    """Attach stable envelope fields while preserving legacy payload keys."""
    if isinstance(message, str):
        normalized: dict[str, Any] = {"type": "message", "message": message}
    elif isinstance(message, dict):
        normalized = dict(message)
    else:
        normalized = {"type": "message", "message": str(message)}

    kind = str(normalized.get("kind") or normalized.get("type") or "message")
    event_id = str(normalized.get("event_id") or normalized.get("id") or uuid.uuid4().hex)
    raw_ts = normalized.get("event_ts") or normalized.get("timestamp")
    if raw_ts:
        try:
            event_ts = float(raw_ts)
        except (TypeError, ValueError):
            try:
                from datetime import datetime
                ts_str = str(raw_ts).rstrip("Z")
                event_ts = datetime.fromisoformat(ts_str).timestamp()
            except (TypeError, ValueError):
                event_ts = time.time()
    else:
        event_ts = time.time()

    if "payload" not in normalized:
        normalized["payload"] = {
            key: value
            for key, value in normalized.items()
            if key not in {"payload", "event_id", "event_ts", "id", "kind"}
        }

    normalized["kind"] = kind
    normalized.setdefault("type", kind)
    normalized["event_id"] = event_id
    normalized.setdefault("id", event_id)
    normalized["event_ts"] = event_ts
    normalized.setdefault("timestamp", event_ts)
    return normalized


class WebSocketManager:
    """Manages WebSocket connections with priority-based queues and heartbeat monitoring."""

    def __init__(self, task_spawner: TaskSpawner | None = None) -> None:
        self.active_connections: dict[WebSocket, ClientQueue] = {}
        self._connection_scopes: dict[WebSocket, str] = {}
        self._pump_tasks: dict[WebSocket, asyncio.Task[Any]] = {}
        self._heartbeat_tasks: dict[WebSocket, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = 20.0
        self._task_spawner = task_spawner or _spawn_websocket_task

    def set_task_spawner(self, spawner: TaskSpawner) -> None:
        """Set the task spawner function (e.g. _spawn_server_task)."""
        self._task_spawner = spawner

    @staticmethod
    def _worse_than(lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> bool:
        return (lhs[0], -float(lhs[1])) > (rhs[0], -float(rhs[1]))

    @staticmethod
    def _drain_queue_snapshot(queue: ClientQueue) -> list[ClientItem]:
        """Drain the current queue contents without relying on an unbounded loop."""
        drained: list[ClientItem] = []
        for _ in range(queue.qsize()):
            drained.append(queue.get_nowait())
        return drained

    async def _replace_lowest_priority_item(
        self,
        queue: ClientQueue,
        item: ClientItem,
    ) -> None:
        drained = self._drain_queue_snapshot(queue)
        if not drained:
            return

        for _ in drained:
            try:
                queue.task_done()
            except ValueError:
                break

        worst_idx = 0
        worst_item = drained[0]
        for idx, existing in enumerate(drained[1:], start=1):
            if self._worse_than(existing, worst_item):
                worst_idx = idx
                worst_item = existing

        if item[0] < worst_item[0]:
            drained[worst_idx] = item

        for existing in drained:
            queue.put_nowait(existing)

    async def connect(
        self,
        websocket: WebSocket,
        *,
        accepted: bool = False,
        scope: str = "owner",
    ) -> None:
        """Accept connection and start a dedicated message pump for this client."""
        if not accepted:
            await websocket.accept()
        if os.environ.get("AURA_TRACE_MODE") == "1":
            logger.info("📡 [TRACE] WebSocket connected (ID: %s)", id(websocket))
        queue: ClientQueue = asyncio.PriorityQueue(maxsize=1000)
        async with self._lock:
            self.active_connections[websocket] = queue
            self._connection_scopes[websocket] = (
                "conversation" if scope == "conversation" else "owner"
            )

        task = self._task_spawner(
            self._pump_messages(websocket, queue), name="ws_pump"
        )
        self._pump_tasks[websocket] = task

        def _on_pump_done(
            t: asyncio.Task[Any],
            ws: WebSocket = websocket,
        ) -> None:
            exc = t.exception() if not t.cancelled() else None
            if exc:
                logger.error("WS pump task died with exception: %s", exc)
            self._pump_tasks.pop(ws, None)
        task.add_done_callback(_on_pump_done)

        self._heartbeat_tasks[websocket] = self._task_spawner(
            self._heartbeat_runner(websocket),
            name="ws_heartbeat",
        )

        logger.info("WS: Client connected. Total: %d", len(self.active_connections))

    async def _pump_messages(self, websocket: WebSocket, queue: ClientQueue) -> None:
        """Hardened message pump with heartbeat and timeout protection."""
        try:
            while not is_shutdown_requested():
                try:
                    _p, _t, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    try:
                        await websocket.send_text(payload)
                    except _WEBSOCKET_DELIVERY_ERRORS as e:
                        if isinstance(e, (WebSocketDisconnect, ConnectionClosed)):
                            logger.debug(
                                "WS send skipped; client disconnected before delivery (type=%s).",
                                type(e).__name__,
                            )
                        else:
                            logger.warning(
                                "WS send failed (message lost, type=%s): %s",
                                type(e).__name__,
                                e,
                            )
                        break
                    finally:
                        try:
                            queue.task_done()
                        except ValueError as exc:
                            record_degradation('websocket_manager', exc)
                            logger.warning("WS queue task accounting failed: %s", exc)
                except TimeoutError:
                    try:
                        await websocket.send_json(
                            self.heartbeat_payload(websocket, "heartbeat")
                        )
                    except _WEBSOCKET_DELIVERY_ERRORS as exc:
                        logger.debug("WS heartbeat failed; closing pump: %s", exc)
                        break
                except asyncio.CancelledError:
                    raise
                except _QUEUE_REPAIR_ERRORS as e:
                    record_degradation('websocket_manager', e)
                    logger.error("Error in WS pump loop: %s", e)
                    break
        except (WebSocketDisconnect, ConnectionClosed):
            pass  # no-op: intentional
        except asyncio.CancelledError:
            raise
        except _WEBSOCKET_DELIVERY_ERRORS as e:
            record_degradation('websocket_manager', e)
            logger.error("Error pumping messages to client: %s", e)
        finally:
            await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Cleanly remove connection and cancel pump task."""
        async with self._lock:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
            self._connection_scopes.pop(websocket, None)
        task = self._pump_tasks.pop(websocket, None)
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
        heartbeat_task = self._heartbeat_tasks.pop(websocket, None)
        if heartbeat_task and heartbeat_task is not asyncio.current_task() and not heartbeat_task.done():
            heartbeat_task.cancel()
        logger.debug("WS: Client disconnected. Total: %d", len(self.active_connections))

    async def _heartbeat_runner(self, websocket: WebSocket) -> None:
        """Aggressively reaps zombie connections to prevent FD exhaustion."""
        try:
            while not is_shutdown_requested():
                jitter = random.uniform(0, 5.0)
                await asyncio.sleep(self._heartbeat_interval + jitter)

                async with self._lock:
                    if websocket not in self.active_connections:
                        break

                try:
                    await asyncio.wait_for(
                        websocket.send_json(self.heartbeat_payload(websocket, "ping")),
                        timeout=5.0,
                    )
                except _WEBSOCKET_HEARTBEAT_ERRORS:
                    logger.warning("🧟 WS ZOMBIE: Reaping connection %s", id(websocket))
                    await self.disconnect(websocket)
                    break
        except asyncio.CancelledError:
            pass  # no-op: intentional

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON-serializable dict to all clients via their queues."""
        if not self.active_connections:
            return
        owner_connections = [
            (websocket, queue)
            for websocket, queue in self.active_connections.items()
            if self._connection_scopes.get(websocket, "owner") == "owner"
        ]
        if not owner_connections:
            return
        message = _normalize_ui_event(message)
        msg_type = message.get("type", "")
        priority = 10
        if msg_type in ("aura_message", "chat_response", "chat_stream_chunk", "agent_step", "tool_execution"):
            priority = 0
        elif msg_type in ("thought", "neural_event", "log", "telemetry"):
            priority = 20

        from enum import Enum

        class _EnumEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, Enum):
                    return obj.value
                return super().default(obj)

        payload = await asyncio.to_thread(
            json.dumps,
            message,
            cls=_EnumEncoder,
            separators=(",", ":"),
        )
        item = (priority, time.monotonic(), payload)

        disconnect_later: list[WebSocket] = []
        async with self._lock:
            for websocket, queue in owner_connections:
                if websocket not in self.active_connections:
                    continue
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    try:
                        await self._replace_lowest_priority_item(queue, item)
                    except _QUEUE_REPAIR_ERRORS as exc:
                        record_degradation('websocket_manager', exc)
                        logger.warning("WS client queue replacement failed; disconnecting client: %s", exc)
                        disconnect_later.append(websocket)
        for websocket in disconnect_later:
            self._task_spawner(self.disconnect(websocket), name="ws_disconnect")

    def count(self) -> int:
        return len(self.active_connections)

    def owner_count(self) -> int:
        """Count authenticated local owner UI clients, excluding paired chat."""

        return sum(
            1
            for websocket in self.active_connections
            if self._connection_scopes.get(websocket, "owner") == "owner"
        )

    def heartbeat_payload(self, websocket: WebSocket, kind: str) -> dict[str, Any]:
        if self._connection_scopes.get(websocket, "owner") == "conversation":
            return conversation_heartbeat_payload(kind)
        return runtime_heartbeat_payload(kind)


# ── Module-level singletons ──────────────────────────────────

broadcast_bus = MessageBroadcastBus(maxsize=1000)
ws_manager = WebSocketManager()
log_queue: collections.deque[Any] = collections.deque(
    maxlen=_env_positive_int("AURA_UI_LOG_QUEUE_MAXLEN", 2000, minimum=500)
)
