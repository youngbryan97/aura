"""interface/websocket_manager.py
─────────────────────────────────
Extracted from server.py — WebSocket connection management,
broadcast infrastructure, and UI event normalization.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import collections
import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    class ConnectionClosed(Exception):   # type: ignore[no-redef]
        pass  # no-op: intentional

from fastapi import WebSocketDisconnect

logger = logging.getLogger("Aura.Server.WebSocket")


# ── Broadcast Bus ────────────────────────────────────────────

class MessageBroadcastBus:
    """A simple pub/sub distributor for server events.
    Ensures that multiple consumers (WebSockets, SSE) get all messages.
    """

    def __init__(self, maxsize: int = 2000):
        self._subs: List[asyncio.PriorityQueue] = []
        self._lock = asyncio.Lock()
        self._maxsize = maxsize

    async def subscribe(self) -> asyncio.PriorityQueue:
        q = asyncio.PriorityQueue(maxsize=self._maxsize)
        async with self._lock:
            self._subs.append(q)
        return q

    async def unsubscribe(self, q: asyncio.PriorityQueue):
        async with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def subscriber_count(self) -> int:
        return len(self._subs)

    @staticmethod
    def _worse_than(lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> bool:
        return (lhs[0], -float(lhs[1])) > (rhs[0], -float(rhs[1]))

    async def _replace_lowest_priority_item(
        self,
        q: asyncio.PriorityQueue,
        item: tuple[int, float, Any],
    ) -> None:
        drained: list[tuple[int, float, Any]] = []
        while True:
            try:
                drained.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
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

    async def publish(self, message: Any, priority: int = 10):
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
                    except Exception as _exc:
                        record_degradation('websocket_manager', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)


def _normalize_ui_event(message: Any) -> Dict[str, Any]:
    """Attach stable envelope fields while preserving legacy payload keys."""
    if isinstance(message, str):
        normalized: Dict[str, Any] = {"type": "message", "message": message}
    elif isinstance(message, dict):
        normalized = dict(message)
    else:
        normalized = {"type": "message", "message": str(message)}

    kind = str(normalized.get("kind") or normalized.get("type") or "message")
    event_id = str(normalized.get("event_id") or normalized.get("id") or uuid.uuid4().hex)
    event_ts = float(normalized.get("event_ts") or normalized.get("timestamp") or time.time())

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

    def __init__(self, task_spawner=None):
        self.active_connections: Dict[WebSocket, asyncio.PriorityQueue] = {}
        self._pump_tasks: Dict[WebSocket, asyncio.Task] = {}
        self._heartbeat_tasks: Dict[WebSocket, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = 20.0
        self._task_spawner = task_spawner or asyncio.create_task

    def set_task_spawner(self, spawner):
        """Set the task spawner function (e.g. _spawn_server_task)."""
        self._task_spawner = spawner

    @staticmethod
    def _worse_than(lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> bool:
        return (lhs[0], -float(lhs[1])) > (rhs[0], -float(rhs[1]))

    async def _replace_lowest_priority_item(
        self,
        queue: asyncio.PriorityQueue,
        item: tuple[int, float, str],
    ) -> None:
        drained: list[tuple[int, float, str]] = []
        while True:
            try:
                drained.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
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

    async def connect(self, websocket: WebSocket):
        """Accept connection and start a dedicated message pump for this client."""
        await websocket.accept()
        if os.environ.get("AURA_TRACE_MODE") == "1":
            logger.info("📡 [TRACE] WebSocket connected (ID: %s)", id(websocket))
        queue = asyncio.PriorityQueue(maxsize=1000)
        async with self._lock:
            self.active_connections[websocket] = queue

        task = self._task_spawner(
            self._pump_messages(websocket, queue), name="ws_pump"
        )
        self._pump_tasks[websocket] = task

        def _on_pump_done(t: asyncio.Task, ws=websocket):
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

    async def _pump_messages(self, websocket: WebSocket, queue: asyncio.Queue):
        """Hardened message pump with heartbeat and timeout protection."""
        try:
            while True:
                try:
                    _p, _t, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    try:
                        await websocket.send_text(payload)
                    except BaseException as e:
                        logger.warning("WS send failed (message lost, type=%s): %s", type(e).__name__, e)
                        break
                    finally:
                        queue.task_done()
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
                    except Exception:
                        break
                except Exception as e:
                    record_degradation('websocket_manager', e)
                    logger.error("Error in WS pump loop: %s", e)
                    break
        except (WebSocketDisconnect, ConnectionClosed):
            pass  # no-op: intentional
        except Exception as e:
            record_degradation('websocket_manager', e)
            logger.error("Error pumping messages to client: %s", e)
        finally:
            await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Cleanly remove connection and cancel pump task."""
        async with self._lock:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
        task = self._pump_tasks.pop(websocket, None)
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
        heartbeat_task = self._heartbeat_tasks.pop(websocket, None)
        if heartbeat_task and heartbeat_task is not asyncio.current_task() and not heartbeat_task.done():
            heartbeat_task.cancel()
        logger.debug("WS: Client disconnected. Total: %d", len(self.active_connections))

    async def _heartbeat_runner(self, websocket: WebSocket):
        """Aggressively reaps zombie connections to prevent FD exhaustion."""
        try:
            while True:
                jitter = random.uniform(0, 5.0)
                await asyncio.sleep(self._heartbeat_interval + jitter)

                async with self._lock:
                    if websocket not in self.active_connections:
                        break

                try:
                    await asyncio.wait_for(
                        websocket.send_json({"type": "ping", "timestamp": time.monotonic()}),
                        timeout=5.0,
                    )
                except (asyncio.TimeoutError, Exception):
                    logger.warning("🧟 WS ZOMBIE: Reaping connection %s", id(websocket))
                    await self.disconnect(websocket)
                    break
        except asyncio.CancelledError:
            pass  # no-op: intentional

    async def broadcast(self, message: dict):
        """Send a JSON-serializable dict to all clients via their queues."""
        if not self.active_connections:
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
            def default(self, obj):
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

        disconnect_later: List[WebSocket] = []
        async with self._lock:
            for websocket, queue in list(self.active_connections.items()):
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    try:
                        await self._replace_lowest_priority_item(queue, item)
                    except Exception:
                        disconnect_later.append(websocket)
        for websocket in disconnect_later:
            self._task_spawner(self.disconnect(websocket), name="ws_disconnect")

    def count(self) -> int:
        return len(self.active_connections)


# ── Module-level singletons ──────────────────────────────────

broadcast_bus = MessageBroadcastBus(maxsize=1000)
ws_manager = WebSocketManager()
log_queue: collections.deque = collections.deque(maxlen=500)
