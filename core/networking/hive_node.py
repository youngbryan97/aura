"""core/networking/hive_node.py - bounded local hive gossip node.

The hive node is external-I/O adjacent, so defaults are intentionally local and
failure modes are explicit. Distributed operation can opt into a wider bind by
setting ``AURA_HIVE_HOST`` or passing ``host`` directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import (
    FallbackClassification,
    NetworkEffectDenied,
    Severity,
    record_degradation,
)
from core.runtime.network_gateway import build_stream_endpoint, get_network_gateway
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.Network.HiveNode")

DEFAULT_HIVE_HOST = "127.0.0.1"
DEFAULT_HIVE_PORT = 9999
MAX_HIVE_MESSAGE_BYTES = 64 * 1024
PEER_TTL_S = 60.0
GOSSIP_INTERVAL_S = 15.0
CONNECT_TIMEOUT_S = 3.0

_HIVE_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
    NetworkEffectDenied,
)


def _record_hive_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "hive_node",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("hive_node", error, severity=severity, action=action)
        except TypeError:
            logger.debug("HiveNode degradation could not be recorded: %s", signature_exc)


def _safe_text(value: object, *, default: str = "", max_chars: int = 256) -> str:
    try:
        text = str(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        text = default
    return text.replace("\x00", "")[:max_chars]


def _derive_node_id(node_id: object) -> str:
    for attr in ("node_id", "instance_id", "id", "name"):
        value = getattr(node_id, attr, None)
        if value:
            return _safe_text(value, max_chars=128)
    text = _safe_text(node_id, max_chars=128)
    return text if text else f"aura-{socket.gethostname()}"


@dataclass
class NodeInfo:
    node_id: str
    ip: str
    port: int
    last_seen: float = field(default_factory=lambda: time.time())


class HiveNode:
    """A bounded gossip node for local Aura instance coordination."""

    def __init__(self, node_id: object, host: str | None = None, port: int | None = None):
        self.node_id = _derive_node_id(node_id)
        self.host = host or os.getenv("AURA_HIVE_HOST", DEFAULT_HIVE_HOST)
        selected_port = port if port is not None else os.getenv("AURA_HIVE_PORT", DEFAULT_HIVE_PORT)
        self.port = int(selected_port)
        self.peers: dict[str, NodeInfo] = {}
        self.server: asyncio.Server | None = None
        self.running = False
        self._gossip_task: asyncio.Task | None = None
        self.known_work_ids: set[str] = set()
        self._last_error = ""
        self._last_error_at = 0.0

    async def start(self) -> None:
        """Start the P2P server and peer maintenance loop."""
        if self.running:
            return

        try:
            self.server = await asyncio.start_server(self._handle_peer, self.host, self.port)
            self.running = True
            sockets = self.server.sockets or []
            if sockets:
                bound = sockets[0].getsockname()
                self.port = int(bound[1])
            logger.info("Hive Node [%s] listening on %s:%d", self.node_id, self.host, self.port)

            self._gossip_task = create_tracked_task(
                self._gossip_loop(),
                name="hive_node.gossip_loop",
            )

            async with self.server:
                await self.server.serve_forever()
        except asyncio.CancelledError:
            raise
        except _HIVE_ERRORS as exc:
            self.running = False
            self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=300)}"
            self._last_error_at = time.time()
            _record_hive_degradation(
                exc,
                action="hive server stayed offline after startup/runtime failure",
                severity="degraded",
                extra={"host": self.host, "port": self.port},
            )
            logger.error("Hive Node failure: %s", exc)
        finally:
            if not self.running:
                await self._close_server()

    async def _handle_peer(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peername = writer.get_extra_info("peername")
        try:
            data = await asyncio.wait_for(reader.read(MAX_HIVE_MESSAGE_BYTES + 1), timeout=CONNECT_TIMEOUT_S)
            if not data:
                return
            if len(data) > MAX_HIVE_MESSAGE_BYTES:
                raise ValueError("hive message exceeded maximum size")

            message = json.loads(data.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("hive message must be a JSON object")

            m_type = _safe_text(message.get("type"), max_chars=64)
            m_node_id = _safe_text(message.get("node_id"), max_chars=128)
            if not m_node_id or m_node_id == self.node_id:
                return

            host = peername[0] if isinstance(peername, tuple) and peername else ""
            port = self._safe_port(message.get("port"), default=self.port)
            self.peers[m_node_id] = NodeInfo(node_id=m_node_id, ip=host, port=port)
            if m_type == "gossip_work_item":
                await self._process_gossip_item(message.get("payload"))
        except _HIVE_ERRORS as exc:
            self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=300)}"
            self._last_error_at = time.time()
            _record_hive_degradation(
                exc,
                action="rejected malformed or failed hive peer message and kept node running",
                severity="warning",
                extra={"peer": str(peername)[:120]},
            )
            logger.debug("Failed to handle peer message: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except _HIVE_ERRORS as exc:
                logger.debug("Hive peer writer close failed: %s", exc)

    async def _process_gossip_item(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            raise ValueError("gossip payload must be a dictionary")
        item_id = _safe_text(payload.get("id"), max_chars=128)
        if not item_id:
            raise ValueError("gossip payload missing id")
        if item_id in self.known_work_ids:
            return False
        self.known_work_ids.add(item_id)
        logger.info("Hive Sync: received WorkItem [%s] from peer", item_id)

        from core.container import ServiceContainer

        workspace = ServiceContainer.get("global_workspace", default=None)
        publish = getattr(workspace, "publish", None)
        if not callable(publish):
            return False
        result = publish(
            priority=self._safe_priority(payload.get("priority", 0.5)),
            source=f"hive_{_safe_text(payload.get('source'), default='peer', max_chars=80)}",
            payload=payload.get("payload", {}),
            reason=f"Hive propagation: {_safe_text(payload.get('reason'), max_chars=200)}",
        )
        if asyncio.iscoroutine(result):
            await asyncio.wait_for(result, timeout=CONNECT_TIMEOUT_S)
        return True

    async def broadcast_work_item(self, work_item_data: dict[str, Any]) -> dict[str, int]:
        item_id = _safe_text(work_item_data.get("id"), max_chars=128)
        if not item_id:
            raise ValueError("work item data missing id")
        self.known_work_ids.add(item_id)
        message = {
            "type": "gossip_work_item",
            "node_id": self.node_id,
            "port": self.port,
            "payload": work_item_data,
        }
        payload = json.dumps(message, ensure_ascii=False, default=str).encode("utf-8")
        if len(payload) > MAX_HIVE_MESSAGE_BYTES:
            raise ValueError("work item gossip payload exceeded maximum size")

        sent = 0
        failed = 0
        for peer in list(self.peers.values()):
            writer: asyncio.StreamWriter | None = None
            try:
                admission = await get_network_gateway().connect_stream(
                    build_stream_endpoint(peer.ip, peer.port),
                    open_timeout=CONNECT_TIMEOUT_S,
                    source="networking:hive_node.broadcast_work_item",
                    read_only=False,
                    allow_private_target=True,
                )
                writer = admission.writer
                writer.write(payload)
                await asyncio.wait_for(writer.drain(), timeout=CONNECT_TIMEOUT_S)
                sent += 1
            except _HIVE_ERRORS as exc:
                failed += 1
                self.peers.pop(peer.node_id, None)
                _record_hive_degradation(
                    exc,
                    action="removed unreachable hive peer after gossip send failed",
                    severity="warning",
                    extra={"peer": peer.node_id, "ip": peer.ip, "port": peer.port},
                )
                logger.debug("Failed to gossip to peer %s: %s", peer.node_id, exc)
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await asyncio.wait_for(
                            writer.wait_closed(),
                            timeout=CONNECT_TIMEOUT_S,
                        )
                    except _HIVE_ERRORS as _exc:
                        logger.debug("Suppressed %s in core.networking.hive_node: %s", type(_exc).__name__, _exc)
        return {"sent": sent, "failed": failed}

    async def _gossip_loop(self) -> None:
        while self.running:
            await asyncio.sleep(GOSSIP_INTERVAL_S)
            self._prune_peers()

    def _prune_peers(self) -> int:
        now = time.time()
        before = len(self.peers)
        self.peers = {key: peer for key, peer in self.peers.items() if now - peer.last_seen < PEER_TTL_S}
        return before - len(self.peers)

    async def stop(self) -> None:
        self.running = False
        await self._close_server()
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.networking.hive_node: %s", type(_exc).__name__, _exc)
            self._gossip_task = None

    async def _close_server(self) -> None:
        if self.server:
            self.server.close()
            try:
                await self.server.wait_closed()
            except _HIVE_ERRORS as exc:
                logger.debug("Hive server close failed: %s", exc)
            self.server = None

    @staticmethod
    def _safe_port(value: object, *, default: int) -> int:
        try:
            port = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return port if 0 < port <= 65535 else default

    @staticmethod
    def _safe_priority(value: object) -> float:
        try:
            priority = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.5
        return max(0.0, min(1.0, priority))

    def status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "peer_count": len(self.peers),
            "known_work_count": len(self.known_work_ids),
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
            "peers": [asdict(peer) for peer in self.peers.values()],
        }
