import asyncio
import json
import logging
import socket
import time
from typing import Any

from core.runtime.errors import NetworkEffectDenied, record_degradation
from core.runtime.network_gateway import build_stream_endpoint, get_network_gateway
from core.runtime.service_registry import get_runtime_service
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Mycelium.Swarm")

class SwarmProtocol:
    """
    Gossip-based protocol for sharing cognitive states between Aura nodes.
    Supports "Consensus Gating" for skill verification.
    """
    def __init__(self, port: int = 10003, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.peers = set()
        self.running = False
        self._server = None
        self._mood_broadcast_task = None
        self.node_id = socket.gethostname()
        self.offline_only = False
        
    async def start(self):
        if self.running:
            return
        self.running = True
        # [BOOT FIX] Default to loopback to avoid macOS firewall dialogs
        # that block the event loop indefinitely on 0.0.0.0 binds.
        bind_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        try:
            self._server = await asyncio.wait_for(
                asyncio.start_server(self._handle_peer, bind_host, self.port),
                timeout=5.0,
            )
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().register_shutdown_resource(
                    self._server,
                    kind="tcp_listener",
                    name=f"swarm_protocol:{bind_host}:{self.port}",
                    source="core.collective.swarm_protocol",
                    closer=self._close_listener,
                    timeout_s=2.0,
                )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                await self._close_listener()
                raise
        except (PermissionError, OSError, TimeoutError) as exc:
            logger.warning("🕸️ SwarmProtocol bind failed on %s:%d (%s). Running offline.", bind_host, self.port, exc)
            self._server = None
            self.offline_only = True
        except (RuntimeError, asyncio.CancelledError, AttributeError) as exc:
            logger.warning("🕸️ SwarmProtocol unexpected bind error: %s. Running offline.", exc)
            self._server = None
            self.offline_only = True
        self.host = bind_host
        self._mood_broadcast_task = get_task_tracker().create_task(self._broadcast_loop())
        if self.offline_only:
            logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
        logger.info(f"🕸️ Mycelial Swarm active on %s:%d (Node: {self.node_id})", self.host, self.port)

    async def stop(self):
        self.running = False
        server = self._server
        await self._close_listener()
        if server is not None:
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().unregister_shutdown_resource(server)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug(
                    "the swarm listener was not unregistered from shutdown hygiene (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
        task, self._mood_broadcast_task = self._mood_broadcast_task, None
        if task:
            await cancel_and_join(task, owner="core.collective.swarm_protocol")

    async def _close_listener(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            try:
                # Bounded: a peer holding its connection open must not wedge
                # listener teardown (A1 bounded-await discipline).
                await asyncio.wait_for(server.wait_closed(), timeout=5.0)
            except TimeoutError:
                logger.debug("Swarm listener close timed out; abandoning socket.")

    async def _handle_peer(self, reader, writer):
        try:
            data = await reader.read(4096)
            if not data or not data.strip():
                return
            try:
                decoded = data.decode('utf-8', errors='ignore')
                message = json.loads(decoded)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                logger.debug("Received malformed/non-JSON data from peer: %s", exc)
                return

            peer_id = message.get("node_id")
            if peer_id:
                self.peers.add(writer.get_extra_info('peername')[0])
                await self._process_gossip(message)
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('swarm_protocol', e)
            logger.debug("Swarm gossip error: %s", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError) as _exc:
                logger.debug("Suppressed %s in core.collective.swarm_protocol: %s", type(_exc).__name__, _exc)

    async def _process_gossip(self, message: dict[str, Any]):
        msg_type = message.get("type")
        if msg_type == "mood_sync":
            peer_id = message.get("node_id", "unknown")
            peer_mood = message.get("mood", {})
            peer_valence = float(peer_mood.get("valence", 0.0)) if isinstance(peer_mood, dict) else 0.0

            # Mood contagion: nudge local affect toward swarm average
            try:
                affect = get_runtime_service("affect_engine", default=None)
                if affect is not None and hasattr(affect, "modify"):
                    # Weak contagion factor — peers influence but don't override
                    contagion_weight = 0.05
                    affect.modify(valence_delta=peer_valence * contagion_weight)
                    logger.debug("Swarm mood contagion from %s: valence nudge %.3f", peer_id, peer_valence * contagion_weight)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('swarm_protocol', e)
                logger.debug("Mood contagion failed: %s", e)
        elif msg_type == "skill_verification":
            # Consensus Gating: Verify a forged skill
            skill_id = message.get("skill_id")
            logger.info("🤝 Swarm Consensus requested for skill: %s", skill_id)
            # Automatically approve for now (In real AGI, other nodes would run tests)
            await self.broadcast({"type": "skill_approved", "skill_id": skill_id, "node_id": self.node_id})

    async def broadcast(self, message: dict[str, Any]):
        payload = json.dumps(
            {**message, "node_id": self.node_id, "timestamp": time.time()},
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        
        for peer in list(self.peers):
            writer = None
            try:
                # Skip if it's our own IP (if discovered)
                admission = await get_network_gateway().connect_stream(
                    build_stream_endpoint(peer, self.port),
                    open_timeout=3.0,
                    source="collective:swarm_protocol.broadcast",
                    read_only=False,
                    allow_private_target=True,
                )
                writer = admission.writer
                writer.write(payload)
                await asyncio.wait_for(writer.drain(), timeout=3.0)
            except asyncio.CancelledError:
                raise
            except (
                NetworkEffectDenied,
                OSError,
                RuntimeError,
                TimeoutError,
                AttributeError,
                ValueError,
            ):
                self.peers.discard(peer)
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await asyncio.wait_for(writer.wait_closed(), timeout=3.0)
                    except (ConnectionError, OSError, RuntimeError):
                        logger.debug("Swarm peer writer close failed", exc_info=True)
                    except TimeoutError:
                        logger.debug("Swarm peer writer close timed out")

    async def _broadcast_loop(self):
        while self.running:
            substrate = get_runtime_service("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_mood()
                await self.broadcast({"type": "mood_sync", "mood": mood})
            await asyncio.sleep(30)
