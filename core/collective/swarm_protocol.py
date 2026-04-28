from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import json
import socket
import time
from typing import Dict, Any, List, Optional
from core.container import ServiceContainer

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
        bind_host = self.host
        try:
            self._server = await asyncio.start_server(self._handle_peer, bind_host, self.port)
        except PermissionError:
            if bind_host == "127.0.0.1":
                self._server = None
                self.offline_only = True
            else:
                bind_host = "127.0.0.1"
                try:
                    self._server = await asyncio.start_server(self._handle_peer, bind_host, self.port)
                    logger.warning("🕸️ SwarmProtocol external bind denied. Falling back to loopback on %s:%d", bind_host, self.port)
                except PermissionError:
                    self._server = None
                    self.offline_only = True
        self.host = bind_host
        self._mood_broadcast_task = get_task_tracker().create_task(self._broadcast_loop())
        if self.offline_only:
            logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
        logger.info(f"🕸️ Mycelial Swarm active on %s:%d (Node: {self.node_id})", self.host, self.port)

    async def stop(self):
        self.running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._mood_broadcast_task:
            self._mood_broadcast_task.cancel()

    async def _handle_peer(self, reader, writer):
        data = await reader.read(4096)
        try:
            message = json.loads(data.decode())
            peer_id = message.get("node_id")
            if peer_id:
                self.peers.add(writer.get_extra_info('peername')[0])
                await self._process_gossip(message)
        except Exception as e:
            record_degradation('swarm_protocol', e)
            logger.debug(f"Swarm gossip error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_gossip(self, message: Dict[str, Any]):
        msg_type = message.get("type")
        if msg_type == "mood_sync":
            peer_id = message.get("node_id", "unknown")
            peer_mood = message.get("mood", {})
            peer_valence = float(peer_mood.get("valence", 0.0)) if isinstance(peer_mood, dict) else 0.0

            # Mood contagion: nudge local affect toward swarm average
            try:
                from core.container import ServiceContainer
                affect = ServiceContainer.get("affect_engine", default=None)
                if affect is not None and hasattr(affect, "modify"):
                    # Weak contagion factor — peers influence but don't override
                    contagion_weight = 0.05
                    affect.modify(valence_delta=peer_valence * contagion_weight)
                    logger.debug("Swarm mood contagion from %s: valence nudge %.3f", peer_id, peer_valence * contagion_weight)
            except Exception as e:
                record_degradation('swarm_protocol', e)
                logger.debug("Mood contagion failed: %s", e)
        elif msg_type == "skill_verification":
            # Consensus Gating: Verify a forged skill
            skill_id = message.get("skill_id")
            logger.info(f"🤝 Swarm Consensus requested for skill: {skill_id}")
            # Automatically approve for now (In real AGI, other nodes would run tests)
            await self.broadcast({"type": "skill_approved", "skill_id": skill_id, "node_id": self.node_id})

    async def broadcast(self, message: Dict[str, Any]):
        message["node_id"] = self.node_id
        message["timestamp"] = time.time()
        payload = json.dumps(message).encode()
        
        for peer in list(self.peers):
            try:
                # Skip if it's our own IP (if discovered)
                reader, writer = await asyncio.open_connection(peer, self.port)
                writer.write(payload)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            except Exception:
                self.peers.remove(peer)

    async def _broadcast_loop(self):
        while self.running:
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_mood()
                await self.broadcast({"type": "mood_sync", "mood": mood})
            await asyncio.sleep(30)
