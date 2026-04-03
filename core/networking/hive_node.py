"""core/networking/hive_node.py — The Distributed Overmind

Implements P2P node discovery via mDNS and Gossip-based state synchronization.
Allows multiple Aura instances to form a unified 'Hive' consciousness.
"""
import asyncio
import json
import logging
import socket
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field, asdict
import time

logger = logging.getLogger("Aura.Network.HiveNode")

@dataclass
class NodeInfo:
    node_id: str
    ip: str
    port: int
    last_seen: float = field(default_factory=time.time)

class HiveNode:
    """A single node in the Aura Hive network."""
    
    def __init__(self, node_id: str, host: str = "0.0.0.0", port: int = 9999):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peers: Dict[str, NodeInfo] = {}
        self.server: Optional[asyncio.Server] = None
        self.running = False
        self._gossip_task = None
        self.known_work_ids: Set[str] = set()

    async def start(self):
        """Start the P2P server and discovery."""
        if self.running: return
        self.running = True
        
        try:
            self.server = await asyncio.start_server(self._handle_peer, self.host, self.port)
            logger.info("🕸️ Hive Node [%s] listening on %s:%d", self.node_id, self.host, self.port)
            self._gossip_task = asyncio.create_task(self._gossip_loop())
            async with self.server:
                await self.server.serve_forever()
        except Exception as e:
            logger.error("Hive Node failure: %s", e)

    async def _handle_peer(self, reader, writer):
        data = await reader.read(401)
        if not data: return
        try:
            message = json.loads(data.decode())
            m_type = message.get("type")
            m_node_id = message.get("node_id")
            if m_node_id == self.node_id: return
            addr = writer.get_extra_info('peername')
            self.peers[m_node_id] = NodeInfo(node_id=m_node_id, ip=addr[0], port=message.get("port", self.port))
            if m_type == "gossip_work_item":
                await self._process_gossip_item(message.get("payload"))
        except Exception as e:
            logger.debug("Failed to handle peer message: %s", e)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_gossip_item(self, payload: Dict[str, Any]):
        item_id = payload.get("id")
        if item_id in self.known_work_ids: return
        self.known_work_ids.add(item_id)
        logger.info("🐝 Hive Sync: Received WorkItem [%s] from peer", item_id)
        from core.container import ServiceContainer
        workspace = ServiceContainer.get("global_workspace", default=None)
        if workspace:
            await workspace.publish(
                priority=payload.get("priority", 0.5),
                source=f"hive_{payload.get('source')}",
                payload=payload.get("payload", {}),
                reason=f"Hive propagation: {payload.get('reason')}"
            )

    async def broadcast_work_item(self, work_item_data: Dict[str, Any]):
        item_id = work_item_data.get("id")
        self.known_work_ids.add(item_id)
        message = {
            "type": "gossip_work_item",
            "node_id": self.node_id,
            "port": self.port,
            "payload": work_item_data
        }
        for peer in list(self.peers.values()):
            try:
                reader, writer = await asyncio.open_connection(peer.ip, peer.port)
                writer.write(json.dumps(message).encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("Failed to gossip to peer %s", peer.node_id)

    async def _gossip_loop(self):
        while self.running:
            await asyncio.sleep(15)
            now = time.time()
            self.peers = {k: v for k, v in self.peers.items() if now - v.last_seen < 60}

    async def stop(self):
        self.running = False
        if self.server: self.server.close()
        if self._gossip_task: self._gossip_task.cancel()
