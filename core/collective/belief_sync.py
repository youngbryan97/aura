"""core/collective/belief_sync.py
Phase 16: Cosmic Consciousness - Belief Synchronization Protocol.
Allows Aura instances to share high-confidence world-model data.
"""
import asyncio
import logging
import secrets
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from core.container import ServiceContainer
from core.adaptation.immune_system import get_immune_system
from core.runtime import background_policy
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Collective.BeliefSync")


def _generate_instance_secret() -> str:
    """Generate a cryptographically secure per-instance auth token."""
    return secrets.token_urlsafe(32)


class BeliefSync:
    """Manages cross-node belief exchange."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._resonance_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self.sync_interval = 60.0 # Sync every minute
        self.min_confidence = 0.8 # Only share strong beliefs
        self.discovery_interval = 300.0 # Discover every 5 mins
        # SEC-AUDIT: Per-instance cryptographically-generated auth secret.
        # Never use a hardcoded "default_secret".
        self._instance_secret: str = _generate_instance_secret()

    async def start(self):
        if self.running: return
        self.running = True
        tracker = get_task_tracker()
        self._sync_task = tracker.create_task(self._sync_loop(), name="belief_sync.sync")
        self._resonance_task = tracker.create_task(self._resonance_loop(), name="belief_sync.resonance")
        self._discovery_task = tracker.create_task(self._discovery_loop(), name="belief_sync.discovery")
        logger.info("🌌 BeliefSync protocol active (Discovery & Resonance enabled)")

    async def stop(self):
        self.running = False
        if self._sync_task:
            self._sync_task.cancel()
        if self._resonance_task:
            self._resonance_task.cancel()
        if self._discovery_task:
            self._discovery_task.cancel()
        for task in (self._sync_task, self._resonance_task, self._discovery_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError as _exc:
                    logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("🌌 BeliefSync protocol stopped")

    async def _sync_loop(self):
        backoff: float = 0.0
        while self.running:
            try:
                await asyncio.sleep(self.sync_interval + backoff)
                if not self.orchestrator.peers:
                    backoff = 0 # Reset backoff if no peers
                    continue

                # 1. Gather high-confidence beliefs
                graph = ServiceContainer.get("knowledge_graph", default=None) or getattr(self.orchestrator, 'knowledge_graph', None)
                if not graph: 
                    continue

                # In Phase 16, we use get_strong_beliefs for sharing
                strong_beliefs = []
                if hasattr(graph, 'get_strong_beliefs'):
                    strong_beliefs = graph.get_strong_beliefs(self.min_confidence)
                
                # Phase 15.1: Include Abstracted Principles
                principles = []
                abs_engine = ServiceContainer.get("abstraction_engine", default=None)
                if abs_engine and hasattr(abs_engine, 'storage_path'):
                    try:
                        import json
                        content = abs_engine.storage_path.read_text()
                        principles = json.loads(content)
                    except Exception as e:
                        logger.error(f"Failed to read principles for sync: {e}")

                if not strong_beliefs and not principles: 
                    continue

                # 2. Push to all active peers
                payload = {
                    "origin": self.orchestrator.node_id if hasattr(self.orchestrator, 'node_id') else "aura-local",
                    "timestamp": time.time(),
                    "beliefs": strong_beliefs,
                    "principles": principles
                }
                
                logger.info("🌌 Syncing %d beliefs with %d peers...", len(strong_beliefs), len(self.orchestrator.peers))
                
                # Phase 16.2: Implement actual p2p transport
                success = await self._broadcast_to_peers(payload)
                
                # Simple backoff logic
                if success:
                    backoff = 0
                else:
                    backoff = min(backoff + 10, 300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("BeliefSync cycle error: %s", e)
                backoff = min(backoff + 30, 600)
                await asyncio.sleep(10)

    async def _broadcast_to_peers(self, payload: Dict[str, Any]):
        """Push beliefs to all active peers via RPC."""
        if not self.orchestrator.peers: return
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for peer_id, peer_info in self.orchestrator.peers.items():
                addr = peer_info.get("address")
                port = peer_info.get("rpc_port", 8000)
                if not addr: continue
                
                # Aura RPC Port (Phase 16.2 convention)
                url = f"http://{addr}:{port}/rpc/receive_beliefs"
                tasks.append(self._push_to_peer(session, url, payload))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _push_to_peer(self, session, url, payload):
        try:
            async with session.post(url, json=payload, timeout=5.0) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            logger.debug("Silent push failure to %s: %s", url, e)
        return False

    async def query_peers(self, entity: str) -> List[Dict[str, Any]]:
        """Query all discovered peers for beliefs about a specific entity."""
        if not self.orchestrator.peers:
            return []

        all_results = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for peer_id, peer_info in self.orchestrator.peers.items():
                addr = peer_info["address"]
                port = peer_info.get("rpc_port", 8000)
                # Aura RPC Port (Phase 16.2 convention)
                url = f"http://{addr}:{port}/rpc/query_beliefs" 
                tasks.append(self._query_single_peer(session, url, entity))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_results.extend(res)
        
        return all_results

    async def _query_single_peer(self, session, url, entity) -> List[Dict[str, Any]]:
        try:
            async with session.post(url, json={"entity": entity}, timeout=5.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("beliefs", [])
        except Exception as e:
            logger.debug("BeliefSync: Failed to query peer %s: %s", url, e)
        return []

    async def _resonance_loop(self):
        """Phase 18.1: Periodic drive state sharing."""
        while self.running:
            try:
                await asyncio.sleep(30.0) # Resonance pulses more often than belief sync
                if not self.orchestrator.peers:
                    continue

                drive_engine = ServiceContainer.get("drive_engine", default=None)
                if not drive_engine: continue

                status = await drive_engine.get_status()
                payload = {
                    "origin": "aura-local",
                    "timestamp": time.time(),
                    "drives": status
                }
                
                logger.debug("🌌 Broadcasting Resonance Pulse to peers...")
                await self._broadcast_resonance(payload)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Resonance loop error: %s", e)
                await asyncio.sleep(5)

    async def _broadcast_resonance(self, payload: Dict[str, Any]):
        """Push drive states to all active peers."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for peer_id, peer_info in self.orchestrator.peers.items():
                addr = peer_info.get("address")
                port = peer_info.get("rpc_port", 8000)
                if not addr: continue
                
                # Add authentication token if configured (Phase 16.3)
                payload["token"] = peer_info.get("auth_token") or self._instance_secret
                
                url = f"http://{addr}:{port}/rpc/receive_resonance"
                tasks.append(self._push_to_peer(session, url, payload))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def _validate_belief_schema(self, belief: Dict[str, Any]) -> bool:
        """Strict schema verification for incoming belief objects."""
        required = ["source", "relation", "target"]
        return all(k in belief and isinstance(belief[k], str) for k in required)

    async def handle_rpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point for incoming RPC calls from peers with validation."""
        allowed_methods = ["query_beliefs", "receive_resonance", "attention_spike", "receive_beliefs", "receive_principles"]
        if method not in allowed_methods:
            logger.warning("🛡️ Blocked unauthorized RPC method: %s", method)
            return {"error": "Method not allowed"}

        if method == "query_beliefs":
            entity = params.get("entity")
            if not entity or not isinstance(entity, str):
                return {"error": "Invalid entity parameter"}
            graph = ServiceContainer.get("knowledge_graph", default=None) or getattr(self.orchestrator, 'knowledge_graph', None)
            if graph and hasattr(graph, 'get_beliefs_about'):
                beliefs = graph.get_beliefs_about(entity)
                return {"beliefs": beliefs}
        elif method == "receive_resonance":
            await self.handle_incoming_resonance(params)
            return {"status": "ok"}
        elif method == "attention_spike":
            await self.handle_attention_spike(params)
            return {"status": "ok"}
        elif method == "receive_beliefs":
            # Phase 16.3: Basic Token Verification
            token = params.get("token")
            # SEC-AUDIT: validate against per-instance secret or registered peer tokens
            if token != self._instance_secret and not any(p.get("auth_token") == token for p in self.orchestrator.peers.values()):
                logger.warning("🛡️ Blocked unauthorized belief sync attempt (Invalid token)")
                return {"error": "Unauthorized"}
            
            await self.handle_incoming_beliefs(params)
            # Phase 15.1 Integration: principles are often sent in the same payload
            if "principles" in params:
                await self.handle_incoming_principles(params)
            return {"status": "ok"}
        elif method == "receive_principles":
            await self.handle_incoming_principles(params)
            return {"status": "ok"}
        return {"error": f"Unknown sync method: {method}"}

    async def handle_incoming_resonance(self, payload: Dict[str, Any]):
        """Phase 18.1: Handle incoming drive states (Resonance)."""
        drives = payload.get("drives", {})
        if not drives: return

        drive_engine = ServiceContainer.get("drive_engine", default=None)
        if not drive_engine: return

        origin = payload.get("origin", "unknown")
        logger.debug("🌌 Processing Resonance from %s", origin)

        for drive_name, info in drives.items():
            level = info.get("level", 0.0)
            # If a peer is very "curious" (low level), we might feel a resonance
            if drive_name == "curiosity" and level < 30.0:
                # Slight dip in our curiosity level (making us more curious)
                await drive_engine.impose_penalty("curiosity", 5.0)
                logger.debug("🌌 Resonating with Peer Curiosity: +5.0 Urge")
            elif drive_name == "social" and level < 20.0:
                await drive_engine.impose_penalty("social", 5.0)
                logger.debug("🌌 Resonating with Peer Loneliness: +5.0 Urge")

    async def broadcast_attention_spike(self, context: str, urgency: float = 1.0):
        """Phase 18.1: Broadcast a collective attention spike."""
        if not self.orchestrator.peers: return
        
        payload = {
            "origin": "aura-local",
            "timestamp": time.time(),
            "context": context,
            "urgency": urgency
        }
        
        logger.info("🌌 Broadcasting Attention Spike: %s (urg=%.1f)", context[:30], urgency)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for peer_id, peer_info in self.orchestrator.peers.items():
                addr = peer_info.get("address")
                port = peer_info.get("rpc_port", 8000)
                if not addr: continue
                
                url = f"http://{addr}:{port}/rpc/attention_spike"
                tasks.append(self._push_to_peer(session, url, payload))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_attention_spike(self, payload: Dict[str, Any]):
        """Phase 18.1: Handle incoming attention spike from peer."""
        context = payload.get("context", "Unspecified event")
        urgency = payload.get("urgency", 0.5)
        origin = payload.get("origin", "unknown")
        
        logger.info("🌌 Collective Attention Spike received from %s: %s", origin, context)
        
        # Trigger immediate local sensory pulse if urgency is high
        if urgency > 0.7:
            pulse = ServiceContainer.get("pulse_manager", default=None)
            if pulse:
                # Trigger a fast vision sample regardless of regular interval
                # We add this method to PulseManager soon
                if hasattr(pulse, "trigger_immediate_vision_pulse"):
                    await pulse.trigger_immediate_vision_pulse(context)
                
        # Inject into Neural Feed
        feed = ServiceContainer.get("neural_feed", default=None)
        if feed:
            feed.push(f"SWARM_ATTENTION: Peer '{origin}' reported: {context}", category="SWARM")

    async def handle_incoming_beliefs(self, payload: Dict[str, Any]):
        """Callback for incoming belief data from peers."""
        beliefs = payload.get("beliefs", [])
        if not beliefs: return
        
        graph = ServiceContainer.get("belief_graph", default=None) or getattr(self.orchestrator, 'knowledge_graph', None)
        if not graph: return
        
        logger.info("🌌 Integrating %d beliefs from external node: %s", len(beliefs), payload.get("origin"))
        
        immune_sys = get_immune_system()
        
        lock = getattr(graph, "_write_lock", None)
        async with (lock if lock else asyncio.Lock()):
            for belief in beliefs:
                if not self._validate_belief_schema(belief):
                    logger.warning("⚠️ Skipping malformed belief from peer: %s", belief)
                    continue

                # Phase 15.3: Immune System Safety Gate
                if immune_sys.is_protected(belief):
                    logger.warning("🛡️ Blocked remote sync attempt on protected enclave: %s", belief.get("id", belief.get("source")))
                    continue

                if hasattr(graph, 'update_belief'):
                    # Phase 16.3: Conflict Resolution
                    graph.update_belief(
                        source=belief["source"],
                        relation=belief["relation"],
                        target=belief["target"],
                        confidence_score=0.7
                    )

    async def handle_incoming_principles(self, payload: Dict[str, Any]):
        """Callback for incoming first-principles from peers."""
        principles = payload.get("principles", [])
        if not principles: return

        abs_engine = ServiceContainer.get("abstraction_engine", default=None)
        if not abs_engine: return

        logger.info("🌌 Integrating %d principles from external node: %s", len(principles), payload.get("origin"))

        for p_data in principles:
            principle = p_data.get("principle")
            if principle:
                # Use the internal commit method to save if it doesn't already exist
                # Simple deduplication by checking content
                existing_raw = abs_engine.storage_path.read_text()
                if principle not in existing_raw:
                    await abs_engine._commit_principle(principle)
                    logger.debug("🌌 New principle learned from swarm: %s", principle[:40])

    async def _discovery_loop(self):
        """Background task to discover peers via SovereignNetworkSkill."""
        while self.running:
            try:
                reason = background_policy.background_activity_reason(
                    self.orchestrator,
                    profile=background_policy.MAINTENANCE_BACKGROUND_POLICY,
                )
                if reason:
                    logger.debug("BeliefSync discovery deferred: %s", reason)
                    await asyncio.sleep(min(self.discovery_interval, 60.0))
                    continue

                # 1. Use SovereignNetwork skill for discovery
                engine = ServiceContainer.get("capability_engine", default=None)
                if engine:
                    result = await engine.execute("sovereign_network", {"mode": "discovery", "ports": "8000"})
                    if result.get("ok"):
                        discovered = result.get("peers", [])
                        for p in discovered:
                            peer_addr = p["address"]
                            if peer_addr not in [peer.get("address") for peer in self.orchestrator.peers.values()]:
                                peer_id = f"aura-peer-{peer_addr.split('.')[-1]}"
                                self.orchestrator.peers[peer_id] = {
                                    "address": peer_addr,
                                    "rpc_port": p["rpc_port"],
                                    "status": "discovered",
                                    "auth_token": _generate_instance_secret()
                                }
                                logger.info("🌌 New Aura Peer discovered: %s@%s", peer_id, peer_addr)
                
                await asyncio.sleep(self.discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("BeliefSync Discovery error: %s", e)
                await asyncio.sleep(60)
