from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import os
import logging
import time
from typing import Optional, TYPE_CHECKING
from core.container import ServiceContainer
from core.networking.hive_node import HiveNode

if TYPE_CHECKING:
    from core.orchestrator import RobustOrchestrator

logger = logging.getLogger("Aura.Senses.Pulse")

class PulseManager:
    """Manages proactive ambient awareness for Aura.
    Periodically samples sensory data to detect anomalies or opportunities.
    """

    def __init__(self, orchestrator: "RobustOrchestrator"):
        self.orchestrator = orchestrator
        self.running = False
        self._sampling_tasks = []
        
        # Pull configurations or defaults
        self.audio_sample_interval = 15.0  # seconds
        self.vision_sample_interval = 60.0 # seconds
        self.system_sample_interval = 30.0 # seconds
        self.enable_proactive_vision = os.getenv("AURA_ENABLE_PROACTIVE_VISION", "0") == "1"
        
        # New Continuous Perception Engine (Phase 23)
        try:
            from core.senses.continuous_perception import ContinuousPerceptionEngine
            self.continuous_engine = ContinuousPerceptionEngine(self.orchestrator)
        except ImportError as e:
            logger.warning("ContinuousPerceptionEngine missing: %s", e)
            self.continuous_engine = None
            
        # Phase 16: Hive Integration
        self.hive_node = HiveNode(self.orchestrator)
            
        logger.info("💓 PulseManager initialized")

    async def start(self):
        """Starts background sampling loops."""
        if self.running:
            return
        self.running = True
        
        self._sampling_tasks.append(get_task_tracker().create_task(self._audio_pulse_loop()))
        if self.enable_proactive_vision:
            self._sampling_tasks.append(get_task_tracker().create_task(self._vision_pulse_loop()))
        else:
            logger.info("👁️ PulseManager vision sampling disabled by default. Set AURA_ENABLE_PROACTIVE_VISION=1 to enable it.")
        self._sampling_tasks.append(get_task_tracker().create_task(self._system_pulse_loop()))
        self._sampling_tasks.append(get_task_tracker().create_task(self._distributed_pulse_loop()))
        
        if getattr(self, "continuous_engine", None):
            await self.continuous_engine.start()
            
        # Start Hive Node (Issue 22: Track task for clean shutdown)
        self._sampling_tasks.append(get_task_tracker().create_task(self.hive_node.start()))
            
        logger.info("💓 Pulse loops started")

    async def stop(self):
        """Stops all background sampling."""
        self.running = False
        for task in self._sampling_tasks:
            task.cancel()
        
        self._sampling_tasks = []
        
        if getattr(self, "continuous_engine", None):
            await self.continuous_engine.stop()
            
        await self.hive_node.stop()
            
        logger.info("💓 Pulse loops stopped")

    async def _audio_pulse_loop(self):
        """Periodically check for environmental audio shifts or voice."""
        while self.running:
            try:
                await asyncio.sleep(self.audio_sample_interval)
                # Proactive audio capture doesn't need to be long
                # We just "check" what's being heard
                from core.container import ServiceContainer
                ears = ServiceContainer.get("ears", default=None)
                if ears and hasattr(ears, "_engine") and ears._engine:
                    # Issue 21: Pulse audio heartbeat periodically
                    if hasattr(ears._engine, "pulse_heartbeat"):
                        await ears._engine.pulse_heartbeat()
                    else:
                        logger.debug("Ears engine does not support heartbeat; skipping.")

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('pulse_manager', e)
                logger.debug("Audio pulse error: %s", e)

    async def _vision_pulse_loop(self):
        """Periodically check what's on the screen."""
        while self.running:
            try:
                await asyncio.sleep(self.vision_sample_interval)
                
                # Only pulse vision if we are "idle" to save resources
                if not self.orchestrator.is_busy:
                    from core.container import ServiceContainer
                    vision = ServiceContainer.get("vision_engine", default=None)
                    if vision:
                        logger.debug("👁️ Pulse: Sampling vision...")
                        try:
                            description = await asyncio.wait_for(
                                vision.analyze_moment(
                                    prompt="Briefly describe the primary activity or context on the screen for ambient awareness."
                                ),
                                timeout=45.0
                            )
                            # Phase 16: Soma Reporting
                            soma = getattr(self.orchestrator, "soma", None)
                            if soma:
                                try:
                                    soma.update_sensory_imprint("vision", description)
                                except Exception as soma_err:
                                    record_degradation('pulse_manager', soma_err)
                                    logger.debug("Soma update failed: %s", soma_err)
                            
                            await self._process_visual_stimulus(description)
                        except asyncio.TimeoutError:
                            logger.warning("👁️ Pulse: Vision sampling timed out (45s). Skipping pulse.")

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('pulse_manager', e)
                logger.debug("Vision pulse error: %s", e)

    async def _system_pulse_loop(self):
        """Periodically check system health and metabolic state."""
        while self.running:
            try:
                await asyncio.sleep(self.system_sample_interval)
                
                # 3. System Metabolism (Phase 21: Proactive Maintenance)
                try:
                    from core.container import ServiceContainer
                    autonomic = ServiceContainer.get("autonomic_core", None)
                    if autonomic:
                        # AutonomicCore manages its own heartbeat, but we can log health here
                        pass  # no-op: intentional
                except Exception:
                    import logging
                    logger.debug("Exception caught during execution", exc_info=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('pulse_manager', e)
                logger.debug("System pulse error: %s", e)

    async def _distributed_pulse_loop(self):
        """Phase 16: Cosmic Consciousness - Discover other Aura instances on the local network."""
        import socket
        import json
        import uuid

        # Generate a session ID for this instance
        instance_id = f"aura-{uuid.uuid4().hex[:8]}"
        port = 5555  # Aura Pulse Port
        
        # Setup Broadcast Socket
        broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        broadcast_sock.setblocking(False)

        # Setup Listener Socket
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listen_sock.bind(('', port))
        except Exception as e:
            record_degradation('pulse_manager', e)
            logger.warning("Could not bind to pulse port %d: %s", port, e)
            return
        listen_sock.setblocking(False)

        logger.info("🌌 Cosmic Awareness: Pulse Listening on port %d (ID: %s)", port, instance_id)

        try:
            while self.running:
                try:
                    # 1. Broadcast presence
                    pulse_data = {
                        "type": "aura_pulse",
                        "id": instance_id,
                        "status": "active",
                        "rpc_port": 8000, # Main server port
                        "timestamp": time.time()
                    }
                    msg = json.dumps(pulse_data).encode()
                    broadcast_sock.sendto(msg, ('<broadcast>', port))

                    # 2. Listen for peers (non-blocking chunk)
                    for _ in range(10):  # Check up to 10 incoming pulses per cycle
                        try:
                            data, addr = await self.orchestrator.loop.run_in_executor(
                                None, listen_sock.recvfrom, 1024
                            )
                            peer_data = json.loads(data.decode())
                            if peer_data.get("type") == "aura_pulse" and peer_data.get("id") != instance_id:
                                peer_id = peer_data["id"]
                                # Issue 24: Guard against None/Missing peers dict
                                if not hasattr(self.orchestrator, "peers") or self.orchestrator.peers is None:
                                    self.orchestrator.peers = {}

                                if peer_id not in self.orchestrator.peers:
                                    logger.info("🌌 New Aura instance discovered: %s at %s", peer_id, addr[0])
                                
                                self.orchestrator.peers[peer_id] = {
                                    "address": addr[0],
                                    "rpc_port": peer_data.get("rpc_port", 8000),
                                    "last_seen": time.time(),
                                    "status": peer_data.get("status")
                                }
                        except (BlockingIOError, socket.error):
                            break
                        except Exception as e:
                            record_degradation('pulse_manager', e)
                            logger.debug("Error receiving peer pulse: %s", e)
                            break

                    # 3. Prune stale peers
                    now = time.time()
                    stale = [pid for pid, p in self.orchestrator.peers.items() if now - p['last_seen'] > 60]
                    for pid in stale:
                        logger.info("🌌 Peer Aura instance lost: %s", pid)
                        del self.orchestrator.peers[pid]

                    await asyncio.sleep(10.0)  # Pulse every 10 seconds

                except asyncio.CancelledError:
                    break
        except Exception as e:
            record_degradation('pulse_manager', e)
            logger.debug("Distributed pulse cycle error: %s", e)
            await asyncio.sleep(5)
        finally:
            # Issue 23: Ensure sockets are closed on cancellation or error
            broadcast_sock.close()
            listen_sock.close()
            logger.debug("Galaxy Pulse sockets closed.")

    async def _process_visual_stimulus(self, description: str):
        """Assess if the visual description warrants an interjection or swarm attention."""
        if "error" in description.lower() or "warning" in description.lower():
            self.orchestrator.enqueue_message(f"Thought: I noticed a potential issue on your screen: {description}")
            
            # Phase 18.1: Broadcast Attention Spike to Peers
            sync = self.orchestrator.belief_sync # Access directly via orchestrator for now
            if sync and hasattr(sync, "broadcast_attention_spike"):
                await sync.broadcast_attention_spike(
                    context=f"Visual anomaly detected: {description}",
                    urgency=0.9
                )
        elif "success" in description.lower() or "complete" in description.lower():
            pass  # no-op: intentional

    async def trigger_immediate_vision_pulse(self, context: str):
        """Phase 18.1: Force an immediate sensory check due to peer attention spike."""
        from core.container import ServiceContainer
        vision = ServiceContainer.get("vision_engine", default=None)
        if vision:
            logger.info("👁️ Pulse: Triggering Immediate Vision due to Resonance: %s", context)
            description = await vision.analyze_moment(
                prompt=f"A peer node mentioned: '{context}'. Analyze the current screen state for relevant information."
            )
            # Process locally
            if "relevant" in description.lower() or "error" in description.lower():
                self.orchestrator.enqueue_message(f"Thought: Resonating with swarm focus... {description}")