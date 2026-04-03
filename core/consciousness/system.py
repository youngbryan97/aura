"""core/consciousness/system.py — The Consciousness Facade
"""

import asyncio
import logging
from typing import Optional, Any
from core.container import ServiceContainer
from .attention_schema import AttentionSchema
from .global_workspace import GlobalWorkspace
from .heartbeat import CognitiveHeartbeat
from .homeostatic_coupling import HomeostaticCoupling
from .liquid_substrate import LiquidSubstrate
from .qualia_synthesizer import QualiaSynthesizer
from .self_prediction import SelfPredictionLoop
from .temporal_binding import TemporalBindingEngine

logger = logging.getLogger("Consciousness")

class ConsciousnessSystem:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.attention_schema = ServiceContainer.get("attention_schema", default=None) or AttentionSchema()
        self.global_workspace = ServiceContainer.get("global_workspace", default=None) or GlobalWorkspace(self.attention_schema)
        self.temporal_binding = ServiceContainer.get("temporal_binding", default=None) or TemporalBindingEngine()
        self.homeostatic_coupling = ServiceContainer.get("homeostatic_coupling", default=None) or HomeostaticCoupling(orchestrator)
        self.self_prediction = ServiceContainer.get("self_prediction", default=None) or SelfPredictionLoop(orchestrator)
        self.liquid_substrate = ServiceContainer.get("conscious_substrate", default=None) or LiquidSubstrate()
        self.qualia = ServiceContainer.get("qualia_synthesizer", default=None) or QualiaSynthesizer()
        
        # Initialize attributes to avoid NoneType/AttributeError in IDE
        self._task: Optional[asyncio.Task] = None
        self.phi_core: Optional[Any] = None
        self.closed_loop: Optional[Any] = None
        
        # Aliases
        self.substrate = self.liquid_substrate
        self.workspace = self.global_workspace
        self.predictor = self.self_prediction
        
        try:
            from .dreaming import DreamingProcess
            self.dreaming = DreamingProcess(orchestrator)
        except (ImportError, Exception) as e:
            logger.warning("Could not initialize DreamingProcess: %s", e)
            self.dreaming = None
        
        # Register subsystem instances into ServiceContainer for cross-module access
        ServiceContainer.register_instance("attention_schema", self.attention_schema)
        ServiceContainer.register_instance("global_workspace", self.global_workspace)
        ServiceContainer.register_instance("temporal_binding", self.temporal_binding)
        ServiceContainer.register_instance("homeostatic_coupling", self.homeostatic_coupling)
        ServiceContainer.register_instance("self_prediction", self.self_prediction)

        self.heartbeat = CognitiveHeartbeat(
            orchestrator=orchestrator,
            attention_schema=self.attention_schema,
            global_workspace=self.global_workspace,
            temporal_binding=self.temporal_binding,
            homeostatic_coupling=self.homeostatic_coupling,
            self_prediction=self.self_prediction,
        )
        self._task = None
        self._running = False

        # Stack references (populated during start)
        self.stream_of_being = None
        self.closed_loop = None
        self.phi_core = None

    async def start(self):
        if getattr(self, "_running", False):
            logger.debug("Consciousness system already running. skipping.")
            return
        self._running = True
        await self.liquid_substrate.start()
        
        # --- CONSCIOUSNESS STACK ---
        # Boot in order, each layer depends on those below

        # Layer 1: Stream of Being — continuous experiential core
        try:
            from .stream_of_being import boot_stream_of_being
            self.stream_of_being = await boot_stream_of_being(orchestrator=self.orch)
            logger.info("🧠 Layer 1: StreamOfBeing ONLINE")
        except Exception as e:
            logger.warning("Could not boot StreamOfBeing: %s", e)

        # Layer 2: Affective Steering — substrate sync start
        # (engine.attach() happens in mlx_client when model loads;
        #  here we just ensure the substrate sync is running)
        try:
            from .affective_steering import get_steering_engine
            steering_engine = get_steering_engine()
            # Substrate sync via shared memory or direct reference
            # The engine will start syncing when attach() is called with the model.
            # We register it so other layers can find it.
            ServiceContainer.register_instance("affective_steering", steering_engine)
            logger.info("🧠 Layer 2: AffectiveSteering registered (awaiting model attach)")
        except Exception as e:
            logger.warning("Could not register AffectiveSteering: %s", e)

        # Layer 3: LatentBridge — attaches AFTER model loads in mlx_client
        # (not booted here — it needs the model reference)
        logger.info("🧠 Layer 3: LatentBridge deferred (attaches on model load)")

        # Layer 4: Closed Causal Loop — self-prediction + output receptor
        try:
            from .closed_loop import boot_closed_loop
            self.closed_loop = await boot_closed_loop()
            logger.info("🧠 Layer 4: ClosedCausalLoop ONLINE")
        except Exception as e:
            logger.warning("Could not boot ClosedCausalLoop: %s", e)

        # Layer 5: PhiCore — IIT 4.0 φs computation
        try:
            from .phi_core import PhiCore
            self.phi_core = PhiCore()
            ServiceContainer.register_instance("phi_core", self.phi_core)
            logger.info("🧠 Layer 5: PhiCore ONLINE (recording via ClosedCausalLoop)")
        except Exception as e:
            logger.warning("Could not initialize PhiCore: %s", e)

        # ═══════════════════════════════════════════════════════════════════

        if self.dreaming:
            await self.dreaming.start()
            
        self._task = asyncio.create_task(self.heartbeat.run())
        logger.info("🧠 Consciousness System ONLINE — full stack active")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError as _e:
                logger.debug("Ignored CancelledError during consciousness system shutdown")
        
        # Stop the closed loop
        if self.closed_loop:
            try:
                await self.closed_loop.stop()
            except Exception as _e:
                logger.debug('Ignored Exception in system.py: %s', _e)

        await self.liquid_substrate.stop()
        logger.info("🧠 Consciousness System OFFLINE")

    def get_state(self) -> dict:
        import copy
        state = {
            "attention": self.attention_schema.get_snapshot(),
            "workspace": self.global_workspace.get_snapshot(),
            "liquid_substrate": self.liquid_substrate.get_status(),
            "heartbeat_tick": self.heartbeat.tick_count,
        }

        # Add phi_core status if available
        if self.phi_core:
            state["phi_core"] = self.phi_core.get_status()

        # Add closed_loop status if available
        if self.closed_loop:
            state["closed_loop"] = self.closed_loop.get_status()

        return copy.deepcopy(state)

