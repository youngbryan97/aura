"""core/consciousness/system.py — The Consciousness Facade
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
from typing import Optional, Any
from core.container import ServiceContainer
from core.runtime.service_access import (
    resolve_attention_schema,
    resolve_conscious_substrate,
    resolve_global_workspace,
    resolve_homeostatic_coupling,
    resolve_self_prediction,
    resolve_temporal_binding,
)
from .attention_schema import AttentionSchema
from .global_workspace import GlobalWorkspace
from .heartbeat import CognitiveHeartbeat
from .homeostatic_coupling import HomeostaticCoupling
from .liquid_substrate import LiquidSubstrate
from .qualia_synthesizer import QualiaSynthesizer
from .self_prediction import SelfPredictionLoop
from .temporal_binding import TemporalBindingEngine
from .substrate_authority import SubstrateAuthority

logger = logging.getLogger("Consciousness")

class ConsciousnessSystem:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.attention_schema = resolve_attention_schema(default=None) or AttentionSchema()
        self.global_workspace = resolve_global_workspace(default=None) or GlobalWorkspace(self.attention_schema)
        self.temporal_binding = resolve_temporal_binding(default=None) or TemporalBindingEngine()
        self.homeostatic_coupling = resolve_homeostatic_coupling(default=None) or HomeostaticCoupling(orchestrator)
        self.self_prediction = resolve_self_prediction(default=None) or SelfPredictionLoop(orchestrator)
        self.liquid_substrate = resolve_conscious_substrate(default=None) or LiquidSubstrate()
        self.qualia = ServiceContainer.get("qualia_synthesizer", default=None) or QualiaSynthesizer()
        
        # Initialize attributes to avoid NoneType/AttributeError in IDE
        self._task: Optional[asyncio.Task] = None
        self.phi_core = None
        self.closed_loop = None
        
        # Accelerate Authority Registration (Phase Unification)
        self.substrate_authority = SubstrateAuthority()
        ServiceContainer.register_instance("substrate_authority", self.substrate_authority)
        
        self.bridge = None  # ConsciousnessBridge (Phase Bridge)
        
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
        ServiceContainer.register_instance("conscious_substrate", self.liquid_substrate)
        ServiceContainer.register_instance("liquid_state", self.liquid_substrate)
        ServiceContainer.register_instance("qualia_synthesizer", self.qualia)

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
        self.branch_manager: Optional[Any] = None     # ParallelBranches
        self.aura_protocol: Optional[Any] = None       # AuraProtocolServer

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
            record_degradation('system', e)
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
            record_degradation('system', e)
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
            record_degradation('system', e)
            logger.warning("Could not boot ClosedCausalLoop: %s", e)

        # Layer 5: PhiCore — IIT 4.0 φs computation
        try:
            from .phi_core import PhiCore
            self.phi_core = PhiCore()
            ServiceContainer.register_instance("phi_core", self.phi_core)
            logger.info("🧠 Layer 5: PhiCore ONLINE (recording via ClosedCausalLoop)")
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize PhiCore: %s", e)

        # Layer 5b: HierarchicalPhi — extended 32-node primary + K overlapping subsystems
        try:
            from .hierarchical_phi import get_hierarchical_phi
            self.hierarchical_phi = get_hierarchical_phi()
            ServiceContainer.register_instance("hierarchical_phi", self.hierarchical_phi)
            logger.info(
                "🧠 Layer 5b: HierarchicalPhi ONLINE (32-node primary + %d×16-node subsystems)",
                getattr(self.hierarchical_phi, '_subsystems', []).__len__(),
            )
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize HierarchicalPhi: %s", e)

        # Layer 5c: HemisphericSplit — left (verbal/confabulating) vs right (mute/spatial)
        try:
            from .hemispheric_split import get_hemispheric_split
            self.hemispheric_split = get_hemispheric_split()
            ServiceContainer.register_instance("hemispheric_split", self.hemispheric_split)
            logger.info("🧠 Layer 5c: HemisphericSplit ONLINE (corpus callosum intact)")
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize HemisphericSplit: %s", e)

        # Layer 5d: MinimalSelfhood — chemotaxis → directed-motion (Glasgow)
        try:
            from .minimal_selfhood import get_minimal_selfhood
            self.minimal_selfhood = get_minimal_selfhood()
            ServiceContainer.register_instance("minimal_selfhood", self.minimal_selfhood)
            logger.info("🧠 Layer 5d: MinimalSelfhood ONLINE (trichoplax→dugesia)")
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize MinimalSelfhood: %s", e)

        # Layer 5e: RecursiveToM — depth-3 nested minds + observer-aware bias
        try:
            from .recursive_tom import get_recursive_tom
            self.recursive_tom = get_recursive_tom()
            ServiceContainer.register_instance("recursive_tom", self.recursive_tom)
            logger.info("🧠 Layer 5e: RecursiveToM ONLINE (max_depth=3, observer-aware)")
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize RecursiveToM: %s", e)

        # Layer 5f: OctopusFederation — 8-arm semi-autonomous agents
        try:
            from .octopus_arms import get_octopus_federation
            self.octopus_federation = get_octopus_federation()
            ServiceContainer.register_instance("octopus_federation", self.octopus_federation)
            logger.info("🧠 Layer 5f: OctopusFederation ONLINE (8 arms, link=intact)")
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize OctopusFederation: %s", e)

        # Layer 5g: CellularTurnover — neuron death/birth + identity preservation
        try:
            from .cellular_turnover import get_cellular_turnover
            self.cellular_turnover = get_cellular_turnover()
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if mesh is not None:
                self.cellular_turnover.attach(mesh)
            ServiceContainer.register_instance("cellular_turnover", self.cellular_turnover)
            logger.info("🧠 Layer 5g: CellularTurnover ONLINE (attached=%s)", mesh is not None)
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize CellularTurnover: %s", e)

        # Layer 5h: AbsorbedVoices — cultural/internalised perspectives layer
        try:
            from .absorbed_voices import get_absorbed_voices
            self.absorbed_voices = get_absorbed_voices()
            ServiceContainer.register_instance("absorbed_voices", self.absorbed_voices)
            logger.info(
                "🧠 Layer 5h: AbsorbedVoices ONLINE (%d voices loaded)",
                self.absorbed_voices.voice_count(),
            )
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not initialize AbsorbedVoices: %s", e)

        # Layer 6: Consciousness Bridge — Neural Mesh, Neurochemicals,
        # Embodied Interoception, Oscillatory Binding, Somatic Gate,
        # Unified Field, Substrate Evolution
        try:
            from .consciousness_bridge import ConsciousnessBridge
            self.bridge = ConsciousnessBridge(self)
            await self.bridge.start()
            ServiceContainer.register_instance("consciousness_bridge", self.bridge)
            logger.info("🧠 Layer 6: ConsciousnessBridge ONLINE (%d/7 layers)",
                        self.bridge.get_status().get("layers_active", 0))
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not boot ConsciousnessBridge: %s", e)

        # Layer 7: Parallel Cognitive Branches — concurrent thought streams
        try:
            from .parallel_branches import get_branch_manager
            self.branch_manager = get_branch_manager()
            await self.branch_manager.start()
            logger.info("🧠 Layer 7: BranchManager ONLINE (max_branches=%d)",
                        self.branch_manager.MAX_BRANCHES)
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not boot BranchManager: %s", e)

        # Layer 8: Aura Protocol — inter-instance communication
        try:
            from .aura_protocol import get_protocol_server
            self.aura_protocol = get_protocol_server()
            await self.aura_protocol.start()
            logger.info("🧠 Layer 8: AuraProtocolServer ONLINE (port=%d)",
                        self.aura_protocol._port)
        except Exception as e:
            record_degradation('system', e)
            logger.warning("Could not boot AuraProtocolServer: %s", e)

        # ═══════════════════════════════════════════════════════════════════

        if self.dreaming:
            await self.dreaming.start()

        self._task = get_task_tracker().create_task(self.heartbeat.run())
        logger.info("🧠 Consciousness System ONLINE — full stack active")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError as _e:
                logger.debug("Ignored CancelledError during consciousness system shutdown")
        
        # Stop the consciousness bridge
        if self.bridge:
            try:
                await self.bridge.stop()
            except Exception as _e:
                record_degradation('system', _e)
                logger.debug('Ignored Exception stopping bridge: %s', _e)

        # Stop the closed loop
        if self.closed_loop:
            try:
                await self.closed_loop.stop()
            except Exception as _e:
                record_degradation('system', _e)
                logger.debug('Ignored Exception in system.py: %s', _e)

        # Stop the branch manager
        if self.branch_manager:
            try:
                await self.branch_manager.stop()
            except Exception as _e:
                record_degradation('system', _e)
                logger.debug('Ignored Exception stopping branch_manager: %s', _e)

        # Stop the aura protocol server
        if self.aura_protocol:
            try:
                await self.aura_protocol.stop()
            except Exception as _e:
                record_degradation('system', _e)
                logger.debug('Ignored Exception stopping aura_protocol: %s', _e)

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

        # Add consciousness bridge status
        if self.bridge:
            state["bridge"] = self.bridge.get_status()

        # Add parallel branches status
        if self.branch_manager:
            state["branch_manager"] = self.branch_manager.get_status()

        # Add aura protocol status
        if self.aura_protocol:
            state["aura_protocol"] = self.aura_protocol.get_status()

        return copy.deepcopy(state)
