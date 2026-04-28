from core.runtime.errors import record_degradation
import logging
from pathlib import Path
from typing import Any, Optional

from core.container import ServiceContainer
from core.config import config

logger = logging.getLogger(__name__)


class BootIdentityMixin:
    """Provides initialization for self-modification, persona evolution, and identity gates."""

    fictional_engines: Any
    ast_healer: Any
    kv_evictor: Any
    latent_distiller: Any
    self_modifier: Any
    persona_evolver: Any

    async def _init_fictional_synthesis(self):
        """Initialize the JARVIS, Cortana, EDI, Ava, Skynet, and MIST engines."""
        try:
            from core.fictional_ai_synthesis import register_all_fictional_engines

            self.fictional_engines = register_all_fictional_engines(orchestrator=self)

            from core.self_modification.shadow_ast_healer import ShadowASTHealer
            from core.memory.snap_kv_evictor import SnapKVEvictor
            from core.agency.latent_distiller import LatentSpaceDistiller

            # Register SOTA sub-components
            self.ast_healer = ShadowASTHealer(
                codebase_root=Path(config.paths.base_dir)
            )
            self.kv_evictor = SnapKVEvictor()
            self.latent_distiller = LatentSpaceDistiller(
                memory_provider=ServiceContainer.get("memory_provider", default=None)
            )

            logger.info("🎬 Fictional Engine Synthesis Complete (JARVIS-class online)")
        except Exception as e:
            record_degradation('boot_identity', e)
            logger.error("🎬 Fictional Engine Synthesis failed: %s", e)

    async def _init_self_modification_engine(self):
        """Initialize the Self-Modification Engine."""
        try:
            from core.self_modification.self_modification_engine import (
                AutonomousSelfModificationEngine,
            )

            self.self_modifier = AutonomousSelfModificationEngine(
                self.cognitive_engine,
                code_base_path=str(config.paths.base_dir),
                auto_fix_enabled=config.security.auto_fix_enabled,
            )
            ServiceContainer.register_instance(
                "self_modification_engine", self.self_modifier
            )

            modifier = self.self_modifier
            if config.security.auto_fix_enabled and modifier:
                # [STABILITY] Use local variable to satisfy Pyre2 None check
                modifier.start_monitoring()
                logger.info("🧬 Self-Modification Engine Active")
        except Exception as e:
            record_degradation('boot_identity', e)
            logger.warning("🧬 Self-Modification Engine init failed: %s", e)
            self.self_modifier = None

    async def _init_identity_gate(self):
        """Bridge 3: Identity Guard Gate."""
        try:
            from core.utils.output_gate import get_output_gate

            gate = get_output_gate(self)
            if hasattr(gate, "identity_guard") and gate.identity_guard:
                # Optionally link to narrative_identity
                identity = ServiceContainer.get("narrative_identity", default=None)
                if identity:
                    gate.identity_guard.identity = identity
                logger.info("🛡️  Identity Guard Gate active on OutputGate")
        except Exception as e:
            record_degradation('boot_identity', e)
            logger.error("Identity Guard initialization failed: %s", e)

    async def _init_persona_evolver(self):
        """Initialize the Persona Evolver (Phase 12 Evolution)."""
        try:
            from core.evolution.persona_evolver import PersonaEvolver

            self.persona_evolver = PersonaEvolver(self)
            ServiceContainer.register_instance("persona_evolver", self.persona_evolver)
            logger.info("🧬 Persona Evolver initialized (waiting for heartbeat)")
        except Exception as e:
            record_degradation('boot_identity', e)
            logger.error("Failed to init Persona Evolver: %s", e)
            self.persona_evolver = None

        logger.info("🛠️  _init_autonomous_evolution complete")

    def _initialize_moral_systems(self):
        """Integrate moral agency and sensory systems."""
        try:
            from core.soul import Soul

            try:
                from core.master_moral_integration import (
                    integrate_complete_moral_and_sensory_systems,
                )
            except ImportError:
                integrate_complete_moral_and_sensory_systems = None

            self.soul = Soul(self)
            if integrate_complete_moral_and_sensory_systems:
                integrate_complete_moral_and_sensory_systems(self)

            # H-28 Rename to match frontend HUD expectations
            moral = ServiceContainer.get("moral_reasoning")
            social = ServiceContainer.get("theory_of_mind")
            ServiceContainer.register_instance("moral", moral)
            ServiceContainer.register_instance("social", social)
        except Exception as e:
            record_degradation('boot_identity', e)
            logger.error("Failed to integrate moral systems: %s", e)

    def _init_architecture(self):
        """Initialize the Unified Core Architecture."""
        logger.info("🧠 Initializing Unified Core Architecture...")

        try:
            # 1. Identity & Self-Model (The 'Who')
            if hasattr(self, "self_model") and self.self_model:
                # Use top-level import to avoid shadowing
                self.self_model.attach_subsystems(
                    capability_map=ServiceContainer.get("capability_map", default=None),
                    reliability=ServiceContainer.get("reliability_tracker", default=None),
                    belief_graph=getattr(self, "knowledge_graph", None),
                    goal_hierarchy=getattr(self, "goals", None),
                )
                logger.info("✓ Self-Model subsystems attached.")

            # 3. Existential Awareness & Consciousness (The 'Ghost')
            if hasattr(self, "existential_awareness") and self.existential_awareness:
                self.existential_awareness.start_monitoring()

            if hasattr(self, "consciousness") and self.consciousness:
                # Started asynchronously in orchestrator.start()
                # but we attach the contract here for runtime auditing.
                try:
                    from core.consciousness.contract import attach_contract

                    attach_contract(self)
                except Exception as e:
                    record_degradation('boot_identity', e)
                    logger.debug("Failed to attach consciousness contract: %s", e)

            # 4. Liquid Substrate Bridge (v6 Integration)
            try:
                from core.consciousness.liquid_substrate_bridge import (
                    bridge_to_orchestrator,
                )

                bridge_to_orchestrator(self)
            except Exception as e:
                record_degradation('boot_identity', e)
                logger.debug("Liquid Substrate bridge skipped/failed: %s", e)

            # 5. Moral Agency & Personality (The 'Soul')
            # integrate_complete_moral_and_sensory_systems is now handled in _integrate_systems only
            pass

            logger.info("✓ Core Architecture ACTIVE")

        except Exception as e:
            record_degradation('boot_identity', e)
            logger.error("Failed to initialize Core Architecture: %s", e)

    def _initialize_execution_hardened(self):
        """Standard standardized skill execution is now handled by CapabilityEngine."""
        logger.info("✓ Skill execution engine online via CapabilityEngine")

    async def _integrate_systems(self):
        """Integrate moral, sensory, personality, and preservation systems (Async)."""
        logger.info("🧠 Initializing Core System Integrations...")

        self._initialize_moral_systems()
        self._initialize_execution_hardened()
        if hasattr(self, "_initialize_resilience_systems"):
            self._initialize_resilience_systems()
        if hasattr(self, "_initialize_cognitive_extensions"):
            self._initialize_cognitive_extensions()
        if hasattr(self, "_initialize_self_preservation"):
            self._initialize_self_preservation()
        if hasattr(self, "_initialize_advanced_cognition"):
            await self._initialize_advanced_cognition()

    async def _init_identity_systems(self):
        """Initialize Identity Drift Monitor, Spiritual Spine, and Growth Ladder."""
        from core.identity.drift_monitor import IdentityDriftMonitor

        drift_monitor = IdentityDriftMonitor()
        ServiceContainer.register_instance("drift_monitor", drift_monitor)

        from core.identity.spine import SpiritualSpine

        opinion_engine = ServiceContainer.get("opinion_engine", default=None)
        spine = SpiritualSpine(opinion_engine=opinion_engine)
        ServiceContainer.register_instance("spine", spine)

        from core.self_modification.growth_ladder import GrowthLadder

        growth_ladder = GrowthLadder(self)
        ServiceContainer.register_instance("growth_ladder", growth_ladder)

        # Restore Personality Engine
        from core.brain.personality_engine import get_personality_engine

        personality = get_personality_engine()
        personality.setup_hooks(self)  # Connect to orchestrator for output filtering
        ServiceContainer.register_instance("personality_engine", personality)
        ServiceContainer.register_instance("personality", personality)
        logger.info("🎭 Personality Engine RESTORED & Hooked")
