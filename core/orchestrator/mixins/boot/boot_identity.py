import logging
import os
from pathlib import Path
from typing import Any

from core.config import config
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.service_names import ServiceNames

logger = logging.getLogger(__name__)

_BOOT_IDENTITY_BOUNDARY_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    RuntimeError,
    OSError,
    TypeError,
    ValueError,
)


def _record_identity_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    """Record degradation inside boot identity initialization."""
    record_degradation("boot_identity", exc, severity=severity, action=action)



def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        if getattr(self, "_fictional_synthesis_initialized", False):
            logger.info("🎬 Fictional Engine Synthesis already initialized.")
            return
        if getattr(self, "_fictional_synthesis_initializing", False):
            logger.info("🎬 Fictional Engine Synthesis initialization already in progress.")
            return
        self._fictional_synthesis_initializing = True
        try:
            from core.fictional_ai_synthesis import register_all_fictional_engines

            existing_engines = getattr(self, "fictional_engines", None)
            if existing_engines:
                self.fictional_engines = existing_engines
                logger.info("🎬 Fictional engines already registered; reusing supervised instances.")
            else:
                self.fictional_engines = register_all_fictional_engines(orchestrator=self)

            from core.orchestrator.initializers.derived_engines import (
                register_derived_engines,
            )

            existing_derived = getattr(self, "derived_engines", None)
            if existing_derived:
                self.derived_engines = existing_derived
                logger.info("🎬 Derived engines already registered; reusing instances.")
            else:
                from core.brain.llm.semantic_neural_serving import prepare_semantic_neural_serving

                await prepare_semantic_neural_serving()
                self.derived_engines = register_derived_engines(orchestrator=self)

            from core.agency.latent_distiller import LatentSpaceDistiller
            from core.memory.snap_kv_evictor import SnapKVEvictor
            from core.self_modification.shadow_ast_healer import ShadowASTHealer

            # Register SOTA sub-components
            self.ast_healer = ShadowASTHealer(
                codebase_root=Path(config.paths.base_dir)
            )
            self.kv_evictor = SnapKVEvictor()
            self.latent_distiller = LatentSpaceDistiller(
                memory_provider=ServiceContainer.get("memory_provider", default=None)
            )

            logger.info("🎬 Fictional Engine Synthesis Complete (JARVIS-class online)")
            self._fictional_synthesis_initialized = True
        except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
            _record_identity_degradation(e, action="continued boot without registering JARVIS-class fictional engines", severity="error")
            logger.error("🎬 Fictional Engine Synthesis failed: %s", e)
        finally:
            self._fictional_synthesis_initializing = False

    async def _init_self_modification_engine(self):
        """Initialize the Self-Modification Engine."""
        if _env_flag("AURA_FOREGROUND_ONLY", False) or not _env_flag("AURA_ENABLE_SELF_MODIFICATION_ENGINE", True):
            logger.info("Self-Modification Engine disabled for foreground-only boot.")
            self.self_modifier = None
            return
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
                # Monitoring is safe in every full runtime: promotion remains
                # separately gated, while proposal-only mode still diagnoses,
                # validates, and quarantines repairs produced from real faults.
                modifier.start_monitoring()
                if modifier.runtime_promotion_enabled():
                    logger.info("🧬 Self-Modification Engine active with supervised promotion")
                else:
                    logger.info(
                        "🧬 Self-Modification Engine active in validation/quarantine mode; "
                        "source promotion requires a supervised operator profile."
                    )
        except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
            _record_identity_degradation(e, action="continued boot with disabled self-modification engine", severity="error")
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
        except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
            _record_identity_degradation(e, action="continued boot with unguarded OutputGate", severity="error")
            logger.error("Identity Guard initialization failed: %s", e)

    async def _init_persona_evolver(self):
        """Initialize the Persona Evolver (Phase 12 Evolution)."""
        try:
            from core.evolution.persona_evolver import PersonaEvolver

            self.persona_evolver = PersonaEvolver(self)
            ServiceContainer.register_instance("persona_evolver", self.persona_evolver)
            logger.info("🧬 Persona Evolver initialized (waiting for heartbeat)")
        except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
            _record_identity_degradation(e, action="continued boot with disabled persona evolution", severity="error")
            logger.error("Failed to init Persona Evolver: %s", e)
            self.persona_evolver = None

        logger.info("🛠️  _init_autonomous_evolution complete")

    def _initialize_moral_systems(self):
        """Integrate moral agency and sensory systems."""
        try:
            from core.soul import Soul

            try:
                from core.morality.master_moral_integration import (
                    integrate_complete_moral_and_sensory_systems,
                )
            except ImportError:
                integrate_complete_moral_and_sensory_systems = None

            self.soul = ServiceContainer.get(ServiceNames.SOUL, default=None)
            if self.soul is None:
                self.soul = Soul(self)
            # LIVE DEFECT, 2026-07-27. The Soul was constructed here and never
            # published to the service spine, so ServiceContainer.get("soul")
            # answered None for the whole life of the process.
            #
            # Two things went wrong downstream. The turn-engagement tracker
            # reported "soul absent for 12 consecutive turns: identity
            # continuity across turns and restarts" as a CRITICAL fault on
            # every conversation — correctly, and permanently, because no
            # amount of waiting was going to register it.
            #
            # Worse, get_panzer_soul() looks the service up and silently
            # substitutes a PanzerSoulProxy carrying `logic = None` when it is
            # missing. PersonalityEngine has therefore been holding a metadata
            # shell rather than the real drive system — an absence presented
            # as a presence, which is the bug class this codebase keeps
            # finding.
            #
            # It is constructed; publishing it is the whole fix.
            ServiceContainer.register_instance(
                ServiceNames.SOUL, self.soul, required=False,
            )
            if integrate_complete_moral_and_sensory_systems:
                integrate_complete_moral_and_sensory_systems(self)

            # H-28 Rename to match frontend HUD expectations
            moral = ServiceContainer.get("moral_reasoning")
            social = ServiceContainer.get("theory_of_mind")
            ServiceContainer.register_instance("moral", moral)
            ServiceContainer.register_instance("social", social)
        except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
            _record_identity_degradation(e, action="continued boot with missing soul/moral system integration", severity="error")
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
                except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
                    _record_identity_degradation(e, action="skipped attaching consciousness contract to boot runtime")
                    logger.debug("Failed to attach consciousness contract: %s", e)

            # 4. Liquid Substrate Bridge (v6 Integration)
            try:
                from core.consciousness.liquid_substrate_bridge import (
                    bridge_to_orchestrator,
                )

                bridge_to_orchestrator(self)
            except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
                _record_identity_degradation(e, action="continued boot with disconnected liquid substrate bridge", severity="error")
                logger.error("Liquid Substrate bridge failed: %s", e, exc_info=True)

            # 5. Moral Agency & Personality (The 'Soul')
            # integrate_complete_moral_and_sensory_systems is now handled in _integrate_systems only
            pass  # no-op: intentional

            logger.info("✓ Core Architecture ACTIVE")

        except _BOOT_IDENTITY_BOUNDARY_ERRORS as e:
            _record_identity_degradation(e, action="continued boot with degraded Core Architecture configuration", severity="error")
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
