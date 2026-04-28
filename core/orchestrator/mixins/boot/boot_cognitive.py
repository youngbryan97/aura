from core.runtime.errors import record_degradation
import asyncio
import inspect
import logging
from typing import Any, Optional

from core.container import ServiceContainer

logger = logging.getLogger(__name__)


class BootCognitiveMixin:
    """Provides initialization for cognitive engines, learning, and belief systems."""

    cognition: Any
    rsi_lab: Any
    concept_bridge: Any
    cryptolalia_decoder: Any
    ontology_genesis: Any
    morphic_forking: Any
    motivation: Any
    continuous_learner: Any
    react_loop: Any
    belief_sync: Any
    meta_learning: Any
    simulator: Any
    goal_hierarchy: Any
    aesthetic_critic: Any
    narrative_engine: Any
    learning_engine: Any
    hooks: Any

    async def _init_cognitive_core(self):
        """Initialize cognitive components — wire the CognitiveEngine to the LLM."""
        logger.info("🧠 Initializing Cognitive Core...")
        from core.brain.cognitive_engine import CognitiveEngine

        try:
            # 1. Resolve or Create CognitiveEngine
            ce = ServiceContainer.get("cognitive_engine", default=None)
            if not ce:
                logger.info("🧠 Creating fresh CognitiveEngine...")
                ce = CognitiveEngine()
                ServiceContainer.register_instance("cognitive_engine", ce)
                self.cognition = ce  # Use the setter
            
            # 2. Wire the engine to the skill router
            engine = ServiceContainer.get("capability_engine", default=None)
            assert engine is not None, (
                "_init_cognitive_core called before _init_skill_system. "
                "Fix _async_init_subsystems ordering."
            )

            if ce and (hasattr(ce, "wire") or hasattr(ce, "setup")):
                if hasattr(ce, "setup"):
                    setup_result = ce.setup(registry=engine, router=engine)
                else:
                    setup_result = ce.wire(engine, engine)
                if inspect.isawaitable(setup_result):
                    await setup_result
                logger.info("🧠 Cognitive Engine wired successfully.")
            else:
                logger.warning("⚠️  CognitiveEngine missing or incompatible.")

            # 3. Start API Adapter (LLM Clients)
            api_adapter = ServiceContainer.get("api_adapter", default=None)
            if api_adapter:
                logger.info("🧠 Starting API Adapter (LLM Infrastructure)...")
                await api_adapter.start()
                logger.info("🧠 API Adapter online.")
            else:
                logger.warning("⚠️  APIAdapter not found in ServiceContainer.")

            # Unified services are now handled in _async_init_subsystems
            pass
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Cognitive core wiring failed: %s", e, exc_info=True)

    async def _init_meta_learning(self):
        """Initialize Meta-Learning (RSI LAB)."""
        try:
            from research.meta_learning_loop import register_rsi_lab

            self.rsi_lab = register_rsi_lab(self)
            logger.info("🔬 RSI Lab online")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.debug("🔬 RSI Lab skipped: %s", e)

    async def _init_concept_bridge(self):
        """Initialize Concept Vector Bridge & Cryptolalia."""
        try:
            from core.brain.concept_vector_bridge import register_concept_bridge
            from core.brain.cryptolalia_decoder import register_cryptolalia_decoder

            self.concept_bridge = register_concept_bridge(self)
            decoder = register_cryptolalia_decoder(self)
            self.cryptolalia_decoder = decoder

            if hasattr(decoder, "init_routes"):
                decoder.init_routes()

            logger.info("🛰️  Cryptolalia Decoder online")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.warning("🛰️  Cryptolalia Decoder failed: %s", e)

    async def _init_advanced_ontology(self):
        """Initialize Advanced Ontology & Morphic Forking."""
        try:
            from core.brain.ontology_genesis import register_ontology_genesis

            self.ontology_genesis = register_ontology_genesis(self)

            from core.brain.morphic_forking import register_morphic_forking

            self.morphic_forking = register_morphic_forking(self)
            logger.info("🌑 Ontology & Morphic Forking online")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.warning("🌑 Ontology/Forking failed: %s", e)

    async def _init_belief_sync_subsystem(self):
        """Initialize the Belief Sync system."""
        try:
            from core.collective.belief_sync import BeliefSync

            self.belief_sync = BeliefSync(self)
            ServiceContainer.register_instance("belief_sync", self.belief_sync)
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("BeliefSync init failed: %s", e)
            ServiceContainer.register_instance("belief_sync", None)

    async def _init_react_loop(self):
        """Initialize the ReAct reasoning loop."""
        try:
            self.react_loop = ServiceContainer.get("react_loop", default=None)
            if not self.react_loop:
                from core.brain.react_loop import ReActLoop

                ce = ServiceContainer.get("cognitive_engine", default=None)
                self.react_loop = ReActLoop(brain=ce, orchestrator=self)
                ServiceContainer.register_instance("react_loop", self.react_loop)
            logger.info("✓ ReAct Loop online (Multi-step reasoning)")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to init ReAct Loop: %s", e)

    async def _init_continuous_learner(self):
        """Initialize the Continuous Learner for weight-level adaptation."""
        try:
            self.continuous_learner = ServiceContainer.get(
                "continuous_learner", default=None
            )
            if not self.continuous_learner:
                from core.learning.genuine_learning_pipeline import (
                    register_continuous_learner,
                )

                self.continuous_learner = register_continuous_learner(self)
                ServiceContainer.register_instance(
                    "continuous_learner", self.continuous_learner
                )
            logger.info("✓ Continuous Learner online")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to init Continuous Learner: %s", e)

    async def _init_live_learner(self):
        """Initialize the LiveLearner subsystem for weight-level evolution."""
        try:
            from core.learning.live_learner import (
                get_live_learner,
                patch_mlx_client_for_hot_swap,
            )

            ll = get_live_learner()

            # Apply hot-swap patch to MLX client if possible
            await patch_mlx_client_for_hot_swap()

            await ll.start()
            ServiceContainer.register_instance("live_learner", ll)
            logger.info("✓ Live Learner online and buffering")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("🛑 Live Learner init failed: %s", e)

    async def _initialize_advanced_cognition(self):
        """Initialize Advanced Cognitive Architecture (Async)."""
        logger.info("🧠 Initializing Advanced Cognitive Integration...")

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                from core.cognitive_integration_layer import CognitiveIntegrationLayer
                from core.config import config

                cil = CognitiveIntegrationLayer(
                    orchestrator=self, base_data_dir=str(config.paths.data_dir)
                )

                if hasattr(cil, "setup"):
                    cil.setup()

                # CRITICAL: Initialize the layer and its async sub-components
                success = await cil.initialize()

                if success and cil.is_active:
                    object.__setattr__(self, "_cognition_layer", cil)
                    if not getattr(ServiceContainer, "_registration_locked", False):
                        ServiceContainer.register_instance("cognitive_integration", cil)
                    else:
                        logger.warning(
                            "⚠️  ServiceContainer locked — skipping cognitive_integration re-registration (already set)"
                        )
                    logger.info(
                        "✅ Advanced Cognition active (attempt %d/%d)",
                        attempt,
                        max_attempts,
                    )

                    # Log sub-component status for diagnostics
                    status = cil.get_status()
                    logger.info(
                        "   Kernel: %s | Monologue: %s | LanguageCenter: %s",
                        "✅" if status.get("kernel_ready") else "❌",
                        "✅" if status.get("monologue_ready") else "⚠️",
                        "✅" if status.get("language_center_ready") else "⚠️",
                    )
                    # Register server for Skynet health monitoring
                    try:
                        from interface.server import app as server_app

                        if not getattr(ServiceContainer, "_registration_locked", False):
                            ServiceContainer.register_instance("server", server_app)
                        logger.info("📡 API Server registered in ServiceContainer.")
                    except ImportError:
                        logger.warning(
                            "📡 API Server (interface.server) not found for registration."
                        )

                    return  # Success — exit
                else:
                    logger.warning(
                        "⚠️  CIL.initialize() returned success=%s, is_active=%s (attempt %d/%d)",
                        success,
                        cil.is_active,
                        attempt,
                        max_attempts,
                    )
                    # Log which component is missing
                    status = cil.get_status()
                    for key, ready in status.items():
                        if not ready:
                            logger.error("   ❌ CIL component not ready: %s", key)

            except Exception as e:
                record_degradation('boot_cognitive', e)
                logger.error(
                    "Failed to init Advanced Cognition (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    e,
                    exc_info=True,
                )

            # Wait before retry (only if not last attempt)
            if attempt < max_attempts:
                logger.info("🔄 Retrying CIL initialization in 2s...")
                await asyncio.sleep(2)

        logger.error(
            "🚨 Advanced Cognition OFFLINE after %d attempts. "
            "Legacy fallback will be used for text responses.",
            max_attempts,
        )

    def _initialize_cognitive_extensions(self):
        """Initialize meta-learning, simulation, and motivation modules."""
        # Meta-Learning
        try:
            from core.meta.meta_learning_engine import MetaLearningEngine

            if hasattr(self, "memory") and self.memory:
                self.meta_learning = MetaLearningEngine(
                    self.memory, self.cognitive_engine
                )
                logger.info("✓ Meta-Learning Engine active")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.debug("Meta-Learning Engine optional: %s", e)

        # World Model & Motivation
        try:
            from core.motivation.aesthetic_critic import AestheticCritic
            from core.motivation.goal_hierarchy import GoalHierarchy
            from core.simulation.mental_simulator import MentalSimulator

            if self.cognitive_engine:
                self.simulator = MentalSimulator(self.cognitive_engine)
                self.goal_hierarchy = GoalHierarchy(self.cognitive_engine)
                self.aesthetic_critic = AestheticCritic(self.cognitive_engine)
                logger.info("✓ Mental Simulation & Intrinsic Motivation active")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.debug("Simulation/Motivation modules optional: %s", e)

        # Narrative Memory (v11.0 Temporal Narrative)
        try:
            from core.brain.narrative_memory import NarrativeEngine

            self.narrative_engine = NarrativeEngine(self)
            from core.container import ServiceContainer
            ServiceContainer.register_instance("narrative_engine", self.narrative_engine)
            logger.info("✓ Narrative Engine initialized")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to init Narrative Memory: %s", e)
            self.narrative_engine = None

        # Hook System Extensions
        try:
            from core.continuous_learning import ContinuousLearningEngine

            self.learning_engine = ContinuousLearningEngine(orchestrator=self)

            # Register Hooks
            async def on_post_think_learning(message, thought):
                if thought and hasattr(thought, "content") and thought.content:
                    await self.learning_engine.record_interaction(
                        message,
                        thought.content,
                        user_name=getattr(self, "user_identity", {}).get("name", "user"),
                    )

            self.hooks.register("post_think", on_post_think_learning)

            async def on_post_action_learning(tool_name, params, result):
                # Learning from research results if it was a search
                if (
                    tool_name == "web_search"
                    and isinstance(result, dict)
                    and result.get("ok")
                ):
                    # results extraction depends on tool schema
                    results = result.get("results", [])
                    if self.learning_engine.knowledge and results:
                        for r in results[:3]:
                            self.learning_engine.knowledge.add_knowledge(
                                content=f"{r.get('title')}: {r.get('summary')}",
                                source=f"research:{r.get('url')}",
                            )

            self.hooks.register("post_action", on_post_action_learning)

            logger.info("✓ Continuous Learning Engine integrated (v6.2 Unified)")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to integrate continuous learning: %s", e)

        try:
            from core.behavior_controller import integrate_behavior_control

            integrate_behavior_control(self)
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to integrate behavior control: %s", e)

    async def _init_cognitive_architecture(self):
        """Initialize the Consciousness System, Affect, and Cognitive Context Manager."""
        from core.brain.composer_node import ComposerNode

        ServiceContainer.register_instance("composer_node", ComposerNode())

        from core.brain.cognitive_context_manager import CognitiveContextManager

        context_manager = CognitiveContextManager(self)
        ServiceContainer.register_instance("context_manager", context_manager)
        # Start context manager in background if it's heavy
        from core.utils.task_tracker import fire_and_track
        fire_and_track(context_manager.start(), name="context_manager_start")
        logger.info("✓ CognitiveContextManager registered and starting in background")

        # Unified Consciousness & Affect Initialization
        try:
            # 1. Affect Engine (Damasio V2)
            from core.affect.damasio_v2 import AffectEngineV2
            affect = ServiceContainer.get("affect_engine")
            ServiceContainer.register_instance("affect_engine", affect)
            ServiceContainer.register_instance("affect_manager", affect)
            logger.info(
                "✓ AffectEngineV2 (affect_engine/affect_manager) registered"
            )

            # 2. Subsystem Audit
            from core.subsystem_audit import SubsystemAudit
            audit = ServiceContainer.get("subsystem_audit")

            # 3. Qualia Synthesizer (Unified)
            # Explicitly register qualia_synthesizer for LoopMonitor tracking.
            # ConsciousnessSystem will resolve it via ServiceContainer or create fresh.
            from core.consciousness.qualia_synthesizer import QualiaSynthesizer
            synth = ServiceContainer.get("qualia_synthesizer")
            logger.info("✓ QualiaSynthesizer registered (initial registration)")

            from core.consciousness import ConsciousnessSystem

            consciousness = ConsciousnessSystem(self)
            ServiceContainer.register_instance("consciousness", consciousness)
            # Individual components are pulled from container or initialized by ConsciousnessSystem
            ServiceContainer.register_instance(
                "global_workspace", consciousness.global_workspace
            )
            ServiceContainer.register_instance(
                "temporal_binding", consciousness.temporal_binding
            )
            ServiceContainer.register_instance(
                "self_prediction", consciousness.self_prediction
            )
            # Ensure the synthesizer used by ConsciousnessSystem is also registered if not already
            if consciousness.qualia and not ServiceContainer.get("qualia_synthesizer", default=None):
                 ServiceContainer.register_instance("qualia_synthesizer", consciousness.qualia)
            logger.info("✓ Consciousness System & components registered")

            # Start Consciousness System in background
            async def _start_consciousness():
                try:
                    await consciousness.start()
                    logger.info("🧠 Consciousness System started in background")
                except Exception as e:
                    record_degradation('boot_cognitive', e)
                    logger.error(
                        "🛑 Consciousness System background start failed: %s", e
                    )

            from core.utils.task_tracker import fire_and_track
            fire_and_track(_start_consciousness(), name="start_consciousness")

            # --- PHASE 8: Phenomenological Integration ---
            from core.consciousness.integration import get_consciousness_integration

            integration = get_consciousness_integration(self)
            await integration.initialize()
            ServiceContainer.register_instance("consciousness_integration", integration)
            logger.info("🌟 Layer 8: Phenomenological Experiencer active")

        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to initialize Unified Consciousness/Affect: %s", e)

        # 🟢 Additional Cognitive Models
        from core.consciousness.mind_model import MindModel

        ServiceContainer.register_instance("mind_model", MindModel())

        # QualiaSynthesizer is now registered above during phenomenological initialization.
        pass

        from core.consciousness.homeostasis import HomeostasisEngine

        ServiceContainer.register_instance("homeostasis", HomeostasisEngine())

    async def _init_language_services(self):
        """Initialize the Language Center (Narrator) and Prompt Compiler."""
        try:
            from core.brain.narrator import register_narrator_service

            register_narrator_service()
            logger.info("✓ NarratorService (The Language Center) registered")

            # Phase 39: JIT Prompt Compiler (The Body)
            from core.brain.llm.compiler import register_prompt_compiler

            register_prompt_compiler()
            logger.info("✓ PromptCompiler (The Body) registered")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to register Language/Identity/Compiler services: %s", e)

    def _init_strategic_planning(self):
        """Initialize the hierarchical Strategic Planner (Phase 17)."""
        try:
            from core.data.project_store import ProjectStore
            from core.neural_feed import NeuralFeed
            from core.strategic_planner import StrategicPlanner

            # 1. Neural Feed
            feed = NeuralFeed()
            ServiceContainer.register_instance("neural_feed", feed)

            # 2. Planner
            db_path = config.paths.data_dir / "projects.db"
            store = ProjectStore(str(db_path))
            planner = StrategicPlanner(self.cognitive_engine, store)
            self.strategic_planner = planner
            self.project_store = store
            ServiceContainer.register_instance("strategic_planner", planner)
            logger.info("🎯 Strategic Planner & Neural Feed online")
        except Exception as e:
            record_degradation('boot_cognitive', e)
            logger.error("Failed to initialize Strategic systems: %s", e)
            self.strategic_planner = None
            self.project_store = None
