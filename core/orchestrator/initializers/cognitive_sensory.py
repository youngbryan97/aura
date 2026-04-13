import asyncio
import logging
from typing import Any
from core.container import ServiceContainer

logger = logging.getLogger(__name__)

async def init_cognitive_sensory_layer(orchestrator: Any):
    """Initialize the higher-order cognitive and sensory services."""
    
    # 1. Identity & Drives
    try:
        from core.self_model import SelfModel
        orchestrator.self_model = await SelfModel.load()
        ServiceContainer.register_instance("self_model", orchestrator.self_model)
        ServiceContainer.register_instance("identity", orchestrator.self_model)
        
        from core.soul import Soul
        orchestrator.soul = Soul(orchestrator)
        ServiceContainer.register_instance("soul", orchestrator.soul)
        
        from core.fictional_ai_synthesis import register_all_fictional_engines
        orchestrator.fictional_engines = register_all_fictional_engines(orchestrator)
        
        # Personality & Identity
        from core.brain.personality_engine import PersonalityEngine
        orchestrator.personality_engine = PersonalityEngine()
        orchestrator.personality_engine.setup_hooks(orchestrator)
        ServiceContainer.register_instance("personality_engine", orchestrator.personality_engine)
        ServiceContainer.register_instance("personality", orchestrator.personality_engine)
        
        logger.info("🆔 Identity, Soul, Personality, and Fictional Engines registered.")
    except Exception as e:
        logger.error("🛑 Failed to load SelfModel: %s", e)
            
    try:
        from core.managers.drive_controller import DriveController
        if hasattr(orchestrator, 'affect') and orchestrator.affect:
            if not hasattr(orchestrator.affect, 'drive_controller') or not orchestrator.affect.drive_controller:
                orchestrator.affect.drive_controller = DriveController()
            
            ServiceContainer.register_instance("drive_engine", orchestrator.affect.drive_controller)
            ServiceContainer.register_instance("drives", orchestrator.affect.drive_controller)
            logger.info("🚗 Drive Engine registered via AffectCoordinator")
        else:
            logger.warning("🚗 Affect system missing; deferring DriveController")
    except Exception as e:
        logger.error("🚗 DriveController init failed: %s", e)
    
    # 2. Senses
    from core.senses.voice_engine import get_voice_engine
    ServiceContainer.register_instance("voice_engine", get_voice_engine())
    
    from core.brain.multimodal_orchestrator import MultimodalOrchestrator
    ServiceContainer.register_instance("multimodal_orchestrator", MultimodalOrchestrator())
    
    from core.brain.composer_node import ComposerNode
    ServiceContainer.register_instance("composer_node", ComposerNode())

    # 3. Consciousness & Resilience (Handled by Modular Providers)
    pass
    
    from core.guardians.memory_guard import MemoryGuard
    memory_guard = MemoryGuard()
    await memory_guard.start()
    ServiceContainer.register_instance("memory_guard", memory_guard)

    from core.soma.resilience_engine import ResilienceEngine
    resilience = ResilienceEngine(orchestrator)
    await resilience.start()
    ServiceContainer.register_instance("soma", resilience)
    ServiceContainer.register_instance("resilience_engine", resilience)

    from core.identity.drift_monitor import IdentityDriftMonitor
    drift_monitor = IdentityDriftMonitor()
    ServiceContainer.register_instance("drift_monitor", drift_monitor)

    from core.identity.spine import SpiritualSpine
    opinion_engine = ServiceContainer.get("opinion_engine", default=None)
    spine = SpiritualSpine(opinion_engine=opinion_engine)
    ServiceContainer.register_instance("spine", spine)

    from core.self_modification.growth_ladder import GrowthLadder
    growth_ladder = GrowthLadder(orchestrator)
    ServiceContainer.register_instance("growth_ladder", growth_ladder)

    from core.memory.sovereign_pruner import SovereignPruner
    pruner = SovereignPruner(orchestrator)
    ServiceContainer.register_instance("sovereign_pruner", pruner)

    from core.guardians.governor import SystemGovernor
    system_governor = SystemGovernor()
    await system_governor.start()
    ServiceContainer.register_instance("system_governor", system_governor)

    # 11. Will Engine (Metabolic Evolution)
    try:
        from core.self.will_engine import WillEngine
        orchestrator.will_engine = WillEngine()
        # Note: Initialization happens here, but the loop needs the event loop to be running.
        # RobustOrchestrator.start() will trigger the scheduler which calls process_cycle.
        await orchestrator.will_engine.initialize()
        ServiceContainer.register_instance("will_engine", orchestrator.will_engine)
        ServiceContainer.register_instance("metabolic_coordinator", orchestrator.will_engine)
        orchestrator.metabolic_coordinator = orchestrator.will_engine
        logger.info("☘️ WillEngine (Metabolic Evolution) registered.")
    except Exception as e:
        logger.error("🛑 WillEngine init failed: %s", e)

    # ── 12. Learned Cognitive Systems ────────────────────────────────────
    # These replace rigid if/else rules with adaptive, data-driven systems.
    # Each is optional — if import fails, the system degrades gracefully.
    _cognitive_services = {
        "sentiment_tracker": ("core.cognitive.sentiment_tracker", "get_sentiment_tracker"),
        "anomaly_detector": ("core.cognitive.anomaly_detector", "AnomalyDetector"),
        "strange_loop": ("core.cognitive.strange_loop", "get_strange_loop"),
        "homeostatic_rl": ("core.cognitive.homeostatic_rl", "get_homeostatic_rl"),
        "topology_evolution": ("core.cognitive.topology_evolution", "TopologyEvolution"),
        "autopoiesis": ("core.cognitive.autopoiesis", "get_autopoiesis_engine"),
    }
    # ALife systems (from Avida, Tierra, Lenia, EcoSim, Evochora, CA research)
    _alife_services = {
        "criticality_regulator": ("core.consciousness.criticality_regulator", "get_criticality_regulator"),
        "alife_dynamics": ("core.consciousness.alife_dynamics", "ALifeDynamics"),
        "alife_extensions": ("core.consciousness.alife_extensions", "ALifeExtensions"),
        "endogenous_fitness": ("core.consciousness.endogenous_fitness", "get_endogenous_fitness"),
    }
    all_services = {**_cognitive_services, **_alife_services}
    total_expected = len(all_services)
    registered_count = 0
    for svc_name, (mod_path, factory_name) in all_services.items():
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            factory = getattr(mod, factory_name)
            instance = factory() if callable(factory) else factory
            ServiceContainer.register_instance(svc_name, instance)
            registered_count += 1
        except Exception as cog_exc:
            logger.debug("Cognitive/ALife service '%s' deferred: %s", svc_name, cog_exc)
    if registered_count:
        logger.info(
            "🧠 Registered %d/%d learned cognitive + ALife systems.",
            registered_count, total_expected,
        )

    # 12. Cellular Substrate (Unified Mutation)
    try:
        from core.state.cellular_substrate import CellularSubstrate
        orchestrator.cellular_substrate = CellularSubstrate()
        await orchestrator.cellular_substrate.initialize()
        ServiceContainer.register_instance("cellular_substrate", orchestrator.cellular_substrate)
        logger.info("♾️ CellularSubstrate (Unified Mutation) registered.")
    except Exception as e:
        logger.error("🛑 CellularSubstrate init failed: %s", e)
    
    logger.info("🧬 [BOOT] Cognitive & Sensory Layer initialized.")
