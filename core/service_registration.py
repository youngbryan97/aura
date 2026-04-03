"""core/service_registration.py — Consolidated Service Registration
=============================================================
Refactored into a modular provider system for Digital Metabolism.
"""

import logging
from .container import ServiceContainer, ServiceLifetime, get_container
from .config import config

# Providers
from .providers.cognitive_provider import register_cognitive_services
from .providers.memory_provider import register_memory_services
from .providers.sensory_provider import register_sensory_services
from .providers.consciousness_provider import register_consciousness_services
from .providers.ops_provider import register_ops_services

# Patch 8: Metabolism
from .services.metabolism import MetabolismService
# Patch 28: Runtime & Control
from .runtime.loop_guard import LoopLagMonitor
from .control.dynamic_router import DynamicRouter

logger = logging.getLogger(__name__)

def register_all_services(is_proxy: bool = False):
    """Register all services via modular providers.
    
    v49: Idempotent registration. If called multiple times, it only 
    registers services that are missing.
    """
    container = get_container()
    
    # Check if we've already done a full registration
    if hasattr(register_all_services, "_full_run") and register_all_services._full_run:
        logger.debug("Modular services already fully registered.")
        return container
    
    if not is_proxy:
        register_all_services._full_run = True

    logger.info("Initializing Modular Service Providers (is_proxy=%s)...", is_proxy)

    # 0. Infrastructure (Remain in main entry for now)
    def create_event_bus():
        from .event_bus import get_event_bus
        return get_event_bus()
    container.register('event_bus', create_event_bus, lifetime=ServiceLifetime.SINGLETON, required=True)

    def create_mycelial():
        from .mycelium import MycelialNetwork
        return MycelialNetwork()
    container.register('mycelial_network', create_mycelial, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('mycelium', lambda: container.get("mycelial_network"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 0.5 Metabolism (Patch 8)
    container.register('metabolism', lambda: MetabolismService(), lifetime=ServiceLifetime.SINGLETON)
    container.register('metabolic_monitor', lambda: container.get("metabolism"), lifetime=ServiceLifetime.SINGLETON)
    
    # Patch 28: Dynamic Router & Loop Monitor
    container.register("loop_monitor", lambda: LoopLagMonitor(), lifetime=ServiceLifetime.SINGLETON)
    container.register("dynamic_router", lambda: DynamicRouter(), lifetime=ServiceLifetime.SINGLETON)

    # Patch 49: Core state binding
    def create_state_repo():
        from .state.state_repository import StateRepository
        from .config import config
        db_path = config.paths.data_dir / "state" / "aura_state.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return StateRepository(db_path=str(db_path), is_vault_owner=not is_proxy)
    container.register('state_repo', create_state_repo, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('state_repository', lambda: container.get("state_repo"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # 1. Modular Provider Execution
    register_cognitive_services(container, is_proxy=is_proxy)
    
    if not is_proxy:
        register_memory_services(container)
        register_sensory_services(container)
        register_consciousness_services(container)
    else:
        logger.info("📡 Proxy Mode: Skipping Memory, Sensory, and Consciousness providers.")

    register_ops_services(container, is_proxy=is_proxy)

    # 1.5 Platform Root (Hardware Binding)
    if not container.has('platform_root'):
        def create_platform_root():
            from core.sovereign.platform_root import get_platform_root
            return get_platform_root()
        container.register('platform_root', create_platform_root, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 2. Final Wiring (Inter-provider dependencies)
    _finalize_wiring(container)

    # 3. Boot Validation Gate (Patch 11/27)
    from core.bootstrap.validation import BootValidator
    v_result = BootValidator.validate_boot(container)
    if not v_result.passed:
        logger.error("🛡️ Boot Validation FAILED: %s", v_result.failures)
        # Defer lock to aura_main.py
        return container

    # 2.2 Digital Organism Extensions (2026 Phase)
    def _create_self_model():
        from core.self.self_model import SelfModel
        return SelfModel()

    def _create_identity_anchor():
        from core.identity.identity_anchor import IdentityAnchor
        return IdentityAnchor()

    def _create_goal_engine():
        from core.goals.goal_engine import GoalEngine
        return GoalEngine()

    def _create_internal_simulator():
        from core.simulation.internal_simulator import InternalSimulator
        return InternalSimulator()

    def _create_meta_cognition_loop():
        from core.meta.meta_cognition import MetaCognition
        return MetaCognition()

    container.register('self_model', _create_self_model, lifetime=ServiceLifetime.SINGLETON)
    container.register('identity_anchor', _create_identity_anchor, lifetime=ServiceLifetime.SINGLETON)
    container.register('goal_engine', _create_goal_engine, lifetime=ServiceLifetime.SINGLETON)
    container.register('internal_simulator', _create_internal_simulator, lifetime=ServiceLifetime.SINGLETON)
    container.register('meta_cognition_loop', _create_meta_cognition_loop, lifetime=ServiceLifetime.SINGLETON)

    # Agency convergence (2026 Phase)
    def _create_tension_engine():
        from core.agency.tension_engine import TensionEngine
        return TensionEngine()

    def _create_initiative_arbiter():
        from core.agency.initiative_arbiter import InitiativeArbiter
        return InitiativeArbiter()

    container.register('tension_engine', _create_tension_engine, lifetime=ServiceLifetime.SINGLETON)
    container.register('initiative_arbiter', _create_initiative_arbiter, lifetime=ServiceLifetime.SINGLETON)

    # Patch 27: Container lock deferred to aura_main.py after all top-level components register
    logger.info("✅ All modular services registered and validated (Lock deferred).")
    return container

def _finalize_wiring(container):
    """Handles cross-component linking (e.g. Mycelium roots)."""
    try:
        mycelial = container.get("mycelial_network")
        if mycelial:
            # Link major layers
            from .meta_cognition import MetaEvolutionEngine
            mycelial.link_layer("meta_cognition", MetaEvolutionEngine)
            
            # Establish base hyphae
            if hasattr(mycelial, 'establish_consciousness_hyphae'):
                mycelial.establish_consciousness_hyphae()
            
            mycelial.establish_connection("cognition", "llm", priority=1.0)
            mycelial.establish_connection("memory", "cognition", priority=0.9)
            
            # Phase 10: Neural Root for Metal persistence
            if hasattr(mycelial, 'establish_neural_root'):
                mycelial.establish_neural_root("llm", hardware_id="gpu_metal")
            
    except Exception as e:
        logger.debug("Wiring deferred: %s", e)

def inject_services_into_context(context: dict) -> dict:
    container = get_container()
    for name, descriptor in container._services.items():
        if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance:
            context[name] = descriptor.instance
    return context