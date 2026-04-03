"""core/providers/ops_provider.py — Ops, Health & Resilience Registration
"""

import logging
from core.container import ServiceLifetime

logger = logging.getLogger("Aura.Providers.Ops")

def register_ops_services(container, is_proxy: bool = False):
    # 9.0 Health Monitor
    def create_health_monitor():
        from core.ops.health_monitor import HealthMonitor
        return HealthMonitor()
    container.register('health_monitor', create_health_monitor, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 9.1 Metabolic Monitor
    def create_metabolic_monitor():
        from core.ops.metabolic_monitor import MetabolicMonitor
        return MetabolicMonitor()
    container.register('metabolic_monitor', create_metabolic_monitor, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 42. Self-Modification Engine
    def create_sme():
        if is_proxy:
            return None
        from core.self_modification.self_modification_engine import AutonomousSelfModificationEngine
        from core.config import config
        brain = container.get("cognitive_engine")
        return AutonomousSelfModificationEngine(
            brain, 
            code_base_path=str(config.paths.base_dir),
            auto_fix_enabled=config.security.auto_fix_enabled
        )
    container.register('self_modification_engine', create_sme, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('sme', lambda: container.get("self_modification_engine"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # Resilience Engine
    def create_resilience():
        try:
            from core.resilience.resilience import ResilienceEngine
            return ResilienceEngine()
        except ImportError: return None
    container.register('resilience', create_resilience, lifetime=ServiceLifetime.SINGLETON, required=False)

    # Subsystem Audit
    def create_subsystem_audit():
        from core.subsystem_audit import SubsystemAudit
        return SubsystemAudit()
    container.register('subsystem_audit', create_subsystem_audit, lifetime=ServiceLifetime.SINGLETON, required=True)

    # 14. Register metabolism service
    def create_metabolic_coordinator():
        from core.coordinators.metabolic_coordinator import MetabolicCoordinator
        # Defer orchestrator resolution until it's actually needed or available
        return MetabolicCoordinator(container=container)

    container.register('metabolic_coordinator', create_metabolic_coordinator, lifetime=ServiceLifetime.SINGLETON, required=True)
    container.register('metabolism', lambda: container.get("metabolic_coordinator"), lifetime=ServiceLifetime.SINGLETON, required=False)

    # Register raw state for components that only need energy tracking
    from core.systems.metabolism import MetabolismService
    container.register('metabolism_state', lambda: MetabolismService(), lifetime=ServiceLifetime.SINGLETON, required=False)
