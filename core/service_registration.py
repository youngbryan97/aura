"""core/service_registration.py — Consolidated Service Registration
=============================================================
Refactored into a modular provider system for Digital Metabolism.
"""

import logging
import threading
from typing import Any

from core.exceptions import ContainerError
from core.runtime.errors import record_degradation

from .config import config
from .container import ServiceLifetime, get_container
from .control.dynamic_router import DynamicRouter

# Providers
from .providers.cognitive_provider import register_cognitive_services
from .providers.consciousness_provider import register_consciousness_services
from .providers.memory_provider import register_memory_services
from .providers.ops_provider import register_ops_services
from .providers.sensory_provider import register_sensory_services

# Patch 28: Runtime & Control
from .runtime.loop_guard import LoopLagMonitor

# Patch 8: Metabolism
from .services.metabolism import MetabolismService

logger = logging.getLogger(__name__)

# Serializes concurrent boot-time registration passes (RLock so the body can
# call container operations that may re-enter).
_REGISTRATION_LOCK = threading.RLock()


def install_immune_enforcement(immune: Any) -> Any:
    """Install the real enforcement backends and record it if they do not.

    CP126 (critical): "Immune system silently ignores enforcement backend
    failure. Firewall, quarantine, process, resource, and ARP backend
    activation errors are swallowed while the immune decision layer is
    returned as available."

    The whole defect was ``except (...): pass``. An immune system that can
    DECIDE to quarantine but cannot ENFORCE it is not a degraded immune
    system, it is a reporting one — and every surface asking the container
    for "immune_system" received an object that looked fully armed.
    Detection without enforcement is the most dangerous shape a security
    control can take, precisely because it is trusted.

    The decision layer genuinely works without the backends, so activation
    stays best-effort and the immune system is still returned. What changed
    is that the gap is recorded and readable off the object.

    Module-level rather than a closure inside the registration body, so the
    behaviour can be exercised without booting the whole container.
    """
    enforcement_active = False
    enforcement_error = ""
    try:
        from core.security.defensive_runtime import ensure_defensive_runtime_active

        ensure_defensive_runtime_active()
        enforcement_active = True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        enforcement_error = f"{type(exc).__name__}: {exc}"
        record_degradation(
            "service_registration.immune_enforcement",
            exc,
            severity="warning",
            action=(
                "returned the immune DECISION layer without enforcement "
                "backends; quarantine and firewall actions will not reach the host"
            ),
        )
    try:
        immune.enforcement_backends_active = enforcement_active
        immune.enforcement_backends_error = enforcement_error
    except (AttributeError, TypeError) as exc:
        record_degradation(
            "service_registration.immune_enforcement",
            exc,
            severity="info",
            action="could not annotate the immune system with its enforcement state",
            enforce_failure_policy=False,
        )
    return immune


def register_all_services(is_proxy: bool = False):
    """Register all services via modular providers.
    
    v49: Idempotent registration. If called multiple times, it only 
    registers services that are missing.
    """
    container = get_container()

    # Serialize concurrent boot callers so two cannot both pass the
    # already-registered check and interleave registration.
    with _REGISTRATION_LOCK:
        # Check if we've already done a full registration
        if getattr(register_all_services, "_full_run", False):
            logger.debug("Modular services already fully registered.")
            return container

        logger.info("Initializing Modular Service Providers (is_proxy=%s)...", is_proxy)
        _register_all_services_body(container, is_proxy)
        # Mark the full run complete ONLY after registration succeeds — a
        # failure mid-registration must let a retry re-run, not return a
        # partially-registered container forever.
        if not is_proxy:
            register_all_services._full_run = True
        logger.info(
            "Modular service registration pass complete (is_proxy=%s); some services "
            "are lazy/optional factories and are validated on first use, not at boot.",
            is_proxy,
        )
        return container


def _register_all_services_body(container, is_proxy: bool):
    """Register the modular service providers (idempotence handled by caller)."""

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

    # AtomSpace: PLN metagraph + ECAN attention economy (Hyperon fusion).
    # The belief engine mirrors claims here; the revision loop ticks the
    # economy and runs attention-guided forward chaining.
    def create_atomspace():
        from core.knowledge.atomspace import get_atomspace
        return get_atomspace()
    container.register('atomspace', create_atomspace, lifetime=ServiceLifetime.SINGLETON, required=False)

    # 0.5 Metabolism / resource stakes.  The ledger is separate from the older
    # consciousness.resource_stakes engine so it can persist hard action
    # envelopes and degradation events for audit.
    def create_resource_stakes():
        from core.autonomic.resource_stakes import ResourceStakesLedger
        path = config.paths.data_dir / "resource_stakes" / "stakes.sqlite3"
        return ResourceStakesLedger(path)

    container.register('resource_stakes', create_resource_stakes, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register(
        'metabolism',
        lambda: MetabolismService(stakes=container.get("resource_stakes", default=None)),
        lifetime=ServiceLifetime.SINGLETON,
    )
    container.register('metabolic_monitor', lambda: container.get("metabolism"), lifetime=ServiceLifetime.SINGLETON)

    # Canonical desired-state and resource-admission spine. Domain-specific
    # samplers and the legacy arbitrator are adapters behind this owner.
    def create_runtime_control_plane():
        from core.runtime.control_plane import get_runtime_control_plane

        return get_runtime_control_plane()

    def create_resource_governor():
        from core.resource.resource_governor import get_resource_governor

        return get_resource_governor()

    def create_resource_arbitrator():
        from core.resilience.resource_arbitrator import get_resource_arbitrator

        return get_resource_arbitrator()

    def create_lane_admission():
        from core.brain.lane_admission import get_lane_admission_controller

        return get_lane_admission_controller()

    def create_lane_reconciler():
        from core.runtime.lane_reconciler import get_lane_reconciler

        return get_lane_reconciler()

    def create_actor_supervision():
        from core.supervisor.tree import get_tree

        return get_tree()

    container.register(
        'runtime_control_plane',
        create_runtime_control_plane,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        owner='core/runtime/control_plane.py',
        registered_by='register_all_services',
        required_for='desired-state reconciliation and constrained work admission',
        failure_policy='fail-closed',
    )
    container.register(
        'resource_admission',
        lambda: container.get('runtime_control_plane').admission,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies=['runtime_control_plane'],
        owner='core/runtime/control_plane.py',
        registered_by='register_all_services',
        required_for='pressure-aware resource leases',
        failure_policy='fail-closed',
    )
    container.register(
        'resource_governor',
        create_resource_governor,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies=['runtime_control_plane'],
        owner='core/resource/resource_governor.py',
        registered_by='register_all_services',
        required_for='resource sampling, throttling, and tiered eviction',
        failure_policy='degrade_with_receipt',
    )
    container.register(
        'resource_arbitrator',
        create_resource_arbitrator,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies=['runtime_control_plane'],
        owner='core/resilience/resource_arbitrator.py',
        registered_by='register_all_services',
        required_for='legacy inference and evolution admission compatibility',
        failure_policy='fail-closed',
    )
    container.register(
        'lane_admission',
        create_lane_admission,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies=['resource_admission'],
        owner='core/brain/lane_admission.py',
        registered_by='register_all_services',
        required_for='declared model-lane memory envelope enforcement',
        failure_policy='fail-closed',
    )
    container.register(
        'lane_reconciler',
        create_lane_reconciler,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies=['runtime_control_plane', 'lane_admission'],
        owner='core/runtime/lane_reconciler.py',
        registered_by='register_all_services',
        required_for='model-lane desired-state and crash-loop convergence',
        failure_policy='degrade_with_receipt',
    )
    container.register(
        'actor_supervision',
        create_actor_supervision,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies=['runtime_control_plane'],
        owner='core/supervisor/tree.py',
        registered_by='register_all_services',
        required_for='canonical actor process lifecycle and restart policy',
        failure_policy='fail-closed',
    )

    # Critique-closure services: adaptive mood, mesh cognition, emergent goals,
    # structural mutator, lineage, self-awareness suite, identity chronicle.
    # Every one of these must be container-registered or it is dead code.
    def create_adaptive_mood():
        from core.consciousness.adaptive_mood import get_adaptive_mood
        return get_adaptive_mood()

    def create_mesh_cognition():
        from core.consciousness.mesh_cognition import get_mesh_cognition
        return get_mesh_cognition()

    def create_emergent_goal_engine():
        from core.goals.emergent_goals import get_emergent_goal_engine
        return get_emergent_goal_engine()

    def create_structural_mutator():
        from core.self_modification.structural_mutator import get_structural_mutator
        return get_structural_mutator()

    def create_lineage_manager():
        from core.self_modification.lineage import get_lineage_manager
        return get_lineage_manager()

    def create_self_awareness_suite():
        from core.consciousness.self_awareness_suite import get_self_awareness_suite
        return get_self_awareness_suite()

    def create_identity_chronicle():
        from core.identity.id_rag import get_identity_chronicle
        return get_identity_chronicle()

    def create_reimplementation_lab():
        from core.config import config
        from core.llm.code_generator import LLMCodeGenerator
        from core.self_improvement.reimplementation_lab import ReimplementationLab
        # Use LLM generator configured for local primary tier
        generator = LLMCodeGenerator(prefer_tier="primary")
        return ReimplementationLab(
            project_root=str(config.paths.base_dir),
            generator=generator
        )

    def create_program_dna_reconstruction():
        import importlib

        program_dna = importlib.import_module("core.self_improvement.program_dna")
        return program_dna.ProgramDNAReconstructionEngine(
            project_root=str(config.paths.base_dir),
            internal_lab=container.get("reimplementation_lab", default=None),
        )

    def create_being_runtime():
        from core.being.runtime import get_being_runtime

        return get_being_runtime()

    container.register('adaptive_mood', create_adaptive_mood, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('mesh_cognition', create_mesh_cognition, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register(
        'hierarchical_agency',
        lambda: __import__('core.agency.hierarchical_agency', fromlist=['get_hierarchical_agency']).get_hierarchical_agency(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )

    def _create_defensive_runtime():
        from core.security.defensive_runtime import ensure_defensive_runtime_active

        return ensure_defensive_runtime_active()

    container.register(
        'defensive_runtime',
        _create_defensive_runtime,
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
        required_for="live runtime defense",
        failure_policy="degrade_with_receipt",
    )
    try:
        container.get("defensive_runtime")
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        # ContainerError (and its ServiceNotFoundError subclass) descend from
        # AuraError, not RuntimeError, so the tuple above could never catch
        # them. This registration declares required=False and
        # failure_policy="degrade_with_receipt" — it says out loud that it
        # intends to degrade — yet an absent descriptor raised straight
        # through and killed the lifespan with "Application startup failed".
        ContainerError,
    ) as exc:
        record_degradation("service_registration.defensive_runtime", exc)

    def _create_immune_system():
        from core.security.immune_system import get_immune_system

        return install_immune_enforcement(get_immune_system())


    try:
        from core.resilience.fault_taxonomy import get_fault_registry
        from core.resilience.recovery_bridge import get_recovery_bridge

        bridge = get_recovery_bridge()
        if bridge.start():
            get_fault_registry().add_listener(bridge.on_fault)
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("service_registration.recovery_bridge", exc,
                           severity="debug",
                           action="recovery bridge not started")

    container.register(
        'immune_system', _create_immune_system,
        lifetime=ServiceLifetime.SINGLETON, required=False,
    )
    container.register(
        'deletion_guard',
        lambda: __import__('core.security.deletion_guard', fromlist=['get_deletion_guard']).get_deletion_guard(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'threat_detectors',
        lambda: __import__('core.security.threat_detectors', fromlist=['get_threat_detectors']).get_threat_detectors(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'perception_sentinel',
        lambda: __import__('core.perception.perception_sentinel', fromlist=['get_perception_sentinel']).get_perception_sentinel(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'network_sentinel',
        lambda: __import__('core.security.network_sentinel', fromlist=['get_network_sentinel']).get_network_sentinel(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'sensory_runtime',
        lambda: __import__('core.perception.sensory_runtime', fromlist=['get_sensory_runtime']).get_sensory_runtime(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'other_agent_model',
        lambda: __import__('core.social.other_agent_model', fromlist=['get_other_agent_model']).get_other_agent_model(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Canonical single world-model surface — composes the four complementary facets
    # (forward dynamics, causal graph, outcome prediction, MCTS planning).
    container.register(
        'unified_world_model',
        lambda: __import__('core.world_model.unified_world_model', fromlist=['get_unified_world_model']).get_unified_world_model(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'world_model',
        lambda: container.get('unified_world_model'),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # The ontogenetic organ: the persistent learned state, the experience
    # corpus, and the authority ledger that decides which of Aura's decisions
    # a learned head is currently allowed to make.
    container.register(
        'ontogeny',
        lambda: __import__('core.ontogeny.service', fromlist=['get_ontogeny']).get_ontogeny(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Task-driven retrieval router over the typed memory taxonomy (intentional, not blind
    # similarity). Stores plug in as adapters; default sync stores wired best-effort.
    def _create_intentional_retriever():
        mod = __import__('core.memory.intentional_retrieval', fromlist=['get_intentional_retriever'])
        retriever = mod.get_intentional_retriever()
        try:
            retriever.wire_default_stores()
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            from core.runtime.errors import record_degradation
            record_degradation("service_registration.intentional_retriever", exc, severity="debug")
        return retriever
    container.register(
        'intentional_retriever',
        _create_intentional_retriever,
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Learned-but-bounded value model — preferences fenced by an immutable constitution,
    # anchored in Will; also backs the intentional retriever's VALUE store.
    container.register(
        'value_model',
        lambda: __import__('core.values.value_model', fromlist=['get_value_model']).get_value_model(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Permanent epistemic critic — audits a claim/response (overclaiming, action-actually-done,
    # receipt-exists, stale memory, persona leak, ungrounded user projection) before it's trusted.
    container.register(
        'adversarial_auditor',
        lambda: __import__('core.cognition.adversarial_audit', fromlist=['get_adversarial_auditor']).get_adversarial_auditor(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # World-scale ingestion — unrestricted web reach (owner-authorized) that updates the world
    # model + memory; state-changing writes route through the value model + Will.
    container.register(
        'world_ingestion',
        lambda: __import__('core.world_model.world_ingestion', fromlist=['get_world_ingestion_engine']).get_world_ingestion_engine(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Drive-integration volition — temporal accumulation + competition + hysteresis, replacing
    # the legacy instantaneous-VAD-threshold + flat-refractory volition path.
    container.register(
        'drive_integration',
        lambda: __import__('core.consciousness.drive_integration', fromlist=['get_drive_integration_engine']).get_drive_integration_engine(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Conation — the motivational faculty. Prices what is wanted by where the
    # value came from and whose mind was involved, which valence and arousal
    # cannot distinguish: five situations that behave nothing alike return one
    # identical point from the affect path.
    container.register(
        'conation',
        lambda: __import__('core.conation.engine', fromlist=['get_conation']).get_conation(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Concept formation — abstracts new conceptual primitives from repeated prediction errors
    # (complements AbstractionEngine, which distills from successes).
    container.register(
        'concept_formation',
        lambda: __import__('core.cognition.concept_formation', fromlist=['get_concept_formation_engine']).get_concept_formation_engine(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    # Affect grounding — affect labels from sustained multi-signal conditions with explicit factor
    # attribution, replacing single-float-threshold labels ("x < -0.2 → boredom").
    container.register(
        'affect_grounding',
        lambda: __import__('core.affect.affect_grounding', fromlist=['get_affect_grounding_engine']).get_affect_grounding_engine(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register('emergent_goal_engine', create_emergent_goal_engine, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('structural_mutator', create_structural_mutator, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('lineage_manager', create_lineage_manager, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('self_awareness_suite', create_self_awareness_suite, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('identity_chronicle', create_identity_chronicle, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('reimplementation_lab', create_reimplementation_lab, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register(
        'program_dna_reconstruction_engine',
        create_program_dna_reconstruction,
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'program_dna',
        lambda: container.get("program_dna_reconstruction_engine"),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register(
        'being_runtime',
        create_being_runtime,
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register('aura_now_runtime', lambda: container.get("being_runtime"), lifetime=ServiceLifetime.SINGLETON, required=False)
    try:
        # Materialize during boot registration so deep repair is available even
        # before another subsystem lazily asks for it.
        container.get("reimplementation_lab", default=None)
    except (OSError, ConnectionError, TimeoutError) as exc:
        record_degradation('service_registration', exc)
        logger.warning("ReimplementationLab boot singleton unavailable: %s", exc)

    def create_life_trace():
        from core.runtime.life_trace import get_life_trace
        return get_life_trace()

    def create_evidence_mode():
        from core.evaluation.evidence_mode import get_evidence_mode
        return get_evidence_mode()

    def create_markdown_workspace():
        from core.workspace.markdown_workspace import MarkdownWorkspace
        return MarkdownWorkspace()

    def create_aura_workspace():
        from core.workspace.aura_workspace import AuraWorkspace
        return AuraWorkspace(store=container.get("markdown_workspace"))

    def create_simulation_well():
        from core.data.simulation_well import default_simulation_well
        return default_simulation_well()

    def create_temporal_atlas_factory():
        from core.media.temporal_atlas import TemporalAtlas
        return lambda duration_s, **kwargs: TemporalAtlas(duration_s, **kwargs)

    def create_architecture_governor():
        from core.architect.config import ASAConfig
        from core.architect.governor import AutonomousArchitectureGovernor
        return AutonomousArchitectureGovernor(ASAConfig.from_env(config.paths.base_dir))

    def create_source_body():
        from core.soma.source_body import get_source_body
        return get_source_body()

    container.register('source_body', create_source_body, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('life_trace', create_life_trace, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('evidence_mode', create_evidence_mode, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('markdown_workspace', create_markdown_workspace, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('aura_workspace', create_aura_workspace, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('agent_workspace', lambda: container.get("aura_workspace"), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('simulation_well', create_simulation_well, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('temporal_atlas_factory', create_temporal_atlas_factory, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('architecture_governor', create_architecture_governor, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('autonomous_architecture_governor', lambda: container.get("architecture_governor"), lifetime=ServiceLifetime.SINGLETON, required=False)

    def create_neural_intent_router():
        from core.agency.neural_intent_router import get_neural_intent_router
        return get_neural_intent_router()

    def create_permission_setup():
        # Permission setup has no singleton state; expose the module itself
        # so callers can invoke check_all_permissions()/open_settings_pane()
        # through the container.
        import core.security.permission_setup as ps
        return ps

    container.register('neural_intent_router', create_neural_intent_router, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('permission_setup', create_permission_setup, lifetime=ServiceLifetime.SINGLETON, required=False)

    # Patch 28: Dynamic Router & Loop Monitor. Register under the CANONICAL
    # name the health contract + runtime_pressure look up (event_loop_monitor),
    # keeping loop_monitor as an alias — otherwise a live monitor was reported
    # missing by health because of the name drift.
    container.register("event_loop_monitor", lambda: LoopLagMonitor(), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register("loop_monitor", lambda: container.get("event_loop_monitor"), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register("dynamic_router", lambda: DynamicRouter(), lifetime=ServiceLifetime.SINGLETON)

    # Patch 49: Core state binding
    def create_state_repo():
        from .config import config
        from .state.state_repository import StateRepository
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
    from core.startup.boot_validator import BootValidator
    v_result = BootValidator.validate_boot(container)
    if not v_result.passed:
        logger.error("🛡️ Boot Validation FAILED: %s", v_result.failures)
        # Defer lock to aura_main.py
        return container

    # 2.2 Digital Organism Extensions (2026 Phase)
    def _create_self_model():
        from uuid import uuid4

        from core.self_model import SelfModel
        return SelfModel(id=str(uuid4()))

    def _create_canonical_self_engine():
        from core.self.canonical_self import get_canonical_self_engine

        return get_canonical_self_engine()

    def _create_identity_anchor():
        from core.identity.identity_anchor import IdentityAnchor
        return IdentityAnchor()

    def _create_goal_engine():
        from core.goals.goal_engine import GoalEngine
        return GoalEngine()

    def _create_goal_hierarchy():
        from core.motivation.goal_hierarchy import GoalHierarchy

        cognitive_engine = container.get("cognitive_engine", default=None)
        return GoalHierarchy(cognitive_engine)

    def _create_internal_simulator():
        from core.simulation.internal_simulator import InternalSimulator
        return InternalSimulator()

    def _create_meta_cognition_loop():
        from core.meta.meta_cognition import MetaCognition
        return MetaCognition()

    container.register('self_model', _create_self_model, lifetime=ServiceLifetime.SINGLETON)
    container.register('canonical_self_engine', _create_canonical_self_engine, lifetime=ServiceLifetime.SINGLETON)
    container.register(
        'canonical_self',
        lambda canonical_self_engine: canonical_self_engine.get_self(),
        lifetime=ServiceLifetime.SINGLETON,
        required=False,
    )
    container.register('identity_anchor', _create_identity_anchor, lifetime=ServiceLifetime.SINGLETON)
    container.register('goal_engine', _create_goal_engine, lifetime=ServiceLifetime.SINGLETON)
    container.register('goal_hierarchy', _create_goal_hierarchy, lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('goal_manager', lambda: container.get("goal_engine"), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('goal_memory', lambda: container.get("goal_engine"), lifetime=ServiceLifetime.SINGLETON, required=False)
    container.register('internal_simulator', _create_internal_simulator, lifetime=ServiceLifetime.SINGLETON)
    container.register('meta_cognition_loop', _create_meta_cognition_loop, lifetime=ServiceLifetime.SINGLETON)

    # Agency convergence (2026 Phase)
    def _create_tension_engine():
        from core.agency.tension_engine import TensionEngine
        return TensionEngine()

    def _create_initiative_arbiter():
        from core.agency.initiative_arbiter import InitiativeArbiter
        return InitiativeArbiter()

    def _create_tool_orchestrator():
        from core.agency.tool_orchestrator import get_tool_orchestrator
        return get_tool_orchestrator()

    container.register('tension_engine', _create_tension_engine, lifetime=ServiceLifetime.SINGLETON)
    container.register('initiative_arbiter', _create_initiative_arbiter, lifetime=ServiceLifetime.SINGLETON)
    container.register('tool_orchestrator', _create_tool_orchestrator, lifetime=ServiceLifetime.SINGLETON, required=False)

    # Patch 27: Container lock deferred to aura_main.py after all top-level components register
    logger.debug("Modular service providers registered (container lock deferred).")
    return container

def _finalize_wiring(container):
    """Handles cross-component linking (e.g. Mycelium roots)."""
    try:
        mycelial = container.get("mycelial_network")
        if mycelial:
            # Link major layers
            from core.cognition.meta_cognition import MetaEvolutionEngine
            mycelial.link_layer("meta_cognition", MetaEvolutionEngine)
            
            # Establish base hyphae
            if hasattr(mycelial, 'establish_consciousness_hyphae'):
                mycelial.establish_consciousness_hyphae()
            
            mycelial.establish_connection("cognition", "llm", priority=1.0)
            mycelial.establish_connection("memory", "cognition", priority=0.9)
            
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('service_registration', e)
        logger.debug("Wiring deferred: %s", e)

# Services never injected wholesale into cognition context — they hold keys,
# trust roots, or governance authority that must be reached only through their
# owning gateways, not handed to arbitrary context consumers.
_CONTEXT_INJECTION_DENYLIST = frozenset({
    "trust_engine", "keypair", "signing_key", "governance_vault", "vault",
    "credential_store", "secret_store", "will", "constitutional_core",
    "authority_gateway",
})


def inject_services_into_context(context: dict) -> dict:
    container = get_container()
    # Iterate a SNAPSHOT (not the live registry) so a concurrent registration
    # cannot mutate the dict mid-iteration, and skip sensitive owners.
    try:
        items = list(container._services.items())
    except (AttributeError, RuntimeError):
        return context
    for name, descriptor in items:
        if name in _CONTEXT_INJECTION_DENYLIST:
            continue
        if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance:
            context[name] = descriptor.instance
    return context
