from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import Severity, record_degradation
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger(__name__)

_COGNITIVE_SENSORY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


def _boot_report(orchestrator: Any) -> dict[str, Any]:
    report = getattr(orchestrator, "cognitive_sensory_boot", None)
    if not isinstance(report, dict):
        report = {
            "completed": [],
            "degraded": {},
            "registered": {},
            "learned_services": {"registered": 0, "expected": 0},
        }
        orchestrator.cognitive_sensory_boot = report
    else:
        report.setdefault("completed", [])
        report.setdefault("degraded", {})
        report.setdefault("registered", {})
        report.setdefault("learned_services", {"registered": 0, "expected": 0})
    return report


def _record_cognitive_sensory_degradation(
    orchestrator: Any,
    error: BaseException,
    *,
    phase: str,
    action: str,
    severity: Severity = "warning",
) -> None:
    report = _boot_report(orchestrator)
    report["degraded"][phase] = {
        "error": _error_summary(error),
        "action": action,
        "severity": severity,
    }
    record_degradation(
        "cognitive_sensory",
        error,
        severity=severity,
        action=action,
        extra={"phase": phase},
    )


def _register(report: dict[str, Any], name: str, instance: Any) -> None:
    ServiceContainer.register_instance(name, instance)
    report["registered"][name] = instance.__class__.__name__


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _run_phase(
    orchestrator: Any,
    phase: str,
    action_on_failure: str,
    runner: Callable[[], Any],
    *,
    severity: Severity = "warning",
) -> Any | None:
    report = _boot_report(orchestrator)
    try:
        result = await _maybe_await(runner())
        if phase not in report["completed"]:
            report["completed"].append(phase)
        return result
    except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
        _record_cognitive_sensory_degradation(
            orchestrator,
            exc,
            phase=phase,
            action=action_on_failure,
            severity=severity,
        )
        logger.error("%s init degraded: %s", phase, exc)
        return None


async def init_cognitive_sensory_layer(orchestrator: Any) -> dict[str, Any]:
    """Initialize the higher-order cognitive and sensory services."""
    report = _boot_report(orchestrator)

    async def _identity_and_personality() -> None:
        from core.brain.identity import IdentityService
        from core.brain.llm.semantic_neural_serving import prepare_semantic_neural_serving
        from core.brain.personality_engine import PersonalityEngine
        from core.fictional_ai_synthesis import register_all_fictional_engines
        from core.orchestrator.initializers.derived_engines import (
            register_derived_engines,
        )
        from core.self_model import SelfModel
        from core.soul import Soul

        orchestrator.self_model = await SelfModel.load()
        _register(report, "self_model", orchestrator.self_model)
        _register(report, "identity", orchestrator.self_model)

        identity_service = ServiceContainer.get("identity_service", default=None)
        if identity_service is None:
            identity_service = IdentityService()
            _register(report, "identity_service", identity_service)
        orchestrator.identity_service = identity_service

        orchestrator.soul = ServiceContainer.get("soul", default=None)
        if orchestrator.soul is None:
            orchestrator.soul = Soul(orchestrator)
        _register(report, "soul", orchestrator.soul)

        orchestrator.fictional_engines = register_all_fictional_engines(orchestrator)
        await prepare_semantic_neural_serving()
        orchestrator.derived_engines = register_derived_engines(orchestrator)

        orchestrator.personality_engine = PersonalityEngine()
        orchestrator.personality_engine.setup_hooks(orchestrator)
        _register(report, "personality_engine", orchestrator.personality_engine)
        _register(report, "personality", orchestrator.personality_engine)
        logger.info("🆔 Identity, Soul, Personality, and Fictional Engines registered.")

    await _run_phase(
        orchestrator,
        "identity_personality",
        "Skipped identity/personality services and left boot report degraded for health contract review",
        _identity_and_personality,
        severity="critical",
    )

    async def _drive_engine() -> None:
        from core.managers.drive_controller import DriveController

        if hasattr(orchestrator, "affect") and orchestrator.affect:
            controller = getattr(orchestrator.affect, "_drive_controller", None)
            if controller is None:
                controller = getattr(orchestrator, "drive_controller", None)
            if controller is None or not callable(getattr(controller, "is_alive", None)):
                controller = DriveController(orchestrator)
            orchestrator.affect.drive_controller = controller
            orchestrator.drive_controller = controller
            _register(report, "drive_engine", controller)
            _register(report, "drives", controller)
            logger.info("🚗 Drive Engine registered via AffectCoordinator")
            return
        raise RuntimeError("affect system unavailable; drive controller deferred")

    await _run_phase(
        orchestrator,
        "drive_engine",
        "Deferred drive engine registration; motivation restoration remains unavailable until affect is online",
        _drive_engine,
        severity="warning",
    )

    async def _voice_engine() -> None:
        from core.senses.voice_engine import get_voice_engine

        _register(report, "voice_engine", get_voice_engine())

    await _run_phase(
        orchestrator,
        "voice_engine",
        "Skipped voice engine registration; chat continues without voice I/O",
        _voice_engine,
        severity="warning",
    )

    async def _reality_reach() -> None:
        from core.environment.runtime_workspace import environment_runtime_file
        from core.reality_reach.acceptance_mandate import AcceptanceMandateStore
        from core.reality_reach.acceptance_service import (
            RealityAcceptanceService,
            capture_runtime_source_identity,
        )
        from core.reality_reach.digital_twin import RealityDigitalTwinGraph
        from core.reality_reach.event_flow import RealityEventFlowRuntime
        from core.reality_reach.historian import RealityHistorian
        from core.reality_reach.live import get_reality_reach_service
        from core.reality_reach.metrology import RealityMetrologyService
        from core.reality_reach.middleware import RealityMiddlewareRuntime
        from core.reality_reach.transactions import get_reality_actuation_coordinator

        service = get_reality_reach_service()
        await asyncio.to_thread(service.refresh)
        coordinator = await asyncio.to_thread(
            get_reality_actuation_coordinator,
            service,
        )
        orchestrator.reality_reach = service
        orchestrator.reality_actuation = coordinator
        metrology = None
        try:
            metrology_path = environment_runtime_file(
                "shared",
                "reality_metrology.json",
                purpose="state",
            )
            metrology = RealityMetrologyService(
                service,
                state_path=metrology_path,
            )
            await metrology.start()
        except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
            _record_cognitive_sensory_degradation(
                orchestrator,
                exc,
                phase="reality_metrology",
                action=(
                    "Kept bounded raw physical sensing available but closed calibrated, "
                    "synchronized, simulation, and hardware-in-loop evidence until "
                    "metrology-state recovery"
                ),
                severity="critical",
            )
        orchestrator.reality_metrology = metrology
        acceptance = None
        if metrology is not None:
            mandate_store = None
            try:
                mandate_path = environment_runtime_file(
                    "shared",
                    "reality_acceptance_mandates.encrypted.json",
                    purpose="state",
                )
                mandate_store = await asyncio.to_thread(
                    AcceptanceMandateStore.provision_system,
                    mandate_path,
                )
                source_identity = await asyncio.to_thread(capture_runtime_source_identity)
                acceptance = RealityAcceptanceService(
                    service,
                    metrology,
                    mandate_store=mandate_store,
                    pinned_source_identity=source_identity,
                )
            except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
                if mandate_store is not None:
                    await asyncio.to_thread(mandate_store.close)
                _record_cognitive_sensory_degradation(
                    orchestrator,
                    exc,
                    phase="reality_acceptance",
                    action=(
                        "Kept physical sensing and ordinary governed actuation available "
                        "but closed connector acceptance until source identity and "
                        "rollback-resistant mandate custody can be pinned"
                    ),
                    severity="critical",
                )
        orchestrator.reality_acceptance = acceptance
        middleware = None
        try:
            middleware_path = environment_runtime_file(
                "shared",
                "reality_middleware.json",
                purpose="state",
            )
            middleware = RealityMiddlewareRuntime(
                service,
                state_path=middleware_path,
            )
            await middleware.start()
        except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
            _record_cognitive_sensory_degradation(
                orchestrator,
                exc,
                phase="reality_middleware",
                action=(
                    "Kept raw sensing, history, and topology available but closed "
                    "managed telemetry, request/response, and long-running physical "
                    "actions until lifecycle-state recovery"
                ),
                severity="critical",
            )
        orchestrator.reality_middleware = middleware
        event_flow = None
        if middleware is not None:
            try:
                flow_path = environment_runtime_file(
                    "shared",
                    "reality_event_flows.json",
                    purpose="state",
                )
                event_flow = RealityEventFlowRuntime(
                    middleware=middleware,
                    state_path=flow_path,
                    worker_enabled=True,
                )
                await event_flow.start()
            except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
                _record_cognitive_sensory_degradation(
                    orchestrator,
                    exc,
                    phase="reality_event_flow",
                    action=(
                        "Kept direct managed physical endpoints available but closed "
                        "durable multi-step event flows until graph-state recovery"
                    ),
                    severity="critical",
                )
        orchestrator.reality_event_flow = event_flow
        historian = None
        try:
            historian_path = environment_runtime_file(
                "shared",
                "reality_historian.sqlite3",
                purpose="history",
            )
            historian = await asyncio.to_thread(RealityHistorian, historian_path)
        except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
            _record_cognitive_sensory_degradation(
                orchestrator,
                exc,
                phase="reality_historian",
                action=(
                    "Kept live physical sensing and actuation available without "
                    "durable history; alarms and store-and-forward remain explicitly "
                    "degraded until historian recovery"
                ),
                severity="warning",
            )
        orchestrator.reality_historian = historian
        digital_twin = None
        try:
            twin_path = environment_runtime_file(
                "shared",
                "reality_digital_twin.sqlite3",
                purpose="state",
            )
            digital_twin = await asyncio.to_thread(
                RealityDigitalTwinGraph,
                twin_path,
                session_id=service.session_id,
            )
            await asyncio.to_thread(digital_twin.reconcile_service, service)
        except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
            _record_cognitive_sensory_degradation(
                orchestrator,
                exc,
                phase="reality_digital_twin",
                action=(
                    "Kept raw physical inventory available but failed the canonical "
                    "sensory-fabric boot contract until topology and state projection recover"
                ),
                severity="critical",
            )
        orchestrator.reality_digital_twin = digital_twin
        ServiceContainer.register_instance(
            "reality_reach",
            service,
            required=False,
            owner="core/reality_reach/live.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="physical reachability and experiment evidence",
            failure_policy="degrade_with_receipt",
        )
        ServiceContainer.register_instance(
            "reality_actuation",
            coordinator,
            required=False,
            owner="core/reality_reach/transactions.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="governed physical actuation and effect reconciliation",
            failure_policy="degrade_with_receipt",
        )
        if metrology is not None:
            ServiceContainer.register_instance(
                "reality_metrology",
                metrology,
                required=True,
                owner="core/reality_reach/metrology.py",
                registered_by="init_cognitive_sensory_layer",
                required_for=(
                    "calibrated synchronized acquisition, uncertainty propagation, "
                    "and live/simulation/HIL evidence separation"
                ),
                failure_policy="fail-closed",
            )
        if acceptance is not None:
            ServiceContainer.register_instance(
                "reality_acceptance",
                acceptance,
                required=True,
                owner="core/reality_reach/acceptance_service.py",
                registered_by="init_cognitive_sensory_layer",
                required_for=(
                    "governed live and hardware-in-loop connector acceptance"
                ),
                failure_policy="fail-closed",
            )
        if middleware is not None:
            ServiceContainer.register_instance(
                "reality_middleware",
                middleware,
                required=True,
                owner="core/reality_reach/middleware.py",
                registered_by="init_cognitive_sensory_layer",
                required_for=(
                    "managed telemetry QoS, bounded services, and cancellable "
                    "restart-safe physical actions"
                ),
                failure_policy="fail-closed",
            )
        if event_flow is not None:
            ServiceContainer.register_instance(
                "reality_event_flow",
                event_flow,
                required=True,
                owner="core/reality_reach/event_flow.py",
                registered_by="init_cognitive_sensory_layer",
                required_for=("typed durable sensing, reasoning, and governed physical workflows"),
                failure_policy="fail-closed",
            )
        if historian is not None:
            ServiceContainer.register_instance(
                "reality_historian",
                historian,
                required=False,
                owner="core/reality_reach/historian.py",
                registered_by="init_cognitive_sensory_layer",
                required_for=(
                    "restart-safe physical history, alarms, quarantine, and "
                    "cognitive store-and-forward"
                ),
                failure_policy="degrade_with_receipt",
            )
        if digital_twin is not None:
            ServiceContainer.register_instance(
                "reality_digital_twin",
                digital_twin,
                required=True,
                owner="core/reality_reach/digital_twin.py",
                registered_by="init_cognitive_sensory_layer",
                required_for=("stable physical identity, topology, state, and migration receipts"),
                failure_policy="fail-closed",
            )
        report["registered"]["reality_reach"] = service.__class__.__name__
        report["registered"]["reality_actuation"] = coordinator.__class__.__name__
        if metrology is not None:
            report["registered"]["reality_metrology"] = metrology.__class__.__name__
        if middleware is not None:
            report["registered"]["reality_middleware"] = middleware.__class__.__name__
        if event_flow is not None:
            report["registered"]["reality_event_flow"] = event_flow.__class__.__name__
        if historian is not None:
            report["registered"]["reality_historian"] = historian.__class__.__name__
        if digital_twin is not None:
            report["registered"]["reality_digital_twin"] = digital_twin.__class__.__name__

    await _run_phase(
        orchestrator,
        "reality_reach",
        "Registered no physical reachability service; physical experiments remain unavailable",
        _reality_reach,
        severity="warning",
    )

    async def _reality_sensory_fabric() -> None:
        from core.environment.runtime_workspace import environment_runtime_file
        from core.reality_reach.attachments import DeviceAttachmentBroker
        from core.reality_reach.observation_router import RealityObservationRouter
        from core.reality_reach.trust_custody import KeychainAttachmentTrustStore

        reality_reach = getattr(orchestrator, "reality_reach", None)
        if reality_reach is None:
            raise RuntimeError("Reality Reach must be initialized before its sensory fabric")
        historian = getattr(orchestrator, "reality_historian", None)
        digital_twin = getattr(orchestrator, "reality_digital_twin", None)
        if digital_twin is None:
            raise RuntimeError("Reality digital twin is required for the canonical sensory fabric")
        router = RealityObservationRouter(
            reality_reach,
            historian=historian,
            digital_twin=digital_twin,
        )
        await router.start()
        try:
            trust_store = None
            trust_store_error = ""
            trust_state_path = environment_runtime_file(
                "shared",
                "reality_attachment_trust.json",
                purpose="identity",
            )
            try:
                trust_store = await asyncio.to_thread(
                    KeychainAttachmentTrustStore.provision_system,
                    trust_state_path,
                )
            except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
                trust_store_error = _error_summary(exc)
                _record_cognitive_sensory_degradation(
                    orchestrator,
                    exc,
                    phase="reality_attachment_trust_custody",
                    action=(
                        "Kept physical discovery and session observation active; "
                        "persistent attachment trust remains fail-closed until "
                        "macOS Keychain custody recovers"
                    ),
                    severity="warning",
                )
            broker = DeviceAttachmentBroker(
                reality_reach,
                router,
                digital_twin=digital_twin,
                middleware=getattr(orchestrator, "reality_middleware", None),
                state_path=trust_state_path,
                trust_store=trust_store,
                trust_store_error=trust_store_error,
                attachment_observer=getattr(
                    getattr(orchestrator, "reality_actuation", None),
                    "notify_adapter_available",
                    None,
                ),
            )
            await broker.start()
        except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS:
            await router.stop()
            raise
        orchestrator.reality_observation_router = router
        orchestrator.reality_attachment_broker = broker
        ServiceContainer.register_instance(
            "reality_observation_router",
            router,
            required=True,
            owner="core/reality_reach/observation_router.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="bounded physical exteroception and cognitive grounding",
            failure_policy="fail-closed",
        )
        ServiceContainer.register_instance(
            "reality_attachment_broker",
            broker,
            required=False,
            owner="core/reality_reach/attachments.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="portable device discovery, trust, and attachment",
            failure_policy="degrade_with_receipt",
        )
        report["registered"]["reality_observation_router"] = router.__class__.__name__
        report["registered"]["reality_attachment_broker"] = broker.__class__.__name__

    await _run_phase(
        orchestrator,
        "reality_sensory_fabric",
        "Physical channels remain inventoried but are not routed into cognition or portable attachment",
        _reality_sensory_fabric,
        severity="critical",
    )

    async def _reality_connectors() -> None:
        from core.embodiment.reality_connectors import (
            build_configured_reality_connector_catalog,
        )

        broker = getattr(orchestrator, "reality_attachment_broker", None)
        if broker is None:
            raise RuntimeError("Reality attachment broker must exist before connector registration")
        catalog = build_configured_reality_connector_catalog()
        if catalog.connectors:
            catalog.register_with(broker)
        orchestrator.reality_connector_catalog = catalog
        ServiceContainer.register_instance(
            "reality_connector_catalog",
            catalog,
            required=False,
            owner="core/embodiment/reality_connectors.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="portable protocol connector boot and configuration truth",
            failure_policy="degrade_with_receipt",
        )
        report["registered"]["reality_connector_catalog"] = catalog.__class__.__name__
        for status in catalog.status()["connectors"]:
            if status["state"] != "invalid":
                continue
            error = RuntimeError(
                f"{status['connector_id']}:{status['error'] or 'invalid_configuration'}"
            )
            _record_cognitive_sensory_degradation(
                orchestrator,
                error,
                phase=f"reality_connector.{status['connector_id']}",
                action=(
                    "Kept every other physical connector active; this connector "
                    "remains unavailable until its explicit configuration is valid"
                ),
                severity="warning",
            )

    await _run_phase(
        orchestrator,
        "reality_connectors",
        "Concrete physical connectors were not registered; the core Reality Reach fabric remains active",
        _reality_connectors,
        severity="warning",
    )

    async def _hardware_manager() -> None:
        from core.embodiment.hardware_manager import get_hardware_manager

        manager = get_hardware_manager()
        reality_reach = getattr(orchestrator, "reality_reach", None)
        if reality_reach is None:
            raise RuntimeError("Reality Reach must be initialized before hardware")
        manager.bind_reality_reach(reality_reach)
        observation_router = getattr(orchestrator, "reality_observation_router", None)
        if observation_router is None:
            raise RuntimeError("Reality observation routing must be initialized before hardware")
        manager.bind_observation_router(observation_router)
        manager.register_configured_devices()
        await manager.start()
        orchestrator.hardware_manager = manager
        ServiceContainer.register_instance(
            "hardware_manager",
            manager,
            required=False,
            owner="core/embodiment/hardware_manager.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="registered physical-device lifecycle and interlocks",
            failure_policy="degrade_with_receipt",
        )
        report["registered"]["hardware_manager"] = manager.__class__.__name__

    await _run_phase(
        orchestrator,
        "hardware_manager",
        "Registered no physical hardware manager; device actuation remains unavailable",
        _hardware_manager,
        severity="warning",
    )

    async def _iot_bridge() -> None:
        from core.embodiment.iot_bridge import get_iot_bridge

        reality_reach = getattr(orchestrator, "reality_reach", None)
        reality_actuation = getattr(orchestrator, "reality_actuation", None)
        if reality_reach is None or reality_actuation is None:
            raise RuntimeError("Reality Reach must be initialized before the IoT bridge")
        observation_router = getattr(orchestrator, "reality_observation_router", None)
        attachment_broker = getattr(orchestrator, "reality_attachment_broker", None)
        if observation_router is None or attachment_broker is None:
            raise RuntimeError("Reality sensory fabric must be initialized before the IoT bridge")
        bridge = get_iot_bridge()
        bridge.bind_reality_reach(reality_reach, reality_actuation)
        bridge.bind_sensory_fabric(observation_router, attachment_broker)
        await bridge.start()
        orchestrator.iot_bridge = bridge
        ServiceContainer.register_instance(
            "iot_bridge",
            bridge,
            required=False,
            owner="core/embodiment/iot_bridge.py",
            registered_by="init_cognitive_sensory_layer",
            required_for="causal environmental sensing and governed physical effects",
            failure_policy="degrade_with_receipt",
        )
        report["registered"]["iot_bridge"] = bridge.__class__.__name__

    await _run_phase(
        orchestrator,
        "iot_bridge",
        "Environmental coupling remained offline; physical IoT effects stay unavailable",
        _iot_bridge,
        severity="warning",
    )

    async def _reality_restart_recovery() -> None:
        coordinator = getattr(orchestrator, "reality_actuation", None)
        start = getattr(coordinator, "start_recovery_supervisor", None)
        wait_for_attempt = getattr(coordinator, "wait_for_recovery_attempt", None)
        if coordinator is None or not callable(start) or not callable(wait_for_attempt):
            raise RuntimeError("Reality actuation recovery supervision is unavailable")

        recovery_status = coordinator.status().get("restart_recovery") or {}
        recovery_generation = int(recovery_status.get("generation") or 0)
        orchestrator.reality_actuation_recovery_supervisor_task = await start(
            max_transactions=64,
        )

        async def _observe_initial_recovery() -> None:
            try:
                recovery_report = await wait_for_attempt(
                    after_generation=recovery_generation,
                    timeout_s=30.0,
                )
            except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
                _record_cognitive_sensory_degradation(
                    orchestrator,
                    exc,
                    phase="reality_actuation_restart_recovery",
                    action=(
                        "Kept requested physical effects closed and retained pending "
                        "transactions for a later source-bound recovery sweep"
                    ),
                    severity="critical",
                )
                return
            orchestrator.reality_actuation_recovery = recovery_report
            if not bool(recovery_report.get("complete")):
                error = RuntimeError(
                    "reality_actuation_restart_recovery_incomplete:"
                    f"failures={len(recovery_report.get('failures') or [])}:"
                    f"unresolved={len(recovery_report.get('unresolved') or [])}:"
                    "legacy="
                    f"{len(recovery_report.get('legacy_unrecoverable_transaction_sha256') or [])}:"
                    "capsule_only="
                    f"{len(recovery_report.get('capsule_without_transaction_sha256') or [])}:"
                    f"deferred={int(recovery_report.get('deferred') or 0)}"
                )
                _record_cognitive_sensory_degradation(
                    orchestrator,
                    error,
                    phase="reality_actuation_restart_recovery",
                    action=(
                        "Recovered every currently source-bound effect that reached "
                        "observed safe state and retained the remainder as explicit "
                        "manual or deferred reconciliation debt"
                    ),
                    severity="critical",
                )

        orchestrator.reality_actuation_recovery_task = create_tracked_task(
            _observe_initial_recovery(),
            name="reality_reach.restart_recovery_observer",
            owner="core.orchestrator.initializers.cognitive_sensory",
            bounded=True,
        )

    await _run_phase(
        orchestrator,
        "reality_restart_recovery",
        "Physical restart recovery was not scheduled; pending effects remain closed",
        _reality_restart_recovery,
        severity="critical",
    )

    async def _multimodal_orchestrator() -> None:
        from core.brain.multimodal_orchestrator import MultimodalOrchestrator

        _register(report, "multimodal_orchestrator", MultimodalOrchestrator())

    await _run_phase(
        orchestrator,
        "multimodal_orchestrator",
        "Skipped multimodal orchestrator registration; text cognition remains available",
        _multimodal_orchestrator,
        severity="warning",
    )

    async def _composer_node() -> None:
        from core.brain.composer_node import ComposerNode

        _register(report, "composer_node", ComposerNode())

    await _run_phase(
        orchestrator,
        "composer_node",
        "Skipped composer node registration; downstream composition will use direct response paths",
        _composer_node,
        severity="warning",
    )

    async def _memory_guard() -> None:
        from core.guardians.memory_guard import MemoryGuard

        memory_guard = MemoryGuard()
        await _maybe_await(memory_guard.start())
        _register(report, "memory_guard", memory_guard)

    await _run_phase(
        orchestrator,
        "memory_guard",
        "Skipped memory guard startup; health contract should treat memory protection as degraded",
        _memory_guard,
        severity="critical",
    )

    async def _resilience_engine() -> None:
        from core.soma.resilience_engine import ResilienceEngine

        resilience = ResilienceEngine(orchestrator)
        await _maybe_await(resilience.start())
        _register(report, "soma", resilience)
        _register(report, "resilience_engine", resilience)

    await _run_phase(
        orchestrator,
        "resilience_engine",
        "Skipped resilience engine startup; runtime repair loop remains degraded",
        _resilience_engine,
        severity="critical",
    )

    async def _identity_monitors() -> None:
        from core.identity.drift_monitor import IdentityDriftMonitor
        from core.identity.spine import SpiritualSpine

        drift_monitor = IdentityDriftMonitor()
        _register(report, "drift_monitor", drift_monitor)

        opinion_engine = ServiceContainer.get("opinion_engine", default=None)
        spine = SpiritualSpine(opinion_engine=opinion_engine)
        _register(report, "spine", spine)

    await _run_phase(
        orchestrator,
        "identity_monitors",
        "Skipped identity drift/spine monitors; self-integrity telemetry is degraded",
        _identity_monitors,
        severity="warning",
    )

    async def _self_modification_scaffolds() -> None:
        from core.memory.sovereign_pruner import SovereignPruner
        from core.self_modification.growth_ladder import GrowthLadder

        growth_ladder = GrowthLadder(orchestrator)
        _register(report, "growth_ladder", growth_ladder)

        pruner = SovereignPruner(orchestrator)
        _register(report, "sovereign_pruner", pruner)

    await _run_phase(
        orchestrator,
        "self_modification_scaffolds",
        "Skipped growth ladder/pruner registration; self-improvement maintenance is degraded",
        _self_modification_scaffolds,
        severity="warning",
    )

    async def _system_governor() -> None:
        from core.guardians.governor import SystemGovernor

        system_governor = SystemGovernor()
        await _maybe_await(system_governor.start())
        _register(report, "system_governor", system_governor)

    await _run_phase(
        orchestrator,
        "system_governor",
        "Skipped system governor startup; boot health must remain degraded",
        _system_governor,
        severity="critical",
    )

    async def _will_engine() -> None:
        from core.self.will_engine import WillEngine

        orchestrator.will_engine = WillEngine()
        await _maybe_await(orchestrator.will_engine.initialize())
        _register(report, "will_engine", orchestrator.will_engine)
        _register(report, "metabolic_coordinator", orchestrator.will_engine)
        orchestrator.metabolic_coordinator = orchestrator.will_engine
        logger.info("☘️ WillEngine (Metabolic Evolution) registered.")

    await _run_phase(
        orchestrator,
        "will_engine",
        "Skipped WillEngine registration; agency/metabolic coordination remains degraded",
        _will_engine,
        severity="critical",
    )

    # Learned cognitive systems replace rigid if/else rules with adaptive,
    # data-driven systems. Each is optional, but every deferral is recorded.
    cognitive_services = {
        "sentiment_tracker": ("core.cognitive.sentiment_tracker", "get_sentiment_tracker"),
        "anomaly_detector": ("core.cognitive.anomaly_detector", "AnomalyDetector"),
        "strange_loop": ("core.cognitive.strange_loop", "get_strange_loop"),
        "homeostatic_rl": ("core.cognitive.homeostatic_rl", "get_homeostatic_rl"),
        "topology_evolution": ("core.cognitive.topology_evolution", "TopologyEvolution"),
        "autopoiesis": ("core.cognitive.autopoiesis", "get_autopoiesis_engine"),
        "adaptive_immune_system": (
            "core.adaptation.adaptive_immunity",
            "get_adaptive_immune_system",
        ),
        "autonomous_resilience_mesh": (
            "core.adaptation.autonomous_resilience",
            "get_autonomous_resilience_mesh",
        ),
    }
    alife_services = {
        "criticality_regulator": (
            "core.consciousness.criticality_regulator",
            "get_criticality_regulator",
        ),
        "alife_dynamics": ("core.consciousness.alife_dynamics", "ALifeDynamics"),
        "alife_extensions": ("core.consciousness.alife_extensions", "ALifeExtensions"),
        "endogenous_fitness": ("core.consciousness.endogenous_fitness", "get_endogenous_fitness"),
    }
    all_services = {**cognitive_services, **alife_services}
    registered_count = 0
    for service_name, (module_path, factory_name) in all_services.items():
        try:
            module = importlib.import_module(module_path)
            factory = getattr(module, factory_name)
            instance = factory() if callable(factory) else factory
            _register(report, service_name, instance)
            registered_count += 1
        except _COGNITIVE_SENSORY_RECOVERABLE_ERRORS as exc:
            _record_cognitive_sensory_degradation(
                orchestrator,
                exc,
                phase=f"learned_service:{service_name}",
                action=f"Deferred optional learned/ALife service {service_name}; boot continues without it",
                severity="warning",
            )
            logger.debug("Cognitive/ALife service '%s' deferred: %s", service_name, exc)
    report["learned_services"] = {"registered": registered_count, "expected": len(all_services)}
    if registered_count:
        logger.info(
            "🧠 Registered %d/%d learned cognitive + ALife systems.",
            registered_count,
            len(all_services),
        )

    async def _cellular_substrate() -> None:
        from core.state.cellular_substrate import CellularSubstrate

        orchestrator.cellular_substrate = CellularSubstrate()
        await _maybe_await(orchestrator.cellular_substrate.initialize())
        _register(report, "cellular_substrate", orchestrator.cellular_substrate)
        logger.info("♾️ CellularSubstrate (Unified Mutation) registered.")

    await _run_phase(
        orchestrator,
        "cellular_substrate",
        "Skipped cellular substrate registration; unified mutation substrate remains degraded",
        _cellular_substrate,
        severity="critical",
    )

    async def _research_cognition() -> None:
        """Phase 3/4 research services the dream phase reads every cycle.

        ``MindTick._dream_research`` asks the container for these four by name
        and comments that they are "boot-registered singletons". Nothing ever
        registered them: the only registrar was a second, unreachable copy of
        this initializer at ``core/initializers/cognitive_sensory.py``, retired
        in 053b0a8ab because nothing called it. Every read returned None and
        every branch was skipped silently, so metacognitive reflection,
        intrinsic-motivation feeding and EWC consolidation never ran.
        """
        from core.adaptation.intrinsic_motivation import IntrinsicMotivationEngine
        from core.adaptation.plasticity_governor import get_plasticity_governor
        from core.meta.experience_distillery import ExperienceDistillery
        from core.meta.metacognitive_monitor import MetaCognitiveMonitor

        services: list[tuple[str, Any, str, str]] = [
            (
                "metacognitive_monitor",
                MetaCognitiveMonitor(),
                "core/meta/metacognitive_monitor.py",
                "dream-phase learning-condition assessment and strategy actions",
            ),
            (
                "intrinsic_motivation",
                IntrinsicMotivationEngine(),
                "core/adaptation/intrinsic_motivation.py",
                "competence/novelty rewards fed into the dynamic value graph",
            ),
            (
                "experience_distillery",
                ExperienceDistillery(),
                "core/meta/experience_distillery.py",
                "turning failure contexts into reusable lessons",
            ),
            (
                "plasticity_governor",
                get_plasticity_governor(),
                "core/adaptation/plasticity_governor.py",
                "EWC consolidation of weights during dream",
            ),
        ]
        for name, instance, owner, required_for in services:
            ServiceContainer.register_instance(
                name,
                instance,
                required=False,
                owner=owner,
                registered_by="init_cognitive_sensory_layer",
                required_for=required_for,
                failure_policy="degrade_without_dream_research",
            )
            report["registered"][name] = instance.__class__.__name__
        logger.info("🧠 Registered %d research cognition services.", len(services))

    await _run_phase(
        orchestrator,
        "research_cognition",
        "Skipped Phase 3/4 research cognition; dream-phase reflection and EWC stay inert",
        _research_cognition,
    )

    logger.info("🧬 [BOOT] Cognitive & Sensory Layer initialized.")
    return report
