from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import Severity, record_degradation

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
        from core.brain.personality_engine import PersonalityEngine
        from core.fictional_ai_synthesis import register_all_fictional_engines
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

        orchestrator.soul = Soul(orchestrator)
        _register(report, "soul", orchestrator.soul)

        orchestrator.fictional_engines = register_all_fictional_engines(orchestrator)

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
            if not hasattr(orchestrator.affect, "drive_controller") or not orchestrator.affect.drive_controller:
                orchestrator.affect.drive_controller = DriveController()
            _register(report, "drive_engine", orchestrator.affect.drive_controller)
            _register(report, "drives", orchestrator.affect.drive_controller)
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
        "adaptive_immune_system": ("core.adaptation.adaptive_immunity", "get_adaptive_immune_system"),
        "autonomous_resilience_mesh": ("core.adaptation.autonomous_resilience", "get_autonomous_resilience_mesh"),
    }
    alife_services = {
        "criticality_regulator": ("core.consciousness.criticality_regulator", "get_criticality_regulator"),
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

    logger.info("🧬 [BOOT] Cognitive & Sensory Layer initialized.")
    return report
