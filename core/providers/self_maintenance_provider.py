"""core/providers/self_maintenance_provider.py — her self-maintenance and artificial-life organs.

Every kernel tick, ``CognitiveIntegrationPhase`` asks the container for twelve
services by name, and the dream phase in ``MindTick`` asks for four more. Their
only registrar was ``init_cognitive_sensory_layer``, and nothing has called that
function since at least April 2026: ``boot.py`` imported it and never called it,
and Checkpoint 55 dropped even the import. So every read returned None and each
runner returned at its first line. Her mesh never got its Lenia coupling, her
entropy never accumulated or dissipated, the homeostatic drives never moved, the
autopoiesis engine never ran a repair cycle, adaptive immunity never saw an
antigen, and dream-phase consolidation never ran. Her app logged "Activation
audit passed: 100%" over all of it, because the audit's list did not name them.
The dead-letter queue the memory phase hands a failed commit to was missing the
same way, so a memory that failed to commit was kept only in the cold store.

They are registered here because this is the one path her app and the subject
battery both build her through (``register_all_services``), so the organism the
battery measures and the one Bryan talks to are the same organism.

Each is lazy: built on the first read, by the phase that uses it. A name that is
already registered is left alone, so a boot path that registers its own instance
first keeps it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.runtime.service_access import optional_service
from core.runtime.service_registry import SERVICE_LIFETIME_SINGLETON

logger = logging.getLogger("Aura.Providers.SelfMaintenance")


def _sentiment_tracker() -> Any:
    from core.cognitive.sentiment_tracker import get_sentiment_tracker

    return get_sentiment_tracker()


def _anomaly_detector() -> Any:
    from core.cognitive.anomaly_detector import AnomalyDetector

    return AnomalyDetector()


def _strange_loop() -> Any:
    from core.cognitive.strange_loop import get_strange_loop

    return get_strange_loop()


def _homeostatic_rl() -> Any:
    from core.cognitive.homeostatic_rl import get_homeostatic_rl

    return get_homeostatic_rl()


def _topology_evolution() -> Any:
    from core.cognitive.topology_evolution import TopologyEvolution

    return TopologyEvolution()


def _autopoiesis() -> Any:
    from core.cognitive.autopoiesis import get_autopoiesis_engine

    return get_autopoiesis_engine()


def _adaptive_immune_system() -> Any:
    from core.adaptation.adaptive_immunity import get_adaptive_immune_system

    return get_adaptive_immune_system()


def _autonomous_resilience_mesh() -> Any:
    from core.adaptation.autonomous_resilience import get_autonomous_resilience_mesh

    return get_autonomous_resilience_mesh()


def _criticality_regulator() -> Any:
    from core.consciousness.criticality_regulator import get_criticality_regulator

    return get_criticality_regulator()


def _alife_dynamics() -> Any:
    from core.consciousness.alife_dynamics import ALifeDynamics

    return ALifeDynamics()


def _mesh_columns() -> int:
    """How many columns her mesh has, so the extension layer sizes itself to it."""
    mesh = optional_service("neural_mesh", default=None)
    columns = getattr(getattr(mesh, "cfg", None), "columns", None)
    try:
        return int(columns) if columns else 64
    except (TypeError, ValueError):
        return 64


def _alife_extensions() -> Any:
    from core.consciousness.alife_extensions import ALifeExtensions

    return ALifeExtensions(n_columns=_mesh_columns())


def _endogenous_fitness() -> Any:
    from core.consciousness.endogenous_fitness import get_endogenous_fitness

    return get_endogenous_fitness()


def _metacognitive_monitor() -> Any:
    from core.meta.metacognitive_monitor import MetaCognitiveMonitor

    return MetaCognitiveMonitor()


def _intrinsic_motivation() -> Any:
    from core.adaptation.intrinsic_motivation import IntrinsicMotivationEngine

    return IntrinsicMotivationEngine()


def _experience_distillery() -> Any:
    from core.meta.experience_distillery import ExperienceDistillery

    return ExperienceDistillery()


def _plasticity_governor() -> Any:
    from core.adaptation.plasticity_governor import get_plasticity_governor

    return get_plasticity_governor()


def _dead_letter_queue() -> Any:
    from core.tasks.dead_letter_queue import get_dlq

    return get_dlq()


#: (name, factory, owner, what reads it). The phase that reads each name is the
#: one the test pairs it with, so a reader added there without a row here fails.
SELF_MAINTENANCE_SERVICES: tuple[tuple[str, Callable[[], Any], str, str], ...] = (
    ("sentiment_tracker", _sentiment_tracker, "core/cognitive/sentiment_tracker.py",
     "CognitiveIntegrationPhase: the person's tone into her affect"),
    ("anomaly_detector", _anomaly_detector, "core/cognitive/anomaly_detector.py",
     "CognitiveIntegrationPhase: the threat level immunity reads"),
    ("autonomous_resilience_mesh", _autonomous_resilience_mesh, "core/adaptation/autonomous_resilience.py",
     "CognitiveIntegrationPhase: runtime audit, and the health probes and repair handlers autopoiesis acts through"),
    ("adaptive_immune_system", _adaptive_immune_system, "core/adaptation/adaptive_immunity.py",
     "CognitiveIntegrationPhase: antigens, clonal selection and bounded effectors"),
    ("strange_loop", _strange_loop, "core/cognitive/strange_loop.py",
     "CognitiveIntegrationPhase: her prediction of her own next state"),
    ("homeostatic_rl", _homeostatic_rl, "core/cognitive/homeostatic_rl.py",
     "CognitiveIntegrationPhase: energy, drives and action preferences"),
    ("topology_evolution", _topology_evolution, "core/cognitive/topology_evolution.py",
     "CognitiveIntegrationPhase: connections her mesh grows and prunes"),
    ("autopoiesis", _autopoiesis, "core/cognitive/autopoiesis.py",
     "CognitiveIntegrationPhase: health monitoring and the repair cycle"),
    ("endogenous_fitness", _endogenous_fitness, "core/consciousness/endogenous_fitness.py",
     "CognitiveIntegrationPhase: the survival watch"),
    ("criticality_regulator", _criticality_regulator, "core/consciousness/criticality_regulator.py",
     "CognitiveIntegrationPhase: gain and noise toward the critical point"),
    ("alife_dynamics", _alife_dynamics, "core/consciousness/alife_dynamics.py",
     "CognitiveIntegrationPhase: Lenia coupling, entropy and compute credit"),
    ("alife_extensions", _alife_extensions, "core/consciousness/alife_extensions.py",
     "CognitiveIntegrationPhase: pattern replication, speciation and thermodynamic cost"),
    ("metacognitive_monitor", _metacognitive_monitor, "core/meta/metacognitive_monitor.py",
     "MindTick dream phase: learning-condition assessment"),
    ("intrinsic_motivation", _intrinsic_motivation, "core/adaptation/intrinsic_motivation.py",
     "MindTick dream phase: competence and novelty rewards"),
    ("experience_distillery", _experience_distillery, "core/meta/experience_distillery.py",
     "MindTick dream phase: failures into reusable lessons"),
    ("plasticity_governor", _plasticity_governor, "core/adaptation/plasticity_governor.py",
     "MindTick dream phase: EWC consolidation of weights"),
    ("dead_letter_queue", _dead_letter_queue, "core/tasks/dead_letter_queue.py",
     "MemoryConsolidationPhase: a memory commit that failed, kept whole for retry; stall-recovery receipts"),
)

#: Names her code reads that mean a service registered under another name.
#: ``homeostasis_engine`` is read by the latent bridge, the IoT bridge and the
#: substrate voice; the engine has only ever been registered as ``homeostasis``.
ALIASES: tuple[tuple[str, str], ...] = (
    ("homeostasis_engine", "homeostasis"),
    ("homeostatic_engine", "homeostasis"),
)


def register_self_maintenance_services(container: Any) -> list[str]:
    """Register each organ that is not registered yet. Returns the names registered."""
    added: list[str] = []
    for name, factory, owner, required_for in SELF_MAINTENANCE_SERVICES:
        if container.has(name):
            continue
        container.register(
            name,
            factory,
            lifetime=SERVICE_LIFETIME_SINGLETON,
            required=False,
            owner=owner,
            registered_by="register_self_maintenance_services",
            required_for=required_for,
            failure_policy="degrade_without_this_organ",
        )
        added.append(name)
    for alias, target in ALIASES:
        if container.has(alias):
            continue
        container.register(
            alias,
            lambda target=target: container.get(target, default=None),
            lifetime=SERVICE_LIFETIME_SINGLETON,
            required=False,
            owner="core/providers/self_maintenance_provider.py",
            registered_by="register_self_maintenance_services",
            required_for=f"readers that ask for {target} as {alias}",
            failure_policy="degrade_without_this_organ",
        )
        added.append(alias)
    if added:
        logger.info("Registered %d self-maintenance and artificial-life organs.", len(added))
    return added
