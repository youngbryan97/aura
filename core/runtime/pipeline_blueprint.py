from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.phases.affect_update import AffectUpdatePhase
from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.phases.consciousness_phase import ConsciousnessPhase
from core.phases.executive_closure import ExecutiveClosurePhase
from core.phases.identity_reflection import IdentityReflectionPhase
from core.phases.initiative_generation import InitiativeGenerationPhase
from core.phases.memory_consolidation import MemoryConsolidationPhase
from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.phases.proprioceptive_loop import ProprioceptiveLoop
from core.phases.response_generation import ResponseGenerationPhase
from core.phases.sensory_ingestion import SensoryIngestionPhase
from core.phases.social_context_phase import SocialContextPhase


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    attribute_name: str
    phase_cls: type[Any]


_LEGACY_PIPELINE_PREFIX: tuple[PhaseSpec, ...] = (
    PhaseSpec("proprioceptive_loop", "proprioceptive_phase", ProprioceptiveLoop),
    PhaseSpec("social_context", "social_context_phase", SocialContextPhase),
    PhaseSpec("sensory_ingestion", "sensory_ingestion_phase", SensoryIngestionPhase),
    PhaseSpec("memory_retrieval", "memory_retrieval_phase", MemoryRetrievalPhase),
    PhaseSpec("affect_update", "affect_phase", AffectUpdatePhase),
)

_LEGACY_PIPELINE_SUFFIX: tuple[PhaseSpec, ...] = (
    PhaseSpec("cognitive_routing", "routing_phase", CognitiveRoutingPhase),
    PhaseSpec("response_generation", "response_phase", ResponseGenerationPhase),
    PhaseSpec("memory_consolidation", "memory_consolidation_phase", MemoryConsolidationPhase),
    PhaseSpec("identity_reflection", "identity_reflection_phase", IdentityReflectionPhase),
    PhaseSpec("initiative_generation", "initiative_generation_phase", InitiativeGenerationPhase),
    PhaseSpec("consciousness", "consciousness_phase", ConsciousnessPhase),
)

_EXECUTIVE_CLOSURE_SPEC = PhaseSpec(
    "executive_closure",
    "executive_closure_phase",
    ExecutiveClosurePhase,
)

_KERNEL_PIPELINE_ATTRIBUTE_ORDER: tuple[str, ...] = (
    "proprioceptive_phase",
    "social_context_phase",
    "sensory_ingestion_phase",
    "multimodal",
    "eternal",
    "memory_retrieval_phase",
    "perfect_emotion",
    "affect_phase",
    "phi_phase",
    "motivation_phase",
    "cognitive_integration",  # Learned cognitive systems (sentiment, anomaly, self-model, RL, plasticity)
    "executive_closure_phase",
    "evolution_guard",
    "growth",
    "evolution",
    "inference_phase",
    "conversational_dynamics_phase",
    "bonding_phase",
    "routing_phase",
    "godmode_tools",
    "response_phase",
    "repair_phase",
    "memory_consolidation_phase",
    "identity_reflection_phase",
    "initiative_generation_phase",
    "consciousness_phase",
    "self_review_phase",
    "learning_phase",
    "legacy_bridge",
)


def legacy_runtime_phase_specs(*, include_executive_closure: bool) -> tuple[PhaseSpec, ...]:
    specs = list(_LEGACY_PIPELINE_PREFIX)
    if include_executive_closure:
        specs.append(_EXECUTIVE_CLOSURE_SPEC)
    specs.extend(_LEGACY_PIPELINE_SUFFIX)
    return tuple(specs)


def instantiate_legacy_runtime_phases(
    owner: Any,
    *,
    include_executive_closure: bool,
) -> list[tuple[str, Any]]:
    return [
        (spec.name, spec.phase_cls(owner))
        for spec in legacy_runtime_phase_specs(
            include_executive_closure=include_executive_closure,
        )
    ]


def bind_legacy_runtime_phase_attributes(
    target: Any,
    owner: Any,
    *,
    include_executive_closure: bool,
) -> None:
    for spec in legacy_runtime_phase_specs(
        include_executive_closure=include_executive_closure,
    ):
        setattr(target, spec.attribute_name, spec.phase_cls(owner))


def kernel_phase_attribute_order() -> tuple[str, ...]:
    return _KERNEL_PIPELINE_ATTRIBUTE_ORDER


def resolve_phase_instances(target: Any, attribute_order: tuple[str, ...]) -> list[Any]:
    instances: list[Any] = []
    for attribute_name in attribute_order:
        phase = getattr(target, attribute_name, None)
        if phase is None:
            raise AttributeError(
                f"Expected phase attribute '{attribute_name}' to be initialized before pipeline resolution."
            )
        instances.append(phase)
    return instances
