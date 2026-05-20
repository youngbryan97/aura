# ruff: noqa: N999
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state.aura_state import AuraState

# Canonical base class lives in kernel.bridge. All phases — whether part of the
# legacy CognitiveEngine pipeline or the AuraKernel unitary pipeline — share it.
from core.kernel.bridge import Phase

PhaseCallable = Callable[["AuraState"], Awaitable["AuraState"]]


class BasePhase(Phase):
    """Legacy-pipeline phase base class.

    Extends the canonical ``Phase`` so that ``isinstance(x, Phase)`` works
    across both pipelines. The legacy pipeline does not use a kernel reference,
    so ``kernel`` defaults to ``None``; subclasses may receive a
    ``container`` argument instead.
    """

    def __init__(self, container=None):
        super().__init__(kernel=None)
        self.container = container

_PHASE_EXPORTS = {
    "AuraState": ("..state.aura_state", "AuraState"),
    "SensoryIngestionPhase": (".sensory_ingestion", "SensoryIngestionPhase"),
    "MemoryRetrievalPhase": (".memory_retrieval", "MemoryRetrievalPhase"),
    "AffectUpdatePhase": (".affect_update", "AffectUpdatePhase"),
    "ExecutiveClosurePhase": (".executive_closure", "ExecutiveClosurePhase"),
    "CognitiveRoutingPhase": (".cognitive_routing", "CognitiveRoutingPhase"),
    "CognitiveRoutingPhaseUnitary": (".cognitive_routing_unitary", "CognitiveRoutingPhase"),
    "ResponseGenerationPhase": (".response_generation", "ResponseGenerationPhase"),
    "MemoryConsolidationPhase": (".memory_consolidation", "MemoryConsolidationPhase"),
    "IdentityReflectionPhase": (".identity_reflection", "IdentityReflectionPhase"),
    "InitiativeGenerationPhase": (".initiative_generation", "InitiativeGenerationPhase"),
    "ConsciousnessPhase": (".consciousness_phase", "ConsciousnessPhase"),
    "SocialContextPhase": (".social_context_phase", "SocialContextPhase"),
    "InferencePhase": (".inference_phase", "InferencePhase"),
    "BondingPhase": (".bonding_phase", "BondingPhase"),
    "RepairPhase": (".repair_phase", "RepairPhase"),
    "UnityBindingPhase": (".unity_binding", "UnityBindingPhase"),
}


def __getattr__(name: str):
    """Lazy phase exports keep lightweight submodule imports dependency-light."""
    if name not in _PHASE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _PHASE_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BasePhase",
    "Phase",
    "PhaseCallable",
    *_PHASE_EXPORTS.keys(),
]
