from typing import Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state.aura_state import AuraState
else:
    from ..state.aura_state import AuraState

# Canonical base class lives in kernel.bridge. All phases — whether part of the
# legacy CognitiveEngine pipeline or the AuraKernel unitary pipeline — share it.
from core.kernel.bridge import Phase

PhaseCallable = Callable[[AuraState], Awaitable[AuraState]]


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

from .sensory_ingestion import SensoryIngestionPhase
from .memory_retrieval import MemoryRetrievalPhase
from .affect_update import AffectUpdatePhase
from .executive_closure import ExecutiveClosurePhase
from .cognitive_routing import CognitiveRoutingPhase
from .cognitive_routing_unitary import CognitiveRoutingPhase as CognitiveRoutingPhaseUnitary
from .response_generation import ResponseGenerationPhase
from .memory_consolidation import MemoryConsolidationPhase
from .identity_reflection import IdentityReflectionPhase
from .initiative_generation import InitiativeGenerationPhase
from .consciousness_phase import ConsciousnessPhase
from .social_context_phase import SocialContextPhase
from .inference_phase import InferencePhase
from .bonding_phase import BondingPhase
from .repair_phase import RepairPhase
from .unity_binding import UnityBindingPhase
