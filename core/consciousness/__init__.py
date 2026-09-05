"""
core/consciousness/__init__.py
==============================
Exports for the consciousness evolutionary layers.
"""

from core.consciousness.continuous_experience import (
    ContinuousExperienceStream,
    ExperienceFrame,
    get_continuous_experience_stream,
)
from core.consciousness.coordinator import (
    ConsciousnessCoordinator,
    get_consciousness_coordinator,
)
from core.consciousness.identity_driver import (
    IdentityDriver,
    get_identity_driver,
)
from core.consciousness.integration import (
    ConsciousnessAugmentor,
    ConsciousnessIntegration,
    get_consciousness_integration,
)
from core.consciousness.phenomenological_experiencer import (
    AttentionSchema,
    PhenomenologicalExperiencer,
    Quale,
    get_experiencer,
)
from core.consciousness.self_awareness import (
    SelfAwareness,
    get_self_awareness,
)
from core.consciousness.unified_self import (
    SelfState,
    UnifiedSelf,
    get_unified_self,
)

__all__ = [
    "PhenomenologicalExperiencer",
    "AttentionSchema",
    "Quale",
    "get_experiencer",
    "ConsciousnessAugmentor",
    "ConsciousnessIntegration",
    "get_consciousness_integration",
    "ContinuousExperienceStream",
    "ExperienceFrame",
    "get_continuous_experience_stream",
    "UnifiedSelf",
    "get_unified_self",
    "SelfState",
    "SelfAwareness",
    "get_self_awareness",
    "IdentityDriver",
    "get_identity_driver",
    "ConsciousnessCoordinator",
    "get_consciousness_coordinator",
    "ConsciousnessSystem",
    "PhenomenalKnowingKernel",
    "RecursiveSelfKnowingKernel",
    "AutomaticSelfKnowingKernel",
]


def __getattr__(name: str):
    if name == "ConsciousnessSystem":
        from core.consciousness.system import ConsciousnessSystem

        return ConsciousnessSystem
    if name == "PhenomenalKnowingKernel":
        from core.consciousness.phenomenal_knowing import PhenomenalKnowingKernel

        return PhenomenalKnowingKernel
    if name == "RecursiveSelfKnowingKernel":
        from core.consciousness.recursive_self_knowing import RecursiveSelfKnowingKernel

        return RecursiveSelfKnowingKernel
    if name == "AutomaticSelfKnowingKernel":
        from core.consciousness.automatic_self_knowing import AutomaticSelfKnowingKernel

        return AutomaticSelfKnowingKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
