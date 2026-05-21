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
    "ConsciousnessSystem",
]


def __getattr__(name: str):
    if name == "ConsciousnessSystem":
        from core.consciousness.system import ConsciousnessSystem

        return ConsciousnessSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
