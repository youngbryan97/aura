"""
core/consciousness/__init__.py
==============================
Exports for the consciousness evolutionary layers.
"""

from core.consciousness.integration import (
    ConsciousnessAugmentor,
    ConsciousnessIntegration,
    get_consciousness_integration,
)
from core.consciousness.continuous_experience import (
    ContinuousExperienceStream,
    ExperienceFrame,
    get_continuous_experience_stream,
)
from core.consciousness.phenomenological_experiencer import (
    AttentionSchema,
    PhenomenologicalExperiencer,
    Quale,
    get_experiencer,
)
from core.consciousness.system import ConsciousnessSystem

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
]
