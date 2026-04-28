"""Symbol grounding via a semiotic network + neuromodulated plasticity.

This subpackage implements the first 9 phases of the AURA-style
grounding plan: types, semiotic network, sensory kernel, prediction
ledger linkage, and a plasticity-fed grounding service.  See
``core/plasticity/`` for the Backpropamine plastic adapter that gates
weight updates on external semantic reward.
"""
from core.grounding.types import (
    GroundedConcept,
    GroundingEvent,
    GroundingMethod,
    PerceptualEvidence,
    SymbolLink,
    new_id,
)
from core.grounding.semiotic_network import SemioticNetwork
from core.grounding.grounding_kernel import GroundingKernel, GroundingObservation
from core.grounding.grounding_service import GroundingService

__all__ = [
    "GroundedConcept",
    "GroundingEvent",
    "GroundingKernel",
    "GroundingMethod",
    "GroundingObservation",
    "GroundingService",
    "PerceptualEvidence",
    "SemioticNetwork",
    "SymbolLink",
    "new_id",
]
