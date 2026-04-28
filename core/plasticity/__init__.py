"""Backpropamine-style neuromodulated plasticity.

Small bounded plastic adapters whose weights update during life via
Hebbian + eligibility traces gated by a neuromodulatory signal.
External semantic reward (from the F2 prediction ledger or the F10
grounding service) drives the update; a governor decides when to
allow it.

The base LLM weights are NEVER touched by this module.  Plasticity is
encapsulated in adapter layers that reweight features before
downstream consumers see them.
"""
from core.plasticity.neuromodulated_plasticity import (
    NeuromodulatedPlasticLayer,
    PlasticityConfig,
)
from core.plasticity.plastic_adapter import GroundingPlasticAdapter
from core.plasticity.semantic_weight_governor import (
    PlasticityDecision,
    SemanticWeightGovernor,
)

__all__ = [
    "GroundingPlasticAdapter",
    "NeuromodulatedPlasticLayer",
    "PlasticityConfig",
    "PlasticityDecision",
    "SemanticWeightGovernor",
]
