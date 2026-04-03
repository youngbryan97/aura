"""PNEUMA — Precision-weighted Neural Epistemic Unified Manifold Architecture.

Active inference engine that modulates Aura's attention, belief integration,
and response selection via free-energy minimization.
"""
from .pneuma import PNEUMA, get_pneuma

__all__ = ["PNEUMA", "get_pneuma"]
