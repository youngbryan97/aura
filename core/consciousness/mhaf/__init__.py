"""MHAF — Mycelial Hypergraph Attractor Field.

A living hypergraph consciousness substrate that encodes Aura's cognitive
state as HRR vectors, minimizes free energy via gradient descent, and
auto-modifies its own structure based on epistemic outcomes.
"""
from .hrr import HRREncoder
from .phi_estimator import compute_local_phi

__all__ = ["HRREncoder", "compute_local_phi"]
