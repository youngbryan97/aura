"""Semantic verification — multi-channel checks beyond a single evaluator.

The honest answer to "verify deep semantic improvements" is to refuse
to trust any one signal.  This module provides:

  * ``HashEmbedder`` — dependency-free signed n-gram embedder for
    paraphrase / consistency comparisons.
  * ``SemanticVerifier`` — three independent channels:
      - self-consistency over multiple samples
      - paraphrase invariance over an answer set
      - proof-carrying code (assertion presence + sandbox execution)
  * Optional integration with the F10 grounding service: a verifier
    can route grounded predictions through the prediction ledger so
    semantic checks share the same Brier track record everything else
    in Aura uses.
"""
from core.verification.embedder import HashEmbedder
from core.verification.semantic_verifier import (
    InvarianceResult,
    ProofCarryingResult,
    SelfConsistencyResult,
    SemanticVerifier,
)

__all__ = [
    "HashEmbedder",
    "InvarianceResult",
    "ProofCarryingResult",
    "SelfConsistencyResult",
    "SemanticVerifier",
]
