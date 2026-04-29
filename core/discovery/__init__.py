"""Bounded algorithm discovery for Aura.

Honest answer to "reliably discover new scientific algorithms": a
constrained evolutionary search over small expression trees plus an
AST-restricted Python sandbox that delegates subprocess isolation to
F4's ``SafeMutationEvaluator``.

The discovery loop:
  1. ``ExpressionEvolver`` samples + mutates ``SafeExpression`` ASTs
     and scores them against a deterministic example set.
  2. ``SafeCodeEvaluator`` runs candidate Python in an isolated
     subprocess with an AST allowlist + rlimits + timeout.  Any
     non-PASSED outcome lands in F4's quarantine.
  3. Discovered winners are emitted as ``StateMutationReceipt`` to
     the F1 audit chain so a future auditor can replay every
     successful candidate.

This is "discovery in bounded domains," not "discover novel
mathematics on its own."  The bounded search is the honest version.
"""
from core.discovery.expression import SafeExpression
from core.discovery.evolver import EvolverResult, ExpressionEvolver
from core.discovery.code_eval import (
    DiscoveryEvaluation,
    SafeCodeEvaluator,
)

__all__ = [
    "DiscoveryEvaluation",
    "EvolverResult",
    "ExpressionEvolver",
    "SafeCodeEvaluator",
    "SafeExpression",
]
