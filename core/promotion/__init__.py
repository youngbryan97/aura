"""Promotion gate + anti-overfit substrate.

The honest answer to "guarantee monotonic intelligence gain" plus
"prevent benchmark overfitting": a checkpoint promotion gate that
refuses to promote regressions on a locked, statistically meaningful
score vector, fed by a procedural ``DynamicBenchmark`` plus a
``HoldoutVault`` whose answers never enter training and a
``LeakageDetector`` that catches contamination.

Wires into existing Aura modules:
  * F1 audit chain — every PromotionDecision emits a
    ``GovernanceReceipt`` so the chain has a forensic record.
  * F2 prediction ledger — the gate accepts a callable that produces
    a ``ScoreEstimate`` per metric.
"""
from core.promotion.gate import (
    PromotionDecision,
    PromotionGate,
    ScoreEstimate,
)
from core.promotion.dynamic_benchmark import DynamicBenchmark, Task
from core.promotion.holdout_vault import HoldoutVault, LeakageDetector

__all__ = [
    "DynamicBenchmark",
    "HoldoutVault",
    "LeakageDetector",
    "PromotionDecision",
    "PromotionGate",
    "ScoreEstimate",
    "Task",
]
