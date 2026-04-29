"""SelfImprovingResearchCore — Aura's autonomous self-improvement substrate.

This is the integration layer that owns:

  F17 LatticeLM           — the model itself
  F18 PromotionGate       — monotone checkpoint promotion
      DynamicBenchmark    — secret evals
      HoldoutVault        — anti-contamination
  F19 ExpressionEvolver   — bounded algorithm discovery
      SafeCodeEvaluator   — AST-restricted sandbox
  F20 SemanticVerifier    — multi-channel semantic checks
  F21 NoveltyArchive      — failure-finding test diversity
      UnknownUnknownGen   — adversarial task synthesis
      EmbeddingEntropyProbe — model-internal blind-spot probes
  F22 DistributedGradientSync + Int8Compressor — hardware escape

Wired into existing Aura:

  F1  audit chain         — every promotion + discovery emits a
                            tamper-evident receipt
  F2  prediction ledger   — Brier/calibration of capability estimates
  F3  task lifecycle      — initiatives tracked through proposed →
                            accepted → planned → testing → completed
  F4  mutation safety     — subprocess isolation for code candidates
  F5  doctor --bundle     — research_core status surfaces in the
                            diagnostics tarball
  F9  curriculum loop     — gap detector + generated unknowns feed
                            new initiatives
  F10 grounding service   — semantic verification leans on grounded
                            predictions when wired
  F12 plugin allowlist    — promoted candidates must be hash-approved
  F14 tenant boundary     — single-tenant install boundary respected
  F16 Will gate           — promotions consult the Will before commit

Aura-owned, not operator-owned: the core registers itself in the
ServiceContainer at boot, runs cycles when the autonomy budget allows
it, and emits its own receipts.  No operator intervention is required.
"""
from core.research_core.core import (
    CycleReport,
    ResearchCoreConfig,
    SelfImprovingResearchCore,
)
from core.research_core.doctor import collect_research_core_status
from core.research_core.registry import register_research_core

__all__ = [
    "CycleReport",
    "ResearchCoreConfig",
    "SelfImprovingResearchCore",
    "collect_research_core_status",
    "register_research_core",
]
