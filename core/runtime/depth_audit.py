"""Depth Audit framework — Tier 0-5 classification.

The audit explicitly demands that every flagship module declares its
operational depth:

  Tier 0  name only / stub
  Tier 1  wrapper / adapter
  Tier 2  heuristic signal
  Tier 3  stateful subsystem
  Tier 4  closed-loop subsystem (input -> state -> behavior -> verify -> learn)
  Tier 5  causally necessary integrated subsystem (ablation tests prove it)

Modules register a ``DepthReport`` describing how many native computational
steps they take, how many LLM delegations they perform, whether they have
durable state, whether they close the loop with verification, whether an
ablation test exists, and whether they integrate with governance.

A ``DepthRegistry`` aggregates reports. In strict runtime mode, any
flagship module below Tier 4 fails the audit. The harness is callable
from boot (``enforce_depth_audit``) and from tests.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("Aura.DepthAudit")


# Modules whose ambitious names (per the audit) *demand* Tier 4+.
FLAGSHIP_MODULES = frozenset(
    {
        "intersubjectivity_engine",
        "abstraction_engine",
        "adaptive_immune_system",
        "latent_bridge",
        "consciousness_integration",
        "alignment_auditor",
        "cognitive_trainer",
        "ghost_probe",
        "self_modification_engine",
        "memory_write_gateway",
        "state_gateway",
        "unified_will",
    }
)


@dataclass
class DepthReport:
    module: str
    native_steps: int
    llm_delegations: int
    durable_state: bool
    closed_loop: bool
    ablation_test: bool
    governance_integrated: bool
    tier: int
    notes: str = ""

    def is_flagship(self) -> bool:
        return self.module in FLAGSHIP_MODULES

    def __post_init__(self) -> None:
        if not 0 <= self.tier <= 5:
            raise ValueError("tier must be 0..5")
        if self.native_steps < 0 or self.llm_delegations < 0:
            raise ValueError("native_steps/llm_delegations must be >= 0")


@dataclass
class DepthAuditResult:
    reports: List[DepthReport] = field(default_factory=list)
    failures: List[DepthReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


class DepthRegistry:
    def __init__(self):
        self._reports: Dict[str, DepthReport] = {}

    def register(self, report: DepthReport) -> None:
        if report.module in self._reports:
            existing = self._reports[report.module]
            if existing.tier > report.tier:
                logger.warning(
                    "DepthAudit: refusing to lower tier for %s (was %d, attempted %d)",
                    report.module,
                    existing.tier,
                    report.tier,
                )
                return
        self._reports[report.module] = report

    def get(self, module: str) -> Optional[DepthReport]:
        return self._reports.get(module)

    def all(self) -> List[DepthReport]:
        return list(self._reports.values())

    def clear(self) -> None:
        self._reports.clear()

    def audit(self) -> DepthAuditResult:
        result = DepthAuditResult(reports=self.all())
        for r in result.reports:
            if r.is_flagship() and r.tier < 4:
                result.failures.append(r)
        return result


_global_registry: Optional[DepthRegistry] = None


def get_depth_registry() -> DepthRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = DepthRegistry()
    return _global_registry


def reset_depth_registry() -> None:
    global _global_registry
    _global_registry = None


def enforce_depth_audit() -> DepthAuditResult:
    """Run the audit and (in strict mode) raise on failure."""
    result = get_depth_registry().audit()
    if not result.passed:
        msg = "; ".join(
            f"{r.module}@T{r.tier}" for r in result.failures
        )
        if os.environ.get("AURA_STRICT_RUNTIME") == "1":
            raise RuntimeError(
                f"AURA_STRICT_RUNTIME: flagship modules below Tier 4: {msg}"
            )
        logger.warning("DepthAudit: flagship modules below Tier 4: %s", msg)
    return result


# Convenience ----------------------------------------------------------------


def report(
    module: str,
    *,
    native_steps: int,
    llm_delegations: int = 0,
    durable_state: bool = False,
    closed_loop: bool = False,
    ablation_test: bool = False,
    governance_integrated: bool = False,
    tier: int,
    notes: str = "",
) -> DepthReport:
    rep = DepthReport(
        module=module,
        native_steps=native_steps,
        llm_delegations=llm_delegations,
        durable_state=durable_state,
        closed_loop=closed_loop,
        ablation_test=ablation_test,
        governance_integrated=governance_integrated,
        tier=tier,
        notes=notes,
    )
    get_depth_registry().register(rep)
    return rep
