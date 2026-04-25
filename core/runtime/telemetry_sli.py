"""Aura SLI / SLO declarations.

The audit calls for measurable golden signals plus Aura-specific signals
(receipt coverage, event loop lag, memory write success, checkpoint
success, actor restart, tool verification, hallucination correction,
movie comment timing, conversation repair). This module captures the SLO
catalog as data so dashboards and alerts can be derived from a single
source of truth and tested for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SLO:
    name: str
    description: str
    target: float
    unit: str
    pageable: bool


SLO_CATALOG: Dict[str, SLO] = {
    "runtime_availability": SLO(
        name="runtime_availability",
        description="Aura accepts local user input during single-user operation",
        target=0.995,
        unit="ratio",
        pageable=True,
    ),
    "event_loop_lag_p99_idle": SLO(
        name="event_loop_lag_p99_idle",
        description="p99 event loop lag while idle/background",
        target=0.250,
        unit="seconds",
        pageable=True,
    ),
    "event_loop_lag_p99_active": SLO(
        name="event_loop_lag_p99_active",
        description="p99 event loop lag during active local inference",
        target=1.000,
        unit="seconds",
        pageable=False,
    ),
    "fast_path_response_p99": SLO(
        name="fast_path_response_p99",
        description="99% of non-model fast-path responses complete < 500ms",
        target=0.500,
        unit="seconds",
        pageable=False,
    ),
    "memory_write_durability": SLO(
        name="memory_write_durability",
        description="acknowledged memory writes survive clean restart",
        target=1.0,
        unit="ratio",
        pageable=True,
    ),
    "governance_receipt_coverage": SLO(
        name="governance_receipt_coverage",
        description="consequential actions with a governance receipt",
        target=1.0,
        unit="ratio",
        pageable=True,
    ),
    "ungoverned_tool_executions_strict": SLO(
        name="ungoverned_tool_executions_strict",
        description="ungoverned tool executions in strict mode",
        target=0.0,
        unit="count",
        pageable=True,
    ),
    "clean_restart_recovery_seconds": SLO(
        name="clean_restart_recovery_seconds",
        description="restart from clean shutdown restores state",
        target=30.0,
        unit="seconds",
        pageable=False,
    ),
    "actor_critical_unready_seconds": SLO(
        name="actor_critical_unready_seconds",
        description="crashed critical actor flips readiness false within",
        target=2.0,
        unit="seconds",
        pageable=True,
    ),
    "actor_optional_unready_seconds": SLO(
        name="actor_optional_unready_seconds",
        description="crashed optional actor detected within",
        target=5.0,
        unit="seconds",
        pageable=False,
    ),
    "self_mod_unvalidated_commits": SLO(
        name="self_mod_unvalidated_commits",
        description="self-modification commits without ladder validation",
        target=0.0,
        unit="count",
        pageable=True,
    ),
    "checkpoint_success_rate": SLO(
        name="checkpoint_success_rate",
        description="checkpoint write success rate",
        target=0.999,
        unit="ratio",
        pageable=True,
    ),
}


REQUIRED_SLO_NAMES = frozenset(SLO_CATALOG.keys())


def required_pageable_slos() -> Dict[str, SLO]:
    return {name: slo for name, slo in SLO_CATALOG.items() if slo.pageable}
