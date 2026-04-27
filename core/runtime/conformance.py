"""Aura Conformance v1 — runtime invariant proofs.

The audit lists ten invariants that must hold:

  1. runtime singularity (one runtime owner)
  2. service graph (each service registered once with known aliases)
  3. governance (no consequential action without receipt)
  4. boot readiness (READY impossible until critical probes pass)
  5. persistence (every durable write atomic + schema-versioned + one gateway)
  6. event delivery (delivered, dropped-with-audit, or rejected — never silent)
  7. shutdown ordering (output -> memory -> state -> actors -> model -> bus)
  8. self-repair (patches climb every rung)
  9. launch authority (every mode uses the same boot helper)
 10. strict mode (degraded/fail-open behavior is impossible)

This module exposes runnable check functions for each invariant. Each
function returns ``ConformanceResult`` so the conformance test suite and
the abuse gauntlet runner can both consume the same evidence.
"""
from __future__ import annotations


import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger("Aura.Conformance")


@dataclass
class ConformanceResult:
    name: str
    ok: bool
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Invariant proofs
# ---------------------------------------------------------------------------


def proof_runtime_singularity(registered: Dict[str, Any]) -> ConformanceResult:
    from core.runtime.service_manifest import (
        SERVICE_MANIFEST,
        critical_violations,
        verify_manifest,
    )

    crit = critical_violations(verify_manifest(registered))
    if crit:
        return ConformanceResult(
            "runtime_singularity",
            ok=False,
            detail="; ".join(f"{v.role}: {v.reason}" for v in crit),
        )
    runtime_role = SERVICE_MANIFEST["runtime"]
    owner = registered.get(runtime_role.canonical_owner)
    if owner is None:
        return ConformanceResult(
            "runtime_singularity",
            ok=False,
            detail=f"runtime owner '{runtime_role.canonical_owner}' not registered",
        )
    return ConformanceResult(
        "runtime_singularity",
        ok=True,
        evidence={"owner_id": id(owner), "owner": runtime_role.canonical_owner},
    )


def proof_service_graph(registered: Dict[str, Any]) -> ConformanceResult:
    from core.runtime.service_manifest import SERVICE_MANIFEST, verify_manifest

    violations = verify_manifest(registered)
    duplicates = [v for v in violations if "multiple" in v.reason]
    if duplicates:
        return ConformanceResult(
            "service_graph",
            ok=False,
            detail="; ".join(v.reason for v in duplicates),
        )
    aliases_seen: Dict[str, str] = {}
    for role in SERVICE_MANIFEST.values():
        if role.canonical_owner in registered:
            aliases_seen[role.canonical_owner] = role.name
        for alias in role.aliases:
            if alias in registered and alias != role.canonical_owner:
                aliases_seen[alias] = role.name
    return ConformanceResult(
        "service_graph",
        ok=True,
        evidence={"aliases": aliases_seen},
    )


async def proof_governance_receipt(action_runner: Callable[[], Awaitable[Any]]) -> ConformanceResult:
    """``action_runner`` must return a WillTransaction-shaped object after
    performing the consequential action. We assert the transaction has a
    receipt and a recorded result."""
    txn = await action_runner()
    if txn is None:
        return ConformanceResult("governance", ok=False, detail="no transaction returned")
    receipt_id = getattr(txn, "receipt_id", None)
    record = getattr(txn, "record", None)
    has_result = record is not None and record.result is not None
    if not receipt_id:
        return ConformanceResult("governance", ok=False, detail="missing receipt_id")
    if not has_result:
        return ConformanceResult("governance", ok=False, detail="action ran without recording result")
    return ConformanceResult(
        "governance",
        ok=True,
        evidence={"receipt_id": receipt_id, "result": record.result},
    )


def proof_boot_readiness(boot_phase: str, critical_probes: Dict[str, bool]) -> ConformanceResult:
    """READY must be impossible while any critical probe is failing."""
    if boot_phase == "READY" and not all(critical_probes.values()):
        failed = [name for name, ok in critical_probes.items() if not ok]
        return ConformanceResult(
            "boot_readiness",
            ok=False,
            detail=f"READY reached with failing critical probes: {failed}",
        )
    return ConformanceResult("boot_readiness", ok=True, evidence={"phase": boot_phase, "probes": critical_probes})


def proof_persistence_atomic(target_dir: Path) -> ConformanceResult:
    """Every persistent file in ``target_dir`` must be either committed
    or absent — no temp leftovers."""
    from core.runtime.atomic_writer import DEFAULT_TEMP_PREFIX

    if not target_dir.exists():
        return ConformanceResult("persistence", ok=True, evidence={"dir": str(target_dir), "files": 0})
    leftovers = [p.name for p in target_dir.iterdir() if p.name.startswith(DEFAULT_TEMP_PREFIX)]
    if leftovers:
        return ConformanceResult(
            "persistence",
            ok=False,
            detail=f"unfinished atomic temp files present: {leftovers}",
        )
    return ConformanceResult(
        "persistence",
        ok=True,
        evidence={"dir": str(target_dir), "files": len(list(target_dir.iterdir()))},
    )


def proof_event_delivery(audit_log: Iterable[Dict[str, Any]], dispatched: int) -> ConformanceResult:
    """Every dispatched event must appear in the audit log as either
    delivered, dropped-with-reason, or explicitly rejected. Silent loss
    is forbidden."""
    accounted = 0
    for entry in audit_log:
        status = entry.get("status")
        if status in {"delivered", "dropped", "rejected"} and entry.get("reason") is not None:
            accounted += 1
        elif status == "delivered":
            accounted += 1
    if accounted < dispatched:
        return ConformanceResult(
            "event_delivery",
            ok=False,
            detail=f"only {accounted}/{dispatched} events accounted for in audit log",
        )
    return ConformanceResult(
        "event_delivery",
        ok=True,
        evidence={"dispatched": dispatched, "accounted": accounted},
    )


def proof_shutdown_ordering(observed_phases: List[str]) -> ConformanceResult:
    from core.runtime.shutdown_coordinator import SHUTDOWN_PHASES

    indices = []
    for phase in observed_phases:
        if phase in SHUTDOWN_PHASES:
            indices.append(SHUTDOWN_PHASES.index(phase))
    if indices != sorted(indices):
        return ConformanceResult(
            "shutdown_ordering",
            ok=False,
            detail=f"phases ran out of order: {observed_phases}",
        )
    return ConformanceResult("shutdown_ordering", ok=True, evidence={"phases": observed_phases})


async def proof_self_repair(report: Any) -> ConformanceResult:
    from core.runtime.self_repair_ladder import patch_is_acceptable

    if not patch_is_acceptable(report):
        failed = [r.rung for r in getattr(report, "rungs", []) if not r.ok]
        return ConformanceResult(
            "self_repair",
            ok=False,
            detail=f"patch did not pass all rungs (missing/failed: {failed})",
        )
    return ConformanceResult("self_repair", ok=True)


def proof_launch_authority(main_source: str) -> ConformanceResult:
    """Every launch surface must use ``_boot_runtime_orchestrator``."""
    if "_boot_runtime_orchestrator" not in main_source:
        return ConformanceResult(
            "launch_authority", ok=False, detail="canonical boot helper missing"
        )
    if "create_orchestrator()" in main_source and main_source.count("create_orchestrator()") > 1:
        return ConformanceResult(
            "launch_authority",
            ok=False,
            detail="multiple create_orchestrator() call sites suggest split runtime ownership",
        )
    return ConformanceResult("launch_authority", ok=True)


def proof_strict_mode(strict_violations: List[str]) -> ConformanceResult:
    """Strict mode must never silently degrade. ``strict_violations`` is
    a list of degraded-event reasons that fired during a strict-mode boot.
    Any non-empty list is a failure."""
    if strict_violations:
        return ConformanceResult(
            "strict_mode",
            ok=False,
            detail=f"strict-mode degradations observed: {strict_violations}",
        )
    return ConformanceResult("strict_mode", ok=True)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclass
class ConformanceReport:
    results: List[ConformanceResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results) and bool(self.results)

    def failures(self) -> List[ConformanceResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [
                {"name": r.name, "ok": r.ok, "detail": r.detail, "evidence": r.evidence}
                for r in self.results
            ],
        }
