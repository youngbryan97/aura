"""core/resilience/fmea_registry.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure Mode and Effects Analysis (FMEA) registry — production reliability oriented.

Provides queryable access to all Aura failure modes with:
- Risk Priority Number (RPN) scoring
- Automated mitigation action mapping
- Runbook linkage
- Runtime occurrence tracking (bridges to FaultRegistry)

Usage:
    from core.resilience.fmea_registry import get_fmea_registry

    fmea = get_fmea_registry()
    report = fmea.full_report()
    high_risk = fmea.faults_above_rpn(30)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from core.resilience.fault_taxonomy import get_fault_registry

logger = logging.getLogger("Aura.FMEA")


@dataclass
class MitigationAction:
    """A concrete mitigation action for a failure mode."""
    action_id: str
    description: str
    automated: bool
    implementation_path: str  # file path to the implementing code
    verified: bool = False    # whether the mitigation has been tested
    last_verified: float | None = None


@dataclass
class FMEAEntry:
    """A complete FMEA row linking fault definition to mitigations."""
    fault_id: str
    mitigations: list[MitigationAction] = field(default_factory=list)
    notes: str = ""
    last_reviewed: float = field(default_factory=time.time)
    review_owner: str = "system"

    def is_mitigated(self) -> bool:
        """A fault is considered mitigated if it has at least one verified
        automated mitigation."""
        return any(m.automated and m.verified for m in self.mitigations)


class FMEARegistry:
    """FMEA registry bridging fault definitions to mitigation actions.

    Thread-safe. Read-heavy workload optimized.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, FMEAEntry] = {}
        self._register_builtins()

    def register(self, entry: FMEAEntry) -> None:
        with self._lock:
            self._entries[entry.fault_id] = entry

    def get_entry(self, fault_id: str) -> FMEAEntry | None:
        with self._lock:
            return self._entries.get(fault_id)

    def faults_above_rpn(self, threshold: int) -> list[dict[str, Any]]:
        """Return all faults with RPN above the given threshold."""
        fault_reg = get_fault_registry()
        results = []
        with self._lock:
            for fault_id, entry in self._entries.items():
                defn = fault_reg.get_definition(fault_id)
                if defn and defn.rpn >= threshold:
                    results.append({
                        "fault_id": fault_id,
                        "name": defn.name,
                        "rpn": defn.rpn,
                        "severity": defn.severity.name,
                        "mitigated": entry.is_mitigated(),
                        "mitigation_count": len(entry.mitigations),
                    })
        results.sort(key=lambda x: x["rpn"], reverse=True)
        return results

    def unmitigated_faults(self) -> list[str]:
        """Return fault IDs that lack verified automated mitigations."""
        with self._lock:
            return [fid for fid, entry in self._entries.items()
                    if not entry.is_mitigated()]

    def full_report(self) -> list[dict[str, Any]]:
        """Generate a complete FMEA report."""
        fault_reg = get_fault_registry()
        report = []
        with self._lock:
            for fault_id, entry in self._entries.items():
                defn = fault_reg.get_definition(fault_id)
                if not defn:
                    continue
                report.append({
                    "fault_id": fault_id,
                    "name": defn.name,
                    "description": defn.description,
                    "domain": defn.domain.value,
                    "severity": defn.severity.name,
                    "probability": defn.probability.name,
                    "detection": defn.detection.name,
                    "rpn": defn.rpn,
                    "recovery": defn.recovery.value,
                    "mttr_target_s": defn.mttr_seconds,
                    "blast_radius": defn.blast_radius,
                    "runbook": defn.runbook,
                    "mitigated": entry.is_mitigated(),
                    "mitigation_count": len(entry.mitigations),
                    "mitigations": [
                        {
                            "action_id": m.action_id,
                            "description": m.description,
                            "automated": m.automated,
                            "verified": m.verified,
                            "impl": m.implementation_path,
                        }
                        for m in entry.mitigations
                    ],
                    "notes": entry.notes,
                    "review_owner": entry.review_owner,
                })
        report.sort(key=lambda x: x["rpn"], reverse=True)
        return report

    def coverage_summary(self) -> dict[str, Any]:
        """Summary stats for reliability gate checks."""
        fault_reg = get_fault_registry()
        total_defined = len(fault_reg.all_definitions())
        with self._lock:
            total_fmea = len(self._entries)
            mitigated = sum(1 for e in self._entries.values() if e.is_mitigated())
            unmitigated = total_fmea - mitigated
            missing = total_defined - total_fmea
        return {
            "total_fault_definitions": total_defined,
            "total_fmea_entries": total_fmea,
            "mitigated": mitigated,
            "unmitigated": unmitigated,
            "missing_fmea_entries": missing,
            "coverage_pct": round(total_fmea / max(total_defined, 1) * 100, 1),
            "mitigation_pct": round(mitigated / max(total_fmea, 1) * 100, 1),
        }

    def _register_builtins(self) -> None:
        """Register FMEA entries for all built-in fault definitions."""
        entries = {
            "F01": FMEAEntry(fault_id="F01", mitigations=[
                MitigationAction("MIT-F01-1", "Boot probe validates model files",
                                 automated=True, implementation_path="core/brain/llm/mlx_worker.py",
                                 verified=True),
                MitigationAction("MIT-F01-2", "`make doctor` validates runtime prerequisites",
                                 automated=False, implementation_path="Makefile",
                                 verified=True),
            ]),
            "F02": FMEAEntry(fault_id="F02", mitigations=[
                MitigationAction("MIT-F02-1", "MLX client respawns crashed worker within 30s",
                                 automated=True, implementation_path="core/brain/llm/mlx_client.py",
                                 verified=True),
                MitigationAction("MIT-F02-2", "Worker health probe detects crash",
                                 automated=True, implementation_path="infrastructure/watchdog.py",
                                 verified=True),
            ]),
            "F03": FMEAEntry(fault_id="F03", mitigations=[
                MitigationAction("MIT-F03-1", "SQLite integrity check on boot",
                                 automated=True, implementation_path="core/resilience/startup_validator.py",
                                 verified=True),
                MitigationAction("MIT-F03-2", "`make restore` from last backup + WAL replay",
                                 automated=False, implementation_path="core/backup.py",
                                 verified=True),
            ]),
            "F04": FMEAEntry(fault_id="F04", mitigations=[
                MitigationAction("MIT-F04-1", "Bounded shutdown with 12s budget",
                                 automated=True, implementation_path="core/graceful_shutdown.py",
                                 verified=True),
                MitigationAction("MIT-F04-2", "Stall watchdog forces exit",
                                 automated=True, implementation_path="core/resilience/stall_watchdog.py",
                                 verified=True),
            ]),
            "F05": FMEAEntry(fault_id="F05", mitigations=[
                MitigationAction("MIT-F05-1", "Privacy classification gate before cloud fallback",
                                 automated=True, implementation_path="core/brain/llm/provider_contract.py",
                                 verified=True),
                MitigationAction("MIT-F05-2", "Cloud fallback audit log",
                                 automated=True, implementation_path="core/audit_logger.py",
                                 verified=True),
            ]),
            "F06": FMEAEntry(fault_id="F06", mitigations=[
                MitigationAction("MIT-F06-1", "Multi-layer prompt sanitizer + integrity check",
                                 automated=True, implementation_path="core/security/",
                                 verified=True),
                MitigationAction("MIT-F06-2", "Will receipt chain for anomaly detection",
                                 automated=True, implementation_path="core/governance/will.py",
                                 verified=True),
            ]),
            "F07": FMEAEntry(fault_id="F07", mitigations=[
                MitigationAction("MIT-F07-1", "Metabolic monitor + resource governor",
                                 automated=True, implementation_path="core/resilience/resource_governor.py",
                                 verified=True),
                MitigationAction("MIT-F07-2", "Automatic tier demotion under pressure",
                                 automated=True, implementation_path="core/resilience/memory_governor.py",
                                 verified=True),
            ]),
            "F08": FMEAEntry(fault_id="F08", mitigations=[
                MitigationAction("MIT-F08-1", "Task tracker orphan detection",
                                 automated=True, implementation_path="core/reaper.py",
                                 verified=True),
            ]),
            "F09": FMEAEntry(fault_id="F09", mitigations=[
                MitigationAction("MIT-F09-1", "Periodic memory consolidation cycle",
                                 automated=True, implementation_path="core/memory/",
                                 verified=True),
            ]),
            "F10": FMEAEntry(fault_id="F10", mitigations=[
                MitigationAction("MIT-F10-1", "CanonicalSelf hash integrity check",
                                 automated=True, implementation_path="core/identity/",
                                 verified=True),
                MitigationAction("MIT-F10-2", "Stem cell reversion from canonical snapshot",
                                 automated=True, implementation_path="core/resilience/stem_cell.py",
                                 verified=True),
            ]),
            "F11": FMEAEntry(fault_id="F11", mitigations=[
                MitigationAction("MIT-F11-1", "Timeout enforcement + degradation recording",
                                 automated=True, implementation_path="core/tools/",
                                 verified=True),
            ]),
            "F12": FMEAEntry(fault_id="F12", mitigations=[
                MitigationAction("MIT-F12-1", "Lock watchdog releases stale locks",
                                 automated=True, implementation_path="core/resilience/lock_watchdog.py",
                                 verified=True),
            ]),
            "F13": FMEAEntry(fault_id="F13", mitigations=[
                MitigationAction("MIT-F13-1", "Log write error detection",
                                 automated=True, implementation_path="core/logging_config.py",
                                 verified=True),
            ]),
            "F14": FMEAEntry(fault_id="F14", mitigations=[
                MitigationAction("MIT-F14-1", "Telemetry health check",
                                 automated=True, implementation_path="core/observability/telemetry.py",
                                 verified=True),
            ]),
            "F15": FMEAEntry(fault_id="F15", mitigations=[
                MitigationAction("MIT-F15-1", "Verified state machine enforces transitions",
                                 automated=True, implementation_path="core/resilience/verified_state_machine.py",
                                 verified=False),
            ], notes="Mitigation added during reliability hardening"),
            "F16": FMEAEntry(fault_id="F16", mitigations=[
                MitigationAction("MIT-F16-1", "Event bus backpressure via bounded queue",
                                 automated=True, implementation_path="core/event_bus.py",
                                 verified=True),
            ]),
            "F17": FMEAEntry(fault_id="F17", mitigations=[
                MitigationAction("MIT-F17-1", "Verified state machine rejects illegal transitions",
                                 automated=True, implementation_path="core/resilience/verified_state_machine.py",
                                 verified=False),
            ], notes="Mitigation added during reliability hardening"),
            "F18": FMEAEntry(fault_id="F18", mitigations=[
                MitigationAction("MIT-F18-1", "SLO burn-rate monitor with alerts",
                                 automated=True, implementation_path="slo/slo_monitor.py",
                                 verified=False),
            ], notes="Mitigation added during reliability hardening"),
            "F19": FMEAEntry(fault_id="F19", mitigations=[
                MitigationAction("MIT-F19-1", "TMR divergence detection and quarantine",
                                 automated=True, implementation_path="core/resilience/tmr.py",
                                 verified=False),
            ], notes="Mitigation added during reliability hardening"),
            "AFFECT-TRAP": FMEAEntry(fault_id="AFFECT-TRAP", mitigations=[
                MitigationAction("MIT-AT-1",
                                 "Anti-trap guard: bounded exploration escape when "
                                 "temperature pins at floor with non-improving distress; "
                                 "repair/ideation lanes get unconditional exploration floors",
                                 automated=True,
                                 implementation_path="core/brain/affective_antitrap.py",
                                 verified=True),
            ], notes="Digital-depression loop identified by external review July 3"),
            "WILL-REFUSE": FMEAEntry(fault_id="WILL-REFUSE", mitigations=[
                MitigationAction("MIT-WR-1",
                                 "No mitigation required: refusal IS governance working "
                                 "as designed; occurrences recorded (recovered=True) for "
                                 "forensic traceability only",
                                 automated=True,
                                 implementation_path="core/governance/will.py",
                                 verified=True),
            ], notes="Not a defect class — traceability entry"),
            "F20": FMEAEntry(fault_id="F20", mitigations=[
                MitigationAction("MIT-F20-1", "Design-by-contract log+continue enforcement",
                                 automated=True, implementation_path="core/resilience/contracts.py",
                                 verified=False),
            ], notes="Mitigation added during reliability hardening"),
            "ACTION-CLAIM-MISMATCH": FMEAEntry(
                fault_id="ACTION-CLAIM-MISMATCH",
                mitigations=[
                    MitigationAction(
                        "MIT-ACM-1",
                        "Action expectation contract downgrades successful "
                        "returns when effect evidence or acceptance criteria are missing",
                        automated=True,
                        implementation_path="core/runtime/skill_contract.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-ACM-2",
                        "CapabilityEngine enforces explicit action expectations "
                        "before returning success to callers",
                        automated=True,
                        implementation_path="core/capability_engine.py",
                        verified=True,
                    ),
                ],
                notes="Grounding gap closed by Pass F action-depth contract",
            ),
            "PASSF-ACTION-SHALLOW-SUCCESS": FMEAEntry(
                fault_id="PASSF-ACTION-SHALLOW-SUCCESS",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-ACTION-1",
                        "ActionExpectation evaluates user-visible effect, "
                        "acceptance criteria, evidence, and repair hints before "
                        "a result can remain success_verified",
                        automated=True,
                        implementation_path="core/runtime/skill_contract.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-ACTION-2",
                        "Desktop chat objectives pass expectation contracts "
                        "through the live capability execution lane",
                        automated=True,
                        implementation_path="interface/routes/chat.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-FALSE-HEALTH": FMEAEntry(
                fault_id="PASSF-FALSE-HEALTH",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-HEALTH-1",
                        "Production readiness gate separates proof readiness, "
                        "chat readiness, and health blockers",
                        automated=True,
                        implementation_path="tools/aura_production_readiness_gate.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-HEALTH-2",
                        "Live boot proof reports explicit readiness blockers "
                        "instead of treating boot as full user-path proof",
                        automated=True,
                        implementation_path="tools/live_boot_proof.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-RESOURCE-SPAWN-LOOP": FMEAEntry(
                fault_id="PASSF-RESOURCE-SPAWN-LOOP",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-RESOURCE-1",
                        "Resource governor tracks pressure and applies automatic "
                        "degradation before unbounded work piles up",
                        automated=True,
                        implementation_path="core/resilience/resource_governor.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-RESOURCE-2",
                        "Memory governor can demote tiers under pressure",
                        automated=True,
                        implementation_path="core/resilience/memory_governor.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-DESKTOP-PERMISSION-DRIFT": FMEAEntry(
                fault_id="PASSF-DESKTOP-PERMISSION-DRIFT",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-DESKTOP-1",
                        "Chat desktop verifier rejects critical-step success "
                        "without observable effect receipts",
                        automated=True,
                        implementation_path="interface/routes/chat.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-DESKTOP-2",
                        "Browser/desktop actor lifecycle is covered by runtime "
                        "hardening runbooks and leakage tests",
                        automated=True,
                        implementation_path="tests/test_server_runtime_hardening.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-REPAIR-STORM": FMEAEntry(
                fault_id="PASSF-REPAIR-STORM",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-REPAIR-1",
                        "Will/governance receipts give repair actions an "
                        "auditable decision boundary",
                        automated=True,
                        implementation_path="core/governance/will.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-REPAIR-2",
                        "Self-repair runbook defines safe mitigation, rollback, "
                        "and postmortem requirements",
                        automated=False,
                        implementation_path="docs/runbooks/self-repair-failed.md",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-STALE-OBLIGATION": FMEAEntry(
                fault_id="PASSF-STALE-OBLIGATION",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-STALE-1",
                        "Grounded recall keeps remembered context tied to "
                        "evidence instead of letting stale assertions dominate",
                        automated=True,
                        implementation_path="core/conversation/grounded_recall.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-STALE-2",
                        "Remaining checkpoint contract keeps open obligations "
                        "explicit rather than implicit prompt residue",
                        automated=True,
                        implementation_path="tools/closeout/remaining_checkpoint_contract.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-NEURAL-STREAM-FLOOD": FMEAEntry(
                fault_id="PASSF-NEURAL-STREAM-FLOOD",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-STREAM-1",
                        "Diagnostics dashboard exposes summarized FMEA, high-risk, "
                        "and unmitigated state instead of raw stream volume",
                        automated=True,
                        implementation_path="core/resilience/diagnostics_dashboard.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-STREAM-2",
                        "Distributed tracing uses per-trace sampling decisions "
                        "to prevent incoherent observability floods",
                        automated=True,
                        implementation_path="core/observability/tracing.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-VISIBLE-WEB-PROOF-ACCESS": FMEAEntry(
                fault_id="PASSF-VISIBLE-WEB-PROOF-ACCESS",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-WEB-1",
                        "Web interlocutor proof path records browser-access "
                        "blockers instead of silently substituting weak proof",
                        automated=True,
                        implementation_path="tools/proof/run_web_interlocutor_live_proof.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-WEB-2",
                        "Live boot proof keeps visible browser proof separate "
                        "from core boot readiness",
                        automated=True,
                        implementation_path="tools/live_boot_proof.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-PROOF-ARTIFACT-CONTAMINATION": FMEAEntry(
                fault_id="PASSF-PROOF-ARTIFACT-CONTAMINATION",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-PROOF-1",
                        "Proof fabrication guard rejects hardcoded pass/fail "
                        "scores and fixture-backed victory claims",
                        automated=True,
                        implementation_path="tools/proof_fabrication_guard.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-PROOF-2",
                        "Proof step runner wraps each proof command with hard "
                        "timeout and evidence artifact metadata",
                        automated=True,
                        implementation_path="tools/run_proof_step.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
            "PASSF-SEMANTIC-REVIEW-GAP": FMEAEntry(
                fault_id="PASSF-SEMANTIC-REVIEW-GAP",
                mitigations=[
                    MitigationAction(
                        "MIT-PASSF-SEMANTIC-1",
                        "Semantic review ledger records per-file behavioral "
                        "review evidence separate from mechanical hashing",
                        automated=True,
                        implementation_path="tools/closeout/semantic_review_ledger.py",
                        verified=True,
                    ),
                    MitigationAction(
                        "MIT-PASSF-SEMANTIC-2",
                        "Codebase closeout audit reports semantic review "
                        "incompleteness instead of closing on text counts alone",
                        automated=True,
                        implementation_path="tools/closeout/run_codebase_closeout_audit.py",
                        verified=True,
                    ),
                ],
                notes="Pass F structural maturity risk",
            ),
        }
        for fid, entry in entries.items():
            self._entries[fid] = entry


# ── Module singleton ─────────────────────────────────────────────────

_fmea: FMEARegistry | None = None
_fmea_lock = threading.Lock()


def get_fmea_registry() -> FMEARegistry:
    global _fmea
    if _fmea is None:
        with _fmea_lock:
            if _fmea is None:
                _fmea = FMEARegistry()
    return _fmea
