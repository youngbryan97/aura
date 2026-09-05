"""core/resilience/fault_taxonomy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Formal fault classification system modeled after general hazard
severity categories and production reliability analysis practices.

Every fault in Aura is classified by severity, affected subsystem,
detection method, recovery strategy, and MTTR target. All
record_degradation() calls feed into this taxonomy for unified
fault accounting and reliability-grade traceability.

Usage:
    from core.resilience.fault_taxonomy import (
        FaultSeverity, FaultEntry, FaultRegistry, get_fault_registry,
    )

    registry = get_fault_registry()
    registry.record_fault("FAULT-MEM-001", subsystem="memory_facade",
                          details="SQLite integrity check failed")

    # Query
    critical = registry.faults_by_severity(FaultSeverity.CATASTROPHIC)
    recent = registry.recent_faults(window_s=3600)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

logger = logging.getLogger("Aura.FaultTaxonomy")


# ── Severity Categories ──────────────────────────────────

class FaultSeverity(IntEnum):
    """general hazard severity categories.

    Lower numeric value = higher severity. This ordering allows
    natural comparison: ``if fault.severity <= FaultSeverity.CRITICAL``.
    """
    CATASTROPHIC = 1   # System death, data loss, identity corruption
    CRITICAL = 2       # Major subsystem failure, degraded with user impact
    MARGINAL = 3       # Minor degradation, automatic recovery expected
    NEGLIGIBLE = 4     # Cosmetic, no functional impact


class FaultProbability(IntEnum):
    """general probability levels.

    Used in Risk Priority Number calculation.
    """
    FREQUENT = 5       # Likely to occur repeatedly
    PROBABLE = 4       # Will occur several times in system life
    OCCASIONAL = 3     # Likely to occur sometime
    REMOTE = 2         # Unlikely but possible
    IMPROBABLE = 1     # So unlikely it can be assumed it will not occur


class DetectionDifficulty(IntEnum):
    """How hard the fault is to detect before user impact.

    Lower = harder to detect = higher risk.
    """
    UNDETECTABLE = 5   # No automated detection possible
    LOW = 4            # Manual inspection only
    MODERATE = 3       # Detected by periodic health check
    HIGH = 2           # Detected by real-time monitor
    CERTAIN = 1        # Self-reporting or impossible to miss


class RecoveryStrategy(StrEnum):
    """Canonical recovery strategies."""
    AUTOMATIC_RESTART = "automatic_restart"
    AUTOMATIC_FALLBACK = "automatic_fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    STEM_CELL_REVERT = "stem_cell_revert"
    MANUAL_INTERVENTION = "manual_intervention"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    QUARANTINE = "quarantine"
    IGNORE = "ignore"


class FaultDomain(StrEnum):
    """Top-level fault domains mapping to Aura subsystems."""
    INFERENCE = "inference"
    MEMORY = "memory"
    IDENTITY = "identity"
    GOVERNANCE = "governance"
    NETWORK = "network"
    STORAGE = "storage"
    BOOT = "boot"
    SHUTDOWN = "shutdown"
    RESOURCE = "resource"
    SECURITY = "security"
    OBSERVABILITY = "observability"
    TOOL_EXECUTION = "tool_execution"
    CONSCIOUSNESS = "consciousness"
    AGENCY = "agency"


# ── Fault Definition ─────────────────────────────────────────────────

@dataclass(frozen=True)
class FaultDefinition:
    """Static definition of a known fault mode.

    This is the reliability-grade catalog entry — defined at code time,
    not at runtime. Think of it as the FMEA row.
    """
    fault_id: str                          # e.g., "FAULT-MEM-001"
    name: str                              # Human-readable short name
    description: str                       # What happens
    domain: FaultDomain
    severity: FaultSeverity
    probability: FaultProbability
    detection: DetectionDifficulty
    recovery: RecoveryStrategy
    mttr_seconds: float                    # Mean Time To Recovery target
    blast_radius: str                      # What else breaks
    runbook: str = ""                      # Path to runbook

    @property
    def rpn(self) -> int:
        """Risk Priority Number = Severity × Probability × Detection difficulty.

        Range: 1–125. Higher = more risk. Used to prioritize mitigation.
        """
        return int(self.severity) * int(self.probability) * int(self.detection)


# ── Runtime Fault Record ─────────────────────────────────────────────

@dataclass
class FaultRecord:
    """Runtime occurrence of a fault."""
    fault_id: str
    subsystem: str
    severity: FaultSeverity
    timestamp: float = field(default_factory=lambda: time.time())
    details: str = ""
    error_type: str = ""
    error_message: str = ""
    recovered: bool = False
    recovery_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "subsystem": self.subsystem,
            "severity": self.severity.name,
            "timestamp": self.timestamp,
            "details": self.details[:500],
            "error_type": self.error_type,
            "error_message": self.error_message[:200],
            "recovered": self.recovered,
            "recovery_time_s": self.recovery_time_s,
        }


# ── Fault Registry ───────────────────────────────────────────────────

class FaultRegistry:
    """Central registry for fault definitions and runtime occurrences.

    Thread-safe. All mutations are lock-protected.
    """

    def __init__(self, max_records: int = 2000, *, persistent_evidence: bool = False) -> None:
        self._lock = threading.Lock()
        self._definitions: dict[str, FaultDefinition] = {}
        self._records: deque[FaultRecord] = deque(maxlen=max_records)
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int),
        )
        # Cross-boot occurrence evidence (core/resilience/fault_evidence.py).
        # Off by default so throwaway registries in tests never touch the
        # live evidence file; the process singleton opts in.
        self._persistent_evidence = persistent_evidence
        # Listeners fire OUTSIDE the lock, guarded, and must be O(1)
        # (the recovery bridge only enqueues). Bounded to prevent
        # accidental listener sprawl.
        self._listeners: list[Any] = []
        self._register_builtin_faults()

    def add_listener(self, listener: Any) -> bool:
        """Register a fault-record listener (max 8; O(1) callbacks only)."""
        with self._lock:
            if len(self._listeners) >= 8 or listener in self._listeners:
                return False
            self._listeners.append(listener)
            return True

    # ── Definition management ────────────────────────────────────────

    def define(self, defn: FaultDefinition) -> None:
        """Register a fault definition in the catalog."""
        with self._lock:
            self._definitions[defn.fault_id] = defn

    def get_definition(self, fault_id: str) -> FaultDefinition | None:
        with self._lock:
            return self._definitions.get(fault_id)

    def all_definitions(self) -> list[FaultDefinition]:
        with self._lock:
            return list(self._definitions.values())

    # ── Runtime fault recording ──────────────────────────────────────

    def record_fault(
        self,
        fault_id: str,
        subsystem: str,
        *,
        details: str = "",
        error: Exception | None = None,
        recovered: bool = False,
        recovery_time_s: float | None = None,
        severity: FaultSeverity | None = None,
    ) -> FaultRecord:
        """Record a runtime fault occurrence.

        Severity resolution order: explicit ``severity`` argument, then the
        matching fault definition, then MARGINAL. The explicit override
        exists so callers with live severity knowledge (e.g. the degradation
        tracker) keep fidelity for fault IDs without a static definition.
        """
        if severity is None:
            defn = self.get_definition(fault_id)
            severity = defn.severity if defn else FaultSeverity.MARGINAL

        record = FaultRecord(
            fault_id=fault_id,
            subsystem=subsystem,
            severity=severity,
            details=details,
            error_type=type(error).__name__ if error else "",
            error_message=str(error)[:200] if error else "",
            recovered=recovered,
            recovery_time_s=recovery_time_s,
        )

        with self._lock:
            self._records.append(record)
            self._counts[fault_id][severity.name] += 1

        if self._persistent_evidence:
            try:
                from core.resilience.fault_evidence import get_fault_evidence_store
                get_fault_evidence_store().record(fault_id)
            except (ImportError, AttributeError, RuntimeError, OSError) as exc:
                logger.debug("Fault evidence unavailable: %s", exc)

        for listener in list(self._listeners):
            try:
                listener(record)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Fault listener failed: %s", exc)

        log_level = {
            FaultSeverity.CATASTROPHIC: logging.CRITICAL,
            FaultSeverity.CRITICAL: logging.ERROR,
            FaultSeverity.MARGINAL: logging.WARNING,
            FaultSeverity.NEGLIGIBLE: logging.DEBUG,
        }.get(severity, logging.WARNING)

        logger.log(
            log_level,
            "FAULT %s [%s] in %s: %s%s",
            fault_id,
            severity.name,
            subsystem,
            details[:120],
            f" | error={record.error_type}: {record.error_message[:80]}" if error else "",
        )

        return record

    def mark_recovered(self, record: FaultRecord, recovery_time_s: float) -> None:
        """Mark a fault record as recovered."""
        record.recovered = True
        record.recovery_time_s = recovery_time_s

    # ── Queries ──────────────────────────────────────────────────────

    def faults_by_severity(self, severity: FaultSeverity) -> list[FaultRecord]:
        with self._lock:
            return [r for r in self._records if r.severity == severity]

    def faults_by_subsystem(self, subsystem: str) -> list[FaultRecord]:
        with self._lock:
            return [r for r in self._records if r.subsystem == subsystem]

    def recent_faults(
        self, window_s: float = 3600, severity_at_most: FaultSeverity | None = None,
    ) -> list[FaultRecord]:
        cutoff = time.time() - max(0.0, window_s)
        with self._lock:
            results = [r for r in self._records if r.timestamp >= cutoff]
            if severity_at_most is not None:
                results = [r for r in results if r.severity <= severity_at_most]
            return results

    def fault_count(self, fault_id: str) -> int:
        with self._lock:
            return sum(self._counts.get(fault_id, {}).values())

    def rpn_report(self) -> list[dict[str, Any]]:
        """Generate a Risk Priority Number report for all defined faults.

        Sorted by RPN descending (highest risk first).
        """
        with self._lock:
            return self._rpn_report_locked()

    def _rpn_report_locked(self) -> list[dict[str, Any]]:
        # Callers must hold self._lock; the lock is non-reentrant, so any
        # method already inside it must use this core, never rpn_report().
        entries = []
        for defn in self._definitions.values():
            occurrence_count = sum(self._counts.get(defn.fault_id, {}).values())
            entries.append({
                "fault_id": defn.fault_id,
                "name": defn.name,
                "severity": defn.severity.name,
                "probability": defn.probability.name,
                "detection": defn.detection.name,
                "rpn": defn.rpn,
                "domain": defn.domain.value,
                "recovery": defn.recovery.value,
                "mttr_target_s": defn.mttr_seconds,
                "runtime_occurrences": occurrence_count,
            })
        entries.sort(key=lambda e: e["rpn"], reverse=True)
        return entries

    def status(self) -> dict[str, Any]:
        """Summary status for health endpoints."""
        with self._lock:
            total = len(self._records)
            by_sev = defaultdict(int)
            for r in self._records:
                by_sev[r.severity.name] += 1
            unrecovered = sum(1 for r in self._records if not r.recovered)
            return {
                "total_faults": total,
                "by_severity": dict(by_sev),
                "unrecovered": unrecovered,
                "definitions_count": len(self._definitions),
                "top_rpn": self._rpn_report_locked()[:5],
            }

    # ── Built-in fault definitions (from KNOWN_FAILURE_MODES.md) ─────

    def _register_builtin_faults(self) -> None:
        """Pre-register all known failure modes from KNOWN_FAILURE_MODES.md."""
        builtins = [
            FaultDefinition(
                fault_id="F01", name="Model fails to load",
                description="Insufficient RAM, corrupted weights, missing model files",
                domain=FaultDomain.INFERENCE, severity=FaultSeverity.CATASTROPHIC,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.MANUAL_INTERVENTION, mttr_seconds=300,
                blast_radius="No inference capability",
                runbook="docs/runbooks/model-fails-to-load.md",
            ),
            FaultDefinition(
                fault_id="F02", name="Worker process crash during inference",
                description="GPU memory pressure, MLX runtime error, corrupted prompt",
                domain=FaultDomain.INFERENCE, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.AUTOMATIC_RESTART, mttr_seconds=30,
                blast_radius="Current request fails; auto-recovery spawns new worker",
                runbook="docs/runbooks/worker-crash.md",
            ),
            FaultDefinition(
                fault_id="F03", name="Memory database corruption",
                description="Dirty shutdown, disk full, concurrent write race",
                domain=FaultDomain.STORAGE, severity=FaultSeverity.CATASTROPHIC,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.STEM_CELL_REVERT, mttr_seconds=120,
                blast_radius="Memory retrieval fails; boot may degrade",
                runbook="docs/runbooks/memory-corruption.md",
            ),
            FaultDefinition(
                fault_id="F04", name="Shutdown hangs",
                description="Blocked async task, hung worker, deadlocked service",
                domain=FaultDomain.SHUTDOWN, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.MANUAL_INTERVENTION, mttr_seconds=15,
                blast_radius="Process requires SIGKILL",
                runbook="docs/runbooks/shutdown-hang.md",
            ),
            FaultDefinition(
                fault_id="F05", name="External-service egress privacy leak",
                description="Sensitive content crosses an external service boundary",
                domain=FaultDomain.SECURITY, severity=FaultSeverity.CATASTROPHIC,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.MODERATE,
                recovery=RecoveryStrategy.QUARANTINE, mttr_seconds=60,
                blast_radius="Sensitive data sent to an external service",
                runbook="docs/runbooks/external-egress.md",
            ),
            FaultDefinition(
                fault_id="F06", name="Prompt injection succeeds",
                description="Novel injection technique bypasses sanitizer",
                domain=FaultDomain.SECURITY, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.STEM_CELL_REVERT, mttr_seconds=60,
                blast_radius="Aura performs unintended action",
                runbook="docs/runbooks/prompt-injection.md",
            ),
            FaultDefinition(
                fault_id="F07", name="Resource exhaustion (RAM/GPU)",
                description="Large context, multiple concurrent requests, memory leak",
                domain=FaultDomain.RESOURCE, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.OCCASIONAL, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.GRACEFUL_DEGRADATION, mttr_seconds=30,
                blast_radius="Degraded performance; potential OOM kill",
                runbook="docs/runbooks/resource-exhaustion.md",
            ),
            FaultDefinition(
                fault_id="F08", name="Background task orphaning",
                description="Task creator dies without cleaning up background work",
                domain=FaultDomain.RESOURCE, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.AUTOMATIC_RESTART, mttr_seconds=60,
                blast_radius="Wasted resources; potential stale state",
                runbook="docs/runbooks/orphaned-tasks.md",
            ),
            FaultDefinition(
                fault_id="F09", name="Stale memory retrieval",
                description="Vector DB index drift; outdated embeddings",
                domain=FaultDomain.MEMORY, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.OCCASIONAL, detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.AUTOMATIC_FALLBACK, mttr_seconds=300,
                blast_radius="Irrelevant context in responses",
            ),
            FaultDefinition(
                fault_id="F10", name="Identity drift",
                description="Sustained adversarial prompting; corrupted CanonicalSelf",
                domain=FaultDomain.IDENTITY, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.MODERATE,
                recovery=RecoveryStrategy.STEM_CELL_REVERT, mttr_seconds=30,
                blast_radius="Personality/identity becomes inconsistent",
            ),
            FaultDefinition(
                fault_id="F11", name="Tool execution timeout",
                description="Slow external service; large file operation; network timeout",
                domain=FaultDomain.TOOL_EXECUTION, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.OCCASIONAL, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.RETRY_WITH_BACKOFF, mttr_seconds=10,
                blast_radius="Individual tool call fails",
                runbook="docs/runbooks/tool-timeout-storm.md",
            ),
            FaultDefinition(
                fault_id="F12", name="Lock contention / deadlock",
                description="Multiple subsystems contending for same resource",
                domain=FaultDomain.RESOURCE, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.AUTOMATIC_RESTART, mttr_seconds=15,
                blast_radius="Request stalls until watchdog releases",
            ),
            FaultDefinition(
                fault_id="F13", name="Log rotation failure",
                description="Disk full; permission error",
                domain=FaultDomain.OBSERVABILITY, severity=FaultSeverity.NEGLIGIBLE,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.IGNORE, mttr_seconds=600,
                blast_radius="Logs stop writing; no data loss",
            ),
            FaultDefinition(
                fault_id="F14", name="Telemetry emission failure",
                description="Metrics endpoint unavailable",
                domain=FaultDomain.OBSERVABILITY, severity=FaultSeverity.NEGLIGIBLE,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.IGNORE, mttr_seconds=300,
                blast_radius="Missing observability data",
            ),
            # ── New faults identified during reliability hardening ─────
            FaultDefinition(
                fault_id="F15", name="Circuit breaker stuck open",
                description="Circuit breaker fails to transition to half-open after cooldown",
                domain=FaultDomain.INFERENCE, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.AUTOMATIC_RESTART, mttr_seconds=60,
                blast_radius="Subsystem permanently unavailable until restart",
            ),
            FaultDefinition(
                fault_id="F16", name="Event bus backpressure overflow",
                description="Event bus queue full; events dropped",
                domain=FaultDomain.AGENCY, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.GRACEFUL_DEGRADATION, mttr_seconds=5,
                blast_radius="Internal events lost; subsystem desync",
            ),
            FaultDefinition(
                fault_id="F17", name="State machine illegal transition",
                description="Component attempts a forbidden state transition",
                domain=FaultDomain.GOVERNANCE, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.QUARANTINE, mttr_seconds=0,
                blast_radius="Component quarantined; may affect dependent subsystems",
            ),
            FaultDefinition(
                fault_id="F18", name="SLO budget exhausted",
                description="Error budget for an SLO consumed; reliability degraded",
                domain=FaultDomain.OBSERVABILITY, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.OCCASIONAL, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.GRACEFUL_DEGRADATION, mttr_seconds=0,
                blast_radius="Feature freeze recommended until budget replenished",
            ),
            FaultDefinition(
                fault_id="F19", name="TMR channel divergence",
                description="Triple modular redundancy channels produce conflicting results",
                domain=FaultDomain.GOVERNANCE, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.IMPROBABLE, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.QUARANTINE, mttr_seconds=0,
                blast_radius="Divergent channel quarantined; majority result used",
            ),
            FaultDefinition(
                fault_id="ACTION-CLAIM-MISMATCH", name="Confabulated action",
                description="A tool reported success while the predicted "
                            "world-state was absent — the agent believed an "
                            "unexecuted intention (sensorimotor grounding "
                            "caught the claim/reality divergence)",
                domain=FaultDomain.TOOL_EXECUTION, severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE,
                detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.RETRY_WITH_BACKOFF, mttr_seconds=30,
                blast_radius="False belief about completed work; downstream "
                             "plans built on a phantom effect",
            ),
            FaultDefinition(
                fault_id="AFFECT-TRAP", name="Affective sampling trap",
                description="Sampling temperature pinned at the deterministic "
                            "floor while distress telemetry failed to improve — "
                            "the safety clamp starving self-repair variance "
                            "(digital-depression loop); a bounded exploration "
                            "escape was opened",
                domain=FaultDomain.CONSCIOUSNESS, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.REMOTE,
                detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.AUTOMATIC_FALLBACK, mttr_seconds=60,
                blast_radius="Degraded creative/self-repair capability until escape",
            ),
            FaultDefinition(
                fault_id="WILL-REFUSE", name="Will refusal issued",
                description="The Will refused an action — governance functioning "
                            "as designed; recorded for forensic traceability only",
                domain=FaultDomain.GOVERNANCE, severity=FaultSeverity.NEGLIGIBLE,
                probability=FaultProbability.OCCASIONAL,
                detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.IGNORE, mttr_seconds=0,
                blast_radius="None — the refused action never executes",
            ),
            FaultDefinition(
                fault_id="F20", name="Contract violation",
                description="Design-by-contract precondition/postcondition/invariant failed",
                domain=FaultDomain.GOVERNANCE, severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.REMOTE, detection=DetectionDifficulty.CERTAIN,
                recovery=RecoveryStrategy.GRACEFUL_DEGRADATION, mttr_seconds=0,
                blast_radius="Logged and monitored; function continues (log+continue mode)",
            ),
            FaultDefinition(
                fault_id="PASSF-ACTION-SHALLOW-SUCCESS",
                name="Shallow action reports success",
                description="An action fires and returns a technically true success "
                            "without satisfying the user's implied acceptance "
                            "criteria or preserving effect evidence.",
                domain=FaultDomain.TOOL_EXECUTION,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.PROBABLE,
                detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.RETRY_WITH_BACKOFF,
                mttr_seconds=30,
                blast_radius="User-facing task appears complete while the useful "
                             "work is missing or too shallow to rely on.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-FALSE-HEALTH",
                name="False health/readiness signal",
                description="A readiness, liveness, or proof-health path reports "
                            "green while the live user path remains blocked.",
                domain=FaultDomain.OBSERVABILITY,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.PROBABLE,
                detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.QUARANTINE,
                mttr_seconds=60,
                blast_radius="Operators trust a green signal that does not match "
                             "actual demo or daily-use readiness.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-RESOURCE-SPAWN-LOOP",
                name="Resource spawn loop",
                description="Model, memory, or worker orchestration repeatedly "
                            "spawns work under pressure instead of admitting, "
                            "backing off, or degrading explicitly.",
                domain=FaultDomain.RESOURCE,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.OCCASIONAL,
                detection=DetectionDifficulty.MODERATE,
                recovery=RecoveryStrategy.CIRCUIT_BREAKER,
                mttr_seconds=45,
                blast_radius="GPU/RAM pressure cascades into stalled requests, "
                             "wedged workers, or unreliable boot.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-DESKTOP-PERMISSION-DRIFT",
                name="Desktop permission drift",
                description="Desktop, Chrome, accessibility, camera, microphone, "
                            "or browser-control permissions drift after boot and "
                            "surface only as shallow action failure.",
                domain=FaultDomain.TOOL_EXECUTION,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.OCCASIONAL,
                detection=DetectionDifficulty.MODERATE,
                recovery=RecoveryStrategy.GRACEFUL_DEGRADATION,
                mttr_seconds=120,
                blast_radius="Visible computer-use workflows cannot be demoed or "
                             "completed despite nominal capability registration.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-REPAIR-STORM",
                name="Self-repair storm",
                description="Autonomy or self-repair loops continue patching, "
                            "retrying, or re-planning without cooldown, budget, "
                            "causal progress, or operator-visible stop condition.",
                domain=FaultDomain.AGENCY,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE,
                detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.CIRCUIT_BREAKER,
                mttr_seconds=60,
                blast_radius="Repair behavior consumes resources and can make "
                             "the original fault harder to diagnose.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-STALE-OBLIGATION",
                name="Stale obligation blocks current work",
                description="Memory, prompt, or task-state residue from an older "
                            "objective keeps steering unrelated current work.",
                domain=FaultDomain.MEMORY,
                severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.PROBABLE,
                detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.AUTOMATIC_FALLBACK,
                mttr_seconds=30,
                blast_radius="Aura acts federated and distracted instead of "
                             "unified around the user's current objective.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-NEURAL-STREAM-FLOOD",
                name="Neural stream flood hides state",
                description="High-volume internal streams, events, or logs drown "
                            "out the actionable user-visible state needed for "
                            "debugging and daily operation.",
                domain=FaultDomain.OBSERVABILITY,
                severity=FaultSeverity.MARGINAL,
                probability=FaultProbability.PROBABLE,
                detection=DetectionDifficulty.MODERATE,
                recovery=RecoveryStrategy.GRACEFUL_DEGRADATION,
                mttr_seconds=30,
                blast_radius="Operators see noise rather than crisp state, "
                             "blockers, receipts, and next actions.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-VISIBLE-WEB-PROOF-ACCESS",
                name="Visible web proof inaccessible",
                description="Visible Chrome or web-interlocutor proof depends on "
                            "a browser profile, extension, or session that is not "
                            "available to the active runtime.",
                domain=FaultDomain.TOOL_EXECUTION,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.OCCASIONAL,
                detection=DetectionDifficulty.HIGH,
                recovery=RecoveryStrategy.MANUAL_INTERVENTION,
                mttr_seconds=300,
                blast_radius="External web proof and demo-critical browser "
                             "workflows remain honestly blocked.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-PROOF-ARTIFACT-CONTAMINATION",
                name="Proof artifact contamination",
                description="A proof, certification, or report consumes stale, "
                            "fixture-backed, hardcoded, or cross-run artifact "
                            "data as if it were fresh live evidence.",
                domain=FaultDomain.OBSERVABILITY,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.REMOTE,
                detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.QUARANTINE,
                mttr_seconds=120,
                blast_radius="Certification can overstate maturity and hide "
                             "runtime regressions until live demo.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
            FaultDefinition(
                fault_id="PASSF-SEMANTIC-REVIEW-GAP",
                name="Semantic review gap",
                description="Mechanical closeout, line counting, or path hashing "
                            "is mistaken for semantic review of code behavior, "
                            "runtime contracts, and user-facing obligations.",
                domain=FaultDomain.GOVERNANCE,
                severity=FaultSeverity.CRITICAL,
                probability=FaultProbability.PROBABLE,
                detection=DetectionDifficulty.LOW,
                recovery=RecoveryStrategy.MANUAL_INTERVENTION,
                mttr_seconds=600,
                blast_radius="Remaining code debt is hidden behind a green "
                             "mechanical audit surface.",
                runbook="docs/runbooks/pass-f-maturity-risks.md",
            ),
        ]
        for defn in builtins:
            self._definitions[defn.fault_id] = defn


# ── Module singleton ─────────────────────────────────────────────────

_registry: FaultRegistry | None = None
_registry_lock = threading.Lock()


def get_fault_registry() -> FaultRegistry:
    """Return the global fault registry singleton.

    The singleton records cross-boot occurrence evidence unless running
    under pytest / test mode (throwaway test registries must never write
    to the live evidence file).
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                testing = bool(
                    os.environ.get("AURA_TEST_MODE")
                    or os.environ.get("PYTEST_CURRENT_TEST"),
                )
                _registry = FaultRegistry(persistent_evidence=not testing)
        # What every known fault actually looks like, learned from its
        # instances. Without this the recogniser in unknown_failure.py has
        # nothing to compare a new failure against, and it had nothing: the
        # module was correct, complete, and reached by no fault in the tree.
        try:
            from core.resilience.unknown_failure import attach_to_the_fault_registry

            attach_to_the_fault_registry(_registry)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Failure ontology not attached: %s", exc)
    return _registry
