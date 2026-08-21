"""System integrity audit — make silent subsystem failures speak.

The critique's third gap: "the silence of subsystem failures … the degradation
receipt system is comprehensive but requires active reading." Failures are recorded
(``record_degradation``), the CRSM loop can quietly stop closing, and CAA steering can
quietly run below capacity — but nothing pulls these together and says so out loud.

This audit consolidates the three signals — degradation receipts, CRSM→LoRA loop
closure, and CAA steering readiness — into one report and logs a single loud summary
when anything is wrong. Health callers consume a bounded stale-while-revalidate read
model; filesystem hashing, dataset parsing, and CAA readiness checks never run on the
HTTP event loop. Under ``AURA_STRICT_RUNTIME=1`` the collector emits the report even
when clean, so production runs surface the activation state without requiring someone
to go read receipts.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from typing import Any

from core.health.read_model import HealthReadModelConfig, HealthSnapshotReadModel

logger = logging.getLogger("Aura.IntegrityAudit")

_last_run = 0.0
_last_report: dict[str, Any] | None = None
_lock = threading.Lock()
_PROCESS_STARTED_AT: float | None = None
_SESSION_LOCK = threading.Lock()
_SESSION_STARTED_AT: float | None = None
_SESSION_ACTIVE = False
_SESSION_GENERATION = 0
_INCIDENT_LOCK = threading.Lock()
_ACTIVE_CONCERN_COUNTS: dict[str, int] = {}
_LAST_WARNED_CONCERN_COUNTS: dict[str, int] = {}

# Degradations above this for a single subsystem are flagged as a concern.
_DEGRADATION_CONCERN = 10
# Concern verdicts look at a trailing window so the runtime can recover after
# a degradation storm instead of staying "unhealthy" for its whole lifetime.
_DEGRADATION_CONCERN_WINDOW_S = 1800.0
# Individual degradation records already report each failure. The integrity
# summary is a higher-level incident signal and only re-announces after another
# full concern threshold accumulates for an affected subsystem.
_DEGRADATION_CONCERN_RELOG_DELTA = _DEGRADATION_CONCERN


def _reset_integrity_incident() -> None:
    with _INCIDENT_LOCK:
        _ACTIVE_CONCERN_COUNTS.clear()
        _LAST_WARNED_CONCERN_COUNTS.clear()


def _observe_integrity_incident(
    current_counts: dict[str, int],
) -> dict[str, Any]:
    """Advance the runtime-concern incident and return its transition.

    The health report remains level-triggered, while logs and counters are
    edge-triggered. This prevents a read-model refresh from manufacturing a
    fresh warning for the same already-recorded degradation history.
    """

    normalized = {
        str(subsystem): max(0, int(count))
        for subsystem, count in current_counts.items()
        if str(subsystem) and int(count) >= _DEGRADATION_CONCERN
    }
    with _INCIDENT_LOCK:
        previous = dict(_ACTIVE_CONCERN_COUNTS)
        newly_active = sorted(set(normalized) - set(previous))
        resolved = sorted(set(previous) - set(normalized))
        materially_worsened = sorted(
            subsystem
            for subsystem, count in normalized.items()
            if subsystem in previous
            and count
            >= _LAST_WARNED_CONCERN_COUNTS.get(subsystem, previous[subsystem])
            + _DEGRADATION_CONCERN_RELOG_DELTA
        )
        warning_required = bool(newly_active or materially_worsened)
        fully_recovered = bool(previous and not normalized)

        _ACTIVE_CONCERN_COUNTS.clear()
        _ACTIVE_CONCERN_COUNTS.update(normalized)
        for subsystem in resolved:
            _LAST_WARNED_CONCERN_COUNTS.pop(subsystem, None)
        if warning_required:
            _LAST_WARNED_CONCERN_COUNTS.update(normalized)
        if fully_recovered:
            _LAST_WARNED_CONCERN_COUNTS.clear()

        return {
            "active": bool(normalized),
            "current_counts": dict(normalized),
            "new_subsystems": newly_active,
            "materially_worsened_subsystems": materially_worsened,
            "resolved_subsystems": resolved,
            "warning_required": warning_required,
            "fully_recovered": fully_recovered,
            "relog_delta": _DEGRADATION_CONCERN_RELOG_DELTA,
        }


def _integrity_incident_snapshot(current_counts: dict[str, int]) -> dict[str, Any]:
    """Describe current pressure without consuming a logging transition."""

    return {
        "active": bool(current_counts),
        "current_counts": dict(current_counts),
        "new_subsystems": [],
        "materially_worsened_subsystems": [],
        "resolved_subsystems": [],
        "warning_required": False,
        "fully_recovered": False,
        "relog_delta": _DEGRADATION_CONCERN_RELOG_DELTA,
    }


def _process_started_at() -> float:
    """Return the wall-clock start of this process incarnation.

    A trailing wall-clock window alone cannot distinguish records from an old
    runtime when a tracker is retained by an embedded host. Process identity is
    the lower bound for evidence attributed to this runtime.
    """

    global _PROCESS_STARTED_AT
    if _PROCESS_STARTED_AT is not None:
        return _PROCESS_STARTED_AT
    try:
        from core.runtime.resource_psutil import Process

        observed = float(Process(os.getpid()).create_time())
        if observed <= 0.0:
            raise ValueError("process start time must be positive")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # Keep the established trailing-window behavior when epoch evidence is
        # unavailable. Guessing "now" would erase valid records from this boot.
        observed = 0.0
    _PROCESS_STARTED_AT = observed
    return observed


def _runtime_epoch_started_at() -> float:
    with _SESSION_LOCK:
        session_started_at = _SESSION_STARTED_AT
    if session_started_at is not None:
        return session_started_at
    return _process_started_at()


def _begin_integrity_session(*, now: float | None = None) -> float:
    """Advance the evidence epoch only for a new runtime lifespan."""

    global _SESSION_ACTIVE, _SESSION_GENERATION, _SESSION_STARTED_AT
    with _SESSION_LOCK:
        if _SESSION_ACTIVE:
            return float(_SESSION_STARTED_AT or 0.0)
        if _SESSION_GENERATION == 0:
            _SESSION_STARTED_AT = _process_started_at()
        else:
            _SESSION_STARTED_AT = float(time.time() if now is None else now)
        _SESSION_GENERATION += 1
        _SESSION_ACTIVE = True
        session_started_at = float(_SESSION_STARTED_AT or 0.0)
    _reset_integrity_incident()
    return session_started_at


def _end_integrity_session() -> None:
    global _SESSION_ACTIVE
    with _SESSION_LOCK:
        _SESSION_ACTIVE = False


def _epoch_scoped_degradation_counts(
    records: Iterable[Any],
    *,
    window_s: float,
    now: float,
    process_started_at: float,
) -> tuple[dict[str, dict[str, int]], dict[str, float | str]]:
    """Count degradation records attributable to the current process epoch."""

    observed_at = float(now)
    epoch_started_at = max(0.0, float(process_started_at))
    window_started_at = max(
        epoch_started_at,
        observed_at - max(0.0, float(window_s)),
    )
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        try:
            timestamp = float(record.timestamp)
        except (AttributeError, TypeError, ValueError):
            continue
        if timestamp < window_started_at or timestamp > observed_at:
            continue
        subsystem = str(getattr(record, "subsystem", "") or "").strip()
        severity = str(getattr(record, "severity", "") or "").strip()
        if not subsystem or not severity:
            continue
        counts.setdefault(subsystem, {}).setdefault(severity, 0)
        counts[subsystem][severity] += 1
    scope: dict[str, float | str] = {
        "kind": (
            "process_epoch_trailing_window"
            if epoch_started_at > 0.0
            else "trailing_window_epoch_unavailable"
        ),
        "process_started_at": epoch_started_at,
        "window_started_at": window_started_at,
        "observed_at": observed_at,
    }
    return counts, scope


def strict_mode() -> bool:
    return os.environ.get("AURA_STRICT_RUNTIME") == "1"


def _env_positive_float(name: str, default: float) -> float:
    try:
        return max(0.05, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return float(default)


def run_integrity_audit(*, log: bool = True) -> dict[str, Any]:
    """Aggregate degradations + CRSM loop + CAA readiness; log loudly if degraded."""
    # RUNTIME-HEALTH concerns (can the process actually serve?) vs ADVISORY concerns
    # (operational facts like "training hasn't run yet"). Only the former may gate
    # health — an open CRSM loop or runtime-derived CAA vectors are real and worth
    # surfacing, but they do NOT mean Aura can't converse, so they must never make the
    # runtime report "degraded"/not-ready.
    concerns: list[str] = []
    advisory: list[str] = []
    concern_counts: dict[str, int] = {}

    degradations: dict[str, Any] = {}
    try:
        from core.runtime.errors import get_degradation_tracker

        tracker = get_degradation_tracker()
        degradations = tracker.status()
        # Health verdicts use the intersection of a trailing window and this
        # process incarnation. A host that retains a tracker across a runtime
        # restart must not attribute the old runtime's records to the new one.
        observed_at = time.time()
        recent_counts, recent_scope = _epoch_scoped_degradation_counts(
            tracker.recent(limit=1_000_000),
            window_s=_DEGRADATION_CONCERN_WINDOW_S,
            now=observed_at,
            process_started_at=_runtime_epoch_started_at(),
        )
        degradations["recent_window_s"] = _DEGRADATION_CONCERN_WINDOW_S
        degradations["recent_counts_by_subsystem"] = recent_counts
        degradations["recent_scope"] = recent_scope
        for sub, sevs in recent_counts.items():
            total = sum(sevs.values())
            if total >= _DEGRADATION_CONCERN:
                concern_counts[sub] = total
                concerns.append(
                    f"{sub}: {total} degradations in the last "
                    f"{int(_DEGRADATION_CONCERN_WINDOW_S // 60)}m"
                )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        degradations = {}

    crsm_loop: dict[str, Any] = {}
    try:
        from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

        crsm_loop = get_crsm_loop_monitor().loop_state()
        if crsm_loop.get("state") == "open":
            advisory.append(f"CRSM→LoRA loop OPEN ({crsm_loop.get('unconsumed')} captures untrained)")
    except (ImportError, AttributeError, RuntimeError, TypeError):
        crsm_loop = {}

    caa_readiness: dict[str, Any] = {}
    try:
        from core.consciousness.caa.readiness_report import verify_readiness

        caa_readiness = verify_readiness()
        if caa_readiness.get("below_design_capacity"):
            advisory.append(
                f"CAA steering at {caa_readiness.get('steering_capacity_pct')}% "
                f"({caa_readiness.get('level')})"
            )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        caa_readiness = {}

    # Failure pressure with its top contributors: when background policy
    # reports failure_lockdown_X, this names the feeder without log archaeology.
    failure_state: dict[str, Any] = {}
    try:
        from core.health.degraded_events import get_unified_failure_state

        unified = get_unified_failure_state()
        failure_state = {
            "pressure": unified.get("pressure", 0.0),
            "count": unified.get("count", 0),
            "critical": unified.get("critical", 0),
            "top_subsystems": unified.get("top_subsystems", []),
        }
        if float(failure_state.get("pressure") or 0.0) >= 0.5:
            advisory.append(
                f"failure pressure {failure_state['pressure']:.2f} "
                f"(top: {', '.join(str(t) for t in failure_state['top_subsystems'][:3]) or 'n/a'})"
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        failure_state = {}

    integrity_incident = (
        _observe_integrity_incident(concern_counts)
        if log
        else _integrity_incident_snapshot(concern_counts)
    )
    report = {
        # 'healthy' reflects RUNTIME health only — advisory operational facts never make
        # the runtime unhealthy. 'concerns' (the health-blocking list) holds runtime
        # concerns; 'advisory' holds the surfaced-but-non-blocking operational notes.
        "healthy": not concerns,
        "concerns": concerns,
        "advisory": advisory,
        "strict_mode": strict_mode(),
        "degradations": degradations,
        "crsm_loop": crsm_loop,
        "caa_readiness": caa_readiness,
        "failure_state": failure_state,
        "integrity_incident": integrity_incident,
        "at": time.time(),
    }

    global _last_report
    _last_report = report

    if log:
        if concerns and integrity_incident["warning_required"]:
            logger.warning(
                "🩺 [Integrity] runtime concern incident opened or escalated "
                "(%d active): %s",
                len(concerns),
                " | ".join(concerns),
            )
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("integrity_concern_total")
            except (ImportError, AttributeError, RuntimeError, TypeError):
                pass
        if integrity_incident["resolved_subsystems"]:
            remaining = sorted(concern_counts)
            logger.info(
                "🩺 [Integrity] runtime concern recovered for %s; remaining=%s",
                ", ".join(integrity_incident["resolved_subsystems"]),
                ", ".join(remaining) if remaining else "none",
            )
        if advisory:
            logger.info("🩺 [Integrity] advisory (non-blocking): %s", " | ".join(advisory))
        if not concerns and not advisory and strict_mode():
            logger.info("🩺 [Integrity] all subsystems nominal (strict mode).")
    return report


def _integrity_snapshot_fallback() -> dict[str, Any]:
    return {
        "healthy": False,
        "concerns": ["integrity_snapshot_initializing"],
        "advisory": [],
        "strict_mode": strict_mode(),
        "degradations": {},
        "crsm_loop": {},
        "caa_readiness": {},
        "failure_state": {},
        "integrity_incident": {
            "active": False,
            "current_counts": {},
            "new_subsystems": [],
            "materially_worsened_subsystems": [],
            "resolved_subsystems": [],
            "warning_required": False,
            "fully_recovered": False,
            "relog_delta": _DEGRADATION_CONCERN_RELOG_DELTA,
        },
        "at": None,
    }


def _new_integrity_read_model() -> HealthSnapshotReadModel:
    refresh_s = _env_positive_float("AURA_INTEGRITY_REFRESH_S", 15.0)
    return HealthSnapshotReadModel(
        run_integrity_audit,
        _integrity_snapshot_fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=refresh_s,
            max_stale_s=max(
                refresh_s,
                _env_positive_float("AURA_INTEGRITY_MAX_STALE_S", 90.0),
            ),
            collection_timeout_s=_env_positive_float(
                "AURA_INTEGRITY_COLLECTION_TIMEOUT_S", 8.0
            ),
            retry_base_s=_env_positive_float("AURA_INTEGRITY_RETRY_BASE_S", 2.0),
            retry_max_s=_env_positive_float("AURA_INTEGRITY_RETRY_MAX_S", 30.0),
            schema_version="aura.integrity.snapshot.v1",
            metadata_key="integrity_read_model",
            worker_name_prefix="AuraIntegritySnapshot",
            incident_prefix="integrity-refresh",
            log_label="Integrity snapshot",
        ),
    )


_INTEGRITY_READ_MODEL = _new_integrity_read_model()


def start_integrity_read_model() -> bool:
    """Prewarm integrity evidence without joining the collector."""

    _begin_integrity_session()
    return _INTEGRITY_READ_MODEL.start()


def stop_integrity_read_model() -> None:
    _INTEGRITY_READ_MODEL.close()
    _end_integrity_session()


def reset_integrity_read_model_for_test() -> None:
    global _SESSION_ACTIVE, _SESSION_GENERATION, _SESSION_STARTED_AT
    _INTEGRITY_READ_MODEL.reset_for_test()
    with _SESSION_LOCK:
        _SESSION_STARTED_AT = None
        _SESSION_ACTIVE = False
        _SESSION_GENERATION = 0
    _reset_integrity_incident()


def read_integrity_audit() -> dict[str, Any]:
    """Return immediately with current or explicitly stale integrity evidence."""

    return _INTEGRITY_READ_MODEL.read()


def maybe_run(*, interval_s: float = 300.0) -> dict[str, Any] | None:
    """Legacy synchronous throttle for CLI/background callers.

    Event-loop and request paths must use :func:`read_integrity_audit`.
    """
    global _last_run
    now = time.time()
    with _lock:
        if now - _last_run < interval_s:
            return _last_report
        _last_run = now
    return run_integrity_audit()


def last_report() -> dict[str, Any] | None:
    return _last_report
