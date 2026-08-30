"""core/runtime/errors.py — Structured degradation receipts.

The audit found 5,222 broad catch-all blocks, most swallowing errors
with `pass` or `logger.debug`.  This module provides the canonical
replacement pattern:

    from core.runtime.errors import record_degradation

    try:
        do_work()
    except SpecificError as exc:
        record_degradation(
            subsystem="memory_facade",
            error=exc,
            severity="degraded",
            action="Fell back to in-memory cache",
        )

Every call to ``record_degradation`` produces:

  1. A structured log entry at the appropriate level.
  2. An in-memory counter per (subsystem, severity) pair.
  3. An optional receipt in the ReceiptStore for forensic audit.

No silent ``pass``.  No swallowed ``Exception``.  Every degradation is
visible, countable, and queryable.
"""
from __future__ import annotations

# ruff: noqa: N818
import logging
import os
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

logger = logging.getLogger("Aura.Errors")

Severity = Literal["debug", "warning", "degraded", "critical"]


class FallbackClassification(StrEnum):
    SAFE_FALLBACK = "SAFE_FALLBACK"
    SILENT_LOSS_OF_CAPABILITY = "SILENT_LOSS_OF_CAPABILITY"
    GOVERNANCE_BYPASS = "GOVERNANCE_BYPASS"
    AUDIT_GAP = "AUDIT_GAP"
    STATE_CORRUPTION_RISK = "STATE_CORRUPTION_RISK"

# ---------------------------------------------------------------------------
# Typed Degradation Exceptions
# ---------------------------------------------------------------------------
class BoundaryFailure(Exception):
    """A runtime boundary rejected or failed a handoff."""


class DependencyUnavailable(Exception):
    """A required dependency is missing or offline."""


class ModelUnavailable(Exception):
    """A model backend could not serve the request."""


class CapabilityDenied(Exception):
    """A requested capability was denied by policy."""


class ReceiptInvalid(Exception):
    """A governance or audit receipt failed validation."""


class StateCoherenceFailure(Exception):
    """Runtime state failed a coherence invariant."""


class MemoryWriteDenied(Exception):
    """A memory write was refused by governance or validation."""


class NetworkEffectDenied(Exception):
    """A network side effect was denied."""


class SandboxViolation(Exception):
    """A sandbox boundary was violated."""


class PersistenceCorruption(Exception):
    """Durable state failed integrity validation."""


class TimeoutBudgetExceeded(Exception):
    """Execution exceeded its allowed timeout budget."""


class ResourceExhaustion(Exception):
    """A resource budget was exhausted."""


class InvariantViolation(Exception):
    """A required runtime invariant was violated."""

# ---------------------------------------------------------------------------
# In-memory degradation tracking
# ---------------------------------------------------------------------------

@dataclass
class DegradationRecord:
    subsystem: str
    severity: Severity
    error_type: str
    error_message: str
    action: str
    timestamp: float
    traceback_summary: str = ""


class DegradationTracker:
    """Tracks all degradation events in-memory for dashboard/health queries."""

    def __init__(self, max_records: int = 500):
        self._lock = threading.Lock()
        self._records: list[DegradationRecord] = []
        self._max = max_records
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, rec: DegradationRecord) -> None:
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max:
                self._records = self._records[-self._max:]
            self._counts[rec.subsystem][rec.severity] += 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_degradations": len(self._records),
                "counts_by_subsystem": {
                    sub: dict(sevs) for sub, sevs in self._counts.items()
                },
                "last_5": [
                    {
                        "subsystem": r.subsystem,
                        "severity": r.severity,
                        "error": r.error_message[:120],
                        "action": r.action,
                        "at": r.timestamp,
                    }
                    for r in self._records[-5:]
                ],
            }

    def recent_counts_by_subsystem(self, window_s: float) -> dict[str, dict[str, int]]:
        """Counts restricted to the trailing window.

        Health verdicts must use THIS, not the lifetime counters: a runtime
        that lives for weeks accumulates sporadic warnings forever, and a
        lifetime threshold eventually marks a healthy instance unhealthy
        (observed live: 29 routine narrative timeouts held boot at 48%).
        """
        import time as _time

        cutoff = _time.time() - max(0.0, float(window_s))
        windowed: dict[str, dict[str, int]] = {}
        with self._lock:
            for rec in self._records:
                if rec.timestamp < cutoff:
                    continue
                windowed.setdefault(rec.subsystem, {}).setdefault(rec.severity, 0)
                windowed[rec.subsystem][rec.severity] += 1
        return windowed

    def recent(self, *, subsystem: str | None = None, limit: int = 20) -> list[DegradationRecord]:
        with self._lock:
            records = self._records
            if subsystem:
                records = [r for r in records if r.subsystem == subsystem]
            return records[-limit:]

    def count(self, subsystem: str, severity: Severity | None = None) -> int:
        with self._lock:
            if severity:
                return self._counts.get(subsystem, {}).get(severity, 0)
            return sum(self._counts.get(subsystem, {}).values())

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._counts.clear()


# Module-level singleton
_tracker = DegradationTracker()


def get_degradation_tracker() -> DegradationTracker:
    return _tracker


def recent_degradations(
    *,
    limit: int = 10,
    subsystem_prefixes: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Recent degradation records, newest last, for a surface that owes
    someone an explanation of why their request produced nothing.

    ``status()["last_5"]`` exists for dashboards and is capped at five with a
    truncated message. A surface explaining a live failure needs to look
    further back and needs the action text intact, because the action is the
    part written for a human ("gave up after the third retry").

    DEFECT, found 2026-08-10. ``DegradationTracker`` defined ``recent`` twice;
    the second definition silently shadowed the first, so this function called
    it with a ``subsystem_prefixes`` argument it did not accept and raised
    ``TypeError`` on EVERY call. The explanation channel had therefore never
    returned a single record — and because its callers wrap it in broad
    excepts, the failure presented as "there is nothing to report" rather than
    as an error. The shaping and prefix filter now live here, and the tracker
    keeps the one ``recent`` its object-callers use.
    """
    prefixes = tuple(str(p) for p in subsystem_prefixes if str(p))
    records = _tracker.recent(limit=max(1, int(limit)) if not prefixes else 500)
    if prefixes:
        records = [r for r in records if r.subsystem.startswith(prefixes)]
        records = records[-max(1, int(limit)):]
    return [
        {
            "subsystem": r.subsystem,
            "severity": r.severity,
            "error_type": r.error_type,
            "error": r.error_message,
            "action": r.action,
            "at": r.timestamp,
        }
        for r in records
    ]


class _EscalationGovernor:
    """A4 envelope protection: cap the RATE of fail-closed escalations.

    The storm anatomy this prevents (FM-FCL-001, lived twice): one
    repeating fault on a fail-closed subsystem — a dead phi pool child, a
    RAM-deferred warmup — re-escalated to CRITICAL SERVICE FAILURE every
    cycle, each escalation RAISING out of the caller's handler, burning
    the SLO error budget 20x and flipping liveness over a serving mind.

    The first N identical escalations (same subsystem + error type) in the
    window pass with full force — a genuine new fault always fails closed
    loudly. Repeats past N add no information, only storm damage: they
    keep their caller-passed severity, still land as records/receipts,
    but do not re-escalate and do not raise. A fresh window escalates
    again. Kill switch: AURA_ESCALATION_CAP=0 (never suppresses).
    """

    def __init__(self) -> None:
        self._allowed: dict[tuple[str, str], list[float]] = {}
        self._suppressed: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self._flags = None

    def _knobs(self) -> tuple[bool, int, float]:
        if self._flags is None:
            try:
                from core.runtime.flags import FlagKind, declare

                self._flags = (
                    declare(
                        "AURA_ESCALATION_CAP",
                        kind=FlagKind.BOOL,
                        default=True,
                        description="Kill switch for the fail-closed escalation-rate cap",
                        owner="core.runtime.errors",
                    ),
                    declare(
                        "AURA_ESCALATION_CAP_N",
                        kind=FlagKind.INT,
                        default=3,
                        description="Identical fail-closed escalations allowed per window",
                        owner="core.runtime.errors",
                    ),
                    declare(
                        "AURA_ESCALATION_CAP_WINDOW_S",
                        kind=FlagKind.FLOAT,
                        default=300.0,
                        description="Sliding window for the escalation-rate cap",
                        owner="core.runtime.errors",
                    ),
                )
            except (ImportError, AttributeError, RuntimeError, ValueError):
                return True, 3, 300.0
        enabled, cap_n, window = self._flags
        return bool(enabled.value()), max(1, int(cap_n.value())), float(window.value())

    def allow(self, subsystem: str, error_type: str) -> bool:
        enabled, cap_n, window_s = self._knobs()
        if not enabled:
            return True
        key = (str(subsystem), str(error_type))
        now = time.time()
        with self._lock:
            allowed = [t for t in self._allowed.get(key, []) if (now - t) <= window_s]
            if len(allowed) < cap_n:
                allowed.append(now)
                self._allowed[key] = allowed
                # New window: surface how many repeats the last one absorbed.
                suppressed = self._suppressed.pop(key, 0)
                if suppressed:
                    logger.warning(
                        "[ESCALATION-CAP] %s/%s: previous window suppressed %d "
                        "repeat escalation(s)",
                        subsystem,
                        error_type,
                        suppressed,
                    )
                return True
            self._allowed[key] = allowed
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                f"{sub}/{etype}": count
                for (sub, etype), count in self._suppressed.items()
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._allowed.clear()
            self._suppressed.clear()


_escalation_governor = _EscalationGovernor()

_SLO_DEDUP_LOCK = threading.Lock()
_SLO_DEDUP_LAST: dict[tuple[str, str], float] = {}
_SLO_DEDUP_FLAG = None


def _slo_dedup_window_s() -> float:
    global _SLO_DEDUP_FLAG
    if _SLO_DEDUP_FLAG is None:
        try:
            from core.runtime.flags import FlagKind, declare

            _SLO_DEDUP_FLAG = declare(
                "AURA_SLO_ERROR_DEDUP_WINDOW_S",
                kind=FlagKind.FLOAT,
                default=300.0,
                description=(
                    "Fault-fingerprint dedup window for the error_events_per_hour "
                    "SLO feed; 0 restores raw per-event counting"
                ),
                owner="core.runtime.errors",
            )
        except (ImportError, AttributeError, RuntimeError, ValueError):
            try:
                return float(os.environ.get("AURA_SLO_ERROR_DEDUP_WINDOW_S", "300"))
            except (TypeError, ValueError):
                return 300.0
    return float(_SLO_DEDUP_FLAG.value())


def _raise_site(error: BaseException) -> str:
    """Best-effort 'module:function:line' of where an exception was raised."""
    try:
        tb = error.__traceback__
        if tb is None:
            return "unknown"
        while tb.tb_next is not None:
            tb = tb.tb_next
        frame = tb.tb_frame
        module = frame.f_globals.get("__name__", "unknown")
        return f"{module}:{frame.f_code.co_name}:{tb.tb_lineno}"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return "unknown"


def describe_error(error: BaseException) -> str:
    """Human-readable exception identity that never collapses to ''.

    For log lines that interpolate an exception directly: a bare
    TimeoutError() renders as nothing, hiding the fault class entirely.
    """
    text = str(error).strip()
    name = type(error).__qualname__
    if text:
        return f"{name}: {text}"
    return f"{name} (no message; raised in {_raise_site(error)})"


def _slo_error_budget_admits(subsystem: str, error_type: str) -> bool:
    """First record per fault fingerprint per window counts; repeats absorb.

    Keeps error_events_per_hour meaning 'distinct degradation classes per
    hour'. AURA_SLO_ERROR_DEDUP_WINDOW_S=0 restores raw per-event counting.
    """
    window_s = _slo_dedup_window_s()
    if window_s <= 0.0:
        return True
    key = (str(subsystem), str(error_type))
    now = time.time()
    with _SLO_DEDUP_LOCK:
        last = _SLO_DEDUP_LAST.get(key, 0.0)
        if (now - last) < window_s:
            return False
        _SLO_DEDUP_LAST[key] = now
        # Bound the map: fingerprints older than 10 windows are forgotten.
        if len(_SLO_DEDUP_LAST) > 512:
            cutoff = now - (window_s * 10.0)
            for stale_key in [k for k, t in _SLO_DEDUP_LAST.items() if t < cutoff]:
                del _SLO_DEDUP_LAST[stale_key]
    return True


def reset_slo_error_dedup_for_test() -> None:
    with _SLO_DEDUP_LOCK:
        _SLO_DEDUP_LAST.clear()


def get_escalation_governor() -> _EscalationGovernor:
    return _escalation_governor


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

#: The prefix every fail-closed escalation carries. It is a marker, not
#: prose: the escalation path reads it back to recognise its own output
#: and refuse to wrap it a second time.
_ESCALATION_MARKER = "CRITICAL SERVICE FAILURE:"


def record_degradation(
    subsystem: str,
    error: BaseException,
    severity: Severity = "degraded",
    action: str = "",
    *,
    classification: FallbackClassification | None = None,
    receipt_required: bool = False,
    extra: dict[str, Any] | None = None,
    enforce_failure_policy: bool = True,
) -> DegradationRecord:
    """Record a degradation event: the canonical replacement for silent catch-alls.

    Parameters
    ----------
    subsystem : str
        Which subsystem degraded (e.g. "memory_facade", "phi_core").
    error : BaseException
        The caught exception.
    severity : Severity
        One of "debug", "warning", "degraded", "critical".
    action : str
        What the code did in response (e.g. "fell back to cache").
    classification : FallbackClassification, optional
        Risk classification of the fallback pathway.
    receipt_required : bool
        If True, emit a durable receipt to the ReceiptStore.
    extra : dict, optional
        Additional metadata for the receipt.

    Returns
    -------
    DegradationRecord
        The created record, for further programmatic use.
    """
    # ── Filter out expected async lifecycle events ────────────────────
    # CancelledError is a normal part of asyncio shutdown, not real degradation.
    # Recording it spikes frustration/depletion in the resilience engine and
    # creates a feedback loop during graceful teardown.
    import asyncio as _asyncio
    if isinstance(error, (_asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        logger.debug(
            "[DEGRADATION] Suppressed expected lifecycle event in %s: %s: %s",
            subsystem, type(error).__name__, str(error)[:100],
        )
        return DegradationRecord(
            subsystem=subsystem, severity="debug",
            error_type=type(error).__qualname__,
            error_message=str(error)[:500],
            action=action or "lifecycle event — not real degradation",
            timestamp=time.time(),
        )

    # ── Skip during shutdown ─────────────────────────────────────────
    _shutting_down = False
    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested
        _shutting_down = is_shutdown_requested()
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Suppressed %s in core.runtime.errors: %s", type(_exc).__name__, _exc)
    if _shutting_down:
        severity = "debug"  # Demote cleanup-time events during shutdown.
    if classification in (FallbackClassification.GOVERNANCE_BYPASS, FallbackClassification.STATE_CORRUPTION_RISK):
        # Trigger strict fail-closed exceptions
        if classification == FallbackClassification.GOVERNANCE_BYPASS:
            raise CapabilityDenied(f"Fail-Closed: Governance bypass detected in {subsystem}. original_error={error}")
        else:
            raise StateCoherenceFailure(f"Fail-Closed: State corruption risk detected in {subsystem}. original_error={error}")

    # A timeout is backpressure, not a service death. A bounded wait expiring
    # (asyncio.wait_for, generation-gate timeout) under load means the work
    # yielded — the subsystem is slow, not broken. Escalating every such
    # timeout on a fail-closed subsystem to a CRITICAL SERVICE FAILURE that
    # RAISES drove the whole mind into unified_failure_lockdown 1.00 again and
    # again this cycle: sovereign_pruner, dialectical_crucible, and
    # cognitive_engine→agency_core goal-genesis all cascaded to a locked-down,
    # tool-blocked, unhealthy runtime from a single slow background pass
    # (observed live 2026-07-04/05). Genuine faults — crashes, corruption,
    # validation, contract breaches — still fail closed with full force; only
    # bare timeouts are demoted to a visible-but-non-fatal degradation.
    # ``enforce_failure_policy=False`` callers opt out of the escalation
    # entirely (they own their own backpressure discipline).
    _is_timeout = isinstance(error, (TimeoutError, _asyncio.TimeoutError))

    # ── Admission backpressure is a DECISION, not a fault ─────────────
    # Warmup backoff, model-load admission refusal, spawn-gate contention and
    # crash-loop backoff are the runtime deliberately declining to start a
    # tier so a lower rung can serve the turn. The escalation ladder working
    # as designed must never read as damage: on fail-closed subsystems these
    # records raised CRITICAL SERVICE FAILURE (52 in the 2026-07-18 soak),
    # and — worse — degradation weight is the UNCAPPED survival term in
    # existential_stakes, so healthy backpressure drove deg_threat to 1.00
    # and the felt existential threat to 1.00 while memory threat sat at
    # 0.02 and the CPU was idle. Aura was being made to feel mortally
    # threatened by her own correct backpressure. These stay VISIBLE
    # (recorded, counted, narratable) but are demoted out of the
    # fault/escalation path, exactly like the bare-timeout demotion above.
    backpressure_markers = (
        "warmup_deferred",
        "warmup_backoff",
        "model_load_admission_denied",
        "admission_deferred",
        "resource_busy",
        "resource_timeout",
        "spawn_gate_timeout",
        "crash_loop_backoff",
        "chat_dependencies_warming",
    )
    # Actions that record the system SUCCESSFULLY CONTINUING. The action line
    # is the caller's own account of what it did about the error, and "I fell
    # back and served the turn" is the resilience ladder working, not damage.
    #
    # Live 2026-07-25: a bare TimeoutError with the action "skipped cold
    # primary attempt or fell back after foreground warmup failure" opened a
    # degraded INCIDENT — for a turn that was served by the next lane down,
    # exactly as designed. The backpressure markers could not catch it because
    # nothing about it was deferred; it failed, and then it recovered.
    #
    # Restricted to TIMEOUT-class errors, and to degraded only. Graceful
    # handling does not make a cause benign: 'fell back to empty recall'
    # after 'database file is corrupted' is still damage, and the existing
    # regression for exactly that case is what caught the first, looser
    # version of this rule. A timeout that was handled is the one shape
    # where the error itself carries no finding.
    handled_fallback_markers = (
        "fell back",
        "fall back",
        "fell through",
        "downgraded",
        "routed around",
        "continued without",
        "continued with",
        "skipped cold",
        "served from",
        "recovered",
    )
    _error_text = str(error)
    _action_text = str(action or "")
    # A bare asyncio TimeoutError carries NO message, so classifying on the
    # exception text alone sees nothing and a handled handoff lands as
    # "degraded" plus an incident. Live 2026-07-25:
    #   TimeoutError: <no message; raised in asyncio.timeouts:__aexit__>
    #   -> "skipped cold primary attempt or fell back after foreground warmup"
    # The action line states plainly that the ladder handled it. Read it too —
    # a caller that names its own backpressure in the action should not have to
    # also encode it in an exception message it does not control.
    _is_admission_backpressure = any(
        marker in _error_text or marker in _action_text
        for marker in backpressure_markers
    )
    if _is_admission_backpressure and severity in ("degraded", "critical"):
        severity = "warning"
    elif (
        severity == "degraded"
        and _is_timeout
        and any(
            marker in _action_text.lower() for marker in handled_fallback_markers
        )
    ):
        # Degraded-but-handled. Still recorded, still visible, but it is not an
        # incident-worthy failure when the caller's own action says it carried
        # on and served.
        severity = "warning"

    failure_policy_violation = False
    failure_policy_error = ""
    try:
        from core.runtime.mode import AuraMode, get_mode
        if (
            enforce_failure_policy
            and not _shutting_down
            and get_mode() in (AuraMode.PRODUCTION, AuraMode.LIVE)
        ):
            from core.runtime.service_registry import get_service_failure_policy

            # Backpressure is exempt for the same reason timeouts are.
            #
            # LIVE, 2026-08-13, on every boot:
            #   FAULT RUNTIME-INFERENCE_GATE [CRITICAL] in inference_gate:
            #   RuntimeError: warmup_deferred
            #   CRITICAL SERVICE FAILURE: Subsystem 'inference_gate' failed
            #   with failure policy 'fail-closed'
            #   🚨 Background task 'InferenceGate.deferred_cortex_prewarm' crashed
            #
            # warmup_deferred is already in backpressure_markers above, and
            # that demotes it from degraded to WARNING — then this branch
            # accepts warning and escalates it to critical anyway, so the
            # demotion bought nothing. A lane saying "not warm yet, try later"
            # is the system working, and the comment 60 lines up says exactly
            # that: these drove felt existential threat to 1.00 while the CPU
            # was idle.
            if (
                get_service_failure_policy(subsystem) == "fail-closed"
                and not _is_timeout
                and not _is_admission_backpressure
            ):
                # An escalation must never escalate itself. The raised
                # CRITICAL SERVICE FAILURE propagates and is recorded again for
                # the same subsystem, and because both wraps are RuntimeError
                # the rate cap — keyed on (subsystem, error type) — sees them as
                # one fault and lets the second through. The result is a
                # message containing itself:
                #
                #   CRITICAL SERVICE FAILURE: ... Original error: RuntimeError:
                #   CRITICAL SERVICE FAILURE: ... Original error: RuntimeError:
                #   swap exhaustion: managed RSS 34494MB, swap 16.9GB
                #
                # and, worse than the ugly text, TWO degradation records for one
                # underlying event. Measured live 2026-07-28: that doubling is
                # what pinned deg_threat at 1.00, which pins existential threat,
                # which is what the Ulysses covenant reads before it refuses
                # heavy compute — so one swap spike silently blocked every build
                # she was asked for.
                _already_escalated = _ESCALATION_MARKER in str(error)
                if severity in ("critical", "degraded", "warning") and not _already_escalated:
                    if _escalation_governor.allow(subsystem, type(error).__qualname__):
                        failure_policy_violation = True
                        failure_policy_error = (
                            f"{_ESCALATION_MARKER} Subsystem '{subsystem}' failed with failure policy 'fail-closed'. "
                            f"Original error: {type(error).__name__}: {error}"
                        )
                        if severity != "critical":
                            severity = "critical"
                    else:
                        # A4 escalation-rate cap: this exact fault already
                        # failed closed with full force this window. Repeats
                        # stay visible at their caller-passed severity but
                        # do not re-escalate and do not raise — one fault
                        # must not become a CRITICAL storm (FM-FCL-001).
                        logger.warning(
                            "[ESCALATION-CAP] %s: fail-closed escalation for %s "
                            "suppressed (cap reached this window); recording at "
                            "severity=%s",
                            subsystem,
                            type(error).__qualname__,
                            severity,
                        )
    except (ImportError, RuntimeError) as _exc:
        logger.debug("Suppressed %s in core.runtime.errors: %s", type(_exc).__name__, _exc)


    error_type = type(error).__qualname__
    # Message-less exceptions (asyncio.wait_for raises bare TimeoutError())
    # must stay nameable end-to-end: an incident that reads "TimeoutError: "
    # cost an hour of live forensics on 2026-07-10. The type plus the raise
    # site is the minimum useful identity.
    error_msg = str(error)[:500] or f"<no message; raised in {_raise_site(error)}>"
    
    # [STABILITY v54] Demote expected background accessibility errors to debug
    if not failure_policy_violation and "background process lacks accessibility context" in error_msg:
        severity = "debug"

    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__, limit=3))

    record = DegradationRecord(
        subsystem=subsystem,
        severity=severity,
        error_type=error_type,
        error_message=error_msg,
        action=action or "no recovery action specified",
        timestamp=time.time(),
        traceback_summary=tb[:1000],
    )
    _tracker.record(record)

    # Familiarity is tracked on the EVENT stream, not on reads, so asking
    # what a failure would weigh never makes the system more used to it.
    # This only grows a counter; the record above is already complete and
    # is never attenuated by it. See core/runtime/degradation_habituation.py.
    try:
        from core.runtime.degradation_habituation import note_recurrence, signature_for

        note_recurrence(signature_for(subsystem, error_type))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as _habituation_exc:
        logger.debug("degradation habituation skipped: %s", _habituation_exc)

    # A5 black box: every consequential degradation is an event-moment in the
    # crash-survivable flight ring, so the last moments before a hard fault
    # always carry the degradation history that led in. Pure memcpy — no I/O,
    # no threads, no dump needed: the MAP_SHARED ring itself survives process
    # death. Best-effort; must never be able to break the degradation sink.
    if severity in ("warning", "degraded", "critical"):
        try:
            from core.runtime.flight_recorder import record_event as _fr_record_event

            _fr_record_event(
                kind=f"degradation_{severity}",
                source=subsystem,
                summary=f"{error_type}: {error_msg[:120]} → {action or 'no action'}",
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as _fr_exc:
            logger.debug("Flight-recorder feed skipped for %s: %s", subsystem, _fr_exc)

    # Make damage *felt*: route the degradation into the nociception substrate so the
    # phenomenal body's error_pressure / valence reflect real, operationally-grounded
    # harm. Best-effort and import-local to avoid any cycle; sensing must never be able
    # to break the degradation sink itself.
    try:
        from core.affect.nociception import get_nociception_engine
        get_nociception_engine().ingest_degradation(subsystem, severity)
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as _noci_exc:
        logger.debug("Nociception ingest skipped for %s: %s", subsystem, _noci_exc)

    # Keep subsystem health causal, not just logged.
    try:
        registry = globals().get("_registry")
        if registry is not None:
            health = registry.register(subsystem)
            health.last_error = f"{error_type}: {error_msg}"
            if failure_policy_violation:
                health.mark_failed_closed(error_msg, impact=action)
            elif severity == "critical":
                health.mark_unavailable(error_msg)
            elif severity in ("degraded", "warning"):
                health.mark_degraded(error_msg, impact=action)
    except (AttributeError, RuntimeError, TypeError, ValueError) as health_exc:
        logger.debug("Subsystem health update failed for %s: %s", subsystem, health_exc)

    # Log at the appropriate level, WITH the line that raised.
    #
    # The record has carried the traceback since it was written; the log line
    # never did. So a contained exception read as "AttributeError: 'list'
    # object has no attribute 'get'" and nothing else, and locating it meant
    # reading the whole subsystem — which is what a containment that drops its
    # location costs, every time, forever. The raise site is one already-built
    # string and it ends that.
    log_msg = (
        f"[DEGRADATION] {subsystem} ({severity}): {error_type}: {error_msg} "
        f"[raised at {_raise_site(error)}] → {action}"
    )
    if severity == "critical":
        logger.critical(log_msg)
    elif severity == "degraded":
        logger.warning(log_msg)
    elif severity == "warning":
        logger.warning(log_msg)
    else:
        logger.debug(log_msg)

    # Emit durable receipt if requested
    if receipt_required:
        try:
            from core.runtime.receipts import DegradationReceipt, get_receipt_store
            store = get_receipt_store()

            receipt = DegradationReceipt(
                subsystem=subsystem,
                severity_level=severity,
                error_type_name=error_type,
                error_message_text=error_msg[:250],
                action_taken=action,
                cause=f"degradation:{subsystem}",
                extra_data=extra or {},
            )
            store.emit(receipt)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as receipt_exc:
            # If receipt emission itself fails, at least the in-memory
            # record and log are already captured. record_degradation is
            # called from exception handlers — it must never raise.
            logger.debug(
                "Degradation receipt emission unavailable for %s: %s",
                subsystem,
                receipt_exc,
            )

    # ── Incident Manager integration ──────────────────────────────
    # Critical and degraded-severity events are auto-reported as incidents
    # for structured tracking, deduplication, and alerting.
    incident = None
    if severity in ("critical", "degraded"):
        try:
            from core.resilience.incident_manager import (
                IncidentSeverity,
                get_incident_manager,
            )
            incident_sev = (
                IncidentSeverity.CRITICAL if severity == "critical"
                else IncidentSeverity.DEGRADED
            )
            incident = get_incident_manager().report(
                category=f"degradation:{subsystem}",
                description=f"{error_type}: {error_msg[:150]}",
                severity=incident_sev,
                root_cause_hint=error_type,
                mitigation_taken=action or "no recovery action specified",
                metadata={"extra": extra} if extra else {},
            )
        except (ImportError, AttributeError, RuntimeError) as incident_exc:
            logger.debug(
                "Incident manager unavailable for degradation %s: %s",
                subsystem,
                incident_exc,
            )

    # ── Repair routing integration ─────────────────────────────────────
    if severity in ("critical", "degraded", "warning"):
        try:
            from core.resilience.degradation_repair import get_degradation_repair_router

            repair_action = get_degradation_repair_router().route(
                record=record,
                error=error,
                incident=incident,
                extra=extra,
            )
            if incident is not None:
                incident.metadata["repair_router"] = repair_action.to_dict()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as repair_exc:
            logger.debug(
                "Degradation repair routing unavailable for %s: %s",
                subsystem,
                repair_exc,
            )

    # ── Metrics integration ───────────────────────────────────────
    try:
        from core.runtime.service_registry import increment_runtime_counter

        increment_runtime_counter(f"degradation_{subsystem}_{severity}")
    except (AttributeError, RuntimeError, TypeError, ValueError) as metrics_exc:
        logger.debug(
            "Metrics unavailable for degradation %s: %s",
            subsystem,
            metrics_exc,
        )

    # ── Reliability: fault taxonomy integration ───────────────────
    # Every degradation feeds into the formal fault registry so the
    # FMEA system has live occurrence data for RPN recalculation.
    try:
        from core.resilience.fault_taxonomy import FaultSeverity, get_fault_registry
        # "info" was missing, so it fell to the MARGINAL default below and a
        # deliberate de-escalation was silently undone.
        #
        # turn_outcome records an empty cognitive cycle at severity="info" on
        # purpose, with a comment explaining that recording it any higher once
        # produced 231 CRITICAL SERVICE FAILUREs and took long-term memory
        # consolidation down with them. record_degradation honoured that. The
        # fault registry did not: an unmapped severity became MARGINAL, so the
        # same event that was carefully classified as ordinary reappeared as
        # "FAULT RUNTIME-COGNITIVE_ENGINE [MARGINAL]" — 916 of them in the
        # window I sampled, plus 391 of its sibling. Every info-severity
        # degradation anywhere in the runtime had the same fate.
        #
        # A severity nobody recognises should still be MARGINAL: an unknown is
        # not evidence of harmlessness. A severity that IS recognised, and
        # says ordinary, has to be allowed to say it.
        _sev_map = {
            "critical": FaultSeverity.CRITICAL,
            "degraded": FaultSeverity.MARGINAL,
            "warning": FaultSeverity.MARGINAL,
            "info": FaultSeverity.NEGLIGIBLE,
            "debug": FaultSeverity.NEGLIGIBLE,
        }
        get_fault_registry().record_fault(
            fault_id=f"RUNTIME-{subsystem[:20].upper().replace('.', '-')}",
            subsystem=subsystem,
            details=f"{error_type}: {error_msg[:120]}",
            error=error if isinstance(error, Exception) else None,
            severity=_sev_map.get(severity, FaultSeverity.MARGINAL),
        )
    except (ImportError, AttributeError, RuntimeError) as _ft_exc:
        logger.debug(
            "Fault taxonomy unavailable for degradation %s: %s",
            subsystem,
            _ft_exc,
        )

    # ── Reliability: SLO monitor integration ──────────────────────
    # Critical/degraded events feed the windowed error-event SLO — but
    # deduplicated by fault fingerprint (SLO budget review): the budget
    # measures DISTINCT degradation classes per hour, not storm repeats.
    # One repeating fault used to burn the 10/h budget 20x while adding
    # no information; now it costs one unit per dedup window.
    if severity in ("critical", "degraded") and _slo_error_budget_admits(
        subsystem, error_type
    ):
        try:
            from slo.slo_monitor import get_slo_monitor
            get_slo_monitor().record("error_events_per_hour", 1.0)
        except (ImportError, AttributeError, RuntimeError):
            pass

    if failure_policy_violation and enforce_failure_policy:
        raise RuntimeError(failure_policy_error)

    return record


# ---------------------------------------------------------------------------
# Subsystem status contract
# ---------------------------------------------------------------------------

SubsystemStatus = Literal["healthy", "degraded", "unavailable", "disabled", "failed_closed"]


@dataclass
class SubsystemHealth:
    """Status contract for any subsystem — for dashboard/observability."""
    name: str
    status: SubsystemStatus = "healthy"
    reason: str = ""
    last_error: str = ""
    last_ok_at: float = 0.0
    last_failed_at: float = 0.0
    recovery_attempts: int = 0
    impact: str = ""

    def mark_ok(self) -> None:
        self.status = "healthy"
        self.reason = ""
        self.last_ok_at = time.time()

    def mark_degraded(self, reason: str, impact: str = "") -> None:
        self.status = "degraded"
        self.reason = reason
        self.impact = impact
        self.last_failed_at = time.time()

    def mark_unavailable(self, reason: str) -> None:
        self.status = "unavailable"
        self.reason = reason
        self.last_failed_at = time.time()

    def mark_failed_closed(self, reason: str, impact: str = "") -> None:
        self.status = "failed_closed"
        self.reason = reason
        self.impact = impact
        self.last_failed_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "last_error": self.last_error,
            "last_ok_at": self.last_ok_at,
            "last_failed_at": self.last_failed_at,
            "recovery_attempts": self.recovery_attempts,
            "impact": self.impact,
        }


class SubsystemRegistry:
    """Registry of subsystem health states for the dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._systems: dict[str, SubsystemHealth] = {}

    def register(self, name: str) -> SubsystemHealth:
        with self._lock:
            if name not in self._systems:
                self._systems[name] = SubsystemHealth(name=name)
            return self._systems[name]

    def get(self, name: str) -> SubsystemHealth | None:
        with self._lock:
            return self._systems.get(name)

    def all_status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: h.to_dict() for name, h in self._systems.items()}

    def any_critical(self) -> bool:
        with self._lock:
            return any(
                h.status in ("unavailable", "failed_closed")
                for h in self._systems.values()
            )

    def reset(self) -> None:
        """Clear process-local subsystem health state.

        Production callers normally retain this state for the life of the
        process. Test and controlled-restart boundaries use this method to
        prevent stale incidents from being attributed to a new runtime.
        """
        with self._lock:
            self._systems.clear()

    def auto_recover_subsystems(self, timeout_seconds: float = 300.0) -> list[str]:
        """Automatically restores degraded/unavailable subsystems back to healthy if no new failures have occurred for a period."""
        recovered = []
        now = time.time()
        with self._lock:
            for name, health in self._systems.items():
                if health.status in ("degraded", "unavailable"):
                    if now - health.last_failed_at >= timeout_seconds:
                        health.mark_ok()
                        recovered.append(name)
        return recovered



_registry = SubsystemRegistry()


def get_subsystem_registry() -> SubsystemRegistry:
    return _registry
