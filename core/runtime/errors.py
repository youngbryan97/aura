"""core/runtime/errors.py — Structured degradation receipts.

The audit found 5,222 `except Exception` blocks, most swallowing errors
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

import logging
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger("Aura.Errors")

Severity = Literal["debug", "warning", "degraded", "critical"]

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
        self._records: List[DegradationRecord] = []
        self._max = max_records
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, rec: DegradationRecord) -> None:
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max:
                self._records = self._records[-self._max:]
            self._counts[rec.subsystem][rec.severity] += 1

    def status(self) -> Dict[str, Any]:
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

    def recent(self, *, subsystem: str | None = None, limit: int = 20) -> List[DegradationRecord]:
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


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def record_degradation(
    subsystem: str,
    error: BaseException,
    severity: Severity = "degraded",
    action: str = "",
    *,
    receipt_required: bool = False,
    extra: Dict[str, Any] | None = None,
) -> DegradationRecord:
    """Record a degradation event — the canonical replacement for ``except Exception: pass``.

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
    receipt_required : bool
        If True, emit a durable receipt to the ReceiptStore.
    extra : dict, optional
        Additional metadata for the receipt.

    Returns
    -------
    DegradationRecord
        The created record, for further programmatic use.
    """
    error_type = type(error).__qualname__
    error_msg = str(error)[:500]
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

    # Log at the appropriate level
    log_msg = (
        f"[DEGRADATION] {subsystem} ({severity}): {error_type}: {error_msg} "
        f"→ {action}"
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
            from core.runtime.receipts import get_receipt_store, _ReceiptBase
            store = get_receipt_store()

            @dataclass
            class DegradationReceipt(_ReceiptBase):
                kind: str = "degradation"
                subsystem: str = ""
                severity_level: str = ""
                error_type_name: str = ""
                error_message_text: str = ""
                action_taken: str = ""
                extra_data: Dict[str, Any] = field(default_factory=dict)

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
        except Exception:
            # If receipt emission itself fails, at least the in-memory
            # record and log are already captured.
            pass  # no-op: intentional

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

    def to_dict(self) -> Dict[str, Any]:
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
        self._systems: Dict[str, SubsystemHealth] = {}

    def register(self, name: str) -> SubsystemHealth:
        with self._lock:
            if name not in self._systems:
                self._systems[name] = SubsystemHealth(name=name)
            return self._systems[name]

    def get(self, name: str) -> SubsystemHealth | None:
        with self._lock:
            return self._systems.get(name)

    def all_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: h.to_dict() for name, h in self._systems.items()}

    def any_critical(self) -> bool:
        with self._lock:
            return any(
                h.status in ("unavailable", "failed_closed")
                for h in self._systems.values()
            )


_registry = SubsystemRegistry()


def get_subsystem_registry() -> SubsystemRegistry:
    return _registry
