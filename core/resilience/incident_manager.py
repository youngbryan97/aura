"""core/resilience/incident_manager.py
======================================
Structured incident tracking for indefinite autonomous operation.

Converts repeated degradation events (starvation, fallback cascades,
event-loop lag, substrate divergence) into formal incidents with severity,
root-cause hints, mitigation taken, and recovery status.

This replaces silent degradation recording with actionable incident tracking.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("Aura.Resilience.IncidentManager")


class IncidentSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentStatus(str, Enum):
    ACTIVE = "active"
    MITIGATING = "mitigating"
    RECOVERED = "recovered"
    UNRECOVERABLE = "unrecoverable"


@dataclass
class Incident:
    """A structured incident record."""
    incident_id: str
    severity: IncidentSeverity
    category: str  # e.g., "substrate_divergence", "model_fallback", "memory_pressure"
    description: str
    root_cause_hint: str = ""
    mitigation_taken: str = ""
    status: IncidentStatus = IncidentStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0
    occurrence_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncidentManager:
    """Manages structured incidents for autonomous operation.

    Features:
    - Deduplication: repeated events of same category are merged
    - Escalation: repeated incidents auto-escalate severity
    - Recovery tracking: incidents are resolved when conditions normalize
    - Alert hooks: fires callbacks on critical/emergency severity
    """

    MAX_ACTIVE = 100
    MAX_HISTORY = 500
    ESCALATION_THRESHOLD = 5  # occurrences before severity escalation

    def __init__(self) -> None:
        self._active: Dict[str, Incident] = {}
        self._history: Deque[Incident] = deque(maxlen=self.MAX_HISTORY)
        self._alert_callbacks: List[Any] = []
        self._incident_counter = 0
        self._total_incidents = 0

    def _make_id(self) -> str:
        self._incident_counter += 1
        return f"INC-{int(time.time())}-{self._incident_counter:04d}"

    def report(
        self,
        category: str,
        description: str,
        severity: IncidentSeverity = IncidentSeverity.WARNING,
        root_cause_hint: str = "",
        mitigation_taken: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Incident:
        """Report an incident. Deduplicates by category.

        If an active incident of the same category exists, increments its
        occurrence count and possibly escalates severity.
        """
        self._total_incidents += 1

        # Dedup check
        if category in self._active:
            existing = self._active[category]
            existing.occurrence_count += 1
            existing.updated_at = time.time()
            existing.description = description  # Update to latest
            if root_cause_hint:
                existing.root_cause_hint = root_cause_hint
            if mitigation_taken:
                existing.mitigation_taken = mitigation_taken
            if metadata:
                existing.metadata.update(metadata)

            # Auto-escalate
            if existing.occurrence_count >= self.ESCALATION_THRESHOLD:
                existing.severity = self._escalate(existing.severity)

            logger.info(
                "Incident %s updated (count=%d, severity=%s): %s",
                existing.incident_id,
                existing.occurrence_count,
                existing.severity.value,
                description[:100],
            )

            self._fire_alerts(existing)
            return existing

        # New incident
        incident = Incident(
            incident_id=self._make_id(),
            severity=severity,
            category=category,
            description=description,
            root_cause_hint=root_cause_hint,
            mitigation_taken=mitigation_taken,
            metadata=dict(metadata or {}),
        )

        self._active[category] = incident

        # Evict oldest if at capacity
        while len(self._active) > self.MAX_ACTIVE:
            oldest_key = min(
                self._active, key=lambda k: self._active[k].created_at
            )
            evicted = self._active.pop(oldest_key)
            evicted.status = IncidentStatus.UNRECOVERABLE
            self._history.append(evicted)

        logger.warning(
            "NEW INCIDENT %s [%s] %s: %s",
            incident.incident_id,
            incident.severity.value,
            category,
            description[:100],
        )

        self._fire_alerts(incident)
        return incident

    def resolve(self, category: str, resolution: str = "") -> Optional[Incident]:
        """Resolve an active incident."""
        if category not in self._active:
            return None

        incident = self._active.pop(category)
        incident.status = IncidentStatus.RECOVERED
        incident.resolved_at = time.time()
        incident.updated_at = time.time()
        if resolution:
            incident.mitigation_taken = resolution
        self._history.append(incident)

        logger.info(
            "Incident %s RESOLVED [%s]: %s",
            incident.incident_id,
            category,
            resolution or "auto-recovered",
        )
        return incident

    def get_active(self) -> List[Dict[str, Any]]:
        """Get all active incidents."""
        return [
            {
                "incident_id": i.incident_id,
                "severity": i.severity.value,
                "category": i.category,
                "description": i.description[:200],
                "root_cause_hint": i.root_cause_hint[:100],
                "mitigation_taken": i.mitigation_taken[:100],
                "status": i.status.value,
                "occurrence_count": i.occurrence_count,
                "age_seconds": round(time.time() - i.created_at, 1),
            }
            for i in self._active.values()
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get incident manager summary."""
        active_by_severity: Dict[str, int] = {}
        for i in self._active.values():
            active_by_severity[i.severity.value] = (
                active_by_severity.get(i.severity.value, 0) + 1
            )

        return {
            "total_incidents": self._total_incidents,
            "active_count": len(self._active),
            "history_count": len(self._history),
            "active_by_severity": active_by_severity,
            "has_critical": any(
                i.severity in (IncidentSeverity.CRITICAL, IncidentSeverity.EMERGENCY)
                for i in self._active.values()
            ),
        }

    def register_alert_callback(self, callback: Any) -> None:
        """Register callback for critical/emergency incidents."""
        self._alert_callbacks.append(callback)

    def _fire_alerts(self, incident: Incident) -> None:
        """Fire alert callbacks for critical+ incidents."""
        if incident.severity not in (
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        ):
            return

        for callback in self._alert_callbacks:
            try:
                callback(incident)
            except Exception as e:
                logger.debug("Alert callback failed: %s", e)

    @staticmethod
    def _escalate(current: IncidentSeverity) -> IncidentSeverity:
        """Escalate severity one level."""
        order = [
            IncidentSeverity.INFO,
            IncidentSeverity.WARNING,
            IncidentSeverity.DEGRADED,
            IncidentSeverity.CRITICAL,
            IncidentSeverity.EMERGENCY,
        ]
        idx = order.index(current) if current in order else 0
        return order[min(idx + 1, len(order) - 1)]


# Singleton
_incident_manager: Optional[IncidentManager] = None


def get_incident_manager() -> IncidentManager:
    global _incident_manager
    if _incident_manager is None:
        _incident_manager = IncidentManager()
    return _incident_manager
