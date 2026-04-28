"""core/consciousness/executive_inhibitor.py

Executive Inhibitor — Prefrontal Cortex Analogue for Ganglion Governance.

Prevents context collapse by authorizing or vetoing ganglion actions based on
the current system state (Φ, workspace ignition). Protects high-integration
states from disruption by non-critical stimuli.

Key behaviors:
1. HIGH-Φ PROTECTION: When Φ is above threshold and workspace is ignited,
   only critical actions are allowed through. This preserves integrated
   conscious states from being shattered by impulsive ganglion firings.

2. CRITICAL PASSTHROUGH: Safety-flagged actions always pass regardless of state.

3. VETO LOGGING: All vetoed actions are logged for debugging/telemetry.
"""

from core.runtime.errors import record_degradation
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("Consciousness.Executive")


class ExecutiveInhibitor:
    """Authorization layer for ganglion actions.

    Sits between the ganglion action queue and actual execution.
    Checks current system state before allowing actions to proceed.

    Usage:
        inhibitor = ExecutiveInhibitor()
        if inhibitor.authorize(action, phi=0.8, ignited=True):
            await execute(action)
    """

    # Default thresholds
    PHI_PROTECTION_THRESHOLD: float = 0.5     # Φ above which we protect integration
    IGNITION_REQUIRED: bool = True            # Also require workspace ignition for protection
    FIELD_COHERENCE_CRISIS: float = 0.25      # Below this → hard block non-critical
    FIELD_COHERENCE_WARNING: float = 0.40     # Below this → veto non-critical during ignition
    MAX_VETO_LOG: int = 200                   # Max veto entries to retain

    def __init__(
        self,
        phi_threshold: float = PHI_PROTECTION_THRESHOLD,
        require_ignition: bool = IGNITION_REQUIRED,
    ):
        """
        Args:
            phi_threshold: Φ level above which non-critical actions are inhibited.
            require_ignition: If True, protection only activates when workspace is also ignited.
        """
        self._phi_threshold = phi_threshold
        self._require_ignition = require_ignition

        # Counters
        self._authorized_count: int = 0
        self._vetoed_count: int = 0
        self._critical_passthrough_count: int = 0
        self._field_vetoed_count: int = 0

        # Veto and Audit logs
        self._veto_log: List[Dict[str, Any]] = []
        self._audit_trail: List[Dict[str, Any]] = []  # CS-02: Critical bypass tracking

        logger.info(
            "Executive Inhibitor online (phi_threshold=%.2f, require_ignition=%s, field_crisis=%.2f)",
            phi_threshold, require_ignition, self.FIELD_COHERENCE_CRISIS,
        )

    def authorize(
        self,
        action: Any,
        phi: float = 0.0,
        ignited: bool = False,
    ) -> bool:
        """Decide whether to authorize a ganglion action.

        Args:
            action: A GanglionAction instance (or any object with
                    is_critical, source_domain, action_type attributes).
            phi: Current Φ value from the substrate.
            ignited: Whether the global workspace is currently ignited.

        Returns:
            True if the action is authorized, False if vetoed.
        """
        # Critical actions ALWAYS pass (CS-02: Added Audit Trail)
        is_critical = getattr(action, "is_critical", False)
        if is_critical:
            self._critical_passthrough_count += 1
            self._authorized_count += 1
            
            # CS-02: Log the bypass for security auditing
            audit_entry = {
                "timestamp": time.time(),
                "source": getattr(action, "source_domain", "unknown"),
                "action": getattr(action, "action_type", "unknown"),
                "reason": "critical_bypass",
                "phi": round(phi, 4)
            }
            self._audit_trail.append(audit_entry)
            if len(self._audit_trail) > self.MAX_VETO_LOG:
                self._audit_trail.pop(0)
                
            logger.warning("🛡️ Executive BYPASS [Critical]: %s/%s", audit_entry["source"], audit_entry["action"])
            return True

        # ── UNIFIED FIELD COHERENCE GATE (mandatory) ────────────────
        # If the experiential field is fragmented, non-critical actions halt.
        # This is the embodied substrate saying "I am not coherent enough to act."
        try:
            from core.container import ServiceContainer
            unified_field = ServiceContainer.get("unified_field", default=None)
            if unified_field:
                field_coherence = unified_field.get_coherence()
                if field_coherence < self.FIELD_COHERENCE_CRISIS:
                    self._field_vetoed_count += 1
                    self._vetoed_count += 1
                    veto_entry = {
                        "timestamp": time.time(),
                        "source": getattr(action, "source_domain", "unknown"),
                        "action": getattr(action, "action_type", "unknown"),
                        "phi": round(phi, 4),
                        "field_coherence": round(field_coherence, 4),
                        "reason": "field_coherence_crisis",
                    }
                    self._veto_log.append(veto_entry)
                    if len(self._veto_log) > self.MAX_VETO_LOG:
                        self._veto_log = self._veto_log[-self.MAX_VETO_LOG:]
                    logger.info(
                        "🛑 Executive VETO [Field Crisis]: [%s/%s] (coherence=%.3f < %.3f)",
                        veto_entry["source"], veto_entry["action"],
                        field_coherence, self.FIELD_COHERENCE_CRISIS,
                    )
                    return False
        except Exception as e:
            record_degradation('executive_inhibitor', e)
            logger.debug("Field coherence check failed (allowing): %s", e)

        # ── HIGH-Φ INTEGRATION PROTECTION ────────────────────────────
        # Check if we're in a protected state
        in_protected_state = phi >= self._phi_threshold
        if self._require_ignition:
            in_protected_state = in_protected_state and ignited

        if in_protected_state:
            # VETO: Non-critical action during high-Φ integrated state
            self._vetoed_count += 1
            veto_entry = {
                "timestamp": time.time(),
                "source": getattr(action, "source_domain", "unknown"),
                "action": getattr(action, "action_type", "unknown"),
                "phi": round(phi, 4),
                "ignited": ignited,
                "reason": "high_phi_protection",
            }
            self._veto_log.append(veto_entry)
            if len(self._veto_log) > self.MAX_VETO_LOG:
                self._veto_log = self._veto_log[-self.MAX_VETO_LOG:]

            logger.debug(
                "Executive VETO: [%s/%s] (phi=%.3f, ignited=%s)",
                veto_entry["source"], veto_entry["action"], phi, ignited,
            )
            return False

        # Low-Φ or non-ignited: allow through
        self._authorized_count += 1
        return True

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot."""
        return {
            "authorized": self._authorized_count,
            "vetoed": self._vetoed_count,
            "critical_passthrough": self._critical_passthrough_count,
            "phi_threshold": self._phi_threshold,
            "require_ignition": self._require_ignition,
            "recent_vetoes": len(self._veto_log),
        }

    def get_recent_vetoes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the N most recent veto entries for debugging."""
        return self._veto_log[-n:]
