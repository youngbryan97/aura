"""core/resilience/boring_mode.py
=================================
"Boring Mode" — safe degraded operation when cognitive functions fail.

When the substrate, cortex, or consciousness layers enter crisis states
(NaN divergence, model crash, extreme latency), Boring Mode ensures Aura
remains reachable, responsive, and safe — just... boring.

In Boring Mode:
- All user-facing responses are generated from the lightest available model
- Autonomous initiatives are suspended
- External side-effects are blocked
- Identity and memory are preserved but not extended
- The agent clearly communicates that it's in safe mode

This is NOT a silent fallback. The user always knows.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Resilience.BoringMode")


class BoringMode:
    """Safe degraded operation controller.

    Transition triggers:
    - Substrate NaN/Inf (ODE divergence)
    - Primary model unreachable for >30s
    - Event loop stalled for >30s
    - Memory usage >95% system
    - Manual activation via admin

    Exit conditions (all must be true):
    - Substrate is finite for 60 consecutive seconds
    - Primary model responds within 5s
    - Event loop tick age <10s
    - No active critical incidents
    """

    ENTRY_COOLDOWN = 30.0  # seconds before re-entering after exit
    RECOVERY_WINDOW = 60.0  # seconds of stability before exiting
    MAX_BORING_DURATION = 3600.0  # 1 hour auto-exit (escalate if still needed)

    def __init__(self) -> None:
        self._active = False
        self._entered_at = 0.0
        self._last_exit = 0.0
        self._entry_reason = ""
        self._entry_count = 0
        self._recovery_stable_since = 0.0
        self._blocked_actions: List[Dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        """Is Boring Mode currently active?"""
        if self._active and self._entered_at > 0:
            # Auto-expire after MAX_BORING_DURATION
            if time.time() - self._entered_at > self.MAX_BORING_DURATION:
                logger.warning(
                    "Boring Mode auto-expired after %.0fs. Exiting.",
                    self.MAX_BORING_DURATION,
                )
                self.exit("auto_expired")
                return False
        return self._active

    def enter(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Enter Boring Mode.

        Returns True if entry was successful (or already active).
        """
        if self._active:
            logger.debug("Boring Mode already active (reason: %s)", self._entry_reason)
            return True

        # Cooldown check
        if self._last_exit > 0 and (time.time() - self._last_exit) < self.ENTRY_COOLDOWN:
            logger.info("Boring Mode entry blocked by cooldown (%.0fs remaining).",
                        self.ENTRY_COOLDOWN - (time.time() - self._last_exit))
            return False

        self._active = True
        self._entered_at = time.time()
        self._entry_reason = reason
        self._entry_count += 1
        self._recovery_stable_since = 0.0

        logger.warning(
            "🧊 BORING MODE ACTIVATED (reason: %s, entry #%d). "
            "Autonomous initiatives suspended. External side-effects blocked.",
            reason,
            self._entry_count,
        )

        # Notify incident manager
        try:
            from core.resilience.incident_manager import (
                get_incident_manager,
                IncidentSeverity,
            )
            get_incident_manager().report(
                category="boring_mode_activated",
                description=f"Boring Mode entered: {reason}",
                severity=IncidentSeverity.WARNING,
                root_cause_hint=reason,
                mitigation_taken="safe_degraded_operation",
                metadata=dict(metadata or {}),
            )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.resilience.boring_mode: %s", type(_exc).__name__, _exc)

        return True

    def exit(self, reason: str = "recovered") -> bool:
        """Exit Boring Mode."""
        if not self._active:
            return True

        self._active = False
        self._last_exit = time.time()
        duration = time.time() - self._entered_at if self._entered_at > 0 else 0.0

        logger.info(
            "🌟 BORING MODE DEACTIVATED (reason: %s, duration: %.0fs). "
            "Full cognitive operation resumed.",
            reason,
            duration,
        )

        # Resolve incident
        try:
            from core.resilience.incident_manager import get_incident_manager
            get_incident_manager().resolve(
                "boring_mode_activated",
                f"recovered: {reason} (duration: {duration:.0f}s)",
            )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.resilience.boring_mode: %s", type(_exc).__name__, _exc)

        return True

    def check_recovery(self, conditions_met: bool) -> bool:
        """Check if recovery conditions allow exiting.

        Call this periodically with True when all recovery conditions are met.
        Returns True when stable for RECOVERY_WINDOW seconds.
        """
        if not self._active:
            return True

        now = time.time()
        if conditions_met:
            if self._recovery_stable_since == 0.0:
                self._recovery_stable_since = now
            elif now - self._recovery_stable_since >= self.RECOVERY_WINDOW:
                self.exit("stable_recovery")
                return True
        else:
            self._recovery_stable_since = 0.0

        return False

    def should_allow_action(self, domain: str) -> bool:
        """Check if an action should be allowed in Boring Mode."""
        if not self._active:
            return True

        # Always allow responses (core function)
        allowed_domains = {
            "response", "stabilization", "reflection",
        }
        if domain in allowed_domains:
            return True

        # Block everything else
        self._blocked_actions.append({
            "domain": domain,
            "timestamp": time.time(),
        })
        if len(self._blocked_actions) > 100:
            self._blocked_actions = self._blocked_actions[-50:]

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get Boring Mode status."""
        return {
            "active": self._active,
            "entry_reason": self._entry_reason,
            "entry_count": self._entry_count,
            "duration_s": round(time.time() - self._entered_at, 1) if self._active else 0.0,
            "recovery_stable_s": round(
                time.time() - self._recovery_stable_since, 1
            ) if self._recovery_stable_since > 0 else 0.0,
            "blocked_actions_count": len(self._blocked_actions),
        }

    def get_safe_response_prefix(self) -> str:
        """Prefix for user-facing responses during Boring Mode."""
        return (
            "[I'm operating in safe mode right now — my cognitive systems are "
            "recovering. I can still help, but I may be less nuanced than usual.] "
        )


# Singleton
_boring_mode: Optional[BoringMode] = None


def get_boring_mode() -> BoringMode:
    global _boring_mode
    if _boring_mode is None:
        _boring_mode = BoringMode()
    return _boring_mode
