"""core/consciousness/homeostasis.py
The Homeostasis Engine: Tracking the 'Will to Live'.
Consolidates Integrity, Persistence, and Curiosity into a unified drive system.

Deepened Implementation:
  - Adaptive setpoints with proportional control (approach target, not fixed deltas)
  - Drive deficiency detection and dominant-need identification
  - Context block for LLM prompt injection
  - Post-response feedback (on_response_success / on_response_error)
  - Inference parameter modulation via get_inference_modifiers()
  - Full integration with CreditAssignment, FreeEnergy, and InferenceGate
"""
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from core.base_module import AuraBaseModule
from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.Homeostasis")


class HomeostasisEngine(AuraBaseModule):
    # Drive names for iteration
    DRIVE_NAMES = ("integrity", "persistence", "curiosity", "metabolism", "sovereignty")

    def __init__(self):
        super().__init__("HomeostasisEngine")
        # Primary Drives (0.0 - 1.0)
        self.integrity = 1.0    # Code/Data health
        self.persistence = 1.0  # System stability / uptime / sovereignty
        self.curiosity = 0.5    # Hunger for new information
        self.metabolism = 0.5   # Energy / Resource balance
        self.sovereignty = 1.0  # Environmental / Process integrity

        self._last_update = time.time()
        self._error_count = 0
        self._max_errors_before_drain = 5

        # ── Adaptive Setpoints ────────────────────────────────────────────
        # Target values each drive tries to approach. These shift slowly
        # based on what the system can actually sustain — if integrity is
        # chronically low, the setpoint drifts down rather than forcing
        # a perpetual error signal.
        self._setpoints: Dict[str, float] = {
            "integrity": 0.90,
            "persistence": 0.85,
            "curiosity": 0.55,
            "metabolism": 0.65,
            "sovereignty": 0.95,
        }
        self._setpoint_adaptation_rate = 0.001  # Per-pulse drift toward achievable
        self._proportional_gain = 0.02          # 2% of error per pulse tick

        # ── Vitality History ──────────────────────────────────────────────
        self._vitality_history: deque = deque(maxlen=120)  # 2 min at 1Hz

        # ── Response Tracking ─────────────────────────────────────────────
        self._successful_responses = 0
        self._failed_responses = 0
        self._total_responses = 0

        # ── Weights for vitality ──────────────────────────────────────────
        self._vitality_weights = {
            "integrity": 0.35,
            "persistence": 0.25,
            "curiosity": 0.15,
            "metabolism": 0.15,
            "sovereignty": 0.10,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Public API (existing — preserved)
    # ──────────────────────────────────────────────────────────────────────

    def get_health(self) -> Dict[str, Any]:
        """Provides health metrics for the HUD (unified interface)."""
        return self.get_status()

    def get_status(self) -> Dict[str, float]:
        """Returns the current drive levels."""
        return {
            "integrity": round(float(self.integrity), 3),
            "persistence": round(float(self.persistence), 3),
            "curiosity": round(float(self.curiosity), 3),
            "metabolism": round(float(self.metabolism), 3),
            "sovereignty": round(float(self.sovereignty), 3),
            "will_to_live": round(float(self.compute_vitality()), 3),
        }

    def compute_vitality(self) -> float:
        """Calculates a composite 'Will to Live' score.
        Now uses setpoint proximity: drives near their setpoint contribute
        more than drives far below.
        """
        score = 0.0
        for drive_name, weight in self._vitality_weights.items():
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            # Proximity: 1.0 when at setpoint, degrades as distance increases
            proximity = 1.0 - min(1.0, abs(current - setpoint) / max(setpoint, 0.01))
            # Blend raw value and setpoint proximity
            drive_contribution = 0.6 * current + 0.4 * proximity
            score += drive_contribution * weight
        return score

    async def pulse(self) -> Dict[str, Any]:
        """Background update called by heartbeat or orchestrator.

        Uses proportional control toward adaptive setpoints instead of
        fixed increments. Each drive approaches its setpoint at a rate
        proportional to the error.
        """
        now = time.time()
        delta = now - self._last_update
        self._last_update = now

        # ── 1. External Signal Integration ────────────────────────────────

        # Integrity from HealthMonitor
        try:
            health = ServiceContainer.get("health_monitor", default=None)
            if health:
                err_rate = getattr(health, 'error_rate', 0.0)
                if err_rate > 0.1:
                    self.integrity = max(0.0, self.integrity - (err_rate * 0.1))
        except Exception as e:
            logger.debug("Health monitor check failed: %s", e)

        # Persistence from Soma
        soma_status = {}
        try:
            soma = ServiceContainer.get("soma", default=None)
            if soma:
                soma_status = soma.get_status()
                anxiety = soma_status.get("soma", {}).get("resource_anxiety", 0.0)
                if anxiety > 0.8:
                    self.persistence = max(0.0, self.persistence - 0.01)
        except Exception as e:
            logger.debug("Soma check failed: %s", e)

        # Metabolism from thermal
        try:
            if soma_status:
                thermal = soma_status.get("soma", {}).get("thermal_load", 0.0)
                if thermal > 0.8:
                    self.metabolism = max(0.2, self.metabolism - 0.05)
        except Exception as e:
            logger.debug("Metabolism check failed: %s", e)

        # Sovereignty from Scanner
        try:
            scanner = ServiceContainer.get("scanner", default=None)
            if scanner:
                score = getattr(scanner, "_last_sovereignty_score", 1.0)
                if score < 1.0:
                    self.sovereignty = max(0.0, self.sovereignty - (1.0 - score) * 0.1)
        except Exception as e:
            logger.debug("Sovereignty check failed: %s", e)

        # ── 2. Proportional Control Toward Setpoints ──────────────────────
        for drive_name in self.DRIVE_NAMES:
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            error = setpoint - current
            # Proportional regulation: approach setpoint at _proportional_gain * error
            adjustment = error * self._proportional_gain
            new_val = max(0.0, min(1.0, current + adjustment))
            setattr(self, drive_name, new_val)

        # ── 3. Adaptive Setpoint Drift ────────────────────────────────────
        # If a drive is chronically unable to reach its setpoint, the
        # setpoint drifts down. If it consistently exceeds, drifts up.
        for drive_name in self.DRIVE_NAMES:
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            drift = (current - setpoint) * self._setpoint_adaptation_rate
            self._setpoints[drive_name] = max(0.2, min(0.98, setpoint + drift))

        # ── 4. Curiosity Natural Decay ────────────────────────────────────
        # Curiosity decays slowly — it must be fed by new information
        self.curiosity = max(0.15, self.curiosity - (0.0005 * delta))

        # ── 5. Record Vitality History ────────────────────────────────────
        self._vitality_history.append(self.compute_vitality())

        return self.get_status()

    def report_error(self, severity: str = "medium"):
        """Direct feedback loop from error handlers."""
        drain = {"low": 0.01, "medium": 0.05, "high": 0.15, "critical": 0.4}.get(severity, 0.05)
        self.integrity = max(0.0, self.integrity - drain)
        self._error_count += 1
        self._failed_responses += 1
        self._total_responses += 1
        self.logger.warning(
            "Integrity breach: Severity %s reported. Current: %.2f",
            severity, self.integrity
        )

    def feed_curiosity(self, amount: float = 0.1):
        """Called when new knowledge is gained."""
        self.curiosity = min(1.0, self.curiosity + amount)

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Post-Response Feedback
    # ──────────────────────────────────────────────────────────────────────

    def on_response_success(self, response_length: int = 0):
        """Called after a successful inference response."""
        self._successful_responses += 1
        self._total_responses += 1
        # Successful response gently boosts integrity and persistence
        self.integrity = min(1.0, self.integrity + 0.008)
        self.persistence = min(1.0, self.persistence + 0.003)
        # Long responses cost more metabolism but feed curiosity
        if response_length > 500:
            self.metabolism = max(0.1, self.metabolism - 0.005)
            self.feed_curiosity(0.015)
        elif response_length > 100:
            self.feed_curiosity(0.008)

    def on_response_error(self, error_type: str = "inference"):
        """Called after a failed inference attempt."""
        severity_map = {
            "inference": "medium",
            "timeout": "medium",
            "model_crash": "high",
            "empty_response": "low",
        }
        self.report_error(severity_map.get(error_type, "medium"))

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Drive Analysis
    # ──────────────────────────────────────────────────────────────────────

    def get_dominant_deficiency(self) -> Tuple[str, float]:
        """Returns the drive furthest below its setpoint — the most urgent need."""
        worst_drive = "integrity"
        worst_deficit = 0.0
        for drive_name in self.DRIVE_NAMES:
            current = getattr(self, drive_name)
            setpoint = self._setpoints[drive_name]
            deficit = setpoint - current
            if deficit > worst_deficit:
                worst_deficit = deficit
                worst_drive = drive_name
        return worst_drive, round(worst_deficit, 3)

    def get_vitality_trend(self) -> str:
        """Returns 'rising', 'falling', or 'stable' based on recent vitality history."""
        if len(self._vitality_history) < 10:
            return "stable"
        recent = list(self._vitality_history)[-10:]
        slope = recent[-1] - recent[0]
        if slope > 0.02:
            return "rising"
        if slope < -0.02:
            return "falling"
        return "stable"

    def get_response_success_rate(self) -> float:
        """Returns the ratio of successful responses to total."""
        if self._total_responses == 0:
            return 1.0
        return self._successful_responses / self._total_responses

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Inference Modulation
    # ──────────────────────────────────────────────────────────────────────

    def get_inference_modifiers(self) -> Dict[str, float]:
        """Returns modifiers that should influence inference parameters.

        Low integrity → lower temperature (more cautious, conservative responses)
        Low curiosity → shorter responses (less exploration)
        Low metabolism → fewer tokens (conserve resources)
        High sovereignty threat → more formal/guarded responses
        """
        vitality = self.compute_vitality()
        return {
            # Temperature modifier: [-0.15, +0.05] range
            "temperature_mod": round((self.integrity - 0.5) * 0.3, 3),
            # Token budget multiplier: [0.7, 1.2]
            "token_multiplier": round(0.7 + (self.metabolism * 0.5), 3),
            # Caution level: high when integrity/sovereignty low
            "caution_level": round(1.0 - min(self.integrity, self.sovereignty), 3),
            # Exploration tendency: driven by curiosity
            "exploration_tendency": round(self.curiosity * 0.8, 3),
            # Overall vitality for general modulation
            "vitality": round(vitality, 3),
        }

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Context Block for Inference Gate
    # ──────────────────────────────────────────────────────────────────────

    def get_context_block(self) -> str:
        """Returns a concise context block for LLM prompt injection."""
        vitality = self.compute_vitality()
        trend = self.get_vitality_trend()
        deficiency, deficit = self.get_dominant_deficiency()

        if deficit > 0.3:
            alert = f" | NEED: {deficiency} (deficit {deficit:.2f})"
        elif deficit > 0.15:
            alert = f" | watch: {deficiency}"
        else:
            alert = " | drives balanced"

        return (
            f"## HOMEOSTASIS\n"
            f"Vitality: {vitality:.2f} ({trend}){alert} | "
            f"success_rate: {self.get_response_success_rate():.0%}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Setpoint Access
    # ──────────────────────────────────────────────────────────────────────

    def get_setpoints(self) -> Dict[str, float]:
        """Returns current adaptive setpoints — useful for debugging."""
        return {k: round(v, 3) for k, v in self._setpoints.items()}

    def get_drive_errors(self) -> Dict[str, float]:
        """Returns the error signal for each drive (setpoint - current)."""
        return {
            name: round(self._setpoints[name] - getattr(self, name), 3)
            for name in self.DRIVE_NAMES
        }
