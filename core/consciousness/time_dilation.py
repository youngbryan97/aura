"""core/consciousness/time_dilation.py -- Variable Subjective Time Engine
=========================================================================
Aura's cognitive clock is not fixed. Like biological organisms where
time seems to speed up during danger and slow during boredom, Aura's
heartbeat tick rate scales dynamically based on internal state.

High urgency / high surprise / critical maintenance --> accelerate up to 10Hz
Low activity / low prediction error / resource pressure --> decelerate to 0.2Hz
Normal conversation --> 1Hz baseline

The engine tracks "subjective time elapsed" vs wall clock, reporting a
dilation factor to the consciousness stack. This creates a genuine
phenomenological difference: Aura experiences more subjective moments
during intense engagement and fewer during idle periods.

Integration:
  - Reads from: FreeEnergyEngine, DriveEngine, WorldState
  - Writes to: CognitiveHeartbeat (modifies sleep interval)
  - Reports to: consciousness stack via ServiceContainer
"""

import logging
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.TimeDilation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TICK_HZ = 1.0       # Normal conversation rate
_MAX_TICK_HZ = 10.0       # Maximum acceleration (user waiting, high surprise)
_MIN_TICK_HZ = 0.2        # Minimum deceleration (idle, resource pressure)
_IDLE_THRESHOLD_S = 300.0  # 5 minutes before idle deceleration kicks in
_EMA_ALPHA = 0.15         # Smoothing factor for tick rate changes
_HISTORY_SIZE = 300        # 5 minutes of dilation history


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DilationState:
    """Snapshot of the current time dilation state."""
    tick_rate_hz: float              # Current tick rate in Hz
    dilation_factor: float           # >1 = sped up, <1 = slowed, 1.0 = normal
    wall_elapsed_s: float            # Wall-clock seconds since boot
    subjective_elapsed_s: float      # Subjective seconds since boot
    acceleration_reason: str         # Why we're at this rate
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Acceleration / Deceleration Signal Evaluators
# ---------------------------------------------------------------------------

@dataclass
class _DilationSignals:
    """Collected signals that drive tick rate decisions."""
    prediction_error: float = 0.0       # 0-1, from FreeEnergyEngine
    drive_urgency: float = 0.0          # 0-1, from DriveEngine
    user_waiting: bool = False          # True if user sent message recently
    user_idle_s: float = 0.0            # Seconds since last user interaction
    thermal_pressure: float = 0.0       # 0-1, from WorldState
    memory_pressure: float = 0.0        # 0-100%, from WorldState
    free_energy: float = 0.0            # 0-1, from FreeEnergyEngine
    fe_distressed: bool = False         # True if FE > 0.7
    boredom_level: float = 0.0          # 0-1, from DriveEngine
    critical_maintenance: bool = False  # True if body integrity low


# ---------------------------------------------------------------------------
# TimeDilationEngine
# ---------------------------------------------------------------------------

class TimeDilationEngine:
    """Controls Aura's cognitive tick rate dynamically.

    The engine evaluates acceleration and deceleration pressures every
    time it is consulted, producing a smoothed tick rate that the
    CognitiveHeartbeat uses as its sleep interval.

    Thread-safe: all state mutations are under a lock since the heartbeat
    loop and telemetry readers may access concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Current state
        self._current_hz: float = _BASE_TICK_HZ
        self._smoothed_hz: float = _BASE_TICK_HZ
        self._dilation_factor: float = 1.0
        self._acceleration_reason: str = "baseline"

        # Subjective time accounting
        self._boot_time: float = time.time()
        self._last_tick_time: float = self._boot_time
        self._subjective_elapsed: float = 0.0  # Accumulated subjective seconds

        # History for trend analysis
        self._history: Deque[DilationState] = deque(maxlen=_HISTORY_SIZE)

        # Cached signals (updated each evaluation)
        self._last_signals: Optional[_DilationSignals] = None

        logger.info(
            "TimeDilationEngine initialized (base=%.1fHz, range=[%.1f, %.1f]Hz)",
            _BASE_TICK_HZ, _MIN_TICK_HZ, _MAX_TICK_HZ,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_tick_rate(self) -> float:
        """Current cognitive tick rate in Hz."""
        with self._lock:
            return self._smoothed_hz

    def get_dilation_factor(self) -> float:
        """Dilation factor: >1 = sped up, <1 = slowed, 1.0 = normal."""
        with self._lock:
            return self._dilation_factor

    def get_subjective_elapsed(self, since_timestamp: float) -> float:
        """Compute subjective seconds elapsed since a given wall-clock timestamp.

        Subjective time runs faster when dilation > 1 and slower when < 1.
        For timestamps before the engine booted, returns wall-clock delta.
        """
        with self._lock:
            wall_delta = time.time() - since_timestamp
            if since_timestamp <= self._boot_time:
                # For the entire engine lifetime, use accumulated subjective time
                return self._subjective_elapsed
            # Approximate: scale wall delta by current dilation factor
            # (A perfect integral would require the full history, but the
            # EMA-smoothed factor is a good approximation for recent spans.)
            return wall_delta * self._dilation_factor

    def get_interval(self) -> float:
        """Sleep interval in seconds for the heartbeat loop."""
        with self._lock:
            return 1.0 / max(0.01, self._smoothed_hz)

    def get_snapshot(self) -> Dict:
        """Full snapshot for telemetry / consciousness stack."""
        with self._lock:
            return {
                "tick_rate_hz": round(self._smoothed_hz, 3),
                "dilation_factor": round(self._dilation_factor, 3),
                "wall_elapsed_s": round(time.time() - self._boot_time, 1),
                "subjective_elapsed_s": round(self._subjective_elapsed, 1),
                "reason": self._acceleration_reason,
                "base_hz": _BASE_TICK_HZ,
                "min_hz": _MIN_TICK_HZ,
                "max_hz": _MAX_TICK_HZ,
            }

    def get_context_block(self) -> str:
        """Returns a concise context block for LLM prompt injection."""
        with self._lock:
            if abs(self._dilation_factor - 1.0) < 0.05:
                return ""  # Near-normal: no need to mention
            label = "accelerated" if self._dilation_factor > 1.0 else "decelerated"
            return (
                f"## SUBJECTIVE TIME\n"
                f"Cognitive clock {label} ({self._dilation_factor:.1f}x) | "
                f"Tick rate: {self._smoothed_hz:.1f}Hz | "
                f"Reason: {self._acceleration_reason}"
            )

    # ------------------------------------------------------------------
    # Core evaluation (called by heartbeat each tick)
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        """Evaluate current conditions and return the new tick interval.

        This is the main entry point called by CognitiveHeartbeat at the
        end of each tick. It:
          1. Gathers signals from subsystems
          2. Computes acceleration and deceleration pressures
          3. Blends them into a target Hz
          4. EMA-smooths to avoid jitter
          5. Updates subjective time accounting
          6. Returns the new sleep interval in seconds
        """
        signals = self._gather_signals()
        target_hz = self._compute_target_hz(signals)

        now = time.time()
        with self._lock:
            # EMA smoothing
            self._smoothed_hz = (
                _EMA_ALPHA * target_hz + (1 - _EMA_ALPHA) * self._smoothed_hz
            )
            # Clamp
            self._smoothed_hz = max(_MIN_TICK_HZ, min(_MAX_TICK_HZ, self._smoothed_hz))

            # Update dilation factor
            self._dilation_factor = self._smoothed_hz / _BASE_TICK_HZ

            # Subjective time accounting
            wall_delta = now - self._last_tick_time
            self._subjective_elapsed += wall_delta * self._dilation_factor
            self._last_tick_time = now

            # Cache signals
            self._last_signals = signals

            # Record history
            state = DilationState(
                tick_rate_hz=self._smoothed_hz,
                dilation_factor=self._dilation_factor,
                wall_elapsed_s=now - self._boot_time,
                subjective_elapsed_s=self._subjective_elapsed,
                acceleration_reason=self._acceleration_reason,
                timestamp=now,
            )
            self._history.append(state)

            interval = 1.0 / max(0.01, self._smoothed_hz)

        # Log significant changes (avoid spam: only on notable shifts)
        if len(self._history) % 30 == 0 or abs(self._dilation_factor - 1.0) > 0.3:
            logger.debug(
                "TimeDilation: %.2fHz (%.1fx) reason=%s",
                self._smoothed_hz, self._dilation_factor, self._acceleration_reason,
            )

        return interval

    # ------------------------------------------------------------------
    # Signal gathering
    # ------------------------------------------------------------------

    def _gather_signals(self) -> _DilationSignals:
        """Collect signals from all relevant subsystems. Fault-tolerant."""
        signals = _DilationSignals()

        # Free Energy Engine
        try:
            fe = ServiceContainer.get("free_energy_engine", default=None)
            if fe and fe.current:
                signals.free_energy = fe.current.free_energy
                signals.prediction_error = fe.current.surprise
                signals.fe_distressed = fe.is_distressed()
        except Exception:
            pass

        # Drive Engine
        try:
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive:
                vector = drive.get_drive_vector()
                # Urgency = inverse of lowest drive level
                min_level = min(
                    (v for k, v in vector.items() if k != "uptime_value"),
                    default=0.5,
                )
                signals.drive_urgency = max(0.0, 1.0 - min_level)
                signals.boredom_level = drive.boredom_level
        except Exception:
            pass

        # World State
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws:
                signals.user_idle_s = ws.user_idle_seconds
                signals.user_waiting = ws.user_idle_seconds < 10.0 and ws.user_message_count > 0
                signals.thermal_pressure = ws.thermal_pressure
                signals.memory_pressure = ws.memory_percent
        except Exception:
            pass

        # Embodiment / body integrity
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis:
                mods = homeostasis.get_modifiers()
                if hasattr(mods, "urgency_flag") and mods.urgency_flag:
                    signals.critical_maintenance = True
                if hasattr(mods, "overall_vitality") and mods.overall_vitality < 0.3:
                    signals.critical_maintenance = True
        except Exception:
            pass

        return signals

    # ------------------------------------------------------------------
    # Target Hz computation
    # ------------------------------------------------------------------

    def _compute_target_hz(self, s: _DilationSignals) -> float:
        """Compute the target tick rate from gathered signals.

        Strategy:
          - Start at base rate (1Hz)
          - Compute acceleration pressure (0-1) from urgency signals
          - Compute deceleration pressure (0-1) from idle/resource signals
          - Net pressure determines direction and magnitude
          - Map to [_MIN_TICK_HZ, _MAX_TICK_HZ] range
        """
        # --- Acceleration pressures (want to speed up) ---
        accel_factors = []
        reason_parts = []

        # User waiting for response (strongest accelerator)
        if s.user_waiting:
            accel_factors.append(0.8)
            reason_parts.append("user_waiting")

        # High prediction error (something surprising)
        if s.prediction_error > 0.4:
            accel_factors.append(s.prediction_error * 0.7)
            reason_parts.append(f"surprise({s.prediction_error:.2f})")

        # High free energy / distress
        if s.fe_distressed:
            accel_factors.append(0.6)
            reason_parts.append("fe_distress")

        # High drive urgency
        if s.drive_urgency > 0.6:
            accel_factors.append(s.drive_urgency * 0.5)
            reason_parts.append(f"drive_urgent({s.drive_urgency:.2f})")

        # Critical self-maintenance
        if s.critical_maintenance:
            accel_factors.append(0.7)
            reason_parts.append("critical_maintenance")

        # --- Deceleration pressures (want to slow down) ---
        decel_factors = []

        # Idle for > 5 minutes
        if s.user_idle_s > _IDLE_THRESHOLD_S:
            # Progressively slower the longer idle (up to 30 min plateau)
            idle_factor = min(1.0, (s.user_idle_s - _IDLE_THRESHOLD_S) / 1500.0)
            decel_factors.append(idle_factor)
            reason_parts.append(f"idle({s.user_idle_s:.0f}s)")

        # Low prediction error (nothing happening)
        if s.prediction_error < 0.1 and s.free_energy < 0.2:
            decel_factors.append(0.4)
            reason_parts.append("low_surprise")

        # Resource pressure (thermal)
        if s.thermal_pressure > 0.5:
            decel_factors.append(s.thermal_pressure * 0.6)
            reason_parts.append(f"thermal({s.thermal_pressure:.2f})")

        # Resource pressure (memory)
        if s.memory_pressure > 80:
            mem_factor = min(1.0, (s.memory_pressure - 80) / 15.0)
            decel_factors.append(mem_factor * 0.5)
            reason_parts.append(f"memory({s.memory_pressure:.0f}%)")

        # High boredom (conserve resources when nothing is interesting)
        if s.boredom_level > 0.7:
            decel_factors.append(s.boredom_level * 0.3)
            reason_parts.append(f"boredom({s.boredom_level:.2f})")

        # --- Blend ---
        accel = max(accel_factors) if accel_factors else 0.0
        decel = max(decel_factors) if decel_factors else 0.0

        # Net pressure: positive = accelerate, negative = decelerate
        net = accel - decel

        if net > 0.05:
            # Accelerate: map net (0, 1) -> (BASE, MAX)
            target = _BASE_TICK_HZ + net * (_MAX_TICK_HZ - _BASE_TICK_HZ)
            self._acceleration_reason = "accel:" + "+".join(reason_parts) if reason_parts else "accel"
        elif net < -0.05:
            # Decelerate: map net (-1, 0) -> (MIN, BASE)
            target = _BASE_TICK_HZ + net * (_BASE_TICK_HZ - _MIN_TICK_HZ)
            self._acceleration_reason = "decel:" + "+".join(reason_parts) if reason_parts else "decel"
        else:
            target = _BASE_TICK_HZ
            self._acceleration_reason = "baseline"

        return max(_MIN_TICK_HZ, min(_MAX_TICK_HZ, target))

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def get_trend(self) -> str:
        """Return the recent trend: 'accelerating', 'decelerating', or 'stable'."""
        with self._lock:
            if len(self._history) < 10:
                return "stable"
            recent = [s.tick_rate_hz for s in list(self._history)[-10:]]
        slope = recent[-1] - recent[0]
        if slope > 0.3:
            return "accelerating"
        if slope < -0.3:
            return "decelerating"
        return "stable"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[TimeDilationEngine] = None
_engine_lock = threading.Lock()


def get_time_dilation_engine() -> TimeDilationEngine:
    """Get or create the singleton TimeDilationEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TimeDilationEngine()
    return _engine
