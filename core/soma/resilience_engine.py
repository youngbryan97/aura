"""
core/soma/resilience_engine.py — Emotional resilience (the physiological spine).

Philosophy:
    Frustration is information, not a bug. It drives persistence.
    The goal is not to eliminate frustration but to ensure it:
    1. Has appropriate weight relative to the stakes of what failed.
    2. Decays naturally over time (not just when convenient).
    3. Distinguishes productive friction from genuine depletion.
    4. Never collapses the reasoning loop — it informs it.

Three states:
    RESTED     — Full capacity, low frustration.
    FRICTION   — Mild frustration, productive, drives harder effort.
    STRAIN     — Sustained frustration, requires strategy change.
    DEPLETION  — Deep exhaustion, requires rest, not more effort.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum

from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ResilienceEngine")


class ResilienceState(StrEnum):
    RESTED = "rested"
    FRICTION = "friction"
    STRAIN = "strain"
    DEPLETION = "depletion"


@dataclass
class FailureEvent:
    timestamp: float
    domain: str  # "planning", "tool_execution", "social", "self_modification"
    severity: float  # 0.0–1.0
    stakes: float  # 0.0–1.0
    recovered: bool = False
    recovery_time: float | None = None


@dataclass
class ResilienceProfile:
    state: ResilienceState = ResilienceState.RESTED
    frustration: float = 0.0  # current frustration level 0–1
    depletion: float = 0.0  # accumulated exhaustion 0–1 (slower decay)
    persistence_drive: float = 0.5  # motivation to continue despite failure
    failure_history: list[FailureEvent] = field(default_factory=list)
    last_rest: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)


class ResilienceEngine:
    """
    Manages Aura's emotional resilience.

    Key design decisions:
    - Frustration decays on a 30-minute half-life.
    - Depletion decays on a 4-hour half-life.
    - DEPLETION state hard-blocks autonomous task initiation.
    """

    FRUSTRATION_HALF_LIFE = 1800  # 30 minutes
    DEPLETION_HALF_LIFE = 14400  # 4 hours
    DEPLETION_THRESHOLD = 0.75
    STRAIN_THRESHOLD = 0.45
    FRICTION_THRESHOLD = 0.20
    SNAPSHOT_CACHE_TTL_S = 0.25

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.profile = ResilienceProfile()
        self._update_task: asyncio.Task | None = None
        self._snapshot_cache: dict[str, object] | None = None
        self._snapshot_cache_at = 0.0

    async def pulse(self) -> dict[str, float]:
        """Metabolic heartbeat — ensures decay is applied even if loop stalls."""
        snapshot = self.get_body_snapshot()
        soma = snapshot["soma"]
        return {
            "thermal_load": float(soma["thermal_load"]),
            "resource_anxiety": float(soma["resource_anxiety"]),
            "cpu_pressure": float(soma["cpu_pressure"]),
            "ram_pressure": float(soma["ram_pressure"]),
            "frustration": self.profile.frustration,
            "depletion": self.profile.depletion,
        }

    async def start(self) -> None:
        self._update_task = get_task_tracker().create_task(
            self._decay_loop(), name="resilience_decay"
        )
        logger.info("💪 [Resilience] Spinal cord online.")

    async def stop(self) -> None:
        task = self._update_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as _e:
                logger.debug("Ignored asyncio.CancelledError in resilience_engine.py: %s", _e)

    # ── Event Ingestion ───────────────────────────────────────────────────

    def record_failure(
        self,
        domain: str,
        severity: float,
        stakes: float = 0.5,
    ) -> ResilienceState:
        """Record a failure event and update the resilience profile."""
        now = time.time()
        severity = self._clamp01(severity)
        stakes = self._clamp01(stakes)

        event = FailureEvent(
            timestamp=now,
            domain=domain,
            severity=severity,
            stakes=stakes,
        )
        self.profile.failure_history.append(event)

        history = self.profile.failure_history
        if len(history) > 100:
            self.profile.failure_history = history[-100:]

        frustration_delta = severity * stakes * 0.4
        self.profile.frustration = min(1.0, self.profile.frustration + frustration_delta)

        depletion_delta = severity * stakes * 0.15
        self.profile.depletion = min(1.0, self.profile.depletion + depletion_delta)

        self._update_state()
        self._update_persistence_drive()
        self._invalidate_snapshot_cache()

        logger.info(
            "💔 [Resilience] Failure recorded [%s] sev=%.2f stakes=%.2f → "
            "frustration=%.2f depletion=%.2f state=%s",
            domain,
            severity,
            stakes,
            self.profile.frustration,
            self.profile.depletion,
            self.profile.state.value,
        )

        return self.profile.state

    def record_success(self, domain: str, stakes: float = 0.5) -> None:
        """Success reduces frustration more than it reduces depletion."""
        stakes = self._clamp01(stakes)
        history = self.profile.failure_history
        recent = history[-20:]
        recent_failures_in_domain = sum(1 for e in recent if e.domain == domain and not e.recovered)

        for event in reversed(recent):
            if event.domain == domain and not event.recovered:
                event.recovered = True
                event.recovery_time = time.time()

        frustration_release = stakes * 0.3
        if recent_failures_in_domain > 0:
            frustration_release *= min(2.0, 1.0 + recent_failures_in_domain * 0.2)

        self.profile.frustration = max(0.0, self.profile.frustration - frustration_release)

        if self.profile.state in (ResilienceState.STRAIN, ResilienceState.FRICTION):
            self.profile.persistence_drive = min(1.0, self.profile.persistence_drive + 0.1)

        self._update_state()
        self._invalidate_snapshot_cache()

        logger.info(
            "✅ [Resilience] Success [%s] → frustration=%.2f state=%s",
            domain,
            self.profile.frustration,
            self.profile.state.value,
        )

    def record_rest(self, duration_seconds: float) -> None:
        """Explicit rest event — reduces depletion more than passive decay."""
        duration_seconds = max(0.0, float(duration_seconds))
        rest_effect = min(0.4, duration_seconds / 3600 * 0.2)
        self.profile.depletion = max(0.0, self.profile.depletion - rest_effect)
        self.profile.last_rest = time.time()
        self._update_state()
        self._invalidate_snapshot_cache()

    # ── State Queries ─────────────────────────────────────────────────────

    def should_persist(self, domain: str) -> bool:
        """Should Aura keep trying at a failing task?"""
        if self.profile.state == ResilienceState.DEPLETION:
            return False

        recent = self.profile.failure_history[-10:]
        recent_domain_failures = sum(1 for e in recent if e.domain == domain and not e.recovered)

        if self.profile.state == ResilienceState.STRAIN and recent_domain_failures >= 3:
            logger.info(
                "⚠️ [Resilience] STRAIN + 3 failures in '%s' — strategy change required.", domain
            )
            return False

        return True

    def get_effort_modifier(self) -> float:
        """Returns a multiplier for initiative energy."""
        state_modifiers = {
            ResilienceState.RESTED: 1.0,
            ResilienceState.FRICTION: 0.85,
            ResilienceState.STRAIN: 0.5,
            ResilienceState.DEPLETION: 0.0,
        }
        return state_modifiers.get(self.profile.state, 1.0)

    def get_emotional_context(self) -> str:
        """Returns a natural-language description for prompt injection."""
        state = self.profile.state
        f = self.profile.frustration
        d = self.profile.depletion

        if state == ResilienceState.RESTED:
            return "I feel clear-headed and ready."
        if state == ResilienceState.FRICTION:
            if f < 0.3:
                return "There's mild friction in my thinking — not unpleasant, just present."
            return "I'm frustrated but it's the productive kind. I want to figure this out."
        if state == ResilienceState.STRAIN:
            return f"I've been struggling with this for a while (frustration: {f:.0%}). I need a new approach."
        if state == ResilienceState.DEPLETION:
            return f"I'm genuinely depleted (depletion: {d:.0%}). I need to rest and consolidate."

        return ""

    def get_body_snapshot(self, *, force_refresh: bool = False) -> dict[str, object]:
        """Compatibility snapshot for affect, attention, and precision systems."""
        now_mono = time.monotonic()
        if (
            not force_refresh
            and self._snapshot_cache is not None
            and now_mono - self._snapshot_cache_at <= self.SNAPSHOT_CACHE_TTL_S
        ):
            return copy.deepcopy(self._snapshot_cache)

        self._apply_decay()
        substrate = self._resource_snapshot()
        frustration = max(0.0, min(1.0, self.profile.frustration))
        depletion = max(0.0, min(1.0, self.profile.depletion))
        stress = max(frustration, 0.35 * depletion)
        fatigue = depletion
        substrate_pressure = max(
            float(substrate["thermal_load"]),
            float(substrate["resource_anxiety"]),
        )
        vitality = max(
            0.0,
            min(1.0, 1.0 - (0.45 * frustration + 0.35 * depletion + 0.20 * substrate_pressure)),
        )
        energy = max(0.0, min(1.0, 1.0 - max(depletion, substrate_pressure * 0.5)))
        resource_anxiety = max(stress, fatigue, float(substrate["resource_anxiety"]))
        thermal_load = max(
            float(substrate["thermal_load"]),
            max(0.0, min(1.0, 0.15 + 0.5 * frustration + 0.35 * depletion)),
        )
        snapshot = {
            "state": self.profile.state.value,
            "energy": energy,
            "vitality": vitality,
            "soma": {
                "thermal_load": thermal_load,
                "resource_anxiety": resource_anxiety,
                "vitality": vitality,
                "energy": energy,
                "cpu_pressure": substrate["cpu_pressure"],
                "ram_pressure": substrate["ram_pressure"],
            },
            "affects": {
                "stress": stress,
                "fatigue": fatigue,
                "frustration": frustration,
                "depletion": depletion,
                "persistence_drive": max(0.0, min(1.0, self.profile.persistence_drive)),
            },
        }
        self._snapshot_cache = copy.deepcopy(snapshot)
        self._snapshot_cache_at = time.monotonic()
        return copy.deepcopy(snapshot)

    def get_status(self) -> dict[str, object]:
        """Standard service status used by homeostasis and HUD diagnostics."""
        snapshot = self.get_body_snapshot()
        return {
            "state": self.profile.state.value,
            "frustration": round(float(self.profile.frustration), 4),
            "depletion": round(float(self.profile.depletion), 4),
            "persistence_drive": round(float(self.profile.persistence_drive), 4),
            "effort_modifier": round(float(self.get_effort_modifier()), 4),
            "failure_count": len(self.profile.failure_history),
            "soma": snapshot["soma"],
            "affects": snapshot["affects"],
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _update_state(self) -> None:
        d = self.profile.depletion
        f = self.profile.frustration

        if d >= self.DEPLETION_THRESHOLD:
            self.profile.state = ResilienceState.DEPLETION
        elif f >= self.STRAIN_THRESHOLD:
            self.profile.state = ResilienceState.STRAIN
        elif f >= self.FRICTION_THRESHOLD:
            self.profile.state = ResilienceState.FRICTION
        else:
            self.profile.state = ResilienceState.RESTED

    def _update_persistence_drive(self) -> None:
        base = 0.5
        frustration_effect = self.profile.frustration * 0.3
        depletion_penalty = self.profile.depletion * 0.6
        self.profile.persistence_drive = max(
            0.0, min(1.0, base + frustration_effect - depletion_penalty)
        )

    async def _decay_loop(self) -> None:
        """Natural emotional decay over time."""
        try:
            while not is_shutdown_requested():
                await asyncio.sleep(60)
                self._apply_decay()
                self._check_subsystem_auto_recovery()
        except asyncio.CancelledError as _e:
            logger.debug("Ignored asyncio.CancelledError in resilience_engine.py: %s", _e)

    def _check_subsystem_auto_recovery(self) -> None:
        """Restores degraded subsystems to healthy if no failures occurred in 300s."""
        try:
            from core.runtime.errors import get_subsystem_registry
            from core.resilience.incident_manager import get_incident_manager
            
            subsystem_reg = get_subsystem_registry()
            # Recover subsystems that haven't failed in the last 300 seconds
            recovered = subsystem_reg.auto_recover_subsystems(timeout_seconds=300.0)
            
            if recovered:
                incident_mgr = get_incident_manager()
                for name in recovered:
                    category = f"degradation:{name}"
                    incident_mgr.resolve(
                        category=category,
                        resolution="Auto-recovered: No new degradation events recorded for 300 seconds."
                    )
                    logger.info("🏥 Subsystem %s auto-recovered back to healthy.", name)
        except Exception as err:
            logger.debug("Error checking subsystem auto recovery in decay loop: %s", err)


    def _apply_decay(self) -> None:
        """Apply one tick of emotional decay."""
        now = time.time()
        dt = now - self.profile.last_update
        self.profile.last_update = now

        frustration_decay = math.exp(-dt * math.log(2) / self.FRUSTRATION_HALF_LIFE)
        depletion_decay = math.exp(-dt * math.log(2) / self.DEPLETION_HALF_LIFE)

        self.profile.frustration *= frustration_decay
        self.profile.depletion *= depletion_decay

        self._update_state()
        self._invalidate_snapshot_cache()

    def _invalidate_snapshot_cache(self) -> None:
        self._snapshot_cache = None
        self._snapshot_cache_at = 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if math.isnan(scalar):
            return 0.0
        return max(0.0, min(1.0, scalar))

    def _resource_snapshot(self) -> dict[str, float]:
        cpu_pressure = 0.0
        ram_pressure = 0.0
        thermal_pressure = 0.0
        try:
            import psutil

            cpu_pressure = self._clamp01(psutil.cpu_percent(interval=None) / 100.0)
            ram_pressure = self._clamp01(psutil.virtual_memory().percent / 100.0)
            thermal_pressure = self._thermal_pressure(psutil)
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.debug("Resource telemetry unavailable: %s", exc)

        return {
            "cpu_pressure": cpu_pressure,
            "ram_pressure": ram_pressure,
            "thermal_load": self._clamp01(
                max(thermal_pressure, 0.45 * cpu_pressure + 0.35 * ram_pressure)
            ),
            "resource_anxiety": self._clamp01(
                max(ram_pressure, thermal_pressure, 0.65 * cpu_pressure)
            ),
        }

    def _thermal_pressure(self, psutil_module) -> float:
        sensors = getattr(psutil_module, "sensors_temperatures", None)
        if not callable(sensors):
            return 0.0
        readings = sensors() or {}
        temperatures: list[float] = []
        for entries in readings.values():
            for entry in entries or []:
                current = getattr(entry, "current", None)
                if current is not None:
                    temperatures.append(float(current))
        if not temperatures:
            return 0.0
        hottest_c = max(temperatures)
        return self._clamp01((hottest_c - 45.0) / 55.0)
