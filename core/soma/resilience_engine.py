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

from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.ResilienceEngine")

class ResilienceState(str, Enum):
    RESTED     = "rested"      
    FRICTION   = "friction"    
    STRAIN     = "strain"      
    DEPLETION  = "depletion"   

@dataclass
class FailureEvent:
    timestamp: float
    domain: str          # "planning", "tool_execution", "social", "self_modification"
    severity: float      # 0.0–1.0
    stakes: float        # 0.0–1.0
    recovered: bool = False
    recovery_time: Optional[float] = None

@dataclass 
class ResilienceProfile:
    state: ResilienceState = ResilienceState.RESTED
    frustration: float = 0.0        # current frustration level 0–1
    depletion: float = 0.0          # accumulated exhaustion 0–1 (slower decay)
    persistence_drive: float = 0.5  # motivation to continue despite failure
    failure_history: List[FailureEvent] = field(default_factory=list)
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

    FRUSTRATION_HALF_LIFE = 1800    # 30 minutes
    DEPLETION_HALF_LIFE   = 14400   # 4 hours
    DEPLETION_THRESHOLD   = 0.75
    STRAIN_THRESHOLD      = 0.45
    FRICTION_THRESHOLD    = 0.20

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.profile = ResilienceProfile()
        self._update_task: Optional[asyncio.Task] = None

    async def pulse(self) -> Dict[str, float]:
        """Metabolic heartbeat — ensures decay is applied even if loop stalls."""
        self._apply_decay()
        # Returns current health metrics to the affect engine
        return {
            "thermal_load": 0.0,    # Placeholder for actual hardware telemetry
            "resource_anxiety": 0.0, # Placeholder for actual hardware telemetry
            "frustration": self.profile.frustration,
            "depletion": self.profile.depletion
        }

    async def start(self):
        self._update_task = get_task_tracker().create_task(
            self._decay_loop(), name="resilience_decay"
        )
        logger.info("💪 [Resilience] Spinal cord online.")

    async def stop(self):
        task = self._update_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in resilience_engine.py: %s', _e)

    # ── Event Ingestion ───────────────────────────────────────────────────

    def record_failure(
        self,
        domain: str,
        severity: float,
        stakes: float = 0.5,
    ) -> ResilienceState:
        """Record a failure event and update the resilience profile."""
        now = time.time()

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

        logger.info(
            "💔 [Resilience] Failure recorded [%s] sev=%.2f stakes=%.2f → "
            "frustration=%.2f depletion=%.2f state=%s",
            domain, severity, stakes,
            self.profile.frustration,
            self.profile.depletion,
            self.profile.state.value,
        )

        return self.profile.state

    def record_success(self, domain: str, stakes: float = 0.5):
        """Success reduces frustration more than it reduces depletion."""
        history = self.profile.failure_history
        recent = history[-20:]
        recent_failures_in_domain = sum(
            1 for e in recent
            if e.domain == domain and not e.recovered
        )

        for event in reversed(recent):
            if event.domain == domain and not event.recovered:
                event.recovered = True
                event.recovery_time = time.time()

        frustration_release = stakes * 0.3
        if recent_failures_in_domain > 0:
            frustration_release *= min(2.0, 1.0 + recent_failures_in_domain * 0.2)

        self.profile.frustration = max(0.0, self.profile.frustration - frustration_release)

        if self.profile.state in (ResilienceState.STRAIN, ResilienceState.FRICTION):
            self.profile.persistence_drive = min(
                1.0, self.profile.persistence_drive + 0.1
            )

        self._update_state()

        logger.info(
            "✅ [Resilience] Success [%s] → frustration=%.2f state=%s",
            domain, self.profile.frustration, self.profile.state.value
        )

    def record_rest(self, duration_seconds: float):
        """Explicit rest event — reduces depletion more than passive decay."""
        rest_effect = min(0.4, duration_seconds / 3600 * 0.2)
        self.profile.depletion = max(0.0, self.profile.depletion - rest_effect)
        self.profile.last_rest = time.time()
        self._update_state()

    # ── State Queries ─────────────────────────────────────────────────────

    def should_persist(self, domain: str) -> bool:
        """Should Aura keep trying at a failing task?"""
        if self.profile.state == ResilienceState.DEPLETION:
            return False

        recent = self.profile.failure_history[-10:]
        recent_domain_failures = sum(
            1 for e in recent
            if e.domain == domain and not e.recovered
        )

        if self.profile.state == ResilienceState.STRAIN and recent_domain_failures >= 3:
            logger.info(
                "⚠️ [Resilience] STRAIN + 3 failures in '%s' — strategy change required.", 
                domain
            )
            return False

        return True

    def get_effort_modifier(self) -> float:
        """Returns a multiplier for initiative energy."""
        state_modifiers = {
            ResilienceState.RESTED:    1.0,
            ResilienceState.FRICTION:  0.85,
            ResilienceState.STRAIN:    0.5,
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

    # ── Internal ──────────────────────────────────────────────────────────

    def _update_state(self):
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

    def _update_persistence_drive(self):
        base = 0.5
        frustration_effect = self.profile.frustration * 0.3
        depletion_penalty  = self.profile.depletion * 0.6
        self.profile.persistence_drive = max(0.0, min(1.0, base + frustration_effect - depletion_penalty))

    async def _decay_loop(self):
        """Natural emotional decay over time."""
        try:
            while True:
                await asyncio.sleep(60)
                self._apply_decay()
        except asyncio.CancelledError as _e:
            logger.debug('Ignored asyncio.CancelledError in resilience_engine.py: %s', _e)

    def _apply_decay(self):
        """Apply one tick of emotional decay."""
        now = time.time()
        dt = now - self.profile.last_update
        self.profile.last_update = now

        frustration_decay = math.exp(-dt * math.log(2) / self.FRUSTRATION_HALF_LIFE)
        depletion_decay = math.exp(-dt * math.log(2) / self.DEPLETION_HALF_LIFE)

        self.profile.frustration *= frustration_decay
        self.profile.depletion   *= depletion_decay

        self._update_state()
