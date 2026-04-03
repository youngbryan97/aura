"""
core/session_guardian.py
─────────────────────────
Manages session health and prevents the cascade failures that kill long
conversations. This is the missing layer between the orchestrator and
the subsystems.

What it does:
  1. Tracks per-session health metrics
  2. Gates expensive operations (consolidation, evolution, RL) on system health
  3. Detects when the LLM response pipeline has gone silent and recovers it
  4. Prevents the reply_queue race condition by tagging responses with session IDs
  5. Provides a clean "safe mode" that disables non-essential subsystems

Install: instantiate in orchestrator_boot.py after the orchestrator is created,
call guardian.attach(orchestrator). Then call guardian.start() in the run loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.SessionGuardian")


# ── Health Levels ─────────────────────────────────────────────────────────────

class HealthLevel(Enum):
    OPTIMAL = "optimal"       # All systems go
    NOMINAL = "nominal"       # Minor issues, all features active
    STRESSED = "stressed"     # RAM/CPU pressure, expensive ops gated
    DEGRADED = "degraded"     # Multiple failures, core-only mode
    CRITICAL = "critical"     # Emergency — only LLM + reply pipeline


HEALTH_THRESHOLDS = {
    "ram_stressed": 75.0,
    "ram_degraded": 85.0,
    "ram_critical": 92.0,
    "consecutive_empty_responses_stressed": 2,
    "consecutive_empty_responses_degraded": 5,
    "silence_timeout_seconds": 45.0,  # No response for this long = stall
}


# ── Session State ─────────────────────────────────────────────────────────────

@dataclass
class SessionMetrics:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: float = field(default_factory=time.time)
    message_count: int = 0
    successful_responses: int = 0
    empty_responses: int = 0
    consecutive_empty: int = 0
    last_successful_response_at: float = field(default_factory=time.time)
    last_message_at: float = 0.0
    health_level: HealthLevel = HealthLevel.OPTIMAL
    ram_pct: float = 0.0
    recovery_attempts: int = 0

    @property
    def success_rate(self) -> float:
        if self.message_count == 0:
            return 1.0
        return self.successful_responses / self.message_count

    @property
    def silence_duration(self) -> float:
        if self.last_message_at == 0:
            return 0.0
        return time.time() - max(self.last_message_at, self.last_successful_response_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "uptime_s": round(time.time() - self.started_at),
            "messages": self.message_count,
            "success_rate": round(self.success_rate, 2),
            "consecutive_empty": self.consecutive_empty,
            "health": self.health_level.value,
            "ram_pct": round(self.ram_pct, 1),
            "silence_s": round(self.silence_duration),
        }


# ── Gated Operation ───────────────────────────────────────────────────────────

@dataclass
class GatedOperation:
    """An operation that requires a minimum health level to run."""
    name: str
    min_health: HealthLevel
    fn: Callable
    cooldown_seconds: float = 0.0
    last_run: float = 0.0

    def can_run(self, current_health: HealthLevel) -> bool:
        levels = list(HealthLevel)
        current_idx = levels.index(current_health)
        required_idx = levels.index(self.min_health)
        # Lower index = better health (OPTIMAL=0, CRITICAL=4)
        if current_idx > required_idx:
            return False
        if self.cooldown_seconds > 0:
            if time.time() - self.last_run < self.cooldown_seconds:
                return False
        return True


# ── Main Guardian ─────────────────────────────────────────────────────────────

class SessionGuardian:
    """
    Attaches to the orchestrator and monitors session health.
    Gates expensive operations. Detects and recovers from stalls.
    """

    def __init__(self, safe_mode: bool = False):
        self.safe_mode = safe_mode
        self.metrics = SessionMetrics()
        self._orchestrator = None
        self._gated_ops: List[GatedOperation] = []
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._recovery_callbacks: List[Callable] = []

        # Register default gated operations
        self._register_defaults()
        logger.info(
            "SessionGuardian initialized (safe_mode=%s, session=%s)",
            safe_mode, self.metrics.session_id
        )

    def _register_defaults(self):
        """Register the expensive operations that should be gated."""
        # Memory consolidation — needs at least NOMINAL health
        self.register_gated_op(
            "memory_consolidation",
            min_health=HealthLevel.NOMINAL,
            cooldown_seconds=120.0,
        )
        # Persona evolution — needs OPTIMAL
        self.register_gated_op(
            "persona_evolution",
            min_health=HealthLevel.OPTIMAL,
            cooldown_seconds=300.0,
        )
        # RL training — needs OPTIMAL
        self.register_gated_op(
            "rl_training",
            min_health=HealthLevel.OPTIMAL,
            cooldown_seconds=600.0,
        )
        # Context pruning — needs NOMINAL (important but LLM-dependent)
        self.register_gated_op(
            "context_pruning",
            min_health=HealthLevel.NOMINAL,
            cooldown_seconds=60.0,
        )
        # Autonomous thought — needs NOMINAL
        self.register_gated_op(
            "autonomous_thought",
            min_health=HealthLevel.NOMINAL,
            cooldown_seconds=0.0,
        )

    def register_gated_op(
        self,
        name: str,
        min_health: HealthLevel = HealthLevel.NOMINAL,
        cooldown_seconds: float = 0.0,
        fn: Optional[Callable] = None,
    ):
        self._gated_ops.append(GatedOperation(
            name=name,
            min_health=min_health,
            fn=fn or (lambda: None),
            cooldown_seconds=cooldown_seconds,
        ))

    def attach(self, orchestrator) -> "SessionGuardian":
        """Attach to an orchestrator instance."""
        self._orchestrator = orchestrator
        logger.info("SessionGuardian attached to orchestrator")
        return self

    def add_recovery_callback(self, fn: Callable):
        """Register a function to call when stall is detected."""
        self._recovery_callbacks.append(fn)

    # ── Operation Gates ───────────────────────────────────────────────────────

    def can_run(self, operation_name: str) -> bool:
        """Check if an operation is allowed at current health level."""
        if self.safe_mode:
            # In safe mode, only core operations allowed
            allowed_in_safe = {"llm_call", "reply_pipeline", "user_message"}
            return operation_name in allowed_in_safe

        op = next((o for o in self._gated_ops if o.name == operation_name), None)
        if op is None:
            return True  # Unknown ops allowed by default
        result = op.can_run(self.metrics.health_level)
        if not result:
            logger.debug(
                "GATED: %s blocked (health=%s, required=%s)",
                operation_name, self.metrics.health_level.value, op.min_health.value
            )
        return result

    def mark_op_ran(self, operation_name: str):
        """Mark an operation as having run (for cooldown tracking)."""
        op = next((o for o in self._gated_ops if o.name == operation_name), None)
        if op:
            op.last_run = time.time()

    # ── Response Tracking ─────────────────────────────────────────────────────

    def on_message_sent(self, message: str):
        """Call when user sends a message."""
        self.metrics.message_count += 1
        self.metrics.last_message_at = time.time()

    def on_response_received(self, response: str, is_valid: bool):
        """Call when LLM response comes back."""
        if is_valid and response and response.strip():
            self.metrics.successful_responses += 1
            self.metrics.consecutive_empty = 0
            self.metrics.last_successful_response_at = time.time()
        else:
            self.metrics.empty_responses += 1
            self.metrics.consecutive_empty += 1
            logger.warning(
                "Empty/invalid response #%d (consecutive=%d)",
                self.metrics.empty_responses,
                self.metrics.consecutive_empty,
            )

        self._recalculate_health()

    def on_ram_update(self, ram_pct: float):
        """Call when RAM usage is updated (from SystemSoma telemetry)."""
        self.metrics.ram_pct = ram_pct
        self._recalculate_health()

    # ── Health Calculation ────────────────────────────────────────────────────

    def _recalculate_health(self):
        """Recalculate health level from current metrics."""
        old_level = self.metrics.health_level
        ram = self.metrics.ram_pct
        empty = self.metrics.consecutive_empty

        if (ram >= HEALTH_THRESHOLDS["ram_critical"] or
                empty >= HEALTH_THRESHOLDS["consecutive_empty_responses_degraded"] * 2):
            level = HealthLevel.CRITICAL
        elif (ram >= HEALTH_THRESHOLDS["ram_degraded"] or
              empty >= HEALTH_THRESHOLDS["consecutive_empty_responses_degraded"]):
            level = HealthLevel.DEGRADED
        elif (ram >= HEALTH_THRESHOLDS["ram_stressed"] or
              empty >= HEALTH_THRESHOLDS["consecutive_empty_responses_stressed"]):
            level = HealthLevel.STRESSED
        elif ram >= 60.0 or empty >= 1:
            level = HealthLevel.NOMINAL
        else:
            level = HealthLevel.OPTIMAL

        if level != old_level:
            logger.info(
                "Health level: %s → %s (RAM=%.1f%%, empty_streak=%d)",
                old_level.value, level.value, ram, empty
            )

        self.metrics.health_level = level

    # ── Stall Detection & Recovery ────────────────────────────────────────────

    async def _monitor_loop(self):
        """Background loop that detects stalls and triggers recovery."""
        logger.info("SessionGuardian monitor loop started")
        while self._running:
            try:
                await asyncio.sleep(10.0)
                await self._check_for_stall()
                self._log_status()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Monitor loop error: %s", exc)

    async def _check_for_stall(self):
        """Detect if the response pipeline has stalled and attempt recovery."""
        silence = self.metrics.silence_duration

        # Only check for stall if user has recently sent a message
        if self.metrics.last_message_at == 0:
            return
        if time.time() - self.metrics.last_message_at > 300:
            # User hasn't sent anything in 5 minutes — not a stall, just idle
            return

        if silence > HEALTH_THRESHOLDS["silence_timeout_seconds"]:
            logger.warning(
                "STALL DETECTED: No valid response for %.0fs (attempt %d)",
                silence, self.metrics.recovery_attempts + 1
            )
            self.metrics.recovery_attempts += 1
            await self._attempt_recovery()

    async def _attempt_recovery(self):
        """Try to recover from a stall."""
        if not self._orchestrator:
            return

        # Strategy 1: Reset consecutive empty counter to allow ops again
        self.metrics.consecutive_empty = 0
        self._recalculate_health()

        # Strategy 2: Call registered recovery callbacks
        for callback in self._recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as exc:
                logger.error("Recovery callback failed: %s", exc)

        # Strategy 3: If orchestrator has a retry mechanism, trigger it
        if hasattr(self._orchestrator, '_reconnect_cognitive_engine'):
            try:
                await self._orchestrator._reconnect_cognitive_engine()
                logger.info("Guardian triggered cognitive engine reconnect")
            except Exception as exc:
                logger.error("Cognitive engine reconnect failed: %s", exc)

        # Reset the silence timer so we don't spam recovery
        self.metrics.last_successful_response_at = time.time()

    def _log_status(self):
        if self.metrics.message_count % 10 == 0 and self.metrics.message_count > 0:
            logger.info("Session status: %s", self.metrics.to_dict())

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the background monitor loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="session_guardian"
        )
        logger.info("SessionGuardian started")

    def stop(self):
        """Stop the guardian."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    def get_status(self) -> Dict[str, Any]:
        return {
            "guardian": self.metrics.to_dict(),
            "gated_ops": [
                {
                    "name": op.name,
                    "min_health": op.min_health.value,
                    "can_run": op.can_run(self.metrics.health_level),
                    "last_run_ago_s": round(time.time() - op.last_run) if op.last_run else None,
                }
                for op in self._gated_ops
            ],
        }


# ── Global Instance ───────────────────────────────────────────────────────────

_guardian: Optional[SessionGuardian] = None


def get_guardian(safe_mode: bool = False) -> SessionGuardian:
    global _guardian
    if _guardian is None:
        _guardian = SessionGuardian(safe_mode=safe_mode)
    return _guardian


def reset_guardian():
    """Create a fresh guardian for a new session."""
    global _guardian
    _guardian = None
    return get_guardian()