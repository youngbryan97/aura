"""core/world_state.py -- Live Perceptual World State
=====================================================
Separate from EpistemicState (knowledge graph). This tracks the LIVE
state of Aura's environment -- what is happening RIGHT NOW.

The WorldState holds:
  - User activity (last interaction, idle duration, estimated mood)
  - System telemetry (CPU, RAM, thermal, battery)
  - Environment facts (time of day, ambient context)
  - Salient event queue (recent changes worth noticing)
  - Standing beliefs about environment with TTLs

This feeds into initiative scoring: Aura acts based on what's
happening in the world, not just internal timers.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.WorldState")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SalientEvent:
    """Something that happened in the environment worth noticing."""
    description: str
    source: str              # "system", "user", "perception", "terminal"
    salience: float = 0.5    # 0-1, how important
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600.0      # expires after 1h by default
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


@dataclass
class EnvironmentBelief:
    """A standing belief about the environment."""
    key: str                 # "user_mood", "time_of_day", etc.
    value: Any
    confidence: float = 0.7
    source: str = "inferred"
    updated_at: float = field(default_factory=time.time)
    ttl: float = 1800.0      # 30 min default

    @property
    def expired(self) -> bool:
        return (time.time() - self.updated_at) > self.ttl


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

class WorldState:
    """Live perceptual world state.

    Updated continuously by:
      - System telemetry (psutil)
      - ContinuousPerceptionEngine
      - Terminal monitor
      - User interaction tracking

    Read by:
      - InitiativeSynthesizer (to generate environment-aware impulses)
      - InitiativeArbiter (to modulate social_appropriateness)
      - CognitiveKernel (to add world context to briefings)
    """

    _MAX_EVENTS = 50

    def __init__(self) -> None:
        # User state
        self.last_user_interaction: float = 0.0
        self.user_idle_seconds: float = 0.0
        self.user_message_count: int = 0
        self.estimated_user_mood: str = "unknown"  # positive, neutral, negative, frustrated

        # System telemetry
        self.cpu_percent: float = 0.0
        self.memory_percent: float = 0.0
        self.thermal_pressure: float = 0.0  # 0-1
        self.battery_percent: float = 100.0
        self.battery_charging: bool = True

        # Environment
        self.time_of_day: str = "unknown"  # morning, afternoon, evening, night, late_night
        self.session_duration_s: float = 0.0
        self.is_user_coding: bool = False
        self.active_app_context: str = ""

        # Event queue (salient changes)
        self._events: Deque[SalientEvent] = deque(maxlen=self._MAX_EVENTS)

        # Standing beliefs
        self._beliefs: Dict[str, EnvironmentBelief] = {}

        # Timing
        self._boot_time = time.time()
        self._last_telemetry_update: float = 0.0
        self._telemetry_interval: float = 10.0  # update every 10s

        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("world_state", self, required=False)
        self._update_time_of_day()
        self._started = True
        logger.info("WorldState ONLINE -- live perceptual feed active")

    # ------------------------------------------------------------------
    # Telemetry update (called periodically)
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Pull fresh telemetry from system. Fast, sync, no LLM."""
        now = time.time()
        if (now - self._last_telemetry_update) < self._telemetry_interval:
            return
        self._last_telemetry_update = now

        # User idle time
        if self.last_user_interaction > 0:
            self.user_idle_seconds = now - self.last_user_interaction
        self.session_duration_s = now - self._boot_time

        # System telemetry via psutil
        try:
            import psutil
            self.cpu_percent = psutil.cpu_percent(interval=0)
            self.memory_percent = psutil.virtual_memory().percent
            battery = psutil.sensors_battery()
            if battery:
                self.battery_percent = battery.percent
                self.battery_charging = battery.power_plugged or False
            # Thermal (macOS)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    max_temp = max(t.current for sensors in temps.values() for t in sensors)
                    self.thermal_pressure = min(1.0, max(0.0, (max_temp - 60) / 40))
            except (AttributeError, Exception):
                self.thermal_pressure = 0.0
        except ImportError:
            pass  # no-op: intentional
        except Exception as e:
            record_degradation('world_state', e)
            logger.debug("WorldState telemetry failed: %s", e)

        # Time of day
        self._update_time_of_day()

        # Auto-generate salient events from telemetry
        if self.cpu_percent > 85:
            self._add_event("System under heavy CPU load", "system", salience=0.6, ttl=300)
        if self.memory_percent > 85:
            self._add_event("Memory pressure elevated", "system", salience=0.7, ttl=300)
        if self.thermal_pressure > 0.7:
            self._add_event("Thermal pressure high", "system", salience=0.8, ttl=300)
        if self.user_idle_seconds > 3600:
            hours = int(self.user_idle_seconds / 3600)
            self._add_event(f"User has been idle for {hours}+ hours", "user", salience=0.4, ttl=600)

        # Evict expired events and beliefs
        self._evict_expired()

    def _update_time_of_day(self) -> None:
        hour = time.localtime().tm_hour
        if 5 <= hour < 12:
            self.time_of_day = "morning"
        elif 12 <= hour < 17:
            self.time_of_day = "afternoon"
        elif 17 <= hour < 21:
            self.time_of_day = "evening"
        elif 21 <= hour < 24:
            self.time_of_day = "night"
        else:
            self.time_of_day = "late_night"

    # ------------------------------------------------------------------
    # Event management
    # ------------------------------------------------------------------

    def record_event(self, description: str, source: str = "system",
                     salience: float = 0.5, ttl: float = 3600.0,
                     **metadata) -> None:
        """Record a salient environment event."""
        self._add_event(description, source, salience, ttl, metadata)

    def _add_event(self, description: str, source: str,
                   salience: float = 0.5, ttl: float = 3600.0,
                   metadata: Optional[Dict] = None) -> None:
        # Dedup: don't add identical events within 60s
        for existing in self._events:
            if (existing.description == description and
                    existing.source == source and
                    (time.time() - existing.timestamp) < 60):
                return
        self._events.append(SalientEvent(
            description=description, source=source,
            salience=salience, ttl=ttl,
            metadata=metadata or {},
        ))

    def get_salient_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most salient non-expired events."""
        self._evict_expired()
        events = sorted(self._events, key=lambda e: e.salience, reverse=True)
        return [
            {
                "description": e.description,
                "source": e.source,
                "salience": round(e.salience, 3),
                "age_s": round(time.time() - e.timestamp, 1),
                "metadata": e.metadata,
            }
            for e in events[:limit]
        ]

    # ------------------------------------------------------------------
    # User tracking
    # ------------------------------------------------------------------

    def on_user_message(self, message: str = "", mood_hint: str = "") -> None:
        """Called when the user sends a message."""
        self.last_user_interaction = time.time()
        self.user_idle_seconds = 0.0
        self.user_message_count += 1
        if mood_hint:
            self.estimated_user_mood = mood_hint

    def on_user_error(self, error_text: str = "") -> None:
        """Called when a user-relevant error is detected (e.g., terminal)."""
        self.record_event(
            f"User encountered error: {error_text[:100]}",
            source="terminal", salience=0.8, ttl=1800,
        )
        # Late night + error = likely frustrated
        if self.time_of_day in ("night", "late_night"):
            self.estimated_user_mood = "frustrated"
            self.set_belief("user_likely_frustrated", True, confidence=0.7, source="inference")

    # ------------------------------------------------------------------
    # Beliefs
    # ------------------------------------------------------------------

    def set_belief(self, key: str, value: Any, confidence: float = 0.7,
                   source: str = "inferred", ttl: float = 1800.0) -> None:
        self._beliefs[key] = EnvironmentBelief(
            key=key, value=value, confidence=confidence,
            source=source, ttl=ttl,
        )

    def get_belief(self, key: str) -> Optional[Any]:
        belief = self._beliefs.get(key)
        if belief and not belief.expired:
            return belief.value
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        self.update()
        return {
            "user_idle_s": round(self.user_idle_seconds, 1),
            "user_mood": self.estimated_user_mood,
            "user_messages": self.user_message_count,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "thermal_pressure": round(self.thermal_pressure, 3),
            "battery": round(self.battery_percent, 1),
            "time_of_day": self.time_of_day,
            "session_duration_m": round(self.session_duration_s / 60, 1),
            "salient_events": len(self._events),
            "beliefs": {k: v.value for k, v in self._beliefs.items() if not v.expired},
        }

    def get_context_summary(self) -> str:
        """One-line summary for injecting into cognitive context."""
        self.update()
        parts = [f"Time: {self.time_of_day}"]
        if self.user_idle_seconds > 300:
            parts.append(f"User idle {int(self.user_idle_seconds/60)}min")
        if self.estimated_user_mood != "unknown":
            parts.append(f"User mood: {self.estimated_user_mood}")
        if self.cpu_percent > 70:
            parts.append(f"CPU: {self.cpu_percent:.0f}%")
        if self.thermal_pressure > 0.5:
            parts.append("thermal pressure")
        events = [e for e in self._events if not e.expired and e.salience > 0.5]
        if events:
            parts.append(f"{len(events)} salient events")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        # Events
        self._events = deque(
            (e for e in self._events if not e.expired),
            maxlen=self._MAX_EVENTS,
        )
        # Beliefs
        expired_keys = [k for k, v in self._beliefs.items() if v.expired]
        for k in expired_keys:
            del self._beliefs[k]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ws_instance: Optional[WorldState] = None


def get_world_state() -> WorldState:
    global _ws_instance
    if _ws_instance is None:
        _ws_instance = WorldState()
    return _ws_instance
