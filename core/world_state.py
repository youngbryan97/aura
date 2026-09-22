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

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import has_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.WorldState")

_ALLOWED_USER_MOODS = frozenset(
    {"unknown", "positive", "neutral", "negative", "frustrated", "focused", "tired"}
)


def _now() -> float:
    """Read the live wall clock without binding to a patched time function."""
    return time.time()


def _finite(value: Any, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    """Coerce to a finite float in [lo, hi], falling back to default."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SalientEvent:
    """Something that happened in the environment worth noticing."""
    description: str
    source: str              # "system", "user", "perception", "terminal"
    salience: float = 0.5    # 0-1, how important
    timestamp: float = field(default_factory=_now)
    ttl: float = 3600.0      # expires after 1h by default
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl


@dataclass
class GameState:
    """Structured representation of a game or complex environment."""
    game_id: str = "nethack"
    active: bool = False
    hp: int = 0
    hp_max: int = 0
    hunger: str = "neutral"
    level: int = 1
    coordinates: tuple[int, int] = (0, 0)
    inventory: list[dict[str, Any]] = field(default_factory=list)
    map_hash: str = ""
    last_action: str = ""
    salient_changes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=_now)


@dataclass
class EnvironmentBelief:
    """A standing belief about the environment."""
    key: str                 # "user_mood", "time_of_day", etc.
    value: Any
    confidence: float = 0.7
    source: str = "inferred"
    updated_at: float = field(default_factory=_now)
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
        # One lock guards the many concurrent producers (perception, terminal
        # monitor, user tracking, telemetry, game updates) that mutate the
        # shared fields, deque, and belief dict.
        self._state_lock = threading.RLock()

        # User state
        self.last_user_interaction: float = 0.0
        self.user_idle_seconds: float = 0.0
        self.user_message_count: int = 0
        self.estimated_user_mood: str = "unknown"  # positive, neutral, negative, frustrated

        # System telemetry. `_telemetry_measured` distinguishes a real reading
        # from the optimistic defaults below (battery full/charging, zero CPU/
        # thermal) so consumers never mistake an unobserved host for a healthy
        # idle one.
        self._telemetry_measured: bool = False
        self._thermal_measured: bool = False
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

        # ── Perceptual grounding (fed by PerceptualPump at 10 Hz) ────
        self.active_foreground_app: str = ""       # currently focused application
        self.active_window_title: str = ""         # window title of focused app
        self.screen_content_hash: str = ""         # hash of OCR text (change detection)
        self.ambient_audio_level: float = 0.0      # 0-1 mic RMS energy
        self.voice_activity_detected: bool = False  # VAD flag
        self.last_voice_activity_at: float = 0.0     # wall-clock timestamp for non-transcribed voice activity
        self.last_voice_transcript: str = ""       # most recent speech snippet
        self.last_voice_transcript_at: float = 0.0  # wall-clock timestamp for recency checks
        self.last_audio_source_assessment: dict[str, Any] = {}
        self.installed_apps: list[str] = []        # discovered installed applications
        self.automation_permissions: dict[str, bool] = {
            "accessibility": False,
            "screen_recording": False,
            "microphone": False,
            "camera": False,
            "full_disk_access": False,
        }

        # Event queue (salient changes)
        self._events: deque[SalientEvent] = deque(maxlen=self._MAX_EVENTS)

        # Standing beliefs
        self._beliefs: dict[str, EnvironmentBelief] = {}

        # Game/Stress-test state
        self.game_state = GameState()

        # Timing
        self._boot_time = time.time()
        self._last_telemetry_update: float = 0.0
        self._telemetry_interval: float = 10.0  # update every 10s

        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        register_runtime_service("world_state", self, required=False)
        self._update_time_of_day()
        self._started = True
        logger.info("WorldState ONLINE -- live perceptual feed active")

    # ------------------------------------------------------------------
    # Telemetry update (called periodically)
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Pull fresh telemetry from system. Fast, sync, no LLM."""
        now = time.time()
        with self._state_lock:
            if (now - self._last_telemetry_update) < self._telemetry_interval:
                return
            self._last_telemetry_update = now

            # User idle time (clamp: wall-clock rollback must not make idle/
            # session durations negative).
            if self.last_user_interaction > 0:
                self.user_idle_seconds = max(0.0, now - self.last_user_interaction)
            self.session_duration_s = max(0.0, now - self._boot_time)

            # System telemetry via psutil
            try:
                from core.runtime import resource_psutil as psutil
                self.cpu_percent = _finite(psutil.cpu_percent(interval=0), self.cpu_percent, lo=0.0, hi=100.0)
                self.memory_percent = _finite(psutil.virtual_memory().percent, self.memory_percent, lo=0.0, hi=100.0)
                battery = psutil.sensors_battery()
                if battery:
                    self.battery_percent = _finite(battery.percent, self.battery_percent, lo=0.0, hi=100.0)
                    self.battery_charging = bool(battery.power_plugged) if battery.power_plugged is not None else self.battery_charging
                self._telemetry_measured = True
                # Thermal pressure through psutil when available, with substrate fallback.
                try:
                    temps = psutil.sensors_temperatures()
                except (AttributeError, OSError, RuntimeError, ValueError) as exc:
                    logger.debug(
                        "no temperature sensors, so the substrate fallback decides thermal pressure (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                    temps = None
                if temps:
                    max_temp = max(t.current for sensors in temps.values() for t in sensors)
                    self.thermal_pressure = min(1.0, max(0.0, (max_temp - 60) / 40))
                    self._thermal_measured = True
                else:
                    try:
                        from core.resilience.substrate_monitor import SubstrateMonitor
                        _level, pressure, _source = SubstrateMonitor().thermal()
                        self.thermal_pressure = _finite(pressure, self.thermal_pressure, lo=0.0, hi=1.0)
                        self._thermal_measured = True
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as thermal_exc:
                        # Unknown thermal state must NOT be reported as zero
                        # pressure (a safety signal resource admission relies
                        # on) — retain the last reading and mark it stale.
                        self._thermal_measured = False
                        record_degradation('world_state', thermal_exc)
            except ImportError as import_exc:
                # Missing psutil leaves telemetry indistinguishable from a
                # healthy idle host unless we record it and mark unmeasured.
                self._telemetry_measured = False
                self._thermal_measured = False
                record_degradation('world_state', import_exc)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
                self._telemetry_measured = False
                record_degradation('world_state', e)
                logger.debug("WorldState telemetry failed: %s", e)

            # Time of day
            self._update_time_of_day()

            # Auto-generate salient events from telemetry — only from REAL
            # measurements, so unobserved defaults never mint events.
            if self._telemetry_measured:
                if self.cpu_percent > 85:
                    self._add_event("System under heavy CPU load", "system", salience=0.6, ttl=300)
                if self.memory_percent > 85:
                    self._add_event("Memory pressure elevated", "system", salience=0.7, ttl=300)
            if self._thermal_measured and self.thermal_pressure > 0.7:
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

    def push_event(
        self,
        description: str,
        *,
        source: str = "system",
        salience: float = 0.5,
        ttl: float = 3600.0,
        metadata: dict[str, Any] | None = None,
        **extra_metadata: Any,
    ) -> None:
        """Compatibility event ingress for motor/somatic reflex producers."""
        payload = dict(metadata or {})
        payload.update(extra_metadata)
        self._add_event(description, source, salience, ttl, payload)

    def _add_event(self, description: str, source: str,
                   salience: float = 0.5, ttl: float = 3600.0,
                   metadata: dict | None = None) -> None:
        # Reject non-finite salience/TTL that would break sorting/expiry.
        salience = _finite(salience, 0.5, lo=0.0, hi=1.0)
        ttl = _finite(ttl, 3600.0, lo=1.0, hi=86400.0)
        with self._state_lock:
            # Dedup: don't add identical events within 60s
            for existing in self._events:
                if (existing.description == description and
                        existing.source == source and
                        (time.time() - existing.timestamp) < 60):
                    return
            # Store a shallow copy of metadata so a caller cannot mutate the
            # stored event's internal dict after the fact.
            self._events.append(SalientEvent(
                description=str(description)[:500], source=str(source)[:64],
                salience=salience, ttl=ttl,
                metadata=dict(metadata or {}),
            ))

    def get_salient_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most salient non-expired events."""
        with self._state_lock:
            self._evict_expired()
            events = sorted(self._events, key=lambda e: e.salience, reverse=True)
            # Return metadata COPIES so consumers cannot mutate internal event
            # state through the returned dicts.
            return [
                {
                    "description": e.description,
                    "source": e.source,
                    "salience": round(e.salience, 3),
                    "age_s": round(max(0.0, time.time() - e.timestamp), 1),
                    "metadata": dict(e.metadata),
                }
                for e in events[:limit]
            ]

    # ------------------------------------------------------------------
    # User tracking
    # ------------------------------------------------------------------

    def on_user_message(self, message: str = "", mood_hint: str = "") -> None:
        """Called when the user sends a message."""
        with self._state_lock:
            self.last_user_interaction = time.time()
            self.user_idle_seconds = 0.0
            self.user_message_count += 1
            # Only accept a mood hint from the recognized label set; free-form
            # values are not treated as truth.
            if mood_hint and str(mood_hint).strip().lower() in _ALLOWED_USER_MOODS:
                self.estimated_user_mood = str(mood_hint).strip().lower()

    def on_user_error(self, error_text: str = "") -> None:
        """Called when a user-relevant error is detected (e.g., terminal)."""
        self.record_event(
            f"User encountered error: {str(error_text)[:100]}",
            source="terminal", salience=0.8, ttl=1800,
        )
        # A late-night error is NOT evidence the user is frustrated. Record a
        # LOW-confidence, explicitly-labeled time-heuristic hypothesis instead
        # of asserting a mood with no linguistic/behavioral signal.
        if self.time_of_day in ("night", "late_night"):
            self.set_belief(
                "user_possibly_frustrated_late_night_error",
                True,
                confidence=0.25,
                source="time_of_day_heuristic",
                ttl=900.0,
            )

    # ------------------------------------------------------------------
    # Beliefs
    # ------------------------------------------------------------------

    def set_belief(self, key: str, value: Any, confidence: float = 0.7,
                   source: str = "inferred", ttl: float = 1800.0) -> None:
        with self._state_lock:
            self._beliefs[str(key)[:120]] = EnvironmentBelief(
                key=str(key)[:120], value=value,
                confidence=_finite(confidence, 0.7, lo=0.0, hi=1.0),
                source=str(source)[:64],
                # Allow short positive TTLs; only non-finite/negative are
                # rejected (negative clamps to 0 → immediately expired).
                ttl=_finite(ttl, 1800.0, lo=0.0, hi=86400.0),
            )

    def get_belief(self, key: str) -> Any | None:
        with self._state_lock:
            belief = self._beliefs.get(key)
            if belief and not belief.expired:
                return belief.value
            return None

    def get_belief_full(self, key: str) -> dict[str, Any] | None:
        """Return a belief with its provenance so callers can distinguish
        inference from observation instead of treating the bare value as fact."""
        with self._state_lock:
            belief = self._beliefs.get(key)
            if belief and not belief.expired:
                return {
                    "value": belief.value,
                    "confidence": round(belief.confidence, 3),
                    "source": belief.source,
                    "age_s": round(max(0.0, time.time() - belief.updated_at), 1),
                }
            return None

    # ------------------------------------------------------------------
    # Game State Updates
    # ------------------------------------------------------------------

    # Known game-state fields with their coercion, so arbitrary/invalid values
    # cannot poison the structure or make later arithmetic fail.
    _GAME_INT_FIELDS = ("hp", "hp_max", "level")
    _GAME_STR_FIELDS = ("game_id", "hunger", "last_action", "map_hash")

    def update_game_state(self, **kwargs) -> None:
        """Update the game state with new data points.

        Usage: world_state.update_game_state(hp=15, hunger="weak")
        """
        with self._state_lock:
            for key, value in kwargs.items():
                if not hasattr(self.game_state, key):
                    continue
                if key in self._GAME_INT_FIELDS:
                    try:
                        setattr(self.game_state, key, max(0, int(value)))
                    except (TypeError, ValueError):
                        continue
                elif key in self._GAME_STR_FIELDS:
                    setattr(self.game_state, key, str(value)[:200])
                elif key == "active":
                    setattr(self.game_state, key, bool(value))
                else:
                    setattr(self.game_state, key, value)

            self.game_state.updated_at = time.time()

            # If HP is low, trigger a salient event
            if self.game_state.hp > 0 and self.game_state.hp_max > 0:
                hp_percent = self.game_state.hp / self.game_state.hp_max
                if hp_percent <= 0.25:
                    self.record_event(
                        f"CRITICAL: Low health ({self.game_state.hp}/{self.game_state.hp_max})",
                        source="game", salience=0.95, ttl=300
                    )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
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
            # Freshness/availability so a consumer never mistakes optimistic
            # defaults for a real observation of a healthy host.
            "telemetry_measured": self._telemetry_measured,
            "thermal_measured": self._thermal_measured,
            "telemetry_age_s": round(max(0.0, time.time() - self._last_telemetry_update), 1) if self._last_telemetry_update else None,
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

_ws_instance: WorldState | None = None
_ws_instance_lock = threading.RLock()


def get_world_state() -> WorldState:
    global _ws_instance
    with _ws_instance_lock:
        if _ws_instance is None:
            _ws_instance = WorldState()
        instance = _ws_instance
        if not has_runtime_service("world_state"):
            register_runtime_service(
                "world_state",
                instance,
                required=False,
                owner="core/world_state.py",
                registered_by="core.world_state.get_world_state",
                required_for="live environment grounding",
                failure_policy="degrade_without_environment_initiative",
            )
        return instance


def reset_world_state_for_test() -> None:
    """Deactivate and forget the process singleton between hermetic tests."""
    global _ws_instance
    with _ws_instance_lock:
        instance, _ws_instance = _ws_instance, None
    if instance is not None:
        instance._started = False
