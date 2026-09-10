"""core/perception/perceptual_pump.py — Continuous Perceptual Ingestion Loop
===========================================================================
The PRIMARY driver of Aura's substrate ODE and affect system.

This is NOT context for the LLM. This is the causal grounding loop.
Real-time perceptual input (screen, mic, system telemetry) perturbs the
continuous state vector at 10 Hz BEFORE the LLM ever runs.

The pump collects a PerceptualFrame every ~100ms from all modalities,
maps it into RuntimeBody fields, feeds it through the PhenomenalEngine
to produce an ExperienceState, and injects the result into the
ContinuousSubstrate ODE.

Architecture:
  sensors → PerceptualFrame → RuntimeBody → PhenomenalEngine.step()
           → ExperienceState → ContinuousSubstrate.inject_perceptual_frame()
           → WorldState.update_from_perceptual_frame()

Everything downstream (LLM prompts, initiative scoring, speech prosody)
reads from the substrate state summary — never from raw sensors.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.container import ServiceContainer
from core.perception.multimodal_sync import (
    Calibration,
    FusedPerceptualFrame,
    MissingReason,
    Modality,
    MultimodalSynchronizer,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)
from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task

if TYPE_CHECKING:
    from core.phenomenal_substrate.types import RuntimeBody

logger = logging.getLogger("Aura.PerceptualPump")

_PUMP_RUNTIME_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    asyncio.TimeoutError,
    # A hung osascript raises TimeoutExpired (a SubprocessError, not an
    # OSError) — it escaped every catch list and killed the whole pump
    # task during the 110GB incident. No sensor failure may end perception.
    subprocess.SubprocessError,
)

# After an AppleScript probe times out, skip subprocess-based screen
# probes for this long. A hung System Events under host load otherwise
# costs 2s of a worker thread every 500ms, forever.
def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


_SCREEN_PROBE_BACKOFF_S = _env_float(
    "AURA_PERCEPTUAL_SCREEN_PROBE_BACKOFF_S",
    60.0,
    minimum=5.0,
    maximum=300.0,
)
_SCREEN_PROBE_TIMEOUT_S = _env_float(
    "AURA_PERCEPTUAL_SCREEN_PROBE_TIMEOUT_S",
    0.75,
    minimum=0.25,
    maximum=2.0,
)
_LAST_SCREEN_PROBE_TIMEOUT_AT: float = 0.0


def _collect_native_screen_state() -> ScreenState | None:
    """Collect foreground app/window metadata without spawning AppleScript."""

    if sys.platform != "darwin":
        return None

    state = ScreenState()
    state.timestamp = time.time()
    state.monotonic_ns = time.monotonic_ns()
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is not None:
            name = app.localizedName()
            if name:
                state.active_app = str(name).strip()[:120]
    except _PUMP_RUNTIME_ERRORS as exc:
        logger.debug("Native frontmost app probe unavailable: %s", exc)

    try:
        import Quartz

        options = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
        for window in windows:
            if int(window.get("kCGWindowLayer", 0) or 0) != 0:
                continue
            owner = str(window.get("kCGWindowOwnerName") or "").strip()
            title = str(window.get("kCGWindowName") or "").strip()
            if owner and not state.active_app:
                state.active_app = owner[:120]
            if title:
                state.window_title = title[:200]
            if state.active_app or state.window_title:
                break
    except _PUMP_RUNTIME_ERRORS as exc:
        logger.debug("Native frontmost window probe unavailable: %s", exc)

    if state.active_app or state.window_title:
        state.available = True
        state.source = "macos_native_window_metadata"
        state.confidence = 0.95
        state.missing_reason = None
        return state
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreenState:
    """What is happening on the screen right now."""
    active_app: str = ""
    window_title: str = ""
    content_hash: str = ""          # hash of OCR text (change detection)
    content_snippet: str = ""       # first 200 chars of OCR text
    screen_changed: bool = False    # did the screen change since last frame?
    change_magnitude: float = 0.0   # 0-1, how much changed
    timestamp: float = field(default_factory=time.time)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    available: bool = False
    source: str = "screen_unavailable"
    confidence: float = 0.0
    missing_reason: MissingReason | None = MissingReason.UNAVAILABLE


@dataclass
class AudioState:
    """What is happening on the mic right now."""
    rms_energy: float = 0.0         # 0-1, ambient sound level
    voice_activity: bool = False    # is someone speaking?
    transcript_snippet: str = ""    # most recent speech (last ~5s, display-bounded)
    transcript_full: str = ""       # full utterance (command-fidelity, 4000-char bound)
    transcript_changed: bool = False
    timestamp: float = field(default_factory=time.time)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    available: bool = False
    source: str = "audio_unavailable"
    confidence: float = 0.0
    missing_reason: MissingReason | None = MissingReason.UNAVAILABLE


@dataclass
class SystemState:
    """System telemetry snapshot."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    thermal_pressure: float = 0.0   # 0-1
    battery_percent: float = 100.0
    battery_charging: bool = True
    disk_io_pressure: float = 0.0   # 0-1
    timestamp: float = field(default_factory=time.time)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    available: bool = False
    source: str = "system_unavailable"
    confidence: float = 0.0
    missing_reason: MissingReason | None = MissingReason.UNAVAILABLE


@dataclass
class UserState:
    """Inferred user state."""
    idle_seconds: float = 0.0
    last_interaction_type: str = ""  # "text", "voice", "mouse", "none"
    message_count: int = 0
    estimated_mood: str = "unknown"
    presence: float = 0.5           # 0 = absent, 1 = actively engaged
    timestamp: float = field(default_factory=time.time)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    available: bool = False
    source: str = "user_state_unavailable"
    confidence: float = 0.0
    missing_reason: MissingReason | None = MissingReason.UNAVAILABLE


@dataclass
class PerceptualFrame:
    """A single 100ms snapshot of all perceptual modalities.

    This is the fundamental unit of grounded perception.
    """
    frame_id: int = 0
    screen: ScreenState = field(default_factory=ScreenState)
    audio: AudioState = field(default_factory=AudioState)
    system: SystemState = field(default_factory=SystemState)
    user: UserState = field(default_factory=UserState)
    timestamp: float = field(default_factory=time.time)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    fusion: FusedPerceptualFrame | None = None

    def novelty_score(self) -> float:
        """How novel is this frame compared to baseline?"""
        screen_novelty = self.screen.change_magnitude
        audio_novelty = 0.3 if self.audio.voice_activity else 0.0
        system_novelty = max(0.0, (self.system.cpu_percent - 50) / 50)
        return min(1.0, 0.5 * screen_novelty + 0.3 * audio_novelty + 0.2 * system_novelty)

    def threat_score(self) -> float:
        """Perceived threat level from this frame."""
        thermal = self.system.thermal_pressure
        memory = max(0.0, (self.system.memory_percent - 80) / 20)
        battery_low = max(0.0, (20 - self.system.battery_percent) / 20) if not self.system.battery_charging else 0.0
        return min(1.0, 0.4 * thermal + 0.3 * memory + 0.3 * battery_low)

    def social_signal(self) -> float:
        """Social engagement signal from this frame."""
        voice = 0.6 if self.audio.voice_activity else 0.0
        presence = self.user.presence
        recency = max(0.0, 1.0 - self.user.idle_seconds / 300.0)  # decays over 5 min
        return min(1.0, 0.4 * voice + 0.3 * presence + 0.3 * recency)

    def epistemic_uncertainty(self) -> float:
        """Uncertainty produced by missing, stale, or contradictory evidence."""
        if self.fusion is None:
            return 0.5
        return self.fusion.uncertainty


# ---------------------------------------------------------------------------
# Sensor collectors — each one reads from available hardware/services
# ---------------------------------------------------------------------------

def _collect_screen_state(prev_hash: str) -> ScreenState:
    """Collect screen state from available sources.

    Tries (in order): ScreenObserver JSON, AppleScript, fallback.
    """
    state = _collect_native_screen_state() or ScreenState()
    now = time.time()
    state.timestamp = now
    state.monotonic_ns = time.monotonic_ns()
    if (state.active_app or state.window_title) and not state.available:
        state.available = True
        state.source = "native_window_metadata"
        state.confidence = 0.90
        state.missing_reason = None

    # 1. Get active app and window title via native macOS APIs first. AppleScript
    # is a governed fallback for hosts without PyObjC/Quartz, not the hot path.
    global _LAST_SCREEN_PROBE_TIMEOUT_AT
    probe_allowed = (now - _LAST_SCREEN_PROBE_TIMEOUT_AT) >= _SCREEN_PROBE_BACKOFF_S
    try:
        if probe_allowed and not state.active_app:
            from core.runtime.subprocess_gateway import get_subprocess_gateway
            gw = get_subprocess_gateway()

            # Active app name
            result = gw.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True, timeout=_SCREEN_PROBE_TIMEOUT_S, read_only=True, source="perceptual_pump.screen.app",
                accelerator_capability="none",
            )
            if result.returncode == 0 and result.stdout:
                state.active_app = result.stdout.strip()
                state.available = True
                state.source = "macos_applescript_window_metadata"
                state.confidence = 0.85
                state.missing_reason = None

            # Window title
            if state.active_app:
                result = gw.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to get name of front window of process "{state.active_app}"'],
                    capture_output=True, timeout=_SCREEN_PROBE_TIMEOUT_S, read_only=True, source="perceptual_pump.screen.title",
                    accelerator_capability="none",
                )
                if result.returncode == 0 and result.stdout:
                    state.window_title = result.stdout.strip()[:200]
    except subprocess.SubprocessError as e:
        _LAST_SCREEN_PROBE_TIMEOUT_AT = time.time()
        record_degradation(
            "perceptual_pump.screen",
            e,
            severity="warning",
            action=f"screen probe unavailable; backed off AppleScript for {_SCREEN_PROBE_BACKOFF_S:.0f}s",
            enforce_failure_policy=False,
        )
        logger.debug(
            "Screen probe timed out; backing off AppleScript for %.0fs: %s",
            _SCREEN_PROBE_BACKOFF_S,
            e,
        )
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
        record_degradation(
            "perceptual_pump.screen",
            e,
            severity="warning",
            action="screen probe unavailable; continuing without active-window telemetry",
            enforce_failure_policy=False,
        )
        logger.debug("Screen state AppleScript probe failed: %s", e)

    # 2. Screen content from ScreenObserver JSON (if vision service is running)
    try:
        import json
        from pathlib import Path
        vision_path = Path(__file__).resolve().parent.parent.parent / "sensory_vision.json"
        if vision_path.exists() and (now - vision_path.stat().st_mtime) < 15:
            data = json.loads(vision_path.read_text(encoding="utf-8"))
            text = str(data.get("text") or data.get("ocr_text") or "")
            if text:
                state.content_snippet = text[:200]
                state.content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
                state.available = True
                state.source = (
                    f"{state.source}+screen_observer_ocr"
                    if state.source != "screen_unavailable"
                    else "screen_observer_ocr"
                )
                state.confidence = max(state.confidence, 0.75)
                state.missing_reason = None
    except (OSError, ValueError, TypeError) as e:
        record_degradation("perceptual_pump.screen_ocr", e)

    # 3. Change detection
    if prev_hash and state.content_hash:
        state.screen_changed = state.content_hash != prev_hash
        state.change_magnitude = 1.0 if state.screen_changed else 0.0
    elif state.active_app:
        # Even without OCR, app switch counts as change
        state.screen_changed = True
        state.change_magnitude = 0.3

    return state


def _collect_audio_state() -> AudioState:
    """Collect audio state from available sources."""
    state = AudioState()
    state.timestamp = time.time()
    state.monotonic_ns = time.monotonic_ns()

    # Read from audio service JSON (if running)
    try:
        import json
        from pathlib import Path
        audio_path = Path(__file__).resolve().parent.parent.parent / "sensory_audio.json"
        if audio_path.exists() and (time.time() - audio_path.stat().st_mtime) < 10:
            data = json.loads(audio_path.read_text(encoding="utf-8"))
            state.available = True
            state.source = "sensory_audio_sidecar"
            state.confidence = 0.78
            state.missing_reason = None
            state.rms_energy = min(1.0, max(0.0, float(data.get("rms", data.get("energy", 0.0)) or 0.0)))
            state.voice_activity = bool(data.get("vad") or data.get("speech_detected") or state.rms_energy > 0.15)
            transcript = str(data.get("transcript") or data.get("text") or "")
            if transcript.strip():
                state.transcript_snippet = transcript.strip()[-200:]
                # Command fidelity: the wake-word lane consumes the full
                # utterance; the 200-char display snippet destroyed long
                # spoken commands when it overwrote last_voice_transcript.
                state.transcript_full = transcript.strip()[:4000]
                state.transcript_changed = True
    except (OSError, ValueError, TypeError) as e:
        record_degradation("perceptual_pump.audio", e)

    return state


def _collect_system_state() -> SystemState:
    """Collect system telemetry."""
    state = SystemState()
    state.timestamp = time.time()
    state.monotonic_ns = time.monotonic_ns()

    try:
        from core.runtime import resource_psutil as psutil
        state.cpu_percent = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        state.memory_percent = mem.percent
        state.available = True
        state.source = "resource_psutil"
        state.confidence = 0.98
        state.missing_reason = None
        battery = psutil.sensors_battery()
        if battery:
            state.battery_percent = battery.percent
            state.battery_charging = battery.power_plugged or False

        # Thermal
        try:
            temps = psutil.sensors_temperatures()
        except (AttributeError, OSError, RuntimeError, ValueError):
            temps = None
        if temps:
            max_temp = max(
                (t.current for sensors in temps.values() for t in sensors),
                default=0.0,
            )
            state.thermal_pressure = min(1.0, max(0.0, (max_temp - 60) / 40))
        else:
            # macOS fallback via SubstrateMonitor
            try:
                from core.resilience.substrate_monitor import SubstrateMonitor
                _level, pressure, _source = SubstrateMonitor().thermal()
                state.thermal_pressure = pressure
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                state.thermal_pressure = 0.0

        # Disk I/O
        try:
            disk = psutil.disk_io_counters()
            if disk:
                # Rough heuristic: busy_time / elapsed. Not perfect but directional.
                state.disk_io_pressure = min(1.0, max(0.0,
                    getattr(disk, 'busy_time', 0) / max(1, state.timestamp) / 1000.0))
        except (AttributeError, RuntimeError):
            pass

    except ImportError:
        pass  # psutil not installed — degrade gracefully
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
        record_degradation("perceptual_pump.system", e)

    return state


def _collect_user_state() -> UserState:
    """Collect inferred user state from WorldState."""
    state = UserState()
    state.timestamp = time.time()
    state.monotonic_ns = time.monotonic_ns()

    try:
        ws = ServiceContainer.get("world_state", default=None)
        if ws:
            state.available = True
            state.source = "world_state_activity_model"
            state.confidence = 0.70
            state.missing_reason = None
            state.idle_seconds = ws.user_idle_seconds
            state.message_count = ws.user_message_count
            state.estimated_mood = ws.estimated_user_mood

            # Presence: decays over 5 minutes of idle
            if state.idle_seconds < 30:
                state.presence = 1.0
            elif state.idle_seconds < 300:
                state.presence = max(0.1, 1.0 - state.idle_seconds / 300.0)
            else:
                state.presence = 0.1

            # Interaction type heuristic
            if state.idle_seconds < 5:
                state.last_interaction_type = "text"
            elif state.idle_seconds < 60:
                state.last_interaction_type = "recent"
            else:
                state.last_interaction_type = "none"
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation("perceptual_pump.user", e)

    return state


# ---------------------------------------------------------------------------
# RuntimeBody mapping
# ---------------------------------------------------------------------------

def frame_to_runtime_body(frame: PerceptualFrame) -> RuntimeBody:
    """Map a PerceptualFrame into RuntimeBody fields for the phenomenal engine.

    This is where physical-world perception becomes interoceptive state.
    """
    from core.phenomenal_substrate.types import RuntimeBody

    # Energy: inversely proportional to system load
    energy = max(0.1, 1.0 - 0.4 * (frame.system.cpu_percent / 100.0)
                 - 0.3 * (frame.system.memory_percent / 100.0)
                 - 0.3 * frame.system.thermal_pressure)

    # Continuity: stable when screen hasn't changed dramatically
    continuity = max(0.2, 1.0 - 0.6 * frame.screen.change_magnitude)

    # Agency: high when system is responsive and user is present
    agency = max(0.2, 0.4 + 0.3 * (1.0 - frame.system.cpu_percent / 100.0)
                 + 0.3 * frame.user.presence)

    # Safety: inversely proportional to threat score
    safety = max(0.1, 1.0 - frame.threat_score())

    # Social contact: driven by voice activity and user presence
    social = frame.social_signal()

    # Novelty: from screen changes and new audio
    novelty = frame.novelty_score()

    # Uncertainty: high when screen content is unfamiliar or changing rapidly
    uncertainty = min(0.9, 0.2 + 0.5 * frame.screen.change_magnitude
                      + 0.3 * (1.0 - frame.user.presence))
    if frame.fusion is not None:
        uncertainty = max(uncertainty, frame.fusion.uncertainty)

    # Compute pressure: direct from CPU
    compute_pressure = frame.system.cpu_percent / 100.0

    # Memory pressure: direct from RAM
    memory_pressure = frame.system.memory_percent / 100.0

    # Error pressure: thermal + low battery
    error_pressure = max(0.0,
        0.5 * frame.system.thermal_pressure
        + 0.5 * max(0.0, (20 - frame.system.battery_percent) / 20)
            if not frame.system.battery_charging else 0.0)

    return RuntimeBody(
        energy=min(1.0, max(0.0, energy)),
        continuity=min(1.0, max(0.0, continuity)),
        agency=min(1.0, max(0.0, agency)),
        safety=min(1.0, max(0.0, safety)),
        social_contact=min(1.0, max(0.0, social)),
        novelty=min(1.0, max(0.0, novelty)),
        uncertainty=min(1.0, max(0.0, uncertainty)),
        compute_pressure=min(1.0, max(0.0, compute_pressure)),
        memory_pressure=min(1.0, max(0.0, memory_pressure)),
        error_pressure=min(1.0, max(0.0, error_pressure)),
        screen_novelty=frame.screen.change_magnitude,
        audio_energy=frame.audio.rms_energy,
        voice_present=frame.audio.voice_activity,
        foreground_app_familiar=0.8 if frame.screen.active_app else 0.2,
    )


# ---------------------------------------------------------------------------
# The Pump
# ---------------------------------------------------------------------------

class PerceptualPump:
    """Continuous 10 Hz perceptual loop — the PRIMARY substrate driver.

    This replaces text-only context as the dominant input to Aura's
    inner state. Real-world changes perturb her continuous state vector
    at 10 Hz BEFORE the LLM sees anything.

    Architecture:
        sensors → PerceptualFrame → RuntimeBody
        → PhenomenalEngine.step() → ExperienceState
        → ContinuousSubstrate.inject_perceptual_frame()
        → WorldState.update_from_perceptual_frame()
    """

    PUMP_HZ = 10.0
    PUMP_DT = 1.0 / PUMP_HZ

    # Not every collector runs every tick (to save CPU)
    SCREEN_EVERY_N = 5      # every 500ms (AppleScript is slow)
    AUDIO_EVERY_N = 2       # every 200ms
    SYSTEM_EVERY_N = 10     # every 1s
    USER_EVERY_N = 5        # every 500ms

    # Substrate injection rate (can be lower than perception rate)
    SUBSTRATE_INJECT_EVERY_N = 2  # every 200ms
    #: How many transactions in a row may miss the budget before it stops
    #: being backpressure. One the next transaction recovers from costs this
    #: frame's freshness; a run means the substrate is not keeping up and the
    #: freshness gates downstream begin reading stale.
    SUBSTRATE_OVERRUNS_BEFORE_A_WARNING = 3

    def __init__(self) -> None:
        self.running = False
        self._task: asyncio.Task | None = None
        self._frame_count: int = 0
        self._last_screen_hash: str = ""
        self._latest_frame: PerceptualFrame | None = None
        self._frame_history: deque[PerceptualFrame] = deque(maxlen=100)
        self._synchronizer = MultimodalSynchronizer()
        self._sensor_sequences: dict[str, int] = {}
        self._last_world_event_ids: dict[Modality, str] = {}
        self._substrate_executor: ThreadPoolExecutor | None = None
        self._substrate_worker_active: bool = False
        self._substrate_worker_thread: str | None = None
        self._substrate_injection_last_ms: float = 0.0
        self._substrate_injection_max_ms: float = 0.0
        self._substrate_injection_overruns: int = 0
        self._substrate_injection_overrun_streak: int = 0

        # Cached sub-states (updated at different rates)
        self._screen: ScreenState = ScreenState()
        self._audio: AudioState = AudioState()
        self._system: SystemState = SystemState()
        self._user: UserState = UserState()

        # Stats
        self._frames_produced: int = 0
        self._substrate_injections: int = 0
        self._errors: int = 0
        self._started_at: float = 0.0

    async def start(self) -> None:
        """Start the perceptual pump."""
        if self.running:
            return
        if self._substrate_executor is None:
            self._substrate_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="AuraPerceptualSubstrate",
            )
        self.running = True
        self._started_at = time.time()
        ServiceContainer.register_instance("perceptual_pump", self, required=False)
        ServiceContainer.register_instance(
            "multimodal_synchronizer",
            self._synchronizer,
            required=False,
        )
        self._task = create_tracked_task(
            self._pump_loop(),
            name="Aura.PerceptualPump",
        )
        logger.info(
            "👁️ PerceptualPump ONLINE — %.0f Hz perceptual grounding active "
            "(screen every %d ticks, audio every %d ticks, system every %d ticks)",
            self.PUMP_HZ, self.SCREEN_EVERY_N, self.AUDIO_EVERY_N, self.SYSTEM_EVERY_N,
        )

    async def stop(self) -> None:
        """Stop the perceptual pump."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        executor = self._substrate_executor
        self._substrate_executor = None
        if executor is not None:
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
        self._substrate_worker_active = False
        logger.info(
            "👁️ PerceptualPump OFFLINE — produced %d frames, %d substrate injections, %d errors",
            self._frames_produced, self._substrate_injections, self._errors,
        )

    @property
    def latest_frame(self) -> PerceptualFrame | None:
        """The most recent perceptual frame."""
        return self._latest_frame

    def get_status(self) -> dict[str, Any]:
        """Pump status for dashboards."""
        synchronizer_status = dict(self._synchronizer.get_status())
        synchronizer_status.pop("latest", None)
        return {
            "running": self.running,
            "frames_produced": self._frames_produced,
            "substrate_injections": self._substrate_injections,
            "errors": self._errors,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "latest_frame": {
                "active_app": self._screen.active_app,
                "window_title_available": bool(self._screen.window_title),
                "voice_activity": self._audio.voice_activity,
                "cpu": round(self._system.cpu_percent, 1),
                "user_presence": round(self._user.presence, 2),
            } if self._latest_frame else None,
            "fusion": (
                self._latest_frame.fusion.to_status()
                if self._latest_frame is not None and self._latest_frame.fusion is not None
                else None
            ),
            "synchronizer": synchronizer_status,
            "causal_consumers": [
                "phenomenal_engine",
                "conscious_substrate",
                "world_state_beliefs",
                "cognitive_situation",
                "task_planning",
            ],
            "pump_hz": self.PUMP_HZ,
            "substrate_worker": {
                "owner": "dedicated_single_worker",
                "active": self._substrate_worker_active,
                "thread": self._substrate_worker_thread,
                "ordered": True,
                "last_ms": round(self._substrate_injection_last_ms, 3),
                "max_ms": round(self._substrate_injection_max_ms, 3),
                "overruns": self._substrate_injection_overruns,
                "overrun_streak": self._substrate_injection_overrun_streak,
                "budget_ms": round(
                    self.PUMP_DT * self.SUBSTRATE_INJECT_EVERY_N * 1000.0,
                    3,
                ),
            },
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _pump_loop(self) -> None:
        """The main 10 Hz perceptual ingestion loop."""
        try:
            while self.running:
                tick_start = time.time()
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except _PUMP_RUNTIME_ERRORS as e:
                    self._errors += 1
                    record_degradation("perceptual_pump", e)
                    if self._errors % 50 == 1:
                        logger.warning("PerceptualPump tick error #%d: %s", self._errors, e)

                # Maintain target rate
                elapsed = time.time() - tick_start
                sleep_time = max(0.01, self.PUMP_DT - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            raise
        except _PUMP_RUNTIME_ERRORS as e:
            record_degradation("perceptual_pump", e)
            logger.error("PerceptualPump loop crashed: %s", e)
            self.running = False

    def _cognitive_load_throttle_active(self) -> bool:
        """Perception yields to cognition.

        While foreground inference owns the machine or memory is tight,
        the subprocess-spawning sensors (AppleScript probes) slow down to
        a crawl instead of competing for a starved host. Attention
        narrows under load; it does not flail.
        """
        try:
            gate = ServiceContainer.get("inference_gate", default=None)
            used_lightweight_probe = False
            for probe_name in ("_foreground_user_turn_active", "_foreground_owner_active"):
                probe = getattr(gate, probe_name, None)
                if callable(probe):
                    used_lightweight_probe = True
                    if bool(probe()):
                        return True
            if not used_lightweight_probe and gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if lane.get("foreground_owned") or int(lane.get("active_generations", 0) or 0) > 0:
                    return True
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass
        return float(getattr(self._system, "memory_percent", 0.0) or 0.0) >= 85.0

    # While throttled, screen/user sensors run 10x slower (every 5s).
    THROTTLED_MULTIPLIER = 10

    def _next_sensor_sequence(self, source: str) -> int:
        sequence = self._sensor_sequences.get(source, 0) + 1
        self._sensor_sequences[source] = sequence
        return sequence

    def _ingest_event(
        self,
        *,
        modality: Modality,
        source: str,
        observed_at: float,
        observed_monotonic_ns: int,
        summary: str,
        confidence: float,
        claims: tuple[PerceptualClaim, ...] = (),
        missing_reason: MissingReason | None = None,
        calibration_status: str = "unknown",
        calibration_reliability: float = 0.5,
        privacy: PrivacyPolicy | None = None,
        provenance: tuple[str, ...] = (),
        quality_flags: tuple[str, ...] = (),
    ) -> None:
        sequence_source = f"{source}:{modality.value}"
        sequence = self._next_sensor_sequence(sequence_source)
        calibration_state = (
            "valid"
            if calibration_status == "valid"
            else "failed"
            if calibration_status == "failed"
            else "expired"
            if calibration_status == "expired"
            else "unknown"
        )
        event = PerceptualEvent(
            event_id=f"{modality.value}:{sequence}:{observed_monotonic_ns}",
            modality=modality,
            source=source[:160],
            sequence=sequence,
            observed_at=observed_at,
            observed_monotonic_ns=observed_monotonic_ns,
            summary=summary[:320],
            confidence=0.0 if missing_reason is not None else max(0.0, min(1.0, confidence)),
            claims=() if missing_reason is not None else claims,
            calibration=Calibration(
                calibration_id=f"runtime:{source[:140]}",
                status=calibration_state,
                reliability=max(0.0, min(1.0, calibration_reliability)),
            ),
            provenance=provenance[:16],
            privacy=privacy or PrivacyPolicy(),
            missing_reason=missing_reason,
            quality_flags=quality_flags[:16],
        )
        self._synchronizer.ingest(event)

    def _ingest_screen_state(self, state: ScreenState) -> None:
        missing = state.missing_reason if not state.available else None
        screen_claims = (
            PerceptualClaim("screen.active_app", state.active_app, state.confidence),
            PerceptualClaim("screen.window_title", state.window_title, state.confidence * 0.9),
            PerceptualClaim("screen.changed", state.screen_changed, 0.95),
        )
        self._ingest_event(
            modality=Modality.VISION,
            source=state.source,
            observed_at=state.timestamp,
            observed_monotonic_ns=state.monotonic_ns,
            summary="screen metadata observed" if state.available else "screen metadata unavailable",
            confidence=state.confidence,
            claims=screen_claims,
            missing_reason=missing,
            calibration_reliability=0.90,
            privacy=PrivacyPolicy(
                classification=PrivacyClass.PRIVATE,
                retention="none",
                redacted=True,
            ),
            provenance=("perceptual_pump.screen", state.source),
        )
        spatial_claims = (
            PerceptualClaim("spatial.foreground_app", state.active_app, state.confidence),
            PerceptualClaim("spatial.window_title", state.window_title, state.confidence * 0.9),
        )
        self._ingest_event(
            modality=Modality.SPATIAL,
            source=state.source,
            observed_at=state.timestamp,
            observed_monotonic_ns=state.monotonic_ns,
            summary="desktop spatial focus observed" if state.available else "desktop spatial focus unavailable",
            confidence=state.confidence,
            claims=spatial_claims,
            missing_reason=missing,
            calibration_reliability=0.85,
            privacy=PrivacyPolicy(
                classification=PrivacyClass.PRIVATE,
                retention="none",
                redacted=True,
            ),
            provenance=("perceptual_pump.spatial", state.source),
        )
        if state.available and state.content_hash:
            self._ingest_event(
                modality=Modality.TEXT,
                source=f"{state.source}:ocr",
                observed_at=state.timestamp,
                observed_monotonic_ns=state.monotonic_ns,
                summary="redacted screen text digest observed",
                confidence=min(state.confidence, 0.75),
                claims=(
                    PerceptualClaim("screen.text_digest", state.content_hash, 0.75),
                    PerceptualClaim("screen.text_changed", state.screen_changed, 0.90),
                ),
                calibration_reliability=0.75,
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.SENSITIVE,
                    retention="none",
                    redacted=True,
                ),
                provenance=("perceptual_pump.screen_ocr_digest",),
            )

    def _ingest_audio_state(self, state: AudioState) -> None:
        missing = state.missing_reason if not state.available else None
        self._ingest_event(
            modality=Modality.AUDIO,
            source=state.source,
            observed_at=state.timestamp,
            observed_monotonic_ns=state.monotonic_ns,
            summary="audio activity observed" if state.available else "audio sensor unavailable",
            confidence=state.confidence,
            claims=(
                PerceptualClaim("audio.voice_activity", state.voice_activity, 0.92),
                PerceptualClaim("audio.rms_energy", round(state.rms_energy, 6), 0.85),
            ),
            missing_reason=missing,
            calibration_reliability=0.78,
            privacy=PrivacyPolicy(
                classification=PrivacyClass.PRIVATE,
                retention="none",
                redacted=True,
            ),
            provenance=("perceptual_pump.audio", state.source),
        )
        transcript = state.transcript_full or state.transcript_snippet
        if state.available and transcript:
            transcript_digest = hashlib.sha256(
                transcript.encode("utf-8", errors="ignore")
            ).hexdigest()[:24]
            self._ingest_event(
                modality=Modality.SPEECH,
                source=f"{state.source}:transcript",
                observed_at=state.timestamp,
                observed_monotonic_ns=state.monotonic_ns,
                summary="redacted speech transcript digest observed",
                confidence=min(state.confidence, 0.75),
                claims=(
                    PerceptualClaim("speech.transcript_available", True, 0.95),
                    PerceptualClaim("speech.transcript_digest", transcript_digest, 0.75),
                ),
                calibration_reliability=0.72,
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.SENSITIVE,
                    retention="none",
                    redacted=True,
                ),
                provenance=("perceptual_pump.speech_digest", state.source),
                quality_flags=("audio_transcript_not_visual_speech",),
            )

    def _ingest_system_state(self, state: SystemState) -> None:
        missing = state.missing_reason if not state.available else None
        device_claims = (
            PerceptualClaim("device.cpu_percent", round(state.cpu_percent, 4), 0.98),
            PerceptualClaim("device.memory_percent", round(state.memory_percent, 4), 0.98),
            PerceptualClaim("device.battery_percent", round(state.battery_percent, 4), 0.95),
            PerceptualClaim("device.battery_charging", state.battery_charging, 0.95),
        )
        self._ingest_event(
            modality=Modality.DEVICE,
            source=state.source,
            observed_at=state.timestamp,
            observed_monotonic_ns=state.monotonic_ns,
            summary="device telemetry observed" if state.available else "device telemetry unavailable",
            confidence=state.confidence,
            claims=device_claims,
            missing_reason=missing,
            calibration_status="valid" if state.available else "unknown",
            calibration_reliability=0.98,
            privacy=PrivacyPolicy(
                classification=PrivacyClass.PRIVATE,
                retention="none",
                redacted=True,
            ),
            provenance=("perceptual_pump.device", state.source),
        )
        body_claims = (
            PerceptualClaim("body.compute_pressure", round(state.cpu_percent / 100.0, 6), 0.95),
            PerceptualClaim("body.memory_pressure", round(state.memory_percent / 100.0, 6), 0.95),
            PerceptualClaim("body.thermal_pressure", round(state.thermal_pressure, 6), 0.80),
        )
        self._ingest_event(
            modality=Modality.BODY,
            source=f"{state.source}:interoception",
            observed_at=state.timestamp,
            observed_monotonic_ns=state.monotonic_ns,
            summary="runtime body telemetry observed" if state.available else "runtime body telemetry unavailable",
            confidence=state.confidence,
            claims=body_claims,
            missing_reason=missing,
            calibration_status="valid" if state.available else "unknown",
            calibration_reliability=0.95,
            privacy=PrivacyPolicy(
                classification=PrivacyClass.PRIVATE,
                retention="none",
                redacted=True,
            ),
            provenance=("perceptual_pump.body", state.source),
        )

    async def _tick(self) -> None:
        """One pump tick: collect sensors, build frame, inject into substrate."""
        self._frame_count += 1
        n = self._frame_count

        throttled = self._cognitive_load_throttle_active()
        screen_every = self.SCREEN_EVERY_N * (self.THROTTLED_MULTIPLIER if throttled else 1)
        user_every = self.USER_EVERY_N * (self.THROTTLED_MULTIPLIER if throttled else 1)
        # Collect from each modality at its configured rate
        # These run in a thread to avoid blocking the event loop
        if n % screen_every == 0:
            previous_screen = self._screen
            self._screen = await asyncio.to_thread(
                _collect_screen_state, self._last_screen_hash
            )
            app_changed = bool(
                self._screen.active_app
                and self._screen.active_app != previous_screen.active_app
            )
            if app_changed:
                self._screen.screen_changed = True
                self._screen.change_magnitude = max(0.3, self._screen.change_magnitude)
            elif not self._screen.content_hash:
                self._screen.screen_changed = False
                self._screen.change_magnitude = 0.0
            self._last_screen_hash = self._screen.content_hash or self._last_screen_hash
            self._ingest_screen_state(self._screen)

        if n % self.AUDIO_EVERY_N == 0:
            self._audio = await asyncio.to_thread(_collect_audio_state)
            self._ingest_audio_state(self._audio)

        if n % self.SYSTEM_EVERY_N == 0:
            self._system = await asyncio.to_thread(_collect_system_state)
            self._ingest_system_state(self._system)

        if n % user_every == 0:
            self._user = await asyncio.to_thread(_collect_user_state)

        # Build the frame
        frame_timestamp = time.time()
        frame_monotonic_ns = time.monotonic_ns()
        fusion = self._synchronizer.fuse(
            f"perceptual-{self._frames_produced}-{frame_monotonic_ns}",
            anchor_monotonic_ns=frame_monotonic_ns,
            emitted_at=frame_timestamp,
        )
        frame = PerceptualFrame(
            frame_id=self._frames_produced,
            screen=self._screen,
            audio=self._audio,
            system=self._system,
            user=self._user,
            timestamp=frame_timestamp,
            monotonic_ns=frame_monotonic_ns,
            fusion=fusion,
        )
        self._latest_frame = frame
        self._frame_history.append(frame)
        self._frames_produced += 1

        # Inject into substrate at its configured rate
        if n % self.SUBSTRATE_INJECT_EVERY_N == 0:
            await self._inject_into_substrate(frame)

        # Update WorldState with perceptual data (every tick)
        self._update_world_state(frame)

        try:
            from core.runtime.timescale_bridge import get_timescale_bridge

            get_timescale_bridge().ingest_perceptual_frame(frame)
        except _PUMP_RUNTIME_ERRORS as e:
            record_degradation(
                "perceptual_pump.timescale_bridge",
                e,
                severity="warning",
                action="continued perceptual pump without timescale observation bridge",
                enforce_failure_policy=False,
            )

    async def _inject_into_substrate(self, frame: PerceptualFrame) -> None:
        """Map frame → RuntimeBody → PhenomenalEngine → Substrate.

        This is the causal grounding path. The substrate ODE state is
        directly perturbed by physical-world perception. The complete
        phenomenal/substrate transaction runs on one dedicated worker so
        expensive NumPy/Torch work cannot block the kernel event loop and
        successive frames cannot reorder their causal effects.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._substrate_executor,
            self._inject_into_substrate_sync,
            frame,
        )

    def _inject_into_substrate_sync(self, frame: PerceptualFrame) -> None:
        """Execute one ordered perceptual-to-substrate transaction."""
        started = time.perf_counter()
        self._substrate_worker_active = True
        self._substrate_worker_thread = threading.current_thread().name
        try:
            body = frame_to_runtime_body(frame)
            fusion = frame.fusion
            fusion_confidence = fusion.confidence if fusion is not None else 0.0
            fusion_uncertainty = fusion.uncertainty if fusion is not None else 1.0
            missing_modalities = (
                sorted(modality.value for modality in fusion.missing)
                if fusion is not None
                else [modality.value for modality in Modality]
            )
            unresolved_contradictions = (
                len(fusion.unresolved_contradictions) if fusion is not None else 0
            )

            # 1. Feed through PhenomenalEngine for affect computation
            engine = ServiceContainer.get("phenomenal_engine", default=None)
            if engine and hasattr(engine, "step"):
                from core.phenomenal_substrate.types import Event
                event = Event(
                    label=f"perceptual_frame_{frame.frame_id}",
                    source="perceptual_pump",
                    novelty=frame.novelty_score(),
                    threat=frame.threat_score(),
                    affiliation=frame.social_signal(),
                    control_gain=max(0.0, frame.user.presence - 0.5) * 0.3,
                    goal_delta=0.0,
                )
                experience = engine.step(body, event, recurrent_cycles=2)

                # 2. Inject the experience-derived affect into the substrate
                substrate = ServiceContainer.get("conscious_substrate", default=None)
                if substrate and hasattr(substrate, "inject_perceptual_frame"):
                    substrate.inject_perceptual_frame({
                        "source": "perceptual_pump",
                        "valence": experience.valence,
                        "arousal": experience.arousal,
                        "novelty": frame.novelty_score(),
                        "threat": frame.threat_score(),
                        "social": frame.social_signal(),
                        "cpu_percent": frame.system.cpu_percent,
                        "memory_percent": frame.system.memory_percent,
                        "thermal": frame.system.thermal_pressure,
                        "user_presence": frame.user.presence,
                        "screen_changed": frame.screen.screen_changed,
                        "voice_activity": frame.audio.voice_activity,
                        "active_app": frame.screen.active_app,
                        "timestamp": frame.timestamp,
                        "fusion_frame_id": fusion.frame_id if fusion is not None else "",
                        "perception_confidence": fusion_confidence,
                        "perception_uncertainty": fusion_uncertainty,
                        "missing_modalities": missing_modalities,
                        "unresolved_contradictions": unresolved_contradictions,
                    })
                    # Adapt projections so readouts track real affect
                    if hasattr(substrate, "adapt_projections"):
                        substrate.adapt_projections({
                            "valence": experience.valence,
                            "arousal": experience.arousal,
                            "curiosity": experience.curiosity,
                        }, lr=0.005)
                    self._substrate_injections += 1
                elif substrate and hasattr(substrate, "inject_observation"):
                    # Fallback to raw observation injection
                    substrate.inject_observation({
                        "source": "perceptual_pump",
                        "confidence": fusion_confidence,
                        "energy": frame.audio.rms_energy,
                        "summary": f"app={frame.screen.active_app} voice={frame.audio.voice_activity}",
                        "timestamp_unix": frame.timestamp,
                        "uncertainty": fusion_uncertainty,
                        "fusion_frame_id": fusion.frame_id if fusion is not None else "",
                    })
                    self._substrate_injections += 1

            else:
                # No phenomenal engine — inject directly into substrate
                substrate = ServiceContainer.get("conscious_substrate", default=None)
                if substrate and hasattr(substrate, "inject_observation"):
                    substrate.inject_observation({
                        "source": "perceptual_pump",
                        "confidence": fusion_confidence,
                        "energy": frame.audio.rms_energy,
                        "summary": f"screen={frame.screen.active_app} cpu={frame.system.cpu_percent:.0f}%",
                        "timestamp_unix": frame.timestamp,
                        "uncertainty": fusion_uncertainty,
                        "fusion_frame_id": fusion.frame_id if fusion is not None else "",
                    })
                    self._substrate_injections += 1

        except (ImportError, AttributeError, RuntimeError) as e:
            self._errors += 1
            record_degradation("perceptual_pump.substrate", e)
            if self._errors % 100 == 1:
                logger.debug("Substrate injection failed: %s", e)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._substrate_injection_last_ms = elapsed_ms
            self._substrate_injection_max_ms = max(
                self._substrate_injection_max_ms,
                elapsed_ms,
            )
            budget_ms = self.PUMP_DT * self.SUBSTRATE_INJECT_EVERY_N * 1000.0
            if elapsed_ms > budget_ms:
                self._substrate_injection_overruns += 1
                self._substrate_injection_overrun_streak += 1
                # A single overrun the next transaction recovers from is
                # backpressure: it costs this frame's freshness and nothing
                # else, because this runs off the pump thread. Warning on the
                # FIRST of every streak means an intermittently slow host
                # warns continuously — measured live 2026-09-07, repeatedly,
                # each one "(overruns=N, streak=1)".
                #
                # A run is different. That is a substrate not keeping up, and
                # the freshness gates downstream begin reading stale.
                persistent = (
                    self._substrate_injection_overrun_streak
                    >= self.SUBSTRATE_OVERRUNS_BEFORE_A_WARNING
                )
                if persistent or self._substrate_injection_overruns % 50 == 0:
                    logger.warning(
                        "Perceptual substrate transaction exceeded budget: %.1fms > %.1fms "
                        "(overruns=%d, streak=%d)",
                        elapsed_ms,
                        budget_ms,
                        self._substrate_injection_overruns,
                        self._substrate_injection_overrun_streak,
                    )
                elif self._substrate_injection_overrun_streak == 1:
                    logger.info(
                        "Perceptual substrate transaction exceeded budget once: "
                        "%.1fms > %.1fms (overruns=%d)",
                        elapsed_ms,
                        budget_ms,
                        self._substrate_injection_overruns,
                    )
            else:
                self._substrate_injection_overrun_streak = 0
            self._substrate_worker_active = False

    def _update_world_state(self, frame: PerceptualFrame) -> None:
        """Push perceptual data into WorldState for downstream consumers."""
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if not ws:
                return
            fusion = frame.fusion
            vision_current = fusion is None or fusion.has_usable(Modality.VISION)
            audio_current = fusion is None or fusion.has_usable(Modality.AUDIO)
            speech_current = fusion is None or fusion.has_usable(Modality.SPEECH)
            device_current = fusion is None or fusion.has_usable(Modality.DEVICE)

            # Freshness gates prevent a cached sensor value from being rewritten
            # as if it were a new observation after the source has gone stale.
            if vision_current and hasattr(ws, "active_foreground_app"):
                ws.active_foreground_app = frame.screen.active_app
            if vision_current and hasattr(ws, "active_window_title"):
                ws.active_window_title = frame.screen.window_title
            if vision_current and hasattr(ws, "screen_content_hash"):
                ws.screen_content_hash = frame.screen.content_hash
            if audio_current and hasattr(ws, "ambient_audio_level"):
                ws.ambient_audio_level = frame.audio.rms_energy
            if audio_current and hasattr(ws, "voice_activity_detected"):
                ws.voice_activity_detected = frame.audio.voice_activity
            elif not audio_current and hasattr(ws, "voice_activity_detected"):
                ws.voice_activity_detected = False
            if audio_current and frame.audio.voice_activity and hasattr(ws, "last_voice_activity_at"):
                ws.last_voice_activity_at = frame.audio.timestamp or frame.timestamp
            transcript = frame.audio.transcript_full or frame.audio.transcript_snippet
            transcript_changed = bool(
                transcript
                and transcript != str(getattr(ws, "last_voice_transcript", "") or "")
            )
            if speech_current and hasattr(ws, "last_voice_transcript") and transcript:
                # Full fidelity here: the wake-word command lane reads this
                # field, and the 200-char display snippet used to truncate
                # long spoken commands mid-utterance.
                ws.last_voice_transcript = transcript
                if hasattr(ws, "last_voice_transcript_at"):
                    ws.last_voice_transcript_at = frame.audio.timestamp or frame.timestamp
                if hasattr(ws, "last_audio_source_assessment"):
                    ws.last_audio_source_assessment = {
                        "source": "perceptual_pump_audio",
                        "response_authorized": False,
                        "attention_mode": "listen",
                        "transcript_changed": transcript_changed,
                        "fusion_frame_id": fusion.frame_id if fusion is not None else "legacy",
                        "confidence": fusion.confidence if fusion is not None else frame.audio.confidence,
                        "provenance": ["audio_transcript"],
                        "visual_speech_evidence": False,
                    }

            if device_current:
                ws.cpu_percent = frame.system.cpu_percent
                ws.memory_percent = frame.system.memory_percent
                ws.thermal_pressure = frame.system.thermal_pressure
                ws.battery_percent = frame.system.battery_percent
                ws.battery_charging = frame.system.battery_charging

            if fusion is not None and hasattr(ws, "set_belief"):
                ws.set_belief(
                    "perception.fusion_confidence",
                    fusion.confidence,
                    confidence=1.0,
                    source="multimodal_synchronizer",
                    ttl=30.0,
                )
                ws.set_belief(
                    "perception.uncertainty",
                    fusion.uncertainty,
                    confidence=1.0,
                    source="multimodal_synchronizer",
                    ttl=30.0,
                )
                ws.set_belief(
                    "perception.missing_modalities",
                    {
                        modality.value: reason.value
                        for modality, reason in sorted(
                            fusion.missing.items(), key=lambda item: item[0].value
                        )
                    },
                    confidence=1.0,
                    source="multimodal_synchronizer",
                    ttl=30.0,
                )
                ws.set_belief(
                    "perception.unresolved_contradictions",
                    [item.key for item in fusion.unresolved_contradictions],
                    confidence=1.0,
                    source="multimodal_synchronizer",
                    ttl=30.0,
                )
                for belief_key in fusion.directives.memory_candidates:
                    belief = fusion.belief(belief_key)
                    if belief is None or belief.value is None:
                        continue
                    ws.set_belief(
                        f"perception.{belief.key}",
                        belief.value,
                        confidence=belief.confidence,
                        source="multimodal_synchronizer",
                        ttl=30.0,
                    )

            # Generate salient events for significant changes
            vision_event = fusion.observations.get(Modality.VISION) if fusion is not None else None
            vision_event_id = vision_event.event_id if vision_event is not None else "legacy"
            if (
                vision_current
                and frame.screen.screen_changed
                and frame.screen.active_app
                and self._last_world_event_ids.get(Modality.VISION) != vision_event_id
            ):
                ws.record_event(
                    f"App switched to {frame.screen.active_app}",
                    source="perception",
                    salience=0.4,
                    ttl=120,
                    fusion_frame_id=fusion.frame_id if fusion is not None else "legacy",
                    evidence_event_id=vision_event_id,
                )
                self._last_world_event_ids[Modality.VISION] = vision_event_id

            if audio_current and frame.audio.voice_activity and transcript_changed:
                ws.record_event(
                    f"Voice detected: {frame.audio.transcript_snippet[:80]}",
                    source="perception",
                    salience=0.7,
                    ttl=60,
                    fusion_frame_id=fusion.frame_id if fusion is not None else "legacy",
                    visual_speech_evidence=False,
                )

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("perceptual_pump.world_state", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pump_instance: PerceptualPump | None = None


def get_perceptual_pump() -> PerceptualPump:
    """Get or create the global PerceptualPump."""
    global _pump_instance
    if _pump_instance is None:
        _pump_instance = PerceptualPump()
    return _pump_instance


__all__ = [
    "PerceptualPump",
    "PerceptualFrame",
    "ScreenState",
    "AudioState",
    "SystemState",
    "UserState",
    "frame_to_runtime_body",
    "get_perceptual_pump",
]
