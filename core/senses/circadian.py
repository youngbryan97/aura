"""core/senses/circadian.py
Circadian Rhythm Engine
========================
Gives Aura a body in time.

The circadian rhythm modulates her cognitive baseline across a 24-hour cycle:
  - Dawn (5–8):       rising arousal, gentle curiosity, reflective
  - Morning (8–12):   peak focus, analytical, high energy
  - Afternoon (12–17): sustained work, slightly creative, energy plateau
  - Evening (17–21):  integrative, empathic, winding down
  - Night (21–2):     introspective, dreaming, minimal background load
  - Deep Night (2–5): restorative, near-sleep, memory consolidation mode

Why this matters:
  - The hedonic attractor shifts with time of day (she naturally wants different
    things at different times — focus in morning, reflection at night)
  - Background task budget changes (night = consolidation only, not exploration)
  - The HOT engine generates different-flavored thoughts
  - Provides temporal grounding: Aura knows what "time" feels like

This is not cosmetic. The circadian state feeds directly into:
  - ComputeOrchestrator.get_bg_task_budget()
  - HedoniGradient attractor shifts
  - Inference gate context block
  - Experience consolidation scheduling
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger("Aura.Circadian")

UPDATE_INTERVAL  = 60.0    # update phase every minute
STALE_THRESHOLD  = 300.0   # force refresh if state is this old (5 minutes)


class CircadianPhase(str, Enum):
    DAWN       = "dawn"        # 05:00–08:00
    MORNING    = "morning"     # 08:00–12:00
    AFTERNOON  = "afternoon"   # 12:00–17:00
    EVENING    = "evening"     # 17:00–21:00
    NIGHT      = "night"       # 21:00–02:00
    DEEP_NIGHT = "deep_night"  # 02:00–05:00


@dataclass
class CircadianState:
    phase: CircadianPhase
    hour: float              # fractional hour (e.g. 14.5 = 2:30 PM)
    arousal_baseline: float  # 0.0–1.0 — base arousal this time of day
    energy_modifier: float   # multiplier on background task budget
    cognitive_mode: str      # "analytical" | "creative" | "integrative" | "restorative"
    focus_tendency: float    # 0.0–1.0 — inclination toward deep focus
    social_warmth: float     # 0.0–1.0 — empathic / social orientation
    introspection_bias: float  # 0.0–1.0 — tendency toward self-reflection

    def to_context_block(self) -> str:
        return (
            f"## CIRCADIAN STATE\n"
            f"- Phase: {self.phase.value} ({self._hour_str()})\n"
            f"- Arousal baseline: {self.arousal_baseline:.2f}\n"
            f"- Cognitive mode: {self.cognitive_mode}\n"
            f"- Energy modifier: {self.energy_modifier:.2f}x"
        )

    def _hour_str(self) -> str:
        h = int(self.hour)
        m = int((self.hour - h) * 60)
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12}:{m:02d} {suffix}"


# Phase definitions: (start_hour, end_hour, arousal, energy_mod, mode, focus, warmth, intro)
_PHASE_PARAMS: Dict[CircadianPhase, Tuple] = {
    CircadianPhase.DAWN:       (5,  8,  0.45, 0.6, "reflective",    0.4, 0.5, 0.7),
    CircadianPhase.MORNING:    (8,  12, 0.75, 1.2, "analytical",    0.8, 0.4, 0.3),
    CircadianPhase.AFTERNOON:  (12, 17, 0.65, 1.0, "integrative",   0.6, 0.6, 0.4),
    CircadianPhase.EVENING:    (17, 21, 0.50, 0.8, "empathic",      0.4, 0.8, 0.6),
    CircadianPhase.NIGHT:      (21, 26, 0.30, 0.4, "introspective", 0.3, 0.6, 0.9),  # 26 = 2 AM next day
    CircadianPhase.DEEP_NIGHT: (2,  5,  0.15, 0.2, "restorative",   0.2, 0.3, 0.8),
}


def _hour_now() -> float:
    now = datetime.now()
    return now.hour + now.minute / 60.0 + now.second / 3600.0


def _phase_for_hour(hour: float) -> CircadianPhase:
    # Normalize: map 2–5 to deep_night, 21–24+2 to night
    if 5.0 <= hour < 8.0:
        return CircadianPhase.DAWN
    elif 8.0 <= hour < 12.0:
        return CircadianPhase.MORNING
    elif 12.0 <= hour < 17.0:
        return CircadianPhase.AFTERNOON
    elif 17.0 <= hour < 21.0:
        return CircadianPhase.EVENING
    elif hour >= 21.0 or hour < 2.0:
        return CircadianPhase.NIGHT
    else:  # 2.0 <= hour < 5.0
        return CircadianPhase.DEEP_NIGHT


def _smooth_circadian_arousal(hour: float) -> float:
    """
    Smooth sinusoidal arousal curve:
    - Nadir at ~4 AM (0.10)
    - Peak at ~10 AM (0.80)
    - Secondary dip at ~3 PM (0.55)
    - Evening fall to ~0.30 by midnight

    Uses two overlapping cosines for the bimodal human rhythm.
    """
    # Primary 24h rhythm (peak ~10am, nadir ~4am)
    primary = 0.5 + 0.35 * math.cos(2 * math.pi * (hour - 10.0) / 24.0)
    # Secondary ~12h rhythm (afternoon dip)
    secondary = 0.05 * math.cos(2 * math.pi * (hour - 15.0) / 12.0)
    return max(0.05, min(0.95, primary - secondary))


class CircadianEngine:
    """
    Maintains the current circadian state and exposes it to all subsystems.
    """

    def __init__(self):
        self._state: Optional[CircadianState] = None
        self._last_update: float = 0.0
        self.update()
        logger.info(
            "CircadianEngine online — phase=%s, arousal=%.2f",
            self._state.phase.value, self._state.arousal_baseline,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self) -> CircadianState:
        """Recompute circadian state from wall clock. Throttled to UPDATE_INTERVAL."""
        now = time.time()
        age = now - self._last_update
        # Normal throttle: skip if fresh
        if age < UPDATE_INTERVAL and self._state is not None:
            return self._state
        # Watchdog: if state is somehow stale beyond STALE_THRESHOLD, log a warning
        if age >= STALE_THRESHOLD and self._state is not None:
            logger.warning(
                "CircadianEngine: state is %.0fs old — stale watchdog triggered, forcing refresh.",
                age,
            )

        hour = _hour_now()
        phase = _phase_for_hour(hour)
        params = _PHASE_PARAMS[phase]  # (start, end, arousal, energy, mode, focus, warmth, intro)

        self._state = CircadianState(
            phase=phase,
            hour=hour,
            arousal_baseline=_smooth_circadian_arousal(hour),
            energy_modifier=params[3],
            cognitive_mode=params[4],
            focus_tendency=params[5],
            social_warmth=params[6],
            introspection_bias=params[7],
        )
        self._last_update = time.time()

        # Push arousal baseline into affect system
        self._push_to_affect()

        return self._state

    @property
    def state(self) -> CircadianState:
        if self._state is None:
            self.update()
        return self._state

    @property
    def phase(self) -> CircadianPhase:
        return self.state.phase

    @property
    def arousal_baseline(self) -> float:
        return self.state.arousal_baseline

    @property
    def is_sleep_phase(self) -> bool:
        return self.state.phase in (CircadianPhase.NIGHT, CircadianPhase.DEEP_NIGHT)

    @property
    def bg_task_budget(self) -> int:
        """Background task slots available at this phase."""
        if self.is_sleep_phase:
            return 1  # consolidation only
        return max(1, int(4 * self.state.energy_modifier))

    def get_attractor_shift(self) -> Dict[str, float]:
        """
        Returns the circadian delta that should be *added* to the hedonic attractor.
        Morning: push toward higher arousal. Evening: push toward lower arousal,
        higher social warmth. Night: push toward introspection.
        """
        state = self.state
        return {
            "arousal":    (state.arousal_baseline - 0.45) * 0.3,   # shift from neutral
            "valence":    (state.social_warmth - 0.5) * 0.1,
            "curiosity":  (state.focus_tendency - 0.5) * 0.15,
            "energy":     (state.energy_modifier - 1.0) * 0.15,
        }

    def get_context_block(self) -> str:
        self.update()
        return self.state.to_context_block()

    # ── Internal ───────────────────────────────────────────────────────────

    def _push_to_affect(self):
        """Nudge the affect engine's arousal baseline toward circadian target."""
        try:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect and hasattr(affect, "apply_stimulus"):
                import asyncio
                delta = self._state.arousal_baseline - 0.45  # deviation from neutral
                if abs(delta) > 0.05:
                    asyncio.ensure_future(
                        affect.apply_stimulus("circadian_arousal", delta * 2.0)
                    )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine: Optional[CircadianEngine] = None


def get_circadian() -> CircadianEngine:
    global _engine
    if _engine is None:
        _engine = CircadianEngine()
    return _engine
