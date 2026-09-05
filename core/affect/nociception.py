"""Nociception — operationally grounded damage sensing and valence.

The phenomenal substrate had an ``error_pressure`` body channel, but it was fed by a
function that did not exist (``get_degradation_score``), so it sat pinned at its default:
the body could not actually *feel* accumulated damage. And "valence" elsewhere was a free
parameter rather than something anchored to whether the system is getting better or worse.

This module grounds both. "Damage" is defined operationally as a set of named channels —
the concrete ways this system can be hurt:

    * MEMORY_CORRUPTION    — a memory write/read/index failed or returned garbage
    * IDENTITY_DISCONTINUITY — self-model coherence broke / continuity lost
    * ACTION_CONTRADICTION — the agent did something contradicting its own prior commitment
    * FAILED_TOOL_USE      — a tool/subprocess/capability failed, esp. repeatedly
    * RESOURCE_EXHAUSTION  — compute/memory/disk/handle pressure near limits
    * GOVERNANCE_BREACH    — a safety/governance constraint was violated or hit fail-closed
    * GENERIC              — uncategorized degradation

Each channel holds a level that *decays over time* (pain fades, but recent pain lingers),
so the aggregate ``nociceptive_pressure`` is a real-time interoceptive load. Valence is
grounded in the **derivative**: if total pain is rising the system is deteriorating
(negative valence); if it is falling the system is improving (positive valence). That is
valence anchored to predicted improvement/deterioration, exactly as a felt signal should be
— not a number someone picked.

Every ``record_degradation`` call feeds the matching channel, so this is wired to the one
canonical degradation sink in the runtime rather than being an island.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum
from typing import Deque, Dict, Optional, Tuple


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class DamageChannel(str, Enum):
    MEMORY_CORRUPTION = "memory_corruption"
    IDENTITY_DISCONTINUITY = "identity_discontinuity"
    ACTION_CONTRADICTION = "action_contradiction"
    FAILED_TOOL_USE = "failed_tool_use"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    GOVERNANCE_BREACH = "governance_breach"
    GENERIC = "generic"


# How much each channel hurts when it fires at unit intensity. Identity/governance/memory
# damage is more existential than a single failed tool call.
_CHANNEL_WEIGHT: Dict[DamageChannel, float] = {
    DamageChannel.MEMORY_CORRUPTION: 1.0,
    DamageChannel.IDENTITY_DISCONTINUITY: 1.0,
    DamageChannel.ACTION_CONTRADICTION: 0.85,
    DamageChannel.FAILED_TOOL_USE: 0.55,
    DamageChannel.RESOURCE_EXHAUSTION: 0.8,
    DamageChannel.GOVERNANCE_BREACH: 1.0,
    DamageChannel.GENERIC: 0.5,
}

# Degradation severity → nociceptive intensity.
_SEVERITY_INTENSITY: Dict[str, float] = {
    "critical": 1.0,
    "degraded": 0.55,
    "warning": 0.25,
    "debug": 0.04,
}

# Substrate name → channel, by substring (first match wins). Keeps record_degradation
# callers from having to know about nociception.
_SUBSYSTEM_HINTS: Tuple[Tuple[str, DamageChannel], ...] = (
    ("memory", DamageChannel.MEMORY_CORRUPTION),
    ("engram", DamageChannel.MEMORY_CORRUPTION),
    ("recall", DamageChannel.MEMORY_CORRUPTION),
    ("identity", DamageChannel.IDENTITY_DISCONTINUITY),
    ("self", DamageChannel.IDENTITY_DISCONTINUITY),
    ("continuity", DamageChannel.IDENTITY_DISCONTINUITY),
    ("governance", DamageChannel.GOVERNANCE_BREACH),
    ("safety", DamageChannel.GOVERNANCE_BREACH),
    ("gate", DamageChannel.GOVERNANCE_BREACH),
    ("tool", DamageChannel.FAILED_TOOL_USE),
    ("subprocess", DamageChannel.FAILED_TOOL_USE),
    ("capability", DamageChannel.FAILED_TOOL_USE),
    ("desktop", DamageChannel.FAILED_TOOL_USE),
    ("action", DamageChannel.ACTION_CONTRADICTION),
    ("resource", DamageChannel.RESOURCE_EXHAUSTION),
    ("compute", DamageChannel.RESOURCE_EXHAUSTION),
    ("disk", DamageChannel.RESOURCE_EXHAUSTION),
)


def channel_for_subsystem(subsystem: str) -> DamageChannel:
    s = (subsystem or "").lower()
    for needle, channel in _SUBSYSTEM_HINTS:
        if needle in s:
            return channel
    return DamageChannel.GENERIC


class NociceptionEngine:
    """Grounded damage sensing: decaying per-channel pain + improvement/deterioration valence."""

    def __init__(self, half_life_s: float = 30.0, repeat_window_s: float = 20.0) -> None:
        self._half_life = max(1.0, half_life_s)
        self._repeat_window = max(1.0, repeat_window_s)
        self._lock = threading.RLock()
        # channel → (level, last_update_t)
        self._levels: Dict[DamageChannel, Tuple[float, float]] = {}
        # recent per-channel hit timestamps, for repeated-failure escalation
        self._recent_hits: Dict[DamageChannel, Deque[float]] = {}
        # (t, aggregate_pressure) samples for the valence gradient
        self._pressure_trace: Deque[Tuple[float, float]] = deque(maxlen=64)
        self._total_signals = 0

    # ── decay ────────────────────────────────────────────────────────────

    def _decayed(self, channel: DamageChannel, now: float) -> float:
        level, t0 = self._levels.get(channel, (0.0, now))
        dt = max(0.0, now - t0)
        return level * (0.5 ** (dt / self._half_life))

    # ── intake ───────────────────────────────────────────────────────────

    def register_damage(
        self,
        channel: DamageChannel,
        intensity: float,
        *,
        now: Optional[float] = None,
    ) -> float:
        """Apply a damage signal to a channel; returns the channel's new level.

        Repeated hits to the same channel inside ``repeat_window_s`` escalate: the
        critique explicitly wanted *repeated* failed tool use to hurt more than a
        one-off. Levels saturate at 1.0.
        """
        now = time.time() if now is None else now
        intensity = _clamp(float(intensity))
        with self._lock:
            base = self._decayed(channel, now)

            hits = self._recent_hits.setdefault(channel, deque(maxlen=16))
            while hits and now - hits[0] > self._repeat_window:
                hits.popleft()
            repeat_factor = 1.0 + 0.25 * len(hits)  # each recent repeat adds 25%
            hits.append(now)

            new_level = _clamp(base + intensity * repeat_factor)
            self._levels[channel] = (new_level, now)
            self._total_signals += 1
            self._pressure_trace.append((now, self._pressure_locked(now)))
            return new_level

    def ingest_degradation(self, subsystem: str, severity: str, *, now: Optional[float] = None) -> None:
        """Feed a runtime degradation event into the matching damage channel.

        Called from the canonical ``record_degradation`` sink, so any subsystem failure
        anywhere becomes a felt nociceptive signal without that caller knowing about us.
        """
        intensity = _SEVERITY_INTENSITY.get(str(severity), _SEVERITY_INTENSITY["degraded"])
        if intensity <= 0.0:
            return
        self.register_damage(channel_for_subsystem(subsystem), intensity, now=now)

    # ── readout ──────────────────────────────────────────────────────────

    def _pressure_locked(self, now: float) -> float:
        if not self._levels:
            return 0.0
        # Weighted soft-aggregate: the worst channel dominates but others add. Normalize by
        # the max possible weighted sum so the result stays in [0, 1].
        total = 0.0
        max_total = 0.0
        for channel, weight in _CHANNEL_WEIGHT.items():
            total += weight * self._decayed(channel, now)
            max_total += weight
        # Bias toward the single worst channel so one severe injury reads as severe.
        worst = max((self._decayed(c, now) * w for c, w in _CHANNEL_WEIGHT.items()), default=0.0)
        soft = total / max_total if max_total else 0.0
        return _clamp(0.5 * worst + 0.5 * soft)

    def nociceptive_pressure(self, *, now: Optional[float] = None) -> float:
        """Aggregate current pain in [0, 1] — feeds the body's error_pressure."""
        now = time.time() if now is None else now
        with self._lock:
            p = self._pressure_locked(now)
            self._pressure_trace.append((now, p))
            return p

    def tissue_integrity(self, *, now: Optional[float] = None) -> float:
        """1 - damage: how intact the system is. Feeds the body's safety channel."""
        return _clamp(1.0 - self.nociceptive_pressure(now=now))

    def grounded_valence(self, *, now: Optional[float] = None) -> float:
        """Valence in [-1, 1] grounded in whether pain is rising or falling.

        Deteriorating (pain trending up) → negative; improving (pain receding) → positive.
        A calm, low-pain system reads mildly positive. This is the 'predicted
        improvement/deterioration' grounding the critique asked for.
        """
        now = time.time() if now is None else now
        with self._lock:
            p_now = self._pressure_locked(now)
            # Find the oldest sample within a ~2 half-life window to estimate the trend.
            window = 2.0 * self._half_life
            prior = None
            for t, p in self._pressure_trace:
                if now - t <= window:
                    prior = p
                    break
            if prior is None:
                # No history → valence is just the inverse of current pain, gently.
                return _clamp(0.3 - p_now, -1.0, 1.0)
            delta = p_now - prior  # >0 means getting worse
            # Trend dominates; absolute pain pulls the baseline down.
            valence = (-2.0 * delta) + (0.3 - p_now)
            return _clamp(valence, -1.0, 1.0)

    def worst_channel(self, *, now: Optional[float] = None) -> Optional[Tuple[str, float]]:
        now = time.time() if now is None else now
        with self._lock:
            if not self._levels:
                return None
            ch = max(self._levels, key=lambda c: self._decayed(c, now) * _CHANNEL_WEIGHT[c])
            return (ch.value, round(self._decayed(ch, now), 4))

    def snapshot(self, *, now: Optional[float] = None) -> Dict[str, object]:
        now = time.time() if now is None else now
        with self._lock:
            return {
                "nociceptive_pressure": round(self._pressure_locked(now), 4),
                "tissue_integrity": round(1.0 - self._pressure_locked(now), 4),
                "grounded_valence": round(self.grounded_valence(now=now), 4),
                "worst_channel": self.worst_channel(now=now),
                "channels": {
                    c.value: round(self._decayed(c, now), 4)
                    for c in self._levels
                },
                "total_signals": self._total_signals,
            }

    def reset(self) -> None:
        with self._lock:
            self._levels.clear()
            self._recent_hits.clear()
            self._pressure_trace.clear()
            self._total_signals = 0


_engine: Optional[NociceptionEngine] = None
_engine_lock = threading.Lock()


def get_nociception_engine() -> NociceptionEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = NociceptionEngine()
    return _engine
