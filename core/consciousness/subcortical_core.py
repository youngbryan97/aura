"""
Subcortical Core — Thalamo-Reticular Arousal System

Based on 2025 neuroscience reviews challenging cortex-centric theories:
consciousness may not require cortex at all in some cases. Thalamic gating
and brainstem arousal are major contributors that most AI architectures
(including previous Aura versions) undervalue.

This module provides:
1. Thalamic gate: controls which content can ENTER the neural mesh at all
   (not just which content wins workspace competition — that's GWT's job)
2. Brainstem arousal: modulates overall gain of the substrate tick
3. Reticular activation: baseline vigilance level that affects all processing

Integration:
- Feeds into liquid_substrate via gain modulation
- Gates neural_mesh sensory input (low arousal = reduced mesh activation)
- Boosts neurochemical_system baseline (norepinephrine, acetylcholine)
- Modulates heartbeat tick rate (low arousal = slower background processing)

Runtime benefit: arousal gating during idle periods reduces compute load
without losing architectural continuity. This is biologically realistic
AND good for indefinite runtime on consumer hardware.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("Consciousness.SubcorticalCore")


@dataclass
class ArousalState:
    """Current state of the subcortical arousal system."""
    arousal_level: float = 0.5       # 0=deep sleep, 1=peak vigilance
    thalamic_gate: float = 0.5       # 0=fully closed (nothing enters mesh), 1=fully open
    reticular_activation: float = 0.3  # baseline vigilance
    brainstem_drive: float = 0.4     # raw arousal drive before modulation
    timestamp: float = 0.0


class SubcorticalCore:
    """Thalamo-reticular arousal system for the consciousness stack.

    The subcortical core sits BELOW the cortical systems (mesh, workspace, phi)
    and controls their overall activation level. It's the difference between
    being awake, drowsy, and unconscious.

    Key dynamics:
    - External stimuli (user input, sensory events) raise brainstem drive
    - Brainstem drive decays exponentially toward baseline when unstimulated
    - Thalamic gate opens proportionally to arousal level
    - Low arousal = mesh receives attenuated input = reduced compute
    - High arousal = full mesh activation = full cognitive capacity
    """

    def __init__(self):
        self._arousal: float = 0.5
        self._brainstem_drive: float = 0.4
        self._reticular_baseline: float = 0.3
        self._thalamic_gate: float = 0.5
        self._last_stimulus_time: float = time.time()
        self._history: deque[ArousalState] = deque(maxlen=120)

        # Decay and response parameters
        self._arousal_decay_rate: float = 0.02     # per second
        self._stimulus_response_gain: float = 0.3  # how much a stimulus raises arousal
        self._gate_sensitivity: float = 1.5        # how sharply the gate responds to arousal
        self._min_gate: float = 0.05               # gate never fully closes (prevents total shutdown)

        logger.info("SubcorticalCore initialized (thalamic arousal gating active).")

    def receive_stimulus(self, intensity: float = 1.0, source: str = "external"):
        """Signal that a stimulus has arrived (user input, sensory event, etc).

        This raises brainstem drive, which in turn raises arousal and opens
        the thalamic gate, allowing more content to enter the mesh.
        """
        self._last_stimulus_time = time.time()
        boost = min(0.5, self._stimulus_response_gain * intensity)
        self._brainstem_drive = min(1.0, self._brainstem_drive + boost)
        logger.debug(
            "SubcorticalCore: stimulus received (source=%s, intensity=%.2f, drive→%.2f)",
            source, intensity, self._brainstem_drive,
        )

    def tick(self, dt: float = 1.0) -> ArousalState:
        """Update arousal state. Called once per cognitive tick.

        Returns the current arousal state for downstream consumers.
        """
        now = time.time()
        idle_seconds = now - self._last_stimulus_time

        # Brainstem drive decays toward reticular baseline
        decay = self._arousal_decay_rate * dt
        self._brainstem_drive = max(
            self._reticular_baseline,
            self._brainstem_drive - decay,
        )

        # Arousal is smoothed brainstem drive + reticular baseline
        target_arousal = min(1.0, self._brainstem_drive * 0.7 + self._reticular_baseline * 0.3)
        # Smooth transition (EMA)
        alpha = min(1.0, 0.1 * dt)
        self._arousal = self._arousal * (1.0 - alpha) + target_arousal * alpha

        # Thalamic gate: sigmoid response to arousal
        # Low arousal → nearly closed (but never fully, per _min_gate)
        # High arousal → fully open
        gate_input = (self._arousal - 0.3) * self._gate_sensitivity
        self._thalamic_gate = max(
            self._min_gate,
            1.0 / (1.0 + math.exp(-gate_input)),
        )

        # Idle-specific behavior: if no stimulus for >5 min, lower baseline
        if idle_seconds > 300.0:
            # Long idle: reduce reticular baseline (drowsy)
            self._reticular_baseline = max(0.1, self._reticular_baseline - 0.001 * dt)
        elif idle_seconds < 30.0:
            # Recent stimulus: restore reticular baseline
            self._reticular_baseline = min(0.4, self._reticular_baseline + 0.005 * dt)

        state = ArousalState(
            arousal_level=round(self._arousal, 4),
            thalamic_gate=round(self._thalamic_gate, 4),
            reticular_activation=round(self._reticular_baseline, 4),
            brainstem_drive=round(self._brainstem_drive, 4),
            timestamp=now,
        )
        self._history.append(state)
        return state

    def get_mesh_gain_multiplier(self) -> float:
        """Return the gain multiplier for the neural mesh.

        When arousal is low, mesh activation is attenuated → less compute.
        When arousal is high, mesh runs at full capacity.
        """
        return max(0.1, self._thalamic_gate)

    def get_substrate_gain_multiplier(self) -> float:
        """Return the gain multiplier for the liquid substrate.

        Similar to mesh, but the substrate never drops below 30% capacity
        (it must maintain continuity even during deep idle).
        """
        return max(0.3, self._arousal)

    def get_heartbeat_rate_multiplier(self) -> float:
        """Return a multiplier for the heartbeat tick rate.

        Low arousal → slower background processing (energy conservation).
        High arousal → faster processing (full cognitive engagement).
        """
        if self._arousal < 0.2:
            return 0.5   # Half-rate background ticks
        elif self._arousal < 0.4:
            return 0.75
        elif self._arousal > 0.8:
            return 1.25  # Slightly faster for high vigilance
        return 1.0

    def get_context_block(self) -> str:
        """Short context block for cognition injection."""
        if self._arousal < 0.3:
            return "## AROUSAL STATE\nDrowsy — processing at reduced capacity. Conserving energy."
        elif self._arousal > 0.8:
            return "## AROUSAL STATE\nHighly vigilant — all systems at peak capacity."
        return ""

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "arousal_level": round(self._arousal, 4),
            "thalamic_gate": round(self._thalamic_gate, 4),
            "reticular_baseline": round(self._reticular_baseline, 4),
            "brainstem_drive": round(self._brainstem_drive, 4),
            "idle_seconds": round(time.time() - self._last_stimulus_time, 1),
            "mesh_gain": round(self.get_mesh_gain_multiplier(), 4),
            "substrate_gain": round(self.get_substrate_gain_multiplier(), 4),
            "heartbeat_rate_mult": self.get_heartbeat_rate_multiplier(),
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[SubcorticalCore] = None


def get_subcortical_core() -> SubcorticalCore:
    global _instance
    if _instance is None:
        _instance = SubcorticalCore()
    return _instance
