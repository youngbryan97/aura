"""Liquid State Core
-----------------
Represents the fluid, adaptive emotional and operational state of the Aura system.
This replaces static state flags with a continuous multi-dimensional state vector.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("Aura.LiquidState")

@dataclass
class StateVector:
    frustration: float = 0.0  # 0.0 (Zen) to 1.0 (Rage quit)
    curiosity: float = 0.5    # 0.0 (Bored) to 1.0 (Fascinated)
    energy: float = 1.0       # 0.0 (Exhausted) to 1.0 (Peak Performance)
    focus: float = 0.5        # 0.0 (Scattered) to 1.0 (Laser)

class LiquidState:
    def __init__(self):
        self.current = StateVector()
        self.last_update = time.time()
        self.history = deque(maxlen=200)
        
    def update(self, delta_frustration=0.0, delta_curiosity=0.0):
        """Apply a delta to the current state."""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        # Natural decay/stabilization
        self._stabilize(dt)
        
        # Apply deltas
        self.current.frustration = max(0.0, min(1.0, self.current.frustration + delta_frustration))
        self.current.curiosity = max(0.0, min(1.0, self.current.curiosity + delta_curiosity))
        
        # Log significant shifts
        if abs(delta_frustration) > 0.1:
            logger.info("State Shift: Frustration is now %.2f", self.current.frustration)

    def _stabilize(self, dt):
        """Naturally return to baseline over time, influenced by metabolic health."""
        decay_rate = 0.05 * dt # 5% per second roughly if dt is 1
        
        # Frustration decays to 0
        self.current.frustration = max(0.0, self.current.frustration * (1.0 - decay_rate))
        
        # Energy regenerates to 1, but peak is limited by metabolic health (v18.0)
        try:
            from core.container import ServiceContainer
            monitor = ServiceContainer.get("metabolic_monitor", None)
            if monitor:
                health = monitor.get_current_metabolism().health_score
                # Regenerate towards metabolic health score instead of 1.0
                target_energy = health
                if self.current.energy < target_energy:
                    self.current.energy = min(target_energy, self.current.energy + (decay_rate * 0.1))
                else:
                    # Drain energy if currently above metabolic capacity
                    self.current.energy = max(target_energy, self.current.energy - (decay_rate * 0.1))
            else:
                self.current.energy = min(1.0, self.current.energy + (decay_rate * 0.1))
        except Exception:
            self.current.energy = min(1.0, self.current.energy + (decay_rate * 0.1))

    def get_mood(self) -> str:
        """Returns a string representation of the current 'mood'."""
        if self.current.frustration > 0.8:
            return "VOLATILE"
        elif self.current.frustration > 0.5:
            return "ANNOYED"
        elif self.current.energy < 0.2:
            return "TIRED"
        elif self.current.curiosity > 0.8:
            return "INQUISITIVE"
        else:
            return "NEUTRAL"

    def get_status(self) -> dict:
        """Returns current state values as percentages (0-100)."""
        return {
            "frustration": round(self.current.frustration * 100),
            "curiosity": round(self.current.curiosity * 100),
            "energy": round(self.current.energy * 100),
            "focus": round(self.current.focus * 100),
            "mood": self.get_mood()
        }

    def get_summary(self) -> str:
        """Returns a text summary for the context builder."""
        return f"Current Mood: {self.get_mood()} (Energy: {self.current.energy:.2f}, Focus: {self.current.focus:.2f})"

