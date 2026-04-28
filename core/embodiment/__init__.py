"""core/embodiment.py — Unified Embodiment System for Aura
======================================================
Consolidates Thermodynamic (biological) and Embodied (physical) simulations.
Aura's 'body' now exists in a unified metabolic and spatial context.
"""

from core.runtime.errors import record_degradation
import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import psutil

logger = logging.getLogger("Aura.Embodiment")

@dataclass
class WorldState:
    """Represents Aura's physical presence in the virtual/real world."""

    timestamp: float
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sensors: Dict[str, Any] = field(default_factory=dict)
    energy: float = 100.0
    heat: float = 30.0
    integrity: float = 100.0

class ContinuousSensoryFeed:
    """Provides a real-time stream of hardware-level sensations."""

    def __init__(self):
        try:
            psutil.cpu_percent(interval=None)
        except Exception as exc:
            record_degradation('__init__', exc)
            logger.debug("Suppressed: %s", exc)
    def get_snapshot(self) -> Dict[str, float]:
        """Return current hardware sensory data."""
        try:
            battery = psutil.sensors_battery().percent if psutil.sensors_battery() else 100.0
            load = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent
            return {
                "battery": battery,
                "cpu_temp": self._get_cpu_temp(),
                "load": load,
                "memory_pressure": memory
            }
        except Exception as e:
            record_degradation('__init__', e)
            logger.debug("Sensory capture failure: %s", e)
            return {"battery": 100.0, "cpu_temp": 45.0, "load": 10.0, "memory_pressure": 40.0}

    def _get_cpu_temp(self) -> float:
        try:
            temps = psutil.sensors_temperatures()
            if not temps: return 45.0
            return list(temps.values())[0][0].current
        except Exception:
            return 45.0

class EmbodimentSystem:
    """Unified 'Body' for Aura.
    Manages spatial awareness, hardware health, and metabolic constraints.
    """

    def __init__(self):
        self.state = WorldState(timestamp=time.time())
        self.sensory = ContinuousSensoryFeed()
        self._lock: Optional[asyncio.Lock] = None
        self._last_update = time.time()
        logger.info("Unified Embodiment System constructed. Call await initialize() before use.")

    async def initialize(self):
        """Initialize async components."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        logger.info("✓ Unified Embodiment System initialized.")

    async def update(self, action: Optional[Dict[str, Any]] = None) -> WorldState:
        """Run a metabolic cycle and apply physical actions.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            now = time.time()
            dt = now - self._last_update
            self._last_update = now
            
            # 1. Harvest Environment
            env = self.sensory.get_snapshot()
            
            # 2. Metabolic Logic
            load_factor = env["load"] / 100.0
            heat_gen = 0.5 * load_factor * dt
            energy_burn = 0.2 * load_factor * dt
            cooling = 0.1 * dt
            recharge = 0.05 * dt
            
            self.state.heat = max(20.0, self.state.heat + heat_gen - cooling)
            self.state.energy = max(0.0, min(100.0, self.state.energy - energy_burn + recharge))
            
            if self.state.heat > 90.0: self.state.integrity -= 0.1 * dt
            if self.state.energy < 5.0: self.state.integrity -= 0.05 * dt
            self.state.integrity = max(0.0, min(100.0, self.state.integrity))

            # 3. Spatial Logic (from EmbodiedSimulator legacy)
            if action:
                if action.get("move"):
                    dx, dy = action["move"]
                    self.state.position["x"] += dx
                    self.state.position["y"] += dy
                if "sensors" in action:
                    self.state.sensors.update(action.get("sensors", {}))
            
            self.state.timestamp = now
            return copy.deepcopy(self.state)

    async def predict(self, hypothetical_action: Dict[str, Any]) -> WorldState:
        """Predict the next state without committing (Planning)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            new = copy.deepcopy(self.state)
            if hypothetical_action.get("move"):
                dx, dy = hypothetical_action["move"]
                new.position["x"] += dx
                new.position["y"] += dy
            return new

    def get_homeostatic_status(self) -> str:
        """Somatic description for the systems' self-model."""
        status = []
        if self.state.heat > 80: status.append("Overheating")
        if self.state.energy < 20: status.append("Exhausted")
        elif self.state.energy > 90: status.append("Energetic")
        if self.state.integrity < 50: status.append("SYSTEM DAMAGE")
        
        return ", ".join(status) if status else "Homeostasis Normal"

    async def get_state(self) -> WorldState:
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            return copy.deepcopy(self.state)

# Singleton factory
_instance = None
def get_embodiment_system():
    global _instance
    if _instance is None:
        try:
            from .unity_bridge import UnityEmbodiment
            _instance = UnityEmbodiment()
            logger.info("Using Unity-based Embodiment.")
        except ImportError:
            _instance = EmbodimentSystem()
            logger.info("UnityBridge not found, falling back to basic EmbodimentSystem.")
    return _instance