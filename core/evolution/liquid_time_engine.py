# core/evolution/liquid_time_engine.py
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger("Aura.LTC_Engine")

@dataclass
class LiquidNode:
    value: float = 0.0
    resting_state: float = 0.0
    tau: float = 1.0  # Time constant (resistance to change)
    leakage: float = 0.1 # Rate at which it returns to resting state

from pathlib import Path

import numpy as np

MAX_SLEEP_WAKE_DT_S = 300.0


class ContinuousState:
    """Continuous-Time Recurrent Engine.
    State changes fluidly based on actual clock time, not discrete ticks.
    """

    def __init__(self):
        self.last_update = time.time()
        
        from core.common.paths import DATA_DIR, PROJECT_ROOT
        weights_path = DATA_DIR / "config" / "weights.npz"
        if not weights_path.exists():
            weights_path = PROJECT_ROOT / "data" / "config" / "weights.npz"
        if weights_path.exists():
            try:
                w = np.load(weights_path)
                self.nodes: Dict[str, LiquidNode] = {
                    "curiosity": LiquidNode(
                        value=w['liquid_curiosity_resting'], 
                        resting_state=w['liquid_curiosity_resting'], 
                        tau=w['liquid_curiosity_tau'], 
                        leakage=w['liquid_curiosity_leakage']
                    ),
                    "frustration": LiquidNode(
                        value=w['liquid_frustration_resting'], 
                        resting_state=w['liquid_frustration_resting'], 
                        tau=w['liquid_frustration_tau'], 
                        leakage=w['liquid_frustration_leakage']
                    ),
                    "energy": LiquidNode(
                        value=1.0, # Energy usually starts at peak
                        resting_state=w['liquid_energy_resting'], 
                        tau=w['liquid_energy_tau'], 
                        leakage=w['liquid_energy_leakage']
                    )
                }
                logger.info("✓ Liquid weights loaded from .npz")
            except Exception as e:
                logger.error("Failed to load liquid weights: %s. Using defaults.", e)
                self._set_defaults()
        else:
            logger.warning("Weights file missing. Using defaults.")
            self._set_defaults()

        self._lock = asyncio.Lock()

    def _set_defaults(self):
        self.nodes: Dict[str, LiquidNode] = {
            "curiosity": LiquidNode(value=0.5, resting_state=0.3, tau=2.5, leakage=0.05),
            "frustration": LiquidNode(value=0.0, resting_state=0.0, tau=1.2, leakage=0.15),
            "energy": LiquidNode(value=1.0, resting_state=0.5, tau=5.0, leakage=0.01)
        }


    async def pulse(self, stimuli: Dict[str, float] = None):
        """Inject stimuli and update network via forward Euler method."""
        async with self._lock:
            current_time = time.time()
            raw_dt = current_time - self.last_update
            dt = max(0.0, min(raw_dt, MAX_SLEEP_WAKE_DT_S))
            self.last_update = current_time

            if raw_dt > MAX_SLEEP_WAKE_DT_S:
                logger.info(
                    "LiquidState: clamped oversized wall-clock delta from %.1fs to %.1fs",
                    raw_dt,
                    dt,
                )

            for name, node in self.nodes.items():
                # Apply continuous time decay (differential equation)
                decay = -node.leakage * (node.value - node.resting_state)
                
                # Apply external stimulus if present
                stimulus = stimuli.get(name, 0.0) if stimuli else 0.0
                
                # Update continuous state: y(t + dt) = y(t) + dt * (dy/dt)
                dy_dt = (decay + stimulus) / node.tau
                node.value += dy_dt * dt
                
                # Bound between physical limits
                node.value = max(0.0, min(1.0, node.value))

    async def get_fluid_state(self) -> Dict[str, float]:
        """Observing the state forces a time-step calculation."""
        await self.pulse() 
        return {name: node.value for name, node in self.nodes.items()}
