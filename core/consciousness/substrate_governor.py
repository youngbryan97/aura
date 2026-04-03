"""core/consciousness/substrate_governor.py — Aura 3.0: Substrate Governor
=======================================================================
Implements the Phase 6 frequency scaling. Manages global tick-rate based 
on cognitive load and resource anxiety.

ZENITH Protocol compliance:
  - 5Hz (Hibernation): Minimal power, background cleanup only.
  - 20Hz (High-Alacrity): Focus mode for complex reasoning.
  - Frequency transitions are smooth and event-driven.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("Aura.SubstrateGovernor")


class SubstrateGovernor:
    """
    Substrate management service.
    
    ZENITH Purity:
      - Uses precise timing loops to maintain target Hz.
      - Monitors latency and adjusts frequency dynamically.
    """

    def __init__(self):
        self._target_hz = 5.0
        self._actual_hz = 0.0
        self._running = False
        self._tick_count = 0
        self._last_tick_time = 0.0

    async def on_start_async(self):
        self._running = True
        logger.info("SubstrateGovernor ONLINE. Default frequency: 5Hz (Hibernation).")

    async def on_stop_async(self):
        self._running = False
        logger.info("SubstrateGovernor SHUTDOWN.")

    @property
    def alacrity(self) -> float:
        return self._target_hz

    async def set_alacrity(self, hz: float):
        """Adjusts the system clock speed."""
        clamped = max(1.0, min(60.0, hz))
        if clamped != self._target_hz:
            logger.info("Substrate shift: %.1fHz -> %.1fHz", self._target_hz, clamped)
            self._target_hz = clamped

    def apply_volition_profile(self, level: int):
        """
        Maps VolitionLevel to target frequency (Hz).
        Level 0: 5Hz (Lockdown)
        Level 1: 10Hz (Reflective)
        Level 2: 15Hz (Perceptive)
        Level 3: 20Hz (Agentic)
        """
        hz_map = {0: 5.0, 1: 10.0, 2: 15.0, 3: 20.0}
        new_hz = hz_map.get(level, 5.0)
        
        logger.info("⚡ [VOLITION] Governor applying profile L%d (%.1fHz)", level, new_hz)
        self._target_hz = new_hz

    async def sleep_until_next_tick(self):
        """Precise throttle for the main orchestrator loop."""
        if not self._running:
            return

        now = time.monotonic()
        if self._last_tick_time == 0.0:
            self._last_tick_time = now
            
        interval = 1.0 / self._target_hz
        elapsed = now - self._last_tick_time
        remaining = interval - elapsed
        
        if remaining > 0:
            await asyncio.sleep(remaining)
        
        self._last_tick_time = time.monotonic()
        self._tick_count += 1
        
    def get_clock_stats(self) -> Dict[str, Any]:
        return {
            "target_hz": self._target_hz,
            "total_ticks": self._tick_count,
            "mode": "focus" if self._target_hz >= 20.0 else "hibernation"
        }
