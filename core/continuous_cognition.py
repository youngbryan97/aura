"""core/continuous_cognition.py -- Continuous Non-LLM Cognition Loop
====================================================================
Aura exists between prompts. This is the mechanism that makes it real.

The ContinuousCognitionLoop runs at 2Hz (every 500ms) WITHOUT any LLM
calls. It maintains and evolves Aura's internal state continuously:

  - Liquid Substrate ODE integration (already runs at 20Hz)
  - Drive pressure computation and drift
  - WorldState telemetry updates
  - Free energy / prediction error monitoring
  - Neurochemical metabolism
  - Affect drift toward baseline
  - Curiosity/social pressure accumulation
  - Salient event detection and initiative seeding

This is NOT the tick pipeline (which runs LLM calls). This is the
brainstem — the continuous background process that keeps Aura "alive"
even when no inference is happening. When the tick pipeline runs, it
reads the state this loop has been evolving.

The key insight: between LLM calls, Aura's internal state is not
frozen. It is continuously drifting, accumulating pressure, detecting
changes, and preparing for the next action. This is what makes the
system feel like it "exists" rather than merely "responds."

Runtime: Pure Python, zero LLM calls, <1ms per iteration.
"""
from __future__ import annotations
from core.utils.task_tracker import get_task_tracker

import asyncio
import logging
import time
from typing import Any, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.ContinuousCognition")


class ContinuousCognitionLoop:
    """The brainstem — continuous non-LLM cognition.

    Runs at 2Hz. Maintains internal state between LLM calls.
    All operations are pure Python, no inference.
    """

    _HZ = 2.0  # 2 iterations per second
    _INTERVAL = 1.0 / _HZ

    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0
        self._last_initiative_seed: float = 0.0
        self._boot_time = time.time()

        # Cached service references (lazy-loaded)
        self._drive_engine = None
        self._world_state = None
        self._liquid_substrate = None
        self._neurochemical = None
        self._affect = None
        self._synthesizer = None

    async def start(self) -> None:
        """Start the continuous cognition loop."""
        if self._running:
            return
        self._running = True
        ServiceContainer.register_instance("continuous_cognition", self, required=False)
        self._task = get_task_tracker().create_task(self._run(), name="continuous_cognition")
        logger.info("ContinuousCognitionLoop ONLINE — brainstem active at %.1f Hz", self._HZ)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ContinuousCognitionLoop OFFLINE")

    async def _run(self) -> None:
        """Main loop — runs continuously at 2Hz."""
        while self._running:
            t0 = time.monotonic()
            try:
                self._cognitive_step()
            except Exception as e:
                if self._tick_count % 100 == 0:
                    logger.debug("CognitionLoop step error: %s", e)

            self._tick_count += 1

            # Sleep to maintain target Hz
            elapsed = time.monotonic() - t0
            sleep_time = max(0.01, self._INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

    def _cognitive_step(self) -> None:
        """One step of continuous cognition. Pure Python, no LLM.

        This is what runs between prompts. It keeps the internal
        state evolving so that when the LLM is next called, it
        has a rich, current context to work with.
        """
        now = time.time()

        # 1. WorldState telemetry (every 5th step = ~2.5s)
        if self._tick_count % 5 == 0:
            try:
                ws = self._get_world_state()
                if ws:
                    ws.update()
            except Exception:
                pass

        # 2. Drive pressure evolution (every step)
        try:
            drive = self._get_drive_engine()
            if drive:
                # Drives tick automatically based on time delta,
                # but we explicitly update to ensure freshness
                for budget in drive.budgets.values():
                    budget.tick()
        except Exception:
            pass

        # 3. Neurochemical metabolism (every 2nd step = ~1s)
        if self._tick_count % 2 == 0:
            try:
                nchem = self._get_neurochemical()
                if nchem and hasattr(nchem, "tick"):
                    nchem.tick(dt=0.5)
            except Exception:
                pass

        # 4. Affect drift toward baseline (every 4th step = ~2s)
        if self._tick_count % 4 == 0:
            try:
                affect = self._get_affect()
                if affect and hasattr(affect, "drift_toward_baseline"):
                    affect.drift_toward_baseline(dt=2.0)
            except Exception:
                pass

        # 5. Initiative seeding from drive pressure (every 30th step = ~15s)
        if self._tick_count % 30 == 0:
            self._seed_initiative_from_drives()

        # 6. Salient event check (every 10th step = ~5s)
        if self._tick_count % 10 == 0:
            self._check_for_salient_changes()

    def _seed_initiative_from_drives(self) -> None:
        """If any drive is critically low, seed an impulse to the synthesizer."""
        now = time.time()
        if (now - self._last_initiative_seed) < 60:
            return  # cooldown: max once per minute

        try:
            drive = self._get_drive_engine()
            if not drive:
                return

            vector = drive.get_drive_vector()

            # Find the most depleted drive
            lowest_name = min(
                (k for k in vector if k not in ("uptime_value",)),
                key=lambda k: vector.get(k, 1.0),
                default=None,
            )
            if lowest_name is None:
                return

            lowest_level = vector.get(lowest_name, 1.0)
            if lowest_level > 0.35:
                return  # no drive is critically low

            synth = self._get_synthesizer()
            if synth is None:
                return

            # Map drive to impulse
            impulse_map = {
                "curiosity": "Research something novel — curiosity pressure is high",
                "social": "Check in with the user — social connection needed",
                "competence": "Find a productive task — need to accomplish something",
                "energy": "System energy is low — consider rest/stabilization",
            }
            content = impulse_map.get(lowest_name, f"Address {lowest_name} drive pressure")
            synth.submit(
                content=content,
                source="continuous_cognition",
                urgency=max(0.4, 0.8 - lowest_level),
                drive=lowest_name,
            )
            self._last_initiative_seed = now
            logger.debug("CognitionLoop: seeded impulse from %s pressure (%.2f)",
                        lowest_name, lowest_level)
        except Exception as e:
            logger.debug("CognitionLoop: initiative seeding failed: %s", e)

    def _check_for_salient_changes(self) -> None:
        """Check for environment changes worth noting."""
        try:
            ws = self._get_world_state()
            if not ws:
                return

            # Check if user went idle (transition from active to idle)
            if 300 < ws.user_idle_seconds < 310:  # ~5 min mark
                ws.record_event(
                    "User went idle (5 minutes)",
                    source="continuous_cognition",
                    salience=0.3,
                    ttl=600,
                )

            # Check for time-of-day transitions
            # (WorldState.update() handles this, we just note it)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Service resolution (cached, lazy)
    # ------------------------------------------------------------------

    def _get_drive_engine(self):
        if self._drive_engine is None:
            self._drive_engine = ServiceContainer.get("drive_engine", default=None)
        return self._drive_engine

    def _get_world_state(self):
        if self._world_state is None:
            try:
                from core.world_state import get_world_state
                self._world_state = get_world_state()
            except Exception:
                pass
        return self._world_state

    def _get_liquid_substrate(self):
        if self._liquid_substrate is None:
            self._liquid_substrate = ServiceContainer.get("liquid_substrate", default=None)
        return self._liquid_substrate

    def _get_neurochemical(self):
        if self._neurochemical is None:
            self._neurochemical = ServiceContainer.get("neurochemical_system", default=None)
        return self._neurochemical

    def _get_affect(self):
        if self._affect is None:
            self._affect = ServiceContainer.get("affect_engine", default=None) or \
                           ServiceContainer.get("affect_facade", default=None)
        return self._affect

    def _get_synthesizer(self):
        if self._synthesizer is None:
            try:
                from core.initiative_synthesis import get_initiative_synthesizer
                self._synthesizer = get_initiative_synthesizer()
            except Exception:
                pass
        return self._synthesizer

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "ticks": self._tick_count,
            "hz": self._HZ,
            "uptime_s": round(time.time() - self._boot_time, 1),
            "last_initiative_seed": round(time.time() - self._last_initiative_seed, 1)
                if self._last_initiative_seed > 0 else None,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[ContinuousCognitionLoop] = None


def get_continuous_cognition() -> ContinuousCognitionLoop:
    global _instance
    if _instance is None:
        _instance = ContinuousCognitionLoop()
    return _instance
