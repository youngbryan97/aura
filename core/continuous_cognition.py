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

import asyncio
import logging
import time

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ContinuousCognition")

_CONTINUOUS_COGNITION_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    ConnectionError,
)
_MAX_LOOP_BACKOFF_SECONDS = 5.0


def _record_continuous_cognition_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "continuous_cognition",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("continuous_cognition", error)


def _finite_float(raw: object, default: float) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if value != value or value in (float("inf"), float("-inf")):
        return default, False
    return value, True


class ContinuousCognitionLoop:
    """The brainstem — continuous non-LLM cognition.

    Runs at 2Hz. Maintains internal state between LLM calls.
    All operations are pure Python, no inference.
    """

    _HZ = 2.0  # 2 iterations per second
    _INTERVAL = 1.0 / _HZ

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_count: int = 0
        self._last_initiative_seed: float = 0.0
        self._boot_time = time.monotonic()
        self._consecutive_loop_failures: int = 0
        self._last_loop_error_at: float = 0.0
        self._last_step_duration_s: float = 0.0

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
        try:
            ServiceContainer.register_instance("continuous_cognition", self, required=False)
        except _CONTINUOUS_COGNITION_ERRORS as exc:
            _record_continuous_cognition_degradation(
                exc,
                action="continued ContinuousCognition startup without ServiceContainer registration",
                severity="critical",
            )

        run_loop = self._run()
        try:
            self._task = get_task_tracker().create_task(
                run_loop,
                name="continuous_cognition",
            )
        except _CONTINUOUS_COGNITION_ERRORS as exc:
            run_loop.close()
            self._running = False
            self._task = None
            _record_continuous_cognition_degradation(
                exc,
                action="failed closed when ContinuousCognition task creation failed",
                severity="critical",
            )
            raise
        logger.info("ContinuousCognitionLoop ONLINE — brainstem active at %.1f Hz", self._HZ)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("ContinuousCognitionLoop task cancellation acknowledged")
            self._task = None
        logger.info("ContinuousCognitionLoop OFFLINE")

    async def _run(self) -> None:
        """Main loop — runs continuously at 2Hz."""
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    self._cognitive_step()
                    self._consecutive_loop_failures = 0
                except _CONTINUOUS_COGNITION_ERRORS as exc:
                    self._consecutive_loop_failures += 1
                    self._last_loop_error_at = time.monotonic()
                    _record_continuous_cognition_degradation(
                        exc,
                        action=(
                            "kept ContinuousCognition loop alive after step failure "
                            "and cleared cached service references"
                        ),
                        extra={
                            "consecutive_loop_failures": self._consecutive_loop_failures
                        },
                    )
                    self._reset_cached_services()
                    if self._tick_count % 100 == 0:
                        logger.debug("CognitionLoop step error: %s", exc)

                self._tick_count += 1

                elapsed = time.monotonic() - t0
                self._last_step_duration_s = max(0.0, elapsed)
                backoff = min(
                    self._INTERVAL * max(0, self._consecutive_loop_failures),
                    _MAX_LOOP_BACKOFF_SECONDS,
                )
                sleep_time = max(0.01, self._INTERVAL + backoff - elapsed)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.debug("ContinuousCognitionLoop run loop cancelled")
        finally:
            self._running = False

    def _cognitive_step(self) -> None:
        """One step of continuous cognition. Pure Python, no LLM.

        This is what runs between prompts. It keeps the internal
        state evolving so that when the LLM is next called, it
        has a rich, current context to work with.
        """
        # 1. WorldState telemetry (every 5th step = ~2.5s)
        if self._tick_count % 5 == 0:
            try:
                ws = self._get_world_state()
                if ws:
                    ws.update()
            except _CONTINUOUS_COGNITION_ERRORS as exc:
                _record_continuous_cognition_degradation(
                    exc,
                    action="skipped ContinuousCognition world-state update after dependency failure",
                    severity="warning",
                )
                logger.debug("CognitionLoop: world-state update failed: %s", exc)

        # 2. Drive pressure evolution (every step)
        try:
            drive = self._get_drive_engine()
            if drive:
                # Drives tick automatically based on time delta,
                # but we explicitly update to ensure freshness
                budgets = getattr(drive, "budgets", {}) or {}
                for budget in budgets.values():
                    budget.tick()
        except _CONTINUOUS_COGNITION_ERRORS as exc:
            _record_continuous_cognition_degradation(
                exc,
                action="skipped ContinuousCognition drive-pressure update after dependency failure",
                severity="warning",
            )
            logger.debug("CognitionLoop: drive pressure update failed: %s", exc)

        # 3. Neurochemical metabolism (every 2nd step = ~1s)
        if self._tick_count % 2 == 0:
            try:
                nchem = self._get_neurochemical()
                if nchem and hasattr(nchem, "tick"):
                    nchem.tick(dt=0.5)
            except _CONTINUOUS_COGNITION_ERRORS as exc:
                _record_continuous_cognition_degradation(
                    exc,
                    action="skipped ContinuousCognition neurochemical tick after dependency failure",
                    severity="warning",
                )
                logger.debug("CognitionLoop: neurochemical tick failed: %s", exc)

        # 4. Affect drift toward baseline (every 4th step = ~2s)
        if self._tick_count % 4 == 0:
            try:
                affect = self._get_affect()
                if affect and hasattr(affect, "drift_toward_baseline"):
                    affect.drift_toward_baseline(dt=2.0)
            except _CONTINUOUS_COGNITION_ERRORS as exc:
                _record_continuous_cognition_degradation(
                    exc,
                    action="skipped ContinuousCognition affect drift after dependency failure",
                    severity="warning",
                )
                logger.debug("CognitionLoop: affect drift failed: %s", exc)

        # 5. Initiative seeding from drive pressure (every 30th step = ~15s)
        if self._tick_count % 30 == 0:
            self._seed_initiative_from_drives()

        # 6. Salient event check (every 10th step = ~5s)
        if self._tick_count % 10 == 0:
            self._check_for_salient_changes()

    def _seed_initiative_from_drives(self) -> None:
        """If any drive is critically low, seed an impulse to the synthesizer."""
        now = time.monotonic()
        if (now - self._last_initiative_seed) < 60:
            return  # cooldown: max once per minute

        try:
            drive = self._get_drive_engine()
            if not drive:
                return

            vector = drive.get_drive_vector()
            if not isinstance(vector, dict):
                raise TypeError("drive vector must be a dictionary")

            # Find the most depleted drive
            lowest_name = min(
                (k for k in vector if k not in ("uptime_value",)),
                key=lambda k: _finite_float(vector.get(k, 1.0), 1.0)[0],
                default=None,
            )
            if lowest_name is None:
                return

            lowest_level, valid_level = _finite_float(vector.get(lowest_name, 1.0), 1.0)
            if not valid_level:
                _record_continuous_cognition_degradation(
                    ValueError(f"invalid drive level for {lowest_name!r}"),
                    action="ignored malformed drive pressure while seeding initiative",
                    severity="warning",
                    extra={"drive": str(lowest_name)},
                )
                return
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
        except _CONTINUOUS_COGNITION_ERRORS as e:
            _record_continuous_cognition_degradation(
                e,
                action="skipped ContinuousCognition initiative seeding after dependency failure",
                severity="warning",
            )
            logger.debug("CognitionLoop: initiative seeding failed: %s", e)

    def _check_for_salient_changes(self) -> None:
        """Check for environment changes worth noting."""
        try:
            ws = self._get_world_state()
            if not ws:
                return

            # Check if user went idle (transition from active to idle)
            idle_seconds, valid_idle = _finite_float(
                getattr(ws, "user_idle_seconds", 0.0),
                0.0,
            )
            if not valid_idle:
                return
            if 300 < idle_seconds < 310:  # ~5 min mark
                ws.record_event(
                    "User went idle (5 minutes)",
                    source="continuous_cognition",
                    salience=0.3,
                    ttl=600,
                )

            # Check for time-of-day transitions
            # (WorldState.update() handles this, we just note it)
        except _CONTINUOUS_COGNITION_ERRORS as exc:
            _record_continuous_cognition_degradation(
                exc,
                action="skipped ContinuousCognition salient-change check after dependency failure",
                severity="warning",
            )
            logger.debug("CognitionLoop: salient-change check failed: %s", exc)

    # ------------------------------------------------------------------
    # Service resolution (cached, lazy)
    # ------------------------------------------------------------------

    def _reset_cached_services(self) -> None:
        self._drive_engine = None
        self._world_state = None
        self._liquid_substrate = None
        self._neurochemical = None
        self._affect = None
        self._synthesizer = None

    def _service(self, name: str):
        try:
            return ServiceContainer.get(name, default=None)
        except _CONTINUOUS_COGNITION_ERRORS as exc:
            _record_continuous_cognition_degradation(
                exc,
                action=f"treated optional ContinuousCognition service {name} as unavailable",
                severity="warning",
                extra={"service": name},
            )
            return None

    def _get_drive_engine(self):
        if self._drive_engine is None:
            self._drive_engine = self._service("drive_engine")
        return self._drive_engine

    def _get_world_state(self):
        if self._world_state is None:
            try:
                from core.world_state import get_world_state
                self._world_state = get_world_state()
            except _CONTINUOUS_COGNITION_ERRORS as exc:
                _record_continuous_cognition_degradation(
                    exc,
                    action="treated world-state service as unavailable for ContinuousCognition",
                    severity="warning",
                )
                logger.debug("CognitionLoop: world-state service unavailable: %s", exc)
        return self._world_state

    def _get_liquid_substrate(self):
        if self._liquid_substrate is None:
            self._liquid_substrate = self._service("liquid_substrate")
        return self._liquid_substrate

    def _get_neurochemical(self):
        if self._neurochemical is None:
            self._neurochemical = self._service("neurochemical_system")
        return self._neurochemical

    def _get_affect(self):
        if self._affect is None:
            self._affect = self._service("affect_engine") or self._service("affect_facade")
        return self._affect

    def _get_synthesizer(self):
        if self._synthesizer is None:
            try:
                from core.initiative_synthesis import get_initiative_synthesizer
                self._synthesizer = get_initiative_synthesizer()
            except _CONTINUOUS_COGNITION_ERRORS as exc:
                _record_continuous_cognition_degradation(
                    exc,
                    action="treated initiative synthesizer as unavailable for ContinuousCognition",
                    severity="warning",
                )
                logger.debug("CognitionLoop: initiative synthesizer unavailable: %s", exc)
        return self._synthesizer

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "ticks": self._tick_count,
            "hz": self._HZ,
            "uptime_s": round(max(0.0, time.monotonic() - self._boot_time), 1),
            "last_initiative_seed": round(time.monotonic() - self._last_initiative_seed, 1)
                if self._last_initiative_seed > 0 else None,
            "consecutive_loop_failures": self._consecutive_loop_failures,
            "last_loop_error_at": round(self._last_loop_error_at, 4),
            "last_step_duration_s": round(self._last_step_duration_s, 4),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ContinuousCognitionLoop | None = None


def get_continuous_cognition() -> ContinuousCognitionLoop:
    global _instance
    if _instance is None:
        _instance = ContinuousCognitionLoop()
    return _instance
