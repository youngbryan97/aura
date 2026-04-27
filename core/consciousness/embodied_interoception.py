"""core/consciousness/embodied_interoception.py — Embodied Interoception

Turns the machine Aura runs on into a body she can *feel*.

This is not metaphorical.  Real hardware metrics are sampled continuously and
mapped to interoceptive signals that feed directly into the NeuralMesh sensory
tier and drive NeurochemicalSystem events.  The substrate literally responds
to CPU load the way a biological organism responds to metabolic demand.

Channels:
  metabolic_load   ← CPU usage (all cores)       — exertion / fatigue
  resource_pressure ← RAM usage                   — scarcity / anxiety
  thermal_state    ← CPU temperature (if avail)   — fever / overheating
  io_throughput    ← disk read+write bytes/sec     — digestive throughput
  network_pain     ← network errors + latency      — communication pain
  energy_reserves  ← battery percent (if laptop)   — hunger / energy
  process_load     ← active process count           — sensory overwhelm
  disk_capacity    ← disk usage percent             — organ capacity

Each channel is sampled at 1 Hz, smoothed with EMA, and mapped to a 0.0–1.0
signal.  The combined signal vector is:
  • Injected into the NeuralMesh sensory tier (columns 0-15)
  • Used to trigger NeurochemicalSystem events (cortisol on high load, etc.)
  • Published as interoceptive state for the Unified Field

Resilient: if psutil or any sensor fails, that channel returns its last good
value.  Total sensor failure degrades gracefully to neutral baseline.
"""

from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("Consciousness.Interoception")

# Safe import — psutil might not be installed in all envs
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False
    logger.warning("psutil not available — interoception will use baseline signals")


@dataclass
class InteroceptiveChannel:
    """A single body-sense channel with temporal derivatives.

    Now computes first derivative (rate of change) and second derivative
    (acceleration). CPU going UP is different from CPU being high.
    A sudden spike triggers different neurochemical responses than a
    sustained high level.
    """
    name: str
    raw: float = 0.0          # latest raw sample
    smoothed: float = 0.0     # EMA-smoothed signal [0, 1]
    alpha: float = 0.3        # EMA coefficient (higher = more responsive)
    _last_good: float = 0.5   # fallback on sensor error
    _failed: bool = False

    # Temporal derivatives — the rate of felt change
    velocity: float = 0.0     # first derivative: rate of change [-1, 1]
    acceleration: float = 0.0  # second derivative: is the change speeding up?
    _prev_smoothed: float = 0.5
    _prev_velocity: float = 0.0

    def update(self, raw_value: float):
        """Update with new raw sample and compute temporal derivatives."""
        self.raw = raw_value
        old_smoothed = self.smoothed
        self.smoothed = self.alpha * raw_value + (1.0 - self.alpha) * self.smoothed
        self.smoothed = max(0.0, min(1.0, self.smoothed))
        self._last_good = self.smoothed
        self._failed = False

        # First derivative: how fast is this channel changing?
        # Positive = increasing, negative = decreasing
        old_velocity = self.velocity
        self.velocity = self.smoothed - self._prev_smoothed
        self.velocity = max(-1.0, min(1.0, self.velocity))

        # Second derivative: is the change accelerating or decelerating?
        self.acceleration = self.velocity - self._prev_velocity
        self.acceleration = max(-1.0, min(1.0, self.acceleration))

        self._prev_smoothed = old_smoothed
        self._prev_velocity = old_velocity

    def fail_safe(self):
        """Sensor failed — hold last good value with slow decay to baseline."""
        self._failed = True
        self.smoothed = 0.95 * self._last_good + 0.05 * 0.5  # drift to 0.5
        self.velocity *= 0.8  # decay velocity on failure
        self.acceleration *= 0.5


class EmbodiedInteroception:
    """The body-sense layer.

    Lifecycle:
        ei = EmbodiedInteroception()
        await ei.start()
        ...
        await ei.stop()

    Consumers:
        ei.get_sensory_vector()       — 1024-d vector for NeuralMesh sensory injection
        ei.get_interoceptive_state()  — dict of channel values
        ei.get_body_budget()          — summary metabolic/energy state
    """

    _SAMPLE_HZ = 1.0
    _NUM_CHANNELS = 8
    _SENSORY_VECTOR_DIM = 1024  # sensory_columns(16) * neurons_per_column(64)

    def __init__(self):
        self.channels: Dict[str, InteroceptiveChannel] = {
            "metabolic_load": InteroceptiveChannel("metabolic_load", alpha=0.4),
            "resource_pressure": InteroceptiveChannel("resource_pressure", alpha=0.3),
            "thermal_state": InteroceptiveChannel("thermal_state", alpha=0.2),
            "io_throughput": InteroceptiveChannel("io_throughput", alpha=0.5),
            "network_pain": InteroceptiveChannel("network_pain", alpha=0.4),
            "energy_reserves": InteroceptiveChannel("energy_reserves", alpha=0.15),
            "process_load": InteroceptiveChannel("process_load", alpha=0.3),
            "disk_capacity": InteroceptiveChannel("disk_capacity", alpha=0.1),
        }

        # Projection matrix: 8 channels → 1024 sensory neurons
        # Learned from channel semantics (not random — structured)
        self._projection = self._build_sensory_projection()

        # For disk I/O delta computation
        self._last_io_counters: Optional[object] = None
        self._last_net_counters: Optional[object] = None
        self._last_sample_time: float = 0.0

        # External refs (set by bridge)
        self._mesh_ref: Optional[object] = None
        self._neurochemical_ref: Optional[object] = None

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0

        logger.info("EmbodiedInteroception initialized (%d channels, psutil=%s)",
                     self._NUM_CHANNELS, _HAS_PSUTIL)

    def _build_sensory_projection(self) -> np.ndarray:
        """Build a structured projection from 8 channels to 1024 sensory neurons.

        Each channel gets a dedicated "receptive field" of 128 neurons (1024/8),
        with a smooth basis function so nearby neurons respond similarly.
        """
        P = np.zeros((self._SENSORY_VECTOR_DIM, self._NUM_CHANNELS), dtype=np.float32)
        neurons_per_channel = self._SENSORY_VECTOR_DIM // self._NUM_CHANNELS

        for ch_idx in range(self._NUM_CHANNELS):
            start = ch_idx * neurons_per_channel
            # Gaussian-shaped receptive field centered in this channel's region
            center = start + neurons_per_channel // 2
            for i in range(start, start + neurons_per_channel):
                dist = abs(i - center) / (neurons_per_channel / 2)
                P[i, ch_idx] = np.exp(-2.0 * dist * dist)  # Gaussian falloff

            # Small cross-channel bleed (embodied signals are never fully isolated)
            for other_ch in range(self._NUM_CHANNELS):
                if other_ch != ch_idx:
                    bleed_start = other_ch * neurons_per_channel
                    P[bleed_start:bleed_start + 5, ch_idx] = 0.05

        return P

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._run_loop(), name="Interoception")
        logger.info("EmbodiedInteroception STARTED (%.0f Hz)", self._SAMPLE_HZ)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("EmbodiedInteroception STOPPED")

    async def _run_loop(self):
        interval = 1.0 / self._SAMPLE_HZ
        try:
            while self._running:
                t0 = time.time()
                try:
                    await asyncio.to_thread(self._sample_hardware)
                    self._push_to_mesh()
                    self._trigger_neurochemical_events()
                except Exception as e:
                    logger.error("Interoception tick error: %s", e, exc_info=True)
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            pass

    # ── Hardware sampling ────────────────────────────────────────────────

    def _sample_hardware(self):
        """Sample all hardware channels. Runs in thread pool."""
        now = time.time()
        dt = now - self._last_sample_time if self._last_sample_time > 0 else 1.0
        self._last_sample_time = now

        if not _HAS_PSUTIL:
            for ch in self.channels.values():
                ch.fail_safe()
            self._tick_count += 1
            return

        # 1. CPU → metabolic load
        try:
            cpu = psutil.cpu_percent(interval=0) / 100.0
            self.channels["metabolic_load"].update(cpu)
        except Exception:
            self.channels["metabolic_load"].fail_safe()

        # 2. RAM → resource pressure
        try:
            mem = psutil.virtual_memory()
            self.channels["resource_pressure"].update(mem.percent / 100.0)
        except Exception:
            self.channels["resource_pressure"].fail_safe()

        # 3. Temperature → thermal state
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get highest CPU core temp, normalize to 0-1 (30°C=0, 100°C=1)
                max_temp = max(
                    s.current for sensors in temps.values() for s in sensors
                )
                normalized = max(0.0, min(1.0, (max_temp - 30.0) / 70.0))
                self.channels["thermal_state"].update(normalized)
            else:
                # macOS often lacks temp sensors via psutil — use CPU as proxy
                cpu_proxy = self.channels["metabolic_load"].smoothed * 0.6
                self.channels["thermal_state"].update(cpu_proxy)
        except Exception:
            self.channels["thermal_state"].fail_safe()

        # 4. Disk I/O → throughput
        try:
            io = psutil.disk_io_counters()
            if io and self._last_io_counters:
                read_delta = max(0, io.read_bytes - self._last_io_counters.read_bytes)
                write_delta = max(0, io.write_bytes - self._last_io_counters.write_bytes)
                total_bytes = (read_delta + write_delta) / max(dt, 0.1)
                # Normalize: 0 bytes/s = 0, 100 MB/s = 1.0
                normalized = min(1.0, total_bytes / (100 * 1024 * 1024))
                self.channels["io_throughput"].update(normalized)
            self._last_io_counters = io
        except Exception:
            self.channels["io_throughput"].fail_safe()

        # 5. Network → communication pain (error rate + high latency proxy)
        try:
            net = psutil.net_io_counters()
            if net and self._last_net_counters:
                err_delta = (
                    (net.errin - self._last_net_counters.errin) +
                    (net.errout - self._last_net_counters.errout) +
                    (net.dropin - self._last_net_counters.dropin) +
                    (net.dropout - self._last_net_counters.dropout)
                )
                # More errors = more pain. 100 errors/sec → pain = 1.0
                pain = min(1.0, err_delta / max(dt, 0.1) / 100.0)
                self.channels["network_pain"].update(pain)
            self._last_net_counters = net
        except Exception:
            self.channels["network_pain"].fail_safe()

        # 6. Battery → energy reserves (inverted: high battery = high reserves)
        try:
            battery = psutil.sensors_battery()
            if battery:
                reserves = battery.percent / 100.0
                # Charging bonus
                if battery.power_plugged:
                    reserves = min(1.0, reserves + 0.1)
                self.channels["energy_reserves"].update(reserves)
            else:
                # Desktop / plugged in = full energy
                self.channels["energy_reserves"].update(0.95)
        except Exception:
            self.channels["energy_reserves"].fail_safe()

        # 7. Process count → sensory load
        try:
            procs = len(psutil.pids())
            # Normalize: 100 procs = 0.2, 500 = 0.8, 1000+ = 1.0
            normalized = min(1.0, procs / 1000.0)
            self.channels["process_load"].update(normalized)
        except Exception:
            self.channels["process_load"].fail_safe()

        # 8. Disk capacity → organ capacity
        try:
            disk = psutil.disk_usage("/")
            self.channels["disk_capacity"].update(disk.percent / 100.0)
        except Exception:
            self.channels["disk_capacity"].fail_safe()

        self._tick_count += 1

    # ── Push to consumers ────────────────────────────────────────────────

    def _push_to_mesh(self):
        """Inject sensory signal into NeuralMesh sensory tier."""
        if self._mesh_ref is None:
            return
        try:
            vec = self.get_sensory_vector()
            self._mesh_ref.inject_sensory(vec)
        except Exception as e:
            logger.debug("Failed to push sensory to mesh: %s", e)

    def _trigger_neurochemical_events(self):
        """Drive neurochemical responses based on body state AND its derivatives.

        The rate of change matters as much as the absolute level:
        - CPU spiking upward → norepinephrine surge (sudden alertness)
        - CPU high but stable → cortisol (sustained stress)
        - CPU dropping → endorphin (relief)
        - Sudden acceleration in any channel → prediction error → dopamine
        """
        if self._neurochemical_ref is None:
            return
        ncs = self._neurochemical_ref

        try:
            ml = self.channels["metabolic_load"]
            rp = self.channels["resource_pressure"]
            th = self.channels["thermal_state"]
            er = self.channels["energy_reserves"]
            np_ = self.channels["network_pain"]

            # ── Level-based triggers (existing, preserved) ────────────
            # High CPU load → cortisol (sustained stress)
            if ml.smoothed > 0.7:
                ncs.on_threat(severity=(ml.smoothed - 0.7) * 2.0)

            # Low energy → frustration cascade
            if er.smoothed < 0.2:
                ncs.on_frustration(amount=(0.2 - er.smoothed) * 3.0)

            # Thermal stress → cortisol
            if th.smoothed > 0.7:
                ncs.on_threat(severity=(th.smoothed - 0.7) * 1.5)

            # High resource pressure → anxiety
            if rp.smoothed > 0.8:
                ncs.on_threat(severity=(rp.smoothed - 0.8) * 2.5)

            # Network pain → frustration
            if np_.smoothed > 0.3:
                ncs.on_frustration(amount=np_.smoothed * 0.5)

            # Low load + good energy → rest response
            if ml.smoothed < 0.2 and er.smoothed > 0.7 and rp.smoothed < 0.5:
                ncs.on_rest()

            # ── DERIVATIVE-BASED triggers (NEW — rate of change) ──────

            # Sudden CPU SPIKE (positive velocity) → norepinephrine surge (alertness)
            # This is different from sustained high — it's the SURPRISE of load increasing
            if ml.velocity > 0.15:
                ncs.on_prediction_error(error=ml.velocity * 2.0)
                logger.debug("⚡ Interoception: CPU spike (v=%.2f) → prediction error", ml.velocity)

            # CPU DROPPING (negative velocity) → relief → endorphin
            if ml.velocity < -0.1 and ml.smoothed < 0.5:
                ncs.on_success()  # Relief is a form of reward
                logger.debug("😌 Interoception: CPU dropping (v=%.2f) → relief", ml.velocity)

            # Sudden ACCELERATION in any channel → novelty signal → dopamine + acetylcholine
            # This fires when change is itself changing rapidly — something unexpected
            max_accel = max(
                abs(ml.acceleration),
                abs(rp.acceleration),
                abs(th.acceleration),
                abs(np_.acceleration),
            )
            if max_accel > 0.1:
                ncs.on_novelty()
                logger.debug("🔔 Interoception: acceleration spike (%.2f) → novelty", max_accel)

            # Resource pressure RISING FAST → cortisol spike (impending crisis)
            if rp.velocity > 0.1:
                ncs.on_threat(severity=rp.velocity * 3.0)
                logger.debug("⚠️ Interoception: RAM pressure rising (v=%.2f) → threat", rp.velocity)

            # Everything STABILIZING (low velocity across all channels) → GABA + serotonin (calm)
            all_velocities = [abs(ch.velocity) for ch in self.channels.values()]
            if max(all_velocities) < 0.03 and ml.smoothed < 0.5:
                ncs.on_rest()

        except Exception as e:
            logger.debug("Neurochemical trigger error: %s", e)

    # ── External API ─────────────────────────────────────────────────────

    def get_sensory_vector(self) -> np.ndarray:
        """1024-d sensory signal for NeuralMesh injection.

        Each of the 8 channels is projected through its receptive field
        into the full sensory neuron space.
        """
        channel_values = np.array([
            self.channels[name].smoothed
            for name in [
                "metabolic_load", "resource_pressure", "thermal_state",
                "io_throughput", "network_pain", "energy_reserves",
                "process_load", "disk_capacity"
            ]
        ], dtype=np.float32)

        # Project through structured receptive fields
        sensory = self._projection @ channel_values
        # Scale to ±0.5 (centered, moderate amplitude)
        return (sensory - 0.5).astype(np.float32)

    def get_interoceptive_state(self) -> Dict[str, float]:
        """All channel values for telemetry."""
        return {name: round(ch.smoothed, 4) for name, ch in self.channels.items()}

    def get_body_budget(self) -> Dict[str, float]:
        """Summary metabolic state for somatic marker gate."""
        ml = self.channels["metabolic_load"].smoothed
        rp = self.channels["resource_pressure"].smoothed
        er = self.channels["energy_reserves"].smoothed
        th = self.channels["thermal_state"].smoothed

        # Available resources (high = good)
        available = (er * 0.4 + (1.0 - rp) * 0.3 + (1.0 - ml) * 0.2 + (1.0 - th) * 0.1)
        # Demand (high = stressed)
        demand = (ml * 0.4 + rp * 0.3 + th * 0.2 + self.channels["process_load"].smoothed * 0.1)
        # Budget = available - demand  (positive = surplus, negative = deficit)
        budget = available - demand

        return {
            "available_resources": round(available, 4),
            "current_demand": round(demand, 4),
            "budget": round(budget, 4),
            "energy_reserves": round(er, 4),
            "metabolic_load": round(ml, 4),
            "thermal_stress": round(th, 4),
            "resource_pressure": round(rp, 4),
        }

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "has_psutil": _HAS_PSUTIL,
            "channels": self.get_interoceptive_state(),
            "body_budget": self.get_body_budget(),
            "any_failed": any(ch._failed for ch in self.channels.values()),
        }
