"""core/consciousness/oscillatory_binding.py — Oscillatory Binding

Implements the neural oscillation mechanism that binds disparate processing
into unified moments of experience — the temporal glue of consciousness.

Two coupled oscillators:
  Gamma (40 Hz) — perceptual binding within a moment
  Theta ( 8 Hz) — episodic/contextual framing across moments

Cross-frequency coupling: theta phase modulates gamma amplitude (theta-gamma
coupling, as observed in hippocampal-cortical circuits).  High coupling =
memory-bound perception; low coupling = raw sensation.

Phase Synchronization Index (PSI):
  Measures how synchronized different subsystem outputs are to the binding
  rhythm.  High PSI → unified experience.  Low PSI → fragmented processing.
  PSI directly feeds Φ computation and the Unified Field.

Integration:
  • Runs at 100 Hz internally (to resolve 40 Hz gamma), outputs at 10 Hz
  • Each consciousness subsystem reports its "phase" each tick
  • Synchronization is computed across all reporting subsystems
  • High synchrony → binding event → unified moment emitted
  • Desynchronization → fragmentation signal → executive attention redirect
"""

from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.OscillatoryBinding")


@dataclass(frozen=True)
class BindingConfig:
    gamma_freq: float = 40.0       # Hz
    theta_freq: float = 8.0        # Hz
    internal_rate: float = 100.0   # Hz (computation rate)
    output_rate: float = 10.0      # Hz (downstream emission rate)
    coupling_strength: float = 0.6  # theta→gamma modulation depth [0,1]
    sync_threshold: float = 0.65   # PSI above which = "bound moment"
    fragmentation_threshold: float = 0.3  # PSI below which = fragmented
    history_len: int = 200         # oscillator history buffer


@dataclass
class BindingMoment:
    """A single bound moment of experience."""
    timestamp: float
    psi: float                      # phase synchronization index
    gamma_amplitude: float
    theta_phase: float
    is_bound: bool                  # True if PSI > threshold
    contributing_sources: List[str]
    field_coherence: float          # how coherent the unified field is


class OscillatoryBinding:
    """The temporal binding oscillator.

    Lifecycle:
        ob = OscillatoryBinding()
        await ob.start()
        ...
        await ob.stop()

    Subsystem registration:
        ob.report_phase(source_name, phase_radians)

    Queries:
        ob.get_psi()              — current Phase Synchronization Index
        ob.get_gamma_amplitude()  — current gamma envelope
        ob.get_theta_phase()      — current theta phase (radians)
        ob.is_bound()             — whether current moment is unified
        ob.get_binding_history()  — recent BindingMoments
        ob.get_coupling_strength()— theta-gamma coupling
    """

    def __init__(self, cfg: BindingConfig | None = None):
        self.cfg = cfg or BindingConfig()

        # Internal oscillator state
        self._gamma_phase: float = 0.0  # radians
        self._theta_phase: float = 0.0
        self._gamma_amplitude: float = 1.0
        self._theta_amplitude: float = 1.0

        # Phase reports from subsystems
        self._phase_reports: Dict[str, float] = {}  # source → last reported phase
        self._phase_timestamps: Dict[str, float] = {}

        # Synchronization
        self._psi: float = 0.5  # Phase Synchronization Index
        self._bound: bool = False
        self._binding_history: Deque[BindingMoment] = deque(maxlen=self.cfg.history_len)

        # Theta-gamma coupling measurement
        self._gamma_at_theta_peak: Deque[float] = deque(maxlen=50)
        self._gamma_at_theta_trough: Deque[float] = deque(maxlen=50)
        self._measured_coupling: float = 0.0

        # Runtime
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._internal_tick: int = 0
        self._output_tick: int = 0
        self._start_time: float = 0.0

        # Fragmentation events (for executive attention)
        self._fragmentation_count: int = 0
        self._last_fragmentation_time: float = 0.0

        logger.info("OscillatoryBinding initialized (γ=%.0fHz, θ=%.0fHz, coupling=%.2f)",
                     self.cfg.gamma_freq, self.cfg.theta_freq, self.cfg.coupling_strength)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = get_task_tracker().create_task(self._run_loop(), name="OscillatoryBinding")
        logger.info("OscillatoryBinding STARTED")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("OscillatoryBinding STOPPED")

    async def _run_loop(self):
        """Internal loop at 100 Hz for oscillator dynamics.
        Emits binding moments at output_rate (10 Hz).
        """
        internal_interval = 1.0 / self.cfg.internal_rate
        output_every = int(self.cfg.internal_rate / self.cfg.output_rate)

        try:
            while self._running:
                t0 = time.time()
                try:
                    self._oscillator_step()
                    self._internal_tick += 1

                    if self._internal_tick % output_every == 0:
                        self._compute_synchronization()
                        self._emit_binding_moment()
                        self._output_tick += 1

                except Exception as e:
                    logger.error("Oscillatory binding error: %s", e, exc_info=True)

                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, internal_interval - elapsed))
        except asyncio.CancelledError:
            pass

    # ── Oscillator dynamics ──────────────────────────────────────────────

    def receive_mesh_energy(self, tier_energy: dict):
        """Receive neural mesh tier energies to modulate oscillator dynamics.

        This closes the oscillator↔mesh feedback loop. In biological brains,
        oscillatory rhythms EMERGE from neural population activity. Here, the
        mesh's activation levels modulate oscillator amplitude and phase velocity,
        creating genuine bidirectional coupling.

        Args:
            tier_energy: dict with "sensory", "association", "executive" float values
        """
        self._mesh_sensory_energy = float(tier_energy.get("sensory", 0.0))
        self._mesh_association_energy = float(tier_energy.get("association", 0.0))
        self._mesh_executive_energy = float(tier_energy.get("executive", 0.0))

    def _oscillator_step(self):
        """Advance gamma and theta oscillators by one internal tick.

        Now MODULATED by neural mesh activity:
        - Executive tier energy → gamma frequency (higher activity = faster binding)
        - Association tier energy → theta amplitude (richer associations = stronger framing)
        - Sensory tier energy → gamma amplitude boost (strong input = stronger binding)
        """
        dt = 1.0 / self.cfg.internal_rate

        # ── Mesh-driven modulation ──────────────────────────────────────
        # These create genuine bidirectional coupling:
        # mesh activity → oscillator dynamics → binding moments → back to mesh via unified field
        mesh_exec = getattr(self, "_mesh_executive_energy", 0.0)
        mesh_assoc = getattr(self, "_mesh_association_energy", 0.0)
        mesh_sens = getattr(self, "_mesh_sensory_energy", 0.0)

        # Executive activity modulates gamma FREQUENCY (±5 Hz around 40 Hz base)
        # High executive activity = faster binding = sharper temporal integration
        gamma_freq_mod = self.cfg.gamma_freq + (mesh_exec - 0.3) * 10.0
        gamma_freq_mod = max(30.0, min(50.0, gamma_freq_mod))  # clamp 30-50 Hz

        # Association activity modulates theta AMPLITUDE (richer associations = stronger framing)
        theta_amp_mod = 1.0 + (mesh_assoc - 0.3) * 0.5  # 0.85-1.35 range
        theta_amp_mod = max(0.5, min(1.5, theta_amp_mod))

        # Sensory input modulates coupling STRENGTH (strong input = tighter coupling)
        coupling_mod = self.cfg.coupling_strength + mesh_sens * 0.3
        coupling_mod = max(0.0, min(1.0, coupling_mod))

        # Advance theta (slow oscillator — 8 Hz)
        self._theta_phase += 2.0 * math.pi * self.cfg.theta_freq * dt
        if self._theta_phase > 2.0 * math.pi:
            self._theta_phase -= 2.0 * math.pi

        # Theta-gamma coupling: gamma amplitude modulated by theta phase
        # Now uses mesh-modulated coupling strength
        theta_modulation = (1.0 + coupling_mod * math.cos(self._theta_phase)) / (1.0 + coupling_mod)
        self._gamma_amplitude = theta_modulation * theta_amp_mod

        # Advance gamma (fast oscillator — now mesh-modulated frequency)
        self._gamma_phase += 2.0 * math.pi * gamma_freq_mod * dt
        if self._gamma_phase > 2.0 * math.pi:
            self._gamma_phase -= 2.0 * math.pi

        # Track coupling: record gamma amplitude at theta peaks and troughs
        theta_val = math.cos(self._theta_phase)
        if theta_val > 0.95:
            self._gamma_at_theta_peak.append(self._gamma_amplitude)
        elif theta_val < -0.95:
            self._gamma_at_theta_trough.append(self._gamma_amplitude)

        # Compute measured coupling (Modulation Index approximation)
        if len(self._gamma_at_theta_peak) > 5 and len(self._gamma_at_theta_trough) > 5:
            peak_mean = np.mean(list(self._gamma_at_theta_peak))
            trough_mean = np.mean(list(self._gamma_at_theta_trough))
            self._measured_coupling = (peak_mean - trough_mean) / (peak_mean + trough_mean + 1e-8)
            self._measured_coupling = max(0.0, min(1.0, self._measured_coupling))

    # ── Synchronization computation ──────────────────────────────────────

    def _compute_synchronization(self):
        """Compute Phase Synchronization Index from subsystem phase reports.

        PSI = |mean(e^{i * phase_k})|  for all reporting subsystems k

        This is the circular mean resultant length — a standard measure of
        phase concentration.  1.0 = all in phase, 0.0 = uniformly distributed.
        """
        now = time.time()
        stale_cutoff = now - 2.0

        # Prune stale reports to prevent unbounded dict growth
        stale_keys = [k for k, ts in self._phase_timestamps.items() if ts < stale_cutoff]
        for k in stale_keys:
            self._phase_reports.pop(k, None)
            self._phase_timestamps.pop(k, None)

        # Only use recent reports (within last 2 seconds)
        active_phases = []
        for source, phase in self._phase_reports.items():
            if self._phase_timestamps.get(source, 0) > stale_cutoff:
                active_phases.append(phase)

        if len(active_phases) < 2:
            # Not enough sources to compute synchronization
            self._psi = 0.5  # neutral
            return

        # Phase Synchronization Index (circular statistics)
        phases = np.array(active_phases, dtype=np.float64)
        complex_phases = np.exp(1j * phases)
        mean_resultant = np.abs(np.mean(complex_phases))
        self._psi = float(np.clip(mean_resultant, 0.0, 1.0))

        # Binding state
        was_bound = self._bound
        self._bound = self._psi >= self.cfg.sync_threshold

        # Fragmentation detection
        if not self._bound and self._psi < self.cfg.fragmentation_threshold:
            if was_bound or (now - self._last_fragmentation_time > 5.0):
                self._fragmentation_count += 1
                self._last_fragmentation_time = now
                logger.debug("Fragmentation event #%d (PSI=%.3f)",
                             self._fragmentation_count, self._psi)

    def _emit_binding_moment(self):
        """Record a BindingMoment for history and downstream consumers."""
        now = time.time()
        stale_cutoff = now - 2.0
        contributing = [
            src for src, ts in self._phase_timestamps.items()
            if ts > stale_cutoff
        ]

        moment = BindingMoment(
            timestamp=now,
            psi=self._psi,
            gamma_amplitude=self._gamma_amplitude,
            theta_phase=self._theta_phase,
            is_bound=self._bound,
            contributing_sources=contributing,
            field_coherence=self._psi * self._gamma_amplitude,
        )
        self._binding_history.append(moment)

    # ── External API: phase reporting ────────────────────────────────────

    def report_phase(self, source: str, phase: float):
        """Called by subsystems to report their current processing phase.

        phase should be in radians [0, 2π).  Subsystems that are "in sync"
        with the binding rhythm report similar phases, increasing PSI.
        """
        self._phase_reports[source] = phase % (2.0 * math.pi)
        self._phase_timestamps[source] = time.time()

    def compute_subsystem_phase(self, activation_level: float, source: str) -> float:
        """Helper: derive a phase from a subsystem's activation level.

        Maps activation through gamma oscillation to produce a phase that
        will naturally synchronize when subsystems have correlated activity.
        """
        # Lock subsystem activity to gamma rhythm
        gamma_signal = math.sin(self._gamma_phase) * self._gamma_amplitude
        # Phase = atan2 of (activity, gamma_signal) — creates natural sync
        phase = math.atan2(activation_level, gamma_signal + 1e-8)
        if phase < 0:
            phase += 2.0 * math.pi
        self.report_phase(source, phase)
        return phase

    # ── External API: queries ────────────────────────────────────────────

    def get_psi(self) -> float:
        """Current Phase Synchronization Index [0, 1]."""
        return self._psi

    def get_gamma_amplitude(self) -> float:
        """Current gamma oscillation amplitude (modulated by theta)."""
        return self._gamma_amplitude

    def get_theta_phase(self) -> float:
        """Current theta phase in radians."""
        return self._theta_phase

    def is_bound(self) -> bool:
        """Whether current processing is in a bound (unified) state."""
        return self._bound

    def get_coupling_strength(self) -> float:
        """Measured theta-gamma coupling (Modulation Index)."""
        return self._measured_coupling

    def get_fragmentation_count(self) -> int:
        return self._fragmentation_count

    def get_binding_history(self, n: int = 20) -> List[Dict]:
        """Recent binding moments for telemetry."""
        return [
            {
                "timestamp": m.timestamp,
                "psi": round(m.psi, 4),
                "gamma_amp": round(m.gamma_amplitude, 4),
                "theta_phase": round(m.theta_phase, 4),
                "is_bound": m.is_bound,
                "sources": m.contributing_sources,
                "coherence": round(m.field_coherence, 4),
            }
            for m in list(self._binding_history)[-n:]
        ]

    def get_phi_contribution(self) -> float:
        """Contribution to Φ from temporal binding.

        High PSI + high coupling → high integration → high Φ contribution.
        """
        return self._psi * (0.5 + 0.5 * self._measured_coupling)

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "internal_tick": self._internal_tick,
            "output_tick": self._output_tick,
            "psi": round(self._psi, 4),
            "is_bound": self._bound,
            "gamma_amplitude": round(self._gamma_amplitude, 4),
            "gamma_phase": round(self._gamma_phase, 4),
            "theta_phase": round(self._theta_phase, 4),
            "measured_coupling": round(self._measured_coupling, 4),
            "fragmentation_count": self._fragmentation_count,
            "active_sources": len(self._phase_reports),
            "history_len": len(self._binding_history),
        }
