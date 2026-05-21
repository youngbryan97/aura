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
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Consciousness.OscillatoryBinding")

_RECOVERABLE_BINDING_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_TAU = 2.0 * math.pi
_MAX_PHASE_SOURCES = 256
_MAX_SOURCE_NAME_CHARS = 128


def _record_binding_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "oscillatory_binding",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("oscillatory_binding", error)


def _finite_float(raw: object, default: float) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if not math.isfinite(value):
        return default, False
    return value, True


def _clamp_float(value: float, *, lower: float, upper: float) -> tuple[float, bool]:
    clamped = max(lower, min(upper, value))
    return clamped, clamped == value


def _normalize_phase(raw: object) -> tuple[float, bool]:
    phase, valid = _finite_float(raw, 0.0)
    return phase % _TAU, valid


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
    contributing_sources: list[str]
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
        self._validate_config()

        # Internal oscillator state
        self._gamma_phase: float = 0.0  # radians
        self._theta_phase: float = 0.0
        self._gamma_amplitude: float = 1.0
        self._theta_amplitude: float = 1.0

        # Phase reports from subsystems
        self._phase_reports: dict[str, float] = {}  # source → last reported phase
        self._phase_timestamps: dict[str, float] = {}

        # Synchronization
        self._psi: float = 0.5  # Phase Synchronization Index
        self._bound: bool = False
        self._binding_history: deque[BindingMoment] = deque(maxlen=self.cfg.history_len)

        # Theta-gamma coupling measurement
        self._gamma_at_theta_peak: deque[float] = deque(maxlen=50)
        self._gamma_at_theta_trough: deque[float] = deque(maxlen=50)
        self._measured_coupling: float = 0.0

        # Runtime
        self._running = False
        self._task: asyncio.Task | None = None
        self._internal_tick: int = 0
        self._output_tick: int = 0
        self._start_time: float = 0.0
        self._consecutive_tick_failures: int = 0
        self._last_tick_error_at: float = 0.0

        # Fragmentation events (for executive attention)
        self._fragmentation_count: int = 0
        self._last_fragmentation_time: float = 0.0

        logger.info("OscillatoryBinding initialized (γ=%.0fHz, θ=%.0fHz, coupling=%.2f)",
                     self.cfg.gamma_freq, self.cfg.theta_freq, self.cfg.coupling_strength)

    def _validate_config(self) -> None:
        numeric_ranges = {
            "gamma_freq": (self.cfg.gamma_freq, 1.0, 200.0),
            "theta_freq": (self.cfg.theta_freq, 0.1, 40.0),
            "internal_rate": (self.cfg.internal_rate, 1.0, 1000.0),
            "output_rate": (self.cfg.output_rate, 0.1, 200.0),
            "coupling_strength": (self.cfg.coupling_strength, 0.0, 1.0),
            "sync_threshold": (self.cfg.sync_threshold, 0.0, 1.0),
            "fragmentation_threshold": (self.cfg.fragmentation_threshold, 0.0, 1.0),
        }
        for name, (raw, lower, upper) in numeric_ranges.items():
            value, valid = _finite_float(raw, 0.0)
            if not valid or not (lower <= value <= upper):
                raise ValueError(
                    f"OscillatoryBinding config {name} must be finite in [{lower}, {upper}]"
                )
        if self.cfg.output_rate > self.cfg.internal_rate:
            raise ValueError("OscillatoryBinding output_rate must not exceed internal_rate")
        if self.cfg.history_len <= 0:
            raise ValueError("OscillatoryBinding history_len must be positive")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        try:
            self._task = get_task_tracker().create_task(
                self._run_loop(),
                name="OscillatoryBinding",
            )
        except _RECOVERABLE_BINDING_ERRORS as exc:
            self._running = False
            self._task = None
            _record_binding_degradation(
                exc,
                action="failed closed when OscillatoryBinding task creation failed",
                severity="critical",
            )
            raise
        logger.info("OscillatoryBinding STARTED")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("OscillatoryBinding task cancellation acknowledged")
            self._task = None
        logger.info("OscillatoryBinding STOPPED")

    async def _run_loop(self):
        """Internal loop at 100 Hz for oscillator dynamics.
        Emits binding moments at output_rate (10 Hz).
        """
        internal_rate, valid_internal_rate = _finite_float(self.cfg.internal_rate, 100.0)
        output_rate, valid_output_rate = _finite_float(self.cfg.output_rate, 10.0)
        internal_rate, internal_rate_unchanged = _clamp_float(
            internal_rate,
            lower=1.0,
            upper=1000.0,
        )
        output_rate, output_rate_unchanged = _clamp_float(
            output_rate,
            lower=0.1,
            upper=internal_rate,
        )
        if not all(
            (
                valid_internal_rate,
                valid_output_rate,
                internal_rate_unchanged,
                output_rate_unchanged,
            )
        ):
            _record_binding_degradation(
                ValueError("OscillatoryBinding loop rates were invalid"),
                action="normalized OscillatoryBinding loop rates before runtime loop",
                severity="warning",
                extra={"internal_rate": internal_rate, "output_rate": output_rate},
            )
        internal_interval = 1.0 / internal_rate
        output_every = max(1, int(round(internal_rate / output_rate)))

        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    self._oscillator_step()
                    self._internal_tick += 1

                    if self._internal_tick % output_every == 0:
                        self._compute_synchronization()
                        self._emit_binding_moment()
                        self._output_tick += 1

                    self._consecutive_tick_failures = 0
                except _RECOVERABLE_BINDING_ERRORS as exc:
                    self._consecutive_tick_failures += 1
                    self._last_tick_error_at = time.monotonic()
                    _record_binding_degradation(
                        exc,
                        action=(
                            "kept OscillatoryBinding loop alive after tick failure "
                            "and reset to neutral phase state"
                        ),
                        extra={
                            "consecutive_tick_failures": self._consecutive_tick_failures
                        },
                    )
                    logger.error("Oscillatory binding error: %s", exc, exc_info=True)
                    self._enter_fail_safe_state()

                elapsed = time.monotonic() - t0
                backoff = min(
                    internal_interval * max(0, self._consecutive_tick_failures),
                    2.0,
                )
                await asyncio.sleep(max(0.0, internal_interval + backoff - elapsed))
        except asyncio.CancelledError:
            logger.debug("OscillatoryBinding run loop cancelled")
        finally:
            self._running = False

    def _enter_fail_safe_state(self) -> None:
        self._gamma_phase, _ = _normalize_phase(self._gamma_phase)
        self._theta_phase, _ = _normalize_phase(self._theta_phase)
        gamma_amp, _ = _finite_float(self._gamma_amplitude, 1.0)
        theta_amp, _ = _finite_float(self._theta_amplitude, 1.0)
        measured, _ = _finite_float(self._measured_coupling, 0.0)
        psi, _ = _finite_float(self._psi, 0.5)
        self._gamma_amplitude, _ = _clamp_float(gamma_amp, lower=0.0, upper=2.0)
        self._theta_amplitude, _ = _clamp_float(theta_amp, lower=0.0, upper=2.0)
        self._measured_coupling, _ = _clamp_float(measured, lower=0.0, upper=1.0)
        self._psi, _ = _clamp_float(psi, lower=0.0, upper=1.0)
        self._bound = self._psi >= self.cfg.sync_threshold

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
        if not isinstance(tier_energy, dict):
            _record_binding_degradation(
                TypeError("mesh tier energy must be a dict"),
                action="ignored malformed mesh energy update for oscillator binding",
                severity="warning",
            )
            return
        values: dict[str, float] = {}
        for key in ("sensory", "association", "executive"):
            value, valid = _finite_float(tier_energy.get(key, 0.0), 0.0)
            value, unchanged = _clamp_float(value, lower=0.0, upper=1.0)
            if not valid or not unchanged:
                _record_binding_degradation(
                    ValueError(f"invalid mesh {key} energy for oscillator binding"),
                    action="normalized mesh energy before oscillator modulation",
                    severity="warning",
                    extra={"tier": key, "value": value},
                )
            values[key] = value
        self._mesh_sensory_energy = values["sensory"]
        self._mesh_association_energy = values["association"]
        self._mesh_executive_energy = values["executive"]

    def _oscillator_step(self):
        """Advance gamma and theta oscillators by one internal tick.

        Now MODULATED by neural mesh activity:
        - Executive tier energy → gamma frequency (higher activity = faster binding)
        - Association tier energy → theta amplitude (richer associations = stronger framing)
        - Sensory tier energy → gamma amplitude boost (strong input = stronger binding)
        """
        internal_rate, valid_internal_rate = _finite_float(self.cfg.internal_rate, 100.0)
        internal_rate, rate_unchanged = _clamp_float(
            internal_rate,
            lower=1.0,
            upper=1000.0,
        )
        if not valid_internal_rate or not rate_unchanged:
            _record_binding_degradation(
                ValueError("OscillatoryBinding internal rate was invalid"),
                action="normalized internal rate before oscillator step",
                severity="warning",
                extra={"internal_rate": internal_rate},
            )
        dt = 1.0 / internal_rate

        # ── Mesh-driven modulation ──────────────────────────────────────
        # These create genuine bidirectional coupling:
        # mesh activity → oscillator dynamics → binding moments → back to mesh via unified field
        mesh_exec, _ = _finite_float(getattr(self, "_mesh_executive_energy", 0.0), 0.0)
        mesh_assoc, _ = _finite_float(getattr(self, "_mesh_association_energy", 0.0), 0.0)
        mesh_sens, _ = _finite_float(getattr(self, "_mesh_sensory_energy", 0.0), 0.0)
        mesh_exec, _ = _clamp_float(mesh_exec, lower=0.0, upper=1.0)
        mesh_assoc, _ = _clamp_float(mesh_assoc, lower=0.0, upper=1.0)
        mesh_sens, _ = _clamp_float(mesh_sens, lower=0.0, upper=1.0)

        # Executive activity modulates gamma FREQUENCY (±5 Hz around 40 Hz base)
        # High executive activity = faster binding = sharper temporal integration
        gamma_freq_mod = self.cfg.gamma_freq + (mesh_exec - 0.3) * 10.0
        gamma_freq_mod, _ = _finite_float(gamma_freq_mod, 40.0)
        gamma_freq_mod, _ = _clamp_float(gamma_freq_mod, lower=30.0, upper=50.0)

        # Association activity modulates theta AMPLITUDE (richer associations = stronger framing)
        theta_amp_mod = 1.0 + (mesh_assoc - 0.3) * 0.5  # 0.85-1.35 range
        theta_amp_mod, _ = _finite_float(theta_amp_mod, 1.0)
        theta_amp_mod, _ = _clamp_float(theta_amp_mod, lower=0.5, upper=1.5)

        # Sensory input modulates coupling STRENGTH (strong input = tighter coupling)
        coupling_mod = self.cfg.coupling_strength + mesh_sens * 0.3
        coupling_mod, _ = _finite_float(coupling_mod, self.cfg.coupling_strength)
        coupling_mod, _ = _clamp_float(coupling_mod, lower=0.0, upper=1.0)

        # Advance theta (slow oscillator — 8 Hz)
        theta_freq, _ = _finite_float(self.cfg.theta_freq, 8.0)
        theta_freq, _ = _clamp_float(theta_freq, lower=0.1, upper=40.0)
        theta_phase, _ = _normalize_phase(self._theta_phase)
        self._theta_phase = (theta_phase + _TAU * theta_freq * dt) % _TAU

        # Theta-gamma coupling: gamma amplitude modulated by theta phase
        # Now uses mesh-modulated coupling strength
        theta_modulation = (1.0 + coupling_mod * math.cos(self._theta_phase)) / (1.0 + coupling_mod)
        self._gamma_amplitude, _ = _finite_float(theta_modulation * theta_amp_mod, 1.0)
        self._gamma_amplitude, _ = _clamp_float(
            self._gamma_amplitude,
            lower=0.0,
            upper=2.0,
        )

        # Advance gamma (fast oscillator — now mesh-modulated frequency)
        gamma_phase, _ = _normalize_phase(self._gamma_phase)
        self._gamma_phase = (gamma_phase + _TAU * gamma_freq_mod * dt) % _TAU

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
            self._measured_coupling, _ = _finite_float(self._measured_coupling, 0.0)
            self._measured_coupling, _ = _clamp_float(
                self._measured_coupling,
                lower=0.0,
                upper=1.0,
            )

    # ── Synchronization computation ──────────────────────────────────────

    def _compute_synchronization(self):
        """Compute Phase Synchronization Index from subsystem phase reports.

        PSI = |mean(e^{i * phase_k})|  for all reporting subsystems k

        This is the circular mean resultant length — a standard measure of
        phase concentration.  1.0 = all in phase, 0.0 = uniformly distributed.
        """
        now = time.monotonic()
        stale_cutoff = now - 2.0

        # Prune stale reports to prevent unbounded dict growth
        stale_keys = []
        for key, timestamp in self._phase_timestamps.items():
            ts, valid_ts = _finite_float(timestamp, 0.0)
            if not valid_ts or ts < stale_cutoff:
                stale_keys.append(key)
        for k in stale_keys:
            self._phase_reports.pop(k, None)
            self._phase_timestamps.pop(k, None)

        # Only use recent reports (within last 2 seconds)
        active_phases = []
        for source, phase in self._phase_reports.items():
            if self._phase_timestamps.get(source, 0) > stale_cutoff:
                normalized_phase, valid_phase = _normalize_phase(phase)
                if valid_phase:
                    active_phases.append(normalized_phase)
                else:
                    _record_binding_degradation(
                        ValueError(f"invalid phase report from {source}"),
                        action="ignored invalid phase report during synchronization",
                        severity="warning",
                        extra={"source": source},
                    )

        if len(active_phases) < 2:
            # Not enough sources to compute synchronization
            self._psi = 0.5  # neutral
            self._bound = False
            return

        # Phase Synchronization Index (circular statistics)
        phases = np.array(active_phases, dtype=np.float64)
        complex_phases = np.exp(1j * phases)
        mean_resultant = np.abs(np.mean(complex_phases))
        self._psi = float(np.clip(mean_resultant, 0.0, 1.0))

        # Binding state
        was_bound = self._bound
        sync_threshold, _ = _finite_float(self.cfg.sync_threshold, 0.65)
        sync_threshold, _ = _clamp_float(sync_threshold, lower=0.0, upper=1.0)
        self._bound = self._psi >= sync_threshold

        # Fragmentation detection
        fragmentation_threshold, _ = _finite_float(self.cfg.fragmentation_threshold, 0.3)
        fragmentation_threshold, _ = _clamp_float(
            fragmentation_threshold,
            lower=0.0,
            upper=1.0,
        )
        if not self._bound and self._psi < fragmentation_threshold:
            if was_bound or (now - self._last_fragmentation_time > 5.0):
                self._fragmentation_count += 1
                self._last_fragmentation_time = now
                logger.debug("Fragmentation event #%d (PSI=%.3f)",
                             self._fragmentation_count, self._psi)

    def _emit_binding_moment(self):
        """Record a BindingMoment for history and downstream consumers."""
        now = time.monotonic()
        stale_cutoff = now - 2.0
        contributing = [
            src for src, ts in self._phase_timestamps.items()
            if ts > stale_cutoff
        ]
        gamma_amplitude, _ = _finite_float(self._gamma_amplitude, 1.0)
        gamma_amplitude, _ = _clamp_float(gamma_amplitude, lower=0.0, upper=2.0)
        theta_phase, _ = _normalize_phase(self._theta_phase)
        psi, _ = _finite_float(self._psi, 0.5)
        psi, _ = _clamp_float(psi, lower=0.0, upper=1.0)

        moment = BindingMoment(
            timestamp=time.time(),
            psi=psi,
            gamma_amplitude=gamma_amplitude,
            theta_phase=theta_phase,
            is_bound=self._bound,
            contributing_sources=contributing,
            field_coherence=max(0.0, min(1.0, psi * gamma_amplitude)),
        )
        self._binding_history.append(moment)

    # ── External API: phase reporting ────────────────────────────────────

    def report_phase(self, source: str, phase: float):
        """Called by subsystems to report their current processing phase.

        phase should be in radians [0, 2π).  Subsystems that are "in sync"
        with the binding rhythm report similar phases, increasing PSI.
        """
        if not isinstance(source, str) or not source.strip():
            _record_binding_degradation(
                ValueError("phase report source was empty or non-string"),
                action="ignored phase report with invalid source",
                severity="warning",
            )
            return
        source = source.strip()[:_MAX_SOURCE_NAME_CHARS]
        normalized_phase, valid_phase = _normalize_phase(phase)
        if not valid_phase:
            _record_binding_degradation(
                ValueError(f"phase report from {source} was non-finite"),
                action="ignored invalid phase report",
                severity="warning",
                extra={"source": source},
            )
            return
        if source not in self._phase_reports and len(self._phase_reports) >= _MAX_PHASE_SOURCES:
            if self._phase_timestamps:
                oldest_source = min(
                    self._phase_timestamps,
                    key=lambda item: self._phase_timestamps.get(item, 0.0),
                )
            else:
                oldest_source = next(iter(self._phase_reports))
            self._phase_reports.pop(oldest_source, None)
            self._phase_timestamps.pop(oldest_source, None)
            _record_binding_degradation(
                RuntimeError("OscillatoryBinding phase source capacity reached"),
                action="evicted oldest phase source before accepting new report",
                severity="warning",
                extra={"evicted_source": oldest_source},
            )
        self._phase_reports[source] = normalized_phase
        self._phase_timestamps[source] = time.monotonic()

    def compute_subsystem_phase(self, activation_level: float, source: str) -> float:
        """Helper: derive a phase from a subsystem's activation level.

        Maps activation through gamma oscillation to produce a phase that
        will naturally synchronize when subsystems have correlated activity.
        """
        # Lock subsystem activity to gamma rhythm
        activation_level, valid_activation = _finite_float(activation_level, 0.0)
        activation_level, activation_unchanged = _clamp_float(
            activation_level,
            lower=-1.0,
            upper=1.0,
        )
        if not valid_activation or not activation_unchanged:
            _record_binding_degradation(
                ValueError("subsystem activation level was invalid"),
                action="normalized subsystem activation before phase computation",
                severity="warning",
                extra={"source": source, "activation_level": activation_level},
            )
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
        psi, _ = _finite_float(self._psi, 0.5)
        psi, _ = _clamp_float(psi, lower=0.0, upper=1.0)
        return psi

    def get_gamma_amplitude(self) -> float:
        """Current gamma oscillation amplitude (modulated by theta)."""
        amplitude, _ = _finite_float(self._gamma_amplitude, 1.0)
        amplitude, _ = _clamp_float(amplitude, lower=0.0, upper=2.0)
        return amplitude

    def get_theta_phase(self) -> float:
        """Current theta phase in radians."""
        theta_phase, _ = _normalize_phase(self._theta_phase)
        return theta_phase

    def is_bound(self) -> bool:
        """Whether current processing is in a bound (unified) state."""
        return self._bound

    def get_coupling_strength(self) -> float:
        """Measured theta-gamma coupling (Modulation Index)."""
        coupling, _ = _finite_float(self._measured_coupling, 0.0)
        coupling, _ = _clamp_float(coupling, lower=0.0, upper=1.0)
        return coupling

    def get_fragmentation_count(self) -> int:
        return self._fragmentation_count

    def get_binding_history(self, n: int = 20) -> list[dict]:
        """Recent binding moments for telemetry."""
        if not isinstance(n, int):
            n = 20
        n = max(0, min(500, n))
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
        contribution = self.get_psi() * (0.5 + 0.5 * self.get_coupling_strength())
        contribution, _ = _clamp_float(contribution, lower=0.0, upper=1.0)
        return contribution

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "internal_tick": self._internal_tick,
            "output_tick": self._output_tick,
            "psi": round(self.get_psi(), 4),
            "is_bound": self._bound,
            "gamma_amplitude": round(self.get_gamma_amplitude(), 4),
            "gamma_phase": round(_normalize_phase(self._gamma_phase)[0], 4),
            "theta_phase": round(self.get_theta_phase(), 4),
            "measured_coupling": round(self.get_coupling_strength(), 4),
            "fragmentation_count": self._fragmentation_count,
            "active_sources": len(self._phase_reports),
            "history_len": len(self._binding_history),
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_tick_error_age_s": round(time.monotonic() - self._last_tick_error_at, 1)
            if self._last_tick_error_at
            else 0,
            "uptime_s": round(time.monotonic() - self._start_time, 1) if self._start_time else 0,
        }
