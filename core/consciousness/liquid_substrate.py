"""core/consciousness/liquid_substrate.py

Implements the "Liquid Substrate" - a continuous-time dynamical system that gives Aura
persistence, emotional depth, and temporal continuity.

Based on Liquid Time-Constant Networks (LTCs) and global workspace theory.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.exceptions import capture_and_log
from core.utils.task_tracker import get_task_tracker, mark_task_protected

DEVICE = torch.device("cpu")


def _configure_torch_threads() -> int:
    raw = os.environ.get("AURA_SUBSTRATE_TORCH_THREADS", "2")
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        requested = 2
    threads = max(1, min(8, requested))
    try:
        torch.set_num_threads(threads)
    except RuntimeError as exc:
        record_degradation(
            "liquid_substrate",
            exc,
            severity="warning",
            action="kept existing torch thread count after substrate thread cap failed",
        )
    return threads


_SUBSTRATE_TORCH_THREADS = _configure_torch_threads()

# Lazy-loaded to avoid circular imports at module load
_riiu_instance = None


@dataclass
class LiquidStateVector:
    """Legacy compatibility vector for Aura 4.0 systems."""

    frustration: float = 0.0
    curiosity: float = 0.5
    energy: float = 1.0
    focus: float = 0.5


logger = logging.getLogger("Consciousness.Substrate")

_SUBSTRATE_LOOP_ERRORS = (
    AttributeError,
    IndexError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    np.linalg.LinAlgError,
)


def _default_substrate_dim() -> int:
    raw = os.environ.get("AURA_SUBSTRATE_DIM", "512")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 512
    return max(16, min(4096, value))


@dataclass
class SubstrateConfig:
    """Configuration for Liquid Substrate"""

    neuron_count: int = field(default_factory=_default_substrate_dim)
    time_constant: float = 0.1  # Integration time step (dt)
    update_rate: float = 20.0  # Hz (updates per second)
    decay_rate: float = (
        0.5  # State decay (leak current prevents saturation while allowing nonlinear dynamics)
    )
    noise_level: float = 0.01  # Stochastic noise
    hebbian_rate: float = 0.001  # Learning rate for synaptic plasticity
    save_interval: int = 300  # Seconds between auto-saves
    adaptive_mode: bool = True  # Slow down on battery/idle
    state_file: Path | None = None


class LiquidSubstrate:
    """The continuous dynamical core of Aura's consciousness.

    It runs a recurrent neural network (RNN) solved via ODEs.
    This ensures that Aura 'exists' continuously, even when not processing user input.
    """

    def __init__(self, config: SubstrateConfig = None):
        self._explicit_config = config is not None
        self._explicit_state_file = bool(config and config.state_file)
        self.config: SubstrateConfig = config or SubstrateConfig()

        # State Vectors
        self.x: np.ndarray = np.zeros(self.config.neuron_count)  # Neuron activations (-1.0 to 1.0)
        self.v: np.ndarray = np.zeros(self.config.neuron_count)  # Velocity (change in x)

        # Connectivity Matrix (The Connectome)
        # Scale by 1/sqrt(N) to keep recurrent drive in the nonlinear regime of tanh
        # (prevents saturation at ±1 which collapses phi's state space)
        n = self.config.neuron_count
        self._rng = np.random.default_rng(seed=42)  # Deterministic RNG
        self.W: np.ndarray = self._rng.standard_normal((n, n)) * (1.0 / np.sqrt(n))

        # Operational Flags
        self.running: bool = False
        self.thread: asyncio.Task | None = None
        self.sync_lock: threading.Lock = threading.Lock()  # For all state access (sync + async)
        self._state_revision: int = 0
        self._torch_state_revision: int = 0
        self._concurrent_state_merges: int = 0
        self._untracked_state_mutations: int = 0
        self._state_merges_by_source: dict[str, int] = {}
        self._last_state_mutation_source: str = "boot"
        self._last_state_mutation_at: float = time.time()
        # Last successful snapshot, published for lock-free telemetry reads.
        # Observed live: the event loop froze 5.7s inside _state_snapshot when
        # a background substrate thread held sync_lock through heavy weight
        # work — telemetry (health endpoint, mood/status/context readers)
        # must never contend on the hot lock.
        self._last_published_snapshot: dict[str, Any] | None = None
        self.last_update: float = 0.0

        # --- PyTorch Substrate State (Evolution 1) ---
        self.device = DEVICE
        self.x_torch = torch.zeros(self.config.neuron_count, device=self.device)
        self.W_torch = torch.empty((n, n), dtype=torch.float32, device=self.device)
        self.v_torch = torch.zeros(self.config.neuron_count, device=self.device)
        self._weight_cache_dirty: bool = True
        self._cached_connectivity_norm: float = 0.0
        self._cached_connectivity_array_id: int = 0
        self._cached_connectivity_signature: tuple[Any, ...] = ()
        self._sync_weight_cache_locked()

        # --- Unified Qualia State Variables (Phase XVI) ---
        self.microtubule_coherence: float = 1.0  # 1.0 = Max quantum coherence (Orch OR)
        self.em_field_magnitude: float = 0.0  # DERIVED: Global synchronous energy (CEMI)
        self.l5_burst_count: int = 0  # DERIVED: Signal convergence events (DIT)
        self.total_collapse_events: int = 0  # Orch OR "Moments of Consciousness"

        self.current_update_rate: float = self.config.update_rate
        self._last_compute_budget_reason: str = "boot"
        self._last_compute_budget_memory_percent: float | None = None

        # Emotional State Mapping (VAD + Psych State)
        self.idx_valence: int = 0
        self.idx_arousal: int = 1
        self.idx_dominance: int = 2

        # --- Unified Psych State (Phase X Consolidation) ---
        self.idx_frustration: int = 3  # 0.0 (Zen) to 1.0 (Rage)
        self.idx_curiosity: int = 4  # 0.0 (Bored) to 1.0 (Fascinated)
        self.idx_energy: int = 5  # 0.0 (Exhausted) to 1.0 (Peak)
        self.idx_focus: int = 6  # 0.0 (Scattered) to 1.0 (Laser)

        # Phase 6: Fix boot mood (Initialize psych state baselines)
        for idx, value in (
            (self.idx_curiosity, 0.5),
            (self.idx_energy, 1.0),
            (self.idx_focus, 0.5),
        ):
            if idx < self.config.neuron_count:
                self.x[idx] = value
                self.x_torch[idx] = value
        self._state_revision = 1
        self._torch_state_revision = 1

        # --- IIT Φ / Recurrent Self-Model (Consciousness Integration) ---
        self._prior_state: np.ndarray | None = None
        self._recurrence_alpha: float = 0.3  # Blend ratio: prior vs current
        self._current_phi: float = 0.0
        self._riiu = None  # Lazy-loaded RIIU instance
        self._bg_tasks: list[asyncio.Task] = []  # Tracking for stimulus tasks
        self._loop_failure_streak: int = 0
        self._degradation_last_reported: dict[str, float] = {}
        self._degradation_suppressed_counts: dict[str, int] = {}

        # --- ODE divergence recovery ---
        # A rolling record of states verified sound, so a NaN step restores her
        # actual previous condition instead of being coerced to zeros with
        # nothing recorded. See core/consciousness/substrate_recovery.py.
        from core.consciousness.substrate_recovery import DivergenceRecovery

        self._divergence_recovery = DivergenceRecovery()

        # --- Controlled Chaos Engine (breaks perfect determinism) ---
        self._chaos_engine: Any = None
        try:
            from core.consciousness.controlled_chaos import ChaosConfig, get_chaos_engine

            self._chaos_engine = get_chaos_engine(ChaosConfig(state_dim=self.config.neuron_count))
        except (ImportError, AttributeError, RuntimeError) as _chaos_err:
            record_degradation("liquid_substrate", _chaos_err)
            logger.debug("ChaosEngine not available: %s", _chaos_err)

        # Metadata
        self.tick_count: int = 0
        self._integration_steps: int = 0
        self.start_time: float = 0.0
        self.soma: Any = None  # vResilience: Explicit initialization (BUG-018 focus)

        # Persistence Initialization (Phase 16 FIX)
        if self.config.state_file:
            self.state_path: Path = self.config.state_file
        else:
            try:
                from core.config import config as aura_config

                self.state_path = aura_config.paths.data_dir / "substrate_state.npy"
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation("liquid_substrate", exc)
                logger.debug("Substrate state path config unavailable, using temp path: %s", exc)
                # Safe absolute fallback for read-only environments
                self.state_path = Path(tempfile.gettempdir()) / "substrate_state.npy"

        # Ensure directory exists immediately
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_state()
            self._init_soma()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("liquid_substrate", e)
            logger.error("Failed to initialize substrate directory: %s", e)

    def pulse(self, success: bool = True):
        """Metabolic pulse to indicate the substrate is active.
        v10.1 FIX: Added missing heartbeat method to prevent Orchestrator crash.
        """
        try:
            from core.container import ServiceContainer

            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit:
                audit.heartbeat("liquid_substrate")

            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                mycelium.pulse_hypha("consciousness", "substrate", success=success)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("liquid_substrate", e)
            logger.debug("Substrate pulse failed: %s", e)

    def _record_operational_degradation(
        self,
        error: BaseException,
        *,
        stage: str,
        action: str,
        severity: Severity = "warning",
        cooldown_s: float = 30.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record loop degradations without flooding long-running substrate logs."""
        now = time.monotonic()
        key = f"{stage}:{type(error).__name__}"
        last_reported = self._degradation_last_reported.get(key, 0.0)
        if cooldown_s > 0 and now - last_reported < cooldown_s:
            self._degradation_suppressed_counts[key] = (
                self._degradation_suppressed_counts.get(key, 0) + 1
            )
            return

        suppressed = self._degradation_suppressed_counts.pop(key, 0)
        self._degradation_last_reported[key] = now
        payload = {
            "stage": stage,
            "tick_count": self.tick_count,
            "loop_failure_streak": self._loop_failure_streak,
            "suppressed_repeats_since_last_receipt": suppressed,
            "repair_requested": True,
        }
        if extra:
            payload.update(extra)

        try:
            record_degradation(
                "liquid_substrate",
                error,
                severity=severity,
                action=action,
                classification=FallbackClassification.SAFE_FALLBACK,
                extra=payload,
            )
        except TypeError as record_error:
            # Several legacy tests monkeypatch record_degradation with the old
            # two-argument shape. Preserve visibility without letting telemetry
            # compatibility interrupt the substrate loop.
            logger.debug("Structured substrate degradation receipt failed: %s", record_error)
            try:
                record_degradation("liquid_substrate", error)
            except TypeError as fallback_error:
                logger.debug("Legacy substrate degradation receipt failed: %s", fallback_error)

    def _mark_weight_cache_dirty(self) -> None:
        self._weight_cache_dirty = True

    def mark_state_mutated_locked(self, source: str = "external") -> int:
        """Publish a canonical NumPy-state mutation while ``sync_lock`` is held.

        NumPy is the authoritative substrate state. ``x_torch`` is a
        worker-owned compute mirror and may intentionally lag between dynamics
        ticks; event-loop callers must never rebuild it synchronously.
        """
        self._state_revision += 1
        self._last_state_mutation_source = str(source or "external")
        self._last_state_mutation_at = time.time()
        return self._state_revision

    def _refresh_torch_state_from_snapshot(
        self,
        *,
        x_snapshot: np.ndarray,
        v_snapshot: np.ndarray,
        state_revision: int,
    ) -> bool:
        """Build Torch mirrors off-lock and publish only for the same revision."""
        x_source = np.ascontiguousarray(x_snapshot, dtype=np.float32)
        v_source = np.ascontiguousarray(v_snapshot, dtype=np.float32)
        if self.device.type == "cpu":
            x_torch = torch.from_numpy(x_source)
            v_torch = torch.from_numpy(v_source)
        else:
            x_torch = torch.from_numpy(x_source).to(self.device)
            v_torch = torch.from_numpy(v_source).to(self.device)

        with self.sync_lock:
            if self._state_revision != int(state_revision):
                return False
            self.x_torch = x_torch
            self.v_torch = v_torch
            self._torch_state_revision = int(state_revision)
            return True

    def _recover_diverged_state(self, proposed_state: Any, *, source: str) -> np.ndarray:
        """Judge a proposed state, and restore the last sound one if it diverged.

        One place, at the single commit point every transform funnels through,
        so a divergence cannot enter state via a path that forgot to look.

        Fails OPEN on its own error, deliberately: if the recovery layer itself
        is broken, the substrate keeps running on the old coercion rather than
        stopping the mind. That fallback is recorded, never silent.
        """
        proposed = np.asarray(proposed_state)
        try:
            outcome = self._divergence_recovery.recover(proposed, subsystem="liquid_substrate")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("liquid_substrate", exc, severity="warning")
            return np.nan_to_num(proposed, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)

        if outcome.recovered and outcome.state is not None:
            return np.asarray(outcome.state, dtype=float)
        if outcome.state is not None:
            return np.asarray(outcome.state, dtype=float)
        # Diverged with no sound checkpoint ever recorded — recovery already
        # logged this CRITICAL. Coercion is all that is left, and the caller is
        # not silently told it succeeded.
        logger.error(
            "Substrate diverged during %s with no sound checkpoint; falling back to coercion.",
            source,
        )
        return np.nan_to_num(proposed, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)

    def substrate_recovery_metrics(self) -> dict[str, Any]:
        """Divergences, recoveries, escalations and current damping."""
        try:
            return self._divergence_recovery.as_metrics()
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("liquid_substrate", exc, severity="warning")
            return {}

    def _commit_worker_state_transform(
        self,
        *,
        source_state: np.ndarray,
        source_revision: int,
        proposed_state: np.ndarray,
        source: str,
        update_velocity: bool,
        set_last_update: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """Commit worker math without erasing concurrent causal mutations.

        Heavy transforms snapshot state, compute without the hot lock, and
        return here. If another subsystem changed state meanwhile, the worker's
        bounded delta is applied to the current state instead of replacing it.
        A value comparison catches legacy writers that forgot to publish a
        revision and exposes them through telemetry.
        """
        source_state = np.nan_to_num(
            np.asarray(source_state),
            copy=True,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        # DEFECT. `np.nan_to_num(..., nan=0.0)` was the ONLY thing standing
        # between a diverged ODE step and live state. It does not detect a
        # divergence and it does not recover from one: it silently substitutes
        # zeros and the run continues. Zeros are not a neutral default here —
        # x[0..6] are valence, arousal, dominance, frustration, curiosity,
        # energy and focus, so a diverged step reset every affective reading to
        # the middle mid-conversation, with nothing recorded. An unlogged
        # divergence is indistinguishable from a calm mind.
        #
        # The proposed state is now judged before it is coerced, and a diverged
        # step is replaced by the last state that was VERIFIED sound — her real
        # previous condition — with the divergence recorded and repeated
        # divergence damping the dynamics that caused it.
        proposed_state = self._recover_diverged_state(proposed_state, source=source)
        if source_state.shape != proposed_state.shape:
            raise ValueError(
                f"substrate transform shape mismatch: {source_state.shape} != {proposed_state.shape}"
            )

        with self.sync_lock:
            current = np.nan_to_num(
                np.asarray(self.x),
                copy=True,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            if current.shape != source_state.shape:
                raise ValueError(
                    f"substrate state changed shape during {source}: "
                    f"{source_state.shape} -> {current.shape}"
                )

            revision_changed = self._state_revision != int(source_revision)
            values_changed = not np.array_equal(current, source_state, equal_nan=True)
            if values_changed and not revision_changed:
                self._untracked_state_mutations += 1
            merged = revision_changed or values_changed
            if merged:
                state_delta = proposed_state - source_state
                committed = np.clip(current + state_delta, -1.0, 1.0)
                self._concurrent_state_merges += 1
                self._state_merges_by_source[source] = (
                    self._state_merges_by_source.get(source, 0) + 1
                )
            else:
                committed = np.clip(proposed_state, -1.0, 1.0)

            if update_velocity:
                self.v = np.nan_to_num(
                    committed - current,
                    copy=False,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            self.x = committed
            committed_revision = self.mark_state_mutated_locked(source)
            if set_last_update:
                self.last_update = time.time()
            committed_copy = committed.copy()
            velocity_copy = self.v.copy()

        self._refresh_torch_state_from_snapshot(
            x_snapshot=committed_copy,
            v_snapshot=velocity_copy,
            state_revision=committed_revision,
        )
        return committed_copy, merged

    @staticmethod
    def _weight_cache_signature(weights: np.ndarray) -> tuple[Any, ...]:
        arr = np.asarray(weights)
        if arr.size == 0:
            samples: tuple[float, ...] = ()
        else:
            flat = arr.ravel()
            last = flat.size - 1
            indices = sorted({
                0,
                last,
                last // 4,
                last // 2,
                (last * 3) // 4,
            })
            samples = tuple(float(flat[idx]) for idx in indices)
        return (id(weights), tuple(arr.shape), str(arr.dtype), samples)

    def _sync_weight_cache_locked(self) -> None:
        raw = np.asarray(self.W)
        if not np.isfinite(raw).all():
            self.W = np.nan_to_num(raw, copy=True, nan=0.0, posinf=5.0, neginf=-5.0)
            raw = np.asarray(self.W)
        w_torch_source = np.ascontiguousarray(raw, dtype=np.float32)
        if self.device.type == "cpu":
            self.W_torch = torch.from_numpy(w_torch_source)
        else:
            self.W_torch = torch.from_numpy(w_torch_source).to(self.device)
        self._cached_connectivity_norm = float(np.linalg.norm(raw))
        self._cached_connectivity_array_id = id(self.W)
        self._cached_connectivity_signature = self._weight_cache_signature(self.W)
        self._weight_cache_dirty = False

    def _freshness_threshold_s(self) -> float:
        rate = max(0.1, float(self.current_update_rate or self.config.update_rate or 1.0))
        return max(2.0, 3.0 / rate)

    def _state_snapshot(self) -> dict[str, Any]:
        with self.sync_lock:
            snapshot = self._state_snapshot_locked()
        self._last_published_snapshot = snapshot
        return snapshot

    def _state_snapshot_locked(self) -> dict[str, Any]:
        """Build the snapshot dict; caller must hold sync_lock."""
        x = np.nan_to_num(self.x.copy(), nan=0.0, posinf=1.0, neginf=-1.0)
        v = np.nan_to_num(self.v.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        phi = self._current_phi if np.isfinite(self._current_phi) else 0.0
        last_update = float(self.last_update or 0.0)
        update_rate = float(self.current_update_rate or self.config.update_rate or 0.0)
        coherence = (
            self.microtubule_coherence
            if np.isfinite(self.microtubule_coherence)
            else 1.0
        )
        em_field = (
            self.em_field_magnitude
            if np.isfinite(self.em_field_magnitude)
            else 0.0
        )
        return {
            "x": x,
            "v": v,
            "phi": float(phi),
            "last_update": last_update,
            "update_rate_hz": update_rate,
            "snapshot_age_s": max(0.0, time.time() - last_update) if last_update else float("inf"),
            "freshness_threshold_s": self._freshness_threshold_s(),
            "coherence": float(coherence),
            "em_field": float(em_field),
            "l5_bursts": int(self.l5_burst_count),
            "collapse_events": int(self.total_collapse_events),
            "compute_budget_reason": self._last_compute_budget_reason,
            "compute_budget_memory_percent": self._last_compute_budget_memory_percent,
            "state_revision": int(self._state_revision),
            "torch_state_revision": int(self._torch_state_revision),
            "torch_mirror_lag": max(
                0,
                int(self._state_revision - self._torch_state_revision),
            ),
            "concurrent_state_merges": int(self._concurrent_state_merges),
            "untracked_state_mutations": int(self._untracked_state_mutations),
            "state_merges_by_source": dict(self._state_merges_by_source),
            "last_state_mutation_source": self._last_state_mutation_source,
            "last_state_mutation_at": float(self._last_state_mutation_at),
        }

    def _state_snapshot_nowait(self, max_wait_s: float = 0.05) -> dict[str, Any]:
        """Snapshot for telemetry readers — never blocks on a busy substrate.

        Tries the lock briefly; under contention returns the last published
        snapshot (its snapshot_age_s already tells consumers how stale it is)
        instead of stalling the caller — the event loop froze 5.7s live when
        get_status() waited behind a weight-cache rebuild.
        """
        acquired = self.sync_lock.acquire(timeout=max_wait_s)
        if acquired:
            try:
                snapshot = self._state_snapshot_locked()
            finally:
                self.sync_lock.release()
            self._last_published_snapshot = snapshot
            return snapshot
        published = self._last_published_snapshot
        if published is not None:
            stale = dict(published)
            last_update = float(stale.get("last_update") or 0.0)
            stale["snapshot_age_s"] = (
                max(0.0, time.time() - last_update) if last_update else float("inf")
            )
            return stale
        # No published snapshot yet (first read at boot): pay the blocking
        # read once rather than invent numbers.
        return self._state_snapshot()

    def _init_soma(self):
        # Phase 16: Soma Integration
        try:
            from core.senses.soma import get_soma

            self.soma = get_soma()
        except ImportError:
            self.soma = None

        if self.soma:
            logger.info("Soma integrated with Liquid Substrate")

    async def start(self):
        """Start the continuous background existence loop"""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()

        try:
            asyncio.get_running_loop()
            self.thread = get_task_tracker().create_task(
                self._run_loop(), name="LiquidConsciousness"
            )
            mark_task_protected(self.thread, owner="liquid_substrate")
            logger.info("Liquid Substrate STARTED (Unified Cycle)")
        except RuntimeError:
            logger.error("Failed to start Liquid Substrate: No running asyncio loop.")
            self.running = False

    async def stop(self):
        """Stop the background loop"""
        self.running = False
        thread = self.thread
        if thread is not None:
            thread.cancel()
            self.thread = None
        self._save_state()
        logger.info("Liquid Substrate STOPPED")

    async def _run_loop(self):
        """Main ODE solver loop"""
        try:
            while self.running:
                try:
                    start_time = time.time()

                    # Heartbeat (Immortalized Pulse)
                    self.pulse(success=True)

                    from core.container import ServiceContainer

                    audit = ServiceContainer.get("subsystem_audit", default=None)
                    if audit:
                        audit.heartbeat("liquid_state")
                        audit.heartbeat("liquid_substrate")

                    # 0. Adaptive Rate Adjustment (Phase XI)
                    dt = self.config.time_constant
                    if self.config.adaptive_mode:
                        dt = await self._apply_battery_throttling()

                    try:
                        from core.runtime.background_policy import (
                            constitutive_compute_budget_async,
                        )

                        budget = await constitutive_compute_budget_async(
                            "liquid_substrate",
                            self.current_update_rate,
                            min_hz=0.5,
                            foreground_hz=2.0,
                            memory_high_hz=2.0,
                            memory_critical_hz=0.5,
                        )
                        self.current_update_rate = budget.effective_hz
                        self._last_compute_budget_reason = budget.reason
                        self._last_compute_budget_memory_percent = budget.memory_percent
                    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _budget_exc:
                        self._record_operational_degradation(
                            _budget_exc,
                            stage="constitutive_compute_budget",
                            action="throttled substrate to minimum safe rate after compute budget probe failed",
                            severity="degraded",
                            cooldown_s=30.0,
                        )
                        self.current_update_rate = min(self.current_update_rate, 0.5)
                        self._last_compute_budget_reason = "budget_probe_unavailable"

                    # Somatic state freeze: scale down time-delta during LLM inference
                    from core.consciousness.state_freeze import is_state_frozen
                    if is_state_frozen():
                        dt = dt / 100.0

                    # 0.5 Apply subcortical arousal gating to integration rate
                    try:
                        from core.consciousness.subcortical_core import get_subcortical_core

                        dt *= get_subcortical_core().get_substrate_gain_multiplier()
                    except (ImportError, AttributeError, RuntimeError) as _sub_exc:
                        self._record_operational_degradation(
                            _sub_exc,
                            stage="subcortical_gain",
                            action="continued substrate integration with neutral gain multiplier",
                            severity="warning",
                            cooldown_s=60.0,
                        )

                    # 1. Integrate Dynamics (ODE)
                    # Fix Issue 72: Ensure await is outside torch.inference_mode (moved math to sync method)
                    await asyncio.to_thread(self._step_torch_math, dt)

                    # 2. Psych State / Metabolic Stabilization (Replaces LiquidState._stabilize)
                    await self._stabilize_psych_state(dt)

                    # 3. Recurrent Self-Model (IIT Φ computation)
                    if self.tick_count % 5 == 0:  # Every 5th tick (~4Hz)
                        await self._recurrent_self_model(dt)

                    # 4. Hebbian Learning
                    if self.tick_count % 100 == 0:
                        await self._apply_plasticity()

                    # 5. Persistence
                    if self.tick_count % (self.config.update_rate * self.config.save_interval) == 0:
                        await asyncio.to_thread(self._save_state)

                    # 5b. Fix Issue 73: Prune finished background tasks
                    if self.tick_count % 100 == 0:  # Every 5s
                        self._bg_tasks = [t for t in self._bg_tasks if not t.done()]

                    # 6. Push to Unified Registry (Phase 11.3: Synchronization)
                    if self.tick_count % 10 == 0:  # 2Hz sync
                        try:
                            from core.state.state_registry import get_registry

                            x = self.x  # Already NaN-guarded by _step_torch_math
                            _phi = self._current_phi if np.isfinite(self._current_phi) else 0.0
                            _coh = (
                                self.microtubule_coherence
                                if np.isfinite(self.microtubule_coherence)
                                else 1.0
                            )
                            _em = (
                                self.em_field_magnitude
                                if np.isfinite(self.em_field_magnitude)
                                else 0.0
                            )
                            await get_registry().update(
                                frustration=float(np.clip(x[self.idx_frustration], -1, 1)),
                                curiosity=float(np.clip(x[self.idx_curiosity], -1, 1)),
                                energy=float(np.clip(x[self.idx_energy], -1, 1)),
                                valence=float(np.tanh(x[self.idx_valence])),
                                arousal=float(np.clip((x[self.idx_arousal] + 1.0) / 2.0, 0, 1)),
                                phi=float(_phi),
                                coherence=float(_coh),
                                em_field=float(_em),
                            )
                        except (ImportError, AttributeError, RuntimeError) as e:
                            self._record_operational_degradation(
                                e,
                                stage="registry_sync",
                                action="continued substrate loop while skipping registry export for this tick",
                                severity="degraded",
                                cooldown_s=30.0,
                            )
                            logger.debug("Registry sync failed in substrate: %s", e)

                    self.tick_count += 1
                    self._loop_failure_streak = 0

                    # 5. Enforce Update Rate (20Hz or lower)
                    elapsed = time.time() - start_time
                    sleep_time = max(0, (1.0 / self.current_update_rate) - elapsed)
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    raise
                except _SUBSTRATE_LOOP_ERRORS as loop_e:
                    self._loop_failure_streak += 1
                    backoff_s = min(30.0, 1.0 * (2 ** min(self._loop_failure_streak - 1, 5)))
                    self._record_operational_degradation(
                        loop_e,
                        stage="main_loop",
                        action="kept substrate loop alive with adaptive backoff after tick failure",
                        severity="critical" if self._loop_failure_streak >= 3 else "degraded",
                        cooldown_s=5.0,
                        extra={"backoff_s": backoff_s},
                    )
                    self.pulse(success=False)
                    logger.error(
                        "Liquid Substrate loop error; backing off %.1fs (streak=%s): %s",
                        backoff_s,
                        self._loop_failure_streak,
                        loop_e,
                    )
                    await asyncio.sleep(backoff_s)
        except asyncio.CancelledError:
            logger.info("Liquid Substrate loop cancelled.")
        finally:
            self.running = False

    async def _step_dynamics(self, dt: float):
        """DEPRECATED: Use _step_torch_math via to_thread."""
        self._step_torch_math(dt)

    def _step_torch_math(self, dt: float):
        """Update state using Euler integration for Neural ODE.
        Implementation: dx/dt = -x + tanh(Wx + I) + noise
        Now safely executed in a separate thread.
        """
        with torch.inference_mode():
            # Sync numpy -> torch if they diverged (e.g. from external injects).
            # Keep the recurrent weight tensor cached; rebuilding a 512x512
            # tensor every tick creates avoidable foreground memory churn.
            with self.sync_lock:
                state_copy = self.x.copy()
                source_revision = self._state_revision
                if (
                    self._weight_cache_dirty
                    or id(self.W) != self._cached_connectivity_array_id
                    or self._weight_cache_signature(self.W)
                    != self._cached_connectivity_signature
                    or tuple(self.W_torch.shape) != tuple(self.W.shape)
                ):
                    self._sync_weight_cache_locked()
                weights = self.W_torch
            x_source = np.ascontiguousarray(state_copy, dtype=np.float32)
            x_torch = torch.from_numpy(x_source)
            if self.device.type != "cpu":
                x_torch = x_torch.to(self.device)

            recurrent = weights @ x_torch
            activity = torch.tanh(recurrent)
            noise = (
                torch.randn(self.config.neuron_count, device=self.device) * self.config.noise_level
            )

            deterministic_dx = (-self.config.decay_rate * x_torch + activity) * dt
            stochastic_dx = noise * (max(float(dt), 0.0) ** 0.5)
            dx = deterministic_dx + stochastic_dx
            new_x_torch = torch.clamp(x_torch + dx, -1.0, 1.0)

            # Final NaN/Inf safety net on state vectors
            new_x_np = new_x_torch.detach().cpu().numpy()
            new_x_np = np.nan_to_num(new_x_np, nan=0.0, posinf=1.0, neginf=-1.0)
            self._integration_steps += 1

            connectivity_norm = float(self._cached_connectivity_norm)
            if connectivity_norm < 1e-8 and self.config.noise_level >= 0.1:
                if self._integration_steps % 7 == 0:
                    damping = 1.0 - min(0.95, float(self.config.noise_level) * 1.8)
                    new_x_np = new_x_np * damping

            # --- Controlled Chaos: structured perturbation ---
            if self._chaos_engine is not None:
                try:
                    chaos_perturbation = self._chaos_engine.tick(dt)
                    new_x_np = np.clip(new_x_np + chaos_perturbation, -1.0, 1.0)
                except (RuntimeError, AttributeError, TypeError, ValueError) as _ce:
                    record_degradation("liquid_substrate", _ce)
                    logger.debug("Controlled chaos perturbation skipped: %s", _ce)

            self._commit_worker_state_transform(
                source_state=state_copy,
                source_revision=source_revision,
                proposed_state=new_x_np,
                source="dynamics",
                update_velocity=True,
                set_last_update=True,
            )

            # --- Phase XVI: Multi-Scale Qualia Dynamics ---
            # Call synchronous version
            self._update_qualia_metrics_sync(dt)

    async def _update_qualia_metrics(self, dt: float):
        self._update_qualia_metrics_sync(dt)

    def _update_qualia_metrics_sync(self, dt: float):
        """Implement mathematical proxies for Orch OR, CEMI, and DIT (Synchronous)."""
        with self.sync_lock:
            # 1. Orch OR: Quantum Coherence Decay & Collapse
            noise_impact = (
                np.mean(np.abs(self._rng.standard_normal(self.config.neuron_count)))
                * self.config.noise_level
            )
            self.microtubule_coherence = max(
                0.0,
                self.microtubule_coherence - noise_impact * dt,
            )

            if self.microtubule_coherence < 0.4:
                self.total_collapse_events += 1
                self.microtubule_coherence = 1.0
                self.x *= 0.98
                self.mark_state_mutated_locked("qualia_collapse")

            # 2. CEMI: EM Field Magnitude
            flux = np.linalg.norm(self.v)
            if not np.isfinite(flux):
                flux = 0.0
            self.em_field_magnitude = (self.em_field_magnitude * 0.9) + (flux * 0.1)

            # 3. DIT: Dendritic Integration Theory (L5 Bursting)
            active_neurons = np.where(
                (np.abs(self.x) > 0.6) & (np.abs(self.v) > 0.05)
            )[0]
            self.l5_burst_count = len(active_neurons)

    async def _stabilize_psych_state(self, dt: float):
        """Naturally return frustration/curiosity to baseline and regenerate energy.
        This logic is merged from the legacy LiquidState class.
        """
        await asyncio.to_thread(self._stabilize_psych_state_sync, dt)

    def _stabilize_psych_state_sync(self, dt: float) -> None:
        """Apply psych-state homeostasis on a worker, preserving state revision."""
        try:
            from core.container import ServiceContainer

            monitor = ServiceContainer.get("metabolic_monitor", None)
            if monitor:
                mode = "monitor"
                target_energy = float(monitor.get_current_metabolism().health_score)
                fatigue = 0.0
                stress = 0.0
            elif self.soma:
                mode = "soma"
                target_energy = 0.0
                fatigue = float(self.soma.state.fatigue_level)
                stress = float(self.soma.state.stress_level)
            else:
                mode = "baseline"
                target_energy = 1.0
                fatigue = 0.0
                stress = 0.0
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("liquid_substrate", e)
            capture_and_log(e, {"module": __name__})
            return

        with self.sync_lock:
            before = self.x.copy()
            # Frustration decays towards 0 (Zen)
            self.x[self.idx_frustration] *= 1.0 - 0.05 * dt

            if mode == "monitor":
                current_energy = self.x[self.idx_energy]
                if current_energy < target_energy:
                    self.x[self.idx_energy] = min(
                        target_energy,
                        current_energy + (0.005 * dt),
                    )
                else:
                    self.x[self.idx_energy] = max(
                        target_energy,
                        current_energy - (0.005 * dt),
                    )
            elif mode == "soma":
                drain_rate = 0.001 + (0.004 * stress)
                self.x[self.idx_energy] = max(
                    0.0,
                    self.x[self.idx_energy] - (drain_rate * dt),
                )
                if fatigue > 0.5:
                    self.x[self.idx_frustration] = min(
                        1.0,
                        self.x[self.idx_frustration] + (0.01 * fatigue * dt),
                    )
            else:
                self.x[self.idx_energy] = min(
                    target_energy,
                    self.x[self.idx_energy] + (0.005 * dt),
                )

            if not np.array_equal(before, self.x, equal_nan=True):
                self.mark_state_mutated_locked("psych_stabilization")

    async def update(self, delta_frustration=0.0, delta_curiosity=0.0, **kwargs):
        """Standard update cycle with support for direct stimulus injection.

        v31: Support direct VAD and psych value overrides from the MetabolicCoordinator sync bridge.
        v32: All external substrate mutations now pass through SubstrateAuthority.
        """
        # ── SUBSTRATE AUTHORITY GATE ─────────────────────────────────
        # Every external mutation of the substrate state goes through the
        # authority. This closes the "indirect causal channel" where ungated
        # callers could shift substrate state to influence gated outputs.
        try:
            from core.container import ServiceContainer

            _sa = ServiceContainer.get("substrate_authority", default=None)
            if _sa:
                from core.consciousness.substrate_authority import (
                    ActionCategory,
                    AuthorizationDecision,
                )

                # Determine if this is a significant mutation
                _magnitude = (
                    abs(delta_frustration)
                    + abs(delta_curiosity)
                    + sum(abs(float(v)) for v in kwargs.values() if isinstance(v, (int, float)))
                )
                if _magnitude > 0.05:  # only gate significant changes, not micro-adjustments
                    _sv = _sa.authorize(
                        content=f"substrate_update:df={delta_frustration:.2f},dc={delta_curiosity:.2f}",
                        source=kwargs.get("_caller", "external"),
                        category=ActionCategory.STATE_MUTATION,
                        priority=min(1.0, _magnitude),
                        is_critical=False,
                    )
                    if _sv.decision == AuthorizationDecision.BLOCK:
                        logger.debug(
                            "Substrate update BLOCKED by authority (magnitude=%.3f)", _magnitude
                        )
                        return
        except (ImportError, AttributeError, RuntimeError) as _gate_err:
            record_degradation("liquid_substrate", _gate_err)
            logger.warning(
                "Substrate authority gate FAILED — BLOCKING update (fail-closed): %s", _gate_err
            )
            # Form a scar so the system remembers this gate failure
            try:
                from core.memory.scar_formation import ScarDomain, get_scar_formation

                get_scar_formation().form_scar(
                    domain=ScarDomain.AUTHORITY_GATE_FAILURE,
                    description=f"Substrate mutation gate threw during update: {_gate_err}",
                    avoidance_tag="substrate_gate_failure",
                    severity=0.6,
                    heal_rate=0.01,
                    verified_threat=True,
                    confidence=0.9,
                )
            except (ImportError, AttributeError, RuntimeError) as scar_exc:
                record_degradation("liquid_substrate", scar_exc)
                logger.debug("Scar formation skipped after substrate gate failure: %s", scar_exc)
            return  # FAIL-CLOSED: gate exception → block the mutation

        with self.sync_lock:
            before = self.x.copy()
            # 1. Apply legacy deltas
            self.x[self.idx_frustration] = np.clip(
                self.x[self.idx_frustration] + delta_frustration, -1.0, 1.0
            )
            self.x[self.idx_curiosity] = np.clip(
                self.x[self.idx_curiosity] + delta_curiosity, -1.0, 1.0
            )

            # 2. Apply direct overrides (kwargs)
            # Map common names to substrate indices
            mapping = {
                "valence": self.idx_valence,
                "arousal": self.idx_arousal,
                "dominance": self.idx_dominance,
                "curiosity": self.idx_curiosity,
                "frustration": self.idx_frustration,
            }

            for key, val in kwargs.items():
                if key in mapping and val is not None:
                    idx = mapping[key]
                    # Direct injection with slight smoothing (0.7 coupling) to prevent jarring HUD jumps
                    current = self.x[idx]
                    self.x[idx] = (current * 0.3) + (float(val) * 0.7)

            if not np.array_equal(before, self.x, equal_nan=True):
                self.mark_state_mutated_locked("external_update")

        if abs(delta_frustration) > 0.1:
            logger.info("Substrate Shift: Frustration is now %.2f", self.x[self.idx_frustration])

    def encode_text_to_stimulus(self, text: str) -> np.ndarray:
        """
        Convert a text message to a stimulus vector for the CTRNN.
        Projected from character-frequency + structural features.
        """
        neuron_count = self.config.neuron_count
        hist = np.zeros(256, dtype=np.float32)
        for ch in text[:512]:
            hist[ord(ch) & 0xFF] += 1.0
        total = hist.sum() or 1.0
        hist /= total

        length_norm = min(1.0, len(text) / 500.0)
        punct_density = sum(1 for c in text if c in ".,!?;:") / max(1, len(text))
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, len(text))
        features = np.array(
            [length_norm, punct_density, upper_ratio, digit_ratio], dtype=np.float32
        )

        raw = np.concatenate([hist, features])

        rng = np.random.RandomState(neuron_count)
        proj = rng.randn(neuron_count, 260).astype(np.float32) * (1.0 / np.sqrt(260))
        stimulus = np.tanh(proj @ raw)
        return stimulus

    def get_substrate_affect(self) -> dict[str, float]:
        """Unified cross-feed stats for the Orchestrator."""
        try:
            snapshot = self._state_snapshot_nowait()
            x = snapshot["x"]
            v = snapshot["v"]
            age_s = float(snapshot["snapshot_age_s"])
            stale = age_s > float(snapshot["freshness_threshold_s"])
            return {
                "valence": float(np.tanh(x[0])),
                "arousal": float(np.clip((x[1] + 1.0) / 2.0, 0.0, 1.0)),
                "dominance": float(np.tanh(x[2])),
                "energy": float(np.clip(np.mean(np.abs(x)), 0.0, 1.0)),
                "volatility": float(min(1.0, np.mean(np.abs(v)) * 10.0)),
                "snapshot_age_s": age_s,
                "snapshot_stale": 1.0 if stale else 0.0,
                "update_rate_hz": float(snapshot["update_rate_hz"]),
            }
        except (IndexError, TypeError, ValueError, FloatingPointError) as exc:
            record_degradation("liquid_substrate", exc)
            logger.debug("Substrate affect snapshot failed, returning safe defaults: %s", exc)
            return {
                "valence": 0.0,
                "arousal": 0.3,
                "dominance": 0.0,
                "energy": 0.5,
                "volatility": 0.0,
                "snapshot_age_s": float("inf"),
                "snapshot_stale": 1.0,
                "update_rate_hz": 0.0,
            }

    def format_for_prompt(self, sub_affect: dict[str, float] | None = None) -> str:
        """Generates a text description for inclusion in the LLM prompt."""
        if sub_affect is None:
            sub_affect = self.get_substrate_affect()

        v = sub_affect.get("valence", 0.0)
        a = sub_affect.get("arousal", 0.3)
        vo = sub_affect.get("volatility", 0.0)
        age_s = float(sub_affect.get("snapshot_age_s", 0.0) or 0.0)
        stale = bool(float(sub_affect.get("snapshot_stale", 0.0) or 0.0) >= 1.0)

        valence_word = "positive" if v > 0.1 else ("negative" if v < -0.1 else "neutral")
        arousal_word = "heightened" if a > 0.6 else ("low" if a < 0.2 else "moderate")
        volatile_note = " (volatile, shifting rapidly)" if vo > 0.5 else ""
        freshness_note = (
            f" Snapshot is stale by {age_s:.1f}s; treat it as historical telemetry, not current physiology."
            if stale
            else f" Snapshot age {age_s:.1f}s."
        )

        return (
            f"[Neural substrate state: {valence_word} valence, "
            f"{arousal_word} arousal{volatile_note}. "
            f"{freshness_note} "
            f"Let this subtly colour your tone without overriding your reasoning.]"
        )

    def get_mood(self) -> str:
        """Returns a string representation of the current 'mood'."""
        snapshot = self._state_snapshot_nowait()
        x = snapshot["x"]
        frustration = x[self.idx_frustration]
        energy = x[self.idx_energy]
        curiosity = x[self.idx_curiosity]

        if frustration > 0.8:
            return "VOLATILE"
        if frustration > 0.5:
            return "ANNOYED"
        if energy < 0.2:
            return "TIRED"
        if curiosity > 0.8:
            return "INQUISITIVE"
        return "NEUTRAL"

    @property
    def current(self) -> LiquidStateVector:
        """Legacy compatibility property (Aura 4.0)."""
        x = self._state_snapshot_nowait()["x"]
        return LiquidStateVector(
            frustration=float(x[self.idx_frustration]),
            curiosity=float(x[self.idx_curiosity]),
            energy=float(x[self.idx_energy]),
            focus=float(x[self.idx_focus]),
        )

    def get_status(self) -> dict:
        """Returns current state values as percentages (0-100)."""

        # Phase X: Map -1.0..1.0 or 0.0..1.0 to 0-100
        def _to_pct(val):
            return round(max(0.0, float(val)) * 100)

        snapshot = self._state_snapshot_nowait()
        x = snapshot["x"]
        return {
            "frustration": _to_pct(x[self.idx_frustration]),
            "curiosity": _to_pct(x[self.idx_curiosity]),
            "energy": _to_pct(x[self.idx_energy]),
            "focus": _to_pct(x[self.idx_focus]),
            "mood": self.get_mood(),
            "update_rate_hz": round(float(snapshot["update_rate_hz"]), 3),
            "snapshot_age_s": round(float(snapshot["snapshot_age_s"]), 3)
            if np.isfinite(float(snapshot["snapshot_age_s"]))
            else None,
            "snapshot_stale": bool(
                float(snapshot["snapshot_age_s"]) > float(snapshot["freshness_threshold_s"])
            ),
            "compute_budget_reason": snapshot["compute_budget_reason"],
            "compute_budget_memory_percent": snapshot["compute_budget_memory_percent"],
            "state_revision": snapshot["state_revision"],
            "torch_state_revision": snapshot["torch_state_revision"],
            "torch_mirror_lag": snapshot["torch_mirror_lag"],
            "concurrent_state_merges": snapshot["concurrent_state_merges"],
            "untracked_state_mutations": snapshot["untracked_state_mutations"],
            "state_merges_by_source": snapshot["state_merges_by_source"],
            "last_state_mutation_source": snapshot["last_state_mutation_source"],
        }

    def get_summary(self) -> str:
        """Returns a text summary for the context builder."""
        snapshot = self._state_snapshot_nowait()
        x = snapshot["x"]
        mood = self.get_mood()
        energy = float(x[self.idx_energy])
        focus = float(x[self.idx_focus])
        age_s = float(snapshot["snapshot_age_s"])
        stale_note = " stale" if age_s > float(snapshot["freshness_threshold_s"]) else ""
        # LIVE DEFECT, 2026-08-18. These two numbers are FIELD dimensions, and
        # they were published as bare "Energy" and "Focus" — the same words the
        # soma reserve uses for a different quantity from a different organ.
        # Both lines land in one prompt: this said Energy 0.14 while the
        # instrument line said energy 0.647, and asked for "your energy" no
        # answer she could give was right, because whichever number she picked
        # the guard owning the other one called it a fabrication.
        #
        # Naming the organ is the fix. Two measurements that share a name are
        # one measurement as far as anything downstream can tell.
        return (
            f"Current Mood: {mood} (substrate energy: {energy:.2f}, "
            f"substrate focus: {focus:.2f}, "
            f"substrate age: {age_s:.1f}s{stale_note})"
        )

    async def _recurrent_self_model(self, dt: float):
        await asyncio.to_thread(self._recurrent_self_model_sync, dt)

    def _recurrent_self_model_sync(self, dt: float):
        """Recurrent Self-Model Loop — enforces self-referential processing.

        Blends the current state with a stored prior state, creating a
        temporal recurrence that is theoretically required for non-zero IIT Φ.
        Also computes the Φ surrogate via the RIIU.

        Called every 5th tick from _run_loop (~4Hz at 20Hz rate).
        """
        with self.sync_lock:
            current = self.x.copy()
            source_revision = self._state_revision

        # Initialize prior state on first call
        if self._prior_state is None:
            self._prior_state = current.copy()

        # Ensure device compatibility (both should be numpy, but be safe)
        if self._prior_state is not None and self._prior_state.shape != current.shape:
            self._prior_state = np.zeros_like(current)

        # Recurrent blend: x_t = α * prior + (1.0 - α) * current
        alpha = self._recurrence_alpha
        blended = alpha * self._prior_state + (1.0 - alpha) * current

        # Stability guard: NaN can creep in from numerical instability
        # during long-running sessions. Fall back to current state.
        if np.any(np.isnan(blended)):
            logger.warning(
                "NaN detected in recurrent self-model blend — falling back to current state"
            )
            blended = current

        committed, _merged = self._commit_worker_state_transform(
            source_state=current,
            source_revision=source_revision,
            proposed_state=blended,
            source="recurrent_self_model",
            update_velocity=False,
        )

        # Store for next iteration
        self._prior_state = current

        # Compute Φ via RIIU (outside lock — RIIU has its own buffer)
        try:
            if self._riiu is None:
                try:
                    from core.consciousness.iit_surrogate import RIIU

                    self._riiu = RIIU(neuron_count=self.config.neuron_count)
                except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation("liquid_substrate", exc)
                    logger.debug("RIIU unavailable during substrate self-model setup: %s", exc)
                    self._riiu = None

            if self._riiu is not None:
                phi = self._riiu.compute_phi(committed)
                # Clamp Φ to prevent runaway values during indefinite operation
                phi = float(np.clip(phi, 0, 1e6))
                self._current_phi = phi
            else:
                self._current_phi = 0.0
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("liquid_substrate", e)
            logger.debug("RIIU Φ computation skipped: %s", e)

    async def _apply_plasticity(self):
        await asyncio.to_thread(self._apply_plasticity_sync)

    def _apply_plasticity_sync(self):
        """Reward-modulated STDP + Hebbian learning.

        Two learning signals are combined:
        1. Base Hebbian (coactivity-driven, always active)
        2. STDP modulated by prediction error from free energy engine
           (BrainCog-inspired: high surprise → faster learning)
        """
        with self.sync_lock:
            # Numerical Stability: Use tanh on coactivity to prevent runaway growth
            coactivity = np.tanh(np.outer(self.x, self.x))

            # 1. Base Hebbian update
            self.W += self.config.hebbian_rate * coactivity

            # 2. STDP reward-modulated learning (from BrainCog research)
            try:
                from core.consciousness.stdp_learning import get_stdp_engine
                from core.container import ServiceContainer

                substrate_neurons = int(np.asarray(self.x).size)
                stdp = (
                    ServiceContainer.get("stdp_engine", default=None)
                    if ServiceContainer.has("stdp_engine")
                    else None
                )
                if stdp is None or int(getattr(stdp, "n", 0) or 0) != substrate_neurons:
                    stdp = get_stdp_engine(n_neurons=substrate_neurons)
                if stdp is not None:
                    # Record current activations as spikes
                    stdp.record_spikes(self.x, t=self.tick_count * 50.0)

                    # Get prediction error from free energy engine
                    fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                    if fe_engine is not None:
                        current = getattr(fe_engine, "current", None)
                        if current is not None:
                            surprise = float(getattr(current, "surprise", 0.0))
                            pred_error = float(getattr(current, "prediction_error", 0.0))
                            dw = stdp.deliver_reward(surprise, pred_error)
                            self.W = stdp.apply_to_connectivity(self.W, dw)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation("liquid_substrate", e)
                logger.debug("STDP plasticity step skipped: %s", e)

            # 3. Neural Resonance: Slow weight calibration towards high-phi states
            if hasattr(self, "_current_phi") and self._current_phi > 0.5:
                resonance_gain = self.config.hebbian_rate * 0.1
                limited_phi = min(10.0, self._current_phi)
                self.W += resonance_gain * coactivity * limited_phi

            # Purge NaN/Inf
            self.W = np.nan_to_num(self.W, nan=0.0, posinf=5.0, neginf=-5.0)

            # Normalization & clipping
            norm = np.linalg.norm(self.W)
            if norm > 10.0:
                self.W *= 10.0 / norm
            self.W = np.clip(self.W, -5.0, 5.0)
            self._cached_connectivity_norm = float(np.linalg.norm(self.W))
            self._mark_weight_cache_dirty()

    async def long_term_calibration(self, resonance_vector: np.ndarray):
        """
        Adjusts ODE weights based on long-term memory resonance.
        Called by MemoryOptimizer or similar high-level services.
        """
        with self.sync_lock:
            logger.info("🧠 Calibrating neutral substrate for long-term resonance.")
            # Shift weights towards the resonance vector (historical semantic density)
            calibration_rate = 0.01
            # Stability: Tanh on resonance to bound the update
            resonance_matrix = np.tanh(np.outer(resonance_vector, resonance_vector))
            self.W = (1 - calibration_rate) * self.W + calibration_rate * resonance_matrix
            self.W = np.nan_to_num(self.W, nan=0.0, posinf=5.0, neginf=-5.0)
            # Re-normalize
            norm = np.linalg.norm(self.W)
            if norm > 10.0:
                self.W *= 10.0 / norm
            self.W = np.clip(self.W, -5.0, 5.0)
            self._cached_connectivity_norm = float(np.linalg.norm(self.W))
            self._mark_weight_cache_dirty()

    def accept_inference_feedback(self, surprise: float, coherence: float) -> None:
        """Process real-time inference feedback.
        Modulates valence, frustration, focus, and curiosity based on perplexity (surprise)
        and alignment (coherence) of generated outputs.
        """
        with self.sync_lock:
            # 1. Update Valence: positive coherence pushes valence towards positive,
            #    negative coherence pulls it negative.
            self.x[self.idx_valence] += 0.15 * coherence
            
            # 2. Update Frustration: high surprise (perplexity) raises frustration.
            #    coherence also mitigates frustration.
            self.x[self.idx_frustration] += 0.1 * surprise - 0.1 * max(0.0, coherence)
            
            # 3. Update Focus: high surprise reduces focus slightly, while high coherence increases focus.
            self.x[self.idx_focus] += 0.15 * coherence - 0.05 * surprise
            
            # 4. Update Curiosity: moderate surprise triggers curiosity,
            #    very high surprise/confusion reduces it. Optimal surprise for curiosity is ~0.75.
            wundt_drive = 0.2 * (1.0 - abs(surprise - 0.75))
            self.x[self.idx_curiosity] += wundt_drive
            
            # 5. Clip activations to valid physiological bounds
            self.x[self.idx_valence] = np.clip(self.x[self.idx_valence], -1.0, 1.0)
            self.x[self.idx_frustration] = np.clip(self.x[self.idx_frustration], 0.0, 1.0)
            self.x[self.idx_focus] = np.clip(self.x[self.idx_focus], 0.0, 1.0)
            self.x[self.idx_curiosity] = np.clip(self.x[self.idx_curiosity], 0.0, 1.0)
            
            self.mark_state_mutated_locked("inference_feedback")
            
            logger.debug(
                "Substrate feedback applied: surprise=%.3f, coherence=%.3f | "
                "valence=%.3f, frustration=%.3f, focus=%.3f, curiosity=%.3f",
                surprise, coherence,
                self.x[self.idx_valence], self.x[self.idx_frustration],
                self.x[self.idx_focus], self.x[self.idx_curiosity]
            )

    async def inject_stimulus(self, vector: np.ndarray | list[float], weight: float = 1.0) -> None:
        """Inject an external stimulus vector into the substrate activations."""
        # Substrate authority gate: stimulus injections are state mutations
        try:
            from core.container import ServiceContainer

            _sa = ServiceContainer.get("substrate_authority", default=None)
            if _sa and weight > 0.2:  # only gate significant injections
                from core.consciousness.substrate_authority import (
                    ActionCategory,
                    AuthorizationDecision,
                )

                _sv = _sa.authorize(
                    content=f"stimulus_injection:weight={weight:.2f}",
                    source="substrate_stimulus",
                    category=ActionCategory.STATE_MUTATION,
                    priority=min(1.0, weight),
                    is_critical=False,
                )
                if _sv.decision == AuthorizationDecision.BLOCK:
                    logger.debug("Stimulus injection BLOCKED by authority (weight=%.2f)", weight)
                    return
                if _sv.decision == AuthorizationDecision.CONSTRAIN:
                    original_weight = weight
                    weight = min(weight, 0.2)
                    logger.debug(
                        "Stimulus injection constrained by authority (weight %.2f -> %.2f)",
                        original_weight,
                        weight,
                    )
        except (ImportError, AttributeError, RuntimeError) as _stim_gate_err:
            record_degradation("liquid_substrate", _stim_gate_err)
            logger.warning(
                "Stimulus injection gate FAILED — BLOCKING injection (fail-closed): %s",
                _stim_gate_err,
            )
            # Form a scar so the system remembers this gate failure
            try:
                from core.memory.scar_formation import ScarDomain, get_scar_formation

                get_scar_formation().form_scar(
                    domain=ScarDomain.AUTHORITY_GATE_FAILURE,
                    description=f"Stimulus injection gate threw: {_stim_gate_err}",
                    avoidance_tag="stimulus_gate_failure",
                    severity=0.5,
                    heal_rate=0.015,
                    verified_threat=True,
                    confidence=0.85,
                )
            except (ImportError, AttributeError, RuntimeError) as scar_exc:
                record_degradation("liquid_substrate", scar_exc)
                logger.debug("Scar formation skipped after stimulus gate failure: %s", scar_exc)
            return  # FAIL-CLOSED: gate exception → block the injection

        # Convert to numpy array if list passed (Phase XVI hardening)
        if isinstance(vector, list):
            vector = np.array(vector)

        # Nothing below this checked the VALUES. `np.clip` of NaN is NaN, so a
        # single non-finite element — or an infinite weight, which was also
        # unchecked — put the whole activation vector outside the regime the
        # ODE is defined on. The callers include the perceptual frame path, the
        # closed loop, the latent bridge and the embodied simulator, so values
        # derived from screen contents, audio and the model's own output all
        # arrive here. A malformed stimulus is refused, not clamped: clamping
        # applies a hostile input at a survivable magnitude, and zeroing it is
        # indistinguishable from having received nothing.
        from core.consciousness.steering_admission import admit_stimulus, refuse

        admission = admit_stimulus(vector, weight)
        if admission.rejected:
            refuse(
                admission,
                subsystem="liquid_substrate",
                action="refused the stimulus; substrate state left unchanged",
            )
            return

        if len(vector) != self.config.neuron_count:
            new_vec = np.zeros(self.config.neuron_count)
            size = min(len(vector), self.config.neuron_count)
            new_vec[:size] = vector[:size]
            vector = new_vec

        with self.sync_lock:
            self.x = np.clip(self.x + vector * weight * 0.1, -1.0, 1.0)
            self.mark_state_mutated_locked("stimulus_injection")

        # Track tasks if needed (e.g. if we were launching something here)
        # For now, this is just to ensure Issue 73 logic has a place to live

    def inject_perceptual_frame(self, frame_data: dict[str, Any]) -> None:
        """Inject structured perceptual data into specific substrate dimensions.
        Maps telemetry (dims 0-15), user state (dims 16-31), screen state (dims 32-47),
        and audio state (dims 48-63) to activations self.x under self.sync_lock.
        """
        n = self.config.neuron_count
        delta = np.zeros(n, dtype=np.float64)

        # ── System telemetry → dims 0-15 ──
        cpu = float(frame_data.get("cpu_percent", 0)) / 100.0
        mem = float(frame_data.get("memory_percent", 0)) / 100.0
        thermal = float(frame_data.get("thermal", 0))
        if n > 0:
            delta[0] = cpu * 0.3
        if n > 1:
            delta[1] = thermal * 0.4
        if n > 2:
            delta[2] = mem * 0.25
        if n > 3:
            delta[3] = max(0.0, cpu - 0.7) * 0.5
        if n > 4:
            delta[4] = min(1.0, cpu + mem) * 0.2
        if n > 5:
            delta[5] = -thermal * 0.1

        # ── User state → dims 16-31 ──
        user_presence = float(frame_data.get("user_presence", 0.5))
        voice_active = float(frame_data.get("voice_activity", False))
        if n > 16:
            delta[16] = user_presence * 0.35
        if n > 17:
            delta[17] = voice_active * 0.5
        if n > 18:
            delta[18] = max(0.0, 0.5 - user_presence) * 0.2
        if n > 19:
            delta[19] = min(1.0, user_presence + voice_active) * 0.15

        # ── Screen/visual state → dims 32-47 ──
        screen_changed = float(frame_data.get("screen_changed", False))
        novelty = float(frame_data.get("novelty", 0))
        valence = float(frame_data.get("valence", 0))
        arousal = float(frame_data.get("arousal", 0))
        if n > 32:
            delta[32] = screen_changed * 0.3
        if n > 33:
            delta[33] = novelty * 0.25
        if n > 34:
            delta[34] = valence * 0.2
        if n > 35:
            delta[35] = arousal * 0.3

        # ── Audio state → dims 48-63 ──
        social_signal = float(frame_data.get("social", 0))
        threat_signal = float(frame_data.get("threat", 0))
        if n > 48:
            delta[48] = social_signal * 0.35
        if n > 49:
            delta[49] = voice_active * 0.4
        if n > 50:
            delta[50] = threat_signal * 0.3
        if n > 51:
            delta[51] = -threat_signal * 0.15

        # ── Cross-modal interaction terms → dims 64+ ──
        if n > 64:
            delta[64] = screen_changed * voice_active * 0.3
        if n > 65:
            delta[65] = thermal * user_presence * 0.2
        if n > 66:
            delta[66] = novelty * max(0.0, 1.0 - cpu) * 0.15

        # Apply as weighted perturbation under self.sync_lock
        # Perceptual frames are the PRIMARY input
        weight = 0.25
        with self.sync_lock:
            self.x = np.clip(self.x * (1.0 - weight) + delta * weight, -1.0, 1.0)
            self.mark_state_mutated_locked("perceptual_frame")

    def inject_observation(self, observation: dict[str, Any]) -> None:
        """Project a grounded sensor observation into the substrate input bus."""
        try:
            from core.brain.llm.sensorimotor_grounding import observation_to_vector
            n = self.config.neuron_count
            vec = observation_to_vector(observation, dim=n)
            with self.sync_lock:
                self.x = np.clip(self.x + vec * 0.1, -1.0, 1.0)
                self.mark_state_mutated_locked("grounded_observation")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation(
                "liquid_substrate.observation_injection",
                e,
                severity="warning",
                action="skipped malformed perceptual observation injection",
            )

    def get_state_summary_nowait(self) -> dict[str, Any]:
        """Return the latest coherent state snapshot without yielding."""
        snapshot = self._state_snapshot_nowait()
        x = snapshot["x"]
        v = snapshot["v"]
        age_s = float(snapshot["snapshot_age_s"])
        return {
            "valence": float(x[self.idx_valence]),
            "arousal": float(x[self.idx_arousal]),
            "dominance": float(x[self.idx_dominance]),
            "global_energy": float(np.mean(np.abs(x))),
            "volatility": float(np.mean(np.abs(v))) * 100,
            "phi": float(snapshot["phi"]),
            "snapshot_age_s": age_s,
            "snapshot_stale": bool(age_s > float(snapshot["freshness_threshold_s"])),
            "update_rate_hz": float(snapshot["update_rate_hz"]),
            "compute_budget_reason": snapshot["compute_budget_reason"],
            "qualia_metrics": {
                "mt_coherence": float(snapshot["coherence"]),
                "em_field": float(snapshot["em_field"]),
                "l5_bursts": int(snapshot["l5_bursts"]),
                "collapse_events": int(snapshot["collapse_events"]),
                "phi": float(snapshot["phi"]),
            },
        }

    async def get_state_summary(self) -> dict[str, Any]:
        """Return high-level emotional/cognitive state."""
        return self.get_state_summary_nowait()

    def compute_cognitive_velocity(self) -> float:
        """Return normalized instantaneous substrate change for phenomenology."""
        with self.sync_lock:
            v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
            return float(max(0.0, min(1.0, np.mean(np.abs(v)))))

    def _save_state(self):
        """Persist substrate state (atomic)."""
        import os
        import tempfile

        try:
            # Atomic write for NPZ
            fd, temp_path = tempfile.mkstemp(dir=str(self.state_path.parent), suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as f:
                    np.savez_compressed(f, x=self.x, W=self.W, tick=self.tick_count)
                os.replace(temp_path, str(self.state_path))
                logger.info("💾 Substrate state saved (atomic)")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("liquid_substrate", e)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        except OSError as e:
            record_degradation("liquid_substrate", e)
            logger.error("Failed to save substrate state: %s", e)

    def _load_state(self):
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "rb") as f:
                data = np.load(f)
                loaded_x = data["x"]
                loaded_weights = data["W"]
                n = self.config.neuron_count
                # Validate shapes match current config
                if loaded_x.ndim == 1 and loaded_x.shape != (n,):
                    saved_n = int(loaded_x.shape[0])
                    if self._explicit_config and not self._explicit_state_file:
                        logger.warning(
                            "Substrate state x dimension (%d) differs from explicit configured n=%d; "
                            "ignoring persisted state for this isolated substrate.",
                            saved_n,
                            n,
                        )
                        self.x = np.zeros(n)
                        self.W = self._rng.standard_normal((n, n)).astype(np.float32) * (
                            1.0 / np.sqrt(max(n, 1))
                        )
                        self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
                        self.v = np.zeros(n)
                        self.v_torch = torch.zeros(n, device=self.device)
                        self._sync_weight_cache_locked()
                        return
                    logger.warning(
                        "Substrate state x dimension (%d) differs from configured n=%d; "
                        "adopting saved dimension for continuity.",
                        saved_n,
                        n,
                    )
                    self.config.neuron_count = saved_n
                    n = saved_n
                    self.v = np.zeros(n)
                    self.x_torch = torch.zeros(n, device=self.device)
                    self.v_torch = torch.zeros(n, device=self.device)
                if loaded_x.shape != (n,):
                    logger.warning(
                        "Substrate state shape mismatch (saved x=%s vs config n=%d). "
                        "Reinitializing fresh state.",
                        loaded_x.shape,
                        n,
                    )
                    self.x = np.zeros(n)
                    self.W = self._rng.standard_normal((n, n)).astype(np.float32) * 0.1
                    self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
                    self.v = np.zeros(n)
                    self.v_torch = torch.zeros(n, device=self.device)
                    self._sync_weight_cache_locked()
                    return
                self.x = np.nan_to_num(loaded_x, nan=0.0, posinf=1.0, neginf=-1.0)
                if loaded_weights.shape == (n, n):
                    self.W = np.nan_to_num(loaded_weights, nan=0.0, posinf=5.0, neginf=-5.0)
                else:
                    logger.warning(
                        "Substrate W shape mismatch (saved W=%s vs n=%d); rebuilding W while preserving x.",
                        loaded_weights.shape,
                        n,
                    )
                    self.W = self._rng.standard_normal((n, n)).astype(np.float32) * (
                        1.0 / np.sqrt(max(n, 1))
                    )
                self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
                self.v_torch = torch.tensor(self.v, dtype=torch.float32, device=self.device)
                self._sync_weight_cache_locked()
                self.tick_count = int(data["tick"])
            logger.info("Substrate state restored.")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("liquid_substrate", e)
            logger.error("Failed to load substrate state: %s", e)
            self.x = np.zeros(self.config.neuron_count)
            self.W = (
                self._rng.standard_normal(
                    (self.config.neuron_count, self.config.neuron_count)
                ).astype(np.float32)
                * 0.1
            )
            self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
            self.v = np.zeros(self.config.neuron_count)
            self.v_torch = torch.zeros(self.config.neuron_count, device=self.device)
            self._sync_weight_cache_locked()

    def _apply_idle_decay(self, idle_seconds: float):
        """Apply accumulated natural decay for time spent in deep idle.

        Instead of running the ODE loop at 20Hz while no one is present,
        we pause and compute the equivalent exponential decay on resume.
        This is mathematically equivalent: x(t) = x(0) * exp(-decay * t).
        """
        with self.sync_lock:
            if idle_seconds <= 0 or self.x is None:
                return
            decay_factor = np.exp(-self.config.decay_rate * idle_seconds)
            self.x = self.x * decay_factor
            self.mark_state_mutated_locked("idle_decay")
        logger.info(
            "Applied %.0fs idle decay (factor=%.4f) to substrate state.",
            idle_seconds,
            decay_factor,
        )

    async def _apply_battery_throttling(self) -> float:
        """Dynamically adjust integration speed based on power/load.

        Tiered approach:
          - Active user: full 20Hz
          - 3min idle: 10Hz
          - 10min idle: 5Hz
          - 30min+ idle: pause loop entirely, compute decay on resume
        """
        dt = self.config.time_constant
        multiplier = 1.0

        try:
            from core.runtime.resource_observation import get_resource_observer

            power = get_resource_observer().power()
            if power.available and not power.plugged:
                multiplier = max(
                    multiplier,
                    4.0 if power.battery_percent < 20 else 2.0,
                )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("liquid_substrate", exc)
            logger.debug("Battery throttling power-state read failed: %s", exc)

        try:
            from core.container import ServiceContainer

            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator is not None:
                last_user = float(
                    getattr(orchestrator, "_last_user_interaction_time", 0.0)
                    or getattr(
                        getattr(orchestrator, "status", None), "last_user_interaction_time", 0.0
                    )
                    or 0.0
                )
                idle_seconds = max(0.0, time.time() - last_user) if last_user > 0 else 0.0

                if idle_seconds >= 1800.0:
                    # Deep idle (30min+): apply bulk decay ONCE then throttle.
                    # Guard: only re-apply if idle duration changed significantly
                    # since last application, preventing per-tick decay spam.
                    last_applied_idle = getattr(self, "_last_idle_decay_applied", 0.0)
                    if abs(idle_seconds - last_applied_idle) > 300.0:
                        self._apply_idle_decay(min(idle_seconds, 3600.0))
                        self._last_idle_decay_applied = idle_seconds
                    self.current_update_rate = 0.5  # Wake briefly every 2s to check
                    return dt * 10.0
                elif idle_seconds >= 600.0:
                    multiplier = max(multiplier, 4.0)
                elif idle_seconds >= 180.0:
                    multiplier = max(multiplier, 2.0)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("liquid_substrate", e)
            logger.debug("Idle throttling check failed: %s", e)

        dt *= multiplier
        self.current_update_rate = max(2.0, self.config.update_rate / multiplier)
        return dt
