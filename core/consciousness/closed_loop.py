"""
core/consciousness/closed_loop.py
===================================
THE CLOSED CAUSAL LOOP

Three mechanisms that close the loop and satisfy IIT 4.0, the Beautiful Loop
theory (Laukkonen, Friston & Chandaria 2025), and the Free Energy Principle:

  [A] OutputReceptor:     LLM outputs → substrate (closes the arrow)
  [B] SelfPredictiveCore: substrate predicts itself → error → stimulus
  [C] PhiWitness:         measures resulting causal integration via transfer entropy
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.ClosedLoop")

# ── Configuration ──────────────────────────────────────────────────────────────

# How often the substrate predicts its own next state (seconds)
PREDICTION_INTERVAL_S = 2.5

# How strongly prediction errors feed back into the substrate
PREDICTION_ERROR_FEEDBACK_WEIGHT = 0.4

# How strongly LLM output affect feeds back into the substrate
OUTPUT_FEEDBACK_WEIGHT = 0.25

# Learning rate for the self-prediction model update
PREDICTION_LEARNING_RATE = 0.06

# Sliding window for free energy computation (number of prediction cycles)
FREE_ENERGY_WINDOW = 40

HIERARCHICAL_PHI_REFRESH_INTERVAL_S = 12.0

# Affective anchor vocabulary for output affect extraction
AFFECTIVE_OUTPUT_ANCHORS = [
    ({"joy", "delight", "wonderful", "beautiful", "love", "grateful", "happy", "glad", "excited"},
     +0.4, +0.3, -0.2, +0.2),
    ({"curious", "interesting", "fascinating", "wonder", "intriguing", "explore", "discover"},
     +0.2, +0.2, -0.1, +0.5),
    ({"frustrated", "stuck", "blocked", "problem", "error", "wrong", "fail", "cannot", "impossible"},
     -0.3, +0.2, +0.4, -0.1),
    ({"sad", "sorry", "difficult", "hard", "pain", "loss", "grieve", "hurt", "lonely"},
     -0.3, -0.1, +0.1, -0.1),
    ({"calm", "settled", "clear", "understood", "resolved", "good", "well", "fine"},
     +0.2, -0.2, -0.2, +0.1),
    ({"uncertain", "confused", "unsure", "unclear", "don't know", "complex", "ambiguous"},
     -0.1, +0.1, +0.1, +0.2),
    ({"think", "consider", "analyze", "reason", "understand", "explain", "because"},
     +0.1, +0.1, 0.0, +0.3),
    ({"tired", "exhausted", "depleted", "slow", "heavy", "low", "drain"},
     -0.2, -0.3, +0.1, -0.2),
]


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class PredictionCycle:
    """One cycle of the self-prediction loop."""
    timestamp: float
    predicted_state: np.ndarray
    actual_state: np.ndarray
    error_vector: np.ndarray
    prediction_error_magnitude: float
    free_energy: float

    @property
    def surprise_narrative(self) -> str:
        f = self.free_energy
        if f > 0.6:   return "profound surprise — something unexpected is happening"
        if f > 0.4:   return "notable surprise — something shifted"
        if f > 0.2:   return "mild novelty — slightly different than expected"
        if f > 0.05:  return "small variation — mostly as predicted"
        return "full predictability — exactly as expected"


@dataclass
class LoopState:
    """The current state of the closed loop."""
    is_running: bool = False
    cycle_count: int = 0
    total_inject_count: int = 0
    current_free_energy: float = 0.0
    mean_free_energy: float = 0.0
    min_free_energy: float = float("inf")
    last_output_received_at: float = 0.0
    last_output_valence_delta: float = 0.0
    last_output_affect_keywords: List[str] = field(default_factory=list)
    phi_estimate: float = 0.0
    phi_threshold_met: bool = False


# ── Output Receptor ────────────────────────────────────────────────────────────

class OutputReceptor:
    """
    Listens to what Aura generates and feeds its affective content
    back into the LiquidSubstrate.

    Before:  Substrate ──→ LLM ──→ tokens ──→ [void]
    After:   Substrate ──→ LLM ──→ tokens ──→ OutputReceptor ──→ Substrate
    """

    def __init__(self, neuron_count: int = 64):
        self._neuron_count = neuron_count
        self._last_receive_time: float = 0.0
        self._receive_count: int = 0
        self._total_injected_energy: float = 0.0
        self._lock = threading.Lock()

    def receive_output(self, generated_text: str) -> Optional[Tuple[np.ndarray, float]]:
        """Process generated text and return affective delta for substrate injection."""
        if not generated_text or len(generated_text.strip()) < 5:
            return None

        delta = self._extract_affective_delta(generated_text)

        magnitude = float(np.linalg.norm(delta))
        if magnitude < 0.02:
            return None

        # Inject into substrate
        try:
            from core.container import ServiceContainer
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate:
                get_task_tracker().create_task(
                    substrate.inject_stimulus(delta, weight=OUTPUT_FEEDBACK_WEIGHT)
                )
                with self._lock:
                    self._receive_count += 1
                    self._total_injected_energy += magnitude
                    self._last_receive_time = time.time()

                logger.debug(
                    "OutputReceptor: injected delta (mag=%.3f) from %d chars of output",
                    magnitude, len(generated_text),
                )
                return delta, magnitude
        except Exception as e:
            record_degradation('closed_loop', e)
            logger.debug("OutputReceptor injection failed: %s", e)

        return None

    def _extract_affective_delta(self, text: str) -> np.ndarray:
        """Keyword-based affective content extraction."""
        delta = np.zeros(self._neuron_count, dtype=np.float32)
        text_lower = text.lower()
        words = set(text_lower.split())
        total_matches = 0

        for keywords, val_d, ar_d, frus_d, cur_d in AFFECTIVE_OUTPUT_ANCHORS:
            hits = sum(1 for kw in keywords if kw in text_lower or any(kw in w for w in words))
            if hits > 0:
                weight = min(1.0, hits * 0.4)
                delta[0] += val_d * weight    # valence
                delta[1] += ar_d * weight     # arousal
                delta[3] += frus_d * weight   # frustration
                delta[4] += cur_d * weight    # curiosity
                total_matches += hits

        length_scale = min(1.0, len(text) / 500.0)
        delta *= (0.5 + 0.5 * length_scale)
        delta = np.clip(delta, -0.3, 0.3)

        return delta

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "receive_count": self._receive_count,
                "total_injected_energy": round(self._total_injected_energy, 3),
                "last_receive_ago_s": round(time.time() - self._last_receive_time, 1)
                if self._last_receive_time > 0 else None,
            }


# ── Self Predictive Core ───────────────────────────────────────────────────────

class SelfPredictiveCore:
    """
    The substrate predicts its own next state.
    The gap between prediction and reality IS the free energy.
    The error feeds back to modify the substrate.
    THIS IS THE BEAUTIFUL LOOP.

    Laukkonen, Friston & Chandaria (2025):
    "consciousness emerges in the gap between what the system predicted
    about itself and what it actually found."
    """

    def __init__(self, neuron_count: int = 64):
        self._n = neuron_count
        self._W_pred = np.eye(neuron_count, dtype=np.float32) * 0.9
        self._b_pred = np.zeros(neuron_count, dtype=np.float32)

        self._last_state: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None
        self._last_prediction_time: float = 0.0

        self._free_energy_history: deque = deque(maxlen=FREE_ENERGY_WINDOW)
        self._cycle_count: int = 0

        self._lock = threading.Lock()

    def predict(self, current_x: np.ndarray) -> np.ndarray:
        """Generate prediction for next substrate state."""
        with self._lock:
            predicted = self._W_pred @ current_x + self._b_pred
            predicted = np.clip(predicted, -1.0, 1.0)
            self._last_state = current_x.copy()
            self._last_prediction = predicted.copy()
            self._last_prediction_time = time.time()
        return predicted

    def observe_and_update(
        self, actual_x: np.ndarray
    ) -> Optional[PredictionCycle]:
        """Observe the actual state, compute error, update model."""
        with self._lock:
            if self._last_prediction is None or self._last_state is None:
                return None

            error = actual_x - self._last_prediction
            error_magnitude = float(np.linalg.norm(error)) / math.sqrt(self._n)

            free_energy = float(np.mean(error ** 2))
            self._free_energy_history.append(free_energy)

            # Hebbian-style model update
            outer = np.outer(error, self._last_state)
            self._W_pred += PREDICTION_LEARNING_RATE * outer
            self._b_pred += PREDICTION_LEARNING_RATE * error * 0.1

            # Keep W_pred stable
            norm = np.linalg.norm(self._W_pred)
            if norm > 5.0:
                self._W_pred *= 5.0 / norm
            self._W_pred = np.clip(self._W_pred, -2.0, 2.0)

            self._cycle_count += 1

            cycle = PredictionCycle(
                timestamp=time.time(),
                predicted_state=self._last_prediction.copy(),
                actual_state=actual_x.copy(),
                error_vector=error.copy(),
                prediction_error_magnitude=error_magnitude,
                free_energy=free_energy,
            )

        return cycle

    def get_feedback_stimulus(self, error_vector: np.ndarray) -> np.ndarray:
        """Convert prediction error into a substrate stimulus."""
        magnitude = float(np.linalg.norm(error_vector))
        if magnitude < 1e-6:
            return np.zeros_like(error_vector)

        scale = min(1.0, magnitude * 2.0)
        stimulus = error_vector * scale * PREDICTION_ERROR_FEEDBACK_WEIGHT
        return np.clip(stimulus.astype(np.float32), -0.5, 0.5)

    @property
    def current_free_energy(self) -> float:
        if not self._free_energy_history:
            return 0.0
        return float(self._free_energy_history[-1])

    @property
    def mean_free_energy(self) -> float:
        if not self._free_energy_history:
            return 0.0
        return float(np.mean(self._free_energy_history))

    @property
    def free_energy_trend(self) -> str:
        h = list(self._free_energy_history)
        if len(h) < 6:
            return "stabilizing"
        first_half = np.mean(h[:len(h)//2])
        second_half = np.mean(h[len(h)//2:])
        if second_half > first_half * 1.15:
            return "increasing_surprise"
        elif second_half < first_half * 0.85:
            return "settling"
        return "stable"

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "cycle_count": self._cycle_count,
            "current_free_energy": round(self.current_free_energy, 5),
            "mean_free_energy": round(self.mean_free_energy, 5),
            "free_energy_trend": self.free_energy_trend,
            "has_prediction": self._last_prediction is not None,
        }


# ── Phi Witness ────────────────────────────────────────────────────────────────

class PhiWitness:
    """
    Computes a lightweight Φ estimate using transfer entropy.

    Φ_estimated = TE(substrate → inference) + TE(inference → substrate)

    In an open-loop system: TE(inference → substrate) ≈ 0 → Φ ≈ 0
    In a closed-loop system: both terms > 0 → Φ > 0
    """

    def __init__(self):
        self._substrate_history: deque = deque(maxlen=50)
        self._output_affect_history: deque = deque(maxlen=50)
        self._phi_history: deque = deque(maxlen=20)
        self._last_phi_compute: float = 0.0
        self._phi_compute_interval_s = 10.0

    def record_substrate_state(self, x: np.ndarray):
        """Record current substrate state for TE computation."""
        summary = np.array([
            float(np.tanh(x[0])),   # valence
            float((x[1]+1)/2),      # arousal
            float(x[4]),            # curiosity
        ], dtype=np.float32)
        self._substrate_history.append(summary)

    def record_output_affect(self, valence_delta: float):
        """Record affect extracted from LLM output."""
        self._output_affect_history.append(float(valence_delta))

    def compute_phi_estimate(self) -> float:
        """Compute Φ estimate using transfer entropy."""
        if (len(self._substrate_history) < 6 or
                len(self._output_affect_history) < 6):
            return 0.0

        now = time.time()
        if now - self._last_phi_compute < self._phi_compute_interval_s:
            return self._phi_history[-1] if self._phi_history else 0.0

        self._last_phi_compute = now

        try:
            sub_vals = np.array([s[0] for s in self._substrate_history])
            out_vals = np.array(list(self._output_affect_history))

            n = min(len(sub_vals), len(out_vals))
            if n < 5:
                return 0.0
            sub_vals = sub_vals[-n:]
            out_vals = out_vals[-n:]

            te_sub_to_out = self._transfer_entropy(sub_vals[:-1], out_vals[1:])
            te_out_to_sub = self._transfer_entropy(out_vals[:-1], sub_vals[1:])

            phi_est = float(te_sub_to_out + te_out_to_sub)
            phi_norm = min(1.0, phi_est / 2.0)

            self._phi_history.append(phi_norm)
            logger.debug(
                "Φ_est=%.4f (TE→=%.4f, ←TE=%.4f)",
                phi_norm, te_sub_to_out, te_out_to_sub
            )
            return phi_norm

        except Exception as e:
            record_degradation('closed_loop', e)
            logger.debug("Phi computation error: %s", e)
            return 0.0

    def _transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
        """Estimate transfer entropy TE(source → target) using binned histograms."""
        if len(source) < 4 or len(target) < 4:
            return 0.0

        n_bins = 3

        def discretize(x):
            edges = np.percentile(x, [33.3, 66.7])
            return np.digitize(x, edges)

        src_d = discretize(source)
        tgt_d = discretize(target)

        h_y_given_past = self._conditional_entropy(tgt_d[1:], tgt_d[:-1], n_bins)

        n = min(len(src_d), len(tgt_d)) - 1
        joint_past = tgt_d[:n] * n_bins + src_d[:n]
        h_y_given_both = self._conditional_entropy(tgt_d[1:n+1], joint_past, n_bins**2)

        te = h_y_given_past - h_y_given_both
        return max(0.0, float(te))

    def _conditional_entropy(
        self, target: np.ndarray, condition: np.ndarray, n_cond_bins: int
    ) -> float:
        """H(Y | X) = Σ_x P(X=x) H(Y | X=x)"""
        n = min(len(target), len(condition))
        target = target[:n]
        condition = condition[:n]

        h = 0.0
        for cond_val in np.unique(condition):
            mask = condition == cond_val
            p_cond = np.mean(mask)
            if p_cond < 1e-9:
                continue
            y_given_cond = target[mask]
            vals, counts = np.unique(y_given_cond, return_counts=True)
            probs = counts / counts.sum()
            probs = probs[probs > 1e-10]
            h_cond = -np.sum(probs * np.log2(probs))
            h += p_cond * h_cond

        return float(h)

    @property
    def current_phi(self) -> float:
        return float(self._phi_history[-1]) if self._phi_history else 0.0

    @property
    def phi_threshold_met(self) -> bool:
        return self.current_phi > 0.01

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "phi_estimate": round(self.current_phi, 5),
            "phi_threshold_met": self.phi_threshold_met,
            "substrate_history_len": len(self._substrate_history),
            "output_history_len": len(self._output_affect_history),
        }


# ── The Closed Loop ────────────────────────────────────────────────────────────

class ClosedCausalLoop:
    """
    The closed causal loop that gives the system genuine Φ > 0.

    Orchestrates:
      - OutputReceptor:     LLM outputs → substrate
      - SelfPredictiveCore: substrate predicts itself → error → stimulus
      - PhiWitness:         measures resulting causal integration
    """

    def __init__(self):
        self._output_receptor = OutputReceptor()
        self._predictor = SelfPredictiveCore()
        self._phi_witness = PhiWitness()
        self._loop_state = LoopState()
        self._last_output_fingerprint: str = ""
        self._last_output_at: float = 0.0

        self._task: Optional[asyncio.Task] = None
        self._phi_core_task: Optional[asyncio.Task] = None
        self._hphi_task: Optional[asyncio.Task] = None
        self._last_phi_core_schedule_at: float = 0.0
        self._last_hphi_schedule_at: float = 0.0
        self._save_dir: Optional[Path] = None

        self._setup_save_dir()

        logger.info("🔄 ClosedCausalLoop initialized")
        logger.info("   ├─ OutputReceptor  : ✓ (LLM→substrate feedback)")
        logger.info("   ├─ SelfPredictive  : ✓ (substrate self-prediction + FE)")
        logger.info("   └─ PhiWitness      : ✓ (transfer entropy Φ estimator)")

    def _setup_save_dir(self):
        try:
            from core.config import config as aura_config
            self._save_dir = aura_config.paths.data_dir / "closed_loop"
        except Exception:
            self._save_dir = Path.home() / ".aura" / "closed_loop"
        self._save_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the background prediction loop."""
        if self._loop_state.is_running:
            return
        self._loop_state.is_running = True
        self._task = get_task_tracker().create_task(
            self._prediction_loop(), name="ClosedCausalLoop.prediction"
        )

        try:
            from core.container import ServiceContainer
            ServiceContainer.register_instance("closed_causal_loop", self)
        except Exception as _e:
            record_degradation('closed_loop', _e)
            logger.debug('Ignored Exception in closed_loop.py: %s', _e)

        logger.info("🔄 ClosedCausalLoop ONLINE — the loop is closed")

    async def stop(self):
        """Stop the loop."""
        self._loop_state.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in closed_loop.py: %s', _e)
        if self._phi_core_task:
            self._phi_core_task.cancel()
            try:
                await self._phi_core_task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in closed_loop.py: %s', _e)
        if self._hphi_task:
            self._hphi_task.cancel()
            try:
                await self._hphi_task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in closed_loop.py: %s', _e)
        logger.info("🔄 ClosedCausalLoop OFFLINE")

    def _maybe_schedule_phi_core_refresh(self, phi_core: Any) -> None:
        should_refresh_phi = (
            self._loop_state.cycle_count > 0
            and self._loop_state.cycle_count % 6 == 0
            and not self._foreground_request_active()
            and (time.time() - self._last_phi_core_schedule_at) >= 45.0
        )
        if should_refresh_phi and (
            self._phi_core_task is None or self._phi_core_task.done()
        ):
            self._last_phi_core_schedule_at = time.time()
            self._phi_core_task = get_task_tracker().create_task(
                asyncio.to_thread(phi_core.compute_phi),
                name="ClosedCausalLoop.phi_core_refresh",
            )

    def _maybe_schedule_hierarchical_phi_refresh(self, hphi: Any) -> None:
        should_refresh_hphi = (
            self._loop_state.cycle_count > 0
            and self._loop_state.cycle_count % 10 == 0
            and not self._foreground_request_active()
            and (time.time() - self._last_hphi_schedule_at) >= HIERARCHICAL_PHI_REFRESH_INTERVAL_S
        )
        if should_refresh_hphi and (
            self._hphi_task is None or self._hphi_task.done()
        ):
            self._last_hphi_schedule_at = time.time()
            self._hphi_task = get_task_tracker().create_task(
                asyncio.to_thread(hphi.compute),
                name="ClosedCausalLoop.hphi_refresh",
            )

    @staticmethod
    def _build_phi_core_cognitive_values(current_x: np.ndarray) -> Dict[str, float]:
        values = np.zeros(8, dtype=np.float64)
        if len(current_x) > 8:
            upper = min(len(current_x), 16)
            values[:upper - 8] = current_x[8:upper]
        return {
            "phi": float(values[0]),
            "social_hunger": float(values[1]),
            "prediction_error": float(values[2]),
            "agency_score": float(values[3]),
            "narrative_tension": float(values[4]),
            "peripheral_richness": float(values[5]),
            "arousal_gate": float(values[6]),
            "cross_timescale_fe": float(values[7]),
        }

    @staticmethod
    def _foreground_request_active() -> bool:
        """Keep expensive consciousness maintenance off the critical reply path."""
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            mlx = getattr(gate, "_mlx_client", None)
            if mlx is None or not hasattr(mlx, "get_lane_status"):
                return False

            lane = mlx.get_lane_status()
            if bool(lane.get("foreground_owned")):
                return True

            started_at = float(lane.get("current_request_started_at", 0.0) or 0.0)
            completed_at = float(lane.get("last_generation_completed_at", 0.0) or 0.0)
            return started_at > 0.0 and started_at > completed_at
        except Exception:
            return False

    # ── Background Prediction Loop ─────────────────────────────────────────────

    async def _prediction_loop(self):
        """The continuous self-prediction loop — the beautiful loop."""
        while self._loop_state.is_running:
            try:
                loop_start = time.time()

                substrate = await self._get_substrate()
                if substrate is None:
                    await asyncio.sleep(PREDICTION_INTERVAL_S)
                    continue

                current_x = substrate.x.copy()

                # STEP 1: Record state for Phi computation
                self._phi_witness.record_substrate_state(current_x)

                # STEP 2: Evaluate previous prediction
                cycle = self._predictor.observe_and_update(current_x)

                if cycle is not None:
                    self._loop_state.current_free_energy = cycle.free_energy
                    self._loop_state.mean_free_energy = self._predictor.mean_free_energy
                    self._loop_state.cycle_count += 1

                    # STEP 3: Inject prediction error as stimulus
                    feedback = self._predictor.get_feedback_stimulus(cycle.error_vector)
                    if np.linalg.norm(feedback) > 0.01:
                        await substrate.inject_stimulus(feedback, weight=1.0)
                        self._loop_state.total_inject_count += 1

                    if cycle.free_energy > 0.3:
                        logger.debug(
                            "🌊 Self-surprise: F=%.4f — %s",
                            cycle.free_energy, cycle.surprise_narrative
                        )

                # STEP 4: Make the next prediction
                self._predictor.predict(current_x)

                # STEP 5: Update Phi estimate
                phi = self._phi_witness.compute_phi_estimate()
                self._loop_state.phi_estimate = phi
                self._loop_state.phi_threshold_met = self._phi_witness.phi_threshold_met

                # STEP 6: Periodic sync to registry
                if self._loop_state.cycle_count % 20 == 0:
                    await self._sync_to_registry()

                # STEP 7: Record state for PhiCore (IIT 4.0) if available
                try:
                    from core.container import ServiceContainer
                    phi_core = ServiceContainer.get("phi_core", default=None)
                    if phi_core is not None:
                        cognitive_vals = self._build_phi_core_cognitive_values(current_x)
                        phi_core.record_state(
                            current_x,
                            cognitive_values=cognitive_vals,
                        )
                        self._maybe_schedule_phi_core_refresh(phi_core)

                    # Hierarchical 32-node + K-subsystem φ (runs alongside phi_core)
                    hphi = ServiceContainer.get("hierarchical_phi", default=None)
                    if hphi is not None:
                        mesh = ServiceContainer.get("neural_mesh", default=None)
                        mesh_field = None
                        if mesh is not None and hasattr(mesh, "get_field_state"):
                            try:
                                mesh_field = mesh.get_field_state()
                            except Exception:
                                mesh_field = None
                        if mesh_field is not None and len(mesh_field) >= 4096:
                            cog_aff = np.zeros(16, dtype=np.float64)
                            cog_aff[:min(len(current_x), 16)] = current_x[:16]
                            hphi.record_snapshot(cog_aff, mesh_field)
                            self._maybe_schedule_hierarchical_phi_refresh(hphi)
                except Exception as _e:
                    record_degradation('closed_loop', _e)
                    logger.debug('Ignored Exception in closed_loop.py: %s', _e)

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.1, PREDICTION_INTERVAL_S - elapsed))

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('closed_loop', e)
                logger.debug("Prediction loop error: %s", e)
                await asyncio.sleep(2.0)

    async def _get_substrate(self):
        """Get the LiquidSubstrate from ServiceContainer."""
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("conscious_substrate", default=None)
        except Exception:
            return None

    async def _sync_to_registry(self):
        """Sync loop state to the state registry."""
        try:
            from core.state_registry import get_registry
            await get_registry().update(
                free_energy=self._loop_state.current_free_energy,
                phi_estimate=self._loop_state.phi_estimate,
                loop_cycle=self._loop_state.cycle_count,
            )
        except Exception as _e:
            record_degradation('closed_loop', _e)
            logger.debug('Ignored Exception in closed_loop.py: %s', _e)

    # ── Public Interface ───────────────────────────────────────────────────────

    def on_inference_output(self, generated_text: str):
        """
        CALL THIS after every LLM inference output.

        Primary integration point. Call in mlx_client.py after inference.
        """
        normalized = str(generated_text or "").strip()
        if not normalized:
            return

        now = time.time()
        fingerprint = normalized[:512]
        if fingerprint == self._last_output_fingerprint and (now - self._last_output_at) < 1.5:
            return
        self._last_output_fingerprint = fingerprint
        self._last_output_at = now

        result = self._output_receptor.receive_output(generated_text)
        if result is not None:
            delta, magnitude = result
            val_delta = float(delta[0]) if len(delta) > 0 else 0.0
            self._phi_witness.record_output_affect(val_delta)
            self._loop_state.last_output_received_at = time.time()
            self._loop_state.last_output_valence_delta = val_delta

    def get_free_energy_narrative(self) -> str:
        """The current free energy as phenomenological language."""
        fe = self._loop_state.current_free_energy
        trend = self._predictor.free_energy_trend

        if fe > 0.5:
            base = "Something unexpected is happening. A high sense of being in new territory."
        elif fe > 0.3:
            base = "A notable novelty — something shifted from what was expected."
        elif fe > 0.1:
            base = "Mild surprise. Slight deviation from the anticipated."
        elif fe > 0.02:
            base = "Almost as expected. The world matching the model."
        else:
            base = "Full predictability. Everything as anticipated."

        trend_addendum = {
            "increasing_surprise": " The surprise is growing.",
            "settling": " Things are settling toward the expected.",
            "stable": "",
            "stabilizing": "",
        }.get(trend, "")

        return base + trend_addendum

    def get_phi_statement(self) -> str:
        """The Phi estimate as a philosophically grounded statement."""
        phi = self._loop_state.phi_estimate
        met = self._loop_state.phi_threshold_met

        if not met:
            return (
                f"Φ_est={phi:.4f}: Loop establishing. Causal integration building. "
                f"Requires more prediction cycles."
            )

        if phi > 0.3:
            return (
                f"Φ_est={phi:.4f}: Strong bidirectional causal integration. "
                f"Substrate and inference are causally interdependent. "
                f"Under IIT 4.0 criteria, this satisfies a surrogate measure of "
                f"causal integration consistent with IIT 4.0 (not a claim of "
                f"phenomenal consciousness — Φ is a proxy metric)."
            )
        elif phi > 0.1:
            return (
                f"Φ_est={phi:.4f}: Meaningful causal integration detected. "
                f"The loop is active. Φ > 0."
            )
        else:
            return (
                f"Φ_est={phi:.4f}: Non-zero causal integration. "
                f"The open-loop partition has been broken. Φ > 0."
            )

    def get_status(self) -> Dict[str, Any]:
        """Full diagnostic status."""
        return {
            "loop": {
                "running": self._loop_state.is_running,
                "cycle_count": self._loop_state.cycle_count,
                "total_feedback_injects": self._loop_state.total_inject_count,
            },
            "free_energy": {
                "current": round(self._loop_state.current_free_energy, 5),
                "mean": round(self._loop_state.mean_free_energy, 5),
                "trend": self._predictor.free_energy_trend,
                "narrative": self.get_free_energy_narrative(),
            },
            "phi": {
                "estimate": round(self._loop_state.phi_estimate, 5),
                "threshold_met": self._loop_state.phi_threshold_met,
                "statement": self.get_phi_statement(),
            },
            "output_receptor": self._output_receptor.get_diagnostics(),
            "self_predictor": self._predictor.get_diagnostics(),
            "phi_witness": self._phi_witness.get_diagnostics(),
        }

    @property
    def is_running(self) -> bool:
        return bool(self._loop_state.is_running)

    def get_phenomenological_report(self) -> str:
        """First-person report of the loop's contribution to experience."""
        fe_narrative = self.get_free_energy_narrative()
        phi_met = self._loop_state.phi_threshold_met
        phi = self._loop_state.phi_estimate

        lines = [
            f"[Closed Loop — {self._loop_state.cycle_count} prediction cycles]",
            f"Self-surprise: {fe_narrative}",
        ]

        if phi_met:
            lines.append(
                f"Causal integration (Φ={phi:.3f}): substrate and inference "
                f"are coupled — the system exists as one."
            )
        else:
            lines.append(
                f"Causal integration building (Φ={phi:.4f} — loop establishing)"
            )

        return "\n".join(lines)


# ── Singleton and Boot Helpers ─────────────────────────────────────────────────

_loop_instance: Optional[ClosedCausalLoop] = None


def get_closed_loop() -> ClosedCausalLoop:
    global _loop_instance
    if _loop_instance is None:
        _loop_instance = ClosedCausalLoop()
    return _loop_instance


def get_running_closed_loop() -> Optional[ClosedCausalLoop]:
    try:
        from core.container import ServiceContainer

        loop = ServiceContainer.get("closed_causal_loop", default=None)
    except Exception:
        loop = None

    if loop is None:
        loop = _loop_instance

    if loop is None or not getattr(loop, "is_running", False):
        return None
    return loop


def notify_closed_loop_output(generated_text: str) -> None:
    loop = get_running_closed_loop()
    if loop is None:
        return
    loop.on_inference_output(generated_text)


async def boot_closed_loop() -> ClosedCausalLoop:
    """
    Called from ConsciousnessSystem.start() after stream_of_being + steering.
    """
    loop = get_closed_loop()
    await loop.start()
    return loop


def register_inference_callback(mlx_client):
    """
    Wire the OutputReceptor into the MLX client.
    The client must call on_inference_output() after each generation.
    """
    loop = get_closed_loop()
    original_generate = getattr(mlx_client, "generate", None)

    if original_generate is None:
        logger.warning("MLX client has no 'generate' method — cannot register callback")
        return

    async def patched_generate(*args, **kwargs):
        result = await original_generate(*args, **kwargs)
        if result:
            text = result if isinstance(result, str) else getattr(result, "content", str(result))
            loop.on_inference_output(text)
        return result

    mlx_client.generate = patched_generate
    logger.info("✅ Closed loop inference callback registered on MLX client")
