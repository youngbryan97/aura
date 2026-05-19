"""core/adaptation/plasticity_governor.py -- Elastic Weight Consolidation
=========================================================================
Prevents catastrophic forgetting by computing a diagonal Fisher Information
Matrix over important substrate and world-model parameters, then penalizing
changes to those parameters during online learning.

Algorithm (Online EWC, Schwarz et al. 2018):
  1. After each successful adaptation cycle, estimate diagonal Fisher:
       F_i = E[ (d log p(data|theta) / d theta_i)^2 ]
     Approximated as: F_i = mean over recent gradients of grad_i^2
  2. Maintain a running average of Fisher matrices (online EWC):
       F_running = gamma * F_old + (1 - gamma) * F_new
  3. When parameters are updated, add a quadratic penalty:
       L_ewc = lambda/2 * sum_i F_i * (theta_i - theta*_i)^2
     where theta* are the parameters at the last consolidation point.
  4. The penalty is applied to the parameter update delta, not to the
     loss function directly (since we use numpy, not autograd).

Protected parameter sets:
  - ContinuousSubstrate.W (coupling matrix)
  - LearnedWorldModel weights (W_enc, W_dec, W_prior, GRU weights)
  - DynamicValueGraph node weights

References:
    Kirkpatrick et al. (2017) Overcoming catastrophic forgetting in NNs.
    Schwarz et al. (2018) Progress & Compress: online EWC.
    Zenke et al. (2017) Continual learning through synaptic intelligence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.PlasticityGovernor")

_DATA_DIR = Path.home() / ".aura" / "data" / "plasticity"
_FISHER_PATH = _DATA_DIR / "fisher_state.npz"


@dataclass
class PlasticityConfig:
    """Configuration for EWC plasticity governance."""
    ewc_lambda: float = 100.0          # EWC penalty strength
    fisher_gamma: float = 0.95         # Online Fisher running average decay
    consolidation_interval: int = 50   # Steps between Fisher updates
    min_samples_for_fisher: int = 10   # Minimum gradient samples
    max_fisher_entries: int = 50000    # Cap total Fisher diagonal entries
    gradient_clip: float = 10.0        # Clip gradient norms for stability
    seed: int = 77


@dataclass
class ConsolidationRecord:
    """Record of a parameter consolidation event."""
    timestamp: float
    parameter_set: str
    fisher_norm: float
    n_parameters: int
    mean_importance: float
    max_importance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "parameter_set": self.parameter_set,
            "fisher_norm": round(self.fisher_norm, 6),
            "n_parameters": self.n_parameters,
            "mean_importance": round(self.mean_importance, 6),
            "max_importance": round(self.max_importance, 6),
        }


@dataclass
class PenaltyReport:
    """Report from applying an EWC penalty to a parameter update."""
    parameter_set: str
    original_delta_norm: float
    penalized_delta_norm: float
    penalty_magnitude: float
    n_suppressed: int  # Parameters where penalty > delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_set": self.parameter_set,
            "original_delta_norm": round(self.original_delta_norm, 6),
            "penalized_delta_norm": round(self.penalized_delta_norm, 6),
            "penalty_magnitude": round(self.penalty_magnitude, 6),
            "n_suppressed": self.n_suppressed,
        }


class ParameterSnapshot:
    """Stores a frozen copy of parameters at consolidation time."""

    def __init__(self, name: str, params: np.ndarray) -> None:
        self.name = name
        self.theta_star = params.copy().ravel().astype(np.float64)
        self.fisher_diag = np.zeros_like(self.theta_star)
        self._gradient_accumulator: List[np.ndarray] = []
        self._consolidation_count = 0

    @property
    def n_params(self) -> int:
        return self.theta_star.size

    def accumulate_gradient(self, gradient: np.ndarray) -> None:
        """Accumulate a gradient sample for Fisher estimation."""
        g = np.asarray(gradient, dtype=np.float64).ravel()
        if g.size != self.n_params:
            # Pad or truncate
            padded = np.zeros(self.n_params, dtype=np.float64)
            n = min(g.size, self.n_params)
            padded[:n] = g[:n]
            g = padded
        self._gradient_accumulator.append(g)

    def estimate_fisher(self, gamma: float = 0.95) -> float:
        """Estimate diagonal Fisher from accumulated gradients.

        Uses online EWC: running average of Fisher matrices.
        Returns the norm of the new Fisher estimate.
        """
        if not self._gradient_accumulator:
            return 0.0

        grads = np.array(self._gradient_accumulator, dtype=np.float64)
        new_fisher = np.mean(grads ** 2, axis=0)

        if self._consolidation_count == 0:
            self.fisher_diag = new_fisher
        else:
            # Online EWC: running average
            self.fisher_diag = gamma * self.fisher_diag + (1 - gamma) * new_fisher

        self._gradient_accumulator.clear()
        self._consolidation_count += 1

        return float(np.linalg.norm(self.fisher_diag))

    def compute_penalty(self, current_params: np.ndarray,
                        ewc_lambda: float) -> Tuple[np.ndarray, float]:
        """Compute EWC penalty for current parameters vs consolidated.

        Returns:
            (penalty_vector, total_penalty_magnitude)

        The penalty is: lambda/2 * F_i * (theta_i - theta*_i)^2
        The gradient of this penalty (to subtract from the update) is:
            lambda * F_i * (theta_i - theta*_i)
        """
        current = np.asarray(current_params, dtype=np.float64).ravel()
        if current.size != self.n_params:
            padded = np.zeros(self.n_params, dtype=np.float64)
            n = min(current.size, self.n_params)
            padded[:n] = current[:n]
            current = padded

        diff = current - self.theta_star
        penalty_grad = ewc_lambda * self.fisher_diag * diff
        penalty_magnitude = float(0.5 * ewc_lambda * np.sum(
            self.fisher_diag * diff ** 2
        ))

        return penalty_grad, penalty_magnitude

    def update_anchor(self, new_params: np.ndarray) -> None:
        """Update the consolidation anchor point."""
        self.theta_star = np.asarray(new_params, dtype=np.float64).ravel().copy()


class PlasticityGovernor:
    """Governs parameter plasticity using Elastic Weight Consolidation.

    Usage:
        gov = get_plasticity_governor()

        # Register parameter sets to protect
        gov.register_parameters("substrate_W", substrate.W)
        gov.register_parameters("world_model_enc", world_model.W_enc)

        # During learning, record gradients
        gov.record_gradient("substrate_W", gradient)

        # Periodically consolidate (during dream cycles)
        gov.consolidate()

        # Before applying an update, compute the penalized delta
        report = gov.penalize_update("substrate_W", current_W, proposed_delta)
    """

    def __init__(self, config: Optional[PlasticityConfig] = None) -> None:
        self.config = config or PlasticityConfig()
        self._snapshots: Dict[str, ParameterSnapshot] = {}
        self._step_count = 0
        self._consolidation_history: Deque[ConsolidationRecord] = deque(maxlen=50)
        self._penalty_history: Deque[PenaltyReport] = deque(maxlen=100)

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info("PlasticityGovernor initialized: %d parameter sets",
                     len(self._snapshots))

    def register_parameters(self, name: str, params: np.ndarray) -> None:
        """Register a parameter set for EWC protection."""
        flat = np.asarray(params, dtype=np.float64).ravel()
        if flat.size > self.config.max_fisher_entries:
            logger.warning(
                "Parameter set '%s' has %d entries (max %d) -- truncating",
                name, flat.size, self.config.max_fisher_entries,
            )
            flat = flat[:self.config.max_fisher_entries]
        self._snapshots[name] = ParameterSnapshot(name, flat)
        logger.debug("Registered parameter set '%s': %d parameters", name, flat.size)

    def record_gradient(self, name: str, gradient: np.ndarray) -> None:
        """Record a gradient sample for Fisher estimation."""
        snap = self._snapshots.get(name)
        if snap is None:
            return
        g = np.asarray(gradient, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(g))
        if norm > self.config.gradient_clip:
            g = g * (self.config.gradient_clip / norm)
        snap.accumulate_gradient(g)
        self._step_count += 1

    def consolidate(self, name: Optional[str] = None) -> List[ConsolidationRecord]:
        """Estimate Fisher and consolidate parameter anchors.

        Args:
            name: If given, consolidate only this parameter set.
                  If None, consolidate all.

        Returns:
            List of consolidation records.
        """
        records = []
        targets = [name] if name else list(self._snapshots.keys())

        for pname in targets:
            snap = self._snapshots.get(pname)
            if snap is None:
                continue

            fisher_norm = snap.estimate_fisher(gamma=self.config.fisher_gamma)

            if fisher_norm > 0:
                record = ConsolidationRecord(
                    timestamp=time.time(),
                    parameter_set=pname,
                    fisher_norm=fisher_norm,
                    n_parameters=snap.n_params,
                    mean_importance=float(np.mean(snap.fisher_diag)),
                    max_importance=float(np.max(snap.fisher_diag)),
                )
                records.append(record)
                self._consolidation_history.append(record)
                logger.info(
                    "Consolidated '%s': Fisher norm=%.4f, mean importance=%.6f",
                    pname, fisher_norm, record.mean_importance,
                )

        if records:
            self._save()

        return records

    def penalize_update(self, name: str, current_params: np.ndarray,
                        proposed_delta: np.ndarray) -> Tuple[np.ndarray, PenaltyReport]:
        """Apply EWC penalty to a proposed parameter update.

        Returns:
            (penalized_delta, report)

        The penalized delta is: delta - lambda * F * (theta - theta*)
        This pushes the update away from directions that would damage
        previously learned representations.
        """
        snap = self._snapshots.get(name)
        delta = np.asarray(proposed_delta, dtype=np.float64).ravel()
        original_norm = float(np.linalg.norm(delta))

        if snap is None or np.sum(snap.fisher_diag) < 1e-10:
            report = PenaltyReport(
                parameter_set=name,
                original_delta_norm=original_norm,
                penalized_delta_norm=original_norm,
                penalty_magnitude=0.0,
                n_suppressed=0,
            )
            self._penalty_history.append(report)
            return delta, report

        penalty_grad, penalty_mag = snap.compute_penalty(
            current_params, self.config.ewc_lambda
        )

        # Truncate/pad penalty to match delta size
        if penalty_grad.size != delta.size:
            pg = np.zeros_like(delta)
            n = min(penalty_grad.size, delta.size)
            pg[:n] = penalty_grad[:n]
            penalty_grad = pg

        penalized = delta - penalty_grad
        penalized_norm = float(np.linalg.norm(penalized))

        # Count suppressed parameters (where penalty dominates delta)
        n_suppressed = int(np.sum(np.abs(penalty_grad) > np.abs(delta)))

        report = PenaltyReport(
            parameter_set=name,
            original_delta_norm=original_norm,
            penalized_delta_norm=penalized_norm,
            penalty_magnitude=penalty_mag,
            n_suppressed=n_suppressed,
        )
        self._penalty_history.append(report)

        if penalty_mag > 0.01:
            logger.debug(
                "EWC penalty on '%s': %.4f -> %.4f (penalty=%.4f, suppressed=%d)",
                name, original_norm, penalized_norm, penalty_mag, n_suppressed,
            )

        return penalized, report

    def get_importance_map(self, name: str) -> Optional[np.ndarray]:
        """Get the Fisher diagonal (importance map) for a parameter set."""
        snap = self._snapshots.get(name)
        return snap.fisher_diag.copy() if snap else None

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            save_dict: Dict[str, np.ndarray] = {
                "step_count": np.array([self._step_count]),
            }
            for name, snap in self._snapshots.items():
                save_dict[f"{name}__theta_star"] = snap.theta_star
                save_dict[f"{name}__fisher"] = snap.fisher_diag
                save_dict[f"{name}__count"] = np.array([snap._consolidation_count])
            np.savez_compressed(str(_FISHER_PATH), **save_dict)
            logger.debug("Plasticity state saved")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Plasticity save failed: %s", exc)

    def _load(self) -> None:
        try:
            if not _FISHER_PATH.exists():
                return
            data = np.load(str(_FISHER_PATH), allow_pickle=False)
            self._step_count = int(data.get("step_count", [0])[0])
            for name in list(self._snapshots.keys()):
                ts_key = f"{name}__theta_star"
                f_key = f"{name}__fisher"
                c_key = f"{name}__count"
                snap = self._snapshots[name]
                if ts_key in data and data[ts_key].size == snap.n_params:
                    snap.theta_star = data[ts_key]
                    snap.fisher_diag = data[f_key]
                    snap._consolidation_count = int(data[c_key][0])
            logger.info("Plasticity state restored (step %d)", self._step_count)
        except (OSError, ConnectionError, TimeoutError) as exc:
            logger.debug("Plasticity load failed: %s", exc)

    # ── Public API ──────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        param_status = {}
        for name, snap in self._snapshots.items():
            param_status[name] = {
                "n_parameters": snap.n_params,
                "consolidation_count": snap._consolidation_count,
                "fisher_norm": round(float(np.linalg.norm(snap.fisher_diag)), 4),
                "mean_importance": round(float(np.mean(snap.fisher_diag)), 6),
                "pending_gradients": len(snap._gradient_accumulator),
            }
        return {
            "step_count": self._step_count,
            "parameter_sets": param_status,
            "n_consolidations": len(self._consolidation_history),
            "n_penalties_applied": len(self._penalty_history),
            "ewc_lambda": self.config.ewc_lambda,
        }


_instance: Optional[PlasticityGovernor] = None


def get_plasticity_governor() -> PlasticityGovernor:
    """Get or create the singleton PlasticityGovernor."""
    global _instance
    if _instance is None:
        _instance = PlasticityGovernor()
    return _instance
