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

import io
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.PlasticityGovernor")

_DATA_DIR = state_root() / "data" / "plasticity"
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
    #: Ceiling on the EWC penalty as a fraction of the proposed update's norm.
    #: Without this the raw penalty is subtracted directly and can exceed the
    #: update, REVERSING its direction — turning a brake into an accelerator
    #: pointing backwards. 1.0 means the penalty may cancel an update exactly
    #: but never invert it.
    max_penalty_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Reject configuration that would silently disable or invert the governor."""
        def _finite(value: Any, default: float, *, lo: float, hi: float) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(v):
                return default
            return max(lo, min(hi, v))

        self.ewc_lambda = _finite(self.ewc_lambda, 100.0, lo=0.0, hi=1e6)
        # gamma is a running-average decay: outside [0,1) it either ignores
        # history entirely or diverges.
        self.fisher_gamma = _finite(self.fisher_gamma, 0.95, lo=0.0, hi=0.999999)
        self.gradient_clip = _finite(self.gradient_clip, 10.0, lo=1e-6, hi=1e9)
        self.max_penalty_ratio = _finite(self.max_penalty_ratio, 1.0, lo=0.0, hi=10.0)
        try:
            self.consolidation_interval = max(1, int(self.consolidation_interval))
        except (TypeError, ValueError):
            self.consolidation_interval = 50
        try:
            self.min_samples_for_fisher = max(1, int(self.min_samples_for_fisher))
        except (TypeError, ValueError):
            self.min_samples_for_fisher = 10
        try:
            self.max_fisher_entries = max(1, int(self.max_fisher_entries))
        except (TypeError, ValueError):
            self.max_fisher_entries = 50000


@dataclass
class ConsolidationRecord:
    """Record of a parameter consolidation event."""
    timestamp: float
    parameter_set: str
    fisher_norm: float
    n_parameters: int
    mean_importance: float
    max_importance: float

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
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
        self._gradient_accumulator: list[np.ndarray] = []
        self._consolidation_count = 0

    @property
    def n_params(self) -> int:
        return self.theta_star.size

    def accumulate_gradient(self, gradient: np.ndarray) -> None:
        """Accumulate a gradient sample for Fisher estimation."""
        g = np.asarray(gradient, dtype=np.float64).ravel()
        if g.size != self.n_params:
            # Padding or truncating here HIDES index-alignment corruption: the
            # Fisher entry for parameter i would be estimated from the gradient
            # of some other parameter, silently protecting the wrong weights.
            # An incompatible parameter generation is rejected instead.
            raise ValueError(
                f"gradient size {g.size} does not match parameter set "
                f"'{self.name}' ({self.n_params}); refusing to misalign Fisher"
            )
        if not np.all(np.isfinite(g)):
            raise ValueError(
                f"non-finite gradient for '{self.name}'; refusing to poison Fisher"
            )
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
                        ewc_lambda: float) -> tuple[np.ndarray, float]:
        """Compute EWC penalty for current parameters vs consolidated.

        Returns:
            (penalty_vector, total_penalty_magnitude)

        The penalty is: lambda/2 * F_i * (theta_i - theta*_i)^2
        The gradient of this penalty (to subtract from the update) is:
            lambda * F_i * (theta_i - theta*_i)
        """
        current = np.asarray(current_params, dtype=np.float64).ravel()
        if current.size != self.n_params:
            raise ValueError(
                f"parameter size {current.size} does not match snapshot "
                f"'{self.name}' ({self.n_params}); refusing to misalign penalty"
            )

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

    def __init__(self, config: PlasticityConfig | None = None) -> None:
        self.config = config or PlasticityConfig()
        self._snapshots: dict[str, ParameterSnapshot] = {}
        self._step_count = 0
        self._consolidation_history: deque[ConsolidationRecord] = deque(maxlen=50)
        self._penalty_history: deque[PenaltyReport] = deque(maxlen=100)
        # Guards snapshots, accumulators, histories and persistence together.
        # Concurrent learners previously lost gradients and could persist a
        # half-consolidated state.
        self._lock = checked_lock("plasticity_governor", reentrant=True)
        # Persisted payload held until the matching parameter set registers.
        # _load() used to iterate self._snapshots, which is EMPTY here, so
        # every restored Fisher was discarded and register_parameters then
        # built a fresh zero-Fisher snapshot — the governor protected nothing
        # across a restart.
        self._persisted: dict[str, dict[str, Any]] = {}

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info("PlasticityGovernor initialized: %d parameter sets",
                     len(self._snapshots))

    def register_parameters(self, name: str, params: np.ndarray) -> None:
        """Register a parameter set for EWC protection.

        Restores any persisted Fisher state for this name, so protection
        learned before a restart still applies.
        """
        flat = np.asarray(params, dtype=np.float64).ravel()
        if not np.all(np.isfinite(flat)):
            raise ValueError(f"parameter set '{name}' contains non-finite values")
        if flat.size > self.config.max_fisher_entries:
            # Truncation leaves the REST of the tensor entirely ungoverned and
            # biases protection by memory layout — the first entries are
            # protected simply for being first. Loud, because a silently
            # half-governed parameter set looks identical to a governed one.
            logger.warning(
                "Parameter set '%s' has %d entries (max %d): only the first %d "
                "are Fisher-protected; the remainder is UNGOVERNED and may be "
                "overwritten freely.",
                name, flat.size, self.config.max_fisher_entries,
                self.config.max_fisher_entries,
            )
            flat = flat[:self.config.max_fisher_entries]

        with self._lock:
            snap = ParameterSnapshot(name, flat)
            saved = self._persisted.get(name)
            if saved is not None:
                theta = saved.get("theta_star")
                fisher = saved.get("fisher")
                if (isinstance(theta, np.ndarray) and theta.size == snap.n_params
                        and isinstance(fisher, np.ndarray)
                        and fisher.size == snap.n_params
                        and np.all(np.isfinite(theta)) and np.all(np.isfinite(fisher))):
                    snap.theta_star = theta.astype(np.float64)
                    snap.fisher_diag = fisher.astype(np.float64)
                    snap._consolidation_count = int(saved.get("count", 0))
                    logger.info(
                        "Restored persisted Fisher for '%s' (%d consolidations)",
                        name, snap._consolidation_count,
                    )
                else:
                    logger.warning(
                        "Discarded persisted Fisher for '%s': shape or finiteness "
                        "mismatch against the newly registered parameters.", name,
                    )
            self._snapshots[name] = snap
        logger.debug("Registered parameter set '%s': %d parameters", name, flat.size)

    def record_gradient(self, name: str, gradient: np.ndarray) -> None:
        """Record a gradient sample for Fisher estimation."""
        snap = self._snapshots.get(name)
        if snap is None:
            return
        g = np.asarray(gradient, dtype=np.float64).ravel()
        # A NaN norm compares False against every threshold, so the clip below
        # silently passed non-finite gradients straight into Fisher, where one
        # poisons the diagonal permanently.
        if not np.all(np.isfinite(g)):
            record_degradation(
                "plasticity_governor",
                ValueError(f"non-finite gradient for '{name}'"),
                severity="warning",
                action="dropped a non-finite gradient sample",
            )
            return
        norm = float(np.linalg.norm(g))
        if math.isfinite(norm) and norm > self.config.gradient_clip:
            g = g * (self.config.gradient_clip / norm)

        with self._lock:
            try:
                snap.accumulate_gradient(g)
            except ValueError as exc:
                record_degradation("plasticity_governor", exc, severity="warning",
                                   action="rejected a misaligned gradient sample")
                return
            self._step_count += 1
            due = self._step_count % self.config.consolidation_interval == 0

        # consolidation_interval was exposed as a safety control and then never
        # read: the governor would never consolidate unless something external
        # remembered to call it. It now fires on the documented schedule.
        if due:
            self.consolidate()

    def consolidate(
        self,
        name: str | None = None,
        current_params: dict[str, np.ndarray] | None = None,
    ) -> list[ConsolidationRecord]:
        """Estimate Fisher and consolidate parameter anchors.

        Args:
            name: If given, consolidate only this parameter set.
                  If None, consolidate all.

        Returns:
            List of consolidation records.
        """
        records = []
        with self._lock:
            targets = [name] if name else list(self._snapshots.keys())

        for pname in targets:
            snap = self._snapshots.get(pname)
            if snap is None:
                continue

            # min_samples_for_fisher was exposed as a safety parameter and never
            # read, so a Fisher estimate could be built from a SINGLE gradient
            # and then treated as an importance map.
            pending = len(snap._gradient_accumulator)
            if pending < self.config.min_samples_for_fisher:
                # An explicit consolidate() that finds nothing pending is
                # ambiguous, and the ambiguity is a real hazard rather than a
                # cosmetic one.
                #
                # record_gradient() auto-consolidates every
                # consolidation_interval steps and CLEARS the accumulator, so
                # a caller that records N samples and then consolidates can
                # find its samples already consumed — through no fault of its
                # own, purely because the shared step counter happened to
                # cross a multiple. Returning [] then reads as "consolidation
                # failed" when the parameter is in fact consolidated, and the
                # caller's next move is usually to retry or to treat the
                # parameter as unprotected.
                #
                # So report the consolidation that DOES exist. An empty list
                # now means genuinely not consolidated.
                if snap._consolidation_count > 0 and snap.fisher_diag is not None:
                    fisher = np.asarray(snap.fisher_diag, dtype=np.float64)
                    if fisher.size and float(np.sum(fisher)) > 0:
                        records.append(
                            ConsolidationRecord(
                                timestamp=time.time(),
                                parameter_set=pname,
                                fisher_norm=float(np.linalg.norm(fisher)),
                                n_parameters=snap.n_params,
                                mean_importance=float(np.mean(fisher)),
                                max_importance=float(np.max(fisher)),
                            )
                        )
                        logger.debug(
                            "'%s' already consolidated (%d prior pass(es)); "
                            "%d/%d new samples pending.",
                            pname, snap._consolidation_count, pending,
                            self.config.min_samples_for_fisher,
                        )
                        continue
                logger.debug(
                    "Skipped consolidating '%s': %d/%d gradient samples.",
                    pname, pending, self.config.min_samples_for_fisher,
                )
                continue

            with self._lock:
                fisher_norm = snap.estimate_fisher(gamma=self.config.fisher_gamma)
                # EWC consolidation means anchoring at the weights that were
                # just consolidated. theta_star was left at REGISTRATION values
                # unless a caller separately remembered update_anchor(), so the
                # penalty pulled toward the initial weights forever and the
                # documented consolidation points never existed.
                if current_params is not None and pname in current_params:
                    proposed = np.asarray(current_params[pname], dtype=np.float64).ravel()
                    if proposed.size == snap.n_params and np.all(np.isfinite(proposed)):
                        snap.update_anchor(proposed)
                    else:
                        logger.warning(
                            "Anchor for '%s' not updated: shape/finiteness mismatch.",
                            pname,
                        )
                elif snap._consolidation_count > 1:
                    logger.debug(
                        "Consolidated '%s' without current parameters: the anchor "
                        "remains where it was last set.", pname,
                    )

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
                        proposed_delta: np.ndarray) -> tuple[np.ndarray, PenaltyReport]:
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

        if penalty_grad.size != delta.size:
            # Silently resizing here misaligns the penalty against the update,
            # so parameter i is braked by parameter j's importance.
            raise ValueError(
                f"proposed delta size {delta.size} does not match parameter set "
                f"'{name}' ({penalty_grad.size}); refusing to misalign the penalty"
            )
        if not np.all(np.isfinite(penalty_grad)):
            record_degradation(
                "plasticity_governor",
                ValueError(f"non-finite EWC penalty for '{name}'"),
                severity="warning",
                action="applied the update unpenalized rather than corrupting it",
            )
            penalty_grad = np.zeros_like(delta)

        # Trust region. The raw EWC gradient was subtracted directly from the
        # proposed delta with no learning-rate integration or norm cap, so a
        # large lambda or a sharp Fisher could produce a penalty BIGGER than the
        # update — reversing its direction and amplifying its magnitude. A
        # brake that pushes is worse than no brake. The penalty is scaled so it
        # can cancel an update but never invert it.
        penalty_norm = float(np.linalg.norm(penalty_grad))
        max_penalty = self.config.max_penalty_ratio * original_norm
        clipped = False
        if penalty_norm > max_penalty > 0.0:
            penalty_grad = penalty_grad * (max_penalty / penalty_norm)
            clipped = True
        elif original_norm <= 0.0:
            # No proposed movement means nothing to brake.
            penalty_grad = np.zeros_like(delta)

        penalized = delta - penalty_grad
        penalized_norm = float(np.linalg.norm(penalized))
        if clipped:
            logger.debug(
                "EWC penalty on '%s' clipped to the trust region "
                "(%.4f -> %.4f, update norm %.4f).",
                name, penalty_norm, max_penalty, original_norm,
            )

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

    def get_importance_map(self, name: str) -> np.ndarray | None:
        """Get the Fisher diagonal (importance map) for a parameter set."""
        snap = self._snapshots.get(name)
        return snap.fisher_diag.copy() if snap else None

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            save_dict: dict[str, np.ndarray] = {
                "step_count": np.array([self._step_count]),
            }
            for name, snap in self._snapshots.items():
                save_dict[f"{name}__theta_star"] = snap.theta_star
                save_dict[f"{name}__fisher"] = snap.fisher_diag
                save_dict[f"{name}__count"] = np.array([snap._consolidation_count])
            # Atomic replace through the GOVERNED gateway. A crash mid-write
            # previously left a truncated .npz the next boot would fail to
            # load, silently discarding every consolidation ever made. An
            # earlier version of this fix did its own open()+fsync+replace,
            # which bought atomicity by bypassing the write gateway — the
            # durable-write ratchet caught it. Serialising to a buffer and
            # handing the bytes to the gateway gets both properties.
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **save_dict)
            get_file_write_gateway().write_bytes(
                _FISHER_PATH,
                buffer.getvalue(),
                source="adaptation.plasticity_governor.fisher_state",
            )
            logger.debug("Plasticity state saved")
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            # OSError was outside the handler, so a full disk or permission
            # failure propagated out of a routine save.
            record_degradation("plasticity_governor", exc, severity="warning",
                               action="plasticity state not persisted this cycle")

    def _load(self) -> None:
        try:
            if not _FISHER_PATH.exists():
                return
            data = np.load(str(_FISHER_PATH), allow_pickle=False)
            self._step_count = int(data.get("step_count", [0])[0])
            # Iterate the FILE, not self._snapshots — nothing is registered yet
            # at construction time, which is why every restore was a no-op.
            # The payload is held until the matching set registers.
            for key in data.files:
                if not key.endswith("__theta_star"):
                    continue
                name = key[: -len("__theta_star")]
                f_key, c_key = f"{name}__fisher", f"{name}__count"
                if f_key not in data.files:
                    continue
                entry: dict[str, Any] = {
                    "theta_star": np.asarray(data[key], dtype=np.float64),
                    "fisher": np.asarray(data[f_key], dtype=np.float64),
                }
                if c_key in data.files:
                    try:
                        entry["count"] = int(np.asarray(data[c_key]).ravel()[0])
                    except (IndexError, ValueError, TypeError):
                        entry["count"] = 0
                self._persisted[name] = entry
            logger.info(
                "Plasticity state restored (step %d, %d parameter set(s) pending "
                "registration)", self._step_count, len(self._persisted),
            )
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as exc:
            record_degradation("plasticity_governor", exc, severity="warning",
                               action="started with no restored Fisher protection")

    # ── Public API ──────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
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


_instance: PlasticityGovernor | None = None


def get_plasticity_governor() -> PlasticityGovernor:
    """Get or create the singleton PlasticityGovernor."""
    global _instance
    if _instance is None:
        _instance = PlasticityGovernor()
    return _instance
