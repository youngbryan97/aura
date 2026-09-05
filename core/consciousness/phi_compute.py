"""core/consciousness/phi_compute.py -- Integrated Information (Phi) Computation
==============================================================================
Computes integrated information for the neural substrate using Geometric
Integrated Information (Phi_G) from Barrett & Seth (2011).

For continuous dynamical systems like the ODE substrate, Phi_G is computed
from the covariance structure of the system's trajectory.  The Minimum
Information Partition (MIP) is found via spectral graph methods for
scalability (O(N^2 log N) vs O(2^N) exhaustive).

Algorithm:
  1. Collect T trajectory samples from the substrate state vector
  2. Compute covariance matrix Sigma
  3. For candidate bipartitions {A, B}:
     Phi_G(A,B) = H(A) + H(B) - H(A u B)
     Using Gaussian entropy: H(X) = 0.5 * ln det(2*pi*e * Sigma_X)
  4. MIP = argmin_{A,B} Phi_G(A,B)
  5. Phi = Phi_G(MIP)

Scalability:
  N <= 16: exhaustive bipartition search
  N >  16: spectral partition via Fiedler vector + Kernighan-Lin swaps

References:
    Barrett & Seth (2011) PLoS Comp Bio
    Oizumi et al. (2014) Neurosci Consciousness
    Tononi (2004) BMC Neuroscience
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.PhiCompute")


@dataclass
class PhiResult:
    """Result of an integrated information computation."""
    phi: float
    mip_partition: Tuple[List[int], List[int]]
    mip_loss: float
    system_entropy: float
    part_entropies: Tuple[float, float]
    n_neurons: int
    n_partitions_evaluated: int
    computation_time_ms: float
    method: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phi": round(self.phi, 6),
            "mip_sizes": (len(self.mip_partition[0]), len(self.mip_partition[1])),
            "system_entropy": round(self.system_entropy, 6),
            "n_neurons": self.n_neurons,
            "n_partitions_evaluated": self.n_partitions_evaluated,
            "computation_time_ms": round(self.computation_time_ms, 2),
            "method": self.method,
        }


@dataclass
class PhiConfig:
    """Configuration for phi computation."""
    trajectory_length: int = 200
    sample_interval: int = 1
    regularization: float = 1e-6
    max_exhaustive_n: int = 16
    kl_refinement_swaps: int = 50
    min_partition_size: int = 1
    seed: int = 91


class PhiComputer:
    """Compute integrated information (Phi) for a neural substrate.

    Usage:
        computer = PhiComputer()
        for state_vector in substrate_trajectory:
            computer.record_state(state_vector)
        result = computer.compute()
        print(f"Phi = {result.phi:.4f}")
    """

    def __init__(self, config: Optional[PhiConfig] = None) -> None:
        self.config = config or PhiConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._trajectory: Deque[np.ndarray] = deque(maxlen=self.config.trajectory_length)
        self._step_counter = 0
        self._last_result: Optional[PhiResult] = None
        self._phi_history: Deque[float] = deque(maxlen=100)

    def record_state(self, state: np.ndarray) -> None:
        """Record a substrate state snapshot."""
        self._step_counter += 1
        if self._step_counter % self.config.sample_interval == 0:
            self._trajectory.append(np.asarray(state, dtype=np.float64).ravel().copy())

    @property
    def has_sufficient_data(self) -> bool:
        return len(self._trajectory) >= max(20, self._trajectory.maxlen // 2)

    def compute(self, coupling_matrix: Optional[np.ndarray] = None) -> PhiResult:
        """Compute Phi from collected trajectory."""
        t_start = time.monotonic()

        if not self.has_sufficient_data:
            raise ValueError(
                f"Insufficient data: {len(self._trajectory)} samples "
                f"(need >= {max(20, self._trajectory.maxlen // 2)})"
            )

        states = np.array(list(self._trajectory), dtype=np.float64)
        T, N = states.shape

        if N < 2:
            r = PhiResult(phi=0.0, mip_partition=([0], []), mip_loss=0.0,
                          system_entropy=0.0, part_entropies=(0.0, 0.0),
                          n_neurons=N, n_partitions_evaluated=0,
                          computation_time_ms=0.0, method="trivial")
            self._last_result = r
            return r

        cov = np.cov(states.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov += self.config.regularization * np.eye(cov.shape[0])

        system_entropy = self._gaussian_entropy(cov)

        if N <= self.config.max_exhaustive_n:
            result = self._exhaustive_phi(cov, system_entropy, N)
        else:
            result = self._spectral_phi(cov, system_entropy, N, coupling_matrix)

        result.computation_time_ms = (time.monotonic() - t_start) * 1000
        self._last_result = result
        self._phi_history.append(result.phi)
        logger.debug("Phi=%.4f N=%d method=%s %.1fms",
                      result.phi, N, result.method, result.computation_time_ms)
        return result

    # ── Entropy ─────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_entropy(cov: np.ndarray) -> float:
        """H(X) = 0.5 * ln det(2*pi*e * Sigma) via Cholesky."""
        n = cov.shape[0]
        try:
            L = np.linalg.cholesky(cov)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-15)
            log_det = float(np.sum(np.log(eigvals)))
        return float(0.5 * (n * np.log(2 * np.pi * np.e) + log_det))

    def _partition_entropy(self, cov: np.ndarray,
                           a: List[int], b: List[int]) -> Tuple[float, float]:
        if not a or not b:
            return (0.0, 0.0)
        return (self._gaussian_entropy(cov[np.ix_(a, a)]),
                self._gaussian_entropy(cov[np.ix_(b, b)]))

    def _phi_for_partition(self, cov: np.ndarray, sys_h: float,
                           a: List[int], b: List[int]) -> float:
        h_a, h_b = self._partition_entropy(cov, a, b)
        return max(0.0, h_a + h_b - sys_h)

    # ── Exhaustive (N <= 16) ────────────────────────────────────────────

    def _exhaustive_phi(self, cov: np.ndarray, sys_h: float, N: int) -> PhiResult:
        min_phi = float("inf")
        best_a: List[int] = []
        best_b: List[int] = []
        n_eval = 0
        ms = self.config.min_partition_size
        indices = list(range(N))

        for mask in range(1, 2 ** N - 1):
            a = [i for i in indices if mask & (1 << i)]
            b = [i for i in indices if not (mask & (1 << i))]
            if len(a) < ms or len(b) < ms:
                continue
            if len(a) > len(b) or (len(a) == len(b) and a[0] > b[0]):
                continue
            phi = self._phi_for_partition(cov, sys_h, a, b)
            n_eval += 1
            if phi < min_phi:
                min_phi = phi
                best_a, best_b = a, b

        if min_phi == float("inf"):
            min_phi = 0.0
        h_a, h_b = self._partition_entropy(cov, best_a, best_b)
        return PhiResult(phi=min_phi, mip_partition=(best_a, best_b),
                         mip_loss=min_phi, system_entropy=sys_h,
                         part_entropies=(h_a, h_b), n_neurons=N,
                         n_partitions_evaluated=n_eval,
                         computation_time_ms=0.0, method="exhaustive")

    # ── Spectral (N > 16) ──────────────────────────────────────────────

    def _spectral_phi(self, cov: np.ndarray, sys_h: float, N: int,
                      coupling: Optional[np.ndarray] = None) -> PhiResult:
        adj = np.abs(coupling) if coupling is not None and coupling.shape == (N, N) else np.abs(cov)
        np.fill_diagonal(adj, 0.0)
        degree = np.sum(adj, axis=1)
        laplacian = np.diag(degree) - adj
        d_inv = np.zeros_like(degree)
        nz = degree > 1e-10
        d_inv[nz] = 1.0 / np.sqrt(degree[nz])
        L_norm = np.diag(d_inv) @ laplacian @ np.diag(d_inv)

        eigvals, eigvecs = np.linalg.eigh(L_norm)
        fiedler = eigvecs[:, min(1, N - 1)]

        a = [i for i in range(N) if fiedler[i] >= 0]
        b = [i for i in range(N) if fiedler[i] < 0]

        ms = self.config.min_partition_size
        while len(a) < ms and b:
            a.append(b.pop())
        while len(b) < ms and a:
            b.append(a.pop())

        best_phi = self._phi_for_partition(cov, sys_h, a, b)
        n_eval = 1

        # Kernighan-Lin swap refinement
        for _ in range(self.config.kl_refinement_swaps):
            improved = False
            for ii in range(len(a)):
                for jj in range(len(b)):
                    ta, tb = a.copy(), b.copy()
                    ta[ii], tb[jj] = tb[jj], ta[ii]
                    if len(ta) < ms or len(tb) < ms:
                        continue
                    tp = self._phi_for_partition(cov, sys_h, ta, tb)
                    n_eval += 1
                    if tp < best_phi:
                        best_phi, a, b = tp, ta, tb
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        # Single-element moves
        for _ in range(self.config.kl_refinement_swaps):
            improved = False
            for ii in range(len(a)):
                if len(a) <= ms:
                    break
                ta = a[:ii] + a[ii + 1:]
                tb = b + [a[ii]]
                tp = self._phi_for_partition(cov, sys_h, ta, tb)
                n_eval += 1
                if tp < best_phi:
                    best_phi, a, b = tp, ta, tb
                    improved = True
                    break
            if not improved:
                for jj in range(len(b)):
                    if len(b) <= ms:
                        break
                    ta = a + [b[jj]]
                    tb = b[:jj] + b[jj + 1:]
                    tp = self._phi_for_partition(cov, sys_h, ta, tb)
                    n_eval += 1
                    if tp < best_phi:
                        best_phi, a, b = tp, ta, tb
                        improved = True
                        break
            if not improved:
                break

        a.sort()
        b.sort()
        h_a, h_b = self._partition_entropy(cov, a, b)
        return PhiResult(phi=best_phi, mip_partition=(a, b), mip_loss=best_phi,
                         system_entropy=sys_h, part_entropies=(h_a, h_b),
                         n_neurons=N, n_partitions_evaluated=n_eval,
                         computation_time_ms=0.0, method="spectral")

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def latest_phi(self) -> float:
        return self._last_result.phi if self._last_result else 0.0

    @property
    def mean_phi(self) -> float:
        return float(np.mean(list(self._phi_history))) if self._phi_history else 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            "trajectory_length": len(self._trajectory),
            "has_sufficient_data": self.has_sufficient_data,
            "latest_phi": round(self.latest_phi, 6),
            "mean_phi": round(self.mean_phi, 6),
            "n_computations": len(self._phi_history),
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }


_instance: Optional[PhiComputer] = None


def get_phi_computer() -> PhiComputer:
    """Get or create the singleton PhiComputer."""
    global _instance
    if _instance is None:
        _instance = PhiComputer()
    return _instance
