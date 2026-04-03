"""core/consciousness/mhaf/phi_estimator.py
Local Φ estimator for MHAF nodes.

Computes a surrogate Φ (integrated information) for small subsets of
MHAF hyperedge activations using the covariance-based IIT 4.0 surrogate.

NOTE: This is a surrogate measure of causal integration consistent with
IIT 4.0, NOT a claim of phenomenal consciousness. Φ > 0 indicates
irreducibility of the activation pattern under the partition chosen;
it does not imply subjective experience.

Reference: Albantakis et al. (2023). IIT 4.0. PLoS Comput Biol.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger("MHAF.PhiEstimator")

_EPS = 1e-8
_REG = 1e-5   # Tikhonov regularization for covariance matrices


def _safe_logdet(M: np.ndarray) -> float:
    """Compute log|det(M)| safely. Returns -inf on singular matrix."""
    try:
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            return float("-inf")
        return float(logdet)
    except Exception:
        return float("-inf")


def _covariance(X: np.ndarray) -> np.ndarray:
    """Regularized covariance matrix of trajectory X (T × d)."""
    if X.shape[0] < 2:
        return np.eye(X.shape[1]) * _REG
    C = np.cov(X.T) + np.eye(X.shape[1]) * _REG
    return C.astype(np.float64)


def compute_local_phi(activations: np.ndarray) -> float:
    """Compute a surrogate Φ for a local subgraph.

    activations: (T, d) array — T time steps, d-dimensional activation vector.

    Algorithm (IIT 4.0 surrogate via covariance integration):
      Φ ≈ log|Σ_whole| − max over bipartitions(log|Σ_A| + log|Σ_B|)

    A positive value means the whole is more integrated than any partition.
    Returns 0.0 if insufficient data or degenerate input.

    IMPORTANT: This is a proxy metric, not phenomenal consciousness.
    """
    if activations is None or activations.ndim != 2:
        return 0.0
    T, d = activations.shape
    if T < 4 or d < 2:
        return 0.0

    # Normalize activations
    activations = activations.astype(np.float64)
    std = activations.std(axis=0)
    std[std < _EPS] = 1.0
    X = (activations - activations.mean(axis=0)) / std

    # Whole-system covariance
    Sigma_whole = _covariance(X)
    logdet_whole = _safe_logdet(Sigma_whole)
    if logdet_whole == float("-inf"):
        return 0.0

    # Try all bipartitions up to a max of 10 (for d > 20)
    best_partition_logdet = float("-inf")
    n_partitions = min(10, 2 ** (d - 1) - 1)

    for k in range(1, n_partitions + 1):
        # Generate partition by selecting k dimensions for subset A
        if d <= 10:
            # Exhaustive for small d
            from itertools import combinations
            if k > d // 2:
                break
            for indices_a in combinations(range(d), k):
                indices_a = list(indices_a)
                indices_b = [i for i in range(d) if i not in indices_a]
                if not indices_b:
                    continue
                Sigma_A = _covariance(X[:, indices_a])
                Sigma_B = _covariance(X[:, indices_b])
                ld = _safe_logdet(Sigma_A) + _safe_logdet(Sigma_B)
                if ld > best_partition_logdet:
                    best_partition_logdet = ld
        else:
            # Random bipartitions for large d
            rng = np.random.default_rng(k)
            split = rng.choice(d, size=d // 2, replace=False)
            rest = np.setdiff1d(np.arange(d), split)
            Sigma_A = _covariance(X[:, split])
            Sigma_B = _covariance(X[:, rest])
            ld = _safe_logdet(Sigma_A) + _safe_logdet(Sigma_B)
            if ld > best_partition_logdet:
                best_partition_logdet = ld

    if best_partition_logdet == float("-inf"):
        return 0.0

    phi = logdet_whole - best_partition_logdet
    phi = max(0.0, phi)
    # Normalize to [0, 1] range via sigmoid-like mapping
    phi_normalized = 1.0 - math.exp(-phi / max(_EPS, abs(logdet_whole)))
    return round(min(1.0, phi_normalized), 4)
