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

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("MHAF.PhiEstimator")

_EPS = 1e-8
_REG = 1e-5   # Tikhonov regularization for covariance matrices
_RECOVERABLE_PHI_ERRORS = (
    FloatingPointError,
    np.linalg.LinAlgError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _safe_logdet(matrix: np.ndarray) -> float:
    """Compute log|det(matrix)| safely. Returns -inf on singular matrix."""
    try:
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            return float("-inf")
        return float(logdet)
    except _RECOVERABLE_PHI_ERRORS as exc:
        record_degradation("mhaf_phi_estimator", exc)
        logger.debug("logdet computation failed: %s", exc)
        return float("-inf")


def _covariance(samples: np.ndarray) -> np.ndarray:
    """Regularized covariance matrix of trajectory X (T × d)."""
    if samples.shape[0] < 2:
        return np.eye(samples.shape[1]) * _REG
    covariance = np.cov(samples.T) + np.eye(samples.shape[1]) * _REG
    return covariance.astype(np.float64)


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
    timesteps, d = activations.shape
    if timesteps < 4 or d < 2:
        return 0.0

    # Normalize activations
    activations = activations.astype(np.float64)
    std = activations.std(axis=0)
    std[std < _EPS] = 1.0
    normalized = (activations - activations.mean(axis=0)) / std

    # Whole-system covariance
    sigma_whole = _covariance(normalized)
    logdet_whole = _safe_logdet(sigma_whole)
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
                sigma_a = _covariance(normalized[:, indices_a])
                sigma_b = _covariance(normalized[:, indices_b])
                ld = _safe_logdet(sigma_a) + _safe_logdet(sigma_b)
                if ld > best_partition_logdet:
                    best_partition_logdet = ld
        else:
            # Random bipartitions for large d
            rng = np.random.default_rng(k)
            split = rng.choice(d, size=d // 2, replace=False)
            rest = np.setdiff1d(np.arange(d), split)
            sigma_a = _covariance(normalized[:, split])
            sigma_b = _covariance(normalized[:, rest])
            ld = _safe_logdet(sigma_a) + _safe_logdet(sigma_b)
            if ld > best_partition_logdet:
                best_partition_logdet = ld

    if best_partition_logdet == float("-inf"):
        return 0.0

    phi = logdet_whole - best_partition_logdet
    phi = max(0.0, phi)
    # Normalize to [0, 1] range via sigmoid-like mapping
    phi_normalized = 1.0 - math.exp(-phi / max(_EPS, abs(logdet_whole)))
    return round(min(1.0, phi_normalized), 4)
