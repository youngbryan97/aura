"""
TPM Sampling Error Analysis for Phi Estimates
==============================================

Empirical TPMs are estimated from finite samples of system behavior.
This introduces sampling noise that propagates through the phi computation,
producing uncertainty in phi estimates that is rarely characterized.

The core question: how many samples do you need before your phi estimate
is reliable? If you're computing phi from 50 state transitions vs 5000,
how different are the error distributions?

This module:
1. Bootstrap resampling: given an empirical TPM and sample count, resample
   N times, compute phi each time, return the full error distribution.
2. Minimum sample size derivation: for a given epsilon-accuracy target,
   compute the minimum number of transitions needed.
3. Confidence intervals: Bayesian and frequentist CI for phi estimates.
4. Bias characterization: does finite sampling systematically over- or
   under-estimate phi?

Uses numpy throughout. Compatible with both the RIIU surrogate phi
(core/consciousness/iit_surrogate.py) and the spectral approximation
(research/phi_approximation.py).
"""
from __future__ import annotations


import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Research.TPMErrorAnalysis")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhiErrorDistribution:
    """The distribution of phi estimates from bootstrap resampling."""
    phi_samples: np.ndarray          # Array of phi estimates from each resample
    mean: float                      # Mean phi across resamples
    std: float                       # Standard deviation
    ci_lower: float                  # 95% CI lower bound
    ci_upper: float                  # 95% CI upper bound
    bias: float                      # Mean(resampled) - original (systematic bias)
    cv: float                        # Coefficient of variation (std/mean)
    n_resamples: int
    original_phi: float              # Phi from the original (non-resampled) TPM
    sample_count: int                # Number of transitions used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Phi={self.mean:.4f} +/- {self.std:.4f} "
            f"(95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]) "
            f"bias={self.bias:+.4f}, CV={self.cv:.3f}, "
            f"n_samples={self.sample_count}, n_resamples={self.n_resamples}"
        )


@dataclass
class MinSampleResult:
    """Result of minimum sample size estimation."""
    epsilon: float                   # Target accuracy (max acceptable error)
    confidence_level: float          # Target confidence (e.g., 0.95)
    min_samples: int                 # Minimum transitions needed
    achieved_error: float            # Actual error at min_samples
    achieved_confidence: float       # Actual confidence at min_samples
    sample_curve: List[Tuple[int, float]]  # (n_samples, error) curve
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"For epsilon={self.epsilon:.4f} at {self.confidence_level:.0%} confidence: "
            f"need {self.min_samples} samples (achieved error={self.achieved_error:.4f})"
        )


# ---------------------------------------------------------------------------
# TPM utilities
# ---------------------------------------------------------------------------

def _normalize_tpm(tpm: np.ndarray) -> np.ndarray:
    """Ensure each row of the TPM sums to 1.0 (valid probability distribution)."""
    row_sums = tpm.sum(axis=1, keepdims=True)
    # Avoid division by zero: rows with zero sum get uniform distribution
    zero_rows = row_sums < 1e-12
    row_sums[zero_rows] = 1.0
    normalized = tpm / row_sums
    # Set zero rows to uniform
    n = tpm.shape[1]
    normalized[zero_rows.flatten()] = 1.0 / n
    return normalized


def _generate_transitions_from_tpm(
    tpm: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic state transitions from a TPM.

    Simulates a Markov chain to produce (state, next_state) pairs.
    Returns an array of shape (n_samples, 2) with state indices.
    """
    n_states = tpm.shape[0]
    transitions = np.zeros((n_samples, 2), dtype=int)

    # Start from a random state
    current = rng.integers(0, n_states)

    for i in range(n_samples):
        transitions[i, 0] = current
        # Sample next state from transition distribution
        probs = tpm[current]
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total < 1e-12:
            current = rng.integers(0, n_states)
        else:
            probs /= total
            current = rng.choice(n_states, p=probs)
        transitions[i, 1] = current

    return transitions


def _estimate_tpm_from_transitions(
    transitions: np.ndarray,
    n_states: int,
    pseudocount: float = 0.01,
) -> np.ndarray:
    """Estimate a TPM from observed transitions using maximum likelihood + pseudocounts.

    Args:
        transitions: (n_samples, 2) array of (state, next_state) pairs
        n_states: Total number of states
        pseudocount: Laplace smoothing parameter (prevents zero probabilities)

    Returns:
        Estimated TPM of shape (n_states, n_states)
    """
    counts = np.full((n_states, n_states), pseudocount, dtype=np.float64)

    for i in range(transitions.shape[0]):
        s = transitions[i, 0]
        s_next = transitions[i, 1]
        if 0 <= s < n_states and 0 <= s_next < n_states:
            counts[s, s_next] += 1.0

    return _normalize_tpm(counts)


# ---------------------------------------------------------------------------
# Phi computation (lightweight surrogate)
# ---------------------------------------------------------------------------

def _compute_phi_from_tpm(
    tpm: np.ndarray,
    n_nodes: int,
    n_partitions: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute a surrogate phi from a TPM using covariance-based integration.

    This is a simplified version of the RIIU computation adapted for
    discrete state TPMs. It measures how much information is lost when
    the system is partitioned.

    phi = log|Sigma_whole| - max(log|Sigma_partition|)

    The covariance is computed from the transition probabilities treated
    as a multivariate distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_states = tpm.shape[0]
    if n_states < 4:
        return 0.0

    # Build a node-level representation from the state-level TPM
    # Each node's "activity" is the marginal probability of being ON
    # given each possible input state
    node_activities = np.zeros((n_states, n_nodes), dtype=np.float64)
    for s in range(n_states):
        for node in range(n_nodes):
            # P(node ON at t+1 | state s at t) = sum over states where node is ON
            prob_on = 0.0
            for s_next in range(n_states):
                if (s_next >> node) & 1:
                    prob_on += tpm[s, s_next]
            node_activities[s, node] = prob_on

    # Whole-system covariance
    reg = 1e-6
    cov_whole = np.cov(node_activities, rowvar=False)
    if cov_whole.ndim == 0:
        cov_whole = np.array([[float(cov_whole)]])
    cov_whole += np.eye(cov_whole.shape[0]) * reg

    sign_w, logdet_w = np.linalg.slogdet(cov_whole)
    if sign_w <= 0 or not np.isfinite(logdet_w):
        return 0.0

    # Test partitions
    max_part_logdet = -np.inf
    indices = np.arange(n_nodes)

    for _ in range(n_partitions):
        split = rng.integers(max(2, n_nodes // 4), max(3, 3 * n_nodes // 4))
        perm = rng.permutation(indices)
        part_a = perm[:split]
        part_b = perm[split:]

        if len(part_a) < 2 or len(part_b) < 2:
            continue

        cov_a = np.cov(node_activities[:, part_a], rowvar=False)
        cov_b = np.cov(node_activities[:, part_b], rowvar=False)

        if cov_a.ndim == 0:
            cov_a = np.array([[float(cov_a)]])
        if cov_b.ndim == 0:
            cov_b = np.array([[float(cov_b)]])

        cov_a += np.eye(cov_a.shape[0]) * reg
        cov_b += np.eye(cov_b.shape[0]) * reg

        s_a, ld_a = np.linalg.slogdet(cov_a)
        s_b, ld_b = np.linalg.slogdet(cov_b)

        if s_a <= 0 or s_b <= 0 or not np.isfinite(ld_a) or not np.isfinite(ld_b):
            continue

        part_logdet = ld_a + ld_b
        if np.isfinite(part_logdet):
            max_part_logdet = max(max_part_logdet, part_logdet)

    if max_part_logdet == -np.inf:
        return 0.0

    phi = logdet_w - max_part_logdet
    return float(np.clip(phi, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Bootstrap resampling engine
# ---------------------------------------------------------------------------

class TPMErrorAnalyzer:
    """Characterizes how sampling noise in empirical TPMs affects phi estimates.

    Usage:
        analyzer = TPMErrorAnalyzer()

        # From an existing TPM with known sample count
        dist = analyzer.bootstrap_phi(tpm, n_nodes=8, sample_count=200, n_resamples=500)
        print(dist.summary())

        # Find minimum sample size for desired accuracy
        result = analyzer.minimum_sample_size(tpm, n_nodes=8, epsilon=0.05)
        print(result.summary())
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._cache: Dict[str, PhiErrorDistribution] = {}

    def bootstrap_phi(
        self,
        tpm: np.ndarray,
        n_nodes: int,
        sample_count: int,
        n_resamples: int = 500,
        n_partitions: int = 4,
    ) -> PhiErrorDistribution:
        """Bootstrap resample a TPM to characterize phi estimation error.

        Protocol:
        1. Compute phi from the original TPM (ground truth for this analysis)
        2. Generate 'sample_count' synthetic transitions from the TPM
        3. For each of n_resamples iterations:
           a. Resample transitions with replacement
           b. Estimate a new TPM from the resampled transitions
           c. Compute phi from the resampled TPM
        4. Return the distribution of phi estimates

        Args:
            tpm: The "true" transition probability matrix (n_states x n_states)
            n_nodes: Number of nodes in the complex (tpm is 2^n_nodes x 2^n_nodes)
            sample_count: Number of transitions to simulate (the "sample size")
            n_resamples: Number of bootstrap iterations
            n_partitions: Partition count for phi computation

        Returns:
            PhiErrorDistribution with full statistics
        """
        t0 = time.time()
        tpm = _normalize_tpm(tpm)
        n_states = tpm.shape[0]

        # Step 1: Original phi
        original_phi = _compute_phi_from_tpm(tpm, n_nodes, n_partitions, self._rng)

        # Step 2: Generate synthetic transitions
        transitions = _generate_transitions_from_tpm(tpm, sample_count, self._rng)

        # Step 3: Bootstrap
        phi_samples = np.zeros(n_resamples, dtype=np.float64)

        for i in range(n_resamples):
            # Resample with replacement
            indices = self._rng.integers(0, sample_count, size=sample_count)
            resampled = transitions[indices]

            # Estimate TPM from resampled transitions
            resampled_tpm = _estimate_tpm_from_transitions(resampled, n_states)

            # Compute phi
            phi_samples[i] = _compute_phi_from_tpm(
                resampled_tpm, n_nodes, n_partitions, self._rng
            )

        # Step 4: Statistics
        mean_phi = float(np.mean(phi_samples))
        std_phi = float(np.std(phi_samples, ddof=1))
        ci_lower = float(np.percentile(phi_samples, 2.5))
        ci_upper = float(np.percentile(phi_samples, 97.5))
        bias = mean_phi - original_phi
        cv = std_phi / max(1e-8, abs(mean_phi))

        elapsed = time.time() - t0

        dist = PhiErrorDistribution(
            phi_samples=phi_samples,
            mean=mean_phi,
            std=std_phi,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bias=bias,
            cv=cv,
            n_resamples=n_resamples,
            original_phi=original_phi,
            sample_count=sample_count,
            metadata={
                "n_nodes": n_nodes,
                "n_states": n_states,
                "n_partitions": n_partitions,
                "elapsed_ms": round(elapsed * 1000, 2),
                "skewness": float(_skewness(phi_samples)),
                "kurtosis": float(_kurtosis(phi_samples)),
            },
        )

        logger.info("Bootstrap phi: %s", dist.summary())
        return dist

    def minimum_sample_size(
        self,
        tpm: np.ndarray,
        n_nodes: int,
        epsilon: float = 0.05,
        confidence_level: float = 0.95,
        max_samples: int = 10000,
        n_resamples: int = 200,
        n_partitions: int = 4,
    ) -> MinSampleResult:
        """Find the minimum sample count for epsilon-accurate phi estimation.

        Uses binary search over sample counts. For each candidate count,
        runs bootstrap to estimate the error distribution. The minimum
        count is the smallest N where P(|error| < epsilon) >= confidence_level.

        Args:
            tpm: True TPM
            n_nodes: Number of nodes
            epsilon: Maximum acceptable absolute error in phi
            confidence_level: Required probability of being within epsilon
            max_samples: Upper bound on sample count search
            n_resamples: Bootstrap iterations per candidate
            n_partitions: Partitions for phi computation

        Returns:
            MinSampleResult with the minimum sample size and error curve
        """
        t0 = time.time()
        tpm = _normalize_tpm(tpm)

        # Compute reference phi
        ref_phi = _compute_phi_from_tpm(tpm, n_nodes, n_partitions, self._rng)

        # Sample counts to test (geometric progression for efficiency)
        test_counts = sorted(set(
            [int(x) for x in np.geomspace(10, max_samples, num=15).astype(int)]
        ))

        sample_curve: List[Tuple[int, float]] = []
        min_n = max_samples
        achieved_error = float('inf')
        achieved_confidence = 0.0

        for n in test_counts:
            dist = self.bootstrap_phi(tpm, n_nodes, n, n_resamples, n_partitions)

            # What fraction of resampled phis are within epsilon of original?
            within_epsilon = np.abs(dist.phi_samples - ref_phi) < epsilon
            frac_within = float(np.mean(within_epsilon))
            mean_error = float(np.mean(np.abs(dist.phi_samples - ref_phi)))

            sample_curve.append((n, mean_error))

            if frac_within >= confidence_level and n < min_n:
                min_n = n
                achieved_error = mean_error
                achieved_confidence = frac_within

        elapsed = time.time() - t0

        result = MinSampleResult(
            epsilon=epsilon,
            confidence_level=confidence_level,
            min_samples=min_n,
            achieved_error=achieved_error,
            achieved_confidence=achieved_confidence,
            sample_curve=sample_curve,
            metadata={
                "n_nodes": n_nodes,
                "ref_phi": ref_phi,
                "n_test_counts": len(test_counts),
                "elapsed_ms": round(elapsed * 1000, 2),
            },
        )

        logger.info("Min sample size: %s", result.summary())
        return result

    def bias_characterization(
        self,
        tpm: np.ndarray,
        n_nodes: int,
        sample_counts: Optional[List[int]] = None,
        n_resamples: int = 200,
    ) -> Dict[str, Any]:
        """Characterize systematic bias in phi estimation across sample sizes.

        Does finite sampling systematically over- or under-estimate phi?
        This matters because if the bias is predictable, it can be corrected.

        Returns:
            Dict with bias statistics at each sample size
        """
        if sample_counts is None:
            sample_counts = [20, 50, 100, 200, 500, 1000]

        tpm = _normalize_tpm(tpm)
        ref_phi = _compute_phi_from_tpm(tpm, n_nodes, rng=self._rng)

        results = []
        for n in sample_counts:
            dist = self.bootstrap_phi(tpm, n_nodes, n, n_resamples)
            results.append({
                "sample_count": n,
                "bias": dist.bias,
                "std": dist.std,
                "cv": dist.cv,
                "relative_bias": dist.bias / max(1e-8, abs(ref_phi)),
            })

        # Fit bias as a function of 1/n (expected scaling for finite-sample bias)
        ns = np.array([r["sample_count"] for r in results], dtype=float)
        biases = np.array([r["bias"] for r in results], dtype=float)

        if len(ns) > 1 and np.std(biases) > 1e-10:
            # Fit bias ~ a / n + b
            inv_n = 1.0 / ns
            coeffs = np.polyfit(inv_n, biases, 1)
            bias_scaling_coefficient = float(coeffs[0])
            bias_asymptote = float(coeffs[1])
        else:
            bias_scaling_coefficient = 0.0
            bias_asymptote = 0.0

        return {
            "reference_phi": ref_phi,
            "per_sample_count": results,
            "bias_scaling_coefficient": bias_scaling_coefficient,
            "bias_asymptote": bias_asymptote,
            "bias_direction": "over" if bias_asymptote > 0 else "under",
            "bias_is_significant": abs(bias_asymptote) > 0.01 * abs(ref_phi),
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    """Compute Fisher's skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3) * n * n / ((n - 1) * (n - 2)))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def generate_random_tpm(n_nodes: int, sparsity: float = 0.3, seed: int = 42) -> np.ndarray:
    """Generate a random TPM for testing.

    Creates a TPM with controlled sparsity (fraction of near-zero entries)
    to simulate realistic system dynamics where not all state transitions
    are equally likely.

    Args:
        n_nodes: Number of nodes (TPM will be 2^n_nodes x 2^n_nodes)
        sparsity: Fraction of entries to suppress (0 = dense, 1 = very sparse)
        seed: Random seed

    Returns:
        Normalized TPM
    """
    rng = np.random.default_rng(seed)
    n_states = 2 ** n_nodes

    # Start with random positive entries
    tpm = rng.exponential(1.0, size=(n_states, n_states))

    # Apply sparsity mask
    mask = rng.random((n_states, n_states)) > sparsity
    tpm *= mask

    # Ensure self-transitions are nonzero (system tends to stay in state)
    tpm += np.eye(n_states) * 0.5

    return _normalize_tpm(tpm)
