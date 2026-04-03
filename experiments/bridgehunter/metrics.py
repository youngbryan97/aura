"""experiments/bridgehunter/metrics.py

Consciousness metric computations for BridgeHunter.
Standalone functions that can be called on substrate state trajectories.
"""

import numpy as np
from typing import Dict, List


def compute_phi_surrogate(trajectory: np.ndarray, n_partitions: int = 4) -> float:
    """Compute Φ surrogate from a state trajectory.

    Args:
        trajectory: Shape (T, N) — T timesteps, N neurons.
        n_partitions: Number of random bipartitions to test.

    Returns:
        Φ estimate (float ≥ 0).
    """
    T, N = trajectory.shape
    if T < 8 or N < 4:
        return 0.0

    reg = 1e-6

    # Whole-system covariance
    cov_whole = np.cov(trajectory, rowvar=False) + np.eye(N) * reg
    sign_w, logdet_w = np.linalg.slogdet(cov_whole)
    if sign_w <= 0:
        return 0.0

    # Test partitions
    rng = np.random.RandomState(42)
    max_part_logdet = -np.inf

    for _ in range(n_partitions):
        split = rng.randint(2, N - 2)
        perm = rng.permutation(N)
        a, b = perm[:split], perm[split:]

        cov_a = np.cov(trajectory[:, a], rowvar=False) + np.eye(len(a)) * reg
        cov_b = np.cov(trajectory[:, b], rowvar=False) + np.eye(len(b)) * reg

        s_a, ld_a = np.linalg.slogdet(cov_a)
        s_b, ld_b = np.linalg.slogdet(cov_b)

        if s_a > 0 and s_b > 0:
            max_part_logdet = max(max_part_logdet, ld_a + ld_b)

    if max_part_logdet == -np.inf:
        return 0.0

    return max(0.0, float(logdet_w - max_part_logdet))


def compute_ignition_rate(priorities: List[float], threshold: float = 0.6) -> float:
    """Compute the fraction of ticks where workspace priority exceeded ignition threshold.

    Args:
        priorities: List of winner priorities per tick.
        threshold: Ignition threshold.

    Returns:
        Ignition rate (0.0-1.0).
    """
    if not priorities:
        return 0.0
    ignited = sum(1 for p in priorities if p >= threshold)
    return ignited / len(priorities)


def compute_causal_emergence(trajectory: np.ndarray, grain: int = 4) -> float:
    """Compute causal emergence: does the macro-scale have more determinism
    than the micro-scale?

    Uses effective information (EI) ratio between coarse-grained and fine-grained.

    Args:
        trajectory: Shape (T, N).
        grain: Coarse-graining factor (average every `grain` neurons).

    Returns:
        Causal emergence score (> 0 means macro > micro).
    """
    T, N = trajectory.shape
    if T < 8 or N < grain * 2:
        return 0.0

    reg = 1e-6

    # Micro-scale: transition matrix determinism
    micro_ei = _effective_information(trajectory, reg)

    # Macro-scale: coarse-grain by averaging groups of neurons
    n_macro = N // grain
    macro_traj = np.zeros((T, n_macro))
    for i in range(n_macro):
        macro_traj[:, i] = trajectory[:, i * grain:(i + 1) * grain].mean(axis=1)

    macro_ei = _effective_information(macro_traj, reg)

    return float(macro_ei - micro_ei)


def compute_spectral_entropy(state: np.ndarray) -> float:
    """Shannon entropy of the activation distribution."""
    abs_x = np.abs(state) + 1e-10
    p = abs_x / abs_x.sum()
    return float(-np.sum(p * np.log2(p + 1e-10)))


def compute_self_reference(trajectory: np.ndarray, lag: int = 5) -> float:
    """Measure self-referential dynamics: autocorrelation at lag.

    High autocorrelation at moderate lag suggests the system is
    referencing its own past states (recurrent processing).
    """
    T, N = trajectory.shape
    if T <= lag:
        return 0.0

    current = trajectory[lag:]
    past = trajectory[:-lag]

    # Average cosine similarity
    sims = []
    for t in range(len(current)):
        n_c = np.linalg.norm(current[t])
        n_p = np.linalg.norm(past[t])
        if n_c > 1e-10 and n_p > 1e-10:
            sims.append(float(np.dot(current[t], past[t]) / (n_c * n_p)))

    return float(np.mean(sims)) if sims else 0.0


def _effective_information(trajectory: np.ndarray, reg: float = 1e-6) -> float:
    """Compute effective information as log-determinism of transition matrix."""
    T, N = trajectory.shape
    if T < 3:
        return 0.0

    # Estimate transition: x_{t+1} ≈ A @ x_t
    X = trajectory[:-1].T  # (N, T-1)
    Y = trajectory[1:].T   # (N, T-1)

    # Least-squares: A = Y @ X^+ (pseudo-inverse)
    try:
        A = Y @ np.linalg.pinv(X)
        # EI ≈ log|det(A)| (determinism of transitions)
        sign, logdet = np.linalg.slogdet(A)
        if sign > 0:
            return float(logdet)
    except Exception:
        pass

    return 0.0
