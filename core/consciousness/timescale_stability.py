"""
Formal Stability Analysis of Cross-Timescale Binding
=====================================================

The CrossTimescaleBinding module (core/consciousness/timescale_binding.py)
couples five temporal layers bidirectionally: reflex, moment, episode,
horizon, identity. The coupling coefficients (top-down, bottom-up)
determine whether the coupled system converges to a stable attractor
or diverges into oscillation/chaos.

This matters because:
- Too strong top-down coupling: slow layers paralyze fast layers
  (the system becomes rigid, unresponsive to moment-to-moment changes)
- Too strong bottom-up coupling: fast layers destabilize slow layers
  (identity and commitments are washed away by noise)
- Just right: stable attractor with finite-time convergence, where
  long-term commitments bias but don't dominate fast decisions

This module provides:
1. Linearized stability analysis via Jacobian eigenvalue computation
2. Lyapunov function construction for the coupled system
3. Stability margin computation (distance to instability boundary)
4. Maximum safe coupling strength derivation
5. Phase portrait characterization (attractors, limit cycles, chaos)

Mathematical model:
    dx_i/dt = f_i(x_i) + alpha * (x_{i+1} - x_i) + beta * (x_{i-1} - x_i)

    Where:
      x_i = state of layer i
      f_i = layer-local dynamics (decay toward prediction)
      alpha = top-down coupling (slow -> fast)
      beta = bottom-up coupling (fast -> slow)

Stability criterion (Lyapunov):
    V(x) = sum_i w_i * ||x_i - x_i*||^2
    dV/dt < 0 for all x != x* iff eigenvalues of Jacobian have negative real parts
"""
from __future__ import annotations


import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import eig, eigvals

logger = logging.getLogger("Research.TimescaleStability")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Result of stability analysis for a given coupling configuration."""
    alpha: float                     # Top-down coupling strength
    beta: float                      # Bottom-up coupling strength
    n_layers: int                    # Number of coupled layers
    dim_per_layer: int               # State dimensions per layer

    # Eigenvalue analysis
    eigenvalues: np.ndarray          # Complex eigenvalues of the Jacobian
    max_real_eigenvalue: float       # Largest real part (< 0 = stable)
    spectral_abscissa: float         # Same as max_real_eigenvalue
    spectral_radius: float           # Max absolute eigenvalue

    # Stability assessment
    is_stable: bool                  # All eigenvalues have negative real part
    stability_margin: float          # Distance from instability (positive = stable)
    convergence_rate: float          # Rate of fastest convergence mode
    slowest_mode_rate: float         # Rate of slowest convergence mode

    # Lyapunov analysis
    lyapunov_exponent: float         # Maximal Lyapunov exponent
    lyapunov_function_valid: bool    # Whether V(x) is a valid Lyapunov function

    # Coupling limits
    max_safe_alpha: float            # Maximum top-down coupling before instability
    max_safe_beta: float             # Maximum bottom-up coupling before instability

    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        return (
            f"Stability: {status} (margin={self.stability_margin:.4f}) | "
            f"alpha={self.alpha:.3f}, beta={self.beta:.3f} | "
            f"max_real_eig={self.max_real_eigenvalue:.4f} | "
            f"convergence_rate={self.convergence_rate:.4f} | "
            f"safe_alpha<={self.max_safe_alpha:.3f}, safe_beta<={self.max_safe_beta:.3f}"
        )


@dataclass
class PhasePortrait:
    """Characterization of the dynamical regime."""
    regime: str                      # "stable_node", "stable_focus", "limit_cycle", "chaos"
    attractor_dimension: int         # Topological dimension of the attractor
    oscillatory: bool                # Whether the system oscillates on approach
    damping_ratio: float             # 0 = undamped, 1 = critically damped, >1 = overdamped
    natural_frequencies: np.ndarray  # Oscillation frequencies of dominant modes
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Jacobian construction
# ---------------------------------------------------------------------------

def _build_layer_dynamics_matrix(
    n_layers: int,
    dim: int,
    decay_rates: np.ndarray,
) -> np.ndarray:
    """Build the block-diagonal matrix of layer-local dynamics.

    Each layer has dynamics: dx_i/dt = -gamma_i * x_i  (decay toward zero)
    where gamma_i is the decay rate for layer i.

    Returns:
        Block-diagonal matrix of shape (n_layers * dim, n_layers * dim)
    """
    total = n_layers * dim
    A = np.zeros((total, total), dtype=np.float64)

    for i in range(n_layers):
        start = i * dim
        end = start + dim
        A[start:end, start:end] = -decay_rates[i] * np.eye(dim)

    return A


def _build_coupling_matrix(
    n_layers: int,
    dim: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Build the coupling matrix for bidirectional timescale interactions.

    Top-down: layer i+1 (slow) influences layer i (fast) with strength alpha
    Bottom-up: layer i (fast) influences layer i+1 (slow) with strength beta

    The coupling is diffusive: influence proportional to state difference.

    Returns:
        Coupling matrix of shape (n_layers * dim, n_layers * dim)
    """
    total = n_layers * dim
    C = np.zeros((total, total), dtype=np.float64)

    for i in range(n_layers):
        start_i = i * dim
        end_i = start_i + dim

        # Top-down: from layer i+1 to layer i
        if i + 1 < n_layers:
            start_j = (i + 1) * dim
            end_j = start_j + dim
            C[start_i:end_i, start_j:end_j] += alpha * np.eye(dim)
            C[start_i:end_i, start_i:end_i] -= alpha * np.eye(dim)

        # Bottom-up: from layer i-1 to layer i
        if i - 1 >= 0:
            start_j = (i - 1) * dim
            end_j = start_j + dim
            C[start_i:end_i, start_j:end_j] += beta * np.eye(dim)
            C[start_i:end_i, start_i:end_i] -= beta * np.eye(dim)

    return C


def build_jacobian(
    n_layers: int,
    dim: int,
    alpha: float,
    beta: float,
    decay_rates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the full Jacobian of the coupled timescale system.

    J = A + C

    Where A is the layer-local dynamics and C is the inter-layer coupling.

    Args:
        n_layers: Number of temporal layers (default 5 for Aura)
        dim: State dimensions per layer
        alpha: Top-down coupling coefficient
        beta: Bottom-up coupling coefficient
        decay_rates: Per-layer decay rates (if None, uses Aura's defaults)

    Returns:
        Jacobian matrix of shape (n_layers * dim, n_layers * dim)
    """
    if decay_rates is None:
        # Aura's default decay rates (faster layers decay faster)
        # Corresponds to: reflex=20Hz, moment=1Hz, episode=1/60Hz, horizon=1/3600Hz, identity=1/86400Hz
        base_rates = np.array([0.5, 0.1, 0.01, 0.001, 0.0001])
        if n_layers <= len(base_rates):
            decay_rates = base_rates[:n_layers]
        else:
            decay_rates = np.geomspace(0.5, 0.0001, n_layers)

    A = _build_layer_dynamics_matrix(n_layers, dim, decay_rates)
    C = _build_coupling_matrix(n_layers, dim, alpha, beta)

    return A + C


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

class TimescaleStabilityAnalyzer:
    """Formal stability analysis of the cross-timescale binding system.

    Usage:
        analyzer = TimescaleStabilityAnalyzer()

        # Analyze current coupling configuration
        result = analyzer.analyze(alpha=0.15, beta=0.08)
        print(result.summary())

        # Find maximum safe coupling
        max_alpha, max_beta = analyzer.find_stability_boundary()

        # Full phase portrait
        portrait = analyzer.phase_portrait(alpha=0.15, beta=0.08)
    """

    # Aura's default configuration
    DEFAULT_N_LAYERS = 5
    DEFAULT_DIM_PER_LAYER = 8    # Matches TimescaleLayer.state dimension
    DEFAULT_DECAY_RATES = np.array([0.5, 0.1, 0.01, 0.001, 0.0001])

    def __init__(
        self,
        n_layers: int = DEFAULT_N_LAYERS,
        dim_per_layer: int = DEFAULT_DIM_PER_LAYER,
        decay_rates: Optional[np.ndarray] = None,
    ):
        self._n_layers = n_layers
        self._dim = dim_per_layer
        self._decay_rates = decay_rates if decay_rates is not None else self.DEFAULT_DECAY_RATES[:n_layers]
        self._results: List[StabilityResult] = []

    def analyze(
        self,
        alpha: float,
        beta: float,
    ) -> StabilityResult:
        """Perform full stability analysis for given coupling strengths.

        Computes:
        1. Jacobian eigenvalues (linearized stability)
        2. Stability margin (distance to instability boundary)
        3. Convergence rates (how fast the system returns to equilibrium)
        4. Lyapunov exponent (nonlinear stability indicator)
        5. Maximum safe coupling strengths

        Args:
            alpha: Top-down coupling coefficient (slow -> fast)
            beta: Bottom-up coupling coefficient (fast -> slow)

        Returns:
            StabilityResult with complete analysis
        """
        J = build_jacobian(self._n_layers, self._dim, alpha, beta, self._decay_rates)

        # Eigenvalue analysis
        evals = eigvals(J)
        real_parts = np.real(evals)
        imag_parts = np.imag(evals)

        max_real = float(np.max(real_parts))
        spectral_radius = float(np.max(np.abs(evals)))

        # Stability: all eigenvalues must have negative real part
        is_stable = max_real < -1e-10

        # Stability margin: how far from the imaginary axis is the rightmost eigenvalue
        stability_margin = -max_real  # Positive = stable, negative = unstable

        # Convergence rate: the most negative real eigenvalue (fastest decay)
        negative_reals = real_parts[real_parts < -1e-10]
        convergence_rate = float(np.min(negative_reals)) if len(negative_reals) > 0 else 0.0
        slowest_rate = float(np.max(negative_reals)) if len(negative_reals) > 0 else 0.0

        # Lyapunov exponent: for the linearized system, equals max real eigenvalue
        lyapunov_exp = max_real

        # Lyapunov function validity: V(x) = x^T P x where P solves A^T P + P A = -Q
        # For the linearized system, if all eigenvalues are negative, P exists
        lyap_valid = is_stable

        # Maximum safe coupling strengths (via bisection search)
        max_safe_alpha = self._find_max_safe_coupling(
            fixed_param="beta", fixed_value=beta,
            search_param="alpha", search_range=(0.0, 2.0),
        )
        max_safe_beta = self._find_max_safe_coupling(
            fixed_param="alpha", fixed_value=alpha,
            search_param="beta", search_range=(0.0, 2.0),
        )

        result = StabilityResult(
            alpha=alpha,
            beta=beta,
            n_layers=self._n_layers,
            dim_per_layer=self._dim,
            eigenvalues=evals,
            max_real_eigenvalue=max_real,
            spectral_abscissa=max_real,
            spectral_radius=spectral_radius,
            is_stable=is_stable,
            stability_margin=stability_margin,
            convergence_rate=convergence_rate,
            slowest_mode_rate=slowest_rate,
            lyapunov_exponent=lyapunov_exp,
            lyapunov_function_valid=lyap_valid,
            max_safe_alpha=max_safe_alpha,
            max_safe_beta=max_safe_beta,
            metadata={
                "jacobian_condition": float(np.linalg.cond(J)),
                "n_oscillatory_modes": int(np.sum(np.abs(imag_parts) > 1e-8)),
                "n_real_modes": int(np.sum(np.abs(imag_parts) <= 1e-8)),
            },
        )

        self._results.append(result)
        logger.info("Stability: %s", result.summary())
        return result

    def phase_portrait(
        self,
        alpha: float,
        beta: float,
    ) -> PhasePortrait:
        """Characterize the dynamical regime of the coupled system.

        Classifies the system's behavior based on eigenvalue structure:
        - Stable node: all eigenvalues real and negative
        - Stable focus: complex eigenvalues with negative real parts (spiraling)
        - Limit cycle: eigenvalues on the imaginary axis
        - Unstable: eigenvalues with positive real parts

        Returns:
            PhasePortrait with regime classification and oscillation frequencies
        """
        J = build_jacobian(self._n_layers, self._dim, alpha, beta, self._decay_rates)
        evals = eigvals(J)

        real_parts = np.real(evals)
        imag_parts = np.imag(evals)

        max_real = float(np.max(real_parts))
        has_imaginary = np.any(np.abs(imag_parts) > 1e-8)

        # Classification
        if max_real > 1e-6:
            regime = "unstable"
        elif has_imaginary and max_real < -1e-8:
            regime = "stable_focus"
        elif not has_imaginary and max_real < -1e-8:
            regime = "stable_node"
        elif abs(max_real) < 1e-6 and has_imaginary:
            regime = "limit_cycle"
        else:
            regime = "marginally_stable"

        # Damping ratio for dominant complex pair
        oscillatory_mask = np.abs(imag_parts) > 1e-8
        if np.any(oscillatory_mask):
            # Find the dominant (rightmost) oscillatory pair
            osc_reals = real_parts[oscillatory_mask]
            osc_imags = imag_parts[oscillatory_mask]
            dominant_idx = np.argmax(osc_reals)
            sigma = osc_reals[dominant_idx]
            omega = abs(osc_imags[dominant_idx])
            omega_n = np.sqrt(sigma ** 2 + omega ** 2)
            damping_ratio = -sigma / omega_n if omega_n > 1e-12 else 1.0
        else:
            damping_ratio = 1.0  # Overdamped (no oscillation)

        # Natural frequencies of oscillatory modes
        natural_freqs = np.abs(imag_parts[oscillatory_mask]) / (2 * np.pi) if np.any(oscillatory_mask) else np.array([])
        # Keep unique frequencies (complex conjugate pairs give the same freq)
        if len(natural_freqs) > 0:
            natural_freqs = np.unique(np.round(natural_freqs, 6))

        # Attractor dimension (number of non-zero dimensions in stable manifold)
        attractor_dim = int(np.sum(real_parts < -1e-8))

        portrait = PhasePortrait(
            regime=regime,
            attractor_dimension=attractor_dim,
            oscillatory=has_imaginary and max_real < 0,
            damping_ratio=float(np.clip(damping_ratio, 0.0, 10.0)),
            natural_frequencies=natural_freqs,
            metadata={
                "alpha": alpha,
                "beta": beta,
                "n_eigenvalues": len(evals),
                "n_positive_real": int(np.sum(real_parts > 1e-8)),
                "n_zero_real": int(np.sum(np.abs(real_parts) <= 1e-8)),
                "n_negative_real": int(np.sum(real_parts < -1e-8)),
            },
        )

        logger.info(
            "Phase portrait: regime=%s, damping=%.3f, oscillatory=%s, freqs=%s",
            portrait.regime, portrait.damping_ratio, portrait.oscillatory,
            portrait.natural_frequencies.tolist() if len(portrait.natural_frequencies) > 0 else "none",
        )
        return portrait

    def stability_map(
        self,
        alpha_range: Tuple[float, float] = (0.0, 1.0),
        beta_range: Tuple[float, float] = (0.0, 1.0),
        resolution: int = 20,
    ) -> Dict[str, Any]:
        """Compute a 2D stability map over coupling parameter space.

        For each (alpha, beta) pair, computes whether the system is stable
        and the stability margin. Returns a grid that can be visualized
        as a heatmap showing the stable region.

        Args:
            alpha_range: Range of top-down coupling to scan
            beta_range: Range of bottom-up coupling to scan
            resolution: Number of points in each dimension

        Returns:
            Dict with alpha_values, beta_values, stability_grid, margin_grid
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
        betas = np.linspace(beta_range[0], beta_range[1], resolution)

        stability_grid = np.zeros((resolution, resolution), dtype=bool)
        margin_grid = np.zeros((resolution, resolution), dtype=float)

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                J = build_jacobian(self._n_layers, self._dim, a, b, self._decay_rates)
                evals = eigvals(J)
                max_real = float(np.max(np.real(evals)))
                stability_grid[i, j] = max_real < -1e-10
                margin_grid[i, j] = -max_real

        # Find the stability boundary (contour where margin = 0)
        boundary_points = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                if stability_grid[i, j] != stability_grid[i + 1, j]:
                    boundary_points.append((float(alphas[i]), float(betas[j])))
                if stability_grid[i, j] != stability_grid[i, j + 1]:
                    boundary_points.append((float(alphas[i]), float(betas[j])))

        return {
            "alpha_values": alphas.tolist(),
            "beta_values": betas.tolist(),
            "stability_grid": stability_grid.tolist(),
            "margin_grid": margin_grid.tolist(),
            "stable_fraction": float(np.mean(stability_grid)),
            "max_margin": float(np.max(margin_grid)),
            "boundary_points": boundary_points[:50],  # Cap for readability
        }

    def _find_max_safe_coupling(
        self,
        fixed_param: str,
        fixed_value: float,
        search_param: str,
        search_range: Tuple[float, float],
        tolerance: float = 0.001,
    ) -> float:
        """Binary search for maximum coupling strength that maintains stability.

        Args:
            fixed_param: Which parameter to hold fixed ("alpha" or "beta")
            fixed_value: Value of the fixed parameter
            search_param: Which parameter to search over
            search_range: (min, max) for the search
            tolerance: Convergence tolerance

        Returns:
            Maximum safe value of search_param
        """
        lo, hi = search_range

        for _ in range(50):  # Max iterations
            mid = (lo + hi) / 2.0

            if search_param == "alpha":
                J = build_jacobian(self._n_layers, self._dim, mid, fixed_value, self._decay_rates)
            else:
                J = build_jacobian(self._n_layers, self._dim, fixed_value, mid, self._decay_rates)

            evals = eigvals(J)
            max_real = float(np.max(np.real(evals)))

            if max_real < -1e-10:
                lo = mid  # Still stable, try higher
            else:
                hi = mid  # Unstable, try lower

            if hi - lo < tolerance:
                break

        return round(lo, 4)

    def analyze_aura_defaults(self) -> StabilityResult:
        """Analyze Aura's default coupling configuration.

        Uses the actual values from CrossTimescaleBinding:
          _TOP_DOWN_COUPLING = 0.15
          _BOTTOM_UP_COUPLING = 0.08
        """
        return self.analyze(alpha=0.15, beta=0.08)

    def sensitivity_analysis(
        self,
        alpha: float,
        beta: float,
        perturbation: float = 0.01,
    ) -> Dict[str, Any]:
        """How sensitive is stability to small changes in coupling?

        Computes the gradient of the stability margin with respect to
        alpha and beta. High sensitivity means the system is near the
        stability boundary and small parameter changes could destabilize it.

        Returns:
            Dict with gradients and sensitivity scores
        """
        base = self.analyze(alpha, beta)

        # Perturb alpha
        result_a_plus = self.analyze(alpha + perturbation, beta)
        result_a_minus = self.analyze(alpha - perturbation, beta)
        d_margin_d_alpha = (result_a_plus.stability_margin - result_a_minus.stability_margin) / (2 * perturbation)

        # Perturb beta
        result_b_plus = self.analyze(alpha, beta + perturbation)
        result_b_minus = self.analyze(alpha, beta - perturbation)
        d_margin_d_beta = (result_b_plus.stability_margin - result_b_minus.stability_margin) / (2 * perturbation)

        # Overall sensitivity magnitude
        sensitivity = np.sqrt(d_margin_d_alpha ** 2 + d_margin_d_beta ** 2)

        return {
            "base_margin": base.stability_margin,
            "d_margin_d_alpha": round(d_margin_d_alpha, 6),
            "d_margin_d_beta": round(d_margin_d_beta, 6),
            "sensitivity_magnitude": round(float(sensitivity), 6),
            "most_sensitive_param": "alpha" if abs(d_margin_d_alpha) > abs(d_margin_d_beta) else "beta",
            "perturbation_used": perturbation,
            "margin_at_alpha_plus": result_a_plus.stability_margin,
            "margin_at_alpha_minus": result_a_minus.stability_margin,
            "margin_at_beta_plus": result_b_plus.stability_margin,
            "margin_at_beta_minus": result_b_minus.stability_margin,
        }

    def get_results_history(self) -> List[StabilityResult]:
        """Return all analysis results computed so far."""
        return list(self._results)
