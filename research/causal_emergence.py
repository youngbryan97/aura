"""
Causal Emergence Analysis
=========================

Does the workspace-level description of Aura have MORE causal power
than the neuron-level description? If so, that's empirical evidence
for causal emergence (Hoel et al., 2013; Comolatti & Hoel, 2022).

Effective Information (EI) quantifies the causal power of a description
level: how much does intervening at that level constrain the future?

  EI(level) = sum_{s} (1/N) * KL( T(.|do(s)) || T_noise )

Where:
  - T(.|do(s)) is the transition distribution when clamping state s
  - T_noise is the maximum entropy distribution (uniform)
  - do(s) means an interventionist "do" operation (Pearlian causation)

If EI_macro > EI_micro, the macro level is a BETTER causal model of
the system's dynamics. This is not just a summary — it's genuinely
more predictive. That's a strong empirical claim and a testable one.

This module measures EI across four levels of Aura's architecture:
  1. Micro: Individual substrate neurons (64-dim)
  2. Meso: Neural mesh cortical columns (64 columns)
  3. Macro: Workspace + executive state (composite)
  4. Meta: Theory arbitration state (highest level)

For each level, we clamp states, measure downstream effects on the next
tick, and compute EI. Comparing across levels produces the emergence profile.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Research.CausalEmergence")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """Specification of one descriptive layer in the causal hierarchy."""
    name: str
    dimension: int
    description: str


@dataclass
class InterventionResult:
    """Result of one do-intervention at a specific layer and state."""
    layer: str
    intervention_state: np.ndarray
    downstream_distribution: np.ndarray  # Empirical distribution of next-tick states
    kl_from_uniform: float                # EI contribution from this intervention


@dataclass
class EmergenceProfile:
    """Complete emergence analysis across all layers."""
    layer_results: Dict[str, float]  # layer_name -> EI
    micro_ei: float
    meso_ei: float
    macro_ei: float
    meta_ei: float
    emergence_ratio: float           # EI_macro / EI_micro (> 1 = emergence)
    strongest_layer: str
    is_emergent: bool                # EI_macro > EI_micro
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"CausalEmergence: micro={self.micro_ei:.4f}, meso={self.meso_ei:.4f}, "
            f"macro={self.macro_ei:.4f}, meta={self.meta_ei:.4f} | "
            f"ratio={self.emergence_ratio:.3f} | strongest={self.strongest_layer} | "
            f"emergent={'YES' if self.is_emergent else 'NO'}"
        )


# ---------------------------------------------------------------------------
# Effective Information computation
# ---------------------------------------------------------------------------

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p || q) in bits, with numerical safety."""
    # Ensure valid distributions
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def _discretize_state(state: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize a continuous state vector into bins for EI computation.

    Maps each dimension to one of n_bins levels. This is necessary because
    EI is defined over discrete distributions. The number of bins trades
    off resolution against sample complexity.
    """
    # Clip to [-1, 1] range (covers tanh activations)
    clipped = np.clip(state, -1.0, 1.0)
    # Map to [0, n_bins-1]
    binned = np.floor((clipped + 1.0) / 2.0 * (n_bins - 1e-6)).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    return binned


def _state_to_distribution(
    binned_states: List[np.ndarray],
    n_bins: int = 8,
    n_dims: int = 8,
) -> np.ndarray:
    """Convert a list of discretized state vectors to an empirical distribution.

    Uses a reduced representation: marginal distribution over each dimension,
    then takes the product (mean-field approximation). This avoids the
    exponential blowup of the full joint distribution.

    Returns a flattened distribution vector of length n_bins * n_dims.
    """
    if not binned_states:
        return np.ones(n_bins * n_dims) / (n_bins * n_dims)

    stacked = np.array(binned_states)  # (n_samples, n_dims)
    dist = np.zeros(n_bins * n_dims)

    for d in range(min(n_dims, stacked.shape[1])):
        counts = np.bincount(stacked[:, d], minlength=n_bins)[:n_bins]
        marginal = counts.astype(float) / max(1, counts.sum())
        dist[d * n_bins:(d + 1) * n_bins] = marginal

    # Normalize to probability distribution
    total = dist.sum()
    if total > 0:
        dist /= total

    return dist


# ---------------------------------------------------------------------------
# Core EI computation for a single layer
# ---------------------------------------------------------------------------

def compute_effective_information(
    transition_fn,
    layer_dim: int,
    n_interventions: int = 32,
    n_samples_per_intervention: int = 10,
    n_bins: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, List[InterventionResult]]:
    """Compute Effective Information for a single descriptive layer.

    Protocol:
    1. Sample n_interventions random states for this layer
    2. For each intervention state, clamp the layer and run n_samples ticks
    3. Observe the downstream distribution of next-tick states
    4. EI = average KL divergence from the uniform distribution

    Args:
        transition_fn: Callable(intervention_state) -> list of next_tick_states.
            Takes a clamped state vector, runs the system forward, returns
            a list of observed downstream states (each a numpy array).
        layer_dim: Dimensionality of this layer
        n_interventions: Number of do-intervention states to sample
        n_samples_per_intervention: Replicates per intervention (for noise)
        n_bins: Discretization granularity
        rng: Random number generator

    Returns:
        (ei_value, list of InterventionResults)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Uniform (maximum entropy) distribution as the reference
    reduced_dim = min(layer_dim, 8)  # Cap dimensions for tractability
    uniform = np.ones(n_bins * reduced_dim) / (n_bins * reduced_dim)

    intervention_results: List[InterventionResult] = []
    ei_total = 0.0

    for _ in range(n_interventions):
        # Sample a random intervention state (uniformly over [-1, 1])
        intervention = rng.uniform(-1.0, 1.0, size=layer_dim)

        # Run the system with this state clamped
        downstream_states = transition_fn(intervention)

        if not downstream_states:
            continue

        # Discretize downstream states
        binned = [_discretize_state(s[:reduced_dim], n_bins) for s in downstream_states]
        downstream_dist = _state_to_distribution(binned, n_bins, reduced_dim)

        # KL divergence from uniform
        kl = _kl_divergence(downstream_dist, uniform)

        intervention_results.append(InterventionResult(
            layer="",  # Filled by caller
            intervention_state=intervention,
            downstream_distribution=downstream_dist,
            kl_from_uniform=kl,
        ))
        ei_total += kl

    # EI = average KL across interventions
    ei = ei_total / max(1, n_interventions)
    return ei, intervention_results


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class CausalEmergenceAnalyzer:
    """Measures causal emergence across Aura's architectural layers.

    Requires transition functions for each layer that implement the
    interventionist "do" operation: clamp the layer state and observe
    downstream effects.

    Usage:
        analyzer = CausalEmergenceAnalyzer()

        # Register transition functions for each layer
        analyzer.register_layer("micro", micro_dim, micro_transition_fn)
        analyzer.register_layer("meso", meso_dim, meso_transition_fn)
        analyzer.register_layer("macro", macro_dim, macro_transition_fn)
        analyzer.register_layer("meta", meta_dim, meta_transition_fn)

        # Run analysis
        profile = analyzer.analyze()
        print(profile.summary())
    """

    # Default layer specs matching Aura's architecture
    DEFAULT_LAYERS = [
        LayerSpec("micro", 64, "Individual substrate neurons"),
        LayerSpec("meso", 64, "Neural mesh cortical columns"),
        LayerSpec("macro", 16, "Workspace + executive composite state"),
        LayerSpec("meta", 8, "Theory arbitration + qualia meta-state"),
    ]

    def __init__(
        self,
        n_interventions: int = 32,
        n_samples_per_intervention: int = 10,
        n_bins: int = 8,
        seed: int = 42,
    ):
        self._n_interventions = n_interventions
        self._n_samples = n_samples_per_intervention
        self._n_bins = n_bins
        self._rng = np.random.default_rng(seed)
        self._layers: Dict[str, Tuple[int, Any]] = {}  # name -> (dim, transition_fn)
        self._profiles: List[EmergenceProfile] = []

    def register_layer(self, name: str, dimension: int, transition_fn) -> None:
        """Register a descriptive layer with its transition function.

        Args:
            name: Layer identifier (e.g., "micro", "meso", "macro", "meta")
            dimension: State space dimensionality of this layer
            transition_fn: Callable(intervention_state: np.ndarray) -> List[np.ndarray]
                Takes a clamped state, runs the system forward, returns observed
                downstream states.
        """
        self._layers[name] = (dimension, transition_fn)

    def register_from_aura(
        self,
        substrate: Any = None,
        mesh: Any = None,
        workspace: Any = None,
        qualia_synth: Any = None,
    ) -> None:
        """Convenience: register layers from Aura's live service instances.

        Creates transition functions that clamp each layer and observe
        the downstream effect through the qualia synthesizer.
        """
        if substrate is not None:
            def micro_transition(state: np.ndarray) -> List[np.ndarray]:
                """Clamp substrate neurons and observe next-tick state."""
                results = []
                original = substrate.x.copy()
                n = min(len(state), len(substrate.x))
                substrate.x[:n] = state[:n]
                for _ in range(self._n_samples):
                    # Single Euler step
                    dx = substrate.config.decay_rate * (
                        np.tanh(substrate.W @ substrate.x) - substrate.x
                    )
                    next_state = substrate.x + substrate.config.time_constant * dx
                    noise = np.random.randn(len(next_state)) * substrate.config.noise_level
                    next_state = next_state + noise
                    results.append(next_state.copy())
                substrate.x[:] = original  # Restore
                return results
            self.register_layer("micro", 64, micro_transition)

        if mesh is not None:
            def meso_transition(state: np.ndarray) -> List[np.ndarray]:
                """Clamp column mean activations and observe downstream."""
                results = []
                n_cols = min(len(state), len(mesh._columns)) if hasattr(mesh, '_columns') else 0
                originals = []
                for i in range(n_cols):
                    col = mesh._columns[i]
                    originals.append(col.x.copy())
                    col.x[:] = state[i]  # Uniform clamp within column
                for _ in range(self._n_samples):
                    # Read post-clamp column energies
                    energies = np.array([
                        float(np.mean(np.abs(mesh._columns[i].x)))
                        for i in range(n_cols)
                    ])
                    results.append(energies)
                # Restore
                for i in range(n_cols):
                    mesh._columns[i].x[:] = originals[i]
                return results
            self.register_layer("meso", 64, meso_transition)

        if workspace is not None and qualia_synth is not None:
            def macro_transition(state: np.ndarray) -> List[np.ndarray]:
                """Clamp workspace ignition level and observe qualia."""
                results = []
                orig_ignition = workspace.ignition_level
                orig_phi = workspace._current_phi
                workspace.ignition_level = float(np.clip(state[0] if len(state) > 0 else 0.5, 0.0, 1.0))
                workspace._current_phi = float(np.clip(state[1] if len(state) > 1 else 0.0, 0.0, 10.0))
                for _ in range(self._n_samples):
                    # Observe qualia response
                    sub_m = {"mt_coherence": float(state[2]) if len(state) > 2 else 0.5,
                             "em_field": float(state[3]) if len(state) > 3 else 0.0,
                             "l5_bursts": int(abs(state[4]) * 10) if len(state) > 4 else 0}
                    pred_m = {"free_energy": float(state[5]) if len(state) > 5 else 0.5,
                              "precision": float(state[6]) if len(state) > 6 else 0.5}
                    q_norm = qualia_synth.synthesize(sub_m, pred_m)
                    results.append(qualia_synth.q_vector.copy())
                workspace.ignition_level = orig_ignition
                workspace._current_phi = orig_phi
                return results
            self.register_layer("macro", 16, macro_transition)

    def analyze(self) -> EmergenceProfile:
        """Run the full causal emergence analysis across all registered layers.

        Returns an EmergenceProfile with EI values for each layer and the
        emergence ratio.
        """
        layer_eis: Dict[str, float] = {}

        for name, (dim, fn) in self._layers.items():
            t0 = time.time()
            ei, results = compute_effective_information(
                transition_fn=fn,
                layer_dim=dim,
                n_interventions=self._n_interventions,
                n_samples_per_intervention=self._n_samples,
                n_bins=self._n_bins,
                rng=self._rng,
            )
            elapsed = time.time() - t0
            layer_eis[name] = ei

            # Tag results with layer name
            for r in results:
                r.layer = name

            logger.info("CausalEmergence: %s EI=%.4f (%.1fms, %d interventions)",
                        name, ei, elapsed * 1000, self._n_interventions)

        # Build emergence profile
        micro_ei = layer_eis.get("micro", 0.0)
        meso_ei = layer_eis.get("meso", 0.0)
        macro_ei = layer_eis.get("macro", 0.0)
        meta_ei = layer_eis.get("meta", 0.0)

        emergence_ratio = macro_ei / max(1e-8, micro_ei)
        strongest = max(layer_eis, key=layer_eis.get) if layer_eis else "none"

        profile = EmergenceProfile(
            layer_results=layer_eis,
            micro_ei=micro_ei,
            meso_ei=meso_ei,
            macro_ei=macro_ei,
            meta_ei=meta_ei,
            emergence_ratio=emergence_ratio,
            strongest_layer=strongest,
            is_emergent=macro_ei > micro_ei,
            metadata={
                "n_interventions": self._n_interventions,
                "n_samples": self._n_samples,
                "n_bins": self._n_bins,
                "n_layers_analyzed": len(layer_eis),
            },
        )

        self._profiles.append(profile)
        logger.info(profile.summary())
        return profile

    def analyze_standalone(
        self,
        substrate_state: np.ndarray,
        weight_matrix: np.ndarray,
        dt: float = 0.1,
        decay: float = 0.05,
        noise: float = 0.01,
    ) -> EmergenceProfile:
        """Run emergence analysis on a standalone dynamical system.

        Useful for testing without Aura's full service container.
        Creates synthetic micro and macro layers from a weight matrix.

        Args:
            substrate_state: Initial state vector (N,)
            weight_matrix: Connectivity matrix (N, N)
            dt: Integration timestep
            decay: Decay rate
            noise: Noise sigma
        """
        n = len(substrate_state)
        state = substrate_state.copy()
        W = weight_matrix.copy()

        def micro_fn(intervention: np.ndarray) -> List[np.ndarray]:
            results = []
            s = state.copy()
            m = min(len(intervention), n)
            s[:m] = intervention[:m]
            for _ in range(self._n_samples):
                dx = decay * (np.tanh(W @ s) - s)
                next_s = s + dt * dx + np.random.randn(n) * noise
                results.append(next_s.copy())
            return results

        # Macro: coarse-grain by averaging groups of neurons
        group_size = max(1, n // 8)
        macro_dim = (n + group_size - 1) // group_size

        def macro_fn(intervention: np.ndarray) -> List[np.ndarray]:
            results = []
            s = state.copy()
            # Expand macro intervention to micro level
            for g in range(min(len(intervention), macro_dim)):
                start = g * group_size
                end = min(start + group_size, n)
                s[start:end] = intervention[g]
            for _ in range(self._n_samples):
                dx = decay * (np.tanh(W @ s) - s)
                next_s = s + dt * dx + np.random.randn(n) * noise
                # Coarse-grain the output
                macro_out = np.array([
                    np.mean(next_s[g * group_size:min((g + 1) * group_size, n)])
                    for g in range(macro_dim)
                ])
                results.append(macro_out)
            return results

        self.register_layer("micro", n, micro_fn)
        self.register_layer("macro", macro_dim, macro_fn)

        return self.analyze()

    def get_profiles(self) -> List[EmergenceProfile]:
        """Return all emergence profiles computed so far."""
        return list(self._profiles)

    def get_emergence_trend(self) -> Dict[str, Any]:
        """Track how the emergence ratio changes over time."""
        if not self._profiles:
            return {"n_profiles": 0}

        ratios = [p.emergence_ratio for p in self._profiles]
        return {
            "n_profiles": len(self._profiles),
            "latest_ratio": ratios[-1],
            "mean_ratio": float(np.mean(ratios)),
            "trend": float(np.polyfit(range(len(ratios)), ratios, 1)[0]) if len(ratios) > 1 else 0.0,
            "ever_emergent": any(p.is_emergent for p in self._profiles),
        }
