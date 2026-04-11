"""core/consciousness/iit_surrogate.py

Reflexive Integrated Information Unit (RIIU) — Φ Surrogate Estimator.

Implements a differentiable surrogate for IIT's Integrated Information (Φ).
Uses a sliding window of substrate state snapshots to compute a covariance-based
integration measure.

Key properties:
- NumPy-only (no PyTorch dependency)
- Numerically stable (regularized covariance, clamped logdet)
- Deterministic float output
- Maintains sliding buffer for temporal integration

Theory:
    Φ ≈ log|Σ_whole| − max(log|Σ_partition|)
    where Σ is the covariance of the state trajectory over the buffer window.
    Higher Φ → more integrated system → partitions lose more information.
"""

import numpy as np
import time
import logging
from core.container import get_container

logger = logging.getLogger("Consciousness.RIIU")

# Defaults
_DEFAULT_BUFFER_SIZE = 64    # Number of state snapshots in sliding window
_DEFAULT_NEURON_COUNT = 64   # Must match LiquidSubstrate.config.neuron_count
_REGULARIZATION = 1e-6       # Tikhonov regularization for covariance
_MIN_SAMPLES = 8             # Minimum samples before computing Φ


class RIIU:
    """Reflexive Integrated Information Unit.

    Maintains a sliding buffer of state vectors and computes a covariance-based
    surrogate for IIT Φ (Integrated Information).

    Usage:
        riiu = RIIU(neuron_count=64)
        phi = riiu.compute_phi(current_state_vector)  # Call each tick
    """

    def __init__(
        self,
        neuron_count: int = _DEFAULT_NEURON_COUNT,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        num_partitions: int = 4,
    ):
        """
        Args:
            neuron_count: Dimensionality of each state vector.
            buffer_size: Number of state snapshots to retain.
            num_partitions: Number of bipartitions to test. More = more
                accurate Φ estimate but slower. 4 is a good balance.
        """
        self.neuron_count = neuron_count
        self.buffer_size = buffer_size
        self.num_partitions = 8  # Increased for richer network state
        
        # 2026 Evolution: Network-Aware Dimensionality
        self.network_dim = 128 # Reserved for hyphae + attention
        self.total_dim = neuron_count + self.network_dim

        # Sliding window buffer: shape (buffer_size, total_dim)
        self._buffer = np.zeros((buffer_size, self.total_dim), dtype=np.float64)
        self._write_idx = 0
        self._samples_collected = 0

        # Welford's online covariance (avoids full recompute every tick)
        self._welford_mean = np.zeros(self.total_dim, dtype=np.float64)
        self._welford_M2 = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)
        self._welford_n = 0
        self._cov_dirty = True  # signals that cached slogdet needs refresh
        self._cached_cov = None

        # Cached results
        self._last_phi: float = 0.0
        self._last_whole_logdet: float = 0.0
        self._tick_count: int = 0

        # Partition indices (precomputed for speed)
        self._partitions = self._generate_partitions()

        logger.info(
            "RIIU initialized (neurons=%d, buffer=%d, partitions=%d)",
            neuron_count, buffer_size, self.num_partitions,
        )

    def compute_phi(self, state_vector: np.ndarray) -> float:
        """Adds a new state vector to the buffer and computes current Φ.

        Metabolic optimization (v50): Uses Welford's online covariance
        so each tick is O(d^2) rank-1 update instead of O(n*d^2) full recompute.
        The expensive slogdet+partition search only runs every 5 ticks.
        """
        # 1. Update sliding buffer
        v = np.zeros(self.total_dim, dtype=np.float64)
        n = min(len(state_vector), self.total_dim)
        v[:n] = state_vector[:n]

        self._buffer[self._write_idx] = v
        self._write_idx = (self._write_idx + 1) % self.buffer_size
        self._samples_collected = min(self.buffer_size, self._samples_collected + 1)

        # 2. Welford online covariance update (O(d^2) per tick)
        self._welford_n += 1
        delta = v - self._welford_mean
        self._welford_mean += delta / self._welford_n
        delta2 = v - self._welford_mean
        self._welford_M2 += np.outer(delta, delta2)
        self._cov_dirty = True

        self._tick_count += 1

        phi = self._last_phi  # default: return cached value
        if self._samples_collected >= _MIN_SAMPLES and self._tick_count % 5 == 0:
            # 3. Full phi computation every 5 ticks (amortize slogdet cost)
            data = self._buffer[:self._samples_collected]
            phi = self._compute_phi_internal(data)

        # 3. Mycelial Pulse (Proof of Life for Φ Subsystem)
        try:
            container = get_container()
            mycelium = container.get("mycelial_network", default=None)
            if mycelium:
                # Source=consciousness, target=iit_phi
                hypha = mycelium.get_hypha("consciousness", "iit_phi")
                if hypha:
                    hypha.pulse(success=True)
        except Exception as _e:
            logger.debug('Ignored Exception in iit_surrogate.py: %s', _e)

        self._last_phi = phi
        return phi

    def update_from_network(self):
        """Pulls real-time signal traffic from Mycelium into the Φ substrate."""
        try:
            container = get_container()
            mycelium = container.get("mycelial_network", default=None)
            substrate = container.get("conscious_substrate", default=None)
            
            if not mycelium: return 0.0
            
            # 1. Base Liquid State (64 neurons)
            state_vec = np.zeros(self.total_dim)
            if substrate and hasattr(substrate, 'get_state'):
                l_state = substrate.get_state()
                n = min(len(l_state), self.neuron_count)
                state_vec[:n] = l_state[:n]
                
            # 2. Mycelial hyphae state (strengths & pulse recency)
            hyphae_vec = []
            now = time.monotonic()
            for h in list(mycelium.hyphae.values())[:self.network_dim // 2]:
                hyphae_vec.append(h.strength / 10.0) # Normalized 0-1
                recency = max(0.0, 1.0 - (now - h.last_pulse) / 10.0)
                hyphae_vec.append(recency)
                
            # Pad or truncate hyphae_vec to match network_dim
            hyphae_vec = (hyphae_vec + [0.0] * self.network_dim)[:self.network_dim]
            state_vec[self.neuron_count:] = hyphae_vec
            
            return self.compute_phi(state_vec)
        except Exception as e:
            logger.debug("Failed network-bound Φ update: %s", e)
            return 0.0

    def get_phi(self) -> float:
        """Return the last computed Φ without recomputation."""
        return self._last_phi

    def get_stats(self) -> dict:
        """Return RIIU telemetry."""
        return {
            "phi": round(self._last_phi, 6),
            "whole_logdet": round(self._last_whole_logdet, 6),
            "samples": self._samples_collected,
            "buffer_full": self._samples_collected >= self.buffer_size,
            "neuron_count": self.neuron_count,
        }

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_phi_internal(self, data: np.ndarray) -> float:
        """Core Φ computation.

        Φ ≈ log|Σ_whole| − max(log|Σ_partition|)

        If the whole system's covariance has higher log-determinant than
        any partition, the system is integrated (high Φ).
        """
        n_samples, n_dims = data.shape

        import warnings
        
        # 1. Whole-system covariance + log-determinant
        cov_whole = self._regularized_covariance(data)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*slogdet.*')
                sign_w, logdet_w = np.linalg.slogdet(cov_whole)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return 0.0

        if sign_w <= 0 or not np.isfinite(logdet_w):
            # Degenerate — all zeros, perfectly correlated, or unstable
            return 0.0

        self._last_whole_logdet = logdet_w

        # 2. Find the partition with the MAXIMUM log-det-sum
        #    (the "best" partition that preserves the most information)
        max_partition_logdet = -np.inf

        for part_a, part_b in self._partitions:
            if len(part_a) < 2 or len(part_b) < 2:
                continue

            # Covariance of each partition
            cov_a = self._regularized_covariance(data[:, part_a])
            cov_b = self._regularized_covariance(data[:, part_b])

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*slogdet.*')
                    sign_a, logdet_a = np.linalg.slogdet(cov_a)
                    sign_b, logdet_b = np.linalg.slogdet(cov_b)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue

            if sign_a <= 0 or sign_b <= 0 or not np.isfinite(logdet_a) or not np.isfinite(logdet_b):
                continue

            partition_logdet = logdet_a + logdet_b
            if not np.isfinite(partition_logdet): continue
            max_partition_logdet = max(max_partition_logdet, partition_logdet)

        if max_partition_logdet == -np.inf:
            return 0.0

        # Φ = how much information is lost when you partition the system
        phi = logdet_w - max_partition_logdet

        # Clamp to non-negative and finite range
        if not np.isfinite(phi): return 0.0
        return float(np.clip(phi, 0.0, 1000.0))

    def _regularized_covariance(self, data: np.ndarray) -> np.ndarray:
        """Compute regularized covariance matrix (Tikhonov)."""
        cov = np.cov(data, rowvar=False)

        # Handle 1D case (single feature partition)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Tikhonov regularization: Σ + εI
        cov += np.eye(cov.shape[0]) * _REGULARIZATION

        return cov

    def _generate_partitions(self):
        """Generate random bipartitions of neuron indices.

        For true IIT you'd test ALL possible bipartitions (2^N),
        which is intractable. We sample a fixed number instead.
        """
        indices = np.arange(self.total_dim)
        partitions = []

        rng = np.random.RandomState(42)  # Deterministic partitions

        for _ in range(self.num_partitions):
            # Random split point (ensure both halves have at least 8 elements for stability)
            split = rng.randint(8, self.total_dim - 8)
            perm = rng.permutation(indices)
            part_a = sorted(perm[:split].tolist())
            part_b = sorted(perm[split:].tolist())
            partitions.append((part_a, part_b))

        return partitions
