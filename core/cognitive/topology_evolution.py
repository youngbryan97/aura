"""core/cognitive/topology_evolution.py — NEAT-Inspired Structural Plasticity

The neural mesh has 4096 neurons in 64 cortical columns.  STDP (in neural_mesh.py)
can strengthen or weaken *existing* synapses, but it cannot create new connections
or remove dead ones.  That is like a brain that can adjust synapse strength but
cannot grow new axons or prune unused pathways.

This module fills that gap.  It watches how columns co-activate over time and
proposes *structural* changes to the inter-column wiring:

  - **Synaptogenesis** (connection birth): when two unconnected columns fire
    together repeatedly, a new inter-column connection is grown.
  - **Synaptic pruning** (connection death): when a connection's weight is
    near-zero and it has not carried useful traffic in a long time, it is
    removed to free capacity for better pathways.
  - **Column specialization**: over many ticks each column develops a
    preference for certain kinds of information (language, emotion, spatial,
    etc.), mirroring how biological cortical columns specialize.
  - **Fitness tracking**: every connection accumulates a fitness score so the
    system can tell whether recent structural changes helped or hurt.
  - **Innovation protection**: brand-new connections get a grace period (50
    ticks) before they can be pruned — borrowed from the NEAT algorithm's
    idea that novel structures need time to prove their worth.

The module never writes directly into the mesh.  Instead, it returns a
``TopologyDelta`` object that *proposes* births and deaths.  The mesh (or its
bridge) decides whether to apply them.  This keeps ownership clean.

Operates entirely on numpy arrays.  Thread-safe via an internal lock.
"""
from __future__ import annotations


import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TopologyEvolution",
    "TopologyDelta",
    "TopologyMetrics",
    "TopologyConfig",
    "ConnectionRecord",
]

logger = logging.getLogger("Cognitive.TopologyEvolution")

# ---------------------------------------------------------------------------
# Tier helpers (mirror neural_mesh.py boundaries)
# ---------------------------------------------------------------------------

_SENSORY_END = 16       # columns  0..15
_ASSOCIATION_END = 48   # columns 16..47
_NUM_COLUMNS = 64       # columns 48..63 = executive

_SPECIALIZATION_LABELS: Tuple[str, ...] = (
    "language",
    "emotion",
    "spatial",
    "temporal",
    "social",
    "abstract",
    "motor",
    "self-referential",
)


def _tier_for(col: int) -> int:
    """Return tier index: 0 = sensory, 1 = association, 2 = executive."""
    if col < _SENSORY_END:
        return 0
    if col < _ASSOCIATION_END:
        return 1
    return 2


def _same_tier(a: int, b: int) -> bool:
    return _tier_for(a) == _tier_for(b)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyConfig:
    """Tuning knobs for structural plasticity.  All defaults are conservative
    enough to run on a single MacBook without destabilizing the mesh."""

    # -- Co-activation sliding window --
    correlation_window: int = 100          # ticks of history to keep
    birth_correlation_threshold: float = 0.6   # same-tier or adjacent-tier
    cross_tier_correlation_threshold: float = 0.75  # sensory <-> executive

    # -- Birth limits --
    max_births_per_tick: int = 2
    new_weight_scale: float = 0.02         # stddev of initial random weight

    # -- Pruning --
    prune_weight_threshold: float = 0.01   # |w| below this → candidate
    prune_idle_ticks: int = 100            # unused for this many ticks
    max_prunes_per_tick: int = 1
    min_inter_connectivity: float = 0.03   # never drop below 3 %

    # -- Innovation protection (NEAT heritage) --
    novelty_protection_ticks: int = 50

    # -- Specialization --
    specialization_ema_alpha: float = 0.01  # exponential moving average speed

    # -- Fitness --
    fitness_ema_alpha: float = 0.05         # how fast fitness adapts


@dataclass
class ConnectionRecord:
    """Book-keeping for one inter-column connection.

    Tracks when it was born, how much it has been used, and whether it is
    still inside its novelty-protection window.
    """
    src: int                          # source column index
    dst: int                          # destination column index
    born_tick: int                    # tick at which the connection was created
    last_used_tick: int               # last tick where traffic flowed
    cumulative_usage: float = 0.0     # running sum of |weight × activation|
    fitness: float = 0.0              # weight_mag × usage_freq × output_contribution

    @property
    def age(self) -> int:
        """Ticks since birth (must be computed relative to current tick)."""
        # Caller is responsible for passing current tick; this is a convenience
        # alias used in __repr__ only.
        return 0  # sentinel — real age computed externally

    def is_protected(self, current_tick: int, protection_ticks: int) -> bool:
        """True if the connection is still inside its novelty grace period."""
        return (current_tick - self.born_tick) < protection_ticks


@dataclass
class TopologyDelta:
    """The output of one ``evolve()`` call.

    Contains the *proposed* structural changes.  The mesh (or its bridge)
    applies them at its discretion.
    """
    births: List[Tuple[int, int, float]]   # (src, dst, initial_weight)
    deaths: List[Tuple[int, int]]          # (src, dst)
    metrics: "TopologyMetrics"
    tick: int


@dataclass
class TopologyMetrics:
    """Snapshot of network-level topology statistics."""
    connectivity_ratio: float       # existing / possible inter-column edges
    modularity: float               # 0 = uniform, 1 = strongly clustered
    small_world_coefficient: float  # clustering / mean_path_length
    tier_integration: float         # fraction of edges that cross tiers
    total_connections: int          # absolute count of live inter-column edges
    births_this_tick: int
    deaths_this_tick: int
    mean_connection_fitness: float
    topology_fitness: float         # overall score for the current wiring


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TopologyEvolution:
    """NEAT-inspired structural plasticity engine for the cortical neural mesh.

    This class watches column-level activation patterns over a sliding window,
    proposes new connections between columns that fire together, and prunes
    connections that have fallen silent.  It also tracks how columns specialize
    over time and assigns human-readable labels (e.g. "language", "emotion").

    Usage::

        topo = TopologyEvolution()

        # Each kernel tick:
        delta = await topo.evolve(column_activations, inter_column_weights, tick)
        # delta.births / delta.deaths → apply to mesh._inter_W

        # Diagnostics:
        metrics = topo.get_metrics()
        specs   = topo.get_column_specializations()

    Thread-safety: all public methods acquire an internal lock.  The ``evolve``
    coroutine is async only for interface consistency — the heavy lifting is
    pure numpy behind the lock.
    """

    def __init__(self, cfg: TopologyConfig | None = None):
        self.cfg = cfg or TopologyConfig()
        self._rng = np.random.default_rng()
        self._lock = threading.Lock()

        # -- Activation history (ring buffer) --
        # Shape: (window, columns).  Filled as ticks arrive.
        self._history = np.zeros(
            (self.cfg.correlation_window, _NUM_COLUMNS), dtype=np.float32
        )
        self._history_ptr: int = 0       # next write position in ring buffer
        self._history_filled: int = 0    # how many slots have been written

        # -- Connection registry --
        # Key: (src, dst).  Only inter-column connections are tracked here.
        self._connections: Dict[Tuple[int, int], ConnectionRecord] = {}

        # -- Column specialization profiles --
        # Each column gets an 8-d vector (one per label).  Updated via EMA.
        self._specialization = np.zeros(
            (_NUM_COLUMNS, len(_SPECIALIZATION_LABELS)), dtype=np.float32
        )

        # -- Fitness history --
        self._topology_fitness: float = 0.0
        self._fitness_history: List[float] = []

        # -- Cached metrics --
        self._last_metrics = TopologyMetrics(
            connectivity_ratio=0.0,
            modularity=0.0,
            small_world_coefficient=0.0,
            tier_integration=0.0,
            total_connections=0,
            births_this_tick=0,
            deaths_this_tick=0,
            mean_connection_fitness=0.0,
            topology_fitness=0.0,
        )

        logger.info(
            "TopologyEvolution initialized (window=%d, birth_thresh=%.2f, "
            "cross_tier_thresh=%.2f, protection=%d ticks)",
            self.cfg.correlation_window,
            self.cfg.birth_correlation_threshold,
            self.cfg.cross_tier_correlation_threshold,
            self.cfg.novelty_protection_ticks,
        )

    # =====================================================================
    # Public API
    # =====================================================================

    async def evolve(
        self,
        column_activations: np.ndarray,
        inter_column_weights: np.ndarray,
        tick: int,
    ) -> TopologyDelta:
        """Run one structural-plasticity step.

        Parameters
        ----------
        column_activations : ndarray, shape (64,)
            Mean activation of each cortical column this tick.
        inter_column_weights : ndarray, shape (64, 64)
            Current inter-column weight matrix from the neural mesh.
        tick : int
            Monotonically increasing tick counter from the kernel.

        Returns
        -------
        TopologyDelta
            Proposed births, deaths, and current topology metrics.
        """
        with self._lock:
            return self._evolve_inner(column_activations, inter_column_weights, tick)

    def get_metrics(self) -> TopologyMetrics:
        """Return the most recent topology metrics snapshot.

        Safe to call from any thread.
        """
        with self._lock:
            return self._last_metrics

    def get_column_specializations(self) -> Dict[int, str]:
        """Return each column's dominant specialization label.

        Example return value::

            {0: "spatial", 1: "emotion", 2: "language", ...}

        Columns that have not yet accumulated enough data return "unspecialized".
        """
        with self._lock:
            return self._compute_specialization_labels()

    def get_connection_records(self) -> Dict[Tuple[int, int], ConnectionRecord]:
        """Return a shallow copy of all tracked connection records."""
        with self._lock:
            return dict(self._connections)

    def get_fitness_history(self) -> List[float]:
        """Return the topology-level fitness over time.

        Each entry corresponds to one ``evolve()`` call.
        """
        with self._lock:
            return list(self._fitness_history)

    # =====================================================================
    # Core algorithm (all calls happen under self._lock)
    # =====================================================================

    def _evolve_inner(
        self,
        col_act: np.ndarray,
        weights: np.ndarray,
        tick: int,
    ) -> TopologyDelta:
        # Defensive copies / shape guards
        col_act = np.asarray(col_act, dtype=np.float32).ravel()[:_NUM_COLUMNS]
        if col_act.shape[0] < _NUM_COLUMNS:
            padded = np.zeros(_NUM_COLUMNS, dtype=np.float32)
            padded[: col_act.shape[0]] = col_act
            col_act = padded

        weights = np.asarray(weights, dtype=np.float32)[:_NUM_COLUMNS, :_NUM_COLUMNS]

        # 1. Record activation into ring buffer
        self._record_activation(col_act)

        # 2. Sync connection registry with the live weight matrix.
        #    The mesh may have connections we do not track yet (e.g. from init).
        self._sync_registry(weights, tick)

        # 3. Update usage stats for every tracked connection
        self._update_usage(weights, col_act, tick)

        # 4. Update column specialization profiles
        self._update_specialization(col_act)

        # 5. Propose births (synaptogenesis)
        births = self._propose_births(weights, tick)

        # 6. Propose deaths (pruning)
        deaths = self._propose_deaths(weights, tick)

        # 7. Update connection fitness scores
        self._update_fitness(weights, col_act, tick)

        # 8. Compute topology metrics
        metrics = self._compute_metrics(weights, len(births), len(deaths))
        self._last_metrics = metrics

        # 9. Update topology-level fitness history
        self._topology_fitness = metrics.topology_fitness
        self._fitness_history.append(self._topology_fitness)
        # Cap history length to prevent unbounded memory growth
        if len(self._fitness_history) > 10_000:
            self._fitness_history = self._fitness_history[-5_000:]

        return TopologyDelta(
            births=births,
            deaths=deaths,
            metrics=metrics,
            tick=tick,
        )

    # -----------------------------------------------------------------
    # 1. Ring-buffer activation history
    # -----------------------------------------------------------------

    def _record_activation(self, col_act: np.ndarray) -> None:
        """Store column activations into the sliding window ring buffer."""
        self._history[self._history_ptr] = col_act
        self._history_ptr = (self._history_ptr + 1) % self.cfg.correlation_window
        self._history_filled = min(self._history_filled + 1, self.cfg.correlation_window)

    def _get_correlation_matrix(self) -> np.ndarray:
        """Compute pairwise Pearson correlation between columns over the
        filled portion of the history window.

        Returns shape (64, 64) with values in [-1, 1].  Diagonal is 1.
        If fewer than 10 samples exist, returns zeros (not enough data).
        """
        n = self._history_filled
        if n < 10:
            return np.zeros((_NUM_COLUMNS, _NUM_COLUMNS), dtype=np.float32)

        # Extract the filled portion in chronological order
        if n < self.cfg.correlation_window:
            data = self._history[:n]
        else:
            # Ring buffer is full — unroll so oldest is first
            data = np.concatenate(
                [self._history[self._history_ptr:], self._history[: self._history_ptr]],
                axis=0,
            )

        # Numpy corrcoef wants (variables, observations), so transpose.
        # corrcoef can produce NaN for constant columns; replace with 0.
        corr = np.corrcoef(data.T)  # (64, 64)
        return np.nan_to_num(corr, nan=0.0).astype(np.float32)

    # -----------------------------------------------------------------
    # 2. Registry sync
    # -----------------------------------------------------------------

    def _sync_registry(self, weights: np.ndarray, tick: int) -> None:
        """Make sure every nonzero inter-column weight has a ConnectionRecord.

        Connections that exist in the weight matrix but not in our registry
        are assumed to be original mesh connections (from initialization).
        We create records for them so they participate in fitness tracking
        and pruning like everyone else.
        """
        nonzero = np.argwhere(weights != 0.0)
        existing_keys = set(self._connections.keys())

        for idx in range(nonzero.shape[0]):
            src, dst = int(nonzero[idx, 0]), int(nonzero[idx, 1])
            if src == dst:
                continue  # skip diagonal (shouldn't exist, but be safe)
            key = (src, dst)
            if key not in existing_keys:
                # Treat pre-existing mesh connections as old (born at tick 0)
                # so they are not novelty-protected.
                self._connections[key] = ConnectionRecord(
                    src=src, dst=dst, born_tick=0, last_used_tick=tick,
                )

        # Remove registry entries whose weight has already been zeroed out
        # externally (e.g. by the evolution engine).
        dead_keys = [
            k for k in self._connections
            if weights[k[0], k[1]] == 0.0
        ]
        for k in dead_keys:
            del self._connections[k]

    # -----------------------------------------------------------------
    # 3. Usage tracking
    # -----------------------------------------------------------------

    def _update_usage(
        self,
        weights: np.ndarray,
        col_act: np.ndarray,
        tick: int,
    ) -> None:
        """Update usage statistics for every tracked connection.

        A connection is considered "used" this tick if the product of its
        weight and the source column's activation exceeds a small threshold.
        This models the idea that a synapse is "active" only when current
        actually flows through it.
        """
        usage_threshold = 1e-4
        for key, rec in self._connections.items():
            src, dst = key
            traffic = abs(float(weights[src, dst]) * float(col_act[src]))
            if traffic > usage_threshold:
                rec.last_used_tick = tick
                rec.cumulative_usage += traffic

    # -----------------------------------------------------------------
    # 4. Specialization
    # -----------------------------------------------------------------

    def _update_specialization(self, col_act: np.ndarray) -> None:
        """Update each column's specialization profile via exponential moving
        average.

        The 8 specialization dimensions are mapped from the column activation
        in a biologically-inspired way: the raw activation is projected through
        a fixed random basis (seeded for reproducibility) that models the idea
        that different columns sit in different anatomical regions and therefore
        receive different mixes of input modalities.

        Over time, columns that consistently respond to one kind of pattern
        will develop a strong peak in one dimension.
        """
        if not hasattr(self, "_spec_basis"):
            # Deterministic basis so specialization labels are stable across
            # restarts.  Shape: (columns, labels).
            rng = np.random.default_rng(seed=12345)
            self._spec_basis = rng.standard_normal(
                (_NUM_COLUMNS, len(_SPECIALIZATION_LABELS))
            ).astype(np.float32)
            # Normalize rows so that the projection is bounded
            norms = np.linalg.norm(self._spec_basis, axis=1, keepdims=True)
            self._spec_basis /= np.clip(norms, 1e-6, None)

        # Project activation through the basis: each column's scalar
        # activation becomes an 8-d vector weighted by its basis row.
        projection = col_act[:, None] * self._spec_basis  # (64, 8)

        # EMA update
        alpha = self.cfg.specialization_ema_alpha
        self._specialization = (
            (1.0 - alpha) * self._specialization + alpha * projection
        ).astype(np.float32)

    def _compute_specialization_labels(self) -> Dict[int, str]:
        """Assign a human-readable label to each column based on its
        specialization profile."""
        result: Dict[int, str] = {}
        for col in range(_NUM_COLUMNS):
            profile = self._specialization[col]
            peak = float(np.max(np.abs(profile)))
            if peak < 0.005:
                result[col] = "unspecialized"
            else:
                idx = int(np.argmax(np.abs(profile)))
                result[col] = _SPECIALIZATION_LABELS[idx]
        return result

    # -----------------------------------------------------------------
    # 5. Synaptogenesis (connection birth)
    # -----------------------------------------------------------------

    def _propose_births(
        self,
        weights: np.ndarray,
        tick: int,
    ) -> List[Tuple[int, int, float]]:
        """Propose new inter-column connections based on co-activation.

        Two columns that are NOT currently connected but show high correlation
        in the sliding window become candidates for a new connection.

        Rules:
          - Same-tier or adjacent-tier pairs need correlation > 0.6.
          - Cross-tier pairs (sensory <-> executive) need correlation > 0.75.
          - At most 2 new connections per tick.
          - New connections start with small random weights.
        """
        corr = self._get_correlation_matrix()
        if corr.max() == 0.0:
            return []  # not enough data yet

        births: List[Tuple[int, int, float]] = []

        # Build mask of currently unconnected pairs (where weight == 0)
        unconnected = (weights == 0.0)
        np.fill_diagonal(unconnected, False)  # ignore self-loops

        # Determine threshold per pair
        same_thresh = self.cfg.birth_correlation_threshold
        cross_thresh = self.cfg.cross_tier_correlation_threshold

        # Score each candidate: correlation minus threshold → higher = better
        scores = np.full((_NUM_COLUMNS, _NUM_COLUMNS), -np.inf, dtype=np.float32)
        for i in range(_NUM_COLUMNS):
            ti = _tier_for(i)
            for j in range(_NUM_COLUMNS):
                if not unconnected[i, j]:
                    continue
                tj = _tier_for(j)
                # Determine whether this is a cross-tier (non-adjacent) pair
                tier_gap = abs(ti - tj)
                thresh = cross_thresh if tier_gap >= 2 else same_thresh
                c = float(corr[i, j])
                if c > thresh:
                    scores[i, j] = c - thresh  # margin above threshold

        # Pick the top candidates by score
        flat_order = np.argsort(scores.ravel())[::-1]
        for flat_idx in flat_order:
            if len(births) >= self.cfg.max_births_per_tick:
                break
            i, j = divmod(int(flat_idx), _NUM_COLUMNS)
            if scores[i, j] <= 0.0:
                break  # no more candidates above threshold

            # Assign a small random initial weight
            init_w = float(self._rng.standard_normal() * self.cfg.new_weight_scale)

            births.append((i, j, init_w))

            # Register the new connection
            self._connections[(i, j)] = ConnectionRecord(
                src=i, dst=j, born_tick=tick, last_used_tick=tick,
            )

        if births:
            logger.debug(
                "Synaptogenesis tick=%d: %d new connections proposed", tick, len(births)
            )

        return births

    # -----------------------------------------------------------------
    # 6. Synaptic pruning (connection death)
    # -----------------------------------------------------------------

    def _propose_deaths(
        self,
        weights: np.ndarray,
        tick: int,
    ) -> List[Tuple[int, int]]:
        """Propose removal of inter-column connections that are effectively dead.

        A connection is a pruning candidate if:
          - Its absolute weight is below the prune threshold (0.01).
          - It has not been used (carried meaningful traffic) in 100+ ticks.
          - It is NOT inside its novelty-protection window.
          - Removing it would not drop total inter-column connectivity below
            the minimum (3 %).

        Intra-column connections are never touched — they are structural.
        At most 1 connection is removed per tick.
        """
        deaths: List[Tuple[int, int]] = []

        # Current connectivity count
        total_existing = int(np.count_nonzero(weights)) - _NUM_COLUMNS  # minus diagonal
        # Note: diagonal *should* be zero, but be defensive
        total_existing = max(total_existing, 0)

        # Maximum possible inter-column edges (exclude diagonal)
        max_possible = _NUM_COLUMNS * (_NUM_COLUMNS - 1)
        current_ratio = total_existing / max_possible if max_possible > 0 else 0.0

        # Do not prune if already at minimum connectivity
        if current_ratio <= self.cfg.min_inter_connectivity:
            return []

        # Build list of pruning candidates sorted by fitness (worst first)
        candidates: List[Tuple[float, Tuple[int, int]]] = []
        for key, rec in self._connections.items():
            src, dst = key
            w = abs(float(weights[src, dst]))

            # Skip if weight is still significant
            if w >= self.cfg.prune_weight_threshold:
                continue

            # Skip if recently used
            idle_ticks = tick - rec.last_used_tick
            if idle_ticks < self.cfg.prune_idle_ticks:
                continue

            # Skip if novelty-protected
            if rec.is_protected(tick, self.cfg.novelty_protection_ticks):
                continue

            # Candidate.  Score by fitness (lower = worse = prune first).
            candidates.append((rec.fitness, key))

        # Sort ascending by fitness → prune the least fit first
        candidates.sort(key=lambda t: t[0])

        for _, key in candidates:
            if len(deaths) >= self.cfg.max_prunes_per_tick:
                break

            # Double-check that removing this edge won't breach minimum
            if (total_existing - len(deaths) - 1) / max_possible < self.cfg.min_inter_connectivity:
                break

            deaths.append(key)
            # Remove from registry
            self._connections.pop(key, None)

        if deaths:
            logger.debug(
                "Pruning tick=%d: %d connections proposed for removal", tick, len(deaths)
            )

        return deaths

    # -----------------------------------------------------------------
    # 7. Fitness tracking
    # -----------------------------------------------------------------

    def _update_fitness(
        self,
        weights: np.ndarray,
        col_act: np.ndarray,
        tick: int,
    ) -> None:
        """Update per-connection fitness scores.

        Fitness for a single connection is defined as:

            fitness = weight_magnitude  x  usage_frequency  x  output_contribution

        where:
          - weight_magnitude = |w|
          - usage_frequency  = cumulative_usage / age  (how often traffic flows)
          - output_contribution = |activation_dst| (proxy for whether the target
            column actually uses the signal)

        The score is smoothed with an exponential moving average so that a
        single quiet tick does not kill an otherwise healthy connection.
        """
        alpha = self.cfg.fitness_ema_alpha
        for key, rec in self._connections.items():
            src, dst = key
            w_mag = abs(float(weights[src, dst]))
            age = max(1, tick - rec.born_tick)
            usage_freq = rec.cumulative_usage / age
            output_contrib = abs(float(col_act[dst]))

            instant_fitness = w_mag * usage_freq * output_contrib
            rec.fitness = (1.0 - alpha) * rec.fitness + alpha * instant_fitness

    # -----------------------------------------------------------------
    # 8. Topology metrics
    # -----------------------------------------------------------------

    def _compute_metrics(
        self,
        weights: np.ndarray,
        births: int,
        deaths: int,
    ) -> TopologyMetrics:
        """Compute network-level topology statistics from the weight matrix."""

        # -- Basic counts --
        adj = (weights != 0.0).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        total_connections = int(adj.sum())
        max_possible = _NUM_COLUMNS * (_NUM_COLUMNS - 1)
        connectivity_ratio = total_connections / max_possible if max_possible > 0 else 0.0

        # -- Tier integration --
        cross_tier_count = 0
        for i in range(_NUM_COLUMNS):
            for j in range(_NUM_COLUMNS):
                if adj[i, j] > 0 and _tier_for(i) != _tier_for(j):
                    cross_tier_count += 1
        tier_integration = cross_tier_count / max(total_connections, 1)

        # -- Modularity (Newman-style approximation) --
        # Group columns by tier and measure fraction of within-group edges
        # vs expected under random wiring.
        modularity = self._compute_modularity(adj, total_connections)

        # -- Small-world coefficient --
        small_world = self._compute_small_world(adj)

        # -- Mean connection fitness --
        fitnesses = [rec.fitness for rec in self._connections.values()]
        mean_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0

        # -- Topology-level fitness --
        # Balanced score: we want moderate connectivity, high modularity,
        # small-world properties, and healthy individual connections.
        topology_fitness = (
            0.3 * min(connectivity_ratio / 0.05, 1.0)   # reward having some edges
            + 0.2 * modularity
            + 0.2 * min(small_world, 1.0)
            + 0.15 * tier_integration
            + 0.15 * min(mean_fitness * 100.0, 1.0)     # scale up for numerical range
        )

        return TopologyMetrics(
            connectivity_ratio=round(connectivity_ratio, 6),
            modularity=round(modularity, 4),
            small_world_coefficient=round(small_world, 4),
            tier_integration=round(tier_integration, 4),
            total_connections=total_connections,
            births_this_tick=births,
            deaths_this_tick=deaths,
            mean_connection_fitness=round(mean_fitness, 6),
            topology_fitness=round(topology_fitness, 4),
        )

    def _compute_modularity(self, adj: np.ndarray, m: int) -> float:
        """Newman modularity Q based on the three cortical tiers.

        Q = (1/2m) * sum_ij [ A_ij - (k_i * k_j) / (2m) ] * delta(c_i, c_j)

        where c_i is the tier assignment of column i.  Returns 0 if no edges.
        """
        if m == 0:
            return 0.0

        degrees = adj.sum(axis=1)  # out-degree of each column
        two_m = float(m)  # directed graph, so total edges = m
        if two_m == 0:
            return 0.0

        q = 0.0
        for i in range(_NUM_COLUMNS):
            ti = _tier_for(i)
            for j in range(_NUM_COLUMNS):
                if _tier_for(j) != ti:
                    continue
                expected = (degrees[i] * degrees[j]) / two_m
                q += float(adj[i, j]) - expected
        q /= two_m
        # Clamp to [0, 1] range for downstream consumers
        return float(np.clip(q, 0.0, 1.0))

    def _compute_small_world(self, adj: np.ndarray) -> float:
        """Approximate small-world coefficient: C / L.

        C = mean local clustering coefficient.
        L = approximate mean shortest path length (via BFS on a sample).

        For computational efficiency on a 64-node graph this is exact for C
        and sampled for L.  Returns 0.0 if the graph is disconnected or
        has fewer than 3 nodes with edges.
        """
        n = _NUM_COLUMNS

        # -- Clustering coefficient --
        # For each node, C_i = (edges among neighbors) / (k_i * (k_i - 1))
        clustering_sum = 0.0
        counted = 0
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count edges among neighbors
            sub = adj[np.ix_(neighbors, neighbors)]
            edges_among = float(sub.sum())
            clustering_sum += edges_among / (k * (k - 1))
            counted += 1

        if counted == 0:
            return 0.0

        C = clustering_sum / counted

        # -- Mean path length (BFS from 8 random source nodes) --
        sample_sources = self._rng.choice(n, size=min(8, n), replace=False)
        total_dist = 0.0
        total_pairs = 0

        for src in sample_sources:
            dist = self._bfs_distances(adj, int(src))
            reachable = dist[dist < n]  # unreachable nodes get sentinel = n
            total_dist += float(reachable.sum())
            total_pairs += len(reachable)

        if total_pairs == 0:
            return 0.0

        L = total_dist / total_pairs
        if L < 1e-6:
            return 0.0

        return float(C / L)

    @staticmethod
    def _bfs_distances(adj: np.ndarray, source: int) -> np.ndarray:
        """BFS shortest distances from ``source``.  Unreachable nodes get
        distance = n (sentinel).  Operates on the adjacency matrix directly
        — no external graph library needed."""
        n = adj.shape[0]
        dist = np.full(n, n, dtype=np.int32)
        dist[source] = 0
        frontier = [source]
        while frontier:
            next_frontier: List[int] = []
            for node in frontier:
                neighbors = np.where(adj[node] > 0)[0]
                for nb in neighbors:
                    nb_int = int(nb)
                    if dist[nb_int] == n:  # not yet visited
                        dist[nb_int] = dist[node] + 1
                        next_frontier.append(nb_int)
            frontier = next_frontier
        return dist

    # =====================================================================
    # Introspection / status
    # =====================================================================

    def get_status(self) -> Dict:
        """Return a JSON-serializable status dictionary for dashboards."""
        with self._lock:
            specs = self._compute_specialization_labels()
            # Count per specialization label
            label_counts: Dict[str, int] = {}
            for label in specs.values():
                label_counts[label] = label_counts.get(label, 0) + 1

            return {
                "total_tracked_connections": len(self._connections),
                "history_filled": self._history_filled,
                "topology_fitness": round(self._topology_fitness, 4),
                "connectivity_ratio": self._last_metrics.connectivity_ratio,
                "modularity": self._last_metrics.modularity,
                "small_world_coefficient": self._last_metrics.small_world_coefficient,
                "tier_integration": self._last_metrics.tier_integration,
                "specialization_distribution": label_counts,
                "fitness_history_length": len(self._fitness_history),
            }
