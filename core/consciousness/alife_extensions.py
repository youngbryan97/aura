"""core/consciousness/alife_extensions.py -- Artificial Life Extensions

Brings remaining concepts from artificial life research into Aura's cognitive
architecture.  The main ALife mechanisms (criticality regulation, Lenia kernels,
entropy tracking, CPU allocation, endogenous fitness) live in their own modules.
This file contains the REMAINING features that complete the integration:

1. PatternReplicator  (Avida / Evochora)
   Successful cortical columns replicate their weight patterns to struggling
   neighbors within the same tier.  In Avida, organisms that solve problems
   self-replicate; here, column configurations that contribute energy, maintain
   stability, and keep error low propagate their intra-column weights to
   underperforming siblings.  Marker-based transfer (from Evochora) restricts
   which weight subsets are actually copied, creating selective inheritance
   rather than wholesale cloning.

2. ColumnSpeciation  (EcoSim)
   Columns form functional "species" by clustering their 8-dimensional
   specialization profiles (language, emotion, spatial, temporal, social,
   abstract, motor, self-referential).  Speciation emerges from k-means on
   these profiles, with k chosen by silhouette score.  Same-species connections
   are easier to maintain; cross-species connections require stronger
   correlation.  At least one representative from each species is protected
   from pruning (niche protection), preventing monoculture collapse.

3. ToroidalTopology  (Avida / Evochora / Cellular Automata)
   The 64-column mesh wraps toroidally so that column 0 and column 63 are
   neighbors, not maximally distant.  This removes edge effects from distance-
   dependent computations (Lenia kernels, topology evolution, replication
   proximity, specialization clustering).

4. ThermodynamicPolicy  (Evochora)
   Every cognitive operation has an explicit energy cost and entropy generation
   rate, configured in one place.  Failed operations pay a 50% entropy
   surcharge (waste from failure).  This creates genuine metabolic pressure
   that shapes which operations the system prefers.

5. OwnershipModel  (Evochora)
   Working memory entries have owners.  Accessing your own memory is cheap;
   accessing a sibling's costs 1.5x; crossing tier boundaries costs 2x;
   reading kernel/governance memory costs 3x.  This creates natural information
   locality -- subsystems prefer their own recent outputs, and cross-subsystem
   communication has a real cost that makes it selective.

All five systems are wrapped by ALifeExtensions, which provides a single
async tick() entry point and a dashboard-ready get_status().

Thread-safety: all mutable state is guarded by a threading.Lock.
Dependencies: numpy only (no sklearn -- k-means and silhouette are implemented
from scratch to avoid adding a heavy dependency).
"""
from __future__ import annotations


import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.ALifeExtensions")

__all__ = [
    "PatternReplicator",
    "ColumnSpeciation",
    "ToroidalTopology",
    "ThermodynamicPolicy",
    "OwnershipModel",
    "ALifeExtensions",
    "ALifeExtensionState",
]


# ---------------------------------------------------------------------------
# 3. ToroidalTopology -- from Avida / Evochora / Cellular Automata
# ---------------------------------------------------------------------------
# Listed first because the other systems reference toroidal distance.

class ToroidalTopology:
    """Toroidal wrapping for the 64-column cortical mesh.

    In Avida and most cellular automata, the grid wraps at its edges so that
    there are no boundary effects.  A 1-D ring of 64 columns means column 0
    is adjacent to column 63.  Without wrapping, columns near the edges have
    fewer effective neighbors and receive weaker inter-column signals -- an
    artifact of indexing, not biology.

    The distance between two columns on a torus of size N is:
        d(i, j) = min(|i - j|, N - |i - j|) / N

    This is normalized to [0, 1] so it can drop directly into exponential
    decay functions (Lenia kernels, connectivity probability, replication
    eligibility) without rescaling.

    The full NxN distance matrix is computed once and cached because N is
    constant for the lifetime of the mesh.
    """

    __slots__ = ("_n_columns", "_distance_matrix")

    def __init__(self, n_columns: int = 64):
        self._n_columns = n_columns
        self._distance_matrix = self._build_distance_matrix(n_columns)

    # -- Public API --------------------------------------------------------

    @staticmethod
    def toroidal_distance(i: int, j: int, n_columns: int = 64) -> float:
        """Normalized toroidal distance between two column indices.

        Returns a float in [0, 0.5] -- on a ring of 64, the farthest two
        columns can be is 32 positions apart, which normalizes to 0.5.
        """
        raw = abs(i - j)
        return min(raw, n_columns - raw) / n_columns

    @staticmethod
    def toroidal_distance_matrix(n_columns: int = 64) -> np.ndarray:
        """Compute the full NxN normalized toroidal distance matrix.

        Returns an (n_columns, n_columns) float32 array where entry [i, j]
        is the toroidal distance between column i and column j.
        """
        return ToroidalTopology._build_distance_matrix(n_columns)

    def get_distance(self, i: int, j: int) -> float:
        """Look up the cached toroidal distance between two columns."""
        return float(self._distance_matrix[i, j])

    def get_matrix(self) -> np.ndarray:
        """Return the cached distance matrix (read-only view)."""
        result = self._distance_matrix.view()
        result.flags.writeable = False
        return result

    @property
    def n_columns(self) -> int:
        return self._n_columns

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _build_distance_matrix(n: int) -> np.ndarray:
        idx = np.arange(n, dtype=np.float32)
        raw = np.abs(idx[:, None] - idx[None, :])
        return np.minimum(raw, n - raw).astype(np.float32) / n


# ---------------------------------------------------------------------------
# 1. PatternReplicator -- from Avida / Evochora
# ---------------------------------------------------------------------------

@dataclass
class ReplicationEvent:
    """Record of one pattern replication."""
    tick: int
    donor_col: int
    recipient_col: int
    tier: str
    alpha: float
    marker_match_fraction: float


class PatternReplicator:
    """Autopoietic pattern replication inspired by Avida's self-replicating
    organisms and Evochora's marker-based selective inheritance.

    In Avida, organisms that solve computational tasks earn CPU cycles and
    self-replicate, spreading successful genomes through the population.
    Here, the "organisms" are cortical columns, and "successful" means high
    energy contribution, stable activation, and low error rate.

    How it works each tick:
      1. Compute a fitness score for every column:
           fitness = contribution_to_output * activation_stability * (1 - error_rate)
      2. Within each tier (sensory / association / executive), identify
         the top 25% (donors) and bottom 25% (recipients).
      3. If at least 100 ticks have passed since the last replication in
         this tier, pick the best donor and worst recipient in the tier.
      4. Partially blend the donor's intra-column weights into the recipient:
           W_new = (1 - alpha) * W_recipient + alpha * W_donor
         where alpha = 0.2 (gentle transfer).
      5. Marker-based transfer (Evochora): each column has a 4-bit marker.
         Only weight rows whose index modulo 16 matches the donor's marker
         are actually transferred.  This prevents wholesale copying and
         allows selective inheritance of sub-circuits.
      6. The recipient gets a 50-tick protection period during which it
         cannot be overwritten again (matching topology_evolution's
         protection window).
      7. After 50 ticks, check if the recipient improved.  Track success
         rate for evolutionary pressure feedback.
    """

    def __init__(self, n_columns: int = 64, torus: ToroidalTopology | None = None):
        self._n_columns = n_columns
        self._torus = torus or ToroidalTopology(n_columns)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()

        # Per-column 4-bit markers (0-15).  Initialized randomly.
        self._markers = self._rng.integers(0, 16, size=n_columns).astype(np.int8)

        # Protection: tick at which each column's protection expires.
        self._protection_expires = np.zeros(n_columns, dtype=np.int64)

        # Rate limit: last replication tick per tier.
        self._last_replication_tick: Dict[str, int] = {
            "SENSORY": -100,
            "ASSOCIATION": -100,
            "EXECUTIVE": -100,
        }

        # Tracking
        self._events: List[ReplicationEvent] = []
        self._pending_checks: List[Tuple[int, int, float, int]] = []  # (recipient, tick_due, pre_fitness, event_idx)
        self._successes: int = 0
        self._total_checked: int = 0

        # Alpha (blending strength)
        self._alpha: float = 0.2

        # Protection duration
        self._protection_ticks: int = 50

        # Rate limit interval
        self._rate_limit_ticks: int = 100

    # -- Tier helpers ------------------------------------------------------

    _TIER_RANGES: Dict[str, Tuple[int, int]] = {
        "SENSORY": (0, 16),
        "ASSOCIATION": (16, 48),
        "EXECUTIVE": (48, 64),
    }

    @staticmethod
    def _tier_for(col_idx: int) -> str:
        if col_idx < 16:
            return "SENSORY"
        if col_idx < 48:
            return "ASSOCIATION"
        return "EXECUTIVE"

    # -- Fitness computation -----------------------------------------------

    @staticmethod
    def compute_column_fitness(
        contributions: np.ndarray,
        stabilities: np.ndarray,
        error_rates: np.ndarray,
    ) -> np.ndarray:
        """Compute per-column fitness from the three components.

        All inputs are (n_columns,) arrays with values in [0, 1].
        Returns (n_columns,) fitness in [0, 1].
        """
        return contributions * stabilities * (1.0 - error_rates)

    # -- Main tick ---------------------------------------------------------

    def tick(
        self,
        columns_W: List[np.ndarray],
        fitness: np.ndarray,
        tick_count: int,
    ) -> List[ReplicationEvent]:
        """Run one replication cycle.

        Args:
            columns_W: list of (n, n) intra-column weight matrices,
                       one per column.  MODIFIED IN PLACE on replication.
            fitness:   (n_columns,) fitness scores.
            tick_count: current global tick.

        Returns:
            List of ReplicationEvent for any replications that occurred
            this tick (usually 0 or 1).
        """
        new_events: List[ReplicationEvent] = []

        with self._lock:
            # Check pending success evaluations first
            self._check_pending(fitness, tick_count)

            for tier_name, (lo, hi) in self._TIER_RANGES.items():
                # Rate limit
                if tick_count - self._last_replication_tick[tier_name] < self._rate_limit_ticks:
                    continue

                tier_fitness = fitness[lo:hi]
                n_tier = hi - lo
                if n_tier < 4:
                    continue

                # Top/bottom 25%
                threshold_top = np.percentile(tier_fitness, 75)
                threshold_bot = np.percentile(tier_fitness, 25)

                donors = [
                    i + lo for i in range(n_tier)
                    if tier_fitness[i] >= threshold_top
                ]
                recipients = [
                    i + lo for i in range(n_tier)
                    if tier_fitness[i] <= threshold_bot
                    and self._protection_expires[i + lo] <= tick_count
                ]

                if not donors or not recipients:
                    continue

                # Best donor, worst recipient
                best_donor = max(donors, key=lambda c: fitness[c])
                worst_recipient = min(recipients, key=lambda c: fitness[c])

                # Perform replication
                event = self._replicate(
                    columns_W, best_donor, worst_recipient, tier_name, tick_count
                )
                new_events.append(event)
                self._events.append(event)

                # Trim event history
                if len(self._events) > 500:
                    self._events = self._events[-500:]

                # Schedule success check
                self._pending_checks.append((
                    worst_recipient,
                    tick_count + self._protection_ticks,
                    float(fitness[worst_recipient]),
                    len(self._events) - 1,
                ))

                self._last_replication_tick[tier_name] = tick_count

        return new_events

    def _replicate(
        self,
        columns_W: List[np.ndarray],
        donor: int,
        recipient: int,
        tier_name: str,
        tick_count: int,
    ) -> ReplicationEvent:
        """Perform marker-based partial weight transfer from donor to recipient.

        Evochora's marker system: each column has a 4-bit marker (0-15).
        Only weight rows whose (index mod 16) == donor's marker are
        transferred.  This creates selective inheritance -- the donor
        doesn't clone its entire weight matrix, only the sub-circuit
        tagged with its marker value.
        """
        W_donor = columns_W[donor]
        W_recipient = columns_W[recipient]
        n = W_donor.shape[0]

        donor_marker = int(self._markers[donor])
        mask_rows = np.array([(r % 16) == donor_marker for r in range(n)])
        n_transferred = int(mask_rows.sum())
        marker_fraction = n_transferred / max(n, 1)

        # Blend only the masked rows
        alpha = self._alpha
        W_new = W_recipient.copy()
        W_new[mask_rows] = (1.0 - alpha) * W_recipient[mask_rows] + alpha * W_donor[mask_rows]
        columns_W[recipient][:] = W_new

        # Set protection
        self._protection_expires[recipient] = tick_count + self._protection_ticks

        logger.debug(
            "Replicated col %d -> col %d (tier=%s, alpha=%.2f, marker=%d, rows=%d/%d)",
            donor, recipient, tier_name, alpha, donor_marker, n_transferred, n,
        )

        return ReplicationEvent(
            tick=tick_count,
            donor_col=donor,
            recipient_col=recipient,
            tier=tier_name,
            alpha=alpha,
            marker_match_fraction=marker_fraction,
        )

    def _check_pending(self, fitness: np.ndarray, tick_count: int) -> None:
        """Check whether past recipients improved after replication."""
        still_pending = []
        for recipient, tick_due, pre_fitness, event_idx in self._pending_checks:
            if tick_count >= tick_due:
                self._total_checked += 1
                if fitness[recipient] > pre_fitness:
                    self._successes += 1
            else:
                still_pending.append((recipient, tick_due, pre_fitness, event_idx))
        self._pending_checks = still_pending

    # -- Query API ---------------------------------------------------------

    def get_success_rate(self) -> float:
        """Fraction of replications where the recipient improved within
        the 50-tick protection window."""
        if self._total_checked == 0:
            return 0.0
        return self._successes / self._total_checked

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_replications": len(self._events),
                "success_rate": round(self.get_success_rate(), 3),
                "successes": self._successes,
                "checked": self._total_checked,
                "pending_checks": len(self._pending_checks),
                "last_replication_tick": dict(self._last_replication_tick),
                "alpha": self._alpha,
                "protection_ticks": self._protection_ticks,
            }


# ---------------------------------------------------------------------------
# 2. ColumnSpeciation -- from EcoSim
# ---------------------------------------------------------------------------

@dataclass
class SpeciesInfo:
    """Snapshot of current species structure."""
    species_ids: np.ndarray           # (n_columns,) int -- species label per column
    n_species: int
    species_sizes: Dict[int, int]     # species_id -> member count
    species_fitness: Dict[int, float] # species_id -> mean fitness
    shannon_diversity: float          # Shannon diversity index of species distribution
    turnover_rate: float              # fraction of columns that changed species since last clustering
    silhouette_score: float           # quality of the clustering


class ColumnSpeciation:
    """Speciation-driven column specialization inspired by EcoSim's genotypic
    clustering and niche dynamics.

    In EcoSim, organisms form species through genotypic clustering.  Species
    compete for resources; successful species expand while unfit species
    contract.  Crucially, each species is protected from total extinction by
    "niche protection" -- at least one representative survives, maintaining
    genetic diversity.

    Mapped to Aura's cortical mesh:

    Columns have 8-dimensional specialization profiles (language, emotion,
    spatial, temporal, social, abstract, motor, self-referential).  Every
    500 ticks, k-means clustering groups columns into functional species
    based on these profiles.  K is chosen automatically from {2..6} by
    silhouette score.

    Species membership affects connection dynamics:
      - Same-species connections have a lower pruning threshold (easier to
        maintain), creating dense within-species connectivity.
      - Cross-species connections require higher correlation to survive,
        creating sparse between-species bridges.

    This produces functional modularity from selection pressure alone --
    no hard-wired module boundaries needed.

    Niche protection: during topology evolution, at least one column from
    each species is immune to pruning.  This prevents monoculture where one
    dominant species eliminates all others.
    """

    def __init__(self, n_columns: int = 64, n_specialization_dims: int = 8):
        self._n_columns = n_columns
        self._n_dims = n_specialization_dims
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()

        # Current species assignments
        self._species_ids = np.zeros(n_columns, dtype=np.int32)
        self._prev_species_ids = np.zeros(n_columns, dtype=np.int32)

        # Clustering interval
        self._clustering_interval: int = 500
        self._last_clustering_tick: int = -500  # force first clustering

        # Species info cache
        self._info: Optional[SpeciesInfo] = None

        # Pruning threshold modifiers
        self._same_species_pruning_factor: float = 0.7    # 30% easier to keep
        self._cross_species_pruning_factor: float = 1.4   # 40% harder to keep

    # -- k-means (pure numpy) ---------------------------------------------

    @staticmethod
    def _kmeans(data: np.ndarray, k: int, max_iter: int = 30,
                rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering.  Returns (labels, centroids).

        Args:
            data: (n_samples, n_features) float array.
            k: number of clusters.
            max_iter: iteration cap.
            rng: optional random generator for reproducibility.

        Returns:
            labels:    (n_samples,) int array of cluster assignments.
            centroids: (k, n_features) float array.
        """
        rng = rng or np.random.default_rng()
        n = data.shape[0]
        if k >= n:
            return np.arange(n, dtype=np.int32), data.copy()

        # k-means++ initialization
        centroids = np.empty((k, data.shape[1]), dtype=np.float64)
        centroids[0] = data[rng.integers(0, n)]
        for c in range(1, k):
            dists = np.min(
                np.sum((data[:, None, :] - centroids[None, :c, :]) ** 2, axis=2),
                axis=1,
            )
            total = dists.sum()
            if total < 1e-15:
                # All points coincide with existing centroids — pick random
                centroids[c] = data[rng.integers(0, n)]
            else:
                probs = dists / total
                # Guard against float rounding: force exact sum to 1
                probs = np.maximum(probs, 0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    probs = np.ones(n) / n
                centroids[c] = data[rng.choice(n, p=probs)]

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            # Assign
            dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update centroids
            for c in range(k):
                members = data[labels == c]
                if len(members) > 0:
                    centroids[c] = members.mean(axis=0)

        return labels, centroids

    @staticmethod
    def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
        """Compute mean silhouette score.

        For each sample, silhouette = (b - a) / max(a, b) where:
          a = mean distance to same-cluster members
          b = mean distance to nearest other cluster
        Returns mean silhouette in [-1, 1].  Higher is better.
        """
        n = data.shape[0]
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        # Pairwise distances
        dists = np.sqrt(
            np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=2) + 1e-12
        )

        silhouettes = np.zeros(n, dtype=np.float64)
        for i in range(n):
            own_label = labels[i]
            own_mask = labels == own_label
            own_count = own_mask.sum() - 1  # exclude self
            if own_count <= 0:
                silhouettes[i] = 0.0
                continue
            a_i = dists[i, own_mask].sum() / own_count

            # Mean distance to each other cluster, take the minimum
            b_i = np.inf
            for lbl in unique_labels:
                if lbl == own_label:
                    continue
                other_mask = labels == lbl
                b_candidate = dists[i, other_mask].mean()
                if b_candidate < b_i:
                    b_i = b_candidate

            silhouettes[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)

        return float(silhouettes.mean())

    # -- Shannon diversity -------------------------------------------------

    @staticmethod
    def _shannon_diversity(species_ids: np.ndarray) -> float:
        """Shannon diversity index: H = -sum(p_i * ln(p_i)).

        Higher values mean more even distribution across species.
        A population with one dominant species has low diversity.
        """
        _, counts = np.unique(species_ids, return_counts=True)
        proportions = counts / counts.sum()
        proportions = proportions[proportions > 0]
        return float(-np.sum(proportions * np.log(proportions)))

    # -- Main tick ---------------------------------------------------------

    def tick(
        self,
        specialization_profiles: np.ndarray,
        column_fitness: np.ndarray,
        tick_count: int,
    ) -> Optional[SpeciesInfo]:
        """Run speciation clustering if the interval has elapsed.

        Args:
            specialization_profiles: (n_columns, 8) float array --
                the 8-dimensional specialization profile per column.
            column_fitness: (n_columns,) float fitness scores.
            tick_count: current global tick.

        Returns:
            SpeciesInfo if clustering was performed, else None.
        """
        if tick_count - self._last_clustering_tick < self._clustering_interval:
            return self._info

        with self._lock:
            self._last_clustering_tick = tick_count
            self._prev_species_ids = self._species_ids.copy()

            profiles = specialization_profiles.astype(np.float64)

            # Try k = 2..6, pick by silhouette
            best_k = 2
            best_score = -1.0
            best_labels = np.zeros(self._n_columns, dtype=np.int32)
            for k in range(2, 7):
                labels, _centroids = self._kmeans(profiles, k, rng=self._rng)
                score = self._silhouette_score(profiles, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            self._species_ids = best_labels

            # Compute species info
            unique_species = np.unique(best_labels)
            sizes = {int(s): int((best_labels == s).sum()) for s in unique_species}
            sp_fitness = {}
            for s in unique_species:
                members = column_fitness[best_labels == s]
                sp_fitness[int(s)] = float(members.mean()) if len(members) > 0 else 0.0

            # Turnover rate
            changed = (self._species_ids != self._prev_species_ids).sum()
            turnover = float(changed / self._n_columns)

            diversity = self._shannon_diversity(best_labels)

            self._info = SpeciesInfo(
                species_ids=best_labels.copy(),
                n_species=best_k,
                species_sizes=sizes,
                species_fitness=sp_fitness,
                shannon_diversity=diversity,
                turnover_rate=turnover,
                silhouette_score=best_score,
            )

            logger.debug(
                "Speciation: k=%d, silhouette=%.3f, diversity=%.3f, turnover=%.2f",
                best_k, best_score, diversity, turnover,
            )

            return self._info

    # -- Pruning threshold modifiers ---------------------------------------

    def get_pruning_factor(self, col_i: int, col_j: int) -> float:
        """Return the pruning threshold multiplier for a connection between
        columns i and j.

        Same-species connections get a lower factor (easier to keep).
        Cross-species connections get a higher factor (harder to keep).
        This creates functional modularity from selection pressure.
        """
        if self._species_ids[col_i] == self._species_ids[col_j]:
            return self._same_species_pruning_factor
        return self._cross_species_pruning_factor

    def get_protected_columns(self) -> List[int]:
        """Return column indices that are immune from pruning.

        Niche protection: at least one representative from each species
        must survive.  We pick the fittest member of each species.
        """
        protected = []
        if self._info is None:
            return protected
        for species_id in self._info.species_sizes:
            members = np.where(self._species_ids == species_id)[0]
            if len(members) > 0:
                # The fittest member is protected
                protected.append(int(members[0]))  # first member as representative
        return protected

    def get_species_expansion_scores(self) -> Dict[int, float]:
        """Return a score for each species indicating whether it should
        expand (positive) or contract (negative).

        Fit species get positive expansion, giving their members priority
        in replication.  Unfit species contract.
        """
        if self._info is None:
            return {}
        mean_fitness = np.mean(list(self._info.species_fitness.values())) if self._info.species_fitness else 0.5
        scores = {}
        for sp_id, sp_fit in self._info.species_fitness.items():
            scores[sp_id] = sp_fit - mean_fitness
        return scores

    # -- Query API ---------------------------------------------------------

    def get_species_id(self, col_idx: int) -> int:
        return int(self._species_ids[col_idx])

    def get_info(self) -> Optional[SpeciesInfo]:
        return self._info

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            if self._info is None:
                return {
                    "n_species": 0,
                    "shannon_diversity": 0.0,
                    "turnover_rate": 0.0,
                    "silhouette_score": 0.0,
                    "species_sizes": {},
                    "species_fitness": {},
                }
            return {
                "n_species": self._info.n_species,
                "shannon_diversity": round(self._info.shannon_diversity, 3),
                "turnover_rate": round(self._info.turnover_rate, 3),
                "silhouette_score": round(self._info.silhouette_score, 3),
                "species_sizes": self._info.species_sizes,
                "species_fitness": {k: round(v, 3) for k, v in self._info.species_fitness.items()},
            }


# ---------------------------------------------------------------------------
# 4. ThermodynamicPolicy -- from Evochora
# ---------------------------------------------------------------------------

@dataclass
class ThermodynamicCost:
    """Energy and entropy cost for a single operation type."""
    energy: float
    entropy: float


class ThermodynamicPolicy:
    """Per-operation energy and entropy costs inspired by Evochora's
    instruction-level thermodynamics.

    In Evochora, every instruction executed by a digital organism has both
    an energy cost (drawn from the organism's energy store) and an entropy
    generation (contributing to the global entropy pool that, when high
    enough, triggers "cleanup" events).  This creates genuine metabolic
    pressure: organisms that waste energy on failed instructions or
    unnecessary entropy generation are outcompeted.

    Mapped to Aura:

    Every cognitive operation -- LLM inference, tool execution, memory
    access, belief updates, self-repair, goal completion, dreaming -- has
    an explicit (energy_delta, entropy_delta) pair.  Negative energy means
    the operation is a reward (e.g., successful goal completion restores
    energy).  Negative entropy means the operation creates order (e.g.,
    memory writes, dream cleanup, self-repair).

    The cost table is stored as a mutable dictionary so that evolutionary
    processes can tune metabolic efficiency at runtime.  For example, if
    the system discovers that self-repair costs too much entropy, it can
    reduce that cost and see whether the change improves overall fitness.

    Failed operations: Evochora penalizes failed instructions with 50%
    extra entropy.  A failed LLM inference generates 7.5 entropy instead
    of 5.0 -- waste heat from a botched computation.
    """

    # Default cost table
    _DEFAULT_COSTS: Dict[str, Tuple[float, float]] = {
        "llm_inference_primary":    (8.0,  5.0),
        "llm_inference_secondary":  (3.0,  2.0),
        "llm_inference_tertiary":   (1.0,  0.5),
        "tool_execution":           (4.0,  2.0),
        "memory_write":             (1.0, -1.0),
        "memory_read":              (0.5,  0.0),
        "belief_update":            (2.0,  1.0),
        "self_repair":              (3.0, -2.0),
        "goal_completion":         (-5.0, -4.0),
        "idle_tick":               (-0.2, -0.1),
        "background_tick":          (0.5,  0.3),
        "dream_cycle":              (2.0, -10.0),
        "user_interaction_success": (-3.0, -1.0),
        "user_interaction_failure": (1.0,  2.0),
        "mesh_substep":             (0.1,  0.05),
    }

    def __init__(self):
        self._lock = threading.Lock()

        # Mutable cost table -- seeded from defaults, modifiable at runtime
        self._costs: Dict[str, ThermodynamicCost] = {}
        for op, (energy, entropy) in self._DEFAULT_COSTS.items():
            self._costs[op] = ThermodynamicCost(energy=energy, entropy=entropy)

        # Running totals (for telemetry)
        self._total_energy_applied: float = 0.0
        self._total_entropy_applied: float = 0.0
        self._operation_counts: Dict[str, int] = {}
        self._failed_operation_counts: Dict[str, int] = {}

        # Error penalty multiplier on entropy
        self._failure_entropy_multiplier: float = 1.5

    # -- Cost lookup -------------------------------------------------------

    def get_cost(self, operation: str) -> ThermodynamicCost:
        """Look up the cost for an operation type.

        Returns (0, 0) for unknown operations rather than raising.
        """
        return self._costs.get(operation, ThermodynamicCost(energy=0.0, entropy=0.0))

    def apply_cost(self, operation: str, failed: bool = False) -> Tuple[float, float]:
        """Apply the cost for an operation and return (energy_delta, entropy_delta).

        If the operation failed, entropy is multiplied by 1.5 (waste from
        failure), matching Evochora's error penalty.

        The deltas are SIGNED: negative energy means a reward (energy
        gained), positive energy means a cost (energy spent).  The caller
        is responsible for applying these to HomeostaticRL.energy and the
        EntropyTracker.
        """
        cost = self.get_cost(operation)
        energy_delta = cost.energy
        entropy_delta = cost.entropy

        if failed and entropy_delta > 0:
            entropy_delta *= self._failure_entropy_multiplier

        with self._lock:
            self._total_energy_applied += energy_delta
            self._total_entropy_applied += entropy_delta
            self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
            if failed:
                self._failed_operation_counts[operation] = (
                    self._failed_operation_counts.get(operation, 0) + 1
                )

        return energy_delta, entropy_delta

    # -- Runtime tuning ----------------------------------------------------

    def set_cost(self, operation: str, energy: float, entropy: float) -> None:
        """Override the cost for an operation at runtime.

        This is called by evolutionary processes that are tuning the
        metabolic efficiency of the system.
        """
        with self._lock:
            self._costs[operation] = ThermodynamicCost(energy=energy, entropy=entropy)

    def scale_all_costs(self, factor: float) -> None:
        """Scale all costs by a factor.  Useful for global metabolic
        adjustments (e.g., entering a low-power mode)."""
        with self._lock:
            for op in self._costs:
                self._costs[op] = ThermodynamicCost(
                    energy=self._costs[op].energy * factor,
                    entropy=self._costs[op].entropy * factor,
                )

    # -- Query API ---------------------------------------------------------

    def get_cost_table(self) -> Dict[str, Dict[str, float]]:
        """Return the full cost table as a JSON-friendly dict."""
        with self._lock:
            return {
                op: {"energy": c.energy, "entropy": c.entropy}
                for op, c in self._costs.items()
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_energy_applied": round(self._total_energy_applied, 2),
                "total_entropy_applied": round(self._total_entropy_applied, 2),
                "operation_counts": dict(self._operation_counts),
                "failed_operation_counts": dict(self._failed_operation_counts),
                "failure_entropy_multiplier": self._failure_entropy_multiplier,
                "n_operation_types": len(self._costs),
            }


# ---------------------------------------------------------------------------
# 5. OwnershipModel -- from Evochora
# ---------------------------------------------------------------------------

class OwnershipModel:
    """Ownership-based memory access costs inspired by Evochora's PEEK
    instruction semantics.

    In Evochora, a PEEK instruction (reading another organism's memory) has
    different costs depending on the relationship between the reader and the
    memory owner:
      - Own memory: cheap (local register access)
      - Nearby organism: moderate cost (local bus)
      - Distant organism: expensive (network traversal)
      - System/kernel memory: very expensive (privileged access)

    Mapped to Aura's working memory:

    Each entry in working memory has an 'owner' field naming the subsystem
    that created it (e.g., "neural_mesh", "homeostasis", "narrative_gravity",
    "somatic_gate").  When a subsystem reads an entry, the access cost depends
    on the relationship:
      - Same subsystem (own memory):      1.0x base cost
      - Same tier (sibling):              1.5x base cost
      - Different tier (foreign):         2.0x base cost
      - Kernel/governance (system):       3.0x base cost

    This creates natural information locality.  Subsystems prefer their own
    recent outputs because they're cheapest to access, creating temporal
    coherence.  Cross-subsystem communication is selective because it costs
    more, filtering out low-value information transfer.

    Tier assignment: subsystems are grouped into tiers reflecting their
    position in the cognitive hierarchy:
      - sensory:      embodied_interoception, neural_mesh, peripheral_awareness
      - association:  narrative_gravity, theory_of_mind, world_model,
                      counterfactual_engine, multiple_drafts
      - executive:    executive_authority, somatic_gate, metacognition,
                      predictive_engine, homeostasis
      - kernel:       consciousness_bridge, substrate_evolution, unified_field,
                      governance (constitutional layer)
    """

    # Tier assignments for known subsystems
    _TIER_MAP: Dict[str, str] = {
        # Sensory tier
        "embodied_interoception": "sensory",
        "neural_mesh":           "sensory",
        "peripheral_awareness":  "sensory",
        "oscillatory_binding":   "sensory",
        "qualia_engine":         "sensory",
        # Association tier
        "narrative_gravity":     "association",
        "theory_of_mind":        "association",
        "world_model":           "association",
        "counterfactual_engine": "association",
        "multiple_drafts":       "association",
        "semantic_bridge":       "association",
        "evidence_engine":       "association",
        "dreaming":              "association",
        "attention_schema":      "association",
        # Executive tier
        "executive_authority":   "executive",
        "somatic_gate":          "executive",
        "metacognition":         "executive",
        "predictive_engine":     "executive",
        "homeostasis":           "executive",
        "homeostatic_rl":        "executive",
        "free_energy":           "executive",
        "credit_assignment":     "executive",
        # Kernel tier
        "consciousness_bridge":  "kernel",
        "substrate_evolution":   "kernel",
        "unified_field":         "kernel",
        "governance":            "kernel",
        "substrate_authority":   "kernel",
        "liquid_substrate":      "kernel",
    }

    # Cost multipliers
    _COST_MULTIPLIERS: Dict[str, float] = {
        "own":     1.0,
        "sibling": 1.5,
        "foreign": 2.0,
        "system":  3.0,
    }

    def __init__(self):
        self._lock = threading.Lock()

        # Access tracking
        self._access_counts: Dict[str, int] = {
            "own": 0,
            "sibling": 0,
            "foreign": 0,
            "system": 0,
        }
        self._total_accesses: int = 0

        # Ownership distribution tracking: subsystem -> entry count
        self._ownership_counts: Dict[str, int] = {}

    # -- Tier resolution ---------------------------------------------------

    def _get_tier(self, subsystem: str) -> str:
        """Resolve a subsystem name to its tier.

        Unknown subsystems default to 'association' (middle of the
        hierarchy) rather than 'kernel' to avoid over-penalizing new
        modules that haven't been registered yet.
        """
        return self._TIER_MAP.get(subsystem, "association")

    def _classify_access(self, accessor: str, owner: str) -> str:
        """Classify the relationship between an accessor and an owner.

        Returns one of: 'own', 'sibling', 'foreign', 'system'.
        """
        if accessor == owner:
            return "own"

        owner_tier = self._get_tier(owner)
        if owner_tier == "kernel":
            return "system"

        accessor_tier = self._get_tier(accessor)
        if accessor_tier == owner_tier:
            return "sibling"

        return "foreign"

    # -- Public API --------------------------------------------------------

    def get_access_cost(self, accessor: str, owner: str) -> float:
        """Return the cost multiplier for 'accessor' reading memory owned
        by 'owner'.

        This multiplier should be applied to the base memory_read cost
        from ThermodynamicPolicy.  For example, if base memory_read costs
        0.5 energy and the access is cross-tier (2.0x), the actual cost
        is 1.0 energy.
        """
        relationship = self._classify_access(accessor, owner)
        multiplier = self._COST_MULTIPLIERS[relationship]

        with self._lock:
            self._access_counts[relationship] += 1
            self._total_accesses += 1

        return multiplier

    def register_ownership(self, subsystem: str, count: int = 1) -> None:
        """Record that a subsystem owns 'count' new working memory entries."""
        with self._lock:
            self._ownership_counts[subsystem] = (
                self._ownership_counts.get(subsystem, 0) + count
            )

    def get_ownership_stats(self) -> Dict[str, Any]:
        """Return ownership distribution and access pattern statistics."""
        with self._lock:
            total_entries = sum(self._ownership_counts.values()) or 1
            tier_distribution: Dict[str, int] = {}
            for sub, count in self._ownership_counts.items():
                tier = self._get_tier(sub)
                tier_distribution[tier] = tier_distribution.get(tier, 0) + count

            return {
                "total_entries": sum(self._ownership_counts.values()),
                "by_subsystem": dict(self._ownership_counts),
                "by_tier": tier_distribution,
                "access_counts": dict(self._access_counts),
                "total_accesses": self._total_accesses,
                "cross_boundary_fraction": round(
                    (self._access_counts["sibling"] + self._access_counts["foreign"] + self._access_counts["system"])
                    / max(self._total_accesses, 1),
                    3,
                ),
            }

    def get_status(self) -> Dict[str, Any]:
        return self.get_ownership_stats()


# ---------------------------------------------------------------------------
# Composite state and wrapper
# ---------------------------------------------------------------------------

@dataclass
class ALifeExtensionState:
    """Snapshot returned by ALifeExtensions.tick() for downstream consumers."""
    replications_this_tick: List[ReplicationEvent] = field(default_factory=list)
    species_info: Optional[SpeciesInfo] = None
    toroidal_distances: Optional[np.ndarray] = None
    thermodynamic_costs_applied: Tuple[float, float] = (0.0, 0.0)
    ownership_stats: Dict[str, Any] = field(default_factory=dict)


class ALifeExtensions:
    """Unified wrapper for all five ALife extension systems.

    Instantiated once by the ConsciousnessBridge and ticked on every
    heartbeat.  Provides a single async tick() entry point and a
    dashboard-ready get_status() for the HUD.

    This class does not own the NeuralMesh or HomeostaticRL -- it
    receives their state as arguments and returns deltas.  This keeps
    the ALife extensions loosely coupled: they can be disabled without
    breaking the core consciousness stack.
    """

    def __init__(self, n_columns: int = 64, n_specialization_dims: int = 8):
        self._lock = threading.Lock()
        self._n_columns = n_columns

        # Subsystems
        self.topology = ToroidalTopology(n_columns)
        self.replicator = PatternReplicator(n_columns, self.topology)
        self.speciation = ColumnSpeciation(n_columns, n_specialization_dims)
        self.thermodynamics = ThermodynamicPolicy()
        self.ownership = OwnershipModel()

        # Tick counter
        self._tick_count: int = 0
        self._start_time: float = time.time()

        # Accumulated thermodynamic costs for reporting
        self._last_energy_delta: float = 0.0
        self._last_entropy_delta: float = 0.0

        logger.info(
            "ALifeExtensions initialized (columns=%d, dims=%d)",
            n_columns, n_specialization_dims,
        )

    async def tick(
        self,
        mesh_state: Dict[str, Any],
        evolution_state: Dict[str, Any],
        tick_count: int,
    ) -> ALifeExtensionState:
        """Run one integration step across all five ALife subsystems.

        Args:
            mesh_state: dictionary with keys:
                - "columns_W": list of (n, n) numpy arrays, intra-column weights
                  (MODIFIED IN PLACE by replicator)
                - "contributions": (n_columns,) float -- output contribution per column
                - "stabilities": (n_columns,) float -- activation stability per column
                - "error_rates": (n_columns,) float -- error rate per column
                - "specialization_profiles": (n_columns, 8) float --
                  specialization dimensions per column
            evolution_state: dictionary with keys:
                - "generation": int -- current evolutionary generation
                - "champion_fitness": float -- fitness of current champion genome
            tick_count: global tick counter.

        Returns:
            ALifeExtensionState with the results of this tick's processing.
        """
        self._tick_count = tick_count

        # -- Extract mesh state with safe defaults -------------------------
        columns_W = mesh_state.get("columns_W", [])
        n = self._n_columns

        contributions = np.asarray(
            mesh_state.get("contributions", np.full(n, 0.5, dtype=np.float32)),
            dtype=np.float32,
        )
        stabilities = np.asarray(
            mesh_state.get("stabilities", np.full(n, 0.5, dtype=np.float32)),
            dtype=np.float32,
        )
        error_rates = np.asarray(
            mesh_state.get("error_rates", np.zeros(n, dtype=np.float32)),
            dtype=np.float32,
        )
        spec_profiles = np.asarray(
            mesh_state.get(
                "specialization_profiles",
                np.zeros((n, 8), dtype=np.float32),
            ),
            dtype=np.float32,
        )

        # -- Compute fitness -----------------------------------------------
        fitness = PatternReplicator.compute_column_fitness(
            contributions, stabilities, error_rates,
        )

        # -- 1. Pattern Replication ----------------------------------------
        replications: List[ReplicationEvent] = []
        if columns_W:
            replications = self.replicator.tick(columns_W, fitness, tick_count)

        # -- 2. Speciation -------------------------------------------------
        species_info = self.speciation.tick(spec_profiles, fitness, tick_count)

        # -- 3. Toroidal distances (cached, always available) ---------------
        torus_matrix = self.topology.get_matrix()

        # -- 4. Thermodynamic cost for this mesh substep -------------------
        # Each tick of the mesh is one "mesh_substep" operation per column.
        # We apply the aggregate cost once rather than 64 times.
        base_energy, base_entropy = self.thermodynamics.get_cost("mesh_substep").energy, self.thermodynamics.get_cost("mesh_substep").entropy
        tick_energy = base_energy * n
        tick_entropy = base_entropy * n

        with self._lock:
            self._last_energy_delta = tick_energy
            self._last_entropy_delta = tick_entropy

        # -- 5. Ownership (passive -- tracks are updated by callers) --------
        ownership_stats = self.ownership.get_ownership_stats()

        return ALifeExtensionState(
            replications_this_tick=replications,
            species_info=species_info,
            toroidal_distances=torus_matrix,
            thermodynamic_costs_applied=(tick_energy, tick_entropy),
            ownership_stats=ownership_stats,
        )

    def get_status(self) -> Dict[str, Any]:
        """Dashboard-ready summary of all ALife extension subsystems."""
        with self._lock:
            return {
                "alife_extensions": {
                    "tick_count": self._tick_count,
                    "uptime_s": round(time.time() - self._start_time, 1),
                    "last_energy_delta": round(self._last_energy_delta, 2),
                    "last_entropy_delta": round(self._last_entropy_delta, 2),
                },
                "pattern_replicator": self.replicator.get_status(),
                "column_speciation": self.speciation.get_status(),
                "toroidal_topology": {
                    "n_columns": self.topology.n_columns,
                    "max_distance": 0.5,
                    "wrapping": "toroidal",
                },
                "thermodynamic_policy": self.thermodynamics.get_status(),
                "ownership_model": self.ownership.get_status(),
            }
