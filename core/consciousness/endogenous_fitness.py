"""core/consciousness/endogenous_fitness.py — Endogenous Fitness Evaluation

Replaces the explicit hand-designed fitness formula in substrate_evolution.py
with a survival-based approach inspired by two landmark artificial life systems:

Tierra (Tom Ray, 1991):
    In Tierra, organisms are self-replicating programs in a shared memory space.
    There is NO explicit fitness function. Fitness is endogenous: a program that
    replicates itself survives, and one that doesn't goes extinct. The key insight
    is that real fitness emerges from the process of surviving, not from an
    external formula saying "this is good."  Our system applies the same idea:
    instead of computing fitness = phi * coherence * efficiency, we let a genome
    configuration run for a window of time and observe whether the system stays
    stable. Survival IS fitness.

EcoSim (Larry Yaeger, 1994):
    EcoSim added two ideas on top of Tierra-style endogenous fitness:
    1. Behavioral rules should evolve alongside structural parameters. Organisms
       in EcoSim had neural networks whose weights evolved, giving them learned
       behavioral strategies. We implement this via a Fuzzy Cognitive Map (FCM)
       that maps internal states to action preferences.
    2. Speciation and niche protection. In EcoSim, the population would diverge
       into distinct species that exploited different ecological niches. This
       prevents evolutionary collapse to a single strategy. We implement this
       via k-means clustering on genome vectors with silhouette-score-based
       species detection.

Integration:
    This module provides an alternative fitness evaluator that the existing
    SubstrateEvolution can call instead of its internal _evaluate_fitness().
    It does NOT replace substrate_evolution.py — it is a drop-in upgrade.

    Usage:
        from core.consciousness.endogenous_fitness import get_endogenous_fitness
        ef = get_endogenous_fitness()
        result = await ef.evaluate_fitness(genome_params)
"""
from __future__ import annotations


__all__ = [
    "EndogenousFitness",
    "EndogenousFitnessConfig",
    "FitnessResult",
    "SpeciesInfo",
    "get_endogenous_fitness",
]

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.EndogenousFitness")


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class Action(Enum):
    """Behavioral actions the FCM can express preference for.

    These map to the top-level action modes of Aura's cognitive heartbeat:
      RESPOND  — answer a user query or react to an event
      EXPLORE  — seek out new information or novel experiences
      REFLECT  — introspect, consolidate, journal
      REST     — reduce activity, conserve resources
      REPAIR   — self-heal, defragment, fix anomalies
      INITIATE — proactively start a conversation or task
    """
    RESPOND = 0
    EXPLORE = 1
    REFLECT = 2
    REST = 3
    REPAIR = 4
    INITIATE = 5


NUM_ACTIONS = len(Action)  # 6

# The seven internal state dimensions the FCM reads as input.
STATE_DIM_NAMES = [
    "energy",          # 0: metabolic reserves (0-1)
    "social_hunger",   # 1: desire for interaction (0-1)
    "curiosity",       # 2: novelty-seeking drive (0-1)
    "competence",      # 3: recent task success rate (0-1)
    "coherence_need",  # 4: how much the system needs re-integration (0-1)
    "free_energy",     # 5: active inference free energy (0-1, lower = calmer)
    "threat_level",    # 6: anomaly/security threat (0-1)
]
STATE_DIM = len(STATE_DIM_NAMES)  # 7


# ---------------------------------------------------------------------------
# Crisis reasons — why an evaluation ended early
# ---------------------------------------------------------------------------

class CrisisReason(Enum):
    """Why the survival evaluation was terminated before the window ended.

    Each of these mirrors a real biological death/collapse mode:
      ENERGY_DEPLETED — ran out of metabolic fuel (starvation)
      VITALITY_COLLAPSED — autopoiesis loop broke (organ failure)
      SUSTAINED_THREAT — persistent external danger (predation)
      SUSTAINED_FREE_ENERGY — stuck in high surprise (confusion death)
      ENTROPY_OVERFLOW — internal disorder too high (heat death)
      PHI_ZERO — integrated information vanished (brain death)
    """
    ENERGY_DEPLETED = auto()
    VITALITY_COLLAPSED = auto()
    SUSTAINED_THREAT = auto()
    SUSTAINED_FREE_ENERGY = auto()
    ENTROPY_OVERFLOW = auto()
    PHI_ZERO = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndogenousFitnessConfig:
    """Tuning knobs for the endogenous fitness evaluator.

    All thresholds are chosen conservatively so that only genuinely
    unstable genomes trigger early termination. A genome that keeps
    the system barely alive for the full window still earns a score
    of 1.0 before bonuses.
    """
    # Survival evaluation
    evaluation_window_s: float = 60.0   # seconds per genome evaluation
    tick_interval_s: float = 1.0        # how often to sample system state

    # Crisis thresholds
    energy_critical: float = 15.0       # raw energy units (substrate scale)
    vitality_critical: float = 0.3      # autopoiesis vitality minimum
    threat_sustained_threshold: float = 0.8   # anomaly threat level
    threat_sustained_ticks: int = 5           # consecutive ticks at high threat
    free_energy_sustained_threshold: float = 0.8
    free_energy_sustained_ticks: int = 5
    entropy_max_fraction: float = 0.8   # fraction of max entropy
    phi_zero_ticks: int = 10            # consecutive ticks with phi == 0

    # Behavioral genome
    behavioral_mutation_rate: float = 0.05   # per-gene mutation probability
    structural_mutation_rate: float = 0.01   # per-gene for structural genes
    fcm_blend_alpha_init: float = 0.3        # initial weight for FCM prefs
    fcm_blend_alpha_adapt_rate: float = 0.01 # how fast alpha adapts

    # Speciation
    speciation_check_interval: int = 10   # every N evolutionary cycles
    speciation_min_k: int = 2
    speciation_max_k: int = 5
    speciation_silhouette_threshold: float = 0.4

    # Entropy ceiling for crisis detection (log2 of state space)
    max_entropy: float = 16.0  # bits (log2(2^16) for 16-node complex)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    """The outcome of one survival-based fitness evaluation.

    Fields:
        fitness_score: Final combined score in [0, ~2.0]. The base score is
            survival_time / evaluation_window (0-1), then multiplied by
            efficiency and integration bonuses.
        survival_time: How many seconds the system ran stably.
        crisis_reason: Why the evaluation ended early, or None if the genome
            survived the full window.
        efficiency_bonus: Multiplier from energy efficiency (0-1 range as
            a factor, so the multiplier is in [0, 1]).
        phi_bonus: Multiplier from mean integrated information during the
            run. Systems that naturally achieve high phi without being told
            to maximize it get a bonus.
        behavioral_fitness: How well the FCM behavioral genome contributed
            to survival (correlation between FCM action preferences and
            actions that actually helped).
    """
    fitness_score: float
    survival_time: float
    crisis_reason: Optional[CrisisReason]
    efficiency_bonus: float
    phi_bonus: float
    behavioral_fitness: float


@dataclass
class SpeciesInfo:
    """Summary of the current speciation state of the genome population.

    Fields:
        species_count: How many distinct species (genome clusters) exist.
            1 means the population is homogeneous (no speciation).
        sizes: List of how many genomes belong to each species, sorted
            largest-first.
        turnover_rate: Fraction of species that went extinct or appeared
            in the last speciation check. High turnover means the
            evolutionary landscape is volatile.
        silhouette_score: Quality of the clustering (0-1). Above 0.4
            indicates genuine speciation rather than noise.
        champions_per_species: Best fitness score in each species.
    """
    species_count: int
    sizes: List[int]
    turnover_rate: float
    silhouette_score: float
    champions_per_species: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Behavioral Genome — Fuzzy Cognitive Map
# ---------------------------------------------------------------------------

class BehavioralGenome:
    """A Fuzzy Cognitive Map (FCM) that maps internal state to action preferences.

    Inspired by EcoSim: instead of hard-coding rules like "if energy < 0.2
    then REST," the behavioral genome is a weight matrix that EVOLVES. The
    system discovers its own behavioral rules through selection pressure.

    The FCM is a matrix R of shape (NUM_ACTIONS, STATE_DIM) where each entry
    R[a, i] is a signed weight in [-1, 1] that says how much state dimension i
    contributes to the preference for action a.

    For example, if R[REST, energy] = -0.8, it means low energy strongly
    increases the preference for REST (because the negative weight times a
    low energy value yields a high preference via the sigmoid).

    The output uses fuzzy AND semantics:
        pref[a] = product over i of sigmoid(R[a, i] * s[i])

    This means ALL relevant state dimensions must "agree" for an action to
    get a high preference — one strongly negative signal can suppress an
    action. This is more conservative than a simple weighted sum and leads
    to more nuanced behavioral strategies.
    """

    def __init__(self, weights: Optional[np.ndarray] = None,
                 rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng()
        if weights is not None:
            if weights.shape != (NUM_ACTIONS, STATE_DIM):
                raise ValueError(
                    f"FCM weight matrix must be ({NUM_ACTIONS}, {STATE_DIM}), "
                    f"got {weights.shape}"
                )
            self.weights: np.ndarray = np.clip(weights, -1.0, 1.0).astype(np.float32)
        else:
            # Initialize with small random weights — no innate behavioral bias
            self.weights = (self._rng.standard_normal((NUM_ACTIONS, STATE_DIM))
                            * 0.1).astype(np.float32)

        # Adaptive blending parameter: how much the FCM overrides the
        # top-down (td) action preferences from the existing system.
        # Starts at 0.3 and adapts based on whether FCM-influenced
        # decisions lead to better survival outcomes.
        self.alpha: float = 0.3

    def compute_preferences(self, state: np.ndarray) -> np.ndarray:
        """Compute action preferences from the current internal state.

        Args:
            state: Array of shape (STATE_DIM,) with values in [0, 1].

        Returns:
            Array of shape (NUM_ACTIONS,) with preference scores in (0, 1).
            Higher values mean stronger preference for that action.
        """
        state = np.asarray(state, dtype=np.float32).flatten()[:STATE_DIM]
        if state.shape[0] < STATE_DIM:
            padded = np.full(STATE_DIM, 0.5, dtype=np.float32)
            padded[:state.shape[0]] = state
            state = padded

        # Fuzzy AND: product of sigmoid(R[a,i] * s[i]) for all i
        # sigmoid(x) = 1 / (1 + exp(-x))
        # We scale the input by 4 to make the sigmoid more decisive
        # (without scaling, sigmoid(0.5) ~ 0.62 which is barely above 0.5)
        raw = self.weights * state[np.newaxis, :]  # (NUM_ACTIONS, STATE_DIM)
        activated = _sigmoid(raw * 4.0)            # (NUM_ACTIONS, STATE_DIM)

        # Product across state dimensions (fuzzy AND)
        preferences = np.prod(activated, axis=1)   # (NUM_ACTIONS,)
        return preferences

    def blend_with_td(self, td_prefs: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Blend top-down preferences with FCM preferences.

        Args:
            td_prefs: Top-down action preferences from the existing system,
                shape (NUM_ACTIONS,).
            state: Current internal state, shape (STATE_DIM,).

        Returns:
            Blended preferences, shape (NUM_ACTIONS,). When alpha is 0,
            this returns pure top-down prefs. When alpha is 1, pure FCM.
        """
        fcm_prefs = self.compute_preferences(state)
        td_prefs = np.asarray(td_prefs, dtype=np.float32).flatten()[:NUM_ACTIONS]
        if td_prefs.shape[0] < NUM_ACTIONS:
            padded = np.full(NUM_ACTIONS, 0.5, dtype=np.float32)
            padded[:td_prefs.shape[0]] = td_prefs
            td_prefs = padded

        blended = (1.0 - self.alpha) * td_prefs + self.alpha * fcm_prefs
        return blended

    def adapt_alpha(self, reward_signal: float, rate: float = 0.01) -> None:
        """Nudge the blending parameter based on whether FCM-influenced
        decisions helped or hurt survival.

        Args:
            reward_signal: Positive if the FCM-influenced decision improved
                survival, negative if it hurt. Typically in [-1, 1].
            rate: How fast alpha adapts per call.
        """
        # If the FCM contributed to a good outcome, increase its influence.
        # If it contributed to a bad outcome, decrease its influence.
        self.alpha = float(np.clip(self.alpha + rate * reward_signal, 0.05, 0.95))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    # Clip to avoid overflow in exp()
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean silhouette score for a clustering.

    The silhouette score measures how similar each point is to its own
    cluster versus the nearest other cluster. Ranges from -1 (terrible)
    to +1 (perfect). Above 0.4 indicates meaningful clusters.

    This is a pure-numpy implementation to avoid requiring scikit-learn.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        labels: Cluster assignment for each sample, shape (n_samples,).

    Returns:
        Mean silhouette score across all samples.
    """
    n = len(labels)
    if n < 2:
        return 0.0
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Pairwise distance matrix (Euclidean)
    # Using broadcasting: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i.x_j
    norms_sq = np.sum(X ** 2, axis=1)
    dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2.0 * (X @ X.T)
    dist = np.sqrt(np.maximum(dist_sq, 0.0))

    silhouettes = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = np.sum(own_mask) - 1  # exclude self

        if own_count <= 0:
            silhouettes[i] = 0.0
            continue

        # a(i) = mean distance to points in same cluster
        a_i = np.sum(dist[i, own_mask]) / own_count

        # b(i) = min over other clusters of mean distance to that cluster
        b_i = np.inf
        for label in unique_labels:
            if label == own_label:
                continue
            other_mask = labels == label
            other_count = np.sum(other_mask)
            if other_count == 0:
                continue
            mean_dist = np.sum(dist[i, other_mask]) / other_count
            b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)

    return float(np.mean(silhouettes))


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simple k-means clustering. Returns label array of shape (n_samples,).

    Pure numpy implementation — no sklearn dependency.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        k: Number of clusters.
        max_iter: Maximum iterations.
        rng: Random number generator for centroid initialization.

    Returns:
        Integer array of cluster labels, shape (n_samples,).
    """
    rng = rng or np.random.default_rng()
    n, d = X.shape
    if n <= k:
        return np.arange(n, dtype=np.int32)

    # k-means++ initialization for better convergence
    centroids = np.empty((k, d), dtype=X.dtype)
    centroids[0] = X[rng.integers(0, n)]
    for c in range(1, k):
        dist_sq = np.min(
            np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :c, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dist_sq / (dist_sq.sum() + 1e-12)
        centroids[c] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assign
        dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EndogenousFitness:
    """Survival-based fitness evaluation inspired by Tierra and EcoSim.

    Instead of computing fitness as an explicit formula
    (fitness = phi * coherence * efficiency * binding_strength), this class
    evaluates fitness by letting a genome configuration run and observing
    whether the system stays alive.

    The three components of fitness:

    1. SURVIVAL (Tierra-inspired):
       A genome's base fitness is simply how long the system runs stably
       with that genome's configuration, divided by the evaluation window.
       If it survives the full window, base fitness = 1.0.

    2. BEHAVIORAL COMPETENCE (EcoSim-inspired):
       Each genome includes a Fuzzy Cognitive Map that maps internal states
       to action preferences. The FCM evolves alongside structural parameters,
       so behavioral strategy is discovered rather than hand-coded.

    3. SPECIATION (EcoSim-inspired):
       The population of genomes is periodically checked for clustering.
       If distinct species emerge, niche protection ensures each species'
       best genome survives, preventing evolutionary collapse.

    Thread safety:
       All mutable state is protected by a threading.Lock. The evaluation
       coroutine itself uses asyncio.Lock for async safety within the
       event loop.

    Lifecycle:
        ef = get_endogenous_fitness()
        result = await ef.evaluate_fitness(genome_params)
        info = ef.get_species_info()
    """

    def __init__(self, cfg: EndogenousFitnessConfig | None = None):
        self.cfg = cfg or EndogenousFitnessConfig()
        self._rng = np.random.default_rng()

        # Thread-safe lock for all mutable state
        self._lock = threading.Lock()

        # Async lock for evaluation serialization (only one eval at a time)
        self._eval_lock = asyncio.Lock()

        # Behavioral genome — the FCM that evolves alongside structural genes
        self._behavioral_genome = BehavioralGenome(rng=self._rng)

        # Speciation tracking
        self._species_labels: np.ndarray = np.array([], dtype=np.int32)
        self._species_history: List[SpeciesInfo] = []
        self._evolution_cycle_count: int = 0
        self._previous_species_count: int = 1

        # Genome archive for speciation analysis
        # Each entry: (genome_vector, fitness_score)
        self._genome_archive: List[Tuple[np.ndarray, float]] = []
        self._archive_max_size: int = 200

        # Running statistics for efficiency bonus calculation
        self._energy_observations: List[float] = []
        self._max_energy_observed: float = 100.0  # will be updated dynamically

        logger.info(
            "EndogenousFitness initialized (window=%.0fs, tick=%.1fs, "
            "behavioral_mut=%.3f, structural_mut=%.3f)",
            self.cfg.evaluation_window_s,
            self.cfg.tick_interval_s,
            self.cfg.behavioral_mutation_rate,
            self.cfg.structural_mutation_rate,
        )

    # ------------------------------------------------------------------
    # Core API: survival-based fitness evaluation
    # ------------------------------------------------------------------

    async def evaluate_fitness(
        self,
        genome_params: dict,
        evaluation_window_s: Optional[float] = None,
    ) -> FitnessResult:
        """Evaluate a genome's fitness by observing system survival.

        This is the Tierra insight: instead of computing a score from a
        formula, we let the genome run and see if the system stays alive.
        A genome that keeps energy above critical, phi nonzero, entropy
        bounded, and threats handled IS fit — we don't need to weight
        those factors by hand.

        Args:
            genome_params: Dictionary of genome parameters. Must contain
                at least 'inter_weights' (np.ndarray). May also contain
                structural gene values (mu, sigma, beta, etc.) and a
                'behavioral_weights' key with the FCM matrix.
            evaluation_window_s: Override the default evaluation window.
                Shorter windows are useful for quick screening; longer
                windows catch slow-onset instability.

        Returns:
            FitnessResult with the survival-based fitness score and
            diagnostic information.
        """
        window = evaluation_window_s or self.cfg.evaluation_window_s

        async with self._eval_lock:
            return await self._run_survival_evaluation(genome_params, window)

    async def _run_survival_evaluation(
        self,
        genome_params: dict,
        window: float,
    ) -> FitnessResult:
        """Internal: run the survival evaluation loop.

        Samples system state at regular intervals and checks for crisis
        conditions. The genome's fitness is determined by how long it
        survives and how efficiently it does so.
        """
        tick = self.cfg.tick_interval_s
        max_ticks = int(window / tick)
        if max_ticks < 1:
            max_ticks = 1

        # Extract behavioral genome if present
        behavioral_weights = genome_params.get("behavioral_weights", None)
        if behavioral_weights is not None:
            eval_behavior = BehavioralGenome(
                weights=np.asarray(behavioral_weights, dtype=np.float32),
                rng=self._rng,
            )
        else:
            eval_behavior = self._behavioral_genome

        # Crisis streak counters
        threat_streak: int = 0
        free_energy_streak: int = 0
        phi_zero_streak: int = 0

        # Accumulators for bonus calculations
        energy_consumed_total: float = 0.0
        phi_total: float = 0.0
        tick_count: int = 0
        crisis_reason: Optional[CrisisReason] = None
        start_time = time.monotonic()
        previous_energy: Optional[float] = None

        for t in range(max_ticks):
            # Yield control so the rest of the system keeps running.
            # This is critical: we are observing the LIVE system, not
            # a simulation. The genome is already applied to the mesh,
            # and we are watching what happens.
            await asyncio.sleep(tick)

            # Sample system state from live services
            state = self._sample_system_state()
            tick_count += 1

            energy = state["energy"]
            vitality = state["vitality"]
            threat = state["threat_level"]
            free_energy = state["free_energy"]
            entropy = state["entropy"]
            phi = state["phi"]

            # Track energy consumption
            if previous_energy is not None:
                consumed = max(0.0, previous_energy - energy)
                energy_consumed_total += consumed
            previous_energy = energy

            # Accumulate phi for integration bonus
            phi_total += phi

            # ----------------------------------------------------------
            # Crisis detection: any of these ends the evaluation early.
            # The philosophy is Darwinian: if the system dies, the genome
            # that caused it is unfit. Period.
            # ----------------------------------------------------------

            # 1. Energy depletion (starvation)
            if energy < self.cfg.energy_critical:
                crisis_reason = CrisisReason.ENERGY_DEPLETED
                break

            # 2. Vitality collapse (autopoiesis failure)
            if vitality < self.cfg.vitality_critical:
                crisis_reason = CrisisReason.VITALITY_COLLAPSED
                break

            # 3. Sustained threat (predation)
            if threat > self.cfg.threat_sustained_threshold:
                threat_streak += 1
            else:
                threat_streak = 0
            if threat_streak >= self.cfg.threat_sustained_ticks:
                crisis_reason = CrisisReason.SUSTAINED_THREAT
                break

            # 4. Sustained free energy (confusion death)
            if free_energy > self.cfg.free_energy_sustained_threshold:
                free_energy_streak += 1
            else:
                free_energy_streak = 0
            if free_energy_streak >= self.cfg.free_energy_sustained_ticks:
                crisis_reason = CrisisReason.SUSTAINED_FREE_ENERGY
                break

            # 5. Entropy overflow (heat death)
            if entropy > self.cfg.entropy_max_fraction * self.cfg.max_entropy:
                crisis_reason = CrisisReason.ENTROPY_OVERFLOW
                break

            # 6. Phi zero for too long (brain death)
            if phi <= 0.0:
                phi_zero_streak += 1
            else:
                phi_zero_streak = 0
            if phi_zero_streak >= self.cfg.phi_zero_ticks:
                crisis_reason = CrisisReason.PHI_ZERO
                break

        # ----------------------------------------------------------
        # Compute fitness from survival + bonuses
        # ----------------------------------------------------------
        elapsed = time.monotonic() - start_time
        survival_time = min(elapsed, window)

        # Base fitness: fraction of the window survived (0 to 1.0)
        base_fitness = survival_time / window

        # Efficiency bonus: reward low energy consumption
        # A genome that keeps the system alive on less energy is more fit,
        # just like a biological organism with lower metabolic rate has
        # an advantage in lean times.
        if tick_count > 0 and self._max_energy_observed > 0:
            avg_consumed = energy_consumed_total / tick_count
            efficiency_bonus = 1.0 - min(1.0, avg_consumed / self._max_energy_observed)
        else:
            efficiency_bonus = 0.5  # neutral if no data

        # Integration bonus: reward naturally emergent phi
        # This is the key philosophical point: phi is not a target to
        # maximize, it is a CONSEQUENCE of good integration. A genome
        # that produces high phi without being told to is genuinely
        # integrated, not just gaming a metric.
        if tick_count > 0:
            mean_phi = phi_total / tick_count
        else:
            mean_phi = 0.0
        phi_bonus = 1.0 + mean_phi * 0.5  # range [1.0, 1.5]

        # Behavioral fitness: how well the FCM's preferences correlated
        # with survival. This is a simplified metric — in a full EcoSim
        # implementation, we would track each action's contribution to
        # survival independently.
        behavioral_fitness = self._assess_behavioral_fitness(
            eval_behavior, self._sample_system_state()
        )

        # Final fitness: survival * efficiency * integration
        fitness_score = base_fitness * efficiency_bonus * phi_bonus

        # Update energy tracking for future evaluations
        with self._lock:
            if energy_consumed_total > 0:
                self._energy_observations.append(energy_consumed_total)
                if len(self._energy_observations) > 100:
                    self._energy_observations = self._energy_observations[-100:]
                self._max_energy_observed = max(
                    self._max_energy_observed,
                    max(self._energy_observations),
                )

            # Archive this genome for speciation analysis
            genome_vec = self._genome_to_vector(genome_params)
            if genome_vec is not None:
                self._genome_archive.append((genome_vec, fitness_score))
                if len(self._genome_archive) > self._archive_max_size:
                    self._genome_archive = self._genome_archive[-self._archive_max_size:]

            # Increment cycle count and check for speciation
            self._evolution_cycle_count += 1
            if self._evolution_cycle_count % self.cfg.speciation_check_interval == 0:
                self._detect_speciation()

        if crisis_reason is not None:
            logger.info(
                "Genome evaluation ended early: %s at t=%.1fs (fitness=%.4f)",
                crisis_reason.name, survival_time, fitness_score,
            )
        else:
            logger.debug(
                "Genome survived full window (fitness=%.4f, phi_mean=%.3f, "
                "efficiency=%.3f)",
                fitness_score, mean_phi, efficiency_bonus,
            )

        return FitnessResult(
            fitness_score=float(np.clip(fitness_score, 0.0, 2.0)),
            survival_time=survival_time,
            crisis_reason=crisis_reason,
            efficiency_bonus=efficiency_bonus,
            phi_bonus=phi_bonus,
            behavioral_fitness=behavioral_fitness,
        )

    # ------------------------------------------------------------------
    # System state sampling
    # ------------------------------------------------------------------

    def _sample_system_state(self) -> Dict[str, float]:
        """Read the current system state from live services.

        This reaches into the ServiceContainer to read real values from
        the running consciousness stack. If a service is unavailable,
        safe defaults are used (the system is assumed healthy unless we
        can prove otherwise — innocent until proven dead).
        """
        state: Dict[str, float] = {
            "energy": 50.0,       # safe default: mid-range
            "vitality": 0.8,      # safe default: healthy
            "threat_level": 0.0,  # safe default: no threat
            "free_energy": 0.3,   # safe default: moderate
            "entropy": 4.0,       # safe default: moderate
            "phi": 1.0,           # safe default: integrated
        }

        # Energy from the liquid substrate
        try:
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate is None:
                substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate is not None:
                # The substrate stores energy as a normalized value in its
                # state vector at idx_energy. Scale to raw units.
                raw_energy = float(getattr(substrate, "x", np.zeros(1))[
                    getattr(substrate, "idx_energy", 5)
                ])
                state["energy"] = raw_energy * 100.0  # scale to 0-100 range
        except Exception:
            pass

        # Vitality (autopoiesis) from homeostasis
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis is not None:
                state["vitality"] = float(homeostasis.compute_vitality())
        except Exception:
            pass

        # Threat level from anomaly detector / ICE layer
        try:
            ice = ServiceContainer.get("ice_layer", default=None)
            if ice is not None:
                state["threat_level"] = float(getattr(ice, "_threat_level", 0.0))
            else:
                anomaly = ServiceContainer.get("anomaly_detector", default=None)
                if anomaly is not None and hasattr(anomaly, "get_threat_level"):
                    state["threat_level"] = float(anomaly.get_threat_level())
        except Exception:
            pass

        # Free energy from the active inference engine
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine is not None:
                current = getattr(fe_engine, "_current", None)
                if current is not None:
                    state["free_energy"] = float(current.free_energy)
                else:
                    state["free_energy"] = float(
                        getattr(fe_engine, "_smoothed_fe", 0.3)
                    )
        except Exception:
            pass

        # Entropy from phi_core or substrate
        try:
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and hasattr(phi_core, "get_status"):
                phi_status = phi_core.get_status()
                state["entropy"] = float(phi_status.get("entropy", 4.0))
                state["phi"] = float(phi_status.get("phi", 0.0))
            else:
                # Fallback: read phi from substrate
                substrate = ServiceContainer.get("liquid_substrate", default=None)
                if substrate is not None:
                    state["phi"] = float(getattr(substrate, "_current_phi", 1.0))
        except Exception:
            pass

        return state

    def _get_behavioral_state_vector(self) -> np.ndarray:
        """Build the 7-dimensional state vector that the FCM reads.

        Maps the raw system state to the normalized [0, 1] dimensions
        that the behavioral genome expects.
        """
        raw = self._sample_system_state()

        vec = np.zeros(STATE_DIM, dtype=np.float32)
        vec[0] = float(np.clip(raw["energy"] / 100.0, 0.0, 1.0))     # energy
        vec[1] = 0.5  # social_hunger — from affect if available
        vec[2] = 0.5  # curiosity — from homeostasis if available
        vec[3] = 0.5  # competence — from recent success rate
        vec[4] = float(np.clip(1.0 - raw.get("phi", 1.0), 0.0, 1.0)) # coherence_need
        vec[5] = float(np.clip(raw["free_energy"], 0.0, 1.0))         # free_energy
        vec[6] = float(np.clip(raw["threat_level"], 0.0, 1.0))        # threat_level

        # Try to fill in social_hunger and curiosity from live services
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis is not None:
                vec[2] = float(np.clip(getattr(homeostasis, "curiosity", 0.5), 0.0, 1.0))
        except Exception:
            pass

        try:
            affect = ServiceContainer.get("affective_steering", default=None)
            if affect is None:
                affect = ServiceContainer.get("affect_facade", default=None)
            if affect is not None and hasattr(affect, "get_status"):
                status = affect.get_status()
                # Social hunger might be in the affect or homeostasis status
                sh = status.get("social_hunger", None)
                if sh is not None:
                    vec[1] = float(np.clip(sh, 0.0, 1.0))
        except Exception:
            pass

        return vec

    # ------------------------------------------------------------------
    # Behavioral fitness assessment
    # ------------------------------------------------------------------

    def _assess_behavioral_fitness(
        self,
        behavior: BehavioralGenome,
        state: Dict[str, float],
    ) -> float:
        """Assess how well the behavioral genome's preferences align with
        the system's current needs.

        A simple heuristic: if the system is in a crisis state and the
        FCM recommends the appropriate action, that is good behavioral
        fitness. If the system is healthy and the FCM recommends REST or
        EXPLORE, that is also good.

        Returns a score in [0, 1].
        """
        vec = self._get_behavioral_state_vector()
        prefs = behavior.compute_preferences(vec)

        # What SHOULD the system be doing based on its state?
        energy_norm = state.get("energy", 50.0) / 100.0
        threat = state.get("threat_level", 0.0)
        fe = state.get("free_energy", 0.3)

        # Ideal action preference (soft, not hard-coded)
        ideal = np.full(NUM_ACTIONS, 0.3, dtype=np.float32)
        if energy_norm < 0.3:
            ideal[Action.REST.value] = 0.9
            ideal[Action.REPAIR.value] = 0.7
        if threat > 0.5:
            ideal[Action.REPAIR.value] = 0.9
            ideal[Action.RESPOND.value] = 0.7
        if fe > 0.6:
            ideal[Action.REFLECT.value] = 0.8
            ideal[Action.EXPLORE.value] = 0.6
        if energy_norm > 0.7 and threat < 0.2:
            ideal[Action.EXPLORE.value] = 0.8
            ideal[Action.INITIATE.value] = 0.7

        # Normalize both to unit vectors and compute cosine similarity
        norm_prefs = prefs / (np.linalg.norm(prefs) + 1e-12)
        norm_ideal = ideal / (np.linalg.norm(ideal) + 1e-12)
        cosine_sim = float(np.dot(norm_prefs, norm_ideal))

        # Map from [-1, 1] to [0, 1]
        return float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Genome mutation
    # ------------------------------------------------------------------

    def mutate_behavioral_genome(
        self,
        genome: np.ndarray,
        rate: Optional[float] = None,
    ) -> np.ndarray:
        """Mutate a behavioral genome (FCM weight matrix).

        Each weight has a probability of `rate` of being perturbed by
        Gaussian noise. This is EcoSim-style behavioral mutation: the
        rules themselves evolve, not just the structural parameters.

        Behavioral genes mutate at a HIGHER rate (default 0.05) than
        structural genes (default 0.01) because behavioral adaptation
        needs to be faster — an organism that can quickly change its
        behavior in response to a new environment has an advantage.

        Args:
            genome: FCM weight matrix of shape (NUM_ACTIONS, STATE_DIM).
            rate: Per-gene mutation probability. Defaults to config value.

        Returns:
            New mutated weight matrix (the original is not modified).
        """
        rate = rate if rate is not None else self.cfg.behavioral_mutation_rate
        genome = np.array(genome, dtype=np.float32, copy=True)

        # Which genes mutate this generation?
        mask = self._rng.random(genome.shape) < rate
        noise = self._rng.standard_normal(genome.shape).astype(np.float32) * 0.1
        genome = genome + mask * noise

        # Clip to valid range
        return np.clip(genome, -1.0, 1.0).astype(np.float32)

    def mutate_structural_genome(
        self,
        genome_params: dict,
        rate: Optional[float] = None,
    ) -> dict:
        """Mutate structural genome parameters.

        Structural genes include kernel parameters (mu, sigma, beta,
        growth_mu, growth_sigma), inter-column weights, and neuromodulation
        baselines. These mutate at a LOWER rate than behavioral genes
        because structural changes are more disruptive — like changing
        the body plan versus changing behavior.

        Args:
            genome_params: Dictionary of structural parameters.
            rate: Per-gene mutation probability. Defaults to config value.

        Returns:
            New dictionary with mutated parameters (original not modified).
        """
        rate = rate if rate is not None else self.cfg.structural_mutation_rate
        result = {}

        for key, value in genome_params.items():
            if key == "behavioral_weights":
                # Behavioral genes use their own mutation rate
                result[key] = self.mutate_behavioral_genome(
                    np.asarray(value), self.cfg.behavioral_mutation_rate
                )
            elif isinstance(value, np.ndarray):
                # Array-valued structural gene: Gaussian perturbation
                arr = value.copy().astype(np.float32)
                mask = self._rng.random(arr.shape) < rate
                noise = self._rng.standard_normal(arr.shape).astype(np.float32)
                noise *= 0.02  # small perturbation for structural stability
                arr = arr + mask * noise
                result[key] = np.clip(arr, -1.0, 1.0)
            elif isinstance(value, (int, float)):
                # Scalar structural gene
                if self._rng.random() < rate:
                    perturbation = self._rng.standard_normal() * 0.02
                    result[key] = float(np.clip(value + perturbation, -1.0, 1.0))
                else:
                    result[key] = value
            else:
                result[key] = value

        return result

    # ------------------------------------------------------------------
    # Speciation detection (EcoSim)
    # ------------------------------------------------------------------

    def _detect_speciation(self) -> None:
        """Check if the population has diverged into distinct species.

        Runs k-means clustering for k=2..5 on the genome archive vectors,
        picks the k with the best silhouette score, and declares a
        speciation event if the silhouette exceeds the threshold.

        This prevents evolutionary collapse: if all genomes converge to
        a single strategy, there is no speciation, and the search is
        at risk of getting stuck. By protecting the best genome from
        each species, we maintain diversity.

        Must be called with self._lock held.
        """
        if len(self._genome_archive) < self.cfg.speciation_min_k * 3:
            # Not enough genomes to cluster meaningfully
            return

        # Build the data matrix from archived genome vectors
        X = np.array([vec for vec, _ in self._genome_archive], dtype=np.float32)
        fitnesses = np.array([fit for _, fit in self._genome_archive], dtype=np.float32)

        # Normalize features for fair distance computation
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        X_norm = (X - X.mean(axis=0)) / std

        best_k = 1
        best_score = -1.0
        best_labels = np.zeros(len(X_norm), dtype=np.int32)

        for k in range(self.cfg.speciation_min_k, self.cfg.speciation_max_k + 1):
            if k >= len(X_norm):
                break
            labels = _kmeans(X_norm, k, rng=self._rng)
            score = _silhouette_score(X_norm, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        # Compute species sizes and champions
        species_count = best_k if best_score > self.cfg.speciation_silhouette_threshold else 1
        sizes: List[int] = []
        champions: List[float] = []

        if species_count > 1:
            for s in range(best_k):
                mask = best_labels == s
                species_size = int(np.sum(mask))
                sizes.append(species_size)
                if species_size > 0:
                    champions.append(float(np.max(fitnesses[mask])))
                else:
                    champions.append(0.0)
            # Sort by size descending
            order = np.argsort(sizes)[::-1]
            sizes = [sizes[i] for i in order]
            champions = [champions[i] for i in order]
        else:
            sizes = [len(X_norm)]
            champions = [float(np.max(fitnesses)) if len(fitnesses) > 0 else 0.0]

        # Turnover: how much the species landscape changed
        prev_count = self._previous_species_count
        if prev_count > 0:
            turnover = abs(species_count - prev_count) / max(prev_count, species_count)
        else:
            turnover = 0.0

        info = SpeciesInfo(
            species_count=species_count,
            sizes=sizes,
            turnover_rate=turnover,
            silhouette_score=best_score,
            champions_per_species=champions,
        )

        self._species_labels = best_labels
        self._previous_species_count = species_count
        self._species_history.append(info)
        if len(self._species_history) > 50:
            self._species_history = self._species_history[-50:]

        if species_count > 1:
            logger.info(
                "Speciation detected: %d species (silhouette=%.3f, sizes=%s)",
                species_count, best_score, sizes,
            )

    def get_species_info(self) -> SpeciesInfo:
        """Get the current speciation state of the genome population.

        Returns the most recent speciation analysis, or a default
        single-species result if speciation detection has not yet run.
        """
        with self._lock:
            if self._species_history:
                return self._species_history[-1]
            return SpeciesInfo(
                species_count=1,
                sizes=[len(self._genome_archive)] if self._genome_archive else [0],
                turnover_rate=0.0,
                silhouette_score=0.0,
                champions_per_species=[0.0],
            )

    def get_niche_protected_genomes(
        self,
        genomes: List[Tuple[np.ndarray, float]],
    ) -> List[int]:
        """Return indices of genomes that should be protected via niche protection.

        For each species, the best-performing genome is protected from
        elimination. This ensures that even a small species with a unique
        strategy survives to compete in the next generation.

        Args:
            genomes: List of (genome_vector, fitness_score) tuples.

        Returns:
            List of indices into `genomes` that should be preserved.
        """
        with self._lock:
            if len(genomes) < self.cfg.speciation_min_k * 2:
                # Not enough genomes; protect the single best
                if not genomes:
                    return []
                best_idx = max(range(len(genomes)), key=lambda i: genomes[i][1])
                return [best_idx]

            # Cluster the provided genomes
            X = np.array([g[0] for g in genomes], dtype=np.float32)
            fitnesses = np.array([g[1] for g in genomes], dtype=np.float32)

            std = X.std(axis=0)
            std[std < 1e-8] = 1.0
            X_norm = (X - X.mean(axis=0)) / std

            best_k = 1
            best_score = -1.0
            best_labels = np.zeros(len(X_norm), dtype=np.int32)

            for k in range(self.cfg.speciation_min_k, min(self.cfg.speciation_max_k + 1, len(X_norm))):
                labels = _kmeans(X_norm, k, rng=self._rng)
                score = _silhouette_score(X_norm, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            protected: List[int] = []
            if best_score > self.cfg.speciation_silhouette_threshold:
                for s in range(best_k):
                    mask = best_labels == s
                    species_indices = np.where(mask)[0]
                    if len(species_indices) > 0:
                        best_in_species = species_indices[
                            np.argmax(fitnesses[species_indices])
                        ]
                        protected.append(int(best_in_species))
            else:
                # No real speciation — protect the global best
                protected.append(int(np.argmax(fitnesses)))

            return protected

    # ------------------------------------------------------------------
    # Behavioral genome access
    # ------------------------------------------------------------------

    def get_behavioral_genome(self) -> np.ndarray:
        """Return the current FCM weight matrix.

        This is the evolved behavioral strategy — the system's "personality"
        in terms of how it maps internal states to action preferences.

        Returns:
            Copy of the FCM weight matrix, shape (NUM_ACTIONS, STATE_DIM).
        """
        with self._lock:
            return self._behavioral_genome.weights.copy()

    def set_behavioral_genome(self, weights: np.ndarray) -> None:
        """Replace the current FCM weight matrix.

        Args:
            weights: New FCM matrix of shape (NUM_ACTIONS, STATE_DIM).
        """
        with self._lock:
            self._behavioral_genome = BehavioralGenome(
                weights=np.asarray(weights, dtype=np.float32),
                rng=self._rng,
            )

    def get_behavioral_preferences(self) -> Dict[str, float]:
        """Compute and return current action preferences from the FCM.

        Useful for debugging and visualization: shows what actions the
        evolved behavioral genome currently favors given the live system
        state.

        Returns:
            Dictionary mapping action names to preference scores (0-1).
        """
        vec = self._get_behavioral_state_vector()
        prefs = self._behavioral_genome.compute_preferences(vec)
        return {action.name: float(prefs[action.value]) for action in Action}

    # ------------------------------------------------------------------
    # Genome vector conversion for speciation
    # ------------------------------------------------------------------

    def _genome_to_vector(self, genome_params: dict) -> Optional[np.ndarray]:
        """Flatten a genome parameter dictionary into a 1D vector for clustering.

        Concatenates all numeric arrays and scalar values into a single
        vector that can be used for distance-based speciation detection.

        Returns None if the genome has no numeric content.
        """
        parts: List[np.ndarray] = []

        for key in sorted(genome_params.keys()):
            value = genome_params[key]
            if isinstance(value, np.ndarray):
                parts.append(value.flatten().astype(np.float32))
            elif isinstance(value, (int, float)):
                parts.append(np.array([float(value)], dtype=np.float32))

        if not parts:
            return None
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a status dictionary for dashboards and monitoring."""
        with self._lock:
            species = self._species_history[-1] if self._species_history else None
            return {
                "evolution_cycles": self._evolution_cycle_count,
                "genome_archive_size": len(self._genome_archive),
                "behavioral_alpha": round(self._behavioral_genome.alpha, 3),
                "species_count": species.species_count if species else 1,
                "silhouette_score": round(species.silhouette_score, 3) if species else 0.0,
                "max_energy_observed": round(self._max_energy_observed, 1),
                "behavioral_preferences": self.get_behavioral_preferences(),
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[EndogenousFitness] = None


def get_endogenous_fitness(
    config: EndogenousFitnessConfig | None = None,
) -> EndogenousFitness:
    """Get or create the module-level EndogenousFitness singleton.

    Args:
        config: Optional configuration. Only used on first call (when the
            singleton is created). Subsequent calls return the existing
            instance regardless of the config argument.

    Returns:
        The singleton EndogenousFitness instance.
    """
    global _instance
    if _instance is None:
        _instance = EndogenousFitness(config)
    return _instance
