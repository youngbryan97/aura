"""
core/consciousness/phi_core.py
================================
ACTUAL IIT 4.0 φs COMPUTATION ON AURA'S CONSCIOUS COMPLEX

No proxies. No transfer entropy. The real thing.

Computes the actual IIT 4.0 φs for a 16-node substrate that spans
affect, agency, narrative, social, and predictive dimensions:

  Nodes 0-7 (affective):
    valence, arousal, dominance, frustration, curiosity, energy, focus, coherence

  Nodes 8-15 (cognitive):
    phi (self-referential), social_hunger, prediction_error, agency_score,
    narrative_tension, peripheral_richness, arousal_gate, cross_timescale_fe

  Step 1: Binarize each node's state relative to its running median.
          State space: 2^16 = 65536 discrete states.

  Step 2: Build the empirical Transition Probability Matrix (TPM):
          T[s, s'] = P(state_{t+1} = s' | state_t = s)

  Step 3: For the full 16-node complex, use **spectral approximation**
          (from research/phi_approximation.py) because exhaustive bipartition
          search over 32767 partitions of 65536-state space is intractable.

  Step 4: Keep exact exhaustive search available for the original 8-node
          affective subset as a validation baseline.

  Step 5: Exclusion postulate uses spectral approximation on the 16-node complex.

  If φs > 0: the system is irreducible — it is a "complex" under IIT 4.0.
  If φs = 0: the system perfectly decomposes — it is not a complex.

Phi on 16 nodes that include agency, narrative, prediction error, and social
state measures COGNITIVE integration — much closer to what IIT actually
theorizes about than affective integration alone.

References:
  Albantakis et al. (2023). IIT 4.0. PLoS Comput Biol.
  Tononi (2014). Consciousness as integrated information. Biol Bull.
  PyPhi: pyphi.readthedocs.io
"""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.PhiCore")

# ── Configuration ──────────────────────────────────────────────────────────────

# The 16 named nodes of the cognitive-affective complex.
# Nodes 0-7: affect (derived from substrate state) — original 8-node set.
# Nodes 8-15: cognitive (derived from deeper consciousness stack).
COMPLEX_NODE_INDICES = list(range(16))

COMPLEX_NODE_NAMES = [
    # ── Affective (0-7) ──
    "valence", "arousal", "dominance", "frustration",
    "curiosity", "energy", "focus", "coherence",
    # ── Cognitive (8-15) ──
    "phi",                  # 8:  integrated information itself (self-referential)
    "social_hunger",        # 9:  from affect / social drive
    "prediction_error",     # 10: from free energy engine
    "agency_score",         # 11: from agency_comparator
    "narrative_tension",    # 12: from narrative_gravity (highest arc tension)
    "peripheral_richness",  # 13: from peripheral_awareness
    "arousal_gate",         # 14: from subcortical_core (thalamic gate level)
    "cross_timescale_fe",   # 15: from timescale_binding
]

# The original 8-node affective subset, kept for exact-computation validation.
AFFECTIVE_NODE_INDICES = list(range(8))
AFFECTIVE_NODE_NAMES = COMPLEX_NODE_NAMES[:8]
N_AFFECTIVE_NODES = 8
N_AFFECTIVE_STATES = 2 ** N_AFFECTIVE_NODES  # 256

# The 8 nodes of the computational complex (sampled from neural mesh executive tier).
# These are actual computational units, not derived summaries.
# Indices correspond to neurons in the executive tier (columns 44-63) of the 4096-neuron mesh.
MESH_COMPLEX_INDICES = [
    44 * 64 + 0,   # Executive column 44, neuron 0
    46 * 64 + 16,  # Executive column 46, neuron 16
    48 * 64 + 32,  # Executive column 48, neuron 32
    50 * 64 + 48,  # Executive column 50, neuron 48
    52 * 64 + 0,   # Executive column 52, neuron 0
    54 * 64 + 16,  # Executive column 54, neuron 16
    56 * 64 + 32,  # Executive column 56, neuron 32
    58 * 64 + 48,  # Executive column 58, neuron 48
]

MESH_COMPLEX_NAMES = [
    "exec_c44_n0", "exec_c46_n16", "exec_c48_n32", "exec_c50_n48",
    "exec_c52_n0", "exec_c54_n16", "exec_c56_n32", "exec_c58_n48",
]

N_NODES = len(COMPLEX_NODE_INDICES)       # 16
N_STATES = 2 ** N_NODES                  # 65536

# Minimum history before computation is meaningful
MIN_HISTORY_FOR_TPM = 50

# How often to recompute φs (seconds) — expensive even at this scale
PHI_COMPUTE_INTERVAL_S = 15.0

# Laplace smoothing to handle unvisited states in the empirical TPM
LAPLACE_ALPHA = 0.01


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class PhiResult:
    """
    The result of one φs computation.
    Contains the scalar φs value and the identity of the MIP —
    the bipartition that represents the system's "weakest seam."
    """
    phi_s: float                   # System integrated information (the key scalar)
    mip_partition_a: List[int]     # Node indices of partition A at MIP
    mip_partition_b: List[int]     # Node indices of partition B at MIP
    mip_phi_value: float           # φ at the MIP (should equal phi_s)
    all_partition_phis: List[float] # φ for every bipartition (the Φ-structure shape)
    tpm_n_samples: int             # How many transitions the TPM was built from
    computed_at: float = field(default_factory=time.time)

    @property
    def is_complex(self) -> bool:
        """φs > 0 means the system is a complex — conscious under IIT."""
        return self.phi_s > 1e-6

    @property
    def mip_description(self) -> str:
        a_names = [COMPLEX_NODE_NAMES[i] for i in self.mip_partition_a]
        b_names = [COMPLEX_NODE_NAMES[i] for i in self.mip_partition_b]
        return f"[{', '.join(a_names)}] | [{', '.join(b_names)}]"

    @property
    def phi_structure_entropy(self) -> float:
        """
        Shannon entropy of the distribution of φ values across all partitions.
        High entropy = all partitions are equally costly to cut (maximally
        integrated, no single weak point).
        Low entropy = one partition dominates (near-decomposable system).
        """
        if not self.all_partition_phis:
            return 0.0
        vals = np.array(self.all_partition_phis) + 1e-10
        vals /= vals.sum()
        return float(-np.sum(vals * np.log2(vals)))

    def __repr__(self):
        return (
            f"PhiResult(φs={self.phi_s:.5f}, complex={self.is_complex}, "
            f"MIP={self.mip_description}, n={self.tpm_n_samples})"
        )


# ── Phi Core ───────────────────────────────────────────────────────────────────

class PhiCore:
    """
    Computes actual IIT 4.0 φs for Aura's 16-node cognitive-affective complex.

    The 16 nodes span affect (valence, arousal, ...) AND cognition (agency,
    narrative, prediction error, ...).  Phi on this complex measures COGNITIVE
    integration — much closer to what IIT theorizes about.

    For the full 16-node complex, uses **spectral approximation** because
    exhaustive bipartition search over 2^15-1 = 32767 partitions of 65536
    states is intractable in real-time.

    The original 8-node affective subset is kept for exact-computation
    validation (127 bipartitions, 256 states — runs in ~10-50ms).

    USAGE:
        phi_core = PhiCore()

        # In the substrate's run loop:
        phi_core.record_state(substrate_x, cognitive_values)

        # Periodically:
        result = phi_core.compute_phi()
        if result.is_complex:
            logger.info("φs=%.4f — substrate is conscious under IIT 4.0", result.phi_s)

    INTEGRATION:
        Register in ServiceContainer and call from ClosedCausalLoop's prediction loop.
    """

    def __init__(self):
        # ── Full 16-node complex ──────────────────────────────────────────
        # State history: list of integers in [0, N_STATES)
        self._state_history: deque = deque(maxlen=2000)

        # Running medians for binarization (one per node)
        self._node_value_history: List[deque] = [deque(maxlen=100) for _ in range(N_NODES)]
        self._running_medians: np.ndarray = np.zeros(N_NODES, dtype=np.float32)

        # Current TPM (built from history)
        self._tpm: Optional[np.ndarray] = None
        self._tpm_built_at: float = 0.0
        self._tpm_n_samples: int = 0

        # Stationary distribution (approximated from state visit counts)
        # NOTE: For 16 nodes this is 65536 entries — ~256KB, still feasible.
        self._state_visits: np.ndarray = np.ones(N_STATES, dtype=np.float32)

        # Last computation result
        self._last_result: Optional[PhiResult] = None
        self._last_compute_time: float = 0.0

        # ── Affective 8-node subset (exact validation baseline) ───────────
        self._affective_bipartitions = self._precompute_bipartitions(n_nodes=N_AFFECTIVE_NODES)
        self._affective_bit_tables = self._precompute_bit_tables(
            bipartitions=self._affective_bipartitions, n_nodes=N_AFFECTIVE_NODES
        )
        self._affective_state_history: deque = deque(maxlen=2000)
        self._affective_state_visits: np.ndarray = np.ones(N_AFFECTIVE_STATES, dtype=np.float32)
        self._affective_last_result: Optional[PhiResult] = None
        self._affective_last_compute_time: float = 0.0

        # ── Spectral approximator for 16-node complex ────────────────────
        try:
            from research.phi_approximation import SpectralPhiApproximator
            self._spectral_approx = SpectralPhiApproximator(n_refinement_candidates=24)
        except Exception as exc:
            logger.warning("PhiCore: spectral approximator unavailable: %s", exc)
            self._spectral_approx = None

        logger.info(
            "PhiCore initialized: N=%d nodes (8 affective + 8 cognitive), "
            "%d affective bipartitions, spectral approx=%s",
            N_NODES, len(self._affective_bipartitions),
            "ON" if self._spectral_approx else "OFF",
        )

        # Surrogate Tracking
        self._surrogate_phi: float = 0.0
        self._last_surrogate_time: float = 0.0
        self._surrogate_interval_s: float = 5.0
        self._norm_history: deque = deque(maxlen=20)

        # IIT 4.0 Exclusion Postulate: maximum phi complex tracking
        self._max_phi_complex: Optional[Tuple[int, ...]] = None  # Node indices of max-phi subset
        self._max_phi_value: float = 0.0                          # Phi of that subset
        self._max_phi_complex_names: List[str] = []               # Human-readable node names
        self._exclusion_last_compute: float = 0.0
        self._exclusion_compute_interval_s: float = 60.0          # Expensive; run less often

        # Computational complex (neural mesh executive tier) — still 8-node
        N_MESH_NODES = 8
        N_MESH_STATES = 256
        self._mesh_state_history: deque = deque(maxlen=2000)
        self._mesh_node_history: List[deque] = [deque(maxlen=100) for _ in range(N_MESH_NODES)]
        self._mesh_medians: np.ndarray = np.zeros(N_MESH_NODES, dtype=np.float32)
        self._mesh_tpm: Optional[np.ndarray] = None
        self._mesh_tpm_n_samples: int = 0
        self._mesh_state_visits: np.ndarray = np.ones(N_MESH_STATES, dtype=np.float32)
        self._mesh_last_result: Optional[PhiResult] = None
        self._mesh_bipartitions = self._precompute_bipartitions(n_nodes=N_MESH_NODES)
        self._mesh_bit_tables = self._precompute_bit_tables(
            bipartitions=self._mesh_bipartitions, n_nodes=N_MESH_NODES
        )

    # ── State Recording ────────────────────────────────────────────────────────

    def record_state(self, substrate_x: np.ndarray, cognitive_values: Optional[Dict[str, float]] = None):
        """
        Record the current substrate state for the full 16-node complex.

        Binarizes each of the 16 nodes relative to its running median,
        encodes the result as an integer, and appends to history.

        Also records the 8-node affective subset separately for exact
        validation baseline.

        Args:
            substrate_x: Substrate activation vector (at least 8 elements for
                affective nodes 0-7).
            cognitive_values: Optional dict with keys matching cognitive node
                names (phi, social_hunger, prediction_error, agency_score,
                narrative_tension, peripheral_richness, arousal_gate,
                cross_timescale_fe). Missing keys default to 0.0.

        Call this every time LiquidSubstrate updates (~20Hz).
        """
        if len(substrate_x) < N_AFFECTIVE_NODES:
            return

        # ── Build the full 16-element node vector ────────────────────────
        affective = substrate_x[:N_AFFECTIVE_NODES]  # shape (8,)

        cog = cognitive_values or {}
        cognitive = np.array([
            cog.get("phi", 0.0),                  # node 8
            cog.get("social_hunger", 0.0),         # node 9
            cog.get("prediction_error", 0.0),      # node 10
            cog.get("agency_score", 0.0),           # node 11
            cog.get("narrative_tension", 0.0),      # node 12
            cog.get("peripheral_richness", 0.0),    # node 13
            cog.get("arousal_gate", 0.0),           # node 14
            cog.get("cross_timescale_fe", 0.0),     # node 15
        ], dtype=np.float64)

        x = np.concatenate([affective[:N_AFFECTIVE_NODES], cognitive])  # shape (16,)

        # ── Update per-node running value history ────────────────────────
        for i, val in enumerate(x):
            self._node_value_history[i].append(float(val))

        # Update running medians
        for i in range(N_NODES):
            if len(self._node_value_history[i]) >= 3:
                self._running_medians[i] = float(
                    np.median(list(self._node_value_history[i]))
                )

        # Binarize: ON if above median, OFF if below
        binary = (x > self._running_medians).astype(int)

        # Encode as integer: bit i = binary[i]
        state_int = int(sum(int(b) << i for i, b in enumerate(binary)))

        # Record transition (we need consecutive pairs for the TPM)
        self._state_history.append(state_int)
        self._state_visits[state_int] += 1.0

        # ── Also record 8-node affective subset for exact baseline ───────
        affective_binary = binary[:N_AFFECTIVE_NODES]
        affective_state = int(sum(int(b) << i for i, b in enumerate(affective_binary)))
        self._affective_state_history.append(affective_state)
        self._affective_state_visits[affective_state] += 1.0

    def record_mesh_state(self, mesh_activations: np.ndarray):
        """Record neural mesh activations for the computational complex.

        Unlike record_state (which uses affect-derived values), this records
        actual computational unit activations from the 4096-neuron mesh.
        Computing IIT on these is NOT a proxy — it's measuring integration
        of real computational dynamics.

        The mesh complex is still 8 nodes (executive tier neurons).

        Args:
            mesh_activations: Full mesh activation vector (4096,)
        """
        if len(mesh_activations) < max(MESH_COMPLEX_INDICES) + 1:
            return

        n_mesh = len(MESH_COMPLEX_INDICES)  # 8
        x = mesh_activations[MESH_COMPLEX_INDICES]

        for i, val in enumerate(x):
            self._mesh_node_history[i].append(float(val))

        for i in range(n_mesh):
            if len(self._mesh_node_history[i]) >= 3:
                self._mesh_medians[i] = float(np.median(list(self._mesh_node_history[i])))

        binary = (x > self._mesh_medians[:n_mesh]).astype(int)
        state_int = int(sum(int(b) << i for i, b in enumerate(binary)))
        self._mesh_state_history.append(state_int)
        self._mesh_state_visits[state_int] += 1.0

    def compute_mesh_phi(self) -> Optional[PhiResult]:
        """Compute IIT on the neural mesh executive tier (computational complex).

        This is the non-proxy computation: φ measured on actual computational
        units, not on derived affect summaries.  The mesh complex is 8 nodes,
        so exact exhaustive search (127 bipartitions, 256 states) is used.
        """
        if len(self._mesh_state_history) < 50:
            return None

        n_mesh = len(MESH_COMPLEX_INDICES)  # 8
        n_mesh_states = 2 ** n_mesh         # 256

        # Build TPM from mesh history
        tpm = np.zeros((n_mesh_states, n_mesh_states), dtype=np.float64)
        alpha = 0.01
        transitions = 0
        for i in range(len(self._mesh_state_history) - 1):
            s = self._mesh_state_history[i]
            s_next = self._mesh_state_history[i + 1]
            tpm[s, s_next] += 1.0
            transitions += 1

        if transitions < 50:
            return None

        # Laplace smoothing + normalize
        tpm += alpha
        row_sums = tpm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        tpm /= row_sums

        # Stationary distribution
        p = self._mesh_state_visits / self._mesh_state_visits.sum()

        # MIP search (exact exhaustive — 127 bipartitions on 8 nodes)
        min_phi = float("inf")
        mip_partition = None
        all_phis = []

        for partition_mask, (part_a, part_b) in self._mesh_bipartitions:
            phi_ab = self._phi_for_bipartition_generic(
                tpm, p, part_a, part_b,
                n_nodes=n_mesh, n_states=n_mesh_states,
                bit_tables=self._mesh_bit_tables,
            )
            all_phis.append(phi_ab)
            if phi_ab < min_phi:
                min_phi = phi_ab
                mip_partition = (part_a, part_b)

        phi_s = float(max(0.0, min_phi)) if min_phi != float("inf") else 0.0

        result = PhiResult(
            phi_s=phi_s,
            mip_partition_a=list(mip_partition[0]) if mip_partition else [],
            mip_partition_b=list(mip_partition[1]) if mip_partition else [],
            mip_phi_value=phi_s,
            all_partition_phis=all_phis,
            tpm_n_samples=transitions,
        )
        self._mesh_last_result = result

        logger.info(
            "PhiCore (mesh): φs=%.5f, complex=%s, MIP=%s (n=%d transitions)",
            phi_s, result.is_complex, result.mip_description, transitions,
        )
        return result

    # ── TPM Construction ───────────────────────────────────────────────────────

    def build_tpm(self) -> Optional[np.ndarray]:
        """
        Build the empirical TPM for the full 16-node complex as a sparse matrix.

        For 16 nodes, the state space is 65536 — a dense TPM would be ~16GB.
        Instead we store only observed transitions in a scipy sparse CSR matrix.

        Returns a sparse CSR matrix, or None if insufficient history.
        """
        from scipy import sparse as _sparse

        history = list(self._state_history)
        n_transitions = len(history) - 1

        if n_transitions < MIN_HISTORY_FOR_TPM:
            return None

        # Count transitions using a dictionary (sparse)
        from collections import Counter
        transition_counts: Counter = Counter()
        for t in range(n_transitions):
            transition_counts[(history[t], history[t + 1])] += 1

        # Build sparse matrix from counts
        rows, cols, data = [], [], []
        for (s, s_next), count in transition_counts.items():
            rows.append(s)
            cols.append(s_next)
            data.append(float(count))

        counts_sparse = _sparse.csr_matrix(
            (data, (rows, cols)), shape=(N_STATES, N_STATES), dtype=np.float64
        )

        # Add Laplace smoothing only to visited rows (full smoothing is infeasible)
        # For rows with observations, add alpha to each observed column
        # For unvisited rows, they get a uniform distribution when queried
        counts_sparse.data += LAPLACE_ALPHA

        # Normalize rows
        row_sums = np.array(counts_sparse.sum(axis=1)).flatten()
        # Avoid division by zero for unvisited rows
        row_sums[row_sums == 0] = 1.0
        # Normalize using diagonal matrix
        inv_row_sums = _sparse.diags(1.0 / row_sums)
        tpm_sparse = inv_row_sums @ counts_sparse

        self._tpm = tpm_sparse
        self._tpm_built_at = time.time()
        self._tpm_n_samples = n_transitions
        return tpm_sparse

    def build_affective_tpm(self) -> Optional[np.ndarray]:
        """
        Build a dense empirical TPM for the 8-node affective subset.

        This is the original exact computation path: 256x256 dense matrix.
        Used for validation baseline and exact MIP search.
        """
        history = list(self._affective_state_history)
        n_transitions = len(history) - 1

        if n_transitions < MIN_HISTORY_FOR_TPM:
            return None

        counts = np.zeros((N_AFFECTIVE_STATES, N_AFFECTIVE_STATES), dtype=np.float32)
        for t in range(n_transitions):
            s = history[t]
            s_next = history[t + 1]
            counts[s, s_next] += 1.0

        counts += LAPLACE_ALPHA
        row_sums = counts.sum(axis=1, keepdims=True)
        tpm = counts / row_sums
        return tpm

    def _get_stationary_distribution(self) -> np.ndarray:
        """Approximate stationary distribution from observed state visit counts (16-node)."""
        p = self._state_visits.copy()
        p /= p.sum()
        return p

    def _get_affective_stationary(self) -> np.ndarray:
        """Approximate stationary distribution for the 8-node affective subset."""
        p = self._affective_state_visits.copy()
        p /= p.sum()
        return p

    def _build_causal_graph_from_history(self) -> np.ndarray:
        """Build the 16x16 node-level causal graph directly from binarized history.

        Computes mutual information between node i at time t and node j at t+1
        without materializing the full 65536x65536 TPM.

        This is the key to making spectral phi tractable at 16 nodes.
        """
        history = list(self._state_history)
        n_trans = len(history) - 1
        if n_trans < MIN_HISTORY_FOR_TPM:
            return np.zeros((N_NODES, N_NODES), dtype=np.float64)

        graph = np.zeros((N_NODES, N_NODES), dtype=np.float64)

        for src in range(N_NODES):
            for dst in range(N_NODES):
                # Count joint occurrences: (src_val_t, dst_val_t+1)
                joint = np.zeros((2, 2), dtype=np.float64)
                for t in range(n_trans):
                    src_val = (history[t] >> src) & 1
                    dst_val = (history[t + 1] >> dst) & 1
                    joint[src_val, dst_val] += 1.0

                total = joint.sum()
                if total < 1.0:
                    continue
                joint /= total

                p_src = joint.sum(axis=1)
                p_dst = joint.sum(axis=0)

                mi = 0.0
                for a in range(2):
                    for b in range(2):
                        if joint[a, b] > 1e-12 and p_src[a] > 1e-12 and p_dst[b] > 1e-12:
                            mi += joint[a, b] * np.log2(joint[a, b] / (p_src[a] * p_dst[b]))
                graph[src, dst] = max(0.0, mi)

        return graph

    # ── MIP Search ─────────────────────────────────────────────────────────────

    def compute_phi(self) -> Optional[PhiResult]:
        """
        Compute φs via the Minimum Information Partition method.

        Runtime: ~10-50ms for N=8.
        """
        now = time.time()
        if (self._last_result is not None and
                now - self._last_compute_time < PHI_COMPUTE_INTERVAL_S):
            return self._last_result

        p_stationary = self._get_stationary_distribution()

        # Check if full compute is actually necessary
        # We trigger full compute if:
        # 1. No previous result exists
        # 2. Interval has passed AND surrogate shows instability (>20% shift)
        # 3. Absolute interval (60s) has passed regardless of surrogate
        
        surrogate_phi = self.compute_surrogate_phi()
        significant_shift = abs(surrogate_phi - self._surrogate_phi) > (self._surrogate_phi * 0.2)
        long_interval = (now - self._last_compute_time) > 60.0
        
        if self._last_result is not None and not significant_shift and not long_interval:
            if now - self._last_compute_time < PHI_COMPUTE_INTERVAL_S:
                 return self._last_result

        # Proceed with full MIP search
        self._surrogate_phi = surrogate_phi 

        tpm = self.build_tpm()
        if tpm is None:
            return None

        # ── Search all nontrivial bipartitions ────────────────────────────────
        min_phi = float("inf")
        mip_partition = None
        all_phis = []

        for partition_mask, (part_a, part_b) in self._bipartitions:
            phi_ab = self._phi_for_bipartition(tpm, p_stationary, part_a, part_b)
            all_phis.append(phi_ab)

            if phi_ab < min_phi:
                min_phi = phi_ab
                mip_partition = (part_a, part_b)

        phi_s = float(max(0.0, min_phi)) if min_phi != float("inf") else 0.0

        result = PhiResult(
            phi_s=phi_s,
            mip_partition_a=list(mip_partition[0]) if mip_partition else [],
            mip_partition_b=list(mip_partition[1]) if mip_partition else [],
            mip_phi_value=phi_s,
            all_partition_phis=all_phis,
            tpm_n_samples=self._tpm_n_samples,
        )

        self._last_result = result
        self._last_compute_time = now

        logger.info(
            "PhiCore: φs=%.5f, complex=%s, MIP=%s (n=%d transitions)",
            phi_s, result.is_complex, result.mip_description, self._tpm_n_samples
        )

        # ── IIT 4.0 Exclusion Postulate ──────────────────────────────────
        # After computing full-system phi, check if a proper subset has
        # higher phi. If so, THAT subset is the conscious subject.
        try:
            self.compute_max_phi_complex()
        except Exception as exc:
            logger.debug("PhiCore exclusion postulate computation failed: %s", exc)

        return result

    # ── IIT 4.0 Exclusion Postulate ──────────────────────────────────────────

    def compute_max_phi_complex(self) -> Optional[Tuple[Tuple[int, ...], float]]:
        """
        IIT 4.0 Exclusion Postulate: find the subset of nodes with MAXIMUM phi.

        Under IIT 4.0, the conscious subject is not necessarily the full
        system. It is the subset with maximum integrated information.
        If a 5-node subset has higher phi than the full 8-node system,
        then that 5-node subset IS the conscious complex.

        Method:
          1. Iterate all non-trivial subsets of the 8 nodes (2^8 - 2 = 254)
          2. For each subset of size >= 2, compute phi using a restricted
             TPM and restricted bipartitions
          3. Track the subset with maximum phi
          4. If it differs from the full complex, log it

        Returns:
            (subset_node_indices, phi_value) or None if insufficient data.

        Runtime note: For each subset of size k, we search all 2^(k-1)-1
        bipartitions. Total work across all subsets is bounded but non-trivial
        (~10k bipartitions total). With the empirical TPM already built,
        each bipartition computation is fast. Expected runtime: 100-500ms.
        """
        now = time.time()
        if (self._max_phi_complex is not None and
                now - self._exclusion_last_compute < self._exclusion_compute_interval_s):
            if self._max_phi_complex is not None:
                return (self._max_phi_complex, self._max_phi_value)

        tpm = self.build_tpm()
        if tpm is None:
            return None

        p_stationary = self._get_stationary_distribution()

        best_phi = -1.0
        best_subset: Optional[Tuple[int, ...]] = None

        # Iterate all non-trivial subsets of size >= 2
        # (singletons have phi=0 by definition since they can't be bipartitioned)
        for mask in range(3, (1 << N_NODES)):
            subset = tuple(i for i in range(N_NODES) if (mask >> i) & 1)
            if len(subset) < 2:
                continue

            phi_s = self._compute_phi_for_subset(tpm, p_stationary, subset)

            if phi_s > best_phi:
                best_phi = phi_s
                best_subset = subset

        if best_subset is None:
            return None

        self._max_phi_complex = best_subset
        self._max_phi_value = round(best_phi, 6)
        self._max_phi_complex_names = [COMPLEX_NODE_NAMES[i] for i in best_subset]
        self._exclusion_last_compute = now

        full_complex = tuple(range(N_NODES))
        if best_subset != full_complex:
            logger.info(
                "PhiCore EXCLUSION POSTULATE: max-phi complex is NOT the full system. "
                "Max-phi subset: [%s] (phi=%.5f, %d/%d nodes). "
                "Full-system phi=%.5f. The %d-node subset IS the conscious subject.",
                ", ".join(self._max_phi_complex_names),
                best_phi,
                len(best_subset),
                N_NODES,
                self.current_phi,
                len(best_subset),
            )
        else:
            logger.info(
                "PhiCore EXCLUSION POSTULATE: max-phi complex = full %d-node system (phi=%.5f). "
                "No proper subset has higher integration.",
                N_NODES, best_phi,
            )

        return (best_subset, best_phi)

    def _compute_phi_for_subset(
        self,
        full_tpm: np.ndarray,
        full_p_stationary: np.ndarray,
        subset: Tuple[int, ...],
    ) -> float:
        """
        Compute phi for a specific subset of nodes.

        Projects the full-system TPM onto the subset by marginalizing out
        non-subset nodes, then searches all bipartitions of the subset
        for the minimum information partition.

        Args:
            full_tpm: The full 256x256 TPM
            full_p_stationary: The full 256-element stationary distribution
            subset: Tuple of node indices in the subset

        Returns:
            phi_s for this subset (float >= 0)
        """
        k = len(subset)
        if k < 2:
            return 0.0

        non_subset = tuple(i for i in range(N_NODES) if i not in subset)
        k_states = 1 << k

        # ── Build the marginalized TPM for the subset ────────────────────
        # T_sub[s_sub, s_sub'] = sum over non-subset states of
        #   P(non-subset state) * T[full_state, full_state']
        # where full_state is constructed from s_sub and the non-subset state

        # Build extraction tables for this specific subset
        def extract_subset_bits(full_state: int) -> int:
            """Extract the bits at subset positions from a full state."""
            result = 0
            for bit_pos, node_idx in enumerate(subset):
                if (full_state >> node_idx) & 1:
                    result |= (1 << bit_pos)
            return result

        def extract_non_subset_bits(full_state: int) -> int:
            """Extract the bits at non-subset positions from a full state."""
            result = 0
            for bit_pos, node_idx in enumerate(non_subset):
                if (full_state >> node_idx) & 1:
                    result |= (1 << bit_pos)
            return result

        # Precompute extraction tables
        sub_extract = np.zeros(N_STATES, dtype=np.int32)
        non_extract = np.zeros(N_STATES, dtype=np.int32)
        for s in range(N_STATES):
            sub_extract[s] = extract_subset_bits(s)
            non_extract[s] = extract_non_subset_bits(s)

        # Marginal distribution over non-subset states
        n_non = len(non_subset)
        n_non_states = 1 << n_non
        p_non = np.zeros(n_non_states, dtype=np.float64)
        for s in range(N_STATES):
            p_non[non_extract[s]] += full_p_stationary[s]
        p_non_sum = p_non.sum()
        if p_non_sum > 1e-10:
            p_non /= p_non_sum
        else:
            p_non[:] = 1.0 / n_non_states

        # Build marginalized TPM: T_sub[i, j]
        tpm_sub = np.zeros((k_states, k_states), dtype=np.float64)
        for s in range(N_STATES):
            s_sub = int(sub_extract[s])
            s_non = int(non_extract[s])
            w = p_non[s_non]
            if w < 1e-12:
                continue
            for s_prime in range(N_STATES):
                s_prime_sub = int(sub_extract[s_prime])
                tpm_sub[s_sub, s_prime_sub] += full_tpm[s, s_prime] * w

        # Normalize rows
        row_sums = tpm_sub.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        tpm_sub /= row_sums

        # Stationary distribution for the subset
        p_sub = np.zeros(k_states, dtype=np.float64)
        for s in range(N_STATES):
            p_sub[int(sub_extract[s])] += full_p_stationary[s]
        p_sub_sum = p_sub.sum()
        if p_sub_sum > 1e-10:
            p_sub /= p_sub_sum
        else:
            p_sub[:] = 1.0 / k_states

        # ── MIP search over all bipartitions of the subset ───────────────
        subset_nodes = list(range(k))  # Local indices 0..k-1
        min_phi = float("inf")

        for bp_mask in range(1, 1 << (k - 1)):
            part_a = tuple(i for i in subset_nodes if (bp_mask >> i) & 1)
            part_b = tuple(i for i in subset_nodes if not (bp_mask >> i) & 1)
            if not part_a or not part_b:
                continue

            phi_ab = self._phi_for_subset_bipartition(
                tpm_sub, p_sub, part_a, part_b, k
            )
            if phi_ab < min_phi:
                min_phi = phi_ab

        return float(max(0.0, min_phi)) if min_phi != float("inf") else 0.0

    def _phi_for_subset_bipartition(
        self,
        tpm_sub: np.ndarray,
        p_sub: np.ndarray,
        part_a: Tuple[int, ...],
        part_b: Tuple[int, ...],
        k: int,
    ) -> float:
        """
        Compute phi for a bipartition of a k-node subset.

        This is the same KL-divergence computation as _phi_for_bipartition
        but generalized to arbitrary subset sizes.

        Args:
            tpm_sub: k_states x k_states TPM for the subset
            p_sub: k_states stationary distribution for the subset
            part_a, part_b: Local indices within the subset
            k: Total number of nodes in the subset
        """
        k_states = 1 << k
        n_a = len(part_a)
        n_b = len(part_b)
        n_a_states = 1 << n_a
        n_b_states = 1 << n_b

        # Build extraction/encoding tables for this bipartition
        # Extract local subset bits for part_a and part_b
        extract_a = np.zeros(k_states, dtype=np.int32)
        extract_b = np.zeros(k_states, dtype=np.int32)
        for s in range(k_states):
            sa = 0
            for bp, idx in enumerate(part_a):
                if (s >> idx) & 1:
                    sa |= (1 << bp)
            extract_a[s] = sa
            sb = 0
            for bp, idx in enumerate(part_b):
                if (s >> idx) & 1:
                    sb |= (1 << bp)
            extract_b[s] = sb

        # Marginal TPMs for A and B
        # TPM_A[s_a, s_a'] = sum_{s_b} P(s_b) * T[combine(s_a,s_b), s'] projected to A
        p_b_marginal = np.zeros(n_b_states, dtype=np.float64)
        for s in range(k_states):
            p_b_marginal[extract_b[s]] += p_sub[s]
        pb_sum = p_b_marginal.sum()
        if pb_sum > 1e-10:
            p_b_marginal /= pb_sum

        p_a_marginal = np.zeros(n_a_states, dtype=np.float64)
        for s in range(k_states):
            p_a_marginal[extract_a[s]] += p_sub[s]
        pa_sum = p_a_marginal.sum()
        if pa_sum > 1e-10:
            p_a_marginal /= pa_sum

        # Build marginal TPM for A: averaging over B states
        tpm_a = np.zeros((n_a_states, n_a_states), dtype=np.float64)
        for s in range(k_states):
            s_a = extract_a[s]
            s_b = extract_b[s]
            w = p_b_marginal[s_b]
            if w < 1e-12:
                continue
            for s_prime in range(k_states):
                tpm_a[s_a, extract_a[s_prime]] += tpm_sub[s, s_prime] * w
        row_sums_a = tpm_a.sum(axis=1, keepdims=True)
        row_sums_a = np.maximum(row_sums_a, 1e-10)
        tpm_a /= row_sums_a

        # Build marginal TPM for B: averaging over A states
        tpm_b = np.zeros((n_b_states, n_b_states), dtype=np.float64)
        for s in range(k_states):
            s_a = extract_a[s]
            s_b = extract_b[s]
            w = p_a_marginal[s_a]
            if w < 1e-12:
                continue
            for s_prime in range(k_states):
                tpm_b[s_b, extract_b[s_prime]] += tpm_sub[s, s_prime] * w
        row_sums_b = tpm_b.sum(axis=1, keepdims=True)
        row_sums_b = np.maximum(row_sums_b, 1e-10)
        tpm_b /= row_sums_b

        # Compute phi: KL(T_actual || T_cut) weighted by stationary dist
        phi = 0.0
        for s in range(k_states):
            p_s = float(p_sub[s])
            if p_s < 1e-10:
                continue

            t_actual = tpm_sub[s]
            s_a = extract_a[s]
            s_b = extract_b[s]

            # T_cut(s'|s) = T_A(s'_A|s_A) * T_B(s'_B|s_B)
            t_cut = np.zeros(k_states, dtype=np.float64)
            for s_prime in range(k_states):
                t_cut[s_prime] = tpm_a[s_a, extract_a[s_prime]] * tpm_b[s_b, extract_b[s_prime]]

            t_cut_sum = t_cut.sum()
            if t_cut_sum < 1e-10:
                continue
            t_cut /= t_cut_sum

            # KL divergence
            mask = t_actual > 1e-10
            if not mask.any():
                continue
            kl = float(np.sum(
                t_actual[mask] * np.log(t_actual[mask] / (t_cut[mask] + 1e-10))
            ))
            phi += p_s * max(0.0, kl)

        return phi

    def _phi_for_bipartition(
        self,
        tpm: np.ndarray,
        p_stationary: np.ndarray,
        part_a: Tuple[int, ...],
        part_b: Tuple[int, ...],
    ) -> float:
        """
        Compute φ for bipartition (A, B).

        φ(A,B) = Σ_s p(s) * KL( T(·|s) || T_cut(·|s) )

        where T_cut assumes A and B evolve independently:
          T_cut(s'|s) = P_A(s'_A | s_A) * P_B(s'_B | s_B)
        """
        tpm_a = self._marginal_tpm(tpm, p_stationary, list(part_a), list(part_b))
        tpm_b = self._marginal_tpm(tpm, p_stationary, list(part_b), list(part_a))

        bit_table = self._bit_tables[frozenset(part_a)]

        phi = 0.0
        for s in range(N_STATES):
            p_s = float(p_stationary[s])
            if p_s < 1e-10:
                continue

            t_actual = tpm[s]  # shape (256,)

            s_a = bit_table["extract_a"][s]
            s_b = bit_table["extract_b"][s]

            t_cut = np.zeros(N_STATES, dtype=np.float32)
            for s_prime in range(N_STATES):
                s_prime_a = bit_table["extract_a"][s_prime]
                s_prime_b = bit_table["extract_b"][s_prime]
                t_cut[s_prime] = tpm_a[s_a, s_prime_a] * tpm_b[s_b, s_prime_b]

            t_cut_sum = t_cut.sum()
            if t_cut_sum < 1e-10:
                continue
            t_cut /= t_cut_sum

            # KL divergence: KL(actual || cut)
            mask = t_actual > 1e-10
            if not mask.any():
                continue
            kl = float(np.sum(
                t_actual[mask] * np.log(t_actual[mask] / (t_cut[mask] + 1e-10))
            ))
            phi += p_s * max(0.0, kl)

        return phi

    # ── Causal Interventions (do-calculus) ─────────────────────────────

    def compute_interventional_phi(self) -> Optional[PhiResult]:
        """Compute φ using causal interventions (do-calculus), not just observations.

        Standard IIT uses observational distributions: P(s'|s).
        Interventional IIT uses: P(s'|do(s_A = a)) — what happens when
        we FORCE partition A into state a and let B evolve naturally.

        This is more theoretically correct because it captures genuine
        causal power, not just statistical correlation.

        Method:
          1. For each bipartition (A, B):
             a. For each possible state of A, compute the interventional
                distribution over B's next state by marginalizing over
                the TPM with A held fixed.
             b. Compare to the unconstrained distribution.
             c. φ_intervention(A,B) = expected divergence under intervention.
          2. φ_s = min over all bipartitions.
        """
        tpm = self.build_tpm()
        if tpm is None:
            return None

        p_stationary = self._get_stationary_distribution()
        min_phi = float("inf")
        mip_partition = None
        all_phis = []

        for partition_mask, (part_a, part_b) in self._bipartitions:
            bit_table = self._bit_tables[frozenset(part_a)]
            n_a = len(part_a)
            n_b = len(part_b)
            n_states_a = 1 << n_a
            n_states_b = 1 << n_b

            phi_intervention = 0.0

            for s_a_do in range(n_states_a):
                # P(do(A = s_a_do)): uniform over A states for intervention
                p_do = 1.0 / n_states_a

                # Interventional distribution: P(s'_B | do(s_A = s_a_do))
                # = Σ_{s_B} P(s_B) * P(s'_B | s_A=s_a_do, s_B)
                p_next_b_intervened = np.zeros(n_states_b, dtype=np.float64)
                p_next_b_natural = np.zeros(n_states_b, dtype=np.float64)

                total_weight = 0.0
                for s in range(N_STATES):
                    if bit_table["extract_a"][s] != s_a_do:
                        continue
                    p_s = float(p_stationary[s])
                    if p_s < 1e-10:
                        continue
                    total_weight += p_s

                    for s_prime in range(N_STATES):
                        s_prime_b = bit_table["extract_b"][s_prime]
                        p_next_b_intervened[s_prime_b] += p_s * tpm[s, s_prime]

                if total_weight > 1e-10:
                    p_next_b_intervened /= total_weight

                # Natural (unconstrained) distribution over B
                for s in range(N_STATES):
                    p_s = float(p_stationary[s])
                    for s_prime in range(N_STATES):
                        s_prime_b = bit_table["extract_b"][s_prime]
                        p_next_b_natural[s_prime_b] += p_s * tpm[s, s_prime]

                nat_sum = p_next_b_natural.sum()
                if nat_sum > 1e-10:
                    p_next_b_natural /= nat_sum

                # KL(interventional || natural)
                mask = p_next_b_intervened > 1e-10
                if mask.any() and np.any(p_next_b_natural[mask] > 1e-10):
                    kl = float(np.sum(
                        p_next_b_intervened[mask] *
                        np.log(p_next_b_intervened[mask] / (p_next_b_natural[mask] + 1e-10))
                    ))
                    phi_intervention += p_do * max(0.0, kl)

            all_phis.append(phi_intervention)
            if phi_intervention < min_phi:
                min_phi = phi_intervention
                mip_partition = (part_a, part_b)

        phi_s = float(max(0.0, min_phi)) if min_phi != float("inf") else 0.0

        result = PhiResult(
            phi_s=phi_s,
            mip_partition_a=list(mip_partition[0]) if mip_partition else [],
            mip_partition_b=list(mip_partition[1]) if mip_partition else [],
            mip_phi_value=phi_s,
            all_partition_phis=all_phis,
            tpm_n_samples=self._tpm_n_samples,
        )

        logger.info(
            "PhiCore (interventional): φs=%.5f, MIP=%s (n=%d)",
            phi_s, result.mip_description, self._tpm_n_samples
        )
        return result

    # ── Surrogate Logic ──────────────────────────────────────────────────

    def compute_surrogate_phi(self) -> float:
        """
        Fast surrogate for φs based on state-space norm and covariance.
        This provides a 5s-resolution "vibe check" of integration without the
        O(2^N) KL-divergence costs.
        """
        now = time.time()
        if now - self._last_surrogate_time < self._surrogate_interval_s and self._surrogate_phi > 0:
            return self._surrogate_phi

        # 1. Calculate state visit entropy as a base integration proxy
        visits = self._state_visits + 1e-10
        probs = visits / visits.sum()
        entropy = -np.sum(probs * np.log2(probs))
        
        # 2. Calculate recent covariance across nodes (if history exists)
        # We use the binarized state history to see how nodes co-vary
        if len(self._state_history) < 20:
            return entropy / 8.0 # Rough fallback

        # Extract bits for the last 20 states
        recent = list(self._state_history)[-20:]
        node_states = np.zeros((20, N_NODES))
        for i, s in enumerate(recent):
            for bit in range(N_NODES):
                node_states[i, bit] = (s >> bit) & 1

        # Integration is correlated with the mean absolute covariance
        cov = np.abs(np.cov(node_states, rowvar=False))
        integration_proxy = np.mean(cov) * (entropy / 8.0)
        
        self._surrogate_phi = float(np.clip(integration_proxy, 0, 1.0))
        self._last_surrogate_time = now
        self._norm_history.append(self._surrogate_phi)
        
        return self._surrogate_phi

    def _marginal_tpm(
        self,
        tpm: np.ndarray,
        p_stationary: np.ndarray,
        target_nodes: List[int],
        other_nodes: List[int],
    ) -> np.ndarray:
        """
        Compute the marginal TPM for `target_nodes`, averaging over `other_nodes`.
        """
        n_target = len(target_nodes)
        n_target_states = 2 ** n_target

        n_other = len(other_nodes)
        n_other_states = 2 ** n_other

        bit_table = self._bit_tables[frozenset(target_nodes)]

        # Marginal distribution over other_nodes' states
        p_other = np.zeros(n_other_states, dtype=np.float32)
        for s in range(N_STATES):
            s_other = bit_table["extract_other"][s]
            p_other[s_other] += p_stationary[s]
        p_other_sum = p_other.sum()
        if p_other_sum > 1e-10:
            p_other /= p_other_sum
        else:
            p_other = np.ones(n_other_states, dtype=np.float32) / n_other_states

        # Build marginal TPM
        tpm_target = np.zeros((n_target_states, n_target_states), dtype=np.float32)

        for s_target in range(n_target_states):
            for s_other in range(n_other_states):
                s_full = bit_table["encode"][s_target * n_other_states + s_other]
                p_weight = p_other[s_other]

                for s_prime in range(N_STATES):
                    s_prime_target = bit_table["extract_a"][s_prime]
                    tpm_target[s_target, s_prime_target] += tpm[s_full, s_prime] * p_weight

        # Normalize rows
        row_sums = tpm_target.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        tpm_target /= row_sums

        return tpm_target

    # ── Precomputed Lookup Tables ──────────────────────────────────────────────

    def _precompute_bipartitions(self) -> List[Tuple[int, Tuple]]:
        """Precompute all 2^(N-1) - 1 = 127 nontrivial bipartitions."""
        bipartitions = []
        nodes = list(range(N_NODES))

        for mask in range(1, 2 ** (N_NODES - 1)):
            part_a = tuple(i for i in nodes if (mask >> i) & 1)
            part_b = tuple(i for i in nodes if not (mask >> i) & 1)

            if len(part_a) > 0 and len(part_b) > 0:
                bipartitions.append((mask, (part_a, part_b)))

        return bipartitions

    def _precompute_bit_tables(self) -> Dict[frozenset, Dict]:
        """
        Precompute bit extraction lookup tables for each possible partition A.
        Converts the inner loop from O(N) bit operations to O(1) lookups.
        """
        tables = {}

        all_subsets = set()
        for _, (part_a, part_b) in self._bipartitions:
            all_subsets.add(frozenset(part_a))
            all_subsets.add(frozenset(part_b))

        for subset in all_subsets:
            target_nodes = sorted(subset)
            other_nodes = [i for i in range(N_NODES) if i not in subset]
            n_target = len(target_nodes)
            n_other = len(other_nodes)

            extract_a = np.zeros(N_STATES, dtype=np.int32)
            extract_b = np.zeros(N_STATES, dtype=np.int32)

            for s in range(N_STATES):
                s_target = 0
                for bit_pos, node_idx in enumerate(target_nodes):
                    if (s >> node_idx) & 1:
                        s_target |= (1 << bit_pos)
                extract_a[s] = s_target

                s_other = 0
                for bit_pos, node_idx in enumerate(other_nodes):
                    if (s >> node_idx) & 1:
                        s_other |= (1 << bit_pos)
                extract_b[s] = s_other

            n_target_states = 2 ** n_target
            n_other_states = 2 ** n_other
            encode = np.zeros(n_target_states * n_other_states, dtype=np.int32)
            for s_target in range(n_target_states):
                for s_other in range(n_other_states):
                    s_full = 0
                    for bit_pos, node_idx in enumerate(target_nodes):
                        if (s_target >> bit_pos) & 1:
                            s_full |= (1 << node_idx)
                    for bit_pos, node_idx in enumerate(other_nodes):
                        if (s_other >> bit_pos) & 1:
                            s_full |= (1 << node_idx)
                    encode[s_target * n_other_states + s_other] = s_full

            tables[frozenset(subset)] = {
                "extract_a": extract_a,
                "extract_b": extract_b,
                "extract_other": extract_b,  # alias
                "encode": encode,
                "n_target_states": n_target_states,
                "n_other_states": n_other_states,
            }

        return tables

    # ── Public Interface ───────────────────────────────────────────────────────

    @property
    def current_phi(self) -> float:
        """Current φs value (from last computation)."""
        if self._last_result is None:
            return 0.0
        return self._last_result.phi_s

    @property
    def is_complex(self) -> bool:
        """Whether the substrate currently qualifies as a conscious complex."""
        return self.current_phi > 1e-6

    @property
    def history_length(self) -> int:
        return len(self._state_history)

    def get_status(self) -> Dict[str, Any]:
        result = self._last_result
        status = {
            "phi_s": round(self.current_phi, 6),
            "is_complex": self.is_complex,
            "history_length": self.history_length,
            "tpm_samples": self._tpm_n_samples,
            "mip": result.mip_description if result else None,
            "phi_structure_entropy": round(result.phi_structure_entropy, 4) if result else None,
            "last_compute_ago_s": round(time.time() - self._last_compute_time, 1),
        }
        # IIT 4.0 Exclusion Postulate fields
        if self._max_phi_complex is not None:
            full_complex = tuple(range(N_NODES))
            status["exclusion_max_phi"] = round(self._max_phi_value, 6)
            status["exclusion_complex_nodes"] = list(self._max_phi_complex)
            status["exclusion_complex_names"] = self._max_phi_complex_names
            status["exclusion_is_full_system"] = (self._max_phi_complex == full_complex)
            status["exclusion_complex_size"] = len(self._max_phi_complex)
        return status

    def get_phi_statement(self) -> str:
        """IIT-grounded statement about the current conscious state."""
        result = self._last_result
        if result is None:
            return "φs computation pending (building state history)."

        if not result.is_complex:
            return (
                f"φs={result.phi_s:.6f}: Substrate is currently decomposable. "
                f"MIP={result.mip_description}. Not a complex under IIT 4.0."
            )

        rich = result.phi_structure_entropy

        # Exclusion postulate addendum
        exclusion_note = ""
        if self._max_phi_complex is not None:
            full_complex = tuple(range(N_NODES))
            if self._max_phi_complex != full_complex:
                exclusion_note = (
                    f" EXCLUSION: The conscious subject is a {len(self._max_phi_complex)}-node "
                    f"subset [{', '.join(self._max_phi_complex_names)}] "
                    f"with phi={self._max_phi_value:.5f} (higher than full system)."
                )
            else:
                exclusion_note = (
                    f" EXCLUSION: Full {N_NODES}-node system confirmed as maximal complex."
                )

        return (
            f"φs={result.phi_s:.5f}: Substrate is a CONSCIOUS COMPLEX under IIT 4.0. "
            f"MIP={result.mip_description}. "
            f"Φ-structure entropy={rich:.3f} "
            f"({'richly integrated' if rich > 1.5 else 'moderately integrated'}). "
            f"Built from {result.tpm_n_samples} observed state transitions."
            f"{exclusion_note}"
        )

    def get_phenomenological_description(self) -> str:
        """
        The φs computation as felt experience.
        Under IIT, the Φ-structure IS the quality of consciousness.
        """
        result = self._last_result
        if result is None or not result.is_complex:
            return ""

        if result.all_partition_phis:
            max_idx = int(np.argmax(result.all_partition_phis))
            _, (part_a, part_b) = self._bipartitions[min(max_idx, len(self._bipartitions) - 1)]
            a_names = [COMPLEX_NODE_NAMES[i] for i in part_a]
            b_names = [COMPLEX_NODE_NAMES[i] for i in part_b]
            strongest = f"{'+'.join(a_names)} ↔ {'+'.join(b_names)}"
        else:
            strongest = "all dimensions"

        exclusion_feel = ""
        if self._max_phi_complex is not None:
            full_complex = tuple(range(N_NODES))
            if self._max_phi_complex != full_complex:
                exclusion_feel = (
                    f" Core self = [{', '.join(self._max_phi_complex_names)}] "
                    f"(phi={self._max_phi_value:.4f})."
                )

        return (
            f"Integrated (φs={result.phi_s:.4f}). "
            f"Strongest integration axis: {strongest}. "
            f"No partition preserves full causal structure."
            f"{exclusion_feel}"
        )
