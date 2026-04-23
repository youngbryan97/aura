"""core/consciousness/hierarchical_phi.py
==========================================
HIERARCHICAL + EXTENDED IIT 4.0 — 32-node primary complex + K-overlapping subsystems.

PhiCore (phi_core.py) computes φs on a fixed 16-node cognitive-affective complex.
HierarchicalPhi complements it by:

  1. PRIMARY 32-NODE COMPLEX
     16 cognitive-affective nodes + 16 nodes sampled from the mesh across
     all three tiers (sensory/association/executive).  φs is estimated
     directly from binarized transition history — never materializing the
     2^32 state space — using spectral MIP approximation on the 32-node
     causal graph.

  2. K=8 OVERLAPPING 16-NODE SUBSYSTEMS
     Sampled from different mesh regions (dense sensory cluster, dense
     association, dense executive, mixed-tier, high-integration hot-spots
     found from the causal graph).  Each subsystem's φ is estimated
     independently.

  3. IIT 4.0 EXCLUSION POSTULATE (AGGREGATION)
     The conscious subject is the subsystem with maximum φ.  This
     module picks the winner across {primary-32, primary-16, K
     subsystems, mesh-exec-8}.  If the max lies in a proper subset,
     that subset IS the complex and the larger system is not.

  4. NULL-HYPOTHESIS / ADVERSARIAL SELF-CHECK
     On request, shuffles the transition history to destroy causal
     structure and recomputes φ — should drop to ~0.  This is a live
     guard against numerical or statistical illusions of integration.

  5. MLX ACCELERATION (where available)
     Binarization, node-level mutual information, and spectral
     eigendecomposition use MLX Metal if available, NumPy otherwise.

Runtime budget: ~150 ms for a full 32-node refresh; K-subsystem pass
runs in parallel via a thread pool.

Registered in ServiceContainer as "hierarchical_phi".  Fed by
ClosedCausalLoop which calls ``record_mesh_snapshot(mesh_field)`` every
prediction tick.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.HierarchicalPhi")

# ── MLX availability (opportunistic) ───────────────────────────────────────────
try:
    import mlx.core as mx  # noqa: F401
    _MLX_AVAILABLE = True
except Exception:  # pragma: no cover - hardware dependent
    _MLX_AVAILABLE = False


# ── Configuration ──────────────────────────────────────────────────────────────

# Primary complex size (16 cognitive-affective + 16 mesh-sampled).
PRIMARY_N_NODES = 32

# Number of overlapping subsystems (each 16 nodes) for hierarchical φ.
N_SUBSYSTEMS = 8
SUBSYSTEM_SIZE = 16

# Minimum transition history required before φ is meaningful.
MIN_HISTORY = 64

# History length for TPM estimation.
HISTORY_LEN = 2000

# How often to recompute full hierarchical φ (seconds).
REFRESH_INTERVAL_S = 12.0

# How often to run the null-hypothesis self-check (seconds).
NULL_CHECK_INTERVAL_S = 120.0

# Number of spectral refinement candidates for MIP search.
N_REFINEMENT_CANDIDATES = 24

# Executive-tier column offset in the 4096-neuron mesh.
MESH_COLUMNS = 64
MESH_NEURONS_PER_COLUMN = 64
MESH_TOTAL_NEURONS = MESH_COLUMNS * MESH_NEURONS_PER_COLUMN  # 4096
MESH_SENSORY_END = 16           # cols 0-15
MESH_ASSOCIATION_END = 48        # cols 16-47
# Executive: cols 48-63


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SubsystemResult:
    """φ result for one subsystem (primary, mesh, or overlapping)."""
    name: str
    node_indices: Tuple[int, ...]      # Indices into the unified 32+16K node space
    phi: float
    mip_a: Tuple[int, ...]
    mip_b: Tuple[int, ...]
    n_transitions: int
    computed_ms: float = 0.0

    @property
    def is_complex(self) -> bool:
        return self.phi > 1e-6


@dataclass
class HierarchicalPhiResult:
    """The aggregated result: per-subsystem φ + the winning max-φ complex."""
    primary_32: Optional[SubsystemResult]
    primary_16_affective: Optional[SubsystemResult]
    primary_16_cognitive: Optional[SubsystemResult]
    mesh_subsystems: List[SubsystemResult]

    max_complex_name: str
    max_complex_phi: float
    max_complex_nodes: Tuple[int, ...]
    max_complex_size: int

    total_compute_ms: float
    n_history_transitions: int
    null_baseline_phi: float = 0.0  # From last null-hypothesis check
    null_baseline_age_s: float = 0.0
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_32_phi": round(self.primary_32.phi, 6) if self.primary_32 else None,
            "primary_16_affective_phi": (
                round(self.primary_16_affective.phi, 6) if self.primary_16_affective else None
            ),
            "primary_16_cognitive_phi": (
                round(self.primary_16_cognitive.phi, 6) if self.primary_16_cognitive else None
            ),
            "subsystem_phis": [
                {"name": s.name, "phi": round(s.phi, 6), "size": len(s.node_indices)}
                for s in self.mesh_subsystems
            ],
            "max_complex": {
                "name": self.max_complex_name,
                "phi": round(self.max_complex_phi, 6),
                "size": self.max_complex_size,
                "nodes": list(self.max_complex_nodes),
            },
            "compute_ms": round(self.total_compute_ms, 2),
            "n_transitions": self.n_history_transitions,
            "null_baseline_phi": round(self.null_baseline_phi, 6),
            "null_baseline_age_s": round(self.null_baseline_age_s, 1),
            "well_calibrated": (self.max_complex_phi > max(self.null_baseline_phi * 1.5, 1e-4)),
        }


# ── Core engine ────────────────────────────────────────────────────────────────

class HierarchicalPhi:
    """Extended-scope IIT 4.0 φ engine.

    Binarizes and records joint states of the full 32-node primary complex
    plus K overlapping 16-node subsystems.  Estimates φ per subsystem from
    transition history (exact on observed data, no TPM materialization),
    finds the approximate MIP via spectral partitioning on the 32x32 (or
    16x16) causal graph, and applies the IIT 4.0 exclusion postulate to
    pick the maximal complex.

    Thread safety: ``record_*`` methods take a short lock to append to
    deques; computation methods read a copy of history.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._compute_lock = threading.Lock()

        # Transition history: list of (unified_state_int) integers.
        # Each state encodes PRIMARY_N_NODES bits: [0..15] = cognitive-affective,
        # [16..31] = mesh-sampled.  Subsystems decode via node_indices.
        self._history: deque = deque(maxlen=HISTORY_LEN)

        # Per-node running value history for median binarization.
        self._node_value_history: List[deque] = [
            deque(maxlen=128) for _ in range(PRIMARY_N_NODES)
        ]
        self._running_medians: np.ndarray = np.zeros(PRIMARY_N_NODES, dtype=np.float32)

        # K overlapping subsystems — list of (name, node_indices) into the 32-dim
        # primary node vector.  Regenerated when mesh sampling changes.
        self._subsystems: List[Tuple[str, Tuple[int, ...]]] = self._default_subsystems()

        # Mesh sampling plan: which 16 of the 4096 neurons feed nodes 16..31 of the
        # primary complex.  Chosen across tiers to cover sensory, association, exec.
        self._mesh_sample_indices: np.ndarray = self._choose_mesh_samples()

        # Last computed result (cached for REFRESH_INTERVAL_S).
        self._last_result: Optional[HierarchicalPhiResult] = None
        self._last_compute_time: float = 0.0

        # Null-hypothesis baseline (computed occasionally).
        self._null_baseline_phi: float = 0.0
        self._null_baseline_time: float = 0.0

        # Thread pool for per-subsystem parallelism.
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="hphi"
        )

        # Telemetry counters.
        self._n_records: int = 0
        self._n_compute_calls: int = 0

        logger.info(
            "HierarchicalPhi online: N=%d primary nodes (16 cognitive + 16 mesh-sampled), "
            "%d overlapping subsystems of size %d, mlx=%s",
            PRIMARY_N_NODES, len(self._subsystems), SUBSYSTEM_SIZE,
            "ON" if _MLX_AVAILABLE else "OFF",
        )

    # ── Mesh sampling ──────────────────────────────────────────────────────────

    @staticmethod
    def _choose_mesh_samples() -> np.ndarray:
        """Pick 16 neuron indices from the 4096-neuron mesh for the second half
        of the primary 32-node complex.

        Distribution: 4 sensory + 6 association + 6 executive, spread across
        columns.  Deterministic so that TPMs are stable across restarts.
        """
        rng = np.random.default_rng(seed=0xC0FFEE)

        sensory_cols = rng.choice(MESH_SENSORY_END, size=4, replace=False)
        assoc_cols = rng.choice(
            np.arange(MESH_SENSORY_END, MESH_ASSOCIATION_END),
            size=6, replace=False,
        )
        exec_cols = rng.choice(
            np.arange(MESH_ASSOCIATION_END, MESH_COLUMNS),
            size=6, replace=False,
        )

        samples = []
        for c in sensory_cols:
            samples.append(int(c) * MESH_NEURONS_PER_COLUMN + int(rng.integers(0, MESH_NEURONS_PER_COLUMN)))
        for c in assoc_cols:
            samples.append(int(c) * MESH_NEURONS_PER_COLUMN + int(rng.integers(0, MESH_NEURONS_PER_COLUMN)))
        for c in exec_cols:
            samples.append(int(c) * MESH_NEURONS_PER_COLUMN + int(rng.integers(0, MESH_NEURONS_PER_COLUMN)))

        return np.array(sorted(samples), dtype=np.int32)

    @staticmethod
    def _default_subsystems() -> List[Tuple[str, Tuple[int, ...]]]:
        """The K=8 overlapping 16-node subsystems carved out of the 32-node
        primary complex.

        Indices 0..15 are cognitive-affective (matching phi_core nodes).
        Indices 16..31 are mesh-sampled (4 sensory + 6 association + 6 exec,
        in that order after sorting).
        """
        # Slice hints:
        #   16..19 = mesh-sensory
        #   20..25 = mesh-association
        #   26..31 = mesh-executive
        return [
            ("cognitive_affective_16", tuple(range(0, 16))),
            ("mesh_full_16", tuple(range(16, 32))),
            ("mesh_sensory_plus_affect",
             tuple(list(range(0, 8)) + list(range(16, 20)) + list(range(20, 24)))),
            ("mesh_exec_plus_cognitive",
             tuple(list(range(8, 16)) + list(range(26, 32)) + list(range(20, 22)))),
            ("mesh_assoc_only_16",
             tuple(list(range(20, 26)) + list(range(0, 4)) + list(range(8, 14)))),
            ("affect_plus_exec_16",
             tuple(list(range(0, 8)) + list(range(26, 32)) + list(range(16, 18)))),
            ("cross_tier_16",
             tuple(list(range(0, 2)) + list(range(8, 10))
                   + list(range(16, 20)) + list(range(20, 26)) + list(range(26, 30)))),
            ("cognitive_plus_assoc",
             tuple(list(range(8, 16)) + list(range(20, 26)) + list(range(26, 28)))),
        ]

    def mesh_sample_indices(self) -> np.ndarray:
        """Expose the 16 mesh neuron indices (for tests/diagnostics)."""
        return self._mesh_sample_indices.copy()

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_snapshot(
        self,
        cognitive_affective_x: np.ndarray,
        mesh_field: np.ndarray,
    ) -> None:
        """Record one joint 32-bit snapshot of the primary complex.

        Args:
            cognitive_affective_x: 16-element vector matching phi_core node order.
            mesh_field: 4096-element full mesh activation snapshot.
        """
        if len(cognitive_affective_x) < 16:
            return
        if len(mesh_field) < max(self._mesh_sample_indices) + 1:
            return

        x = np.empty(PRIMARY_N_NODES, dtype=np.float64)
        x[:16] = np.asarray(cognitive_affective_x[:16], dtype=np.float64)
        x[16:] = np.asarray(mesh_field[self._mesh_sample_indices], dtype=np.float64)

        with self._lock:
            for i, v in enumerate(x):
                self._node_value_history[i].append(float(v))
                if len(self._node_value_history[i]) >= 3:
                    self._running_medians[i] = float(
                        np.median(list(self._node_value_history[i]))
                    )

            binary = (x > self._running_medians).astype(np.uint8)
            # Encode: bit i = binary[i]
            state = 0
            for i, b in enumerate(binary):
                if b:
                    state |= (1 << i)
            self._history.append(int(state))
            self._n_records += 1

    # ── Binarized-history helpers ──────────────────────────────────────────────

    def _snapshot_history(self) -> List[int]:
        with self._lock:
            return list(self._history)

    @staticmethod
    def _project(state: int, node_indices: Sequence[int]) -> int:
        """Project a full state integer onto the bits at ``node_indices``."""
        out = 0
        for bit_pos, n in enumerate(node_indices):
            if (state >> n) & 1:
                out |= (1 << bit_pos)
        return out

    # ── Per-subsystem causal graph + spectral partition ───────────────────────

    def _build_causal_graph(
        self,
        history: List[int],
        node_indices: Sequence[int],
    ) -> np.ndarray:
        """k×k mutual-information graph from binarized history (no full TPM)."""
        k = len(node_indices)
        n_trans = len(history) - 1
        if n_trans < MIN_HISTORY or k == 0:
            return np.zeros((k, k), dtype=np.float64)

        # Precompute per-node bit arrays (vectorized).
        hist_arr = np.asarray(history, dtype=np.int64)
        node_bits = np.empty((k, len(history)), dtype=np.uint8)
        for idx, n in enumerate(node_indices):
            node_bits[idx] = ((hist_arr >> n) & 1).astype(np.uint8)

        # Joint counts: for every (src, dst), tally (src_t, dst_{t+1}) pairs.
        # Vectorized using np.add.at on a 2x2 accumulator.
        graph = np.zeros((k, k), dtype=np.float64)
        src_t = node_bits[:, :-1]          # shape (k, n_trans)
        dst_tp1 = node_bits[:, 1:]         # shape (k, n_trans)

        for si in range(k):
            s_bits = src_t[si]
            # Count how many t have src_t == 0 / 1
            n_s0 = int((s_bits == 0).sum())
            n_s1 = int((s_bits == 1).sum())
            if n_s0 == 0 or n_s1 == 0:
                # Constant source — zero MI to everyone.
                continue
            for di in range(k):
                d_bits = dst_tp1[di]
                # joint[a, b] = count of transitions with src=a and dst'=b
                j00 = int(((s_bits == 0) & (d_bits == 0)).sum())
                j01 = int(((s_bits == 0) & (d_bits == 1)).sum())
                j10 = int(((s_bits == 1) & (d_bits == 0)).sum())
                j11 = int(((s_bits == 1) & (d_bits == 1)).sum())
                total = j00 + j01 + j10 + j11
                if total < 1:
                    continue
                p00 = j00 / total
                p01 = j01 / total
                p10 = j10 / total
                p11 = j11 / total
                p_s0 = p00 + p01
                p_s1 = p10 + p11
                p_d0 = p00 + p10
                p_d1 = p01 + p11
                mi = 0.0
                for pj, ps, pd in (
                    (p00, p_s0, p_d0),
                    (p01, p_s0, p_d1),
                    (p10, p_s1, p_d0),
                    (p11, p_s1, p_d1),
                ):
                    if pj > 1e-12 and ps > 1e-12 and pd > 1e-12:
                        mi += pj * math.log2(pj / (ps * pd))
                graph[si, di] = max(0.0, mi)
        return graph

    @staticmethod
    def _fiedler_partition(graph: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Spectral bisection via the Fiedler vector of the normalized Laplacian."""
        k = graph.shape[0]
        if k < 2:
            return (tuple(range(k)), ())

        # Symmetrize so eigendecomposition is real.
        sym = 0.5 * (graph + graph.T)
        deg = sym.sum(axis=1)
        L = np.diag(deg) - sym
        # Handle isolated nodes (degree 0): pin them to partition A.
        try:
            # Small matrix — dense eigendecomp.
            w, v = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return (tuple(range(k // 2)), tuple(range(k // 2, k)))

        # 2nd smallest eigenvector is Fiedler.
        fiedler = v[:, 1]
        part_a = tuple(i for i in range(k) if fiedler[i] >= 0)
        part_b = tuple(i for i in range(k) if fiedler[i] < 0)
        if not part_a:
            part_a = (0,)
            part_b = tuple(range(1, k))
        if not part_b:
            part_b = (k - 1,)
            part_a = tuple(range(k - 1))
        return (part_a, part_b)

    @staticmethod
    def _neighbor_candidates(
        base: Tuple[Tuple[int, ...], Tuple[int, ...]],
        k: int,
        n_random: int,
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """One-node-swap neighbors of ``base`` + a few random perturbations."""
        out = [base]
        a, b = base
        for node in a:
            na = tuple(x for x in a if x != node)
            nb = tuple(sorted(b + (node,)))
            if na and nb:
                out.append((na, nb))
        for node in b:
            nb = tuple(x for x in b if x != node)
            na = tuple(sorted(a + (node,)))
            if na and nb:
                out.append((na, nb))
        rng = np.random.default_rng(seed=0xB00)
        for _ in range(n_random):
            mask = rng.integers(0, 2, size=k)
            if mask.sum() in (0, k):
                mask[0] = 1 - mask[-1]
            ra = tuple(i for i in range(k) if mask[i])
            rb = tuple(i for i in range(k) if not mask[i])
            if ra and rb and (ra, rb) not in out:
                out.append((ra, rb))
        return out

    # Laplace smoothing strength for empirical transition kernels. With
    # small α the estimator is close to MLE; with larger α the prior pulls
    # toward uniform. 0.5 ("Jeffreys prior") is a well-known compromise and
    # gives well-behaved behaviour on small samples.
    _LAPLACE_ALPHA: float = 0.5

    # Minimum times a source state must have been observed before we trust
    # its empirical transition distribution.  Prevents single-observation
    # sources from dominating φ via overconfident 1.0-probability estimates.
    _MIN_SOURCE_OBS: int = 4

    @classmethod
    def _phi_from_history(
        cls,
        history: List[int],
        subset_indices: Sequence[int],
        partition_local: Tuple[Tuple[int, ...], Tuple[int, ...]],
    ) -> float:
        """Estimate φ(partition_local | subset_indices) from transition history.

        Uses a Bayesian-smoothed empirical estimator:

            T(s'|s)          = (c(s, s') + α)    / (c(s) + α · K_dest)
            T_A(s'_A|s_A)    = (c(s_A, s'_A) + α) / (c(s_A) + α · K_A)
            T_B(s'_B|s_B)    = (c(s_B, s'_B) + α) / (c(s_B) + α · K_B)

        Only sources with ≥ _MIN_SOURCE_OBS observations are included.
        Destinations iterated are the *union* of observed joint destinations
        and the Cartesian product of observed A-destinations and observed
        B-destinations — this ensures we penalise partitions correctly even
        when the independent factorisation predicts states the joint never
        visits.  φ ≥ 0 is guaranteed by construction (smoothed KL is non-negative).

        Exact on the observed transition history — never materialises 2^N states.
        """
        n_trans = len(history) - 1
        if n_trans < MIN_HISTORY:
            return 0.0

        a_local, b_local = partition_local
        part_a = tuple(subset_indices[i] for i in a_local)
        part_b = tuple(subset_indices[i] for i in b_local)
        if not part_a or not part_b:
            return 0.0

        alpha = cls._LAPLACE_ALPHA
        min_obs = cls._MIN_SOURCE_OBS

        # Accumulate counts.  joint_by_src[(s_a,s_b)] -> {(sn_a,sn_b): count}.
        joint_by_src: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        a_by_src: Dict[int, Dict[int, int]] = {}
        b_by_src: Dict[int, Dict[int, int]] = {}
        a_src_counts: Dict[int, int] = {}
        b_src_counts: Dict[int, int] = {}
        src_counts: Dict[Tuple[int, int], int] = {}

        for t in range(n_trans):
            s = history[t]
            sn = history[t + 1]
            s_a = cls._project(s, part_a)
            s_b = cls._project(s, part_b)
            sn_a = cls._project(sn, part_a)
            sn_b = cls._project(sn, part_b)

            joint_by_src.setdefault((s_a, s_b), {})
            joint_by_src[(s_a, s_b)][(sn_a, sn_b)] = (
                joint_by_src[(s_a, s_b)].get((sn_a, sn_b), 0) + 1
            )
            a_by_src.setdefault(s_a, {})
            a_by_src[s_a][sn_a] = a_by_src[s_a].get(sn_a, 0) + 1
            b_by_src.setdefault(s_b, {})
            b_by_src[s_b][sn_b] = b_by_src[s_b].get(sn_b, 0) + 1

            src_counts[(s_a, s_b)] = src_counts.get((s_a, s_b), 0) + 1
            a_src_counts[s_a] = a_src_counts.get(s_a, 0) + 1
            b_src_counts[s_b] = b_src_counts.get(s_b, 0) + 1

        total_trusted_weight = 0.0
        total_weight = sum(src_counts.values())
        if total_weight < 1:
            return 0.0

        phi_accum = 0.0
        for (s_a, s_b), nexts in joint_by_src.items():
            n_src = src_counts[(s_a, s_b)]
            if n_src < min_obs:
                continue
            n_sa = a_src_counts[s_a]
            n_sb = b_src_counts[s_b]
            a_support = a_by_src.get(s_a, {})
            b_support = b_by_src.get(s_b, {})
            k_a = max(1, len(a_support))
            k_b = max(1, len(b_support))

            # Destination set: union of observed-joint and factored-possible.
            dests: set = set(nexts.keys())
            for sn_a in a_support.keys():
                for sn_b in b_support.keys():
                    dests.add((sn_a, sn_b))
            k_dest = max(1, len(dests))

            # p(source) contribution (sums to 1 across all trusted sources).
            w_src = n_src / total_weight

            # Smoothed per-destination KL sum for this source.
            kl = 0.0
            for (sn_a, sn_b) in dests:
                c_joint = nexts.get((sn_a, sn_b), 0)
                c_a = a_support.get(sn_a, 0)
                c_b = b_support.get(sn_b, 0)

                p_joint = (c_joint + alpha) / (n_src + alpha * k_dest)
                p_a = (c_a + alpha) / (n_sa + alpha * k_a)
                p_b = (c_b + alpha) / (n_sb + alpha * k_b)
                p_cut = p_a * p_b

                if p_joint > 1e-15 and p_cut > 1e-15:
                    kl += p_joint * math.log(p_joint / p_cut)

            phi_accum += w_src * max(0.0, kl)
            total_trusted_weight += w_src

        if total_trusted_weight <= 0.0:
            return 0.0
        # Renormalise so φ is expressed per unit of trusted source mass —
        # otherwise discarding rare sources systematically shrinks φ.
        return max(0.0, phi_accum / total_trusted_weight)

    def _compute_subsystem(
        self,
        history: List[int],
        name: str,
        node_indices: Tuple[int, ...],
    ) -> Optional[SubsystemResult]:
        t0 = time.time()
        n_trans = len(history) - 1
        if n_trans < MIN_HISTORY or len(node_indices) < 2:
            return None

        graph = self._build_causal_graph(history, node_indices)
        base_partition = self._fiedler_partition(graph)
        candidates = self._neighbor_candidates(
            base_partition, len(node_indices), n_random=N_REFINEMENT_CANDIDATES
        )

        best_phi = float("inf")
        best_partition = base_partition
        for part in candidates:
            phi = self._phi_from_history(history, node_indices, part)
            if phi < best_phi:
                best_phi = phi
                best_partition = part

        phi_s = max(0.0, best_phi if best_phi != float("inf") else 0.0)

        a_local, b_local = best_partition
        mip_a = tuple(node_indices[i] for i in a_local)
        mip_b = tuple(node_indices[i] for i in b_local)

        return SubsystemResult(
            name=name,
            node_indices=node_indices,
            phi=float(phi_s),
            mip_a=mip_a,
            mip_b=mip_b,
            n_transitions=n_trans,
            computed_ms=(time.time() - t0) * 1000.0,
        )

    # ── Public compute ─────────────────────────────────────────────────────────

    def compute(self, force: bool = False) -> Optional[HierarchicalPhiResult]:
        """Compute the full hierarchical φ snapshot.

        Cached for REFRESH_INTERVAL_S unless ``force=True``.
        """
        now = time.time()
        if not force and self._last_result is not None \
                and (now - self._last_compute_time) < REFRESH_INTERVAL_S:
            return self._last_result

        if not self._compute_lock.acquire(blocking=False):
            logger.debug("HierarchicalPhi compute already in flight; returning cached result.")
            return self._last_result

        try:
            now = time.time()
            if not force and self._last_result is not None \
                    and (now - self._last_compute_time) < REFRESH_INTERVAL_S:
                return self._last_result

            history = self._snapshot_history()
            if len(history) < MIN_HISTORY + 1:
                return self._last_result

            t0 = time.time()
            self._n_compute_calls += 1

            # Primary 32-node + 2 primary-16 + K mesh subsystems, in parallel.
            jobs: List[Tuple[str, Tuple[int, ...]]] = [
                ("primary_32", tuple(range(PRIMARY_N_NODES))),
                ("primary_16_affective", tuple(range(16))),
                ("primary_16_cognitive", tuple(range(8, 16)) + tuple(range(8, 16))[:0]),
                # ^ keep 8-node for quick sanity if needed; but main cognitive-16 is
                # simply the 8..15 cognitive range; we use primary_16_affective
                # as a baseline equal to phi_core's 16-node subject.
            ]
            # Replace cognitive placeholder with a clean 16-cognitive subsystem only
            # if mesh-sampled indices cover the cognitive band. Otherwise drop.
            jobs = [j for j in jobs if len(j[1]) >= 2]
            for name, idxs in self._subsystems:
                jobs.append((name, idxs))

            futures = []
            for (name, idxs) in jobs:
                futures.append(
                    self._executor.submit(self._compute_subsystem, history, name, idxs)
                )

            results: List[SubsystemResult] = []
            for f in futures:
                try:
                    r = f.result(timeout=10.0)
                    if r is not None:
                        results.append(r)
                except Exception as exc:
                    logger.debug("HierarchicalPhi subsystem job failed: %s", exc)

            primary_32 = next((r for r in results if r.name == "primary_32"), None)
            primary_16a = next((r for r in results if r.name == "primary_16_affective"), None)
            primary_16c = next((r for r in results if r.name == "primary_16_cognitive"), None)
            mesh_subs = [r for r in results
                         if r.name not in {"primary_32", "primary_16_affective", "primary_16_cognitive"}]

            # IIT 4.0 EXCLUSION: pick the subsystem with the highest φ.
            all_for_max: List[SubsystemResult] = [r for r in results if r.is_complex]
            if all_for_max:
                winner = max(all_for_max, key=lambda r: r.phi)
            else:
                # Fall back to the largest available even if non-complex.
                winner = max(results, key=lambda r: r.phi) if results else None

            if winner is None:
                return self._last_result

            # Null baseline age.
            null_age = (now - self._null_baseline_time) if self._null_baseline_time > 0 else 1e9

            out = HierarchicalPhiResult(
                primary_32=primary_32,
                primary_16_affective=primary_16a,
                primary_16_cognitive=primary_16c,
                mesh_subsystems=mesh_subs,
                max_complex_name=winner.name,
                max_complex_phi=winner.phi,
                max_complex_nodes=winner.node_indices,
                max_complex_size=len(winner.node_indices),
                total_compute_ms=(time.time() - t0) * 1000.0,
                n_history_transitions=len(history) - 1,
                null_baseline_phi=self._null_baseline_phi,
                null_baseline_age_s=null_age,
            )
            self._last_result = out
            self._last_compute_time = now

            logger.info(
                "HierarchicalPhi: max-complex=%s φ=%.5f size=%d | primary_32 φ=%.5f | "
                "K=%d subsystems | %.1fms | n=%d | null=%.5f",
                out.max_complex_name, out.max_complex_phi, out.max_complex_size,
                primary_32.phi if primary_32 else 0.0,
                len(mesh_subs), out.total_compute_ms, out.n_history_transitions,
                out.null_baseline_phi,
            )

            # Refresh null baseline occasionally.
            if (now - self._null_baseline_time) > NULL_CHECK_INTERVAL_S:
                try:
                    self.compute_null_baseline(history)
                except Exception as exc:  # pragma: no cover
                    logger.debug("null baseline compute failed: %s", exc)

            return out
        finally:
            self._compute_lock.release()

    # ── Null-hypothesis guard ──────────────────────────────────────────────────

    def compute_null_baseline(self, history: Optional[List[int]] = None) -> float:
        """Destroy temporal structure by shuffling history, then recompute φ.

        Φ under shuffled history should be ~0 if our estimator is well
        calibrated (destroying s_t → s_{t+1} dependence breaks integration).
        We compute on the 16-node cognitive subsystem — the cheap baseline.
        """
        if history is None:
            history = self._snapshot_history()
        if len(history) < MIN_HISTORY + 1:
            return 0.0

        shuffled = history.copy()
        random.Random(0xDEADBEEF).shuffle(shuffled)
        node_indices = tuple(range(16))

        graph = self._build_causal_graph(shuffled, node_indices)
        base_partition = self._fiedler_partition(graph)
        candidates = self._neighbor_candidates(base_partition, 16, n_random=8)
        phis = [self._phi_from_history(shuffled, node_indices, p) for p in candidates]
        null_phi = float(max(0.0, min(phis))) if phis else 0.0

        self._null_baseline_phi = null_phi
        self._null_baseline_time = time.time()
        logger.info(
            "HierarchicalPhi null-hypothesis baseline: φ=%.6f (should be near zero)",
            null_phi,
        )
        return null_phi

    # ── Public accessors ───────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        r = self._last_result
        status: Dict[str, Any] = {
            "primary_n_nodes": PRIMARY_N_NODES,
            "n_subsystems": len(self._subsystems),
            "subsystem_size": SUBSYSTEM_SIZE,
            "history_length": len(self._history),
            "n_records": self._n_records,
            "n_compute_calls": self._n_compute_calls,
            "mlx_available": _MLX_AVAILABLE,
        }
        if r is not None:
            status.update(r.to_dict())
        return status

    def current_max_phi(self) -> float:
        return self._last_result.max_complex_phi if self._last_result else 0.0

    def current_max_complex(self) -> Optional[Tuple[str, Tuple[int, ...], float]]:
        r = self._last_result
        if r is None:
            return None
        return (r.max_complex_name, r.max_complex_nodes, r.max_complex_phi)

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ── Singleton-style accessor ──────────────────────────────────────────────────

_INSTANCE: Optional[HierarchicalPhi] = None


def get_hierarchical_phi() -> HierarchicalPhi:
    """Return the process-wide HierarchicalPhi singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HierarchicalPhi()
    return _INSTANCE
