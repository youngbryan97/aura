"""core/consciousness/plasticity_monitor.py
─────────────────────────────────────────────
Plasticity monitor for the LiquidSubstrate's recurrent connectivity matrix.

Background
----------
The arXiv preprint "Barriers for Learning in an Evolving World: Mathematical
Understanding of Loss of Plasticity" (2510.00304, 2025) provides a
theoretical analysis of why neural networks under continual learning lose
adaptive capacity. The core observation: as training continues on a
non-stationary stream, the weight matrix's singular-value spectrum
collapses toward a low-rank structure. Effective rank drops; new
information cannot reorganize the representation because the geometry has
flattened.

Why this fits Aura
------------------
Aura's STDP system updates the LiquidSubstrate's W matrix every 100 ticks
using prediction-error reward (see ARCHITECTURE.md §7). The
"closed-loop" concern flagged in the technical-critique-response is
precisely about whether this update is doing useful work or chasing its
own outputs. The plasticity monitor doesn't fully resolve that question
(external-input ablation is still needed), but it gives a *direct* read
on whether the weight matrix is losing its capacity to adapt.

What this module does
---------------------
1. Periodic SVD on the W matrix (cheap for 64×64; bounded for the
   neural-mesh up to a few thousand nodes).
2. Computes effective rank via the "stable rank" metric:
       stable_rank(W) = ||W||_F^2 / ||W||_2^2
   = sum of squared singulars / largest squared singular.
   This is the standard plasticity measure used in the continual-learning
   literature; it's stable under small perturbations and reveals collapse
   even when nominal rank stays full.
3. Tracks plasticity history; warns when stable rank drops below a
   configurable floor for a sustained number of measurements.
4. Exposes a per-call and aggregate report.

Defensive: bounded matrix size, swallow SVD failures (very rare on numeric
matrices), graceful when W is unset or wrong shape.

Pure numpy.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Aura.PlasticityMonitor")

# Bounded pool for heavy linear-algebra work (SVD).  Shared across all
# PlasticityMonitor instances so total thread count stays predictable.
_SVD_POOL: ThreadPoolExecutor | None = None


def _get_svd_pool() -> ThreadPoolExecutor:
    global _SVD_POOL
    if _SVD_POOL is None:
        _SVD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aura-svd")
    return _SVD_POOL

# ── Tunables ──────────────────────────────────────────────────────────────

DEFAULT_HISTORY = 32
MAX_MATRIX_DIM = 4096        # refuse SVD on matrices larger than this
PLASTICITY_FLOOR_RATIO = 0.25  # warn when stable_rank / nominal_rank < this
SUSTAINED_BREACHES_FOR_WARN = 3


@dataclass
class PlasticityReport:
    nominal_rank: int            # min(W.shape)
    stable_rank: float           # ||W||_F^2 / ||W||_2^2
    stable_rank_ratio: float     # stable_rank / nominal_rank — in [0, 1]
    largest_singular: float
    smallest_significant_singular: float
    collapse_warning: bool
    sustained_breaches: int      # how many recent measurements are below floor
    measurement_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nominal_rank": int(self.nominal_rank),
            "stable_rank": round(float(self.stable_rank), 3),
            "stable_rank_ratio": round(float(self.stable_rank_ratio), 3),
            "largest_singular": round(float(self.largest_singular), 4),
            "smallest_significant_singular": round(float(self.smallest_significant_singular), 4),
            "collapse_warning": bool(self.collapse_warning),
            "sustained_breaches": int(self.sustained_breaches),
            "measurement_count": int(self.measurement_count),
        }


class PlasticityMonitor:
    def __init__(
        self,
        history: int = DEFAULT_HISTORY,
        floor_ratio: float = PLASTICITY_FLOOR_RATIO,
        sustained_for_warn: int = SUSTAINED_BREACHES_FOR_WARN,
    ) -> None:
        self._floor = float(floor_ratio)
        self._sustained_for_warn = int(sustained_for_warn)
        self._ratio_history: Deque[float] = deque(maxlen=int(history))
        self._measurement_count: int = 0
        self._last_warning_at: int = -1  # measurement index of most recent warning

    # ── Main API ──────────────────────────────────────────────────────────

    def measure(self, W: np.ndarray) -> Optional[PlasticityReport]:
        """Compute a plasticity report for the given weight matrix. Returns
        None on degenerate input (None, wrong shape, oversized).
        """
        if W is None:
            return None
        try:
            mat = np.asarray(W, dtype=np.float64)
        except Exception as e:
            logger.debug("plasticity: cast failed: %s", e)
            return None
        if mat.ndim != 2:
            logger.debug("plasticity: W is not 2-D (ndim=%d)", mat.ndim)
            return None
        m, n = mat.shape
        if m == 0 or n == 0:
            return None
        if max(m, n) > MAX_MATRIX_DIM:
            logger.debug("plasticity: refusing SVD on %dx%d matrix (cap=%d)",
                         m, n, MAX_MATRIX_DIM)
            return None

        # SVD — robust path
        try:
            singulars = np.linalg.svd(mat, compute_uv=False)
        except np.linalg.LinAlgError as e:
            logger.debug("plasticity: SVD failed: %s", e)
            return None

        if singulars.size == 0:
            return None

        s = np.asarray(singulars, dtype=np.float64)
        s_max = float(s.max())
        if s_max <= 0.0:
            # All-zero matrix — nominally degenerate but we can still report.
            stable = 0.0
            ratio = 0.0
            smallest_significant = 0.0
        else:
            sq = s * s
            stable = float(sq.sum() / (s_max * s_max))
            nominal = float(min(m, n))
            ratio = stable / nominal if nominal > 0.0 else 0.0
            # "Smallest significant" — first singular below 1% of max.
            sig_mask = s >= s_max * 0.01
            if sig_mask.any():
                smallest_significant = float(s[sig_mask].min())
            else:
                smallest_significant = float(s.min())

        self._measurement_count += 1
        self._ratio_history.append(ratio)
        breaches = sum(1 for r in self._ratio_history if r < self._floor)

        warn = breaches >= self._sustained_for_warn
        if warn and self._last_warning_at != self._measurement_count:
            logger.warning(
                "🟡 Plasticity floor breached: stable_rank_ratio=%.3f over "
                "%d/%d recent measurements (floor=%.2f). Continual-learning "
                "rank collapse may be in progress on the substrate's W matrix.",
                ratio, breaches, len(self._ratio_history), self._floor,
            )
            self._last_warning_at = self._measurement_count

        return PlasticityReport(
            nominal_rank=int(min(m, n)),
            stable_rank=stable,
            stable_rank_ratio=ratio,
            largest_singular=s_max,
            smallest_significant_singular=smallest_significant,
            collapse_warning=warn,
            sustained_breaches=breaches,
            measurement_count=self._measurement_count,
        )

    @property
    def history(self) -> List[float]:
        return list(self._ratio_history)

    @property
    def measurement_count(self) -> int:
        return self._measurement_count

    def reset(self) -> None:
        self._ratio_history.clear()
        self._measurement_count = 0
        self._last_warning_at = -1

    # ── Async API — event-loop-safe SVD ───────────────────────────────────

    async def measure_async(self, W: np.ndarray) -> Optional[PlasticityReport]:
        """Offload SVD to a bounded thread pool so the event loop stays responsive.

        This is the preferred entry point for any caller running inside an
        asyncio context (substrate ticks, telemetry, heartbeat diagnostics).
        The synchronous ``measure()`` is preserved for tests and offline use.

        CRITICAL FIX (2026-04-28): SVD is O(N³).  On a 1024×1024 matrix it
        takes measurable wall-time.  Running it on the event-loop thread
        blocks all coroutines (socket handling, heartbeat, chat) for the
        duration.  Using a bounded executor keeps the loop alive while the
        heavy math runs in a capped worker pool (max 2 threads).
        """
        if W is None:
            return None
        # Copy the matrix so the worker thread does not race with the caller.
        W_copy = np.array(W, copy=True)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_get_svd_pool(), self.measure, W_copy)


# ── Singleton ─────────────────────────────────────────────────────────────

_singleton: Optional[PlasticityMonitor] = None


def get_plasticity_monitor() -> PlasticityMonitor:
    global _singleton
    if _singleton is None:
        _singleton = PlasticityMonitor()
    return _singleton


def reset_plasticity_monitor() -> None:
    """Test/diagnostic helper."""
    global _singleton
    _singleton = None
