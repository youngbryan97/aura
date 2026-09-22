"""Grassmann-geometry state encoding for phi on the transformer's own substrate.

The IIT critique of Aura's phi is correct: ``record_residual_stream`` collapsed a
~5000-dimensional hidden vector into 8 contiguous chunk-means, which throws away
almost all representational structure. The resulting φ describes a near-arbitrary
projection, not the transformer's actual state dynamics.

This module supplies the principled alternative the critique recommends — using the
**Grassmann distance between layer/representation subspaces to build a real TPM over
the transformer's own state transitions**:

  1. A sliding window of residual-stream vectors is reduced to its top-k principal
     **subspace** (a point on the Grassmann manifold G(k, d)) — this captures the
     *directions* the representation occupies, not arbitrary coordinate chunks.
  2. A small set of recurring subspaces are learned as **geometric anchors** (modes).
  3. The current subspace is encoded against the anchors by Grassmann distance into an
     N-bit state: bit i = "the representation is engaging geometric mode i". This is a
     genuine IIT node decomposition (the nodes are geometric modes), so the existing
     exact-φ machinery in ``phi_core`` runs on it unchanged — but now over the
     transformer's representational geometry rather than 8 means.

The Grassmann distance (principal angles via SVD) is rotation-invariant within a
subspace and respects the manifold's geometry, so two representations that span the
same directions map to the same mode even if individual coordinates differ.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger("Aura.GrassmannPhi")


def subspace_basis(window: np.ndarray, k: int, *, rel_tol: float = 0.05) -> np.ndarray:
    """Dominant subspace of a window of vectors → (d, k') orthonormal basis.

    The span of the *raw* vectors (NOT mean-centered): the common-mode activation
    direction is part of the representational state, so removing it would leave only
    noise. Only directions whose singular value is at least ``rel_tol`` of the top
    one are kept, so a window dominated by one direction yields a 1-D subspace
    instead of padding with noise directions (which would make the state unstable).
    """
    M = np.asarray(window, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] < 2:
        raise ValueError("window must be (m, d) with m >= 2")
    _u, s, vt = np.linalg.svd(M, full_matrices=False)
    k = int(max(1, min(k, vt.shape[0])))
    top = float(s[0]) if s.size else 0.0
    if top <= 0.0:
        return np.ascontiguousarray(vt[:1].T)
    significant = max(1, int(np.sum(s > rel_tol * top)))
    k = min(k, significant)
    return np.ascontiguousarray(vt[:k].T)  # (d, k), orthonormal columns


def principal_angles(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces (orthonormal-column bases)."""
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    k = min(qa.shape[1], qb.shape[1])
    if k == 0:
        return np.zeros(0)
    s = np.linalg.svd(qa[:, :k].T @ qb[:, :k], compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def grassmann_distance(qa: np.ndarray, qb: np.ndarray) -> float:
    """Geodesic distance on the Grassmann manifold = ‖principal angles‖₂."""
    return float(np.linalg.norm(principal_angles(qa, qb)))


class GrassmannResidualComplex:
    """Encodes residual-stream geometry into an N-bit IIT state via Grassmann modes."""

    def __init__(
        self,
        *,
        n_anchors: int = 8,
        subspace_dim: int = 4,
        window: int = 24,
        max_dims: int = 256,
        anchor_separation: float | None = None,
    ) -> None:
        self.n_anchors = int(n_anchors)
        self.k = int(subspace_dim)
        self.window = int(window)
        self.max_dims = int(max_dims)
        # Two subspaces are "different modes" if separated by more than this.
        # Max Grassmann distance is sqrt(k)·(π/2); seed anchors at a third of that.
        self.anchor_separation = (
            anchor_separation
            if anchor_separation is not None
            else (np.sqrt(self.k) * (np.pi / 2.0)) / 3.0
        )
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._anchors: list[np.ndarray] = []
        self._dim_idx: np.ndarray | None = None
        self._dist_ema = float(np.sqrt(self.k) * (np.pi / 2.0) * 0.5)
        self.samples = 0

    def _project(self, v: np.ndarray) -> np.ndarray:
        if v.size > self.max_dims:
            if self._dim_idx is None:
                self._dim_idx = np.linspace(0, v.size - 1, self.max_dims).astype(int)
            return v[self._dim_idx]
        return v

    def observe(self, hidden_vec: np.ndarray) -> int | None:
        """Add a residual vector; once the window is full, return the N-bit state."""
        v = np.asarray(hidden_vec, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return None
        self._buf.append(self._project(v))
        if len(self._buf) < self.window:
            return None
        try:
            q = subspace_basis(np.asarray(self._buf), self.k)
        # not a failure: a window with no subspace basis has no reading to encode.
        except (ValueError, np.linalg.LinAlgError):
            return None
        return self._encode(q)

    def _encode(self, q: np.ndarray) -> int:
        # Seed distinct geometric anchors until the pool is full.
        if len(self._anchors) < self.n_anchors:
            if not self._anchors or min(grassmann_distance(q, a) for a in self._anchors) > self.anchor_separation:
                self._anchors.append(q)
        if not self._anchors:
            return 0
        dists = np.array([grassmann_distance(q, a) for a in self._anchors], dtype=np.float64)
        # Adaptive radius (EMA of typical distance) → ~half the modes active on average.
        self._dist_ema = 0.98 * self._dist_ema + 0.02 * float(np.mean(dists))
        radius = self._dist_ema
        bits = (dists <= radius).astype(int)
        bits[int(np.argmin(dists))] = 1  # the nearest mode is always engaged
        state = 0
        for i in range(min(self.n_anchors, len(bits))):
            state |= int(bits[i]) << i
        self.samples += 1
        return int(state)

    def status(self) -> dict[str, float | int]:
        return {
            "anchors": len(self._anchors),
            "samples": self.samples,
            "radius_ema": round(self._dist_ema, 5),
            "subspace_dim": self.k,
        }
