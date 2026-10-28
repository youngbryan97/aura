"""Prospective observations sufficient for explicitly declared linear reads.

This does not discover what the runtime reads and is not an official recorder.
For an array x and declared read matrix A, construct orthonormal observations
z = B x such that A x = C z. Any transition depending on x only through A x
then loses none of those inputs. Completeness of A remains a separate source
and intervention obligation; fitting A from a desired verdict would not meet it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _row_basis(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    if not matrix.shape[0]:
        return np.empty((0, matrix.shape[1]))
    _u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * singular[0] if singular.size else 0.
    return vt[singular > threshold]


@dataclass(frozen=True)
class ObservationBasis:
    axes: np.ndarray
    reader_coordinates: np.ndarray
    previous_rank: int
    reconstruction_error: float

    @property
    def added_rank(self) -> int:
        return self.axes.shape[0] - self.previous_rank

    def observe(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) @ self.axes.T

    def reconstruct_reads(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations) @ self.reader_coordinates.T


def sufficient_basis(
    existing_axes: np.ndarray, readers: np.ndarray, *, tolerance: float = 1e-12,
) -> ObservationBasis:
    """Keep the existing linear span and minimally add the declared read span.

    Row-normalisation makes rank independent of a reader's physical units.
    The certificate measures reconstruction of the original, unnormalised A.
    The minimum additional rank is rank([B; A]) - rank(B), to this tolerance.
    """
    existing = np.asarray(existing_axes, dtype=float)
    needed = np.asarray(readers, dtype=float)
    if (existing.ndim != 2 or needed.ndim != 2 or existing.shape[1] != needed.shape[1]
            or not existing.shape[1] or not np.isfinite(existing).all()
            or not np.isfinite(needed).all() or not np.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError("finite matrices with the same positive width and a positive tolerance are required")

    def normalise(matrix: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(matrix, axis=1)
        return matrix / np.where(norm > 0., norm, 1.)[:, None]

    before = _row_basis(normalise(existing), tolerance)
    axes = _row_basis(np.vstack([before, normalise(needed)]), tolerance)
    coordinates = needed @ axes.T
    residual = needed - coordinates @ axes
    error = float(np.linalg.norm(residual, ord="fro"))
    return ObservationBasis(axes, coordinates, before.shape[0], error)


def declared_read_canary() -> bool:
    """Small algebraic canary; it does not certify a runtime's read inventory."""
    readers = np.asarray([[1., 2., 0., 0.], [0., 0., 3., -1.]])
    states = np.asarray([[.2, -.4, .7, .9], [1., 0., -2., .1]])
    basis = sufficient_basis(np.eye(4)[:1], readers)
    return bool(basis.reconstruction_error < 1e-12 and np.allclose(
        basis.reconstruct_reads(basis.observe(states)), states @ readers.T, atol=1e-12, rtol=0.,
    ))
