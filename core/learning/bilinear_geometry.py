"""Coordinate geometry for an anchored bilinear map, without a dense map.

For M = Q D.T, use ||dQ D0.T||_F^2 + ||Q0 dD.T||_F^2.
This is invariant under Q -> Q T, D -> D T^-T for invertible T.
It measures the two separate first-order contributions, not their sum or
the full nonlinear change. It does not establish out-of-sample retention.
"""

from dataclasses import dataclass

import numpy as np

from core.verify.invariants import invariant


def _gram_root(factor, scale):
    # Thin SVD avoids squaring the condition number before rank detection.
    _, singular, right = np.linalg.svd(factor, full_matrices=False)
    threshold = np.finfo(float).eps * max(factor.shape) * singular[0]
    if len(singular) != factor.shape[1] or singular[-1] <= threshold:
        raise ValueError("functional bilinear geometry requires full-column-rank factors")
    root = (right.T * (singular * scale)) @ right
    inverse = (right.T / (singular * scale)) @ right
    return root, inverse, float(singular[0] / singular[-1])


@dataclass(frozen=True)
class BilinearFactorGeometry:
    query_root: np.ndarray
    query_inverse: np.ndarray
    definition_root: np.ndarray
    definition_inverse: np.ndarray
    condition_numbers: tuple[float, float]

    @classmethod
    def from_factors(cls, query, definition, *, scale=1.):
        q, d = (np.asarray(value, dtype=np.float64) for value in (query, definition))
        if (q.ndim != 2 or d.shape != q.shape or not all(q.shape)
                or not np.all(np.isfinite(q)) or not np.all(np.isfinite(d))
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError("invalid functional bilinear geometry")
        qr, qi, dc = _gram_root(d, scale)
        dr, di, qc = _gram_root(q, scale)
        return cls(qr, qi, dr, di, (qc, dc))

    def encode(self, parameters):
        q, d, *rest = parameters
        return (q @ self.query_root, d @ self.definition_root, *rest)

    def decode(self, coordinates):
        q, d, *rest = coordinates
        return (q @ self.query_inverse, d @ self.definition_inverse, *rest)

    def pullback(self, gradients):
        q, d, *rest = gradients
        return (q @ self.query_inverse.T, d @ self.definition_inverse.T, *rest)

    def certificate(self):
        return {"metric": "anchored_bilinear_factor_function_v1",
                "factor_condition_numbers": list(self.condition_numbers),
                "dense_bilinear_map_materialized": False,
                "scope": "separate_first_order_factor_contributions",
                "out_of_sample_retention_proven": False}


@invariant("learning.bilinear_metric_ignores_reciprocal_factor_scaling", scope="learning",
           owner="core/learning/bilinear_geometry.py", observational=False)
def _factor_scaling() -> tuple:
    q, d = np.array([[1.], [2.]]), np.array([[3.], [4.]])
    dq, dd = np.array([[.1], [.2]]), np.array([[.3], [-.1]])
    first = BilinearFactorGeometry.from_factors(q, d).encode((dq, dd))
    second = BilinearFactorGeometry.from_factors(8 * q, d / 8).encode((8 * dq, dd / 8))
    assert np.isclose(sum(np.sum(x * x) for x in first), sum(np.sum(x * x) for x in second))
    return ()
