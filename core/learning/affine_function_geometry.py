"""Measure affine updates in both coefficient and observed-function space.

For augmented source features X and coefficient update A, the squared norm is
||A||_F^2 + mean(||X A.T||_F^2). The identity term retains unobserved directions.
Thin SVD applies the square root without materializing a feature covariance.
This controls observed score movement, not correctness on unseen inputs.
"""

from dataclasses import dataclass

import numpy as np

from core.verify.invariants import invariant
from typing import Any


@dataclass(frozen=True)
class AffineFunctionGeometry:
    parameter_index: int
    directions: np.ndarray
    roots: np.ndarray
    source_count: int

    @classmethod
    def from_features(cls, features: Any, *, parameter_index: int) -> Any:
        x = np.asarray(features, dtype=np.float64)
        if (x.ndim != 2 or not all(x.shape) or not np.all(np.isfinite(x))
                or type(parameter_index) is not int or parameter_index < 0):
            raise ValueError("invalid affine function geometry")
        augmented = np.column_stack((x, np.ones(len(x)))) / np.sqrt(len(x))
        _, singular, directions = np.linalg.svd(augmented, full_matrices=False)
        return cls(parameter_index, directions, np.sqrt(1. + singular ** 2), len(x))

    def _transform(self, parameters: Any, *, inverse: bool) -> tuple[Any, ...]:
        index = self.parameter_index
        values = list(parameters)
        weight, bias = values[index:index + 2]
        if (weight.ndim != 2 or bias.shape != (len(weight),)
                or weight.shape[1] + 1 != self.directions.shape[1]):
            raise ValueError("affine function parameter geometry differs")
        matrix = np.column_stack((weight, bias))
        factors = 1. / self.roots - 1. if inverse else self.roots - 1.
        transformed = matrix + ((matrix @ self.directions.T) * factors) @ self.directions
        values[index:index + 2] = (transformed[:, :-1], transformed[:, -1])
        return tuple(values)

    def encode(self, parameters: tuple[Any, ...]) -> Any:
        return self._transform(parameters, inverse=False)

    def decode(self, parameters: Any) -> Any:
        return self._transform(parameters, inverse=True)

    def pullback(self, gradients: Any) -> Any:
        # The inverse square root is symmetric, so it is its own adjoint.
        return self.decode(gradients)

    def certificate(self) -> dict[str, Any]:
        return {"metric": "coefficient_plus_empirical_affine_function_v1",
                "parameter_index": self.parameter_index, "source_feature_count": self.source_count,
                "feature_width": self.directions.shape[1] - 1,
                "coefficient_ridge": 1., "function_weight": 1.,
                "dense_covariance_materialized": False,
                "out_of_sample_retention_proven": False}


@dataclass(frozen=True)
class CompositeParameterGeometry:
    components: tuple

    def encode(self, parameters: tuple[Any, ...]) -> Any:
        for component in self.components:
            parameters = component.encode(parameters)
        return parameters

    def decode(self, parameters: Any) -> Any:
        for component in reversed(self.components):
            parameters = component.decode(parameters)
        return parameters

    def pullback(self, gradients: Any) -> Any:
        for component in self.components:
            gradients = component.pullback(gradients)
        return gradients


@invariant("learning.affine_metric_equals_observed_score_movement", scope="learning",
           owner="core/learning/affine_function_geometry.py", observational=False)
def _score_movement() -> tuple:
    x = np.array([[1., 2.], [-2., 3.]])
    weight, bias = np.array([[.3, -.2]]), np.array([.4])
    coordinates = AffineFunctionGeometry.from_features(x, parameter_index=0).encode((weight, bias))
    expected = np.sum(weight ** 2) + np.sum(bias ** 2) + np.sum((x @ weight.T + bias) ** 2) / len(x)
    assert np.isclose(sum(np.sum(value ** 2) for value in coordinates), expected)
    return ()
