"""Strong baseline comparators for causal-exclusion claims.

The skeptical baseline is not a random network or a zeroed system.  It is a
compact stateful controller that learns the same endpoint mapping.  If Aura's
full stack cannot beat this baseline on coherence/process metrics, the critic
is right: the process is replaceable theater.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ControllerObservation:
    state: Mapping[str, float]
    params: Mapping[str, float]
    coherence: float = 0.0
    process_trace: Sequence[float] = ()


@dataclass(frozen=True)
class MimicComparison:
    endpoint_mae: float
    trace_distance: float
    coherence_delta: float
    process_advantage: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "endpoint_mae": round(self.endpoint_mae, 6),
            "trace_distance": round(self.trace_distance, 6),
            "coherence_delta": round(self.coherence_delta, 6),
            "process_advantage": self.process_advantage,
        }


class LinearStateParameterMimic:
    """Least-squares mimic of the stack's state-to-parameter endpoint mapping."""

    def __init__(self, state_keys: Sequence[str], param_keys: Sequence[str]) -> None:
        if not state_keys or not param_keys:
            raise ValueError("state_keys and param_keys must be non-empty")
        self.state_keys = tuple(state_keys)
        self.param_keys = tuple(param_keys)
        self._coef: np.ndarray | None = None

    def fit(self, observations: Iterable[ControllerObservation]) -> None:
        rows = list(observations)
        if len(rows) < len(self.state_keys) + 1:
            raise ValueError("need enough observations to fit baseline")
        x = np.array([[float(row.state.get(k, 0.0)) for k in self.state_keys] for row in rows], dtype=float)
        y = np.array([[float(row.params.get(k, 0.0)) for k in self.param_keys] for row in rows], dtype=float)
        x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        self._coef = np.linalg.pinv(x_aug) @ y

    def predict(self, state: Mapping[str, float]) -> dict[str, float]:
        if self._coef is None:
            raise RuntimeError("baseline mimic has not been fit")
        x = np.array([1.0] + [float(state.get(k, 0.0)) for k in self.state_keys], dtype=float)
        y = x @ self._coef
        return {key: float(value) for key, value in zip(self.param_keys, y)}


def compare_to_mimic(
    actual: Sequence[ControllerObservation],
    predicted_params: Sequence[Mapping[str, float]],
    *,
    param_keys: Sequence[str],
    min_trace_distance: float = 0.05,
    min_coherence_delta: float = 0.05,
) -> MimicComparison:
    if len(actual) != len(predicted_params):
        raise ValueError("actual observations and predictions must have same length")
    if not actual:
        raise ValueError("need at least one observation")

    endpoint_errors: list[float] = []
    trace_distances: list[float] = []
    coherence_deltas: list[float] = []

    for row, pred in zip(actual, predicted_params):
        endpoint_errors.extend(
            abs(float(row.params.get(key, 0.0)) - float(pred.get(key, 0.0))) for key in param_keys
        )
        trace = np.array(list(row.process_trace), dtype=float)
        if trace.size:
            endpoint = np.array([float(pred.get(key, 0.0)) for key in param_keys], dtype=float)
            endpoint_scalar = float(endpoint.mean()) if endpoint.size else 0.0
            trace_distances.append(float(np.mean(np.abs(trace - endpoint_scalar))))
        coherence_deltas.append(float(row.coherence) - _mimic_coherence(pred))

    endpoint_mae = float(np.mean(endpoint_errors))
    trace_distance = float(np.mean(trace_distances)) if trace_distances else 0.0
    coherence_delta = float(np.mean(coherence_deltas))
    return MimicComparison(
        endpoint_mae=endpoint_mae,
        trace_distance=trace_distance,
        coherence_delta=coherence_delta,
        process_advantage=trace_distance >= min_trace_distance and coherence_delta >= min_coherence_delta,
    )


def _mimic_coherence(params: Mapping[str, float]) -> float:
    if not params:
        return 0.0
    values = np.array(list(params.values()), dtype=float)
    spread = float(np.std(values))
    return max(0.0, min(1.0, 1.0 - spread))

