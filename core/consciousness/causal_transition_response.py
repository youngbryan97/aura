"""Bounded responses to interventions on an independently identified kernel.

The caller supplies the causal identification and its evidence reference. An
observational transition matrix alone cannot supply that identification. The
quantity computed here is a directed intervention-response diagnostic, not IIT
4.0 and not evidence of experience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_RESPONSE_NODES = 8


@dataclass(frozen=True)
class CausalTransitionKernel:
    transition: Any
    background: Any
    identification: str
    evidence_id: str
    sample_count: int = 0

    def __post_init__(self) -> None:
        shape = getattr(self.transition, "shape", None)
        size = shape[0] if shape is not None and len(shape) == 2 else len(self.transition)
        if not 4 <= size <= 2**MAX_RESPONSE_NODES or size & (size - 1):
            raise ValueError("intervention response requires a complete kernel of two to eight binary nodes")
        if shape is not None and tuple(shape) != (size, size):
            raise ValueError("causal kernel must be square")
        if shape is None and any(len(row) != size for row in self.transition):
            raise ValueError("causal kernel must be square")
        if len(self.background) != size:
            raise ValueError("kernel and background dimensions disagree")
        if self.identification not in {"randomized_interventions", "structural_equations"}:
            raise ValueError("observations alone do not identify an intervention kernel")
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("causal identification needs an evidence reference")
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        if self.identification == "randomized_interventions" and self.sample_count == 0:
            raise ValueError("a randomized intervention record needs measured samples")
        transition = np.array(self.transition, dtype=np.float64, copy=True)
        background = np.array(self.background, dtype=np.float64, copy=True)
        if transition.shape != (size, size) or background.shape != (size,):
            raise ValueError("kernel and background dimensions disagree")
        if not np.isfinite(transition).all() or not np.isfinite(background).all():
            raise ValueError("kernel probabilities must be finite")
        if np.any(transition < 0) or np.any(background < 0):
            raise ValueError("kernel probabilities must be nonnegative")
        if not np.allclose(transition.sum(axis=1), 1.0) or not np.isclose(background.sum(), 1.0):
            raise ValueError("every intervention row and the background must be normalized")
        transition /= transition.sum(axis=1, keepdims=True)
        background /= background.sum()
        transition.setflags(write=False)
        background.setflags(write=False)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "background", background)

    @property
    def node_count(self) -> int:
        return int(self.transition.shape[0]).bit_length() - 1


def _extract(states: np.ndarray, nodes: tuple[int, ...]) -> np.ndarray:
    values = np.zeros(len(states), dtype=np.int64)
    for index, node in enumerate(nodes):
        values |= ((states >> node) & 1) << index
    return values


def _directed_response(kernel: CausalTransitionKernel, moved: tuple[int, ...], other: tuple[int, ...]) -> float:
    states = np.arange(len(kernel.background))
    a, b = _extract(states, moved), _extract(states, other)
    # Marginal P(B), shared across assignments, replaces observational P(B|A).
    background_b = np.bincount(b, weights=kernel.background, minlength=2**len(other))
    responses = []
    for assignment in range(2**len(moved)):
        rows = np.flatnonzero(a == assignment)
        next_state = background_b[b[rows]] @ kernel.transition[rows]
        responses.append(np.bincount(b, weights=next_state, minlength=len(background_b)))
    responses = np.asarray(responses)
    reference = responses.mean(axis=0)
    information = 0.0
    for response in responses:
        present = response > 0
        information += float(np.sum(response[present] * np.log(response[present] / reference[present])))
    return max(0.0, information / len(responses))


def weakest_response(kernel: CausalTransitionKernel) -> tuple[float, tuple[int, ...], tuple[int, ...], list[float]]:
    """The weakest directed response across both directions of every cut."""
    count = kernel.node_count
    partitions = []
    for mask in range(1, 2**count - 1):
        if not mask & 1:
            continue
        left = tuple(node for node in range(count) if mask & (1 << node))
        right = tuple(node for node in range(count) if not mask & (1 << node))
        value = min(_directed_response(kernel, left, right), _directed_response(kernel, right, left))
        partitions.append((value, left, right))
    minimum, left, right = min(partitions)
    return minimum, left, right, [value for value, _, _ in partitions]
