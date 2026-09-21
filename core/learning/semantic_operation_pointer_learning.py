"""Expose the shipped operation-boundary score to complete-graph learning."""

from types import SimpleNamespace

import numpy as np

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm, argument_parameters
from core.learning.semantic_paired_pointer_refit import paired_boundary_feature
from core.learning.semantic_program_transducer import LinearPointerHead
from core.verify.invariants import invariant
from typing import Any


def operation_pointer_parameters(model: Any) -> tuple[Any, Any]:
    pointer = model.operation_pointer
    parts = (pointer.start_weight, pointer.end_weight)
    if pointer.pair_weight is not None:
        parts += (pointer.pair_weight,)
    return (
        np.concatenate(parts).astype(np.float64),
        np.asarray(pointer.start_bias + pointer.end_bias, dtype=np.float64),
    )


def operation_pointer_graph_evidence(model: Any, hidden: Any, nodes: Any) -> tuple[Any, ...]:
    """Differentiate only the boundary features declared by the shipped pointer."""
    offset = 2 + 2 * len(model.operation_head.heads) + len(argument_parameters(model))
    terms = []
    for node in nodes:
        if model.operation_pointer.pair_weight is not None:
            feature = paired_boundary_feature(hidden, node.span)
        else:
            node.span.validate_bound(len(hidden))
            feature = np.concatenate((hidden[node.span.start], hidden[node.span.end - 1]))
        terms.append(ArgumentScoreTerm(offset, feature, 1., "conditional_log_odds_v1"))
    return tuple(terms)


def operation_pointer_from_parameters(values: Any, *, pointer: Any) -> Any:
    weight, bias = values
    parts = 2 if pointer.pair_weight is None else 3
    if np.shape(weight) != (parts * pointer.width,) or np.shape(bias) != ():
        raise ValueError("operation pointer parameter geometry differs from declared capacity")
    start, end, *interaction = np.split(weight, parts)
    pair = interaction[0] if interaction else None
    return LinearPointerHead(start, float(bias) / 2, end, float(bias) / 2, pair)


@invariant("learning.operation_pointer_refit_preserves_capacity", scope="learning",
           owner="core/learning/semantic_operation_pointer_learning.py", observational=False)
def _pointer_capacity_roundtrip() -> tuple:
    for pair in (None, np.ones(2)):
        pointer = LinearPointerHead(np.ones(2), 0., np.ones(2), 0., pair)
        restored = operation_pointer_from_parameters(
            operation_pointer_parameters(SimpleNamespace(operation_pointer=pointer)), pointer=pointer)
        assert (restored.pair_weight is None) == (pair is None)
    return ()
