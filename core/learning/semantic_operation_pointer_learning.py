"""Expose the shipped operation-boundary score to complete-graph learning."""

import numpy as np

from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm, argument_parameters
from core.learning.semantic_paired_pointer_refit import paired_boundary_feature
from core.learning.semantic_program_transducer import LinearPointerHead


def operation_pointer_parameters(model):
    pointer = model.operation_pointer
    return (
        np.concatenate((pointer.start_weight, pointer.end_weight,
                        pointer.pair_weight if pointer.pair_weight is not None else np.zeros(pointer.width))).astype(np.float64),
        np.asarray(pointer.start_bias + pointer.end_bias, dtype=np.float64),
    )


def operation_pointer_graph_evidence(model, hidden, nodes):
    """One linear score term per operation span, including diagonal interaction."""
    offset = 2 + 2 * len(model.operation_head.heads) + len(argument_parameters(model))
    return tuple(ArgumentScoreTerm(offset, paired_boundary_feature(hidden, node.span),
                                  1., "conditional_log_odds_v1") for node in nodes)


def operation_pointer_from_parameters(values):
    weight, bias = values
    start, end, pair = np.split(weight, 3)
    return LinearPointerHead(start, float(bias) / 2, end, float(bias) / 2, pair)
