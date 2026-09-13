"""Globally constrain an already scored typed argument chart.

This module supplies search, not semantic scores or an executable answer.
The caller supplies the existing floor-typed options and validates the result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.learning.semantic_program_ir import TokenSpan
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract

type ArgumentAssignment = tuple[
    float,
    tuple[tuple[int, ...], ...],
    tuple[tuple["TokenSpan", ...], ...],
    tuple[tuple[int, ...], ...],
]


class ArgumentOptimizationIncompleteError(RuntimeError):
    """A search limit or solver error is not a proof of graph infeasibility."""


def optimize_argument_chart(
    options: Sequence[Sequence[Sequence[tuple[float, int, TokenSpan]]]],
    *,
    n_inputs: int,
    contract: RegisterUseContract,
    node_limit: int = 10000,
) -> ArgumentAssignment | None:
    """Return an optimal feasible assignment within solver precision, or none.

Binary choices select one mention/register per argument. Continuous ordering
variables forbid cycles. Exactly one unused operation result makes the finite
DAG connected to a single sink. Limits never masquerade as an optimum.
"""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    operation_count = len(options)
    if (
        operation_count < 1
        or type(n_inputs) is not int or n_inputs < 1
        or type(node_limit) is not int or node_limit < 1
    ):
        raise ValueError("argument optimization dimensions must be positive")
    choices = []
    slots = defaultdict(list)
    uses = defaultdict(list)
    tokens = defaultdict(list)
    local_uses = defaultdict(list)
    for node, arguments in enumerate(options):
        if not arguments:
            return None
        for position, candidates in enumerate(arguments):
            if not candidates:
                return None
            for score, register, span in candidates:
                if (
                    not math.isfinite(score)
                    or not 0 <= register < n_inputs + operation_count
                    or register == n_inputs + node
                ):
                    raise ValueError("invalid typed argument option")
                index = len(choices)
                choices.append((node, position, score, register, span))
                slots[node, position].append(index)
                uses[register].append(index)
                local_uses[node, register].append(index)
                for token in range(span.start, span.end):
                    tokens[token].append(index)

    count = len(choices)
    sink_offset = count
    order_offset = count + operation_count
    width = count + 2 * operation_count
    objective = np.zeros(width)
    objective[:count] = [-choice[2] for choice in choices]
    lower = np.zeros(width)
    upper = np.ones(width)
    upper[order_offset:] = operation_count - 1
    integrality = np.zeros(width)
    integrality[:order_offset] = 1
    row_indices, column_indices, coefficients = [], [], []
    lows, highs = [], []

    def constraint(values, lo, hi):
        row = len(lows)
        for column, coefficient in values.items():
            if coefficient:
                row_indices.append(row)
                column_indices.append(column)
                coefficients.append(coefficient)
        lows.append(lo)
        highs.append(hi)

    for indices in slots.values():
        constraint(dict.fromkeys(indices, 1.0), 1.0, 1.0)
    # A token can belong to only one selected argument mention, across the chart.
    for indices in tokens.values():
        constraint(dict.fromkeys(indices, 1.0), 0.0, 1.0)
    if contract.distinct_arguments:
        for indices in local_uses.values():
            constraint(dict.fromkeys(indices, 1.0), 0.0, 1.0)
    for register in range(n_inputs):
        constraint(dict.fromkeys(uses[register], 1.0),
                   contract.input_min_uses, contract.input_max_uses)
    constraint({sink_offset + i: 1.0 for i in range(operation_count)}, 1.0, 1.0)
    minimum = max(1, contract.intermediate_min_uses)
    maximum = contract.intermediate_max_uses
    for node in range(operation_count):
        values = dict.fromkeys(uses[n_inputs + node], 1.0)
        constraint({**values, sink_offset + node: float(minimum)}, minimum, np.inf)
        constraint({**values, sink_offset + node: float(maximum)}, 0.0, maximum)
    for index, (node, _position, _score, register, _span) in enumerate(choices):
        if register >= n_inputs:
            dependency = register - n_inputs
            constraint({order_offset + node: 1.0, order_offset + dependency: -1.0,
                        index: -float(operation_count)}, 1 - operation_count, np.inf)
    matrix = coo_matrix(
        (np.asarray(coefficients, dtype=float), (row_indices, column_indices)),
        shape=(len(lows), width),
    ).tocsc()
    result = milp(
        objective, integrality=integrality, bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, lows, highs),
        options={"node_limit": node_limit, "mip_rel_gap": 0.0},
    )
    if result.status == 2:
        return None
    if result.status != 0 or result.x is None:
        raise ArgumentOptimizationIncompleteError(f"argument_optimizer_status:{result.status}")
    values = np.asarray(result.x)
    # Validate the returned integer point, not a rounded relaxation.
    if (
        values.shape != (width,)
        or not np.all(np.isfinite(values))
        or np.max(np.abs(values[:order_offset] - np.rint(values[:order_offset]))) > 1e-6
        or np.any(values < lower - 1e-6)
        or np.any(values > upper + 1e-6)
        or np.any(matrix @ values < np.asarray(lows) - 1e-6)
        or np.any(matrix @ values > np.asarray(highs) + 1e-6)
    ):
        raise ValueError("argument optimizer returned an invalid solution")
    arguments, spans = [], []
    score = 0.0
    for node, positions in enumerate(options):
        node_arguments, node_spans = [], []
        for position in range(len(positions)):
            selected = [i for i in slots[node, position] if values[i] > 0.5]
            if len(selected) != 1:
                raise ValueError("argument optimizer did not select exactly one option")
            _, _, contribution, register, span = choices[selected[0]]
            score += contribution
            node_arguments.append(register)
            node_spans.append(span)
        arguments.append(tuple(node_arguments))
        spans.append(tuple(node_spans))
    dependencies = tuple(
        tuple(sorted({register - n_inputs for register in values if register >= n_inputs}))
        for values in arguments
    )
    return score, tuple(arguments), tuple(spans), dependencies
