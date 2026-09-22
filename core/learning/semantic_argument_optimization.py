"""Globally constrain an already scored typed argument chart.

This module supplies search, not semantic scores or an executable answer.
The caller supplies the existing floor-typed options and validates the result.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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


def _shortlist_mentions(
    options: Sequence[Sequence[Sequence[tuple[float, int, TokenSpan]]]],
    definition_options: Sequence[Sequence[Sequence[TokenSpan]]] | None,
    limit: int=4,
) -> tuple[tuple[Any, ...], Any]:
    rows, labels = [], []
    for node, arguments in enumerate(options):
        row, names = [], []
        for position, candidates in enumerate(arguments):
            groups = defaultdict(list)
            for index, (_score, register, _span) in enumerate(candidates):
                definition = definition_options[node][position][index] if definition_options is not None else None
                groups[register, definition].append(index)
            indices = sorted(index for group in groups.values()
                             for index in sorted(group, key=lambda i: -candidates[i][0])[:limit])
            row.append(tuple(candidates[index] for index in indices))
            if definition_options is not None:
                names.append(tuple(definition_options[node][position][index] for index in indices))
        rows.append(tuple(row))
        labels.append(tuple(names))
    return tuple(rows), tuple(labels) if definition_options is not None else None


def _dual_bound_screen(
    objective: Any,
    matrix: Any,
    lows: Any,
    highs: Any,
    upper: Any,
    incumbent_cost: Any,
    *,
    binary_count: Any,
    time_limit_s: Any=None,
) -> Any:
    """Fix only binary choices whose dual lower bound exceeds a feasible cost."""
    from scipy.optimize import linprog
    from scipy.sparse import vstack

    lows, highs = np.asarray(lows), np.asarray(highs)
    equal = lows == highs
    finite_high = np.isfinite(highs) & ~equal
    finite_low = np.isfinite(lows) & ~equal
    inequalities = vstack((matrix[finite_high], -matrix[finite_low])).tocsc()
    limits = np.concatenate((highs[finite_high], -lows[finite_low]))
    equations = matrix[equal]
    right = highs[equal]
    # Redundant binary upper bounds can absorb all reduced cost into their
    # duals. Relax them for pricing; the certified bound below still uses the
    # original finite domain, including the continuous ordering variables.
    pricing_upper = upper.copy()
    pricing_upper[:binary_count] = np.inf
    relaxation = linprog(objective, A_ub=inequalities, b_ub=limits,
        A_eq=equations, b_eq=right, bounds=np.column_stack((np.zeros(len(upper)), pricing_upper)),
        method="highs-ds", options={"time_limit": time_limit_s} if time_limit_s is not None else {})
    if not relaxation.success:
        return upper
    # Clipping signs and pricing the remaining stationarity residual against
    # the bounds gives a valid dual bound without trusting solver optimality.
    y = np.minimum(np.asarray(relaxation.ineqlin.marginals), 0.)
    z = np.asarray(relaxation.eqlin.marginals)
    residual = objective - inequalities.T @ y - equations.T @ z
    terms = np.concatenate((limits * y, right * z, np.minimum(residual, 0.) * upper))
    if not np.all(np.isfinite(terms)) or not np.all(np.isfinite(residual)):
        return upper
    bound = math.fsum(terms)
    # The tolerance is the float-error bar on this comparison, and it must
    # not be read off the bound it guards: a threshold derived from the
    # value it judges is a threshold that moves with the answer, which is
    # what `make epistemic-independence` refuses. Both sides of the
    # comparison are bounded by the problem's own cost scale, so that is
    # what it scales with — an input, fixed before the solve, and never
    # smaller than the summation error it stands in for.
    cost_scale = float(np.sum(np.abs(np.asarray(objective)) * np.abs(np.asarray(upper))))
    tolerance = 1e-7 * (1. + abs(incumbent_cost) + cost_scale)
    fixed = upper.copy()
    binary = np.arange(len(upper)) < binary_count
    fixed[binary & (upper == 1.) & (bound + np.maximum(residual, 0.) > incumbent_cost + tolerance)] = 0.
    return fixed


def optimize_argument_chart(
    options: Sequence[Sequence[Sequence[tuple[float, int, TokenSpan]]]],
    *,
    n_inputs: int,
    contract: RegisterUseContract,
    node_limit: int=10000,
    definition_options: Sequence[Sequence[Sequence[TokenSpan]]] | None=None,
    definition_scores: Mapping[tuple[int, TokenSpan], float] | None=None,
    prune_dominated: bool=False,
    excluded_arguments: Sequence[Sequence[int]] | None=None,
    excluded_graphs: Sequence[Sequence[Sequence[int]]]=(),
    time_limit_s: float | None=None,
    selection_observer: Any=None,
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
        or (time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s <= 0))
    ):
        raise ValueError("argument optimization dimensions must be positive")
    deadline = None if time_limit_s is None else time.monotonic() + time_limit_s

    def remaining() -> Any:
        if deadline is None:
            return None
        duration = deadline - time.monotonic()
        if duration <= 0.:
            raise ArgumentOptimizationIncompleteError("argument_optimizer_budget_exhausted")
        return duration

    excluded = tuple(excluded_graphs) + (() if excluded_arguments is None else (excluded_arguments,))
    if any(
        len(graph) != operation_count
        or any(len(target) != len(node)
               for target, node in zip(graph, options, strict=True))
        or any(type(register) is not int or not 0 <= register < n_inputs + operation_count
               for node in graph for register in node)
        for graph in excluded
    ):
        raise ValueError("excluded argument graph differs from chart")
    excluded = tuple(dict.fromkeys(tuple(tuple(node) for node in graph) for graph in excluded))
    choices = []
    slots = defaultdict(list)
    uses = defaultdict(list)
    tokens = defaultdict(list)
    local_uses = defaultdict(list)
    definition_uses = defaultdict(list)
    if definition_options is not None and (
        len(definition_options) != len(options)
        or any(len(labels) != len(arguments) for labels, arguments in zip(definition_options, options, strict=True))
        or any(
            len(labels) != len(candidates)
            for definitions, arguments in zip(definition_options, options, strict=True)
            for labels, candidates in zip(definitions, arguments, strict=True)
        )
    ):
        raise ValueError("definition options differ from argument chart")
    incumbent = None
    if prune_dominated:
        short, names = _shortlist_mentions(options, definition_options)
        try:
            incumbent = optimize_argument_chart(short, n_inputs=n_inputs, contract=contract,
                node_limit=node_limit, definition_options=names, definition_scores=definition_scores,
                excluded_graphs=excluded, time_limit_s=remaining())
        # not a failure: an optimisation that did not complete has no incumbent to keep.
        except ArgumentOptimizationIncompleteError:
            incumbent = None
    for node, arguments in enumerate(options):
        if not arguments:
            return None
        for position, candidates in enumerate(arguments):
            if not candidates:
                return None
            for candidate_index, (score, register, span) in enumerate(candidates):
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
                if definition_options is not None:
                    definition = definition_options[node][position][candidate_index]
                    definition_uses[register, definition].append(index)
                for token in range(span.start, span.end):
                    tokens[token].append(index)

    count = len(choices)
    if definition_scores is not None and (
        definition_options is None
        or any(key not in definition_scores for key in definition_uses)
        or any(not math.isfinite(value) for value in definition_scores.values())
    ):
        raise ValueError("definition attachment scores do not cover the chart")
    definition_offset = count
    sink_offset = count + len(definition_uses)
    order_offset = sink_offset + operation_count
    width = order_offset + operation_count
    objective = np.zeros(width)
    objective[:count] = [-choice[2] for choice in choices]
    if definition_scores is not None:
        objective[definition_offset:sink_offset] = [-definition_scores[key] for key in definition_uses]
    lower = np.zeros(width)
    upper = np.ones(width)
    upper[order_offset:] = operation_count - 1
    integrality = np.zeros(width)
    integrality[:order_offset] = 1
    row_indices, column_indices, coefficients = [], [], []
    lows, highs = [], []

    def constraint(values: dict[str, Any], lo: float, hi: float) -> None:
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
    for graph in excluded:
        # One no-good covers every mention/definition realization of this graph.
        matching = {index: 1.0 for index, (node, position, _score, register, _span)
                    in enumerate(choices) if register == graph[node][position]}
        constraint(matching, -np.inf, len(slots) - 1)
    definitions_by_register = defaultdict(list)
    for label_index, ((register, _definition), indices) in enumerate(definition_uses.items()):
        variable = definition_offset + label_index
        definitions_by_register[register].append(variable)
        for index in indices:
            constraint({index: 1.0, variable: -1.0}, -np.inf, 0.0)
        # Attachment evidence is paid once, and only for a used definition.
        if definition_scores is not None:
            constraint({variable: 1.0, **dict.fromkeys(indices, -1.0)}, -np.inf, 0.0)
    for variables in definitions_by_register.values():
        constraint(dict.fromkeys(variables, 1.0), 0.0, 1.0)
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
    if incumbent is not None:
        upper = _dual_bound_screen(objective, matrix, lows, highs, upper, -incumbent[0],
                                  binary_count=order_offset, time_limit_s=remaining())
    duration = remaining()
    result = milp(
        objective, integrality=integrality, bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, lows, highs),
        options={"node_limit": node_limit, "mip_rel_gap": 0.0,
                 **({"time_limit": duration} if duration is not None else {})},
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
    arguments, spans, selected_indices = [], [], []
    score = 0.0
    for node, positions in enumerate(options):
        node_arguments, node_spans, node_indices = [], [], []
        for position in range(len(positions)):
            selected = [i for i in slots[node, position] if values[i] > 0.5]
            if len(selected) != 1:
                raise ValueError("argument optimizer did not select exactly one option")
            _, _, contribution, register, span = choices[selected[0]]
            score += contribution
            node_arguments.append(register)
            node_spans.append(span)
            node_indices.append(slots[node, position].index(selected[0]))
        arguments.append(tuple(node_arguments))
        spans.append(tuple(node_spans))
        selected_indices.append(tuple(node_indices))
    dependencies = tuple(
        tuple(sorted({register - n_inputs for register in values if register >= n_inputs}))
        for values in arguments
    )
    if definition_scores is not None:
        score += sum(definition_scores[key] for index, key in enumerate(definition_uses)
                     if values[definition_offset + index] > 0.5)
    if selection_observer is not None:
        selection_observer(tuple(selected_indices))
    return score, tuple(arguments), tuple(spans), dependencies
