"""Offline constrained decoding of a learned operation score chart.

Operation score orders feasible interpretations; the existing argument model
chooses bindings within each interpretation. This is a hierarchical objective,
not a claim of jointly normalized operation and argument probabilities.
"""

import math
import time

import numpy as np

from core.learning.semantic_argument_chart import select_operation_argument_graph
from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_graph_counterexamples import argument_graph_program
from core.learning.semantic_operation_search import (
    OperationChartSearch,
    OperationSearchIncompleteError,
)
from core.learning.semantic_program_floor import PRIMITIVES_BY_NAME
from core.learning.semantic_program_ir import TokenSpan, normalize_semantic_value
from core.learning.semantic_program_transducer import _hidden_array
from core.learning.semantic_program_transducer_fitting import (
    _assign_typed_arguments,
    _operation_chart_use_feasible,
    _OperationNode,
)


def resolve_operation_scores(model, *, source_token_ids, hidden_states, public_inputs,
                             scores, labels, time_limit_s=10., max_expansions=100_000):
    """Retain every finite label/span, then select the best feasible operation set.

    The grammar uses the parent's declared public-input geometry and use
    contract. No target count, target spans, target program or answer is read.
    Search exhaustion is distinct from proving that no interpretation exists.
    """
    if (type(time_limit_s) not in (int, float) or not math.isfinite(time_limit_s)
            or time_limit_s <= 0):
        raise ValueError("positive finite search allowance required")
    deadline = time.monotonic() + time_limit_s
    tokens = tuple(source_token_ids)
    hidden = _hidden_array(hidden_states, expected_width=model.hidden_size)
    inputs = tuple(normalize_semantic_value(x) for x in public_inputs)
    values = np.asarray(scores)
    if (values.ndim != 3 or values.shape[0] != len(tokens) or len(hidden) != len(tokens)
            or not all(values.shape) or values.shape[2] != len(labels)
            or len(set(labels)) != len(labels) or set(labels) - set(PRIMITIVES_BY_NAME)
            or np.isnan(values).any() or np.isposinf(values).any()):
        raise ValueError("invalid labeled operation chart")
    for start, column, _ in np.argwhere(np.isfinite(values)):
        if start + column + 1 > len(tokens):
            raise ValueError("finite operation score exceeds source bounds")
    limit = model.inference_step_limit(len(inputs))
    if limit is None:
        return None, "public_input_count_unsupported", {}
    spans, _, pointer = model._runtime_input_grounding(tokens, hidden, inputs)
    nodes = tuple(_OperationNode(TokenSpan(int(s), int(s + w + 1)), labels[int(k)],
                                 float(values[s, w, k]), float(values[s, w, k]), 0.)
                  for s, w, k in np.argwhere(np.isfinite(values)))
    search = OperationChartSearch(
        nodes, max_steps=limit, length_penalty=0., max_expansions=max_expansions,
        feasible=lambda selected: _operation_chart_use_feasible(
            selected, n_inputs=len(inputs), contract=model.register_use_contract,
            input_types=tuple("integer" if type(x) is int else "integer_sequence" for x in inputs)),
    )
    relation_scores, relation_vectors = {}, {}
    definitions = model.definition_pointer.score_sequence(hidden)

    def assign(selected):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ArgumentOptimizationIncompleteError("context_decode_budget_exhausted")
        return _assign_typed_arguments(
            model=model, hidden=hidden, inputs=inputs, input_spans=spans,
            source_token_ids=tokens, operation_nodes=selected, argument_pointer_scores=pointer,
            definition_pointer_scores=definitions, relation_score_cache=relation_scores,
            relation_vector_cache=relation_vectors, time_limit_s=remaining,
        )

    refusal, result = "", None
    try:
        selected = select_operation_argument_graph(search, assign, length_penalty=0., joint=False)
        if selected is None:
            refusal = "typed_operation_grammar_exhausted"
        else:
            result = argument_graph_program(selected.operation_nodes, selected.arguments,
                                            n_inputs=len(inputs))
    except (ArgumentOptimizationIncompleteError, OperationSearchIncompleteError) as exc:
        refusal = str(exc)
    return result, refusal, {"expanded": search.expanded, "feasible_charts": search.yielded,
                             "grammar_exhausted": search.complete, "nodes": len(nodes)}
