"""Resolve predicted operation sets using the existing typed argument machinery.

This diagnostic has no serving authority. Frozen argument heads may have seen
the source fold, so its results do not establish full-program held-out transfer.
"""

from core.learning.semantic_graph_counterexamples import argument_graph_program
from core.learning.semantic_program_ir import TokenSpan, normalize_semantic_value
from core.learning.semantic_program_transducer import _hidden_array
from core.learning.semantic_program_transducer_fitting import (
    _assign_typed_arguments,
    _OperationNode,
)


def resolve_operation_set(model, *, source_token_ids, hidden_states, public_inputs,
                          operations, time_limit_s=10.):
    """Return a program or refusal from public evidence and predicted spans.

    No target program, target spans, or expected output is accepted here.
    Operation scores are constant for this fixed set and cannot select an
    argument assignment; their common additive offset is zero.
    """
    tokens = tuple(source_token_ids)
    hidden = _hidden_array(hidden_states, expected_width=model.hidden_size)
    if len(tokens) != len(hidden):
        raise ValueError("source tokens and hidden evidence differ")
    inputs = tuple(normalize_semantic_value(x) for x in public_inputs)
    limit = model.inference_step_limit(len(inputs))
    if limit is None:
        return None, "public_input_count_unsupported"
    if not operations:
        return None, "empty_predicted_operation_set"
    if len(operations) > limit:
        return None, "predicted_operation_count_exceeds_grammar"
    nodes = []
    previous = 0
    for start, end, operation in operations:
        span = TokenSpan(start, end)
        span.validate_bound(len(tokens))
        if start < previous:
            raise ValueError("predicted operation spans overlap or are unordered")
        nodes.append(_OperationNode(span, operation, 0., 0., 0.))
        previous = end
    input_spans, _, pointer = model._runtime_input_grounding(tokens, hidden, inputs)
    assigned = _assign_typed_arguments(
        model=model, hidden=hidden, inputs=inputs, input_spans=input_spans,
        source_token_ids=tokens, operation_nodes=tuple(nodes), argument_pointer_scores=pointer,
        time_limit_s=time_limit_s,
    )
    if assigned is None:
        return None, "typed_argument_chart_empty"
    return argument_graph_program(assigned.operation_nodes, assigned.arguments,
                                  n_inputs=len(inputs)), ""
