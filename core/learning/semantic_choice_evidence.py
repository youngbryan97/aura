"""Keep categorical denominators differentiable across complete graph choices."""

from dataclasses import replace
import math

import numpy as np

from core.learning.semantic_argument_graph_learning import argument_parameters, argument_slot_evidence
from core.learning.semantic_relation_graph_learning import GraphChoiceNormalizer, RelationGraphContrast, graph_margin


def argument_choice_normalizers(model, hidden, nodes, chart, *, learn_arguments):
    if chart is None or chart.option_relation_evidence is None:
        raise ValueError("conditional graph learning requires the complete scored chart")
    parameters = (model.definition_relation_head.query_projection.astype(np.float64),
                  model.definition_relation_head.definition_projection.astype(np.float64),
                  *(value.astype(np.float64) for head in model.operation_head.heads for value in (head.weight, head.bias)),
                  *argument_parameters(model))
    normalizers = []
    for node_index, (node, slots) in enumerate(zip(nodes, chart.options, strict=True)):
        for position, slot in enumerate(slots):
            rows = []
            for index, (score, _register, mention) in enumerate(slot):
                relation = chart.option_relation_evidence[node_index][position][index]
                terms = argument_slot_evidence(model, hidden, node.span, position, mention) if learn_arguments else ()
                row = RelationGraphContrast((relation,), (), 0., argument_terms=tuple((1., term) for term in terms))
                fixed = score - graph_margin(parameters, row, scale=model.definition_relation_scale)
                rows.append(replace(row, fixed_margin=fixed))
            normalizers.append(GraphChoiceNormalizer(tuple(rows), model.definition_relation_scale))
    measured = math.fsum(term.score(parameters) for term in normalizers)
    if not math.isclose(measured, chart.choice_log_normalizer, rel_tol=1e-6, abs_tol=1e-4):
        raise ValueError("conditional graph denominator does not replay the runtime chart")
    return tuple(normalizers)
