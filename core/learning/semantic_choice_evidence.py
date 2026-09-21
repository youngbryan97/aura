"""Keep categorical denominators differentiable across complete graph choices."""

from dataclasses import replace
import math

import numpy as np

from core.learning.semantic_argument_graph_learning import argument_parameters, argument_slot_evidence
from core.learning.semantic_relation_graph_learning import GraphChoiceNormalizer, RelationGraphContrast


def argument_choice_normalizers(model, hidden, nodes, chart, *, learn_arguments, argument_evidence_cache=None):
    if chart is None or chart.option_relation_evidence is None:
        raise ValueError("conditional graph learning requires the complete scored chart")
    parameters = (model.definition_relation_head.query_projection.astype(np.float64),
                  model.definition_relation_head.definition_projection.astype(np.float64),
                  *(value.astype(np.float64) for head in model.operation_head.heads for value in (head.weight, head.bias)),
                  *argument_parameters(model))
    normalizers, relation_scores = [], {}
    # Caller caches are scoped to one source observation at one model revision.
    slot_terms = {} if argument_evidence_cache is None else argument_evidence_cache
    for node_index, (node, slots) in enumerate(zip(nodes, chart.options, strict=True)):
        for position, slot in enumerate(slots):
            rows = []
            for index, (score, _register, mention) in enumerate(slot):
                relation = chart.option_relation_evidence[node_index][position][index]
                key = (node.span, position, mention)
                if key not in slot_terms:
                    terms = argument_slot_evidence(model, hidden, node.span, position, mention) if learn_arguments else ()
                    slot_terms[key] = (terms, math.fsum(term.score(parameters) for term in terms))
                terms, argument_score = slot_terms[key]
                row = RelationGraphContrast((relation,), (), 0., argument_terms=tuple((1., term) for term in terms))
                bank, selected = relation
                if id(bank) not in relation_scores:
                    relation_scores[id(bank)] = bank.scores(*parameters[:2])
                fixed = score - math.fsum((model.definition_relation_scale * relation_scores[id(bank)][selected],
                                          argument_score))
                rows.append(replace(row, fixed_margin=fixed))
            normalizers.append(GraphChoiceNormalizer(tuple(rows), model.definition_relation_scale))
    measured = math.fsum(term.score(parameters) for term in normalizers)
    if not math.isclose(measured, chart.choice_log_normalizer, rel_tol=1e-6, abs_tol=1e-4):
        raise ValueError("conditional graph denominator does not replay the runtime chart")
    return tuple(normalizers)
