"""Learn complete operation span sets from source-only native depth states.

The partition includes absent, repeated and alternative operations. A typed
program still needs argument grounding; this field supplies no answer oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import TokenSpan
from core.verify.invariants import invariant


def operation_span_inventory(token_count, max_span_tokens, input_spans=()):
    if any(type(value) is not int or value < 1 for value in (token_count, max_span_tokens)):
        raise ValueError("operation inventory needs positive source and span bounds")
    for span in input_spans:
        span.validate_bound(token_count)
    return tuple(TokenSpan(start, end) for end in range(1, token_count + 1)
        for start in range(max(0, end - max_span_tokens), end)
        if not any(start < span.end and span.start < end for span in input_spans))


def operation_set_log_partition(energies, spans, token_count, max_steps):
    """Exact differentiable sum over bounded, nonoverlapping labelled span sets.

    Each unused token advances once. Labels on the same span are alternatives,
    so neither overlapping placements nor duplicate paths inflate the mass.
    """
    if (energies.ndim != 2 or energies.shape[0] != len(spans) or energies.shape[1] < 1
            or type(token_count) is not int or token_count < 1
            or type(max_steps) is not int or max_steps < 1
            or len(set(spans)) != len(spans)):
        raise ValueError("operation partition geometry differs")
    by_end = [[] for _ in range(token_count + 1)]
    for index, span in enumerate(spans):
        span.validate_bound(token_count)
        by_end[span.end].append((index, span.start))
    # None denotes unreachable cardinalities and avoids -inf/-inf derivatives.
    table = [[mx.array(0.)] + [None] * max_steps]
    label_mass = mx.logsumexp(energies, axis=-1)
    for end in range(1, token_count + 1):
        row = [mx.array(0.)]
        for count in range(1, max_steps + 1):
            values = ([table[end - 1][count]] if table[end - 1][count] is not None else [])
            values.extend(table[start][count - 1] + label_mass[index]
                for index, start in by_end[end] if table[start][count - 1] is not None)
            row.append(mx.logsumexp(mx.stack(values)) if values else None)
        table.append(row)
    return mx.logsumexp(mx.stack([value for value in table[-1] if value is not None]))


@dataclass(frozen=True)
class OperationSetSupervision:
    source_id: str
    token_count: int
    input_spans: tuple
    targets: tuple

    def indices(self, field):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("operation supervision needs a source identity")
        spans = operation_span_inventory(self.token_count, field.max_span_tokens, self.input_spans)
        lookup = {span: index for index, span in enumerate(spans)}
        targets = tuple(sorted(self.targets, key=lambda row: (row[1].start, row[1].end)))
        if len(targets) > field.max_steps:
            raise ValueError("source operation count exceeds the declared grammar")
        for index, (label, span) in enumerate(targets):
            if (label not in field.labels or span not in lookup
                    or index and targets[index - 1][1].end > span.start):
                raise ValueError("source operation is absent from the public bounded grammar")
        return spans, tuple((lookup[span], field.labels.index(label)) for label, span in targets)

    def receipt(self):
        return {"source_id": self.source_id, "token_count": self.token_count,
            "input_spans": [(span.start, span.end) for span in self.input_spans],
            "targets": [(label, span.start, span.end) for label, span in self.targets]}


class NativeOperationField(nn.Module):
    def __init__(self, hidden_width, *, depths, labels, relation_width=128,
                 max_span_tokens=29, max_steps=4):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in (
                hidden_width, depths, relation_width, max_span_tokens, max_steps))
                or not labels or len(set(labels)) != len(labels)
                or any(semantic_primitive_type_signature(label) is None for label in labels)):
            raise ValueError("native operation field needs an explicit typed vocabulary and bounds")
        self.hidden_width, self.depths, self.labels = hidden_width, depths, tuple(labels)
        self.relation_width = relation_width
        self.max_span_tokens, self.max_steps = max_span_tokens, max_steps
        self.depth_gate = mx.zeros((depths,))
        self.project = nn.Linear(hidden_width, relation_width)
        self.compose = nn.Linear(6 * relation_width, relation_width)
        self.output = nn.Linear(relation_width, len(labels) + 1)

    def to_contract(self):
        return {"schema": "aura.native_operation_field.v1", "hidden_width": self.hidden_width,
            "depths": self.depths, "labels": list(self.labels), "relation_width": self.relation_width,
            "max_span_tokens": self.max_span_tokens, "max_steps": self.max_steps,
            "score": "primitive_minus_null_logit", "objective": "bounded_span_set_likelihood",
            "absolute_position_features": False, "construction_features": False,
            "whole_program_calibration_proven": False}

    @classmethod
    def from_contract(cls, contract):
        field = cls(contract["hidden_width"], depths=contract["depths"], labels=contract["labels"],
            relation_width=contract["relation_width"], max_span_tokens=contract["max_span_tokens"],
            max_steps=contract["max_steps"])
        if field.to_contract() != contract:
            raise ValueError("unknown native operation field contract")
        return field

    def __call__(self, states, spans):
        if (states.ndim != 3 or states.shape[1:] != (self.depths, self.hidden_width)
                or states.shape[0] < 1):
            raise ValueError("native operation field needs aligned whole-source depth states")
        for span in spans:
            span.validate_bound(len(states))
        if not spans:
            return mx.zeros((0, len(self.labels)))
        mixed = mx.sum(states * mx.softmax(self.depth_gate)[None, :, None], axis=1)
        tokens = nn.silu(self.project(mixed))
        prefix = mx.concatenate((mx.zeros((1, self.relation_width)), mx.cumsum(tokens, axis=0)))
        starts = mx.array([span.start for span in spans], dtype=mx.int32)
        ends = mx.array([span.end for span in spans], dtype=mx.int32)
        first, last = tokens[starts], tokens[ends - 1]
        mean = (prefix[ends] - prefix[starts]) / (ends - starts)[:, None]
        context = mx.broadcast_to((mx.mean(tokens, axis=0) + tokens[-1]) / 2., mean.shape)
        logits = self.output(nn.silu(self.compose(mx.concatenate(
            (first, last, mean, context, last - first, first * last), axis=-1))))
        return logits[:, :-1] - logits[:, -1:]

    def source_loss(self, states, supervision):
        if len(states) != supervision.token_count:
            raise ValueError("native operation source and observations differ")
        spans, targets = supervision.indices(self)
        energies = self(states, spans)
        positive = sum((energies[row, column] for row, column in targets), mx.array(0.))
        return operation_set_log_partition(energies, spans, len(states), self.max_steps) - positive

    def proposal(self, source_id, states, *, max_expansions=200_000, chart_limit=64):
        return NativeOperationProposal(self, source_id, states,
            max_expansions=max_expansions, chart_limit=chart_limit)


class NativeOperationProposal:
    """Source-bound runtime callback; no annotation enters the operation bank."""
    def __init__(self, field, source_id, states, *, max_expansions, chart_limit):
        if (not isinstance(source_id, str) or not source_id
                or type(max_expansions) is not int or max_expansions < 1
                or type(chart_limit) is not int or chart_limit < 1
                or not mx.all(mx.isfinite(states)).item()):
            raise ValueError("native operation proposal needs finite source-qualified observations")
        self.field, self.source_id, self.states = field, source_id, states
        self.max_expansions, self.last_receipt = max_expansions, None
        self.chart_limit = chart_limit

    def __call__(self, *, source_id, source_token_ids, input_spans, max_steps):
        from itertools import islice

        from core.learning.semantic_operation_search import OperationChartSearch
        from core.learning.semantic_program_transducer_fitting import _OperationNode

        if source_id != self.source_id or len(source_token_ids) != len(self.states):
            raise ValueError("native operation proposal changed its public source")
        if type(max_steps) is not int or not 1 <= max_steps <= self.field.max_steps:
            raise ValueError("runtime operation count exceeds the fitted field grammar")
        spans = operation_span_inventory(len(self.states), self.field.max_span_tokens, input_spans)
        energies = self.field(self.states, spans)
        mx.eval(energies)
        values = energies.tolist()
        if any(not math.isfinite(score) for row in values for score in row):
            raise ValueError("native operation proposal produced nonfinite scores")
        nodes = tuple(_OperationNode(span, label, score, score, 1.)
            for span, row in zip(spans, values, strict=True)
            for label, score in zip(self.field.labels, row, strict=True))
        self.last_receipt = {"schema": "aura.native_operation_proposal.v1", "source_id": source_id,
            "span_count": len(spans), "label_count": len(self.field.labels), "node_count": len(nodes),
            "length_penalty": 0., "score": "primitive_minus_null_logit",
            "target_available_to_proposer": False, "inventory_pruned_by_parent": False,
            "chart_limit": self.chart_limit, "runtime_search_exhaustive": False,
            "whole_program_calibration_proven": False, "serving_authority": False}
        return islice(OperationChartSearch(nodes, max_steps=max_steps, length_penalty=0.,
                                    max_expansions=self.max_expansions), self.chart_limit)


@invariant("learning.operation_partition_includes_missing_and_extra_steps", scope="learning",
           owner="core/learning/semantic_native_operation_field.py", observational=False)
def _partition_counts_every_chart_once():
    value = operation_set_log_partition(mx.zeros((2, 2)),
        (TokenSpan(0, 1), TokenSpan(1, 2)), 2, 2)
    # Empty, four one-step choices and four two-step choices.
    assert abs(float(value.item()) - math.log(9.)) < 1e-5
    return ()
