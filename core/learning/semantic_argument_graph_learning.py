"""Differentiate the argument-mention terms already used by the runtime chart."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class ArgumentScoreTerm:
    parameter_index: int
    feature: np.ndarray
    scale: float
    strategy: str

    def __post_init__(self):
        feature = np.asarray(self.feature, dtype=np.float64)
        if (type(self.parameter_index) is not int or self.parameter_index < 2
                or feature.ndim != 1 or not feature.size or not np.all(np.isfinite(feature))
                or not np.isfinite(self.scale) or self.scale < 0
                or self.strategy not in {"conditional_log_odds_v1", "independent_positive_v1"}):
            raise ValueError("invalid argument graph evidence")
        object.__setattr__(self, "feature", feature)

    def _logit(self, parameters):
        if self.parameter_index + 1 >= len(parameters):
            raise ValueError("argument graph parameter block is missing")
        weight, bias = parameters[self.parameter_index:self.parameter_index + 2]
        if weight.shape != self.feature.shape or np.size(bias) != 1:
            raise ValueError("argument graph parameter geometry differs")
        return float(self.feature @ weight + np.asarray(bias).item())

    def score(self, parameters):
        logit = self._logit(parameters)
        return float(self.scale * (logit if self.strategy == "conditional_log_odds_v1"
                                   else -np.logaddexp(0., -logit)))

    def score_gradient(self, parameters):
        logit = self._logit(parameters)
        bias = parameters[self.parameter_index + 1]
        if self.strategy == "conditional_log_odds_v1":
            value, slope = logit, 1.
        elif self.strategy == "independent_positive_v1":
            value, slope = -np.logaddexp(0., -logit), expit(-logit)
        else:
            raise ValueError("unknown argument graph score strategy")
        return float(self.scale * value), self.scale * slope * self.feature, np.full_like(bias, self.scale * slope)


def argument_parameters(model):
    return tuple(np.asarray(value, dtype=np.float64)
                 for pair in zip(model.argument_role_heads, model.argument_proposal_heads, strict=True)
                 for head in pair for value in (head.weight, head.bias))


def argument_slot_evidence(model, hidden, operation_span, position, mention):
    from core.learning.semantic_program_transducer_fitting import (
        _directional_relation_feature, _relation_span_vector,
    )

    def vector(span):
        return _relation_span_vector(hidden, span, hidden_channels=model.hidden_channels,
                                     hidden_channel_widths=model.hidden_channel_widths)

    offset = 2 + 2 * len(model.operation_head.heads)
    strategy = model.training_receipt.get("argument_score_strategy", "independent_positive_v1")
    feature = _directional_relation_feature(vector(mention), vector(operation_span))
    return (
        ArgumentScoreTerm(offset + 4 * position, feature, model.argument_role_scale, strategy),
        ArgumentScoreTerm(offset + 4 * position + 2, feature, model.argument_proposal_scale, strategy),
    )


def argument_graph_evidence(model, hidden, nodes, spans):
    terms = []
    for node, mentions in zip(nodes, spans, strict=True):
        for position, mention in enumerate(mentions):
            terms.extend(argument_slot_evidence(model, hidden, node.span, position, mention))
    return tuple(terms)
