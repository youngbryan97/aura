"""Differentiate the shipped multiview operation score in a complete graph."""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class OperationEvidenceBank:
    features: tuple

    def __post_init__(self):
        features = tuple(np.asarray(value, dtype=np.float64) for value in self.features)
        if not features or any(value.ndim != 1 or not value.size or not np.all(np.isfinite(value))
                               for value in features):
            raise ValueError("operation graph features are invalid")
        object.__setattr__(self, "features", features)

    def score_gradient(self, selected, parameters):
        """Use log(mean(softmax(view))) rather than softmax(mean(view))."""
        if len(parameters) != 2 * len(self.features):
            raise ValueError("operation graph views differ from parameters")
        distributions = []
        for index, feature in enumerate(self.features):
            weight, bias = parameters[2 * index:2 * index + 2]
            if (weight.ndim != 2 or weight.shape[1] != feature.size
                    or bias.shape != (len(weight),) or len(weight) < 2
                    or type(selected) is not int or not 0 <= selected < len(weight)
                    or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias))):
                raise ValueError("operation graph classifier geometry differs")
            distributions.append(softmax(weight @ feature + bias))
        if len({len(row) for row in distributions}) != 1:
            raise ValueError("operation graph class inventories differ")
        mass = sum(row[selected] for row in distributions)
        probability = mass / len(distributions)
        gradients = []
        for feature, distribution in zip(self.features, distributions, strict=True):
            delta = -distribution.copy()
            delta[selected] += 1.
            delta *= distribution[selected] / mass if probability > 1e-12 else 0.
            gradients.extend((np.outer(delta, feature), delta))
        return float(np.log(max(probability, 1e-12))), tuple(gradients)


def operation_graph_evidence(model, hidden, nodes):
    from core.learning.semantic_program_transducer import _operation_feature

    return tuple((OperationEvidenceBank(tuple(_operation_feature(hidden, node.span, mode=mode,
        hidden_channels=model.hidden_channels, hidden_channel_widths=model.hidden_channel_widths)
        for mode in model.operation_head.modes)), model.operation_head.labels.index(node.operation))
        for node in nodes)
