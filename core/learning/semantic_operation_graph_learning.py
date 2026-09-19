"""Differentiate the shipped multiview operation score in a complete graph."""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from core.verify.invariants import invariant


@dataclass(frozen=True)
class OperationEvidenceBank:
    features: tuple
    normalizer_label: int | None = None

    def __post_init__(self):
        features = tuple(np.asarray(value, dtype=np.float64) for value in self.features)
        if not features or any(value.ndim != 1 or not value.size or not np.all(np.isfinite(value))
                               for value in features):
            raise ValueError("operation graph features are invalid")
        if self.normalizer_label is not None and (
                type(self.normalizer_label) is not int or self.normalizer_label < 0):
            raise ValueError("operation graph normalizer is invalid")
        object.__setattr__(self, "features", features)

    def score(self, selected, parameters):
        if len(parameters) != 2 * len(self.features):
            raise ValueError("operation graph views differ from parameters")
        if type(selected) is not int or not 0 <= selected < len(parameters[0]):
            raise ValueError("selected operation hypothesis is invalid")
        distributions = [softmax(parameters[2 * index] @ feature + parameters[2 * index + 1])
                         for index, feature in enumerate(self.features)]
        def log_mass(label):
            if not 0 <= label < len(distributions[0]):
                raise ValueError("operation graph normalizer is invalid")
            return float(np.log(max(sum(row[label] for row in distributions) / len(distributions), 1e-12)))
        return log_mass(selected) - (log_mass(self.normalizer_label) if self.normalizer_label is not None else 0.)

    def score_gradient(self, selected, parameters):
        score, gradient = self._log_probability_gradient(selected, parameters)
        if self.normalizer_label is None:
            return score, gradient
        normalizer, normalizer_gradient = self._log_probability_gradient(self.normalizer_label, parameters)
        return score - normalizer, tuple(a - b for a, b in zip(gradient, normalizer_gradient, strict=True))

    def _log_probability_gradient(self, selected, parameters):
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
    from core.learning.semantic_program_transducer import _operation_feature, OPERATION_BACKGROUND_LABEL

    normalizer = (model.operation_head.labels.index(OPERATION_BACKGROUND_LABEL)
                  if model.training_receipt.get("operation_background_fit", {}).get("score")
                  == "joint_operation_background_log_odds_v2" else None)
    return tuple((OperationEvidenceBank(tuple(_operation_feature(hidden, node.span, mode=mode,
        hidden_channels=model.hidden_channels, hidden_channel_widths=model.hidden_channel_widths)
        for mode in model.operation_head.modes), normalizer), model.operation_head.labels.index(node.operation))
        for node in nodes)


@dataclass(frozen=True)
class OperationSourceSupervision:
    """Batched source-label likelihood using the same multiview mixture as decode."""

    features: tuple
    labels: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        features = tuple(np.asarray(value, dtype=np.float64) for value in self.features)
        labels = np.asarray(self.labels)
        weights = np.asarray(self.weights, dtype=np.float64)
        if (labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'iu'
                or np.any(labels < 0) or weights.shape != labels.shape
                or not np.all(np.isfinite(weights)) or np.any(weights <= 0)
                or not np.isfinite(weights.sum()) or not features
                or any(value.ndim != 2 or value.shape[0] != len(labels) or not value.shape[1]
                       or not np.all(np.isfinite(value)) for value in features)):
            raise ValueError('invalid operation source supervision')
        object.__setattr__(self, 'features', features)
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'weights', weights / weights.sum())

    def loss_gradient(self, parameters):
        if len(parameters) != 2 * len(self.features):
            raise ValueError('operation source views differ from parameters')
        distributions = []
        for index, feature in enumerate(self.features):
            weight, bias = parameters[2 * index:2 * index + 2]
            if (weight.ndim != 2 or weight.shape[1] != feature.shape[1]
                    or bias.shape != (len(weight),) or len(weight) < 2
                    or np.max(self.labels) >= len(weight)
                    or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias))):
                raise ValueError('operation source classifier geometry differs')
            distributions.append(softmax(feature @ weight.T + bias, axis=1))
        if len({value.shape[1] for value in distributions}) != 1:
            raise ValueError('operation source class inventories differ')
        rows = np.arange(len(self.labels))
        mass = sum(value[rows, self.labels] for value in distributions)
        probability = mass / len(distributions)
        loss = -float(self.weights @ np.log(np.maximum(probability, 1e-12)))
        gradients = []
        for feature, distribution in zip(self.features, distributions, strict=True):
            coefficient = np.divide(distribution[rows, self.labels], mass,
                                    out=np.zeros_like(mass), where=probability > 1e-12)
            delta = distribution.copy()
            delta[rows, self.labels] -= 1.
            delta *= (self.weights * coefficient)[:, None]
            gradients.extend((delta.T @ feature, delta.sum(axis=0)))
        return loss, tuple(gradients)


@invariant('learning.source_operation_loss_matches_runtime_mixture', scope='learning',
           owner='core/learning/semantic_operation_graph_learning.py', observational=False)
def _source_loss_matches_decode() -> tuple:
    features = (np.array([1., .5]),)
    parameters = (np.array([[.2, .3], [-.4, .1]]), np.zeros(2))
    value, derivatives = OperationEvidenceBank(features).score_gradient(1, parameters)
    source = OperationSourceSupervision(tuple(row[None, :] for row in features),
                                       np.array([1]), np.ones(1))
    loss, gradients = source.loss_gradient(parameters)
    assert np.isclose(loss, -value)
    assert all(np.allclose(a, -b) for a, b in zip(gradients, derivatives, strict=True))
    return ()
