"""Evaluate shared graph evidence without one dense gradient per witness."""

from math import fsum

import numpy as np
from scipy.special import softmax

from core.verify.invariants import invariant


class GraphConstraintBatch:
    """Chunk operation banks; preserve each witness and the runtime mixture score."""

    def __init__(self, contrasts, scale=1., *, max_feature_bytes=32 * 1024 * 1024):
        if not np.isfinite(scale) or scale <= 0 or type(max_feature_bytes) is not int or max_feature_bytes < 1:
            raise ValueError("invalid graph batch configuration")
        self.rows, self.scale = tuple(contrasts), scale
        self.max_feature_bytes = max_feature_bytes
        self.banks, bank_indices, entries = [], {}, []
        self.relations, self.arguments, self.normalizers = {}, {}, {}
        for row_index, row in enumerate(self.rows):
            for sign, choices in ((1., row.positive_operations), (-1., row.negative_operations)):
                for bank, label in choices:
                    if id(bank) not in bank_indices:
                        bank_indices[id(bank)] = len(self.banks)
                        self.banks.append(bank)
                    entries.append((row_index, bank_indices[id(bank)], label, sign))
                    if bank.normalizer_label is not None:
                        entries.append((row_index, bank_indices[id(bank)], bank.normalizer_label, -sign))
            for sign, choices in ((1., row.positive), (-1., row.negative)):
                for bank, label in choices:
                    key = (id(bank), label)
                    self.relations.setdefault(key, (bank, label, []))[2].append((row_index, sign))
            for sign, term in row.argument_terms:
                self.arguments.setdefault(id(term), (term, []))[1].append((row_index, sign))
            for sign, term in row.normalizer_terms:
                self.normalizers.setdefault(id(term), (term, []))[1].append((row_index, sign))
        self.entries = tuple(entries)
        groups = {}
        for index, bank in enumerate(self.banks):
            geometry = tuple(feature.size for feature in bank.features)
            groups.setdefault(geometry, []).append(index)
        self.chunks = []
        for geometry, indices in groups.items():
            size = max(1, min(128, max_feature_bytes // (8 * sum(geometry))))
            for start in range(0, len(indices), size):
                selected = indices[start:start + size]
                lookup = {index: local for local, index in enumerate(selected)}
                terms = [(row, lookup[bank], label, sign) for row, bank, label, sign in entries if bank in lookup]
                self.chunks.append((tuple(selected), tuple(terms)))

    def _operations(self, parameters):
        for selected, terms in self.chunks:
            banks = [self.banks[index] for index in selected]
            features, distributions = [], []
            for view in range(len(banks[0].features)):
                weight, bias = parameters[2 + 2 * view:4 + 2 * view]
                feature = np.stack([bank.features[view] for bank in banks])
                if (weight.ndim != 2 or weight.shape[1] != feature.shape[1]
                        or bias.shape != (len(weight),) or len(weight) < 2
                        or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias))):
                    raise ValueError("graph batch classifier geometry differs")
                features.append(feature)
                distributions.append(softmax(feature @ weight.T + bias, axis=1))
            if len({value.shape for value in distributions}) != 1:
                raise ValueError("graph batch class inventories differ")
            if any(type(label) is not int or not 0 <= label < distributions[0].shape[1]
                   for _, _, label, _ in terms):
                raise ValueError("graph batch operation label is invalid")
            mass = sum(distributions)
            yield terms, features, distributions, mass, mass / len(distributions)

    def margins(self, parameters):
        # Sum each complete contrast once. Adding a small retained floor before
        # cancelling large shared scores can manufacture a margin violation.
        values = [[row.fixed_margin] for row in self.rows]
        for bank, label, terms in self.relations.values():
            score = self.scale * bank.score(label, *parameters[:2])
            for row, sign in terms:
                values[row].append(sign * score)
        for term, occurrences in self.arguments.values():
            score = term.score(parameters)
            for row, sign in occurrences:
                values[row].append(sign * score)
        for term, occurrences in self.normalizers.values():
            score = term.score(parameters)
            for row, sign in occurrences:
                values[row].append(sign * score)
        for terms, _, _, _, probability in self._operations(parameters):
            scores = np.log(np.maximum(probability, 1e-12))
            for row, bank, label, sign in terms:
                values[row].append(sign * scores[bank, label])
        return np.fromiter((fsum(row) for row in values), dtype=np.float64, count=len(values))

    def weighted_gradient(self, parameters, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (len(self.rows),) or not np.all(np.isfinite(coefficients)):
            raise ValueError("graph batch coefficients differ")
        gradients = [np.zeros_like(value, dtype=np.float64) for value in parameters]
        for bank, label, terms in self.relations.values():
            coefficient = sum(coefficients[row] * sign for row, sign in terms) * self.scale
            if coefficient:
                _, query, definition = bank.score_gradient(label, *parameters[:2])
                gradients[0] += coefficient * query
                gradients[1] += coefficient * definition
        for term, occurrences in self.arguments.values():
            coefficient = sum(coefficients[row] * sign for row, sign in occurrences)
            if coefficient:
                _, weight, bias = term.score_gradient(parameters)
                gradients[term.parameter_index] += coefficient * weight
                gradients[term.parameter_index + 1] += coefficient * bias
        for term, occurrences in self.normalizers.values():
            coefficient = sum(coefficients[row] * sign for row, sign in occurrences)
            if coefficient:
                _, derivatives = term.score_gradient(parameters)
                for gradient, derivative in zip(gradients, derivatives, strict=True):
                    gradient += coefficient * derivative
        for terms, features, distributions, mass, probability in self._operations(parameters):
            cotangent = np.zeros_like(mass)
            for row, bank, label, sign in terms:
                cotangent[bank, label] += coefficients[row] * sign
            cotangent = np.divide(cotangent, mass, out=np.zeros_like(mass), where=probability > 1e-12)
            for view, (feature, distribution) in enumerate(zip(features, distributions, strict=True)):
                weighted = cotangent * distribution
                delta = weighted - distribution * weighted.sum(axis=1, keepdims=True)
                gradients[2 + 2 * view] += delta.T @ feature
                gradients[3 + 2 * view] += delta.sum(axis=0)
        return tuple(gradients)

    def directional_derivative(self, parameters, direction):
        if len(parameters) != len(direction) or any(a.shape != b.shape for a, b in zip(parameters, direction, strict=True)):
            raise ValueError("graph batch direction geometry differs")
        values = np.zeros(len(self.rows))
        for bank, label, terms in self.relations.values():
            _, query, definition = bank.score_gradient(label, *parameters[:2])
            slope = self.scale * (np.sum(query * direction[0]) + np.sum(definition * direction[1]))
            for row, sign in terms:
                values[row] += sign * slope
        for term, occurrences in self.arguments.values():
            _, weight, bias = term.score_gradient(parameters)
            slope = np.sum(weight * direction[term.parameter_index]) + np.sum(bias * direction[term.parameter_index + 1])
            for row, sign in occurrences:
                values[row] += sign * slope
        for term, occurrences in self.normalizers.values():
            _, derivatives = term.score_gradient(parameters)
            slope = sum(np.sum(a * b) for a, b in zip(derivatives, direction, strict=True))
            for row, sign in occurrences:
                values[row] += sign * slope
        for terms, features, distributions, mass, probability in self._operations(parameters):
            derivative = np.zeros_like(mass)
            for view, (feature, distribution) in enumerate(zip(features, distributions, strict=True)):
                logits = feature @ direction[2 + 2 * view].T + direction[3 + 2 * view]
                derivative += distribution * (logits - (distribution * logits).sum(axis=1, keepdims=True))
            slopes = np.divide(derivative, mass, out=np.zeros_like(mass), where=probability > 1e-12)
            for row, bank, label, sign in terms:
                values[row] += sign * slopes[bank, label]
        return values


@invariant("learning.shared_graph_scores_preserve_retained_margin", scope="learning",
           owner="core/learning/semantic_graph_batch.py", observational=False)
def _shared_score_cancellation() -> tuple:
    from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm
    from core.learning.semantic_relation_graph_learning import RelationGraphContrast, graph_margin_gradient

    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([100.]), np.array(0.))
    term = ArgumentScoreTerm(2, np.ones(1), 1., "conditional_log_odds_v1")
    row = RelationGraphContrast((), (), .1, argument_terms=((1., term), (-1., term)))
    assert graph_margin_gradient(parameters, row)[0] == .1
    assert GraphConstraintBatch((row,)).margins(parameters)[0] == .1
    return ()
