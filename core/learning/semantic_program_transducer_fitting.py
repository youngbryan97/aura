"""Fitting the compositional transducer: every head, and what picks its scale.

The transducer itself is a small thing — a set of heads and the contract they
agree on. What takes the room is arriving at them: proposing operation nodes
over a chart and keeping the best non-overlapping cover, fitting the argument
role and proposal heads, choosing the scale each one is fitted at, training the
low-rank relation tissue, and assigning typed arguments once all of that holds.

They live apart from the model they produce so that reading the model does not
mean reading the search that found it.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import (
    SemanticValue,
    TokenSpan,
)
from core.learning.semantic_program_shared_transducer import (
    _geometry,
    _normalized_weights,
    _relation_span_vector,
)
from core.learning.semantic_program_transducer import (
    LinearPointerHead,
    LinearPointerSequenceScores,
    MultiViewClassifierHead,
    SemanticTransducerTrainingExample,
    _fit_binary_head,
    _operation_feature,
)
from core.learning.semantic_relation_tissue import (
    _DIRECTIONAL_RELATION_PARTS,
    DirectionalRelationHead,
    _directional_relation_feature,
)
from core.learning.semantic_relation_tissue import (
    _fit_directional_relation_head as _fit_directional_relation_head,
)
from core.learning.semantic_relation_tissue import (
    _fit_low_rank_relation_tissue as _fit_low_rank_relation_tissue,
)
from core.learning.semantic_relation_tissue import (
    _relation_decision_batch as _relation_decision_batch,
)
from core.learning.semantic_relation_tissue import (
    _relation_tissue_logits as _relation_tissue_logits,
)
from core.learning.semantic_relation_tissue import (
    _relation_tissue_metrics as _relation_tissue_metrics,
)
from core.learning.semantic_relation_tissue import (
    _RelationDecisionBatch as _RelationDecisionBatch,
)

if TYPE_CHECKING:  # the model this module fits imports this module, so the
    # name is needed for the annotation and must not be needed at import time
    from .semantic_program_compositional_transducer import (
        CompositionalSemanticProgramTransducer,
    )

COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v16"


_LEGACY_DEFINITION_CANDIDATE_STRATEGY: Final = "anchored_envelope_v1"


_LOCAL_DEFINITION_CANDIDATE_STRATEGY: Final = "bounded_local_alias_v1"


_STABLE_REGISTER_TABLE_STRATEGY: Final = "stable_register_table_v1"


_OPERATION_MODE: Final = "contextual_mean"


_OPERATION_CANDIDATES: Final = 256


_ARGUMENT_CANDIDATES: Final = 128


_LOCAL_ARGUMENT_CANDIDATES: Final = 256


_ARGUMENT_BEAM: Final = 128


_ARGUMENT_MENTIONS_PER_DEFINITION: Final = 4


_LOCAL_DEFINITION_CANDIDATES: Final = 16


_POINTER_HARD_NEGATIVES: Final = 16


_OPERATION_PENALTY_POINTS: Final = 161


_OPERATION_CHART_BEAM: Final = 16




_RELATION_TISSUE_RANK: Final = 16


_RELATION_TISSUE_SEED: Final = 1729


_RELATION_TISSUE_EPOCHS: Final = 120


_RELATION_TISSUE_BATCH_SIZE: Final = 128


_RELATION_TISSUE_SELECTION_INTERVAL: Final = 5


_RELATION_TISSUE_LEARNING_RATE: Final = 0.01


_RELATION_TISSUE_WEIGHT_DECAY: Final = 0.001


_RELATION_TISSUE_GRADIENT_CLIP: Final = 1.0


_ARGUMENT_PROPOSAL_SCALES: Final = tuple(float(value) for value in np.linspace(0.0, 1.5, 13))


def _log_sigmoid(value: float) -> float:
    """Stable log probability for one binary-head logit."""

    return -math.log1p(math.exp(-abs(value))) + min(value, 0.0)


def _argument_semantic_evidence(
    role_logit: float,
    proposal_logit: float,
    *,
    role_scale: float,
    proposal_scale: float,
    strategy: str,
) -> float:
    """Score a mention under the declared binary-head selection objective."""
    if strategy == "conditional_log_odds_v1":
        # Conditioning binary edge labels on one selected edge cancels the
        # shared negative-label term, leaving the chosen edge's log odds.
        return role_scale * role_logit + proposal_scale * proposal_logit
    if strategy == "independent_positive_v1":
        return role_scale * _log_sigmoid(role_logit) + proposal_scale * _log_sigmoid(proposal_logit)
    raise ValueError("unknown semantic argument scoring objective")


def _mention_invariant_relation_evidence(
    base_logits: Sequence[float],
    combined_logits: Sequence[float],
    *,
    strategy: str = "positive_label_margin_v1",
) -> tuple[float, ...]:
    """Let tissue choose a register without changing mention evidence."""

    if (
        not base_logits
        or len(base_logits) != len(combined_logits)
        or not all(math.isfinite(value) for value in (*base_logits, *combined_logits))
    ):
        raise ValueError("compositional relation logits are invalid")
    base_evidence = tuple(_log_sigmoid(value) for value in base_logits)
    if strategy == "categorical_log_margin_v1":
        # A shared log-softmax normalizer cancels in register score differences.
        combined_evidence = tuple(combined_logits)
    elif strategy == "positive_label_margin_v1":
        combined_evidence = tuple(_log_sigmoid(value) for value in combined_logits)
    else:
        raise ValueError("unknown semantic relation scoring objective")
    mention_evidence = max(base_evidence)
    combined_peak = max(combined_evidence)
    return tuple(mention_evidence + value - combined_peak for value in combined_evidence)


def _overlap(left: TokenSpan, right: TokenSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _all_semantic_spans(
    item: SemanticTransducerTrainingExample,
) -> tuple[TokenSpan, ...]:
    return (
        *item.ir.input_spans,
        *(instruction.operation_span for instruction in item.ir.instructions),
        *(span for instruction in item.ir.instructions for span in instruction.argument_spans),
    )


def _register_definition_spans(
    item: SemanticTransducerTrainingExample,
) -> tuple[TokenSpan, ...]:
    definitions = item.register_definition_spans or (
        *item.ir.input_spans,
        *(instruction.operation_span for instruction in item.ir.instructions),
    )
    if len(definitions) != item.ir.n_inputs + len(item.ir.instructions):
        raise ValueError("compositional register-definition geometry differs")
    return definitions


def _definition_span_candidates(
    anchor: TokenSpan,
    *,
    token_count: int,
    max_span_tokens: int,
    direction: Literal["left", "right"] = "right",
) -> tuple[TokenSpan, ...]:
    """Enumerate register envelopes toward where its name can be introduced.

    Public literals conventionally follow their names (``reserve 7``), while a
    computed value's name follows the operation that defines it.  Keeping the
    direction explicit makes runtime capable of representing the same spans
    used by relation training without opening a quadratic all-span search.
    """

    if direction == "left":
        start = min(anchor.start, max(0, anchor.end - max_span_tokens))
        return tuple(TokenSpan(index, anchor.end) for index in range(start, anchor.start + 1))
    if direction == "right":
        stop = max(anchor.end, min(token_count, anchor.start + max_span_tokens))
        return tuple(TokenSpan(anchor.start, end) for end in range(anchor.end, stop + 1))
    raise ValueError("definition span direction is invalid")


def _register_definition_candidates(
    anchors: Sequence[TokenSpan],
    *,
    input_count: int,
    token_count: int,
    max_span_tokens: int,
    pointer_scores: LinearPointerSequenceScores,
    strategy: str,
    source_ordered_boundaries: bool = False,
) -> tuple[tuple[TokenSpan, ...], ...]:
    """Localize each register inside its bounded defining clause.

    The legacy decoder represented a computed value only with envelopes that
    began at its operation verb. Natural language often names that value at
    the other end of the clause, and a long literal can place an input's name
    outside any value-ending envelope. Local strategies add learned subspans
    inside the defining clause. The stable-table strategy resolves exactly one
    identity span per register before any argument mention is scored, so later
    uses cannot select contradictory definitions for the same register.
    """

    if not 0 <= input_count <= len(anchors):
        raise ValueError("definition candidate input count is invalid")
    if strategy not in {
        _LEGACY_DEFINITION_CANDIDATE_STRATEGY,
        _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
        _STABLE_REGISTER_TABLE_STRATEGY,
    }:
        raise ValueError("definition candidate strategy is invalid")
    result: list[tuple[TokenSpan, ...]] = []
    for index, anchor in enumerate(anchors):
        direction: Literal["left", "right"] = "left" if index < input_count else "right"
        envelopes = _definition_span_candidates(
            anchor,
            token_count=token_count,
            max_span_tokens=max_span_tokens,
            direction=direction,
        )
        if strategy == _LEGACY_DEFINITION_CANDIDATE_STRATEGY:
            result.append(envelopes)
            continue

        if direction == "left":
            clause_start = (
                max((other.end for other in anchors if other.end <= anchor.start), default=0)
                if source_ordered_boundaries
                else anchors[index - 1].end if index else 0
            )
            boundary_start = max(clause_start, anchor.start - max_span_tokens)
            local_bounds = (boundary_start, anchor.start)
            bounded_envelopes = (
                tuple(span for span in envelopes if span.start >= clause_start)
                if source_ordered_boundaries else envelopes
            )
        else:
            next_operation = (
                min((other.start for other in anchors[input_count:]
                     if other.start >= anchor.end), default=token_count)
                if source_ordered_boundaries
                else anchors[index + 1].start if index + 1 < len(anchors) else token_count
            )
            boundary = min(token_count, anchor.start + max_span_tokens, next_operation)
            local_bounds = (anchor.end, boundary)
            bounded_envelopes = tuple(span for span in envelopes if span.end <= boundary)
        local: list[tuple[TokenSpan, float]] = []
        for start in range(*local_bounds):
            stop = min(local_bounds[1], start + max_span_tokens)
            for end in range(start + 1, stop + 1):
                span = TokenSpan(start, end)
                local.append((span, pointer_scores.score_span(span)))
        local.sort(key=lambda item: (-item[1], item[0].start, item[0].end))
        candidates = tuple(
            dict.fromkeys(
                (
                    *(bounded_envelopes or (anchor,)),
                    *(span for span, _score in local[:_LOCAL_DEFINITION_CANDIDATES]),
                )
            )
        )
        if strategy == _STABLE_REGISTER_TABLE_STRATEGY:
            candidates = (
                max(
                    candidates,
                    key=lambda span: (
                        pointer_scores.score_span(span),
                        -(span.end - span.start),
                        -span.start,
                        -span.end,
                    ),
                ),
            )
        result.append(candidates)
    return tuple(result)


def _best_penalized_operation_chart(
    by_count: Sequence[tuple[float, tuple[_OperationNode, ...]]],
    *,
    penalty: float,
) -> tuple[_OperationNode, ...]:
    """Choose a chart, or an empty refusal when this item has no finite chart."""

    candidates = tuple(
        (score - penalty * count, nodes)
        for count, (score, nodes) in enumerate(by_count, start=1)
        if math.isfinite(score)
    )
    if not candidates:
        return ()
    return max(candidates, key=lambda value: (value[0], -len(value[1])))[1]


def _shared_pointer_training_indices(
    item: SemanticTransducerTrainingExample,
    positive_span: TokenSpan,
    *,
    end: bool,
    positive_indices: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    positive = positive_span.end - 1 if end else positive_span.start
    candidates = [(span.end - 1 if end else span.start) for span in _all_semantic_spans(item)]
    candidates.extend(
        (
            positive - 2,
            positive - 1,
            positive + 1,
            positive + 2,
            0,
            item.hidden_states.shape[0] - 1,
        )
    )
    negatives = tuple(
        index
        for index in dict.fromkeys(candidates)
        if 0 <= index < item.hidden_states.shape[0]
        and index != positive
        and index not in positive_indices
    )[:_POINTER_HARD_NEGATIVES]
    return (positive, *negatives)


def _fit_shared_pointer(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    spans: Callable[[SemanticTransducerTrainingExample], Sequence[TokenSpan]],
) -> LinearPointerHead:
    parameters: list[tuple[np.ndarray, float]] = []
    geometry_counts = Counter(_geometry(item) for item in training)
    for end in (False, True):
        features: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for item in training:
            positives = tuple(spans(item))
            if not positives:
                continue
            # This head detects every span of its role, not one exclusive slot.
            positive_indices = frozenset(
                span.end - 1 if end else span.start for span in positives
            )
            item_weight = 1.0 / geometry_counts[_geometry(item)]
            for positive in positives:
                indices = _shared_pointer_training_indices(
                    item,
                    positive,
                    end=end,
                    positive_indices=positive_indices,
                )
                features.extend(item.hidden_states[index] for index in indices)
                labels.extend((1, *(0 for _ in indices[1:])))
                weights.extend([item_weight / len(positives) / len(indices)] * len(indices))
        if not features:
            raise ValueError("compositional pointer has no training support")
        parameters.append(
            _fit_binary_head(
                np.stack(features),
                np.asarray(labels, dtype=np.int8),
                sample_weight=_normalized_weights(weights),
                max_iter=250,
                tolerance=1e-3,
            )
        )
    return LinearPointerHead(
        parameters[0][0],
        parameters[0][1],
        parameters[1][0],
        parameters[1][1],
    )


@dataclass(frozen=True, slots=True)
class LinearArgumentRoleHead:
    """Shared relation from an argument mention to one operation and slot."""

    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        if (
            weight.size < _DIRECTIONAL_RELATION_PARTS
            or weight.size % _DIRECTIONAL_RELATION_PARTS
            or not np.all(np.isfinite(weight))
            or not np.isfinite(self.bias)
        ):
            raise ValueError("compositional argument-role head is invalid")
        object.__setattr__(self, "weight", weight)

    @property
    def channel_width(self) -> int:
        return int(self.weight.size // _DIRECTIONAL_RELATION_PARTS)

    def score(self, reference: np.ndarray, operation: np.ndarray) -> float:
        feature = _directional_relation_feature(reference, operation)
        if feature.shape != self.weight.shape:
            raise ValueError("compositional argument-role feature width differs")
        return float(feature @ self.weight + self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {"weight": self.weight.tolist(), "bias": float(self.bias)}



@dataclass(frozen=True, slots=True)
class RegisterUseContract:
    """Source-learned bounds for a well-formed register-use graph."""

    input_min_uses: int
    input_max_uses: int
    intermediate_min_uses: int
    intermediate_max_uses: int
    distinct_arguments: bool

    def __post_init__(self) -> None:
        if (
            type(self.input_min_uses) is not int
            or type(self.input_max_uses) is not int
            or type(self.intermediate_min_uses) is not int
            or type(self.intermediate_max_uses) is not int
            or not 0 <= self.input_min_uses <= self.input_max_uses
            or not 0 <= self.intermediate_min_uses <= self.intermediate_max_uses
            or type(self.distinct_arguments) is not bool
        ):
            raise ValueError("compositional register-use contract is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_min_uses": self.input_min_uses,
            "input_max_uses": self.input_max_uses,
            "intermediate_min_uses": self.intermediate_min_uses,
            "intermediate_max_uses": self.intermediate_max_uses,
            "distinct_arguments": self.distinct_arguments,
        }

    def allows_partial(self, counts: Counter[int], *, n_inputs: int) -> bool:
        return all(
            count <= (self.input_max_uses if register < n_inputs else self.intermediate_max_uses)
            for register, count in counts.items()
        )

    def accepts_complete(
        self,
        counts: Counter[int],
        *,
        n_inputs: int,
        operation_count: int,
        sink: int,
    ) -> bool:
        return all(
            self.input_min_uses <= counts[index] <= self.input_max_uses for index in range(n_inputs)
        ) and all(
            self.intermediate_min_uses <= counts[n_inputs + index] <= self.intermediate_max_uses
            for index in range(operation_count)
            if index != sink
        )


@dataclass(frozen=True, slots=True)
class _OperationNode:
    span: TokenSpan
    operation: str
    score: float
    pointer_score: float
    confidence: float


def _argument_proposals_by_operation(
    scores: LinearPointerSequenceScores,
    *,
    input_spans: Sequence[TokenSpan],
    operation_nodes: Sequence[_OperationNode],
    max_span_tokens: int,
    clause_local: bool,
) -> tuple[tuple[tuple[TokenSpan, float], ...], ...]:
    """Allocate argument evidence independently to every operation clause."""

    global_proposals = list(
        scores.decode_candidates(
            limit=_ARGUMENT_CANDIDATES,
            max_span_tokens=max_span_tokens,
        )
    )
    observed = {span for span, _score in global_proposals}
    global_proposals.extend(
        (span, scores.score_span(span)) for span in input_spans if span not in observed
    )
    global_proposals = [
        (span, score)
        for span, score in global_proposals
        if not any(_overlap(span, node.span) for node in operation_nodes)
    ]
    if not clause_local:
        shared = tuple(global_proposals)
        return tuple(shared for _node in operation_nodes)

    by_operation: list[tuple[tuple[TokenSpan, float], ...]] = [()] * len(operation_nodes)
    token_count = scores.start.size
    # Training IR is execution-ordered; clauses are always source-ordered.
    source_order = sorted(
        range(len(operation_nodes)),
        key=lambda index: (operation_nodes[index].span.start, operation_nodes[index].span.end),
    )
    for source_index, node_index in enumerate(source_order):
        clause_start = (
            operation_nodes[source_order[source_index - 1]].span.end if source_index else 0
        )
        clause_end = (
            operation_nodes[source_order[source_index + 1]].span.start
            if source_index + 1 < len(source_order)
            else token_count
        )
        local: list[tuple[TokenSpan, float]] = []
        for start in range(clause_start, clause_end):
            stop = min(clause_end, start + max_span_tokens)
            for end in range(start + 1, stop + 1):
                span = TokenSpan(start, end)
                if any(_overlap(span, operation.span) for operation in operation_nodes):
                    continue
                local.append((span, scores.score_span(span)))
        local.sort(key=lambda item: (-item[1], item[0].start, item[0].end))
        merged: dict[TokenSpan, float] = {}
        for span, score in (*global_proposals, *local[:_LOCAL_ARGUMENT_CANDIDATES]):
            merged[span] = max(score, merged.get(span, -float("inf")))
        by_operation[node_index] = tuple(
            sorted(
                merged.items(),
                key=lambda item: (-item[1], item[0].start, item[0].end),
            )
        )
    return tuple(by_operation)


@dataclass(frozen=True, slots=True)
class _TypedArgumentAssignment:
    """One complete typed graph together with its neural evidence score."""

    operation_nodes: tuple[_OperationNode, ...]
    arguments: tuple[tuple[int, ...], ...]
    argument_spans: tuple[tuple[TokenSpan, ...], ...]
    score: float
    runner_up_score: float | None


def _operation_nodes(
    *,
    pointer: LinearPointerHead,
    classifier: MultiViewClassifierHead,
    hidden: np.ndarray,
    input_spans: Sequence[TokenSpan],
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[_OperationNode, ...]:
    nodes: list[_OperationNode] = []
    for span, pointer_score in pointer.decode_candidates(
        hidden,
        limit=_OPERATION_CANDIDATES,
        max_span_tokens=max_span_tokens,
    ):
        if any(_overlap(span, input_span) for input_span in input_spans):
            continue
        operation, confidence = classifier.predict(
            (
                _operation_feature(
                    hidden,
                    span,
                    mode=_OPERATION_MODE,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                ),
            )
        )
        score = float(pointer_score + math.log(max(confidence, 1e-12)))
        nodes.append(
            _OperationNode(
                span=span,
                operation=operation,
                score=score,
                pointer_score=float(pointer_score),
                confidence=confidence,
            )
        )
    return tuple(nodes)


def _best_nonoverlapping_nodes(
    nodes: Sequence[_OperationNode],
    count: int,
) -> tuple[float, tuple[_OperationNode, ...]]:
    """Exact cardinality-constrained weighted interval scheduling."""

    ordered = tuple(
        sorted(
            nodes,
            key=lambda item: (item.span.end, item.span.start, -item.score),
        )
    )
    previous: list[int] = []
    for index, node in enumerate(ordered):
        prior = index - 1
        while prior >= 0 and ordered[prior].span.end > node.span.start:
            prior -= 1
        previous.append(prior)
    impossible = (-float("inf"), ())
    table: list[list[tuple[float, tuple[_OperationNode, ...]]]] = [
        [impossible for _ in range(count + 1)] for _ in range(len(ordered) + 1)
    ]
    table[0][0] = (0.0, ())
    for index, node in enumerate(ordered, start=1):
        for size in range(count + 1):
            table[index][size] = table[index - 1][size]
            if size < 1:
                continue
            prior_score, prior_nodes = table[previous[index - 1] + 1][size - 1]
            if not math.isfinite(prior_score):
                continue
            candidate = (prior_score + node.score, (*prior_nodes, node))
            incumbent = table[index][size]
            if (candidate[0], tuple((n.span.start, n.span.end) for n in candidate[1])) > (
                incumbent[0],
                tuple((n.span.start, n.span.end) for n in incumbent[1]),
            ):
                table[index][size] = candidate
    return table[len(ordered)][count]


def _best_nonoverlapping_node_charts(
    nodes: Sequence[_OperationNode],
    count: int,
    *,
    limit: int,
) -> tuple[tuple[float, tuple[_OperationNode, ...]], ...]:
    """Exact top-k cardinality-constrained weighted interval scheduling."""

    if limit < 1:
        raise ValueError("compositional operation-chart limit must be positive")
    ordered = tuple(sorted(nodes, key=lambda item: (item.span.end, item.span.start, -item.score)))
    previous: list[int] = []
    for index, node in enumerate(ordered):
        prior = index - 1
        while prior >= 0 and ordered[prior].span.end > node.span.start:
            prior -= 1
        previous.append(prior)
    table: list[list[tuple[tuple[float, tuple[_OperationNode, ...]], ...]]] = [
        [() for _ in range(count + 1)] for _ in range(len(ordered) + 1)
    ]
    table[0][0] = ((0.0, ()),)
    for index, node in enumerate(ordered, start=1):
        for size in range(count + 1):
            candidates = list(table[index - 1][size])
            if size >= 1:
                candidates.extend(
                    (score + node.score, (*selected, node))
                    for score, selected in table[previous[index - 1] + 1][size - 1]
                )
            unique: dict[
                tuple[tuple[int, int, str], ...], tuple[float, tuple[_OperationNode, ...]]
            ] = {}
            for candidate in candidates:
                key = tuple(
                    (item.span.start, item.span.end, item.operation) for item in candidate[1]
                )
                incumbent = unique.get(key)
                if incumbent is None or candidate[0] > incumbent[0]:
                    unique[key] = candidate
            table[index][size] = tuple(
                sorted(
                    unique.values(),
                    key=lambda item: (
                        -item[0],
                        tuple((node.span.start, node.span.end) for node in item[1]),
                    ),
                )[:limit]
            )
    return table[len(ordered)][count]


def _operation_chart_candidates(
    nodes: Sequence[_OperationNode],
    *,
    max_steps: int,
    length_penalty: float,
    limit: int = _OPERATION_CHART_BEAM,
) -> tuple[tuple[_OperationNode, ...], ...]:
    candidates = [
        (score - length_penalty * count, selected)
        for count in range(1, max_steps + 1)
        for score, selected in _best_nonoverlapping_node_charts(
            nodes,
            count,
            limit=limit,
        )
    ]
    return tuple(
        selected
        for _score, selected in sorted(
            candidates,
            key=lambda item: (
                -item[0],
                len(item[1]),
                tuple((node.span.start, node.span.end) for node in item[1]),
            ),
        )[:limit]
    )


def _fit_argument_role_heads(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    max_arity: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[LinearArgumentRoleHead, ...]:
    heads: list[LinearArgumentRoleHead] = []
    geometry_counts = Counter(_geometry(item) for item in training)
    for position in range(max_arity):
        features: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for item in training:
            all_arguments = tuple(
                span for instruction in item.ir.instructions for span in instruction.argument_spans
            )
            for instruction in item.ir.instructions:
                if position >= len(instruction.argument_spans):
                    continue
                positive = instruction.argument_spans[position]
                negatives = tuple(
                    span
                    for span in dict.fromkeys(
                        (
                            *item.ir.input_spans,
                            *all_arguments,
                            *(step.operation_span for step in item.ir.instructions),
                        )
                    )
                    if span != positive
                )
                spans = (positive, *negatives)
                operation = _relation_span_vector(
                    item.hidden_states,
                    instruction.operation_span,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                )
                features.extend(
                    _directional_relation_feature(
                        _relation_span_vector(
                            item.hidden_states,
                            span,
                            hidden_channels=hidden_channels,
                            hidden_channel_widths=hidden_channel_widths,
                        ),
                        operation,
                    )
                    for span in spans
                )
                labels.extend((1, *(0 for _ in negatives)))
                decision_weight = 1.0 / geometry_counts[_geometry(item)] / len(spans)
                weights.extend([decision_weight] * len(spans))
        if not features:
            raise ValueError(f"compositional argument slot has no support: {position}")
        weight, bias = _fit_binary_head(
            np.stack(features),
            np.asarray(labels, dtype=np.int8),
            sample_weight=_normalized_weights(weights),
            max_iter=400,
            tolerance=1e-5,
        )
        heads.append(LinearArgumentRoleHead(weight, bias))
    return tuple(heads)


def _argument_proposal_rows(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    argument_pointer: LinearPointerHead,
    position: int,
    max_span_tokens: int,
    max_argument_span_tokens_by_type: Mapping[str, int],
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
    include_semantic_negatives: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    geometry_counts = Counter(_geometry(item) for item in examples)
    positive_rows = 0
    negative_rows = 0
    for item in examples:
        pointer_scores = argument_pointer.score_sequence(item.hidden_states)
        nodes = tuple(
            _OperationNode(instruction.operation_span, instruction.op, 0.0, 0.0, 1.0)
            for instruction in item.ir.instructions
        )
        proposals = _argument_proposals_by_operation(
            pointer_scores,
            input_spans=item.ir.input_spans,
            operation_nodes=nodes,
            max_span_tokens=max_span_tokens,
            clause_local=True,
        )
        for instruction, candidates in zip(item.ir.instructions, proposals, strict=True):
            if position >= len(instruction.argument_spans):
                continue
            signature = semantic_primitive_type_signature(instruction.op)
            if signature is None:
                raise ValueError(f"compositional primitive has no floor type: {instruction.op}")
            argument_types, _result_type = signature
            required_type = argument_types[position]
            positive = instruction.argument_spans[position]
            negatives = tuple(
                span
                for span, _score in candidates
                if span != positive
                and span.end - span.start <= max_argument_span_tokens_by_type[required_type]
            )[:_POINTER_HARD_NEGATIVES]
            if include_semantic_negatives:
                negatives = tuple(dict.fromkeys((
                    *negatives,
                    *(span for span in _all_semantic_spans(item)
                      if span != positive
                      and span.end - span.start <= max_argument_span_tokens_by_type[required_type]),
                )))
            spans = (positive, *negatives)
            operation = _relation_span_vector(
                item.hidden_states,
                instruction.operation_span,
                hidden_channels=hidden_channels,
                hidden_channel_widths=hidden_channel_widths,
            )
            features.extend(
                _directional_relation_feature(
                    _relation_span_vector(
                        item.hidden_states,
                        span,
                        hidden_channels=hidden_channels,
                        hidden_channel_widths=hidden_channel_widths,
                    ),
                    operation,
                )
                for span in spans
            )
            labels.extend((1, *(0 for _ in negatives)))
            decision_weight = 1.0 / geometry_counts[_geometry(item)] / len(spans)
            weights.extend([decision_weight] * len(spans))
            positive_rows += 1
            negative_rows += len(negatives)
    if not features:
        raise ValueError(f"compositional argument proposal slot has no support: {position}")
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        _normalized_weights(weights),
        positive_rows,
        negative_rows,
    )


def _fit_argument_proposal_heads(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    argument_pointer: LinearPointerHead,
    max_arity: int,
    max_span_tokens: int,
    max_argument_span_tokens_by_type: Mapping[str, int],
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[tuple[LinearArgumentRoleHead, ...], dict[str, int]]:
    """Fit slot evidence on the spans the pointer can actually propose at runtime."""

    heads: list[LinearArgumentRoleHead] = []
    positive_rows = 0
    negative_rows = 0
    for position in range(max_arity):
        features, labels, weights, positives, negatives = _argument_proposal_rows(
            training,
            argument_pointer=argument_pointer,
            position=position,
            max_span_tokens=max_span_tokens,
            max_argument_span_tokens_by_type=max_argument_span_tokens_by_type,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        weight, bias = _fit_binary_head(
            features,
            labels,
            sample_weight=weights,
            max_iter=400,
            tolerance=1e-5,
        )
        heads.append(LinearArgumentRoleHead(weight, bias))
        positive_rows += positives
        negative_rows += negatives
    return tuple(heads), {
        "positive_rows": positive_rows,
        "pointer_hard_negative_rows": negative_rows,
    }


def _select_argument_proposal_scale(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    argument_pointer: LinearPointerHead,
    semantic_heads: Sequence[LinearArgumentRoleHead],
    proposal_heads: Sequence[LinearArgumentRoleHead],
    max_span_tokens: int,
    max_argument_span_tokens_by_type: Mapping[str, int],
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    """Calibrate proposal evidence against source-only held-out pointer decisions."""

    totals = {scale: 0.0 for scale in _ARGUMENT_PROPOSAL_SCALES}
    row_correct = {scale: 0 for scale in _ARGUMENT_PROPOSAL_SCALES}
    row_count = 0
    for position, (semantic_head, proposal_head) in enumerate(
        zip(semantic_heads, proposal_heads, strict=True)
    ):
        features, labels, weights, _positives, _negatives = _argument_proposal_rows(
            validation,
            argument_pointer=argument_pointer,
            position=position,
            max_span_tokens=max_span_tokens,
            max_argument_span_tokens_by_type=max_argument_span_tokens_by_type,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        semantic_logits = features @ semantic_head.weight + semantic_head.bias
        proposal_logits = features @ proposal_head.weight + proposal_head.bias
        for scale in _ARGUMENT_PROPOSAL_SCALES:
            logits = semantic_logits + scale * proposal_logits
            losses = np.logaddexp(0.0, logits) - labels * logits
            totals[scale] += float(np.sum(losses * weights))
            row_correct[scale] += int(np.count_nonzero((logits >= 0.0) == labels))
        row_count += int(labels.size)
    if row_count < 1:
        raise ValueError("compositional argument proposal calibration has no support")
    rows = [
        {
            "proposal_scale": scale,
            "validation_cross_entropy": totals[scale],
            "row_correct": row_correct[scale],
            "row_count": row_count,
        }
        for scale in _ARGUMENT_PROPOSAL_SCALES
    ]
    winner = min(
        rows,
        key=lambda row: (
            row["validation_cross_entropy"],
            -row["row_correct"],
            row["proposal_scale"],
        ),
    )
    for row in rows:
        row["selected"] = row is winner
    return float(winner["proposal_scale"]), rows


def _fit_register_use_contract(
    training: Sequence[SemanticTransducerTrainingExample],
) -> RegisterUseContract:
    input_uses: list[int] = []
    intermediate_uses: list[int] = []
    distinct_arguments = True
    for item in training:
        counts: Counter[int] = Counter(
            register for instruction in item.ir.instructions for register in instruction.args
        )
        input_uses.extend(counts[index] for index in range(item.ir.n_inputs))
        intermediate_uses.extend(
            counts[item.ir.n_inputs + step]
            for step in range(len(item.ir.instructions))
            if item.ir.n_inputs + step != item.ir.report_value
        )
        distinct_arguments = distinct_arguments and all(
            len(set(instruction.args)) == len(instruction.args)
            for instruction in item.ir.instructions
        )
    if not input_uses or not intermediate_uses:
        raise ValueError("compositional register-use contract has insufficient support")
    return RegisterUseContract(
        input_min_uses=min(input_uses),
        input_max_uses=max(input_uses),
        intermediate_min_uses=min(intermediate_uses),
        intermediate_max_uses=max(intermediate_uses),
        distinct_arguments=distinct_arguments,
    )


def _select_definition_pointer_scale(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    relation_head: DirectionalRelationHead,
    definition_pointer: LinearPointerHead,
    max_definition_span_tokens: int,
    definition_candidate_strategy: str,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    """Calibrate definition localization only on source validation references."""

    scales = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
    correct = {scale: 0 for scale in scales}
    total = 0
    for item in validation:
        anchors = (
            *item.ir.input_spans,
            *(instruction.operation_span for instruction in item.ir.instructions),
        )
        pointer_scores = definition_pointer.score_sequence(item.hidden_states)
        candidate_spans = _register_definition_candidates(
            anchors,
            input_count=item.ir.n_inputs,
            token_count=item.hidden_states.shape[0],
            max_span_tokens=max_definition_span_tokens,
            pointer_scores=pointer_scores,
            strategy=definition_candidate_strategy,
        )
        candidates = tuple(
            tuple(
                (
                    span,
                    _relation_span_vector(
                        item.hidden_states,
                        span,
                        hidden_channels=hidden_channels,
                        hidden_channel_widths=hidden_channel_widths,
                    ),
                )
                for span in register_candidates
            )
            for register_candidates in candidate_spans
        )
        for step, instruction in enumerate(item.ir.instructions):
            available = item.ir.n_inputs + step
            for reference_span, expected_register in zip(
                instruction.argument_spans,
                instruction.args,
                strict=True,
            ):
                reference = _relation_span_vector(
                    item.hidden_states,
                    reference_span,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                )
                for scale in scales:
                    scores = tuple(
                        max(
                            relation_head.score(reference, definition)
                            + scale * pointer_scores.score_span(span)
                            for span, definition in register_candidates
                        )
                        for register_candidates in candidates[:available]
                    )
                    correct[scale] += int(
                        max(range(available), key=lambda index: scores[index]) == expected_register
                    )
                total += 1
    if total < 1:
        raise ValueError("compositional definition-pointer calibration has no support")
    rows = [
        {"pointer_scale": scale, "register_top1": correct[scale], "total": total}
        for scale in scales
    ]
    winner = max(rows, key=lambda row: (row["register_top1"], -row["pointer_scale"]))
    return float(winner["pointer_scale"]), rows


def _input_type(value: SemanticValue) -> str:
    return "integer" if isinstance(value, int) else "integer_sequence"


def _prefix_feasible_arguments(
    options_by_position: Sequence[Sequence[tuple[float, int, TokenSpan]]],
    *,
    arguments: Sequence[Sequence[int]],
    spans: Sequence[Sequence[TokenSpan]],
    dependencies: Sequence[Sequence[int]],
    operation_nodes: Sequence[_OperationNode],
    n_inputs: int,
    contract: RegisterUseContract,
    beam: int,
) -> list[tuple[float, tuple[int, ...], tuple[TokenSpan, ...]]]:
    """Spend the beam on continuations feasible for this graph prefix."""

    partial: list[tuple[float, tuple[int, ...], tuple[TokenSpan, ...]]] = [(0.0, (), ())]
    prefix_counts = Counter(register for step in arguments for register in step)
    used_spans = tuple(span for step in spans for span in step)
    final_operation = len(arguments) + 1 == len(operation_nodes)
    for position, options in enumerate(options_by_position):
        candidates = []
        for total, registers, mentions in partial:
            for score, register, span in options:
                if any(_overlap(span, previous) for previous in (*used_spans, *mentions)):
                    continue
                if contract.distinct_arguments and register in registers:
                    continue
                next_registers = (*registers, register)
                counts = prefix_counts + Counter(next_registers)
                if not contract.allows_partial(counts, n_inputs=n_inputs):
                    continue
                next_dependencies = (
                    *dependencies,
                    tuple(sorted({value - n_inputs for value in next_registers if value >= n_inputs})),
                )
                complete = final_operation and position + 1 == len(options_by_position)
                if _operation_order(
                    next_dependencies, operation_nodes, require_connected=complete
                ) is None:
                    continue
                if complete:
                    referenced = {value for values in next_dependencies for value in values}
                    sink = next(index for index in range(len(operation_nodes)) if index not in referenced)
                    if not contract.accepts_complete(
                        counts, n_inputs=n_inputs, operation_count=len(operation_nodes), sink=sink
                    ):
                        continue
                candidates.append((total + score, next_registers, (*mentions, span)))
        partial = sorted(
            candidates,
            key=lambda item: (-item[0], item[1], tuple((s.start, s.end) for s in item[2])),
        )[:beam]
        if not partial:
            break
    return partial


def _definition_relation_score_banks(
    head: DirectionalRelationHead,
    reference_vectors: Mapping[TokenSpan, np.ndarray],
    definition_vectors: Sequence[Sequence[tuple[TokenSpan, np.ndarray]]],
    pointer_scores: LinearPointerSequenceScores,
) -> tuple[dict[TokenSpan, tuple[float, ...]], dict[TokenSpan, tuple[float, ...]]]:
    """Reuse projections within one chart without changing scalar score arithmetic."""
    definitions = tuple(
        tuple(
            (definition, definition @ head.definition_projection,
             head.pointer_scale * pointer_scores.score_span(span))
            for span, definition in candidates
        )
        for candidates in definition_vectors
    )
    combined = {}
    base = {}
    for span, reference in reference_vectors.items():
        query = reference @ head.query_projection
        combined_registers = []
        base_registers = []
        for candidates in definitions:
            combined_candidates = []
            base_candidates = []
            for definition, projected, pointer in candidates:
                base_score = head.base_score(reference, definition)
                tissue_score = float(query @ projected)
                combined_candidates.append(base_score + tissue_score + pointer)
                base_candidates.append(base_score + pointer)
            combined_registers.append(max(combined_candidates))
            base_registers.append(max(base_candidates))
        combined[span] = tuple(combined_registers)
        base[span] = tuple(base_registers)
    return combined, base


def _assign_typed_arguments(
    *,
    model: CompositionalSemanticProgramTransducer,
    hidden: np.ndarray,
    inputs: Sequence[SemanticValue],
    input_spans: Sequence[TokenSpan],
    operation_nodes: Sequence[_OperationNode],
    argument_pointer_scores: LinearPointerSequenceScores,
) -> _TypedArgumentAssignment | None:
    if (
        len(operation_nodes) > 1
        and not model.allow_computed_dependencies
        and model.register_use_contract.intermediate_min_uses > 0
    ):
        return None
    proposals_by_operation = _argument_proposals_by_operation(
        argument_pointer_scores,
        input_spans=input_spans,
        operation_nodes=operation_nodes,
        max_span_tokens=model.max_span_tokens,
        clause_local=model.schema == COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA,
    )
    operation_types: list[tuple[tuple[str, ...], str]] = []
    for node in operation_nodes:
        signature = semantic_primitive_type_signature(node.operation)
        if signature is None:
            return None
        operation_types.append(signature)
    definitions = (*input_spans, *(node.span for node in operation_nodes))
    register_types = (
        *(_input_type(value) for value in inputs),
        *(result_type for _argument_types, result_type in operation_types),
    )
    definition_pointer_scores = model.definition_pointer.score_sequence(hidden)
    joint_definitions = model.training_receipt.get("definition_selection_policy") == "joint_graph_v1"
    definition_candidates = _register_definition_candidates(
        definitions,
        input_count=len(inputs),
        token_count=hidden.shape[0],
        max_span_tokens=model.max_definition_span_tokens,
        pointer_scores=definition_pointer_scores,
        strategy=(
            _LOCAL_DEFINITION_CANDIDATE_STRATEGY if joint_definitions
            else model.definition_candidate_strategy
        ),
        source_ordered_boundaries=(
            model.training_receipt.get("definition_boundary_policy") == "source_neighbors_v1"
        ),
    )
    definition_registers = tuple(range(len(definitions)))
    definition_labels = ()
    if joint_definitions:
        # The original anchor remains a hypothesis alongside the best learned
        # local spans. The graph chooses a shared identity across all uses.
        hypotheses = tuple(
            (register, span)
            for register, candidates in enumerate(definition_candidates)
            for span in dict.fromkeys((
                definitions[register],
                *sorted(candidates, key=lambda s: (
                    -definition_pointer_scores.score_span(s), s.end - s.start, s.start, s.end,
                ))[:4],
            ))
        )
        definition_registers = tuple(register for register, _span in hypotheses)
        definition_labels = tuple(span for _register, span in hypotheses)
        definition_candidates = tuple((span,) for span in definition_labels)
    definition_vectors = tuple(
        tuple(
            (
                candidate,
                _relation_span_vector(
                    hidden,
                    candidate,
                    hidden_channels=model.hidden_channels,
                    hidden_channel_widths=model.hidden_channel_widths,
                ),
            )
            for candidate in candidates
        )
        for candidates in definition_candidates
    )
    reference_vectors = {
        span: _relation_span_vector(
            hidden,
            span,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        )
        for proposals in proposals_by_operation
        for span, _score in proposals
    }
    relation_scores, base_relation_scores = _definition_relation_score_banks(
        model.definition_relation_head,
        reference_vectors,
        definition_vectors,
        definition_pointer_scores,
    )
    states: list[
        tuple[
            float,
            tuple[tuple[int, ...], ...],
            tuple[tuple[TokenSpan, ...], ...],
            tuple[tuple[int, ...], ...],
        ]
    ] = [(0.0, (), (), ())]
    strategy = model.training_receipt.get("argument_search_strategy")
    score_strategy = model.training_receipt.get(
        "argument_score_strategy", "independent_positive_v1"
    )
    prefix_feasible = strategy == "prefix_feasible_v1"
    global_constraint = strategy == "global_constraint_v1"
    chart_options = []
    chart_definition_options = []
    for node_index, node in enumerate(operation_nodes):
        argument_types, _result_type = operation_types[node_index]
        if len(argument_types) > len(model.argument_role_heads):
            return None
        operation_vector = _relation_span_vector(
            hidden,
            node.span,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        )
        partial: list[tuple[float, tuple[int, ...], tuple[TokenSpan, ...]]] = [(0.0, (), ())]
        options_by_position: list[list[tuple[float, int, TokenSpan]]] = []
        definitions_by_position = []
        for position, required_type in enumerate(argument_types):
            role_head = model.argument_role_heads[position]
            proposal_head = model.argument_proposal_heads[position]
            by_register: dict[int, list[tuple[float, TokenSpan]]] = {}
            for span, pointer_score in proposals_by_operation[node_index]:
                if span.end - span.start > model.max_argument_span_tokens_by_type[required_type]:
                    continue
                reference = reference_vectors[span]
                role_score = role_head.score(reference, operation_vector)
                proposal_score = proposal_head.score(reference, operation_vector)
                exact_inputs = tuple(
                    index for index, input_span in enumerate(input_spans) if span == input_span
                )
                candidate_registers = (
                    exact_inputs if exact_inputs else tuple(range(len(definitions)))
                )
                eligible_registers = tuple(
                    candidate_index
                    for candidate_index, register in enumerate(definition_registers)
                    if register in candidate_registers
                    and register_types[register] == required_type
                    and register != len(inputs) + node_index
                    and (model.allow_computed_dependencies or register < len(inputs))
                )
                if not eligible_registers:
                    continue
                raw_relation_scores = tuple(
                    relation_scores[span][register] for register in eligible_registers
                )
                base_raw_relation_scores = tuple(
                    base_relation_scores[span][register] for register in eligible_registers
                )
                relation_evidence = _mention_invariant_relation_evidence(
                    base_raw_relation_scores,
                    raw_relation_scores,
                    strategy=model.training_receipt.get(
                        "relation_score_strategy", "positive_label_margin_v1"
                    ),
                )
                for (
                    candidate_index,
                    relation_score,
                    candidate_relation_evidence,
                ) in zip(
                    eligible_registers,
                    raw_relation_scores,
                    relation_evidence,
                    strict=True,
                ):
                    register = definition_registers[candidate_index]
                    if (
                        model.training_receipt.get("forward_reference_policy") != "joint_graph_v1"
                        and register >= len(inputs)
                        and register - len(inputs) > node_index
                        and relation_score <= 0.0
                    ):
                        continue
                    score = (
                        _argument_semantic_evidence(
                            role_score, proposal_score,
                            role_scale=model.argument_role_scale,
                            proposal_scale=model.argument_proposal_scale,
                            strategy=score_strategy,
                        )
                        + model.definition_relation_scale * candidate_relation_evidence
                        + model.argument_pointer_scale * _log_sigmoid(pointer_score)
                    )
                    by_register.setdefault(candidate_index, []).append((score, span))
            if not by_register:
                return None
            ranked_options = sorted(
                (
                    (score, definition_registers[candidate_index], span, candidate_index)
                    for candidate_index, candidates in by_register.items()
                    for score, span in sorted(
                        candidates,
                        key=lambda item: (-item[0], item[1].start, item[1].end),
                    )[:_ARGUMENT_MENTIONS_PER_DEFINITION]
                ),
                key=lambda item: (-item[0], item[1], item[2].start, item[2].end, item[3]),
            )
            options = [(score, register, span) for score, register, span, _index in ranked_options]
            if joint_definitions:
                definitions_by_position.append(tuple(
                    definition_labels[index] for _score, _register, _span, index in ranked_options
                ))
            if prefix_feasible or global_constraint:
                options_by_position.append(options)
                continue
            partial = sorted(
                (
                    (
                        total + score,
                        (*registers, register),
                        (*spans, span),
                    )
                    for total, registers, spans in partial
                    for score, register, span in options
                    if not any(_overlap(span, previous) for previous in spans)
                    and (
                        not model.register_use_contract.distinct_arguments
                        or register not in registers
                    )
                ),
                key=lambda item: (
                    -item[0],
                    item[1],
                    tuple((span.start, span.end) for span in item[2]),
                ),
            )[:_ARGUMENT_BEAM]
            if not partial:
                return None
        if global_constraint:
            chart_options.append(options_by_position)
            chart_definition_options.append(definitions_by_position)
            continue
        candidates: list[
            tuple[
                float,
                tuple[tuple[int, ...], ...],
                tuple[tuple[TokenSpan, ...], ...],
                tuple[tuple[int, ...], ...],
            ]
        ] = []
        for total, arguments, spans, dependencies in states:
            continuations = (
                _prefix_feasible_arguments(
                    options_by_position,
                    arguments=arguments,
                    spans=spans,
                    dependencies=dependencies,
                    operation_nodes=operation_nodes,
                    n_inputs=len(inputs),
                    contract=model.register_use_contract,
                    beam=_ARGUMENT_BEAM,
                )
                if prefix_feasible else partial
            )
            for step_score, step_arguments, step_spans in continuations:
                if any(
                    _overlap(current, previous)
                    for current in step_spans
                    for previous_step in spans
                    for previous in previous_step
                ):
                    continue
                step_dependencies = tuple(
                    sorted(
                        register - len(inputs)
                        for register in set(step_arguments)
                        if register >= len(inputs)
                    )
                )
                candidate_dependencies = (*dependencies, step_dependencies)
                if (
                    _operation_order(
                        candidate_dependencies,
                        operation_nodes,
                        require_connected=False,
                    )
                    is None
                ):
                    continue
                use_counts: Counter[int] = Counter(
                    register for values in (*arguments, step_arguments) for register in values
                )
                if not model.register_use_contract.allows_partial(
                    use_counts,
                    n_inputs=len(inputs),
                ):
                    continue
                candidates.append(
                    (
                        total + step_score,
                        (*arguments, step_arguments),
                        (*spans, step_spans),
                        candidate_dependencies,
                    )
                )
        states = sorted(candidates, key=lambda item: (-item[0], item[1]))[:_ARGUMENT_BEAM]
        if not states:
            return None
    if global_constraint:
        from core.learning.semantic_argument_optimization import optimize_argument_chart

        optimized = optimize_argument_chart(
            chart_options, n_inputs=len(inputs), contract=model.register_use_contract,
            definition_options=chart_definition_options if joint_definitions else None,
        )
        states = [optimized] if optimized is not None else []
    valid: list[_TypedArgumentAssignment] = []
    for score, arguments, spans, dependencies in states:
        order = _operation_order(
            dependencies,
            operation_nodes,
            require_connected=True,
        )
        if order is None:
            continue
        referenced = {dependency for values in dependencies for dependency in values}
        sink = next(index for index in range(len(operation_nodes)) if index not in referenced)
        use_counts = Counter(register for values in arguments for register in values)
        if not model.register_use_contract.accepts_complete(
            use_counts,
            n_inputs=len(inputs),
            operation_count=len(operation_nodes),
            sink=sink,
        ):
            continue
        output_registers = {
            source_index: len(inputs) + target_index
            for target_index, source_index in enumerate(order)
        }
        ordered_arguments = tuple(
            tuple(
                register if register < len(inputs) else output_registers[register - len(inputs)]
                for register in arguments[source_index]
            )
            for source_index in order
        )
        valid.append(
            _TypedArgumentAssignment(
                operation_nodes=tuple(operation_nodes[index] for index in order),
                arguments=ordered_arguments,
                argument_spans=tuple(spans[index] for index in order),
                score=score,
                runner_up_score=None,
            )
        )
        if len(valid) == 2:
            break
    if not valid:
        return None
    winner = valid[0]
    return replace(
        winner,
        runner_up_score=valid[1].score if len(valid) > 1 else None,
    )


def _operation_order(
    dependencies: Sequence[Sequence[int]],
    operation_nodes: Sequence[_OperationNode],
    *,
    require_connected: bool,
) -> tuple[int, ...] | None:
    """Return a stable topological schedule for a partial or complete graph."""

    count = len(dependencies)
    if count > len(operation_nodes):
        return None
    normalized = tuple(tuple(sorted(set(values))) for values in dependencies)
    if any(
        dependency < 0 or dependency >= len(operation_nodes) or dependency == index
        for index, values in enumerate(normalized)
        for dependency in values
    ):
        return None
    completed: set[int] = set()
    order: list[int] = []
    while len(order) < count:
        ready = sorted(
            (
                index
                for index in range(count)
                if index not in completed
                and all(
                    dependency >= count or dependency in completed
                    for dependency in normalized[index]
                )
            ),
            key=lambda index: (
                operation_nodes[index].span.start,
                operation_nodes[index].span.end,
                index,
            ),
        )
        if not ready:
            return None
        completed.add(ready[0])
        order.append(ready[0])
    if not require_connected:
        return tuple(order)
    if count != len(operation_nodes):
        return None
    referenced = {dependency for values in normalized for dependency in values}
    sinks = tuple(index for index in range(count) if index not in referenced)
    if len(sinks) != 1:
        return None
    required = {sinks[0]}
    frontier = [sinks[0]]
    while frontier:
        current = frontier.pop()
        for dependency in normalized[current]:
            if dependency not in required:
                required.add(dependency)
                frontier.append(dependency)
    if required != set(range(count)):
        return None
    return tuple(order)


def _select_operation_length_penalty(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    pointer: LinearPointerHead,
    classifier: MultiViewClassifierHead,
    max_steps: int,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    cached: list[
        tuple[
            SemanticTransducerTrainingExample,
            tuple[tuple[float, tuple[_OperationNode, ...]], ...],
        ]
    ] = []
    average_scores: list[float] = []
    for item in validation:
        nodes = _operation_nodes(
            pointer=pointer,
            classifier=classifier,
            hidden=item.hidden_states,
            input_spans=item.ir.input_spans,
            max_span_tokens=max_span_tokens,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        by_count = tuple(
            _best_nonoverlapping_nodes(nodes, count) for count in range(1, max_steps + 1)
        )
        average_scores.extend(
            score / count
            for count, (score, _selected) in enumerate(by_count, start=1)
            if math.isfinite(score)
        )
        cached.append((item, by_count))
    if not average_scores:
        raise ValueError("compositional operation chart has no validation candidates")
    penalties = np.linspace(
        min(average_scores) - 5.0,
        max(average_scores) + 5.0,
        _OPERATION_PENALTY_POINTS,
    )
    rows: list[dict[str, Any]] = []
    for raw_penalty in penalties:
        penalty = float(raw_penalty)
        span_exact = 0
        operation_exact = 0
        graph_exact = 0
        for item, by_count in cached:
            selected = _best_penalized_operation_chart(by_count, penalty=penalty)
            expected_spans = tuple(
                instruction.operation_span for instruction in item.ir.instructions
            )
            expected_operations = tuple(instruction.op for instruction in item.ir.instructions)
            observed_spans = tuple(node.span for node in selected)
            observed_operations = tuple(node.operation for node in selected)
            span_exact += int(observed_spans == expected_spans)
            operation_exact += int(observed_operations == expected_operations)
            graph_exact += int(
                (observed_spans, observed_operations) == (expected_spans, expected_operations)
            )
        rows.append(
            {
                "length_penalty": penalty,
                "graph_exact": graph_exact,
                "span_exact": span_exact,
                "operation_exact": operation_exact,
                "validation_examples": len(validation),
            }
        )
    winner = max(
        rows,
        key=lambda row: (
            row["graph_exact"],
            row["span_exact"],
            row["operation_exact"],
            -abs(row["length_penalty"]),
        ),
    )
    return float(winner["length_penalty"]), rows
