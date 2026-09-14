"""Source-bounded definition proposals shared by semantic fitting and decoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import (
    LinearPointerSequenceScores,
    SemanticTransducerTrainingExample,
)

_LEGACY_DEFINITION_CANDIDATE_STRATEGY: Final = "anchored_envelope_v1"
_LOCAL_DEFINITION_CANDIDATE_STRATEGY: Final = "bounded_local_alias_v1"
_STABLE_REGISTER_TABLE_STRATEGY: Final = "stable_register_table_v1"
_LOCAL_DEFINITION_CANDIDATES: Final = 16


def _register_definition_spans(item: SemanticTransducerTrainingExample) -> tuple[TokenSpan, ...]:
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
    bidirectional_inputs: bool = False,
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
        if bidirectional_inputs and index < input_count:
            # A literal can precede or follow its name. Its source neighbors,
            # rather than its register number, bound this attachment search.
            left = max((other.end for other in anchors if other.end <= anchor.start), default=0)
            right = min(
                (other.start for other in anchors if other.start >= anchor.end), default=token_count
            )
            lower = max(left, anchor.end - max_span_tokens)
            upper = min(right, anchor.start + max_span_tokens)
            candidates = (
                TokenSpan(start, end)
                for start in range(lower, upper)
                for end in range(start + 1, min(upper, start + max_span_tokens) + 1)
            )
            ranked = sorted(
                candidates,
                key=lambda span: (
                    -pointer_scores.score_span(span),
                    span.end - span.start,
                    span.start,
                    span.end,
                ),
            )
            result.append(tuple(dict.fromkeys((anchor, *ranked[:_LOCAL_DEFINITION_CANDIDATES]))))
            continue
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
                else anchors[index - 1].end
                if index
                else 0
            )
            boundary_start = max(clause_start, anchor.start - max_span_tokens)
            local_bounds = (boundary_start, anchor.start)
            bounded_envelopes = (
                tuple(span for span in envelopes if span.start >= clause_start)
                if source_ordered_boundaries
                else envelopes
            )
        else:
            next_operation = (
                min(
                    (other.start for other in anchors[input_count:] if other.start >= anchor.end),
                    default=token_count,
                )
                if source_ordered_boundaries
                else anchors[index + 1].start
                if index + 1 < len(anchors)
                else token_count
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
