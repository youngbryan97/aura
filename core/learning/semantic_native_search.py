"""Explore native choices globally without replacing the shared typed grammar."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from core.learning.semantic_native_grammar import (
    NativeGrammarDecision,
    NativeGrammarIncompleteError,
    NativeGrammarResult,
    decode_native_grammar,
)
from core.verify.invariants import invariant


@dataclass(frozen=True)
class NativeSearchCandidate:
    result: NativeGrammarResult
    log_probability: float


@dataclass(frozen=True)
class NativeGrammarSearchResult:
    candidates: tuple[NativeSearchCandidate, ...]
    expanded_nodes: int
    scored_decisions: int
    scored_alternatives: int
    disconnected_leaves: int
    frontier_nodes: int
    frontier_log_probability_bound: float | None
    halt_reason: str
    requested_top_k_proven: bool


class _UnscoredDecisionError(Exception):
    def __init__(self, choices):
        self.choices = choices


def search_native_grammar(
    input_types: tuple[str, ...],
    scorer: Callable[[tuple[NativeGrammarDecision, ...]], tuple[float, ...]],
    *, max_steps: int = 8, max_nodes: int = 256, completions: int = 4,
    register_encoding: str = "absolute_v1",
    score_mode: str = "normalized_choices",
) -> NativeGrammarSearchResult:
    """Return connected graphs ranked by a declared nonpositive decision score.

    Each admitted edge has a nonpositive log weight. A partial path's
    score therefore bounds every descendant. Best-first traversal establishes
    top-k ordering for this declared score when it finds k distinct graphs.
    It does not establish semantic correctness or whole-token likelihood.
    Node exhaustion retains candidates but never claims the requested top-k.
    """
    if (type(max_nodes) is not int or max_nodes < 1
            or type(completions) is not int or completions < 1):
        raise ValueError("native search needs positive node and completion bounds")
    if score_mode not in {"normalized_choices", "native_nonpositive"}:
        raise ValueError("native search score mode is not declared")
    frontier = [(0., 0, ())]
    scores_by_prefix = {}
    serial = expanded = disconnected = alternatives = 0
    candidates, seen = [], set()
    while frontier and expanded < max_nodes and len(candidates) < completions:
        negative_bound, _, path = heapq.heappop(frontier)
        expanded += 1
        visited = []

        def replay(choices, *, visited=visited, path=path):
            position = len(visited)
            visited.append(choices)
            if position == len(path):
                raise _UnscoredDecisionError(choices)
            selected = path[position]
            if not 0 <= selected < len(choices):
                raise ValueError("native search replay changed the typed choice inventory")
            return tuple(0. if index == selected else -1e300 for index in range(len(choices)))

        try:
            result = decode_native_grammar(input_types, replay, max_steps=max_steps,
                                            register_encoding=register_encoding)
        except _UnscoredDecisionError as pending:
            choices = pending.choices
            scores = scores_by_prefix.get(choices)
            if scores is None:
                scores = scorer(choices)
                if (not isinstance(scores, tuple) or len(scores) != len(choices)
                        or any(isinstance(value, bool) or not isinstance(value, (float, int))
                               or not math.isfinite(value) for value in scores)):
                    raise ValueError("native search requires every finite decision score") from pending
                scores_by_prefix[choices] = scores
                alternatives += len(choices)
            if score_mode == "native_nonpositive" and any(score > 0. for score in scores):
                raise ValueError("native probability edges must be nonpositive") from pending
            high = max(scores)
            normalizer = math.log(math.fsum(math.exp(value - high) for value in scores))
            for index, score in enumerate(scores):
                log_probability = score if score_mode == "native_nonpositive" else (score - high) - normalizer
                if not math.isfinite(log_probability) or not math.isfinite(negative_bound - log_probability):
                    raise ValueError("native search probability exceeds finite precision") from pending
                serial += 1
                heapq.heappush(frontier, (negative_bound - log_probability,
                                         serial, (*path, index)))
        except NativeGrammarIncompleteError:
            disconnected += 1
        else:
            program = result.program
            key = (program.n_inputs, tuple((step.op, step.args) for step in program.instructions))
            if key not in seen:
                seen.add(key)
                trace = tuple({**row, "scores": scores_by_prefix[choices]}
                              for row, choices in zip(result.trace, visited, strict=True))
                candidates.append(NativeSearchCandidate(replace(result, trace=trace), -negative_bound))
    proven = len(candidates) == completions
    reason = "requested_completions" if proven else "node_bound" if frontier else "frontier_exhausted"
    return NativeGrammarSearchResult(
        tuple(candidates), expanded, len(scores_by_prefix), alternatives, disconnected,
        len(frontier), -frontier[0][0] if frontier else None, reason, proven,
    )


@invariant("learning.native_search_keeps_limits_distinct_from_top_k_proof", scope="learning",
           owner="core/learning/semantic_native_search.py", observational=False)
def _native_search_bound_truth() -> tuple:
    def score(choices):
        return (0.,) * len(choices)
    limited = search_native_grammar(("integer",), score, max_steps=1, max_nodes=1, completions=2)
    assert not limited.requested_top_k_proven and limited.frontier_nodes > 0
    complete = search_native_grammar(("integer",), score, max_steps=1, max_nodes=128, completions=2)
    assert complete.requested_top_k_proven and len(complete.candidates) == 2
    assert complete.candidates[-1].log_probability >= complete.frontier_log_probability_bound
    return ()
