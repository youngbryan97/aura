"""Inference-time candidate search and self-consistency scoring."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable


CandidateGenerator = Callable[[str, int], Awaitable[list[str]] | list[str]]
CandidateScorer = Callable[[str, str], Awaitable[float] | float]


@dataclass(frozen=True)
class SearchCandidate:
    text: str
    score: float
    verifier_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 5),
            "verifier_scores": {k: round(v, 5) for k, v in self.verifier_scores.items()},
        }


@dataclass(frozen=True)
class SearchResult:
    prompt: str
    winner: SearchCandidate
    candidates: tuple[SearchCandidate, ...]
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "winner": self.winner.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "generated_at": self.generated_at,
        }


class InferenceTimeSearchEngine:
    """Generate N reasoning traces, score with verifiers, return best."""

    def __init__(self, generator: CandidateGenerator, scorers: Iterable[tuple[str, CandidateScorer]] = ()) -> None:
        self.generator = generator
        self.scorers = tuple(scorers)

    async def search(self, prompt: str, *, width: int = 4) -> SearchResult:
        raw = self.generator(prompt, width)
        if asyncio.iscoroutine(raw):
            raw = await raw
        candidates: list[SearchCandidate] = []
        for text in list(raw)[:width]:
            verifier_scores: dict[str, float] = {}
            for name, scorer in self.scorers:
                value = scorer(prompt, text)
                if asyncio.iscoroutine(value):
                    value = await value
                verifier_scores[name] = float(value)
            score = self._aggregate(text, verifier_scores)
            candidates.append(SearchCandidate(text=str(text), score=score, verifier_scores=verifier_scores))
        if not candidates:
            empty = SearchCandidate("", 0.0, {})
            return SearchResult(prompt, empty, ())
        candidates.sort(key=lambda c: c.score, reverse=True)
        return SearchResult(prompt, candidates[0], tuple(candidates))

    @staticmethod
    def _aggregate(text: str, scores: dict[str, float]) -> float:
        if not scores:
            length_prior = 1.0 - math.exp(-min(len(text), 400) / 200.0)
            return 0.5 * length_prior
        mean = sum(scores.values()) / max(1, len(scores))
        length_penalty = 0.10 if len(text) < 20 else 0.0
        return max(0.0, min(1.0, mean - length_penalty))


def exact_match_scorer(expected: str) -> CandidateScorer:
    def _score(_prompt: str, candidate: str) -> float:
        return 1.0 if expected.strip().lower() in candidate.strip().lower() else 0.0

    return _score


__all__ = ["InferenceTimeSearchEngine", "SearchCandidate", "SearchResult", "exact_match_scorer"]
