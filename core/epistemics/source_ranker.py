"""core/epistemics/source_ranker.py — Source Reliability Ranker."""
from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger("Aura.SourceRanker")

# Domain credibility defaults
DEFAULT_RELIABILITY: dict[str, float] = {
    "arxiv.org": 0.90,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "nature.com": 0.98,
    "github.com": 0.85,
    "wikipedia.org": 0.80,
    "news.ycombinator.com": 0.70,
    "reddit.com": 0.35,
    "unknown": 0.50,
}


class SourceRanker:
    """Tracks and calibrates reliability scores for academic, web, and user sources."""

    def __init__(self) -> None:
        self.dynamic_scores: dict[str, float] = dict(DEFAULT_RELIABILITY)
        self.evidence_count: dict[str, int] = {}

    @staticmethod
    def _normalized_source(source: str) -> str:
        raw = str(source or "").strip().lower()
        if not raw:
            return "unknown"
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or "").removeprefix("www.").rstrip(".")
        return host or raw[:255]

    def _score_key(self, source: str, *, create: bool = False) -> str:
        normalized = self._normalized_source(source)
        if normalized in self.dynamic_scores:
            return normalized
        domain_matches = [
            key
            for key in self.dynamic_scores
            if key != "unknown"
            and (normalized == key or normalized.endswith(f".{key}"))
        ]
        if domain_matches:
            return max(domain_matches, key=len)
        if create and normalized != "unknown":
            self.dynamic_scores[normalized] = self.dynamic_scores["unknown"]
            return normalized
        return "unknown"

    def get_reliability(self, source: str) -> float:
        """Looks up reliability, normalizing source name first."""
        return self.dynamic_scores[self._score_key(source)]

    def record_outcome(self, source: str, verified_true: bool) -> None:
        """Dynamically adjusts the source's score based on verification outcomes."""
        if not isinstance(verified_true, bool):
            raise TypeError("verified_true must be boolean ground truth")
        key = self._score_key(source, create=True)
        current = self.dynamic_scores[key]
        count = self.evidence_count.get(key, 0) + 1
        self.evidence_count[key] = count

        # Learning rate decays as we get more evidence
        lr = max(0.01, 0.1 / (count ** 0.5))
        target = 1.0 if verified_true else 0.0
        delta = lr * (target - current)

        # Calculate new score clamped between 0.05 and 0.99
        new_score = max(0.05, min(0.99, current + delta))
        self.dynamic_scores[key] = new_score
        logger.info(
            "Recalibrated source %s reliability: %.2f -> %.2f (count=%d)",
            key,
            current,
            new_score,
            count,
        )


_ranker_instance: SourceRanker | None = None


def get_source_ranker() -> SourceRanker:
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = SourceRanker()
    return _ranker_instance
