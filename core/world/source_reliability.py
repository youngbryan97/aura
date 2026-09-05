"""core/world/source_reliability.py — Ingestion Source Reliability Monitor.

Determines reliability ratings and credentials for external information sources.
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger("Aura.SourceReliability")


class SourceReliabilityMonitor:
    """Evaluates and persists credentials/trust ratings for data sources."""

    def __init__(self) -> None:
        # Default baseline credentials
        self.ratings: Dict[str, float] = {
            "arxiv.org": 0.92,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "github.com": 0.90,
            "semanticscholar.org": 0.88,
            "reuters.com": 0.85,
            "apnews.com": 0.85,
            "weather.gov": 0.98,
            "sec.gov": 0.95,
        }

    def get_score(self, source_domain: str) -> float:
        """Returns the reliability rating (0.0 to 1.0) for a given domain."""
        clean_domain = source_domain.strip().lower()
        # Handle subdomains or direct matches
        for domain, score in self.ratings.items():
            if clean_domain == domain or clean_domain.endswith("." + domain):
                return score
        return 0.50  # Default neutral trust score for unverified sources

    def record_feedback(self, source_domain: str, confirmed: bool) -> None:
        """Dynamically adjusts trust based on post-action verification feedback."""
        current = self.get_score(source_domain)
        adjustment = 0.05 if confirmed else -0.10
        new_score = max(0.10, min(1.0, current + adjustment))
        self.ratings[source_domain] = new_score
        logger.info("Updated source trust: %s -> %.2f", source_domain, new_score)


# Singleton
_monitor_instance: SourceReliabilityMonitor | None = None


def get_source_reliability_monitor() -> SourceReliabilityMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SourceReliabilityMonitor()
    return _monitor_instance
