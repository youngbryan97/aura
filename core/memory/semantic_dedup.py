"""core/memory/semantic_dedup.py
================================
Semantic deduplication for memory writes.

Prevents the memory store from accumulating near-duplicate entries
by comparing incoming memories against a recent write cache using
lightweight text similarity (trigram Jaccard + normalized edit distance).

This is NOT a vector similarity check (too expensive for every write).
Instead it uses fast string-based heuristics that catch >90% of
duplicates without requiring embedding computation.

Usage:
    from core.memory.semantic_dedup import get_dedup_gate
    gate = get_dedup_gate()
    if gate.should_store(text, tags=tags):
        # proceed with storage
    else:
        # skip — near-duplicate exists
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from core.memory.retention_policy import working_history_retention_policy

logger = logging.getLogger("Aura.Memory.SemanticDedup")


@dataclass
class RecentWrite:
    """Record of a recently written memory entry."""
    text_hash: str
    trigrams: frozenset
    normalized_text: str
    timestamp: float = field(default_factory=time.time)
    tags: tuple = ()


class SemanticDedupGate:
    """Lightweight semantic deduplication gate for memory writes.

    Maintains a sliding window of recent writes and rejects new entries
    that are too similar to existing ones.
    """

    # Similarity thresholds
    EXACT_MATCH_REJECT = True
    TRIGRAM_SIMILARITY_THRESHOLD = 0.75  # Jaccard > 0.75 = likely duplicate
    NORMALIZED_EDIT_THRESHOLD = 0.85     # Normalized similarity > 0.85 = duplicate

    # Window size
    MAX_RECENT_WRITES = working_history_retention_policy("AURA_SEMANTIC_DEDUP_RECENT_WRITES_MAX").max_items
    WRITE_WINDOW_S = 3600.0  # Only compare against writes in the last hour

    # Durable exact-duplicate memory: bounded LRU of content hashes that is NOT
    # time-pruned, so the same canonical fact ("Bryan prefers Python") written hours
    # or sessions apart is still caught. This is the actual pollution fix.
    DURABLE_EXACT_MAX = 20000

    def __init__(self) -> None:
        self._recent: deque[RecentWrite] = deque(maxlen=self.MAX_RECENT_WRITES)
        self._exact_hashes: set[str] = set()
        self._durable_exact: OrderedDict[str, None] = OrderedDict()
        self._total_checked: int = 0
        self._total_rejected: int = 0
        self._total_passed: int = 0

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def _trigrams(text: str) -> frozenset:
        """Extract character trigrams from text."""
        if len(text) < 3:
            return frozenset([text])
        return frozenset(text[i:i+3] for i in range(len(text) - 2))

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        """Compute Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _normalized_edit_similarity(a: str, b: str) -> float:
        """Fast normalized edit distance similarity (0-1, 1=identical).

        Uses a simplified approach: compare word sets and overlap ratio.
        This is O(n) instead of O(n²) for full Levenshtein.
        """
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        total = max(len(words_a), len(words_b))
        return overlap / total if total > 0 else 0.0

    def _prune_stale(self) -> None:
        """Remove entries older than the write window."""
        cutoff = time.time() - self.WRITE_WINDOW_S
        while self._recent and self._recent[0].timestamp < cutoff:
            old = self._recent.popleft()
            self._exact_hashes.discard(old.text_hash)

    def should_store(
        self,
        text: str,
        *,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> bool:
        """Check whether this memory should be stored.

        Returns True if the memory is sufficiently novel to store.
        Returns False if it's a near-duplicate of a recent write.

        High-importance memories (>= 0.8) bypass dedup to ensure
        critical information is never lost.
        """
        self._total_checked += 1

        # Empty/trivial text always rejected
        clean = text.strip()
        if len(clean) < 10:
            self._total_rejected += 1
            return False

        normalized = self._normalize(clean)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()

        # Exact-duplicate rejection applies to EVERY write, including high-importance
        # ones. Re-storing the identical canonical fact is never useful — it is the
        # exact memory-pollution vector ("Bryan prefers Python" x N). Checked against
        # both the recent window and the durable (non-time-pruned) LRU.
        if self.EXACT_MATCH_REJECT and (
            text_hash in self._exact_hashes or text_hash in self._durable_exact
        ):
            self._durable_exact.move_to_end(text_hash) if text_hash in self._durable_exact else None
            self._total_rejected += 1
            logger.debug("SemanticDedup: Exact duplicate rejected (hash=%s)", text_hash[:8])
            return False

        # High importance bypasses only the FUZZY near-duplicate check below, so a
        # nuanced-but-similar important fact is never lost — but it is still recorded
        # for future exact-duplicate detection.
        if importance >= 0.8:
            self._record_write(
                text, tags, text_hash=text_hash, normalized=normalized,
                trigrams=self._trigrams(normalized),
            )
            self._total_passed += 1
            return True

        # Prune stale entries
        self._prune_stale()

        # Trigram similarity check
        incoming_trigrams = self._trigrams(normalized)

        for recent in reversed(list(self._recent)):
            # Trigram Jaccard
            sim = self._jaccard(incoming_trigrams, recent.trigrams)
            if sim >= self.TRIGRAM_SIMILARITY_THRESHOLD:
                # Double-check with word overlap
                word_sim = self._normalized_edit_similarity(
                    normalized, recent.normalized_text
                )
                if word_sim >= self.NORMALIZED_EDIT_THRESHOLD:
                    self._total_rejected += 1
                    logger.debug(
                        "SemanticDedup: Near-duplicate rejected "
                        "(trigram=%.2f, word=%.2f, text=%s...)",
                        sim, word_sim, clean[:40],
                    )
                    return False

        # Novel — record and allow
        self._record_write(text, tags, text_hash=text_hash,
                          normalized=normalized, trigrams=incoming_trigrams)
        self._total_passed += 1
        return True

    def _record_write(
        self,
        text: str,
        tags: list[str] | None = None,
        *,
        text_hash: str | None = None,
        normalized: str | None = None,
        trigrams: frozenset | None = None,
    ) -> None:
        """Record a successful write for future dedup checks."""
        if normalized is None:
            normalized = self._normalize(text)
        if text_hash is None:
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
        if trigrams is None:
            trigrams = self._trigrams(normalized)

        entry = RecentWrite(
            text_hash=text_hash,
            trigrams=trigrams,
            normalized_text=normalized,
            tags=tuple(tags or ()),
        )
        self._recent.append(entry)
        self._exact_hashes.add(text_hash)
        # Durable LRU: remember this exact content beyond the 1-hour window.
        self._durable_exact[text_hash] = None
        self._durable_exact.move_to_end(text_hash)
        while len(self._durable_exact) > self.DURABLE_EXACT_MAX:
            self._durable_exact.popitem(last=False)

    def get_status(self) -> dict[str, Any]:
        return {
            "total_checked": self._total_checked,
            "total_rejected": self._total_rejected,
            "total_passed": self._total_passed,
            "dedup_rate": round(
                self._total_rejected / max(self._total_checked, 1), 3
            ),
            "recent_window_size": len(self._recent),
        }


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: SemanticDedupGate | None = None


def get_dedup_gate() -> SemanticDedupGate:
    """Get the singleton SemanticDedupGate instance."""
    global _instance
    if _instance is None:
        _instance = SemanticDedupGate()
    return _instance
