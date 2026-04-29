"""NoveltyArchive — embedding-based dedup for failure-finding tests.

Backed by ``HashEmbedder``.  Every candidate prompt is embedded and
compared against the archive's stored embeddings; a prompt is novel
iff its closest neighbour is at most ``1 - novelty_threshold`` cosine
similarity.

The archive is in-memory by design — generated tests usually flow
straight into the curriculum loop or holdout vault.  Callers who
need persistence wrap the archive in their own snapshot rotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.verification.embedder import HashEmbedder


@dataclass
class NoveltyEntry:
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class NoveltyArchive:
    def __init__(
        self,
        *,
        embedder: Optional[HashEmbedder] = None,
        novelty_threshold: float = 0.35,
    ):
        if not 0.0 < novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be in (0, 1]")
        self.embedder = embedder or HashEmbedder()
        self.novelty_threshold = float(novelty_threshold)
        self.entries: List[NoveltyEntry] = []

    # ------------------------------------------------------------------
    def novelty(self, text: str) -> float:
        """Return 1 - max_cosine_to_archive (so >0 means novel)."""
        if not self.entries:
            return 1.0
        candidate = self.embedder.embed(text)
        best = max(self.embedder.cosine(candidate, e.embedding) for e in self.entries)
        return 1.0 - best

    def is_novel(self, text: str) -> bool:
        return self.novelty(text) >= self.novelty_threshold

    def add_if_novel(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not text:
            return False
        emb = self.embedder.embed(text)
        if self.entries:
            best = max(self.embedder.cosine(emb, e.embedding) for e in self.entries)
            if 1.0 - best < self.novelty_threshold:
                return False
        self.entries.append(NoveltyEntry(text=text, embedding=emb, metadata=metadata or {}))
        return True

    def __len__(self) -> int:
        return len(self.entries)

    def reset(self) -> None:
        self.entries.clear()
