from __future__ import annotations

from typing import Any


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in (text or "").split() if t.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Basic text chunking for RAG ingestion."""
    if not text:
        return []
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def compute_term_freq(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    total = float(len(tokens))
    return {k: v / total for k, v in counts.items()}


def compute_cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = float(sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys))
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def retrieve_memories(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.01,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    query = (query or "").lower()
    scored: list[dict[str, Any]] = []
    for m in memories or []:
        text = str(m.get("text", "")).lower()
        score = 1.0 if query and query in text else 0.0
        if score >= threshold:
            item = dict(m)
            item["score"] = score
            scored.append(item)
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return scored[:top_k]


def retrieve_memories_v2(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return retrieve_memories(query, memories, top_k=top_k, **kwargs)
