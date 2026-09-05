"""HashEmbedder — deterministic, dependency-free text embedder.

Signed character n-gram hashing into a fixed-dim vector + L2 norm.
No model load, no network, no hidden state.  Reproducible across
processes by construction (BLAKE2b is the seed).

This is intentionally weak — for production-quality semantic
similarity replace with a real sentence embedder.  But for
self-consistency thresholds, paraphrase invariance, and novelty
archives it's plenty: the goal is "are these texts in the same
neighbourhood," not "what's the optimal embedding."
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Sequence


class HashEmbedder:
    def __init__(self, dim: int = 512, ngram: int = 3):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        self.dim = int(dim)
        self.ngram = int(ngram)

    def embed(self, text: str) -> List[float]:
        normalized = " ".join(str(text).lower().split())
        vec = [0.0] * self.dim
        if not normalized:
            return vec
        if len(normalized) <= self.ngram:
            grams = [normalized]
        else:
            grams = [
                normalized[i : i + self.ngram]
                for i in range(len(normalized) - self.ngram + 1)
            ]
        for g in grams:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            idx = val % self.dim
            sign = 1.0 if (val >> 11) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            raise ValueError("vector length mismatch")
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        denom = (na or 1.0) * (nb or 1.0)
        return dot / denom
