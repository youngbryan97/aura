"""core/consciousness/mhaf/hrr.py
Holographic Reduced Representations (HRR) Engine.

HRR uses circular convolution (implemented via FFT) to bind symbolic
concepts into fixed-width distributed vectors, enabling:
  - Lossless (approximate) storage of structured knowledge
  - Role-filler binding: bind(agent, "John") ⊛ bind(action, "runs")
  - Retrieval by unbinding: x ⊛ ~y ≈ z (where x = y ⊛ z)
  - Superposition of multiple facts in a single vector

Reference: Plate, T.A. (1995). "Holographic reduced representations."
           IEEE Transactions on Neural Networks 6(3): 623-641.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("MHAF.HRR")


class HRREncoder:
    """Holographic Reduced Representation encoder/decoder.

    Encodes arbitrary string keys as random unit vectors and supports
    circular convolution binding/unbinding via FFT.

    All vectors are fixed-width (dim) real-valued float32 arrays,
    normalized to unit length after each operation.
    """

    def __init__(self, dim: int = 512, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._codebook: Dict[str, np.ndarray] = {}
        logger.info("HRREncoder online (dim=%d)", dim)

    def encode(self, key: str) -> np.ndarray:
        """Return the HRR vector for a string key (generated on first call)."""
        if key not in self._codebook:
            v = self._rng.normal(0.0, 1.0 / np.sqrt(self.dim), self.dim).astype(np.float32)
            self._codebook[key] = v / (np.linalg.norm(v) + 1e-8)
        return self._codebook[key].copy()

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution: a ⊛ b (binding operation).

        Result has the same dimensionality. Encodes the association a→b.
        """
        result = np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))).astype(np.float32)
        return self._normalize(result)

    def unbind(self, composite: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular correlation: composite ⊛ ~b ≈ a (unbinding operation).

        ~b is the approximate inverse of b (conjugate in frequency domain).
        """
        b_inv = np.real(np.fft.ifft(np.conj(np.fft.fft(b)))).astype(np.float32)
        return self.bind(composite, b_inv)

    def superpose(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Superposition: element-wise addition of multiple HRR vectors.

        The result is a 'bundle' that is similar to all constituents.
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        result = sum(v.astype(np.float32) for v in vectors) / len(vectors)
        return self._normalize(result)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two HRR vectors ∈ [-1, 1]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def query_codebook(self, query: np.ndarray, top_k: int = 5) -> list[Tuple[str, float]]:
        """Find the top-k most similar keys to a query vector."""
        results = []
        for key, vec in self._codebook.items():
            sim = self.similarity(query, vec)
            results.append((key, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def encode_fact(self, role: str, filler: str) -> np.ndarray:
        """Encode a role-filler pair: bind(encode(role), encode(filler))."""
        return self.bind(self.encode(role), self.encode(filler))

    def decode_filler(self, composite: np.ndarray, role: str, top_k: int = 3) -> list[Tuple[str, float]]:
        """Decode the filler for a known role from a composite vector."""
        role_vec = self.encode(role)
        candidate_filler = self.unbind(composite, role_vec)
        return self.query_codebook(candidate_filler, top_k=top_k)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return (v / norm).astype(np.float32)

    @property
    def codebook_size(self) -> int:
        return len(self._codebook)
