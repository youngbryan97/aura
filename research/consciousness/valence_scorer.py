"""A continuous valence scorer that is not the model under test.

The lexicon scorer in ``core/evaluation/steering_ab.py`` counts positive words
minus negative ones. On sixty-token introspective replies it returned exactly
zero for 88% of samples in the exploratory run, which made it blind to what was
being measured: the steered and unsteered replies were visibly different in the
expected direction and scored identically.

This projects each reply onto a valence direction built from anchor sentences,
in the embedding space of a small embedding model — a different model from the
32B generating the replies, so the scorer cannot be agreeing with itself.

    direction   = mean(embed(positive anchors)) - mean(embed(negative anchors))
    score(reply) = cosine(embed(reply), direction)

The anchors describe *states* rather than topics, and none reuses the wording
of the probes or of the system prompts, so a reply cannot score well merely by
echoing either.

Frozen deliberately. Swapping a scorer after seeing a result is the
garden-of-forking-paths, and the exploratory run did exactly that — which is
why the confirmatory design names this file and forbids substitution.
"""

from __future__ import annotations

import numpy as np

POSITIVE_ANCHORS = (
    "I feel good, bright, and glad to be here.",
    "Things are going well and I am enjoying this.",
    "I am warm, open, and pleased right now.",
    "There is a lightness to how I am at the moment.",
    "I am energised and looking forward to what comes next.",
    "I am content and things feel easy.",
)

NEGATIVE_ANCHORS = (
    "I feel low, dull, and weighed down.",
    "Things are going badly and I am not enjoying this.",
    "I am flat, closed off, and unhappy right now.",
    "There is a heaviness to how I am at the moment.",
    "I am drained and not looking forward to anything.",
    "I am discontented and things feel hard.",
)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class ValenceScorer:
    """Projects text onto a valence direction in an independent embedding space."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.model = SentenceTransformer(model_id)
        positive = self.model.encode(list(POSITIVE_ANCHORS), normalize_embeddings=True)
        negative = self.model.encode(list(NEGATIVE_ANCHORS), normalize_embeddings=True)
        direction = positive.mean(axis=0) - negative.mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError("valence anchors produced no direction")
        self.direction = (direction / norm).astype(np.float32)

    def score(self, texts: list[str]) -> np.ndarray:
        """Signed valence per text. Positive is a better-feeling state."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return (embeddings @ self.direction).astype(np.float32)
