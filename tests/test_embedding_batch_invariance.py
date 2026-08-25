"""Real-checkpoint proof that embedding batch composition cannot change meaning."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.slow]


def test_qwen3_embedding_batch_matches_singleton_encoding_on_mps() -> None:
    """The production path repairs a measured Qwen3/MPS padding defect."""

    from core.memory.embedding_runtime import acquire_shared_embedding_engine

    sentences = [
        "show me a piece of your own program code",
        "which file of your implementation does that function live in",
        "what module or path in your repository contains that code",
        "where in your codebase is that written",
        "let me see the actual source you are built from",
    ]
    with acquire_shared_embedding_engine("batch-invariance-proof") as engine:
        batched = np.asarray(engine.embed_batch(sentences))
        singleton = np.asarray([engine.embed(sentence) for sentence in sentences])

    assert batched.shape == singleton.shape
    cosine = np.sum(batched * singleton, axis=1)
    assert float(np.min(cosine)) >= 0.999, cosine.tolist()
