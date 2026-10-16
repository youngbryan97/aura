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


def test_qwen3_equal_length_batches_match_singleton_encoding_on_mps() -> None:
    """Sentences of one length, which pad nothing, still mean in a batch what they mean alone.

    On MPS they did not when batched together (cosine 0.03 to 0.13, 2026-10-16), so the production path must
    keep encoding them one at a time there.
    """

    from core.memory.embedding_runtime import acquire_shared_embedding_engine

    # Short problems of one template, as a kind's problems are, so lengths collide.
    sentences = [f"What is {a} divided by {b}?" for a, b in
                 ((84, 7), (96, 12), (1001, 13), (45, 9), (360, 24), (81, 3), (144, 16), (999, 37))]
    sentences += [f"Let y = {a}*x - {b}. Solve y = 0 for x." for a, b in ((3, 12), (5, 40), (7, 49), (2, 8))]
    with acquire_shared_embedding_engine("equal-length-batch-proof") as engine:
        batched = np.asarray(engine.embed_batch(sentences))
        singleton = np.asarray([engine.embed(sentence) for sentence in sentences])

    cosine = np.sum(batched * singleton, axis=1)
    assert float(np.min(cosine)) >= 0.999, cosine.tolist()
