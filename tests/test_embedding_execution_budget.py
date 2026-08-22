from __future__ import annotations

import re

import numpy as np
import pytest

from core.memory import embedding_model


class _WordTokenizer:
    """Small offset-aware tokenizer for execution-contract tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        **_kwargs,
    ):
        assert truncation is False
        matches = list(re.finditer(r"\S+", text))
        input_ids = list(range(len(matches)))
        if add_special_tokens:
            input_ids = [-1, *input_ids, -2]
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result

    @staticmethod
    def num_special_tokens_to_add(*, pair: bool = False) -> int:
        assert pair is False
        return 2


class _DecodeOnlyTokenizer(_WordTokenizer):
    def __call__(self, text: str, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError("slow tokenizer has no offsets")
        return super().__call__(text, **kwargs)

    @staticmethod
    def decode(token_ids, **_kwargs) -> str:
        return " ".join(f"token-{token_id}" for token_id in token_ids)


def test_operational_window_is_distinct_from_model_capability() -> None:
    assert 0 < embedding_model.OPERATIONAL_INPUT_TOKENS < embedding_model.MAX_INPUT_TOKENS
    assert (
        embedding_model.OPERATIONAL_PADDED_TOKEN_BUDGET >= embedding_model.OPERATIONAL_INPUT_TOKENS
    )
    assert embedding_model.OPERATIONAL_TRANSIENT_GB > 0.0


def test_operational_views_cover_every_source_token_without_truncation() -> None:
    tokenizer = _WordTokenizer()
    text = " ".join(f"word-{index}" for index in range(2_500))

    views = embedding_model.operational_views(text, tokenizer)

    assert len(views) > 1
    assert all(view.token_count <= embedding_model.OPERATIONAL_INPUT_TOKENS for view in views)
    covered: set[str] = set()
    for view in views:
        covered.update(view.text.split())
    assert covered == set(text.split())
    assert views[0].source_start == 0
    assert views[-1].source_end == len(text)


def test_operational_overlap_preserves_relations_at_a_view_boundary() -> None:
    tokenizer = _WordTokenizer()
    content_budget = embedding_model.OPERATIONAL_INPUT_TOKENS - 2
    words = [f"word-{index}" for index in range(content_budget + 100)]
    words[content_budget - 3] = "left-relation"
    words[content_budget + 3] = "right-relation"

    views = embedding_model.operational_views(" ".join(words), tokenizer)

    assert any("left-relation" in view.text and "right-relation" in view.text for view in views)


def test_operational_views_support_tokenizers_without_offsets() -> None:
    tokenizer = _DecodeOnlyTokenizer()
    text = " ".join(f"word-{index}" for index in range(2_500))

    views = embedding_model.operational_views(text, tokenizer)

    assert len(views) > 1
    assert all(view.source_start == -1 and view.source_end == -1 for view in views)
    assert all(view.token_count <= embedding_model.OPERATIONAL_INPUT_TOKENS for view in views)


def test_microbatches_bound_padded_transformer_work() -> None:
    tokenizer = _WordTokenizer()
    texts = [
        " ".join(f"d{doc}-w{i}" for i in range(size))
        for doc, size in enumerate((20, 300, 900, 1_800))
    ]
    views = [view for text in texts for view in embedding_model.operational_views(text, tokenizer)]

    batches = embedding_model.embedding_microbatches(views)

    assert batches
    assert [view for batch in batches for view in batch] == views
    for batch in batches:
        padded_tokens = len(batch) * max(view.token_count for view in batch)
        assert padded_tokens <= embedding_model.OPERATIONAL_PADDED_TOKEN_BUDGET
        assert len(batch) <= embedding_model.OPERATIONAL_MAX_BATCH_ITEMS


class _FakeEmbeddingModel:
    device = "mps:0"
    tokenizer = _WordTokenizer()

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        cohort = tuple(texts)
        self.calls.append(cohort)
        return np.asarray(
            [[float(len(text.split())), 1.0, 0.0, 0.0] for text in cohort],
            dtype=np.float32,
        )


def _engine_with_model(model: _FakeEmbeddingModel):
    from core.memory.vector_memory_engine import EmbeddingEngine

    engine = EmbeddingEngine()
    engine._model = model
    engine._initialized = True
    return engine


def test_embed_batch_segments_long_inputs_and_preserves_cardinality(monkeypatch) -> None:
    from core.memory.vector_memory_engine import EmbeddingEngine

    model = _FakeEmbeddingModel()
    engine = _engine_with_model(model)
    monkeypatch.setattr(EmbeddingEngine, "_release_mps_cache", staticmethod(lambda _model: None))
    texts = ["short evidence", " ".join(f"long-{i}" for i in range(3_000))]

    vectors = engine.embed_batch(texts)

    assert vectors.shape == (2, 4)
    assert len(model.calls) > 1
    for call in model.calls:
        encoded = [
            len(model.tokenizer(text, add_special_tokens=True)["input_ids"]) for text in call
        ]
        assert max(encoded) <= embedding_model.OPERATIONAL_INPUT_TOKENS
        assert len(call) * max(encoded) <= embedding_model.OPERATIONAL_PADDED_TOKEN_BUDGET


def test_short_input_keeps_the_direct_encoder_result(monkeypatch) -> None:
    from core.memory.vector_memory_engine import EmbeddingEngine

    model = _FakeEmbeddingModel()
    engine = _engine_with_model(model)
    monkeypatch.setattr(EmbeddingEngine, "_release_mps_cache", staticmethod(lambda _model: None))

    vector = engine.embed("short evidence")

    expected = np.asarray([2.0, 1.0, 0.0, 0.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert vector == pytest.approx(expected)
    assert model.calls == [("short evidence",)]


def test_background_embedding_yields_before_starting_a_model_batch(monkeypatch) -> None:
    from core.memory.vector_memory_engine import (
        EmbeddingEngine,
        EmbeddingWorkDeferredError,
    )

    model = _FakeEmbeddingModel()
    engine = _engine_with_model(model)
    monkeypatch.setattr(EmbeddingEngine, "_release_mps_cache", staticmethod(lambda _model: None))
    monkeypatch.setattr(
        "core.runtime.backpressure.primary_inference_active",
        lambda: True,
    )

    with pytest.raises(EmbeddingWorkDeferredError, match="foreground_inference_active"):
        engine.embed_document_views(["background evidence"], background=True)
    assert model.calls == []


def test_multivector_semantic_score_recovers_tail_evidence() -> None:
    from core.memory import rag

    query = np.asarray([1.0, 0.0], dtype=np.float32)
    document = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert rag._document_dense_score(query, document) == pytest.approx(1.0)


def test_background_warm_stops_between_cohorts_and_keeps_completed_work(
    monkeypatch,
) -> None:
    from core.memory import rag

    rag.reset_semantic_state_for_test()

    class _Engine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def embed_document_views(self, texts, *, background=False):
            assert background is True
            cohort = tuple(texts)
            self.calls.append(cohort)
            return [
                np.asarray([[float(index + 1), 1.0]], dtype=np.float32)
                for index, _text in enumerate(cohort)
            ]

    class _ImmediateThread:
        def __init__(self, *, target, **_kwargs) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    engine = _Engine()
    checks = iter((False, False, True))
    monkeypatch.setattr(rag, "_get_embed_engine", lambda: engine)
    monkeypatch.setattr(
        rag,
        "_background_embedding_should_defer",
        lambda: next(checks),
    )
    monkeypatch.setattr(rag.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(rag, "_SEMANTIC_WARM_COHORT", 2)
    texts = [f"memory-{index}" for index in range(5)]

    rag._warm_cache_in_background(texts)

    assert engine.calls == [("memory-0", "memory-1")]
    assert rag._cached_vector("memory-0") is not None
    assert rag._cached_vector("memory-1") is not None
    assert rag._cached_vector("memory-2") is None
    assert rag._WARM_INFLIGHT is False
    rag.reset_semantic_state_for_test()
