from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from core.brain.semantic_memory import SemanticMemory
from core.memory import embedding_model
from core.memory.vector_memory_engine import EmbeddingEngine
from core.runtime.model_lane_control import ModelLaneControlError, ModelLaneController
from core.runtime.receipts import ReceiptStore
from core.runtime.shutdown_coordinator import clear_shutdown_request


@pytest.fixture(autouse=True)
def _owned_receipt_stores(monkeypatch: pytest.MonkeyPatch):
    constructor = ReceiptStore
    stores: list[ReceiptStore] = []

    def _tracked_receipt_store(*args, **kwargs) -> ReceiptStore:
        store = constructor(*args, **kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(sys.modules[__name__], "ReceiptStore", _tracked_receipt_store)
    try:
        yield
    finally:
        for store in reversed(stores):
            store.close()


def _controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelLaneController:
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
    return ModelLaneController(
        state_path=tmp_path / "model_lanes.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_discovery=None,
    )


def _install_fake_embedding_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    controller: ModelLaneController,
    loading_preemptibility: list[bool],
    encode_entered: threading.Event,
    encode_release: threading.Event,
) -> type:
    class _Tokenizer:
        @staticmethod
        def num_special_tokens_to_add(*, pair: bool = False) -> int:
            del pair
            return 0

        @staticmethod
        def __call__(text, **kwargs):
            del kwargs
            words = str(text).split() or [""]
            offsets = []
            cursor = 0
            for word in words:
                start = str(text).find(word, cursor) if word else 0
                start = max(0, start)
                stop = start + len(word)
                offsets.append((start, stop))
                cursor = stop
            return {
                "input_ids": list(range(1, len(words) + 1)),
                "offset_mapping": offsets,
            }

    class _Encoder:
        block = False
        tokenizer = _Tokenizer()

        #: The real encoder is constructed with truncate_dim so the
        #: Matryoshka output stays at VECTOR_DIM. A double that does not
        #: accept it passes while the production call fails, which is the
        #: whole reason this signature is mirrored rather than **kwargs-ed.
        max_seq_length = embedding_model.MAX_INPUT_TOKENS

        def __init__(self, _model_name: str, truncate_dim: int | None = None) -> None:
            assert truncate_dim == embedding_model.VECTOR_DIM, (
                f"encoder must be pinned to {embedding_model.VECTOR_DIM} dims; "
                f"got truncate_dim={truncate_dim!r}"
            )
            loading_preemptibility.append(
                bool(controller.snapshot()["owners"][0]["preemptible"])
            )

        def encode(self, texts, **_kwargs):
            if self.block:
                encode_entered.set()
                assert encode_release.wait(2.0)
            if isinstance(texts, str):
                return np.ones(384, dtype=np.float32)
            return np.ones((len(texts), 384), dtype=np.float32)

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = _Encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    torch = ModuleType("torch")
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)

    class _Index:
        def add(self, _vectors) -> None:
            return None

        def search(self, _query, _top_k: int):
            return np.array([[0.1]], dtype=np.float32), np.array([[0]], dtype=np.int64)

    faiss = ModuleType("faiss")
    faiss.IndexFlatL2 = lambda _dimension: _Index()
    faiss.normalize_L2 = lambda _vectors: None
    faiss.write_index = lambda _index, _path: None
    faiss.read_index = lambda _path: _Index()
    monkeypatch.setitem(sys.modules, "faiss", faiss)
    return _Encoder


def test_embedding_constructor_rejects_an_unowned_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[str] = []
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = lambda *_args, **_kwargs: constructor_calls.append(
        "constructed"
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    with pytest.raises(
        ModelLaneControlError,
        match="model_load_requires_active_synchronous_in_process_model_lane",
    ):
        embedding_model.load_encoder(model_lane_lease=object())

    assert constructor_calls == [], "ownership must be checked before model construction"


def test_mps_causal_encoder_never_batches_independent_sentences() -> None:
    """Padding on the live Qwen3 MPS path must not move semantic vectors."""

    calls: list[tuple[str, ...]] = []

    class _MPSModel:
        device = "mps:0"

        @staticmethod
        def encode(texts, **_kwargs):
            group = tuple(texts)
            calls.append(group)
            return np.ones((len(group), embedding_model.VECTOR_DIM), dtype=np.float32)

    encoded = EmbeddingEngine._encode_model_payload(
        _MPSModel(),
        ["first sentence", "a much longer second sentence"],
        query_task=None,
    )

    assert encoded.shape == (2, embedding_model.VECTOR_DIM)
    assert calls == [("first sentence",), ("a much longer second sentence",)]


def test_cpu_encoder_retains_true_batching() -> None:
    calls: list[tuple[str, ...]] = []

    class _CPUModel:
        device = "cpu"

        @staticmethod
        def encode(texts, **_kwargs):
            group = tuple(texts)
            calls.append(group)
            return np.ones((len(group), embedding_model.VECTOR_DIM), dtype=np.float32)

    EmbeddingEngine._encode_model_payload(
        _CPUModel(),
        ["first sentence", "a much longer second sentence"],
        query_task=None,
    )

    assert calls == [("first sentence", "a much longer second sentence")]


@pytest.fixture(autouse=True)
def _shutdown_state() -> None:
    clear_shutdown_request()
    yield
    clear_shutdown_request()


@pytest.mark.asyncio
async def test_embedding_engine_refuses_eviction_during_encode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.runtime import model_lane_control

    controller = _controller(tmp_path, monkeypatch)
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    loading_preemptibility: list[bool] = []
    encode_entered = threading.Event()
    encode_release = threading.Event()
    encoder_type = _install_fake_embedding_dependencies(
        monkeypatch,
        controller=controller,
        loading_preemptibility=loading_preemptibility,
        encode_entered=encode_entered,
        encode_release=encode_release,
    )
    engine = EmbeddingEngine()

    assert engine.embed("initial").shape == (384,)
    monkeypatch.setattr(
        engine,
        "_init_tfidf_fallback",
        lambda: setattr(engine, "_tfidf_fallback", object()),
    )
    owner = controller.snapshot()["owners"][0]
    assert loading_preemptibility == [False]
    assert owner["preemptible"] is True

    encoder_type.block = True
    embed_task = asyncio.create_task(asyncio.to_thread(engine.embed, "held"))
    assert await asyncio.to_thread(encode_entered.wait, 2.0)
    assert await engine._evict_model_lane(object(), "foreground") is False

    encode_release.set()
    assert (await embed_task).shape == (384,)
    assert await engine._evict_model_lane(object(), "foreground") is True
    assert controller.snapshot()["owners"] == []
    engine.close()


@pytest.mark.asyncio
async def test_semantic_memory_refuses_eviction_during_vector_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.runtime import model_lane_control

    controller = _controller(tmp_path, monkeypatch)
    monkeypatch.setattr(model_lane_control, "get_model_lane_controller", lambda: controller)
    loading_preemptibility: list[bool] = []
    encode_entered = threading.Event()
    encode_release = threading.Event()
    encoder_type = _install_fake_embedding_dependencies(
        monkeypatch,
        controller=controller,
        loading_preemptibility=loading_preemptibility,
        encode_entered=encode_entered,
        encode_release=encode_release,
    )
    memory = SemanticMemory(memory_dir=str(tmp_path / "semantic"))
    await asyncio.to_thread(memory._background_init)
    memory.metadata = [{"id": "one", "text": "remember this", "tags": {}}]

    owner = controller.snapshot()["owners"][0]
    assert loading_preemptibility == [False]
    assert owner["preemptible"] is True

    encoder_type.block = True
    search_task = asyncio.create_task(
        asyncio.to_thread(memory.search_memories, "remember", 1)
    )
    assert await asyncio.to_thread(encode_entered.wait, 2.0)
    assert await memory._evict_vector_model(object(), "foreground") is False

    encode_release.set()
    assert [item["id"] for item in await search_task] == ["one"]
    assert await memory._evict_vector_model(object(), "foreground") is True
    assert controller.snapshot()["owners"] == []
    memory.on_stop()
