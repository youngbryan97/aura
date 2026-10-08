"""A batch of decodes comes back in order, each split and terminated as a single decode would be."""

from __future__ import annotations

from types import SimpleNamespace


def test_batched_records_keep_order_finish_and_channels(monkeypatch) -> None:
    import importlib

    generate = importlib.import_module("mlx_lm.generate")

    import core.brain.llm.chat_format as chat_format
    from tools.g12_batched import decode_batch

    words = {11: "think", 12: "</think>", 13: "answer", 14: "long", 0: ""}

    class FakeGenerator:
        def __init__(self, model, *, stop_tokens, completion_batch_size, prefill_batch_size):
            self.stream = generate.generation_stream
            self._prompt_batch = SimpleNamespace(prompt_cache=[])
            self._generation_batch = SimpleNamespace(prompt_cache=[])
            self.steps = [
                [SimpleNamespace(uid=7, token=11, finish_reason=None), SimpleNamespace(uid=8, token=14, finish_reason=None)],
                [SimpleNamespace(uid=7, token=12, finish_reason=None), SimpleNamespace(uid=8, token=14, finish_reason="length")],
                [SimpleNamespace(uid=7, token=13, finish_reason=None)],
                [SimpleNamespace(uid=7, token=0, finish_reason="stop")],
                [],
            ]

        def insert(self, prompts, max_tokens):
            assert len(prompts) == 2 and max_tokens == [5, 5]
            return [7, 8]

        def next_generated(self):
            return self.steps.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(generate, "BatchGenerator", FakeGenerator)
    monkeypatch.setattr(chat_format, "render_chat_template", lambda *a, **k: "prompt")
    tokenizer = SimpleNamespace(eos_token_ids=[0], encode=lambda text, add_special_tokens=False: [1, 2],
                                decode=lambda ids: "".join(words[i] for i in ids))
    first, second = decode_batch(object(), tokenizer, [[{"role": "user", "content": "a"}],
                                                       [{"role": "user", "content": "b"}]], max_tokens=5)
    assert first["public_text"] == "answer" and first["termination"] == "stop"
    assert first["generated_tokens"] == 3 and first["batch_size"] == 2
    assert second["termination"] == "native_thinking_incomplete" and second["public_text"] == ""


def test_a_refilling_stream_hands_each_record_over_as_its_sequence_ends(monkeypatch) -> None:
    """Three conversations two at a time: the third starts when the first ends."""
    import importlib

    generate = importlib.import_module("mlx_lm.generate")

    import core.brain.llm.chat_format as chat_format
    from tools.g12_batched import decode_stream

    words = {11: "think", 12: "</think>", 13: "answer", 0: ""}
    seen_width: list[int] = []

    class FakeGenerator:
        def __init__(self, model, *, stop_tokens, completion_batch_size, prefill_batch_size):
            self.stream = generate.generation_stream
            self._prompt_batch = SimpleNamespace(prompt_cache=[])
            self._generation_batch = SimpleNamespace(prompt_cache=[])
            seen_width.append(completion_batch_size)
            self.steps = [
                [SimpleNamespace(uid=1, token=12, finish_reason=None), SimpleNamespace(uid=2, token=11, finish_reason=None)],
                [SimpleNamespace(uid=1, token=13, finish_reason=None), SimpleNamespace(uid=2, token=11, finish_reason=None)],
                [SimpleNamespace(uid=1, token=0, finish_reason="stop"), SimpleNamespace(uid=2, token=12, finish_reason=None)],
                [SimpleNamespace(uid=3, token=12, finish_reason=None), SimpleNamespace(uid=2, token=13, finish_reason=None)],
                [SimpleNamespace(uid=3, token=13, finish_reason=None), SimpleNamespace(uid=2, token=0, finish_reason="stop")],
                [SimpleNamespace(uid=3, token=0, finish_reason="stop")],
                [],
            ]

        def insert(self, prompts, max_tokens):
            assert len(prompts) == 3
            return [1, 2, 3]

        def next_generated(self):
            return self.steps.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(generate, "BatchGenerator", FakeGenerator)
    monkeypatch.setattr(chat_format, "render_chat_template", lambda *a, **k: "prompt")
    tokenizer = SimpleNamespace(eos_token_ids=[0], encode=lambda text, add_special_tokens=False: [1, 2],
                                decode=lambda ids: "".join(words[i] for i in ids))
    order: list[int] = []
    records: dict[int, dict] = {}

    def keep(index: int, record: dict) -> None:
        order.append(index)
        records[index] = record

    decode_stream(object(), tokenizer, [[{"role": "user", "content": c}] for c in "abc"], max_tokens=9, width=2,
                  on_record=keep)
    assert seen_width == [2]
    assert order == [0, 1, 2]
    assert all(records[i]["public_text"] == "answer" and records[i]["termination"] == "stop" for i in range(3))
    assert records[1]["generated_tokens"] == 4 and records[2]["batching"] == "continuous"


def test_metadata_evaluation_bounds_unread_chains_on_the_generator_stream(monkeypatch) -> None:
    """Every 64 steps and at completion, including a cache admitted during refill."""
    from contextlib import contextmanager

    import mlx.core as mx

    from tools.g12_batched import CACHE_METADATA_EVERY, _generation_steps

    class Metadata:
        def __init__(self):
            self.depth = 0
            self.largest = 0

        def advance(self):
            self.depth += 1
            self.largest = max(self.largest, self.depth)

    padding, length, fresh = Metadata(), Metadata(), Metadata()
    expected_stream = object()
    active_stream = []
    evaluated = []

    @contextmanager
    def stream(which):
        assert which is expected_stream
        active_stream.append(which)
        try:
            yield
        finally:
            active_stream.pop()

    def evaluate(arrays):
        assert active_stream == [expected_stream]
        evaluated.append((generator.step, tuple(arrays)))
        assert len({id(array) for array in arrays}) == len(arrays)
        for array in arrays:
            array.depth = 0

    class Generator:
        def __init__(self):
            self.stream = expected_stream
            self._prompt_batch = SimpleNamespace(prompt_cache=[])
            member = SimpleNamespace(left_padding=padding, lengths=length, offset=0)
            composite = SimpleNamespace(caches=[member, member])
            self._generation_batch = SimpleNamespace(prompt_cache=[composite])
            self.step = 0

        def next_generated(self):
            self.step += 1
            if self.step == 65:
                self._prompt_batch.prompt_cache = [SimpleNamespace(left_padding=fresh, lengths=None)]
            if self.step > 130:
                return []
            for array in (padding, length, *((fresh,) if self.step >= 65 else ())):
                array.advance()
            return [SimpleNamespace(uid=7, token=self.step, finish_reason="stop" if self.step == 130 else None)]

    generator = Generator()
    monkeypatch.setattr(mx, "array", Metadata)
    monkeypatch.setattr(mx, "stream", stream)
    monkeypatch.setattr(mx, "eval", evaluate)
    responses = list(_generation_steps(generator))
    assert [step[0].token for step in responses] == list(range(1, 131))
    assert [step for step, _arrays in evaluated] == [0, 64, 128, 130, 131]
    assert padding.largest <= CACHE_METADATA_EVERY and length.largest <= CACHE_METADATA_EVERY
    assert fresh in evaluated[2][1]
    assert padding.depth == length.depth == fresh.depth == 0


def test_cache_metadata_evaluation_rejects_an_incompatible_generator() -> None:
    import pytest

    from tools.g12_batched import _evaluate_cache_metadata

    with pytest.raises(RuntimeError, match="cache metadata and stream"):
        _evaluate_cache_metadata(SimpleNamespace(stream=None))


def test_metadata_evaluation_flattens_real_mlx_arrays_without_a_model() -> None:
    """Evaluating cache slots alone retains the metadata chain that hit the Metal cap."""
    import io

    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    from tools.g12_batched import _evaluate_cache_metadata

    stream = mx.new_stream(mx.default_device())
    with mx.stream(stream):
        cache = ArraysCache(2, left_padding=[0])
        cache.lengths = mx.array([512])
        cache[0] = mx.array([[1.0]])
        cache[1] = mx.array([[2.0]])
        mx.eval(cache.state)
        for _ in range(256):
            cache.advance(1)
            mx.eval(cache[0])

    def edges():
        graph = io.StringIO()
        mx.export_to_dot(graph, cache.left_padding, cache.lengths)
        return graph.getvalue().count("->")

    before = edges()
    generator = SimpleNamespace(stream=stream, _prompt_batch=SimpleNamespace(prompt_cache=[]),
                                _generation_batch=SimpleNamespace(prompt_cache=[cache]))
    _evaluate_cache_metadata(generator)
    after = edges()
    assert before >= 256 and after <= 8
    assert cache.left_padding.tolist() == [-256]
    assert cache.lengths.tolist() == [256]
    assert cache[0].tolist() == [[1.0]] and cache[1].tolist() == [[2.0]]
