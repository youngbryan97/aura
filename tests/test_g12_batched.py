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
