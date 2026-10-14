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


def test_a_stream_decodes_fixed_groups_and_hands_each_record_over_as_its_sequence_ends(monkeypatch) -> None:
    """Three conversations two at a time: the third waits for a fresh generator after both of the first end."""
    import importlib

    import mlx.core as mx

    generate = importlib.import_module("mlx_lm.generate")

    import core.brain.llm.chat_format as chat_format
    from tools.g12_batched import decode_stream

    words = {11: "think", 12: "</think>", 13: "answer", 0: ""}
    groups: list[int] = []
    released: list[str] = []

    class FakeGenerator:
        def __init__(self, model, *, stop_tokens, completion_batch_size, prefill_batch_size):
            self.stream = generate.generation_stream
            self._prompt_batch = SimpleNamespace(prompt_cache=[])
            self._generation_batch = SimpleNamespace(prompt_cache=[])
            assert completion_batch_size == prefill_batch_size
            groups.append(completion_batch_size)

        def insert(self, prompts, max_tokens):
            first = 10 * len(groups)
            self.uids = list(range(first, first + len(prompts)))
            self.steps = [[SimpleNamespace(uid=u, token=12, finish_reason=None) for u in self.uids],
                          [SimpleNamespace(uid=u, token=13, finish_reason=None) for u in self.uids],
                          [SimpleNamespace(uid=u, token=0, finish_reason="stop") for u in self.uids], []]
            return self.uids

        def next_generated(self):
            return self.steps.pop(0)

        def close(self):
            released.append("closed")

    monkeypatch.setattr(generate, "BatchGenerator", FakeGenerator)
    monkeypatch.setattr(chat_format, "render_chat_template", lambda *a, **k: "prompt")
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: released.append("synchronized"))
    monkeypatch.setattr(mx, "clear_cache", lambda: released.append("cleared"))
    tokenizer = SimpleNamespace(eos_token_ids=[0], encode=lambda text, add_special_tokens=False: [1, 2],
                                decode=lambda ids: "".join(words[i] for i in ids))
    order: list[int] = []
    records: dict[int, dict] = {}

    def keep(index: int, record: dict) -> None:
        order.append(index)
        records[index] = record

    decode_stream(object(), tokenizer, [[{"role": "user", "content": c}] for c in "abc"], max_tokens=9, width=2,
                  on_record=keep)
    assert groups == [2, 1]
    assert order == [0, 1, 2]
    assert released == ["closed", "synchronized", "cleared"] * 2
    assert all(records[i]["public_text"] == "answer" and records[i]["termination"] == "stop" for i in range(3))
    assert records[1]["batch_size"] == 2 and records[2]["batch_size"] == 1
    assert records[2]["batching"] == "fixed_groups"


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


def test_a_sequence_still_thinking_at_its_budget_has_its_channel_closed_and_writes_its_reply(monkeypatch) -> None:
    """Three requests: one ends in budget; one is cut mid-thought and closed; one is cut mid-reply and continued."""
    import importlib

    import mlx.core as mx

    generate = importlib.import_module("mlx_lm.generate")

    import core.brain.llm.chat_format as chat_format
    from tools.g12_batched import decode_with_thinking_budget

    vocab = {1: "a", 2: "b", 11: "think", 12: "</think>", 13: "code", 0: "", 20: "\n</think>\n\n"}
    seen: list[tuple[list[list[int]], list[int]]] = []

    class FakeGenerator:
        def __init__(self, model, *, stop_tokens, completion_batch_size, prefill_batch_size):
            self.stream = generate.generation_stream
            self._prompt_batch = SimpleNamespace(prompt_cache=[])
            self._generation_batch = SimpleNamespace(prompt_cache=[])

        def insert(self, prompts, max_tokens):
            seen.append(([list(p) for p in prompts], list(max_tokens)))
            self.uids = list(range(len(prompts)))
            if len(seen) == 1:  # the first budget: 3 tokens each
                plan = {0: [11, 12, 0], 1: [11, 11, 11], 2: [12, 13, 13]}
                reasons = {0: ["stop"], 1: ["length"], 2: ["length"]}
            else:  # what is left: replies
                plan = {0: [13, 0], 1: [13, 13]}
                reasons = {0: ["stop"], 1: ["length"]}
            self.steps = []
            for step in range(3):
                row = []
                for uid in self.uids:
                    if step < len(plan[uid]):
                        last = step == len(plan[uid]) - 1
                        row.append(SimpleNamespace(uid=uid, token=plan[uid][step],
                                                   finish_reason=reasons[uid][0] if last else None))
                self.steps.append(row)
            self.steps.append([])
            return self.uids

        def next_generated(self):
            return self.steps.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(generate, "BatchGenerator", FakeGenerator)
    monkeypatch.setattr(chat_format, "render_chat_template", lambda *a, **k: "prompt")
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)

    def encode(text, add_special_tokens=False):
        return [20] if text == "\n</think>\n\n" else [1, 2]

    tokenizer = SimpleNamespace(eos_token_ids=[0], encode=encode, decode=lambda ids: "".join(vocab[i] for i in ids))
    records: dict[int, dict] = {}
    decode_with_thinking_budget(object(), tokenizer, [[{"role": "user", "content": c}] for c in "xyz"],
                                max_tokens=5, answer_tokens=2, width=8,
                                on_record=lambda index, record: records.__setitem__(index, record))
    assert seen[0][1] == [3, 3, 3] and seen[1][1] == [2, 2]
    # The cut mid-thought request is continued from its own text plus the closing marker; the other is not closed.
    assert seen[1][0][0] == [1, 2, 11, 11, 11, 20] and seen[1][0][1] == [1, 2, 12, 13, 13]
    assert records[0]["termination"] == "stop" and not records[0]["thinking_closed_by_budget"]
    assert records[1]["thinking_closed_by_budget"] and records[1]["public_text"] == "code"
    assert records[1]["termination"] == "stop" and records[1]["generated_tokens"] == 5
    assert records[2]["public_text"] == "codecodecodecode" and not records[2]["thinking_closed_by_budget"]


def test_a_sampled_stream_hands_its_sampler_to_each_group_and_seeds_each_group(monkeypatch) -> None:
    import sys
    import types

    from tools import g12_batched

    seen: dict[str, list] = {"samplers": [], "seeds": []}

    class FakeGenerator:
        def __init__(self, model, *, stop_tokens, completion_batch_size, prefill_batch_size, sampler=None):
            seen["samplers"].append(sampler)

        def insert(self, prompts, max_tokens):
            self.uids = list(range(len(prompts)))
            return self.uids

        def next_generated(self):
            if not self.uids:
                return []
            done, self.uids = self.uids, []
            return [types.SimpleNamespace(uid=u, token=0, finish_reason="stop") for u in done]

        def close(self):
            pass

    fake_mx = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda s: seen["seeds"].append(s)),
                                    synchronize=lambda: None, clear_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", types.SimpleNamespace(BatchGenerator=FakeGenerator))
    monkeypatch.setattr(g12_batched, "_evaluate_cache_metadata", lambda generator: None)
    tokenizer = types.SimpleNamespace(eos_token_ids=[1])
    marker = object()
    finished: list[int] = []
    g12_batched._decode_token_groups(None, tokenizer, [[1]] * 5, max_tokens=4, width=2,
                                     on_finish=lambda index, *_: finished.append(index), sampler=marker, seed=7)
    assert seen["samplers"] == [marker] * 3 and seen["seeds"] == [7_000_021, 7_000_023, 7_000_025]
    assert sorted(finished) == [0, 1, 2, 3, 4]


def test_a_group_below_half_full_hands_its_long_sequences_to_the_next_group_whole(monkeypatch) -> None:
    import sys
    import types

    from tools import g12_batched

    groups: list[list[list[int]]] = []

    class FakeGenerator:
        """Each prompt's first token is how many tokens it generates in all; a carried prompt has read some."""

        def __init__(self, model, *, stop_tokens, completion_batch_size, prefill_batch_size, sampler=None):
            self.left: dict[int, int] = {}
            self.made: dict[int, int] = {}

        def insert(self, prompts, max_tokens):
            groups.append([list(p) for p in prompts])
            uids = list(range(len(prompts)))
            for uid, prompt, bound in zip(uids, prompts, max_tokens, strict=True):
                self.left[uid] = min(prompt[0] - (len(prompt) - 1), bound)
                self.made[uid] = len(prompt) - 1
            return uids

        def next_generated(self):
            out = []
            for uid in list(self.left):
                self.left[uid] -= 1
                self.made[uid] += 1
                done = self.left[uid] == 0
                out.append(types.SimpleNamespace(uid=uid, token=100 + self.made[uid],
                                                 finish_reason="length" if done else None))
                if done:
                    del self.left[uid]
            return out

        def close(self):
            pass

    fake_mx = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda s: None),
                                    synchronize=lambda: None, clear_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", types.SimpleNamespace(BatchGenerator=FakeGenerator))
    monkeypatch.setattr(g12_batched, "_evaluate_cache_metadata", lambda generator: None)
    finished: dict[int, list[int]] = {}
    g12_batched._decode_token_groups(None, types.SimpleNamespace(eos_token_ids=[1]), [[3], [9], [2], [2], [2]],
                                     max_tokens=50, width=2, carry=True,
                                     on_finish=lambda index, tokens, *_: finished.setdefault(index, list(tokens)))
    assert {i: len(t) for i, t in finished.items()} == {0: 3, 1: 9, 2: 2, 3: 2, 4: 2}
    assert finished[1] == [101 + k for k in range(9)]
    # The long sequence left its first group once that group was half full, its tokens so far read as prompt.
    assert groups[1][0] == [9, 101, 102, 103]

    groups.clear()
    finished.clear()
    g12_batched._decode_token_groups(None, types.SimpleNamespace(eos_token_ids=[1]), [[3], [9], [2], [2], [2]],
                                     max_tokens=50, width=2,
                                     on_finish=lambda index, tokens, *_: finished.setdefault(index, list(tokens)))
    assert [len(g) for g in groups] == [2, 2, 1] and groups[0] == [[3], [9]] and len(finished[1]) == 9
