"""Her head drafting in her worker's stream keeps what mlx_lm's own step gives: the same tokens, the same history to
each logits processor, the same reading on from a prompt cache, and the tap at the position being chosen after."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _tiny(vocab: int, seed: int):
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(seed)
    text = {"model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 8,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": vocab,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5}}
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text}))
    mx.eval(model.parameters())
    return model


def _head(model):
    from core.brain.llm.two_tokens_a_step import MTPHead

    head = MTPHead(model.language_model.args)
    mx.eval(head.parameters())
    return head


class _Recording:
    """A logits processor with state, as hers have: it counts, remembers each history, and leans on what it saw."""

    def __init__(self):
        self.histories = []

    def __call__(self, tokens, logits):
        self.histories.append([int(t) for t in tokens.tolist()])
        lean = (mx.arange(logits.shape[-1]) == len(self.histories) % logits.shape[-1]).astype(logits.dtype)
        return logits + 0.25 * (len(self.histories) % 3) * lean


PROMPT = [1, 3, 0, 2, 4, 1, 3, 0, 2, 5, 6, 1, 3]


def _plain(model, prompt, count, *, processors=None, cache=None):
    from mlx_lm.generate import generate_step

    return [int(t) for t, _ in generate_step(mx.array(prompt), model, max_tokens=count, logits_processors=processors,
                                             prompt_cache=cache, prefill_step_size=4)]


class _Always:
    """A depth that is always the same, and what came of its passes."""

    def __init__(self, depth):
        self.depth, self.passes = depth, []

    def next(self):
        return self.depth

    def came(self, depth, tokens, seconds):
        self.passes.append((depth, tokens, seconds))


def _with_head(model, head, prompt, count, *, processors=None, cache=None, tap=None, deep=1):
    from core.brain.llm.her_head_in_the_stream import head_generate_step

    how_deep = deep if hasattr(deep, "next") else _Always(deep)
    out = list(head_generate_step(mx.array(prompt), model, head, max_tokens=count, logits_processors=processors,
                                  prompt_cache=cache, prefill_step_size=4, tap=tap, how_deep=how_deep))
    return [int(t) for t, _, _ in out], sum(1 for *_, drafted in out if drafted)


def test_the_same_tokens_as_her_worker_decodes_one_at_a_time(on_cpu):
    kept_from_her_head = 0
    for seed in (0, 1, 2, 3):
        model = _tiny(vocab=7, seed=seed)
        tokens, drafted = _with_head(model, _head(model), PROMPT, 120)
        assert tokens == _plain(model, PROMPT, 120), seed
        kept_from_her_head += drafted
    assert kept_from_her_head > 0  # some of them were her head's drafts, kept


def _with_its_text(model, head, prompt, count, *, processors=None, cache=None, stop_after=None):
    from core.brain.llm.her_head_in_the_stream import head_generate_step

    stream = head_generate_step(mx.array(prompt), model, head, max_tokens=count, logits_processors=processors,
                                prompt_cache=cache, prefill_step_size=4, drafts_from=list(prompt))
    out = []
    for token, _, drafted in stream:
        out.append((int(token), drafted))
        if stop_after is not None and len(out) == stop_after:
            stream.close()
            break
    return [token for token, _ in out], sum(1 for _, drafted in out if drafted)


def test_what_stood_in_its_text_before_is_drafted_several_at_a_time_and_the_tokens_are_the_same(on_cpu):
    by_head = by_text = 0
    for seed in range(8):
        model = _tiny(vocab=7, seed=seed)
        head = _head(model)
        plain = _plain(model, PROMPT, 160)
        one, two = _Recording(), _Recording()
        assert _with_its_text(model, head, PROMPT, 160)[0] == plain, seed
        assert _with_its_text(model, head, PROMPT, 60, processors=[two])[0] == _plain(model, PROMPT, 60, processors=[one]), seed
        assert two.histories == one.histories[:len(two.histories)]
        by_head += _with_head(model, head, PROMPT, 160)[1]
        by_text += _with_its_text(model, head, PROMPT, 160)[1]
    assert by_text > by_head


@pytest.mark.parametrize("stop_after", [1, 2, 3, 5, 8, 13, 21, 34])
def test_stopped_anywhere_it_has_read_no_more_than_it_said(on_cpu, stop_after):
    from mlx_lm.models.cache import make_prompt_cache

    for seed in (0, 1, 2, 3):
        model = _tiny(vocab=7, seed=seed)
        cache = make_prompt_cache(model)
        said, _ = _with_its_text(model, _head(model), PROMPT, 90, cache=cache, stop_after=stop_after)
        read = next(c.offset for c in cache if hasattr(c, "offset"))
        assert len(PROMPT) + len(said) - 1 <= read <= len(PROMPT) + len(said), (seed, read, len(said))
        plain = make_prompt_cache(model)
        model.language_model.model(mx.array([[*PROMPT, *said][:read]]), plain)
        for mine, theirs in zip(cache, plain, strict=True):
            for a, b in zip(mine.state, theirs.state, strict=True):
                assert mx.allclose(a[..., : b.shape[-2], :] if a.shape != b.shape else a, b, atol=1e-3).item(), (seed, stop_after)


def test_the_room_there_is_is_kept_to(on_cpu):
    model = _tiny(vocab=7, seed=3)
    for room in range(1, 14):
        assert _with_its_text(model, _head(model), PROMPT, room)[0] == _plain(model, PROMPT, room)


@pytest.mark.parametrize("seed", [0, 4])
def test_each_logits_processor_sees_what_it_sees_one_token_at_a_time(on_cpu, seed):
    model = _tiny(vocab=7, seed=seed)
    one_at_a_time, with_head = _Recording(), _Recording()
    plain = _plain(model, PROMPT, 80, processors=[one_at_a_time])
    tokens, _ = _with_head(model, _head(model), PROMPT, 80, processors=[with_head])
    assert tokens == plain
    # mlx_lm's step works one token ahead and has processed the token after its last; the positions spoken are alike.
    assert with_head.histories == one_at_a_time.histories[:len(with_head.histories)]
    assert len(one_at_a_time.histories) - len(with_head.histories) <= 1


def test_reading_on_from_a_prompt_cache(on_cpu):
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny(vocab=7, seed=6)
    plain_cache, head_cache = make_prompt_cache(model), make_prompt_cache(model)
    for cache in (plain_cache, head_cache):
        model.language_model.model(mx.array([PROMPT[:7]]), cache)
    tokens, _ = _with_head(model, _head(model), PROMPT[7:], 60, cache=head_cache)
    assert tokens == _plain(model, PROMPT[7:], 60, cache=plain_cache)


def test_the_tap_holds_the_hidden_state_of_the_position_being_chosen_after(on_cpu):
    from types import SimpleNamespace

    from core.brain.llm.several_tokens_a_pass import _text

    model = _tiny(vocab=7, seed=7)
    tap = SimpleNamespace(last_hidden=None)
    seen = []

    def reads_the_tap(tokens, logits):
        seen.append((list(tokens.tolist()), tap.last_hidden))
        return logits

    tokens, _ = _with_head(model, _head(model), PROMPT, 40, processors=[reads_the_tap], tap=tap)
    text = _text(model)
    for history, hidden in seen[:12]:
        whole = list(PROMPT[:-1]) + history  # the prompt, then the history mlx_lm hands its processors
        expected = text.model(mx.array([whole]))[0, -1]
        assert mx.allclose(hidden, expected, atol=1e-4).item()


def test_the_stream_says_what_mlx_lms_stream_says():  # on the GPU: mlx_lm's stream reads the device's working set
    from mlx_lm import stream_generate
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from core.brain.llm.her_head_in_the_stream import stream_generate_with_her_head

    class _Tokenizer:
        eos_token_id = 6
        bos_token = None
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [int(c) for c in text]

        def decode(self, ids, **_kw):
            return "".join(str(int(i)) for i in ids)

        def convert_ids_to_tokens(self, ids):
            return [str(int(i)) for i in ids]

        def get_vocab(self):
            return {str(i): i for i in range(7)}

    # On the GPU whatever an earlier test left as the default: in a full run one left the CPU, and mlx_lm's stream
    # asked the CPU for the GPU's working set (KeyError: 'max_recommended_working_set_size').
    was = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        model = _tiny(vocab=7, seed=8)
        prompt = mx.array(PROMPT)
        plain = [(r.token, r.finish_reason) for r in stream_generate(model, TokenizerWrapper(_Tokenizer()), prompt, max_tokens=50)]
        with_head = [(r.token, r.finish_reason) for r in stream_generate_with_her_head(
            model, TokenizerWrapper(_Tokenizer()), prompt, max_tokens=50, draft_model=_head(model))]
    finally:
        mx.set_default_device(was)
    assert with_head == plain


def test_her_worker_lets_her_head_draft_with_everything_her_generations_carry(monkeypatch):
    from types import SimpleNamespace

    from core.brain.llm import mlx_worker

    model = _tiny(vocab=7, seed=9)
    head = _head(model)
    carried = {"logits_processors": [lambda tokens, logits: logits], "prompt_cache": [], "max_tokens": 8}
    assert mlx_worker._speculative_eligible(head, carried, {}) is True  # her processors and her cache, kept
    assert mlx_worker._speculative_eligible(head, {**carried, "kv_bits": 4}, {}) is False
    assert mlx_worker._speculative_eligible(head, carried, {"schema": {"type": "object"}}) is False
    # What holds a shape replays what a draft had it read (a_shape_the_decoder_enforces.py), so her work's data is drafted.
    assert mlx_worker._speculative_eligible(head, carried, {"schema": {"type": "object"}, "read_by_a_machine": True}) is True
    assert mlx_worker._speculative_eligible(None, carried, {}) is False
    assert mlx_worker._speculative_eligible(SimpleNamespace(), carried, {}) is False  # another draft: as before
    monkeypatch.setenv("AURA_HER_HEAD_DRAFTS", "0")
    from core.brain.llm.her_head_in_the_stream import her_own_head

    assert her_own_head(model) is None  # off until switched on


def test_in_her_heads_pass_every_position_is_steered_and_sampled():
    """MEASURED 20 October 2026: steered at its last position only, the first of her head's two tokens went unsteered
    and her words parted from her one-at-a-time decoding at token 59."""
    from core.consciousness.affective_steering import AffectiveSteeringHook
    from core.consciousness.spoken_positions import speaking_several

    hook = AffectiveSteeringHook.__new__(AffectiveSteeringHook)
    hook._layer_idx = 0
    pass_of_two = mx.zeros((1, 2, 8))
    assert hook._completion_position_mask(pass_of_two) is not None  # a prompt: its last position alone
    with speaking_several():
        assert hook._completion_position_mask(pass_of_two) is None  # two spoken tokens: both


@pytest.mark.parametrize("deep", [2, 3, 4])
def test_however_many_her_head_drafts_the_tokens_are_the_same_and_so_is_what_her_processors_see(on_cpu, deep):
    for seed in (0, 1, 2, 3):
        model = _tiny(vocab=7, seed=seed)
        assert _with_head(model, _head(model), PROMPT, 120, deep=deep)[0] == _plain(model, PROMPT, 120), seed
        one, several = _Recording(), _Recording()
        assert _with_head(model, _head(model), PROMPT, 60, processors=[several], deep=deep)[0] == _plain(model, PROMPT, 60, processors=[one]), seed
        assert several.histories == one.histories[:len(several.histories)]


def test_her_head_reads_what_was_said_and_not_what_it_drafted_from_its_own_output(on_cpu):
    """Whatever it drafted on the way, what it drafts next is from the same hidden states."""
    from core.brain.llm import her_head_in_the_stream as stream

    class _Seen:
        def __init__(self, head):
            self.head, self.offsets = head, []

        def __call__(self, hidden, embedded, cache):
            out = self.head(hidden, embedded, cache)
            self.offsets.append((int(hidden.shape[1]), cache.offset))
            return out

    model = _tiny(vocab=7, seed=2)
    head = _head(model)
    ends = []
    for deep in (1, 3):
        seen = _Seen(head)
        list(stream.head_generate_step(mx.array(PROMPT), model, seen, max_tokens=40, prefill_step_size=4, how_deep=_Always(deep)))
        ends.append(seen.offsets)
    # It stands at what was said after each draft of a pass, however many more it drafted before they were let go of.
    said = [offset for count, offset in ends[0]]
    assert all(offset <= len(PROMPT) + 40 + 3 for _count, offset in ends[1]) and said[-1] <= len(PROMPT) + 40


def test_passes_are_timed_by_how_many_were_drafted_and_not_by_what_is_done_with_each_token(on_cpu):
    import time

    from core.brain.llm.her_head_in_the_stream import head_generate_step

    model, deep = _tiny(vocab=7, seed=0), _Always(2)
    slow = 0.0
    for _ in head_generate_step(mx.array(PROMPT), model, _head(model), max_tokens=30, prefill_step_size=4, how_deep=deep):
        time.sleep(0.02)
        slow += 0.02
    assert deep.passes and all(depth == 2 for depth, _n, _s in deep.passes)
    assert sum(seconds for *_ , seconds in deep.passes) < slow and sum(n for _d, n, _s in deep.passes) <= 30


def test_the_depth_is_the_one_that_has_been_fastest():
    from core.brain.llm.her_head_in_the_stream import HowDeep

    def settles(seconds_for):
        deep = HowDeep()
        for _ in range(400):
            depth = deep.next()
            deep.came(depth, *seconds_for[depth])
        return deep

    # Two drafted come to 2.6 tokens in little more time; a third gains less than it takes.
    faster = settles({1: (1.94, 0.090), 2: (2.6, 0.100), 3: (2.9, 0.130), 4: (3.0, 0.170)})
    assert faster.fastest == 2 and "2: 26.0/s" in faster.says()
    assert settles({1: (1.94, 0.090), 2: (2.0, 0.110), 3: (2.0, 0.130), 4: (2.0, 0.150)}).fastest == 1
    # What was fastest stops being so when it is no longer.
    for _ in range(600):
        depth = faster.next()
        faster.came(depth, *{1: (1.94, 0.080), 2: (1.9, 0.120), 3: (1.9, 0.140), 4: (1.9, 0.160)}[depth])
    assert faster.fastest == 1


@pytest.mark.parametrize("deep", [1, 2, 3])
def test_drawing_her_sequences_follow_the_distribution_they_follow_one_token_at_a_time(on_cpu, deep):
    """The first tokens of many short generations, with her head's drafts drawn and kept by their overlap, and without."""
    from collections import Counter

    from mlx_lm.generate import generate_step

    from core.brain.llm.her_head_in_the_stream import head_generate_step
    from core.brain.llm.her_sampler import HerSampler

    model, sampler, runs = _tiny(vocab=4, seed=1), HerSampler(temp=0.9, top_p=0.95), 500
    head = _head(model)
    mx.random.seed(5)
    plain = Counter(tuple(int(t) for t, _ in generate_step(mx.array([1, 2, 3, 0]), model, max_tokens=3, sampler=sampler, prefill_step_size=4)) for _ in range(runs))
    mx.random.seed(9)
    drawn = Counter(tuple(int(t) for t, _, _ in head_generate_step(mx.array([1, 2, 3, 0]), model, head, max_tokens=3, sampler=sampler, prefill_step_size=4, how_deep=_Always(deep)))
                    for _ in range(runs))
    for said in set(plain) | set(drawn):
        a, b = plain[said] / runs, drawn[said] / runs
        assert abs(a - b) < 4.2 * (max((a + b) / 2, 0.004) * 2 / runs) ** 0.5, (said, a, b)


def _knowing(model, said):
    """Her head, where it knows: what it gives is the hidden state her model has after what was said and the draft."""
    from mlx_lm.models.cache import make_prompt_cache

    from core.brain.llm.several_tokens_a_pass import _text

    text, so_far = _text(model), list(said)

    def head(hidden, embedded, cache):
        ids = [int(mx.argmin(mx.sum((text.model.embed_tokens.weight - row) ** 2, axis=-1)).item()) for row in embedded[0]]
        so_far.extend(ids)
        return text.model(mx.array([so_far]), make_prompt_cache(model))[:, -len(ids):]

    return head


def test_drawn_drafts_are_kept_more_often_than_the_likeliest_ones(on_cpu):
    """A head that predicts what she would say exactly has every draft of its kept, however much she draws."""
    from core.brain.llm.her_head_in_the_stream import head_generate_step
    from core.brain.llm.her_sampler import HerSampler
    model = _tiny(vocab=7, seed=0)

    def kept(sampler):
        mx.random.seed(3)
        deep = _Always(1)
        list(head_generate_step(mx.array([1]), model, _knowing(model, [1]), max_tokens=80, sampler=sampler, prefill_step_size=4, how_deep=deep))
        return sum(n - 1 for _d, n, _s in deep.passes) / len(deep.passes)

    class _OnlyDraws:
        """The same sampler, where only a token can be asked of it."""

        def __init__(self, sampler):
            self.sampler = sampler

        def __call__(self, logprobs):
            return self.sampler(logprobs)

    sampler = HerSampler(temp=1.0)
    assert kept(sampler) > 0.95 > kept(_OnlyDraws(sampler)) + 0.1


def test_what_keeps_her_from_saying_a_thing_again_keeps_her_head_from_drafting_it(on_cpu):
    from mlx_lm.sample_utils import make_logits_processors

    from core.brain.llm.her_head_in_the_stream import (
        WhereItGoes,
        head_generate_step,
        of_what_was_said,
    )
    from core.brain.llm.her_sampler import HerSampler

    (penalty,) = make_logits_processors(repetition_penalty=1.8, repetition_context_size=8)
    assert of_what_was_said(penalty) and not of_what_was_said(_Recording()) and not of_what_was_said(lambda tokens, logits: logits)

    class _Hers:
        """The same, as one that keeps count would be: hers alone."""

        def __init__(self, processor):
            self.processor = processor

        def __call__(self, tokens, logits):
            return self.processor(tokens, logits)

    model = _tiny(vocab=7, seed=0)

    def kept(processor):
        goes = WhereItGoes()
        mx.random.seed(3)
        list(head_generate_step(mx.array([1]), model, _knowing(model, [1]), max_tokens=260, sampler=HerSampler(temp=1.0), logits_processors=[processor],
                                prefill_step_size=4, how_deep=_Always(1), goes=goes))
        of, held = goes.drafts[0]
        return held / of

    assert kept(penalty) > 0.97 > kept(_Hers(penalty)) + 0.03


def test_her_sequences_follow_the_distribution_they_follow_one_token_at_a_time_where_both_are_kept_from_repeating(on_cpu):
    from collections import Counter

    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_logits_processors

    from core.brain.llm.her_head_in_the_stream import head_generate_step
    from core.brain.llm.her_sampler import HerSampler

    model, sampler, runs = _tiny(vocab=4, seed=1), HerSampler(temp=1.0), 500
    head, made = _head(model), lambda: make_logits_processors(repetition_penalty=1.6, repetition_context_size=8)
    mx.random.seed(5)
    plain = Counter(tuple(int(t) for t, _ in generate_step(mx.array([1, 2, 3, 0]), model, max_tokens=3, sampler=sampler, logits_processors=made(), prefill_step_size=4)) for _ in range(runs))
    mx.random.seed(9)
    drawn = Counter(tuple(int(t) for t, _, _ in head_generate_step(mx.array([1, 2, 3, 0]), model, head, max_tokens=3, sampler=sampler, logits_processors=made(), prefill_step_size=4, how_deep=_Always(2)))
                    for _ in range(runs))
    for said in set(plain) | set(drawn):
        a, b = plain[said] / runs, drawn[said] / runs
        assert abs(a - b) < 4.2 * (max((a + b) / 2, 0.004) * 2 / runs) ** 0.5, (said, a, b)


def test_what_building_a_pass_calls_is_told_once_where_it_is_asked_for(on_cpu, monkeypatch, caplog):
    import logging

    from core.brain.llm import her_head_in_the_stream as stream

    model = _tiny(vocab=7, seed=0)

    def told():
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="LLM.HerHead"):
            list(stream.head_generate_step(mx.array(PROMPT), model, _head(model), max_tokens=90, prefill_step_size=4, how_deep=_Always(1)))
        return [r.message for r in caplog.records if r.message.startswith("Building a pass")]

    assert told() == []
    monkeypatch.setenv("AURA_LOOK_INTO_DECODE", "1")
    (said,) = told()
    assert "_linear_reading" in said
